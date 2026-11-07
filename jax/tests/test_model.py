"""The GATv2 model: initialisation, the layer against PyG, masked BatchNorm against torch, pooling,
padding invariance and the isolated-node edge case."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aevplig_jax.aev import radial_aev_numpy
from aevplig_jax.batching import Capacities, assemble, prepare
from aevplig_jax import model
from aevplig_jax.model import (aev_features, encode, gatv2_layer, init_bn_state, init_head,
                               init_params, masked_batchnorm, predict)
from aevplig_jax.pdb_atoms import PocketArrays
from aevplig_jax.rdkit_features import LigandArrays
from aevplig_jax.store import Record
from test_batching import ligand_record
from test_store import random_record

predict_jit = jax.jit(predict, static_argnames="train")
encode_jit = jax.jit(encode, static_argnames="train")


def torch_modules():
    """``torch`` and PyG's ``GATv2Conv``, imported without PyG's import-time FutureWarning."""
    import torch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from torch_geometric.nn import GATv2Conv
    return torch, GATv2Conv


def directed_record(rng, rid, n_lig, n_edges, n_prot=150) -> Record:
    """Random directed bonds without self-loops or repeats among atoms ``0..n_lig-2`` (the last atom
    is isolated), random 4-dim bond features, and a typed pocket around the ligand."""
    pairs = [(i, j) for i in range(n_lig - 1) for j in range(n_lig - 1) if i != j]
    chosen = np.sort(rng.choice(len(pairs), size=n_edges, replace=False))
    edges = np.array([pairs[c] for c in chosen], dtype=np.int32).reshape(-1, 2)
    lig = LigandArrays(
        xyz=rng.uniform(0.0, 8.0, (n_lig, 3)).astype(np.float32),
        feat=rng.random((n_lig, 15), dtype=np.float32),
        bond_src=edges[:, 0].copy(), bond_dst=edges[:, 1].copy(),
        bond_type=rng.random((n_edges, 4), dtype=np.float32),
    )
    pocket = PocketArrays(
        xyz=rng.uniform(-4.0, 12.0, (n_prot, 3)).astype(np.float32),
        species=rng.integers(0, 23, n_prot).astype(np.int8), res=np.zeros(n_prot, np.int32),
        resname=np.zeros(n_prot, np.int8), atom=np.zeros(n_prot, np.int16),
        chain=np.zeros(n_prot, np.int8), bfactor=np.zeros(n_prot, np.float16),
        element=np.zeros(n_prot, np.int8), chain_id=np.zeros(n_prot, np.int8),
        resseq=np.zeros(n_prot, np.int32), icode=np.zeros(n_prot, np.int8),
    )
    return Record(id=rid, lig=lig, pocket=pocket, label=6.0, source="synthetic", split="train")


def reference_features(record: Record) -> np.ndarray:
    """The 367 node features of ``record`` computed without the batch layout (float64 AEV)."""
    aev = radial_aev_numpy(record.lig.xyz, record.pocket.xyz, record.pocket.species)
    return np.concatenate([record.lig.feat, aev], axis=1).astype(np.float32)


def capacities_for(records, ncap=256, mcap=2048, batch=4, extra=2) -> Capacities:
    """Capacities that fit ``records``, with ``extra`` masked slots in every neighbour list."""
    prepared = [prepare(record) for record in records]
    kb = max(item.nbr.shape[1] for item in prepared) + extra
    ka = max(item.aev_nbr.shape[1] for item in prepared) + extra
    return Capacities(ncap=ncap, mcap=mcap, kb=kb, ka=ka, batch=batch)


def device_batch(batch):
    """The jit input: the array fields only (``n_real`` is host metadata)."""
    return batch._replace(n_real=0)


def conv_params(conv) -> dict:
    """A PyG ``GATv2Conv``'s weights as one ``params["gat"][i]`` entry (BatchNorm left at 1/0)."""
    def numpy(tensor):
        return tensor.detach().numpy().copy()
    width = conv.heads * conv.out_channels
    return {"w_l": numpy(conv.lin_l.weight).T, "b_l": numpy(conv.lin_l.bias),
            "w_r": numpy(conv.lin_r.weight).T, "b_r": numpy(conv.lin_r.bias),
            "w_e": numpy(conv.lin_edge.weight).T, "att": numpy(conv.att)[0],
            "bias": numpy(conv.bias),
            "bn": {"scale": np.ones(width, np.float32), "shift": np.zeros(width, np.float32)}}


def assert_trees_close(got, want, atol):
    assert jax.tree.structure(got) == jax.tree.structure(want)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=atol)


def test_init_params_shapes_and_scales():
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    assert len(params["gat"]) == 5 and params["heads"] == {}
    for i, layer in enumerate(params["gat"]):
        fan_in = 367 if i == 0 else 768
        shapes = {name: np.shape(value) for name, value in layer.items() if name != "bn"}
        assert shapes == {"w_l": (fan_in, 768), "b_l": (768,), "w_r": (fan_in, 768), "b_r": (768,),
                          "w_e": (4, 768), "att": (3, 256), "bias": (768,)}, i
        for name in ("b_l", "b_r"):
            assert np.abs(layer[name]).max() <= 1 / math.sqrt(fan_in), (i, name)
        np.testing.assert_array_equal(layer["bias"], 0.0)
        np.testing.assert_array_equal(layer["bn"]["scale"], 1.0)
        np.testing.assert_array_equal(layer["bn"]["shift"], 0.0)
        assert all(np.asarray(leaf).dtype == np.float32 for leaf in jax.tree.leaves(layer))
    # Xavier normal: std sqrt(2 / (fan_in + fan_out)); glorot uniform att: bound
    # sqrt(6 / (3 + 256)); uniform bias U(-1/sqrt(fan_in), 1/sqrt(fan_in)): std bound / sqrt(3).
    w_l = np.asarray(params["gat"][1]["w_l"])
    assert abs(w_l.std() / math.sqrt(2 / (768 + 768)) - 1) < 0.01 and abs(w_l.mean()) < 1e-3
    bound = math.sqrt(6 / (3 + 256))
    att = np.abs(np.asarray(params["gat"][2]["att"]))
    assert att.max() <= bound and att.max() > 0.99 * bound
    b_l = np.asarray(params["gat"][0]["b_l"])
    assert abs(b_l.std() / (1 / math.sqrt(367) / math.sqrt(3)) - 1) < 0.1
    # Every block draws from its own key.
    assert not np.allclose(params["gat"][1]["w_l"], params["gat"][2]["w_l"])
    assert not np.allclose(params["gat"][1]["w_l"], params["gat"][1]["w_r"])

    for name, (fan_in, fan_out) in {"fc1": (1536, 1024), "fc2": (1024, 512), "fc3": (512, 256),
                                    "out": (256, 1)}.items():
        w, b = np.asarray(params[name]["w"]), np.asarray(params[name]["b"])
        assert w.shape == (fan_in, fan_out) and b.shape == (fan_out,), name
        assert np.abs(b).max() <= 1 / math.sqrt(fan_in), name
    fc1 = np.asarray(params["fc1"]["w"])
    assert abs(fc1.std() / math.sqrt(2 / (1536 + 1024)) - 1) < 0.01
    for name, width in {"bn1": 1024, "bn2": 512, "bn3": 256}.items():
        np.testing.assert_array_equal(params[name]["scale"], np.ones(width))
        np.testing.assert_array_equal(params[name]["shift"], np.zeros(width))

    state = init_bn_state()
    assert len(state["gat"]) == 5
    for running in state["gat"]:
        np.testing.assert_array_equal(running["mean"], np.zeros(768))
        np.testing.assert_array_equal(running["var"], np.ones(768))
    for name, width in {"bn1": 1024, "bn2": 512, "bn3": 256}.items():
        np.testing.assert_array_equal(state[name]["mean"], np.zeros(width))
        np.testing.assert_array_equal(state[name]["var"], np.ones(width))

    # A head comes from head_key(key, salt): deterministic, salt-dependent, trunk untouched.
    head = init_head(key, "aux", 7, (768, 2))
    assert set(head) == {"aux"} and head["aux"]["w"].shape == (768, 2)
    assert head["aux"]["b"].shape == (2,)
    assert_trees_close(head, init_head(key, "aux", 7, (768, 2)), atol=0)
    assert not np.allclose(head["aux"]["w"], init_head(key, "aux", 8, (768, 2))["aux"]["w"])
    assert_trees_close(init_params(key), params, atol=0)


def test_head_keys_never_reuse_trunk_keys():
    """Under partitionable threefry ``split(key, n)[s] == fold_in(key, s)``: a head drawn from
    ``fold_in(key, salt)`` would copy the trunk block of index ``salt``. Heads live in their own
    namespace, so no head key equals a trunk key and no head copies a dense layer."""
    key = jax.random.PRNGKey(0)
    trunk = jax.random.split(key, model.LAYERS + len(model.DENSE_NAMES))
    trunk_data = {tuple(np.asarray(jax.random.key_data(k)).ravel()) for k in trunk}
    assert len(trunk_data) == len(trunk)
    for salt in range(17):
        head = tuple(np.asarray(jax.random.key_data(model.head_key(key, salt))).ravel())
        assert head not in trunk_data, salt
    assert model.HEADS_NS >= len(trunk)

    # Same shape as fc3 (trunk block 5 + 2 = 7) at salt 7: not a copy.
    params = init_params(key)
    head = init_head(key, "aux", 7, (512, 256))["aux"]["w"]
    corr = np.corrcoef(np.asarray(head).ravel(), np.asarray(params["fc3"]["w"]).ravel())[0, 1]
    assert abs(corr) < 0.05
    # The head draws from head_key(key, salt).
    want = model._dense(model.head_key(key, 7), 512, 256)
    assert_trees_close(init_head(key, "aux", 7, (512, 256))["aux"], want, atol=0)


@pytest.mark.torch
def test_gatv2_matches_pyg():
    torch, GATv2Conv = torch_modules()
    rng = np.random.default_rng(1)
    record = directed_record(rng, "g", n_lig=12, n_edges=30)
    in_degree = np.bincount(record.lig.bond_dst, minlength=12)
    assert in_degree[11] == 0 and in_degree.max() >= 4  # an isolated node and a wide softmax
    batch = assemble([record], capacities_for([record], batch=1), y_mean=0.0, y_sd=1.0)
    x = np.array(aev_features(batch))  # a writable copy for torch.from_numpy
    np.testing.assert_allclose(x[:12], reference_features(record), atol=1e-5)

    torch.manual_seed(0)
    conv = GATv2Conv(367, 256, heads=3, edge_dim=4)
    with torch.no_grad():
        conv.bias.normal_(0.0, 0.1)  # PyG starts it at zero
    edge_index = torch.from_numpy(
        np.stack([record.lig.bond_src, record.lig.bond_dst]).astype(np.int64))
    with torch.no_grad():
        want = conv(torch.from_numpy(x[:12]), edge_index,
                    torch.from_numpy(record.lig.bond_type)).numpy()
    with jax.default_matmul_precision("highest"):
        got = np.asarray(gatv2_layer(conv_params(conv), x, batch.nbr, batch.nbr_mask,
                                     batch.nbr_feat, heads=3, hidden=256))
    assert got.shape == (256, 768)
    assert np.abs(got[:12] - want).max() < 1e-5


@pytest.mark.torch
def test_masked_batchnorm_matches_torch_train_and_eval():
    torch, _ = torch_modules()
    rng = np.random.default_rng(2)
    real = rng.normal(1.5, 2.0, (7, 6)).astype(np.float32)
    x = np.concatenate([real, np.full((3, 6), 1e3, np.float32)])  # padded rows far off
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], np.float32)
    torch.manual_seed(1)
    norm = torch.nn.BatchNorm1d(6)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.normal_(0.0, 0.5)
        norm.running_mean.normal_(0.0, 0.5)
        norm.running_var.uniform_(0.5, 2.0)
    p = {"scale": norm.weight.detach().numpy().copy(), "shift": norm.bias.detach().numpy().copy()}
    state = {"mean": norm.running_mean.numpy().copy(), "var": norm.running_var.numpy().copy()}

    norm.train()
    with torch.no_grad():
        want_train = norm(torch.from_numpy(real)).numpy()
    got_train, trained = masked_batchnorm(p, state, x, valid, train=True)
    assert np.abs(np.asarray(got_train)[:7] - want_train).max() < 1e-5
    np.testing.assert_allclose(trained["mean"], norm.running_mean.numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(trained["var"], norm.running_var.numpy(), rtol=0, atol=1e-6)

    norm.eval()
    with torch.no_grad():
        want_eval = norm(torch.from_numpy(real)).numpy()
    got_eval, kept = masked_batchnorm(p, trained, x, valid, train=False)
    assert np.abs(np.asarray(got_eval)[:7] - want_eval).max() < 1e-5
    assert_trees_close(kept, trained, atol=0)


def test_isolated_node_graph_runs():
    """A single-atom ligand alone in a batch with padded graph slots: finite predictions, state and
    gradients in both modes (training-mode statistics over one node and one graph)."""
    record = ligand_record("single", [[1.0, 2.0, 3.0]], [],
                           prot_xyz=[[1.0, 2.0, 6.0], [4.0, 2.0, 3.0]], species=[5, 9])
    batch = device_batch(assemble([record], Capacities(ncap=256, mcap=1024, kb=3, ka=32, batch=4),
                                  y_mean=0.0, y_sd=1.0))
    assert np.asarray(batch.aev_mask)[0].sum() == 2  # both pocket atoms within the cutoff
    params, state = init_params(jax.random.PRNGKey(0)), init_bn_state()
    for train in (False, True):
        pred, new_state = predict_jit(params, state, batch, train=train)
        assert pred.shape == (4,) and np.isfinite(np.asarray(pred)).all(), train
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(new_state))

    def loss(p):
        pred, _ = predict(p, state, batch, train=True)
        return jnp.sum(batch.valid * pred ** 2)

    grads = jax.jit(jax.grad(loss))(params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))


def test_padding_does_not_change_prediction(rng):
    records = [random_record(rng, f"r{i}") for i in range(3)]
    small = capacities_for(records, ncap=256, mcap=2048, batch=4, extra=0)
    big = Capacities(ncap=512, mcap=4096, kb=small.kb + 3, ka=small.ka + 32, batch=7)
    batch_small = device_batch(assemble(records, small, y_mean=6.0, y_sd=2.0))
    batch_big = device_batch(assemble(records, big, y_mean=6.0, y_sd=2.0))
    params, state = init_params(jax.random.PRNGKey(1)), init_bn_state()

    # Evaluation mode: identical to 1e-6.
    eval_small, _ = predict_jit(params, state, batch_small, train=False)
    eval_big, _ = predict_jit(params, state, batch_big, train=False)
    np.testing.assert_allclose(eval_small[:3], eval_big[:3], rtol=0, atol=1e-6)
    # Training mode: compiled, the BatchNorm column sums over 256 and 512 rows are fused in a
    # different order (eager execution is bit-identical), a few float32 ulps of the node
    # embeddings; a padded row reaching the statistics or the pooling would move them far more.
    pred_small, trained_small = predict_jit(params, state, batch_small, train=True)
    pred_big, trained_big = predict_jit(params, state, batch_big, train=True)
    np.testing.assert_allclose(pred_small[:3], pred_big[:3], rtol=1e-5, atol=1e-6)
    assert_trees_close(trained_small, trained_big, atol=1e-5)
    trained_eval_small, _ = predict_jit(params, trained_small, batch_small, train=False)
    trained_eval_big, _ = predict_jit(params, trained_small, batch_big, train=False)
    np.testing.assert_allclose(trained_eval_small[:3], trained_eval_big[:3], rtol=0, atol=1e-6)
    assert not np.allclose(trained_eval_small[:3], pred_small[:3])  # running vs batch statistics

    # Pooling: max and mean over each graph's valid rows; padded rows are zero; empty slot -> 0.
    node_emb, graph_emb, _ = encode_jit(params, trained_small, batch_small, train=False)
    node_emb, graph_emb = np.asarray(node_emb), np.asarray(graph_emb)
    n = sum(len(r.lig.xyz) for r in records)
    np.testing.assert_array_equal(node_emb[n:], 0.0)
    graph_of = np.asarray(batch_small.node_graph)
    for g in range(3):
        rows = node_emb[:n][graph_of[:n] == g]
        np.testing.assert_array_equal(graph_emb[g, :768], rows.max(axis=0))
        np.testing.assert_allclose(graph_emb[g, 768:], rows.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(graph_emb[3], 0.0)
