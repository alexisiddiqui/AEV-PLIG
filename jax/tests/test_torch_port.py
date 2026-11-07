"""The PyTorch weight port: parameter count, key checks and predictions of a ported GATv2Net."""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from aevplig_jax.batching import assemble
from aevplig_jax.model import init_bn_state, init_params, predict
from aevplig_jax.torch_port import load_torch_model, port_state_dict
from test_model import (assert_trees_close, capacities_for, device_batch, directed_record,
                        reference_features)

REPO = Path(__file__).resolve().parents[2]  # model_defs.py and utils.py of the original
SRC = Path(__file__).resolve().parents[1] / "src"
CONFIG = SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=256)


def original():
    """``GATv2Net`` (model_defs.py) and ``init_weights`` (utils.py) from the repository root,
    imported without PyG's import-time FutureWarning (utils.py also resets the warning filters)."""
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from model_defs import GATv2Net
        from utils import init_weights
    return GATv2Net, init_weights


def pyg_graphs(records):
    """The records as one PyG batch: features without the batch layout, bonds as given."""
    import torch
    from torch_geometric.data import Batch, Data
    graphs = [Data(x=torch.from_numpy(reference_features(record)),
                   edge_index=torch.from_numpy(
                       np.stack([record.lig.bond_src, record.lig.bond_dst]).astype(np.int64)),
                   edge_attr=torch.from_numpy(record.lig.bond_type))
              for record in records]
    return Batch.from_data_list(graphs)


@pytest.mark.torch
def test_param_count_matches_pytorch():
    GATv2Net, _ = original()
    torch_count = sum(p.numel() for p in GATv2Net(367, 4, CONFIG).parameters())
    jax_count = sum(leaf.size for leaf in jax.tree.leaves(init_params(jax.random.PRNGKey(0))))
    assert jax_count == torch_count == 7_554_561


@pytest.mark.torch
def test_port_rejects_unexpected_and_missing_keys():
    GATv2Net, _ = original()
    sd = {name: value.numpy() for name, value in GATv2Net(367, 4, CONFIG).state_dict().items()}
    assert any(name.endswith("num_batches_tracked") for name in sd)
    params, state = port_state_dict(sd)  # num_batches_tracked is ignored
    assert len(params["gat"]) == 5 and len(state["gat"]) == 5

    missing = {name: value for name, value in sd.items() if name != "GNN_layers.2.lin_edge.weight"}
    with pytest.raises(ValueError, match=r"missing \['GNN_layers.2.lin_edge.weight'\]"):
        port_state_dict(missing)
    extra = {**sd, "GNN_layers.0.lin_edge.bias": np.zeros(768, np.float32)}
    with pytest.raises(ValueError, match=r"unexpected \['GNN_layers.0.lin_edge.bias'\]"):
        port_state_dict(extra)


@pytest.mark.torch
def test_port_round_trip_predictions(tmp_path):
    import torch
    GATv2Net, init_weights = original()
    torch.manual_seed(0)
    net = GATv2Net(367, 4, CONFIG)
    net.apply(init_weights)
    with torch.no_grad():  # move every default-valued tensor so a mapping slip shows
        for layer in net.GNN_layers:
            layer.bias.normal_(0.0, 0.1)
        for norm in [bn.module for bn in net.BN_layers] + [net.bn_connect1, net.bn_connect2,
                                                           net.bn_connect3]:
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.normal_(0.0, 0.2)

    rng = np.random.default_rng(3)
    records = [directed_record(rng, f"g{i}", n_lig, n_edges)
               for i, (n_lig, n_edges) in enumerate([(12, 30), (7, 12), (20, 45)])]
    graphs = pyg_graphs(records)
    net.train()
    with torch.no_grad():  # running statistics from two training-mode passes
        net(graphs)
        net(graphs)
    path = tmp_path / "GATv2Net.model"
    torch.save(net.state_dict(), path)

    sd = load_torch_model(path)
    assert all(isinstance(value, np.ndarray) for value in sd.values())
    params, state = port_state_dict(sd)
    assert jax.tree.structure(params) == jax.tree.structure(init_params(jax.random.PRNGKey(0)))
    assert jax.tree.structure(state) == jax.tree.structure(init_bn_state())
    assert all(np.asarray(leaf).dtype == np.float32 for leaf in jax.tree.leaves((params, state)))

    batch = device_batch(assemble(records, capacities_for(records, batch=5), y_mean=0.0, y_sd=1.0))
    net.eval()
    with torch.no_grad():
        want = net(graphs).numpy()[:, 0]
    with jax.default_matmul_precision("highest"):
        got, kept = predict(params, state, batch, train=False)
    assert np.abs(np.asarray(got)[:3] - want).max() < 1e-4
    assert_trees_close(kept, state, atol=0)

    # Training mode: batch statistics and the running-statistics update match torch too. Dense
    # BatchNorm over three graphs amplifies float32 rounding (torch's own output moves by ~1e-4
    # under 1e-6 relative noise on its graph embedding; in float64 the two agree to 5e-8), so
    # 1e-3 here; a biased-variance or momentum slip would be off by percents.
    net.train()
    with torch.no_grad():
        want_train = net(graphs).numpy()[:, 0]
    with jax.default_matmul_precision("highest"):
        got_train, trained = predict(params, state, batch, train=True)
    assert np.abs(np.asarray(got_train)[:3] - want_train).max() < 1e-3
    _, torch_trained = port_state_dict({name: value.numpy()
                                        for name, value in net.state_dict().items()})
    assert_trees_close(trained, torch_trained, atol=1e-3)


def test_model_and_port_import_without_torch():
    code = ("import sys, aevplig_jax.model, aevplig_jax.torch_port; "
            "print(sorted({'torch', 'torch_geometric', 'rdkit', 'pandas'} & set(sys.modules)))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"
