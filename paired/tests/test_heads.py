"""Tests for src/heads.py (arm S): Siamese head antisymmetry and BatchNorm behaviour, the
encoder/affinity split of GATv2Net and its state-dict renaming, and train_paired.siamese_terms."""
import copy
import io
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

import heads
import pair_losses
import train_paired
from model_defs import GATv2Net

N_FEAT, N_EDGE = 20, 4


def _gnn(seed=0, hidden=8, head=3):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(hidden_dim=hidden, head=head, activation_function="leaky_relu")
    return GATv2Net(N_FEAT, N_EDGE, cfg)


def _graph(k, gen, **fields):
    src = list(range(k - 1)) + list(range(1, k))
    dst = list(range(1, k)) + list(range(k - 1))
    return Data(x=torch.rand(k, N_FEAT, generator=gen), edge_index=torch.tensor([src, dst]),
                edge_attr=torch.rand(len(src), N_EDGE, generator=gen), **fields)


def _batch(n=6, seed=0, series=None, lig=None, y=None, kind=None):
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for k in range(n):
        fields = {}
        if series is not None:
            fields.update(series=torch.tensor([series[k]]), lig=torch.tensor([lig[k]]),
                          y=torch.tensor([float(y[k])]))
        if kind is not None:
            fields["kind"] = torch.tensor([kind[k]])
        graphs.append(_graph(3 + k % 3, gen, **fields))
    return Batch.from_data_list(graphs)


def _trained_head(dim=16, hidden=8, seed=0):
    """A head with non-default weights and BatchNorm running statistics."""
    torch.manual_seed(seed)
    head = heads.SiameseHead(dim, hidden=hidden)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    head.train()
    with torch.no_grad():
        for _ in range(3):
            z = torch.randn(32, dim)
            head(z[:16], z[16:])
    return head


def test_siamese_antisymmetric():
    head = _trained_head()
    torch.manual_seed(1)
    zi, zj = torch.randn(10, 16), torch.randn(10, 16)
    head.eval()
    with torch.no_grad():
        d_ij, d_ji = head(zi, zj), head(zj, zi)
        assert d_ij.shape == (10,)
        assert torch.allclose(d_ij, -d_ji, atol=1e-6, rtol=0)
        assert torch.allclose(head(zi, zi), torch.zeros(10), atol=1e-6, rtol=0)
        # the raw score is not antisymmetric: the symmetrisation does the work
        assert (head.m(zi, zj) + head.m(zj, zi)).abs().max() > 1e-3
        assert d_ij.abs().max() > 1e-3
    # train mode: both orders share one BatchNorm batch, so antisymmetry holds there too
    head.train()
    with torch.no_grad():
        assert torch.allclose(head(zi, zj), -head(zj, zi), atol=1e-5, rtol=0)


def test_siamese_single_pair_eval():
    """Decision: BatchNorm1d (the clone's MLP) is kept. Both orders of every pair run as one batch, so
    a single pair is a batch of 2 (train mode does not raise) and eval mode uses running statistics."""
    head = _trained_head()
    torch.manual_seed(2)
    z = torch.randn(2, 16)
    head.eval()
    with torch.no_grad():
        one = head(z[:1], z[1:])
        assert one.shape == (1,) and torch.isfinite(one).all()
        # eval mode: per-row computation, a pair's value does not depend on the other pairs
        many = head(torch.cat([z[:1], torch.randn(5, 16)]), torch.cat([z[1:], torch.randn(5, 16)]))
        assert torch.allclose(many[:1], one, atol=1e-6)
        assert head(z[:0], z[:0]).shape == (0,)
    head.train()
    one_train = head(z[:1], z[1:])  # one pair in train mode: no "more than 1 value per channel" error
    assert one_train.shape == (1,) and torch.isfinite(one_train).all()
    assert head(z[:0], z[:0]).shape == (0,)  # empty input never reaches BatchNorm
    with pytest.raises(ValueError):
        head(z, z[:1])


def test_wrapper_forward_matches_gatv2net():
    gnn = _gnn(3)
    ref = copy.deepcopy(gnn)
    wrapper = heads.AffinityWithSiamese(gnn, heads.SiameseHead(heads.embed_dim(gnn), hidden=8))
    data = _batch(6, seed=3)
    assert heads.embed_dim(gnn) == 2 * 8 * 3
    for train in (False, True):  # eval first: train-mode passes move the running statistics
        wrapper.train(train)
        ref.train(train)
        assert torch.equal(wrapper(data), ref(data))
        z = wrapper.embed(data)
        assert z.shape == (6, 48)
        assert torch.equal(wrapper.affinity(z), wrapper(data))  # train mode: batch statistics
    # the wrapper shares the GATv2Net's modules: no copy
    assert wrapper.encoder.GNN_layers is gnn.GNN_layers and wrapper.affinity.fc1 is gnn.fc1
    with pytest.raises(ValueError):
        heads.AffinityWithSiamese(_gnn(), heads.SiameseHead(10))


def test_state_dict_roundtrip():
    g1 = _gnn(4)
    data = _batch(6, seed=4)
    g1.train()
    g1(data)  # move BatchNorm running stats / num_batches_tracked off their defaults
    g1.eval()
    sd1 = g1.state_dict()

    wrapped = heads.plain_to_wrapped(sd1)
    assert list(wrapped) == [("encoder." if k.split(".")[0] in heads.ENCODER_KEYS else "affinity.") + k
                             for k in sd1]
    back = heads.wrapped_to_plain(wrapped)
    assert list(back) == list(sd1) and all(back[k] is sd1[k] for k in sd1)

    # a plain dict into the wrapper and back is lossless; the wrapper predicts as the GATv2Net
    wrapper = heads.AffinityWithSiamese(_gnn(5), heads.SiameseHead(48, hidden=8))
    wrapper.from_plain_state_dict(sd1).eval()
    plain = wrapper.to_plain_state_dict()
    assert list(plain) == list(sd1)
    assert all(torch.equal(plain[k], sd1[k]) for k in sd1)
    assert set(plain._metadata) == set(sd1._metadata)
    g3 = _gnn(6)
    g3.load_state_dict(plain)
    g3.eval()
    assert torch.equal(g3(data), g1(data)) and torch.equal(wrapper(data), g1(data))

    # the wrapper's own dict: encoder.* / affinity.* / siamese.* only; save -> load -> same outputs
    sd_w = wrapper.state_dict()
    assert {k.split(".")[0] for k in sd_w} == {"encoder", "affinity", "siamese"}
    assert heads.is_wrapped_state_dict(sd_w) and not heads.is_wrapped_state_dict(sd1)
    assert heads.has_siamese_keys(sd_w) and heads.siamese_hidden_from_state_dict(sd_w) == 8
    buf = io.BytesIO()
    torch.save(sd_w, buf)
    buf.seek(0)
    again = heads.AffinityWithSiamese(_gnn(7), heads.SiameseHead(48, hidden=8))
    again.load_state_dict(torch.load(buf, weights_only=True))
    again.eval()
    with torch.no_grad():
        z = wrapper.embed(data)
        assert torch.equal(again.embed(data), z)
        i, j = torch.tensor([0, 1]), torch.tensor([2, 3])
        assert torch.equal(again.pair_delta(z, i, j), wrapper.pair_delta(z, i, j))
    # strict: unknown or missing keys are errors
    with pytest.raises(KeyError):
        heads.plain_to_wrapped({**sd1, "extra.weight": torch.zeros(1)})
    with pytest.raises(KeyError):
        heads.wrapped_to_plain({**sd_w, "encoder.extra": torch.zeros(1)})
    incomplete = {k: v for k, v in sd1.items() if not k.startswith("fc3.")}
    with pytest.raises(KeyError):
        heads.AffinityWithSiamese(_gnn(8), None).from_plain_state_dict(incomplete)


def _cfg():
    return train_paired.LossConfig(kind="none", weight=0.0, tie=0.0, delta=1.0, weighting="none")


def test_siamese_no_pairs():
    """No same-series different-ligand pair among the MSE-kind rows: the Siamese terms are exact zeros,
    the head is not called, and the gradients are those of the MSE alone."""
    scfg = train_paired.SiameseConfig(weight=1.0, tie=0.3, tau=0.3, delta=0.6)
    cases = {
        "singletons_and_shared_ligand": dict(series=[0, 1, 2, 2], lig=[0, 1, 2, 2], kind=None),
        # rows 2 and 3 share a series with different ligands, but row 3 is a hinge kind (shape twin)
        "hinge_partner": dict(series=[0, 1, 2, 2], lig=[0, 1, 2, 3], kind=[0, 0, 0, 4]),
    }
    for name, spec in cases.items():
        wrapper = heads.AffinityWithSiamese(_gnn(9, head=1), heads.SiameseHead(16, hidden=8))
        ref = copy.deepcopy(wrapper)
        data = _batch(4, seed=9, series=spec["series"], lig=spec["lig"], y=[0.5, -1.0, 1.5, 0.0],
                      kind=spec["kind"])
        bn_before = wrapper.siamese.bn1.running_mean.clone()
        wrapper.train()
        ref.train()
        output, z = train_paired.forward_embed(wrapper, data)
        terms = train_paired.compute_loss(output, data, _cfg())
        weighted, huber, bce = train_paired.siamese_terms(wrapper.siamese, z, data, scfg)
        for t in (weighted, huber, bce):
            assert t.item() == 0.0 and not t.requires_grad, name
        total = terms[0] + weighted
        assert total.item() == terms[1].item(), name  # = the MSE
        total.backward()
        ref_total = train_paired.compute_loss(ref(data), data, _cfg())[0]
        ref_total.backward()
        for (k, p), (_, q) in zip(wrapper.named_parameters(), ref.named_parameters()):
            if k.startswith("siamese."):
                assert p.grad is None, (name, k)
            else:
                assert torch.equal(p.grad, q.grad), (name, k)
        assert torch.equal(wrapper.siamese.bn1.running_mean, bn_before), name


def test_siamese_terms_pairs():
    """With pairs: the term equals siamese_loss over pair_index of the MSE-kind rows, and gradients
    reach both the head and the encoder."""
    scfg = train_paired.SiameseConfig(weight=2.0, tie=0.4, tau=0.5, delta=1.0)
    series, lig = [0, 0, 0, 1, 1, 2], [0, 1, 2, 3, 3, 4]
    kind = [0, 0, 4, 0, 0, 0]  # row 2 is a hinge kind: pairs (0, 1) only; rows 3, 4 share a ligand
    y = [0.0, 1.0, 2.0, -1.0, 0.5, 0.2]
    wrapper = heads.AffinityWithSiamese(_gnn(10, head=1), heads.SiameseHead(16, hidden=8))
    data = _batch(6, seed=10, series=series, lig=lig, y=y, kind=kind)
    wrapper.train()
    output, z = train_paired.forward_embed(wrapper, data)
    weighted, huber, bce = train_paired.siamese_terms(wrapper.siamese, z, data, scfg)
    delta = wrapper.siamese(z[[0]], z[[1]])
    h, b = pair_losses.siamese_loss(delta, data.y, torch.tensor([0]), torch.tensor([1]), 0.4, 0.5, 1.0)
    assert huber.item() == pytest.approx(h.item(), abs=1e-6) and bce.item() == pytest.approx(b.item(), abs=1e-6)
    assert weighted.item() == pytest.approx(2.0 * (h.item() + b.item()), abs=1e-6)
    weighted.backward()
    assert wrapper.siamese.fc1.weight.grad.abs().sum() > 0
    assert wrapper.encoder.GNN_layers[0].lin_l.weight.grad.abs().sum() > 0
    assert wrapper.affinity.fc1.weight.grad is None  # the Siamese term does not touch the pointwise MLP


def test_ligand_only_embed_masks_aev():
    wrapper = heads.AffinityWithSiamese(_gnn(11, head=1), heads.SiameseHead(16, hidden=8)).eval()
    data = _batch(4, seed=11)
    gate = train_paired.LigandOnly(wrapper)
    masked = data.clone()
    masked.x = masked.x.clone()
    masked.x[:, 15:] = 0
    x_before = data.x.clone()
    with torch.no_grad():
        output, z = train_paired.forward_embed(gate, data)
        assert torch.equal(z, wrapper.embed(masked))
        assert torch.equal(output, gate(data)) and torch.equal(output, wrapper(masked))
    assert torch.equal(data.x, x_before)
    assert train_paired.core_of(gate) is wrapper and train_paired.core_of(wrapper) is wrapper


# ----------------------------------------------------------------------------- pair head, moments heads (arm P)

def _pair_head(dim=16, hidden=8, seed=0):
    torch.manual_seed(seed)
    head = heads.PairHead(dim, hidden=hidden)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return head


def test_pairhead_asymmetric_rowwise():
    head = _pair_head()
    torch.manual_seed(3)
    za, zc = torch.randn(6, 16), torch.randn(6, 16)
    head.eval()
    with torch.no_grad():
        l_ac, l_ca = head(za, zc), head(zc, za)
        assert l_ac.shape == (6,) and torch.isfinite(l_ac).all()
        assert (l_ac - l_ca).abs().max() > 1e-3  # the signed block: l(a, c) != l(c, a)
        # LayerNorm: rows are independent, train and eval mode agree, a single pair works
        one = head(za[:1], zc[:1])
        assert torch.allclose(one, l_ac[:1], atol=1e-6)
    head.train()
    with torch.no_grad():
        assert torch.allclose(head(za, zc), l_ac, atol=1e-6)
        assert head(za[:1], zc[:1]).shape == (1,)
    assert head(za[:0], zc[:0]).shape == (0,)
    with pytest.raises(ValueError):
        head(za, zc[:2])
    with pytest.raises(ValueError):
        head(za[:, :8], zc[:, :8])
    # the input blocks are [z_a * z_c, |z_a - z_c|, z_a - z_c]
    assert head.fc1.in_features == 3 * 16 and isinstance(head.ln1, torch.nn.LayerNorm)


def test_moments_head_and_init():
    torch.manual_seed(4)
    head = heads.init_linear(heads.MomentsHead(16, out=343, hidden=8), lambda m: torch.nn.init.constant_(m.weight, 0.1))
    assert torch.all(head.fc1.weight == 0.1) and torch.all(head.ln1.weight == 1.0)  # LayerNorm untouched
    z = torch.randn(5, 16)
    assert head(z).shape == (5, 343) and head(z[:0]).shape == (0, 343)
    with pytest.raises(ValueError):
        head(z[:, :4])
    # the clone's init_weights itself would fail on LayerNorm's 1-d weight: init_linear avoids it
    from utils import init_weights
    heads.init_linear(heads.PairHead(16, hidden=8), init_weights)
    with pytest.raises(ValueError):
        heads.PairHead(16, hidden=8).apply(init_weights)


def _full_wrapper(seed=12, siamese=False):
    gnn = _gnn(seed, head=1)
    dim = heads.embed_dim(gnn)
    moments = {"ligand": heads.MomentsHead(dim, out=343, hidden=8), "pocket": heads.MomentsHead(dim, out=343, hidden=8)}
    return heads.AffinityWithSiamese(gnn, heads.SiameseHead(dim, hidden=8) if siamese else None,
                                     heads.PairHead(dim, hidden=8), moments)


def test_wrapper_with_pair_and_moments_state_dict():
    wrapper = _full_wrapper()
    sd = wrapper.state_dict()
    assert {k.split(".")[0] for k in sd} == {"encoder", "affinity", "pair", "moments"}
    assert [k.split(".")[1] for k in sd if k.startswith("moments.")][0] == "pocket"  # fixed order
    assert wrapper.moments_targets == ("pocket", "ligand")
    assert heads.pair_hidden_from_state_dict(sd) == 8 and heads.siamese_hidden_from_state_dict(sd) is None
    assert heads.moments_from_state_dict(sd) == {"pocket": (8, 343), "ligand": (8, 343)}
    assert heads.has_pair_keys(sd) and not heads.has_siamese_keys(sd)
    # wrapped_to_plain drops every head; the plain dict loads into a GATv2Net and back into the wrapper
    plain = wrapper.to_plain_state_dict()
    assert not any(k.split(".")[0] in ("pair", "moments", "siamese", "encoder", "affinity") for k in plain)
    g = _gnn(13, head=1)
    g.load_state_dict(plain)
    again = _full_wrapper(14)
    again.from_plain_state_dict(plain)
    data = _batch(4, seed=12)
    wrapper.eval(), again.eval(), g.eval()
    with torch.no_grad():
        assert torch.equal(again(data), g(data)) and torch.equal(wrapper(data), g(data))
        z = wrapper.embed(data)
        assert wrapper.pair_score(z, torch.tensor([0, 1]), torch.tensor([2, 3])).shape == (2,)
        assert wrapper.moments_pred(z, "pocket").shape == (4, 343)
    # a Siamese-only wrapper keeps arm S's keys exactly
    s_only = heads.AffinityWithSiamese(_gnn(15, head=1), heads.SiameseHead(16, hidden=8))
    assert {k.split(".")[0] for k in s_only.state_dict()} == {"encoder", "affinity", "siamese"}
    assert s_only.pair is None and s_only.moments is None and s_only.moments_targets == ()
    with pytest.raises(RuntimeError):
        s_only.pair_score(z, torch.tensor([0]), torch.tensor([1]))
    with pytest.raises(RuntimeError):
        s_only.moments_pred(z, "ligand")
    with pytest.raises(ValueError):
        heads.AffinityWithSiamese(_gnn(16, head=1), None, heads.PairHead(10))
    with pytest.raises(ValueError):
        heads.AffinityWithSiamese(_gnn(16, head=1), None, None, {"protein": heads.MomentsHead(16)})
    with pytest.raises(KeyError):
        heads.moments_from_state_dict({**sd, "moments.protein.fc1.weight": torch.zeros(8, 16),
                                       "moments.protein.out.weight": torch.zeros(343, 8)})


def _aux_batch(n, seed, series, lig, y, kind, row, parent):
    gen = torch.Generator().manual_seed(seed)
    graphs = [_graph(3 + k % 3, gen, series=torch.tensor([series[k]]), lig=torch.tensor([lig[k]]),
                     y=torch.tensor([float(y[k])]), kind=torch.tensor([kind[k]]), row=torch.tensor([row[k]]),
                     parent=torch.tensor([parent[k]])) for k in range(n)]
    return Batch.from_data_list(graphs)


def test_pairhead_terms_no_anchor():
    """No anchor (every series has < 2 candidates): exact zeros, head not called, gradients those of the MSE."""
    cfg = train_paired.PairHeadConfig(weight=0.3, tau=0.5, neg_kinds=(4,))
    data = _aux_batch(5, 20, series=[0, 0, 1, 1, 1], lig=[0, 1, 2, 2, 2], y=[0.1, 0.5, -1.0, 0.0, 0.3],
                      kind=[0, 0, 0, 0, 4], row=[0, 1, 2, 3, 4], parent=[0, 1, 2, 3, 0])
    wrapper = heads.AffinityWithSiamese(_gnn(17, head=1), None, heads.PairHead(16, hidden=8))
    ref = copy.deepcopy(wrapper)
    wrapper.train(), ref.train()
    output, z = train_paired.forward_embed(wrapper, data)
    weighted, loss, n_anchors, n_neg = train_paired.pairhead_terms(wrapper.pair, z, data, cfg)
    assert (weighted.item(), loss.item(), n_anchors, n_neg) == (0.0, 0.0, 0, 0)
    assert not weighted.requires_grad
    (train_paired.compute_loss(output, data, _cfg())[0] + weighted).backward()
    train_paired.compute_loss(ref(data), data, _cfg())[0].backward()
    for (k, p), (_, q) in zip(wrapper.named_parameters(), ref.named_parameters()):
        if k.startswith("pair."):
            assert p.grad is None, k
        else:
            assert torch.equal(p.grad, q.grad), k


def test_pairhead_terms_with_negatives():
    cfg = train_paired.PairHeadConfig(weight=2.0, tau=0.5, neg_kinds=(4,))
    data = _aux_batch(5, 21, series=[0, 0, 0, 0, 1], lig=[0, 1, 2, 0, 3], y=[0.0, 1.0, 2.0, 0.0, 0.5],
                      kind=[0, 0, 0, 4, 0], row=[10, 11, 12, 13, 14], parent=[10, 11, 12, 10, 14])
    wrapper = heads.AffinityWithSiamese(_gnn(18, head=1), None, heads.PairHead(16, hidden=8))
    wrapper.train()
    _, z = train_paired.forward_embed(wrapper, data)
    weighted, loss, n_anchors, n_neg = train_paired.pairhead_terms(wrapper.pair, z, data, cfg)
    a, c, neg = pair_losses.pairhead_candidates(data.series, data.lig, data.kind, data.row, data.parent, (4,))
    expected = pair_losses.listwise_pairhead(wrapper.pair(z[a], z[c]), data.y, a, c, neg, 0.5)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6) and weighted.item() == pytest.approx(2 * loss.item())
    assert (n_anchors, n_neg) == (3, 1)
    weighted.backward()
    assert wrapper.pair.fc1.weight.grad.abs().sum() > 0
    assert wrapper.encoder.GNN_layers[0].lin_l.weight.grad.abs().sum() > 0
    assert wrapper.affinity.fc1.weight.grad is None


class _Side:
    """A MomentsTargets over a hand-made side dict (rows 0..n-1)."""

    def __init__(self, values, masks):
        import numpy as np

        n = len(next(iter(values.values())))
        side = {"lookup": np.arange(n)}
        for t in values:
            side[f"{t}_std"] = np.asarray(values[t], dtype=np.float32)
            side[f"mask_{t}"] = np.asarray(masks[t], dtype=bool)
        self.targets = train_paired.MomentsTargets(side, tuple(values), torch.device("cpu"))


def test_moments_mask():
    """Rows without a descriptor contribute nothing to the moments loss and get no gradient (NaN values in a
    masked row are never read); a target without any row in the batch is an exact zero, head not called."""
    import numpy as np

    rng = np.random.default_rng(0)
    pocket = rng.normal(size=(6, 343))
    ligand = rng.normal(size=(6, 343))
    pocket[2] = np.nan  # a failed pocket row
    ligand[[1, 4]] = np.nan  # edit rows: ligand unavailable
    side = _Side({"pocket": pocket, "ligand": ligand},
                 {"pocket": [1, 1, 0, 1, 1, 1], "ligand": [1, 0, 1, 1, 0, 1]})
    cfg = train_paired.MomentsConfig(weight=0.3, side=side.targets, kinds=(0, 6))  # edit rows enter here
    # batch rows 0, 1, 2, 4 (row ids; row 9 is not in the side file at all)
    data = _aux_batch(5, 22, series=[0] * 5, lig=[0, 1, 2, 3, 4], y=[0.0] * 5, kind=[0, 6, 0, 6, 0],
                      row=[0, 1, 2, 4, 9], parent=[0, 0, 2, 2, 9])
    wrapper = _full_wrapper(19)
    wrapper.train()
    z = wrapper.embed(data).detach().requires_grad_(True)
    weighted, parts = train_paired.moments_terms(wrapper.moments, z, data, cfg)
    with torch.no_grad():
        exp_p = torch.nn.functional.mse_loss(wrapper.moments["pocket"](z[[0, 1, 3]]),
                                             torch.as_tensor(pocket[[0, 1, 4]], dtype=torch.float32))
        exp_l = torch.nn.functional.mse_loss(wrapper.moments["ligand"](z[[0, 2]]),
                                             torch.as_tensor(ligand[[0, 2]], dtype=torch.float32))
    assert parts["pocket"].item() == pytest.approx(exp_p.item(), rel=1e-5)
    assert parts["ligand"].item() == pytest.approx(exp_l.item(), rel=1e-5)
    assert weighted.item() == pytest.approx(0.3 * (exp_p.item() + exp_l.item()), rel=1e-5)
    weighted.backward()
    assert torch.equal(z.grad[4], torch.zeros(16))  # row 9: no descriptor anywhere -> no gradient
    assert z.grad[0].abs().sum() > 0 and torch.isfinite(z.grad).all()
    # the ligand loss alone: row positions 1 (edit, ligand masked), 3 and 4 get no gradient from it
    z2 = z.detach().requires_grad_(True)
    train_paired.moments_terms(wrapper.moments, z2, data, cfg)[1]["ligand"].backward()
    assert all(torch.equal(z2.grad[k], torch.zeros(16)) for k in (1, 3, 4))
    # a batch whose rows have no descriptor: exact zeros, heads not called
    none = _aux_batch(2, 23, series=[0, 0], lig=[0, 1], y=[0.0, 0.0], kind=[0, 0], row=[2, 9], parent=[2, 9])
    side_p = _Side({"pocket": pocket}, {"pocket": [1, 1, 0, 1, 1, 1]})
    w0, p0 = train_paired.moments_terms(wrapper.moments, wrapper.embed(none),
                                        none, train_paired.MomentsConfig(weight=1.0, side=side_p.targets))
    assert w0.item() == 0.0 and p0["pocket"].item() == 0.0 and not w0.requires_grad
    # the default (--moments-kinds 0): the kind-6 rows (positions 1, 3) drop out of both losses
    cfg0 = train_paired.MomentsConfig(weight=0.3, side=side.targets)
    assert cfg0.kinds == (0,)
    z3 = z.detach().requires_grad_(True)
    w3, parts3 = train_paired.moments_terms(wrapper.moments, z3, data, cfg0)
    with torch.no_grad():
        exp_p0 = torch.nn.functional.mse_loss(wrapper.moments["pocket"](z[[0]]),
                                              torch.as_tensor(pocket[[0]], dtype=torch.float32))
    assert parts3["pocket"].item() == pytest.approx(exp_p0.item(), rel=1e-5)
    assert parts3["ligand"].item() == pytest.approx(exp_l.item(), rel=1e-5)  # rows 0, 2 are kind 0 already
    w3.backward()
    assert torch.equal(z3.grad[1], torch.zeros(16)) and torch.equal(z3.grad[3], torch.zeros(16))


# ----------------------------------------------------------------------------- moments as input (arm Z)

def _std_desc(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 343, generator=gen), torch.randn(n, 343, generator=gen)


def test_moments_input_matrix_blocks_and_cross():
    pocket, ligand = _std_desc(4)
    ones = torch.ones(4, dtype=torch.bool)
    dims = {m: heads.input_moments_dim(m) for m in heads.INPUT_MOMENTS_CHOICES}
    assert dims == {"none": 0, "pocket": 344, "ligand": 344, "both": 687, "bothx": 1030}
    assert heads.input_moments_targets("bothx") == ("pocket", "ligand") and heads.input_moments_targets("none") == ()
    m = heads.moments_input_matrix(pocket, ligand, ones, ones, "bothx")
    assert m.shape == (4, 1030) and m.dtype == torch.float32
    assert torch.equal(m[:, :343], pocket) and torch.equal(m[:, 343:686], ligand)
    assert torch.equal(m[:, -1], torch.zeros(4))
    # cross block: pocket channel c' faces ligand channel c with c' = c except pos <-> neg (2, 3), don <-> acc (5, 6)
    cross = m[:, 686:1029].view(4, 7, 49)
    p, lg = pocket.view(4, 7, 49), ligand.view(4, 7, 49)
    for c, c_pocket in enumerate((0, 1, 3, 2, 4, 6, 5)):
        assert torch.allclose(cross[:, c], p[:, c_pocket] * lg[:, c]), c
    assert torch.equal(heads.moments_input_matrix(pocket, None, ones, None, "pocket")[:, :343], pocket)
    assert torch.equal(heads.moments_input_matrix(None, ligand, None, ones, "ligand")[:, :343], ligand)
    assert torch.equal(heads.moments_input_matrix(pocket, ligand, ones, ones, "both"), m[:, list(range(686)) + [1029]])
    with pytest.raises(ValueError):
        heads.moments_input_matrix(pocket, None, ones, None, "both")
    with pytest.raises(ValueError):
        heads.moments_input_matrix(pocket, ligand, ones, ones, "none")


def test_input_moments_missing_flag():
    """A row without a descriptor (masked in the side file, NaN raw values, or not in the side file at all) gets zeros
    and the missing flag 1, and trains / predicts without NaN; the other rows are untouched."""
    import numpy as np

    rng = np.random.default_rng(1)
    pocket = rng.normal(size=(5, 343)).astype(np.float32)
    ligand = rng.normal(size=(5, 343)).astype(np.float32)
    pocket[2] = np.nan  # a failed pocket (the raw value is never read)
    ligand[3] = np.nan  # an edit row's ligand
    side = {"lookup": np.arange(5), "pocket_std": pocket, "ligand_std": ligand,
            "mask_pocket": np.array([1, 1, 0, 1, 1], bool), "mask_ligand": np.array([1, 1, 1, 0, 1], bool)}
    rows = torch.tensor([0, 2, 3, 9, 4])  # row 9 is not in the side file
    for mode in ("pocket", "ligand", "both", "bothx"):
        inputs = train_paired.InputMoments(side, mode, torch.device("cpu"))
        m = inputs.batch(rows)
        assert m.shape == (5, heads.input_moments_dim(mode)) and torch.isfinite(m).all(), mode
        missing = {"pocket": [0, 1, 0, 1, 0], "ligand": [0, 0, 1, 1, 0]}
        expect = [0, 0, 0, 1, 0]
        for t in heads.input_moments_targets(mode):
            expect = [max(a, b) for a, b in zip(expect, missing[t])]
        assert m[:, -1].tolist() == expect, mode
        blocks = heads.INPUT_MOMENTS_BLOCKS[mode]
        if "pocket" in blocks:
            assert torch.equal(m[1, :343], torch.zeros(343)) and torch.equal(m[3, :343], torch.zeros(343))
            assert torch.equal(m[0, :343], torch.as_tensor(pocket[0]))
        if "cross" in blocks:  # zero when either side is missing
            assert torch.equal(m[1:4, 686:1029], torch.zeros(3, 343)) and m[4, 686:1029].abs().sum() > 0
    # end to end: the wrapper trains and predicts on such a batch without NaN
    inputs = train_paired.InputMoments(side, "bothx", torch.device("cpu"))
    data = _batch(5, seed=31)
    data.row = rows
    inputs.attach(data)
    wrapper = heads.AffinityWithSiamese(_gnn(31, head=1), None, None, None,
                                        heads.MomentsInput(1030, proj=8, mode="bothx"))
    wrapper.train()
    out = wrapper(data)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
    wrapper.eval()
    assert torch.isfinite(wrapper(data)).all()


def _moments_batch(n, seed, mode="bothx"):
    data = _batch(n, seed=seed)
    p, lg = _std_desc(n, seed)
    ones = torch.ones(n, dtype=torch.bool)
    data.moments = heads.moments_input_matrix(p, lg, ones, ones, mode)
    return data


@pytest.mark.parametrize("proj", [32, 0], ids=["proj32", "raw"])
def test_wrapper_moments_input(proj):
    gnn = _gnn(21, head=1)
    dim = heads.embed_dim(gnn)
    plain_fc1 = gnn.fc1
    moments_in = heads.MomentsInput(687, proj=proj, mode="both")
    wrapper = heads.AffinityWithSiamese(gnn, heads.SiameseHead(dim, hidden=8), heads.PairHead(dim, hidden=8),
                                        {"pocket": heads.MomentsHead(dim, out=343, hidden=8)}, moments_in)
    assert wrapper.input_moments == "both" and wrapper.dim == dim == 16
    out_dim = proj or 687
    assert wrapper.affinity.fc1 is gnn.fc1 and gnn.fc1 is not plain_fc1
    assert gnn.fc1.in_features == dim + out_dim and gnn.fc1.out_features == plain_fc1.out_features
    sd = wrapper.state_dict()
    assert [k.split(".")[0] for k in sd][-1] == "moments_in" if proj else True
    assert {k.split(".")[0] for k in sd} == {"encoder", "affinity", "siamese", "pair", "moments"} | (
        {"moments_in"} if proj else set())
    assert heads.fc1_input_width(sd) == dim + out_dim
    assert heads.input_moments_from_state_dict(sd) == ((687, proj) if proj else None)
    data = _moments_batch(5, 22, "both")
    wrapper.eval()
    with torch.no_grad():
        z = wrapper.embed(data)
        out = wrapper(data)
        assert z.shape == (5, dim) and out.shape == (5, 1)
        assert torch.equal(out, wrapper.affinity_from_z(z, data))
        # the heads read z only; the moments change the pointwise prediction and nothing else
        other = data.clone()
        other.moments = data.moments.flip(0)
        assert torch.equal(wrapper.embed(other), z) and not torch.allclose(wrapper(other), out)
    with pytest.raises(RuntimeError):
        wrapper(_batch(5, seed=22))  # no data.moments
    with pytest.raises(RuntimeError):
        wrapper.to_plain_state_dict()
    with pytest.raises(KeyError):
        heads.wrapped_to_plain(sd) if proj else heads.wrapped_to_plain({**sd, "moments_in.fc.weight": torch.zeros(1)})
    with pytest.raises(RuntimeError):
        wrapper.from_plain_state_dict(_gnn(23, head=1).state_dict())
    # save -> rebuild -> load gives the same predictions
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    gnn2 = _gnn(24, head=1)
    again = heads.AffinityWithSiamese(gnn2, heads.SiameseHead(16, hidden=8), heads.PairHead(16, hidden=8),
                                      {"pocket": heads.MomentsHead(16, out=343, hidden=8)},
                                      heads.MomentsInput(687, proj=proj, mode="both"))
    again.load_state_dict(torch.load(buf, weights_only=True))
    again.eval()
    with torch.no_grad():
        assert torch.equal(again(data), out)
    with pytest.raises(ValueError):
        heads.MomentsInput(100, mode="both")
    with pytest.raises(ValueError):
        heads.AffinityWithSiamese(gnn, None, None, None, heads.MomentsInput(687, mode="both"))  # wrapped twice


def test_ligand_only_keeps_moments():
    """LigandOnly zeroes x[:, 15:] only: data.moments reaches the pointwise head unmasked, the caller's batch is
    untouched, and z is the masked graph's."""
    wrapper = heads.AffinityWithSiamese(_gnn(25, head=1), None, None, None,
                                        heads.MomentsInput(687, proj=16, mode="both")).eval()
    gate = train_paired.LigandOnly(wrapper)
    data = _moments_batch(4, 25, "both")
    x_before, m_before = data.x.clone(), data.moments.clone()
    masked = data.clone()
    masked.x = masked.x.clone()
    masked.x[:, 15:] = 0
    with torch.no_grad():
        assert torch.equal(gate(data), wrapper(masked))
        assert gate._masked(data).moments is data.moments  # the same tensor, never masked
        output, z = train_paired.forward_embed(gate, data)
        assert torch.equal(z, wrapper.embed(masked)) and torch.equal(output, wrapper(masked))
        other = data.clone()
        other.moments = data.moments * 0.5
        assert not torch.allclose(gate(other), gate(data))  # the moments are fed
    assert torch.equal(data.x, x_before) and torch.equal(data.moments, m_before)
    assert not torch.equal(gate._masked(data).x, data.x)


def test_moments_only_ignores_graph():
    """MomentsOnlyNet reads data.moments only: zeroing x / edge_attr or swapping the graphs leaves its predictions
    unchanged (train and eval mode)."""
    torch.manual_seed(26)
    net = heads.MomentsOnlyNet(1030, mode="bothx")
    assert [type(m).__name__ for m in net.children()] == ["Linear", "BatchNorm1d", "Linear", "BatchNorm1d", "Linear"]
    assert (net.fc1.in_features, net.fc1.out_features, net.fc2.out_features, net.out.out_features) == (1030, 512, 256, 1)
    data = _moments_batch(6, 26, "bothx")
    blank = data.clone()
    blank.x = torch.zeros_like(data.x)
    blank.edge_attr = torch.zeros_like(data.edge_attr)
    other = _batch(6, seed=99)
    other.moments = data.moments
    for train in (False, True):
        net.train(train)
        with torch.no_grad():
            ref = net(data)
            assert torch.equal(net(blank), ref) and torch.equal(net(other), ref)
    with pytest.raises(RuntimeError):
        net(_batch(6, seed=26))
    with pytest.raises(ValueError):
        heads.MomentsOnlyNet(100, mode="bothx")
    sd = net.state_dict()
    assert heads.moments_only_from_state_dict(sd) == (1030, (512, 256))
    assert heads.moments_only_from_state_dict(_gnn(1).state_dict()) is None


def test_input_moments_none_builds_as_before():
    """--input-moments none (the default) builds exactly the arm-S / arm-P models: same modules, same initial
    weights under the same seed, same state-dict bytes."""
    gen = torch.Generator().manual_seed(0)
    data = Batch.from_data_list([_graph(4, gen) for _ in range(3)])
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t", "--hidden-dim", "8", "--head", "1"]
    from utils import init_weights

    for extra in ([], ["--siamese", "--siamese-hidden", "8"]):
        args_none = train_paired.parse_args([*base, *extra, "--input-moments", "none"])
        args_default = train_paired.parse_args([*base, *extra])
        assert args_none.input_moments == "none" and args_none.input_moments_proj is None
        buffers = []
        for args in (args_none, args_default, None):
            torch.manual_seed(5)
            if args is not None:
                model = train_paired.build_model(data, args)
            else:  # the arm-S / arm-P construction
                cfg = SimpleNamespace(activation_function="leaky_relu", head=1, hidden_dim=8)
                model = GATv2Net(data.num_node_features, data.num_edge_features, cfg)
                model.apply(init_weights)
                if extra:
                    head = heads.SiameseHead(heads.embed_dim(model), hidden=8)
                    head.apply(init_weights)
                    model = heads.AffinityWithSiamese(model, head)
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            buffers.append(buf.getvalue())
        assert buffers[0] == buffers[1] == buffers[2], extra
