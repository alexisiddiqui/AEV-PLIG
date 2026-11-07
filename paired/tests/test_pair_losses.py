"""Unit tests for the within-series pair losses and the ligand-only wrapper (Task 2)."""
import math

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

import pair_losses
import train_paired

# Three members of one series, three different ligands.
PRED = torch.tensor([0.0, 1.5, 3.0])
Y = torch.tensor([0.0, 2.0, 1.0])
SERIES = torch.tensor([0, 0, 0])
LIG = torch.tensor([0, 1, 2])


def _huber(r, delta):
    r = abs(r)
    return 0.5 * r * r if r <= delta else delta * (r - 0.5 * delta)


def test_delta_huber_matches_manual():
    # pairs (0,1): r = (0-1.5) - (0-2) = 0.5   -> 0.125  (quadratic branch)
    #       (0,2): r = (0-3)   - (0-1) = -2    -> 1.5    (linear branch)
    #       (1,2): r = (1.5-3) - (2-1) = -2.5  -> 2.0
    expected = (0.125 + 1.5 + 2.0) / 3
    assert expected == pytest.approx(
        (_huber(0.5, 1.0) + _huber(-2.0, 1.0) + _huber(-2.5, 1.0)) / 3)
    got = pair_losses.delta_huber(PRED, Y, SERIES, LIG, tie=0.1, delta=1.0, weighting="none")
    assert got.ndim == 0
    assert got.item() == pytest.approx(expected, abs=1e-6)


def test_delta_huber_tie_excludes_pair():
    # |dy| = 2, 1, 1. tie 1.5 keeps only (0,1).
    got = pair_losses.delta_huber(PRED, Y, SERIES, LIG, tie=1.5, delta=1.0, weighting="none")
    assert got.item() == pytest.approx(0.125, abs=1e-6)
    # |dy| >= tie keeps the boundary pairs.
    got = pair_losses.delta_huber(PRED, Y, SERIES, LIG, tie=1.0, delta=1.0, weighting="none")
    assert got.item() == pytest.approx((0.125 + 1.5 + 2.0) / 3, abs=1e-6)
    # Everything excluded -> 0.
    got = pair_losses.delta_huber(PRED, Y, SERIES, LIG, tie=5.0, delta=1.0, weighting="none")
    assert got.item() == 0.0


def test_delta_huber_no_pairs_is_zero():
    pred = torch.tensor([0.3, -1.0, 2.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0, -2.0])
    # (a) every item in its own series.
    got = pair_losses.delta_huber(pred, y, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
                                  tie=0.0, delta=1.0, weighting="absdelta")
    assert got.item() == 0.0 and math.isfinite(got.item())
    # (b) one series whose members all share one ligand, plus a singleton.
    got = pair_losses.delta_huber(pred, y, torch.tensor([0, 0, 1]), torch.tensor([5, 5, 6]),
                                  tie=0.0, delta=1.0, weighting="none")
    assert got.item() == 0.0 and math.isfinite(got.item())
    got = pair_losses.listwise(pred, y, torch.tensor([0, 0, 1]), torch.tensor([5, 5, 6]),
                               tie=0.0, weighting="none")
    assert got.item() == 0.0 and math.isfinite(got.item())

    # (c) the assembled training loss equals the MSE when the batch has no pairs.
    data = Batch.from_data_list([
        Data(y=torch.tensor([float(v)]), series=torch.tensor([s]), lig=torch.tensor([l]))
        for v, s, l in zip(y.tolist(), [0, 0, 1], [5, 5, 6])])
    for kind in ("delta_huber", "listwise"):
        cfg = train_paired.LossConfig(kind=kind, weight=0.3, tie=0.0, delta=1.0, weighting="none")
        total, mse, pair, decoy, hinge = train_paired.compute_loss(pred.view(-1, 1), data, cfg)
        assert pair.item() == 0.0 and decoy.item() == 0.0  # no `kind` field: every row is kind 0
        assert hinge.item() == 0.0
        assert total.item() == pytest.approx(F.mse_loss(pred, y).item(), abs=1e-7)
        assert total.item() == pytest.approx(mse.item(), abs=1e-7)
        total.backward()  # a zero pair term must not break backward
        pred.grad = None


def test_absdelta_weighting():
    # Same pairs as the manual test, each term multiplied by |dy| = 2, 1, 1; plain mean.
    expected = (2 * 0.125 + 1 * 1.5 + 1 * 2.0) / 3
    got = pair_losses.delta_huber(PRED, Y, SERIES, LIG, tie=0.1, delta=1.0, weighting="absdelta")
    assert got.item() == pytest.approx(expected, abs=1e-6)


def _listnet(ys, ps):
    qz = sum(math.exp(v) for v in ys)
    q = [math.exp(v) / qz for v in ys]
    lse = math.log(sum(math.exp(v) for v in ps))
    return -sum(qi * (pi - lse) for qi, pi in zip(q, ps))


def test_listwise_matches_manual():
    y = torch.tensor([1.0, 0.0, 5.0, 2.0, 0.0, 3.0])
    p = torch.tensor([0.5, -0.5, 9.0, 0.0, 1.0, 7.0])
    series = torch.tensor([0, 0, 0, 1, 1, 2])
    lig = torch.tensor([0, 1, 1, 2, 3, 4])
    # series 0 keeps items 0, 1 (item 2 repeats ligand 1); series 1 keeps 3, 4; series 2 is a singleton.
    l0 = _listnet([1.0, 0.0], [0.5, -0.5])
    l1 = _listnet([2.0, 0.0], [0.0, 1.0])
    assert l0 == pytest.approx(0.582206, abs=1e-5)
    assert l1 == pytest.approx(1.194056, abs=1e-5)
    got = pair_losses.listwise(p, y, series, lig, tie=0.0, weighting="none")
    assert got.item() == pytest.approx((l0 + l1) / 2, abs=1e-5)
    # tie 1.5 drops series 0 (range 1.0) and keeps series 1 (range 2.0).
    got = pair_losses.listwise(p, y, series, lig, tie=1.5, weighting="none")
    assert got.item() == pytest.approx(l1, abs=1e-5)
    # the loss is differentiable in the predictions.
    p = p.clone().requires_grad_(True)
    pair_losses.listwise(p, y, series, lig, tie=0.0, weighting="none").backward()
    assert p.grad is not None and p.grad[2].item() == 0.0 and p.grad[5].item() == 0.0


class _AevSum(torch.nn.Module):
    def forward(self, data):
        return data.x[:, 15:].abs().sum()


class _RdkitSum(torch.nn.Module):
    def forward(self, data):
        return data.x[:, :15].sum()


def test_ligand_only_zeroes_aev():
    torch.manual_seed(0)
    x = torch.randn(7, 367)
    data = Data(x=x.clone(), edge_index=torch.zeros(2, 0, dtype=torch.long))
    assert _AevSum()(data).item() > 0
    assert train_paired.LigandOnly(_AevSum())(data).item() == 0.0
    # RDKit columns 0:15 pass through unchanged, and the caller's data is not modified.
    assert train_paired.LigandOnly(_RdkitSum())(data).item() == pytest.approx(x[:, :15].sum().item())
    assert torch.equal(data.x, x)



# ----------------------------------------------------------------------------- decoy hinge (arm 4)

def test_decoy_hinge_manual():
    pred = torch.tensor([1.0, 0.2, -0.5, 3.0, 0.9])
    y = torch.tensor([1.0, 1.0, 0.0, 2.0, 1.0])
    kind = torch.tensor([0, 2, 2, 1, 2])
    margin = 0.5
    # decoys 1, 2, 4: relu(pred - (y - margin)) = relu(0.2 - 0.5) = 0, relu(-0.5 + 0.5) = 0,
    # relu(0.9 - 0.5) = 0.4; mean over the three decoy rows; rows of kind 0 / 1 never enter.
    got = pair_losses.decoy_hinge(pred, y, kind, margin)
    assert got.ndim == 0
    assert got.item() == pytest.approx(0.4 / 3, abs=1e-6)
    # margin 0: relu(0.2 - 1) = 0, relu(-0.5 - 0) = 0, relu(0.9 - 1) = 0
    assert pair_losses.decoy_hinge(pred, y, kind, 0.0).item() == pytest.approx(0.0, abs=1e-7)
    # margin 1.5: relu(0.7) = 0.7, relu(1.0) = 1.0, relu(1.4) = 1.4
    assert pair_losses.decoy_hinge(pred, y, kind, 1.5).item() == pytest.approx(3.1 / 3, abs=1e-6)
    # gradient flows to the decoys above the threshold only
    p = pred.clone().requires_grad_(True)
    pair_losses.decoy_hinge(p, y, kind, margin).backward()
    assert p.grad.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0 / 3])
    # no decoy rows -> exactly 0
    assert pair_losses.decoy_hinge(pred, y, torch.tensor([0, 1, 0, 1, 0]), margin).item() == 0.0


def _batch(y, kind, series=None, lig=None):
    n = len(y)
    series = series or list(range(n))
    lig = lig or list(range(n))
    return Batch.from_data_list([
        Data(y=torch.tensor([float(v)]), kind=torch.tensor([k]), series=torch.tensor([s]),
             lig=torch.tensor([l])) for v, k, s, l in zip(y, kind, series, lig)])


def test_loss_mse_excludes_decoys():
    pred = torch.tensor([0.5, 2.0, 1.0, 3.0], requires_grad=True)
    y = [1.0, 1.0, 0.0, 1.0]
    data = _batch(y, [0, 1, 2, 2])
    cfg = train_paired.LossConfig(kind="none", weight=0.3, tie=0.0, delta=1.0, weighting="none",
                                  decoy_weight=2.0, decoy_margin=0.5)
    total, mse, pair, decoy, hinge = train_paired.compute_loss(pred.view(-1, 1), data, cfg)
    assert mse.item() == pytest.approx((0.25 + 1.0) / 2)  # rows 0 and 1 only
    assert pair.item() == 0.0
    assert decoy.item() == pytest.approx((1.5 + 2.5) / 2)  # relu(1 - (0 - .5)), relu(3 - (1 - .5))
    assert hinge.item() == pytest.approx(2.0 * 2.0)  # weighted term (decoy_weight 2.0)
    assert total.item() == pytest.approx(0.625 + 2.0 * 2.0)
    total.backward()
    assert pred.grad.tolist() == pytest.approx([-0.5, 1.0, 1.0, 1.0])


def test_loss_decoy_only_batch():
    pred = torch.tensor([0.5, -2.0, 1.0], requires_grad=True)
    data = _batch([1.0, 0.0, 0.0], [2, 2, 2], series=[0, 0, 0], lig=[4, 4, 4])
    for kind in ("none", "delta_huber"):
        cfg = train_paired.LossConfig(kind=kind, weight=0.3, tie=0.0, delta=1.0, weighting="none",
                                      decoy_weight=1.0, decoy_margin=0.75)
        total, mse, pair, decoy, hinge = train_paired.compute_loss(pred.view(-1, 1), data, cfg)
        for t in (total, mse, pair, decoy, hinge):
            assert math.isfinite(t.item())
        assert mse.item() == 0.0 and pair.item() == 0.0
        # relu(0.5 - 0.25) = 0.25, relu(-2 + 0.75) = 0, relu(1 + 0.75) = 1.75
        assert decoy.item() == pytest.approx(2.0 / 3)
        assert total.item() == pytest.approx(decoy.item())
        total.backward()
        assert pred.grad.tolist() == pytest.approx([1 / 3, 0.0, 1 / 3])
        pred.grad = None
    # every decoy below its threshold: the total is 0 and backward still works
    low = torch.tensor([-5.0, -5.0], requires_grad=True)
    cfg = train_paired.LossConfig(kind="none", weight=0.0, tie=0.0, delta=1.0, weighting="none",
                                  decoy_weight=1.0, decoy_margin=0.75)
    total, *_ = train_paired.compute_loss(low.view(-1, 1), _batch([1.0, 1.0], [2, 2]), cfg)
    assert total.item() == 0.0
    total.backward()
    assert low.grad.tolist() == [0.0, 0.0]


# ----------------------------------------------------------------------------- per-kind hinge (kinds 2, 4, 6)

def test_hinge_by_kind_margins():
    pred = torch.tensor([1.0, 0.2, 0.9, 0.5, 2.0, 1.5, 3.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0])
    kind = torch.tensor([0, 2, 4, 6, 6, 3, 5])
    margins = {2: 0.5, 4: 1.0, 6: 0.25}
    # kind 2: relu(0.2 - 0.5) = 0; kind 4: relu(0.9 - 0) = 0.9; kind 6: relu(0.5 - 0.75) = 0,
    # relu(2.0 - 1.75) = 0.25; kinds 0, 3, 5 never enter; mean over the four hinge rows
    got = pair_losses.hinge_by_kind(pred, y, kind, margins)
    assert got.ndim == 0
    assert got.item() == pytest.approx((0.0 + 0.9 + 0.0 + 0.25) / 4, abs=1e-6)
    # only the kinds named in margins are hinge rows
    assert pair_losses.hinge_by_kind(pred, y, kind, {4: 1.0}).item() == pytest.approx(0.9, abs=1e-6)
    assert pair_losses.hinge_by_kind(pred, y, kind, {6: 0.25}).item() == pytest.approx(0.125, abs=1e-6)
    # with margins {2: m} it is arm 4's decoy hinge
    for m in (0.0, 0.5, 1.5):
        assert pair_losses.hinge_by_kind(pred, y, kind, {2: m}).item() == pytest.approx(
            pair_losses.decoy_hinge(pred, y, kind, m).item(), abs=1e-7)
    # weights: one common weight scales the mean; per-kind weights weight each row
    assert pair_losses.hinge_by_kind(pred, y, kind, margins, {2: 3.0, 4: 3.0, 6: 3.0}).item() == \
        pytest.approx(3.0 * got.item(), abs=1e-6)
    assert pair_losses.hinge_by_kind(pred, y, kind, margins, {2: 1.0, 4: 2.0, 6: 4.0}).item() == \
        pytest.approx((0.0 + 2.0 * 0.9 + 0.0 + 4.0 * 0.25) / 4, abs=1e-6)
    # no hinge rows -> exactly 0; gradient only through rows above their threshold
    assert pair_losses.hinge_by_kind(pred, y, torch.tensor([0, 1, 3, 5, 7, 0, 1]), margins).item() == 0.0
    p = pred.clone().requires_grad_(True)
    pair_losses.hinge_by_kind(p, y, kind, margins).backward()
    assert p.grad.tolist() == pytest.approx([0.0, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0])


def test_loss_roles_for_kinds_3_to_7():
    """MSE over kinds 0, 3, 5, 7 (and 1); hinge over 4, 6 with per-kind margins; the pair term never
    pairs a hinge row (its label is the parent's)."""
    pred = torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 1.0, 5.0], requires_grad=True)
    y = [1.5, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0]
    kind = [0, 3, 4, 5, 6, 7, 0]
    # rows 0 and 2 (a twin) are in series 0 with different ligands; rows 0 and 6 are a real pair
    series = [0, 1, 0, 2, 3, 4, 0]
    lig = [0, 1, 9, 3, 4, 5, 6]
    data = _batch(y, kind, series=series, lig=lig)
    cfg = train_paired.LossConfig(kind="delta_huber", weight=1.0, tie=0.0, delta=10.0, weighting="none",
                                  hinge_margins={4: 1.0, 6: 0.5}, hinge_weights={4: 1.0, 6: 1.0})
    total, mse, pair, decoy, hinge = train_paired.compute_loss(pred.view(-1, 1), data, cfg)
    mse_rows = [0, 1, 3, 5, 6]
    assert mse.item() == pytest.approx(sum((pred[i].item() - y[i]) ** 2 for i in mse_rows) / 5)
    # kind 4: relu(0 - (1 - 1)) = 0; kind 6: relu(4 - (2 - 0.5)) = 2.5
    assert hinge.item() == pytest.approx(2.5 / 2) and decoy.item() == 0.0
    # only pair (0, 6): r = (1 - 5) - (1.5 - 1) = -4.5 -> 0.5 * 4.5^2 (quadratic branch, delta 10)
    assert pair.item() == pytest.approx(0.5 * 4.5 ** 2)
    assert total.item() == pytest.approx(mse.item() + pair.item() + hinge.item())
    total.backward()
    assert pred.grad[2].item() == 0.0  # a twin below its threshold gets no gradient at all


# ----------------------------------------------------------------------------- Siamese head loss (arm S)

def _softplus(x):
    return math.log1p(math.exp(x))


def test_siamese_loss_manual():
    y = torch.tensor([0.0, 1.0, 0.2])
    i, j = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    # dy = y_i - y_j = -1.0, -0.2, 0.8
    delta_hat = torch.tensor([-0.5, 0.4, 2.5], requires_grad=True)
    # Huber (delta 1) of delta_hat - dy = 0.5, 0.6, 1.7 -> 0.125, 0.18, 1.2 (linear branch)
    huber_expected = (0.125 + 0.18 + 1.2) / 3
    # BCE on |dy| >= 0.5: pair (0,1) target 0 (y_0 < y_1), logit -0.5 / 0.5 = -1 -> softplus(-1);
    #                     pair (1,2) target 1, logit 2.5 / 0.5 = 5 -> softplus(-5); pair (0,2) excluded
    bce_expected = (_softplus(-1.0) + _softplus(-5.0)) / 2
    huber, bce = pair_losses.siamese_loss(delta_hat, y, i, j, tie_std=0.5, tau_std=0.5, huber_delta=1.0)
    assert huber.ndim == 0 and bce.ndim == 0
    assert huber.item() == pytest.approx(huber_expected, abs=1e-6)
    assert bce.item() == pytest.approx(bce_expected, abs=1e-6)
    (huber + bce).backward()
    # d huber / d delta_hat = clip(resid, -1, 1) / 3; d bce / d delta_hat = (sigmoid(l) - t) / tau / 2
    sig = lambda v: 1 / (1 + math.exp(-v))
    expected_grad = [0.5 / 3 + (sig(-1.0) - 0) / 0.5 / 2, 0.6 / 3, 1.0 / 3 + (sig(5.0) - 1) / 0.5 / 2]
    assert delta_hat.grad.tolist() == pytest.approx(expected_grad, abs=1e-6)
    # Huber delta in standardised units: delta 2 keeps every residual quadratic
    huber2, _ = pair_losses.siamese_loss(delta_hat.detach(), y, i, j, 0.5, 0.5, huber_delta=2.0)
    assert huber2.item() == pytest.approx((0.125 + 0.18 + 0.5 * 1.7 ** 2) / 3, abs=1e-6)
    # a tie above every |dy|: the BCE is an exact zero, the Huber term unchanged
    h3, b3 = pair_losses.siamese_loss(delta_hat.detach(), y, i, j, tie_std=5.0, tau_std=0.5)
    assert b3.item() == 0.0 and h3.item() == pytest.approx(huber_expected, abs=1e-6)
    # no pairs: two exact zeros without a graph
    empty = torch.zeros(0, dtype=torch.long)
    h0, b0 = pair_losses.siamese_loss(torch.zeros(0, requires_grad=True), y, empty, empty, 0.5, 0.5)
    assert (h0.item(), b0.item()) == (0.0, 0.0) and not h0.requires_grad and not b0.requires_grad
    with pytest.raises(ValueError):
        pair_losses.siamese_loss(delta_hat.detach()[:2], y, i, j, 0.5, 0.5)
    with pytest.raises(ValueError):
        pair_losses.siamese_loss(delta_hat.detach(), y, i, j, 0.5, 0.0)


# ----------------------------------------------------------------------------- pair head listwise, moments MSE (arm P)

def _t(values, dtype=torch.long):
    return torch.tensor(values, dtype=dtype)


def _np_log_softmax(x):
    x = [float(v) for v in x]
    m = max(x)
    lse = m + math.log(sum(math.exp(v - m) for v in x))
    return [v - lse for v in x]


def test_listwise_pairhead_manual():
    """Three ligands of one series and one shape twin (kind 4) of the first: every crystal is an anchor with two
    candidates; the twin joins anchor 0's list only, at target 0."""
    series, lig = _t([0, 0, 0, 0]), _t([0, 1, 2, 0])
    kind, row, parent = _t([0, 0, 0, 4]), _t([100, 101, 102, 103]), _t([100, 101, 102, 100])
    a, c, neg = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4,))
    assert list(zip(a.tolist(), c.tolist(), neg.tolist())) == [
        (0, 1, False), (0, 2, False), (0, 3, True), (1, 0, False), (1, 2, False), (2, 0, False), (2, 1, False)]
    y = torch.tensor([0.0, 1.0, 2.0, 0.0])
    scores = torch.tensor([0.3, -0.2, 1.1, 0.5, 0.0, -0.7, 0.4], requires_grad=True)
    tau = 0.5
    loss = pair_losses.listwise_pairhead(scores, y, a, c, neg, tau)
    s = scores.detach().tolist()
    expected = []
    for (lo, hi), ys in (((0, 3), [1.0, 2.0]), ((3, 5), [0.0, 2.0]), ((5, 7), [0.0, 1.0])):
        q = [math.exp(v) for v in _np_log_softmax([yy / tau for yy in ys])]
        q = q + [0.0] * (hi - lo - len(q))  # the negative of anchor 0: target 0
        logp = _np_log_softmax(s[lo:hi])
        expected.append(-sum(qq * lp for qq, lp in zip(q, logp)))
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(sum(expected) / 3, abs=1e-6)
    loss.backward()
    # the negative's score only enters the softmax: its gradient is its softmax probability / n_anchors (> 0)
    p_neg = math.exp(_np_log_softmax(s[0:3])[2])
    assert scores.grad[2].item() == pytest.approx(p_neg / 3, abs=1e-6)
    # without --pair-neg-kinds the twin is in no list
    a2, c2, neg2 = pair_losses.pairhead_candidates(series, lig, kind, row, parent)
    assert 3 not in c2.tolist() and not neg2.any() and len(a2) == 6
    # tau is a temperature on y: a larger tau flattens the targets
    flat = pair_losses.listwise_pairhead(scores.detach(), y, a, c, neg, 100.0)
    assert flat.item() != pytest.approx(loss.item())
    with pytest.raises(ValueError):
        pair_losses.listwise_pairhead(scores.detach(), y, a, c, neg, 0.0)
    with pytest.raises(ValueError):
        pair_losses.listwise_pairhead(scores.detach()[:3], y, a, c, neg, tau)
    with pytest.raises(ValueError):
        pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(3,))


def test_listwise_skips_anchor_without_candidates():
    """Replicates (one ligand key), a two-ligand series (one candidate each) and a three-ligand series: only the
    last series' rows are anchors; the skipped anchor's negative is dropped with it; a batch without any anchor
    gives an exact zero, never NaN."""
    #            replicates  two ligands  three ligands  twin of row 3 (skipped anchor)
    series = _t([0, 0, 0,    1, 1,        2, 2, 2,       1])
    lig = _t([5, 5, 5,       1, 2,        3, 4, 6,       1])
    kind = _t([0] * 8 + [4])
    row = torch.arange(200, 209)
    parent = _t([200, 201, 202, 203, 204, 205, 206, 207, 203])
    a, c, neg = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4,))
    assert set(a.tolist()) == {5, 6, 7} and set(c.tolist()) == {5, 6, 7} and not neg.any()
    y = torch.tensor([1.0, 2.0, 3.0, 0.0, 1.0, 0.5, 1.5, -0.5, 0.0])
    scores = torch.randn(len(a), generator=torch.Generator().manual_seed(0))
    loss = pair_losses.listwise_pairhead(scores, y, a, c, neg, 0.7)
    sub = slice(5, 8)
    a_s, c_s, n_s = pair_losses.pairhead_candidates(series[sub], lig[sub], kind[sub], row[sub], parent[sub])
    alone = pair_losses.listwise_pairhead(scores, y[sub], a_s, c_s, n_s, 0.7)
    assert torch.isfinite(loss) and loss.item() == pytest.approx(alone.item(), abs=1e-6)
    # no anchor at all: empty lists and an exact zero without a graph
    a0, c0, n0 = pair_losses.pairhead_candidates(series[:5], lig[:5], kind[:5], row[:5], parent[:5])
    assert a0.numel() == c0.numel() == n0.numel() == 0
    zero = pair_losses.listwise_pairhead(torch.zeros(0, requires_grad=True), y[:5], a0, c0, n0, 0.7)
    assert zero.item() == 0.0 and not zero.requires_grad
    # min_candidates 1 would admit the two-ligand series: the default (2) is the plan's rule
    a1, _, _ = pair_losses.pairhead_candidates(series, lig, kind, row, parent, min_candidates=1)
    assert {3, 4} <= set(a1.tolist())


def test_negatives_attach_to_own_anchor():
    """Two anchors of one series with a negative each (parent = the anchor's row id, not its batch position), a
    negative whose parent is not in the batch, and a contact edit (kind 6) not selected: each negative is in its
    own anchor's list only."""
    #               A    B    C    twinA twinB orphan edit(A)  A-positive (kind 3, MSE, A's ligand)
    series = _t([7, 7, 7, 7, 7, 7, 7, 7])
    lig = _t([0, 1, 2, 0, 1, 9, 0, 0])
    kind = _t([0, 0, 0, 4, 4, 4, 6, 3])
    row = _t([50, 51, 52, 53, 54, 55, 56, 57])
    parent = _t([50, 51, 52, 50, 51, 99, 50, 50])
    a, c, neg = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4,))
    pairs = set(zip(a.tolist(), c.tolist()))
    negs = {(x, y) for x, y, n in zip(a.tolist(), c.tolist(), neg.tolist()) if n}
    assert negs == {(0, 3), (1, 4)}
    assert all(cc not in (3, 4, 5, 6) for aa, cc in pairs - negs)
    assert (0, 4) not in pairs and (1, 3) not in pairs and all(cc != 5 for _, cc in pairs)
    assert all(cc != 6 for _, cc in pairs)  # kind 6 not in neg_kinds
    # the A-positive (kind 3, an MSE kind) is neither an anchor nor a positive candidate: kind-0 rows only
    assert 7 not in a.tolist() and 7 not in c.tolist()
    # with kind 6 selected, the edit joins A's list only
    _, c6, n6 = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4, 6))
    a6 = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4, 6))[0]
    assert {(x, y) for x, y, n in zip(a6.tolist(), c6.tolist(), n6.tolist()) if n} == {(0, 3), (1, 4), (0, 6)}


def test_pairhead_kind0_and_replicates():
    """Only kind-0 rows anchor or are positive candidates (kinds 1, 3, 5, 7 are MSE rows but never in a list), and
    replicate kind-0 rows (same series and ligand) count once: the first in batch order is kept, as in listwise."""
    #             a    b    b'   c    pose(a) Apos(b) Cpos(c) Cctl(c) twin(b')
    series = _t([3, 3, 3, 3, 3, 3, 3, 3, 3])
    lig = _t([0, 1, 1, 2, 0, 1, 2, 2, 1])
    kind = _t([0, 0, 0, 0, 1, 3, 5, 7, 4])
    row = _t([10, 11, 12, 13, 14, 15, 16, 17, 18])
    parent = _t([10, 11, 12, 13, 10, 11, 13, 13, 12])
    a, c, neg = pair_losses.pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(4,))
    assert set(a.tolist()) == {0, 1, 3} and set(c.tolist()) == {0, 1, 3}  # b' (position 2) is a replicate of b
    assert not neg.any()  # the twin's parent is b' (row 12), which is not an anchor: dropped with it
    assert sorted(zip(a.tolist(), c.tolist())) == [(0, 1), (0, 3), (1, 0), (1, 3), (3, 0), (3, 1)]
    # distinct ligands decide the count: a with b, b' and nothing else has one candidate -> no anchor
    a2, _, _ = pair_losses.pairhead_candidates(series[:3], lig[:3], kind[:3], row[:3], parent[:3])
    assert a2.numel() == 0
    # batch order decides the representative: b' first -> b' anchors and takes the twin
    order = torch.tensor([2, 1, 0, 3, 8])
    a3, c3, n3 = pair_losses.pairhead_candidates(series[order], lig[order], kind[order], row[order], parent[order],
                                                 neg_kinds=(4,))
    assert set(a3.tolist()) == {0, 2, 3} and {(x, y) for x, y, n in zip(a3.tolist(), c3.tolist(), n3.tolist()) if n} == {(0, 4)}


def test_masked_mse():
    pred = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 2.0], [float("nan"), float("nan")], [1.0, 1.0]])
    mask = torch.tensor([True, False, True])
    loss = pair_losses.masked_mse(pred, target, mask)
    assert loss.item() == pytest.approx((1.0 + 0.0 + 4.0 + 4.0) / 4)
    loss.backward()
    assert torch.equal(pred.grad[1], torch.zeros(2))  # the masked row: no gradient, its NaN target ignored
    none = pair_losses.masked_mse(pred, target, torch.zeros(3, dtype=torch.bool))
    assert none.item() == 0.0 and not none.requires_grad
    with pytest.raises(ValueError):
        pair_losses.masked_mse(pred, target[:2], mask)
