"""Within-series pair losses for paired AEV-PLIG training.

All inputs are in standardised label units: `tie` and `delta` must already be divided
by the label scaler's scale (train_paired.py does this). Pairs are formed inside a batch
between members of the same series that carry different ligands (paired_data.pair_index).
Both losses return a 0-d tensor, exactly 0.0 when the batch holds no usable pair or series.

decoy_hinge (arm 4) is the one-sided term for decoy poses (kind 2): their label is the
parent crystal's pK, used only as the reference the prediction must stay `margin` below.
hinge_by_kind generalises it to every negative kind (2 decoy pose, 4 shape twin, 6 contact edit)
with a margin per kind; the label `y` of those rows is always the parent's (build_paired_dataset
copies it), never the twin's or the edited ligand's own.

siamese_loss (arm S) scores the Siamese difference head's delta_hat on given pairs (i, j): a Huber term on
delta_hat - (y_i - y_j) over every pair and a direction BCE on delta_hat / tau over the pairs with
|y_i - y_j| >= tie.

Pair head (arm P, GQT's within listwise): pairhead_candidates lists, per anchor a, its candidates c and negatives.
Anchors and positive candidates are kind-0 rows only, one per distinct (series, lig) of the batch (the first in batch
order, as listwise keeps one member per ligand); c is a candidate of a when it has the same series and a different
ligand id; anchors with fewer than min_candidates candidates (distinct ligands) are dropped. The anchor's own
negatives are the rows of the chosen negative kinds whose parent is the anchor's row. listwise_pairhead is
L_a = -sum_c q_c log softmax_c l(z_a, z_c) over a's candidate list with q = t / sum t, t_c = softmax(y_c / tau) over
the positive candidates and t = 0 for negatives (they enter the softmax only), mean over anchors. masked_mse is the
moments heads' loss (rows outside the mask ignored).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import torch
import torch.nn.functional as F

from paired_data import pair_index

WEIGHTINGS = ("none", "absdelta")
DECOY_KIND = 2
HINGE_KINDS = (2, 4, 6)        # decoy pose, shape twin (A), contact edit (C): one-sided hinge
MSE_KINDS = (0, 1, 3, 5, 7)    # original, near-native, A-positive, C-positive, C-control: MSE


def delta_huber(pred, y, series, lig, tie, delta, weighting):
    """Mean Huber((p_i - p_j) - (y_i - y_j)) over same-series, different-ligand pairs
    with |y_i - y_j| >= tie; each term times |y_i - y_j| when weighting == "absdelta"."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}")
    pred, y = pred.reshape(-1), y.reshape(-1)
    i, j = pair_index(series.reshape(-1), lig.reshape(-1))
    i, j = i.to(pred.device), j.to(pred.device)
    dy = y[i] - y[j]
    keep = dy.abs() >= tie
    if not bool(keep.any()):
        return pred.new_zeros(())
    i, j, dy = i[keep], j[keep], dy[keep]
    resid = (pred[i] - pred[j]) - dy
    terms = F.huber_loss(resid, torch.zeros_like(resid), delta=delta, reduction="none")
    if weighting == "absdelta":
        terms = terms * dy.abs()
    return terms.mean()


def _first_occurrence(series, lig):
    """Boolean mask of the first item of each distinct (series, lig) in batch order."""
    n = series.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=series.device)
    _, inv = torch.unique(torch.stack([series, lig], dim=1), dim=0, return_inverse=True)
    pos = torch.arange(n, device=series.device)
    first = torch.full((int(inv.max()) + 1,), n, dtype=torch.long, device=series.device)
    first = first.scatter_reduce(0, inv, pos, reduce="amin")
    mask = torch.zeros(n, dtype=torch.bool, device=series.device)
    mask[first] = True
    return mask


def _segment_log_softmax(x, seg, n_seg):
    """log_softmax of x within each segment id in seg (0..n_seg-1)."""
    shift = x.new_full((n_seg,), float("-inf")).scatter_reduce(
        0, seg, x.detach(), reduce="amax", include_self=True)
    z = x - shift[seg]
    lse = x.new_zeros(n_seg).index_add(0, seg, z.exp()).log()
    return z - lse[seg]


def listwise(pred, y, series, lig, tie, weighting):
    """ListNet top-one loss per series, mean over series.

    For each series with >= 2 distinct ligands in the batch, one member per ligand (the
    first in batch order) is kept; loss = -(softmax(y) * log_softmax(pred)).sum() over
    those members. Series whose kept labels span less than `tie` are skipped.
    """
    if weighting != "none":
        raise ValueError("listwise supports weighting='none' only")
    pred, y = pred.reshape(-1), y.reshape(-1)
    series, lig = series.reshape(-1).to(pred.device), lig.reshape(-1).to(pred.device)
    keep = _first_occurrence(series, lig)
    if int(keep.sum()) < 2:
        return pred.new_zeros(())
    p, t, s = pred[keep], y[keep], series[keep]
    _, seg = torch.unique(s, return_inverse=True)
    n_seg = int(seg.max()) + 1

    count = torch.bincount(seg, minlength=n_seg)
    y_max = t.new_full((n_seg,), float("-inf")).scatter_reduce(0, seg, t, reduce="amax")
    y_min = t.new_full((n_seg,), float("inf")).scatter_reduce(0, seg, t, reduce="amin")
    use = (count >= 2) & ((y_max - y_min) >= tie)
    if not bool(use.any()):
        return pred.new_zeros(())

    q = _segment_log_softmax(t, seg, n_seg).exp()
    log_p = _segment_log_softmax(p, seg, n_seg)
    per_series = p.new_zeros(n_seg).index_add(0, seg, -(q * log_p))
    return per_series[use].mean()


def decoy_hinge(pred, y, kind, margin):
    """Mean over decoy rows (kind == 2) of relu(pred - (y - margin)), standardised units;
    exactly 0.0 when the batch holds no decoy."""
    pred, y = pred.reshape(-1), y.reshape(-1)
    decoy = kind.reshape(-1).to(pred.device) == DECOY_KIND
    if not bool(decoy.any()):
        return pred.new_zeros(())
    return F.relu(pred[decoy] - (y[decoy] - margin)).mean()


def hinge_by_kind(pred, y, kind, margins, weights=None):
    """Mean over rows whose kind is a key of `margins` of
    weights[kind] * relu(pred - (y - margins[kind])), standardised units (weights default 1);
    exactly 0.0 when the batch holds no such row. With margins {2: m} and no weights this is
    decoy_hinge(pred, y, kind, m); with one common weight w it is w * that mean."""
    pred, y = pred.reshape(-1), y.reshape(-1)
    kind = kind.reshape(-1).to(pred.device)
    sel = torch.zeros_like(kind, dtype=torch.bool)
    margin = pred.new_zeros(pred.shape)
    weight = pred.new_ones(pred.shape)
    for k, m in margins.items():
        rows = kind == int(k)
        sel |= rows
        margin = margin.masked_fill(rows, float(m))
        if weights is not None:
            weight = weight.masked_fill(rows, float(weights[k]))
    if not bool(sel.any()):
        return pred.new_zeros(())
    terms = F.relu(pred[sel] - (y[sel] - margin[sel]))
    common = {1.0} if weights is None else {float(weights[k]) for k in margins}
    if len(common) == 1:
        return common.pop() * terms.mean()
    return (weight[sel] * terms).mean()


def siamese_loss(delta_hat, y, i, j, tie_std, tau_std, huber_delta=1.0):
    """(huber_term, bce_term) for the Siamese head's delta_hat [n_pairs] on the pairs (i[k], j[k]) of y.

    huber_term = mean over all pairs of Huber(delta_hat - (y_i - y_j), huber_delta);
    bce_term   = mean over pairs with |y_i - y_j| >= tie_std of
                 BCE-with-logits(delta_hat / tau_std, 1[y_i > y_j]).
    Standardised label units throughout (tie_std, tau_std, huber_delta already divided by the scaler's
    scale). Each term is an exact 0.0 tensor (no graph) when it has no pair."""
    delta_hat, y = delta_hat.reshape(-1), y.reshape(-1)
    i, j = i.reshape(-1).to(y.device), j.reshape(-1).to(y.device)
    if delta_hat.numel() != i.numel() or i.numel() != j.numel():
        raise ValueError(f"siamese_loss: {delta_hat.numel()} delta_hat values for {i.numel()} / "
                         f"{j.numel()} pair indices")
    if not tau_std > 0:
        raise ValueError(f"siamese_loss: tau_std must be > 0, got {tau_std}")
    if i.numel() == 0:
        return delta_hat.new_zeros(()), delta_hat.new_zeros(())
    dy = y[i] - y[j]
    huber = F.huber_loss(delta_hat, dy, delta=huber_delta, reduction="mean")
    keep = dy.abs() >= tie_std
    if not bool(keep.any()):
        return huber, delta_hat.new_zeros(())
    bce = F.binary_cross_entropy_with_logits(delta_hat[keep] / tau_std, (dy[keep] > 0).to(delta_hat.dtype))
    return huber, bce


def pairhead_candidates(series, lig, kind, row, parent, neg_kinds=(), min_candidates=2):
    """(a, c, is_neg): batch positions of every (anchor, candidate) pair of the pair head's listwise term,
    sorted by anchor, then candidate.

    Anchors and positive candidates are kind-0 rows, one per distinct (series, lig): the first such row in batch
    order (later replicates, and rows of kinds 1-7, are neither anchors nor positive candidates). c is a candidate
    of a when series[c] == series[a] and lig[c] != lig[a], so the candidate count is the number of other distinct
    ligands of the series in the batch. Anchors with fewer than min_candidates candidates are dropped with all
    their pairs. A row n whose kind is in neg_kinds (hinge kinds only) is a negative of the anchor whose row id is
    parent[n] (is_neg True), and of no other anchor (a negative whose parent is a dropped replicate is dropped)."""
    series, lig = series.reshape(-1), lig.reshape(-1)
    kind, row, parent = kind.reshape(-1), row.reshape(-1), parent.reshape(-1)
    dev = series.device
    bad = set(int(k) for k in neg_kinds) - set(HINGE_KINDS)
    if bad:
        raise ValueError(f"pairhead_candidates: negative kinds {sorted(bad)} are not hinge kinds {HINGE_KINDS}")
    crystal = torch.nonzero(kind == 0).view(-1)
    keep = torch.zeros(series.numel(), dtype=torch.bool, device=dev)
    keep[crystal[_first_occurrence(series[crystal], lig[crystal])]] = True
    cand = (series[:, None] == series[None, :]) & (lig[:, None] != lig[None, :]) & keep[:, None] & keep[None, :]
    ok = cand.sum(dim=1) >= int(min_candidates)
    cand &= ok[:, None]
    if len(neg_kinds):
        negk = torch.isin(kind, torch.as_tensor(sorted(int(k) for k in neg_kinds), device=dev))
        neg = (parent[None, :] == row[:, None]) & negk[None, :] & ok[:, None]
    else:
        neg = torch.zeros_like(cand)
    a, c = torch.nonzero(cand | neg, as_tuple=True)
    return a, c, neg[a, c]


def listwise_pairhead(scores, y, a, c, is_neg, tau_std):
    """Mean over anchors of -sum_c q_c log softmax_c(scores) (softmax over each anchor's pairs, negatives
    included), q_c = t_c / sum t, t_c = softmax(y_c / tau_std) over the anchor's positive candidates and 0 for its
    negatives. scores, a, c, is_neg [n_pairs] as from pairhead_candidates (every anchor has >= 1 positive
    candidate); standardised units. Exact 0.0 (no graph) without a pair."""
    scores = scores.reshape(-1)
    y = y.reshape(-1).to(scores.dtype)
    a, c = a.reshape(-1).to(scores.device), c.reshape(-1).to(scores.device)
    is_neg = is_neg.reshape(-1).to(scores.device)
    if not (scores.numel() == a.numel() == c.numel() == is_neg.numel()):
        raise ValueError(f"listwise_pairhead: {scores.numel()} scores for {a.numel()} / {c.numel()} / "
                         f"{is_neg.numel()} pair entries")
    if not tau_std > 0:
        raise ValueError(f"listwise_pairhead: tau_std must be > 0, got {tau_std}")
    if a.numel() == 0:
        return scores.new_zeros(())
    _, seg = torch.unique(a, return_inverse=True)
    n_seg = int(seg.max()) + 1
    pos = ~is_neg
    if not bool(torch.isin(torch.arange(n_seg, device=seg.device), seg[pos]).all()):
        raise ValueError("listwise_pairhead: an anchor has negatives but no positive candidate")
    t = scores.new_zeros(scores.shape)
    t[pos] = _segment_log_softmax((y[c] / tau_std)[pos].detach(), seg[pos], n_seg).exp()
    q = t / scores.new_zeros(n_seg).index_add(0, seg, t)[seg]
    log_p = _segment_log_softmax(scores, seg, n_seg)
    per_anchor = scores.new_zeros(n_seg).index_add(0, seg, -(q * log_p))
    return per_anchor.mean()


def masked_mse(pred, target, mask):
    """Mean squared error over the rows with mask True (mean over rows and columns of pred / target [n, d]);
    other rows (their targets may be NaN) contribute nothing and get no gradient. Exact 0.0 (no graph) when no
    row is selected."""
    mask = mask.reshape(-1).to(pred.device).bool()
    if pred.shape[0] != mask.numel() or target.shape != pred.shape:
        raise ValueError(f"masked_mse: pred {tuple(pred.shape)}, target {tuple(target.shape)}, mask {mask.numel()}")
    if not bool(mask.any()):
        return pred.new_zeros(())
    return F.mse_loss(pred[mask], target[mask].to(pred.dtype))
