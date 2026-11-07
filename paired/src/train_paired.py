"""Train one AEV-PLIG seed with pointwise MSE plus an optional within-series pair loss.

Adapted from the clone's training.py (predict / train / _train, checkpoint = highest running
mean of the last 8 validation Pearson values), with series-aware batches from
paired_data.SeriesBatchSampler, a pair term from pair_losses, and an optional ligand-only
wrapper (AEV columns zeroed) used as the shortcut gate.

Augmented rows (build_paired_dataset --aug / --poses-*; `kind` 1 near-native pose, 2 decoy
pose, 3 A-positive, 4 A-negative shape twin, 5 C-positive, 6 C-negative contact edit, 7 C-control
exposed edit): --train-kinds selects the train rows by index before the sampler is built. Loss roles:
the MSE (and the pair term, when on) runs over rows of kinds {0, 1, 3, 5, 7}; rows of kinds
{2, 4, 6} enter only through the one-sided hinge relu(pred - (y - margin[kind])), y = the PARENT's
pK, margins per kind from --hinge-margins (pK, divided by the scaler's scale), mean over the
batch's hinge rows times --hinge-weight. --decoy-weight / --decoy-margin (arm 4) are aliases that
set kind 2's weight / margin. A selected hinge kind needs a margin and a weight > 0. The label scaler
is fit on the kind-0 train rows (arm 1's train set), whatever the selection. Splits written before
arm 4 have no `kind` field: every row is kind 0.

Siamese head (arm S, --siamese): the GATv2Net is wrapped as heads.AffinityWithSiamese (encoder ->
pooled embedding z -> pointwise MLP, plus heads.SiameseHead on z) and the loss gains
--siamese-weight * (huber + bce) from pair_losses.siamese_loss over the batch's same-series,
different-ligand pairs (paired_data.pair_index, MSE-kind rows only): Huber delta SIAMESE_HUBER_PK,
direction BCE on delta_hat / tau (tau = SIAMESE_TAU_PK) over pairs with |dy| >= --siamese-tie, all
in pK divided by the scaler's scale; exactly zero (head not called) in a batch without such a pair.
The pointwise MSE, the checkpoint rule and the predictions are unchanged (forward = pointwise).

Pair head (arm P, --pair-head): heads.PairHead l(z_a, z_c) on the same wrapper, trained with
--pair-head-weight * pair_losses.listwise_pairhead: anchors = kind-0 rows (one per distinct (series, lig) of the
batch) with >= PAIR_MIN_CANDIDATES candidates (the other distinct ligands of the series, kind 0), soft targets
softmax(y_c / tau) over the candidates (tau = PAIR_TAU_PK / scale), plus, with --pair-neg-kinds, the anchor's own
negatives (rows of those kinds whose parent is the anchor's row) at target 0; exactly zero (head not called)
without an anchor. A hinge kind listed in --pair-neg-kinds may have hinge weight 0: it then enters the listwise term
only (dropped from the hinge). (--pair-weight stays the weight of --pair-loss.)
Moments heads (arm P, --moments pocket|ligand|both, --zernike-side PATH from build_zernike_side.py): one
heads.MomentsHead per target regresses the standardised 343 Zernike invariants of the row's pocket / ligand from z,
MSE over the batch rows of --moments-kinds (default 0) that have that descriptor (side file indexed by data.row;
others masked), weight --moments-weight per target. The side file must hold every row of the four splits with the
same series / lig / kind / parent (else exit; --allow-partial-side masks the missing rows instead).
--sampler-parent-aware builds the series chunks from families (an original row followed by its augmented rows,
paired_data.SeriesBatchSampler parent_aware). With either head an extra line per epoch follows the epoch line:
  aux epoch E pairh W anchors A negatives N moments M pocket P ligand L
(W, M the weighted terms in the loss, P / L the unweighted MSEs; A, N per epoch). The checkpoint rule is unchanged.

Moments as INPUT (arm Z, --input-moments pocket|ligand|both|bothx with --zernike-side): every batch (train and
prediction) gets data.moments [B, d_in] = heads.moments_input_matrix of the side file's standardised descriptors of
its rows (indexed by data.row; the side file's train mean / sd; a row without a descriptor, or missing from the
side file under --allow-partial-side, gets zeros and the missing flag 1). --model gatv2 (default): the wrapper's
MomentsInput (--input-moments-proj, default 256; 0 = raw concatenation) feeds the enlarged fc1 of the pointwise
head; the Siamese / pair / moments heads still read z only; --ligand-only zeroes the AEV columns and leaves
data.moments as it is. --model moments_only: heads.MomentsOnlyNet on data.moments alone (no graph path; no heads, no
--ligand-only). The side file is checked against the data as for --moments.

Outputs under --out-dir (default `output`, relative to the CWD, which is $WORK in the jobs):
  trained_models/<tag>_<i>.model        best-checkpoint state_dict of the bare GATv2Net (with --siamese,
                                        --pair-head, --moments or --input-moments: of the AffinityWithSiamese
                                        wrapper, keys encoder.* / affinity.* / siamese.* / pair.* / moments.* /
                                        moments_in.*; heads.wrapped_to_plain gives the GATv2Net dict when there
                                        is no moments input; --model moments_only: of the MomentsOnlyNet)
  trained_models/<tag>_<i>.json         run configuration and checkpoint summary (model_arch gatv2 |
                                        moments_only, input_moments, input_moments_proj, input_dim, ...)
  trained_models/<tag>.pickle           StandardScaler fit on train pK (atomic write)
  predictions/<tag>_<split>_seed<i>.csv row, truth, pred (pK) for train/valid/test/series_valid
                                        (train: the --train-kinds rows only)
"""
import os
import socket
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import copy
import json
import pickle
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from helpers import pearson, spearman
from model_defs import GATv2Net
from utils import init_weights

import build_zernike_side
import heads
import pair_losses
from pair_losses import DECOY_KIND, HINGE_KINDS, MSE_KINDS
from paired_data import PairedDataset, SeriesBatchSampler, pair_index

SEEDS = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
LR = 0.00012291937615434127
ACTIVATION = "leaky_relu"
SPLITS = ("train", "valid", "test", "series_valid")  # all four get prediction CSVs
N_RDKIT = 15            # node feature columns 0:15 are RDKit atom features, 15: are AEVs
RUNNING_MEAN = 8        # checkpoint on the running mean of the last 8 valid Pearson values
MIN_SERIES_ROWS = 5     # series_valid series entering the mean Spearman
KINDS = tuple(sorted(MSE_KINDS + HINGE_KINDS))  # 0 original, 1-2 poses, 3-4 twins (A), 5-7 edits (C)
DEFAULT_HINGE_MARGINS = "2:0.75,4:1.0,6:1.0"   # pK
SIAMESE_TAU_PK = 0.5     # temperature of the Siamese direction BCE (logit = delta_hat / tau), pK
SIAMESE_HUBER_PK = 1.0   # Huber delta of the Siamese difference term, pK
PAIR_TAU_PK = 1.0        # temperature of the pair head's soft targets softmax(y / tau), pK
PAIR_MIN_CANDIDATES = 2  # an anchor of the pair head's listwise term needs this many positive candidates
MOMENTS_CHOICES = {"none": (), "pocket": ("pocket",), "ligand": ("ligand",), "both": ("pocket", "ligand")}
MODEL_CHOICES = ("gatv2", "moments_only")


class LigandOnly(nn.Module):
    """Shortcut gate: zero the AEV block (columns 15:) before calling the wrapped model."""

    def __init__(self, model, n_keep=N_RDKIT):
        super().__init__()
        self.model = model
        self.n_keep = n_keep

    def _masked(self, data):
        x = data.x.clone()
        x[:, self.n_keep:] = 0
        data = copy.copy(data)  # shallow copy: the caller's batch keeps its x
        data.x = x
        return data

    def forward(self, data):
        return self.model(self._masked(data))

    def embed(self, data):
        """The wrapped AffinityWithSiamese's pooled embedding of the masked graph."""
        return self.model.embed(self._masked(data))


def core_of(model):
    """The model inside a LigandOnly gate (the model itself otherwise)."""
    return model.model if isinstance(model, LigandOnly) else model


def forward_embed(model, data):
    """(pointwise output [B, 1], z [B, D]) from one encoder pass of an AffinityWithSiamese, or of a
    LigandOnly around one (z is then the embedding of the masked graph; data.moments, if any, is read unmasked)."""
    z = model.embed(data)
    return core_of(model).affinity_from_z(z, data), z


@dataclass
class LossConfig:
    kind: str          # none | delta_huber | listwise
    weight: float
    tie: float         # standardised units
    delta: float       # standardised units
    weighting: str     # none | absdelta
    decoy_weight: float = 0.0
    decoy_margin: float = 0.0  # standardised units
    hinge_margins: dict | None = None  # hinge kind -> margin (standardised); None: {2: decoy_margin}
    hinge_weights: dict | None = None  # hinge kind -> weight; None: {2: decoy_weight}

    def margins(self):
        return self.hinge_margins if self.hinge_margins is not None else {DECOY_KIND: self.decoy_margin}

    def weights(self):
        return self.hinge_weights if self.hinge_weights is not None else {DECOY_KIND: self.decoy_weight}


@dataclass
class SiameseConfig:
    weight: float      # --siamese-weight
    tie: float         # standardised units: BCE only on pairs with |dy| >= tie
    tau: float         # standardised units: BCE logit = delta_hat / tau
    delta: float       # standardised units: Huber delta


@dataclass
class PairHeadConfig:
    weight: float      # --pair-head-weight
    tau: float         # standardised units: soft targets softmax(y / tau)
    neg_kinds: tuple   # --pair-neg-kinds
    min_candidates: int = PAIR_MIN_CANDIDATES


class MomentsTargets:
    """The side file's standardised descriptors as tensors on one device, indexed by row id."""

    def __init__(self, side, targets, device):
        self.targets = tuple(targets)
        self.lookup = torch.as_tensor(side["lookup"], dtype=torch.long, device=device)
        self.values = {t: torch.as_tensor(side[f"{t}_std"], device=device) for t in self.targets}
        self.mask = {t: torch.as_tensor(side[f"mask_{t}"], dtype=torch.bool, device=device) for t in self.targets}

    def index(self, row):
        """Index into the side arrays per row id, -1 for rows the file does not hold."""
        row = row.view(-1).to(self.lookup.device)
        ok = (row >= 0) & (row < self.lookup.numel())
        return torch.where(ok, self.lookup[row.clamp(0, max(self.lookup.numel() - 1, 0))],
                           torch.full_like(row, -1))

    def batch(self, row, target):
        """(standardised targets [B, 343], mask [B]) of the rows (masked rows: zeros, mask False)."""
        idx = self.index(row)
        safe = idx.clamp(min=0)
        mask = (idx >= 0) & self.mask[target][safe]
        return self.values[target][safe], mask


class InputMoments:
    """The moments input of arm Z: per batch, data.moments [B, d_in] (heads.moments_input_matrix) from the side
    file's standardised descriptors of the batch rows (data.row); rows without a descriptor or not in the side file
    get zeros and the missing flag."""

    def __init__(self, side, mode, device):
        self.mode = mode
        self.dim = heads.input_moments_dim(mode)
        self.side = MomentsTargets(side, heads.input_moments_targets(mode), device)

    def batch(self, row):
        idx = self.side.index(row)
        found = idx >= 0
        safe = idx.clamp(min=0)
        values = {t: (self.side.values[t][safe], found & self.side.mask[t][safe]) for t in self.side.targets}
        pocket, mask_p = values.get("pocket", (None, None))
        ligand, mask_l = values.get("ligand", (None, None))
        return heads.moments_input_matrix(pocket, ligand, mask_p, mask_l, self.mode)

    def attach(self, data):
        """Set data.moments of a batch (on the side file's device) and return the batch."""
        data.moments = self.batch(data.row)
        return data

    def counts(self, rows_by_split):
        """{split: {"rows", "missing"}}: rows per split (row-id tensors) and those whose input has the missing flag."""
        out = {}
        for split, rows in rows_by_split.items():
            rows = torch.as_tensor(rows).view(-1)
            flag = self.batch(rows)[:, -1] if len(rows) else torch.zeros(0)
            out[split] = {"rows": int(len(rows)), "missing": int((flag > 0).sum())}
        return out


@dataclass
class MomentsConfig:
    weight: float      # --moments-weight (per target)
    side: MomentsTargets
    kinds: tuple = (0,)  # --moments-kinds: only rows of these kinds enter the moments loss


def batch_kind(data, like):
    """The batch's kind column, or zeros for data written before arm 4 (no kind field)."""
    kind = getattr(data, "kind", None)
    if kind is None:
        return torch.zeros(like.numel(), dtype=torch.long, device=like.device)
    return kind.view(-1).to(like.device)


def compute_loss(output, data, cfg):
    """total = MSE over rows of MSE_KINDS + weight * pair term (over the same rows) + hinge.

    hinge = pair_losses.hinge_by_kind over rows of the configured hinge kinds (cfg.margins(),
    cfg.weights()); their `y` is the parent's label. Returns (total, mse, pair, decoy, hinge):
    decoy is the unweighted kind-2 hinge (arm 4's logged term), hinge the weighted term added to
    the total. The pair term is 0 when cfg.kind == 'none'; the MSE is a zero that keeps the graph
    when the batch holds hinge rows only."""
    pred = output.view(-1)
    y = data.y.view(-1)
    kind = batch_kind(data, pred)
    regress = torch.isin(kind, torch.as_tensor(MSE_KINDS, device=kind.device))
    every = bool(regress.all())
    if every:
        mse = F.mse_loss(pred, y)
    elif bool(regress.any()):
        mse = F.mse_loss(pred[regress], y[regress])
    else:
        mse = pred.sum() * 0.0
    if cfg.kind == "none":
        pair = pred.new_zeros(())
    else:
        series, lig = data.series.view(-1), data.lig.view(-1)
        p, t = pred, y
        if not every:  # a hinge row's label is its parent's, not its own: no pairs from it
            p, t, series, lig = pred[regress], y[regress], series[regress], lig[regress]
        if cfg.kind == "delta_huber":
            pair = pair_losses.delta_huber(p, t, series, lig, tie=cfg.tie, delta=cfg.delta,
                                           weighting=cfg.weighting)
        elif cfg.kind == "listwise":
            pair = pair_losses.listwise(p, t, series, lig, tie=cfg.tie, weighting=cfg.weighting)
        else:
            raise ValueError(f"unknown pair loss {cfg.kind!r}")
    margins = cfg.margins()
    decoy = pair_losses.decoy_hinge(pred, y, kind, margins.get(DECOY_KIND, cfg.decoy_margin))
    hinge = pair_losses.hinge_by_kind(pred, y, kind, margins, cfg.weights())
    total = mse + cfg.weight * pair + hinge
    return total, mse, pair, decoy, hinge


def siamese_terms(head, z, data, cfg):
    """(weighted term, huber, bce) of the Siamese head over the batch's same-series, different-ligand
    pairs among rows of MSE_KINDS (pair_index order, i < j). Without such a pair all three are exact
    0.0 tensors without a graph and the head is not called (no BatchNorm update, no gradient to it)."""
    y = data.y.view(-1)
    kind = batch_kind(data, y)
    rows = torch.nonzero(torch.isin(kind, torch.as_tensor(MSE_KINDS, device=kind.device))).view(-1)
    i, j = pair_index(data.series.view(-1)[rows], data.lig.view(-1)[rows])
    if i.numel() == 0:
        return z.new_zeros(()), z.new_zeros(()), z.new_zeros(())
    i, j = rows[i], rows[j]
    delta_hat = head(z[i], z[j])
    huber, bce = pair_losses.siamese_loss(delta_hat, y, i, j, tie_std=cfg.tie, tau_std=cfg.tau,
                                          huber_delta=cfg.delta)
    return cfg.weight * (huber + bce), huber, bce


def batch_parent(data, like):
    """The batch's parent column, or its row ids for data written before arm 4 (no parent field)."""
    parent = getattr(data, "parent", None)
    if parent is None:
        return data.row.view(-1).to(like.device)
    return parent.view(-1).to(like.device)


def pairhead_terms(head, z, data, cfg):
    """(weighted term, listwise loss, n_anchors, n_negatives) of the pair head over the batch
    (pair_losses.pairhead_candidates + listwise_pairhead). Without an anchor the two terms are exact 0.0 tensors
    without a graph and the head is not called."""
    y = data.y.view(-1)
    kind = batch_kind(data, y)
    a, c, neg = pair_losses.pairhead_candidates(data.series.view(-1).to(y.device), data.lig.view(-1).to(y.device),
                                                kind, data.row.view(-1).to(y.device), batch_parent(data, y),
                                                cfg.neg_kinds, cfg.min_candidates)
    if a.numel() == 0:
        return z.new_zeros(()), z.new_zeros(()), 0, 0
    scores = head(z[a], z[c])
    loss = pair_losses.listwise_pairhead(scores, y, a, c, neg, cfg.tau)
    return cfg.weight * loss, loss, int(torch.unique(a).numel()), int(neg.sum())


def moments_terms(moments, z, data, cfg):
    """(weighted term, {target: masked MSE}) of the moments heads (nn.ModuleDict target -> MomentsHead) over the
    batch rows of cfg.kinds with that target's descriptor; a target without such a row gives an exact 0.0 (head
    not called)."""
    parts = {}
    kind = batch_kind(data, data.y.view(-1)).to(cfg.side.lookup.device)
    use = torch.isin(kind, torch.as_tensor(cfg.kinds, device=kind.device))
    for target in cfg.side.targets:
        values, mask = cfg.side.batch(data.row, target)
        mask = mask & use
        if not bool(mask.any()):
            parts[target] = z.new_zeros(())
            continue
        pred = moments[target](z[mask])
        parts[target] = pair_losses.masked_mse(pred, values[mask], torch.ones_like(mask[mask]))
    total = None
    for t in cfg.side.targets:
        total = parts[t] if total is None else total + parts[t]
    return cfg.weight * total, parts


def parse_neg_kinds(text):
    """'4,6' -> (4, 6) (sorted, unique, hinge kinds only); '' -> ()."""
    try:
        kinds = sorted({int(t) for t in text.split(",") if t.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--pair-neg-kinds {text!r}: expected a comma list of hinge kinds "
                                         f"{', '.join(map(str, HINGE_KINDS))}")
    if set(kinds) - set(HINGE_KINDS):
        raise argparse.ArgumentTypeError(f"--pair-neg-kinds {text!r}: kinds must be hinge kinds "
                                         f"({', '.join(map(str, HINGE_KINDS))})")
    return tuple(kinds)


def parse_moments_kinds(text):
    """'0,3' -> (0, 3) (sorted, unique, each in KINDS)."""
    try:
        return tuple(parse_kinds(text))
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(str(exc).replace("--train-kinds", "--moments-kinds"))


def parse_kinds(text):
    """'0,3,4' -> [0, 3, 4] (sorted, unique, each in KINDS)."""
    try:
        kinds = sorted({int(t) for t in text.split(",") if t.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--train-kinds {text!r}: expected a comma list of kinds "
                                         f"{', '.join(map(str, KINDS))}")
    if not kinds or set(kinds) - set(KINDS):
        raise argparse.ArgumentTypeError(f"--train-kinds {text!r}: kinds must be among "
                                         f"{', '.join(map(str, KINDS))}")
    return kinds


def parse_margins(text):
    """'2:0.75,4:1.0,6:1.0' -> {2: 0.75, 4: 1.0, 6: 1.0} (pK); keys must be hinge kinds."""
    out = {}
    for item in (t.strip() for t in text.split(",")):
        if not item:
            continue
        try:
            k, m = item.split(":")
            k, m = int(k), float(m)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--hinge-margins {text!r}: expected kind:margin pairs, "
                                             f"e.g. {DEFAULT_HINGE_MARGINS}")
        if k not in HINGE_KINDS:
            raise argparse.ArgumentTypeError(f"--hinge-margins {text!r}: kind {k} is not a hinge kind "
                                             f"({', '.join(map(str, HINGE_KINDS))})")
        if not m >= 0:
            raise argparse.ArgumentTypeError(f"--hinge-margins {text!r}: margin of kind {k} must be >= 0")
        out[k] = m
    return out


def resolve_hinge(args, error):
    """Per selected hinge kind: margin (pK) and weight, after the arm-4 aliases; error() on a
    selected hinge kind without a margin or with weight 0. Sets args.hinge_kinds,
    args.hinge_margins_pk, args.hinge_weights (selected kinds) and args.decoy_margin_pk."""
    margins = dict(args.hinge_margins)
    if args.decoy_margin is not None:
        margins[DECOY_KIND] = args.decoy_margin
    weights = {k: args.hinge_weight for k in HINGE_KINDS}
    if args.decoy_weight is not None:
        weights[DECOY_KIND] = args.decoy_weight
    selected = [k for k in args.train_kinds if k in HINGE_KINDS]
    # arm P: a negative kind of the pair head's listwise term may have hinge weight 0 (listwise only)
    listwise_only = set(getattr(args, "pair_neg_kinds", ())) if getattr(args, "pair_head", False) else set()
    for k in selected:
        if not weights[k] > 0 and k in listwise_only:
            continue
        if k not in margins:
            error(f"--train-kinds selects hinge kind {k} but --hinge-margins has no margin for it")
        if not weights[k] > 0:
            name = "--decoy-weight" if k == DECOY_KIND and args.decoy_weight is not None else "--hinge-weight"
            error(f"--train-kinds selects hinge kind {k} but its weight ({name}) is {weights[k]}: "
                  "those rows would be silently ignored")
    selected = [k for k in selected if weights[k] > 0 or k not in listwise_only]
    if args.decoy_weight is not None and args.decoy_weight > 0 and DECOY_KIND not in args.train_kinds:
        error("--decoy-weight > 0 needs kind 2 (decoys) in --train-kinds")
    args.hinge_kinds = selected
    args.decoy_margin_pk = margins.get(DECOY_KIND)
    args.hinge_margins_pk = {k: margins[k] for k in selected}
    args.hinge_weights = {k: weights[k] for k in selected}
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="dir holding processed/paired_<split>.pt")
    p.add_argument("--seed_index", type=int, required=True, help="index into SEEDS (0..9)")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="output", help="holds trained_models/ and predictions/")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--pair-loss", choices=["none", "delta_huber", "listwise"], default="none")
    p.add_argument("--pair-weight", type=float, default=0.3)
    p.add_argument("--tie", type=float, default=0.5, help="pK; pairs with |dpK| < tie ignored")
    p.add_argument("--huber-delta", type=float, default=1.0, help="pK")
    p.add_argument("--pair-weighting", choices=list(pair_losses.WEIGHTINGS), default="none")
    p.add_argument("--ligand-only", action="store_true", help="zero AEV columns 15: (gate)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--train-kinds", type=parse_kinds, default=[0],
                   help="comma list of train row kinds: 0 original, 1 near-native pose, 2 decoy, "
                        "3 A-positive, 4 shape twin, 5 C-positive, 6 contact edit, 7 exposed edit")
    p.add_argument("--hinge-margins", type=parse_margins, default=DEFAULT_HINGE_MARGINS,
                   help="pK margin per hinge kind: rows are pushed below their parent's pK minus it")
    p.add_argument("--hinge-weight", type=float, default=1.0, help="weight of the hinge term")
    p.add_argument("--decoy-weight", type=float, default=None,
                   help="arm-4 alias: the hinge weight of kind 2 (decoys)")
    p.add_argument("--decoy-margin", type=float, default=None,
                   help="arm-4 alias: the hinge margin of kind 2 (pK), overriding --hinge-margins")
    p.add_argument("--siamese", action="store_true",
                   help="arm S: Siamese difference head on the pooled embedding, trained with the MSE")
    p.add_argument("--siamese-weight", type=float, default=1.0, help="weight of the Siamese term")
    p.add_argument("--siamese-hidden", type=int, default=512, help="hidden width of the Siamese MLP")
    p.add_argument("--siamese-tie", type=float, default=0.5,
                   help="pK; the Siamese direction BCE ignores pairs with |dpK| < tie")
    p.add_argument("--pair-head", action="store_true",
                   help="arm P: pair head l(z_a, z_c) trained with the within-series listwise loss")
    p.add_argument("--pair-head-weight", type=float, default=0.3,
                   help="weight of the pair head's listwise term (--pair-weight is --pair-loss's)")
    p.add_argument("--pair-hidden", type=int, default=512, help="hidden width of the pair head")
    p.add_argument("--pair-neg-kinds", type=parse_neg_kinds, default=(),
                   help="comma list of hinge kinds (2, 4, 6) whose rows join their parent anchor's candidate list "
                        "of the pair head at target 0")
    p.add_argument("--moments", choices=list(MOMENTS_CHOICES), default="none",
                   help="arm P: regress the standardised pocket / ligand Zernike invariants from z")
    p.add_argument("--moments-weight", type=float, default=0.3, help="weight of each moments head's MSE")
    p.add_argument("--zernike-side", default=None, help="paired_zernike.npz from build_zernike_side.py")
    p.add_argument("--moments-kinds", type=parse_moments_kinds, default=None,
                   help="comma list of row kinds entering the moments loss (default 0: original rows only)")
    p.add_argument("--allow-partial-side", action="store_true",
                   help="mask rows missing from --zernike-side instead of exiting")
    p.add_argument("--sampler-parent-aware", action="store_true",
                   help="chunk each series by families: an original row followed by its augmented rows")
    p.add_argument("--input-moments", choices=list(heads.INPUT_MOMENTS_CHOICES), default="none",
                   help="arm Z: feed the standardised pocket / ligand Zernike invariants (bothx: plus the "
                        "pocket-complement x ligand block) and a missing flag to the pointwise head (needs "
                        "--zernike-side)")
    p.add_argument("--input-moments-proj", type=int, default=None,
                   help=f"width of the moments-input projection (default {heads.INPUT_MOMENTS_PROJ}; 0 = raw "
                        "concatenation); --model gatv2 only")
    p.add_argument("--model", choices=list(MODEL_CHOICES), default="gatv2",
                   help="gatv2 (default) or moments_only: an MLP on the moments input alone (no graph path)")
    # Paper hyper-parameters; exposed only so the CPU smoke test can shrink the model.
    p.add_argument("--hidden-dim", type=int, default=256, help=argparse.SUPPRESS)
    p.add_argument("--head", type=int, default=3, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 0 <= args.seed_index < len(SEEDS):
        p.error(f"--seed_index must be in 0..{len(SEEDS) - 1}")
    if args.pair_loss == "listwise" and args.pair_weighting != "none":
        p.error("--pair-weighting absdelta is defined for delta_huber only")
    if args.decoy_weight is not None and args.decoy_weight < 0:
        p.error("--decoy-weight must be >= 0")
    if args.hinge_weight < 0:
        p.error("--hinge-weight must be >= 0")
    if args.decoy_margin is not None and not args.decoy_margin >= 0:
        p.error("--decoy-margin must be >= 0")
    if args.siamese and not args.siamese_weight > 0:
        p.error("--siamese-weight must be > 0 with --siamese (the head would never be trained)")
    if args.siamese and args.siamese_hidden < 1:
        p.error("--siamese-hidden must be >= 1")
    if args.siamese and not args.siamese_tie >= 0:
        p.error("--siamese-tie must be >= 0")
    if args.pair_head and not args.pair_head_weight > 0:
        p.error("--pair-head-weight must be > 0 with --pair-head (the head would never be trained)")
    if args.pair_head and args.pair_hidden < 1:
        p.error("--pair-hidden must be >= 1")
    if args.pair_neg_kinds and not args.pair_head:
        p.error("--pair-neg-kinds needs --pair-head")
    missing = [k for k in args.pair_neg_kinds if k not in args.train_kinds]
    if missing:
        p.error(f"--pair-neg-kinds {missing} are not in --train-kinds: those negatives would never be in a batch")
    if args.model == "moments_only":
        if args.input_moments == "none":
            p.error("--model moments_only needs --input-moments (pocket, ligand, both or bothx)")
        bad = [flag for flag, on in (("--ligand-only", args.ligand_only), ("--siamese", args.siamese),
                                     ("--pair-head", args.pair_head), ("--moments", args.moments != "none"),
                                     ("--input-moments-proj", args.input_moments_proj is not None)) if on]
        if bad:
            p.error(f"--model moments_only has no graph encoder: {', '.join(bad)} not allowed")
    if args.input_moments == "none" and args.input_moments_proj is not None:
        p.error("--input-moments-proj given with --input-moments none")
    if args.input_moments_proj is not None and args.input_moments_proj < 0:
        p.error("--input-moments-proj must be >= 0 (0 = raw concatenation)")
    if args.model == "gatv2" and args.input_moments != "none" and args.input_moments_proj is None:
        args.input_moments_proj = heads.INPUT_MOMENTS_PROJ
    if args.moments != "none" and not args.zernike_side:
        p.error("--moments needs --zernike-side (build_zernike_side.py)")
    if args.input_moments != "none" and not args.zernike_side:
        p.error("--input-moments needs --zernike-side (build_zernike_side.py)")
    if not uses_side(args) and args.zernike_side:
        p.error("--zernike-side given with --moments none and --input-moments none (did you forget --moments or "
                "--input-moments?)")
    if args.moments != "none" and not args.moments_weight > 0:
        p.error("--moments-weight must be > 0 with --moments (the heads would never be trained)")
    if args.moments == "none" and args.moments_kinds is not None:
        p.error("--moments-kinds given with --moments none")
    if not uses_side(args) and args.allow_partial_side:
        p.error("--allow-partial-side given with --moments none and --input-moments none")
    if args.moments != "none":
        args.moments_kinds = args.moments_kinds if args.moments_kinds is not None else (0,)
        missing = [k for k in args.moments_kinds if k not in args.train_kinds]
        if missing:
            p.error(f"--moments-kinds {missing} are not in --train-kinds: no such row would enter the moments loss")
    return resolve_hinge(args, p.error)


def uses_side(args):
    """True when the run reads --zernike-side (moments heads or moments inputs)."""
    return getattr(args, "moments", "none") != "none" or getattr(args, "input_moments", "none") != "none"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(data_dir):
    return {split: PairedDataset(data_dir, split) for split in SPLITS}


def dataset_kinds(ds):
    """Per-row kind of an InMemoryDataset collate (zeros without a kind field)."""
    kind = getattr(ds._data, "kind", None)
    if kind is None:
        return np.zeros(ds._data.y.numel(), dtype=np.int64)
    return kind.view(-1).numpy()


def standardise(datasets):
    """Fit StandardScaler on the kind-0 train pK (every train row in pre-arm-4 files) and
    transform y of every split in place."""
    y_train = datasets["train"]._data.y.view(-1, 1).double().numpy()
    y_train = y_train[dataset_kinds(datasets["train"]) == 0]
    scaler = StandardScaler().fit(y_train)
    for ds in datasets.values():
        y = ds._data.y.view(-1, 1).double().numpy()
        ds._data.y = torch.as_tensor(scaler.transform(y).ravel(), dtype=torch.float)
        ds._data_list = None  # drop any cached items built with the raw labels
    return scaler


def write_scaler_atomic(scaler, path):
    """Array tasks of one tag share the scaler file: write to a unique temp name, then rename."""
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"  # hostname: array tasks on other nodes share the dir
    with open(tmp, "wb") as fh:
        pickle.dump(scaler, fh)
    os.replace(tmp, path)


def select_kinds(train, kinds):
    """(train restricted to rows whose kind is in kinds, selected indices, rows per kind).
    Uses PyG index selection (a view sharing the collated storage); exits when a requested
    kind has no train rows, so a pose arm never runs silently on a dataset without poses."""
    kind = dataset_kinds(train)
    counts = {str(k): int((kind == k).sum()) for k in kinds}
    missing = [k for k in kinds if counts[str(k)] == 0]
    if missing:
        sys.exit(f"ERROR: --train-kinds {','.join(map(str, kinds))}: the train split has no rows "
                 f"of kind {missing} (kinds present: {sorted(set(kind.tolist()))})")
    idx = np.flatnonzero(np.isin(kind, kinds))
    if len(idx) == len(kind):
        return train, idx, counts
    return train[torch.as_tensor(idx, dtype=torch.long)], idx, counts


def wants_wrapper(args):
    return getattr(args, "model", "gatv2") == "gatv2" and (
        getattr(args, "siamese", False) or getattr(args, "pair_head", False)
        or getattr(args, "moments", "none") != "none" or getattr(args, "input_moments", "none") != "none")


def build_model(train_data, args):
    input_mode = getattr(args, "input_moments", "none")
    if getattr(args, "model", "gatv2") == "moments_only":
        net = heads.MomentsOnlyNet(heads.input_moments_dim(input_mode), act=ACTIVATION, mode=input_mode)
        net.apply(init_weights)  # Linear: xavier normal; BatchNorm1d untouched (the clone's MLP init)
        return net
    config = SimpleNamespace(activation_function=ACTIVATION, head=args.head,
                             hidden_dim=args.hidden_dim)
    model = GATv2Net(node_feature_dim=train_data.num_node_features,
                     edge_feature_dim=train_data.num_edge_features, config=config)
    model.apply(init_weights)
    if not wants_wrapper(args):
        return model
    # heads are initialised after the GATv2Net (Siamese, pair, then moments pocket / ligand), so the encoder /
    # MLP start as in the plain arm and a Siamese-only model starts as in arm S; the moments input (arm Z) last
    dim = heads.embed_dim(model)
    siamese = pair = None
    if getattr(args, "siamese", False):
        siamese = heads.SiameseHead(dim, hidden=args.siamese_hidden, act=ACTIVATION)
        siamese.apply(init_weights)
    if getattr(args, "pair_head", False):
        pair = heads.init_linear(heads.PairHead(dim, hidden=args.pair_hidden, act=ACTIVATION), init_weights)
    moments = {t: heads.init_linear(heads.MomentsHead(dim, act=ACTIVATION), init_weights)
               for t in MOMENTS_CHOICES[getattr(args, "moments", "none")]}
    moments_in = None
    if input_mode != "none":
        moments_in = heads.init_linear(heads.MomentsInput(heads.input_moments_dim(input_mode),
                                                          proj=args.input_moments_proj, act=ACTIVATION,
                                                          mode=input_mode), init_weights)
    wrapper = heads.AffinityWithSiamese(model, siamese, pair, moments or None, moments_in)
    if moments_in is not None:  # the enlarged fc1 replaced the GATv2Net's: the clone's init on it
        heads.init_linear(wrapper.affinity.fc1, init_weights)
    return wrapper


@torch.no_grad()
def predict(model, device, loader, scaler, inputs=None):
    """Predictions in pK as a DataFrame with columns row, series, truth, pred (dataset order); inputs (an
    InputMoments) attaches data.moments to every batch."""
    model.eval()
    rows, series, truth, pred = [], [], [], []
    for data in loader:
        data = data.to(device)
        if inputs is not None:
            inputs.attach(data)
        pred.append(model(data).view(-1).cpu())
        truth.append(data.y.view(-1).cpu())
        rows.append(data.row.view(-1).cpu())
        series.append(data.series.view(-1).cpu())

    def inv(t):
        return scaler.inverse_transform(torch.cat(t).double().numpy().reshape(-1, 1)).ravel()

    return pd.DataFrame({"row": torch.cat(rows).numpy(), "series": torch.cat(series).numpy(),
                         "truth": inv(truth), "pred": inv(pred)})


def series_metrics(df, min_rows=MIN_SERIES_ROWS):
    """(mean per-series Spearman over series with >= min_rows rows, pooled Pearson), pK."""
    rhos = [spearman(g["truth"].to_numpy(), g["pred"].to_numpy())
            for _, g in df.groupby("series") if len(g) >= min_rows]
    n_all = len(rhos)
    rhos = [r for r in rhos if np.isfinite(r)]  # NaN = constant truth or prediction in that series; skipped
    if n_all != len(rhos):
        print(f"series_metrics: {n_all - len(rhos)} of {n_all} series skipped (NaN Spearman)")
    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    return mean_rho, float(pearson(df["truth"].to_numpy(), df["pred"].to_numpy()))


AUX_KEYS = ("pairh", "anchors", "negatives", "moments", "pocket", "ligand")


def train_epoch(model, device, loader, optimizer, loss_cfg, siamese_cfg=None, pair_cfg=None, moments_cfg=None,
                aux=None, inputs=None):
    """One pass over the series-aware batches; returns sample-weighted mean
    total/mse/pair/decoy/hinge/siam (siam: the weighted Siamese term, 0 without siamese_cfg). With pair_cfg /
    moments_cfg and a dict `aux`, aux gets the sample-weighted means of the weighted pair-head term (pairh), the
    weighted moments term (moments) and the per-target MSEs (pocket, ligand), and the epoch's anchor / negative
    counts (anchors, negatives). inputs (an InputMoments) attaches data.moments to every batch."""
    model.train()
    sums = np.zeros(6)
    aux_sums = dict.fromkeys(AUX_KEYS, 0.0)
    n_seen = 0
    heads_on = siamese_cfg is not None or pair_cfg is not None or moments_cfg is not None
    for data in loader:
        data = data.to(device)
        if inputs is not None:
            inputs.attach(data)
        optimizer.zero_grad()
        n = data.num_graphs
        if not heads_on:
            terms = compute_loss(model(data), data, loss_cfg)
            total, siam = terms[0], 0.0
        else:
            output, z = forward_embed(model, data)
            terms = compute_loss(output, data, loss_cfg)
            total, siam = terms[0], 0.0
            core = core_of(model)
            if siamese_cfg is not None:
                siam_t = siamese_terms(core.siamese, z, data, siamese_cfg)[0]
                total = total + siam_t
                siam = siam_t.item()
            if pair_cfg is not None:
                pw, _, n_anchors, n_neg = pairhead_terms(core.pair, z, data, pair_cfg)
                total = total + pw
                aux_sums["pairh"] += n * pw.item()
                aux_sums["anchors"] += n_anchors
                aux_sums["negatives"] += n_neg
            if moments_cfg is not None:
                mw, parts = moments_terms(core.moments, z, data, moments_cfg)
                total = total + mw
                aux_sums["moments"] += n * mw.item()
                for t, v in parts.items():
                    aux_sums[t] += n * v.item()
        total.backward()
        optimizer.step()
        sums += n * np.array([total.item()] + [t.item() for t in terms[1:]] + [siam])
        n_seen += n
    if aux is not None:
        for k, v in aux_sums.items():
            aux[k] = v if k in ("anchors", "negatives") else v / max(n_seen, 1)
    return sums / max(n_seen, 1)


def fit(model, core_model, device, loaders, sampler, optimizer, loss_cfg, scaler, epochs,
        ckpt_path, siamese_cfg=None, pair_cfg=None, moments_cfg=None, inputs=None):
    """training.py's _train with the extra logging; saves the core model's state_dict."""
    best_avg, best_epoch, pcs = -1.1, 0, []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        aux = {}
        total, mse, pair, decoy, hinge, siam = train_epoch(model, device, loaders["train"], optimizer,
                                                           loss_cfg, siamese_cfg, pair_cfg, moments_cfg, aux,
                                                           inputs)
        valid = predict(model, device, loaders["valid"], scaler, inputs)
        current_pc = pearson(valid["truth"].to_numpy(), valid["pred"].to_numpy())
        pcs.append(current_pc)
        avg_pc = np.mean(pcs[max(epoch - (RUNNING_MEAN - 1), 0):epoch + 1])
        if avg_pc > best_avg:
            torch.save(core_model.state_dict(), ckpt_path)
            best_avg, best_epoch = avg_pc, epoch + 1
        sv_rho, sv_pc = series_metrics(predict(model, device, loaders["series_valid"], scaler, inputs))
        print(f"epoch {epoch + 1} loss {total:.6f} mse {mse:.6f} pair {pair:.6f} decoy {decoy:.6f} "
              f"hinge {hinge:.6f} siam {siam:.6f} valid_pearson {current_pc:.6f} "
              f"series_valid_spearman {sv_rho:.6f} "
              f"series_valid_pearson {sv_pc:.6f}", flush=True)
        if pair_cfg is not None or moments_cfg is not None:
            print(f"aux epoch {epoch + 1} pairh {aux['pairh']:.6f} anchors {int(aux['anchors'])} "
                  f"negatives {int(aux['negatives'])} moments {aux['moments']:.6f} pocket {aux['pocket']:.6f} "
                  f"ligand {aux['ligand']:.6f}", flush=True)
    if best_epoch == 0:  # every running mean was NaN; keep the last model rather than none
        print("WARNING: no finite validation Pearson; saving the final epoch", flush=True)
        torch.save(core_model.state_dict(), ckpt_path)
        best_epoch = epochs
    print(f"best epoch {best_epoch} running-mean valid_pearson {best_avg:.6f}", flush=True)
    return best_epoch, float(best_avg)


def write_predictions(model, device, loaders, scaler, pred_dir, tag, seed_index, inputs=None):
    paths = {}
    for split in SPLITS:
        df = predict(model, device, loaders[split], scaler, inputs)
        path = os.path.join(pred_dir, f"{tag}_{split}_seed{seed_index}.csv")
        df[["row", "truth", "pred"]].to_csv(path, index=False)
        rho, pc = series_metrics(df)
        print(f"final {split} n {len(df)} pearson {pc:.6f} series_spearman {rho:.6f} -> {path}",
              flush=True)
        paths[split] = path
    return paths


def check_side(side, datasets, kinds=None, allow_partial=False):
    """Exit unless every row of the four splits is in the side file (allow_partial: missing rows are only
    counted; they are masked in training) with the side file's series_id / lig_id / kind / parent (a side file
    of another build would index other rows). Returns {split: {"rows", "in_side", "pocket", "ligand"}} counts;
    pocket / ligand count rows of `kinds` (all kinds when None) with that descriptor."""
    out, problems = {}, []
    for split, ds in datasets.items():
        d = ds._data
        rows = d.row.view(-1).numpy()
        idx = build_zernike_side.side_index(side, rows)
        found = idx >= 0
        got = {"series_id": d.series.view(-1).numpy(), "lig_id": d.lig.view(-1).numpy(),
               "kind": dataset_kinds(ds),
               "parent": d.parent.view(-1).numpy() if getattr(d, "parent", None) is not None else rows}
        for col, values in got.items():
            ref = side[col][idx[found]]
            if col in ("series_id", "lig_id") and (ref < 0).all():
                continue  # meta without the column
            bad = int((ref != values[found]).sum())
            if bad:
                problems.append(f"{split}: {bad} rows differ in {col}")
        if not found.all() and not allow_partial:
            problems.append(f"{split}: {int((~found).sum())} rows are not in the side file "
                            "(--allow-partial-side masks them)")
        sel = found if kinds is None else found & np.isin(got["kind"], list(kinds))
        out[split] = {"rows": int(len(rows)), "in_side": int(found.sum()),
                      "pocket": int(side["mask_pocket"][idx[sel]].sum()),
                      "ligand": int(side["mask_ligand"][idx[sel]].sum())}
    if problems:
        sys.exit("ERROR: --zernike-side does not belong to --data-dir: " + "; ".join(problems))
    return out


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    seed = SEEDS[args.seed_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(args.out_dir, "trained_models")
    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f"{args.tag}_{args.seed_index}.model")
    scaler_path = os.path.join(model_dir, f"{args.tag}.pickle")
    print(f"tag {args.tag} seed_index {args.seed_index} seed {seed} device {device} "
          f"args {vars(args)}", flush=True)

    datasets = load_splits(args.data_dir)
    n_train_all = len(datasets["train"])
    scaler = standardise(datasets)
    # select after standardising (the view shares the collated storage and must not hold items
    # cached with raw labels) and before writing anything (select_kinds may exit)
    train, train_idx, kind_counts = select_kinds(datasets["train"], args.train_kinds)
    side, side_counts, side_digest = None, None, None
    if uses_side(args):  # checked before anything is written (check_side may exit)
        side = build_zernike_side.load_side(args.zernike_side)
        side_counts = check_side(side, datasets, args.moments_kinds if args.moments != "none" else None,
                                 args.allow_partial_side)
        side_digest = heads.side_stats_digest(side)
    write_scaler_atomic(scaler, scaler_path)
    scale = float(scaler.scale_[0])
    margins_std = {k: m / scale for k, m in args.hinge_margins_pk.items()}
    decoy_weight = args.hinge_weights.get(DECOY_KIND, args.decoy_weight or 0.0)
    decoy_margin = args.decoy_margin_pk  # kind 2's margin (pK) after the alias; None if unset
    loss_cfg = LossConfig(kind=args.pair_loss, weight=args.pair_weight, tie=args.tie / scale,
                          delta=args.huber_delta / scale, weighting=args.pair_weighting,
                          decoy_weight=decoy_weight,
                          decoy_margin=(decoy_margin or 0.0) / scale,
                          hinge_margins=margins_std, hinge_weights=dict(args.hinge_weights))
    siamese_cfg = (SiameseConfig(weight=args.siamese_weight, tie=args.siamese_tie / scale,
                                 tau=SIAMESE_TAU_PK / scale, delta=SIAMESE_HUBER_PK / scale)
                   if args.siamese else None)
    pair_cfg = (PairHeadConfig(weight=args.pair_head_weight, tau=PAIR_TAU_PK / scale,
                               neg_kinds=tuple(args.pair_neg_kinds)) if args.pair_head else None)
    moments_cfg = None
    if args.moments != "none":
        moments_cfg = MomentsConfig(weight=args.moments_weight,
                                    side=MomentsTargets(side, MOMENTS_CHOICES[args.moments], device),
                                    kinds=tuple(args.moments_kinds))
        print(f"moments {args.moments} from {args.zernike_side}: rows with descriptors {side_counts}", flush=True)
    inputs, input_counts = None, None
    if args.input_moments != "none":
        inputs = InputMoments(side, args.input_moments, device)
        rows_by_split = {split: ds._data.row.view(-1) for split, ds in datasets.items()}
        rows_by_split["train"] = rows_by_split["train"][torch.as_tensor(train_idx, dtype=torch.long)]  # selected kinds
        input_counts = inputs.counts(rows_by_split)
        print(f"input moments {args.input_moments} (d_in {inputs.dim}, model {args.model}, proj "
              f"{args.input_moments_proj}) from {args.zernike_side}: rows / rows with the missing flag "
              f"{input_counts}", flush=True)
    side = None
    sizes = {split: len(ds) for split, ds in datasets.items()}
    sizes["train"] = len(train)
    print(f"sizes {sizes} (train split {n_train_all} rows, kinds {args.train_kinds}: {kind_counts}) "
          f"pK mean {scaler.mean_[0]:.4f} scale {scale:.4f} loss {loss_cfg} siamese {siamese_cfg}"
          + (f" pair_head {pair_cfg}" if pair_cfg is not None else ""), flush=True)

    set_seed(seed)
    series = datasets["train"]._data.series.view(-1).numpy()[train_idx]
    if args.sampler_parent_aware:
        tdata = datasets["train"]._data
        rows = tdata.row.view(-1).numpy()[train_idx]
        parents = (tdata.parent.view(-1).numpy()[train_idx] if getattr(tdata, "parent", None) is not None
                   else rows)
        sampler = SeriesBatchSampler(series, batch_size=args.batch_size, chunk=args.chunk, seed=seed,
                                     parent_aware=True, rows=rows, parents=parents)
    else:
        sampler = SeriesBatchSampler(series, batch_size=args.batch_size, chunk=args.chunk, seed=seed)
    loaders = {"train": DataLoader(train, batch_sampler=sampler)}
    for split in ("valid", "test", "series_valid"):
        loaders[split] = DataLoader(datasets[split], batch_size=args.batch_size, shuffle=False)

    core_model = build_model(train, args)
    model = LigandOnly(core_model) if args.ligand_only else core_model
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    best_epoch, best_avg = fit(model, core_model, device, loaders, sampler, optimizer, loss_cfg,
                               scaler, args.epochs, ckpt_path, siamese_cfg, pair_cfg, moments_cfg, inputs)

    core_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # Train predictions use a fixed-order loader (eval mode), not the series sampler.
    loaders["train"] = DataLoader(train, batch_size=args.batch_size, shuffle=False)
    paths = write_predictions(model, device, loaders, scaler, pred_dir, args.tag,
                              args.seed_index, inputs)

    summary = {"args": vars(args), "seed": seed, "scaler_mean": float(scaler.mean_[0]),
               "scaler_scale": scale, "best_epoch": best_epoch,
               "best_running_valid_pearson": best_avg, "model": ckpt_path,
               "scaler": scaler_path, "predictions": paths, "train_kinds": list(args.train_kinds),
               "mse_kinds": [k for k in args.train_kinds if k in MSE_KINDS],
               "hinge_kinds": list(args.hinge_kinds),
               "hinge_margins": {str(k): m for k, m in args.hinge_margins_pk.items()},
               "hinge_margins_std": {str(k): m for k, m in margins_std.items()},
               "hinge_weights": {str(k): w for k, w in args.hinge_weights.items()},
               "hinge_weight": args.hinge_weight,
               "decoy_weight": float(decoy_weight), "decoy_margin": decoy_margin,
               "decoy_margin_std": None if decoy_margin is None else decoy_margin / scale,
               "n_train_rows": int(len(train)),
               "n_train_split_rows": int(n_train_all), "train_kind_counts": kind_counts,
               "model_format": ("moments_only" if args.model == "moments_only"
                                else "affinity_with_siamese" if wants_wrapper(args) else "gatv2net"),
               "siamese": bool(args.siamese),
               "siamese_hidden": int(args.siamese_hidden) if args.siamese else None,
               "siamese_weight": float(args.siamese_weight) if args.siamese else None,
               "siamese_dim": int(core_model.dim) if args.siamese else None,
               "siamese_tie": float(args.siamese_tie) if args.siamese else None,
               "siamese_tie_std": siamese_cfg.tie if args.siamese else None,
               "siamese_tau": SIAMESE_TAU_PK if args.siamese else None,
               "siamese_tau_std": siamese_cfg.tau if args.siamese else None,
               "siamese_huber_delta": SIAMESE_HUBER_PK if args.siamese else None,
               "siamese_huber_delta_std": siamese_cfg.delta if args.siamese else None,
               # arm P (pair_head_weight is the pair HEAD's weight; args.pair_weight is --pair-loss's)
               "pair_head": bool(args.pair_head),
               "pair_hidden": int(args.pair_hidden) if args.pair_head else None,
               "pair_head_weight": float(args.pair_head_weight) if args.pair_head else None,
               "pair_neg_kinds": list(args.pair_neg_kinds) if args.pair_head else None,
               "pair_tau": PAIR_TAU_PK if args.pair_head else None,
               "pair_tau_std": pair_cfg.tau if args.pair_head else None,
               "pair_min_candidates": PAIR_MIN_CANDIDATES if args.pair_head else None,
               "moments": args.moments,
               "moments_weight": float(args.moments_weight) if args.moments != "none" else None,
               "moments_dim": heads.MOMENTS_DIM if args.moments != "none" else None,
               "moments_hidden": heads.MOMENTS_HIDDEN if args.moments != "none" else None,
               "zernike_side": (os.path.abspath(args.zernike_side) if uses_side(args) else None),
               "zernike_side_stats": side_digest,
               "moments_kinds": list(args.moments_kinds) if args.moments != "none" else None,
               "moments_rows": side_counts if args.moments != "none" else None,
               "sampler_parent_aware": bool(args.sampler_parent_aware),
               # arm Z (the top-level "model" above is the checkpoint path; the architecture is model_arch)
               "model_arch": args.model,
               "input_moments": args.input_moments,
               "input_moments_proj": (int(args.input_moments_proj) if args.input_moments_proj is not None
                                      and args.model == "gatv2" and args.input_moments != "none" else None),
               "input_dim": heads.input_moments_dim(args.input_moments) if args.input_moments != "none" else None,
               "input_moments_rows": input_counts,
               "side_rows": side_counts if args.input_moments != "none" else None}
    with open(os.path.join(model_dir, f"{args.tag}_{args.seed_index}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"Total time is {time.time() - t0:.1f} seconds", flush=True)
    return summary


if __name__ == "__main__":
    main()
