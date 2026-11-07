"""Evaluate an AEV-PLIG ensemble on the FEP benchmark, the held-out series and the CASF test set.

Every model predicts every graph (pK via the scaler's inverse transform); the ensemble
prediction `pred` is the mean over models. Per-series metrics are Pearson, Spearman, Kendall
(scipy) and RMSE (pK); summaries are n-weighted and unweighted means over series with
n >= 10 (FEP; the AEV-PLIG paper's rule) and over all series. Outputs in --out-dir:
fep_per_series.csv, fep_summary.csv, series_valid_per_series.csv, series_valid_summary.csv,
test_summary.csv, predictions_{fep,series_valid,test}.csv, paper_check.csv (with --paper-csv)
and SUMMARY.md. Each dataset section runs only when its inputs are given.

Pose sensitivity (arm 4, --pose-pt processed/paired_poses_valid.pt: every PDBbind valid crystal
row with its near-native poses (kind 1) and decoys (kind 2), grouped by `parent`):
pose_summary.csv (one row: n_parents, decoy_rank_acc, decoy_gap_mean, nn_spread,
pearson_crystal, pearson_pos, pearson_neg, then n_parents_with_decoys, n_pos, n_neg) and
predictions_poses_valid.csv.

Augmented-row discrimination (--aug-pt processed/paired_aug_valid.pt: every PDBbind valid crystal
row with all its augmented rows, kinds 1-7, grouped by `parent`): aug_summary.csv, one row per kind
present (columns kind, role, ref_kind, n_parents, n_rows, rank_acc, gap_mean, spread, pearson):
  negative kinds 2 (decoy pose), 4 (shape twin), 6 (contact edit), ref_kind = the positive of the
    same scheme (2 -> 0 crystal, 4 -> 3 A-positive, 6 -> 5 C-positive): rank_acc = fraction of parents
    with both whose pred(positive) > max pred(negatives of the kind), gap_mean = mean of
    pred(positive) - mean pred(negatives), pK (a parent's several positive rows are averaged);
  control kinds 1, 3, 5, 7 (label kept): spread = mean |pred - pred(reference)| over rows whose parent
    has the reference row, pK; ref_kind 0 (crystal) for kinds 1, 3, 5 and 5 (C-positive, same edit
    pipeline) for kind 7;
  crystal kind 0: role crystal;
  pearson = label vs pred over the kind's rows (the parent's label for every augmented row);
  n_parents = parents entering rank_acc / spread (crystals for kind 0).
Plus predictions_aug_valid.csv.

Group-disjoint holdout (--group-valid-pt processed/paired_group_valid.pt, needs --meta):
group_valid_summary.csv (scope pointwise: Pearson / Spearman / Kendall / RMSE over all rows; then the
within-series means as series_valid_summary.csv: scopes n>=5 and all, weighted by n and unweighted),
group_valid_per_series.csv and predictions_group_valid.csv.

Data-dir guard: a model whose sidecar <stem>.json records args.data_dir must be evaluated on that build:
--series-valid-pt, --group-valid-pt and --aug-pt must be <data_dir>/processed/<file>, else exit with an
error (--allow-data-dir-mismatch turns it into a warning, e.g. for a reference model on another build).

Ligand-only models: --ligand-only zeroes node-feature columns 15: (the AEVs) on every graph
before prediction, as training's LigandOnly wrapper does. A model <stem>.model whose sibling
<stem>.json says ligand_only true (top level or under "args") switches this on automatically;
models that disagree are an error.

Siamese models (arm S): a model whose sidecar says siamese true, or whose state dict has the
wrapper's encoder.* / affinity.* keys, is rebuilt as heads.AffinityWithSiamese with its SiameseHead
(hidden width from the sidecar's siamese_hidden, else from the weights); its pointwise prediction is
unchanged. Every other model (no sidecar, e.g. the released ensemble, or no siamese flag) loads as a
plain GATv2Net exactly as before.

Pairwise ranking (--pairwise), for every dataset section that runs and has series: FEP by (subset,
series), series_valid and group_valid by series_key, CASF test by the meta's UniProt (rows without one
are singletons). Pairs: every i < j of a series (dataset order) with different lig_key where the meta
has that column (FEP: all pairs). Pair head: delta_hat_ij from each model's Siamese head on its pooled
embeddings, in pK (standardised delta times the scaler's scale), averaged over the models per pair;
point head: f_i - f_j with f the ensemble pointwise prediction. Per series (series with >= 1 pair; n =
ligands in at least one pair): pair_rmse = RMSE(delta - (y_i - y_j)), pK; pair_dir_acc = fraction of pairs
with |y_i - y_j| >= 0.5 pK whose sign agrees; kendall / spearman of y against the per-ligand score
s_i = mean_j delta_hat_ij (pair head; delta_hat_ji = -delta_hat_ij) or against f (point head).
Outputs pairwise_<dataset>_per_series.csv and pairwise_summary.csv (dataset, head in {pair, point},
scope, weighting, n_series, n_items, pair_rmse, pair_dir_acc, kendall, spearman: n-weighted and
unweighted means over series with n >= 10 (FEP) or n >= 5 / n >= 10 (the others) and over all), plus a
SUMMARY section. Without a Siamese head on every model only the point rows are written.

Pair head (arm P): a model whose sidecar says pair_head true, or whose state dict has pair.* keys, gets its
heads.PairHead (moments.* keys / sidecar moments: its MomentsHeads) in the same wrapper. --pairwise then adds,
when every model has a pair head, head pairhead rows to pairwise_summary.csv. l is averaged over the models per
ordered pair, then centred per anchor: l'(a, c) = l(a, c) - mean of l(a, .) over a's candidates in the series (the
ligands paired with a), so a per-anchor offset of the listwise softmax cancels. The per-ligand score
s_i = mean over the ligands a paired with i of l'(a, i) gives kendall / spearman, and pair_dir_acc is the sign of
s_i - s_j against y_i - y_j over pairs with |dy| >= 0.5 pK (pair_rmse NaN: l is not in pK); the per-series files
gain pair_dir_acc_pairhead, kendall_pairhead, spearman_pairhead. With --aug-pt, aug_summary.csv then gains a head
column (point for the rows above) and head pairhead rows per negative kind: per parent with a crystal row (the
anchor a) and per negative kind, l(a, .) over the candidates {positives of the same scheme, negatives of the kind}
is centred the same way; rank_acc = fraction of parents whose mean l'(a, positive) exceeds max l'(a, negative),
gap_mean = mean l'(a, positive) - mean l'(a, negatives) (logit units). For decoys (kind 2) the positive is the
crystal itself (l(z_a, z_a)). Models without a pair head write aug_summary.csv exactly as before.

Moments heads (--zernike-side paired_zernike.npz, build_zernike_side.py): for every section with row ids and
every target (pocket, ligand) that a model has a head for, moments_summary.csv rows (dataset, target, model in
{ensemble, 0, 1, ...}, n_models, n_rows, r2, r2_uniform, mse) over the kind-0 rows (side file kind) with that
descriptor: series_valid, test, group_valid, and valid (the kind-0 rows of aug_valid); every aug_valid row (kinds
0-7, parent / twin descriptors) only under dataset aug_valid_all. Values are in the side file's standardised
units: r2 = 1 - SS_res / SS_tot pooled over the 343 columns (SS_tot about the evaluated rows' column means),
r2_uniform = mean per-column R^2 (columns with SS_tot 0 skipped), mse = mean squared error; ensemble = the mean of
the models' predictions (models with that head). Ligand-only models report both targets. A model sidecar's
zernike_side different from --zernike-side is a warning.

Moments inputs (arm Z): a model whose sidecar says input_moments pocket | ligand | both | bothx is rebuilt with its
heads.MomentsInput (input_moments_proj; the enlarged fc1), one with model_arch (args.model) moments_only as a
heads.MomentsOnlyNet; both need data.moments on every batch, built as in training (heads.moments_input_matrix) from
descriptors in the side file's standardised units, i.e. with --zernike-side's TRAIN mean / sd for every dataset:
  series_valid, test, group_valid, poses, aug_valid   the side file's descriptors of the batch rows (data.row);
  FEP / 15-PGDH (--fep-csv)                           graph_id (the .pt's y) -> row graph_id of the processed CSV
                                                      (its row order, as make_predictions) -> unique_id -> the
                                                      descriptor in --zernike-eval FILE[,FILE...]
                                                      (featurise_zernike.py --csv; files merged by unique_id),
                                                      standardised with the side file's mean / sd (never the
                                                      eval set's own).
A row without a descriptor (not in the file, featurisation failed) gets zeros and the missing flag, as in training;
the counts are printed and the FEP section of SUMMARY.md states the coverage. Models without moments inputs ignore
all of this. When some model needs inputs, --zernike-side is required (and --zernike-eval when --fep-csv is given),
else exit; a sidecar's zernike_side_stats (sha1 of the training side file's mean / sd) that differs from
--zernike-side's is an error for such a model.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import json
import math
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

N_RDKIT = 15  # node feature columns 0:15 are RDKit atom features, 15: are AEVs
MIN_N_FEP = 10
MIN_N_SERIES_VALID = (5, 10)
METRICS = ["pearson", "spearman", "kendall", "rmse"]
POSE_METRICS = ["n_parents", "decoy_rank_acc", "decoy_gap_mean", "nn_spread", "pearson_crystal",
                "pearson_pos", "pearson_neg", "n_parents_with_decoys", "n_pos", "n_neg"]
# augmented kinds: negative kind -> the positive of the same scheme (0 = crystal)
NEGATIVE_POSITIVE = {2: 0, 4: 3, 6: 5}
CONTROL_KINDS = (1, 3, 5, 7)
CONTROL_REF = {1: 0, 3: 0, 5: 0, 7: 5}  # control kind -> the row its spread is measured against
AUG_COLUMNS = ["kind", "role", "ref_kind", "n_parents", "n_rows", "rank_acc", "gap_mean", "spread",
               "pearson"]
MIN_N_GROUP_VALID = 5
PAIR_DIR_MIN_PK = 0.5       # pair_dir_acc counts pairs with |y_i - y_j| >= this (pK)
PAIR_CHUNK = 8192           # pairs per Siamese-head call
PAIRWISE_MIN_NS = {"fep": (MIN_N_FEP,), "series_valid": MIN_N_SERIES_VALID, "group_valid": (5, 10),
                   "test": (5, 10)}
PAIRWISE_COLUMNS = ["n", "n_pairs", "n_dir_pairs", "pair_rmse", "pair_dir_acc", "kendall_pair",
                    "spearman_pair", "pair_rmse_point", "pair_dir_acc_point", "kendall_point",
                    "spearman_point"]
PAIRWISE_SUMMARY_COLUMNS = ["dataset", "head", "scope", "weighting", "n_series", "n_items", "pair_rmse",
                            "pair_dir_acc", "kendall", "spearman"]
PAIRWISE_HEAD_COLUMNS = {
    "pair": {"pair_rmse": "pair_rmse", "pair_dir_acc": "pair_dir_acc", "kendall": "kendall_pair",
             "spearman": "spearman_pair"},
    "point": {"pair_rmse": "pair_rmse_point", "pair_dir_acc": "pair_dir_acc_point",
              "kendall": "kendall_point", "spearman": "spearman_point"},
    "pairhead": {"pair_rmse": None, "pair_dir_acc": "pair_dir_acc_pairhead", "kendall": "kendall_pairhead",
                 "spearman": "spearman_pairhead"},
}
PAIRHEAD_COLUMNS = ["pair_dir_acc_pairhead", "kendall_pairhead", "spearman_pairhead"]  # appended with a pair head
MOMENTS_COLUMNS = ["dataset", "target", "model", "n_models", "n_rows", "r2", "r2_uniform", "mse"]
MOMENTS_TARGETS = ("pocket", "ligand")
INPUT_MOMENTS_MODES = ("none", "pocket", "ligand", "both", "bothx")  # heads.INPUT_MOMENTS_CHOICES
UNMAPPED_UNIPROT = "------"
DEFAULT_INDEX_TGZ = "/scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_plain_text_index.tar.gz"


# ----------------------------------------------------------------------------- metrics

def series_metrics(truth, pred):
    """n, Pearson, Spearman, Kendall (NaN below 2 items or for constant input) and RMSE."""
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = {"n": int(len(truth)), "pearson": math.nan, "spearman": math.nan, "kendall": math.nan,
           "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))) if len(truth) else math.nan}
    if len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(pred) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["pearson"] = float(stats.pearsonr(truth, pred)[0])
            out["spearman"] = float(stats.spearmanr(truth, pred)[0])
            out["kendall"] = float(stats.kendalltau(truth, pred)[0])
    return out


def per_series_metrics(df, group_cols, truth="truth", pred="pred"):
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **series_metrics(g[truth], g[pred])})
    return pd.DataFrame(rows, columns=list(group_cols) + ["n"] + METRICS)


def weighted_mean(values, weights):
    """Mean of the finite values, weighted; NaN when none is finite."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else math.nan


def _pearson(truth, pred):
    """scipy Pearson r; NaN below 2 items or for constant input."""
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(pred) == 0:
        return math.nan
    return float(stats.pearsonr(truth, pred)[0])


def pose_metrics(df):
    """Pose-sensitivity metrics from rows with columns kind (0 crystal, 1 pos, 2 decoy), parent
    (the crystal's row), truth, pred (pK), grouped by parent:
      n_parents        parents with a crystal row
      decoy_rank_acc   fraction of parents (with >= 1 decoy) where pred(crystal) > max pred(decoys)
      decoy_gap_mean   mean over those parents of pred(crystal) - mean pred(decoys)
      nn_spread        mean over all pos rows of |pred(pos) - pred(crystal)|
      pearson_crystal / pearson_pos / pearson_neg   label vs pred over the crystal / pos / decoy rows
    Pose rows whose parent has no crystal row enter only pearson_pos / pearson_neg."""
    kind = df["kind"].to_numpy()
    crystal = df[kind == 0]
    if crystal["parent"].duplicated().any():
        raise ValueError("more than one crystal (kind 0) row per parent")
    crystal_pred = crystal.set_index("parent")["pred"]
    pos, neg = df[kind == 1], df[kind == 2]
    neg = neg.assign(crystal=neg["parent"].map(crystal_pred)).dropna(subset=["crystal"])
    by_parent = neg.groupby("parent").agg(crystal=("crystal", "first"), best=("pred", "max"),
                                          mean=("pred", "mean"))
    pos_c = pos.assign(crystal=pos["parent"].map(crystal_pred)).dropna(subset=["crystal"])
    nan = math.nan
    return {
        "n_parents": int(len(crystal)),
        "decoy_rank_acc": float((by_parent["crystal"] > by_parent["best"]).mean()) if len(by_parent) else nan,
        "decoy_gap_mean": float((by_parent["crystal"] - by_parent["mean"]).mean()) if len(by_parent) else nan,
        "nn_spread": float((pos_c["pred"] - pos_c["crystal"]).abs().mean()) if len(pos_c) else nan,
        "pearson_crystal": _pearson(crystal["truth"], crystal["pred"]),
        "pearson_pos": _pearson(pos["truth"], pos["pred"]),
        "pearson_neg": _pearson(df.loc[kind == 2, "truth"], df.loc[kind == 2, "pred"]),
        "n_parents_with_decoys": int(len(by_parent)),
        "n_pos": int(len(pos)),
        "n_neg": int((kind == 2).sum()),
    }


def aug_metrics(df):
    """Per-kind discrimination metrics (aug_summary.csv rows, AUG_COLUMNS) from rows with columns
    kind, parent, truth, pred (pK) of a crystal + augmented-rows split; see the module docstring."""
    nan = math.nan
    kind = df["kind"].to_numpy()
    crystal = df[kind == 0]
    if crystal["parent"].duplicated().any():
        raise ValueError("more than one crystal (kind 0) row per parent")
    rows = []
    for k in sorted(int(v) for v in np.unique(kind)):
        sub = df[kind == k]
        row = {"kind": k, "role": "crystal", "ref_kind": pd.NA, "n_parents": int(len(sub)),
               "n_rows": int(len(sub)), "rank_acc": nan, "gap_mean": nan, "spread": nan,
               "pearson": _pearson(sub["truth"], sub["pred"])}
        if k in NEGATIVE_POSITIVE:
            ref = NEGATIVE_POSITIVE[k]
            pos = df[kind == ref].groupby("parent")["pred"].mean()
            neg = sub.groupby("parent")["pred"].agg(["max", "mean"])
            both = neg.join(pos.rename("pos"), how="inner")
            row.update(role="negative", ref_kind=ref, n_parents=int(len(both)))
            if len(both):
                row["rank_acc"] = float((both["pos"] > both["max"]).mean())
                row["gap_mean"] = float((both["pos"] - both["mean"]).mean())
        elif k in CONTROL_KINDS:
            ref_k = CONTROL_REF[k]
            ref = df[kind == ref_k].groupby("parent")["pred"].mean()
            c = sub.assign(ref=sub["parent"].map(ref)).dropna(subset=["ref"])
            row.update(role="control", ref_kind=ref_k, n_parents=int(c["parent"].nunique()))
            if len(c):
                row["spread"] = float((c["pred"] - c["ref"]).abs().mean())
        elif k != 0:
            row["role"] = "other"
        rows.append(row)
    out = pd.DataFrame(rows, columns=AUG_COLUMNS)
    out["ref_kind"] = out["ref_kind"].astype("Int64")
    return out


def aug_pair_candidates(df):
    """(anchor, candidate) row positions of df (kind, parent) for the pair-head aug rows: per parent with a
    crystal row (kind 0, the anchor), every row of a negative kind present and of its positive kind (the
    crystal itself for kind 2) of that parent."""
    kind = df["kind"].to_numpy()
    parent = df["parent"].to_numpy()
    crystal = df[kind == 0]
    if crystal["parent"].duplicated().any():
        raise ValueError("more than one crystal (kind 0) row per parent")
    anchor_of = dict(zip(crystal["parent"].to_numpy(), np.flatnonzero(kind == 0)))
    negs = [k for k in NEGATIVE_POSITIVE if (kind == k).any()]
    wanted = set(negs) | {NEGATIVE_POSITIVE[k] for k in negs}
    cand = [p for p in range(len(df)) if kind[p] in wanted and parent[p] in anchor_of]
    a = np.array([anchor_of[parent[p]] for p in cand], dtype=np.int64)
    return a, np.array(cand, dtype=np.int64)


def aug_pairhead_metrics(df, c, score):
    """head pairhead rows of aug_summary.csv (AUG_COLUMNS + head) from df (kind, parent), the candidate positions
    c of aug_pair_candidates and their scores l(z_anchor, z_c): per negative kind present, the scores of each
    parent's candidates {positives of the scheme, negatives of the kind} are centred on their mean (l'); n_parents
    (parents with both a positive and a negative scored), n_rows (negative rows scored), rank_acc (mean positive
    l' > max negative l') and gap_mean (mean positive l' - mean negative l')."""
    nan = math.nan
    sc = pd.DataFrame({"kind": df["kind"].to_numpy()[c], "parent": df["parent"].to_numpy()[c],
                       "score": np.asarray(score, dtype=float)})
    rows = []
    for k in sorted(NEGATIVE_POSITIVE):
        if not (df["kind"] == k).any():
            continue
        ref = NEGATIVE_POSITIVE[k]
        cands = sc[sc["kind"].isin([ref, k])]
        cands = cands.assign(score=cands["score"] - cands.groupby("parent")["score"].transform("mean"))
        pos = cands[cands["kind"] == ref].groupby("parent")["score"].mean()
        neg_rows = cands[cands["kind"] == k]
        neg = neg_rows.groupby("parent")["score"].agg(["max", "mean"])
        both = neg.join(pos.rename("pos"), how="inner")
        row = {"kind": k, "role": "negative", "ref_kind": ref, "n_parents": int(len(both)),
               "n_rows": int(len(neg_rows)), "rank_acc": nan, "gap_mean": nan, "spread": nan, "pearson": nan,
               "head": "pairhead"}
        if len(both):
            row["rank_acc"] = float((both["pos"] > both["max"]).mean())
            row["gap_mean"] = float((both["pos"] - both["mean"]).mean())
        rows.append(row)
    out = pd.DataFrame(rows, columns=AUG_COLUMNS + ["head"])
    out["ref_kind"] = out["ref_kind"].astype("Int64")
    return out


def moments_r2(pred, target):
    """(r2 pooled over columns, r2_uniform = mean per-column R^2 over columns with variance, mse) of
    pred / target [n, d] (NaN for n < 2)."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if len(target) < 2:
        return math.nan, math.nan, (float(np.mean((pred - target) ** 2)) if len(target) else math.nan)
    res = ((pred - target) ** 2).sum(axis=0)
    tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = float(1.0 - res.sum() / tot.sum()) if tot.sum() > 0 else math.nan
    ok = tot > 0
    r2u = float(np.mean(1.0 - res[ok] / tot[ok])) if ok.any() else math.nan
    return r2, r2u, float(np.mean((pred - target) ** 2))


def summarise(per_series, min_ns=(MIN_N_FEP,)):
    """Weighted (weights = n) and unweighted means over series with n >= each min_n, and over all."""
    rows = []
    scopes = [(f"n>={m}", per_series["n"] >= m) for m in min_ns] + [("all", per_series["n"] >= 1)]
    for scope, sel in scopes:
        sub = per_series[sel]
        for weighting in ("weighted", "unweighted"):
            w = sub["n"] if weighting == "weighted" else np.ones(len(sub))
            row = {"scope": scope, "weighting": weighting, "n_series": int(len(sub)),
                   "n_items": int(sub["n"].sum())}
            row.update({m: weighted_mean(sub[m], w) for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows, columns=["scope", "weighting", "n_series", "n_items"] + METRICS)


# ----------------------------------------------------------------------------- pairwise metrics

def _group_ids(df, group_cols):
    """Dense group index per row, groups in sorted key order."""
    return df.groupby(list(group_cols), sort=True, dropna=False).ngroup().to_numpy()


def _lig_codes(lig):
    """Integer code per row; missing / empty keys get a code of their own (never equal to another)."""
    values = pd.Series(np.asarray(lig, dtype=object)).replace("", np.nan)
    codes, _ = pd.factorize(values)
    codes = codes.astype(np.int64)
    missing = codes < 0
    codes[missing] = -2 - np.arange(int(missing.sum()))
    return codes


def series_pairs(df, group_cols, lig=None):
    """(i, j) row positions of every pair i < j inside each series of df (grouped by group_cols), with
    different ligand keys when lig (one key per row) is given; ordered by series (sorted keys), then i, j."""
    gid = _group_ids(df, group_cols)
    codes = _lig_codes(lig) if lig is not None else None
    order = np.argsort(gid, kind="stable")
    _, starts = np.unique(gid[order], return_index=True)
    out_i, out_j = [], []
    for idx in np.split(order, starts[1:]) if len(order) else []:
        if len(idx) < 2:
            continue
        idx = np.sort(idx)
        a, b = np.triu_indices(len(idx), k=1)
        pi, pj = idx[a], idx[b]
        if codes is not None:
            keep = codes[pi] != codes[pj]
            pi, pj = pi[keep], pj[keep]
        out_i.append(pi)
        out_j.append(pj)
    if not out_i:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64)


def _dir_acc(d, dy, mask):
    return float(np.mean(np.sign(d[mask]) == np.sign(dy[mask]))) if mask.any() else math.nan


def _rmse(r):
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else math.nan


def pairwise_per_series(df, group_cols, i, j, delta=None, dir_min=PAIR_DIR_MIN_PK, pairhead=None):
    """Per-series pairwise metrics (columns group_cols + PAIRWISE_COLUMNS) from df (truth, pred in pK),
    the pairs (i, j) (row positions, same series) and delta: the pair head's delta_hat per pair in pK,
    or None (pair columns NaN). Only series with at least one pair appear; n = ligands in a pair.
    pairhead: (l_ij, l_ji), the pair head's l(z_i, z_j) and l(z_j, z_i) per pair, adds PAIRHEAD_COLUMNS: with
    l'(a, c) = l(a, c) - mean of l(a, .) over a's pairs in the series, kendall / spearman of y against
    s_i = mean over i's pairs of l'(other, i), and pair_dir_acc of s_i - s_j against y_i - y_j."""
    y = df["truth"].to_numpy(dtype=float)
    f = df["pred"].to_numpy(dtype=float)
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    if delta is not None:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != i.shape:
            raise ValueError(f"{delta.shape[0]} delta values for {len(i)} pairs")
    if pairhead is not None:
        l_ij, l_ji = (np.asarray(v, dtype=float) for v in pairhead)
        if l_ij.shape != i.shape or l_ji.shape != i.shape:
            raise ValueError(f"{l_ij.shape} / {l_ji.shape} pair-head scores for {len(i)} pairs")
    gid = _group_ids(df, group_cols)
    if np.any(gid[i] != gid[j]):
        raise ValueError("pairs must not cross series")
    order = np.argsort(gid[i], kind="stable")
    i, j, pg = i[order], j[order], gid[i][order]
    if delta is not None:
        delta = delta[order]
    if pairhead is not None:
        l_ij, l_ji = l_ij[order], l_ji[order]
    groups, starts = np.unique(pg, return_index=True)
    stops = np.append(starts[1:], len(pg))
    nan = math.nan
    rows = []
    for start, stop in zip(starts, stops):
        a, b = i[start:stop], j[start:stop]
        dy = y[a] - y[b]
        dirm = np.abs(dy) >= dir_min
        items = np.unique(np.concatenate([a, b]))
        point = series_metrics(y[items], f[items])
        row = {**{c: df[c].iloc[a[0]] for c in group_cols}, "n": int(len(items)), "n_pairs": int(len(a)),
               "n_dir_pairs": int(dirm.sum()), "pair_rmse": nan, "pair_dir_acc": nan, "kendall_pair": nan,
               "spearman_pair": nan, "pair_rmse_point": _rmse((f[a] - f[b]) - dy),
               "pair_dir_acc_point": _dir_acc(f[a] - f[b], dy, dirm), "kendall_point": point["kendall"],
               "spearman_point": point["spearman"]}
        if delta is not None:
            d = delta[start:stop]
            la, lb = np.searchsorted(items, a), np.searchsorted(items, b)
            k = len(items)
            score = ((np.bincount(la, weights=d, minlength=k) - np.bincount(lb, weights=d, minlength=k))
                     / (np.bincount(la, minlength=k) + np.bincount(lb, minlength=k)))
            pair = series_metrics(y[items], score)
            row.update(pair_rmse=_rmse(d - dy), pair_dir_acc=_dir_acc(d, dy, dirm),
                       kendall_pair=pair["kendall"], spearman_pair=pair["spearman"])
        if pairhead is not None:
            lij, lji = l_ij[start:stop], l_ji[start:stop]
            la, lb = np.searchsorted(items, a), np.searchsorted(items, b)
            k = len(items)
            count = np.bincount(la, minlength=k) + np.bincount(lb, minlength=k)
            # anchor means: l(i, j) has anchor i (= a), l(j, i) has anchor j (= b)
            centre = (np.bincount(la, weights=lij, minlength=k) + np.bincount(lb, weights=lji, minlength=k)) / count
            cij, cji = lij - centre[la], lji - centre[lb]
            score = (np.bincount(la, weights=cji, minlength=k) + np.bincount(lb, weights=cij, minlength=k)) / count
            ph = series_metrics(y[items], score)
            row.update(pair_dir_acc_pairhead=_dir_acc(score[la] - score[lb], dy, dirm),
                       kendall_pairhead=ph["kendall"], spearman_pairhead=ph["spearman"])
        rows.append(row)
    columns = list(group_cols) + PAIRWISE_COLUMNS + (PAIRHEAD_COLUMNS if pairhead is not None else [])
    return pd.DataFrame(rows, columns=columns)


def pairwise_summary(per, dataset, min_ns=(MIN_N_FEP,), with_pair=True, with_pairhead=False):
    """pairwise_summary.csv rows of one dataset: per head (pair only when with_pair, then point, then pairhead
    when with_pairhead), scope (n >= each min_n, then all) and weighting (weighted by n, unweighted), the means
    of the head's pair_rmse / pair_dir_acc / kendall / spearman over the scope's series (pairhead: pair_rmse NaN)."""
    rows = []
    scopes = [(f"n>={m}", per["n"] >= m) for m in min_ns] + [("all", per["n"] >= 1)]
    heads_out = (("pair", "point") if with_pair else ("point",)) + (("pairhead",) if with_pairhead else ())
    for head in heads_out:
        for scope, sel in scopes:
            sub = per[sel]
            for weighting in ("weighted", "unweighted"):
                w = sub["n"] if weighting == "weighted" else np.ones(len(sub))
                row = {"dataset": dataset, "head": head, "scope": scope, "weighting": weighting,
                       "n_series": int(len(sub)), "n_items": int(sub["n"].sum())}
                row.update({k: weighted_mean(sub[c], w) if c else math.nan
                            for k, c in PAIRWISE_HEAD_COLUMNS[head].items()})
                rows.append(row)
    return pd.DataFrame(rows, columns=PAIRWISE_SUMMARY_COLUMNS)


# ----------------------------------------------------------------------------- UniProt lookup

def read_pdbbind_index(tgz_path):
    from build_paired_dataset import read_pdbbind_index as _read  # Task 1's reader

    return _read(tgz_path)


def train_rows_same_uniprot(ref_pdbs, meta, name_index):
    """(uniprot per ref_pdb, count of meta train rows with that UniProt); '' and -1 when the
    ref_pdb is not in the PDBbind name index or has no UniProt there. Pose rows (kind != 0, arm 4)
    are not counted, so the column is the same for a build with and without poses."""
    train = meta[meta["split"] == "train"]
    if "kind" in train:
        train = train[train["kind"] == 0]
    counts = train["uniprot"].fillna("").astype(str).value_counts()
    uniprots, out = [], []
    for ref in ref_pdbs:
        u = name_index.get(str(ref).strip().lower())
        if not u or u == UNMAPPED_UNIPROT:
            uniprots.append("")
            out.append(-1)
        else:
            uniprots.append(u)
            out.append(int(counts.get(u, 0)))
    return uniprots, out


# ----------------------------------------------------------------------------- prediction

def load_pt(path):
    """Load any InMemoryDataset collate file (GraphDatasetPredict or PairedDataset format)."""
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    torch.serialization.add_safe_globals([Data])

    class _Pt(InMemoryDataset):
        def __init__(self, pt_path):
            super().__init__(None)
            self.load(pt_path)

    return _Pt(path)


def sidecar_ligand_only(model_path):
    """ligand_only from <stem>.json next to <stem>.model, or None without a sidecar/flag."""
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        side = json.load(fh)
    if "ligand_only" in side:
        return bool(side["ligand_only"])
    if isinstance(side.get("args"), dict) and "ligand_only" in side["args"]:
        return bool(side["args"]["ligand_only"])
    return None


def sidecar_data_dir(model_path):
    """args.data_dir from <stem>.json next to <stem>.model, or None."""
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        side = json.load(fh)
    args = side.get("args")
    return str(args["data_dir"]) if isinstance(args, dict) and args.get("data_dir") else None


def sidecar_siamese(model_path):
    """{"siamese_hidden": int | None} when <stem>.json says siamese true (top level or under "args"),
    else None (no sidecar, no flag, or false)."""
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        side = json.load(fh)
    args = side.get("args") if isinstance(side.get("args"), dict) else {}
    if not (side.get("siamese", args.get("siamese", False))):
        return None
    hidden = side.get("siamese_hidden", args.get("siamese_hidden"))
    return {"siamese_hidden": None if hidden is None else int(hidden)}


def sidecar_heads(model_path):
    """The heads a model's sidecar declares: {"siamese": bool, "siamese_hidden", "pair_head": bool, "pair_hidden",
    "moments": tuple of targets, "moments_dim", "zernike_side"} (top level, else under "args"), or None without a
    sidecar."""
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        side = json.load(fh)
    args = side.get("args") if isinstance(side.get("args"), dict) else {}

    def get(key, default=None):
        return side.get(key, args.get(key, default))

    moments = get("moments", "none") or "none"
    targets = {"none": (), "pocket": ("pocket",), "ligand": ("ligand",), "both": ("pocket", "ligand")}.get(moments)
    if targets is None:
        raise ValueError(f"{path}: unknown moments {moments!r}")
    as_int = lambda v: None if v is None else int(v)  # noqa: E731
    input_moments = get("input_moments", "none") or "none"
    if input_moments not in INPUT_MOMENTS_MODES:
        raise ValueError(f"{path}: unknown input_moments {input_moments!r}")
    # the top-level "model" of a training sidecar is the checkpoint path: the architecture is model_arch / args.model
    arch = side.get("model_arch") or args.get("model") or "gatv2"
    if arch not in ("gatv2", "moments_only"):
        raise ValueError(f"{path}: unknown model_arch {arch!r}")
    if arch == "moments_only" and input_moments == "none":
        raise ValueError(f"{path}: model_arch moments_only without input_moments")
    return {"siamese": bool(get("siamese", False)), "siamese_hidden": as_int(get("siamese_hidden")),
            "pair_head": bool(get("pair_head", False)), "pair_hidden": as_int(get("pair_hidden")),
            "moments": targets, "moments_dim": as_int(side.get("moments_dim")),
            "zernike_side": side.get("zernike_side"), "zernike_side_stats": side.get("zernike_side_stats"),
            "model_arch": arch, "input_moments": input_moments,
            "input_moments_proj": as_int(get("input_moments_proj")), "input_dim": as_int(side.get("input_dim"))}


def check_data_dirs(model_paths, pts, allow):
    """Each given pt (flag -> path) must sit at <data_dir>/processed/<file> for the data_dir every model
    was trained on (sidecar args.data_dir; models without one are not checked). Exit on a mismatch unless
    allow, then warn."""
    trained = {p: sidecar_data_dir(p) for p in model_paths}
    problems = []
    for flag, pt in pts.items():
        if not pt:
            continue
        where = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(pt))))
        for model, data_dir in trained.items():
            if data_dir is not None and os.path.realpath(data_dir) != where:
                problems.append(f"{flag} {pt} belongs to {where} but {os.path.basename(model)} was trained "
                                f"on {data_dir}")
    if not problems:
        return
    msg = "data-dir mismatch: " + "; ".join(problems)
    if allow:
        print(f"WARNING: {msg} (allowed by --allow-data-dir-mismatch)")
        return
    sys.exit(f"ERROR: {msg}. Evaluate on the training build's processed/ files or pass "
             "--allow-data-dir-mismatch")


def resolve_ligand_only(model_paths, cli_flag):
    flags = {p: sidecar_ligand_only(p) for p in model_paths}
    known = {f for f in flags.values() if f is not None}
    if len(known) > 1:
        sys.exit("ERROR: models disagree on ligand_only: "
                 + ", ".join(f"{os.path.basename(p)}={f}" for p, f in flags.items()))
    if True in known:
        print("ligand-only: model sidecars say ligand_only=true; zeroing node-feature columns 15: "
              "on every graph")
        return True
    if cli_flag:
        if False in known:
            print("WARNING: --ligand-only given for models trained with ligand_only=false")
        print("ligand-only: --ligand-only given; zeroing node-feature columns 15: on every graph")
        return True
    return False


def load_models(paths, node_dim, edge_dim, hidden_dim, head, device):
    """One model per path: a plain GATv2Net, heads.AffinityWithSiamese for a model with heads or moments inputs
    (sidecar siamese / pair_head true, moments or input_moments not none, or wrapper keys in the state dict), or
    heads.MomentsOnlyNet (sidecar model_arch moments_only); see the module docstring."""
    import torch
    from model_defs import GATv2Net

    cfg = SimpleNamespace(hidden_dim=hidden_dim, head=head, activation_function="leaky_relu")
    models = []
    for path in paths:
        state = torch.load(path, map_location=device, weights_only=True)
        try:
            side = sidecar_heads(path)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        if side is not None and side["model_arch"] == "moments_only":
            models.append(_load_moments_only(state, side, path))
            continue
        model = GATv2Net(node_feature_dim=node_dim, edge_feature_dim=edge_dim, config=cfg)
        declared = side is not None and (side["siamese"] or side["pair_head"] or bool(side["moments"])
                                         or side["input_moments"] != "none")
        if not declared and not any(k.startswith(("encoder.", "affinity.")) for k in state):
            if not any(k.startswith(("GNN_layers.", "BN_layers.")) for k in state):
                sys.exit(f"ERROR: {os.path.basename(path)}: not a GATv2Net state dict (a moments_only model needs "
                         "its sidecar's model_arch)")
            model.load_state_dict(state)  # plain GATv2Net, as before
        else:
            model = _load_wrapper(model, state, side, path)
        models.append(model)
    return models


def _load_moments_only(state, side, path):
    import heads

    name = os.path.basename(path)
    shape = heads.moments_only_from_state_dict(state)
    if shape is None:
        sys.exit(f"ERROR: {name}: the sidecar says model_arch moments_only but the state dict is not a MomentsOnlyNet")
    d_in, hidden = shape
    mode = side["input_moments"]
    if heads.input_moments_dim(mode) != d_in or (side["input_dim"] is not None and side["input_dim"] != d_in):
        sys.exit(f"ERROR: {name}: input_moments {mode} (input_dim {side['input_dim']}) but the weights take {d_in} "
                 "inputs")
    model = heads.MomentsOnlyNet(d_in, hidden=hidden, mode=mode)
    model.load_state_dict(state)
    return model


def _load_wrapper(gnn, state, side, path):
    import heads

    name = os.path.basename(path)
    side = side or {"siamese": False, "siamese_hidden": None, "pair_head": False, "pair_hidden": None,
                    "moments": (), "moments_dim": None, "input_moments": "none", "input_moments_proj": None,
                    "input_dim": None}
    flags = [f for f, on in (("siamese true", side["siamese"]), ("pair_head true", side["pair_head"]),
                             (f"moments {'+'.join(side['moments'])}", bool(side["moments"])),
                             (f"input_moments {side['input_moments']}", side["input_moments"] != "none")) if on]
    if not heads.is_wrapped_state_dict(state):
        sys.exit(f"ERROR: {name}: the sidecar says {', '.join(flags)} but the state dict is a plain GATv2Net")
    hidden = heads.siamese_hidden_from_state_dict(state)
    if side["siamese"] and hidden is None:
        sys.exit(f"ERROR: {name}: the sidecar says siamese true but the state dict has no siamese.* keys")
    if side["siamese"] and side["siamese_hidden"] is not None and side["siamese_hidden"] != hidden:
        sys.exit(f"ERROR: {name}: sidecar siamese_hidden {side['siamese_hidden']} but the weights have {hidden}")
    pair_hidden = heads.pair_hidden_from_state_dict(state)
    if side["pair_head"] and pair_hidden is None:
        sys.exit(f"ERROR: {name}: the sidecar says pair_head true but the state dict has no pair.* keys")
    if side["pair_head"] and side["pair_hidden"] is not None and side["pair_hidden"] != pair_hidden:
        sys.exit(f"ERROR: {name}: sidecar pair_hidden {side['pair_hidden']} but the weights have {pair_hidden}")
    try:
        moments = heads.moments_from_state_dict(state)
    except KeyError as exc:
        sys.exit(f"ERROR: {name}: {exc}")
    absent = [t for t in side["moments"] if t not in moments]
    if absent:
        sys.exit(f"ERROR: {name}: the sidecar says moments {'+'.join(side['moments'])} but the state dict has no "
                 f"moments.{absent[0]}.* keys")
    for t, (_, out) in moments.items():
        if side["moments_dim"] is not None and side["moments_dim"] != out:
            sys.exit(f"ERROR: {name}: sidecar moments_dim {side['moments_dim']} but the {t} head outputs {out}")
    dim = heads.embed_dim(gnn)
    moments_in = _moments_input(state, side, name, dim)
    siamese = None if hidden is None else heads.SiameseHead(dim, hidden=hidden)
    pair = None if pair_hidden is None else heads.PairHead(dim, hidden=pair_hidden)
    mheads = {t: heads.MomentsHead(dim, out=out, hidden=h) for t, (h, out) in moments.items()}
    model = heads.AffinityWithSiamese(gnn, siamese, pair, mheads or None, moments_in)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {name}: the state dict does not match the rebuilt model: {str(exc)[:500]}")
    return model


def _moments_input(state, side, name, dim):
    """The heads.MomentsInput a wrapper state dict needs (None without moments inputs), checked against the sidecar's
    input_moments / input_moments_proj / input_dim and the weights (moments_in.fc, the enlarged affinity.fc1)."""
    import heads

    mode = side["input_moments"]
    fc1_in = heads.fc1_input_width(state)
    proj_w = heads.input_moments_from_state_dict(state)
    if mode == "none":
        if proj_w is not None or fc1_in != dim:
            sys.exit(f"ERROR: {name}: the weights take moments inputs (fc1 input {fc1_in}, embedding {dim}) but the "
                     "sidecar has no input_moments (moments inputs need the sidecar)")
        return None
    d_in = heads.input_moments_dim(mode)
    proj = side["input_moments_proj"]
    if proj is None:
        proj = proj_w[1] if proj_w is not None else 0
    if side["input_dim"] is not None and side["input_dim"] != d_in:
        sys.exit(f"ERROR: {name}: sidecar input_dim {side['input_dim']} but input_moments {mode} has {d_in}")
    if (proj > 0) != (proj_w is not None) or (proj_w is not None and proj_w != (d_in, proj)):
        sys.exit(f"ERROR: {name}: sidecar input_moments {mode} / proj {proj} (d_in {d_in}) but the weights have "
                 f"{'no projection' if proj_w is None else f'a projection {proj_w[0]} -> {proj_w[1]}'}")
    moments_in = heads.MomentsInput(d_in, proj=proj, mode=mode)
    if fc1_in != dim + moments_in.out_dim:
        sys.exit(f"ERROR: {name}: fc1 takes {fc1_in} inputs, expected {dim} + {moments_in.out_dim}")
    return moments_in


def has_siamese(model):
    return getattr(model, "siamese", None) is not None


def input_mode(model):
    """The --input-moments mode a model reads from data.moments ("none" for every other model)."""
    return getattr(model, "input_moments", "none") or "none"


def has_pair_head(model):
    return getattr(model, "pair", None) is not None


def moments_targets(model):
    return tuple(getattr(model, "moments_targets", ()))


def embed_dataset(model, dataset, device, ligand_only=False, batch_size=256):
    """z [n, D] (CPU) of every graph of dataset in order (eval mode; ligand-only masking as predict_models)."""
    import torch
    from torch_geometric.loader import DataLoader

    model.eval()
    model.to(device)
    zs = []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            data = data.to(device)
            if ligand_only:
                x = data.x.clone()
                x[:, N_RDKIT:] = 0
                data.x = x
            zs.append(model.embed(data).cpu())
    return torch.cat(zs) if zs else torch.zeros(0)


def section_embeddings(ctx, models, ds, batch_size, which):
    """{k: z} for the model indices `which`, computed once per dataset object (cached in ctx until the next
    dataset)."""
    cache = ctx.setdefault("z_cache", {})
    if cache.get("ds") is not ds:
        cache.clear()
        cache.update(ds=ds, z={})
    for k in which:
        if k not in cache["z"]:
            cache["z"][k] = embed_dataset(models[k], ds, ctx["device"], ctx["ligand_only"], batch_size)
    return {k: cache["z"][k] for k in which}


def pairhead_scores(ctx, models, ds, a, c, batch_size=256, chunk=PAIR_CHUNK):
    """[n_models, n_pairs] l(z_a, z_c) of every model's pair head on the pairs (a, c) of dataset positions (all
    models must have one)."""
    import torch

    which = [k for k, m in enumerate(models) if has_pair_head(m)]
    if len(which) != len(models):
        raise ValueError("pairhead_scores needs a pair head on every model")
    zs = section_embeddings(ctx, models, ds, batch_size, which)
    aa = torch.as_tensor(np.asarray(a, dtype=np.int64))
    cc = torch.as_tensor(np.asarray(c, dtype=np.int64))
    out = []
    for k in which:
        model, z = models[k], zs[k].to(ctx["device"])
        with torch.no_grad():
            parts = [model.pair(z[aa[s:s + chunk].to(ctx["device"])], z[cc[s:s + chunk].to(ctx["device"])]).cpu()
                     for s in range(0, len(aa), chunk)]
        out.append(torch.cat(parts).double().numpy() if parts else np.zeros(0))
    return np.vstack(out)


def pair_deltas(models, scaler, dataset, i, j, device, ligand_only=False, batch_size=256, chunk=PAIR_CHUNK):
    """[n_models_with_head, n_pairs] delta_hat (pK) of every model that has a Siamese head on the pairs
    (i, j) of dataset positions, from its embeddings of the (ligand-only masked, if set) graphs; None
    when no model has a head."""
    import torch
    from torch_geometric.loader import DataLoader

    scale = float(scaler.scale_[0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ii = torch.as_tensor(np.asarray(i, dtype=np.int64))
    jj = torch.as_tensor(np.asarray(j, dtype=np.int64))
    out = []
    for model in models:
        if not has_siamese(model):
            continue
        model.eval()
        model.to(device)
        with torch.no_grad():
            zs = []
            for data in loader:
                data = data.to(device)
                if ligand_only:
                    x = data.x.clone()
                    x[:, N_RDKIT:] = 0
                    data.x = x
                zs.append(model.embed(data))
            z = torch.cat(zs)
            parts = [model.siamese(z[ii[k:k + chunk].to(device)], z[jj[k:k + chunk].to(device)]).cpu()
                     for k in range(0, len(ii), chunk)]
        d = torch.cat(parts).double().numpy() if parts else np.zeros(0)
        out.append(d * scale)
    return np.vstack(out) if out else None


class MomentsSource:
    """Standardised pocket / ligand descriptors (the side file's units) per dataset item, keyed by an integer batch
    attribute (`row` for the paired splits, `y` = graph_id for FEP / 15-PGDH), for the moments inputs of arm-Z
    models: batch(data, mode) -> data.moments [B, d_in] (heads.moments_input_matrix; an item whose key is unknown or
    whose descriptor is missing gets zeros and the missing flag)."""

    def __init__(self, key, keys, pocket, ligand, mask_pocket, mask_ligand, label):
        keys = np.asarray(keys, dtype=np.int64)
        if len(keys) and keys.min() < 0:
            raise ValueError(f"{label}: negative {key} keys")
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"{label}: duplicate {key} keys")
        self.key, self.label = key, label
        self.lookup = np.full(int(keys.max()) + 1 if len(keys) else 0, -1, dtype=np.int64)
        self.lookup[keys] = np.arange(len(keys))
        self.pocket = np.ascontiguousarray(pocket, dtype=np.float32).reshape(len(keys), -1)
        self.ligand = np.ascontiguousarray(ligand, dtype=np.float32).reshape(len(keys), -1)
        self.mask_pocket = np.asarray(mask_pocket, dtype=bool).reshape(-1)
        self.mask_ligand = np.asarray(mask_ligand, dtype=bool).reshape(-1)
        if not len(keys):  # one masked dummy item, so every lookup (all misses) can index the arrays
            self.pocket = np.zeros((1, self.pocket.shape[1] or 343), dtype=np.float32)
            self.ligand = np.zeros((1, self.ligand.shape[1] or 343), dtype=np.float32)
            self.mask_pocket = self.mask_ligand = np.zeros(1, dtype=bool)

    @classmethod
    def from_side(cls, side):
        """Row-keyed source over every row of a load_side() dict (its standardised arrays, masked rows 0)."""
        return cls("row", side["row"], side["pocket_std"], side["ligand_std"], side["mask_pocket"],
                   side["mask_ligand"], "side file")

    def index(self, keys):
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        ok = (keys >= 0) & (keys < len(self.lookup))
        out = np.full(keys.shape, -1, dtype=np.int64)
        out[ok] = self.lookup[keys[ok]]
        return out

    def matrix(self, keys, mode):
        """[len(keys), d_in] float32 CPU tensor of the moments input of mode for the items keyed by keys."""
        import torch

        import heads

        idx = self.index(keys)
        found = idx >= 0
        safe = np.where(found, idx, 0)
        targets = heads.input_moments_targets(mode)
        arr = {"pocket": (self.pocket, self.mask_pocket), "ligand": (self.ligand, self.mask_ligand)}
        got = {t: ((torch.as_tensor(arr[t][0][safe]), torch.as_tensor(found & arr[t][1][safe])) if t in targets
                   else (None, None)) for t in arr}
        return heads.moments_input_matrix(got["pocket"][0], got["ligand"][0], got["pocket"][1], got["ligand"][1],
                                          mode)

    def batch(self, data, mode):
        return self.matrix(getattr(data, self.key).view(-1).cpu().numpy(), mode)

    def coverage(self, keys):
        """{"items", "pocket", "ligand", "both"}: items keyed by keys with each descriptor present."""
        idx = self.index(keys)
        found = idx >= 0
        safe = np.where(found, idx, 0)
        p, lg = found & self.mask_pocket[safe], found & self.mask_ligand[safe]
        return {"items": int(len(idx)), "pocket": int(p.sum()), "ligand": int(lg.sum()), "both": int((p & lg).sum())}


def read_eval_descriptors(paths):
    """(unique_id -> position Series, pocket_F [M, 343], ligand_F [M, 343]) raw float32 from zernike-eval npz files
    (featurise_zernike.py layout; a row with ok False is NaN), merged in file order; a unique_id in two files is an
    error."""
    import build_zernike_side

    uids, pockets, ligands = [], [], []
    for path in paths:
        pos, pocket, ligand = build_zernike_side.read_descriptors(path)
        uids.append(pos.sort_values().index.to_numpy(dtype=object))
        pockets.append(pocket)
        ligands.append(ligand)
    uid = np.concatenate(uids) if uids else np.zeros(0, dtype=object)
    index = pd.Series(np.arange(len(uid)), index=uid)
    if index.index.duplicated().any():
        dup = index.index[index.index.duplicated()].tolist()[:5]
        raise ValueError(f"unique_id in more than one --zernike-eval file: {dup}")
    dim = pockets[0].shape[1] if pockets else 343
    return (index, np.concatenate(pockets) if pockets else np.zeros((0, dim), np.float32),
            np.concatenate(ligands) if ligands else np.zeros((0, dim), np.float32))


def standardise_eval(side, pocket, ligand):
    """(pocket_std, ligand_std, mask_pocket, mask_ligand) of raw eval descriptors [n, 343] in the side file's units:
    (F - side mean) / side sd with the side file's TRAIN statistics (never the eval set's own); a row that is not
    finite is masked (zeros)."""
    out = []
    for target, raw in (("pocket", pocket), ("ligand", ligand)):
        raw = np.asarray(raw, dtype=np.float32)
        mask = np.isfinite(raw).all(axis=1) if raw.ndim == 2 else np.zeros(0, dtype=bool)
        std = (raw - side[f"{target}_mean"]) / side[f"{target}_sd"]
        std[~mask] = 0.0
        out.append((np.ascontiguousarray(std, dtype=np.float32), mask))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def fep_moments_source(side, eval_desc, unique_ids, label="fep"):
    """MomentsSource keyed by graph_id (= position in unique_ids, the processed CSV's row order) over the
    --zernike-eval descriptors of those unique_ids, standardised with the side file's statistics."""
    index, pocket, ligand = eval_desc
    pos = index.reindex(pd.Index(np.asarray(unique_ids, dtype=object))).to_numpy()
    have = ~pd.isna(pos)
    dim = pocket.shape[1] if pocket.ndim == 2 else 343
    raw_p = np.full((len(pos), dim), np.nan, dtype=np.float32)
    raw_l = np.full((len(pos), dim), np.nan, dtype=np.float32)
    take = pos[have].astype(np.int64)
    raw_p[have] = pocket[take]
    raw_l[have] = ligand[take]
    p_std, l_std, mp, ml = standardise_eval(side, raw_p, raw_l)
    return MomentsSource("y", np.arange(len(pos)), p_std, l_std, mp, ml, label)


def models_need_inputs(models):
    return any(input_mode(m) != "none" for m in models)


def predict_models(models, scaler, dataset, device, ligand_only=False, collect=(), batch_size=256, moments=None):
    """Return (preds [n_models, n] in pK, {attr: concatenated values from the batches}). moments (a MomentsSource)
    attaches data.moments to every batch of a model with moments inputs (an error when such a model has none)."""
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, cols = [], {a: [] for a in collect}
    for k, model in enumerate(models):
        model.eval()
        model.to(device)
        mode = input_mode(model)
        if mode != "none" and moments is None:
            raise ValueError(f"model {k} takes moments inputs ({mode}) but no descriptor source was given")
        raw = []
        with torch.no_grad():
            for data in loader:
                if k == 0:
                    for a in collect:
                        cols[a].append(getattr(data, a).view(-1).cpu())
                data = data.to(device)
                if ligand_only:
                    x = data.x.clone()
                    x[:, N_RDKIT:] = 0
                    data.x = x
                if mode != "none":
                    data.moments = moments.batch(data, mode).to(device)
                raw.append(model(data).view(-1).cpu())
        raw = torch.cat(raw).numpy().astype(np.float64)
        preds.append(scaler.inverse_transform(raw.reshape(-1, 1)).ravel())
    cols = {a: torch.cat(v).numpy() for a, v in cols.items()}
    return np.vstack(preds), cols


def _add_preds(df, preds):
    df = df.copy()
    df["pred"] = preds.mean(axis=0)
    for i, p in enumerate(preds):
        df[f"pred_{i}"] = p
    return df


def row_moments(ctx, models, ds):
    """The row-keyed MomentsSource (the side file) for a paired split when some model takes moments inputs, else
    None; exits when the split has no row ids or no side file was given."""
    if not models_need_inputs(models):
        return None
    if "row" not in ds._data.keys():
        sys.exit("ERROR: a model takes moments inputs but this .pt has no row ids to look its descriptors up")
    source = ctx.get("row_source")
    if source is None:
        sys.exit("ERROR: a model takes moments inputs: pass --zernike-side")
    cov = source.coverage(ds._data.row.view(-1).numpy())
    print(f"moments inputs: {cov['items']} rows, pocket descriptor {cov['pocket']}, ligand descriptor "
          f"{cov['ligand']} (side file; the rest get zeros and the missing flag)")
    return source


# ----------------------------------------------------------------------------- report helpers

def md_table(df, digits=3):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- pairwise section

def pairwise_dataset(args, ctx, name, df, group_cols, models, ds, lig=None):
    """--pairwise for one dataset: pairs, ensemble delta_hat (mean over the models' Siamese heads when
    every model has one), pairwise_<name>_per_series.csv, and its rows of pairwise_summary.csv
    (collected in ctx["pairwise"])."""
    if len(df) != len(ds):  # pair positions index both the frame and the dataset
        sys.exit(f"ERROR: pairwise {name}: {len(df)} prediction rows for {len(ds)} graphs")
    i, j = series_pairs(df, group_cols, lig)
    with_head = [has_siamese(m) for m in models]
    delta = None
    if with_head and all(with_head):
        per_model = pair_deltas(models, ctx["scaler"], ds, i, j, ctx["device"], ctx["ligand_only"],
                                batch_size=args.batch_size)
        delta = per_model.mean(axis=0)
    elif any(with_head):
        print(f"WARNING: pairwise {name}: only {sum(with_head)} of {len(models)} models have a Siamese "
              "head; writing the point rows only")
    with_ph = [has_pair_head(m) for m in models]
    pairhead = None
    if with_ph and all(with_ph):
        both = pairhead_scores(ctx, models, ds, np.concatenate([i, j]), np.concatenate([j, i]),
                               batch_size=args.batch_size).mean(axis=0)
        pairhead = (both[:len(i)], both[len(i):])
    elif any(with_ph):
        print(f"WARNING: pairwise {name}: only {sum(with_ph)} of {len(models)} models have a pair head; "
              "no pairhead rows")
    ctx["pairwise_heads"] = (f"Siamese head on all {len(models)} models" if delta is not None
                             else f"no Siamese head on all models ({sum(with_head)} of {len(models)}): "
                                  "point rows only")
    if pairhead is not None:
        ctx["pairwise_heads"] += f"; pair head on all {len(models)} models (pairhead rows)"
    per = pairwise_per_series(df, group_cols, i, j, delta, pairhead=pairhead)
    per.to_csv(os.path.join(args.out_dir, f"pairwise_{name}_per_series.csv"), index=False)
    summ = pairwise_summary(per, name, PAIRWISE_MIN_NS.get(name, (MIN_N_FEP,)), with_pair=delta is not None,
                            with_pairhead=pairhead is not None)
    ctx["pairwise"].append(summ)
    print(f"pairwise {name}: {len(per)} series with pairs, {len(i)} pairs")
    print(summ[summ["weighting"] == "weighted"].to_string(index=False))


def pairwise_report(args, ctx, report):
    parts = [p for p in ctx["pairwise"] if len(p)]
    summ = (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=PAIRWISE_SUMMARY_COLUMNS))
    summ.to_csv(os.path.join(args.out_dir, "pairwise_summary.csv"), index=False)
    report.append("## Pairwise ranking within series (--pairwise)\n")
    report.append("Pairs i < j of a series with different ligand keys (FEP: (subset, series); series_valid / "
                  "group_valid: series_key; CASF test: UniProt). head pair: the Siamese head's delta_hat "
                  "(mean over models per pair), kendall / spearman of the per-ligand score mean_j delta_hat_ij; "
                  "head point: f_i - f_j and f (ensemble pointwise); head pairhead (arm P): per-ligand score "
                  "mean_a l(z_a, z_i), direction from l(z_j, z_i) - l(z_i, z_j), no pair RMSE. pair_rmse in pK; "
                  f"pair_dir_acc over pairs with |dy| >= {PAIR_DIR_MIN_PK} pK. n-weighted means; unweighted rows "
                  "in pairwise_summary.csv. "
                  f"{ctx.get('pairwise_heads', 'no dataset with series evaluated')}.\n")
    report.append(md_table(summ[summ["weighting"] == "weighted"]) + "\n")


def moments_dataset(args, ctx, name, ds, rows, models, all_rows_name=None):
    """moments_summary.csv rows of one dataset (collected in ctx["moments"]): per target with a head on some
    model, R^2 of the ensemble (mean over those models) and of each model, over the kind-0 rows (side file kind;
    dataset order) that have that descriptor, as dataset `name`; with all_rows_name, also over every row with the
    descriptor (augmented rows included) under that name."""
    import torch

    side = ctx.get("side")
    if side is None:
        return
    import build_zernike_side

    idx = build_zernike_side.side_index(side, rows)
    with_targets = {t: [k for k, m in enumerate(models) if t in moments_targets(m)] for t in MOMENTS_TARGETS}
    if not any(with_targets.values()):
        return
    which = sorted({k for ks in with_targets.values() for k in ks})
    zs = section_embeddings(ctx, models, ds, args.batch_size, which)
    kind0 = np.zeros(len(rows), dtype=bool)
    kind0[idx >= 0] = side["kind"][idx[idx >= 0]] == 0
    groups = {name: kind0}
    if all_rows_name:
        groups[all_rows_name] = np.ones(len(rows), dtype=bool)
    for target, ks in with_targets.items():
        if not ks:
            continue
        if len(ks) != len(models):
            print(f"WARNING: moments {name}: only {len(ks)} of {len(models)} models have a {target} head")
        preds = {}
        for k in ks:
            model = models[k]
            with torch.no_grad():
                z = zs[k].to(ctx["device"])
                preds[k] = torch.cat([model.moments_pred(z[s:s + PAIR_CHUNK], target).cpu()
                                      for s in range(0, len(z), PAIR_CHUNK)]).double().numpy() \
                    if len(z) else np.zeros((0, side[f"{target}_std"].shape[1]))
        has = idx >= 0
        has[has] = side[f"mask_{target}"][idx[has]]
        target_std = np.zeros((len(rows), side[f"{target}_std"].shape[1]))
        target_std[has] = side[f"{target}_std"][idx[has]]
        for gname, gmask in groups.items():
            sel = has & gmask
            ens = np.mean([preds[k][sel] for k in ks], axis=0)
            entries = [("ensemble", ens)] + [(str(k), preds[k][sel]) for k in ks]
            for model_name, pred in entries:
                r2, r2u, mse = moments_r2(pred, target_std[sel])
                ctx["moments"].append({"dataset": gname, "target": target, "model": model_name,
                                       "n_models": len(ks) if model_name == "ensemble" else 1,
                                       "n_rows": int(sel.sum()), "r2": r2, "r2_uniform": r2u, "mse": mse})
    print(f"moments {name}: " + ", ".join(f"{r['dataset']} {r['target']} r2 {r['r2']:.3f}" for r in ctx["moments"]
                                          if r["model"] == "ensemble" and r["dataset"] in groups))


def moments_report(args, ctx, report):
    summ = pd.DataFrame(ctx["moments"], columns=MOMENTS_COLUMNS)
    summ.to_csv(os.path.join(args.out_dir, "moments_summary.csv"), index=False)
    report.append("## Zernike moments heads (--zernike-side)\n")
    report.append(f"Side file {args.zernike_side}. R^2 in the side file's standardised units over the kind-0 rows "
                  "with the descriptor (aug_valid_all: every aug_valid row, parent / twin descriptors): r2 pooled "
                  "over the 343 columns, r2_uniform the mean per-column R^2; ensemble = "
                  "mean prediction of the models with the head; per-model rows in moments_summary.csv.\n")
    report.append(md_table(summ[summ["model"] == "ensemble"]) + "\n")


def models_with_inputs(model_paths):
    """{path: input_moments mode} of the models whose sidecar declares moments inputs (or model_arch moments_only)."""
    out = {}
    for path in model_paths:
        try:
            side = sidecar_heads(path)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        if side is not None and side["input_moments"] != "none":
            out[path] = side["input_moments"]
    return out


def check_side_stats(paths, side):
    """Exit when a model with moments inputs recorded a side-file standardisation (zernike_side_stats) other than
    --zernike-side's: its inputs would be standardised with statistics it was not trained on."""
    import heads

    digest = heads.side_stats_digest(side)
    for path in paths:
        trained = sidecar_heads(path).get("zernike_side_stats")
        if trained and trained != digest:
            sys.exit(f"ERROR: {os.path.basename(path)} was trained with another side file's mean / sd "
                     f"(zernike_side_stats {trained[:12]}, --zernike-side {digest[:12]}): pass its training side file")


def check_zernike_side(model_paths, side_path):
    """Warn when a model sidecar's zernike_side is not --zernike-side (after realpath)."""
    for path in model_paths:
        try:
            side = sidecar_heads(path)
        except ValueError:
            continue
        trained = side.get("zernike_side") if side else None
        if trained and os.path.realpath(trained) != os.path.realpath(side_path):
            print(f"WARNING: {os.path.basename(path)} was trained with --zernike-side {trained}, evaluating with "
                  f"{side_path}")


# ----------------------------------------------------------------------------- sections

def eval_fep(args, ctx, report):
    processed_csv = args.fep_processed_csv or os.path.splitext(args.fep_csv)[0] + "_processed.csv"
    # process_data round-trips the CSV through pandas, so numeric-looking codes (ref_pdb 3E63)
    # come back as floats in the processed CSV: take only the row order from it and every other
    # column from fep_benchmark.csv read as strings.
    proc = pd.read_csv(processed_csv, usecols=["unique_id"], dtype=str)
    prepared = pd.read_csv(args.fep_csv, dtype={"unique_id": str, "subset": str, "series": str,
                                               "ref_pdb": str}, keep_default_na=False)
    n_prepared = len(prepared)
    missing = set(proc["unique_id"]) - set(prepared["unique_id"])
    if missing:
        sys.exit(f"ERROR: {len(missing)} processed ids are not in {args.fep_csv}: {sorted(missing)[:5]}")
    ds = load_pt(args.fep_pt)
    if len(ds) != len(proc):
        sys.exit(f"ERROR: {args.fep_pt} has {len(ds)} graphs but {processed_csv} has {len(proc)} rows")
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    source, coverage_note = None, ""
    if models_need_inputs(models):  # graph_id = row of the processed CSV -> unique_id -> --zernike-eval descriptor
        if ctx.get("side") is None or ctx.get("eval_desc") is None:
            sys.exit("ERROR: FEP: a model takes moments inputs: pass --zernike-side and --zernike-eval")
        source = fep_moments_source(ctx["side"], ctx["eval_desc"], proc["unique_id"].to_numpy())
        cov = source.coverage(np.arange(len(proc)))
        if len(proc) and cov["pocket"] == 0 and cov["ligand"] == 0:
            sys.exit(f"ERROR: FEP: none of the {len(proc)} ligands of {args.fep_csv} has a descriptor in --zernike-eval "
                     f"{', '.join(getattr(args, 'zernike_eval_files', []))} (wrong file?)")
        if cov["both"] < cov["items"]:
            print(f"WARNING: FEP: {cov['items'] - cov['both']} of {cov['items']} ligands lack a pocket or ligand "
                  "descriptor: zeros and the missing flag")
        coverage_note = (f"Moments inputs (--zernike-eval {', '.join(getattr(args, 'zernike_eval_files', []))}, "
                         "standardised with "
                         f"the side file's train mean / sd): pocket descriptor for {cov['pocket']} of {cov['items']} "
                         f"ligands, ligand descriptor for {cov['ligand']}; the rest get zeros and the missing flag.\n")
        print(f"moments inputs fep: {cov}")
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=("y",), batch_size=args.batch_size, moments=source)
    gid = cols["y"].astype(np.int64)
    df = proc.iloc[gid].reset_index(drop=True).merge(
        prepared[["unique_id", "subset", "series", "ref_pdb", "pK"]], on="unique_id", how="left")
    df = df.rename(columns={"pK": "truth"})
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_fep.csv"), index=False)

    per = per_series_metrics(df, ["subset", "series"])
    refs = df.groupby(["subset", "series"], sort=True)["ref_pdb"].first().astype(str).tolist()
    if ctx["name_index"] is not None:
        uniprots, counts = train_rows_same_uniprot(refs, ctx["meta"] if ctx["meta"] is not None
                                                   else pd.DataFrame({"split": [], "uniprot": []}),
                                                   ctx["name_index"])
        if ctx["meta"] is None:
            counts = [pd.NA] * len(refs)
    else:
        uniprots, counts = [""] * len(refs), [pd.NA] * len(refs)
    per.insert(2, "ref_pdb", refs)
    per.insert(3, "uniprot", uniprots)
    per.insert(5, "train_rows_same_uniprot", pd.array(counts, dtype="Int64"))
    per.to_csv(os.path.join(args.out_dir, "fep_per_series.csv"), index=False)
    summ = summarise(per, (MIN_N_FEP,))
    summ.to_csv(os.path.join(args.out_dir, "fep_summary.csv"), index=False)
    if getattr(args, "pairwise", False):
        pairwise_dataset(args, ctx, "fep", df, ["subset", "series"], models, ds)

    report.append("## FEP benchmark\n")
    report.append(f"Ligands evaluated {len(df)} of {n_prepared} prepared (featurisation kept "
                  f"{len(proc)}); series {len(per)}, with n >= {MIN_N_FEP}: {int((per['n'] >= MIN_N_FEP).sum())}.\n")
    report.append(md_table(summ) + "\n")
    if coverage_note:
        report.append(coverage_note)
    known = per["train_rows_same_uniprot"].dropna()
    if len(known):
        report.append(f"Series whose reference UniProt has train rows: {int((known > 0).sum())} of "
                      f"{len(per)} (unknown ref_pdb / UniProt: {int((known < 0).sum())}).\n")
    print(summ.to_string(index=False))
    return df


def paper_check(args, fep_df, report):
    paper = pd.read_csv(args.paper_csv)
    j = fep_df[["unique_id", "subset", "series", "truth", "pred"]].merge(
        paper[["unique_id", "preds"]].rename(columns={"preds": "paper_pred"}), on="unique_id")
    j["diff"] = j["pred"] - j["paper_pred"]
    j.to_csv(os.path.join(args.out_dir, "paper_check.csv"), index=False)
    r = series_metrics(j["paper_pred"], j["pred"])["pearson"] if len(j) else math.nan
    ours_only = sorted(set(fep_df["unique_id"]) - set(paper["unique_id"]))
    paper_only = sorted(set(paper["unique_id"]) - set(fep_df["unique_id"]))
    report.append("## Featurisation check against the paper's predictions\n")
    report.append(f"Pearson(ours, paper preds) = {r:.4f} over joined {len(j)} ligands; "
                  f"mean |diff| {j['diff'].abs().mean():.4f} pK, max |diff| {j['diff'].abs().max():.4f} pK.\n")
    report.append(f"Ours not in paper: {len(ours_only)} {ours_only[:20]}; paper not in ours: "
                  f"{len(paper_only)} {paper_only[:20]}.\n")
    if len(j):
        paper_per = per_series_metrics(j.assign(pred=j["paper_pred"]), ["subset", "series"])
        report.append("Paper predictions scored by this code on the joined ligands:\n")
        report.append(md_table(summarise(paper_per, (MIN_N_FEP,))) + "\n")
    print(f"paper check: Pearson {r:.4f} over {len(j)} joined ligands")


def eval_series_valid(args, ctx, report):
    ds = load_pt(args.series_valid_pt)
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=("y", "row"), batch_size=args.batch_size,
                                 moments=row_moments(ctx, models, ds))
    meta = ctx["meta"].set_index("row")
    rows = cols["row"].astype(np.int64)
    m = meta.loc[rows]
    df = pd.DataFrame({"row": rows, "unique_id": m["unique_id"].to_numpy(),
                       "series_key": m["series_key"].to_numpy(), "source": m["source"].to_numpy(),
                       "uniprot": m["uniprot"].fillna("").astype(str).to_numpy(),
                       "truth": cols["y"].astype(np.float64)})
    if not np.allclose(df["truth"], m["pK"].to_numpy(), atol=1e-3):
        print("WARNING: series_valid .pt labels differ from paired_meta pK")
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_series_valid.csv"), index=False)
    if getattr(args, "pairwise", False):
        pairwise_dataset(args, ctx, "series_valid", df, ["series_key"], models, ds,
                         lig=m["lig_key"].to_numpy() if "lig_key" in m.columns else None)
    moments_dataset(args, ctx, "series_valid", ds, rows, models)
    per = per_series_metrics(df, ["series_key"])
    first = df.groupby("series_key", sort=True)[["source", "uniprot"]].first()
    per.insert(1, "source", first["source"].to_numpy())
    per.insert(2, "uniprot", first["uniprot"].to_numpy())
    per.to_csv(os.path.join(args.out_dir, "series_valid_per_series.csv"), index=False)
    parts = [summarise(per, MIN_N_SERIES_VALID).assign(source="all")]
    for source in sorted(per["source"].unique()):
        parts.append(summarise(per[per["source"] == source], MIN_N_SERIES_VALID).assign(source=source))
    summ = pd.concat(parts, ignore_index=True)
    summ = summ[["source"] + [c for c in summ.columns if c != "source"]]
    summ.to_csv(os.path.join(args.out_dir, "series_valid_summary.csv"), index=False)
    report.append("## Held-out series (series_valid)\n")
    report.append(f"Rows {len(df)}, series {len(per)}.\n")
    report.append(md_table(summ[summ["scope"] != "all"]) + "\n")
    print(summ.to_string(index=False))


def eval_test(args, ctx, report):
    ds = load_pt(args.test_pt)
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    collect = ("y", "row") if "row" in ds._data.keys() else ("y",)
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=collect, batch_size=args.batch_size,
                                 moments=row_moments(ctx, models, ds))
    df = pd.DataFrame({"truth": cols["y"].astype(np.float64)})
    if "row" in cols:
        df.insert(0, "row", cols["row"].astype(np.int64))
        if ctx["meta"] is not None:
            df.insert(1, "unique_id", ctx["meta"].set_index("row").loc[df["row"], "unique_id"].to_numpy())
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)
    summ = pd.DataFrame([series_metrics(df["truth"], df["pred"])])[["n"] + METRICS]
    summ.to_csv(os.path.join(args.out_dir, "test_summary.csv"), index=False)
    if "row" in cols:
        moments_dataset(args, ctx, "test", ds, df["row"].to_numpy(), models)
    if getattr(args, "pairwise", False):
        if "row" in cols and ctx["meta"] is not None:
            m = ctx["meta"].set_index("row").loc[df["row"]]
            uni = m["uniprot"].fillna("").astype(str).str.strip() if "uniprot" in m.columns else \
                pd.Series([""] * len(m))
            unmapped = uni.isin(["", UNMAPPED_UNIPROT]).to_numpy()
            groups = np.where(unmapped, [f"single:{r}" for r in df["row"]], uni.to_numpy())
            pairwise_dataset(args, ctx, "test", df.assign(uniprot=groups), ["uniprot"], models, ds,
                             lig=m["lig_key"].to_numpy() if "lig_key" in m.columns else None)
        else:
            print("pairwise test: skipped (needs a row field in --test-pt and --meta for the UniProt groups)")
    report.append("## CASF test (pooled)\n")
    report.append(md_table(summ) + "\n")
    print(summ.to_string(index=False))


def eval_poses(args, ctx, report):
    ds = load_pt(args.pose_pt)
    missing = [a for a in ("row", "kind", "parent") if a not in ds._data.keys()]
    if missing:
        sys.exit(f"ERROR: {args.pose_pt} lacks {missing}; build it with build_paired_dataset --poses-*")
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=("y", "row", "kind", "parent"), batch_size=args.batch_size,
                                 moments=row_moments(ctx, models, ds))
    df = pd.DataFrame({"row": cols["row"].astype(np.int64), "kind": cols["kind"].astype(np.int64),
                       "parent": cols["parent"].astype(np.int64), "truth": cols["y"].astype(np.float64)})
    if ctx["meta"] is not None:
        uid = ctx["meta"].set_index("row")["unique_id"].reindex(df["row"].to_numpy())
        df.insert(1, "unique_id", uid.fillna("").to_numpy())  # "" for rows of another build's meta
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_poses_valid.csv"), index=False)
    summ = pd.DataFrame([pose_metrics(df)], columns=POSE_METRICS)
    summ.to_csv(os.path.join(args.out_dir, "pose_summary.csv"), index=False)
    report.append("## Pose sensitivity (PDBbind valid)\n")
    report.append(f"Rows {len(df)} (crystal {int((df['kind'] == 0).sum())}, near-native "
                  f"{int((df['kind'] == 1).sum())}, decoy {int((df['kind'] == 2).sum())}). decoy_rank_acc: "
                  "fraction of complexes whose crystal pose is predicted above every decoy; "
                  "decoy_gap_mean: pred(crystal) - mean pred(decoys), pK; nn_spread: mean "
                  "|pred(near-native) - pred(crystal)|, pK.\n")
    report.append(md_table(summ) + "\n")
    print(summ.to_string(index=False))


def eval_aug(args, ctx, report):
    ds = load_pt(args.aug_pt)
    missing = [a for a in ("row", "kind", "parent") if a not in ds._data.keys()]
    if missing:
        sys.exit(f"ERROR: {args.aug_pt} lacks {missing}; build it with build_paired_dataset --aug")
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=("y", "row", "kind", "parent"), batch_size=args.batch_size,
                                 moments=row_moments(ctx, models, ds))
    df = pd.DataFrame({"row": cols["row"].astype(np.int64), "kind": cols["kind"].astype(np.int64),
                       "parent": cols["parent"].astype(np.int64), "truth": cols["y"].astype(np.float64)})
    if ctx["meta"] is not None:
        uid = ctx["meta"].set_index("row")["unique_id"].reindex(df["row"].to_numpy())
        df.insert(1, "unique_id", uid.fillna("").to_numpy())  # "" for rows of another build's meta
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_aug_valid.csv"), index=False)
    summ = aug_metrics(df)
    with_ph = [has_pair_head(m) for m in models]
    pairhead_note = ""
    if with_ph and all(with_ph):
        a, c = aug_pair_candidates(df)
        score = pairhead_scores(ctx, models, ds, a, c, batch_size=args.batch_size).mean(axis=0)
        summ = pd.concat([summ.assign(head="point"), aug_pairhead_metrics(df, c, score)], ignore_index=True)
        summ["ref_kind"] = summ["ref_kind"].astype("Int64")
        pairhead_note = (" head pairhead (arm P): rank_acc / gap_mean of the pair head's l(z_crystal, .) over the "
                         "positive of the same scheme vs the kind's negatives (logit units; kind 2: the crystal "
                         "itself is the positive).")
    elif any(with_ph):
        print(f"WARNING: aug: only {sum(with_ph)} of {len(models)} models have a pair head; no pairhead rows")
    summ.to_csv(os.path.join(args.out_dir, "aug_summary.csv"), index=False)
    moments_dataset(args, ctx, "valid", ds, df["row"].to_numpy(), models, all_rows_name="aug_valid_all")
    counts = ", ".join(f"kind {k} {int((df['kind'] == k).sum())}" for k in sorted(df["kind"].unique()))
    report.append("## Augmented-row discrimination (PDBbind valid)\n")
    report.append(f"Rows {len(df)} ({counts}). Negative kinds (2 decoy pose, 4 shape twin, 6 contact "
                  "edit): rank_acc = fraction of parents whose positive of the same scheme (ref_kind: "
                  "0 crystal, 3 A-positive, 5 C-positive) is predicted above every negative of the "
                  "kind; gap_mean = pred(positive) - mean pred(negatives), pK. Control kinds (1, 3, 5, "
                  "7): spread = mean |pred - pred(ref_kind)|, pK (crystal; kind 7 against kind 5). pearson: label (the parent's) vs pred "
                  f"over the kind's rows.{pairhead_note}\n")
    report.append(md_table(summ) + "\n")
    print(summ.to_string(index=False))


def eval_group_valid(args, ctx, report):
    ds = load_pt(args.group_valid_pt)
    models = load_models(args.models, ds.num_node_features, ds.num_edge_features, args.hidden_dim,
                         args.head, ctx["device"])
    preds, cols = predict_models(models, ctx["scaler"], ds, ctx["device"], ctx["ligand_only"],
                                 collect=("y", "row"), batch_size=args.batch_size,
                                 moments=row_moments(ctx, models, ds))
    meta = ctx["meta"].set_index("row")
    rows = cols["row"].astype(np.int64)
    m = meta.loc[rows]
    df = pd.DataFrame({"row": rows, "unique_id": m["unique_id"].to_numpy(),
                       "series_key": m["series_key"].to_numpy(), "source": m["source"].to_numpy(),
                       "group_id": (m["group_id"].fillna("").astype(str).to_numpy()
                                    if "group_id" in m else ""),
                       "truth": cols["y"].astype(np.float64)})
    if not np.allclose(df["truth"], m["pK"].to_numpy(), atol=1e-3):
        print("WARNING: group_valid .pt labels differ from paired_meta pK")
    df = _add_preds(df, preds)
    df.to_csv(os.path.join(args.out_dir, "predictions_group_valid.csv"), index=False)
    if getattr(args, "pairwise", False):
        pairwise_dataset(args, ctx, "group_valid", df, ["series_key"], models, ds,
                         lig=m["lig_key"].to_numpy() if "lig_key" in m.columns else None)
    moments_dataset(args, ctx, "group_valid", ds, rows, models)
    per = per_series_metrics(df, ["series_key"])
    first = df.groupby("series_key", sort=True)[["source", "group_id"]].first()
    per.insert(1, "source", first["source"].to_numpy())
    per.insert(2, "group_id", first["group_id"].to_numpy())
    per.to_csv(os.path.join(args.out_dir, "group_valid_per_series.csv"), index=False)
    point = series_metrics(df["truth"], df["pred"])
    pointwise = pd.DataFrame([{"scope": "pointwise", "weighting": "none", "n_series": int(len(per)),
                               "n_items": point["n"], **{k: point[k] for k in METRICS}}])
    summ = pd.concat([pointwise, summarise(per, (MIN_N_GROUP_VALID,))], ignore_index=True)
    summ.to_csv(os.path.join(args.out_dir, "group_valid_summary.csv"), index=False)
    n_groups = df.loc[df["group_id"] != "", "group_id"].nunique()
    report.append("## Group-disjoint holdout (group_valid)\n")
    report.append(f"Rows {len(df)}, target groups {n_groups}, series {len(per)} (with n >= "
                  f"{MIN_N_GROUP_VALID}: {int((per['n'] >= MIN_N_GROUP_VALID).sum())}).\n")
    report.append(md_table(summ) + "\n")
    print(summ.to_string(index=False))


def census_report(args, report):
    path = args.residue_census or os.path.join(os.path.dirname(os.path.abspath(args.fep_csv)),
                                               "fep_residue_census.csv")
    if not os.path.isfile(path):
        return
    c = pd.read_csv(path)
    if not len(c):
        report.append("## Protein atoms the featuriser drops\n\nNone.\n")
        return
    tot = (c.groupby(["record", "residue_name"])
           .agg(n_series=("series", "size"), n_residues=("n_residues", "sum"),
                n_dropped_atoms=("n_dropped_atoms", "sum")).reset_index()
           .sort_values(["record", "n_dropped_atoms"], ascending=[True, False]))
    report.append("## Protein atoms the featuriser drops\n")
    report.append("The clone's GetMolAEVs_extended reads only ATOM records and keeps atoms whose RES-ATOM "
                  "key is in PDB_Atom_Keys.csv, so all HETATM (waters, ions, modified residues such as "
                  "TPO/PTR/SEP) and ATOM caps (ACE/NMA/NME) are ignored; the PDBs are not rewritten. "
                  f"Source: {path}.\n")
    report.append(md_table(tot) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta", default=None, help="paired_meta.csv")
    ap.add_argument("--fep-csv", default=None, help="fep_benchmark.csv")
    ap.add_argument("--fep-pt", default=None, help="processed/fep_benchmark.pt")
    ap.add_argument("--fep-processed-csv", default=None,
                    help="default: <fep-csv stem>_processed.csv (graph_id = its row order)")
    ap.add_argument("--series-valid-pt", default=None)
    ap.add_argument("--test-pt", default=None)
    ap.add_argument("--pose-pt", default=None,
                    help="processed/paired_poses_valid.pt (arm 4) -> pose_summary.csv")
    ap.add_argument("--aug-pt", default=None,
                    help="processed/paired_aug_valid.pt -> aug_summary.csv (per-kind discrimination)")
    ap.add_argument("--group-valid-pt", default=None,
                    help="processed/paired_group_valid.pt -> group_valid_summary.csv (needs --meta)")
    ap.add_argument("--paper-csv", default=None, help="paper predictions (unique_id, preds) to compare")
    ap.add_argument("--pdbbind-index-tgz", default=DEFAULT_INDEX_TGZ,
                    help="PDBbind plain-text index tarball, for the ref_pdb UniProt lookup")
    ap.add_argument("--residue-census", default=None,
                    help="default: fep_residue_census.csv next to --fep-csv, if present")
    ap.add_argument("--allow-data-dir-mismatch", action="store_true",
                    help="warn instead of exiting when a split file is not under a model's training data_dir")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--head", type=int, default=3)
    ap.add_argument("--ligand-only", action="store_true", help="zero node-feature columns 15: before predicting")
    ap.add_argument("--pairwise", action="store_true",
                    help="within-series pairwise ranking (Siamese head and pointwise) -> pairwise_summary.csv")
    ap.add_argument("--zernike-side", default=None,
                    help="paired_zernike.npz (build_zernike_side.py) -> moments_summary.csv for moments heads; the "
                         "descriptors and train mean / sd of models with moments inputs")
    ap.add_argument("--zernike-eval", default=None,
                    help="FILE[,FILE...]: featurise_zernike.py --csv descriptors (e.g. zernike_fep.npz, "
                         "zernike_pgdh.npz) of the --fep-csv ligands, for models with moments inputs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = ap.parse_args(argv)

    import torch

    for path in args.models + [args.scaler]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: missing {path}")
    if (args.fep_csv is None) != (args.fep_pt is None):
        sys.exit("ERROR: --fep-csv and --fep-pt go together")
    if args.series_valid_pt and not args.meta:
        sys.exit("ERROR: --series-valid-pt needs --meta")
    if args.group_valid_pt and not args.meta:
        sys.exit("ERROR: --group-valid-pt needs --meta")
    if args.zernike_side and not os.path.isfile(args.zernike_side):
        sys.exit(f"ERROR: missing {args.zernike_side}")
    args.zernike_eval_files = [f.strip() for f in (args.zernike_eval or "").split(",") if f.strip()]
    for path in args.zernike_eval_files:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: missing --zernike-eval file {path}")
    need_inputs = models_with_inputs(args.models)
    if need_inputs:
        names = ", ".join(f"{os.path.basename(p)} ({m})" for p, m in need_inputs.items())
        if not args.zernike_side:
            sys.exit(f"ERROR: models take moments inputs ({names}): pass --zernike-side (the training side file: "
                     "descriptors and train mean / sd)")
        if args.fep_csv and not args.zernike_eval_files:
            sys.exit(f"ERROR: models take moments inputs ({names}): the --fep-csv section needs --zernike-eval "
                     "(featurise_zernike.py --csv)")
    check_data_dirs(args.models, {"--series-valid-pt": args.series_valid_pt,
                                  "--group-valid-pt": args.group_valid_pt, "--aug-pt": args.aug_pt},
                    args.allow_data_dir_mismatch)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.scaler, "rb") as fh:
        scaler = pickle.load(fh)
    ctx = {"device": device, "scaler": scaler, "ligand_only": resolve_ligand_only(args.models, args.ligand_only),
           "meta": pd.read_csv(args.meta, keep_default_na=False, na_values=[""]) if args.meta else None,
           "name_index": None, "pairwise": [], "moments": [], "side": None, "row_source": None, "eval_desc": None}
    if args.zernike_side:
        import build_zernike_side

        ctx["side"] = build_zernike_side.load_side(args.zernike_side)
        check_zernike_side(args.models, args.zernike_side)
    if need_inputs:
        check_side_stats(need_inputs, ctx["side"])
        ctx["row_source"] = MomentsSource.from_side(ctx["side"])
        if args.zernike_eval_files:
            try:
                ctx["eval_desc"] = read_eval_descriptors(args.zernike_eval_files)
            except ValueError as exc:
                sys.exit(f"ERROR: {exc}")
            print(f"moments inputs: {len(ctx['eval_desc'][0])} descriptors from --zernike-eval "
                  f"{', '.join(args.zernike_eval_files)}")
    elif args.zernike_eval_files:
        print("--zernike-eval: no model takes moments inputs; ignored")
    if args.fep_csv:
        if os.path.isfile(args.pdbbind_index_tgz):
            ctx["name_index"] = read_pdbbind_index(args.pdbbind_index_tgz)[0]
        else:
            print(f"WARNING: {args.pdbbind_index_tgz} not found; uniprot / train_rows_same_uniprot left empty")

    report = [f"# Evaluation {os.path.basename(os.path.abspath(args.out_dir))}\n",
              f"Models ({len(args.models)}): " + ", ".join(args.models) + "\n",
              f"Scaler: {args.scaler} (mean {float(scaler.mean_[0]):.4f}, scale {float(scaler.scale_[0]):.4f}); "
              f"ligand_only {ctx['ligand_only']}; device {device}; hidden {args.hidden_dim}, heads {args.head}.\n"]
    if args.fep_csv:
        fep_df = eval_fep(args, ctx, report)
        if args.paper_csv:
            paper_check(args, fep_df, report)
        census_report(args, report)
    if args.series_valid_pt:
        eval_series_valid(args, ctx, report)
    if args.test_pt:
        eval_test(args, ctx, report)
    if args.group_valid_pt:
        eval_group_valid(args, ctx, report)
    if args.pose_pt:
        eval_poses(args, ctx, report)
    if args.aug_pt:
        eval_aug(args, ctx, report)
    if args.pairwise:
        pairwise_report(args, ctx, report)
    if ctx["side"] is not None:
        if ctx["moments"]:
            moments_report(args, ctx, report)
        else:
            print("moments: --zernike-side given but no model has a moments head (or no section with row ids ran); "
                  "moments_summary.csv not written")
    ctx.pop("z_cache", None)
    with open(os.path.join(args.out_dir, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(report))
    print(f"wrote {os.path.join(args.out_dir, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
