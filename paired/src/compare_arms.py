"""Compare evaluation outputs of several training arms (tags) seed by seed and as 3-seed ensembles.

Reads work/output/eval/<tag>/{fep_summary.csv, series_valid_summary.csv, test_summary.csv} (ensemble) and
work/output/eval/<tag>/seed<i>/... (single seeds) written by eval_paired.py, and writes a markdown report with:
  * per-tag ensemble numbers,
  * per-seed mean +- sd and the per-seed paired difference to the reference tag (same seed index),
  * the shortcut gate: an arm's gain over the reference must exceed the ligand-only arm's gain over the
    ligand-only reference (same losses, AEVs zeroed), otherwise the gain is not evidence of structure use.
Pose-sensitivity rows (pose_summary.csv, arm 4), group-disjoint holdout rows (group_valid_summary.csv) and
per-kind augmented-row rows (aug_summary.csv: rank_acc and gap_mean per negative kind, spread per control
kind, for the kinds every tag has) are added when every tag has that file in its ensemble directory or in one
of its seed directories. Pairwise rows (pairwise_summary.csv, eval_paired.py --pairwise; arm S) likewise, for the
datasets every tag has: per dataset the Siamese head's ("pair-head" / "pair") and the pointwise head's
("point-head" / "point pair") n-weighted Kendall, Spearman, pair RMSE and direction accuracy, at the scope of the
dataset's pointwise rows (FEP n>=10, series_valid and group_valid n>=5); a tag without a Siamese head has NaN
pair rows and real point rows. A "Pair head vs pointwise head" table then sets each tag's pair head against its own
point head and the reference's point head (per-seed paired differences; ligand-only tags of --lig-pairs against the
ligand-only reference). In the shortcut gate the pair-head rows compare the arm's pair head with the reference's
point head (labelled "(ref: point head)"), as the reference has no pair head.
Arm P (when some tag has them; tags without them read NaN): pairwise_summary.csv rows of head pairhead (the pair
head's n-weighted Kendall, Spearman and direction accuracy: "pairhead wKendall" etc., linked to the point head in the
pair-vs-point table and the gate), aug_summary.csv rows of head pairhead per negative kind ("pairhead rank acc", linked
to the pointwise rank acc of the same kind; "pairhead gap (logit)" is reported as a value only: logit gaps of
different models are not comparable, so it gets no paired differences and no gate row; aug_summary.csv files with a
head column are read at head point for the pointwise rows), and moments_summary.csv rows "Moments R2 <target>
(<dataset>)" (ensemble r2 of the pocket / ligand moments heads; kind-0 rows, aug_valid_all every aug_valid row).

Usage: python compare_arms.py --eval-root WORK/output/eval --tags base,dh03,dh10 --ref base \
           [--lig-pairs dh03-lig:dh03,base-lig:base] --seeds 0,1,2 --out REPORT.md
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

# metric columns reported, in order: (file, row selector, column, label)
METRICS = [
    ("fep_summary.csv", {"scope": "n>=10", "weighting": "weighted"}, "pearson", "FEP wPCC (n>=10)"),
    ("fep_summary.csv", {"scope": "n>=10", "weighting": "weighted"}, "kendall", "FEP wKendall (n>=10)"),
    ("fep_summary.csv", {"scope": "n>=10", "weighting": "weighted"}, "spearman", "FEP wSpearman (n>=10)"),
    ("fep_summary.csv", {"scope": "all", "weighting": "weighted"}, "rmse", "FEP wRMSE (all)"),
    ("series_valid_summary.csv", {"source": "all", "scope": "n>=5", "weighting": "weighted"}, "spearman",
     "series_valid wSpearman (n>=5)"),
    ("series_valid_summary.csv", {"source": "all", "scope": "n>=5", "weighting": "weighted"}, "kendall",
     "series_valid wKendall (n>=5)"),
    ("series_valid_summary.csv", {"source": "all", "scope": "all", "weighting": "weighted"}, "rmse",
     "series_valid wRMSE (all)"),
    ("test_summary.csv", {}, "pearson", "CASF Pearson"),
    ("test_summary.csv", {}, "rmse", "CASF RMSE"),
]
# appended to METRICS when every tag has a pose_summary.csv (eval_paired.py --pose-pt)
POSE_METRICS = [
    ("pose_summary.csv", {}, "decoy_rank_acc", "Pose decoy-rank acc"),
    ("pose_summary.csv", {}, "decoy_gap_mean", "Pose decoy gap (pK)"),
    ("pose_summary.csv", {}, "nn_spread", "Pose NN spread (pK)"),
    ("pose_summary.csv", {}, "pearson_crystal", "Pose crystal Pearson"),
]
# appended when every tag has a group_valid_summary.csv (eval_paired.py --group-valid-pt)
GROUP_METRICS = [
    ("group_valid_summary.csv", {"scope": "pointwise", "weighting": "none"}, "pearson", "group_valid Pearson"),
    ("group_valid_summary.csv", {"scope": "pointwise", "weighting": "none"}, "spearman", "group_valid Spearman"),
    ("group_valid_summary.csv", {"scope": "pointwise", "weighting": "none"}, "rmse", "group_valid RMSE"),
    ("group_valid_summary.csv", {"scope": "n>=5", "weighting": "weighted"}, "spearman",
     "group_valid within-series wSpearman (n>=5)"),
]
# pairwise rows (eval_paired.py --pairwise): (dataset in the file, name in the report, scope)
PAIRWISE_FILE = "pairwise_summary.csv"
PAIRWISE_DATASETS = [("fep", "FEP", "n>=10"), ("series_valid", "series_valid", "n>=5"),
                     ("group_valid", "group_valid", "n>=5")]
# (column, label for the Siamese head, label for the pointwise head)
PAIRWISE_COLUMNS = [("kendall", "pair-head wKendall", "point-head wKendall"),
                    ("spearman", "pair-head wSpearman", "point-head wSpearman"),
                    ("pair_rmse", "pair RMSE", "point pair RMSE"),
                    ("pair_dir_acc", "pair dir acc", "point pair dir acc")]
# arm P: (column, label) of the pair head's pairwise rows, linked to the pointwise head's label
PAIRHEAD_COLUMNS = [("kendall", "pairhead wKendall", "point-head wKendall"),
                    ("spearman", "pairhead wSpearman", "point-head wSpearman"),
                    ("pair_dir_acc", "pairhead dir acc", "point pair dir acc")]
MOMENTS_FILE = "moments_summary.csv"
MOMENTS_DATASETS = ("valid", "series_valid", "group_valid", "test", "aug_valid_all")
VALUE_ONLY_SUFFIX = "pairhead gap (logit)"   # rows reported without paired differences or gate rows
MOMENTS_TARGETS = ("pocket", "ligand")
# selector keys a file may lack: a file without the column holds that value only (aug_summary.csv before arm P)
SELECTOR_DEFAULTS = {"head": "point"}
AUG_FILE = "aug_summary.csv"
AUG_NEGATIVE = {"rank_acc": "rank acc", "gap_mean": "gap (pK)"}   # per negative kind (2, 4, 6)
AUG_CONTROL = {"spread": "spread vs {ref} (pK)"}                  # per control kind (1, 3, 5, 7)
AUG_KIND_NAME = {0: "crystal", 1: "near-native", 2: "decoy", 3: "A-pos", 4: "A-twin", 5: "C-pos",
                 6: "C-contact", 7: "C-exposed"}
CONTROL_REF = {1: 0, 3: 0, 5: 0, 7: 5}  # as eval_paired.CONTROL_REF: kind 7's spread is against kind 5
HIGHER_IS_BETTER = {"pearson": True, "kendall": True, "spearman": True, "rmse": False,
                    "decoy_rank_acc": True, "decoy_gap_mean": True, "nn_spread": False,
                    "pearson_crystal": True, "rank_acc": True, "gap_mean": True, "spread": False,
                    "pair_rmse": False, "pair_dir_acc": True, "r2": True}


def read_metric(eval_dir, fname, selector, column):
    """One number from an eval directory, NaN when the file or row is missing."""
    path = os.path.join(eval_dir, fname)
    if not os.path.isfile(path):
        return float("nan")
    df = pd.read_csv(path)
    for key, value in selector.items():
        if key not in df.columns:
            if SELECTOR_DEFAULTS.get(key) == value:
                continue
            return float("nan")
        df = df[df[key].astype(str) == value]
    if len(df) != 1:
        return float("nan")
    return float(df[column].iloc[0])


def _tag_dirs(eval_root, tag, seeds):
    return [os.path.join(eval_root, tag)] + [os.path.join(eval_root, tag, f"seed{s}") for s in seeds]


def has_file(eval_root, tag, seeds, fname):
    return any(os.path.isfile(os.path.join(d, fname)) for d in _tag_dirs(eval_root, tag, seeds))


def has_pose_summary(eval_root, tag, seeds):
    return has_file(eval_root, tag, seeds, POSE_METRICS[0][0])


def aug_roles(eval_root, tag, seeds):
    """{kind: role} over every aug_summary.csv of a tag (ensemble and seed dirs)."""
    roles = {}
    for d in _tag_dirs(eval_root, tag, seeds):
        path = os.path.join(d, AUG_FILE)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            roles.update({int(k): str(r) for k, r in zip(df["kind"], df["role"])})
    return roles


def _aug_selector(eval_root, tags, seeds, kind):
    """Selector of the pointwise aug rows of a kind; head point only when some tag's file has a head column
    (so the selectors of earlier comparisons are unchanged)."""
    sel = {"kind": str(kind)}
    if any(aug_pairhead_kinds(eval_root, t, seeds) for t in tags):
        sel["head"] = "point"
    return sel


def aug_pairhead_kinds(eval_root, tag, seeds):
    """Kinds with head pairhead rows in any aug_summary.csv of a tag."""
    found = set()
    for d in _tag_dirs(eval_root, tag, seeds):
        path = os.path.join(d, AUG_FILE)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if "head" in df.columns:
                found |= {int(k) for k in df.loc[df["head"].astype(str) == "pairhead", "kind"]}
    return found


def aug_metrics_for(eval_root, tags, seeds, with_links=False):
    """Metric rows for the augmented kinds present in every tag's aug_summary.csv, plus (arm P) the pair head's
    rank acc / gap rows of the negative kinds that some tag has pairhead rows for; with_links also returns
    [(pairhead label, pointwise label, column)]."""
    roles = [aug_roles(eval_root, t, seeds) for t in tags]
    common = sorted(set.intersection(*(set(r) for r in roles))) if roles else []
    pairhead = set().union(*(aug_pairhead_kinds(eval_root, t, seeds) for t in tags)) if tags else set()
    out, links = [], []
    for kind in common:
        role = roles[0][kind]
        columns = AUG_NEGATIVE if role == "negative" else AUG_CONTROL if role == "control" else {}
        ref = AUG_KIND_NAME.get(CONTROL_REF.get(kind, 0), "?")
        sel = _aug_selector(eval_root, tags, seeds, kind)
        name = f"Aug kind {kind} ({AUG_KIND_NAME.get(kind, '?')})"
        for column, label in columns.items():
            out.append((AUG_FILE, sel, column, f"{name} {label.format(ref=ref)}"))
        if role == "negative" and kind in pairhead:
            sel_ph = {"kind": str(kind), "head": "pairhead"}
            out.append((AUG_FILE, sel_ph, "rank_acc", f"{name} pairhead rank acc"))
            links.append((f"{name} pairhead rank acc", f"{name} {AUG_NEGATIVE['rank_acc']}", "rank_acc"))
            out.append((AUG_FILE, sel_ph, "gap_mean", f"{name} {VALUE_ONLY_SUFFIX}"))
    return (out, links) if with_links else out


def moments_rows(eval_root, tags, seeds):
    """moments_summary.csv rows (ensemble r2 per dataset and target) present in some tag's file."""
    present = set()
    for t in tags:
        for d in _tag_dirs(eval_root, t, seeds):
            path = os.path.join(d, MOMENTS_FILE)
            if os.path.isfile(path):
                df = pd.read_csv(path)
                present |= set(zip(df["dataset"].astype(str), df["target"].astype(str)))
    return [(MOMENTS_FILE, {"dataset": ds, "target": tg, "model": "ensemble"}, "r2", f"Moments R2 {tg} ({ds})")
            for ds in MOMENTS_DATASETS for tg in MOMENTS_TARGETS if (ds, tg) in present]


def pairwise_datasets(eval_root, tag, seeds, head=None):
    """Datasets named in any pairwise_summary.csv of a tag (ensemble and seed dirs); with head, only the
    datasets that have rows of that head."""
    found = set()
    for d in _tag_dirs(eval_root, tag, seeds):
        path = os.path.join(d, PAIRWISE_FILE)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if head is not None:
                df = df[df["head"].astype(str) == head]
            found |= set(df["dataset"].astype(str))
    return found


def pairwise_rows(eval_root, tags, seeds):
    """(metric rows, [(pair label, point label, column)]) for the datasets every tag's
    pairwise_summary.csv has; empty unless every tag has the file."""
    if not tags or not all(has_file(eval_root, t, seeds, PAIRWISE_FILE) for t in tags):
        return [], []
    common = set.intersection(*(pairwise_datasets(eval_root, t, seeds) for t in tags))
    rows, links = [], []
    for dataset, name, scope in PAIRWISE_DATASETS:
        if dataset not in common:
            continue
        sel = {head: {"dataset": dataset, "head": head, "scope": scope, "weighting": "weighted"}
               for head in ("pair", "point")}
        pair_rows = [(PAIRWISE_FILE, sel["pair"], c, f"{name} {lp} ({scope})") for c, lp, _ in PAIRWISE_COLUMNS]
        point_rows = [(PAIRWISE_FILE, sel["point"], c, f"{name} {lq} ({scope})") for c, _, lq in PAIRWISE_COLUMNS]
        rows += pair_rows + point_rows
        links += [(a[3], b[3], a[2]) for a, b in zip(pair_rows, point_rows)]
        # arm P: the pair head's rows, when some tag has them for this dataset
        if any(dataset in pairwise_datasets(eval_root, t, seeds, head="pairhead") for t in tags):
            sel_ph = {"dataset": dataset, "head": "pairhead", "scope": scope, "weighting": "weighted"}
            for c, label, point_label in PAIRHEAD_COLUMNS:
                rows.append((PAIRWISE_FILE, sel_ph, c, f"{name} {label} ({scope})"))
                links.append((f"{name} {label} ({scope})", f"{name} {point_label} ({scope})", c))
    return rows, links


def metrics_for(eval_root, tags, seeds):
    """METRICS, plus POSE_METRICS / GROUP_METRICS when every tag has that file and the aug rows of the
    kinds every tag has; and the tags lacking a pose_summary.csv."""
    lacking = [t for t in tags if not has_pose_summary(eval_root, t, seeds)]
    metrics = METRICS if lacking else METRICS + POSE_METRICS
    if all(has_file(eval_root, t, seeds, GROUP_METRICS[0][0]) for t in tags):
        metrics = metrics + GROUP_METRICS
    metrics = metrics + pairwise_rows(eval_root, tags, seeds)[0]
    return metrics + aug_metrics_for(eval_root, tags, seeds) + moments_rows(eval_root, tags, seeds), lacking


def collect(eval_root, tags, seeds, metrics=METRICS):
    """DataFrame with columns tag, seed ('ens' or the index), metric label, value."""
    rows = []
    for tag in tags:
        dirs = [("ens", os.path.join(eval_root, tag))] + [(str(s), os.path.join(eval_root, tag, f"seed{s}")) for s in seeds]
        for seed, d in dirs:
            for fname, selector, column, label in metrics:
                rows.append({"tag": tag, "seed": seed, "metric": label, "column": column,
                             "value": read_metric(d, fname, selector, column)})
    return pd.DataFrame(rows)


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table_ensemble(df, tags, metrics=METRICS):
    labels = [m[3] for m in metrics]
    lines = ["| metric | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for label in labels:
        vals = [df[(df.tag == t) & (df.seed == "ens") & (df.metric == label)]["value"]
                for t in tags]
        lines.append(f"| {label} | " + " | ".join(fmt(v.iloc[0]) if len(v) else "nan" for v in vals) + " |")
    return "\n".join(lines)


def seed_stats(df, tag, label, seeds):
    v = np.array([df[(df.tag == tag) & (df.seed == str(s)) & (df.metric == label)]["value"].iloc[0] for s in seeds])
    return v


def value_only(label):
    """Rows shown as values only: no paired differences, no gate row (the pair head's logit gap)."""
    return label.endswith(VALUE_ONLY_SUFFIX)


def table_seeds(df, tags, ref, seeds, metrics=METRICS):
    labels = [m[3] for m in metrics]
    lines = ["| metric | " + " | ".join(f"{t} mean (sd)" for t in tags) + " | "
             + " | ".join(f"{t} - {ref} (paired)" for t in tags if t != ref) + " |",
             "|---|" + "---|" * (2 * len(tags) - 1)]
    for label in labels:
        cells, deltas = [], []
        base = seed_stats(df, ref, label, seeds)
        for t in tags:
            v = seed_stats(df, t, label, seeds)
            cells.append(f"{fmt(np.nanmean(v))} ({fmt(np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else float('nan'))})")
            if t != ref:
                if value_only(label):
                    deltas.append("-")
                    continue
                d = v - base
                deltas.append(f"{fmt(np.nanmean(d), 4)} [{', '.join(fmt(x, 3) for x in d)}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    return "\n".join(lines)


def table_pair_vs_point(df, tags, ref, seeds, links, refs=None):
    """Per tag with a Siamese head (any finite pair value) and pairwise metric: the pair head, the tag's own
    point head and its reference's point head (ensembles; refs: tag -> reference, default ref), and the per-seed
    paired differences pair - own point and pair - reference point (raw differences: for RMSE lower is better).
    '' when no tag has a head."""
    refs = refs or {}
    def ens(tag, label):
        v = df[(df.tag == tag) & (df.seed == "ens") & (df.metric == label)]["value"]
        return v.iloc[0] if len(v) else float("nan")

    def diff(a, b):
        d = a - b
        return f"{fmt(np.nanmean(d), 4) if np.any(np.isfinite(d)) else 'nan'} [{', '.join(fmt(x, 3) for x in d)}]"

    lines = []
    for tag in tags:
        for pair_label, point_label, _ in links:
            pair = seed_stats(df, tag, pair_label, seeds)
            if not (np.any(np.isfinite(pair)) or np.isfinite(ens(tag, pair_label))):
                continue
            tag_ref = refs.get(tag, ref)
            own = seed_stats(df, tag, point_label, seeds)
            base = seed_stats(df, tag_ref, point_label, seeds)
            lines.append(f"| {pair_label} | {tag} | {tag_ref} | {fmt(ens(tag, pair_label))} | "
                         f"{fmt(ens(tag, point_label))} | {fmt(ens(tag_ref, point_label))} | {diff(pair, own)} | "
                         f"{diff(pair, base)} |")
    if not lines:
        return ""
    head = ["| metric | tag | reference | pair head (ens) | own point head (ens) | reference point head (ens) | "
            "pair - own point (per seed) | pair - reference point (per seed) |", "|---|---|---|---|---|---|---|---|"]
    return "\n".join(head + lines)


def gate(df, arm, ref, arm_lig, ref_lig, seeds, metrics=METRICS, ref_labels=None):
    """Per metric: arm-ref gain, lig-arm-lig-ref gain, and whether the structure-based gain exceeds the ligand-only gain
    by more than the seed sd of the paired differences. ref_labels maps a metric label to the label read on the
    reference side (pair-head rows -> the reference's point head)."""
    ref_labels = ref_labels or {}
    lines = ["| metric | " + f"{arm} - {ref}" + " | " + f"{arm_lig} - {ref_lig}" + " | structure-specific gain | verdict |",
             "|---|---|---|---|---|"]
    for fname, selector, column, label in metrics:
        if value_only(label):
            continue
        ref_label = ref_labels.get(label, label)
        d_struct = seed_stats(df, arm, label, seeds) - seed_stats(df, ref, ref_label, seeds)
        d_lig = seed_stats(df, arm_lig, label, seeds) - seed_stats(df, ref_lig, ref_label, seeds)
        sign = 1.0 if HIGHER_IS_BETTER[column] else -1.0
        extra = sign * (np.nanmean(d_struct) - np.nanmean(d_lig))
        spread = np.nanstd(d_struct - d_lig, ddof=1) if np.sum(~np.isnan(d_struct - d_lig)) > 1 else float("nan")
        if np.isnan(extra) or np.isnan(spread):
            verdict = "n/a"
        elif extra > spread:
            verdict = "passes (gain beyond ligand-only, > 1 sd)"
        elif extra > 0:
            verdict = "within seed noise"
        else:
            verdict = "no structure-specific gain"
        shown = label if ref_label == label else f"{label} (ref: point head)"
        lines.append(f"| {shown} | {fmt(np.nanmean(d_struct), 4)} | {fmt(np.nanmean(d_lig), 4)} | "
                     f"{fmt(extra, 4)} (sd {fmt(spread, 4)}) | {verdict} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated tags, e.g. base,dh03,dh10,base-lig,dh03-lig")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--lig-pairs", default="", help="comma-separated ligand-only:structure pairs, e.g. base-lig:base,dh03-lig:dh03")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    # all-NaN rows (e.g. pair-head rows of a tag without a Siamese head) are expected: report nan quietly
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    tags = args.tags.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    metrics, lacking = metrics_for(args.eval_root, tags, seeds)
    df = collect(args.eval_root, tags, seeds, metrics)
    df.to_csv(os.path.splitext(args.out)[0] + "_values.csv", index=False)
    parts = [f"# Arm comparison ({', '.join(tags)}; reference {args.ref}; seeds {args.seeds})\n"]
    if lacking and len(lacking) < len(tags):
        parts.append(f"pose_summary.csv missing for {', '.join(lacking)}: pose rows omitted.\n")
    lacking_group = [t for t in tags if not has_file(args.eval_root, t, seeds, GROUP_METRICS[0][0])]
    if lacking_group and len(lacking_group) < len(tags):
        parts.append(f"group_valid_summary.csv missing for {', '.join(lacking_group)}: group_valid rows omitted.\n")
    lacking_pw = [t for t in tags if not has_file(args.eval_root, t, seeds, PAIRWISE_FILE)]
    if lacking_pw and len(lacking_pw) < len(tags):
        parts.append(f"{PAIRWISE_FILE} missing for {', '.join(lacking_pw)}: pairwise rows omitted.\n")
    parts += ["## 3-seed ensembles\n", table_ensemble(df, tags, metrics), "",
              "## Single seeds: mean (sd) and per-seed paired difference to the reference\n",
              table_seeds(df, tags, args.ref, seeds, metrics), ""]
    lig = dict(p.split(":") for p in args.lig_pairs.split(",") if p)
    ref_lig = next((l for l, s in lig.items() if s == args.ref), None)
    links = pairwise_rows(args.eval_root, tags, seeds)[1] + aug_metrics_for(args.eval_root, tags, seeds,
                                                                             with_links=True)[1]
    refs = {t: ref_lig for t in lig if ref_lig is not None}  # ligand-only tags against the ligand-only reference
    pvp = table_pair_vs_point(df, tags, args.ref, seeds, links, refs)
    if pvp:
        parts += ["## Pair head vs pointwise head\n", pvp, ""]
    ref_labels = {pair_label: point_label for pair_label, point_label, _ in links}
    for arm_lig, arm in lig.items():
        if arm == args.ref or ref_lig is None:
            continue
        parts += [f"## Shortcut gate: {arm} vs {args.ref}, against {arm_lig} vs {ref_lig}\n",
                  gate(df, arm, args.ref, arm_lig, ref_lig, seeds, metrics, ref_labels), ""]
    with open(args.out, "w") as fh:
        fh.write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main(sys.argv[1:])
