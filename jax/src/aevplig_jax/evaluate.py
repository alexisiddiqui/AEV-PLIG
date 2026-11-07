"""Metrics of prediction CSVs in predict.py's layout (``unique_id, subset, series, ref_pdb, truth,
pred, pred_0, ...``).

    python -m aevplig_jax.evaluate casf --predictions CSV [--out-dir DIR]
    python -m aevplig_jax.evaluate fep --predictions CSV [--out-dir DIR]
    python -m aevplig_jax.evaluate pgdh --predictions CSV [--out-dir DIR]
    python -m aevplig_jax.evaluate summary --eval-dir DIR [--title TEXT]

- casf: Pearson and RMSE of ``truth`` against ``pred`` and each ``pred_i``, as helpers.py
  (``casf_summary.csv``: model, n, pearson, rmse).
- fep: per (subset, series) n, Pearson, Spearman, Kendall (NaN below 2 ligands or for constant
  input) and RMSE (``fep_per_series.csv``), and their means over the series with n >= 10 and over
  all series, weighted by n and unweighted, over the finite values (``fep_summary.csv``: scope,
  weighting, n_series, n_items, pearson, spearman, kendall, rmse). The definitions of
  ``paired/src/eval_paired.py`` (``series_metrics``, ``per_series_metrics``, ``weighted_mean``,
  ``summarise``), reimplemented.
- pgdh: Spearman over all ligands and per series (``pgdh_summary.csv``: scope, n, spearman).
- summary: ``SUMMARY.md`` in DIR, one table from the summary CSVs found there.

``--out-dir`` defaults to the predictions CSV's directory.

Imports numpy, pandas, scipy, the standard library and package modules.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from aevplig_jax.inference import pearson, rmse

MIN_N_FEP = 10
METRICS = ["pearson", "spearman", "kendall", "rmse"]
SUMMARY_COLUMNS = ["scope", "weighting", "n_series", "n_items", *METRICS]
TEXT_COLUMNS = ("unique_id", "subset", "series", "ref_pdb")
MEMBER = re.compile(r"pred_\d+$")


def read_predictions(path: Path) -> pd.DataFrame:
    """A predictions CSV with the text columns kept as written (``3E63`` stays a string)."""
    return pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=False,
                       na_values={"truth": [""]})


def member_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if MEMBER.match(c)]


def casf_table(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["truth"].to_numpy(float)
    rows = [{"model": "ensemble" if column == "pred" else column, "n": len(frame),
             "pearson": pearson(truth, frame[column].to_numpy(float)),
             "rmse": rmse(truth, frame[column].to_numpy(float))}
            for column in ["pred", *member_columns(frame)]]
    return pd.DataFrame(rows, columns=["model", "n", "pearson", "rmse"])


def _correlated(truth: np.ndarray, pred: np.ndarray) -> bool:
    """Correlations are defined: at least 2 items and neither side constant."""
    return len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(pred) > 0


def series_metrics(truth, pred) -> dict:
    """n, Pearson, Spearman, Kendall (NaN below 2 items or for constant input) and RMSE."""
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    out = {"n": int(len(truth)), "pearson": math.nan, "spearman": math.nan, "kendall": math.nan,
           "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))) if len(truth) else math.nan}
    if _correlated(truth, pred):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["pearson"] = float(stats.pearsonr(truth, pred)[0])
            out["spearman"] = float(stats.spearmanr(truth, pred)[0])
            out["kendall"] = float(stats.kendalltau(truth, pred)[0])
    return out


def per_series_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **series_metrics(group["truth"], group["pred"])})
    return pd.DataFrame(rows, columns=[*group_cols, "n", *METRICS])


def weighted_mean(values, weights) -> float:
    """Mean of the finite values, weighted; NaN when none is finite."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else math.nan


def summarise(per_series: pd.DataFrame, min_ns=(MIN_N_FEP,)) -> pd.DataFrame:
    """Weighted (weights = n) and unweighted means over series with n >= each min_n, and over all."""
    rows = []
    scopes = [(f"n>={m}", per_series["n"] >= m) for m in min_ns] + [("all", per_series["n"] >= 1)]
    for scope, selected in scopes:
        sub = per_series[selected]
        for weighting in ("weighted", "unweighted"):
            weights = sub["n"] if weighting == "weighted" else np.ones(len(sub))
            row = {"scope": scope, "weighting": weighting, "n_series": int(len(sub)),
                   "n_items": int(sub["n"].sum())}
            row.update({m: weighted_mean(sub[m], weights) for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def fep_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(per_series, summary)``; per-series rows carry the series' first ``ref_pdb``."""
    groups = ["subset", "series"]
    per = per_series_metrics(frame, groups)
    refs = frame.groupby(groups, sort=True)["ref_pdb"].first().tolist()
    per.insert(2, "ref_pdb", refs)
    return per, summarise(per.drop(columns="ref_pdb"))


def spearman(truth, pred) -> float:
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    if not _correlated(truth, pred):
        return math.nan
    return float(stats.spearmanr(truth, pred)[0])


def pgdh_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "pooled", "n": len(frame), "spearman": spearman(frame["truth"], frame["pred"])}]
    rows += [{"scope": series, "n": len(group), "spearman": spearman(group["truth"], group["pred"])}
             for series, group in frame.groupby("series", sort=True)]
    return pd.DataFrame(rows, columns=["scope", "n", "spearman"])


def _fmt(value) -> str:
    if value is None or (isinstance(value, str) and not value):
        return ""
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def render_summary(eval_dir: Path, title: str) -> str:
    """One markdown table (dataset, scope, n, pearson, spearman, kendall, rmse) from the
    ``casf_summary.csv``, ``fep_summary.csv`` and ``pgdh_summary.csv`` present in ``eval_dir``."""
    lines, notes, table = [f"# {title}", ""], [], []

    def add(dataset, scope, n, pearson_=None, spearman_=None, kendall_=None, rmse_=None):
        values = " | ".join(_fmt(v) for v in (pearson_, spearman_, kendall_, rmse_))
        table.append(f"| {dataset} | {scope} | {int(n)} | {values} |")

    casf_path, fep_path, pgdh_path = (eval_dir / f"{name}_summary.csv" for name in ("casf", "fep", "pgdh"))
    if casf_path.exists():
        casf = pd.read_csv(casf_path)
        ensemble = casf[casf["model"] == "ensemble"].iloc[0]
        add("CASF-2016", "ensemble", ensemble["n"], ensemble["pearson"], rmse_=ensemble["rmse"])
        members = casf[casf["model"] != "ensemble"]
        notes.append(f"members: {len(members)}; CASF member Pearson mean {members['pearson'].mean():.3f} "
                     f"(min {members['pearson'].min():.3f}, max {members['pearson'].max():.3f})")
    if fep_path.exists():
        fep = pd.read_csv(fep_path)
        for _, row in fep[fep["scope"] == f"n>={MIN_N_FEP}"].iterrows():
            add("FEP", f"{row['scope']} {row['weighting']}", row["n_items"], row["pearson"],
                row["spearman"], row["kendall"], row["rmse"])
        every = fep[fep["scope"] == "all"].iloc[0]
        selected = fep[fep["scope"] == f"n>={MIN_N_FEP}"].iloc[0]
        notes.append(f"FEP: {selected['n_series']} of {every['n_series']} series have n >= "
                     f"{MIN_N_FEP} ({selected['n_items']} of {every['n_items']} ligands)")
    if pgdh_path.exists():
        for _, row in pd.read_csv(pgdh_path, dtype={"scope": str}).iterrows():
            add("PGDH", row["scope"], row["n"], spearman_=row["spearman"])
    lines += ["| dataset | scope | n | pearson | spearman | kendall | rmse |",
              "|---|---|---|---|---|---|---|", *table, ""]
    lines += [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("casf", "fep", "pgdh"):
        command = commands.add_parser(name)
        command.add_argument("--predictions", type=Path, required=True)
        command.add_argument("--out-dir", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("--eval-dir", type=Path, required=True)
    summary.add_argument("--title", default="Evaluation")
    args = parser.parse_args(argv)

    if args.command == "summary":
        text = render_summary(args.eval_dir, args.title)
        (args.eval_dir / "SUMMARY.md").write_text(text)
        print(text, end="")
        return 0
    frame = read_predictions(args.predictions)
    out_dir = args.out_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "casf":
        tables = {"casf_summary.csv": casf_table(frame)}
    elif args.command == "fep":
        per, summary_table = fep_tables(frame)
        tables = {"fep_per_series.csv": per, "fep_summary.csv": summary_table}
    else:
        tables = {"pgdh_summary.csv": pgdh_table(frame)}
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False)
    print(next(reversed(tables.values())).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
