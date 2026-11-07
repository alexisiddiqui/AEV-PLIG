"""CASF, FEP and PGDH metrics of evaluate.py on fixture prediction CSVs."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aevplig_jax import evaluate

EVAL_PAIRED = Path(__file__).resolve().parents[2] / "paired" / "src" / "eval_paired.py"


def predictions(truth, pred, series=None, subset=None) -> pd.DataFrame:
    """A predictions CSV frame in predict.py's layout, with two members around ``pred``."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    n = len(truth)
    return pd.DataFrame({"unique_id": [f"lig {i}, x" for i in range(n)],
                         "subset": subset if subset is not None else [""] * n,
                         "series": series if series is not None else [""] * n,
                         "ref_pdb": ["3E63"] * n, "truth": truth, "pred": pred,
                         "pred_0": pred - 0.25, "pred_1": pred + 0.25})


def write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def fep_frame(rng) -> pd.DataFrame:
    """Series a (10 ligands, pred = 2 truth + 1: r = +1), b (12, pred = 3 - truth: r = -1),
    c (4, random) and d (10, constant pred: correlations undefined)."""
    parts = []
    t = rng.uniform(4, 10, 10)
    parts.append(("s1", "a", t, 2 * t + 1))
    t = rng.uniform(4, 10, 12)
    parts.append(("s1", "b", t, 3 - t))
    t = rng.uniform(4, 10, 4)
    parts.append(("s2", "c", t, rng.uniform(4, 10, 4)))
    t = rng.uniform(4, 10, 10)
    parts.append(("s2", "d", t, np.full(10, 5.0)))
    order = rng.permutation(36)  # rows interleaved: grouping must not rely on order
    frame = pd.concat([predictions(truth, pred, [series] * len(truth), [subset] * len(truth))
                       for subset, series, truth, pred in parts], ignore_index=True)
    return frame.iloc[order].reset_index(drop=True)


def test_evaluate_casf_matches_helpers(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.uniform(2, 12, 50)
    pred = truth + rng.normal(0, 1, 50)
    csv = write(predictions(truth, pred), tmp_path / "predictions_test.csv")
    assert evaluate.main(["casf", "--predictions", str(csv), "--out-dir", str(tmp_path)]) == 0
    summary = pd.read_csv(tmp_path / "casf_summary.csv").set_index("model")
    assert list(summary.index) == ["ensemble", "pred_0", "pred_1"]
    for name, values in (("ensemble", pred), ("pred_0", pred - 0.25), ("pred_1", pred + 0.25)):
        row = summary.loc[name]
        assert row["n"] == 50
        assert row["pearson"] == pytest.approx(np.corrcoef(truth, values)[0, 1], rel=1e-12)  # helpers.pearson
        assert row["rmse"] == pytest.approx(math.sqrt(((truth - values) ** 2).mean(axis=0)), rel=1e-12)


def test_evaluate_fep_weighted_metrics(tmp_path):
    rng = np.random.default_rng(1)
    frame = fep_frame(rng)
    csv = write(frame, tmp_path / "predictions_fep.csv")
    assert evaluate.main(["fep", "--predictions", str(csv), "--out-dir", str(tmp_path)]) == 0

    per = pd.read_csv(tmp_path / "fep_per_series.csv", dtype={"ref_pdb": str})
    assert list(per.columns) == ["subset", "series", "ref_pdb", "n", "pearson", "spearman", "kendall", "rmse"]
    assert list(per["series"]) == ["a", "b", "c", "d"] and list(per["n"]) == [10, 12, 4, 10]
    assert list(per["ref_pdb"]) == ["3E63"] * 4
    per = per.set_index("series")
    for metric in ("pearson", "spearman", "kendall"):
        assert per.loc["a", metric] == pytest.approx(1.0) and per.loc["b", metric] == pytest.approx(-1.0)
        assert math.isnan(per.loc["d", metric])
    c = frame[frame["series"] == "c"]
    assert per.loc["c", "pearson"] == pytest.approx(stats.pearsonr(c["truth"], c["pred"])[0])
    assert per.loc["c", "kendall"] == pytest.approx(stats.kendalltau(c["truth"], c["pred"])[0])
    rmse = {s: math.sqrt(((g["pred"] - g["truth"]) ** 2).mean()) for s, g in frame.groupby("series")}
    for s, value in rmse.items():
        assert per.loc[s, "rmse"] == pytest.approx(value)

    summary = pd.read_csv(tmp_path / "fep_summary.csv")
    assert list(summary.columns) == ["scope", "weighting", "n_series", "n_items", "pearson", "spearman",
                                     "kendall", "rmse"]
    rows = summary.set_index(["scope", "weighting"])
    # n >= 10: series a, b, d (32 ligands); d has no correlation, so correlations average a and b.
    weighted, unweighted = rows.loc[("n>=10", "weighted")], rows.loc[("n>=10", "unweighted")]
    assert (weighted["n_series"], weighted["n_items"]) == (3, 32)
    for metric in ("pearson", "spearman", "kendall"):
        assert weighted[metric] == pytest.approx((10 * 1 - 12 * 1) / 22)
        assert unweighted[metric] == pytest.approx(0.0, abs=1e-12)
    assert weighted["rmse"] == pytest.approx((10 * rmse["a"] + 12 * rmse["b"] + 10 * rmse["d"]) / 32)
    assert unweighted["rmse"] == pytest.approx((rmse["a"] + rmse["b"] + rmse["d"]) / 3)
    every = rows.loc[("all", "weighted")]
    assert (every["n_series"], every["n_items"]) == (4, 36)
    assert every["pearson"] == pytest.approx((10 - 12 + 4 * per.loc["c", "pearson"]) / 26)


@pytest.mark.skipif(not EVAL_PAIRED.exists(), reason="paired/ tree absent (code-only mirror)")
def test_fep_tables_agree_with_eval_paired(tmp_path, monkeypatch):
    monkeypatch.setenv("AEVPLIG_REPO", str(tmp_path))  # the module puts it on sys.path at import
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("eval_paired_reference", EVAL_PAIRED)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    frame = fep_frame(np.random.default_rng(2))
    want_per = reference.per_series_metrics(frame, ["subset", "series"])
    want = reference.summarise(want_per, (reference.MIN_N_FEP,))
    per, summary = evaluate.fep_tables(frame)
    pd.testing.assert_frame_equal(per.drop(columns="ref_pdb"), want_per, check_dtype=False)
    pd.testing.assert_frame_equal(summary, want, check_dtype=False)


def test_evaluate_pgdh_spearman(tmp_path):
    rng = np.random.default_rng(3)
    truth = rng.uniform(5, 9, 18)
    pred = truth + rng.normal(0, 0.8, 18)
    series = ["S2"] * 7 + ["S1"] * 11
    csv = write(predictions(truth, pred, series, ["pgdh"] * 18), tmp_path / "predictions_pgdh.csv")
    assert evaluate.main(["pgdh", "--predictions", str(csv), "--out-dir", str(tmp_path)]) == 0
    summary = pd.read_csv(tmp_path / "pgdh_summary.csv")
    assert list(summary.columns) == ["scope", "n", "spearman"]
    assert list(summary["scope"]) == ["pooled", "S1", "S2"] and list(summary["n"]) == [18, 11, 7]
    assert summary["spearman"][0] == pytest.approx(stats.spearmanr(truth, pred)[0])
    assert summary["spearman"][1] == pytest.approx(stats.spearmanr(truth[7:], pred[7:])[0])
    assert summary["spearman"][2] == pytest.approx(stats.spearmanr(truth[:7], pred[:7])[0])


def test_evaluate_summary_table(tmp_path):
    rng = np.random.default_rng(4)
    truth = rng.uniform(2, 12, 30)
    write(predictions(truth, truth + rng.normal(0, 1, 30)), tmp_path / "test.csv")
    write(fep_frame(rng), tmp_path / "fep.csv")
    pgdh = predictions(truth[:12], truth[:12] + 1, ["S1"] * 12, ["pgdh"] * 12)
    write(pgdh, tmp_path / "pgdh.csv")
    for kind, name in (("casf", "test"), ("fep", "fep"), ("pgdh", "pgdh")):
        assert evaluate.main([kind, "--predictions", str(tmp_path / f"{name}.csv"),
                              "--out-dir", str(tmp_path)]) == 0
    assert evaluate.main(["summary", "--eval-dir", str(tmp_path), "--title", "jax"]) == 0
    text = (tmp_path / "SUMMARY.md").read_text()
    assert text.startswith("# jax")
    casf = pd.read_csv(tmp_path / "casf_summary.csv").set_index("model").loc["ensemble"]
    assert f"| CASF-2016 | ensemble | 30 | {casf['pearson']:.3f} |" in text
    assert "| FEP | n>=10 weighted | 32 |" in text
    assert "| PGDH | pooled | 12 |" in text
    assert "members: 2" in text
