"""profile.py's report table and port_parity.py's join and verdict (no JAX computation)."""

import math

import numpy as np
import pandas as pd
import pytest

from aevplig_jax import port_parity, profile


def jax_row(step_mean: float, seconds_per_epoch: float, wait: float, gpu: float | None) -> dict:
    """A ``profile.json`` JAX row as profile.py writes it, plus the ``gpu_util`` collect adds."""
    return {"steps": 300, "trace_steps": 20, "warmup_steps": 5, "epochs": 200,
            "step_seconds": {"mean": step_mean, "p50": step_mean * 0.98, "p95": step_mean * 1.2},
            "first_step_seconds": 21.5, "segment_seconds": 300 * step_mean, "valid_seconds": 0.4,
            "steps_per_epoch": 748, "seconds_per_epoch": seconds_per_epoch,
            "minutes_per_seed": seconds_per_epoch * 200 / 60, "wait_fraction": wait,
            "padding": {"node": 0.18, "env": 0.305, "aev": 0.9},
            "capacities": {"ncap": 5120, "mcap": 14336, "kb": 7, "ka": 736, "batch": 128},
            "op_summary": {"source": "compiled HLO", "rows": [["fusion:kLoop", 120],
                                                              ["custom-call:__cublas$gemm", 60]]},
            "window": [1000.0, 1004.0], "gpu_util": gpu}


def torch_loader(seconds_per_epoch: float, wait: float) -> dict:
    return {"seconds_per_epoch": seconds_per_epoch,
            "minutes_per_seed": seconds_per_epoch * 200 / 60, "wait_fraction": wait,
            "window": [0.0, 60.0], "gpu_util": 0.45}


PROFILE = {"jax": {"onthefly": jax_row(0.011, 8.6, 0.004, 0.91),
                   "precomputed": jax_row(0.010, 7.9, 0.012, None)}}
TORCH = {"loaders": {"plain": torch_loader(20.0, 0.30), "workers": torch_loader(15.0, 0.05)},
         "profiler": {"steps": 200, "seconds": {"data": 2.0, "forward": 3.0, "backward": 4.0,
                                                "optimiser": 1.0},
                      "table": "key_averages.txt"}}
HEADER = ["loop", "s / epoch", "min / seed (200 epochs)", "GPU util", "loader wait", "node padding",
          "env padding", "AEV-slot padding"]


def main_table(report: str) -> list[list[str]]:
    """The cells of the table whose first column is ``loop``, header row first."""
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| loop |"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return [rows[0], *rows[2:]]  # drop the | --- | separator


def test_profile_report_table():
    report = profile.render_report({**PROFILE, "torch": TORCH})
    header, *rows = main_table(report)
    assert header == HEADER
    assert [row[0] for row in rows] == ["PyTorch", "PyTorch workers", "JAX precomputed",
                                        "JAX on-the-fly"]
    by_loop = {row[0]: row[1:] for row in rows}
    assert by_loop["PyTorch"] == ["20.0", "66.7", "45.0%", "30.0%", "n/a", "n/a", "n/a"]
    assert by_loop["PyTorch workers"] == ["15.0", "50.0", "45.0%", "5.0%", "n/a", "n/a", "n/a"]
    assert by_loop["JAX precomputed"] == ["7.9", "26.3", "n/a", "1.2%", "18.0%", "30.5%", "90.0%"]
    assert by_loop["JAX on-the-fly"] == ["8.6", "28.7", "91.0%", "0.4%", "18.0%", "30.5%", "90.0%"]
    # mean step 11 ms on the fly against 10 ms precomputed, on the same batches
    assert "on-the-fly overhead: 10.0%" in report
    assert "compute only" in report  # the env arrays are transferred in both modes
    assert "fusion:kLoop" in report and "custom-call:__cublas$gemm" in report  # op summary

    without_torch = profile.render_report(PROFILE)
    header, *rows = main_table(without_torch)
    assert [row[0] for row in rows] == ["PyTorch", "PyTorch workers", "JAX precomputed",
                                        "JAX on-the-fly"]
    for row in rows[:2]:
        assert row[1:] == ["n/a"] * 7
    assert rows[3][1:] == by_loop["JAX on-the-fly"]
    assert "on-the-fly overhead: 10.0%" in without_torch

    only_onthefly = profile.render_report({"jax": {"onthefly": PROFILE["jax"]["onthefly"]}})
    assert main_table(only_onthefly)[3][1:] == ["n/a"] * 7
    assert "on-the-fly overhead: n/a" in only_onthefly


def test_port_parity_join_and_verdict():
    ours = pd.DataFrame({"unique_id": ["1e66", "b", "c"], "truth": [6.3, 7.5, 5.25],
                         "pred": [6.1, 7.4, 5.0]})
    offsets = np.array([2e-4, -5e-4, 1e-4])

    # Reference with ids in another order; truth went through float32 (as the PyTorch dataset's y).
    reference = pd.DataFrame({"unique_id": ["c", "1e66", "b"],
                              "truth": np.float32([5.25, 6.3, 7.5]).astype(np.float64),
                              "pred": 0.0, "pred_0": ours["pred"].to_numpy()[[2, 0, 1]]
                              + offsets[[2, 0, 1]], "pred_1": 9.0})
    rows, summary = port_parity.compare(ours, reference)
    assert summary["join"] == "unique_id"
    assert (summary["n_ours"], summary["n_reference"], summary["n_compared"]) == (3, 3, 3)
    assert summary["complete"] and summary["truth_equal"]
    assert not summary["truth_by_row"]  # the reference rows are in another order
    assert summary["max_abs"] == pytest.approx(5e-4, rel=1e-6)
    assert summary["mean_abs"] == pytest.approx(np.abs(offsets).mean(), rel=1e-6)
    assert summary["worst"] == "b" and summary["passed"]
    assert list(rows["unique_id"]) == ["1e66", "b", "c"]  # our order
    np.testing.assert_allclose(rows["abs_diff"], np.abs(offsets), rtol=1e-6)

    # Reference without ids: aligned row by row, which the truth check vouches for.
    by_row = reference.drop(columns="unique_id").iloc[[1, 2, 0]].reset_index(drop=True)
    rows, summary = port_parity.compare(ours, by_row)
    assert summary["join"] == "row order" and summary["n_compared"] == 3
    assert summary["truth_by_row"] and summary["truth_equal"] and summary["passed"]
    assert summary["worst"] == 1 and summary["max_abs"] == pytest.approx(5e-4, rel=1e-6)

    # A difference of 2e-3 pK fails the 1e-3 acceptance.
    worse = reference.assign(pred_0=reference["pred_0"] + np.array([0.0, 2e-3, 0.0]))
    _, summary = port_parity.compare(ours, worse)
    assert summary["max_abs"] == pytest.approx(2e-3 + 2e-4, rel=1e-6)
    assert summary["worst"] == "1e66" and not summary["passed"]
    assert port_parity.compare(ours, worse, tolerance=3e-3)[1]["passed"]

    # A label that differs beyond float32 precision fails the truth check, and so the set.
    _, summary = port_parity.compare(ours, reference.assign(truth=reference["truth"] + 1e-3))
    assert not summary["truth_equal"] and not summary["passed"]

    # A reference row missing from the join: the comparison is incomplete, so it cannot pass.
    _, summary = port_parity.compare(ours, reference.iloc[:2])
    assert summary["n_compared"] == 2 and not summary["complete"] and not summary["passed"]
    _, summary = port_parity.compare(ours, by_row.iloc[:2])
    assert not summary["complete"] and not summary["truth_by_row"] and not summary["passed"]
    assert math.isfinite(summary["max_abs"])
