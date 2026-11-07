"""Weight-port parity: a PyTorch GATv2Net ported to JAX against that model's PyTorch predictions.

    python -m aevplig_jax.port_parity --npz FILE --store-root DIR --reference-dir DIR --out DIR
        [--reference-column pred_0] [--tolerance 1e-3] [--workers 15] [--prefetch 4]
        [--prepare-workers 12]

``--npz`` (``jobs/export_torch_model.py``, run under the AEV-PLIG env) holds every state-dict tensor
under its key plus the label scaler's ``y_mean`` and ``y_sd``. The arrays go through
``torch_port.port_state_dict``; the ported model predicts, in evaluation mode at matmul precision
``highest``, the ``test`` split of ``<store-root>/trainside`` (CASF) and every record of
``<store-root>/fep``, with capacities planned from each store's own records (as predict.py), and the
predictions are de-standardised with the scaler. Both stores must pass
``store.require_verified``.

Each set is compared with ``--reference-column`` of the PyTorch evaluator's CSV in
``--reference-dir`` (``predictions_test.csv``, ``predictions_fep.csv``): joined on ``unique_id``
when the reference has that column, otherwise row by row (the paired tree's ``eval_paired.py``
writes ``unique_id`` into ``predictions_test.csv`` only for test sets with row ids, which the
parent's ``.pt`` files lack). ``truth_equal``: the reference ``truth`` equals the store label on
every compared row (to ``TRUTH_ATOL``: the PyTorch datasets hold labels in float32);
``truth_by_row``: the same holds row by row in file order, which shows that the store's split order
is the reference's. A set passes when every row of both sides is compared, the truths are equal and
the max abs prediction difference is below ``--tolerance``.

Writes ``<out>/parity.csv`` (one row per set), ``<out>/parity_rows.csv`` (one row per complex) and
``<out>/REPORT.md``; exits 1 when a set fails.

Imports jax, numpy, pandas, the standard library and package modules (no torch).
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from aevplig_jax import inference
from aevplig_jax.batching import StoreSource, prepare_all
from aevplig_jax.loading import LoaderConfig
from aevplig_jax.store import Store, require_verified
from aevplig_jax.torch_port import port_state_dict

TOLERANCE = 1e-3  # pK, design acceptance 2
TRUTH_ATOL = 1e-5  # labels up to ~15 pK round-trip through float32 within 1e-6
SCALER_KEYS = ("y_mean", "y_sd")
# name, store under --store-root, split (None: every record), reference CSV
SETS = (("casf", "trainside", "test", "predictions_test.csv"),
        ("fep", "fep", None, "predictions_fep.csv"))


def _close(a, b) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0.0, atol=TRUTH_ATOL,
                            equal_nan=True))


def compare(ours: pd.DataFrame, reference: pd.DataFrame, column: str = "pred_0",
            tolerance: float = TOLERANCE) -> tuple[pd.DataFrame, dict]:
    """The per-complex table and the summary of ``ours`` (``unique_id, truth, pred`` in store
    order) against ``reference`` (``[unique_id,] truth, <column>`` in file order).

    Rows (in our order): ``unique_id, truth, truth_reference, pred, reference, abs_diff``. Summary:
    ``join``, ``n_ours``, ``n_reference``, ``n_compared``, ``complete``, ``truth_equal``,
    ``truth_by_row``, ``max_abs``, ``mean_abs``, ``worst`` (the id, or the row index for a row join)
    and ``passed``. A repeated id on either side raises ``pandas.errors.MergeError``.
    """
    ours = ours[["unique_id", "truth", "pred"]].reset_index(drop=True)
    reference = reference.reset_index(drop=True)
    same_length = len(ours) == len(reference)
    truth_by_row = same_length and _close(ours["truth"], reference["truth"])
    picked = reference[["truth", column]].rename(columns={"truth": "truth_reference",
                                                          column: "reference"})
    if "unique_id" in reference.columns:
        join = "unique_id"
        picked.insert(0, "unique_id", reference["unique_id"].astype(str))
        rows = ours.merge(picked, on="unique_id", how="inner", validate="one_to_one", sort=False)
        complete = same_length and len(rows) == len(ours)
    else:
        join = "row order"
        n = min(len(ours), len(reference))
        rows = pd.concat([ours.iloc[:n], picked.iloc[:n]], axis=1)
        complete = same_length
    rows = rows[["unique_id", "truth", "truth_reference", "pred", "reference"]].copy()
    rows["abs_diff"] = (rows["pred"] - rows["reference"]).abs()
    diffs = rows["abs_diff"].to_numpy(dtype=float)
    max_abs = float(np.max(diffs)) if len(diffs) else math.nan  # NaN propagates
    worst = None
    if len(diffs) and not np.all(np.isnan(diffs)):
        at = int(np.nanargmax(diffs))
        worst = str(rows["unique_id"].iloc[at]) if join == "unique_id" else at
    truth_equal = len(rows) > 0 and _close(rows["truth"], rows["truth_reference"])
    summary = {"join": join, "n_ours": len(ours), "n_reference": len(reference),
               "n_compared": len(rows), "complete": bool(complete),
               "truth_equal": bool(truth_equal), "truth_by_row": bool(truth_by_row),
               "max_abs": max_abs,
               "mean_abs": float(np.mean(diffs)) if len(diffs) else math.nan, "worst": worst,
               "passed": bool(complete and truth_equal and max_abs < tolerance)}
    return rows, summary


def read_reference(path: Path) -> pd.DataFrame:
    """A reference CSV with ``unique_id`` as text (PDB codes such as 1e66 are not numbers) and only
    empty cells as missing."""
    return pd.read_csv(path, dtype={"unique_id": str}, keep_default_na=False, na_values=[""])


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def render_report(summaries: dict[str, dict], info: dict) -> str:
    """REPORT.md: inputs, one table row per set and the verdict."""
    lines = ["# Weight-port parity", "",
             f"Ported `{info['npz']}` (state dict and scaler: y_mean {info['y_mean']:.6f}, y_sd "
             f"{info['y_sd']:.6f}) with `torch_port.port_state_dict`; JAX predictions in "
             f"evaluation mode at matmul precision highest on {info['devices']}, against column "
             f"`{info['column']}` of `{info['reference_dir']}/predictions_*.csv`. Acceptance: "
             f"max abs difference below {info['tolerance']:g} pK on every set, every row compared, "
             f"truths equal (to {TRUTH_ATOL:g}).", "",
             "| set | join | rows ours / reference / compared | truth equal | truth by row "
             "| max abs | mean abs | worst | passed |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['join']} | {s['n_ours']} / {s['n_reference']} / "
                     f"{s['n_compared']} | {_yes(s['truth_equal'])} | {_yes(s['truth_by_row'])} | "
                     f"{s['max_abs']:.3g} | {s['mean_abs']:.3g} | {s['worst']} | "
                     f"{_yes(s['passed'])} |")
    passed = all(s["passed"] for s in summaries.values())
    lines += ["", f"Verdict: {'PASSED' if passed else 'FAILED'}.", "",
              "`truth by row` compares the labels in file order: yes shows the store's split order "
              "is the reference's (needed for the per-seed CSVs, which carry no ids).", ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", type=Path, required=True,
                        help="jobs/export_torch_model.py output: state dict + y_mean, y_sd")
    parser.add_argument("--store-root", type=Path, required=True, help="holds trainside/ and fep/")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="holds predictions_test.csv and predictions_fep.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-column", default="pred_0")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with np.load(args.npz) as data:
        state_dict = {key: data[key] for key in data.files if key not in SCALER_KEYS}
        y_mean, y_sd = float(data["y_mean"]), float(data["y_sd"])
    params, bn_state = port_state_dict(state_dict)  # numpy only
    inference.enable_compilation_cache()
    summaries, tables = {}, []
    with contextlib.ExitStack() as stack:
        sets = []
        for name, store_name, split, reference in SETS:  # prepare_all forks before JAX starts
            store = stack.enter_context(Store(args.store_root / store_name))
            require_verified(store)
            ids = store.split_ids(split) if split else list(store.ids)
            if not ids:
                raise ValueError(f"{store.path}: no records{f' in split {split}' if split else ''}")
            prepared = prepare_all(store, ids, args.prepare_workers)
            sets.append((name, store, ids, prepared, reference))
        config = LoaderConfig(workers=args.workers, prefetch=args.prefetch)
        member = jax.device_put((params, bn_state))
        with jax.default_matmul_precision("highest"):
            predict_fn = inference.predict_function()
            for name, store, ids, prepared, reference in sets:
                plans = inference.sequential_plans(ids)
                source = StoreSource(store, inference.plan_capacities(prepared, plans), y_mean,
                                     y_sd, args.workers, prepared=prepared)
                pred = inference.predict_ids(predict_fn, [member], source, plans, config)[0]
                labels = [store.records[rid]["label"] for rid in ids]
                ours = pd.DataFrame({"unique_id": ids,
                                     "truth": [math.nan if v is None else float(v) for v in labels],
                                     "pred": inference.destandardise(pred, y_mean, y_sd)})
                rows, summaries[name] = compare(
                    ours, read_reference(args.reference_dir / reference), args.reference_column,
                    args.tolerance)
                tables.append(rows.assign(set=name))
                print(f"{name}: {summaries[name]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"set": name, **s} for name, s in summaries.items()]).to_csv(
        args.out / "parity.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(args.out / "parity_rows.csv", index=False)
    info = {"npz": str(args.npz), "y_mean": y_mean, "y_sd": y_sd, "column": args.reference_column,
            "reference_dir": str(args.reference_dir), "tolerance": args.tolerance,
            "devices": ", ".join(str(device) for device in jax.devices())}
    (args.out / "REPORT.md").write_text(render_report(summaries, info))
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)
    return 0 if all(s["passed"] for s in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
