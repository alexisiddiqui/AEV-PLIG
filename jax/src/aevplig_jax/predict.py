"""Ensemble prediction over a store with the ``best`` checkpoints of training runs.

    python -m aevplig_jax.predict --store DIR --checkpoints GLOB --manifest FILE --out CSV
        [--split NAME] [--precision default|highest] [--workers 15] [--prefetch 4]
        [--prepare-workers 12] [--skip-verification]

``GLOB`` matches ``best`` checkpoint directories or their ``state.npz`` files; matches are sorted
in natural order (seed-2 before seed-10) and member ``i`` is ``pred_i``. Batches are the store's
ids (or ``--split``'s) in order, ``BATCH`` each, with capacities planned from this store's own
records, so a ligand larger than any training ligand still fits; the training manifest only
gives the label scaler. Writes ``unique_id, subset, series, ref_pdb, truth, pred, pred_0, ...``
(the paired tree's ``predictions_fep.csv`` layout): ``subset``, ``series`` and ``ref_pdb`` from
the record meta (empty when absent), ``truth`` the store label (empty when absent), ``pred`` the
member mean, all in label units. The store must pass ``store.require_verified`` unless
``--skip-verification`` opts out with a warning (synthetic stores, smoke runs).

Imports jax, numpy, pandas, the standard library and package modules.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import re
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from aevplig_jax import inference
from aevplig_jax.batching import StoreSource, prepare_all
from aevplig_jax.loading import LoaderConfig
from aevplig_jax.store import Store, check_verified

META_COLUMNS = ("subset", "series", "ref_pdb")


def _natural(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def checkpoint_files(pattern: str) -> list[Path]:
    """The ``state.npz`` files of the ``pattern`` matches, in natural order."""
    matches = sorted(glob.glob(pattern), key=_natural)
    files = [Path(m) / "state.npz" if Path(m).is_dir() else Path(m) for m in matches]
    if not files:
        raise FileNotFoundError(f"no checkpoints match {pattern}")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"no state.npz in {missing}")
    return files


def _text(value) -> str:
    return "" if value is None else str(value)


def predictions_frame(store: Store, ids: list[str], preds: np.ndarray) -> pd.DataFrame:
    """The output table for ``preds`` ``(members, len(ids))`` in label units."""
    meta = [store.records[rid]["meta"] for rid in ids]
    labels = [store.records[rid]["label"] for rid in ids]
    frame = pd.DataFrame({"unique_id": ids,
                          **{key: [_text(m.get(key)) for m in meta] for key in META_COLUMNS},
                          "truth": np.array([np.nan if v is None else v for v in labels], float),
                          "pred": preds.mean(axis=0)})
    for i, pred in enumerate(preds):
        frame[f"pred_{i}"] = pred
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--checkpoints", required=True,
                        help="glob of best checkpoint directories or state.npz files")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="the training run's manifest.json (label scaler)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", help="predict only this split (default: every record)")
    parser.add_argument("--precision", choices=("default", "highest"), default="default")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=12)
    parser.add_argument("--skip-verification", action="store_true",
                        help="use a store without a passing verification.json (synthetic stores "
                             "and smoke runs only; prints a warning)")
    args = parser.parse_args(argv)
    inference.enable_compilation_cache()

    with Store(args.store) as store:
        check_verified(store, args.skip_verification)
        manifest = json.loads(args.manifest.read_text())
        files = checkpoint_files(args.checkpoints)
        ids = store.split_ids(args.split) if args.split else list(store.ids)
        if not ids:
            raise ValueError(f"{args.store}: no records{f' in split {args.split}' if args.split else ''}")
        prepared = prepare_all(store, ids, args.prepare_workers)  # forks before JAX's backend starts
        plans = inference.sequential_plans(ids)
        cap = inference.plan_capacities(prepared, plans)
        source = StoreSource(store, cap, manifest["y_mean"], manifest["y_sd"], args.workers,
                             prepared=prepared)
        precision = (jax.default_matmul_precision("highest") if args.precision == "highest"
                     else contextlib.nullcontext())
        with precision:
            members = [inference.load_member(path)[:2] for path in files]
            preds = inference.predict_ids(inference.predict_function(), members, source, plans,
                                          LoaderConfig(workers=args.workers, prefetch=args.prefetch))
        frame = predictions_frame(store, ids, inference.destandardise(preds, manifest["y_mean"],
                                                                      manifest["y_sd"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    for i, path in enumerate(files):
        print(f"pred_{i}: {path}")
    print(f"wrote {args.out}: {len(frame)} rows, {len(files)} members, capacities {cap.to_json()}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
