"""Precomputed radial AEVs for the on-the-fly vs precomputed overhead A/B (``train.py --aev
precomputed``); the default training path computes them on the fly and never reads this table.

    python -m aevplig_jax.precompute_aev --store DIR --out FILE.npz [--workers 12] [--prefetch 4]

Writes, in store order, the 352 AEV columns of every ligand atom, computed with ``radial_aev_jax``
on sequential padded batches (the batch layout training uses): ``aev`` ``(sum L, 352)`` float32,
``offsets`` ``(N + 1,)`` int64 (record ``i`` owns rows ``offsets[i]:offsets[i + 1]``) and ``ids``.
``load_table`` maps them back to ids for ``StoreSource(aev_table=...)``.

Imports jax, numpy, the standard library and package modules.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import numpy as np

from aevplig_jax import inference
from aevplig_jax.aev import radial_aev_jax
from aevplig_jax.batching import Prepared, StoreSource, prepare_all
from aevplig_jax.loading import LoaderConfig, Prefetcher
from aevplig_jax.store import Store


@jax.jit
def _batch_aev(batch):
    return radial_aev_jax(batch.lig_xyz, batch.env_xyz, batch.env_species, batch.aev_nbr,
                          batch.aev_mask > 0)


def precompute(store: Store, prepared: dict[str, Prepared],
               config: LoaderConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(offsets, aev)`` for every record of ``store`` in store order."""
    plans = inference.sequential_plans(store.ids)
    cap = inference.plan_capacities(prepared, plans)
    source = StoreSource(store, cap, 0.0, 1.0, config.workers, prepared=prepared)
    sizes = np.array([len(prepared[rid].xyz) for rid in store.ids], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    chunks = []
    for plan, (batch, _) in Prefetcher(source, plans, config, to_device=inference.device_batch):
        n_nodes = int(sum(len(prepared[str(rid)].xyz) for rid in plan))
        chunks.append(np.asarray(_batch_aev(batch))[:n_nodes])
    return offsets, np.concatenate(chunks).astype(np.float32)


def write_table(path: Path, ids: list[str], offsets: np.ndarray, aev: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with open(pending, "wb") as handle:
        np.savez(handle, ids=np.array(ids, dtype=str), offsets=offsets, aev=aev)
    os.replace(pending, path)


def load_table(path: Path, store: Store) -> dict[str, np.ndarray]:
    """id -> ``(L, 352)`` rows of a table written for ``store`` (same ids in the same order)."""
    with np.load(path) as data:
        ids = [str(rid) for rid in data["ids"]]
        offsets, aev = data["offsets"], data["aev"]
    if ids != store.ids:
        raise ValueError(f"{path} was not computed from {store.path}: the ids differ")
    return {rid: aev[offsets[i]:offsets[i + 1]] for i, rid in enumerate(ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the table, .npz")
    parser.add_argument("--workers", type=int, default=12, help="prepare processes")
    parser.add_argument("--prefetch", type=int, default=4)
    args = parser.parse_args(argv)
    inference.enable_compilation_cache()
    with Store(args.store) as store:
        prepared = prepare_all(store, store.ids, args.workers)  # forks before JAX's backend starts
        offsets, aev = precompute(store, prepared, LoaderConfig(workers=args.workers,
                                                                prefetch=args.prefetch))
        write_table(args.out, store.ids, offsets, aev)
    print(f"wrote {args.out}: {len(offsets) - 1} records, {len(aev)} ligand atoms x {aev.shape[1]}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
