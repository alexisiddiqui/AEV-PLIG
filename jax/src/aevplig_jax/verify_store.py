"""Verify a record store against the original pickled graphs.

Each pickle is the original featurisers' ``graphs_dict[unique_id] = (c_size, features, edge_index,
edge_features)``: ``features`` a list of per-atom arrays of ``NODE_DIM`` columns (15 RDKit then 352
AEV), ``edge_index`` ``[src, dst]`` lists sorted by (src, dst), ``edge_features`` 4-element lists.
For every store id found in a pickle (or a seeded sample of them) the checks are:

- ``c_size``: ``c_size`` equals the store's ``L``;
- ``features_shape``: ``features`` is an ``(L, NODE_DIM)`` array (otherwise the column checks are
  skipped);
- ``rdkit``: ``lig_feat`` equals ``features[:, :15]`` exactly;
- ``aev``: the radial AEV of the JAX path training uses (``typed_within`` subset, ``neighbour_lists``,
  ``radial_aev_jax``) is within ``AEV_TOLERANCE`` (max abs) of ``features[:, 15:]``;
- ``edge_index`` / ``edge_features``: ``bond_src, bond_dst`` and ``bond_type`` equal them exactly,
  in the same order.

The AEV function is jitted once on static capacities (ligand rows, neighbour slots, typed atoms:
the store's largest ``L``, ``KA`` and ``A`` rounded up to a multiple of 32) and only the valid rows
are compared. ``verification.json`` holds the counts, the worst AEV difference, the
``store_fingerprint`` of the files verified (``store.store_fingerprint``: size of ``records.bin``,
sha256 of ``index.npz``; ``store.require_verified`` checks it) and ``passed``: every compared
record passes, at least one record is compared and, with ``--require-all-pickle-ids``, every
pickled id is in the store. The exit code is 1 when ``passed`` is false.

Imports numpy, the standard library and the package's store and AEV modules (numpy, scipy,
zstandard); pickles are read with ``pickle`` and numpy only, never torch, and jax is imported when
verification runs.

    python -m aevplig_jax.verify_store --store work/store/trainside --require-all-pickle-ids \\
        --pickle data/pdbbind.pickle --pickle data/bindingnet.pickle --pickle data/bindingdb.pickle \\
        --out work/store/trainside/verification.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aevplig_jax.aev import neighbour_lists, radial_aev_jax
from aevplig_jax.constants import EDGE_DIM, NODE_DIM, RDKIT_DIM
from aevplig_jax.pdb_atoms import typed_within
from aevplig_jax.store import Record, Store, store_fingerprint

AEV_TOLERANCE = 1e-4  # max abs difference of the AEV columns
PAD = 32  # capacities are multiples of this
EXAMPLES = 20  # failures and missing ids listed in the report
PROGRESS_EVERY = 10_000


def load_graphs(paths: list[Path]) -> dict[str, tuple]:
    """The pickles' ``graphs_dict`` entries by id (str); a later pickle's entry replaces an earlier
    one, as ``{**a, **b}`` in create_pytorch_data.py does.
    """
    graphs: dict[str, tuple] = {}
    for path in paths:
        with open(path, "rb") as handle:
            graphs.update((str(rid), graph) for rid, graph in pickle.load(handle).items())
    return graphs


def select_ids(ids: list[str], sample: int | None, seed: int = 0) -> list[str]:
    """``sample`` ids drawn without replacement (seeded) in their original order; all of them when
    ``sample`` is None or not smaller than ``len(ids)``.
    """
    if sample is None or sample >= len(ids):
        return list(ids)
    picks = np.sort(np.random.default_rng(seed).choice(len(ids), size=sample, replace=False))
    return [ids[i] for i in picks]


def _round_up(n: int) -> int:
    return max(PAD, -(-int(n) // PAD) * PAD)


def capacities(dims: np.ndarray) -> dict[str, int]:
    """Static shapes of the jitted AEV: ligand rows ``L``, neighbour slots ``K`` and typed atoms
    ``A``, from the store's ``dims`` ``[L, E, P, A, KA]``.
    """
    most = np.asarray(dims).reshape(-1, 5).max(axis=0, initial=0)
    return {"L": _round_up(most[0]), "K": _round_up(most[4]), "A": _round_up(most[3])}


def padded_inputs(record: Record, caps: dict[str, int]) -> tuple[np.ndarray, ...]:
    """``radial_aev_jax`` arguments for ``record`` padded to ``caps``: ligand rows past ``L`` and
    neighbour slots past each atom's count are masked, typed atoms past ``A`` have species 0.
    """
    lig_xyz = record.lig.xyz
    typed = typed_within(record.pocket, lig_xyz)
    env_xyz, env_species = record.pocket.xyz[typed], record.pocket.species[typed]
    n_lig, n_env = len(lig_xyz), len(env_xyz)
    lig = np.zeros((caps["L"], 3), dtype=np.float32)
    lig[:n_lig] = lig_xyz
    env = np.zeros((caps["A"], 3), dtype=np.float32)
    env[:n_env] = env_xyz
    species = np.zeros(caps["A"], dtype=np.int32)
    species[:n_env] = env_species
    nbr = np.zeros((caps["L"], caps["K"]), dtype=np.int32)
    mask = np.zeros((caps["L"], caps["K"]), dtype=bool)
    if n_env:  # A = 0: no neighbours, the AEV is zero
        nbr[:n_lig], mask[:n_lig] = neighbour_lists(lig_xyz, env_xyz, k=caps["K"])
    return lig, env, species, nbr, mask


def compare_record(record: Record, graph: tuple, aev: np.ndarray) -> tuple[list[str], float | None]:
    """Failed checks (in the module docstring's order) and the max abs AEV difference (None when
    the features are not ``(L, NODE_DIM)``). ``aev`` is the record's ``(L, AEV_DIM)`` JAX AEV.
    """
    c_size, features, edge_index, edge_features = graph
    n_lig = len(record.lig.xyz)
    failed = [] if c_size == n_lig else ["c_size"]
    try:
        features = np.asarray(features, dtype=np.float64)
    except ValueError:  # rows of different lengths
        features = None
    if features is None or features.shape != (n_lig, NODE_DIM):
        return [*failed, "features_shape"], None

    if not np.array_equal(features[:, :RDKIT_DIM], record.lig.feat):
        failed.append("rdkit")
    diff = float(np.abs(features[:, RDKIT_DIM:] - aev).max(initial=0.0))
    if math.isnan(diff):
        diff = math.inf
    if not diff < AEV_TOLERANCE:
        failed.append("aev")
    bonds = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    if not np.array_equal(bonds, np.stack([record.lig.bond_src, record.lig.bond_dst], axis=1)):
        failed.append("edge_index")
    bond_type = np.asarray(edge_features, dtype=np.float64).reshape(-1, EDGE_DIM)
    if not np.array_equal(bond_type, record.lig.bond_type):
        failed.append("edge_features")
    return failed, diff


def verify(store_path: Path, pickles: list[Path], sample: int | None = None, seed: int = 0,
           require_all_pickle_ids: bool = False) -> dict:
    """Check the store against the pickles; returns the report written to ``verification.json``."""
    import jax  # here, not at import: the module itself needs numpy only

    start = time.perf_counter()
    graphs = load_graphs(pickles)
    load_seconds = time.perf_counter() - start
    print(f"[verify_store] loaded {len(graphs)} graphs from {len(pickles)} pickles in "
          f"{load_seconds:.1f} s", flush=True)

    aev_fn = jax.jit(radial_aev_jax)
    fingerprint = store_fingerprint(store_path)  # of the files this run reads
    with Store(store_path) as store:
        caps = capacities(store.dims)
        in_store = set(store.ids)
        common = [rid for rid in store.ids if rid in graphs]
        missing_in_pickle = [rid for rid in store.ids if rid not in graphs]
        missing_in_store = [rid for rid in graphs if rid not in in_store]
        ids = select_ids(common, sample, seed)

        failures, by_source, by_check = [], Counter(), Counter()
        worst_id, worst = None, None
        for done, rid in enumerate(ids, 1):
            record = store.get(rid)
            aev = np.asarray(aev_fn(*padded_inputs(record, caps)))[:len(record.lig.xyz)]
            failed, diff = compare_record(record, graphs[rid], aev)
            if diff is not None and (worst is None or diff > worst):
                worst_id, worst = rid, diff
            if failed:
                by_source[record.source] += 1
                by_check.update(failed)
                if len(failures) < EXAMPLES:
                    failures.append({"id": rid, "source": record.source, "checks": failed,
                                     "aev_max_abs": diff})
            if done % PROGRESS_EVERY == 0:
                print(f"[verify_store] {done}/{len(ids)} records, {sum(by_source.values())} failed, "
                      f"{time.perf_counter() - start:.1f} s", flush=True)
        n_store = len(store)

    n_failed = sum(by_source.values())
    passed = bool(ids) and n_failed == 0 and not (require_all_pickle_ids and missing_in_store)
    return {
        "store": str(store_path),
        "pickles": [str(path) for path in pickles],
        "sample": sample,
        "seed": seed,
        "require_all_pickle_ids": require_all_pickle_ids,
        "store_fingerprint": fingerprint,
        "tolerance": AEV_TOLERANCE,
        "counts": {"store": n_store, "pickle": len(graphs), "compared": len(ids),
                   "passed": len(ids) - n_failed, "failed": n_failed,
                   "missing_in_pickle": len(missing_in_pickle),
                   "missing_in_store": len(missing_in_store)},
        "worst": {"id": worst_id, "aev_max_abs": worst},
        "failures_by_source": dict(by_source),
        "failures_by_check": dict(by_check),
        "failures": failures,
        "missing_in_pickle_ids": missing_in_pickle[:EXAMPLES],
        "missing_in_store_ids": missing_in_store[:EXAMPLES],
        "capacities": caps,
        "device": str(jax.devices()[0]),
        "load_seconds": round(load_seconds, 1),
        "seconds": round(time.perf_counter() - start, 1),
        "passed": passed,
    }


def write_report(path: Path, report: dict) -> None:
    """Write ``report`` as JSON through a temporary file and a rename."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=1) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True, help="store directory")
    parser.add_argument("--pickle", type=Path, action="append", required=True,
                        help="graphs pickle (repeat for several)")
    parser.add_argument("--out", type=Path, required=True, help="report, e.g. <store>/verification.json")
    parser.add_argument("--sample", type=int, help="verify this many random ids (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed of --sample")
    parser.add_argument("--require-all-pickle-ids", action="store_true",
                        help="fail when a pickled id is not in the store")
    args = parser.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be at least 1")

    report = verify(args.store, args.pickle, args.sample, args.seed, args.require_all_pickle_ids)
    write_report(args.out, report)
    counts, worst = report["counts"], report["worst"]
    print(f"[verify_store] {args.store}: {counts['compared']} compared, {counts['failed']} failed, "
          f"{counts['missing_in_pickle']} missing in pickle, {counts['missing_in_store']} missing in "
          f"store; worst AEV {worst['aev_max_abs']} ({worst['id']}); capacities {report['capacities']}; "
          f"{report['seconds']} s -> {'PASSED' if report['passed'] else 'FAILED'} ({args.out})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
