"""Build a record store from a processed CSV and the extracted archives, or from a generic CSV.

Training-side sources (``--source``) follow the three generate scripts for paths and ligand kind
(generate_pdbbind_graphs.py:236-262, generate_bindingnet_graphs.py:236-260,
generate_bindingdb_graphs.py:228-250) and create_pytorch_data.py:26-44 for splits: rows with
``max_tanimoto_fep_benchmark < 0.9`` keep their split (PDBbind ``split_core``, otherwise ``train``),
every other row is stored with split ``excluded``. Records keep the processed CSV's row order, so
merging the pdbbind, bindingnet and bindingdb stores in that order gives create_pytorch_data.py's
order. Generic mode (``--generic``) reads ``unique_id, sdf_file, pdb_file`` and an optional label
column; its records have split ``predict``.

A complex whose ligand RDKit cannot read, whose ligand has an element or bond type outside the
feature sets (``ValueError``) or whose files are missing is skipped and listed in
``build_summary.json`` with the reason, as the original featurisers skip them. A repeated id keeps
its first row. With ``--workers N`` each worker writes a part store for a contiguous slice of rows
and the parts are merged in slice order.

    python -m aevplig_jax.build_store --source pdbbind --processed-csv data/pdbbind_processed.csv \\
        --root EXTRACTED --keys data/PDB_Atom_Keys.csv --out store/pdbbind --workers 24
    python -m aevplig_jax.build_store --generic --csv fep_benchmark.csv --label-column pK \\
        --keys data/PDB_Atom_Keys.csv --out store/fep
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import platform
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import rdkit
import scipy
import zstandard

from aevplig_jax.constants import species_table
from aevplig_jax.pdb_atoms import read_pocket
from aevplig_jax.rdkit_features import ligand_arrays, read_ligand
from aevplig_jax.store import STORE_FILES, Record, Store, StoreWriter, merge_stores

POCKET_RADIUS = 10.0  # A, protein atoms kept around the ligand
TANIMOTO_LIMIT = 0.9  # create_pytorch_data.py keeps max_tanimoto_fep_benchmark < 0.9
EXCLUDED = "excluded"
GENERIC_SPLIT = "predict"


class Row(NamedTuple):
    """One complex to build: where its files are and what the store records about it."""

    row: int  # position in the CSV, for ordering failures
    id: str
    ligand: str  # ligand file path
    kind: str  # "sdf" or "mol2", for read_ligand
    protein: str  # protein PDB path
    label: float
    split: str
    source: str
    meta: dict


@dataclass(frozen=True)
class Source:
    """Columns and path templates of one processed CSV."""

    id_column: str
    label_column: str
    split_column: str | None  # None: rows that are not excluded are "train"
    kind: str
    paths: Callable[[dict], tuple[str, str]]  # row -> (ligand, protein) relative to --root
    text_columns: tuple[str, ...]  # read as str
    meta_columns: tuple[str, ...]


def _pdbbind_paths(row: dict) -> tuple[str, str]:
    pdb = row["PDB_code"]
    folder = f"data/pdbbind/{'refined-set' if row['refined'] else 'general-set'}/{pdb}"
    return f"{folder}/{pdb}_ligand.mol2", f"{folder}/{pdb}_protein.pdb"


def _bindingnet_paths(row: dict) -> tuple[str, str]:
    pdb, target, compnd = row["pdb"], row["target"], row["compnd"]
    folder = f"data/bindingnet/from_chembl_client/{pdb}"
    return f"{folder}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf", f"{folder}/rec_h_opt.pdb"


def _bindingdb_paths(row: dict) -> tuple[str, str]:
    folder = f"data/bindingdb/surflex/{row['folder']}"
    return f"{folder}/{row['mol2_file']}", f"{folder}/{row['pdb_file']}"


SOURCES = {
    "pdbbind": Source("PDB_code", "-logKd/Ki", "split_core", "mol2", _pdbbind_paths,
                      ("PDB_code",),
                      ("refined", "core", "split_core", "max_tanimoto_fep_benchmark")),
    "bindingnet": Source("unique_identify", "-logAffi", None, "sdf", _bindingnet_paths,
                         ("unique_identify", "pdb", "target", "compnd"),
                         ("pdb", "target", "compnd", "max_tanimoto_fep_benchmark")),
    "bindingdb": Source("unique_id", "pK", None, "mol2", _bindingdb_paths,
                        ("unique_id", "folder", "pdb_file", "mol2_file"),
                        ("folder", "pdb_file", "mol2_file", "max_tanimoto_fep_benchmark")),
}


def split_rule(split: str, max_tanimoto_fep_benchmark: float) -> str:
    """``split`` for the rows create_pytorch_data.py keeps, ``excluded`` for the rest (NaN too)."""
    return split if max_tanimoto_fep_benchmark < TANIMOTO_LIMIT else EXCLUDED


def _plain(value):
    """A JSON-ready scalar: numpy scalars unwrapped, NaN as None."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def source_rows(name: str, processed_csv: Path, root: Path, limit: int | None = None) -> list[Row]:
    """Rows of a processed CSV (``index_col=0``, file order), paths under ``root``."""
    spec = SOURCES[name]
    frame = pd.read_csv(processed_csv, index_col=0, nrows=limit,
                        dtype={column: str for column in spec.text_columns})
    rows = []
    for i, row in enumerate(frame.to_dict("records")):
        ligand, protein = spec.paths(row)
        split = row[spec.split_column] if spec.split_column else "train"
        meta = {"ligand": ligand, "protein": protein,
                **{column: _plain(row[column]) for column in spec.meta_columns}}
        rows.append(Row(row=i, id=row[spec.id_column], ligand=str(Path(root) / ligand),
                        kind=spec.kind, protein=str(Path(root) / protein),
                        label=float(row[spec.label_column]),
                        split=split_rule(split, row["max_tanimoto_fep_benchmark"]),
                        source=name, meta=meta))
    return rows


def generic_rows(csv: Path, label_column: str | None = None, limit: int | None = None) -> list[Row]:
    """Rows of a ``unique_id, sdf_file, pdb_file`` CSV; source = the CSV's file stem.

    Labels come from ``label_column`` (which must exist) or, when it is None, from a ``label``
    column if there is one, else NaN. Every other column goes into ``meta``.
    """
    frame = pd.read_csv(csv, nrows=limit,
                        dtype={"unique_id": str, "sdf_file": str, "pdb_file": str})
    if label_column is not None and label_column not in frame.columns:
        raise ValueError(f"{csv} has no column {label_column!r}")
    column = label_column or "label"
    has_label = column in frame.columns
    rows = []
    for i, row in enumerate(frame.to_dict("records")):
        meta = {key: _plain(value) for key, value in row.items() if key not in ("unique_id", column)}
        rows.append(Row(row=i, id=str(row["unique_id"]), ligand=str(row["sdf_file"]), kind="sdf",
                        protein=str(row["pdb_file"]),
                        label=float(row[column]) if has_label else math.nan,
                        split=GENERIC_SPLIT, source=Path(csv).stem, meta=meta))
    return rows


def make_record(row: Row, species: dict[str, int]) -> Record:
    """Read one complex. Raises ``OSError`` for missing files and ``ValueError`` for ligands RDKit
    cannot read or the features cannot encode.
    """
    for what, path in (("ligand", row.ligand), ("protein", row.protein)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing {what} file {path}")
    mol = read_ligand(Path(row.ligand), row.kind)
    if mol is None:
        raise ValueError(f"RDKit returned no molecule for {row.ligand}")
    lig = ligand_arrays(mol)
    pocket = read_pocket(Path(row.protein), lig.xyz, species, radius=POCKET_RADIUS)
    return Record(id=row.id, lig=lig, pocket=pocket, label=row.label, source=row.source,
                  split=row.split, meta=row.meta)


def _build_part(task: tuple[Path, list[Row], dict, dict]) -> list[tuple[int, dict]]:
    """Write a store for ``rows`` at ``out``; returns ``(row, {id, reason})`` for skipped rows."""
    out, rows, species, metadata = task
    failures = []
    with StoreWriter(out, metadata) as writer:
        for row in rows:
            try:
                record = make_record(row, species)
            except (OSError, ValueError) as error:
                failures.append((row.row, {"id": row.id, "reason": f"{type(error).__name__}: {error}"}))
                continue
            writer.add(record)
    return failures


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "rdkit": rdkit.__version__, "scipy": scipy.__version__,
            "zstandard": zstandard.__version__}


def build(rows: list[Row], out: Path, species: dict[str, int], workers: int = 1) -> tuple[list[dict], int]:
    """Write the store for ``rows`` at ``out``. Returns the failures ``{id, reason}`` in row order
    and the number of workers used (at most one per row).
    """
    out = Path(out)
    if any((out / name).exists() for name in STORE_FILES):
        raise FileExistsError(f"{out} already holds a store")
    metadata = {"pocket_radius": POCKET_RADIUS, "species": species, "versions": _versions()}

    kept, failures, first = [], [], {}
    for row in rows:
        if row.id in first:
            failures.append((row.row, {"id": row.id, "reason": f"duplicate id: row {row.row} "
                                       f"repeats row {first[row.id]}, first kept"}))
        else:
            first[row.id] = row.row
            kept.append(row)

    n = max(1, min(workers, len(kept)))
    if n == 1:
        failures += _build_part((out, kept, species, metadata))
    else:
        bounds = np.linspace(0, len(kept), n + 1).round().astype(int)
        parts = [out / "parts" / f"part-{i:03d}" for i in range(n)]
        tasks = [(part, kept[a:b], species, metadata)
                 for part, a, b in zip(parts, bounds[:-1], bounds[1:])]
        with multiprocessing.get_context("spawn").Pool(n) as pool:
            for part_failures in pool.map(_build_part, tasks, chunksize=1):
                failures += part_failures
        merge_stores(parts, out)
        shutil.rmtree(out / "parts")
    failures.sort(key=lambda failure: failure[0])
    return [failure for _, failure in failures], n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", choices=list(SOURCES), help="a training-side processed CSV")
    mode.add_argument("--generic", action="store_true",
                      help="a CSV with unique_id, sdf_file, pdb_file (split 'predict')")
    parser.add_argument("--processed-csv", type=Path, help="data/<source>_processed.csv")
    parser.add_argument("--root", type=Path, help="directory the data/... structure paths are under")
    parser.add_argument("--csv", type=Path, help="generic CSV (absolute structure paths)")
    parser.add_argument("--label-column", help="generic label column (default: 'label' if present)")
    parser.add_argument("--keys", type=Path, required=True, help="PDB_Atom_Keys.csv")
    parser.add_argument("--out", type=Path, required=True, help="new store directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N rows only")
    args = parser.parse_args(argv)
    if args.source and (args.processed_csv is None or args.root is None):
        parser.error("--source needs --processed-csv and --root")
    if args.generic and args.csv is None:
        parser.error("--generic needs --csv")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    start = time.perf_counter()
    if args.source:
        rows = source_rows(args.source, args.processed_csv, args.root, args.limit)
    else:
        try:
            rows = generic_rows(args.csv, args.label_column, args.limit)
        except ValueError as error:
            parser.error(str(error))
    failures, workers = build(rows, args.out, species_table(args.keys), args.workers)
    with Store(args.out) as store:
        records = len(store)
        splits = Counter(store.records[rid]["split"] for rid in store.ids)
    summary = {
        "source": args.source or Path(args.csv).stem,
        "csv": str(args.processed_csv or args.csv),
        "root": None if args.root is None else str(args.root),
        "rows": len(rows),
        "records": records,
        "splits": dict(splits),
        "failures": failures,
        "workers": workers,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (args.out / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[build_store] {summary['source']}: {records} records from {len(rows)} rows, "
          f"{len(failures)} failures, splits {dict(splits)}, {workers} workers, "
          f"{summary['seconds']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
