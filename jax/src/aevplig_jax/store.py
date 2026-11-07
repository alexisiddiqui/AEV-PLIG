"""Record store: one zstd frame per complex, read back by offset.

A store is a directory of four files:

- ``records.bin``: one zstd frame (level 3) per record, holding the twelve ``FIELDS`` arrays'
  C-order bytes concatenated in ``FIELDS`` order. Leading dimensions are ``L`` (ligand heavy atoms)
  for ``lig_*``, ``E`` (directed bonds) for ``bond_*`` and ``P`` (pocket atoms) for ``prot_*``.
- ``index.npz``: ``ids`` (N,) str, ``offset`` and ``length`` (N,) int64 into ``records.bin``,
  ``dims`` (N, 5) int32 = ``[L, E, P, A, KA]``; ``A`` = typed pocket atoms within ``RC`` of any
  ligand atom (``typed_within``), ``KA`` = the most of those within ``RC`` of one ligand atom.
- ``records.json``: id -> ``{source, split, label, meta}`` in store order (NaN labels as null).
- ``metadata.json``: ``schema``, ``fields``, ``cutoff`` plus the builder's keys (pocket radius,
  species table, package versions).

Imports numpy, zstandard, json and the standard library, plus package modules that use numpy and
scipy only (``LigandArrays`` imports without RDKit).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zstandard

from aevplig_jax.aev import neighbour_lists
from aevplig_jax.constants import EDGE_DIM, RC, RDKIT_DIM
from aevplig_jax.pdb_atoms import PocketArrays, typed_within
from aevplig_jax.rdkit_features import LigandArrays

SCHEMA = 1
ZSTD_LEVEL = 3
FIELDS = (
    ("lig_xyz", "float32", (3,)),
    ("lig_feat", "float32", (RDKIT_DIM,)),
    ("bond_src", "int32", ()),
    ("bond_dst", "int32", ()),
    ("bond_type", "float32", (EDGE_DIM,)),
    ("prot_xyz", "float32", (3,)),
    ("prot_species", "int8", ()),
    ("prot_res", "int32", ()),
    ("prot_resname", "int8", ()),
    ("prot_atom", "int16", ()),
    ("prot_chain", "int8", ()),
    ("prot_bfactor", "float16", ()),
)
_LEADING = {"lig": "L", "bond": "E", "prot": "P"}  # field-name prefix -> leading dimension
STORE_FILES = ("records.bin", "index.npz", "records.json", "metadata.json")


@dataclass
class Record:
    """One complex: ligand graph, pocket atoms, label (NaN when absent) and provenance."""

    id: str
    lig: LigandArrays
    pocket: PocketArrays
    label: float
    source: str
    split: str
    meta: dict = field(default_factory=dict)


def _field_arrays(record: Record) -> dict[str, np.ndarray]:
    lig, pocket = record.lig, record.pocket
    return {
        "lig_xyz": lig.xyz, "lig_feat": lig.feat, "bond_src": lig.bond_src,
        "bond_dst": lig.bond_dst, "bond_type": lig.bond_type,
        "prot_xyz": pocket.xyz, "prot_species": pocket.species, "prot_res": pocket.res,
        "prot_resname": pocket.resname, "prot_atom": pocket.atom, "prot_chain": pocket.chain,
        "prot_bfactor": pocket.bfactor,
    }


def _shape(name: str, trailing: tuple, sizes: dict[str, int]) -> tuple:
    return (sizes[_LEADING[name.split("_", 1)[0]]], *trailing)


def record_dims(record: Record) -> np.ndarray:
    """``[L, E, P, A, KA]`` of a record, int32."""
    lig_xyz = record.lig.xyz
    typed = typed_within(record.pocket, lig_xyz, cutoff=RC)
    n_typed = int(typed.sum())
    most = 0
    if n_typed:
        _, mask = neighbour_lists(lig_xyz, record.pocket.xyz[typed], cutoff=RC)
        most = int(mask.sum(axis=1).max(initial=0))
    return np.array([len(lig_xyz), len(record.lig.bond_src), len(record.pocket.xyz), n_typed, most],
                    dtype=np.int32)


def _encode(record: Record, dims: np.ndarray) -> bytes:
    """The frame payload: the arrays' bytes in FIELDS order, after checking dtypes and shapes."""
    sizes = dict(zip("LEP", (int(d) for d in dims[:3])))
    arrays = _field_arrays(record)
    chunks = []
    for name, dtype, trailing in FIELDS:
        array, shape = arrays[name], _shape(name, trailing, sizes)
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"{record.id}: {name} is {array.dtype} {array.shape}, "
                             f"expected {dtype} {shape}")
        chunks.append(np.ascontiguousarray(array).tobytes())
    return b"".join(chunks)


def _decode(payload: bytes, dims: np.ndarray, rid: str) -> tuple[LigandArrays, PocketArrays]:
    sizes = dict(zip("LEP", (int(d) for d in dims[:3])))
    arrays, position = {}, 0
    for name, dtype, trailing in FIELDS:
        shape = _shape(name, trailing, sizes)
        count = math.prod(shape)
        # Copied out of the buffer: after the int8 fields the offsets are not aligned.
        arrays[name] = np.frombuffer(payload, dtype, count, position).reshape(shape).copy()
        position += count * np.dtype(dtype).itemsize
    if position != len(payload):
        raise ValueError(f"{rid}: frame holds {len(payload)} bytes, dims {list(dims)} need {position}")
    lig = LigandArrays(xyz=arrays["lig_xyz"], feat=arrays["lig_feat"], bond_src=arrays["bond_src"],
                       bond_dst=arrays["bond_dst"], bond_type=arrays["bond_type"])
    pocket = PocketArrays(xyz=arrays["prot_xyz"], species=arrays["prot_species"],
                          res=arrays["prot_res"], resname=arrays["prot_resname"],
                          atom=arrays["prot_atom"], chain=arrays["prot_chain"],
                          bfactor=arrays["prot_bfactor"])
    return lig, pocket


def _fields_metadata() -> list[dict]:
    return [{"name": name, "dtype": np.dtype(dtype).str,
             "shape": [_LEADING[name.split("_", 1)[0]], *trailing]}
            for name, dtype, trailing in FIELDS]


def _refuse_existing(path: Path) -> None:
    present = [name for name in STORE_FILES if (path / name).exists()]
    if present:
        raise FileExistsError(f"{path} already holds a store ({', '.join(present)})")


def _write_json(path: Path, value) -> None:
    # allow_nan=False: NaN must reach the files as null (labels) or not at all.
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def _write_tables(path: Path, ids: list[str], offset, length, dims, records: dict,
                  metadata: dict) -> None:
    np.savez(path / "index.npz", ids=np.array(ids, dtype=str),
             offset=np.asarray(offset, dtype=np.int64), length=np.asarray(length, dtype=np.int64),
             dims=np.asarray(dims, dtype=np.int32).reshape(-1, 5))
    _write_json(path / "records.json", records)
    _write_json(path / "metadata.json", metadata)


class StoreWriter:
    """Appends records to a new store at ``path``; ``close`` writes the index and JSON files.

    ``metadata`` holds the builder's keys (``pocket_radius``, ``species``, ``versions``); the writer
    adds ``schema``, ``fields`` and ``cutoff``. Refuses a directory that already holds a store.
    """

    def __init__(self, path: Path, metadata: dict):
        self.path = Path(path)
        reserved = {"schema", "fields", "cutoff"} & set(metadata)
        if reserved:
            raise ValueError(f"metadata keys {sorted(reserved)} are set by the writer")
        self.path.mkdir(parents=True, exist_ok=True)
        _refuse_existing(self.path)
        self._metadata = {"schema": SCHEMA, "fields": _fields_metadata(), "cutoff": RC, **metadata}
        self._file = open(self.path / "records.bin", "wb")
        self._compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._ids: list[str] = []
        self._offset: list[int] = []
        self._length: list[int] = []
        self._dims: list[np.ndarray] = []
        self._records: dict[str, dict] = {}
        self._position = 0

    def add(self, record: Record) -> None:
        if record.id in self._records:
            raise ValueError(f"duplicate id {record.id!r}")
        dims = record_dims(record)
        frame = self._compressor.compress(_encode(record, dims))
        self._file.write(frame)
        self._ids.append(record.id)
        self._offset.append(self._position)
        self._length.append(len(frame))
        self._dims.append(dims)
        self._position += len(frame)
        label = None if math.isnan(record.label) else float(record.label)
        self._records[record.id] = {"source": record.source, "split": record.split, "label": label,
                                    "meta": record.meta}

    def close(self) -> None:
        self._file.close()
        _write_tables(self.path, self._ids, self._offset, self._length, self._dims, self._records,
                      self._metadata)

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:  # leave no index behind for a partial store
            self._file.close()


class Store:
    """Read-only store. ``get`` uses ``os.pread`` on one descriptor and a decompressor per thread,
    so threads may call it concurrently.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        with np.load(self.path / "index.npz") as index:
            self.ids: list[str] = [str(i) for i in index["ids"]]
            self.offset: np.ndarray = index["offset"]
            self.length: np.ndarray = index["length"]
            self.dims: np.ndarray = index["dims"]
        self.records: dict = json.loads((self.path / "records.json").read_text())
        self.metadata: dict = json.loads((self.path / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError(f"{self.path}: schema {self.metadata.get('schema')}, expected {SCHEMA}")
        if list(self.records) != self.ids:
            raise ValueError(f"{self.path}: records.json and index.npz list different ids")
        self._row = {rid: i for i, rid in enumerate(self.ids)}
        self._local = threading.local()
        self._fd = os.open(self.path / "records.bin", os.O_RDONLY)

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, id: str) -> Record:  # noqa: A002 - the brief's signature
        i = self._row[id]
        length, offset = int(self.length[i]), int(self.offset[i])
        frame = os.pread(self._fd, length, offset)
        if len(frame) != length:
            raise OSError(f"{self.path}: short read for {id!r} ({len(frame)} of {length} bytes)")
        decompressor = getattr(self._local, "decompressor", None)
        if decompressor is None:  # zstandard decompressors are not safe to share across threads
            decompressor = self._local.decompressor = zstandard.ZstdDecompressor()
        lig, pocket = _decode(decompressor.decompress(frame), self.dims[i], id)
        entry = self.records[id]
        label = math.nan if entry["label"] is None else float(entry["label"])
        return Record(id=id, lig=lig, pocket=pocket, label=label, source=entry["source"],
                      split=entry["split"], meta=dict(entry["meta"]))

    def labels(self) -> np.ndarray:
        """Labels in store order, float64, NaN where absent."""
        return np.array([np.nan if self.records[i]["label"] is None else self.records[i]["label"]
                         for i in self.ids], dtype=np.float64)

    def split_ids(self, split: str) -> list[str]:
        """Ids of ``split`` in store order."""
        return [i for i in self.ids if self.records[i]["split"] == split]

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> Store:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def merge_stores(parts: list[Path], out: Path) -> None:
    """Concatenate stores in the given order into a new store at ``out``.

    Frames are copied byte for byte and offsets shifted; ids must be unique across the parts and
    every part's metadata identical.
    """
    if not parts:
        raise ValueError("no stores to merge")
    stores = [Store(part) for part in parts]
    try:
        metadata = stores[0].metadata
        for store in stores[1:]:
            if store.metadata != metadata:
                raise ValueError(f"metadata of {store.path} differs from {stores[0].path}")
        seen: dict[str, Path] = {}
        for store in stores:
            for rid in store.ids:
                if rid in seen:
                    raise ValueError(f"duplicate id {rid!r} in {seen[rid]} and {store.path}")
                seen[rid] = store.path
            size = (store.path / "records.bin").stat().st_size
            if size != int(store.length.sum()):
                raise ValueError(f"{store.path}: records.bin has {size} bytes, index covers "
                                 f"{int(store.length.sum())}")

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        _refuse_existing(out)
        offsets, base = [], 0
        with open(out / "records.bin", "wb") as target:
            for store in stores:
                with open(store.path / "records.bin", "rb") as source:
                    shutil.copyfileobj(source, target, 16 << 20)
                offsets.append(store.offset + base)
                base += int(store.length.sum())
        ids = [rid for store in stores for rid in store.ids]
        records = {rid: entry for store in stores for rid, entry in store.records.items()}
        _write_tables(out, ids, np.concatenate(offsets),
                      np.concatenate([store.length for store in stores]),
                      np.concatenate([store.dims for store in stores]), records, metadata)
    finally:
        for store in stores:
            store.close()


def join_side(store: Store, ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align a side table (``values[i]`` belongs to ``ids[i]``) to store order.

    Returns the aligned array (zeros for store ids absent from the table, ``values``' dtype and
    trailing shape) and a boolean ``missing`` mask over the store. Table ids not in the store are
    ignored; repeated table ids raise ``ValueError``.
    """
    ids = np.asarray(ids).astype(str)
    values = np.asarray(values)
    if len(ids) != len(values):
        raise ValueError(f"side table has {len(ids)} ids but values of length {len(values)}")
    position: dict[str, int] = {}
    for j, rid in enumerate(ids):
        if rid in position:
            raise ValueError(f"duplicate id {rid!r} in the side table")
        position[rid] = j
    rows = np.array([position.get(rid, -1) for rid in store.ids], dtype=np.int64)
    missing = rows < 0
    aligned = np.zeros((len(store.ids), *values.shape[1:]), dtype=values.dtype)
    aligned[~missing] = values[rows[~missing]]
    return aligned, missing
