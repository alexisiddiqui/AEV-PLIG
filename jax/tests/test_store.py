"""Record store: zstd frames, index, records/metadata JSON, thread-safe reads, merge, side tables.

Records here are synthetic; ``test_build_store.py`` covers records built from structure files.
"""

import json
import math
import os
import subprocess
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest
import zstandard

from aevplig_jax import store_census
from aevplig_jax.pdb_atoms import ELEMENTS, PocketArrays
from aevplig_jax.rdkit_features import LigandArrays
from aevplig_jax.store import FIELDS, Record, Store, StoreWriter, join_side, merge_stores

SRC = Path(__file__).resolve().parents[1] / "src"
FIELD_NAMES = ["lig_xyz", "lig_feat", "bond_src", "bond_dst", "bond_type", "prot_xyz",
               "prot_species", "prot_res", "prot_resname", "prot_atom", "prot_chain", "prot_bfactor",
               "prot_element", "prot_chain_id", "prot_resseq", "prot_icode"]
METADATA = {"pocket_radius": 10.0, "species": {"ALA-CA": 2}, "versions": {"numpy": np.__version__}}

# Ligand of record "a" and its pocket. Distances to the ligand atoms (0,0,0), (1.5,0,0), (0.75,1.3,0):
# p0 (0,0,3) 3.00/3.35/3.35 and p1 (0,0,4.5) 4.50/4.74/4.74 are within 5.1 A of all three atoms,
# p2 (6,0,0) only of atom 1 (4.50; 6.00 and 5.41 for the others), p3 (0,0,2) is close but untyped,
# p4 (0,0,-5.2) is 5.20 A from atom 0 (just outside) and p5 (-9,0,0) is far.
# So A = 3 (p0, p1, p2) and KA = 3 (atom 1 sees p0, p1, p2; atoms 0 and 2 see p0, p1).
LIG_A = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]]
PROT_A = [[0.0, 0.0, 3.0], [0.0, 0.0, 4.5], [6.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -5.2],
          [-9.0, 0.0, 0.0]]
SPECIES_A = [5, 12, 1, 0, 7, 3]


def make_record(rng, rid, lig_xyz, prot_xyz, species, n_bonds, label=5.5, split="train",
                source="synthetic", meta=None) -> Record:
    """A record with the given coordinates and species and random values in every other field.

    The schema-2 pocket fields come from a generator seeded by ``rid``, so ``rng``'s stream (and
    every random record built from it) is what it was before they were added."""
    n_lig, n_prot = len(lig_xyz), len(prot_xyz)
    extra = np.random.default_rng(zlib.crc32(rid.encode()))
    bond_type = np.zeros((n_bonds, 4), dtype=np.float32)
    bond_type[np.arange(n_bonds), rng.integers(0, 4, n_bonds)] = 1.0
    lig = LigandArrays(
        xyz=np.asarray(lig_xyz, dtype=np.float32).reshape(n_lig, 3),
        feat=rng.random((n_lig, 15), dtype=np.float32),
        bond_src=rng.integers(0, max(n_lig, 1), n_bonds, dtype=np.int32),
        bond_dst=rng.integers(0, max(n_lig, 1), n_bonds, dtype=np.int32),
        bond_type=bond_type,
    )
    pocket = PocketArrays(
        xyz=np.asarray(prot_xyz, dtype=np.float32).reshape(n_prot, 3),
        species=np.asarray(species, dtype=np.int8),
        res=rng.integers(0, 50, n_prot, dtype=np.int32),
        resname=rng.integers(0, 21, n_prot, dtype=np.int8),
        atom=rng.integers(-1, 37, n_prot, dtype=np.int16),
        chain=rng.integers(0, 3, n_prot, dtype=np.int8),
        bfactor=rng.uniform(0, 80, n_prot).astype(np.float16),
        element=extra.integers(0, len(ELEMENTS), n_prot, dtype=np.int8),
        chain_id=extra.integers(ord("A"), ord("E"), n_prot, dtype=np.int8),
        resseq=extra.integers(-5, 3000, n_prot, dtype=np.int32),
        icode=extra.choice(np.array([0, ord("A"), ord("B")], dtype=np.int8), n_prot),
    )
    return Record(id=rid, lig=lig, pocket=pocket, label=label, source=source, split=split,
                  meta=meta if meta is not None else {})


def random_record(rng, rid, split="train") -> Record:
    n_lig, n_prot = int(rng.integers(1, 40)), int(rng.integers(0, 300))
    lig_xyz = rng.uniform(0, 12, (n_lig, 3))
    prot_xyz = rng.uniform(-4, 16, (n_prot, 3))
    species = rng.integers(0, 23, n_prot)
    return make_record(rng, rid, lig_xyz, prot_xyz, species, int(rng.integers(0, 2 * n_lig)),
                       label=float(rng.uniform(2, 12)), split=split, meta={"n": n_lig})


def field_arrays(record: Record) -> dict:
    lig, pocket = record.lig, record.pocket
    return {"lig_xyz": lig.xyz, "lig_feat": lig.feat, "bond_src": lig.bond_src,
            "bond_dst": lig.bond_dst, "bond_type": lig.bond_type, "prot_xyz": pocket.xyz,
            "prot_species": pocket.species, "prot_res": pocket.res, "prot_resname": pocket.resname,
            "prot_atom": pocket.atom, "prot_chain": pocket.chain, "prot_bfactor": pocket.bfactor,
            "prot_element": pocket.element, "prot_chain_id": pocket.chain_id,
            "prot_resseq": pocket.resseq, "prot_icode": pocket.icode}


def assert_same_record(got: Record, want: Record):
    assert (got.id, got.source, got.split, got.meta) == (want.id, want.source, want.split, want.meta)
    assert got.label == want.label or (math.isnan(got.label) and math.isnan(want.label))
    for name, expected in field_arrays(want).items():
        array = field_arrays(got)[name]
        assert array.dtype == expected.dtype and array.shape == expected.shape, name
        np.testing.assert_array_equal(array, expected, err_msg=name)


def brute_force_dims(record: Record) -> list[int]:
    """[L, E, P, A, KA] by an all-pairs float64 distance matrix."""
    lig = record.lig.xyz.astype(np.float64)
    prot = record.pocket.xyz.astype(np.float64)
    close = np.linalg.norm(lig[:, None] - prot[None], axis=-1) <= 5.1  # (L, P)
    close &= record.pocket.species[None, :] > 0
    ka = int(close.sum(axis=1).max(initial=0))
    return [len(lig), len(record.lig.bond_src), len(prot), int(close.any(axis=0).sum()), ka]


def write_store(path, records, metadata=METADATA) -> Path:
    with StoreWriter(path, metadata) as writer:
        for record in records:
            writer.add(record)
    return path


def test_fields_order_and_dtypes():
    assert [name for name, _, _ in FIELDS] == FIELD_NAMES
    dtypes = {name: np.dtype(dtype) for name, dtype, _ in FIELDS}
    assert dtypes == {"lig_xyz": np.float32, "lig_feat": np.float32, "bond_src": np.int32,
                      "bond_dst": np.int32, "bond_type": np.float32, "prot_xyz": np.float32,
                      "prot_species": np.int8, "prot_res": np.int32, "prot_resname": np.int8,
                      "prot_atom": np.int16, "prot_chain": np.int8, "prot_bfactor": np.float16,
                      "prot_element": np.int8, "prot_chain_id": np.int8, "prot_resseq": np.int32,
                      "prot_icode": np.int8}
    trailing = {name: tuple(shape) for name, _, shape in FIELDS}
    assert trailing["lig_xyz"] == (3,) and trailing["lig_feat"] == (15,)
    assert trailing["bond_type"] == (4,) and trailing["prot_xyz"] == (3,)
    assert all(trailing[name] == () for name in FIELD_NAMES if name not in
               ("lig_xyz", "lig_feat", "bond_type", "prot_xyz"))


def test_store_round_trip(tmp_path, rng):
    a = make_record(rng, "a", LIG_A, PROT_A, SPECIES_A, n_bonds=4, label=5.5, split="train",
                    source="pdbbind", meta={"pdb": "1abc", "refined": True})
    # E = 0 and A = 0: two typed atoms, both farther than 5.1 A from the ligand.
    b = make_record(rng, "b", [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], [[0.0, 0.0, 7.0], [9.0, 0.0, 0.0]],
                    [4, 9], n_bonds=0, label=math.nan, split="excluded", source="bindingdb")
    path = write_store(tmp_path / "store", [a, b])

    assert sorted(p.name for p in path.iterdir()) == [
        "index.npz", "metadata.json", "records.bin", "records.json"]
    with np.load(path / "index.npz") as index:
        assert sorted(index.files) == ["dims", "ids", "length", "offset"]
        assert index["ids"].dtype.kind == "U" and list(index["ids"]) == ["a", "b"]
        assert index["offset"].dtype == np.int64 and index["length"].dtype == np.int64
        assert index["dims"].dtype == np.int32 and index["dims"].shape == (2, 5)
        offset, length = index["offset"], index["length"]
    assert offset[0] == 0 and offset[1] == length[0]
    assert (path / "records.bin").stat().st_size == length.sum()

    # Each frame is one zstd frame of the sixteen arrays' bytes in FIELDS order.
    blob = (path / "records.bin").read_bytes()
    for i, record in enumerate([a, b]):
        frame = blob[offset[i]:offset[i] + length[i]]
        expected = b"".join(np.ascontiguousarray(field_arrays(record)[name]).tobytes()
                            for name, _, _ in FIELDS)
        assert zstandard.ZstdDecompressor().decompress(frame) == expected

    records = json.loads((path / "records.json").read_text())
    assert list(records) == ["a", "b"]
    assert records["a"] == {"source": "pdbbind", "split": "train", "label": 5.5,
                            "meta": {"pdb": "1abc", "refined": True}}
    assert records["b"]["label"] is None  # NaN is written as null

    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["schema"] == 2 and metadata["cutoff"] == 5.1
    assert metadata["pocket_radius"] == 10.0 and metadata["species"] == {"ALA-CA": 2}
    assert metadata["elements"] == list(ELEMENTS) and metadata["elements"][:7] == [
        "other", "C", "N", "O", "S", "P", "H"]
    assert [f["name"] for f in metadata["fields"]] == FIELD_NAMES
    assert metadata["fields"][0] == {"name": "lig_xyz", "dtype": "<f4", "shape": ["L", 3]}
    assert metadata["fields"][2] == {"name": "bond_src", "dtype": "<i4", "shape": ["E"]}
    assert metadata["fields"][11] == {"name": "prot_bfactor", "dtype": "<f2", "shape": ["P"]}
    assert metadata["fields"][-2] == {"name": "prot_resseq", "dtype": "<i4", "shape": ["P"]}

    with Store(path) as store:
        assert store.ids == ["a", "b"] and len(store) == 2
        np.testing.assert_array_equal(store.dims, [[3, 4, 6, 3, 3], [2, 0, 2, 0, 0]])
        np.testing.assert_array_equal(store.dims, [brute_force_dims(a), brute_force_dims(b)])
        assert_same_record(store.get("a"), a)
        assert_same_record(store.get("b"), b)
        np.testing.assert_array_equal(store.labels(), [5.5, np.nan])
        assert store.split_ids("train") == ["a"] and store.split_ids("excluded") == ["b"]
        assert store.split_ids("valid") == []
        assert store.metadata == metadata and store.records == records
        with pytest.raises(KeyError):
            store.get("missing")


def test_store_refuses_schema_1(tmp_path, rng):
    path = write_store(tmp_path / "store", [random_record(rng, "a")])
    metadata = json.loads((path / "metadata.json").read_text())
    (path / "metadata.json").write_text(json.dumps({**metadata, "schema": 1}))
    with pytest.raises(ValueError, match="schema 1.*prot_element.*rebuild"):
        Store(path)


def test_store_dims_match_brute_force_on_random_records(tmp_path, rng):
    records = [random_record(rng, f"r{i}") for i in range(20)]
    with Store(write_store(tmp_path / "store", records)) as store:
        np.testing.assert_array_equal(store.dims, [brute_force_dims(r) for r in records])
        assert (store.dims[:, 3] > 0).any() and (store.dims[:, 4] > 1).any()


def test_writer_rejects_bad_records(tmp_path, rng):
    record = make_record(rng, "a", LIG_A, PROT_A, SPECIES_A, n_bonds=2)
    with StoreWriter(tmp_path / "store", METADATA) as writer:
        writer.add(record)
        with pytest.raises(ValueError, match="duplicate"):
            writer.add(record)
        wrong = make_record(rng, "b", LIG_A, PROT_A, SPECIES_A, n_bonds=2)
        wrong.pocket.bfactor = wrong.pocket.bfactor.astype(np.float32)
        with pytest.raises(ValueError, match="prot_bfactor"):
            writer.add(wrong)
    with pytest.raises(FileExistsError):
        StoreWriter(tmp_path / "store", METADATA)


def test_store_get_is_thread_safe(tmp_path, rng):
    records = {f"r{i}": random_record(rng, f"r{i}") for i in range(12)}
    store = Store(write_store(tmp_path / "store", records.values()))
    picks = rng.integers(0, len(records), size=(8, 50))
    barrier = threading.Barrier(8)
    errors = []

    def worker(row):
        barrier.wait()
        try:
            for i in row:
                rid = f"r{i}"
                assert_same_record(store.get(rid), records[rid])
        except Exception as error:  # noqa: BLE001 - collected and re-raised below
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(row,)) for row in picks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.close()
    assert errors == []


def test_merge_stores_preserves_records(tmp_path, rng):
    first = [random_record(rng, "a"), random_record(rng, "b", split="valid")]
    third = [make_record(rng, "c", LIG_A, [], [], n_bonds=2, split="test")]  # P = 0
    parts = [write_store(tmp_path / "p0", first), write_store(tmp_path / "p1", []),
             write_store(tmp_path / "p2", third)]
    merge_stores(parts, tmp_path / "merged")

    with Store(tmp_path / "merged") as merged, Store(parts[0]) as p0, Store(parts[2]) as p2:
        assert merged.ids == ["a", "b", "c"]
        for record in first + third:
            assert_same_record(merged.get(record.id), record)
        np.testing.assert_array_equal(merged.dims, np.concatenate([p0.dims, p2.dims]))
        np.testing.assert_array_equal(merged.length, np.concatenate([p0.length, p2.length]))
        np.testing.assert_array_equal(merged.offset, [0, p0.length[0], p0.length.sum()])
        assert merged.metadata == p0.metadata
        assert merged.records == {**p0.records, **p2.records}
        assert merged.split_ids("valid") == ["b"] and merged.split_ids("test") == ["c"]
        np.testing.assert_array_equal(merged.dims[2], [3, 2, 0, 0, 0])
    assert (tmp_path / "merged" / "records.bin").read_bytes() == (
        (parts[0] / "records.bin").read_bytes() + (parts[2] / "records.bin").read_bytes())


def test_merge_stores_refuses_conflicts(tmp_path, rng):
    one = write_store(tmp_path / "one", [random_record(rng, "a")])
    again = write_store(tmp_path / "again", [random_record(rng, "a")])
    with pytest.raises(ValueError, match="duplicate id"):
        merge_stores([one, again], tmp_path / "dup")
    other = write_store(tmp_path / "other", [random_record(rng, "b")],
                        metadata={**METADATA, "pocket_radius": 8.0})
    with pytest.raises(ValueError, match="metadata"):
        merge_stores([one, other], tmp_path / "meta")
    assert not (tmp_path / "dup" / "records.bin").exists()
    assert not (tmp_path / "meta" / "records.bin").exists()


def test_join_side_aligns_and_flags_missing(tmp_path, rng):
    records = [random_record(rng, rid) for rid in ("a", "b", "c", "d")]
    with Store(write_store(tmp_path / "store", records)) as store:
        side_ids = np.array(["c", "a", "zz"])
        values = np.array([[3.0, 3.5], [1.0, 1.5], [9.0, 9.5]], dtype=np.float32)
        aligned, missing = join_side(store, side_ids, values)
        assert aligned.dtype == np.float32 and aligned.shape == (4, 2)
        np.testing.assert_array_equal(aligned, [[1.0, 1.5], [0, 0], [3.0, 3.5], [0, 0]])
        assert missing.dtype == bool
        np.testing.assert_array_equal(missing, [False, True, False, True])

        aligned, missing = join_side(store, np.array([b"d", b"b"]), np.array([4, 2]))
        np.testing.assert_array_equal(aligned, [0, 2, 0, 4])
        np.testing.assert_array_equal(missing, [True, False, True, False])

        with pytest.raises(ValueError, match="duplicate"):
            join_side(store, np.array(["a", "a"]), np.zeros(2))
        with pytest.raises(ValueError, match="length"):
            join_side(store, np.array(["a", "b"]), np.zeros(3))


def test_store_module_does_not_import_rdkit_or_pandas():
    """The training venv has no RDKit: store.py must import without it."""
    code = "import sys, aevplig_jax.store; print(sorted({'rdkit', 'pandas'} & set(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"


def test_store_census_counts_species_zero_atoms(tmp_path, rng, capsys):
    h, c, se = ELEMENTS.index("H"), ELEMENTS.index("C"), ELEMENTS.index("SE")

    def pocket_record(rid, source, species, element, atom):
        record = make_record(rng, rid, LIG_A, rng.uniform(0, 8, (len(species), 3)), species,
                             n_bonds=2, source=source)
        record.pocket.element = np.asarray(element, dtype=np.int8)
        record.pocket.atom = np.asarray(atom, dtype=np.int16)
        return record

    records = [  # species 0: H with atom -1 (1HA), C with a known name (MSE-CA), SE (MSE-SE)
        pocket_record("a", "pdbbind", [0, 0, 3, 0, 5], [h, c, c, se, c], [-1, 1, 1, -1, 2]),
        pocket_record("b", "pdbbind", [0, 4], [h, c], [-1, 3]),
        pocket_record("c", "bindingnet", [0, 0, 0, 2], [h, h, c, c], [-1, 5, -1, 0]),
    ]
    path = write_store(tmp_path / "store", records)
    census = store_census.census(path)
    assert census["pdbbind"] == {"records": 2, "atoms": 7, "species0": 4,
                                 "by_element": {"H": {"atom_unknown": 2, "atom_known": 0},
                                                "C": {"atom_unknown": 0, "atom_known": 1},
                                                "SE": {"atom_unknown": 1, "atom_known": 0}}}
    assert census["bindingnet"]["species0"] == 3
    assert census["bindingnet"]["by_element"] == {"H": {"atom_unknown": 1, "atom_known": 1},
                                                  "C": {"atom_unknown": 1, "atom_known": 0}}

    out = tmp_path / "census.json"
    assert store_census.main(["--store", str(path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| pdbbind | H | 2 | 0 | 2 |" in printed and "| bindingnet | C | 1 | 0 | 1 |" in printed
    assert json.loads(out.read_text()) == census
