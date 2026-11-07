"""Store verification against pickled graphs: atom count, RDKit columns, bonds and JAX AEVs.

The pickles are written here in the original format, ``graphs_dict[id] = (c_size, features,
edge_index, edge_features)`` with ``features`` a list of per-atom float64 arrays (15 RDKit columns
then 352 AEV columns). Their AEV columns come from ``radial_aev_numpy`` over the whole pocket, so
the oracle shares neither the typed subset, the neighbour lists nor the JAX code under test.
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aevplig_jax.aev import radial_aev_numpy
from aevplig_jax.constants import species_table
from aevplig_jax.pdb_atoms import read_pocket
from aevplig_jax.rdkit_features import ligand_arrays, read_ligand
from aevplig_jax.store import STORE_FILES, Record, Store
from aevplig_jax.verify_store import main, select_ids
from test_store import LIG_A, PROT_A, SPECIES_A, make_record, write_store

DATA = Path(__file__).resolve().parent / "data"
SDF, POCKET = DATA / "benzamide.sdf", DATA / "pocket.pdb"
KEYS_CSV = Path(__file__).resolve().parents[2] / "data" / "PDB_Atom_Keys.csv"
SRC = Path(__file__).resolve().parents[1] / "src"
BENZAMIDE = "CHEMBL1_1abc_CHEMBL10"
SYNTHETIC = "1abc"
FAR = "4DBU_BT9_4DBU-results_1"


@pytest.fixture(scope="module")
def species():
    if not KEYS_CSV.exists():
        pytest.skip(f"{KEYS_CSV} not present (code-only mirror)")
    return species_table(KEYS_CSV)


@pytest.fixture
def records(species, rng):
    """Benzamide against ``pocket.pdb`` (L 9, E 18, A 24, KA 16), the synthetic record of
    test_store.py (L 3, A 3) and a record with no typed atom within 5.1 A and no bonds (A = 0, E = 0).
    """
    pytest.importorskip("rdkit", reason="needs RDKit (absent from the jax-Ka training venv)")
    lig = ligand_arrays(read_ligand(SDF, "sdf"))
    benzamide = Record(id=BENZAMIDE, lig=lig, pocket=read_pocket(POCKET, lig.xyz, species),
                       label=6.5, source="bindingnet", split="train")
    synthetic = make_record(rng, SYNTHETIC, LIG_A, PROT_A, SPECIES_A, n_bonds=4, source="pdbbind")
    far = make_record(rng, FAR, [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], [[0.0, 0.0, 7.0], [9.0, 0.0, 0.0]],
                      [4, 9], n_bonds=0, source="bindingdb")
    return [synthetic, benzamide, far]


def graph_of(record: Record) -> tuple:
    """``(c_size, features, edge_index, edge_features)`` as ``mol_to_graph`` returns them."""
    aev = radial_aev_numpy(record.lig.xyz, record.pocket.xyz, record.pocket.species)
    features = [np.append(row.astype(np.float64), aev_row) for row, aev_row in zip(record.lig.feat, aev)]
    edge_index = [[int(s), int(d)] for s, d in zip(record.lig.bond_src, record.lig.bond_dst)]
    edge_features = record.lig.bond_type.astype(np.float64).tolist()
    return len(record.lig.xyz), features, edge_index, edge_features


def write_pickle(path: Path, graphs: dict) -> Path:
    with open(path, "wb") as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def run(store: Path, pickles: list[Path], *extra: str) -> tuple[int, dict]:
    out = store / "verification.json"
    argv = ["--store", str(store), "--out", str(out), *extra]
    for path in pickles:
        argv += ["--pickle", str(path)]
    code = main(argv)
    return code, json.loads(out.read_text())


def test_verify_passes_on_consistent_pickle(tmp_path, records):
    store = write_store(tmp_path / "store", records)
    with Store(store) as opened:
        np.testing.assert_array_equal(opened.dims, [[3, 4, 6, 3, 3], [9, 18, 41, 24, 16], [2, 0, 2, 0, 0]])
    graphs = {record.id: graph_of(record) for record in records}
    assert np.abs(np.asarray(graphs[BENZAMIDE][1])[:, 15:]).max() > 0.1  # a non-trivial oracle
    assert not np.asarray(graphs[FAR][1])[:, 15:].any()
    pickles = [write_pickle(tmp_path / "a.pickle", {rid: graphs[rid] for rid in (SYNTHETIC, BENZAMIDE)}),
               write_pickle(tmp_path / "b.pickle", {FAR: graphs[FAR]})]

    code, report = run(store, pickles)

    assert code == 0 and report["passed"] is True
    assert report["counts"] == {"store": 3, "pickle": 3, "compared": 3, "passed": 3, "failed": 0,
                                "missing_in_pickle": 0, "missing_in_store": 0}
    assert report["capacities"] == {"L": 32, "K": 32, "A": 32}
    assert report["worst"]["id"] in graphs and 0.0 <= report["worst"]["aev_max_abs"] < 1e-5
    assert report["tolerance"] == 1e-4
    assert report["failures_by_source"] == {} and report["failures_by_check"] == {}
    assert report["failures"] == []
    assert report["pickles"] == [str(path) for path in pickles] and report["store"] == str(store)
    assert report["seconds"] >= 0.0
    # Written by rename: no temporary file is left next to it.
    assert sorted(p.name for p in store.iterdir()) == sorted([*STORE_FILES, "verification.json"])


def test_verify_fails_on_perturbed_feature(tmp_path, records):
    store = write_store(tmp_path / "store", records)
    graphs = {record.id: graph_of(record) for record in records}
    graphs[BENZAMIDE][1][4][15 + 100] += 1e-3
    pickle_path = write_pickle(tmp_path / "graphs.pickle", graphs)

    code, report = run(store, [pickle_path])

    assert code == 1 and report["passed"] is False
    assert report["counts"]["compared"] == 3
    assert report["counts"]["passed"] == 2 and report["counts"]["failed"] == 1
    assert report["worst"]["id"] == BENZAMIDE
    assert report["worst"]["aev_max_abs"] == pytest.approx(1e-3, abs=1e-5)
    assert report["failures_by_source"] == {"bindingnet": 1}
    assert report["failures_by_check"] == {"aev": 1}
    assert len(report["failures"]) == 1
    failure = report["failures"][0]
    assert failure["id"] == BENZAMIDE and failure["source"] == "bindingnet"
    assert failure["checks"] == ["aev"]
    assert failure["aev_max_abs"] == pytest.approx(1e-3, abs=1e-5)


def _drop_last_atom(graph):
    c_size, features, edge_index, edge_features = graph
    return c_size - 1, features[:-1], edge_index, edge_features


def _bump_rdkit(graph):
    graph[1][0][3] += 1.0
    return graph


def _swap_edges(graph):
    edge_index = list(graph[2])
    edge_index[0], edge_index[1] = edge_index[1], edge_index[0]
    return graph[0], graph[1], edge_index, graph[3]


def _change_bond_type(graph):
    edge_features = [list(row) for row in graph[3]]
    edge_features[0] = [0.0, 0.0, 0.0, 1.0] if edge_features[0][3] == 0.0 else [1.0, 0.0, 0.0, 0.0]
    return graph[0], graph[1], graph[2], edge_features


@pytest.mark.parametrize("perturb, checks", [
    (lambda g: (g[0] + 1, *g[1:]), ["c_size"]),
    (_drop_last_atom, ["c_size", "features_shape"]),
    (_bump_rdkit, ["rdkit"]),
    (_swap_edges, ["edge_index"]),
    (_change_bond_type, ["edge_features"]),
], ids=["c_size", "atom_count", "rdkit", "edge_order", "edge_features"])
def test_verify_flags_graph_mismatches(tmp_path, records, perturb, checks):
    store = write_store(tmp_path / "store", records)
    graphs = {record.id: graph_of(record) for record in records}
    graphs[BENZAMIDE] = perturb(graphs[BENZAMIDE])

    code, report = run(store, [write_pickle(tmp_path / "graphs.pickle", graphs)])

    assert code == 1 and report["passed"] is False
    assert report["counts"]["failed"] == 1 and report["counts"]["passed"] == 2
    assert report["failures"][0]["id"] == BENZAMIDE and report["failures"][0]["checks"] == checks
    assert report["failures_by_check"] == {check: 1 for check in checks}
    assert report["failures_by_source"] == {"bindingnet": 1}


def test_verify_handles_missing_id(tmp_path, records):
    store = write_store(tmp_path / "store", records)
    graphs = {record.id: graph_of(record) for record in records}
    graphs["extra"] = graphs.pop(SYNTHETIC)
    pickle_path = write_pickle(tmp_path / "graphs.pickle", graphs)

    code, report = run(store, [pickle_path])

    assert code == 0 and report["passed"] is True
    assert report["counts"] == {"store": 3, "pickle": 3, "compared": 2, "passed": 2, "failed": 0,
                                "missing_in_pickle": 1, "missing_in_store": 1}
    assert report["missing_in_pickle_ids"] == [SYNTHETIC]
    assert report["missing_in_store_ids"] == ["extra"]

    # The training-side job requires every pickled id to be in the store.
    code, report = run(store, [pickle_path], "--require-all-pickle-ids")
    assert code == 1 and report["passed"] is False
    assert report["counts"]["failed"] == 0 and report["counts"]["missing_in_store"] == 1


def test_verify_sample(tmp_path, records):
    store = write_store(tmp_path / "store", records)
    pickle_path = write_pickle(tmp_path / "graphs.pickle", {r.id: graph_of(r) for r in records})

    code, report = run(store, [pickle_path], "--sample", "2", "--seed", "3")
    assert code == 0 and report["passed"] is True
    assert report["counts"]["compared"] == 2 and report["sample"] == 2 and report["seed"] == 3
    assert report["counts"]["missing_in_pickle"] == 0  # counted over all ids, not the sample

    with pytest.raises(SystemExit):
        main(["--store", str(store), "--out", str(tmp_path / "v.json"), "--pickle", str(pickle_path),
              "--sample", "0"])


def test_select_ids_is_seeded_and_keeps_store_order():
    ids = [f"r{i}" for i in range(50)]
    picked = select_ids(ids, 10, seed=7)
    assert picked == select_ids(ids, 10, seed=7)
    assert picked != select_ids(ids, 10, seed=8)
    assert len(picked) == 10 and len(set(picked)) == 10
    assert picked == [rid for rid in ids if rid in set(picked)]  # store order
    assert select_ids(ids, None) == ids and select_ids(ids, 50) == ids and select_ids(ids, 80) == ids


def test_verify_fails_when_nothing_is_compared(tmp_path, records):
    """A store checked against an unrelated pickle compares nothing: that is not a pass."""
    store = write_store(tmp_path / "store", records)
    pickle_path = write_pickle(tmp_path / "other.pickle", {"other": graph_of(records[1])})

    code, report = run(store, [pickle_path])

    assert code == 1 and report["passed"] is False
    assert report["counts"]["compared"] == 0
    assert report["counts"]["missing_in_pickle"] == 3 and report["counts"]["missing_in_store"] == 1
    assert report["worst"] == {"id": None, "aev_max_abs": None}


def test_verify_module_imports_no_torch_pandas_rdkit_or_jax():
    """Pickles are read with numpy only; jax is imported when verification runs, not at import."""
    code = ("import sys, aevplig_jax.verify_store; "
            "print(sorted({'torch', 'pandas', 'rdkit', 'jax'} & set(sys.modules)))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"
