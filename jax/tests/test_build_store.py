"""Store builder: per-source rows, split rule and order of create_pytorch_data.py, generic CSVs,
failures, workers.

Structure files are the benzamide fixtures (mol2 for PDBbind and BindingDB, sdf for BindingNet and
generic CSVs) and ``pocket.pdb``, placed under the archive layout the generate scripts expect.
Benzamide against ``pocket.pdb``: L = 9, E = 18, P = 41 within 10 A, A = 24, KA = 16.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aevplig_jax.build_store import main
from aevplig_jax.constants import species_table
from aevplig_jax.pdb_atoms import read_pocket
from aevplig_jax.rdkit_features import ligand_arrays, read_ligand
from aevplig_jax.store import Store, merge_stores

DATA = Path(__file__).resolve().parent / "data"
SDF, MOL2, POCKET = DATA / "benzamide.sdf", DATA / "benzamide.mol2", DATA / "pocket.pdb"
KEYS_CSV = Path(__file__).resolve().parents[2] / "data" / "PDB_Atom_Keys.csv"
BENZAMIDE_DIMS = [9, 18, 41, 24, 16]

# Rows of the three processed CSVs (columns as in data/*_processed.csv, written with an index column).
PDBBIND = pd.DataFrame({
    "PDB_code": ["1abc", "2gen", "3tst", "4val", "5edg", "6exc"],
    "-logKd/Ki": [5.0, 6.0, 7.0, 4.0, 3.0, 8.0],
    "refined": [True, False, False, True, True, False],
    "core": [False, False, True, False, True, True],
    "split_core": ["train", "valid", "test", "valid", "test", "train"],
    "max_tanimoto_schrodinger": [0.1] * 6,
    "max_tanimoto_merck": [0.1] * 6,
    "max_tanimoto_fep_benchmark": [0.85, 0.95, 0.2, 0.3, 0.9, 0.99],
})
BINDINGNET = pd.DataFrame({
    "unique_identify": ["CHEMBL1_1abc_CHEMBL10", "CHEMBL2_7bnt_CHEMBL20", "CHEMBL3_7bnt_CHEMBL30"],
    "-logAffi": [6.5, 7.5, 5.25],
    "target": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
    "pdb": ["1abc", "7bnt", "7bnt"],
    "compnd": ["CHEMBL10", "CHEMBL20", "CHEMBL30"],
    "max_tanimoto_schrodinger": [0.1] * 3,
    "max_tanimoto_merck": [0.1] * 3,
    "max_tanimoto_fep_benchmark": [0.85, 0.95, 0.5],
})
BINDINGDB = pd.DataFrame({
    "folder": ["4DBU_BT9", "4DBU_BT9"],
    "pdb_file": ["4DBU.pdb", "4DBU.pdb"],
    "mol2_file": ["4DBU-results_1.mol2", "4DBU-results_2.mol2"],
    "unique_id": ["4DBU_BT9_4DBU-results_1", "4DBU_BT9_4DBU-results_2"],
    "pK": [5.82, 6.03],
    "max_tanimoto_fep_benchmark": [0.95, 0.1],
})
SOURCES = {"pdbbind": (PDBBIND, "PDB_code"), "bindingnet": (BINDINGNET, "unique_identify"),
           "bindingdb": (BINDINGDB, "unique_id")}


@pytest.fixture(scope="module")
def species():
    if not KEYS_CSV.exists():
        pytest.skip(f"{KEYS_CSV} not present (code-only mirror)")
    return species_table(KEYS_CSV)


def _place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _archive_root(root: Path) -> Path:
    """The extracted-archive layout of the three generate scripts, for the rows above."""
    for pdb, refined in zip(PDBBIND["PDB_code"], PDBBIND["refined"]):
        folder = root / "data" / "pdbbind" / ("refined-set" if refined else "general-set") / pdb
        _place(MOL2, folder / f"{pdb}_ligand.mol2")
        _place(POCKET, folder / f"{pdb}_protein.pdb")
    bindingnet = root / "data" / "bindingnet" / "from_chembl_client"
    for row in BINDINGNET.itertuples():
        _place(SDF, bindingnet / row.pdb / f"target_{row.target}" / row.compnd
               / f"{row.pdb}_{row.target}_{row.compnd}.sdf")
        _place(POCKET, bindingnet / row.pdb / "rec_h_opt.pdb")
    for row in BINDINGDB.itertuples():
        folder = root / "data" / "bindingdb" / "surflex" / row.folder
        _place(MOL2, folder / row.mol2_file)
        _place(POCKET, folder / row.pdb_file)
    return root


def _create_pytorch_data_frame(pdbbind_csv, bindingnet_csv, bindingdb_csv) -> pd.DataFrame:
    """create_pytorch_data.py:26-44, with the three CSV paths as arguments."""
    pdbbind = pd.read_csv(pdbbind_csv, index_col=0)
    pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
    pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
    pdbbind = pdbbind[['unique_id','pK','split']]

    bindingnet = pd.read_csv(bindingnet_csv, index_col=0)
    bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
    bindingnet['split'] = 'train'
    bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]
    bindingnet = bindingnet[['unique_id','pK','split']]

    bindingdb = pd.read_csv(bindingdb_csv, index_col=0)
    bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
    bindingdb['split'] = 'train'
    bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]
    bindingdb = bindingdb[['unique_id','pK','split']]

    return pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)


@pytest.fixture(scope="module")
def trainside(tmp_path_factory, species):
    """The three sources built by the CLI (one worker each) and merged in source order."""
    base = tmp_path_factory.mktemp("trainside")
    root = _archive_root(base / "extracted")
    csvs, stores = {}, {}
    for source, (frame, _) in SOURCES.items():
        csvs[source] = base / f"{source}_processed.csv"
        frame.to_csv(csvs[source])
        stores[source] = base / "store" / source
        assert main(["--source", source, "--processed-csv", str(csvs[source]), "--root", str(root),
                     "--keys", str(KEYS_CSV), "--out", str(stores[source])]) == 0
    merge_stores([stores[s] for s in SOURCES], base / "store" / "trainside")
    return csvs, stores, base / "store" / "trainside"


def test_split_rule_matches_create_pytorch_data(trainside):
    csvs, stores, _ = trainside
    expected = {
        # PDBbind: split_core unless max_tanimoto_fep_benchmark >= 0.9 (0.9 itself is excluded).
        "pdbbind": {"1abc": "train", "2gen": "excluded", "3tst": "test", "4val": "valid",
                    "5edg": "excluded", "6exc": "excluded"},
        "bindingnet": {"CHEMBL1_1abc_CHEMBL10": "train", "CHEMBL2_7bnt_CHEMBL20": "excluded",
                       "CHEMBL3_7bnt_CHEMBL30": "train"},
        "bindingdb": {"4DBU_BT9_4DBU-results_1": "excluded", "4DBU_BT9_4DBU-results_2": "train"},
    }
    labels = {"pdbbind": "-logKd/Ki", "bindingnet": "-logAffi", "bindingdb": "pK"}
    for source, (frame, id_column) in SOURCES.items():
        summary = json.loads((stores[source] / "build_summary.json").read_text())
        assert summary["failures"] == [] and summary["records"] == len(frame)
        with Store(stores[source]) as store:
            assert {i: store.records[i]["split"] for i in store.ids} == expected[source]
            assert {store.records[i]["source"] for i in store.ids} == {source}
            np.testing.assert_array_equal(store.labels(), frame[labels[source]])
            np.testing.assert_array_equal(store.dims, [BENZAMIDE_DIMS] * len(frame))

    reference = _create_pytorch_data_frame(csvs["pdbbind"], csvs["bindingnet"], csvs["bindingdb"])
    with Store(trainside[2]) as merged:
        kept = [i for i in merged.ids if merged.records[i]["split"] != "excluded"]
        assert kept == list(reference["unique_id"])
        for split in ("train", "valid", "test"):
            rows = reference[reference["split"] == split]
            assert merged.split_ids(split) == list(rows["unique_id"])
            position = {i: n for n, i in enumerate(merged.ids)}
            np.testing.assert_array_equal(
                merged.labels()[[position[i] for i in merged.split_ids(split)]], rows["pK"])


def test_record_paths_and_meta_follow_generate_scripts(trainside):
    _, stores, _ = trainside
    with Store(stores["pdbbind"]) as store:
        meta = store.records["2gen"]["meta"]
        assert meta["ligand"] == "data/pdbbind/general-set/2gen/2gen_ligand.mol2"
        assert meta["protein"] == "data/pdbbind/general-set/2gen/2gen_protein.pdb"
        assert meta["refined"] is False and meta["split_core"] == "valid"
        assert meta["max_tanimoto_fep_benchmark"] == 0.95
        assert store.records["1abc"]["meta"]["ligand"].startswith("data/pdbbind/refined-set/1abc/")
    with Store(stores["bindingnet"]) as store:
        meta = store.records["CHEMBL2_7bnt_CHEMBL20"]["meta"]
        assert meta["ligand"] == ("data/bindingnet/from_chembl_client/7bnt/target_CHEMBL2/CHEMBL20/"
                                  "7bnt_CHEMBL2_CHEMBL20.sdf")
        assert meta["protein"] == "data/bindingnet/from_chembl_client/7bnt/rec_h_opt.pdb"
    with Store(stores["bindingdb"]) as store:
        meta = store.records["4DBU_BT9_4DBU-results_2"]["meta"]
        assert meta["ligand"] == "data/bindingdb/surflex/4DBU_BT9/4DBU-results_2.mol2"
        assert meta["protein"] == "data/bindingdb/surflex/4DBU_BT9/4DBU.pdb"


def test_store_order_follows_source_order(trainside):
    _, stores, merged_path = trainside
    expected = [*PDBBIND["PDB_code"], *BINDINGNET["unique_identify"], *BINDINGDB["unique_id"]]
    with Store(merged_path) as merged:
        assert merged.ids == expected
        for source in SOURCES:
            with Store(stores[source]) as part:
                for rid in part.ids:
                    got, want = merged.get(rid), part.get(rid)
                    np.testing.assert_array_equal(got.lig.feat, want.lig.feat)
                    np.testing.assert_array_equal(got.pocket.xyz, want.pocket.xyz)


def _generic_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_generic_csv(tmp_path, species):
    csv = _generic_csv(tmp_path / "fep_set.csv", [
        {"unique_id": "series/lig 2, deprot", "sdf_file": str(SDF), "pdb_file": str(POCKET),
         "series": "s1"}])
    assert main(["--generic", "--csv", str(csv), "--keys", str(KEYS_CSV),
                 "--out", str(tmp_path / "store")]) == 0

    lig = ligand_arrays(read_ligand(SDF, "sdf"))
    pocket = read_pocket(POCKET, lig.xyz, species, radius=10.0)
    with Store(tmp_path / "store") as store:
        assert store.ids == ["series/lig 2, deprot"]
        np.testing.assert_array_equal(store.dims, [BENZAMIDE_DIMS])
        record = store.get("series/lig 2, deprot")
        assert math.isnan(record.label) and np.isnan(store.labels()).all()
        assert (record.source, record.split) == ("fep_set", "predict")
        assert record.meta == {"sdf_file": str(SDF), "pdb_file": str(POCKET), "series": "s1"}
        for name in ("xyz", "feat", "bond_src", "bond_dst", "bond_type"):
            np.testing.assert_array_equal(getattr(record.lig, name), getattr(lig, name), err_msg=name)
        for name in ("xyz", "species", "res", "resname", "atom", "chain", "bfactor"):
            np.testing.assert_array_equal(getattr(record.pocket, name), getattr(pocket, name),
                                          err_msg=name)
        assert store.metadata["pocket_radius"] == 10.0
        assert store.metadata["species"] == species
        assert {"rdkit", "numpy", "zstandard"} <= set(store.metadata["versions"])
    summary = json.loads((tmp_path / "store" / "build_summary.json").read_text())
    assert (summary["rows"], summary["records"], summary["failures"]) == (1, 1, [])
    assert summary["splits"] == {"predict": 1}


def test_build_generic_label_column(tmp_path, species):
    csv = _generic_csv(tmp_path / "pgdh.csv", [
        {"unique_id": "pgdh/S1/compound_1", "pK": 10.5, "sdf_file": str(SDF), "pdb_file": str(POCKET)},
        {"unique_id": "pgdh/S1/compound_2", "pK": 9.25, "sdf_file": str(SDF), "pdb_file": str(POCKET)},
    ])
    args = ["--generic", "--csv", str(csv), "--keys", str(KEYS_CSV)]
    assert main([*args, "--label-column", "pK", "--out", str(tmp_path / "pk")]) == 0
    with Store(tmp_path / "pk") as store:
        np.testing.assert_array_equal(store.labels(), [10.5, 9.25])
        assert "pK" not in store.get("pgdh/S1/compound_1").meta
    # Without --label-column a missing "label" column means NaN labels; a named column must exist.
    assert main([*args, "--out", str(tmp_path / "nolabel")]) == 0
    with Store(tmp_path / "nolabel") as store:
        assert np.isnan(store.labels()).all()
        assert store.get("pgdh/S1/compound_1").meta["pK"] == 10.5
    with pytest.raises(SystemExit):
        main([*args, "--label-column", "pIC50", "--out", str(tmp_path / "bad")])


def test_build_records_failures_and_continues(tmp_path, species):
    garbage = tmp_path / "garbage.sdf"
    garbage.write_text("not a molecule\n")
    xenon = tmp_path / "xenon.sdf"  # element outside the ten ligand symbols: ValueError
    xenon.write_text(SDF.read_text().replace(" N   0", " Xe  0", 1))
    rows = [
        {"unique_id": "good", "sdf_file": SDF, "pdb_file": POCKET},
        {"unique_id": "no_ligand", "sdf_file": tmp_path / "missing.sdf", "pdb_file": POCKET},
        {"unique_id": "unreadable", "sdf_file": garbage, "pdb_file": POCKET},
        {"unique_id": "no_protein", "sdf_file": SDF, "pdb_file": tmp_path / "missing.pdb"},
        {"unique_id": "bad_element", "sdf_file": xenon, "pdb_file": POCKET},
        {"unique_id": "good", "sdf_file": SDF, "pdb_file": POCKET},
        {"unique_id": "good2", "sdf_file": SDF, "pdb_file": POCKET},
    ]
    csv = _generic_csv(tmp_path / "set.csv", rows)
    assert main(["--generic", "--csv", str(csv), "--keys", str(KEYS_CSV),
                 "--out", str(tmp_path / "store")]) == 0
    with Store(tmp_path / "store") as store:
        assert store.ids == ["good", "good2"]
    summary = json.loads((tmp_path / "store" / "build_summary.json").read_text())
    assert (summary["rows"], summary["records"]) == (7, 2)
    reasons = {f["id"]: f["reason"] for f in summary["failures"]}
    assert [f["id"] for f in summary["failures"]] == [
        "no_ligand", "unreadable", "no_protein", "bad_element", "good"]
    assert "missing ligand file" in reasons["no_ligand"]
    assert "RDKit" in reasons["unreadable"]
    assert "missing protein file" in reasons["no_protein"]
    assert reasons["bad_element"].startswith("ValueError") and "Xe" in reasons["bad_element"]
    assert "duplicate id" in reasons["good"]


def test_workers_and_limit_keep_row_order(tmp_path, species):
    rows = [{"unique_id": f"lig{i}", "pK": float(i), "sdf_file": SDF, "pdb_file": POCKET}
            for i in range(7)]
    rows[3]["sdf_file"] = tmp_path / "missing.sdf"
    csv = _generic_csv(tmp_path / "set.csv", rows)
    args = ["--generic", "--csv", str(csv), "--keys", str(KEYS_CSV), "--label-column", "pK",
            "--limit", "6"]
    assert main([*args, "--workers", "1", "--out", str(tmp_path / "serial")]) == 0
    assert main([*args, "--workers", "3", "--out", str(tmp_path / "parallel")]) == 0

    assert sorted(p.name for p in (tmp_path / "parallel").iterdir()) == [
        "build_summary.json", "index.npz", "metadata.json", "records.bin", "records.json"]
    with Store(tmp_path / "serial") as serial, Store(tmp_path / "parallel") as parallel:
        assert serial.ids == parallel.ids == ["lig0", "lig1", "lig2", "lig4", "lig5"]
        np.testing.assert_array_equal(parallel.labels(), [0, 1, 2, 4, 5])
        np.testing.assert_array_equal(parallel.dims, serial.dims)
        assert parallel.records == serial.records
        for rid in serial.ids:
            np.testing.assert_array_equal(parallel.get(rid).pocket.xyz, serial.get(rid).pocket.xyz)
    summaries = [json.loads((tmp_path / d / "build_summary.json").read_text())
                 for d in ("serial", "parallel")]
    for summary in summaries:
        assert summary["rows"] == 6 and summary["records"] == 5
        assert [f["id"] for f in summary["failures"]] == ["lig3"]
    assert [s["workers"] for s in summaries] == [1, 3]
