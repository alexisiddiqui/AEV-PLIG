"""Tests for src/pgdh_prepare.py: a tiny synthetic copy of the 15-PGDH repository layout, plus one
test on the real downloaded repository (skipped when it is not on disk; PGDH_REPO_DIR overrides)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

import fep_prepare  # noqa: E402
import pgdh_prepare  # noqa: E402

REAL_REPO = os.environ.get(
    "PGDH_REPO_DIR",
    "/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/"
    "pgdh/ml-affinity-15-PGDH-case-study-main")

PROTEIN = (
    "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1      13.164   6.188  -5.146  1.00  0.00           C\n"
    "HETATM    4  PA  NAD A 301      14.000   7.000  -4.000  1.00  0.00           P\n"
    "END\n"
)

# compound id -> (series tag, experimental pKi, FEP+ pKi, SMILES)
COMPOUNDS = {"compound_1": ("1", 8.5, 6.0, "Oc1ccccc1"), "compound_7": ("4_5", 7.25, 9.9, "CCNC(C)=O")}


def _mol(cid, series, pki, fep, smiles, seed, with_h=True, shift=0.0, extra=None):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    if shift:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + shift, p.y, p.z))
    if not with_h:
        mol = Chem.RemoveHs(mol)
    mol.SetProp("_Name", cid)
    props = {"ID": cid, "series": series, "pKi": f"{pki:.4f}", "FEP_pKi": f"{fep:.4f}", "smiles": smiles}
    props.update(extra or {})
    for k, v in props.items():
        if v is not None:
            mol.SetProp(k, v)
    return mol


def _write_sdf(path, mols):
    writer = Chem.SDWriter(path)
    for m in mols:
        writer.write(m)
    writer.close()


def _make_repo(root, compounds=COMPOUNDS, vina_ids=("compound_1",), extra_poses=()):
    """poses.sdf with every compound (full H); vina_predictions.sdf with vina_ids only (no H, shifted
    by +1 A in x so the pose source is visible); reference.pdb; the two paper CSVs."""
    os.makedirs(os.path.join(root, "data"))
    os.makedirs(os.path.join(root, "scoring"))
    poses, vina = [], []
    for k, (cid, (series, pki, fep, smi)) in enumerate(compounds.items()):
        poses.append(_mol(cid, series, pki, fep, smi, seed=k + 1))
        if cid in vina_ids:
            vina.append(_mol(cid, series, pki, fep, smi, seed=k + 1, with_h=False, shift=1.0,
                             extra={"minimizedAffinity": "-9.1"}))
    poses += list(extra_poses)
    _write_sdf(os.path.join(root, "data", "poses.sdf"), poses)
    _write_sdf(os.path.join(root, "scoring", "vina_predictions.sdf"), vina)
    with open(os.path.join(root, "data", "reference.pdb"), "w") as fh:
        fh.write(PROTEIN)
    pd.DataFrame({"unique_id": ["compound_7", "compound_1", "compound_99"], "preds": [6.1, 7.2, 5.5]}).to_csv(
        os.path.join(root, "scoring", "aevplig_original.csv"), index=False)
    pd.DataFrame({"unique_id": ["compound_1"], "preds": [8.8]}).to_csv(
        os.path.join(root, "scoring", "aevplig_finetuned.csv"), index=False)
    return root


@pytest.fixture
def prepared(tmp_path):
    repo = _make_repo(str(tmp_path / "repo"))
    out = str(tmp_path / "out")
    df = pgdh_prepare.prepare_pgdh(repo, out)
    with open(os.path.join(out, "pgdh_summary.json")) as fh:
        summary = json.load(fh)
    return repo, out, df, summary


def _positions(path, index=0):
    mol = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)[index]
    return mol, mol.GetConformer().GetPositions()


def test_columns_and_unique_ids(prepared):
    repo, out, df, _ = prepared
    assert list(df.columns) == fep_prepare.COLUMNS
    on_disk = pd.read_csv(os.path.join(out, "pgdh.csv"))
    assert list(on_disk.columns) == fep_prepare.COLUMNS
    assert on_disk["unique_id"].tolist() == ["pgdh/S1/compound_1", "pgdh/S4/compound_7"]
    assert on_disk["series"].tolist() == ["S1", "S4"]
    assert set(on_disk["subset"]) == {"pgdh"}
    assert set(on_disk["ref_pdb"]) == {"9PFL"}
    assert on_disk["n_entries"].tolist() == [1, 1]
    assert on_disk["pdb_file"].nunique() == 1
    with open(on_disk["pdb_file"].iloc[0]) as fh:
        assert fh.read() == PROTEIN
    for path in on_disk["sdf_file"]:
        assert len(Chem.SDMolSupplier(path, removeHs=False)) == 1  # process_data asserts one record


def test_pk_is_experimental_not_fepplus(prepared):
    _, out, df, _ = prepared
    pk = dict(zip(df["unique_id"], df["pK"]))
    assert pk["pgdh/S1/compound_1"] == pytest.approx(8.5)
    assert pk["pgdh/S4/compound_7"] == pytest.approx(7.25)
    fep = pd.read_csv(os.path.join(out, "pgdh_fepplus.csv")).set_index("unique_id")["preds"]
    assert fep["pgdh/S1/compound_1"] == pytest.approx(6.0)
    assert fep["pgdh/S4/compound_7"] == pytest.approx(9.9)


def test_paper_csv_mapping(prepared):
    _, out, _, summary = prepared
    paper = pd.read_csv(os.path.join(out, "pgdh_paper.csv"))
    assert list(paper.columns) == ["unique_id", "preds"]
    assert dict(zip(paper["unique_id"], paper["preds"])) == {"pgdh/S1/compound_1": 7.2, "pgdh/S4/compound_7": 6.1}
    book = summary["paper_join"]["pgdh_paper.csv"]
    assert book["n_rows"] == 3 and book["n_mapped"] == 2
    assert book["paper_ids_not_in_pgdh"] == ["compound_99"]
    assert book["pgdh_ids_not_in_paper"] == []
    ft = pd.read_csv(os.path.join(out, "pgdh_paper_finetuned.csv"))
    assert dict(zip(ft["unique_id"], ft["preds"])) == {"pgdh/S1/compound_1": 8.8}
    assert summary["paper_join"]["pgdh_paper_finetuned.csv"]["pgdh_ids_not_in_paper"] == ["compound_7"]


def test_vina_pose_and_fallback_bookkeeping(prepared):
    repo, out, df, summary = prepared
    assert summary["compounds_without_vina_pose"] == ["compound_7"]
    assert summary["compounds"]["compound_1"]["pose_source"] == pgdh_prepare.VINA_SDF
    assert summary["compounds"]["compound_7"]["pose_source"] == pgdh_prepare.POSES_SDF
    assert summary["pose_source_counts"] == {pgdh_prepare.VINA_SDF: 1, pgdh_prepare.POSES_SDF: 1}
    assert summary["coords_identical_all"] is True
    files = dict(zip(df["unique_id"], df["sdf_file"]))
    # compound_1: the (shifted, H-free) Vina pose, source atoms byte-identical, hydrogens appended
    vina, vina_xyz = _positions(os.path.join(repo, pgdh_prepare.VINA_SDF))
    out1, out1_xyz = _positions(files["pgdh/S1/compound_1"])
    n = vina.GetNumAtoms()
    assert np.array_equal(out1_xyz[:n], vina_xyz)
    assert [a.GetSymbol() for a in out1.GetAtoms()][:n] == [a.GetSymbol() for a in vina.GetAtoms()]
    assert out1.GetNumAtoms() == n + summary["compounds"]["compound_1"]["n_h_added"] > n
    assert all(a.GetSymbol() == "H" for a in list(out1.GetAtoms())[n:])
    # compound_7: the poses.sdf entry (already full H), nothing added
    poses = Chem.SDMolSupplier(os.path.join(repo, pgdh_prepare.POSES_SDF), removeHs=False, sanitize=False)
    out7, out7_xyz = _positions(files["pgdh/S4/compound_7"])
    assert np.array_equal(out7_xyz, poses[1].GetConformer().GetPositions())
    assert summary["compounds"]["compound_7"]["n_h_added"] == 0


def test_asis_keeps_entry_as_read(tmp_path):
    repo = _make_repo(str(tmp_path / "repo"))
    df = pgdh_prepare.prepare_pgdh(repo, str(tmp_path / "out"), hydrogens="asis")
    vina, vina_xyz = _positions(os.path.join(repo, pgdh_prepare.VINA_SDF))
    out1, out1_xyz = _positions(df.loc[df["unique_id"] == "pgdh/S1/compound_1", "sdf_file"].iloc[0])
    assert out1.GetNumAtoms() == vina.GetNumAtoms()
    assert np.array_equal(out1_xyz, vina_xyz)


def test_missing_labels_are_recorded_and_excluded(tmp_path):
    extra = [_mol("compound_3", "2", 7.0, 7.0, "CCO", seed=7, extra={"pKi": None}),     # no pKi tag
             _mol("compound_4", "9", 7.0, 7.0, "CCO", seed=8),                          # unknown series
             _mol("compound_5", "", 7.0, 7.0, "CCO", seed=9, extra={"series": None})]   # no series tag
    assert not extra[0].HasProp("pKi") and not extra[2].HasProp("series")
    repo = _make_repo(str(tmp_path / "repo"), extra_poses=extra)
    out = str(tmp_path / "out")
    df = pgdh_prepare.prepare_pgdh(repo, out)
    with open(os.path.join(out, "pgdh_summary.json")) as fh:
        summary = json.load(fh)
    assert len(df) == 2
    assert summary["compounds_without_pK"] == ["compound_3"]
    assert summary["compounds_with_unknown_series"] == {"compound_4": "9"}
    assert summary["compounds_without_series"] == ["compound_5"]


def test_score_eval_dir(prepared, tmp_path):
    _, out, df, _ = prepared
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pred = df[["unique_id", "series"]].assign(subset="pgdh", ref_pdb="9PFL", truth=df["pK"], pred=[7.0, 6.0])
    pred.to_csv(eval_dir / "predictions_fep.csv", index=False)
    table = pgdh_prepare.score_eval_dir(str(eval_dir), out)
    assert (eval_dir / "pgdh_spearman.csv").is_file()
    assert set(table["predictor"]) == {"ours", "paper_original", "paper_finetuned", "fepplus"}
    ours = table[table["predictor"] == "ours"].set_index("group")
    assert ours.loc["all", "n"] == 2 and ours.loc["S1&S2", "n"] == 1 and ours.loc["S4", "n"] == 1
    assert ours.loc["all", "spearman"] == pytest.approx(1.0)  # pK 8.5 > 7.25 and pred 7.0 > 6.0
    fep = table[table["predictor"] == "fepplus"].set_index("group")
    assert fep.loc["all", "spearman"] == pytest.approx(-1.0)  # FEP+ 6.0 < 9.9: the label is not FEP+


@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL_REPO, "data", "poses.sdf")),
                    reason=f"real 15-PGDH repository not found at {REAL_REPO}")
def test_real_repo(tmp_path):
    out = str(tmp_path / "out")
    pgdh_prepare.main(["--repo-dir", REAL_REPO, "--out-dir", out])
    df = pd.read_csv(os.path.join(out, "pgdh.csv"))
    with open(os.path.join(out, "pgdh_summary.json")) as fh:
        summary = json.load(fh)
    assert pgdh_prepare.SERIES_MAP == {"1": "S1", "2": "S2", "3": "S3", "4_5": "S4", "4/5": "S4"}
    assert len(df) == 43 and df["unique_id"].is_unique
    assert df["series"].value_counts().to_dict() == {"S1": 11, "S2": 7, "S3": 13, "S4": 12}
    assert df["unique_id"].str.match(r"^pgdh/S[1-4]/compound_\d+$").all()
    assert set(df["subset"]) == {"pgdh"} and set(df["ref_pdb"]) == {"9PFL"} and set(df["n_entries"]) == {1}
    # series tags as read from poses.sdf map as documented (compound_40 is table "4/5")
    tags = {m.GetProp("ID"): m.GetProp("series")
            for m in Chem.SDMolSupplier(os.path.join(REAL_REPO, "data", "poses.sdf"), removeHs=False, sanitize=False)}
    assert set(tags.values()) == {"1", "2", "3", "4_5"}
    by_uid = dict(zip(df["unique_id"].str.rsplit("/", n=1).str[1], df["series"]))
    assert all(pgdh_prepare.SERIES_MAP[tags[c]] == s for c, s in by_uid.items())
    assert by_uid["compound_40"] == "S4"
    # pK is the experimental pKi, not FEP_pKi (compound_1: pKi 10.5037, FEP_pKi 10.2551)
    assert df.set_index("unique_id").loc["pgdh/S1/compound_1", "pK"] == pytest.approx(10.5037)
    for name in ("pgdh_paper.csv", "pgdh_paper_finetuned.csv"):
        paper = pd.read_csv(os.path.join(out, name))
        assert len(paper) == 43 and set(paper["unique_id"]) == set(df["unique_id"])
        assert summary["paper_join"][name]["paper_ids_not_in_pgdh"] == []
    assert summary["compounds_without_vina_pose"] == []
    assert summary["coords_identical_all"] is True
    assert summary["not_readable_by_process_data"] == []
