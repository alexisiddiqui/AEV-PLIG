"""Tests for src/featurise_edits.py (scheme C featuriser). AEVs are stubbed (torchani is not in the local
venv); ligands are written as minimal Tripos mol2 files, proteins as a few typed PDB atoms placed next to
chosen ligand groups."""
import json
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import aevplig_featurise as aevf
import featurise_edits as fe
import ligand_edits as le
from pose_augment import heavy_xyz

AEV_WIDTH = 352
ATOM_KEYS = ("PDB_ATOM,ATOM_TYPE\nASP-OD1,O;2;1;0;0;0\nASP-CG,C;3;3;0;0;0\nALA-CB,C;4;1;3;0;0\n"
             "ALA-O,O;2;1;0;0;0\n")
# amine at one end (an Asp carboxylate O next to it: contact), hydroxyl at the other (exposed)
LIGANDS = {"1abc": "NCCCCCCCCO", "2def": "NCCCCCCCCCO", "3ghi": "NCCCCCCO", "4jkl": "NCCCCCCCO",
           "5mno": "CC(C)(C)C"}
SPLITS = {"1abc": "train", "2def": "valid", "3ghi": "test", "4jkl": "train", "5mno": "train"}
REFINED = {"1abc": True, "2def": False, "3ghi": False, "4jkl": True, "5mno": False}
MOL2_TYPES = {"C": "C.3", "O": "O.3", "N": "N.3", "H": "H"}
FIRST_TEN = ["unique_id", "pdb", "kind", "k", "method", "split", "rmsd_to_crystal", "min_dist",
             "contact_frac", "n_within_3A"]
C_COLUMNS = ["edit_rule", "site_class", "n_edited_atoms", "n_contacts_edited_group", "partner_type",
             "mmff_restrained_rmsd"]


def coord_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map):
    """Deterministic and coordinate-dependent, so different poses give different graphs."""
    mol_df = aevf.LoadMolasDF(mol)
    vals = mol_df[["X", "Y", "Z"]].to_numpy(dtype=float).sum(axis=1, keepdims=True)
    return mol_df, np.repeat(vals, AEV_WIDTH, axis=1) + np.arange(AEV_WIDTH)


def write_mol2(mol, path):
    conf = mol.GetConformer()
    lines = ["@<TRIPOS>MOLECULE", "LIG", f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0", "SMALL",
             "NO_CHARGES", "", "@<TRIPOS>ATOM"]
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetIdx() + 1:7d} {a.GetSymbol()}{a.GetIdx() + 1:<5d} {p.x:10.4f} {p.y:10.4f} "
                     f"{p.z:10.4f} {MOL2_TYPES[a.GetSymbol()]:<5s} 1 LIG1 0.0000")
    lines.append("@<TRIPOS>BOND")
    for b in mol.GetBonds():
        lines.append(f"{b.GetIdx() + 1:6d} {b.GetBeginAtomIdx() + 1:5d} {b.GetEndAtomIdx() + 1:5d} 1")
    path.write_text("\n".join(lines) + "\n")


def _pdb_line(i, name, res, resseq, xyz, record="ATOM  "):
    x, y, z = xyz
    return (f"{record}{i:5d}  {name:<3s} {res} A{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00           {name[0]}")


def write_protein(path, mol):
    """Asp OD1 3.2 A from the amine N (outward from the ligand centroid) and a water and a cofactor
    atom far away; nothing near the hydroxyl."""
    xyz = mol.GetConformer().GetPositions()
    c = heavy_xyz(mol).mean(axis=0)
    n_idx = ([a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"] + [0])[0]
    v = (xyz[n_idx] - c) / np.linalg.norm(xyz[n_idx] - c)
    lines = [_pdb_line(1, "OD1", "ASP", 1, xyz[n_idx] + v * 3.2),
             _pdb_line(2, "CG", "ASP", 1, xyz[n_idx] + v * 4.4),
             _pdb_line(3, "O", "HOH", 2, c + np.array([30.0, 0, 0]), record="HETATM"),
             _pdb_line(4, "C1", "NAD", 3, c + np.array([0, 30.0, 0]), record="HETATM"),
             "END"]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "pdbbind"
    for i, (pdb, smi) in enumerate(LIGANDS.items()):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 100 + i
        assert AllChem.EmbedMolecule(mol, params) == 0
        d = root / ("refined-set" if REFINED[pdb] else "general-set") / pdb
        d.mkdir(parents=True)
        write_mol2(mol, d / f"{pdb}_ligand.mol2")
        if pdb != "4jkl":  # 4jkl: missing protein -> recorded error
            write_protein(d / f"{pdb}_protein.pdb", mol)
    meta = pd.DataFrame({
        "row": range(7),
        "unique_id": ["4jkl", "2def", "1abc", "3ghi", "5mno", "T1_1swr_C1", "dbfolder/x"],
        "source": ["pdbbind"] * 5 + ["bindingnet", "bindingdb"],
        "split": [SPLITS[p] for p in ("4jkl", "2def", "1abc", "3ghi", "5mno")] + ["train", "train"],
        "pK": [5.0, 6.0, 7.0, 8.0, 4.0, 5.5, 6.5],
    })
    meta.to_csv(tmp_path / "paired_meta.csv", index=False)
    pdbb = pd.DataFrame({"PDB_code": list(LIGANDS), "-logKd/Ki": [7.0, 6.0, 8.0, 5.0, 4.0],
                         "refined": [REFINED[p] for p in LIGANDS], "core": [False] * 5,
                         "split_core": [SPLITS[p] for p in LIGANDS],
                         "max_tanimoto_fep_benchmark": [0.1] * 5})
    pdbb.to_csv(tmp_path / "pdbbind_processed.csv")
    (tmp_path / "keys.csv").write_text(ATOM_KEYS)
    return tmp_path


def base_args(t, out="out"):
    return ["--pdbbind-root", str(t / "pdbbind"), "--meta", str(t / "paired_meta.csv"),
            "--parent-data", str(t), "--out-dir", str(t / out), "--workers", "1",
            "--atom-keys", str(t / "keys.csv")]


def test_featurise_edits_layout(tree, monkeypatch):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    fe.main(base_args(tree))
    out = tree / "out"
    with open(out / fe.EDITS_PICKLE, "rb") as f:
        graphs = pickle.load(f)
    ok = ["1abc", "2def"]  # 3ghi is a test anchor (not processed), 4jkl has no protein, 5mno has no site
    expected = {f"{p}::{t}" for p in ok for t in ("epos0", "cedit0", "xedit0")} | {"5mno::epos0"}
    assert set(graphs) == expected
    assert all(len(g) == 4 and len(g[1][0]) == 367 for g in graphs.values())
    assert not np.array_equal(graphs["1abc::epos0"][1], graphs["1abc::cedit0"][1])
    meta = pd.read_csv(out / fe.EDITS_META, dtype={"pdb": str}, keep_default_na=False)
    assert list(meta.columns[:16]) == FIRST_TEN + C_COLUMNS
    assert list(meta.columns) == fe.META_COLUMNS
    assert set(meta["unique_id"]) == expected
    assert (meta["split"] == meta["pdb"].map(SPLITS)).all()
    m = meta.set_index("unique_id")
    for pdb in ok:
        pos, neg, ctl = m.loc[f"{pdb}::epos0"], m.loc[f"{pdb}::cedit0"], m.loc[f"{pdb}::xedit0"]
        assert (pos["kind"], pos["k"], pos["method"], pos["edit_rule"], pos["site_class"]) == \
            (5, 0, "null", "null", "null")
        assert (neg["kind"], neg["method"], neg["edit_rule"], neg["site_class"], neg["partner_type"]) == \
            (6, "amine_to_me", "amine_to_me", "contact", "neg")
        assert (ctl["kind"], ctl["method"], ctl["edit_rule"], ctl["site_class"], ctl["partner_type"]) == \
            (7, "oh_to_f", "oh_to_f", "exposed", "")
        assert neg["n_contacts_edited_group"] == 1 and neg["n_edited_atoms"] == 1
        assert pos["n_edited_atoms"] == 0
        assert graphs[f"{pdb}::epos0"][0] == pos["n_heavy"] == neg["n_heavy"] == ctl["n_heavy"]
    assert (meta["min_dist"] >= 2.2).all() and (meta["contact_frac"] >= 0.5).all()
    assert (meta["mmff_restrained_rmsd"] < 0.3).all() and (meta["n_within_3A"] >= 0).all()
    assert set(meta["dcharge"]) == {0}


def test_summary_json(tree, monkeypatch):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    fe.main(base_args(tree))
    s = json.loads((tree / "out" / fe.EDITS_SUMMARY).read_text())
    assert s["n_anchors"] == 4 and s["n_anchors_ok"] == 3          # train/valid PDBbind rows only
    assert s["anchors_by_split"] == {"train": 3, "valid": 1}
    assert s["counts_split_kind"] == {"train": {"5": 2, "6": 1, "7": 1}, "valid": {"5": 1, "6": 1, "7": 1}}
    assert s["pose_counts"]["amine_to_me"]["contact"] == {"train": 1, "valid": 1}
    assert s["pose_counts"]["oh_to_f"]["exposed"] == {"train": 1, "valid": 1}
    assert s["candidate_counts"]["amine_to_me"]["contact"]["train"] == 1
    assert s["candidate_counts"]["oh_to_f"]["exposed"]["valid"] == 1
    assert s["n_errors"] == 1 and s["errors"][0][0] == "4jkl"
    assert s["failures"]["anchors_without_null"] == 0
    assert s["failures"]["contact_edits_per_anchor"] == {"0": 1, "1": 2}
    assert set(s["mmff_restrained_rmsd_quantiles"]) == {"5", "6", "7"}
    assert set(s["cue_auroc"]) == {"5_vs_6", "5_vs_7"}
    for pair in s["cue_auroc"].values():
        assert {"min_dist", "n_within_3A", "n_heavy"} <= set(pair)
        assert all(v is None or 0.0 <= v <= 1.0 for v in pair.values())
    # P(pos > neg) + P(tie) / 2 over kind 5 (10, 11, 5 heavy atoms) x kind 6 (10, 11): (0.5+0+1+0.5+0+0)/6
    assert s["cue_auroc"]["5_vs_6"]["n_heavy"] == pytest.approx(2.0 / 6.0)
    assert s["place_method_counts"] and s["k_contact"] == 2 and s["k_exposed"] == 2
    assert s["exposed_radius"] == 3.5
    assert set(s["exposed_by_radius"]) == {"3.5", "4.0", "4.5"}
    for d in s["exposed_by_radius"].values():   # one exposed hydroxyl in each of 1abc and 2def
        assert d == {"exposed_sites": 2, "anchors_with_exposed_site": 2, "kind7_if_all_placed": 2}


def test_limit_and_determinism(tree, monkeypatch):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    fe.main(base_args(tree, "a") + ["--limit", "2"])
    fe.main(base_args(tree, "b") + ["--limit", "2"])
    a = pd.read_csv(tree / "a" / fe.EDITS_META, dtype={"pdb": str})
    b = pd.read_csv(tree / "b" / fe.EDITS_META, dtype={"pdb": str})
    assert set(a["pdb"]) == {"1abc", "2def"}  # first two train/valid anchors by pdb id
    pd.testing.assert_frame_equal(a, b)
    ga = pickle.loads((tree / "a" / fe.EDITS_PICKLE).read_bytes())
    gb = pickle.loads((tree / "b" / fe.EDITS_PICKLE).read_bytes())
    assert all(np.array_equal(ga[k][1], gb[k][1]) for k in ga)


def test_worker_plumbing(tree, monkeypatch):
    """crc32-seeded rng, AEV-typed protein atoms as the target, protein_fields for classification,
    non-water HETATM atoms as extra exposure atoms, K from the command line."""
    calls = []
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)

    def fake(mol_h, fields, rng, k_contact, k_exposed, *, target_xyz, extra_xyz, exposed_radius, log):
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol_h))
        pdb = next(p for p in LIGANDS if smi == Chem.MolToSmiles(Chem.MolFromSmiles(LIGANDS[p])))
        calls.append((pdb, rng.bit_generator.state ==
                      np.random.default_rng(zlib.crc32(pdb.encode())).bit_generator.state,
                      target_xyz.shape, fields[0].shape, fields[1].shape, extra_xyz.shape,
                      k_contact, k_exposed, exposed_radius))
        log.update(n_sites=0, candidates={}, failures={}, null_failure="")
        return []

    monkeypatch.setattr(fe, "generate_edits", fake)
    fe.main(base_args(tree) + ["--k-contact", "1", "--k-exposed", "3", "--exposed-radius", "4.0"])
    assert sorted(c[0] for c in calls) == ["1abc", "2def", "5mno"]
    for c in calls:
        assert c[1] and c[2] == (2, 3) and c[3] == (2, 3) and c[4] == (2, 7)
        assert c[5] == (1, 3)                       # the NAD atom; the water is dropped
        assert c[6:] == (1, 3, 4.0)
    s = json.loads((tree / "out" / fe.EDITS_SUMMARY).read_text())
    assert s["n_poses"] == 0 and s["n_anchors_ok"] == 3 and s["exposed_radius"] == 4.0


def test_null_failure_recorded(tree, monkeypatch):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)

    def boom(*a, **k):
        raise le.EditFailure("embed_failed")

    monkeypatch.setattr(le, "null_edit", boom)
    fe.main(base_args(tree))
    s = json.loads((tree / "out" / fe.EDITS_SUMMARY).read_text())
    assert s["n_poses"] == 0
    assert s["failures"]["anchors_without_null"] == 3
    assert s["failures"]["null"] == {"embed_failed": 3}
