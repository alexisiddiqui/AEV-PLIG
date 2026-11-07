"""Tests for src/featurise_poses.py and src/aevplig_featurise.py. AEVs are stubbed (torchani is not in
the local venv); ligands are written as minimal Tripos mol2 files, proteins as ALA-only PDB files."""
import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import aevplig_featurise as aevf
import featurise_poses as fp
from pose_augment import Pose, heavy_xyz, n_within

CLONE_SCRIPT = ("/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/"
                "scratchpad/census/data/AEV-PLIG/generate_pdbbind_graphs.py")
AEV_WIDTH = 352  # 22 atom types x 16 radial shifts
ATOM_KEYS = ("PDB_ATOM,ATOM_TYPE\nALA-C,C;4;3;0;0;0\nALA-CA,C;4;3;1;0;0\nALA-CB,C;4;1;3;0;0\n"
             "ALA-N,N;3;2;1;0;0\nALA-O,O;2;1;0;0;0\n")
LIGANDS = {"1abc": "CCCCO", "2def": "CCOCC(N)C", "3ghi": "OCCN(C)C", "4jkl": "CC(C)CO"}
SPLITS = {"1abc": "train", "2def": "valid", "3ghi": "test", "4jkl": "train"}
REFINED = {"1abc": True, "2def": False, "3ghi": False, "4jkl": True}
MOL2_TYPES = {"C": "C.3", "O": "O.3", "N": "N.3", "H": "H"}


def zero_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map):
    mol_df = aevf.LoadMolasDF(mol)
    return mol_df, np.zeros((len(mol_df), AEV_WIDTH))


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


def write_protein(path, center, n=60, seed=0):
    rng = np.random.default_rng(seed)
    names = ["N", "CA", "C", "O", "CB"]
    lines = []
    for i in range(n):
        v = rng.normal(size=3)
        x, y, z = center + v / np.linalg.norm(v) * rng.uniform(5.5, 9.0)
        name = names[i % 5]
        lines.append(f"ATOM  {i + 1:5d}  {name:<3s} ALA A{i // 5 + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00           {name[0]}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "pdbbind"
    for i, (pdb, smi) in enumerate(LIGANDS.items()):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        assert AllChem.EmbedMolecule(mol, randomSeed=i + 1) == 0
        d = root / ("refined-set" if REFINED[pdb] else "general-set") / pdb
        d.mkdir(parents=True)
        write_mol2(mol, d / f"{pdb}_ligand.mol2")
        if pdb != "3ghi":  # 3ghi: missing protein -> recorded error
            write_protein(d / f"{pdb}_protein.pdb", heavy_xyz(mol).mean(axis=0), seed=i)
    meta = pd.DataFrame({
        "row": range(6),
        "unique_id": ["4jkl", "2def", "1abc", "3ghi", "T1_1swr_C1", "dbfolder/x"],
        "source": ["pdbbind"] * 4 + ["bindingnet", "bindingdb"],
        "split": [SPLITS["4jkl"], SPLITS["2def"], SPLITS["1abc"], SPLITS["3ghi"], "train", "train"],
        "pK": [5.0, 6.0, 7.0, 8.0, 5.5, 6.5],
    })
    meta.to_csv(tmp_path / "paired_meta.csv", index=False)
    pdbb = pd.DataFrame({"PDB_code": list(LIGANDS), "-logKd/Ki": [7.0, 6.0, 8.0, 5.0],
                         "refined": [REFINED[p] for p in LIGANDS], "core": [False] * 4,
                         "split_core": [SPLITS[p] for p in LIGANDS],
                         "max_tanimoto_fep_benchmark": [0.1] * 4})
    pdbb.to_csv(tmp_path / "pdbbind_processed.csv")
    (tmp_path / "keys.csv").write_text(ATOM_KEYS)
    return tmp_path


def base_args(t):
    return ["--pdbbind-root", str(t / "pdbbind"), "--meta", str(t / "paired_meta.csv"),
            "--pdbbind-csv", str(t / "pdbbind_processed.csv"), "--out-dir", str(t / "out"),
            "--workers", "1", "--atom-keys", str(t / "keys.csv")]


def shifted(mol, dx):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, (p + [dx, 0.0, 0.0]).tolist())
    return out


def make_fake_generate(calls, short_neg=()):
    def fake(mol_h, target_xyz, k_pos, k_neg, rng):
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol_h), isomericSmiles=False)  # mol2 stereo from 3D
        pdb = next(p for p in LIGANDS if smi == Chem.MolToSmiles(Chem.MolFromSmiles(LIGANDS[p])))
        calls.append((pdb, target_xyz.shape, rng.bit_generator.state ==
                      np.random.default_rng(zlib.crc32(pdb.encode())).bit_generator.state))
        poses = [Pose(1, k, "nn", shifted(mol_h, 0.5 + k), 0.8 + k, 3.0, 1.0, 1) for k in range(k_pos)]
        n_neg = 1 if pdb in short_neg else k_neg
        poses += [Pose(2, k, ["flip", "conf"][k % 2], shifted(mol_h, 5.0 + k), 4.5 + k, 2.5, 0.7, 3, 7)
                  for k in range(n_neg)]
        return poses
    return fake


def test_graph_assembly_matches_clone_format(tree, monkeypatch):
    mol = fp.load_ligand(str(tree / "pdbbind" / "refined-set" / "1abc" / "1abc_ligand.mol2"))
    n_heavy = sum(a.GetSymbol() != "H" for a in mol.GetAtoms())
    n_heavy_bonds = sum(b.GetBeginAtom().GetSymbol() != "H" and b.GetEndAtom().GetSymbol() != "H"
                        for b in mol.GetBonds())
    mol_df, aevs = zero_aevs(None, mol, None, None, None)
    c_size, features, edge_index, edge_attr = aevf.mol_to_graph(mol, mol_df, aevs)
    assert c_size == n_heavy == len(features)
    assert all(len(f) == 367 for f in features)
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    assert np.array_equal(features[0][:15], aevf.atom_features(heavy_atoms[0]))
    assert len(edge_index) == len(edge_attr) == 2 * n_heavy_bonds
    assert all(len(e) == 2 for e in edge_index) and all(len(e) == 4 for e in edge_attr)
    # the worker path gives the same tuple
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", zero_aevs)
    fp._init_worker(str(tree / "keys.csv"))
    g = fp.featurise_mol("unused.pdb", mol)
    assert g[0] == c_size and np.array_equal(g[1], features) and g[2] == edge_index and g[3] == edge_attr


def test_unique_ids_and_meta_rows(tree, monkeypatch):
    calls = []
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    monkeypatch.setattr(fp, "generate_poses", make_fake_generate(calls))
    fp.main(base_args(tree))
    out = tree / "out"
    with open(out / "pdbbind_poses.pickle", "rb") as f:
        graphs = pickle.load(f)
    ok = ["1abc", "2def", "4jkl"]  # 3ghi has no protein file
    expected = {f"{p}::{t}{k}" for p in ok for t in ("pos", "neg") for k in (0, 1)}
    assert set(graphs) == expected
    assert all(len(g[1][0]) == 367 for g in graphs.values())
    assert not np.array_equal(graphs["1abc::pos0"][1], graphs["1abc::pos1"][1])  # pose coords reach the AEV
    meta = pd.read_csv(out / "pdbbind_poses_meta.csv", dtype={"pdb": str})
    assert list(meta.columns) == fp.META_COLUMNS
    assert len(meta) == 12 and set(meta["unique_id"]) == expected
    assert (meta["split"] == meta["pdb"].map(SPLITS)).all()
    row = meta.set_index("unique_id").loc["2def::neg1"]
    assert (row["kind"], row["k"], row["method"], row["attempts"]) == (2, 1, "conf", 3)
    assert fp.META_COLUMNS[-2:] == ["relief_steps", "n_within_3A"] and row["relief_steps"] == 7
    assert (meta.loc[meta["kind"] == 1, "relief_steps"] == 0).all()
    # n_within_3A: ligand heavy atoms of the written pose closer than 3.0 A to a protein heavy atom
    keys, _ = aevf.load_atom_keys(str(tree / "keys.csv"))
    for pdb in ok:
        folder = tree / "pdbbind" / ("refined-set" if REFINED[pdb] else "general-set") / pdb
        mol = fp.load_ligand(str(folder / f"{pdb}_ligand.mol2"))
        target = aevf.LoadPDBasDF(str(folder / f"{pdb}_protein.pdb"), keys)[["X", "Y", "Z"]].to_numpy(float)
        for tag, dx0 in (("pos", 0.5), ("neg", 5.0)):
            for k in (0, 1):
                expected = n_within(heavy_xyz(shifted(mol, dx0 + k)), target)
                assert meta.set_index("unique_id").loc[f"{pdb}::{tag}{k}", "n_within_3A"] == expected
    assert meta["n_within_3A"].max() > 0  # some shifted decoys sit against the protein shell
    assert set(meta.loc[meta["kind"] == 1, "method"]) == {"nn"}
    # one call per readable complex, typed protein atoms as the target, crc32-seeded rng
    assert sorted(c[0] for c in calls) == ok
    assert all(c[1] == (60, 3) and c[2] for c in calls)


def test_limit_and_summary_json(tree, monkeypatch):
    calls = []
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    monkeypatch.setattr(fp, "generate_poses", make_fake_generate(calls, short_neg=("2def",)))
    fp.main(base_args(tree) + ["--limit", "3"])
    out = tree / "out"
    summary = json.loads((out / "pdbbind_poses_summary.json").read_text())
    meta = pd.read_csv(out / "pdbbind_poses_meta.csv", dtype={"pdb": str})
    assert set(meta["pdb"]) == {"1abc", "2def"}  # first 3 by pdb id; 3ghi errors
    assert summary["n_complexes"] == 3 and summary["n_complexes_ok"] == 2
    assert summary["counts_kind"] == {"1": 4, "2": 3}
    assert summary["counts_split_kind"]["valid"] == {"1": 2, "2": 1}
    assert summary["counts_method"] == {"nn": 4, "flip": 2, "conf": 1}
    assert summary["failures"]["neg"]["n"] == 1 and summary["failures"]["pos"]["n"] == 0
    assert summary["failure_complexes"]["neg"] == [["2def", "valid", 1]]
    assert summary["n_errors"] == 1 and summary["errors"][0][0] == "3ghi"
    assert set(summary["quantiles"]["2"]) == {"rmsd", "min_dist", "contact_frac", "attempts"}
    assert summary["quantiles"]["2"]["rmsd"]["0.5"] == pytest.approx(4.5)
    assert summary["relief_steps_quantiles"]["2"]["0.5"] == 7
    assert summary["relief_steps_quantiles"]["1"]["1.0"] == 0 and summary["decoys_relieved"] == 3
    assert set(summary["n_within_3A_quantiles"]) == {"1", "2"}
    assert summary["crystal_min_dist_quantiles"]["0.0"] > 0
    assert summary["crystal_n_within_3A_quantiles"]["0.5"] >= 0
    cue = summary["cue_auroc_min_dist"]
    assert cue["nn_vs_decoy"] == 1.0  # fake poses: nn min_dist 3.0 > decoy 2.5
    assert all(0.0 <= cue[c] <= 1.0 for c in ("crystal_vs_decoy", "crystal_vs_nn"))
    assert summary["identity_check"] is None


def test_auroc_rank_form():
    assert fp._auroc([1.0, 2.0], [1.0, 2.0]) == 0.5
    assert fp._auroc([3.0], [1.0, 2.0]) == 1.0
    assert fp._auroc([0.0, 3.0], [1.0, 2.0]) == 0.5
    assert fp._auroc([1.0], [1.0]) == 0.5
    assert fp._auroc([], [1.0]) is None


def test_check_identity_ok_then_mismatch(tree, monkeypatch, capsys):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    parent = {}
    for pdb in ("1abc", "2def"):  # the clone's crystal path, written independently of featurise_poses
        folder = tree / "pdbbind" / ("refined-set" if REFINED[pdb] else "general-set") / pdb
        mol = Chem.AddHs(Chem.MolFromMol2File(str(folder / f"{pdb}_ligand.mol2")), addCoords=True)
        parent[pdb] = aevf.mol_to_graph(mol, *coord_aevs(None, mol, None, None, None))
    pk = tree / "parent.pickle"
    pk.write_bytes(pickle.dumps(parent))
    args = base_args(tree) + ["--check-identity", "2", "--parent-pickle", str(pk)]
    fp.main(args)
    assert "IDENTITY OK n=2" in capsys.readouterr().out
    ident = json.loads((tree / "out" / "pdbbind_poses_identity.json").read_text())
    assert ident["ok"] is True and ident["n"] == 2
    assert not (tree / "out" / "pdbbind_poses.pickle").exists()  # check mode only checks

    monkeypatch.setattr(fp, "generate_poses", make_fake_generate([]))
    fp.main(base_args(tree))
    summary = json.loads((tree / "out" / "pdbbind_poses_summary.json").read_text())
    assert summary["identity_check"]["ok"] is True

    parent["2def"][1][0][20] += 1e-12
    pk.write_bytes(pickle.dumps(parent))
    with pytest.raises(SystemExit) as exc:
        fp.main(args)
    assert exc.value.code == 3
    assert "IDENTITY MISMATCH" in capsys.readouterr().out
    assert json.loads((tree / "out" / "pdbbind_poses_identity.json").read_text())["ok"] is False


@pytest.mark.skipif(not os.path.exists(CLONE_SCRIPT), reason="clone script not available")
def test_featuriser_copy_is_verbatim(tmp_path):
    assert aevf.source_mismatches(CLONE_SCRIPT) == []
    text = open(CLONE_SCRIPT).read()
    assert "            counter += 1\n" in text
    edited = tmp_path / "generate_pdbbind_graphs.py"
    edited.write_text(text.replace("            counter += 1\n", "            counter += 2\n"))
    assert aevf.source_mismatches(str(edited)) == ["mol_to_graph"]
