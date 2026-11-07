"""Channel assignment for ligands and protein pockets, pocket mask, descriptors."""
import os

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from scipy.spatial.transform import Rotation

import pocket_fields as pf


def _mol3d(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _heavy_index(mol, symbol):
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return [k for k, a in enumerate(heavy) if a.GetSymbol() == symbol]


def test_channel_order():
    assert pf.CHANNELS == ("occupancy", "hydrophobicity", "pos", "neg", "aromatic", "donor", "acceptor")


def test_benzene_all_aromatic_and_neutral():
    xyz, w = pf.ligand_fields(_mol3d("c1ccccc1"))
    assert xyz.shape == (6, 3) and w.shape == (6, 7)
    assert np.all(w[:, pf.OCC] == 1)
    assert np.all(w[:, pf.ARO] == 1)
    assert np.all(w[:, pf.DON] == 0) and np.all(w[:, pf.ACC] == 0)
    assert np.abs(w[:, pf.POS]).max() < 1e-6 and np.abs(w[:, pf.NEG]).max() < 1e-6  # united-atom CH
    assert np.all(w[:, pf.HYD] > 0)


def test_acetate_negative_on_both_oxygens():
    mol = _mol3d("CC(=O)[O-]")
    xyz, w = pf.ligand_fields(mol)
    ox = _heavy_index(mol, "O")
    assert len(ox) == 2
    assert np.all(w[ox, pf.NEG] > 0.3)
    assert np.all(w[ox, pf.POS] == 0)
    assert w[:, pf.ARO].sum() == 0
    assert w[ox, pf.ACC].sum() >= 1  # BaseFeatures types the carbonyl O as an acceptor
    # united-atom charges sum to the formal charge
    assert w[:, pf.POS].sum() - w[:, pf.NEG].sum() == pytest.approx(-1.0, abs=1e-6)


def test_ammonium_positive_donor_and_phenol_features():
    mol = _mol3d("CC[NH3+]")
    _, w = pf.ligand_fields(mol)
    n = _heavy_index(mol, "N")[0]
    assert w[n, pf.POS] > 0.3 and w[n, pf.DON] == 1
    mol = _mol3d("c1ccccc1O")
    _, w = pf.ligand_fields(mol)
    o = _heavy_index(mol, "O")[0]
    assert w[o, pf.DON] == 1 and w[o, pf.ACC] == 1
    assert w[:, pf.ARO].sum() == 6


def test_crippen_sum_matches_mollogp_and_implicit_h_input():
    mol = _mol3d("CC(=O)Nc1ccc(O)cc1")
    _, w = pf.ligand_fields(mol)
    assert w[:, pf.HYD].sum() == pytest.approx(Crippen.MolLogP(mol), abs=1e-6)
    _, w_implicit = pf.ligand_fields(Chem.RemoveHs(mol))
    np.testing.assert_allclose(w_implicit, w, atol=1e-9)


def test_ligand_descriptor_rotation_invariant():
    mol = _mol3d("CC(=O)Nc1ccc(O)cc1")
    f0, g0, n0, out0 = pf.ligand_descriptor(mol, order=12, radius=12.0)
    assert f0.shape == (7, 49) and g0.shape == (21, 49) and n0 == 11 and out0 == 0
    rot = Rotation.random(random_state=3).as_matrix()
    conf = mol.GetConformer()
    pos = conf.GetPositions() @ rot.T + np.array([5.0, -3.0, 2.0])
    moved = Chem.Mol(mol)
    for i, p in enumerate(pos):
        moved.GetConformer().SetAtomPosition(i, p.tolist())
    f1, g1, _, _ = pf.ligand_descriptor(moved, order=12, radius=12.0)
    np.testing.assert_allclose(f1, f0, atol=1e-9)
    np.testing.assert_allclose(g1, g0, atol=1e-9)


# ---- protein side

def _atom_line(serial, name, resname, chain, resseq, x, y, z, element, record="ATOM", altloc=" "):
    padded = f" {name:<3}" if len(name) < 4 else name
    return (f"{record:<6}{serial:>5} {padded}{altloc}{resname:>3} {chain}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")


def _write_pdb(path, atoms):
    with open(path, "w") as fh:
        for i, a in enumerate(atoms, 1):
            fh.write(_atom_line(i, *a))
        fh.write("END\n")
    return str(path)


LYS = [("N", "LYS", "A", 1, 0.0, 0.0, 0.0, "N"), ("CA", "LYS", "A", 1, 1.4, 0.0, 0.0, "C"),
       ("C", "LYS", "A", 1, 2.0, 1.4, 0.0, "C"), ("O", "LYS", "A", 1, 1.4, 2.4, 0.0, "O"),
       ("CB", "LYS", "A", 1, 2.0, -1.0, 1.0, "C"), ("CG", "LYS", "A", 1, 3.4, -1.0, 1.4, "C"),
       ("CD", "LYS", "A", 1, 4.0, -2.0, 2.4, "C"), ("CE", "LYS", "A", 1, 5.4, -2.0, 2.8, "C"),
       ("NZ", "LYS", "A", 1, 6.0, -3.0, 3.8, "N"), ("HZ1", "LYS", "A", 1, 6.9, -3.0, 3.8, "H"),
       ("1HB", "LYS", "A", 1, 2.0, -1.5, 1.8, "H")]


def test_lys_residue_channels(tmp_path):
    atoms = LYS + [("N", "PRO", "A", 2, 3.0, 2.0, 0.0, "N"), ("CA", "PRO", "A", 2, 4.0, 2.5, 0.0, "C"),
                   ("CD", "PRO", "A", 2, 3.0, 3.0, 1.0, "C")]
    path = _write_pdb(tmp_path / "lys.pdb", atoms)
    with open(path, "a") as fh:  # a water and a non-standard cap are not read
        fh.write(_atom_line(99, "O", "HOH", "A", 100, 9.0, 9.0, 9.0, "O", record="HETATM"))
        fh.write(_atom_line(98, "CH3", "ACE", "A", 0, 8.0, 8.0, 8.0, "C"))
    xyz, w, resid = pf.protein_fields(path)
    names = [r for r in resid]
    assert xyz.shape == (12, 3)  # 9 LYS heavy atoms + 3 PRO; H, 1HB, water and ACE dropped
    assert names[0] == "A:LYS1" and names[-1] == "A:PRO2"
    nz = 8
    assert w[nz, pf.POS] == 1.0 and w[nz, pf.DON] == 1.0
    assert w[:, pf.POS].sum() == 1.0 and w[:, pf.NEG].sum() == 0.0
    assert w[0, pf.DON] == 1.0          # LYS backbone N
    assert w[9, pf.DON] == 0.0          # PRO backbone N
    assert w[3, pf.ACC] == 1.0          # backbone O
    assert w[:, pf.ARO].sum() == 0
    assert w[nz, pf.HYD] == pytest.approx(pf.CRIPPEN_TABLE["LYS"]["NZ"])


def test_charged_and_aromatic_residues(tmp_path):
    atoms = [("CG", "ASP", "B", 5, 0, 0, 0, "C"), ("OD1", "ASP", "B", 5, 1, 0, 0, "O"),
             ("OD2", "ASP", "B", 5, 0, 1, 0, "O"), ("OXT", "ASP", "B", 5, 0, 0, 1, "O"),
             ("NE", "ARG", "B", 6, 3, 0, 0, "N"), ("NH1", "ARG", "B", 6, 4, 0, 0, "N"),
             ("NH2", "ARG", "B", 6, 4, 1, 0, "N"), ("ND1", "HIE", "B", 7, 6, 0, 0, "N"),
             ("NE2", "HIE", "B", 7, 6, 1, 0, "N"), ("CZ", "PHE", "B", 8, 8, 0, 0, "C"),
             ("CB", "PHE", "B", 8, 8, 1, 0, "C"), ("SD", "MET", "B", 9, 10, 0, 0, "S")]
    xyz, w, resid = pf.protein_fields(_write_pdb(tmp_path / "mix.pdb", atoms))
    assert len(xyz) == 12
    np.testing.assert_allclose(w[:4, pf.NEG], [0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(w[4:7, pf.POS], [1 / 3] * 3)
    np.testing.assert_allclose(w[7:9, pf.POS], [0.1, 0.1])    # HIE read as HIS
    assert resid[7] == "B:HIS7"
    assert w[7, pf.ARO] == 1 and w[7, pf.DON] == 1 and w[7, pf.ACC] == 1
    assert w[9, pf.ARO] == 1 and w[10, pf.ARO] == 0          # PHE CZ ring, CB not
    assert w[11, pf.ACC] == 1 and w[11, pf.DON] == 0         # MET SD
    assert w[1, pf.HYD] == w[2, pf.HYD]                      # symmetrised carboxylate


def test_altloc_and_models_keep_first(tmp_path):
    atoms = [("CA", "GLY", "A", 1, 0, 0, 0, "C"), ("CA", "GLY", "A", 1, 0.3, 0, 0, "C")]
    path = tmp_path / "alt.pdb"
    with open(path, "w") as fh:
        fh.write(_atom_line(1, *atoms[0], altloc="A"))
        fh.write(_atom_line(2, *atoms[1], altloc="B"))
        fh.write("ENDMDL\n")
        fh.write(_atom_line(3, "N", "GLY", "A", 2, 5, 5, 5, "N"))
    xyz, _, _ = pf.protein_fields(str(path))
    np.testing.assert_allclose(xyz, [[0, 0, 0]])


def test_protein_dataframe_input():
    df = pd.DataFrame({"PDB_ATOM": ["LYS-NZ", "GLU-OE1", "XYZ-C1"], "X": [0.0, 1.0, 2.0],
                       "Y": [0.0, 0.0, 0.0], "Z": [0.0, 0.0, 0.0]})
    xyz, w, _ = pf.protein_fields(df)
    assert len(xyz) == 2
    assert w[0, pf.POS] == 1 and w[1, pf.NEG] == 0.5


def test_crippen_table_reproduces_from_rdkit():
    table, oxt = pf.crippen_table_from_rdkit()
    assert oxt == pytest.approx(pf.OXT_CRIPPEN, abs=1e-9)
    assert set(table) == set(pf.CRIPPEN_TABLE)
    for res, atoms in table.items():
        assert set(atoms) == set(pf.CRIPPEN_TABLE[res]), res
        for atom, value in atoms.items():
            assert value == pytest.approx(pf.CRIPPEN_TABLE[res][atom], abs=1e-9), (res, atom)
    assert len(pf.STANDARD_ATOMS) == 187
    keys_csv = os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv")
    if os.path.exists(keys_csv):
        keys = pd.read_csv(keys_csv)
        assert set(tuple(k.split("-", 1)) for k in keys["PDB_ATOM"]) == set(pf.STANDARD_ATOMS)


def test_pocket_mask_and_descriptor():
    rng = np.random.default_rng(0)
    prot_xyz = rng.uniform(-20, 20, size=(4000, 3))
    w = np.zeros((4000, 7))
    w[:, pf.OCC] = 1
    lig_xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [9.0, 0.0, 0.0]])
    mask = pf.pocket_atoms(prot_xyz, lig_xyz, cutoff=8.0)
    dmin = np.linalg.norm(prot_xyz[:, None, :] - lig_xyz[None], axis=2).min(axis=1)
    assert mask.sum() == (dmin <= 8.0).sum() > 0
    f, g, n_atoms, n_out = pf.pocket_descriptor((prot_xyz, w, None), lig_xyz, order=12, radius=12.0, cutoff=8.0)
    assert f.shape == (7, 49) and g.shape == (21, 49)
    assert n_atoms == mask.sum()
    centroid = lig_xyz.mean(axis=0)
    assert n_out == int((np.linalg.norm(prot_xyz[mask] - centroid, axis=1) > 12.0).sum()) > 0
    assert np.all(f[1:] == 0) and f[0, 0] > 0   # only the occupancy channel carries weight
    assert pf.pocket_atoms(prot_xyz, np.zeros((0, 3))).sum() == 0


F00_PER_UNIT_WEIGHT = 3.0 / (4.0 * np.pi) / np.sqrt(4.0 * np.pi)   # F_00 = |sum w| * (3/4pi) * Y_00


def test_sphere_mode_mask_and_no_cropping():
    rng = np.random.default_rng(0)
    prot_xyz = rng.uniform(-20, 20, size=(4000, 3))
    w = np.zeros((4000, 7))
    w[:, pf.OCC] = 1
    lig_xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [9.0, 0.0, 0.0]])
    centroid = lig_xyz.mean(axis=0)
    dist = np.linalg.norm(prot_xyz - centroid, axis=1)
    f, g, n_atoms, n_out = pf.pocket_descriptor((prot_xyz, w, None), lig_xyz, order=12, radius=10.0,
                                                mode="sphere")
    assert n_atoms == (dist <= 10.0).sum() > 0      # every heavy atom within 10 A of the centroid
    assert n_out == 0                               # descriptor radius = sphere radius: nothing cropped
    assert f[pf.OCC, 0] == pytest.approx(n_atoms * F00_PER_UNIT_WEIGHT, rel=1e-12)
    # the 8 A shell around this elongated ligand reaches past 12 A and is cropped by the descriptor radius
    _, _, n_shell, n_shell_out = pf.pocket_descriptor((prot_xyz, w, None), lig_xyz, radius=12.0, cutoff=8.0)
    assert n_shell_out > 0 and n_shell != n_atoms
    # atoms on the sphere surface (rounding either way) are never masked in and then dropped
    surface = centroid + 10.0 * np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0], [0.6, 0.8, 0], [0, 0.28, 0.96]])
    xyz2 = np.vstack([prot_xyz, surface])
    w2 = np.vstack([w, np.tile(w[:1], (len(surface), 1))])
    f2, _, n2, n2_out = pf.pocket_descriptor((xyz2, w2, None), lig_xyz, radius=10.0, mode="sphere")
    assert n2_out == 0 and n2 >= n_atoms + 3
    assert f2[pf.OCC, 0] == pytest.approx(n2 * F00_PER_UNIT_WEIGHT, rel=1e-12)
    with pytest.raises(ValueError):
        pf.pocket_descriptor((prot_xyz, w, None), lig_xyz, mode="cube")


def test_channel_totals_keep_the_sign_f_drops():
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-8, 8, size=(300, 3))
    w = np.zeros((300, 7))
    w[:, pf.OCC] = 1
    w[:, pf.HYD] = rng.normal(-0.2, 0.3, size=300)     # net hydrophobicity negative
    w[:, pf.POS] = (rng.uniform(size=300) < 0.05)
    lig_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    f, _, n_atoms, n_out, total = pf.pocket_descriptor((xyz, w, None), lig_xyz, radius=12.0, cutoff=8.0,
                                                       return_total=True)
    mask = pf.pocket_atoms(xyz, lig_xyz, 8.0)
    np.testing.assert_allclose(total, w[mask].sum(axis=0), atol=1e-12)   # no atom beyond 12 A here
    assert total[pf.HYD] < 0 < f[pf.HYD, 0] and total[pf.OCC] == n_atoms
    np.testing.assert_allclose(np.abs(total) * F00_PER_UNIT_WEIGHT, f[:, 0], rtol=1e-10, atol=1e-14)
    # totals cover only the atoms the descriptor keeps
    small = pf.channel_totals(xyz, w, lig_xyz.mean(axis=0), 4.0)
    keep = np.linalg.norm(xyz - lig_xyz.mean(axis=0), axis=1) <= 4.0
    np.testing.assert_allclose(small, w[keep].sum(axis=0), atol=1e-12)
    assert pf.channel_totals(np.zeros((0, 3)), np.zeros((0, 7)), np.zeros(3), 12.0).shape == (7,)
    mol = _mol3d("CC(=O)Nc1ccc(O)cc1")
    _, _, n_lig, _, lig_total = pf.ligand_descriptor(mol, return_total=True)
    assert lig_total[pf.OCC] == n_lig and lig_total[pf.HYD] == pytest.approx(Crippen.MolLogP(mol), abs=1e-6)


# ---- featurise_zernike.py end to end on a synthetic parent layout (path rules, untrusted rows, outputs)

ACETATE_MOL2 = """@<TRIPOS>MOLECULE
lig
 7 6 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1         {x0:8.4f}    0.0000    0.0000 C.3     1  LIG1        0.0000
      2 C2         {x1:8.4f}    0.0000    0.0000 C.2     1  LIG1        0.0000
      3 O1         {x2:8.4f}    1.1000    0.0000 O.co2   1  LIG1       -0.5000
      4 O2         {x2:8.4f}   -1.1000    0.0000 O.co2   1  LIG1       -0.5000
      5 H1         {xh:8.4f}    1.0000    0.0000 H       1  LIG1        0.0000
      6 H2         {xh:8.4f}   -0.5000    0.8700 H       1  LIG1        0.0000
      7 H3         {xh:8.4f}   -0.5000   -0.8700 H       1  LIG1        0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     2     3   ar
     3     2     4   ar
     4     1     5    1
     5     1     6    1
     6     1     7    1
"""


def _write_mol2(path, x0=3.0):
    path.write_text(ACETATE_MOL2.format(x0=x0, x1=x0 + 1.5, x2=x0 + 2.1, xh=x0 - 0.4))


def _parent_layout(tmp_path, far=False):
    """far=True adds BindingDB row bd3 whose ligand sits 100 A from the protein (empty pocket)."""
    data = tmp_path / "parent"
    data.mkdir()
    pd.DataFrame({"PDB_code": ["1abc", "2xyz"], "refined": ["True", "False"]}).to_csv(data / "pdbbind_processed.csv")
    pd.DataFrame({"unique_identify": ["bn1"], "pdb": ["1abc"], "target": ["101"], "compnd": ["CHEMBL5"]}).to_csv(
        data / "bindingnet_processed.csv")
    bdb = {"unique_id": ["bd1", "bd2"], "folder": ["3def_A", "3def_A"],
           "pdb_file": ["3def_protein.pdb", "3def_protein.pdb"],
           "mol2_file": ["results_7.mol2", "../../../etc/passwd"]}
    if far:
        for k, v in (("unique_id", "bd3"), ("folder", "3def_A"), ("pdb_file", "3def_protein.pdb"),
                     ("mol2_file", "results_9.mol2")):
            bdb[k].append(v)
    pd.DataFrame(bdb).to_csv(data / "bindingdb_processed.csv")
    meta = pd.DataFrame({"unique_id": ["1abc", "2xyz", "1abc::pos0", "bn1", "bd1", "bd2"] + (["bd3"] if far else []),
                         "source": ["pdbbind", "pdbbind", "pdbbind", "bindingnet", "bindingdb", "bindingdb"]
                         + (["bindingdb"] if far else []),
                         "split": "train", "kind": ["0", "0", "1", "0", "0", "0"] + (["0"] if far else [])})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    pdbbind = tmp_path / "pdbbind" / "refined-set" / "1abc"
    pdbbind.mkdir(parents=True)
    _write_pdb(pdbbind / "1abc_protein.pdb", LYS)
    _write_mol2(pdbbind / "1abc_ligand.mol2")
    (tmp_path / "pdbbind" / "general-set").mkdir()        # 2xyz files are missing on purpose
    bn = tmp_path / "bindingnet" / "1abc"
    (bn / "target_101" / "CHEMBL5").mkdir(parents=True)
    _write_pdb(bn / "rec_h_opt.pdb", LYS)
    mol = _mol3d("CC(=O)[O-]")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 4.0, p.y, p.z))
    writer = Chem.SDWriter(str(bn / "target_101" / "CHEMBL5" / "1abc_101_CHEMBL5.sdf"))
    writer.write(mol)
    writer.close()
    bd = tmp_path / "bindingdb" / "3def_A"
    bd.mkdir(parents=True)
    _write_pdb(bd / "3def_protein.pdb", LYS)
    _write_mol2(bd / "results_7.mol2", x0=2.0)
    if far:
        _write_mol2(bd / "results_9.mol2", x0=100.0)
    return data


def _roots(tmp_path):
    return ",".join(f"{s}={tmp_path / s}" for s in ("pdbbind", "bindingnet", "bindingdb"))


def test_safe_join_rejects_traversal():
    import featurise_zernike as fz
    assert fz.safe_join("/r", "a", "b/c.pdb") == "/r/a/b/c.pdb"
    for bad in ("../x", "/etc/passwd", "a/../../x", "", "a\x00b"):
        with pytest.raises(ValueError):
            fz.safe_join("/r", bad)


def test_featurise_zernike_end_to_end(tmp_path):
    import json
    import featurise_zernike as fz
    data = _parent_layout(tmp_path)
    out = tmp_path / "out" / "zernike.npz"
    fz.main(["--meta", str(tmp_path / "meta.csv"), "--parent-data", str(data), "--roots", _roots(tmp_path),
             "--out", str(out), "--workers", "1"])
    z = np.load(out, allow_pickle=False)
    assert z["unique_id"].tolist() == ["1abc", "2xyz", "bn1", "bd1", "bd2"]   # meta order, pose row skipped
    ok = dict(zip(z["unique_id"].tolist(), z["ok"].tolist()))
    err = dict(zip(z["unique_id"].tolist(), z["error"].tolist()))
    assert ok == {"1abc": True, "2xyz": False, "bn1": True, "bd1": True, "bd2": False}
    assert err["2xyz"].startswith("FileNotFoundError") and "unsafe path" in err["bd2"]
    assert z["pocket_F"].shape == (5, 7, 49) and z["pocket_G"].shape == (5, 21, 49)
    assert z["ligand_F"].dtype == np.float32
    assert np.isnan(z["pocket_F"][1]).all() and np.isfinite(z["pocket_F"][0]).all()
    assert z["n_ligand_atoms"].tolist()[:1] == [4] and z["n_pocket_atoms"][0] > 0
    # the same acetate mol2 at another position: the ligand descriptor is translation invariant
    np.testing.assert_allclose(z["ligand_F"][0], z["ligand_F"][3], atol=1e-5)
    summary = json.loads((tmp_path / "out" / "zernike_summary.json").read_text())
    assert summary["skipped_pose_rows"] == 1 and summary["ok"] == 3 and summary["failed"] == 2
    assert set(summary["per_source"]) == {"pdbbind", "bindingnet", "bindingdb"}
    # default shell run: parameters as npz scalars and in the summary
    assert str(z["pocket_mode"]) == "shell" and int(z["order"]) == 12 and float(z["radius"]) == 12.0
    assert np.isnan(z["pocket_radius"]) and float(z["cutoff"]) == 8.0 and float(z["ligand_radius"]) == 12.0
    assert z["pocket_mode"].ndim == 0 and str(z["variant_note"]) == ""
    params = summary["params"]
    assert (params["pocket_mode"], params["pocket_radius"], params["radius"], params["order"]) == ("shell", None, 12.0, 12)


def test_featurise_zernike_failed_rows_are_nan(tmp_path):
    """Every failed row, including one that fails after its ligand was read (empty pocket), stores NaN
    descriptors and totals and -1 counts, so finiteness and `ok` agree."""
    import featurise_zernike as fz
    data = _parent_layout(tmp_path, far=True)
    out = tmp_path / "zernike.npz"
    fz.main(["--meta", str(tmp_path / "meta.csv"), "--parent-data", str(data), "--roots", _roots(tmp_path),
             "--out", str(out), "--workers", "1"])
    z = np.load(out, allow_pickle=False)
    uid = z["unique_id"].tolist()
    i = uid.index("bd3")
    assert not z["ok"][i] and z["error"][i].startswith("ValueError: no protein heavy atom within 8.0 A")
    for name in ("pocket_F", "pocket_G", "ligand_F", "ligand_G", "pocket_total", "ligand_total"):
        assert np.isnan(z[name][~z["ok"]]).all(), name
        assert np.isfinite(z[name][z["ok"]]).all(), name
    for name in ("n_pocket_atoms", "n_ligand_atoms", "n_outside_pocket", "n_outside_ligand"):
        assert (z[name][~z["ok"]] == -1).all(), name
    assert (~z["ok"]).sum() == 3      # 2xyz (missing files), bd2 (unsafe path), bd3 (empty pocket)
    # the same row also fails in sphere mode, with the sphere message
    out2 = tmp_path / "zs.npz"
    fz.main(["--meta", str(tmp_path / "meta.csv"), "--parent-data", str(data), "--roots", _roots(tmp_path),
             "--out", str(out2), "--workers", "1", "--pocket-mode", "sphere"])
    z2 = np.load(out2, allow_pickle=False)
    assert z2["error"][i].startswith("ValueError: no protein heavy atom within 10.0 A of the ligand centroid")
    assert np.isnan(z2["pocket_total"][i]).all() and z2["n_pocket_atoms"][i] == -1


def test_featurise_zernike_signed_totals(tmp_path):
    import featurise_zernike as fz
    data = _parent_layout(tmp_path)
    out = tmp_path / "zernike.npz"
    fz.main(["--meta", str(tmp_path / "meta.csv"), "--parent-data", str(data), "--roots", _roots(tmp_path),
             "--out", str(out), "--workers", "1"])
    z = np.load(out, allow_pickle=False)
    ok = z["ok"]
    assert z["pocket_total"].shape == (5, 7) and z["ligand_total"].dtype == np.float32
    for side in ("pocket", "ligand"):
        total, f = z[f"{side}_total"][ok], z[f"{side}_F"][ok]
        kept = z[f"n_{side}_atoms"][ok] - z[f"n_outside_{side}"][ok]
        np.testing.assert_allclose(total[:, pf.OCC], kept)
        np.testing.assert_allclose(np.abs(total) * F00_PER_UNIT_WEIGHT, f[:, :, 0], rtol=1e-5, atol=1e-7)
    # 1abc: the whole LYS residue is in the pocket; its Crippen sum is negative, which F_00 cannot show
    lys_hyd = sum(pf.CRIPPEN_TABLE["LYS"][a] for a in ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"))
    assert lys_hyd < 0
    np.testing.assert_allclose(z["pocket_total"][0, [pf.OCC, pf.HYD, pf.POS, pf.NEG]], [9, lys_hyd, 1, 0],
                               rtol=1e-6)
    assert z["pocket_F"][0, pf.HYD, 0] > 0
    _, lig_w = pf.ligand_fields(fz.read_ligand(("mol2", str(tmp_path / "pdbbind" / "refined-set" / "1abc" /
                                                            "1abc_ligand.mol2"))))
    np.testing.assert_allclose(z["ligand_total"][0], lig_w.sum(axis=0), rtol=1e-6, atol=1e-6)
    assert z["ligand_total"][0, pf.POS] - z["ligand_total"][0, pf.NEG] == pytest.approx(-1.0, abs=1e-5)


def test_featurise_zernike_sphere_mode(tmp_path):
    import json
    import featurise_zernike as fz
    data = _parent_layout(tmp_path)
    out = tmp_path / "out" / "zs.npz"
    fz.main(["--meta", str(tmp_path / "meta.csv"), "--parent-data", str(data), "--roots", _roots(tmp_path),
             "--out", str(out), "--workers", "1", "--pocket-mode", "sphere", "--variant-note", "A/B sphere 10"])
    z = np.load(out, allow_pickle=False)
    ok = z["ok"]
    assert ok.tolist() == [True, False, True, True, False]
    assert str(z["pocket_mode"]) == "sphere" and float(z["pocket_radius"]) == 10.0 == float(z["radius"])
    assert int(z["order"]) == 12 and np.isnan(z["cutoff"]) and float(z["ligand_radius"]) == 12.0
    assert str(z["variant_note"]) == "A/B sphere 10"
    assert (z["n_outside_pocket"][ok] == 0).all() and (z["n_pocket_atoms"][ok] > 0).all()
    summary = json.loads((tmp_path / "out" / "zs_summary.json").read_text())
    p = summary["params"]
    assert (p["pocket_mode"], p["pocket_radius"], p["radius"], p["cutoff"], p["order"]) == ("sphere", 10.0, 10.0, None, 12)
    assert summary["variant_note"] == "A/B sphere 10"
    # the sphere radius sets the descriptor radius; the ligand keeps its own
    args = fz.parse_args(["--fep-dir", "x", "--out", "o.npz", "--pocket-mode", "sphere", "--pocket-radius", "9",
                          "--ligand-radius", "11"])
    assert (args.radius, args.pocket_radius, args.ligand_radius, args.cutoff) == (9.0, 9.0, 11.0, None)
    for bad in (["--pocket-radius", "10"], ["--pocket-mode", "sphere", "--radius", "12"],
                ["--pocket-mode", "sphere", "--cutoff", "8"], ["--pocket-mode", "ball"], ["--ligand-radius", "0"]):
        with pytest.raises(SystemExit):
            fz.parse_args(["--fep-dir", "x", "--out", "o.npz"] + bad)


def test_featurise_zernike_fep_mode_dedupes_titles(tmp_path):
    import featurise_zernike as fz
    sdir = tmp_path / "fep_benchmark_inputs" / "structure_inputs" / "subsetA"
    sdir.mkdir(parents=True)
    _write_pdb(sdir / "ser1_protein.pdb", LYS)
    writer = Chem.SDWriter(str(sdir / "ser1_ligands.sdf"))
    for title, smi in (("a", "CC(=O)[O-]"), ("b", "c1ccccc1O"), ("a", "CC(=O)O")):
        mol = _mol3d(smi)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + 4.0, p.y - 2.0, p.z + 2.0))
        mol.SetProp("_Name", title)
        writer.write(mol)
    writer.close()
    pd.DataFrame({"Group abbreviation": ["subsetA"], "Input file naming scheme": ["ser1"], "Protein": ["LysX"],
                  "Reference PDB": ["9ZZZ"]}).to_csv(tmp_path / "fep_benchmark_inputs" / "benchmark_metadata.csv",
                                                     index=False)
    out = tmp_path / "zf.npz"
    fz.main(["--fep-dir", str(tmp_path), "--out", str(out), "--workers", "1"])
    z = np.load(out, allow_pickle=False)
    assert z["unique_id"].tolist() == ["subsetA/ser1/a", "subsetA/ser1/b"]
    assert z["ok"].all()
    assert z["n_ligand_atoms"].tolist() == [4, 7]                               # first "a" entry kept
    meta = pd.read_csv(tmp_path / "zf_meta.csv", keep_default_na=False)
    assert meta["uniprot"].tolist() == ["LysX", "LysX"] and meta["series_key"].iloc[0] == "subsetA/ser1"


def _shifted(mol, dx=4.0, dy=-2.0, dz=2.0):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + dx, p.y + dy, p.z + dz))
    return mol


def _write_prep_sdf(path, mol, title):
    """One ligand written as fep_prepare.write_single_sdf does: the entry read with removeHs=False,
    sanitize=False, written without kekulisation."""
    mol.SetProp("_Name", title)
    tmp = str(path) + ".src"
    w = Chem.SDWriter(tmp)
    w.write(mol)
    w.close()
    raw = next(iter(Chem.SDMolSupplier(tmp, removeHs=False, sanitize=False)))
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    writer.write(raw)
    writer.close()
    os.remove(tmp)


def test_featurise_zernike_csv_mode(tmp_path):
    """--csv: one row per CSV row in CSV order (absolute and --work-dir-relative paths), the training layout and
    shell parameters, unsafe / missing paths recorded as failed rows; a prep-written sdf gives the descriptor of the
    training readers (sdf_first, --fep-dir) for the same molecule, with or without explicit hydrogens."""
    import json
    import featurise_zernike as fz

    work = tmp_path / "work"
    sdir = work / "data" / "fep_structures"
    sdir.mkdir(parents=True)
    _write_pdb(sdir / "ser1_protein.pdb", LYS)
    acetate = _shifted(_mol3d("CC(=O)[O-]"))
    phenol = _shifted(_mol3d("c1ccccc1O"))
    _write_prep_sdf(sdir / "a.sdf", Chem.Mol(acetate), "a")
    _write_prep_sdf(sdir / "b.sdf", Chem.Mol(phenol), "b")
    _write_prep_sdf(sdir / "b_noh.sdf", Chem.RemoveHs(Chem.Mol(phenol)), "b")  # heavy atoms only (pgdh-like input)
    rel = os.path.join("data", "fep_structures")
    table = pd.DataFrame({
        "unique_id": ["s/ser1/b", "s/ser1/a", "s/ser1/b_noh", "s/ser1/gone", "s/ser1/evil"],
        "pK": [6.0, 7.0, 6.0, 5.0, 5.0],
        "sdf_file": [str(sdir / "b.sdf"), os.path.join(rel, "a.sdf"), os.path.join(rel, "b_noh.sdf"),
                     os.path.join(rel, "missing.sdf"), "data/../../etc/passwd"],
        "pdb_file": [str(sdir / "ser1_protein.pdb")] + [os.path.join(rel, "ser1_protein.pdb")] * 4})
    csv = work / "data" / "fep_benchmark.csv"
    table.to_csv(csv, index=False)
    out = work / "data" / "zernike_fep.npz"
    fz.main(["--csv", str(csv), "--work-dir", str(work), "--out", str(out), "--workers", "1"])
    z = np.load(out, allow_pickle=False)
    assert z["unique_id"].tolist() == table["unique_id"].tolist()
    assert z["ok"].tolist() == [True, True, True, False, False]
    assert z["error"][3].startswith("FileNotFoundError") and "unsafe path" in z["error"][4]
    assert z["pocket_F"].shape == (5, 7, 49) and z["ligand_F"].shape == (5, 7, 49)
    assert z["pocket_total"].shape == (5, 7) and z["ligand_total"].dtype == np.float32
    assert set(z["source"].tolist()) == {"fep_benchmark"}
    assert str(z["pocket_mode"]) == "shell" and int(z["order"]) == 12 and float(z["radius"]) == 12.0
    assert float(z["cutoff"]) == 8.0 and float(z["ligand_radius"]) == 12.0
    assert np.isnan(z["pocket_F"][3:]).all() and np.isfinite(z["pocket_F"][:3]).all()
    assert z["n_ligand_atoms"].tolist()[:3] == [7, 4, 7]
    # without explicit hydrogens: the same united-atom fields, so the same descriptor
    np.testing.assert_allclose(z["ligand_F"][2], z["ligand_F"][0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(z["pocket_F"][2], z["pocket_F"][0], rtol=1e-5, atol=1e-6)
    # the training readers on the same molecules: sdf_first (BindingNet) and --fep-dir mode
    for k, name in ((0, "b.sdf"), (1, "a.sdf")):
        mol = fz.read_ligand(("sdf_first", str(sdir / name)))
        xyz, w = pf.ligand_fields(mol)
        np.testing.assert_allclose(z["ligand_total"][k], w.sum(axis=0), rtol=1e-6, atol=1e-6)
    fep = tmp_path / "bench" / "fep_benchmark_inputs" / "structure_inputs" / "s"
    fep.mkdir(parents=True)
    _write_pdb(fep / "ser1_protein.pdb", LYS)
    writer = Chem.SDWriter(str(fep / "ser1_ligands.sdf"))
    for title, mol in (("a", acetate), ("b", phenol)):
        m = Chem.Mol(mol)
        m.SetProp("_Name", title)
        writer.write(m)
    writer.close()
    fz.main(["--fep-dir", str(tmp_path / "bench"), "--out", str(tmp_path / "zfep.npz"), "--workers", "1"])
    zf = np.load(tmp_path / "zfep.npz", allow_pickle=False)
    assert zf["unique_id"].tolist() == ["s/ser1/a", "s/ser1/b"]
    for name in ("ligand_F", "pocket_F", "ligand_total", "pocket_total"):
        np.testing.assert_allclose(z[name][[1, 0]], zf[name], rtol=1e-5, atol=1e-6, err_msg=name)
    summary = json.loads((work / "data" / "zernike_fep_summary.json").read_text())
    assert summary["input"] == "csv" and summary["ok"] == 3 and summary["failed"] == 2 and summary["csv_rows"] == 5
    # relative paths resolve against the CWD without --work-dir
    cwd = os.getcwd()
    try:
        os.chdir(work)
        fz.main(["--csv", str(csv), "--out", str(tmp_path / "z2.npz"), "--workers", "1"])
    finally:
        os.chdir(cwd)
    np.testing.assert_array_equal(np.load(tmp_path / "z2.npz")["ok"], z["ok"])
    # a duplicate unique_id is an error; --csv is an alternative to --fep-dir / --meta
    table.iloc[[0, 0]].to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(SystemExit):
        fz.main(["--csv", str(tmp_path / "dup.csv"), "--out", str(tmp_path / "d.npz"), "--workers", "1"])
    for bad in (["--fep-dir", "x"], ["--meta", "m.csv", "--parent-data", "p"]):
        with pytest.raises(SystemExit):
            fz.parse_args(["--csv", str(csv), "--out", "o.npz", *bad])
    with pytest.raises(SystemExit):
        fz.parse_args(["--fep-dir", "x", "--out", "o.npz", "--work-dir", "w"])
