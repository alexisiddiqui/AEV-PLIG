"""Tests for src/pose_augment.py (arm 4 pose generator). Ligands are built with RDKit; the
"protein" is a cloud of random points on a shell around the ligand."""
import zlib

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

import pose_augment as pa

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _mol3d(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _shell(mol, n=200, gap=2.5, width=4.0, seed=0):
    """n random points at distance r_max + gap .. r_max + gap + width from the heavy-atom centroid,
    so the crystal pose is clash-free whatever its size."""
    xyz = pa.heavy_xyz(mol)
    c = xyz.mean(axis=0)
    rmax = np.linalg.norm(xyz - c, axis=1).max()
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(rmax + gap, rmax + gap + width, size=n)
    return c + v * r[:, None]


def _rng(name="1abc"):
    return np.random.default_rng(zlib.crc32(name.encode()))


def _rotate_about(mol, axis, angle_deg, center):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.deg2rad(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    xyz = conf.GetPositions()
    new = (xyz - center) @ rot.T + center
    for i, p in enumerate(new):
        conf.SetAtomPosition(i, p.tolist())
    return out


def test_near_native_rmsd_and_no_clash():
    mol = _mol3d("CCCCCCO")
    target = _shell(mol)
    poses = pa.generate_poses(mol, target, k_pos=2, k_neg=0, rng=_rng())
    pos = [p for p in poses if p.kind == 1]
    assert [p.k for p in pos] == [0, 1]
    for p in pos:
        assert p.method == "nn"
        assert p.mol.GetNumAtoms() == mol.GetNumAtoms()  # Hs kept
        assert p.mol.GetNumConformers() == 1
        r = pa.symm_rmsd(mol, p.mol)
        assert 0.3 <= r <= 2.0
        assert abs(r - p.rmsd) < 1e-6
        md = pa.min_dist(pa.heavy_xyz(p.mol), target)
        assert md >= 2.2 and abs(md - p.min_dist) < 1e-6
        assert 1 <= p.attempts <= 60
    assert pa.symm_rmsd(pos[0].mol, pos[1].mol) >= 0.3


def test_decoy_rmsd_and_contacts():
    mol = _mol3d(IBUPROFEN)
    target = _shell(mol)
    n_cryst = pa.contacts(pa.heavy_xyz(mol), target)
    assert n_cryst > 0
    poses = pa.generate_poses(mol, target, k_pos=0, k_neg=2, rng=_rng())
    neg = [p for p in poses if p.kind == 2]
    assert [p.k for p in neg] == [0, 1]
    assert [p.method for p in neg] == ["flip", "conf"]
    for p in neg:
        xyz = pa.heavy_xyz(p.mol)
        r = pa.symm_rmsd(mol, p.mol)
        assert abs(r - p.rmsd) < 1e-6
        assert r >= (4.0 if p.attempts <= 60 else 3.0)
        assert pa.min_dist(xyz, target) >= 2.2
        frac = pa.contacts(xyz, target) / n_cryst
        assert frac >= 0.5 and abs(frac - p.contact_frac) < 1e-9
        assert p.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert pa.symm_rmsd(neg[0].mol, neg[1].mol) >= 0.3


def test_rmsd_is_symmetry_corrected():
    benzene = _mol3d("c1ccccc1")
    heavy = pa.heavy_xyz(benzene)
    c = heavy.mean(axis=0)
    normal = np.linalg.svd(heavy - c)[2][2]
    turned = _rotate_about(benzene, normal, 60.0, c)
    assert pa.symm_rmsd(benzene, turned) < 0.05
    naive = np.sqrt(((pa.heavy_xyz(turned) - heavy) ** 2).sum(axis=1).mean())
    assert naive > 1.0  # the test is meaningful: index-wise RMSD is large


def test_generator_handles_rigid_small_ligand():
    for smiles in ("C", "c1ccccc1"):
        mol = _mol3d(smiles)
        assert pa.rotatable_torsions(mol) == []
        target = _shell(mol, gap=3.0)
        poses = pa.generate_poses(mol, target, k_pos=2, k_neg=2, rng=_rng(smiles))
        pos = [p for p in poses if p.kind == 1]
        neg = [p for p in poses if p.kind == 2]
        assert len(pos) >= 1, smiles
        for p in pos:
            assert 0.3 <= p.rmsd <= 2.0
        for p in neg:  # fall back to 3.0 A or fail, never raise
            assert p.rmsd >= (4.0 if p.attempts <= 60 else 3.0)
            assert p.attempts <= 120


def test_clash_threshold_relaxes_to_crystal():
    mol = _mol3d(IBUPROFEN)
    ring = mol.GetSubstructMatch(Chem.MolFromSmarts("c1ccccc1"))
    ring_centre = mol.GetConformer().GetPositions()[list(ring)].mean(axis=0)
    target = np.vstack([_shell(mol), ring_centre[None, :]])
    cryst = pa.min_dist(pa.heavy_xyz(mol), target)
    assert cryst < 2.2
    assert pa.effective_clash(2.2, cryst) == cryst
    assert pa.effective_clash(2.2, 3.1) == 2.2
    poses = pa.generate_poses(mol, target, k_pos=2, k_neg=2, rng=_rng("relax"))
    assert len([p for p in poses if p.kind == 1]) >= 1  # the complex is not silently skipped
    for p in poses:
        assert p.min_dist >= cryst - 1e-9
    assert any(p.min_dist < 2.2 for p in poses)  # accepted only under the relaxed threshold


def test_deterministic_given_rng():
    mol = _mol3d(IBUPROFEN)
    target = _shell(mol)
    a = pa.generate_poses(mol, target, 2, 2, _rng("2xyz"))
    b = pa.generate_poses(mol, target, 2, 2, _rng("2xyz"))
    assert len(a) == len(b) == 4
    for p, q in zip(a, b):
        assert (p.kind, p.k, p.method, p.attempts) == (q.kind, q.k, q.method, q.attempts)
        assert np.array_equal(p.mol.GetConformer().GetPositions(), q.mol.GetConformer().GetPositions())
        assert p.rmsd == q.rmsd
    c = pa.generate_poses(mol, target, 2, 2, _rng("3xyz"))
    assert not np.array_equal(a[0].mol.GetConformer().GetPositions(),
                              c[0].mol.GetConformer().GetPositions())


def test_torsion_perturb_keeps_bonds_and_moves_dihedrals():
    mol = _mol3d(IBUPROFEN)
    tors = pa.rotatable_torsions(mol)
    assert len(tors) == 4  # heavy-atom rotatable bonds of ibuprofen (C-OH, C-CH3 are not)
    out = pa.torsion_perturb(mol, np.random.default_rng(0), sd_deg=30.0)
    a, b = mol.GetConformer().GetPositions(), out.GetConformer().GetPositions()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        assert abs(np.linalg.norm(a[i] - a[j]) - np.linalg.norm(b[i] - b[j])) < 1e-6
    assert pa.symm_rmsd(mol, out) > 0.05


def test_random_conformer_centroid():
    mol = _mol3d(IBUPROFEN)
    conf = pa.random_conformer(mol, np.random.default_rng(1))
    assert conf.GetNumConformers() == 1 and conf.GetNumAtoms() == mol.GetNumAtoms()
    assert np.allclose(pa.heavy_xyz(conf).mean(axis=0), pa.heavy_xyz(mol).mean(axis=0), atol=1e-6)


def _pocket(mol, lo=3.5, hi=4.5, spacing=1.0):
    """Enclosed synthetic pocket: grid points whose distance to the nearest ligand heavy atom is
    in [lo, hi] A (a dense shell shaped like the crystal pose)."""
    xyz = pa.heavy_xyz(mol)
    a, b = xyz.min(axis=0) - hi - 1.0, xyz.max(axis=0) + hi + 1.0
    axes = [np.arange(a[i], b[i], spacing) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d = cKDTree(xyz).query(grid)[0]
    return grid[(d >= lo) & (d <= hi)]


def test_relief_resolves_clash():
    mol = _mol3d(IBUPROFEN)
    xyz = pa.heavy_xyz(mol)
    c = xyz.mean(axis=0)
    axes = np.linalg.svd(xyz - c)[2]  # principal axes: long, middle, normal
    cand = _rotate_about(mol, axes[2], 180.0, c)  # end-to-end flip: a decoy-like placement
    assert pa.symm_rmsd(mol, cand) >= 4.0
    # a wall of protein points 0.6 A beyond the candidate's outermost atoms along the middle axis
    off = ((pa.heavy_xyz(cand) - c) @ axes[1]).max() + 0.6
    ticks = np.arange(-12.0, 12.01, 1.0)
    wall = np.array([c + off * axes[1] + a * axes[0] + b * axes[2] for a in ticks for b in ticks])
    assert pa.min_dist(pa.heavy_xyz(cand), wall) < 1.0
    assert pa.clash_score(pa.heavy_xyz(cand), wall, 2.2) > 0.0
    out, steps, ok = pa.relieve_clash(cand, wall, np.random.default_rng(0), 2.2, c, max_steps=300)
    assert ok and 1 <= steps <= 300
    hx = pa.heavy_xyz(out)
    assert pa.min_dist(hx, wall) >= 2.2 and pa.clash_score(hx, wall, 2.2) == 0.0
    assert pa.symm_rmsd(mol, out) >= 4.0
    assert np.linalg.norm(hx.mean(axis=0) - c) <= 2.5 + 1e-9
    # rigid-body moves only: all interatomic distances (Hs too) are kept
    a, b = cand.GetConformer().GetPositions(), out.GetConformer().GetPositions()
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    assert np.allclose(da, db, atol=1e-6)
    # deterministic given rng
    again, steps2, _ = pa.relieve_clash(cand, wall, np.random.default_rng(0), 2.2, c, max_steps=300)
    assert steps2 == steps and np.array_equal(again.GetConformer().GetPositions(), b)
    # an unresolvable clash (ligand buried in a filled box) spends the budget and reports failure
    box = _pocket(mol, lo=0.0, hi=6.0)
    _, steps3, ok3 = pa.relieve_clash(cand, box, np.random.default_rng(0), 2.2, c, max_steps=50)
    assert not ok3 and steps3 == 50


def test_decoys_found_in_buried_synthetic_pocket():
    mol = _mol3d(IBUPROFEN)
    pocket = _pocket(mol)
    assert pa.min_dist(pa.heavy_xyz(mol), pocket) >= 3.5  # crystal pose fits, clash_eff = 2.2
    n_cryst = pa.contacts(pa.heavy_xyz(mol), pocket)
    poses = pa.generate_poses(mol, pocket, k_pos=2, k_neg=2, rng=_rng("pocket"))
    neg = [p for p in poses if p.kind == 2]
    assert [p.k for p in neg] == [0, 1]
    for p in neg:
        xyz = pa.heavy_xyz(p.mol)
        assert abs(pa.symm_rmsd(mol, p.mol) - p.rmsd) < 1e-6
        assert p.rmsd >= (4.0 if p.attempts <= 60 else 3.0)
        assert pa.min_dist(xyz, pocket) >= 2.2 and abs(pa.min_dist(xyz, pocket) - p.min_dist) < 1e-9
        assert pa.contacts(xyz, pocket) / n_cryst >= 0.5
        assert p.relief_steps >= 1  # every placement clashed in the enclosed pocket
    assert pa.symm_rmsd(neg[0].mol, neg[1].mol) >= 0.3
    # without relief the same pocket yields fewer decoys (what the relief step is for)
    plain = pa.generate_poses(mol, pocket, k_pos=0, k_neg=2, rng=_rng("pocket"), relief_steps_max=0)
    assert len(plain) < 2


def test_near_natives_get_relief():
    mol = _mol3d(IBUPROFEN)
    pocket = _pocket(mol, lo=2.6, hi=3.6)  # crystal pose has atoms within 3.0 A of the "protein"
    cryst = pa.heavy_xyz(mol)
    assert 2.2 <= pa.min_dist(cryst, pocket) < 3.0 and pa.n_within(cryst, pocket) > 0
    poses = pa.generate_poses(mol, pocket, k_pos=2, k_neg=0, rng=_rng("nn-relief"))
    pos = [p for p in poses if p.kind == 1]
    assert [p.k for p in pos] == [0, 1]
    for p in pos:
        assert p.relief_steps >= 1  # every candidate started closer than the 3.0 A target
        assert 0.3 <= p.rmsd <= 2.0 and abs(pa.symm_rmsd(mol, p.mol) - p.rmsd) < 1e-6
        assert p.min_dist >= 2.2
        assert abs(pa.min_dist(pa.heavy_xyz(p.mol), pocket) - p.min_dist) < 1e-9
    # same construction path, relief off: no steps recorded
    plain = pa.generate_poses(mol, pocket, k_pos=2, k_neg=0, rng=_rng("nn-relief"), relief_steps_max=0)
    assert all(p.relief_steps == 0 for p in plain)
    # the relief target is a keyword: a 2.2 A target leaves already clash-free candidates alone
    low = pa.generate_poses(mol, pocket, k_pos=2, k_neg=0, rng=_rng("nn-relief"), relief_target_A=2.2)
    assert all(p.min_dist >= 2.2 for p in low)


def test_n_within_counts_ligand_atoms_closer_than_cutoff():
    lig = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0]])
    prot = np.array([[0.0, 2.9, 0.0], [0.0, 2.95, 0.1], [1.5, 3.1, 0.0], [10.0, 0.0, 3.0]])
    assert pa.n_within(lig, prot) == 1  # 2.9 counts; 3.1 and exactly 3.0 do not
    assert pa.n_within(lig, prot, cutoff=3.2) == 3
    assert pa.n_within(lig, np.empty((0, 3))) == 0
