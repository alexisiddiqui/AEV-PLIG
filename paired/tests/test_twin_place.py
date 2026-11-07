"""Tests for src/twin_place.py: O3A overlay of ETKDG conformers onto the crystal ligand, clash relief and
the arm-4 acceptance, on RDKit-built molecules inside a synthetic pocket shell."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import twin_place as tp
from pose_augment import heavy_centroid, heavy_xyz, min_dist, symm_rmsd

ANCHOR_SMILES = "Cc1ccc2ccccc2c1"      # 2-methylnaphthalene: rigid
TWIN_SMILES = "Cc1ccc2ncccc2c1"        # 6-methylquinoline: same shape, different chemistry


def built(smiles, seed=11):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def shell(mol, seed=0, n_dirs=10, gap=(3.5, 4.3), min_gap=3.4, spacing=1.6):
    """Protein-like heavy atoms hugging the ligand: points 3.5-4.3 A out from each ligand heavy atom, kept
    when >= 3.4 A from every ligand heavy atom and >= 1.6 A from each other."""
    rng = np.random.default_rng(seed)
    lig = heavy_xyz(mol)
    pts = []
    for x in lig:
        for _ in range(n_dirs):
            v = rng.normal(size=3)
            p = x + v / np.linalg.norm(v) * rng.uniform(*gap)
            if np.min(np.linalg.norm(lig - p, axis=1)) < min_gap:
                continue
            if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < spacing:
                continue
            pts.append(p)
    return np.array(pts)


@pytest.fixture(scope="module")
def anchor():
    return built(ANCHOR_SMILES)


@pytest.fixture(scope="module")
def pocket(anchor):
    return shell(anchor)


def test_overlay_puts_twin_on_the_anchor(anchor):
    twin = built(TWIN_SMILES, seed=5)
    # move the input twin far away: the overlay must not depend on its input coordinates
    conf = twin.GetConformer()
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, (p + [40.0, -25.0, 10.0]).tolist())
    mol, score = tp.overlay(anchor, twin, np.random.default_rng(1), n_confs=10)
    assert mol is not None and mol.GetNumConformers() == 1
    assert mol.GetNumAtoms() == twin.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.RemoveHs(twin))
    assert score > 0
    assert np.linalg.norm(heavy_centroid(mol) - heavy_centroid(anchor)) < 1.0
    # the best of the conformers: at least as good as a single-conformer overlay with the same seed
    _, score1 = tp.overlay(anchor, twin, np.random.default_rng(1), n_confs=1)
    assert score >= score1 - 1e-9


def test_place_twin_applies_relief_and_acceptance(anchor, pocket):
    twin = built(TWIN_SMILES, seed=5)
    res = tp.place_twin(anchor, twin, pocket, np.random.default_rng(2))
    assert res.ok and res.reason == ""
    crystal_md = min_dist(heavy_xyz(anchor), pocket)
    assert res.min_dist >= min(tp.CLASH, crystal_md) and res.contact_frac >= tp.CONTACT_KEEP
    assert res.min_dist == pytest.approx(min_dist(heavy_xyz(res.mol), pocket))
    assert 1 <= res.attempts <= tp.MAX_OVERLAYS and res.o3a_score > 0
    assert np.isnan(res.rmsd_to_crystal)
    assert res.centroid_shift < 1.0 + tp.RELIEF_CENTROID_MAX
    assert res.n_within_3A == int(np.sum(np.min(np.linalg.norm(
        heavy_xyz(res.mol)[:, None] - pocket[None], axis=2), axis=1) < 3.0))
    # deterministic given the rng
    again = tp.place_twin(anchor, twin, pocket, np.random.default_rng(2))
    assert np.array_equal(again.mol.GetConformer().GetPositions(), res.mol.GetConformer().GetPositions())


def test_relief_moves_a_clashing_overlay(anchor, pocket):
    # one protein atom 2.6 A from a ligand atom: the crystal pose itself sits inside the 3.0 A relief target
    lig = heavy_xyz(anchor)
    i = int(np.argmax(np.linalg.norm(lig - lig.mean(axis=0), axis=1)))
    out = (lig[i] - lig.mean(axis=0)) / np.linalg.norm(lig[i] - lig.mean(axis=0))
    tight = np.vstack([pocket, lig[i] + 2.6 * out])
    res = tp.regenerate_positive(anchor, np.random.default_rng(3), tight)
    assert res.ok
    assert res.relief_steps > 0
    assert res.min_dist >= min(tp.CLASH, min_dist(lig, tight))


def test_regenerate_positive_rigid_rmsd(anchor, pocket):
    res = tp.regenerate_positive(anchor, np.random.default_rng(4), pocket)
    assert res.ok
    assert res.rmsd_to_crystal < 2.0
    assert res.rmsd_to_crystal == pytest.approx(symm_rmsd(anchor, res.mol))
    assert Chem.MolToSmiles(Chem.RemoveHs(res.mol)) == Chem.MolToSmiles(Chem.RemoveHs(anchor))
    assert res.mol.GetNumAtoms() == anchor.GetNumAtoms()
    # the anchor's crystal conformer is not reused: the pose comes from an ETKDG conformer
    assert not np.allclose(res.mol.GetConformer().GetPositions(), anchor.GetConformer().GetPositions())


def test_flexible_positive_and_twin_share_the_path(pocket):
    # a flexible ligand: the regenerated positive goes through the same overlay + relief + acceptance
    anchor = built("OC(=O)CCc1ccc(O)cc1", seed=2)
    pk = shell(anchor, seed=1)
    pos = tp.regenerate_positive(anchor, np.random.default_rng(5), pk)
    twin = tp.place_twin(anchor, built("NC(=O)CCc1ccc(N)cc1", seed=3), pk, np.random.default_rng(5))
    for r in (pos, twin):
        assert set(vars(r)) == set(vars(pos))
        assert r.attempts >= 1
        if r.ok:
            assert r.min_dist >= min(tp.CLASH, min_dist(heavy_xyz(anchor), pk))


def test_rejection_is_recorded_not_raised(anchor, pocket):
    twin = built(TWIN_SMILES, seed=5)
    res = tp.place_twin(anchor, twin, pocket, np.random.default_rng(6), contact_keep=100.0)
    assert not res.ok and res.reason == "contacts" and res.attempts == tp.MAX_OVERLAYS
    assert res.mol is None
    res = tp.place_twin(anchor, twin, pocket, np.random.default_rng(6), max_overlays=2, contact_keep=100.0)
    assert res.attempts == 2


def test_embed_failure_is_recorded(anchor, pocket, monkeypatch):
    monkeypatch.setattr(tp.AllChem, "EmbedMultipleConfs", lambda *a, **k: [])
    res = tp.place_twin(anchor, built(TWIN_SMILES), pocket, np.random.default_rng(7))
    assert not res.ok and res.reason == "embed_failed" and res.attempts == tp.MAX_OVERLAYS
    mol, score = tp.overlay(anchor, built(TWIN_SMILES), np.random.default_rng(7))
    assert mol is None and np.isnan(score)
