"""Tests for src/ligand_edits.py (scheme C: in-place property edits of the cognate ligand).
Ligands are built with RDKit; the "protein" is a handful of typed atoms placed next to chosen groups
(channel weights from pocket_fields.protein_atom_weights)."""
import zlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import ligand_edits as le
import pose_augment as pa
from pocket_fields import protein_atom_weights


def _mol3d(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _rng(name="1abc"):
    return np.random.default_rng(zlib.crc32(name.encode()))


def _canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _smiles(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _n_sites(smiles, rule):
    return len([s for s in le.find_sites(_mol3d(smiles), rule) if not s.error])


def _fields(atoms):
    """protein_fields-style tuple from [(resname, atom, xyz)]."""
    xyz = np.array([a[2] for a in atoms], dtype=float).reshape(-1, 3)
    w = np.array([protein_atom_weights(a[0], a[1]) for a in atoms], dtype=float).reshape(-1, 7)
    return xyz, w, np.array([f"A:{a[0]}{i}" for i, a in enumerate(atoms)])


def _outward(mol, idx, dist):
    """A point `dist` A from atom idx, on the line from the heavy-atom centroid through the atom."""
    xyz = mol.GetConformer().GetPositions()
    c = pa.heavy_xyz(mol).mean(axis=0)
    v = xyz[idx] - c
    return xyz[idx] + v / np.linalg.norm(v) * dist


# ---------------------------------------------------------------- Step 1: rules

def test_rule_table_order_and_names():
    assert [r[0] for r in le.RULES] == ["cooh_to_me", "amine_to_me", "phenyl_to_chx", "oh_to_f",
                                       "nh_to_o", "amide_nh_to_nme"]
    assert all(len(r) == 3 and ":" in r[1] and ":" in r[2] for r in le.RULES)  # atom-mapped SMARTS


@pytest.mark.parametrize("rule,smiles,n", [
    ("cooh_to_me", "OC(=O)Cc1ccccc1", 1),
    ("cooh_to_me", "[O-]C(=O)CCC(=O)O", 2),
    ("cooh_to_me", "COC(=O)C", 0),            # ester
    ("cooh_to_me", "CC(N)=O", 0),             # amide
    ("cooh_to_me", "OC=O", 0),                # formic acid: no carbon to keep
    ("amine_to_me", "NCCc1ccccc1", 1),
    ("amine_to_me", "[NH3+]CCC", 1),
    ("amine_to_me", "Nc1ccccc1", 1),          # aniline is a primary amine on carbon
    ("amine_to_me", "CC(N)=O", 0),            # primary amide
    ("amine_to_me", "NC(N)=N", 0),            # guanidine
    ("amine_to_me", "CNC", 0),                # secondary amine
    ("amine_to_me", "NN", 0),                 # hydrazine: not on carbon
    ("phenyl_to_chx", "Cc1ccc(O)cc1", 1),
    ("phenyl_to_chx", "c1ccc(cc1)-c1ccccc1", 2),
    ("phenyl_to_chx", "c1ccc2ccccc2c1", 0),   # fused
    ("phenyl_to_chx", "Cc1ccc(C)c(C)c1", 0),  # three substituents
    ("phenyl_to_chx", "c1ccncc1", 0),         # not all-carbon
    ("oh_to_f", "OCCc1ccccc1", 1),
    ("oh_to_f", "Oc1ccccc1", 1),
    ("oh_to_f", "CC(=O)O", 0),                # carboxylic acid OH
    ("oh_to_f", "COC", 0),
    ("nh_to_o", "c1ccc2[nH]ccc2c1", 1),
    ("nh_to_o", "c1cc[nH]c1", 1),
    ("nh_to_o", "O=c1cccc[nH]1", 0),          # six-ring NH
    ("nh_to_o", "c1ccncc1", 0),
    ("amide_nh_to_nme", "CC(=O)NCc1ccccc1", 1),
    ("amide_nh_to_nme", "CC(=O)N(C)C", 0),    # tertiary amide
    ("amide_nh_to_nme", "CC(N)=O", 0),        # primary amide
])
def test_rules_match_expected_groups(rule, smiles, n):
    assert _n_sites(smiles, rule) == n


def test_sites_carry_edited_group():
    mol = _mol3d("OC(=O)Cc1ccc(O)cc1")
    by_rule = {s.rule: s for s in le.find_sites(mol)}
    assert set(by_rule) == {"cooh_to_me", "phenyl_to_chx", "oh_to_f"}
    syms = lambda s: sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in s.edited_atoms)
    assert syms(by_rule["cooh_to_me"]) == ["C", "O", "O"]          # carboxyl C re-typed, two O removed
    assert len(by_rule["cooh_to_me"].removed_atoms) == 2
    assert syms(by_rule["phenyl_to_chx"]) == ["C"] * 6
    assert syms(by_rule["oh_to_f"]) == ["O"]
    assert all(mol.GetAtomWithIdx(i).GetSymbol() != "H" for s in by_rule.values() for i in s.edited_atoms)


@pytest.mark.parametrize("rule,smiles,expected", [
    ("cooh_to_me", "OC(=O)Cc1ccccc1", "CCc1ccccc1"),
    ("amine_to_me", "NCCc1ccccc1", "CCCc1ccccc1"),
    ("amine_to_me", "[NH3+]CCc1ccccc1", "CCCc1ccccc1"),
    ("phenyl_to_chx", "Cc1ccc(O)cc1", "CC1CCC(O)CC1"),
    ("oh_to_f", "OCCc1ccccc1", "FCCc1ccccc1"),
    ("oh_to_f", "Cc1ccc(O)cc1", "Cc1ccc(F)cc1"),
    ("nh_to_o", "Cc1ccc2[nH]ccc2c1", "Cc1ccc2occc2c1"),
    ("amide_nh_to_nme", "CC(=O)NCc1ccccc1", "CC(=O)N(C)Cc1ccccc1"),
])
def test_edits_are_sanitisable_with_coords(rule, smiles, expected):
    mol = _mol3d(smiles)
    res = le.apply_edit(mol, rule, 0, _rng())
    out = res.mol
    Chem.SanitizeMol(Chem.Mol(out))                                   # raises if not sanitisable
    assert _smiles(out) == _canon(expected)
    assert out.GetNumConformers() == 1
    xyz = out.GetConformer().GetPositions()
    assert xyz.shape == (out.GetNumAtoms(), 3) and np.isfinite(xyz).all()
    assert Chem.AddHs(out).GetNumAtoms() == out.GetNumAtoms()        # every H explicit, with coordinates
    assert not np.any(np.all(np.abs(xyz) < 1e-9, axis=1))             # no atom left at the origin
    # unchanged heavy atoms keep their crystal coordinates (restrained MMFF)
    ref = mol.GetConformer().GetPositions()
    assert res.unchanged, "edit kept no unchanged heavy atom"
    d = [np.linalg.norm(xyz[i] - ref[j]) for i, j in res.unchanged]
    assert max(d) < 0.3
    assert res.mmff_restrained_rmsd == pytest.approx(np.sqrt(np.mean(np.square(d))), abs=1e-6)
    # every heavy atom has sensible bonded distances (new atoms placed, not stacked)
    for b in out.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1:
            assert 1.1 < np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) < 1.75


def test_formal_charge_updated():
    for smiles, rule, before, after in [("[O-]C(=O)Cc1ccccc1", "cooh_to_me", -1, 0),
                                        ("OC(=O)Cc1ccccc1", "cooh_to_me", 0, 0),
                                        ("[NH3+]CCc1ccccc1", "amine_to_me", 1, 0),
                                        ("NCCc1ccccc1", "amine_to_me", 0, 0),
                                        ("[NH3+]CCC(=O)[O-]", "cooh_to_me", 0, 1)]:
        mol = _mol3d(smiles)
        assert Chem.GetFormalCharge(mol) == before
        res = le.apply_edit(mol, rule, 0, _rng())
        assert Chem.GetFormalCharge(res.mol) == after == res.formal_charge
        assert res.dcharge == after - before
        assert all(a.GetFormalCharge() == 0 for a in res.mol.GetAtoms() if a.GetSymbol() == "C")


def test_phenyl_to_cyclohexyl_geometry():
    mol = _mol3d("CCc1ccc(O)cc1")
    site = le.find_sites(mol, "phenyl_to_chx")[0]
    res = le.apply_edit(mol, "phenyl_to_chx", 0, _rng())
    out = res.mol
    ring_in = sorted(site.edited_atoms)
    ring_out = [res.atom_map[i] for i in ring_in]
    assert all(not out.GetAtomWithIdx(i).GetIsAromatic() for i in ring_out)
    assert all(out.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP3 for i in ring_out)
    a = out.GetConformer().GetPositions()[ring_out]
    b = mol.GetConformer().GetPositions()[ring_in]
    assert np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) < 0.7
    # puckered (chair), not left planar: out-of-plane deviation of the ring atoms
    c = a - a.mean(axis=0)
    normal = np.linalg.svd(c)[2][-1]
    assert np.abs(c @ normal).max() > 0.15
    for i, j in zip(ring_out, ring_out[1:] + ring_out[:1]):
        if out.GetBondBetweenAtoms(i, j) is not None:
            assert 1.45 < np.linalg.norm(out.GetConformer().GetPositions()[i]
                                         - out.GetConformer().GetPositions()[j]) < 1.62


def test_null_edit_same_molecule_same_place():
    mol = _mol3d("CC(=O)Nc1ccc(O)cc1")
    res = le.null_edit(mol, _rng())
    assert _smiles(res.mol) == _smiles(mol)
    assert res.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert len(res.unchanged) == sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms())
    assert res.mmff_restrained_rmsd < 0.2 and res.dcharge == 0
    assert res.site.rule == "null" and res.site.edited_atoms == ()
    # deterministic given the rng
    res2 = le.null_edit(mol, _rng())
    assert np.allclose(res.mol.GetConformer().GetPositions(), res2.mol.GetConformer().GetPositions())


def test_unsanitisable_edit_is_a_recorded_failure():
    mol = _mol3d("CCC")
    bad = ("bad", "[CH2:1]", "[F:1]")                  # F with two bonds
    sites = le.find_sites(mol, bad)
    assert sites and sites[0].error == "not_sanitisable"
    with pytest.raises(le.EditFailure) as exc:
        le.apply_edit(mol, bad, 0, _rng())
    assert exc.value.reason == "not_sanitisable"


def test_apply_edit_with_target_relief_and_acceptance():
    mol = _mol3d("OCCc1ccccc1")
    target = np.array([_outward(mol, 0, 3.3)])           # one atom next to the hydroxyl O
    res = le.apply_edit(mol, "oh_to_f", 0, _rng(), target=target)
    md = pa.min_dist(pa.heavy_xyz(res.mol), target)
    assert res.min_dist == pytest.approx(md) and md >= 2.2
    assert res.contact_frac >= 0.5 and res.relief_steps >= 0
    assert res.n_within_3A == pa.n_within(pa.heavy_xyz(res.mol), target)


# ---------------------------------------------------------------- Step 2: sites and generate_edits

def test_contact_vs_exposed_classification():
    mol = _mol3d("OC(=O)CCc1ccc(O)cc1")
    xyz = mol.GetConformer().GetPositions()
    sites = {s.rule: s for s in le.find_sites(mol)}
    oh = sites["oh_to_f"].edited_atoms
    o = oh[0]
    # acceptor 3.0 A from the hydroxyl O: contact
    f = _fields([("ALA", "O", _outward(mol, o, 3.0))])
    assert le.classify_site(oh, xyz, f, "oh_to_f") == ("contact", 1, "acceptor")
    # acceptor at 3.9 A: no contact (3.5 A), no protein atom within the default exposure radius
    # (3.5 A) -> exposed; with the plan's original 4.5 A it would be ambiguous
    f = _fields([("ALA", "O", _outward(mol, o, 3.9))])
    assert le.EXPOSED_RADIUS == 3.5
    assert le.classify_site(oh, xyz, f, "oh_to_f") == ("exposed", 0, "")
    assert le.classify_site(oh, xyz, f, "oh_to_f", exposed_radius=4.5) == ("ambiguous", 0, "")
    # a non-acceptor at 3.0 A is no contact, and too close for exposed
    f = _fields([("ALA", "CB", _outward(mol, o, 3.0))])
    assert le.classify_site(oh, xyz, f, "oh_to_f") == ("ambiguous", 0, "")
    # nothing near: exposed; a HETATM heavy atom (cofactor, metal) within the radius makes it ambiguous
    f = _fields([("ALA", "O", _outward(mol, o, 6.0))])
    assert le.classify_site(oh, xyz, f, "oh_to_f") == ("exposed", 0, "")
    near = np.array([_outward(mol, o, 3.3)])
    assert le.classify_site(oh, xyz, f, "oh_to_f", extra_xyz=near)[0] == "ambiguous"
    mid = np.array([_outward(mol, o, 4.0)])
    assert le.classify_site(oh, xyz, f, "oh_to_f", extra_xyz=mid)[0] == "exposed"
    assert le.classify_site(oh, xyz, f, "oh_to_f", extra_xyz=mid, exposed_radius=4.5)[0] == "ambiguous"
    # acid: positive atom within 4.0 A of any atom of the group; a negative one does not count
    acid = sites["cooh_to_me"].edited_atoms
    oxy = [i for i in acid if mol.GetAtomWithIdx(i).GetSymbol() == "O"][0]
    f = _fields([("LYS", "NZ", _outward(mol, oxy, 3.8))])
    assert le.classify_site(acid, xyz, f, "cooh_to_me") == ("contact", 1, "pos")
    f = _fields([("ASP", "OD1", _outward(mol, oxy, 3.3))])
    assert le.classify_site(acid, xyz, f, "cooh_to_me") == ("ambiguous", 0, "")
    # phenyl: >= 3 hydrophobic/aromatic atoms within 4.5 A AND at least one aromatic protein atom;
    # the probe atoms sit 3.6 A above the ring plane (in-plane offsets), so all are 3.6-3.9 A from the ring
    ring = sites["phenyl_to_chx"].edited_atoms
    c = xyz[list(ring)].mean(axis=0)
    u, v, normal = np.linalg.svd(xyz[list(ring)] - c)[2]
    above = [c + normal * 3.6 + a * u + b * v for a, b in ((0.0, 0.0), (0.9, 0.3), (-0.4, 0.9))]
    f = _fields([("LEU", "CD1", above[0]), ("PHE", "CZ", above[1]), ("VAL", "CG1", above[2])])
    assert le.classify_site(ring, xyz, f, "phenyl_to_chx") == ("contact", 3, "aromatic")
    f = _fields([("LEU", "CD1", above[0]), ("TRP", "NE1", above[1]), ("SER", "OG", above[2])])
    # an aromatic partner but fewer than 3 apolar atoms: no contact (generic exposure rule applies)
    assert le.classify_site(ring, xyz, f, "phenyl_to_chx", exposed_radius=4.5) == ("ambiguous", 2, "")
    # apolar only, no aromatic partner: ambiguous at every exposure radius
    f = _fields([("LEU", "CD1", above[0]), ("VAL", "CG1", above[2]), ("ILE", "CD1", above[1])])
    for r in (3.5, 4.0, 4.5):
        assert le.classify_site(ring, xyz, f, "phenyl_to_chx", exposed_radius=r) == ("ambiguous", 3, "")
    # fewer than 3 apolar atoms and none within 3.5 A: exposed at 3.5, ambiguous at 4.5
    f = _fields([("LEU", "CD1", above[0]), ("VAL", "CG1", above[2]), ("SER", "OG", above[1])])
    assert le.classify_site(ring, xyz, f, "phenyl_to_chx") == ("exposed", 2, "")
    assert le.classify_site(ring, xyz, f, "phenyl_to_chx", exposed_radius=4.5) == ("ambiguous", 2, "")

GEN_SMILES = "NCCCCC(CCCCN)(CCCC(=O)O)CCC(O)CCC(O)CCc1ccccc1"


def _gen_case(seed=11):
    """Ligand with two amines and a carboxylic acid next to complementary protein atoms (contact
    candidates) and two hydroxyls and a phenyl ring far from every protein atom (exposed candidates)."""
    mol = _mol3d(GEN_SMILES, seed=seed)
    sites = le.find_sites(mol)
    atoms = []
    for s in sites:
        if s.rule == "amine_to_me":
            atoms.append(("ASP", "OD1", _outward(mol, s.edited_atoms[0], 3.2)))
        elif s.rule == "cooh_to_me":
            oxy = [i for i in s.edited_atoms if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
            atoms.append(("LYS", "NZ", _outward(mol, oxy[0], 3.2)))
    return mol, _fields(atoms), sites


def test_generate_edits_counts_and_kinds():
    mol, fields, sites = _gen_case()
    xyz = mol.GetConformer().GetPositions()
    classes = {}
    for s in sites:
        classes.setdefault(s.rule, []).append(le.classify_site(s.edited_atoms, xyz, fields, s.rule)[0])
    assert classes == {"amine_to_me": ["contact", "contact"], "cooh_to_me": ["contact"],
                       "oh_to_f": ["exposed", "exposed"], "phenyl_to_chx": ["exposed"]}
    log = {}
    poses = le.generate_edits(mol, fields, _rng("9xyz"), k_contact=2, k_exposed=2, log=log)
    assert [(p.kind, p.k) for p in poses] == [(5, 0), (6, 0), (6, 1), (7, 0), (7, 1)]
    null, c0, c1, x0, x1 = poses
    assert (null.rule, null.site_class, null.n_edited_atoms) == ("null", "null", 0)
    assert {c0.rule, c1.rule} == {"amine_to_me", "cooh_to_me"}           # different rules preferred
    assert {x0.rule, x1.rule} == {"oh_to_f", "phenyl_to_chx"}
    assert all(p.site_class == "contact" and p.n_contacts >= 1 for p in (c0, c1))
    assert {c0.partner_type, c1.partner_type} == {"neg", "pos"}
    assert all(p.site_class == "exposed" and p.partner_type == "" for p in (x0, x1))
    clash_eff = min(2.2, pa.min_dist(pa.heavy_xyz(mol), fields[0]))
    for p in poses:
        assert p.min_dist >= clash_eff and p.contact_frac >= 0.5
        assert p.mmff_restrained_rmsd < 0.3 and p.rmsd_to_crystal < 2.5
        assert p.n_heavy == sum(a.GetAtomicNum() > 1 for a in p.mol.GetAtoms())
        Chem.SanitizeMol(Chem.Mol(p.mol))
    assert null.n_heavy == sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms())
    cooh = c0 if c0.rule == "cooh_to_me" else c1
    assert cooh.n_heavy == null.n_heavy - 2 and cooh.dcharge == 0
    # candidate log: rule x site class over all sites found
    assert log["candidates"]["amine_to_me"]["contact"] == 2
    assert log["candidates"]["oh_to_f"]["exposed"] == 2
    assert log["exposed_sites_by_radius"] == {"3.5": 3, "4.0": 3, "4.5": 3}
    # deterministic given the rng
    again = le.generate_edits(mol, fields, _rng("9xyz"), k_contact=2, k_exposed=2)
    assert [(p.kind, p.k, p.rule) for p in again] == [(p.kind, p.k, p.rule) for p in poses]
    for p, q in zip(poses, again):
        assert np.allclose(p.mol.GetConformer().GetPositions(), q.mol.GetConformer().GetPositions())


def test_generate_edits_caps_and_no_sites():
    mol, fields, _ = _gen_case()
    poses = le.generate_edits(mol, fields, _rng(), k_contact=1, k_exposed=0)
    assert [(p.kind, p.k) for p in poses] == [(5, 0), (6, 0)]
    # a ligand without any rule site still gives its null-edit positive, and never raises
    benzene_free = _mol3d("CC(C)(C)OC(C)(C)C")
    poses = le.generate_edits(benzene_free, fields, _rng(), log=(log := {}))
    assert [(p.kind, p.rule) for p in poses] == [(5, "null")]
    assert log["n_sites"] == 0


def test_generate_edits_null_failure_gives_nothing(monkeypatch):
    mol, fields, _ = _gen_case()

    def boom(*a, **k):
        raise le.EditFailure("embed_failed")

    monkeypatch.setattr(le, "null_edit", boom)
    log = {}
    assert le.generate_edits(mol, fields, _rng(), log=log) == []
    assert log["null_failure"] == "embed_failed"
