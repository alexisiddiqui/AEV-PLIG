"""Ligand reader and RDKit node/bond features (atom_features and mol_to_graph in generate_bindingnet_graphs.py).

Fixtures in tests/data describe benzamide, SMILES ``NC(=O)c1ccccc1``, heavy atoms in RDKit order
N, C(amide), O, then the six ring carbons (ipso first):
- benzamide.sdf: ``Chem.AddHs`` of the SMILES, ``AllChem.EmbedMolecule(mol, randomSeed=42)``,
  ``Chem.MolToMolBlock`` plus ``$$$$``.
- benzamide.mol2: written by hand in Tripos MOL2 (RDKit has no Mol2 writer) with the same atoms,
  bonds and coordinates, hydrogens included as in PDBbind ligand files (RDKit's Mol2 parser needs
  them to assign formal charges; it drops them on read and ``read_ligand`` adds them back).
"""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from aevplig_jax.constants import ATOM_SYMBOLS, EDGE_DIM, RDKIT_DIM
from aevplig_jax.rdkit_features import LigandArrays, ligand_arrays, read_ligand

DATA = Path(__file__).resolve().parent / "data"
SDF = DATA / "benzamide.sdf"
MOL2 = DATA / "benzamide.mol2"

SYMBOLS = ["N", "C", "O", "C", "C", "C", "C", "C", "C"]
RING = [3, 4, 5, 6, 7, 8]
# Columns after the 10 symbol columns.
HEAVY, HYDROGENS, VALENCE, AROMATIC, IN_RING = 10, 11, 12, 13, 14

SINGLE, AROMATIC_BOND, DOUBLE = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]
BONDS = {(0, 1): SINGLE, (1, 2): DOUBLE, (1, 3): SINGLE,
         (3, 4): AROMATIC_BOND, (4, 5): AROMATIC_BOND, (5, 6): AROMATIC_BOND,
         (6, 7): AROMATIC_BOND, (7, 8): AROMATIC_BOND, (3, 8): AROMATIC_BOND}


def _sdf_arrays() -> LigandArrays:
    return ligand_arrays(read_ligand(SDF, "sdf"))


def test_benzamide_features_sdf():
    mol = read_ligand(SDF, "sdf")
    arrays = ligand_arrays(mol)

    assert arrays.feat.shape == (9, RDKIT_DIM) and arrays.feat.dtype == np.float32
    assert arrays.xyz.shape == (9, 3) and arrays.xyz.dtype == np.float32
    np.testing.assert_allclose(arrays.xyz, mol.GetConformer().GetPositions()[:9], rtol=0, atol=1e-6)

    one_hot = np.zeros((9, len(ATOM_SYMBOLS)), dtype=np.float32)
    one_hot[np.arange(9), [ATOM_SYMBOLS.index(s) for s in SYMBOLS]] = 1.0
    np.testing.assert_array_equal(arrays.feat[:, :10], one_hot)

    np.testing.assert_array_equal(arrays.feat[:, HEAVY], [1, 3, 1, 3, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(arrays.feat[:, HYDROGENS], [2, 0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(arrays.feat[:, VALENCE], [3, 4, 2, 4, 4, 4, 4, 4, 4])
    ring = np.isin(np.arange(9), RING).astype(np.float32)
    np.testing.assert_array_equal(arrays.feat[:, AROMATIC], ring)
    np.testing.assert_array_equal(arrays.feat[:, IN_RING], ring)


def test_bonds_sorted_and_symmetric():
    arrays = _sdf_arrays()

    assert arrays.bond_src.shape == (18,) and arrays.bond_src.dtype == np.int32
    assert arrays.bond_dst.shape == (18,) and arrays.bond_dst.dtype == np.int32
    assert arrays.bond_type.shape == (18, EDGE_DIM) and arrays.bond_type.dtype == np.float32

    pairs = list(zip(arrays.bond_src.tolist(), arrays.bond_dst.tolist()))
    assert pairs == sorted(pairs)
    assert set(pairs) == {(j, i) for i, j in pairs}
    for (i, j), bond_type in zip(pairs, arrays.bond_type.tolist()):
        assert bond_type == BONDS[(min(i, j), max(i, j))]


def test_mol2_adds_hydrogen_counts():
    mol = read_ligand(MOL2, "mol2")
    assert mol.GetNumAtoms() == 16  # hydrogens dropped by the Mol2 reader are added back
    mol2_arrays = ligand_arrays(mol)
    sdf_arrays = _sdf_arrays()

    assert mol2_arrays.feat[0, HYDROGENS] == 2
    np.testing.assert_array_equal(mol2_arrays.feat, sdf_arrays.feat)
    np.testing.assert_array_equal(mol2_arrays.bond_src, sdf_arrays.bond_src)
    np.testing.assert_array_equal(mol2_arrays.bond_dst, sdf_arrays.bond_dst)
    np.testing.assert_array_equal(mol2_arrays.bond_type, sdf_arrays.bond_type)
    np.testing.assert_array_equal(mol2_arrays.xyz, sdf_arrays.xyz)


def test_unknown_element_raises():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[Si](C)(C)C"))
    AllChem.Compute2DCoords(mol)
    with pytest.raises(ValueError, match="Si"):
        ligand_arrays(mol)


def test_molecule_without_bonds():
    mol = Chem.MolFromSmiles("[Cl-]")
    AllChem.Compute2DCoords(mol)
    arrays = ligand_arrays(mol)

    assert arrays.feat.shape == (1, RDKIT_DIM)
    assert arrays.bond_src.shape == (0,) and arrays.bond_src.dtype == np.int32
    assert arrays.bond_dst.shape == (0,) and arrays.bond_dst.dtype == np.int32
    assert arrays.bond_type.shape == (0, EDGE_DIM) and arrays.bond_type.dtype == np.float32


@pytest.mark.parametrize("kind", ["sdf", "mol2"])
def test_read_ligand_returns_none_when_unreadable(tmp_path, kind):
    path = tmp_path / f"broken.{kind}"
    path.write_text("not a molecule\n")
    assert read_ligand(path, kind) is None


def test_read_ligand_rejects_unknown_kind():
    with pytest.raises(ValueError, match="pdb"):
        read_ligand(SDF, "pdb")
