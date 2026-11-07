"""Ligand reading and the 15 RDKit node features and 4 bond features of AEV-PLIG.

Replicates ``atom_features``, ``mol_to_graph`` and the ligand coordinates of ``LoadMolasDF`` in
generate_bindingnet_graphs.py, and the per-source reading paths: sdf (BindingNet, FEP, 15-PGDH) via
``SDMolSupplier(removeHs=False)``, mol2 (PDBbind, BindingDB) via ``MolFromMol2File`` then
``AddHs(addCoords=True)`` (generate_pdbbind_graphs.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from aevplig_jax.constants import ATOM_SYMBOLS, BOND_TYPES, EDGE_DIM, RDKIT_DIM


@dataclass
class LigandArrays:
    """Heavy-atom graph of one ligand, nodes in RDKit atom order."""

    xyz: np.ndarray  # (L, 3) float32
    feat: np.ndarray  # (L, 15) float32
    bond_src: np.ndarray  # (E,) int32, both directions, sorted by (src, dst)
    bond_dst: np.ndarray  # (E,) int32
    bond_type: np.ndarray  # (E, 4) float32, one-hot over BOND_TYPES


def read_ligand(path: Path, kind: str) -> Chem.Mol | None:
    """Read a ligand as the original featurisers do; ``None`` when RDKit returns no molecule.

    ``kind`` is ``"sdf"`` (first record, hydrogens kept) or ``"mol2"`` (hydrogens added with
    coordinates). Missing or empty files raise RDKit's ``OSError``.
    """
    if kind == "sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        return supplier[0] if len(supplier) else None
    if kind == "mol2":
        mol = Chem.MolFromMol2File(str(path))
        return None if mol is None else Chem.AddHs(mol, addCoords=True)
    raise ValueError(f"unknown ligand kind {kind!r}, expected 'sdf' or 'mol2'")


def _one_hot(value, allowed) -> list[float]:
    """``one_of_k_encoding`` of the original, raising ``ValueError`` on values outside ``allowed``."""
    if value not in allowed:
        raise ValueError(f"input {value} not in allowable set {list(allowed)}")
    return [float(value == item) for item in allowed]


def _atom_features(atom: Chem.Atom) -> list[float]:
    neighbour_symbols = [n.GetSymbol() for n in atom.GetNeighbors()]
    return [
        *_one_hot(atom.GetSymbol(), ATOM_SYMBOLS),
        sum(s != "H" for s in neighbour_symbols),
        sum(s == "H" for s in neighbour_symbols),
        atom.GetExplicitValence(),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def ligand_arrays(mol: Chem.Mol) -> LigandArrays:
    """Node features, coordinates and directed bond lists over the heavy atoms of ``mol``.

    Raises ``ValueError`` on an element outside ``ATOM_SYMBOLS`` or a bond type outside
    ``BOND_TYPES``, as the original's ``one_of_k_encoding`` does.
    """
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
    node_of = {atom_idx: node for node, atom_idx in enumerate(heavy)}

    feat = np.array([_atom_features(mol.GetAtomWithIdx(i)) for i in heavy], dtype=np.float32)
    feat = feat.reshape(len(heavy), RDKIT_DIM)
    # LoadMolasDF rounds the coordinates to 4 decimals before computing the AEVs.
    xyz = np.round(mol.GetConformer().GetPositions()[heavy], 4).astype(np.float32).reshape(-1, 3)

    edges = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in node_of and end in node_of:
            bond_type = _one_hot(bond.GetBondType(), BOND_TYPES)
            edges.append((node_of[begin], node_of[end], bond_type))
            edges.append((node_of[end], node_of[begin], bond_type))
    edges.sort(key=lambda edge: (edge[0], edge[1]))

    return LigandArrays(
        xyz=xyz,
        feat=feat,
        bond_src=np.array([e[0] for e in edges], dtype=np.int32),
        bond_dst=np.array([e[1] for e in edges], dtype=np.int32),
        bond_type=np.array([e[2] for e in edges], dtype=np.float32).reshape(-1, EDGE_DIM),
    )
