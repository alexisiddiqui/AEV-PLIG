"""Protein pocket atoms of a PDB file, kept exactly as ``LoadPDBasDF`` keeps them.

``LoadPDBasDF`` (generate_pdbbind_graphs.py:43-63) reads ``ATOM`` records only (no ``HETATM``),
drops hydrogens by a test on the space-stripped atom name (columns 12:16), keys each atom as
``resname(17:20) + "-" + name``, keeps both conformers of alternate locations (it never reads
column 16), inner-merges with the key table and sorts by serial (6:11). Here atoms whose key is not
in the table are kept with species 0, which the AEV ignores, so the AEV is unchanged. Coordinates
(30:38, 38:46, 46:54) are not rounded. Imports numpy and the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from aevplig_jax.constants import RC

STANDARD_RESIDUES = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
OTHER_RESIDUE = len(STANDARD_RESIDUES)  # resname code of anything else
OTHER_ATOM = -1  # atom code of names not in the key table


@dataclass
class PocketArrays:
    """Protein heavy atoms near a ligand, rows in serial order."""

    xyz: np.ndarray  # (P, 3) float32
    species: np.ndarray  # (P,) int8, 1..22 from the species table, 0 when the RES-ATOM key is absent
    res: np.ndarray  # (P,) int32, serial of (chain, resSeq, iCode) in order of first appearance
    resname: np.ndarray  # (P,) int8, index into STANDARD_RESIDUES, OTHER_RESIDUE otherwise
    atom: np.ndarray  # (P,) int16, index into atom_name_table(species), OTHER_ATOM otherwise
    chain: np.ndarray  # (P,) int8, serial of the chain id in order of first appearance
    bfactor: np.ndarray  # (P,) float16, 0 when the field is blank or unparsable


class _Atom(NamedTuple):
    serial: int
    resname: str
    name: str
    chain: str
    residue: tuple[str, str, str]  # (chain, resSeq, iCode)
    xyz: tuple[float, float, float]
    bfactor: float


def atom_name_table(species: dict[str, int]) -> tuple[str, ...]:
    """Sorted distinct atom names (the part after the dash) of the species table's keys."""
    return tuple(sorted({key.split("-", 1)[1] for key in species}))


def _is_heavy(name: str) -> bool:
    """The hydrogen test of LoadPDBasDF on the space-stripped name (a blank name raises IndexError).

    Names shorter than 4 characters are dropped when they start with H; 4-character names when
    either of the first two characters is H. So ``1HA`` is kept, as in the original.
    """
    return (len(name) < 4 and name[0] != "H") or (len(name) == 4 and name[1] != "H" and name[0] != "H")


def _bfactor(field: str) -> float:
    try:
        return float(field)
    except ValueError:
        return 0.0


_SLACK = 1e-3  # A, added to the reach of the centroid prefilter (float64 error is ~1e-12 A)
_BAND = 1e-6  # A^2, Gram distances this close to cutoff^2 are recomputed exactly (error ~1e-11)


def _close_pairs(points: np.ndarray, centres: np.ndarray, cutoff: float,
                 keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(centre_index, point_index)`` of every pair with ``sum((point - centre)**2) <= cutoff**2``
    in float64 (inclusive), sorted by centre and then point; only points where ``keep`` is True
    when it is given.

    Vectorised: points farther than (largest centre distance + cutoff) from the centres' mean are
    dropped (triangle inequality), squared distances to the rest come from the Gram form
    ``|c|^2 + |p|^2 - 2 c.p`` on mean-centred coordinates, and pairs within ``_BAND`` of the cutoff
    are recomputed with the expression above, so the decision is the same as a direct loop's.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(centres) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    middle = centres.mean(axis=0)
    rel_c = centres - middle
    norm_c = np.einsum("ij,ij->i", rel_c, rel_c)
    rel_p = points - middle
    norm_p = np.einsum("ij,ij->i", rel_p, rel_p)
    reach = np.sqrt(norm_c.max()) + cutoff + _SLACK
    close = norm_p <= reach * reach
    near = np.flatnonzero(close if keep is None else close & keep)
    d2 = (rel_c * -2.0) @ rel_p[near].T  # (C, near); scaling by -2 is exact
    d2 += norm_c[:, None]
    d2 += norm_p[near]
    flat = np.flatnonzero(d2 <= cutoff * cutoff + _BAND)
    centre_index, column = np.divmod(flat, len(near))
    band = np.flatnonzero(d2.ravel()[flat] > cutoff * cutoff - _BAND)
    if len(band):
        exact = ((points[near[column[band]]] - centres[centre_index[band]]) ** 2).sum(axis=1)
        outside = band[exact > cutoff**2]
        if len(outside):
            inside = np.ones(len(flat), dtype=bool)
            inside[outside] = False
            centre_index, column = centre_index[inside], column[inside]
    return centre_index, near[column]


def _within(points: np.ndarray, centres: np.ndarray, cutoff: float) -> np.ndarray:
    """True for each point within ``cutoff`` (inclusive) of at least one centre."""
    within = np.zeros(len(np.asarray(points).reshape(-1, 3)), dtype=bool)
    within[_close_pairs(points, centres, cutoff)[1]] = True
    return within


def _read_atoms(pdb_path: Path) -> list[_Atom]:
    """ATOM records with heavy-atom names, sorted by serial (stable)."""
    atoms = []
    with open(pdb_path) as handle:
        for line in handle:
            if line[:4] != "ATOM":
                continue
            name = line[12:16].replace(" ", "")
            if not _is_heavy(name):
                continue
            atoms.append(_Atom(
                serial=int(line[6:11]),
                resname=line[17:20],
                name=name,
                chain=line[21],
                residue=(line[21], line[22:26], line[26]),
                xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
                bfactor=_bfactor(line[60:66]),
            ))
    atoms.sort(key=lambda atom: atom.serial)
    return atoms


def read_pocket(
    pdb_path: Path, lig_xyz: np.ndarray, species: dict[str, int], radius: float = 10.0
) -> PocketArrays:
    """Atoms ``LoadPDBasDF`` keeps, plus unmapped names (species 0), within ``radius`` (inclusive) of
    a ligand atom.

    ``species`` is ``constants.species_table`` of ``data/PDB_Atom_Keys.csv``. Residue and chain
    serials count first appearances along the returned rows (serial order, which is file order
    when serials increase through the file).
    """
    atoms = _read_atoms(pdb_path)
    xyz = np.array([atom.xyz for atom in atoms], dtype=np.float64).reshape(-1, 3)
    keep = _within(xyz, lig_xyz, radius)
    atoms = [atom for atom, kept in zip(atoms, keep) if kept]

    residue_code = {name: i for i, name in enumerate(STANDARD_RESIDUES)}
    atom_code = {name: i for i, name in enumerate(atom_name_table(species))}
    residues: dict[tuple[str, str, str], int] = {}
    chains: dict[str, int] = {}
    res = [residues.setdefault(atom.residue, len(residues)) for atom in atoms]
    chain = [chains.setdefault(atom.chain, len(chains)) for atom in atoms]
    if len(chains) > np.iinfo(np.int8).max + 1:
        raise ValueError(f"{pdb_path}: {len(chains)} chains within {radius} A do not fit int8 codes")

    return PocketArrays(
        xyz=xyz[keep].astype(np.float32),
        species=np.array([species.get(f"{a.resname}-{a.name}", 0) for a in atoms], dtype=np.int8),
        res=np.array(res, dtype=np.int32),
        resname=np.array([residue_code.get(a.resname, OTHER_RESIDUE) for a in atoms], dtype=np.int8),
        atom=np.array([atom_code.get(a.name, OTHER_ATOM) for a in atoms], dtype=np.int16),
        chain=np.array(chain, dtype=np.int8),
        bfactor=np.array([a.bfactor for a in atoms], dtype=np.float16),
    )


def typed_pairs(pocket: PocketArrays, lig_xyz: np.ndarray, cutoff: float = RC
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(ligand_index, pocket_index)`` of every typed pocket atom (species > 0) within ``cutoff``
    (inclusive) of a ligand atom, sorted by ligand atom and then pocket atom: the radial-AEV pairs.
    """
    return _close_pairs(pocket.xyz, lig_xyz, cutoff, keep=pocket.species > 0)


def typed_within(pocket: PocketArrays, lig_xyz: np.ndarray, cutoff: float = RC) -> np.ndarray:
    """Boolean over pocket atoms: typed (species > 0) and within ``cutoff`` (inclusive) of a ligand
    atom. This subset is what goes to the device for the radial AEV.
    """
    within = np.zeros(len(pocket.species), dtype=bool)
    within[typed_pairs(pocket, lig_xyz, cutoff)[1]] = True
    return within
