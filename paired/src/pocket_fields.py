"""Property-weighted atom fields of ligands and protein pockets, and their 3D Zernike descriptors.

Seven channels, fixed order (CHANNELS): occupancy, hydrophobicity, pos, neg, aromatic, donor, acceptor.
Every field point is a heavy atom; hydrogen properties are folded onto the heavy atom they are bonded
to (united-atom), on both sides, so that ligand and protein share one scale.

Ligand (RDKit mol; explicit Hs are added internally when missing, heavy atoms are the points):
  occupancy      1
  hydrophobicity Wildman-Crippen atom logP (rdMolDescriptors._CalcCrippenContribs on the explicit-H
                 molecule), H contributions added to their heavy atom
  pos / neg      max(q, 0) / max(-q, 0), q = united-atom Gasteiger charge (AllChem.ComputeGasteigerCharges
                 on the explicit-H molecule, H charges added to their heavy atom; NaN -> 0)
  aromatic       atom.GetIsAromatic()
  donor/acceptor heavy atoms of the Donor / Acceptor features of RDKit's BaseFeatures.fdef
Protein (PDB ATOM records, heavy atoms of the 20 standard residues; tables below):
  occupancy 1; hydrophobicity CRIPPEN_TABLE; pos/neg POSITIVE/NEGATIVE; aromatic AROMATIC;
  donor DONOR; acceptor ACCEPTOR.

Protein loader (load_protein_atoms): the heavy-atom rule of the clone's LoadPDBasDF is copied (atom
name from columns 13-16; drop names starting with H, and 4-character names whose second character is H)
and atoms are kept only when RES-NAME is one of the 187 residue-atom keys of the clone's
data/PDB_Atom_Keys.csv (20 residues, their heavy atoms and OXT; STANDARD_ATOMS reproduces that set), which
is the clone's inner merge. LoadPDBasDF itself is not called because its output drops residue names and
numbers, which the channel tables need. Differences from the clone's atom set, all deliberate: only the
first MODEL is read; a repeated (chain, residue, insertion code, atom name) keeps its first record
(alternate locations); protonation-state residue names (HID/HIE/HIP/HSD/HSE/HSP, CYX/CYM, ASH, GLH, LYN)
are read as their standard residue.

CRIPPEN_TABLE comes from one RDKit 2024.03.3 run (crippen_table_from_rdkit): for each residue X the middle
residue of Chem.MolFromSequence("G" + X + "G") (neutral, in-chain backbone), Chem.AddHs, Crippen
contributions with H folded onto heavy atoms; OXT from the C-terminal Gly. Atoms that are equivalent in a
charged residue but differ in the neutral template are averaged: ASP OD1/OD2, GLU OE1/OE2, ARG NH1/NH2,
HIS ND1/NE2 (tautomer unknown), and OXT with the carboxyl O of the C-terminal template.
tests/test_pocket_fields.py recomputes the table with RDKit and compares.

Pocket mask (pocket_descriptor, both centred on the ligand heavy-atom centroid):
  shell   protein heavy atoms within `cutoff` (8 A) of any ligand heavy atom; atoms beyond the descriptor
          radius are dropped from the moments and counted (the original definition).
  sphere  protein heavy atoms within `radius` of the ligand centroid, with the descriptor radius equal to
          the sphere radius, so no pocket atom is cropped and the mask does not follow the ligand shape.
channel_totals keeps the signed sum of each channel's weights, which F_00 = |sum| loses.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import zernike3d

CHANNELS = ("occupancy", "hydrophobicity", "pos", "neg", "aromatic", "donor", "acceptor")
N_CHANNELS = len(CHANNELS)
OCC, HYD, POS, NEG, ARO, DON, ACC = range(N_CHANNELS)

RESIDUES = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
ONE_LETTER = dict(zip(RESIDUES, "ARNDCQEGHILKMFPSTWYV"))
RESIDUE_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
                   "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}

_BACKBONE = {"N": -0.4954, "CA": -0.0821, "C": -0.2783, "O": -0.1526}
OXT_CRIPPEN = -0.07195  # mean of the C-terminal template's carboxyl O (-0.1526) and OXT (0.0087)
# Wildman-Crippen united-atom logP per residue heavy atom (RDKit 2024.03.3; see module docstring)
CRIPPEN_TABLE = {
    "ALA": {**_BACKBONE, "CB": 0.5131},
    "ARG": {**_BACKBONE, "CB": 0.3901, "CG": 0.3901, "CD": 0.0425, "NE": -0.4954, "CZ": -0.2783,
            "NH1": -0.146265, "NH2": -0.146265},
    "ASN": {**_BACKBONE, "CB": 0.3901, "CG": -0.2783, "OD1": -0.1526, "ND2": -0.5906},
    "ASP": {**_BACKBONE, "CB": 0.3901, "CG": -0.2783, "OD1": -0.07195, "OD2": -0.07195},
    "CYS": {**_BACKBONE, "CB": 0.0425, "SG": 0.3805},
    "GLN": {**_BACKBONE, "CB": 0.3901, "CG": 0.3901, "CD": -0.2783, "OE1": -0.1526, "NE2": -0.5906},
    "GLU": {**_BACKBONE, "CB": 0.3901, "CG": 0.3901, "CD": -0.2783, "OE1": -0.07195, "OE2": -0.07195},
    "GLY": {**_BACKBONE, "CA": 0.0425},
    "HIS": {**_BACKBONE, "CB": 0.1944, "CG": 0.136, "ND1": -0.2168, "CD2": 0.2811, "CE1": 0.2811,
            "NE2": -0.2168},
    "ILE": {**_BACKBONE, "CB": 0.123, "CG1": 0.5131, "CG2": 0.3901, "CD1": 0.5131},
    "LEU": {**_BACKBONE, "CB": 0.3901, "CG": 0.123, "CD1": 0.5131, "CD2": 0.5131},
    "LYS": {**_BACKBONE, "CB": 0.3901, "CG": 0.3901, "CD": 0.3901, "CE": 0.0425, "NZ": -0.5906},
    "MET": {**_BACKBONE, "CB": 0.3901, "CG": 0.0425, "SD": 0.6482, "CE": 0.1655},
    "PHE": {**_BACKBONE, "CB": 0.1944, "CG": 0.136, "CD1": 0.2811, "CD2": 0.2811, "CE1": 0.2811,
            "CE2": 0.2811, "CZ": 0.2811},
    "PRO": {**_BACKBONE, "N": -0.3187, "CB": 0.3901, "CG": 0.3901, "CD": 0.0425},
    "SER": {**_BACKBONE, "CB": 0.0425, "OG": -0.557},
    "THR": {**_BACKBONE, "CB": -0.0821, "OG1": -0.557, "CG2": 0.5131},
    "TRP": {**_BACKBONE, "CB": 0.1944, "CG": 0.136, "CD1": 0.2811, "CD2": 0.2955, "NE1": -0.1097,
            "CE2": 0.2955, "CE3": 0.2811, "CZ2": 0.2811, "CZ3": 0.2811, "CH2": 0.2811},
    "TYR": {**_BACKBONE, "CB": 0.1944, "CG": 0.136, "CD1": 0.2811, "CD2": 0.2811, "CE1": 0.2811,
            "CE2": 0.2811, "CZ": 0.5437, "OH": -0.557},
    "VAL": {**_BACKBONE, "CB": 0.123, "CG1": 0.5131, "CG2": 0.5131},
}
SYMMETRISED = (("ASP", "OD1", "OD2"), ("GLU", "OE1", "OE2"), ("ARG", "NH1", "NH2"), ("HIS", "ND1", "NE2"))

STANDARD_ATOMS = frozenset((res, atom) for res, atoms in CRIPPEN_TABLE.items() for atom in atoms) | \
    frozenset((res, "OXT") for res in RESIDUES)

# formal-charge channels: (residue or None for any residue, atom) -> weight
POSITIVE = {("LYS", "NZ"): 1.0, ("ARG", "NH1"): 1.0 / 3, ("ARG", "NH2"): 1.0 / 3, ("ARG", "NE"): 1.0 / 3,
            ("HIS", "ND1"): 0.1, ("HIS", "NE2"): 0.1}
NEGATIVE = {("ASP", "OD1"): 0.5, ("ASP", "OD2"): 0.5, ("GLU", "OE1"): 0.5, ("GLU", "OE2"): 0.5,
            (None, "OXT"): 1.0}
AROMATIC = {"PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
            "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
            "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
            "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"}}
DONOR = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"), ("GLN", "NE2"), ("HIS", "ND1"),
         ("HIS", "NE2"), ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"), ("TYR", "OH"),
         ("CYS", "SG")}  # plus backbone N of every residue except PRO
ACCEPTOR = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"), ("ASN", "OD1"), ("GLN", "OE1"),
            ("HIS", "ND1"), ("HIS", "NE2"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("MET", "SD")}
# plus backbone O and OXT of every residue

MAX_PROTEIN_ATOMS = 500_000
MAX_LIGAND_HEAVY_ATOMS = 2_000


@lru_cache(maxsize=None)
def protein_atom_weights(resname, atom):
    """The 7-channel weight vector of one standard residue atom (a tuple, cached)."""
    if (resname, atom) not in STANDARD_ATOMS:
        raise KeyError(f"{resname}-{atom} is not a standard residue heavy atom")
    w = [0.0] * N_CHANNELS
    w[OCC] = 1.0
    w[HYD] = OXT_CRIPPEN if atom == "OXT" else CRIPPEN_TABLE[resname][atom]
    w[POS] = POSITIVE.get((resname, atom), 0.0)
    w[NEG] = NEGATIVE.get((resname, atom), NEGATIVE.get((None, atom), 0.0))
    w[ARO] = 1.0 if atom in AROMATIC.get(resname, ()) else 0.0
    w[DON] = 1.0 if (resname, atom) in DONOR or (atom == "N" and resname != "PRO") else 0.0
    w[ACC] = 1.0 if (resname, atom) in ACCEPTOR or atom in ("O", "OXT") else 0.0
    return tuple(w)


def _is_heavy_name(name):
    """The clone's LoadPDBasDF heavy-atom predicate on the space-stripped atom name."""
    if not name:
        return False
    return (len(name) < 4 and name[0] != "H") or (len(name) == 4 and name[1] != "H" and name[0] != "H")


def load_protein_atoms(path, max_atoms=MAX_PROTEIN_ATOMS):
    """DataFrame (resname, atom, chain, resseq, icode, X, Y, Z) of standard-residue heavy atoms."""
    rows = []
    seen = set()
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            if line[:4] != "ATOM":
                continue
            name = line[12:16].replace(" ", "")
            if not _is_heavy_name(name):
                continue
            resname = line[17:20].strip()
            resname = RESIDUE_ALIASES.get(resname, resname)
            if (resname, name) not in STANDARD_ATOMS:
                continue
            chain, resseq, icode = line[21:22], line[22:26].strip(), line[26:27].strip()
            key = (chain, resseq, icode, resname, name)
            if key in seen:
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            seen.add(key)
            rows.append((resname, name, chain, resseq, icode) + xyz)
            if len(rows) > max_atoms:
                raise ValueError(f"protein has more than {max_atoms} standard heavy atoms")
    df = pd.DataFrame(rows, columns=["resname", "atom", "chain", "resseq", "icode", "X", "Y", "Z"])
    if not np.isfinite(df[["X", "Y", "Z"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite protein coordinates")
    return df


def protein_fields(protein):
    """(xyz[n, 3], W[n, 7], residue_ids[n]) from a PDB path or a load_protein_atoms-style DataFrame.

    A DataFrame needs X, Y, Z and either resname + atom columns or a PDB_ATOM column ("RES-NAME");
    rows that are not standard residue heavy atoms are dropped.
    """
    df = load_protein_atoms(protein) if isinstance(protein, (str, os.PathLike)) else protein.copy()
    if "resname" not in df.columns or "atom" not in df.columns:
        if "PDB_ATOM" not in df.columns:
            raise ValueError("protein DataFrame needs resname/atom or PDB_ATOM columns")
        split = df["PDB_ATOM"].astype(str).str.split("-", n=1, expand=True)
        df["resname"], df["atom"] = split[0], split[1]
    df["resname"] = df["resname"].astype(str).str.strip().map(lambda r: RESIDUE_ALIASES.get(r, r))
    df["atom"] = df["atom"].astype(str).str.strip()
    keep = np.fromiter(((r, a) in STANDARD_ATOMS for r, a in zip(df["resname"], df["atom"])),
                       dtype=bool, count=len(df))
    df = df[keep].reset_index(drop=True)
    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float).reshape(-1, 3)
    w = np.array([protein_atom_weights(r, a) for r, a in zip(df["resname"], df["atom"])],
                 dtype=float).reshape(-1, N_CHANNELS)
    if {"chain", "resseq"}.issubset(df.columns):
        icode = df["icode"].astype(str) if "icode" in df.columns else ""
        resid = (df["chain"].astype(str) + ":" + df["resname"] + df["resseq"].astype(str) + icode).to_numpy()
    else:
        resid = df["resname"].to_numpy()
    return xyz, w, np.asarray(resid, dtype=str)


@lru_cache(maxsize=1)
def _feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    return ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))


def ligand_fields(mol):
    """(xyz[n_heavy, 3], W[n_heavy, 7]) of an RDKit molecule with a conformer (see module docstring)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError("ligand has no conformer")
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        raise ValueError("ligand has no heavy atoms")
    if len(heavy) > MAX_LIGAND_HEAVY_ATOMS:
        raise ValueError(f"ligand has more than {MAX_LIGAND_HEAVY_ATOMS} heavy atoms")
    pos = mol.GetConformer().GetPositions()
    xyz = np.asarray(pos[heavy], dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite ligand coordinates")

    molh = Chem.AddHs(mol)  # adds only missing Hs; existing atom indices are unchanged
    crippen = rdMolDescriptors._CalcCrippenContribs(molh)
    AllChem.ComputeGasteigerCharges(molh)
    n_all = molh.GetNumAtoms()
    logp = np.zeros(n_all)
    charge = np.zeros(n_all)
    for atom in molh.GetAtoms():
        i = atom.GetIdx()
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if atom.HasProp("_GasteigerHCharge"):
            q += atom.GetDoubleProp("_GasteigerHCharge")
        q = q if np.isfinite(q) else 0.0
        target = i
        if atom.GetAtomicNum() == 1:
            nbrs = atom.GetNeighbors()
            if len(nbrs) == 0:
                continue  # an isolated H carries no field point
            target = nbrs[0].GetIdx()
        logp[target] += crippen[i][0]
        charge[target] += q

    donors, acceptors = set(), set()
    for feat in _feature_factory().GetFeaturesForMol(molh):
        fam = feat.GetFamily()
        if fam == "Donor":
            donors.update(feat.GetAtomIds())
        elif fam == "Acceptor":
            acceptors.update(feat.GetAtomIds())

    w = np.zeros((len(heavy), N_CHANNELS))
    w[:, OCC] = 1.0
    w[:, HYD] = logp[heavy]
    q = charge[heavy]
    w[:, POS] = np.maximum(q, 0.0)
    w[:, NEG] = np.maximum(-q, 0.0)
    w[:, ARO] = [1.0 if mol.GetAtomWithIdx(i).GetIsAromatic() else 0.0 for i in heavy]
    w[:, DON] = [1.0 if i in donors else 0.0 for i in heavy]
    w[:, ACC] = [1.0 if i in acceptors else 0.0 for i in heavy]
    return xyz, w


POCKET_MODES = ("shell", "sphere")


def pocket_atoms(protein_xyz, ligand_xyz, cutoff=8.0):
    """Boolean mask of protein atoms within `cutoff` A of any ligand atom."""
    protein_xyz = np.asarray(protein_xyz, dtype=float).reshape(-1, 3)
    ligand_xyz = np.asarray(ligand_xyz, dtype=float).reshape(-1, 3)
    if len(protein_xyz) == 0 or len(ligand_xyz) == 0:
        return np.zeros(len(protein_xyz), dtype=bool)
    dist, _ = cKDTree(ligand_xyz).query(protein_xyz, k=1, distance_upper_bound=cutoff)
    return dist <= cutoff


def inside_ball(xyz, center, radius):
    """Boolean mask of points with |(r - center) / radius| <= 1: the exact test zernike3d.moments uses to
    keep a point, so a descriptor of the same centre and radius drops none of the masked points."""
    x = np.asarray(xyz, dtype=float).reshape(-1, 3) - np.asarray(center, dtype=float).reshape(1, 3)
    x = x / float(radius)
    return np.sqrt((x * x).sum(axis=1)) <= 1.0


def channel_totals(xyz, w, center, radius):
    """Signed per-channel weight sums [C] over the points the descriptor keeps (inside_ball). F_00 is
    |total| * (3 / 4 pi) / sqrt(4 pi), so the total restores the sign F drops (e.g. of hydrophobicity)."""
    w = np.asarray(w, dtype=float)
    w = w[:, None] if w.ndim == 1 else w
    return w[inside_ball(xyz, center, radius)].sum(axis=0)


def pocket_descriptor(protein_fields, ligand_xyz, order=12, radius=12.0, cutoff=8.0, mode="shell",
                      return_total=False):
    """(F[7, K], G[21, K], n_pocket_atoms, n_outside[, total[7]]), centred on the ligand centroid and
    scaled by `radius`.

    mode "shell":  protein atoms within `cutoff` A of any ligand atom; n_outside of them lie beyond `radius`
                   from the centroid and are dropped from the moments.
    mode "sphere": protein atoms within `radius` A of the ligand centroid (inside_ball), so the mask and the
                   descriptor share one radius and n_outside is 0; `cutoff` is not used.
    return_total: also return the signed channel totals over the kept pocket atoms (channel_totals).
    """
    if mode not in POCKET_MODES:
        raise ValueError(f"unknown pocket mode {mode!r}")
    xyz, w = protein_fields[0], protein_fields[1]
    ligand_xyz = np.asarray(ligand_xyz, dtype=float).reshape(-1, 3)
    center = ligand_xyz.mean(axis=0)
    if mode == "shell":
        mask = pocket_atoms(xyz, ligand_xyz, cutoff)
    else:
        mask = inside_ball(xyz, center, radius)
    f, g, _, n_out = zernike3d.descriptor(xyz[mask], w[mask], order=order, radius=radius, center=center)
    if not return_total:
        return f, g, int(mask.sum()), n_out
    return f, g, int(mask.sum()), n_out, channel_totals(xyz[mask], w[mask], center, radius)


def ligand_descriptor(mol, order=12, radius=12.0, return_total=False):
    """(F[7, K], G[21, K], n_heavy_atoms, n_outside[, total[7]]) centred on the heavy-atom centroid."""
    xyz, w = ligand_fields(mol)
    center = xyz.mean(axis=0)
    f, g, _, n_out = zernike3d.descriptor(xyz, w, order=order, radius=radius, center=center)
    if not return_total:
        return f, g, len(xyz), n_out
    return f, g, len(xyz), n_out, channel_totals(xyz, w, center, radius)


def crippen_table_from_rdkit():
    """Recompute CRIPPEN_TABLE (and the OXT value) with RDKit, with the same symmetrisation."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    def folded(mol):
        molh = Chem.AddHs(mol)
        contrib = rdMolDescriptors._CalcCrippenContribs(molh)
        out = np.zeros(mol.GetNumAtoms())
        for atom in molh.GetAtoms():
            i = atom.GetIdx()
            j = atom.GetNeighbors()[0].GetIdx() if atom.GetAtomicNum() == 1 else i
            out[j] += contrib[i][0]
        return out

    table, oxt = {}, None
    for res in RESIDUES:
        mol = Chem.MolFromSequence("G" + ONE_LETTER[res] + "G")
        vals = folded(mol)
        table[res] = {}
        cterm = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info.GetResidueNumber() == 2:
                table[res][info.GetName().strip()] = float(vals[atom.GetIdx()])
            elif info.GetResidueNumber() == 3:
                cterm[info.GetName().strip()] = float(vals[atom.GetIdx()])
        oxt = 0.5 * (cterm["O"] + cterm["OXT"])
    for res, a, b in SYMMETRISED:
        mean = 0.5 * (table[res][a] + table[res][b])
        table[res][a] = table[res][b] = mean
    return table, oxt
