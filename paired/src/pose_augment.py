"""Perturbed ligand poses for PDBbind complexes (arm 4): near-native positives and far decoys.

Rules: docs/plans/2026-11-05-aevplig-pose-decoys-arm4.md, Task A "Generator rules (exact)".
- Near-native (kind 1, method "nn"): rigid_perturb(max_angle 20 deg, trans_sd 0.5 A) then
  torsion_perturb(sd 15 deg) on every rotatable bond, then relief (below); accepted when
  0.3 <= RMSD <= 2.0 and min_dist >= clash_eff (relief that pushes RMSD out of range is just a
  failed attempt).
- Decoys (kind 2): slot k even -> "flip" (rotation U(90, 180) deg about a random axis through the
  heavy-atom centroid, translation N(0, 1) per axis), k odd -> "conf" (random ETKDG conformer,
  uniform random rotation, centroid on the crystal centroid + N(0, 1) per axis; when ETKDG fails
  for the ligand, the slot uses "flip" attempts and records that method). Accepted when
  RMSD >= 4.0, min_dist >= clash_eff and contact_frac >= 0.5; after max_attempts failures the
  RMSD threshold drops to 3.0 for another max_attempts; then the slot gives no pose.
- Relief (soft repulsion; every near-native and decoy candidate, before acceptance; the crystal
  pose is never moved): relieve_clash runs a greedy rigid-body Monte Carlo of up to
  relief_steps_max steps (rotation U(-relief_step_deg, +relief_step_deg) about a random axis
  through the heavy-atom centroid plus a translation of U(0, relief_step_A) A in a random
  direction). A step is kept when it lowers the soft score sum(max(0, r0 - d_ij)^2) over
  ligand-heavy/protein pairs, r0 = relief_target_A (default 3.0 A) lowered by the same amount as
  the clash threshold, r0_eff = r0 - (clash - clash_eff) (so a crystal pose that already clashes,
  e.g. one protein atom at 1.4 A, does not push every near-native out of the 2.0 A RMSD window;
  r0_eff = r0 for every complex whose crystal pose has min_dist >= clash), and keeps the heavy-atom
  centroid within 2.5 A of the crystal centroid (or, for a candidate placed further out, no
  further out than it already is). It stops when min_dist >= r0_eff or the budget is spent; the
  candidate it ends with then meets the unchanged acceptance rules (min_dist >= clash_eff etc.), so
  relief that cannot reach r0_eff does not by itself reject a candidate. The aim is decoys and
  near-natives whose close-contact statistics resemble crystal poses instead of piling up at the
  clash limit. Pose.relief_steps = steps used (0: min_dist >= r0_eff already).
- clash_eff = min(clash, crystal_min_dist); poses of one kind are distinct (RMSD >= 0.3 A).
RMSD = heavy-atom, symmetry-corrected, no alignment (rdMolAlign.CalcRMS). Distances use the heavy
atoms of the ligand (symbol != "H", as the AEV code) against the protein heavy atoms given.
contact_frac = (protein atoms within 4.0 A of any ligand heavy atom in the pose) / (same count for
the crystal pose); 1.0 when the crystal pose has no contacts.
Pose molecules are copies of the input (Hs kept) with one conformer holding the new coordinates.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolAlign, rdMolTransforms
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

NN_MAX_ANGLE_DEG = 20.0
NN_TRANS_SD = 0.5
NN_TORSION_SD_DEG = 15.0
FLIP_ANGLE_DEG = (90.0, 180.0)
DECOY_TRANS_SD = 1.0
CONTACT_CUTOFF = 4.0
DISTINCT_RMSD = 0.3
# Cap on the symmetry matches CalcRMS enumerates (RDKit default 1e6): bounds time and memory per
# call for ligands with many equivalent terminal groups (polysulfates, polyphosphates).
MAX_MATCHES = 100_000
RELIEF_STEPS_MAX = 300
RELIEF_STEP_DEG = 10.0
RELIEF_STEP_A = 0.5
RELIEF_CENTROID_MAX = 2.5   # A from the crystal heavy-atom centroid
RELIEF_TARGET_A = 3.0       # r0 of the soft repulsion score
RELIEF_NEIGHBOURS = 24      # protein atoms per ligand atom in the soft score
CLOSE_CUTOFF = 3.0          # n_within: ligand heavy atoms with a protein atom closer than this


@dataclass
class Pose:
    kind: int          # 1 near-native positive, 2 decoy
    k: int             # index among accepted poses of this kind (0..K-1, contiguous)
    method: str        # nn | flip | conf
    mol: Chem.Mol      # input molecule (Hs kept) with the new conformer
    rmsd: float        # heavy-atom symmetry-corrected RMSD to the input pose (A)
    min_dist: float    # min ligand-heavy to protein-heavy distance (A)
    contact_frac: float
    attempts: int      # attempts used for this pose (1..max_attempts, or up to 2x for decoys)
    relief_steps: int = 0  # relief Monte Carlo steps used (0: min_dist >= relief target already)


# ---------------------------------------------------------------- geometry helpers

def heavy_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]


def heavy_xyz(mol):
    return mol.GetConformer().GetPositions()[heavy_indices(mol)]


def heavy_centroid(mol):
    return heavy_xyz(mol).mean(axis=0)


def _with_xyz(mol, xyz):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, (x, y, z) in enumerate(np.asarray(xyz, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return out


def _single_conformer(mol):
    out = Chem.Mol(mol)
    if out.GetNumConformers() != 1:
        conf = Chem.Conformer(mol.GetConformer())
        out.RemoveAllConformers()
        out.AddConformer(conf, assignId=True)
    return out


def _tree(target):
    if isinstance(target, cKDTree):
        return target
    return cKDTree(np.asarray(target, dtype=float).reshape(-1, 3))


def min_dist(lig_xyz, target_xyz):
    """Minimum distance between ligand atoms and target atoms (inf when the target is empty).
    target_xyz may be an (n, 3) array or a prebuilt cKDTree."""
    tree = _tree(target_xyz)
    if tree.n == 0:
        return float("inf")
    d, _ = tree.query(np.asarray(lig_xyz, dtype=float).reshape(-1, 3), k=1)
    return float(np.min(d))


def contacts(lig_xyz, target_xyz, cutoff=CONTACT_CUTOFF):
    """Number of target atoms within cutoff of any ligand atom."""
    tree = _tree(target_xyz)
    if tree.n == 0:
        return 0
    hits = tree.query_ball_point(np.asarray(lig_xyz, dtype=float).reshape(-1, 3), r=cutoff)
    return len(set().union(*hits))


def n_within(lig_xyz, target_xyz, cutoff=CLOSE_CUTOFF):
    """Number of ligand atoms with a target atom closer than cutoff."""
    tree = _tree(target_xyz)
    if tree.n == 0:
        return 0
    d, _ = tree.query(np.asarray(lig_xyz, dtype=float).reshape(-1, 3), k=1)
    return int(np.sum(d < cutoff))


def effective_clash(clash, crystal_min_dist):
    """Clash threshold for a complex: a crystal pose that already clashes relaxes it."""
    return float(min(clash, crystal_min_dist))


def symm_rmsd(ref, probe):
    """Heavy-atom symmetry-corrected RMSD without alignment."""
    return float(rdMolAlign.CalcRMS(Chem.RemoveHs(probe), Chem.RemoveHs(ref),
                                    maxMatches=MAX_MATCHES))


# ---------------------------------------------------------------- perturbations

def _axis_angle_matrix(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _random_axis(rng):
    while True:
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n > 1e-8:
            return v / n


def _random_rotation(rng):
    """Uniform random rotation (normalised Gaussian quaternion)."""
    while True:
        q = rng.normal(size=4)
        n = np.linalg.norm(q)
        if n > 1e-8:
            break
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform(mol, rot, trans, center):
    xyz = mol.GetConformer().GetPositions()
    return _with_xyz(mol, (xyz - center) @ rot.T + center + trans)


def rigid_perturb(mol, rng, max_angle_deg, trans_sd, min_angle_deg=0.0):
    """Rotate by U(min_angle, max_angle) degrees about a random axis through the heavy-atom
    centroid, then translate by N(0, trans_sd) per axis. All atoms (Hs too) move."""
    axis = _random_axis(rng)
    angle = np.deg2rad(rng.uniform(min_angle_deg, max_angle_deg))
    trans = rng.normal(0.0, trans_sd, size=3)
    return _transform(mol, _axis_angle_matrix(axis, angle), trans, heavy_centroid(mol))


def _side(mol, j, k):
    """Atoms reachable from k without passing through j (k included)."""
    seen = {k}
    stack = [k]
    while stack:
        a = stack.pop()
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            n = nb.GetIdx()
            if n != j and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


def _pick_neighbor(mol, a, exclude):
    nbs = sorted((nb.GetSymbol() == "H", nb.GetIdx()) for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() != exclude)
    return nbs[0][1] if nbs else None


def rotatable_torsions(mol):
    """(i, j, k, l) for every rotatable bond j-k: Lipinski.RotatableBondSmarts matched on the
    heavy-atom graph (so C-CH3 and C-OH do not count), bonds in rings skipped; k is the side with
    fewer atoms (the side SetDihedralRad moves); i, l are neighbours, heavy atoms preferred."""
    tmp = Chem.Mol(mol)
    for a in tmp.GetAtoms():
        a.SetIntProp("_orig_idx", a.GetIdx())
    hmol = Chem.RemoveHs(tmp)
    orig = [a.GetIntProp("_orig_idx") for a in hmol.GetAtoms()]
    out = []
    for b, c in sorted(hmol.GetSubstructMatches(Lipinski.RotatableBondSmarts)):
        j, k = orig[b], orig[c]
        bond = mol.GetBondBetweenAtoms(j, k)
        if bond is None or bond.IsInRing():
            continue
        if len(_side(mol, j, k)) > len(_side(mol, k, j)):
            j, k = k, j
        i, l = _pick_neighbor(mol, j, k), _pick_neighbor(mol, k, j)
        if i is None or l is None:
            continue
        out.append((i, j, k, l))
    return out


def torsion_perturb(mol, rng, sd_deg, torsions=None):
    """Add N(0, sd_deg) to the dihedral of every rotatable bond (in rotatable_torsions order)."""
    out = Chem.Mol(mol)
    if torsions is None:
        torsions = rotatable_torsions(out)
    if not torsions:
        return out
    deltas = rng.normal(0.0, np.deg2rad(sd_deg), size=len(torsions))
    conf = out.GetConformer()
    for (i, j, k, l), d in zip(torsions, deltas):
        cur = rdMolTransforms.GetDihedralRad(conf, i, j, k, l)
        if np.isfinite(cur):
            rdMolTransforms.SetDihedralRad(conf, i, j, k, l, cur + float(d))
    return out


def random_conformer(mol, rng):
    """ETKDG conformer of a copy (randomSeed drawn from rng), translated so that its heavy-atom
    centroid sits on the input's. None when embedding fails."""
    seed = int(rng.integers(0, 2**31 - 1))
    m = Chem.Mol(mol)
    if AllChem.EmbedMolecule(m, randomSeed=seed) < 0:
        return None
    m = _single_conformer(m)
    shift = heavy_centroid(mol) - heavy_centroid(m)
    return _with_xyz(m, m.GetConformer().GetPositions() + shift)


# ---------------------------------------------------------------- clash relief

def clash_score(lig_xyz, target, r0):
    """Soft repulsion score sum(max(0, r0 - d_ij)^2) over ligand atoms i and their nearest
    RELIEF_NEIGHBOURS target atoms j (0 exactly when min_dist >= r0)."""
    tree = _tree(target)
    if tree.n == 0:
        return 0.0
    d, _ = tree.query(np.asarray(lig_xyz, dtype=float).reshape(-1, 3),
                      k=min(RELIEF_NEIGHBOURS, tree.n), distance_upper_bound=r0)
    d = d[d < r0]
    return float(np.sum((r0 - d) ** 2))


def relieve_clash(mol, target, rng, r0, anchor, max_steps=RELIEF_STEPS_MAX,
                  step_deg=RELIEF_STEP_DEG, step_A=RELIEF_STEP_A, max_shift=RELIEF_CENTROID_MAX):
    """Greedy rigid-body Monte Carlo on the soft repulsion score with target distance r0 (see module
    docstring). Returns (mol, steps, ok): the moved copy (all atoms, Hs too; the lowest-score state
    reached), the number of steps proposed (0 when min_dist >= r0 already) and whether
    min_dist >= r0 was reached. Every proposal draws from rng, so the result is deterministic
    given rng."""
    tree = _tree(target)
    xyz = mol.GetConformer().GetPositions()
    heavy = heavy_indices(mol)
    h = xyz[heavy]
    score = clash_score(h, tree, r0)
    if score == 0.0:
        return mol, 0, True
    anchor = np.asarray(anchor, dtype=float)
    c = h.mean(axis=0)
    cap = max(max_shift, float(np.linalg.norm(c - anchor)))
    max_angle = np.deg2rad(step_deg)
    for step in range(1, max_steps + 1):
        rot = _axis_angle_matrix(_random_axis(rng), rng.uniform(-max_angle, max_angle))
        new_c = c + _random_axis(rng) * rng.uniform(0.0, step_A)
        shift = float(np.linalg.norm(new_c - anchor))
        if shift > cap:
            continue
        new_h = (h - c) @ rot.T + new_c
        s = clash_score(new_h, tree, r0)
        if s < score:
            xyz = (xyz - c) @ rot.T + new_c
            h, c, score = new_h, new_c, s
            cap = max(max_shift, shift)
            if score == 0.0:
                return _with_xyz(mol, xyz), step, True
    return _with_xyz(mol, xyz), max_steps, False


# ---------------------------------------------------------------- generator

def generate_poses(mol_h, target_xyz, k_pos, k_neg, rng, rmsd_pos=(0.3, 2.0), rmsd_neg=4.0,
                   rmsd_neg_fallback=3.0, clash=2.2, contact_keep=0.5, max_attempts=60, *,
                   relief_steps_max=RELIEF_STEPS_MAX, relief_step_deg=RELIEF_STEP_DEG,
                   relief_step_A=RELIEF_STEP_A, relief_target_A=RELIEF_TARGET_A):
    """Near-native positives then decoys for one complex (see module docstring). Never raises for
    small or rigid ligands: a slot that cannot be filled gives no pose. relief_steps_max = 0
    disables the relief (candidates are then accepted or rejected as generated)."""
    base = _single_conformer(mol_h)
    tree = _tree(target_xyz)
    ref_heavy = Chem.RemoveHs(base)
    cryst_xyz = heavy_xyz(base)
    cryst_centroid = cryst_xyz.mean(axis=0)
    clash_eff = effective_clash(clash, min_dist(cryst_xyz, tree))
    relief_r0 = relief_target_A - (clash - clash_eff)
    n_cryst = contacts(cryst_xyz, tree)
    torsions = rotatable_torsions(base)

    def measure(m):
        xyz = heavy_xyz(m)
        n = contacts(xyz, tree)
        return min_dist(xyz, tree), (n / n_cryst if n_cryst > 0 else 1.0)

    def relieve(m):
        """(moved candidate, steps, min_dist, contact_frac) after relief."""
        steps = 0
        if relief_steps_max > 0:
            m, steps, _ok = relieve_clash(m, tree, rng, relief_r0, cryst_centroid,
                                          max_steps=relief_steps_max, step_deg=relief_step_deg,
                                          step_A=relief_step_A)
        md, frac = measure(m)
        return m, steps, md, frac

    def rms(m):
        return float(rdMolAlign.CalcRMS(Chem.RemoveHs(m), ref_heavy, maxMatches=MAX_MATCHES))

    def distinct(m, accepted):
        return all(symm_rmsd(p.mol, m) >= DISTINCT_RMSD for p in accepted)

    pos = []
    for _slot in range(k_pos):
        for attempt in range(1, max_attempts + 1):
            m = rigid_perturb(base, rng, NN_MAX_ANGLE_DEG, NN_TRANS_SD)
            m = torsion_perturb(m, rng, NN_TORSION_SD_DEG, torsions)
            m, steps, md, frac = relieve(m)
            if md < clash_eff:
                continue
            r = rms(m)
            if not (rmsd_pos[0] <= r <= rmsd_pos[1]) or not distinct(m, pos):
                continue
            pos.append(Pose(1, len(pos), "nn", m, r, md, frac, attempt, steps))
            break

    neg = []
    conf_ok = True
    for slot in range(k_neg):
        method = "flip" if slot % 2 == 0 else "conf"
        for attempt in range(1, 2 * max_attempts + 1):
            threshold = rmsd_neg if attempt <= max_attempts else rmsd_neg_fallback
            m, used = None, "flip"
            if method == "conf" and conf_ok:
                c = random_conformer(base, rng)
                if c is None:
                    conf_ok = False
                else:
                    used = "conf"
                    m = _transform(c, _random_rotation(rng), rng.normal(0.0, DECOY_TRANS_SD, size=3),
                                   heavy_centroid(c))
            if m is None:
                m = rigid_perturb(base, rng, FLIP_ANGLE_DEG[1], DECOY_TRANS_SD,
                                  min_angle_deg=FLIP_ANGLE_DEG[0])
            m, steps, md, frac = relieve(m)
            if md < clash_eff or frac < contact_keep:
                continue
            r = rms(m)
            if r < threshold or not distinct(m, neg):
                continue
            neg.append(Pose(2, len(neg), used, m, r, md, frac, attempt, steps))
            break
    return pos + neg
