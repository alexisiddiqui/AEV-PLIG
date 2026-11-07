"""Overlay placement for soft-negative scheme A (docs/plans/2026-11-05-aevplig-soft-negatives-AC.md,
Task A): shape twins (kind 4) and the regenerated cognate ligand (kind 3) go through one path, so that
nothing but the ligand's chemistry tells them apart.

Path (place(); the probe is the twin for kind 4 and the anchor's own ligand for kind 3):
1. Conformers: AllChem.EmbedMultipleConfs(copy of the probe with its Hs, numConfs=n_confs (10),
   randomSeed=int(rng.integers(2**31))); when none embeds, once more with useRandomCoords=True and the
   same seed. The probe's input coordinates are never used (the crystal conformer of the anchor is not
   reused for kind 3).
2. Overlay: for every conformer c, rdMolAlign.GetCrippenO3A(probe, ref, prbCid=c).Align(), ref = the
   anchor's crystal ligand with Hs (its first conformer); the conformer with the best O3A Score() is kept
   (ties: lowest conformer id) as a one-conformer copy of the probe (atom order and Hs unchanged).
3. Relief: pose_augment.relieve_clash on the protein heavy atoms, the arm-4 rigid-body soft repulsion with
   r0 = relief_target_A (3.0 A) - (clash - clash_eff), budget relief_steps_max (300) steps, heavy-atom
   centroid kept within 2.5 A of the crystal ligand's (or no further than the overlay put it).
4. Acceptance (arm 4): min_dist >= clash_eff = min(clash (2.2 A), crystal min_dist) and
   contact_frac >= contact_keep (0.5), contact_frac = protein atoms within 4.0 A of the pose's heavy atoms
   / the same count for the crystal pose (1.0 when the crystal pose has none).
An attempt that fails at 1, 2 or 4 is repeated with a new seed, up to max_overlays (3) attempts; after
that the placement has no pose (ok False, mol None, reason of the last attempt: "embed_failed",
"overlay_failed", "clash" or "contacts") and the measurements of the last candidate. Nothing raises for
a ligand that cannot be placed. Every random draw comes from rng, so a placement is deterministic given
rng (EmbedMultipleConfs runs single-threaded).
Distances use ligand heavy atoms (symbol != "H", as the AEV code) against the protein heavy atoms given
(an (n, 3) array or a cKDTree). regenerate_positive also records rmsd_to_crystal: heavy-atom
symmetry-corrected RMSD to the crystal pose without alignment (pose_augment.symm_rmsd); for a twin it is
NaN (different molecules).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from pose_augment import (CLOSE_CUTOFF, RELIEF_CENTROID_MAX, RELIEF_STEPS_MAX, RELIEF_TARGET_A, _tree,
                          contacts, effective_clash, heavy_xyz, min_dist, n_within, relieve_clash, symm_rmsd)

N_CONFS = 10
MAX_OVERLAYS = 3
CLASH = 2.2
CONTACT_KEEP = 0.5
REJECT_REASONS = ("embed_failed", "overlay_failed", "clash", "contacts")
__all__ = ["Placement", "embed_conformers", "overlay", "place", "place_twin", "regenerate_positive",
           "N_CONFS", "MAX_OVERLAYS", "CLASH", "CONTACT_KEEP", "RELIEF_CENTROID_MAX", "RELIEF_STEPS_MAX",
           "RELIEF_TARGET_A"]


@dataclass
class Placement:
    mol: object            # placed copy of the probe (one conformer, Hs kept); None when not accepted
    ok: bool
    reason: str            # "" when accepted, else one of REJECT_REASONS (last attempt)
    o3a_score: float
    min_dist: float        # min ligand-heavy to protein-heavy distance after relief (A)
    contact_frac: float
    n_within_3A: int       # ligand heavy atoms with a protein heavy atom closer than 3.0 A
    attempts: int          # overlay attempts used (1..max_overlays)
    relief_steps: int      # relief Monte Carlo steps of the last attempt (0: none needed)
    centroid_shift: float  # heavy-atom centroid distance to the crystal ligand's (A)
    rmsd_to_crystal: float = float("nan")


def _seed(rng):
    return int(rng.integers(2**31))


def embed_conformers(mol_h, rng, n_confs=N_CONFS):
    """(copy of mol_h holding only the new conformers, conformer ids), or (None, []) when ETKDG fails."""
    m = Chem.Mol(mol_h)
    seed = _seed(rng)
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, randomSeed=seed))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, randomSeed=seed, useRandomCoords=True))
    if not cids:
        return None, []
    return m, cids


def _keep_conformer(mol, cid):
    out = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetConformer(int(cid)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def _overlay(anchor_mol_h, probe_mol_h, rng, n_confs):
    """(overlaid probe copy, O3A score, "") or (None, nan, "embed_failed" | "overlay_failed")."""
    m, cids = embed_conformers(probe_mol_h, rng, n_confs)
    if m is None:
        return None, float("nan"), "embed_failed"
    best, best_score = None, -np.inf
    for c in cids:
        try:
            o3a = rdMolAlign.GetCrippenO3A(m, anchor_mol_h, prbCid=c)
            o3a.Align()
            s = float(o3a.Score())
        except (RuntimeError, ValueError):
            continue
        if np.isfinite(s) and s > best_score:
            best, best_score = c, s
    if best is None:
        return None, float("nan"), "overlay_failed"
    return _keep_conformer(m, best), best_score, ""


def overlay(anchor_mol_h, twin_mol_h, rng, n_confs=N_CONFS):
    """(probe copy with the best-scoring O3A-overlaid conformer, its O3A score); (None, nan) when no
    conformer embeds or every alignment fails."""
    mol, score, _reason = _overlay(anchor_mol_h, twin_mol_h, rng, n_confs)
    return mol, score


def place(anchor_mol_h, probe_mol_h, target_xyz, rng, n_confs=N_CONFS, max_overlays=MAX_OVERLAYS,
          clash=CLASH, contact_keep=CONTACT_KEEP, relief_target_A=RELIEF_TARGET_A,
          relief_steps_max=RELIEF_STEPS_MAX):
    """Overlay probe_mol_h onto the anchor's crystal ligand inside the pocket (module docstring)."""
    tree = _tree(target_xyz)
    cryst = heavy_xyz(anchor_mol_h)
    centroid = cryst.mean(axis=0)
    clash_eff = effective_clash(clash, min_dist(cryst, tree))
    r0 = relief_target_A - (clash - clash_eff)
    n_cryst = contacts(cryst, tree)
    nan = float("nan")
    last = None
    for attempt in range(1, max_overlays + 1):
        mol, score, reason = _overlay(anchor_mol_h, probe_mol_h, rng, n_confs)
        if mol is None:
            last = Placement(None, False, reason, nan, nan, nan, -1, attempt, 0, nan)
            continue
        steps = 0
        if relief_steps_max > 0:
            mol, steps, _ok = relieve_clash(mol, tree, rng, r0, centroid, max_steps=relief_steps_max)
        xyz = heavy_xyz(mol)
        md = min_dist(xyz, tree)
        frac = contacts(xyz, tree) / n_cryst if n_cryst > 0 else 1.0
        reason = "clash" if md < clash_eff else ("contacts" if frac < contact_keep else "")
        res = Placement(None if reason else mol, not reason, reason, score, md, frac,
                        n_within(xyz, tree, CLOSE_CUTOFF), attempt, steps,
                        float(np.linalg.norm(xyz.mean(axis=0) - centroid)))
        if not reason:
            return res
        last = res
    return last


def place_twin(anchor_mol_h, twin_mol_h, target_xyz, rng, **kw):
    """Kind 4: the twin ligand through place()."""
    return place(anchor_mol_h, twin_mol_h, target_xyz, rng, **kw)


def regenerate_positive(anchor_mol_h, rng, target_xyz, **kw):
    """Kind 3: the anchor's own ligand through place() (its own ETKDG conformers), with rmsd_to_crystal."""
    res = place(anchor_mol_h, anchor_mol_h, target_xyz, rng, **kw)
    if res.ok:
        res.rmsd_to_crystal = symm_rmsd(anchor_mol_h, res.mol)
    return res
