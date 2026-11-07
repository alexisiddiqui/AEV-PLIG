"""In-place property edits of the cognate ligand (scheme C): contact-site edits as negatives (kind 6),
solvent-exposed edits as label-preserving controls (kind 7) and a null edit as the matched positive
(kind 5). Plan: docs/plans/2026-11-05-aevplig-soft-negatives-AC.md, Task C.

Rules (RULES, in this order; reaction SMARTS with atom maps, run on the heavy-atom graph):
  cooh_to_me       R-C(=O)O(H/-) -> R-CH3: the carboxyl C stays (re-typed sp3), both O are removed
  amine_to_me      R-NH2 / R-NH3+ on carbon -> R-CH3 (N re-typed to C, charge 0); amides, amidines,
                   guanidines and thioamides (N on a C=O/N/S carbon) are not amines
  phenyl_to_chx    benzene ring (six aromatic carbons, each in one ring only, <= 2 ring substituents,
                   MATCH_FILTERS) -> cyclohexane: atoms aliphatic, ring bonds single, Hs added
  oh_to_f          hydroxyl on carbon (not a carboxylic/imidic/thio acid OH) -> F
  nh_to_o          neutral aromatic [nH] in a five-membered ring -> aromatic o (pyrrole -> furan,
                   imidazole -> oxazole, indole -> benzofuran); a product that does not sanitise is
                   skipped (Site.error = "not_sanitisable")
  amide_nh_to_nme  secondary amide C(=O)[NH]C -> C(=O)N(CH3)C (the donor H is replaced by a methyl)

Topology (find_sites): the reaction runs on the heavy-atom graph (Chem.RemoveAllHs of the input), each
distinct set of template atoms is one Site. Atoms the template touches (mapped and new) get their
explicit-H count reset so that RDKit recomputes implicit Hs from valence; the product is sanitised
(formal charges are those of the product template: +0 on re-typed atoms; a removed carboxylate O
takes its -1 with it). Product atoms are renumbered so that retained atoms keep the input order,
followed by new atoms; explicit Hs are added afterwards (Chem.AddHs). A retained heavy atom is
"unchanged" when its element, formal charge, aromaticity, H count, degree and neighbours are all as in
the input; the edited group (Site.edited_atoms, input indices) is every template atom that changed or
was removed. Site classification and the MMFF restraints use that split.

Placement (apply_edit / null_edit, one path for both):
  1. coordinate-mapped embedding of the edited molecule (Hs explicit): AllChem.EmbedMolecule with
     coordMap = every retained heavy atom (unchanged and re-typed) at its input coordinates,
     useRandomCoords=True (with random starting coordinates RDKit reproduces the mapped coordinates
     exactly; with the default eigenvector start, ~1/3 of FEP benchmark ligands came back mirrored or
     distorted by > 0.3 A), randomSeed from the anchor rng. Fallbacks, in order, recorded in
     place_method: the same with ignoreSmoothingFailures=True ("embed_ign"); coordMap = unchanged heavy
     atoms only ("embed_unch"); for an edit that adds no heavy atom, the retained heavy atoms at their
     input coordinates plus Chem.AddHs(addCoords=True) ("addhs"); else EditFailure("embed_failed").
  2. AlignMol onto the mapped atoms, then the mapped heavy atoms are set to their input coordinates
     exactly (so the restraint reference below is the crystal pose).
  3. MMFF94 (MMFFGetMoleculeForceField) with a position constraint (MMFFAddPositionConstraint(i, 0.0,
     100.0)) on every unchanged heavy atom, Minimize(maxIts=200). Re-typed retained atoms that were
     embedded at their input coordinates get a flat-bottom constraint (MMFFAddPositionConstraint(i,
     RETYPED_TOL = 0.5 A, 100.0)): free to relax within 0.5 A (the cyclohexane chair forms from the
     planar ring), but the group cannot drift (with no restraint a para-disubstituted ring rotated
     about its axis by ~1 A RMSD in vacuo, which moves the edit off its classified site). New atoms
     and all Hs are free. A molecule MMFF cannot type
     (B, Se, metals) uses UFF with the same constraints ("+uff"); neither: EditFailure("ff_untypable").
     mmff_restrained_rmsd = RMSD of the unchanged heavy atoms after this step vs the input pose.
  4. With a target (protein heavy atoms): pose_augment.relieve_clash (rigid-body soft repulsion to
     r0_eff = 3.0 A - (2.2 - clash_eff)), then the arm-4 acceptance: min_dist >= clash_eff =
     min(2.2, crystal min_dist) else EditFailure("clash"), contact_frac >= 0.5 of the crystal's
     contacts (4.0 A) else EditFailure("contacts_lost"). rmsd_to_crystal = RMSD of the unchanged heavy
     atoms after relief vs the input pose (no alignment); for the null edit that is every heavy atom.
The null edit (kind 5) is the same path with no reaction: the heavy-atom graph, re-protonated, every
heavy atom in the coordMap and restrained.

Site classes (classify_site, edited group at its input coordinates; protein = pocket_fields
protein_fields, standard-residue heavy atoms with channel weights):
  contact    complementary protein atoms within the rule's radius of any edited atom, at least
             min_count of them (SITE_RULES): acid -> pos channel > 0 within 4.0 A; amine -> neg > 0
             within 4.0 A; phenyl -> >= 3 atoms with hydrophobicity > 0.1 (apolar C/S; C next to a
             heteroatom is ~0.04) or aromatic > 0 within 4.5 A AND at least one of them aromatic
             (PHE/TYR/TRP/HIS ring atom; partner "aromatic"); a ring that passes only the apolar
             count is "ambiguous" (coordinator decision 2026-11-05: the apolar count alone marked
             54/57 FEP rings as contact); OH, aromatic NH, amide NH -> acceptor > 0 within 3.5 A
  exposed    no protein heavy atom within exposed_radius (default EXPOSED_RADIUS = 3.5 A; the plan's
             4.5 A gave 1 exposed site in 177 on the 92 FEP anchors, coordinator decision 2026-11-05)
             of any edited atom; ATOM records (protein_fields) and extra_xyz, the non-water HETATM
             heavy atoms of the protein file (cofactors, metals; load_hetatm_xyz), both count, so an
             edit next to a cofactor is never a label-preserving control. generate_edits also
             classifies at EXPOSURE_REPORT_RADII (3.5, 4.0, 4.5 A) and logs the exposed-site counts
             (log["exposed_sites_by_radius"]); only exposed_radius decides.
  ambiguous  otherwise (skipped)
generate_edits: per anchor 1 null edit (kind 5; when it fails the anchor gives nothing, because
negatives need their matched positive), then up to k_contact contact edits (kind 6) and up to
k_exposed exposed edits (kind 7). Within a class the candidates are put in rng order and taken
greedily, an unused rule first ("prefer different rules"); a candidate whose placement fails is
recorded and the next one tried, at most max_tries placements per class.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdMolAlign
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

from pocket_fields import ACC, ARO, HYD, NEG, POS
from pose_augment import (CLOSE_CUTOFF, CONTACT_CUTOFF, RELIEF_TARGET_A, _single_conformer, _tree,
                          contacts, effective_clash, heavy_xyz, min_dist, n_within, relieve_clash)

RULES = [
    ("cooh_to_me", "[#6:1]-[CX3:2](=[OX1])-[OX1H0-,OX2H1]", "[#6:1]-[C+0:2]"),
    ("amine_to_me", "[#6;!$([#6]=[#7,#8,#16]):1]-[NX3H2+0,NX4H3+:2]", "[#6:1]-[C+0:2]"),
    ("phenyl_to_chx", "[c;R1:1]1:[c;R1:2]:[c;R1:3]:[c;R1:4]:[c;R1:5]:[c;R1:6]:1",
     "[C:1]1-[C:2]-[C:3]-[C:4]-[C:5]-[C:6]-1"),
    ("oh_to_f", "[#6;!$([#6]=[#7,#8,#16]):1]-[OX2H1:2]", "[#6:1]-[F+0:2]"),
    ("nh_to_o", "[nH1+0;r5:1]", "[o+0:1]"),
    ("amide_nh_to_nme", "[CX3:1](=[OX1:2])-[NX3H1+0:3]-[#6:4]", "[C:1](=[O:2])-[N+0:3](-[CH3])-[#6:4]"),
]
RULE_NAMES = tuple(r[0] for r in RULES)
MAX_SUBSTITUENTS = 2         # phenyl_to_chx: ring atoms with a heavy neighbour outside the ring
MAX_PRODUCTS = 10_000        # RunReactants cap (a benzene ring gives 12 equivalent products)
EMBED_MAX_ATTEMPTS = 30
MMFF_MAX_ITS = 200
RESTRAINT_K = 100.0          # kcal/mol/A^2
RETYPED_TOL = 0.5            # A, flat-bottom tolerance of the restraint on re-typed retained atoms
CLASH = 2.2
CONTACT_KEEP = 0.5
EXPOSED_RADIUS = 3.5
EXPOSURE_REPORT_RADII = (3.5, 4.0, 4.5)
HYDROPHOBIC_MIN = 0.1        # protein Crippen contribution counted as hydrophobic (phenyl contact)
MAX_TRIES = 6                # placements per site class and anchor
KIND_NULL, KIND_CONTACT, KIND_EXPOSED = 5, 6, 7
WATER_RESIDUES = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_ORIG = "_le_orig_idx"


@dataclass(frozen=True)
class SiteRule:
    partner: str      # pos | neg | acceptor | hydrophobic
    radius: float     # A, from any edited atom
    min_count: int
    require_aromatic: bool = False   # contact also needs >= 1 aromatic protein atom among the partners


SITE_RULES = {
    "cooh_to_me": SiteRule("pos", 4.0, 1),
    "amine_to_me": SiteRule("neg", 4.0, 1),
    "phenyl_to_chx": SiteRule("hydrophobic", 4.5, 3, require_aromatic=True),
    "oh_to_f": SiteRule("acceptor", 3.5, 1),
    "nh_to_o": SiteRule("acceptor", 3.5, 1),
    "amide_nh_to_nme": SiteRule("acceptor", 3.5, 1),
}


class EditFailure(Exception):
    """An edit that cannot be made or placed; reason is a short key counted in the summaries."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Site:
    rule: str
    match: tuple                 # input indices of the reactant-template atoms (sorted)
    edited_atoms: tuple          # input indices of template atoms that changed or were removed
    removed_atoms: tuple = ()    # input indices removed by the edit
    product: Chem.Mol = None     # sanitised edited heavy-atom molecule, no conformer
    retained: tuple = ()         # (product idx, input idx) of every retained heavy atom
    unchanged: tuple = ()        # (product idx, input idx) of the unchanged heavy atoms
    new_atoms: tuple = ()        # product indices of atoms the edit adds
    error: str = ""              # "" or the reason the product is unusable


@dataclass
class EditResult:
    site: Site
    mol: Chem.Mol                # edited molecule, explicit Hs, one conformer
    unchanged: list              # (output idx, input idx) of the unchanged heavy atoms
    atom_map: dict               # input idx -> output idx for every retained heavy atom
    mmff_restrained_rmsd: float
    place_method: str
    formal_charge: int
    dcharge: int
    n_heavy: int
    rmsd_to_crystal: float
    min_dist: float = float("nan")
    contact_frac: float = float("nan")
    n_within_3A: int = -1
    relief_steps: int = 0


@dataclass
class EditPose:
    kind: int                    # 5 null-edit positive, 6 contact edit, 7 exposed edit
    k: int                       # index among accepted poses of this kind (contiguous)
    rule: str                    # rule name, "null" for kind 5
    site_class: str              # null | contact | exposed
    mol: Chem.Mol
    edited_atoms: tuple
    n_edited_atoms: int
    n_contacts: int              # complementary protein atoms counted by classify_site
    partner_type: str
    rmsd_to_crystal: float
    mmff_restrained_rmsd: float
    min_dist: float
    contact_frac: float
    n_within_3A: int
    relief_steps: int
    formal_charge: int
    dcharge: int
    n_heavy: int
    place_method: str


@dataclass
class TargetContext:
    tree: cKDTree
    centroid: np.ndarray
    clash_eff: float
    relief_r0: float
    n_cryst: int
    contact_keep: float = CONTACT_KEEP


# ---------------------------------------------------------------- topology

def _rule(rule):
    if isinstance(rule, str):
        for r in RULES:
            if r[0] == rule:
                return r
        raise KeyError(f"unknown edit rule {rule!r}")
    name, reactant, product = rule
    return name, reactant, product


@lru_cache(maxsize=None)
def _reaction(reactant, product):
    rxn = rdChemReactions.ReactionFromSmarts(f"{reactant}>>{product}")
    rxn.Initialize()
    return rxn


def _max_two_substituents(heavy, atoms):
    ring = set(atoms)
    n_sub = sum(any(nb.GetIdx() not in ring for nb in heavy.GetAtomWithIdx(i).GetNeighbors())
                for i in ring)
    return n_sub <= MAX_SUBSTITUENTS


MATCH_FILTERS = {"phenyl_to_chx": _max_two_substituents}


def _heavy_graph(base):
    """(heavy-atom molecule with the input conformer, list heavy idx -> input idx)."""
    m = Chem.Mol(base)
    for a in m.GetAtoms():
        a.SetIntProp(_ORIG, a.GetIdx())
    heavy = Chem.RemoveAllHs(m)
    return heavy, [a.GetIntProp(_ORIG) for a in heavy.GetAtoms()]


def _signature(atom):
    return (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsAromatic(), atom.GetTotalNumHs(),
            atom.GetDegree())


def _make_site(name, heavy, to_input, prod):
    """Site from one reaction product (template atoms identified via react_atom_idx / old_mapno)."""
    react, new, mapped = {}, [], set()
    for a in prod.GetAtoms():
        if a.HasProp("react_atom_idx"):
            react[a.GetIdx()] = a.GetIntProp("react_atom_idx")
            if a.HasProp("old_mapno"):
                mapped.add(react[a.GetIdx()])
        else:
            new.append(a.GetIdx())
    removed = sorted(set(range(heavy.GetNumAtoms())) - set(react.values()))
    template = sorted(mapped | set(removed))
    rw = Chem.RWMol(prod)
    for a in rw.GetAtoms():
        if a.HasProp("old_mapno") or not a.HasProp("react_atom_idx"):
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
    order = sorted(react, key=lambda i: react[i]) + new
    m = Chem.RenumberAtoms(rw.GetMol(), order)
    m.RemoveAllConformers()
    n_ret = len(react)
    site = Site(rule=name, match=tuple(to_input[h] for h in template), edited_atoms=(),
                removed_atoms=tuple(to_input[h] for h in removed),
                new_atoms=tuple(range(n_ret, m.GetNumAtoms())))
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001 - valence / kekulisation failures make the site unusable
        site.error = "not_sanitisable"
        site.edited_atoms = tuple(sorted(set(site.match)))
        return site
    heavy_of = [m.GetAtomWithIdx(i).GetIntProp("react_atom_idx") for i in range(n_ret)]
    changed = []
    for i, h in enumerate(heavy_of):
        a, b = m.GetAtomWithIdx(i), heavy.GetAtomWithIdx(h)
        nb_a = {heavy_of[n.GetIdx()] if n.GetIdx() < n_ret else -1 for n in a.GetNeighbors()}
        nb_b = {n.GetIdx() for n in b.GetNeighbors()}
        if _signature(a) != _signature(b) or nb_a != nb_b:
            changed.append(i)
    site.product = m
    site.retained = tuple((i, to_input[h]) for i, h in enumerate(heavy_of))
    site.unchanged = tuple((i, to_input[h]) for i, h in enumerate(heavy_of) if i not in set(changed))
    site.edited_atoms = tuple(sorted([to_input[heavy_of[i]] for i in changed] + list(site.removed_atoms)))
    return site


def find_sites(mol_h, rule=None):
    """Every distinct site of one rule (name or (name, reactant, product)) or of all RULES, in rule
    order then RDKit match order. Unusable sites are kept with Site.error set, so indices are stable."""
    base = _single_conformer(mol_h) if mol_h.GetNumConformers() else Chem.Mol(mol_h)
    heavy, to_input = _heavy_graph(base)
    rules = RULES if rule is None else [_rule(rule)]
    sites = []
    for name, reactant, product in rules:
        seen = set()
        for prods in _reaction(reactant, product).RunReactants((heavy,), MAX_PRODUCTS):
            prod = prods[0]
            key = frozenset(a.GetIntProp("react_atom_idx") for a in prod.GetAtoms()
                            if a.HasProp("old_mapno") and a.HasProp("react_atom_idx"))
            key |= frozenset(set(range(heavy.GetNumAtoms())) -
                             {a.GetIntProp("react_atom_idx") for a in prod.GetAtoms()
                              if a.HasProp("react_atom_idx")})
            if key in seen:
                continue
            seen.add(key)
            filt = MATCH_FILTERS.get(name)
            if filt is not None and not filt(heavy, key):
                continue
            sites.append(_make_site(name, heavy, to_input, prod))
    return sites


def _null_site(base):
    heavy, to_input = _heavy_graph(base)
    m = Chem.Mol(heavy)
    m.RemoveAllConformers()
    pairs = tuple((i, to_input[i]) for i in range(m.GetNumAtoms()))
    return Site(rule="null", match=(), edited_atoms=(), product=m, retained=pairs, unchanged=pairs)


# ---------------------------------------------------------------- placement

def target_context(mol_h, target_xyz, clash=CLASH, relief_target=RELIEF_TARGET_A,
                   contact_keep=CONTACT_KEEP):
    """Relief/acceptance constants of one complex, as pose_augment.generate_poses computes them."""
    tree = _tree(target_xyz)
    cryst = heavy_xyz(mol_h)
    clash_eff = effective_clash(clash, min_dist(cryst, tree))
    return TargetContext(tree=tree, centroid=cryst.mean(axis=0), clash_eff=clash_eff,
                         relief_r0=relief_target - (clash - clash_eff), n_cryst=contacts(cryst, tree),
                         contact_keep=contact_keep)


def _set_xyz(mol, pairs, ref_xyz):
    conf = mol.GetConformer()
    for i, j in pairs:
        x, y, z = ref_xyz[j]
        conf.SetAtomPosition(int(i), Point3D(float(x), float(y), float(z)))


def _embed(site, ref, ref_xyz, seed):
    """(molecule with explicit Hs and one conformer, mapped pairs, mode); EditFailure when all fail."""
    m = Chem.AddHs(site.product)
    attempts = [("embed", site.retained, {}),
                ("embed_ign", site.retained, {"ignoreSmoothingFailures": True})]
    if site.unchanged and len(site.unchanged) < len(site.retained):
        attempts.append(("embed_unch", site.unchanged, {"ignoreSmoothingFailures": True}))
    for mode, pairs, extra in attempts:
        if not pairs:
            continue
        e = Chem.Mol(m)
        cmap = {int(i): Point3D(*map(float, ref_xyz[j])) for i, j in pairs}
        cid = AllChem.EmbedMolecule(e, coordMap=cmap, randomSeed=seed, useRandomCoords=True,
                                    maxAttempts=EMBED_MAX_ATTEMPTS, **extra)
        if cid >= 0:
            rdMolAlign.AlignMol(e, ref, atomMap=[(int(i), int(j)) for i, j in pairs])
            _set_xyz(e, pairs, ref_xyz)
            return e, pairs, mode
    if site.new_atoms:
        raise EditFailure("embed_failed")
    h = Chem.Mol(site.product)
    conf = Chem.Conformer(h.GetNumAtoms())
    conf.Set3D(True)
    h.AddConformer(conf, assignId=True)
    _set_xyz(h, site.retained, ref_xyz)
    return Chem.AddHs(h, addCoords=True), site.retained, "addhs"


def _restrained_minimise(m, restrained, max_its, loose=()):
    """Minimise in place with position constraints on `restrained` (exact) and `loose` (flat-bottom,
    RETYPED_TOL); returns the force field name."""
    props = AllChem.MMFFGetMoleculeProperties(m)
    ff, name = None, ""
    if props is not None:
        ff, name = AllChem.MMFFGetMoleculeForceField(m, props), "mmff"
        if ff is not None:
            add = ff.MMFFAddPositionConstraint
    if ff is None and AllChem.UFFHasAllMoleculeParams(m):
        ff, name = AllChem.UFFGetMoleculeForceField(m), "uff"
        if ff is not None:
            add = ff.UFFAddPositionConstraint
    if ff is None:
        raise EditFailure("ff_untypable")
    for i in restrained:
        add(int(i), 0.0, RESTRAINT_K)
    for i in loose:
        add(int(i), RETYPED_TOL, RESTRAINT_K)
    ff.Initialize()
    ff.Minimize(maxIts=max_its)
    return name


def _rmsd(mol, pairs, ref_xyz):
    if not pairs:
        return float("nan")
    xyz = mol.GetConformer().GetPositions()
    d = np.array([xyz[i] - ref_xyz[j] for i, j in pairs])
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _run_site(base, site, rng, target, max_its=MMFF_MAX_ITS):
    if site.error:
        raise EditFailure(site.error)
    if rng is None:
        rng = np.random.default_rng(0)
    ref_xyz = base.GetConformer().GetPositions()
    seed = int(rng.integers(0, 2**31 - 1))
    m, pairs, mode = _embed(site, base, ref_xyz, seed)
    fixed = {i for i, _ in site.unchanged}
    ff = _restrained_minimise(m, sorted(fixed), max_its,
                              loose=sorted(i for i, _ in pairs if i not in fixed))
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        raise EditFailure("not_sanitisable") from None
    mmff_rmsd = _rmsd(m, site.unchanged, ref_xyz)
    q0 = Chem.GetFormalCharge(base)
    q1 = Chem.GetFormalCharge(m)
    res = EditResult(site=site, mol=m, unchanged=list(site.unchanged),
                     atom_map={j: i for i, j in site.retained}, mmff_restrained_rmsd=mmff_rmsd,
                     place_method=f"{mode}+{ff}", formal_charge=q1, dcharge=q1 - q0,
                     n_heavy=sum(a.GetAtomicNum() > 1 for a in m.GetAtoms()), rmsd_to_crystal=mmff_rmsd)
    if target is None:
        return res
    ctx = target if isinstance(target, TargetContext) else target_context(base, target)
    m, steps, _ok = relieve_clash(m, ctx.tree, rng, ctx.relief_r0, ctx.centroid)
    xyz = heavy_xyz(m)
    md = min_dist(xyz, ctx.tree)
    frac = contacts(xyz, ctx.tree, CONTACT_CUTOFF) / ctx.n_cryst if ctx.n_cryst > 0 else 1.0
    if md < ctx.clash_eff:
        raise EditFailure("clash")
    if frac < ctx.contact_keep:
        raise EditFailure("contacts_lost")
    res.mol, res.relief_steps, res.min_dist, res.contact_frac = m, steps, md, frac
    res.n_within_3A = n_within(xyz, ctx.tree, CLOSE_CUTOFF)
    res.rmsd_to_crystal = _rmsd(m, site.unchanged, ref_xyz)
    return res


def apply_edit(mol_h, rule, match_idx=0, rng=None, target=None, max_its=MMFF_MAX_ITS):
    """Apply one rule at one site (match_idx: index into find_sites(mol_h, rule), or a Site) and place
    it in situ (module docstring). target: protein heavy-atom coordinates or a TargetContext, enabling
    relief + acceptance. Returns an EditResult; raises EditFailure(reason)."""
    base = _single_conformer(mol_h)
    if isinstance(match_idx, Site):
        site = match_idx
    else:
        sites = find_sites(base, rule)
        if not 0 <= match_idx < len(sites):
            raise EditFailure("no_match")
        site = sites[match_idx]
    return _run_site(base, site, rng, target, max_its)


def null_edit(mol_h, rng=None, target=None, max_its=MMFF_MAX_ITS):
    """The placement path with no change (kind 5): every heavy atom mapped and restrained."""
    base = _single_conformer(mol_h)
    return _run_site(base, _null_site(base), rng, target, max_its)


# ---------------------------------------------------------------- site classification

def _partner_mask(w, partner):
    if partner == "pos":
        return w[:, POS] > 0
    if partner == "neg":
        return w[:, NEG] > 0
    if partner == "acceptor":
        return w[:, ACC] > 0
    if partner == "hydrophobic":
        return (w[:, HYD] > HYDROPHOBIC_MIN) | (w[:, ARO] > 0)
    raise ValueError(f"unknown partner type {partner!r}")


def classify_site(edited_atom_idx, lig_xyz, protein_fields, rule, extra_xyz=None,
                  exposed_radius=EXPOSED_RADIUS):
    """("contact" | "exposed" | "ambiguous", n_contacts, partner_type) of an edited group at its
    input coordinates (module docstring). protein_fields: (xyz, W[, resid]) as pocket_fields
    returns; extra_xyz: further heavy atoms (HETATM) that count for exposure only; exposed_radius:
    the exposure distance (default 3.5 A)."""
    spec = SITE_RULES[rule if isinstance(rule, str) else rule[0]]
    pts = np.asarray(lig_xyz, dtype=float)[list(edited_atom_idx)].reshape(-1, 3)
    pxyz = np.asarray(protein_fields[0], dtype=float).reshape(-1, 3)
    w = np.asarray(protein_fields[1], dtype=float).reshape(-1, 7)
    mask = _partner_mask(w, spec.partner)
    hits = set()
    if mask.any() and len(pts):
        idx = np.flatnonzero(mask)
        for lst in cKDTree(pxyz[idx]).query_ball_point(pts, r=spec.radius):
            hits.update(idx[j] for j in lst)
    n = len(hits)
    if n >= spec.min_count:
        if not spec.require_aromatic:
            return "contact", n, spec.partner
        if any(w[i, ARO] > 0 for i in hits):
            return "contact", n, "aromatic"
        return "ambiguous", n, ""      # apolar contacts without an aromatic partner
    allxyz = pxyz if extra_xyz is None else np.vstack([pxyz, np.asarray(extra_xyz, float).reshape(-1, 3)])
    if min_dist(pts, allxyz) > exposed_radius:
        return "exposed", n, ""
    return "ambiguous", n, ""


def load_hetatm_xyz(path, exclude_xyz=None, exclude_cutoff=1.0):
    """Heavy-atom coordinates of the non-water HETATM records of a PDB file (first model; element
    from columns 77-78, else the first letter of the atom name). Atoms within exclude_cutoff of
    exclude_xyz (a copy of the ligand itself) are dropped."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("HETATM") or line[17:20].strip() in WATER_RESIDUES:
                continue
            elem = line[76:78].strip().upper() or line[12:16].strip()[:1].upper()
            if elem in ("H", "D"):
                continue
            try:
                rows.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    xyz = np.array(rows, dtype=float).reshape(-1, 3)
    if len(xyz) and exclude_xyz is not None and len(exclude_xyz):
        d, _ = cKDTree(np.asarray(exclude_xyz, float).reshape(-1, 3)).query(xyz, k=1)
        xyz = xyz[d > exclude_cutoff]
    return xyz


# ---------------------------------------------------------------- generator

def _pose(kind, k, res, site_class, n_contacts, partner):
    s = res.site
    return EditPose(kind=kind, k=k, rule=s.rule, site_class=site_class, mol=res.mol,
                    edited_atoms=s.edited_atoms, n_edited_atoms=len(s.edited_atoms), n_contacts=n_contacts,
                    partner_type=partner, rmsd_to_crystal=res.rmsd_to_crystal,
                    mmff_restrained_rmsd=res.mmff_restrained_rmsd, min_dist=res.min_dist,
                    contact_frac=res.contact_frac, n_within_3A=res.n_within_3A,
                    relief_steps=res.relief_steps, formal_charge=res.formal_charge, dcharge=res.dcharge,
                    n_heavy=res.n_heavy, place_method=res.place_method)


def _choose(cands, k_max, rng, base, ctx, max_tries, failures):
    """Greedy pick in rng order, a candidate of a rule not yet used first; failed placements are
    counted in failures[rule][reason] and the next candidate is tried."""
    if k_max <= 0 or not cands:
        return []
    remaining = [cands[i] for i in rng.permutation(len(cands))]
    chosen, used, tries = [], set(), 0
    while remaining and len(chosen) < k_max and tries < max_tries:
        pick = remaining.pop(next((j for j, c in enumerate(remaining) if c[0].rule not in used), 0))
        tries += 1
        try:
            res = _run_site(base, pick[0], rng, ctx)
        except EditFailure as e:
            failures.setdefault(pick[0].rule, Counter())[e.reason] += 1
            continue
        chosen.append((res, pick[1], pick[2]))
        used.add(pick[0].rule)
    return chosen


def generate_edits(mol_h, protein_fields, rng, k_contact=2, k_exposed=2, *, target_xyz=None,
                   extra_xyz=None, clash=CLASH, contact_keep=CONTACT_KEEP, max_tries=MAX_TRIES,
                   exposed_radius=EXPOSED_RADIUS, log=None):
    """Null edit + contact edits + exposed edits of one anchor (module docstring). target_xyz: the
    protein heavy atoms used for relief/acceptance/min_dist (default protein_fields' xyz); extra_xyz:
    HETATM heavy atoms for the exposure test. Never raises for ligands without sites or for failed
    placements; log (a dict, optional) receives n_sites, candidates[rule][class], failures[rule][reason],
    null_failure and exposed_sites_by_radius {"3.5": n, "4.0": n, "4.5": n} (sites classified exposed
    at each EXPOSURE_REPORT_RADII radius and at exposed_radius; only exposed_radius decides)."""
    log = {} if log is None else log
    failures = {}
    radii = sorted(set(EXPOSURE_REPORT_RADII) | {float(exposed_radius)})
    exposed_by_radius = {f"{r:.1f}": 0 for r in radii}
    log.update(n_sites=0, candidates={}, failures=failures, null_failure="",
               exposed_sites_by_radius=exposed_by_radius)
    base = _single_conformer(mol_h)
    if target_xyz is None:
        target_xyz = protein_fields[0]
    ctx = target_context(base, target_xyz, clash, contact_keep=contact_keep)
    try:
        null = null_edit(base, rng, target=ctx)
    except EditFailure as e:
        log["null_failure"] = e.reason
        return []
    poses = [_pose(KIND_NULL, 0, null, "null", 0, "")]
    lig_xyz = base.GetConformer().GetPositions()
    by_class = {"contact": [], "exposed": []}
    for site in find_sites(base):
        if site.error:
            failures.setdefault(site.rule, Counter())[site.error] += 1
            continue
        log["n_sites"] += 1
        cls, n, partner = classify_site(site.edited_atoms, lig_xyz, protein_fields, site.rule, extra_xyz,
                                        exposed_radius)
        for r in radii:
            c = cls if r == exposed_radius else classify_site(site.edited_atoms, lig_xyz, protein_fields,
                                                              site.rule, extra_xyz, r)[0]
            exposed_by_radius[f"{r:.1f}"] += c == "exposed"
        log["candidates"].setdefault(site.rule, Counter())[cls] += 1
        if cls in by_class:
            by_class[cls].append((site, n, partner))
    for kind, cls, k_max in ((KIND_CONTACT, "contact", k_contact), (KIND_EXPOSED, "exposed", k_exposed)):
        for k, (res, n, partner) in enumerate(_choose(by_class[cls], k_max, rng, base, ctx, max_tries,
                                                      failures)):
            poses.append(_pose(kind, k, res, cls, n, partner))
    return poses
