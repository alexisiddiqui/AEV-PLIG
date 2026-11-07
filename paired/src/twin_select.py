"""Shape-twin selection for soft-negative scheme A (docs/plans/2026-11-05-aevplig-soft-negatives-AC.md,
Task A).

A shape twin of a PDBbind anchor complex (P, L) is a training ligand L' with nearly the shape of L and
the worst chemical complementarity to P. Everything is read from fixed arrays (zernike.npz, shell mode,
order 12; paired_meta.csv; pocket_groups.csv), so the candidate list of an anchor is deterministic.

Vectors (channel order of pocket_fields.CHANNELS: occupancy, hydrophobicity, pos, neg, aromatic, donor,
acceptor; F arrays [N, 7, 49]):
- Standardisation: every (channel, invariant) column of ligand_F, and separately of pocket_F, is
  z-scored with the mean and sd over the pool rows (below); a column with sd < 1e-12 keeps scale 1.
- shape: occupancy channel of the standardised ligand_F (49 values), L2-normalised.
- chem: channels 1-6 of the standardised ligand_F (6 x 49), L2-normalised.
- pocket_comp: channels 1-6 of the standardised pocket_F with pos<->neg and donor<->acceptor swapped
  (hydrophobicity and aromatic kept), L2-normalised: the chemistry a ligand would need to match P.
- shape_dist(L, L') = 1 - shape(L).shape(L'); complementarity(P, L') = pocket_comp(P).chem(L').
- Scalars from ligand_total (signed channel sums over the kept atoms): logP = ligand_total[:, 1];
  formal charge = rint(ligand_total[:, 2] - ligand_total[:, 3]) (united-atom Gasteiger charges sum to the
  formal charge); aromatic ring count ~ ligand_total[:, 4] / 6, kept continuous; heavy atoms =
  n_ligand_atoms.

Rows: zernike.npz rows with ok True and finite descriptors whose unique_id is a kind-0 row of the meta
table (arm-4 pose rows, kind != 0, are ignored). The pool is the subset with split == "train", any
source; anchors may be any row with a descriptor (train or valid PDBbind rows in practice).

Per anchor, excluded from the pool before anything else (identity rules, not filters):
  (i)   rows in the anchor's group_id;
  (ii)  rows whose lig_key appears in any meta row of that group (known binders: any split, with or
        without a descriptor; this includes the anchor's own lig_key);
  (iii) rows sharing the anchor's lig_key (also when the anchor has no group);
  (iv)  rows whose anchor_pdb equals the anchor's (BindingNet / BindingDB complexes modelled on the
        anchor's structure, whatever group they were given).
Selection: the n_near = max(k, ceil(shape_frac * pool size)) nearest remaining rows by shape_dist (ties
by row order) form the neighbourhood. Its rows that pass the marginal filters
  |n_heavy' - n_heavy| <= size_tol * n_heavy, |logP' - logP| <= max_dlogp,
  |arom' - arom| <= max_darom, sign(charge') == sign(charge) (when same_charge_sign)
are ranked by complementarity ascending (ties: shape_dist, then row order), and the first k with distinct
lig_keys are the twins. dlogp, dcharge, darom are twin minus anchor.
An anchor with fewer than k twins gets what there is and never raises; info["reason"] is "ok" (k twins),
"short" (1..k-1), "no_twin" (info["empty_at"] names the step that emptied the funnel: "excluded", "size",
"logp", "arom", "charge") or "anchor_no_descriptor".

Shape distances come from blocked float32 matmuls of the cached unit vectors (select_twins_batch, block
rows of anchors at a time), never from a full distance matrix; select_twins is the same code with one
anchor.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

N_CHANNELS = 7
OCC, HYD, POS, NEG, ARO, DON, ACC = range(N_CHANNELS)
CHEM_CHANNELS = (HYD, POS, NEG, ARO, DON, ACC)
# the pocket channel that faces each ligand chemistry channel (same order as CHEM_CHANNELS)
COMPLEMENT_CHANNELS = (HYD, NEG, POS, ARO, ACC, DON)
AROMATIC_RING_ATOMS = 6.0
SD_FLOOR = 1e-12
DEFAULT_BLOCK = 256
POOL_SPLIT = "train"
REASONS = ("ok", "short", "no_twin", "anchor_no_descriptor")
FUNNEL = ("excluded", "size", "logp", "arom", "charge")
DESC_KEYS = ("unique_id", "source", "ok", "ligand_F", "pocket_F", "ligand_total", "n_ligand_atoms")


@dataclass
class TwinChoice:
    anchor_uid: str
    rank: int              # 0..k-1 in complementarity order
    twin_uid: str
    twin_source: str
    twin_lig_key: str
    twin_group_id: str
    twin_anchor_pdb: str
    shape_dist: float
    complementarity: float
    dlogp: float           # twin - anchor
    dcharge: int
    darom: float           # aromatic ring count (aromatic atoms / 6), twin - anchor
    n_heavy_anchor: int
    n_heavy_twin: int

    def as_dict(self):
        return asdict(self)


def _codes(values):
    """Integer codes of string values; '' gets -1 (never equal to anything)."""
    values = pd.Series(values, dtype=str).fillna("")
    codes, uniques = pd.factorize(values)
    codes = codes.astype(np.int64)
    empty = np.flatnonzero(np.asarray(uniques) == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    return codes, np.asarray(uniques, dtype=object)


def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class TwinPool:
    """Cached unit vectors and scalars of every row with a descriptor, the pool rows and the identity
    maps. Build it with build_pool or load_pool. Arrays are indexed by row (index[uid] -> row)."""

    def __init__(self, uid, source, split, shape, chem, pocket_comp, logp, charge, arom, n_heavy,
                 lig_key, group_id, anchor_pdb, group_lig_codes, lig_code, group_code, tmpl_code, stats):
        self.uid, self.source, self.split = uid, source, split
        self.shape, self.chem, self.pocket_comp = shape, chem, pocket_comp
        self.logp, self.charge, self.arom, self.n_heavy = logp, charge, arom, n_heavy
        self.lig_key, self.group_id, self.anchor_pdb = lig_key, group_id, anchor_pdb
        self.lig_code, self.group_code, self.tmpl_code = lig_code, group_code, tmpl_code
        self.group_lig_codes = group_lig_codes   # group code -> sorted unique lig codes of its meta rows
        self.stats = stats
        self.index = {u: i for i, u in enumerate(uid)}
        self.pool_rows = np.flatnonzero(split == POOL_SPLIT)
        self.pool_shape = np.ascontiguousarray(shape[self.pool_rows])
        self.pool_size = int(len(self.pool_rows))
        # identity codes of the pool rows, cached for the per-anchor exclusion masks
        self.pool_lig_code = lig_code[self.pool_rows]
        self.pool_group_code = group_code[self.pool_rows]
        self.pool_tmpl_code = tmpl_code[self.pool_rows]

    def __len__(self):
        return len(self.uid)


def build_pool(desc, meta, groups=None):
    """TwinPool from zernike.npz-like arrays and the meta / groups tables.

    desc: mapping with DESC_KEYS (other keys ignored). meta: DataFrame with unique_id, source, split,
    lig_key, anchor_pdb (kind optional; group_id optional when groups is None). groups: DataFrame with
    unique_id, group_id (pocket_groups.csv), merged onto meta when given."""
    meta = meta.copy()
    for col in ("unique_id", "source", "split", "lig_key", "anchor_pdb"):
        if col not in meta.columns:
            raise ValueError(f"meta has no {col!r} column")
        meta[col] = meta[col].fillna("").astype(str)
    if "kind" in meta.columns:
        meta = meta[meta["kind"].astype(str).str.strip() == "0"]
    if groups is not None:
        g = groups[["unique_id", "group_id"]].astype(str).drop_duplicates("unique_id", keep="first")
        meta = meta.drop(columns=["group_id"], errors="ignore").merge(g, on="unique_id", how="left")
    if "group_id" not in meta.columns:
        meta["group_id"] = ""
    meta["group_id"] = meta["group_id"].fillna("").astype(str)
    if meta["unique_id"].duplicated().any():
        raise ValueError("meta unique_id values must be unique")
    meta = meta.reset_index(drop=True)

    lig_code_all, lig_uniques = _codes(meta["lig_key"])
    group_code_all, group_uniques = _codes(meta["group_id"])
    # PDB codes compared case-blind (BindingDB folders give upper case): one code array serves the anchor's own
    # code and every pool row's
    tmpl_code_all, _ = _codes(meta["anchor_pdb"].str.strip().str.lower())
    group_lig = pd.DataFrame({"g": group_code_all, "l": lig_code_all})
    group_lig = group_lig[(group_lig["g"] >= 0) & (group_lig["l"] >= 0)]
    group_lig_codes = {int(g): np.unique(sub["l"].to_numpy()) for g, sub in group_lig.groupby("g")}

    uid = np.asarray(desc["unique_id"]).astype(str)
    lf = np.asarray(desc["ligand_F"], dtype=np.float64)
    pf = np.asarray(desc["pocket_F"], dtype=np.float64)
    lt = np.asarray(desc["ligand_total"], dtype=np.float64)
    nh = np.asarray(desc["n_ligand_atoms"]).astype(np.int64)
    ok = np.asarray(desc["ok"], dtype=bool)
    finite = (np.isfinite(lf).all(axis=(1, 2)) & np.isfinite(pf).all(axis=(1, 2))
              & np.isfinite(lt).all(axis=1))
    pos = pd.Index(meta["unique_id"]).get_indexer(uid)
    rows = np.flatnonzero(ok & finite & (pos >= 0))
    m = meta.iloc[pos[rows]].reset_index(drop=True)
    lf, pf, lt, nh = lf[rows], pf[rows], lt[rows], nh[rows]
    split = m["split"].to_numpy(dtype=str)
    train = split == POOL_SPLIT
    if not train.any():
        raise ValueError("no pool rows (split == train with a descriptor)")

    def zscore(x):
        flat = x.reshape(len(x), -1)
        mu = flat[train].mean(axis=0)
        sd = flat[train].std(axis=0)
        sd = np.where(sd < SD_FLOOR, 1.0, sd)
        return ((flat - mu) / sd).reshape(x.shape), mu, sd

    lz, l_mu, l_sd = zscore(lf)
    pz, p_mu, p_sd = zscore(pf)
    shape = _unit(lz[:, OCC, :])
    chem = _unit(lz[:, list(CHEM_CHANNELS), :].reshape(len(lz), -1))
    pocket_comp = _unit(pz[:, list(COMPLEMENT_CHANNELS), :].reshape(len(pz), -1))
    charge = np.rint(lt[:, POS] - lt[:, NEG]).astype(np.int64)
    stats = {"n_rows": int(len(rows)), "n_pool": int(train.sum()), "n_desc_rows": int(len(uid)),
             "n_desc_not_ok": int((~(ok & finite)).sum()), "n_desc_not_in_meta": int((pos < 0).sum()),
             "ligand_F_mean": l_mu, "ligand_F_sd": l_sd, "pocket_F_mean": p_mu, "pocket_F_sd": p_sd}
    src = m["source"].to_numpy(dtype=str)
    return TwinPool(uid=uid[rows], source=src, split=split, shape=shape, chem=chem, pocket_comp=pocket_comp,
                    logp=lt[:, HYD], charge=charge, arom=lt[:, ARO] / AROMATIC_RING_ATOMS, n_heavy=nh,
                    lig_key=m["lig_key"].to_numpy(dtype=str), group_id=m["group_id"].to_numpy(dtype=str),
                    anchor_pdb=m["anchor_pdb"].to_numpy(dtype=str), group_lig_codes=group_lig_codes,
                    lig_code=lig_code_all[pos[rows]], group_code=group_code_all[pos[rows]],
                    tmpl_code=tmpl_code_all[pos[rows]], stats=stats)


def load_pool(zernike_npz, meta_csv, groups_csv=None):
    """build_pool from the files (only the needed npz members are decompressed)."""
    with np.load(zernike_npz, allow_pickle=False) as z:
        desc = {k: z[k] for k in DESC_KEYS}
    meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    groups = pd.read_csv(groups_csv, dtype=str, keep_default_na=False) if groups_csv else None
    return build_pool(desc, meta, groups)


# ---------------------------------------------------------------- selection

def _excluded(pool, a):
    """Boolean mask over pool.pool_rows of rows excluded for anchor row a (identity rules)."""
    lig = pool.pool_lig_code
    mask = pool.pool_rows == a
    g = int(pool.group_code[a])
    if g >= 0:
        mask |= pool.pool_group_code == g
        known = pool.group_lig_codes.get(g)
        if known is not None and len(known):
            mask |= np.isin(lig, known)
    la = int(pool.lig_code[a])
    if la >= 0:
        mask |= lig == la
    ta = int(pool.tmpl_code[a])
    if ta >= 0:
        mask |= pool.pool_tmpl_code == ta
    return mask


def _n_near(pool, k, shape_frac):
    return max(int(k), int(math.ceil(shape_frac * pool.pool_size - 1e-9)))


def _info(anchor_uid, reason, **kw):
    out = {"anchor_uid": anchor_uid, "reason": reason, "empty_at": "", "n_selected": 0}
    out.update(kw)
    return out


def _choose(pool, anchor_uid, a, dist, k, shape_frac, max_dlogp, max_darom, same_charge_sign, size_tol):
    """Twins of anchor row a given its shape distances to every pool row (float32 [pool_size])."""
    pr = pool.pool_rows
    excl = _excluded(pool, a)
    d = np.array(dist, dtype=np.float32, copy=True)
    d[excl] = np.inf
    n_eligible = int((~excl).sum())
    base = dict(n_pool=pool.pool_size, n_excluded=int(excl.sum()), n_eligible=n_eligible,
                anchor_complementarity=float(pool.pocket_comp[a] @ pool.chem[a]),
                anchor_logp=float(pool.logp[a]), anchor_charge=int(pool.charge[a]),
                anchor_arom=float(pool.arom[a]), anchor_n_heavy=int(pool.n_heavy[a]))
    n_near = min(_n_near(pool, k, shape_frac), n_eligible)
    funnel = {"n_near": n_near}
    if n_near == 0:
        return [], _info(anchor_uid, "no_twin", empty_at="excluded", **base,
                         **{f"n_{s}": 0 for s in FUNNEL[1:]}, **funnel)
    near = np.argpartition(d, n_near - 1)[:n_near] if n_near < len(d) else np.arange(len(d))
    near = near[np.isfinite(d[near])]
    near = near[np.lexsort((near, d[near]))]          # by distance, ties by row order
    rows = pr[near]
    keep = np.abs(pool.n_heavy[rows] - pool.n_heavy[a]) <= size_tol * pool.n_heavy[a]
    funnel["n_size"] = int(keep.sum())
    keep &= np.abs(pool.logp[rows] - pool.logp[a]) <= max_dlogp
    funnel["n_logp"] = int(keep.sum())
    keep &= np.abs(pool.arom[rows] - pool.arom[a]) <= max_darom + 1e-9
    funnel["n_arom"] = int(keep.sum())
    if same_charge_sign:
        keep &= np.sign(pool.charge[rows]) == np.sign(pool.charge[a])
    funnel["n_charge"] = int(keep.sum())
    near, rows = near[keep], rows[keep]
    if not len(rows):
        empty_at = next(s for s in FUNNEL[1:] if funnel[f"n_{s}"] == 0)
        return [], _info(anchor_uid, "no_twin", empty_at=empty_at, **base, **funnel)
    comp = pool.chem[rows] @ pool.pocket_comp[a]
    order = np.lexsort((near, d[near], comp))           # complementarity, then distance, then row order
    out, seen = [], set()
    for j in order:
        r = int(rows[j])
        lk = pool.lig_key[r]
        if lk in seen:
            continue
        seen.add(lk)
        out.append(TwinChoice(
            anchor_uid=anchor_uid, rank=len(out), twin_uid=str(pool.uid[r]), twin_source=str(pool.source[r]),
            twin_lig_key=str(lk), twin_group_id=str(pool.group_id[r]), twin_anchor_pdb=str(pool.anchor_pdb[r]),
            shape_dist=float(d[near[j]]), complementarity=float(comp[j]),
            dlogp=float(pool.logp[r] - pool.logp[a]), dcharge=int(pool.charge[r] - pool.charge[a]),
            darom=float(pool.arom[r] - pool.arom[a]), n_heavy_anchor=int(pool.n_heavy[a]),
            n_heavy_twin=int(pool.n_heavy[r])))
        if len(out) == k:
            break
    reason = "ok" if len(out) == k else "short"
    return out, _info(anchor_uid, reason, n_selected=len(out), **base, **funnel)


def select_twins_batch(anchor_uids, pool, k=2, shape_frac=0.01, max_dlogp=1.5, max_darom=1,
                       same_charge_sign=True, size_tol=0.2, block=DEFAULT_BLOCK):
    """{anchor_uid: (list[TwinChoice], info)} for every anchor (see module docstring). Shape distances
    of `block` anchors at a time come from one float32 matmul against the cached pool unit vectors."""
    out = {}
    known = [(u, pool.index[u]) for u in anchor_uids if u in pool.index]
    for u in anchor_uids:
        if u not in pool.index:
            out[u] = ([], _info(u, "anchor_no_descriptor"))
    for start in range(0, len(known), max(1, int(block))):
        chunk = known[start:start + max(1, int(block))]
        rows = np.array([a for _, a in chunk])
        dist = 1.0 - pool.shape[rows] @ pool.pool_shape.T     # [block, pool_size] float32
        for (u, a), drow in zip(chunk, dist):
            out[u] = _choose(pool, u, a, drow, k, shape_frac, max_dlogp, max_darom, same_charge_sign,
                             size_tol)
    return out


def select_twins_with_info(anchor_uid, pool, k=2, shape_frac=0.01, max_dlogp=1.5, max_darom=1,
                           same_charge_sign=True, size_tol=0.2):
    """(list[TwinChoice], info) for one anchor."""
    return select_twins_batch([anchor_uid], pool, k=k, shape_frac=shape_frac, max_dlogp=max_dlogp,
                              max_darom=max_darom, same_charge_sign=same_charge_sign,
                              size_tol=size_tol)[anchor_uid]


def select_twins(anchor_uid, pool, k=2, shape_frac=0.01, max_dlogp=1.5, max_darom=1, same_charge_sign=True,
                 size_tol=0.2):
    """The k shape twins of one anchor (fewer, possibly none, when not enough pass; never raises)."""
    return select_twins_with_info(anchor_uid, pool, k=k, shape_frac=shape_frac, max_dlogp=max_dlogp,
                                  max_darom=max_darom, same_charge_sign=same_charge_sign,
                                  size_tol=size_tol)[0]
