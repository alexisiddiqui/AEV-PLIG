"""Target groups for held-out splits, and each row's cosine distance to the nearest training pocket of
another group (replaces descriptor clustering for the split; docs/pocket_descriptor_analysis.md §5).

Groups (rows of --meta with kind 0, or every row when there is no kind column):
  1. A row starts in the group of its `uniprot`. BindingNet rows carry their template's UniProt and
     BindingDB rows their folder PDB's (as build_paired_dataset.py writes them), so they join that group.
     A row without a UniProt is in group "single:<anchor_pdb>": "single:<unique_id>" for a PDBbind row, and
     BindingNet/BindingDB rows on the same structure join it, so rows built on one structure never split.
  2. UniProts that share a normalised protein name of INDEX_general_PL_name.2020 are merged (connected
     components of the UniProt-name graph over every index entry with a UniProt). A name is generic, and
     is skipped, when it maps to >= --generic-min-uniprots (5) UniProts in the index; the skipped names
     are listed in the summary. Names are upper-cased with non-alphanumerics read as spaces.
  3. Optional, --merge-cosine T (default off): average-linkage (UPGMA) merge of the groups on pocket
     descriptors while the smallest group-average cosine distance is below T. Units weigh equally: a
     PDBbind row is a unit, BindingNet/BindingDB rows form one unit per series_key (template / folder,
     pocket_clusters.unit_keys), represented by the normalised mean of its rows' unit vectors. With unit
     vectors the mean pairwise cosine distance between two groups is 1 - mean_A . mean_B, so the merge is
     exact UPGMA over units without a pairwise distance matrix.
  group_id: "uniprot:<the member UniProt with most rows>", or the "single:..." key of a group without one.

Descriptor: [pocket_F | sign(pocket_G) sqrt|pocket_G|] flattened (343 + 1,029 columns at order 12), each
column standardised over the rows with a usable descriptor (ok, finite, in meta; zero-variance columns keep
scale 1), no PCA, compared by cosine.

nn_train_dist: cosine distance from a row's descriptor to the nearest descriptor among rows with
split == "train" in a DIFFERENT group (the row itself never counts). NaN when the row has no usable
descriptor, or when every train row is in its group. Computed in blocks of --block query rows (one matmul of
unit vectors per block), never as a full distance matrix.

Outputs: --out CSV with unique_id, source, split, uniprot, group_id, group_size_rows, group_size_units,
nn_train_dist (meta order, every row), and <out stem>_summary.json (group counts and sizes, the largest
groups with their names, generic names skipped, the fraction of rows whose nearest training pocket is in
another group and the nn_train_dist quantiles, per split).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import json
import re
import tarfile
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import build_paired_dataset as bpd
from pocket_clusters import _clean_uniprot, unit_keys

NAME_FILE = "INDEX_general_PL_name.2020"
ROW_UNIT_SOURCES = ("pdbbind",)
QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
N_LARGEST = 20


# ---------------------------------------------------------------- PDBbind names

def normalise_name(name):
    """Upper case, non-alphanumerics as spaces, single spaces ("HIV-1 PROTEASE" -> "HIV 1 PROTEASE")."""
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split())


def read_name_index(tgz=None, index_dir=None):
    """DataFrame (pdb, uniprot, name) of INDEX_general_PL_name.2020, from the plain-text index tarball (the
    file build_paired_dataset.read_pdbbind_index reads, found the same way) or from an extracted index dir.
    uniprot is "" where the index writes "------"; name is normalised (normalise_name)."""
    if (tgz is None) == (index_dir is None):
        raise ValueError("give exactly one of tgz / index_dir")
    if tgz is not None:
        with tarfile.open(tgz, "r:*") as tf:
            text = bpd._member_text(tf, NAME_FILE)
    else:
        with open(os.path.join(index_dir, NAME_FILE), "r", errors="replace") as fh:
            text = fh.read()
    rows = []
    for line in bpd._index_lines(text):  # columns: PDB code, release year, UniProt ID, protein name
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        uniprot = "" if parts[2] == bpd.UNMAPPED_UNIPROT else parts[2]
        rows.append((parts[0].lower(), uniprot, normalise_name(parts[3]) if len(parts) > 3 else ""))
    return pd.DataFrame(rows, columns=["pdb", "uniprot", "name"]).drop_duplicates("pdb", keep="first")


def name_merge(uniprots, index, generic_min=5):
    """(component, generic): component maps each UniProt of `uniprots` to a component id of the
    UniProt-name graph built from every index entry with a UniProt and a non-generic name; generic is a
    DataFrame (name, n_uniprots_index) of the names mapping to >= generic_min UniProts, largest first."""
    idx = index[(index["uniprot"] != "") & (index["name"] != "")]
    n_uni = idx.groupby("name")["uniprot"].nunique()
    generic = n_uni[n_uni >= generic_min].sort_values(ascending=False, kind="stable")
    edges = idx.loc[~idx["name"].isin(generic.index), ["uniprot", "name"]].drop_duplicates()
    nodes_u = pd.Index(sorted(set(uniprots) | set(edges["uniprot"])))
    nodes_n = pd.Index(sorted(set(edges["name"])))
    iu = nodes_u.get_indexer(edges["uniprot"])
    jn = len(nodes_u) + nodes_n.get_indexer(edges["name"])
    size = len(nodes_u) + len(nodes_n)
    graph = coo_matrix((np.ones(len(edges)), (iu, jn)), shape=(size, size))
    _, comp = connected_components(graph, directed=False)
    component = dict(zip(nodes_u, comp[:len(nodes_u)].tolist()))
    generic_df = pd.DataFrame({"name": generic.index.astype(str), "n_uniprots_index": generic.to_numpy()})
    return {u: component[u] for u in uniprots}, generic_df


# ---------------------------------------------------------------- descriptors

def pocket_embedding(pocket_f, pocket_g):
    """Unit rows of the standardised [F | signed-sqrt G] (float32). Fit and applied on the rows given."""
    n = pocket_f.shape[0]
    g = np.asarray(pocket_g, dtype=np.float64).reshape(n, -1)
    x = np.concatenate([np.asarray(pocket_f, dtype=np.float64).reshape(n, -1),
                        np.sign(g) * np.sqrt(np.abs(g))], axis=1)
    del g
    if n:
        x -= x.mean(axis=0)
        sd = x.std(axis=0)
        x /= np.where(sd > 0, sd, 1.0)
    return unit_rows(x).astype(np.float32)


def unit_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def nearest_train_other_group(emb, group, train, block=1024):
    """(dist, other): dist[i] = 1 - max cosine similarity of row i to a train row of another group (NaN
    without a descriptor or without such a train row); other[i] = 1.0 when row i's nearest train row (any
    group, row i excluded) is in another group, 0.0 when it is in its own, NaN when undefined.
    emb: [n, d] unit rows, NaN rows for a missing descriptor; group: int codes [n]; train: bool [n]."""
    n = len(emb)
    dist = np.full(n, np.nan)
    other = np.full(n, np.nan)
    have = np.isfinite(emb).all(axis=1)
    t_idx = np.flatnonzero(have & train)
    if not len(t_idx):
        return dist, other
    t_emb = np.ascontiguousarray(emb[t_idx], dtype=np.float32)
    t_group = group[t_idx]
    pos_in_train = np.full(n, -1)
    pos_in_train[t_idx] = np.arange(len(t_idx))
    q_idx = np.flatnonzero(have)
    for s in range(0, len(q_idx), block):
        qi = q_idx[s:s + block]
        sim = np.asarray(emb[qi], dtype=np.float32) @ t_emb.T          # [b, n_train]
        self_pos = pos_in_train[qi]
        is_train = self_pos >= 0
        sim[np.flatnonzero(is_train), self_pos[is_train]] = -np.inf
        arg_any = sim.argmax(axis=1)
        best_any = sim[np.arange(len(qi)), arg_any]
        found = np.isfinite(best_any)
        other[qi[found]] = (t_group[arg_any[found]] != group[qi[found]]).astype(float)
        sim[group[qi][:, None] == t_group[None, :]] = -np.inf
        best = sim.max(axis=1)
        ok = np.isfinite(best)
        dist[qi[ok]] = np.clip(1.0 - best[ok].astype(np.float64), 0.0, 2.0)
    return dist, other


def upgma_merge(means, counts, threshold):
    """Average-linkage merge of clusters given by mean unit vectors `means` [G, d] and unit counts [G],
    while the smallest mean pairwise cosine distance (1 - means_a . means_b) is below `threshold`.
    Returns (labels [G] in 0..G'-1, number of merges). Rows whose count is 0 never merge."""
    g = len(means)
    labels = np.arange(g)
    if g < 2:
        return labels, 0
    sim = np.asarray(means, dtype=np.float64) @ np.asarray(means, dtype=np.float64).T
    count = np.asarray(counts, dtype=np.float64).copy()
    dead = count <= 0
    sim[dead, :] = -np.inf
    sim[:, dead] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    row_arg = sim.argmax(axis=1)
    row_max = sim[np.arange(g), row_arg]
    parent = np.arange(g)
    merges = 0
    while True:
        i = int(np.argmax(row_max))
        if not np.isfinite(row_max[i]) or 1.0 - row_max[i] >= threshold:
            break
        j = int(row_arg[i])
        new = (count[i] * sim[i] + count[j] * sim[j]) / (count[i] + count[j])
        sim[i, :] = new
        sim[:, i] = new
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        sim[i, i] = -np.inf
        count[i] += count[j]
        count[j] = 0
        parent[j] = i
        row_max[j] = -np.inf
        merges += 1
        # a row's best partner can only get worse when it was i or j (the average never exceeds the max)
        stale = np.flatnonzero((row_arg == i) | (row_arg == j))
        stale = np.union1d(stale[np.isfinite(row_max[stale])], [i])
        row_arg[stale] = sim[stale].argmax(axis=1)
        row_max[stale] = sim[stale, row_arg[stale]]
        # i's new similarity may now be some row's best
        better = sim[:, i] > row_max
        row_arg[better] = i
        row_max[better] = sim[better, i]
    root = parent.copy()
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    return pd.factorize(root)[0], merges


# ---------------------------------------------------------------- groups

def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(values, q)) for q in QUANTILES} | \
        {"mean": float(values.mean()), "n": int(len(values))}


def build_groups(meta, index, emb, *, generic_min=5, merge_cosine=None, block=1024):
    """Core routine. meta: DataFrame (unique_id, source, split, uniprot, series_key, anchor_pdb), one row per
    output row; index: read_name_index output; emb: [len(meta), d] pocket_embedding rows aligned with meta,
    NaN rows where the descriptor is missing. Returns (rows DataFrame, summary dict)."""
    t0 = time.time()
    n = len(meta)
    uid = meta["unique_id"].astype(str).to_numpy()
    source = meta["source"].astype(str).to_numpy()
    split = meta["split"].astype(str).to_numpy()
    uniprot = _clean_uniprot(meta["uniprot"].to_numpy())
    anchor = meta["anchor_pdb"].astype(str).str.strip().str.lower().to_numpy()
    series = meta["series_key"].astype(str).to_numpy()
    units = unit_keys(source, uid, series, set(ROW_UNIT_SOURCES))
    has_uni = uniprot != ""

    # 1-2. UniProt groups merged through non-generic names; "single:<anchor>" for rows without a UniProt
    meta_uniprots = sorted(set(uniprot[has_uni]))
    component, generic = name_merge(meta_uniprots, index, generic_min)
    single_key = np.array([f"single:{a if a else u}" for a, u in zip(anchor, uid)], dtype=object)
    key = np.where(has_uni, np.array([f"c{component.get(u, -1)}" for u in uniprot], dtype=object), single_key)
    gcode, gkeys = pd.factorize(key)
    n_groups_name = len(gkeys)

    # 3. optional descriptor merge over units
    have = np.isfinite(emb).all(axis=1)
    merge_info = {"merge_cosine": merge_cosine, "groups_before": int(n_groups_name)}
    if merge_cosine is not None:
        ucode, ukeys = pd.factorize(units)
        u_sum = np.zeros((len(ukeys), emb.shape[1]))
        np.add.at(u_sum, ucode[have], emb[have].astype(np.float64))
        u_vec = unit_rows(u_sum)
        u_have = np.linalg.norm(u_sum, axis=1) > 0
        if (pd.Series(gcode).groupby(ucode).nunique() > 1).any():
            raise ValueError("a BindingNet/BindingDB series spans two groups (rows with different UniProts)")
        u_group = np.zeros(len(ukeys), dtype=int)
        u_group[ucode] = gcode
        g_sum = np.zeros((len(gkeys), emb.shape[1]))
        np.add.at(g_sum, u_group[u_have], u_vec[u_have])
        g_cnt = np.bincount(u_group[u_have], minlength=len(gkeys)).astype(float)
        means = g_sum / np.maximum(g_cnt, 1.0)[:, None]
        labels, merges = upgma_merge(means, g_cnt, merge_cosine)
        gcode = labels[gcode]
        merge_info.update(merges=int(merges), groups_with_descriptor=int((g_cnt > 0).sum()))
    gcode = pd.factorize(gcode)[0]
    n_groups = int(gcode.max()) + 1 if n else 0

    # group ids: the member UniProt with most rows, else the largest single key
    df = pd.DataFrame({"g": gcode, "uniprot": uniprot, "single": single_key, "unit": units, "split": split,
                       "anchor": anchor})
    uni_rows = df[has_uni].groupby(["g", "uniprot"]).size().reset_index(name="rows")
    uni_rows = uni_rows.sort_values(["g", "rows", "uniprot"], ascending=[True, False, True], kind="stable")
    rep = {g: f"uniprot:{u}" for g, u in uni_rows.drop_duplicates("g")[["g", "uniprot"]].itertuples(index=False)}
    sgl_rows = df[~has_uni].groupby(["g", "single"]).size().reset_index(name="rows")
    sgl_rows = sgl_rows.sort_values(["g", "rows", "single"], ascending=[True, False, True], kind="stable")
    for g, s in sgl_rows.drop_duplicates("g")[["g", "single"]].itertuples(index=False):
        rep.setdefault(g, s)
    group_id = np.array([rep[g] for g in range(n_groups)], dtype=object)
    size_rows = np.bincount(gcode, minlength=n_groups)
    size_units = df.groupby("g")["unit"].nunique().reindex(range(n_groups), fill_value=0).to_numpy()

    # nearest training pocket of another group
    dist, other = nearest_train_other_group(emb, gcode, split == "train", block=block)

    rows = pd.DataFrame({"unique_id": uid, "source": source, "split": split, "uniprot": uniprot,
                         "group_id": group_id[gcode], "group_size_rows": size_rows[gcode],
                         "group_size_units": size_units[gcode], "nn_train_dist": dist})

    # ---- summary
    name_of = dict(zip(index["pdb"], index["name"]))
    row_name = np.array([name_of.get(a, "") for a in anchor], dtype=object)
    df["name"] = row_name
    train_rows_in_group = np.bincount(gcode, weights=(split == "train").astype(float), minlength=n_groups)
    largest = []
    for g in np.argsort(-size_rows, kind="stable")[:N_LARGEST]:
        m = gcode == g
        sub = df[m]
        uni_counts = sub.loc[sub["uniprot"] != "", "uniprot"].value_counts()
        names = sub.loc[sub["name"] != ""].drop_duplicates("anchor")["name"].value_counts()
        largest.append({"group_id": str(group_id[g]), "rows": int(size_rows[g]), "units": int(size_units[g]),
                        "frac_rows": float(size_rows[g] / n), "n_uniprots": int(len(uni_counts)),
                        "uniprots_top": uni_counts.head(10).index.tolist(),
                        "names_top": {k: int(v) for k, v in names.head(5).items()},
                        "rows_by_split": {k: int(v) for k, v in sub["split"].value_counts().items()}})
    n_uni_per_group = df[has_uni].groupby("g")["uniprot"].nunique().reindex(range(n_groups), fill_value=0)
    multi = n_uni_per_group.to_numpy() >= 2
    uni_in_multi = int(n_uni_per_group[multi].sum())
    splits_per_group = df.groupby("g")["split"].nunique()
    per_split = {}
    for s in sorted(set(split)):
        m = split == s
        others_train = train_rows_in_group[gcode[m]] - (split[m] == "train")
        per_split[s] = {
            "rows": int(m.sum()), "with_descriptor": int((m & have).sum()),
            "groups": int(len(np.unique(gcode[m]))),
            "frac_nearest_train_in_other_group": float(np.nanmean(other[m])) if np.isfinite(other[m]).any()
            else None,
            "frac_rows_whose_group_has_other_train_rows": float((others_train > 0).mean()),
            "nn_train_dist": _quantiles(dist[m]),
        }
    gen_rows = generic.copy()
    meta_name_rows = pd.Series(row_name).value_counts()
    meta_name_uni = df[has_uni & (row_name != "")].groupby("name")["uniprot"].nunique()
    gen_rows["meta_rows"] = gen_rows["name"].map(meta_name_rows).fillna(0).astype(int)
    gen_rows["meta_uniprots"] = gen_rows["name"].map(meta_name_uni).fillna(0).astype(int)
    idx_uni = dict(zip(index["pdb"], index["uniprot"]))
    in_index = np.array([a in idx_uni for a in anchor])
    mismatch = int(sum(1 for a, u, h in zip(anchor, uniprot, in_index) if h and idx_uni[a] != u))
    summary = {
        "rows": int(n), "rows_with_descriptor": int(have.sum()),
        "rows_by_source": {k: int(v) for k, v in pd.Series(source).value_counts().items()},
        "rows_without_uniprot_by_source": {k: int(v) for k, v in pd.Series(source[~has_uni]).value_counts().items()},
        "uniprots": len(meta_uniprots),
        "groups": n_groups,
        "groups_after_name_merge": int(n_groups_name),
        "groups_with_uniprot": int(sum(str(x).startswith("uniprot:") for x in group_id)),
        "groups_single": int(sum(str(x).startswith("single:") for x in group_id)),
        "groups_with_ge2_uniprots": int(multi.sum()), "uniprots_in_multi_uniprot_groups": uni_in_multi,
        "group_size_rows": _quantiles(size_rows), "group_size_units": _quantiles(size_units),
        "largest_group_frac_rows": float(size_rows.max() / n) if n else None,
        "largest_group_frac_units": float(size_units.max() / max(len(set(units)), 1)) if n else None,
        "units": int(len(set(units))),
        "largest_groups": largest,
        "groups_spanning_ge2_splits": int((splits_per_group >= 2).sum()),
        "frac_nearest_train_in_other_group": float(np.nanmean(other)) if np.isfinite(other).any() else None,
        "nn_train_dist": _quantiles(dist),
        "per_split": per_split,
        "name_merge": {"index_entries": int(len(index)), "generic_min_uniprots": generic_min,
                       "generic_names_skipped": int(len(gen_rows)),
                       "generic_names": gen_rows.to_dict(orient="records"),
                       "anchor_pdbs_missing_from_index_rows": int((~in_index).sum()),
                       "rows_uniprot_differs_from_index": mismatch},
        "descriptor_merge": merge_info,
        "descriptor": "standardised [pocket_F | signed-sqrt pocket_G], no PCA, cosine",
        "seconds": time.time() - t0,
    }
    return rows, summary


# ---------------------------------------------------------------- CLI

def load_meta(path):
    meta = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "kind" in meta.columns:
        meta = meta[meta["kind"].str.strip() == "0"]
    for col in ("unique_id", "source", "split"):
        if col not in meta.columns:
            raise ValueError(f"{path} has no {col} column")
    for col in ("uniprot", "series_key", "anchor_pdb"):
        if col not in meta.columns:
            meta[col] = ""
    return meta.reset_index(drop=True)


def load_embedding(zernike_npz, unique_id):
    """pocket_embedding rows aligned with `unique_id` (NaN rows where the npz has no usable descriptor)
    and the npz counts for the summary."""
    with np.load(zernike_npz, allow_pickle=False) as zf:
        z_uid = zf["unique_id"].astype(str)
        ok = zf["ok"].astype(bool)
        f, g = zf["pocket_F"], zf["pocket_G"]
        params = {k: zf[k].item() for k in ("pocket_mode", "pocket_radius", "order", "radius", "variant_note")
                  if k in zf.files and zf[k].ndim == 0}
    ok &= np.isfinite(f).all(axis=(1, 2)) & np.isfinite(g).all(axis=(1, 2))
    pos = pd.Series(np.arange(len(z_uid)), index=z_uid)
    pos = pos[~pos.index.duplicated(keep="first")]
    where = pos.reindex(unique_id).to_numpy(dtype=float)
    found = np.isfinite(where)
    idx = np.where(found, where, 0).astype(int)
    usable = found & ok[idx]
    emb_ok = pocket_embedding(f[idx[usable]], g[idx[usable]])
    emb = np.full((len(unique_id), emb_ok.shape[1]), np.nan, dtype=np.float32)
    emb[usable] = emb_ok
    info = {"npz_rows": int(len(z_uid)), "npz_not_usable": int((~ok).sum()),
            "meta_rows_absent_from_npz": int((~found).sum()), "meta_rows_without_descriptor": int((~usable).sum()),
            "npz_params": params}
    return emb, info


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    raise TypeError(type(obj).__name__)


def _clean_nan(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--meta", required=True, help="paired_meta.csv")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdbbind-index-tgz", help="PDBbind_v2020_plain_text_index.tar.gz")
    src.add_argument("--pdbbind-index-dir", help="extracted index dir holding INDEX_general_PL_name.2020")
    p.add_argument("--zernike", required=True, help="featurise_zernike.py npz")
    p.add_argument("--out", required=True, help="pocket_groups.csv")
    p.add_argument("--merge-cosine", type=float, default=None,
                   help="group-average cosine distance below which groups merge (default: no descriptor merge)")
    p.add_argument("--generic-min-uniprots", type=int, default=5,
                   help="a protein name mapping to at least this many UniProts is generic and not merged on")
    p.add_argument("--block", type=int, default=1024, help="query rows per nearest-neighbour block")
    args = p.parse_args(argv)

    t0 = time.time()
    meta = load_meta(args.meta)
    index = read_name_index(tgz=args.pdbbind_index_tgz, index_dir=args.pdbbind_index_dir)
    emb, info = load_embedding(args.zernike, meta["unique_id"].astype(str).to_numpy())
    rows, summary = build_groups(meta, index, emb, generic_min=args.generic_min_uniprots,
                                 merge_cosine=args.merge_cosine, block=args.block)
    summary.update(info)
    summary["inputs"] = {"meta": os.path.abspath(args.meta), "zernike": os.path.abspath(args.zernike),
                         "pdbbind_index": os.path.abspath(args.pdbbind_index_tgz or args.pdbbind_index_dir)}
    summary["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.to_csv(args.out + ".tmp", index=False, float_format="%.6f")
    os.replace(args.out + ".tmp", args.out)
    stem = args.out[:-4] if args.out.endswith(".csv") else args.out
    summary_path = stem + "_summary.json"
    with open(summary_path + ".tmp", "w") as fh:
        json.dump(_clean_nan(summary), fh, indent=1, default=_json_default)
    os.replace(summary_path + ".tmp", summary_path)
    largest = summary["largest_groups"][0] if summary["largest_groups"] else {}
    print(f"{summary['groups']} groups over {summary['rows']} rows ({summary['groups_after_name_merge']} after the "
          f"name merge, {summary['name_merge']['generic_names_skipped']} generic names skipped); largest "
          f"{largest.get('group_id')} {largest.get('rows')} rows; nearest training pocket in another group: "
          f"{summary['frac_nearest_train_in_other_group']}; wrote {args.out} and {summary_path}", flush=True)
    return rows, summary


if __name__ == "__main__":
    main()
