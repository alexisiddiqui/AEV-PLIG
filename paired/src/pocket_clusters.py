"""Cluster pockets by their 3D Zernike descriptors, with the cut chosen from same-UniProt co-clustering.

Steps:
  1. X = [pocket_F | pocket_G] flattened per row (ok rows of --zernike that are in --meta); StandardScaler
     (zero-variance columns keep scale 1), then PCA to --pca-dims (50) dimensions, both fit on all rows.
  2. Units: a row of a --row-unit-sources source (default pdbbind) is its own unit; rows of other sources
     are grouped by `series_key` (BindingNet template, BindingDB folder), so every complex built on one
     template lands in one cluster. A grouped unit is represented by its medoid row (largest mean cosine
     similarity to the unit's other rows; at most --medoid-max rows are scored, sampled with seed 0).
  3. Average-linkage agglomerative clustering of the unit vectors with cosine distance
     (scipy.cluster.hierarchy.linkage(method="average", metric="cosine"), the algorithm of sklearn's
     AgglomerativeClustering(linkage="average", metric="cosine")), cut at distance t.
  4. t: the smallest merge height at which >= --target (0.9) of the pairs of --criterion-source units
     (default pdbbind) sharing a UniProt fall in one cluster (pair-weighted; the rate is monotone in t
     on one dendrogram, so a bisection over merge heights finds it). Units with an empty or "------"
     UniProt do not form pairs. A grid of thresholds is reported as the curve.
Outputs: --out CSV with unique_id, source, uniprot, cluster_id (0 = largest cluster by rows; -1 for rows
without a usable descriptor), cluster_size (rows), nearest_other_cluster_dist (cosine distance from the
row's PCA vector to the closest unit of another cluster), unit_key; and <out stem>_summary.json
(threshold, number of clusters, size distribution, co-clustering rate at the threshold, clusters mixing
>= 2 UniProts, the curve).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

UNMAPPED = {"", "------", "nan", "None"}
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def descriptor_matrix(pocket_f, pocket_g):
    n = pocket_f.shape[0]
    return np.concatenate([pocket_f.reshape(n, -1), pocket_g.reshape(n, -1)], axis=1).astype(np.float64)


def embed(x, dims=50, seed=0):
    """StandardScaler then PCA to min(dims, n - 1, d) dimensions; returns (Z, explained variance ratio)."""
    z = StandardScaler().fit_transform(x)
    k = max(1, min(dims, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=k, random_state=seed, svd_solver="full" if x.shape[0] < 5000 else "randomized")
    return pca.fit_transform(z), pca.explained_variance_ratio_


def _unit_norm(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norm > 0, norm, 1.0)


def medoid(vectors, max_rows=4000, seed=0):
    """Index (into vectors) of the row with the largest mean cosine similarity to the others."""
    if len(vectors) <= 2:
        return 0
    cand = np.arange(len(vectors))
    if len(cand) > max_rows:
        cand = np.sort(np.random.default_rng(seed).choice(cand, size=max_rows, replace=False))
    u = _unit_norm(vectors[cand])
    return int(cand[np.argmax((u @ u.T).mean(axis=1))])


def unit_keys(source, unique_id, series_key, row_unit_sources):
    keys = []
    for s, u, k in zip(source, unique_id, series_key):
        if s in row_unit_sources or not k:
            keys.append(f"{s}:row:{u}")
        else:
            keys.append(f"{s}:series:{k}")
    return np.array(keys, dtype=object)


def pair_rate(labels, groups):
    """Fraction of same-group pairs with equal labels (pair-weighted), the group-mean of that fraction,
    and the number of pairs. groups: array of group names, None/"" for units that form no pairs."""
    df = pd.DataFrame({"g": groups, "c": labels})
    df = df[df["g"].notna() & (df["g"] != "")]
    if df.empty:
        return float("nan"), float("nan"), 0
    n_g = df.groupby("g").size()
    n_gc = df.groupby(["g", "c"]).size()
    total = (n_g * (n_g - 1) / 2)
    same = (n_gc * (n_gc - 1) / 2).groupby(level=0).sum()
    keep = total > 0
    if not keep.any():
        return float("nan"), float("nan"), 0
    rate = float(same[keep].sum() / total[keep].sum())
    group_mean = float((same[keep] / total[keep]).mean())
    return rate, group_mean, int(total[keep].sum())


def choose_threshold(z_link, groups, target):
    """Smallest merge height whose cut reaches `target` same-group co-clustering (None if no pairs)."""
    heights = np.unique(z_link[:, 2])
    rate_at = lambda t: pair_rate(fcluster(z_link, t, criterion="distance"), groups)[0]
    if not np.isfinite(rate_at(heights[-1])):
        return None
    lo, hi = 0, len(heights) - 1
    if rate_at(heights[0]) >= target:
        return float(heights[0])
    while hi - lo > 1:  # invariant: rate(heights[lo]) < target <= rate(heights[hi])
        mid = (lo + hi) // 2
        if rate_at(heights[mid]) >= target:
            hi = mid
        else:
            lo = mid
    return float(heights[hi])


def nearest_other_cluster(z_rows, row_cluster, z_units, unit_cluster, block=2048):
    """Cosine distance from each row to the closest unit vector in a different cluster (NaN if none)."""
    ur = _unit_norm(z_rows).astype(np.float32)
    uu = _unit_norm(z_units).astype(np.float32)
    out = np.full(len(z_rows), np.nan)
    for i in range(0, len(z_rows), block):
        sim = ur[i:i + block] @ uu.T
        same = row_cluster[i:i + block, None] == unit_cluster[None, :]
        sim[same] = -np.inf
        best = sim.max(axis=1)
        ok = np.isfinite(best)
        out[i:i + block][ok] = 1.0 - best[ok]
    return np.clip(out, 0.0, 2.0)


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def _clean_uniprot(values):
    return np.array(["" if str(v).strip() in UNMAPPED else str(v).strip() for v in values], dtype=object)


def cluster_pockets(unique_id, source, uniprot, series_key, pocket_f, pocket_g, *, pca_dims=50, target=0.9,
                    criterion_sources=("pdbbind",), row_unit_sources=("pdbbind",), medoid_max=4000,
                    n_grid=60, fallback_threshold=0.5):
    """Core routine on in-memory arrays (all rows usable). Returns (rows DataFrame, summary dict)."""
    t0 = time.time()
    uniprot = _clean_uniprot(uniprot)
    source = np.asarray(source, dtype=object)
    unique_id = np.asarray(unique_id, dtype=object)
    x = descriptor_matrix(pocket_f, pocket_g)
    z, evr = embed(x, pca_dims)
    keys = unit_keys(source, unique_id, np.asarray(series_key, dtype=object), set(row_unit_sources))
    codes, uniq_keys = pd.factorize(keys)
    members = pd.Series(np.arange(len(keys))).groupby(codes).apply(lambda s: s.to_numpy())
    rep = np.array([m[medoid(z[m], medoid_max)] for m in members])
    z_units = z[rep]
    unit_source = source[rep]
    unit_uniprot = uniprot[rep]
    if len(z_units) < 2:
        labels_units = np.ones(len(z_units), dtype=int)
        z_link, threshold, chose = None, float("nan"), "single unit"
    else:
        zn = z_units.copy()
        tiny = np.linalg.norm(zn, axis=1) < 1e-12
        zn[tiny, 0] = 1e-9  # cosine distance is undefined for a zero vector
        z_link = linkage(zn, method="average", metric="cosine")
        z_link[:, 2] = np.maximum(z_link[:, 2], 0.0)
        crit = np.where(np.isin(unit_source, list(criterion_sources)), unit_uniprot, "")
        threshold = choose_threshold(z_link, crit, target)
        chose = "co-clustering target"
        if threshold is None:
            threshold, chose = float(fallback_threshold), "fallback (no same-UniProt pairs)"
        labels_units = fcluster(z_link, threshold, criterion="distance")
    # relabel clusters by descending row count (0 = largest)
    row_labels_raw = labels_units[codes]
    counts = pd.Series(row_labels_raw).value_counts()
    order = {lab: i for i, lab in enumerate(sorted(counts.index, key=lambda lab: (-counts[lab], lab)))}
    unit_cluster = np.array([order[lab] for lab in labels_units])
    row_cluster = unit_cluster[codes]
    row_size = pd.Series(row_cluster).map(pd.Series(row_cluster).value_counts()).to_numpy()
    nearest = nearest_other_cluster(z, row_cluster, z_units, unit_cluster)

    rows = pd.DataFrame({"unique_id": unique_id, "source": source, "uniprot": uniprot,
                         "cluster_id": row_cluster.astype(int), "cluster_size": row_size.astype(int),
                         "nearest_other_cluster_dist": nearest, "unit_key": keys})

    crit_units = np.where(np.isin(unit_source, list(criterion_sources)), unit_uniprot, "")
    rate, group_mean, n_pairs = pair_rate(unit_cluster, crit_units)
    curve = []
    if z_link is not None:
        top = float(z_link[:, 2].max())
        grid = np.unique(np.concatenate([np.linspace(0.0, top, n_grid), [threshold]]))
        for t in grid:
            lab = fcluster(z_link, t, criterion="distance")
            r, gm, _ = pair_rate(lab, crit_units)
            curve.append({"threshold": float(t), "same_uniprot_rate": r, "group_mean_rate": gm,
                          "n_clusters": int(len(np.unique(lab))),
                          "largest_cluster_units": int(np.bincount(lab).max())})

    def mixing(mask):
        per = pd.DataFrame({"c": row_cluster[mask], "u": uniprot[mask]})
        per = per[per["u"] != ""]
        n_uni = per.groupby("c")["u"].nunique()
        return int((n_uni >= 2).sum())

    sizes_rows = pd.Series(row_cluster).value_counts().to_numpy()
    sizes_units = pd.Series(unit_cluster).value_counts().to_numpy()
    summary = {
        "rows": int(len(rows)), "units": int(len(z_units)),
        "units_by_source": {s: int((unit_source == s).sum()) for s in sorted(set(unit_source))},
        "pca_dims": int(z.shape[1]), "pca_explained_variance": float(evr.sum()),
        "input_dims": int(x.shape[1]),
        "criterion_sources": list(criterion_sources), "row_unit_sources": list(row_unit_sources),
        "target": target, "threshold": threshold, "threshold_rule": chose,
        "same_uniprot_rate_at_threshold": rate, "group_mean_rate_at_threshold": group_mean,
        "same_uniprot_pairs": n_pairs,
        "criterion_groups": int(len(set(crit_units) - {""})),
        "n_clusters": int(len(sizes_units)),
        "cluster_size_rows": _quantiles(sizes_rows), "cluster_size_units": _quantiles(sizes_units),
        "largest_clusters_rows": [int(v) for v in sizes_rows[:10]],
        "singleton_clusters_units": int((sizes_units == 1).sum()),
        "clusters_mixing_ge2_uniprots_all_rows": mixing(np.ones(len(rows), dtype=bool)),
        "clusters_mixing_ge2_uniprots_criterion_rows": mixing(np.isin(source, list(criterion_sources))),
        "nearest_other_cluster_dist": _quantiles(nearest[np.isfinite(nearest)]),
        "curve": curve, "seconds": time.time() - t0,
    }
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zernike", required=True, help="featurise_zernike.py npz")
    p.add_argument("--meta", required=True, help="paired_meta.csv (unique_id, source, uniprot, series_key)")
    p.add_argument("--out", required=True, help="pocket_clusters.csv")
    p.add_argument("--pca-dims", type=int, default=50)
    p.add_argument("--target", type=float, default=0.9)
    p.add_argument("--criterion-source", default="pdbbind", help="comma list")
    p.add_argument("--row-unit-sources", default="pdbbind", help="comma list ('' = group every source)")
    p.add_argument("--medoid-max", type=int, default=4000)
    p.add_argument("--fallback-threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    with np.load(args.zernike, allow_pickle=False) as zf:
        data = {k: zf[k] for k in ("unique_id", "source", "pocket_F", "pocket_G", "ok")}
    meta = pd.read_csv(args.meta, dtype=str, keep_default_na=False)
    if "kind" in meta.columns:
        meta = meta[meta["kind"].str.strip() == "0"]
    for col in ("uniprot", "series_key"):
        if col not in meta.columns:
            meta[col] = ""
    meta = meta.drop_duplicates("unique_id").set_index("unique_id")
    uid = data["unique_id"].astype(str)
    in_meta = np.isin(uid, meta.index.to_numpy())
    finite = np.isfinite(data["pocket_F"]).all(axis=(1, 2)) & np.isfinite(data["pocket_G"]).all(axis=(1, 2))
    usable = data["ok"].astype(bool) & in_meta & finite
    sel = meta.loc[uid[usable]]
    crit = tuple(s for s in args.criterion_source.split(",") if s)
    row_units = tuple(s for s in args.row_unit_sources.split(",") if s)
    rows, summary = cluster_pockets(uid[usable], sel["source"].to_numpy(), sel["uniprot"].to_numpy(),
                                    sel["series_key"].to_numpy(), data["pocket_F"][usable],
                                    data["pocket_G"][usable], pca_dims=args.pca_dims, target=args.target,
                                    criterion_sources=crit, row_unit_sources=row_units,
                                    medoid_max=args.medoid_max, fallback_threshold=args.fallback_threshold)
    # rows without a usable descriptor (failed featurisation, or meta rows absent from the npz)
    clustered = set(rows["unique_id"])
    missing = [u for u in meta.index if u not in clustered]
    if missing:
        extra = pd.DataFrame({"unique_id": missing, "source": meta.loc[missing, "source"].to_numpy(),
                              "uniprot": _clean_uniprot(meta.loc[missing, "uniprot"].to_numpy()),
                              "cluster_id": -1, "cluster_size": 0, "nearest_other_cluster_dist": np.nan,
                              "unit_key": ""})
        rows = pd.concat([rows, extra], ignore_index=True)
    order = {u: i for i, u in enumerate(meta.index)}
    rows = rows.sort_values("unique_id", key=lambda s: s.map(order)).reset_index(drop=True)
    summary.update(npz_rows=int(len(uid)), npz_not_ok=int((~data["ok"].astype(bool)).sum()),
                   npz_not_in_meta=int((~in_meta).sum()), rows_without_descriptor=len(missing))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.to_csv(args.out + ".tmp", index=False)
    os.replace(args.out + ".tmp", args.out)
    stem = args.out[:-4] if args.out.endswith(".csv") else args.out
    with open(stem + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"threshold {summary['threshold']:.4f} ({summary['threshold_rule']}): {summary['n_clusters']} clusters "
          f"over {summary['units']} units / {summary['rows']} rows; same-UniProt co-clustering "
          f"{summary['same_uniprot_rate_at_threshold']:.3f} over {summary['same_uniprot_pairs']} pairs; "
          f"{summary['clusters_mixing_ge2_uniprots_all_rows']} clusters mix >= 2 UniProts", flush=True)
    return rows, summary


if __name__ == "__main__":
    main()
