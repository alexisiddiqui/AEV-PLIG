"""Build the descriptor side file of arm P (the moments heads' targets): one pocket and one ligand 3D Zernike
descriptor F (7 channels x 49 invariants = 343, docs/zernike_descriptors.md) per row of a paired_meta.csv,
aligned to the meta's `row` ids, with masks and the train standardisation.

  build_zernike_side.py --meta work/data_ac/paired_meta.csv --zernike work/data/zernike.npz \\
                        [--twins-meta work/data/twins_meta.csv] --out work/data_ac/paired_zernike.npz

Which descriptor a row gets (the npz holds kind-0 rows only, keyed by unique_id):
  kind 0                    pocket and ligand of its own unique_id;
  kinds 1, 2 (poses)        pocket and ligand of the parent (same pocket frame, same molecule; for a decoy pose the
                            parent's pocket is an approximation: the pose may sit elsewhere);
  kinds 3, 5 (A / C positive: the anchor's own ligand regenerated or null-edited)   pocket and ligand of the parent;
  kind 4 (shape twin)       pocket of the parent, ligand of the row twins_meta.csv names in twin_unique_id;
  kinds 6, 7 (edits)        pocket of the parent, ligand masked (the edited molecule has no descriptor);
  anything else             masked.
A row whose source id is not in the npz, or whose descriptor is not finite (featurisation failed: NaN), is masked
for that descriptor. Meta files without a kind column (arm 1) are all kind 0 (parent = row). A meta with kind-4
rows needs --twins-meta.

Standardisation: mean and sd (ddof 0) per column over the kind-0 rows of split `train` with a descriptor (pocket
and ligand separately); a column with sd < 1e-8 (or no train row) gets sd 1.

Output (np.savez, uncompressed; written atomically): row [N] int64, unique_id / split [N] str, kind / parent
[N] int64, series_id / lig_id [N] int64 (-1 when the meta lacks them; the trainer checks them against the .pt
files), pocket_F / ligand_F [N, 343] float32 raw (NaN where masked), mask_pocket / mask_ligand [N] bool,
pocket_src / ligand_src [N] str (the unique_id whose descriptor was used, "" when masked), pocket_mean / pocket_sd
/ ligand_mean / ligand_sd [343] float32, and 0-d strings meta_path, zernike_path, twins_meta_path, summary (json).
load_side(path) reads it back and adds the standardised arrays (masked rows 0) and a row -> index lookup.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

F_SHAPE = (7, 49)
DIM = F_SHAPE[0] * F_SHAPE[1]
TARGETS = ("pocket", "ligand")
SD_MIN = 1e-8
TWIN_KIND = 4
PARENT_BOTH = (1, 2, 3, 5)      # pocket and ligand of the parent
PARENT_POCKET_ONLY = (6, 7)     # edited ligands: no ligand descriptor
KNOWN_KINDS = (0, TWIN_KIND) + PARENT_BOTH + PARENT_POCKET_ONLY


def read_meta(path):
    """paired_meta.csv with the columns the side file needs (kind 0 / parent = row when absent)."""
    meta = pd.read_csv(path, dtype={"unique_id": str, "split": str}, keep_default_na=False, na_values=[""])
    missing = [c for c in ("row", "unique_id", "split") if c not in meta]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    meta["unique_id"] = meta["unique_id"].fillna("").astype(str)
    meta["split"] = meta["split"].fillna("").astype(str)
    meta["row"] = meta["row"].astype(np.int64)
    if meta["row"].duplicated().any():
        raise ValueError(f"{path}: duplicate row ids")
    meta["kind"] = meta["kind"].astype(np.int64) if "kind" in meta else np.zeros(len(meta), dtype=np.int64)
    meta["parent"] = meta["parent"].astype(np.int64) if "parent" in meta else meta["row"]
    for col in ("series_id", "lig_id"):
        meta[col] = meta[col].astype(np.int64) if col in meta else np.full(len(meta), -1, dtype=np.int64)
    return meta


def read_descriptors(path):
    """(unique_id -> npz position, pocket_F [M, 343], ligand_F [M, 343]) float32; a row with ok False is NaN."""
    with np.load(path, allow_pickle=False) as z:
        uid = z["unique_id"].astype(str)
        pocket = z["pocket_F"].astype(np.float32).reshape(len(uid), -1)
        ligand = z["ligand_F"].astype(np.float32).reshape(len(uid), -1)
        ok = z["ok"].astype(bool) if "ok" in z.files else np.ones(len(uid), dtype=bool)
    for name, arr in (("pocket_F", pocket), ("ligand_F", ligand)):
        if arr.shape[1] != DIM:
            raise ValueError(f"{path}: {name} has {arr.shape[1]} values per row, expected {DIM} {F_SHAPE}")
    pocket[~ok] = np.nan
    ligand[~ok] = np.nan
    pos = pd.Series(np.arange(len(uid)), index=uid)
    if pos.index.duplicated().any():
        dup = pos.index[pos.index.duplicated()].tolist()[:5]
        raise ValueError(f"{path}: duplicate unique_id {dup}")
    return pos, pocket, ligand


def read_twins(path):
    """twins_meta.csv -> Series unique_id (the twin row, '<pdb>::twin<k>') -> twin_unique_id."""
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("unique_id", "twin_unique_id") if c not in t]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    t = t.drop_duplicates("unique_id")
    return t.set_index("unique_id")["twin_unique_id"]


def source_ids(meta, twins=None):
    """(pocket source unique_id, ligand source unique_id) per meta row ('' = masked) and counts by reason."""
    kind = meta["kind"].to_numpy()
    uid = meta["unique_id"].to_numpy(dtype=object)
    parent_uid = meta["parent"].map(meta.set_index("row")["unique_id"]).fillna("").to_numpy(dtype=object)
    pocket = np.where(kind == 0, uid, parent_uid).astype(object)
    ligand = np.full(len(meta), "", dtype=object)
    ligand[kind == 0] = uid[kind == 0]
    both = np.isin(kind, PARENT_BOTH)
    ligand[both] = parent_uid[both]
    twin = kind == TWIN_KIND
    counts = {"rows": int(len(meta)), "kinds": {str(k): int(n) for k, n in zip(*np.unique(kind, return_counts=True))},
              "no_parent_row": int(((kind != 0) & (parent_uid == "")).sum()),
              "ligand_masked_edit": int(np.isin(kind, PARENT_POCKET_ONLY).sum()),
              "unknown_kind": int((~np.isin(kind, KNOWN_KINDS)).sum())}
    if twin.any():
        if twins is None:
            raise ValueError(f"the meta has {int(twin.sum())} kind-4 (shape twin) rows: pass --twins-meta "
                             "(twin_unique_id gives their ligand descriptor)")
        tw = pd.Series(uid[twin]).map(twins)
        counts["twin_without_twin_id"] = int(tw.isna().sum())
        ligand[twin] = tw.fillna("").to_numpy(dtype=object)
    pocket[~np.isin(kind, KNOWN_KINDS)] = ""
    return pocket, ligand, counts


def gather(src, pos, desc):
    """(values [N, 343] float32, mask [N]) for source ids src ('' masked) from the npz arrays."""
    src = pd.Series(src, dtype=object)
    p = src.map(pos)
    found = p.notna().to_numpy() & (src != "").to_numpy()
    out = np.full((len(src), DIM), np.nan, dtype=np.float32)
    idx = p[found].to_numpy(dtype=np.int64)
    out[found] = desc[idx]
    mask = found & np.isfinite(out).all(axis=1)
    out[~mask] = np.nan
    return out, mask, int((~found & (src != "").to_numpy()).sum())


def standardisation(values, mask, train):
    """(mean, sd) float32 [343] over rows mask & train (ddof 0); sd < SD_MIN or no such row -> 1, mean 0 then."""
    sel = mask & train
    if not sel.any():
        return np.zeros(values.shape[1], dtype=np.float32), np.ones(values.shape[1], dtype=np.float32)
    x = values[sel].astype(np.float64)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(np.isfinite(sd) & (sd >= SD_MIN), sd, 1.0)
    return mean.astype(np.float32), sd.astype(np.float32)


def build(meta, pos, pocket_desc, ligand_desc, twins=None):
    """The side-file arrays (dict) and a summary dict."""
    psrc, lsrc, counts = source_ids(meta, twins)
    pocket, mask_p, miss_p = gather(psrc, pos, pocket_desc)
    ligand, mask_l, miss_l = gather(lsrc, pos, ligand_desc)
    train = (meta["split"].to_numpy() == "train") & (meta["kind"].to_numpy() == 0)
    p_mean, p_sd = standardisation(pocket, mask_p, train)
    l_mean, l_sd = standardisation(ligand, mask_l, train)
    kind = meta["kind"].to_numpy()
    counts.update(
        source_not_in_npz={"pocket": miss_p, "ligand": miss_l},
        with_descriptor={"pocket": int(mask_p.sum()), "ligand": int(mask_l.sum())},
        train_rows_for_standardisation={"pocket": int((mask_p & train).sum()), "ligand": int((mask_l & train).sum())},
        coverage_by_kind={str(k): {"rows": int((kind == k).sum()), "pocket": int(mask_p[kind == k].sum()),
                                   "ligand": int(mask_l[kind == k].sum())} for k in np.unique(kind)},
        coverage_by_split={sp: {"rows": int((meta["split"] == sp).sum()),
                                "pocket": int(mask_p[(meta["split"] == sp).to_numpy()].sum()),
                                "ligand": int(mask_l[(meta["split"] == sp).to_numpy()].sum())}
                           for sp in sorted(meta["split"].unique())})
    arrays = {
        "row": meta["row"].to_numpy(dtype=np.int64), "unique_id": meta["unique_id"].to_numpy(dtype=str),
        "split": meta["split"].to_numpy(dtype=str), "kind": kind.astype(np.int64),
        "parent": meta["parent"].to_numpy(dtype=np.int64), "series_id": meta["series_id"].to_numpy(dtype=np.int64),
        "lig_id": meta["lig_id"].to_numpy(dtype=np.int64),
        "pocket_F": pocket, "ligand_F": ligand, "mask_pocket": mask_p, "mask_ligand": mask_l,
        "pocket_src": np.where(mask_p, psrc, "").astype(str), "ligand_src": np.where(mask_l, lsrc, "").astype(str),
        "pocket_mean": p_mean, "pocket_sd": p_sd, "ligand_mean": l_mean, "ligand_sd": l_sd,
    }
    return arrays, counts


def write_npz(path, arrays):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:  # a file object: np.savez must not append ".npz" to the temp name
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def load_side(path):
    """The side file as a dict of numpy arrays, plus per target "<t>_std" [N, 343] float32 ((F - mean) / sd,
    masked rows 0) and "lookup" (row id -> index into the arrays, -1 for a row the file does not hold; length
    max row + 1)."""
    with np.load(path, allow_pickle=False) as z:
        side = {k: z[k] for k in z.files}
    for t in TARGETS:
        std = (side[f"{t}_F"] - side[f"{t}_mean"]) / side[f"{t}_sd"]
        std[~side[f"mask_{t}"]] = 0.0
        side[f"{t}_std"] = np.ascontiguousarray(std, dtype=np.float32)
    rows = side["row"].astype(np.int64)
    if len(rows) and rows.min() < 0:
        raise ValueError(f"{path}: negative row ids")
    lookup = np.full(int(rows.max()) + 1 if len(rows) else 0, -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows))
    side["lookup"] = lookup
    return side


def side_index(side, rows):
    """Index into the side arrays per row id (numpy), -1 where the file has no such row."""
    rows = np.asarray(rows, dtype=np.int64)
    lookup = side["lookup"]
    ok = (rows >= 0) & (rows < len(lookup))
    out = np.full(rows.shape, -1, dtype=np.int64)
    out[ok] = lookup[rows[ok]]
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True, help="paired_meta.csv of the build the side file belongs to")
    ap.add_argument("--zernike", required=True, help="zernike.npz from featurise_zernike.py (kind-0 rows)")
    ap.add_argument("--twins-meta", default=None, help="twins_meta.csv (needed when the meta has kind-4 rows)")
    ap.add_argument("--out", required=True, help="output .npz (e.g. <data dir>/paired_zernike.npz)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for path in [args.meta, args.zernike] + ([args.twins_meta] if args.twins_meta else []):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: missing {path}")
    meta = read_meta(args.meta)
    pos, pocket, ligand = read_descriptors(args.zernike)
    twins = read_twins(args.twins_meta) if args.twins_meta else None
    try:
        arrays, summary = build(meta, pos, pocket, ligand, twins)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    summary.update(meta=os.path.abspath(args.meta), zernike=os.path.abspath(args.zernike),
                   twins_meta=os.path.abspath(args.twins_meta) if args.twins_meta else None,
                   npz_rows=int(len(pos)))
    arrays.update(meta_path=np.array(summary["meta"]), zernike_path=np.array(summary["zernike"]),
                  twins_meta_path=np.array(summary["twins_meta"] or ""), summary=np.array(json.dumps(summary)))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    write_npz(args.out, arrays)
    print(json.dumps(summary, indent=1))
    for t in TARGETS:
        if summary["train_rows_for_standardisation"][t] == 0:
            print(f"WARNING: no kind-0 train row has a {t} descriptor: mean 0, sd 1")
    print(f"wrote {args.out}: {len(meta)} rows, pocket {summary['with_descriptor']['pocket']}, "
          f"ligand {summary['with_descriptor']['ligand']}")
    return summary


if __name__ == "__main__":
    main()
