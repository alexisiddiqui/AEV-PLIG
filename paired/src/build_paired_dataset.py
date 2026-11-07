"""Build work/data/paired_meta.csv, paired_summary.json and processed/paired_<split>.pt from the
parent tree's graph pickles, *_processed.csv files and the PDBbind v2020 index tarball.

Augmented rows (--aug PICKLE:META, repeatable; --poses-pickle/--poses-meta is the arm-4 alias for
one pair): pose-format pickles + meta CSVs whose rows are made from PDBbind crystal anchors, with
`kind` 1 near-native, 2 decoy (arm 4, featurise_poses.py), 3 A-positive, 4 A-negative shape twin
(featurise_twins.py), 5 C-positive, 6 C-negative contact edit, 7 C-control exposed edit
(featurise_edits.py). Orphans are dropped per parent (kind 4 without kind 3, kind 3 without kind 4,
kind 5 without kind 6 or 7, kinds 6/7 without kind 5; counted as dropped_orphan_<kind>). Augmented rows
are appended after every original row, sorted by (parent row, kind, k)
whatever the input order; parent = the crystal's row; pK, series, ligand and group columns are
copied from the parent (so the hinge reference of kinds 2/4/6 is the parent's label, never the
twin's or the edited ligand's own). Rows of train anchors join `train`; rows of valid anchors go only
to the evaluation split: meta split `poses_valid` for kinds 1/2 (arm-4 label, unchanged) and
`aug_valid` for kinds 3-7. processed/paired_aug_valid.pt holds every PDBbind valid crystal row with
all its augmented rows (both labels); processed/paired_poses_valid.pt (arm 4) holds the crystals with
their kind-1/2 rows only and is written whenever an input carries kinds 1/2. Rows of test,
series_valid or group_valid anchors are dropped (counted). A shape twin (kind 4, meta column
twin_unique_id) is kept only while the twin ligand (lig_key of its source row) still has a kind-0
train row after the group holdout (the matching rule: a negative's ligand is a positive somewhere in
train).

Group-disjoint holdout (--groups pocket_groups.csv, --holdout-groups F, --holdout-group-min-rows N):
meta gains group_id ("" for rows the groups file does not list) and group_leak. Among groups with
>= N kind-0 rows in train + series_valid, round(F * n_eligible) groups are drawn with
numpy.random.default_rng(--seed); their train and series_valid rows (all sources) move to the split
`group_valid` (processed/paired_group_valid.pt); their valid / test rows stay and get group_leak = 1.
This runs before the augmented rows are attached, so no augmented row of a held-out anchor reaches
train. Without --groups the meta columns are arm 1's / arm 4's exactly. Original rows keep their
row/series/lig ids (kind 0, parent = row)."""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import gc
import json
import pickle
import re
import tarfile
import time

import numpy as np
import pandas as pd

from paired_data import SOURCE_ID, PairedDataset

SOURCES = ("pdbbind", "bindingnet", "bindingdb")
SPLITS = ("train", "valid", "test", "series_valid")
POSES_SPLIT = "poses_valid"   # valid-anchor rows of kinds 1/2 (arm 4)
AUG_SPLIT = "aug_valid"       # valid-anchor rows of kinds 3-7; the .pt of this name holds all kinds
GROUP_SPLIT = "group_valid"   # group-disjoint holdout
META_COLUMNS = ["row", "unique_id", "source", "split", "pK", "series_key", "series_id",
                "lig_key", "lig_id", "label_type", "uniprot", "anchor_pdb", "kind", "parent"]
GROUP_COLUMNS = ["group_id", "group_leak"]  # appended to META_COLUMNS with --groups
KINDS = (0, 1, 2)  # always reported in the summary (arm 4); other kinds when present
# kind -> unique_id tag: f"{pdb}::{tag}{k}"
AUG_TAG = {1: "pos", 2: "neg", 3: "tpos", 4: "twin", 5: "epos", 6: "cedit", 7: "xedit"}
POSE_KINDS = (1, 2)
POSE_TAG = {k: AUG_TAG[k] for k in POSE_KINDS}
TWIN_KIND = 4
ORPHAN_KINDS = (3, 4, 5, 6, 7)  # kinds whose rows need their scheme partner on the same parent
POSE_REQUIRED = ("unique_id", "pdb", "kind", "k")
POSE_STR_COLUMNS = {"unique_id": str, "pdb": str, "method": str, "split": str,
                    "twin_unique_id": str, "twin_source": str, "edit_rule": str, "site_class": str,
                    "partner_type": str}
# copied from the parent row onto its augmented rows (with its row as `parent`; plus the group
# columns when present)
POSE_COPIED = ["source", "pK", "series_key", "series_id", "lig_key", "lig_id", "label_type",
               "uniprot", "anchor_pdb"]
# the arm-4 summary["poses"] keys (for the --poses-pickle input)
POSE_COUNT_KEYS = ("meta_rows", "dropped_no_graph", "dropped_no_parent", "dropped_test",
                   "dropped_series_valid", "kept", "train", "poses_valid", "split_mismatch",
                   "pickle_only")
ID_COLUMN = {"pdbbind": "PDB_code", "bindingnet": "unique_identify", "bindingdb": "unique_id"}
STR_COLUMNS = {
    "pdbbind": ["PDB_code", "split_core"],
    "bindingnet": ["unique_identify", "target", "pdb", "compnd"],
    "bindingdb": ["folder", "pdb_file", "mol2_file", "Ki(nM)", "IC50(nM)", "Kd(nM)", "unique_id"],
}
TANIMOTO_MAX = 0.9  # create_pytorch_data.py keeps max_tanimoto_fep_benchmark < 0.9
UNMAPPED_UNIPROT = "------"
BINDINGDB_LABELS = (("Kd(nM)", "Kd"), ("Ki(nM)", "Ki"), ("IC50(nM)", "IC50"))  # priority order
LABEL_RE = re.compile(r"^(IC50|Kd|Ki)")
PEPTIDE_RE = re.compile(r"^\d+-mer$")
MOL2_RE = r"results_(\d+)\.mol2"
LIGAND_RE = re.compile(r"\(([^()]*)\)\s*$")


def _member_text(tf: tarfile.TarFile, basename: str) -> str:
    for member in tf.getmembers():
        if member.isfile() and os.path.basename(member.name) == basename:
            return tf.extractfile(member).read().decode("utf-8", errors="replace")
    raise KeyError(f"{basename} not found in {tf.name}")


def _index_lines(text: str):
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            yield line


def read_pdbbind_index(tgz_path: str) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """name_index: pdb -> UniProt as written ('------' when unmapped);
    data_index: pdb -> (affinity string e.g. 'Kd=49uM', ligand name inside the trailing parentheses)."""
    with tarfile.open(tgz_path, "r:*") as tf:
        name_text = _member_text(tf, "INDEX_general_PL_name.2020")
        data_text = _member_text(tf, "INDEX_general_PL_data.2020")
    name_index = {}
    for line in _index_lines(name_text):
        parts = line.split()
        name_index[parts[0].lower()] = parts[2]
    data_index = {}
    for line in _index_lines(data_text):
        parts = line.split()
        match = LIGAND_RE.search(line)
        data_index[parts[0].lower()] = (parts[4], match.group(1).strip() if match else "")
    return name_index, data_index


def _uniprot(name_index: dict[str, str], pdb: str) -> str:
    uniprot = name_index.get(pdb.lower(), "")
    return "" if uniprot == UNMAPPED_UNIPROT else uniprot


def _pdbbind_rows(df: pd.DataFrame, name_index, data_index) -> pd.DataFrame:
    df = df[df["max_tanimoto_fep_benchmark"] < TANIMOTO_MAX]
    bad = set(df["split_core"]) - {"train", "valid", "test"}
    if bad:
        raise ValueError(f"unexpected PDBbind split_core values {sorted(bad)}")
    pdb = df["PDB_code"].astype(str).tolist()
    uniprot = [_uniprot(name_index, p) for p in pdb]
    lig_key, label_type = [], []
    for p in pdb:
        affinity, name = data_index.get(p.lower(), ("", ""))
        token = name.split()[0] if name.split() else ""
        single = not token or PEPTIDE_RE.match(token)
        lig_key.append(f"pdbbind:single:{p}" if single else f"pdbbind:{token}")
        match = LABEL_RE.match(affinity)
        label_type.append(match.group(1) if match else "unknown")
    return pd.DataFrame({
        "unique_id": pdb,
        "source": "pdbbind",
        "split": df["split_core"].to_numpy(),
        "pK": df["-logKd/Ki"].to_numpy(dtype=float),
        "series_key": [f"pdbbind:{u}" if u else f"pdbbind:single:{p}" for p, u in zip(pdb, uniprot)],
        "lig_key": lig_key,
        "label_type": label_type,
        "uniprot": uniprot,
        "anchor_pdb": pdb,
    })


def _bindingnet_rows(df: pd.DataFrame, name_index) -> pd.DataFrame:
    df = df[df["max_tanimoto_fep_benchmark"] < TANIMOTO_MAX]
    pdb = df["pdb"].astype(str)
    return pd.DataFrame({
        "unique_id": df["unique_identify"].astype(str).to_numpy(),
        "source": "bindingnet",
        "split": "train",
        "pK": df["-logAffi"].to_numpy(dtype=float),
        "series_key": ("bindingnet:" + pdb).to_numpy(),
        "lig_key": ("bindingnet:" + df["compnd"].astype(str)).to_numpy(),
        "label_type": "unknown",
        "uniprot": [_uniprot(name_index, p) for p in pdb],
        "anchor_pdb": pdb.to_numpy(),
    })


def _bindingdb_rows(df: pd.DataFrame, name_index) -> pd.DataFrame:
    df = df[df["max_tanimoto_fep_benchmark"] < TANIMOTO_MAX]
    folder = df["folder"].astype(str)
    mol_id = df["mol2_file"].astype(str).str.extract(MOL2_RE, expand=False)
    if mol_id.isna().any():
        raise ValueError(f"mol2_file without results_<id>.mol2: {df.loc[mol_id.isna(), 'mol2_file'].tolist()[:5]}")
    label_type = np.full(len(df), "unknown", dtype=object)
    for column, name in reversed(BINDINGDB_LABELS):  # the highest-priority label is written last
        values = df[column]
        present = values.notna() & (values.astype(str).str.strip() != "")
        label_type[present.to_numpy()] = name
    anchor = folder.str.split("_").str[0].str.lower()
    return pd.DataFrame({
        "unique_id": df["unique_id"].astype(str).to_numpy(),
        "source": "bindingdb",
        "split": "train",
        "pK": df["pK"].to_numpy(dtype=float),
        "series_key": ("bindingdb:" + folder).to_numpy(),
        "lig_key": ("bindingdb:" + mol_id).to_numpy(),
        "label_type": label_type,
        "uniprot": [_uniprot(name_index, p) for p in anchor],
        "anchor_pdb": anchor.to_numpy(),
    })


def _subset(meta: pd.DataFrame, n_series: int) -> pd.DataFrame:
    """Keep the first n_series train series (sorted key) of each source plus every non-train row."""
    keep = meta["split"] != "train"
    for src in SOURCES:
        train_src = (meta["source"] == src) & (meta["split"] == "train")
        first = sorted(meta.loc[train_src, "series_key"].unique())[:n_series]
        keep |= train_src & meta["series_key"].isin(first)
    return meta[keep].reset_index(drop=True)


def _holdout(meta: pd.DataFrame, frac: float, min_size: int, seed: int) -> None:
    """Move whole BindingNet / BindingDB train series with >= min_size rows to series_valid."""
    for src in ("bindingnet", "bindingdb"):
        train_src = (meta["source"] == src) & (meta["split"] == "train")
        sizes = meta.loc[train_src].groupby("series_key").size()
        eligible = sorted(sizes.index[sizes >= min_size])
        size = int(round(frac * len(eligible)))
        if size == 0:
            continue
        chosen = np.random.default_rng(seed).choice(eligible, size, replace=False)
        meta.loc[train_src & meta["series_key"].isin(chosen), "split"] = "series_valid"


def build_meta(pdbbind: pd.DataFrame, bindingnet: pd.DataFrame, bindingdb: pd.DataFrame,
               name_index: dict[str, str], data_index: dict[str, tuple[str, str]],
               holdout_frac: float, holdout_min_size: int, seed: int,
               subset_series: int | None = None) -> pd.DataFrame:
    """The paired_meta.csv table: original filter and row order, series/ligand keys, holdout, dense ids."""
    meta = pd.concat([_pdbbind_rows(pdbbind, name_index, data_index),
                      _bindingnet_rows(bindingnet, name_index),
                      _bindingdb_rows(bindingdb, name_index)], ignore_index=True)
    if subset_series is not None:
        meta = _subset(meta, subset_series)
    _holdout(meta, holdout_frac, holdout_min_size, seed)
    meta["series_id"] = pd.factorize(meta["series_key"], sort=True)[0].astype(np.int64)
    meta["lig_id"] = pd.factorize(meta["lig_key"], sort=True)[0].astype(np.int64)
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    meta["kind"] = np.zeros(len(meta), dtype=np.int64)
    meta["parent"] = meta["row"]
    return meta[META_COLUMNS]


def read_groups(path: str) -> pd.Series:
    """pocket_groups.csv (pocket_groups.py: unique_id, group_id, ...) -> Series unique_id -> group_id."""
    groups = pd.read_csv(path, usecols=["unique_id", "group_id"], dtype=str, keep_default_na=False)
    groups = groups[groups["group_id"] != ""]
    dup = groups["unique_id"].duplicated(keep=False)
    if dup.any():
        clash = groups[dup].groupby("unique_id")["group_id"].nunique()
        if (clash > 1).any():
            raise ValueError(f"{path}: unique_ids with several group_ids, e.g. {clash.index[clash > 1].tolist()[:5]}")
        groups = groups.drop_duplicates("unique_id")
    return groups.set_index("unique_id")["group_id"]


def group_holdout(meta: pd.DataFrame, groups: pd.Series, frac: float, min_rows: int,
                  seed: int) -> tuple[pd.DataFrame, dict]:
    """Add group_id / group_leak to the kind-0 meta and move whole target groups to group_valid.

    Eligible: groups with >= min_rows rows in train + series_valid (the rows that move). Drawn:
    round(frac * n_eligible) of them (sorted ids) with default_rng(seed). Their train and
    series_valid rows of every source become group_valid; their valid / test rows stay and get
    group_leak = 1. Rows the groups file does not list get group_id "" and never move."""
    meta = meta.copy()
    if (meta["kind"] != 0).any():
        raise ValueError("group_holdout runs on the original rows, before augmented rows are added")
    meta["group_id"] = meta["unique_id"].map(groups).fillna("").astype(str)
    has_group = (meta["group_id"] != "").to_numpy()
    movable = meta["split"].isin(["train", "series_valid"]).to_numpy()
    sizes = meta.loc[movable & has_group].groupby("group_id").size()
    eligible = sorted(sizes.index[sizes >= min_rows])
    n_held = int(round(frac * len(eligible)))
    if frac > 0 and n_held == 0:
        raise ValueError(f"--holdout-groups {frac}: no group held out ({len(eligible)} groups with "
                         f">= {min_rows} train/series_valid rows)")
    chosen = (sorted(np.random.default_rng(seed).choice(eligible, n_held, replace=False).tolist())
              if n_held else [])
    held = meta["group_id"].isin(chosen).to_numpy() & has_group
    moved = meta[held & movable]
    leak = held & meta["split"].isin(["valid", "test"]).to_numpy()
    meta.loc[held & movable, "split"] = GROUP_SPLIT
    meta["group_leak"] = leak.astype(np.int64)
    info = {
        "groups_file_rows": int(len(groups)),
        "rows_without_group": {s: int(((meta["source"] == s).to_numpy() & ~has_group).sum()) for s in SOURCES},
        "n_groups": int(meta.loc[has_group, "group_id"].nunique()),
        "n_eligible": len(eligible), "frac": frac, "min_rows": min_rows, "seed": seed,
        "n_held": n_held, "held_groups": chosen,
        "moved_rows": {f"{src}:{split}": int(n) for (src, split), n
                       in moved.groupby(["source", "split"]).size().items()},
        "group_valid_rows": int(len(moved)),
        "leak_rows": {s: int((leak & (meta["split"] == s).to_numpy()).sum()) for s in ("valid", "test")},
    }
    return meta, info


def train_groups_in_group_valid(meta: pd.DataFrame) -> int:
    """Number of train rows whose (non-empty) group_id is also a group_valid row's (must be 0)."""
    if "group_id" not in meta:
        return 0
    held = set(meta.loc[meta["split"] == GROUP_SPLIT, "group_id"]) - {""}
    return int((meta["split"].eq("train") & meta["group_id"].isin(held)).sum())


def _aug_rows(meta: pd.DataFrame, aug: pd.DataFrame, keys, copied: list[str]) -> tuple[pd.DataFrame, dict]:
    """Validate one augmentation meta table and turn its usable rows into meta rows (no row ids yet).
    Returns (new rows with column _k, counts)."""
    missing = [c for c in POSE_REQUIRED if c not in aug]
    if missing:
        raise ValueError(f"augmentation meta lacks columns {missing}")
    aug = aug.copy()
    for col in ("unique_id", "pdb"):
        aug[col] = aug[col].astype(str)
    aug["kind"] = aug["kind"].astype(np.int64)
    aug["k"] = aug["k"].astype(np.int64)
    bad = set(aug["kind"]) - set(AUG_TAG)
    if bad:
        raise ValueError(f"augmentation meta has kinds {sorted(bad)}; expected one of {sorted(AUG_TAG)}")
    if aug["unique_id"].duplicated().any():
        raise ValueError(f"duplicate augmented ids, e.g. {aug.loc[aug.unique_id.duplicated(), 'unique_id'].tolist()[:5]}")
    expected_ids = [f"{p}::{AUG_TAG[kd]}{k}" for p, kd, k in zip(aug["pdb"], aug["kind"], aug["k"])]
    wrong = aug["unique_id"].to_numpy() != np.asarray(expected_ids, dtype=object)
    if wrong.any():
        raise ValueError(f"augmented ids disagree with pdb/kind/k: {aug.loc[wrong, 'unique_id'].tolist()[:5]}")

    counts = {"meta_rows": int(len(aug)),
              "meta_kinds": {str(k): int(n) for k, n in aug["kind"].value_counts().sort_index().items()}}
    keys = set(keys)
    has_graph = aug["unique_id"].isin(keys).to_numpy()
    counts["dropped_no_graph"] = int((~has_graph).sum())
    counts["pickle_only"] = int(len(keys - set(aug["unique_id"])))
    aug = aug[has_graph]
    parents = meta[(meta["source"] == "pdbbind") & (meta["kind"] == 0)].set_index("unique_id")
    has_parent = aug["pdb"].isin(parents.index).to_numpy()
    counts["dropped_no_parent"] = int((~has_parent).sum())
    aug = aug[has_parent]
    par = parents.loc[aug["pdb"].to_numpy()]
    parent_split = par["split"].to_numpy()
    counts["dropped_test"] = int((parent_split == "test").sum())
    counts["dropped_series_valid"] = int((parent_split == "series_valid").sum())
    counts["dropped_group_valid"] = int((parent_split == GROUP_SPLIT).sum())
    keep = np.isin(parent_split, ["train", "valid"])
    aug, par, parent_split = aug[keep], par[keep], parent_split[keep]
    # the generators write the parent's split; a disagreement means the inputs drifted apart
    counts["split_mismatch"] = (int((aug["split"].astype(str).to_numpy() != parent_split).sum())
                                if "split" in aug else 0)

    # matching rule for shape twins: the twin ligand must still be a kind-0 train positive
    # (twin_unique_id = the row the twin ligand was taken from)
    counts["dropped_twin_unknown"] = counts["dropped_twin_not_in_train"] = 0
    twin = (aug["kind"] == TWIN_KIND).to_numpy()
    counts["twin_rule"] = ("no twins" if not twin.any() else
                           "checked" if "twin_unique_id" in aug else "not checked: no twin_unique_id column")
    if twin.any() and "twin_unique_id" in aug:
        base = meta[meta["kind"] == 0]
        lig_of = base.set_index("unique_id")["lig_key"]
        train_ligs = set(base.loc[base["split"] == "train", "lig_key"])
        twin_lig = aug["twin_unique_id"].astype(str).map(lig_of)
        unknown = twin & twin_lig.isna().to_numpy()
        not_train = twin & ~unknown & ~twin_lig.isin(train_ligs).to_numpy()
        counts["dropped_twin_unknown"] = int(unknown.sum())
        counts["dropped_twin_not_in_train"] = int(not_train.sum())
        ok = ~(unknown | not_train)
        aug, par, parent_split = aug[ok], par[ok], parent_split[ok]

    # orphan matching (per parent, within this input): a negative needs the positive of its scheme and
    # the positive needs a negative / control of it, else the pair carries no matched contrast
    ok = _orphans(aug["pdb"].to_numpy(), aug["kind"].to_numpy(), counts)
    aug, par, parent_split = aug[ok], par[ok], parent_split[ok]

    new = pd.DataFrame({col: par[col].to_numpy() for col in copied})
    new["unique_id"] = aug["unique_id"].to_numpy()
    kind = aug["kind"].to_numpy()
    new["split"] = np.where(parent_split == "train", "train",
                            np.where(np.isin(kind, POSE_KINDS), POSES_SPLIT, AUG_SPLIT))
    new["kind"] = kind
    new["parent"] = par["row"].to_numpy().astype(np.int64)
    new["_k"] = aug["k"].to_numpy()
    counts.update(kept=int(len(new)), train=int((new["split"] == "train").sum()),
                  poses_valid=int((new["split"] == POSES_SPLIT).sum()),
                  aug_valid=int((new["split"] == AUG_SPLIT).sum()),
                  kept_kinds={f"{sp}:{k}": int(n) for (sp, k), n
                              in new.groupby(["split", "kind"]).size().items()})
    return new, counts


def _orphans(pdb: np.ndarray, kind: np.ndarray, counts: dict) -> np.ndarray:
    """Keep-mask after the orphan rules, in order: kind 4 without a kind 3 on its parent; kind 3 without a
    kind 4 left; kind 5 with neither a kind 6 nor a kind 7; kinds 6 / 7 without a kind 5. Sets
    counts[f"dropped_orphan_{k}"]."""
    def on(mask_kinds, keep):
        return np.isin(pdb, np.unique(pdb[keep & np.isin(kind, mask_kinds)]))

    keep = np.ones(len(pdb), dtype=bool)
    drops = {}
    drops[4] = (kind == 4) & ~on([3], keep)
    keep &= ~drops[4]
    drops[3] = (kind == 3) & ~on([4], keep)
    keep &= ~drops[3]
    drops[5] = (kind == 5) & ~on([6, 7], keep)
    drops[6] = (kind == 6) & ~on([5], keep)
    drops[7] = (kind == 7) & ~on([5], keep)
    for k in ORPHAN_KINDS:
        keep &= ~drops[k]
        counts[f"dropped_orphan_{k}"] = int(drops[k].sum())
    return keep


def add_aug(meta: pd.DataFrame, inputs) -> tuple[pd.DataFrame, dict]:
    """Append the augmented rows of every (name, aug_meta, pickle_keys) input after the original
    rows, sorted by (parent row, kind, k) over all inputs together; train anchors -> `train`, valid
    anchors -> `poses_valid` (kinds 1/2) or `aug_valid` (kinds 3-7). Returns (meta, {name: counts})."""
    columns = META_COLUMNS + [c for c in GROUP_COLUMNS if c in meta]
    copied = POSE_COPIED + [c for c in GROUP_COLUMNS if c in meta]
    parts, counts, seen = [], {}, {}
    for name, aug, keys in inputs:
        new, counts[name] = _aug_rows(meta, aug, keys, copied)
        clash = [u for u in new["unique_id"] if u in seen]
        if clash:
            raise ValueError(f"augmented ids of {name} repeat those of {seen[clash[0]]}: {clash[:5]}")
        seen.update((u, name) for u in new["unique_id"])
        parts.append(new)
    if not parts:
        return meta[columns], counts
    new = pd.concat(parts, ignore_index=True)
    new = new.sort_values(["parent", "kind", "_k"], kind="stable").reset_index(drop=True)
    new["row"] = np.arange(len(meta), len(meta) + len(new), dtype=np.int64)
    out = pd.concat([meta, new[columns]], ignore_index=True)
    return out[columns], counts


def add_poses(meta: pd.DataFrame, poses: pd.DataFrame, pose_keys) -> tuple[pd.DataFrame, dict]:
    """Arm-4 entry point: add_aug with one input; returns (meta, its counts)."""
    meta, counts = add_aug(meta, [("poses", poses, pose_keys)])
    return meta, counts["poses"]


def _valid_crystals(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[(meta["split"] == "valid") & (meta["kind"] == 0) & (meta["source"] == "pdbbind")]


def poses_valid_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """Every PDBbind valid crystal row plus the poses_valid rows, grouped by parent (crystal first)."""
    rows = pd.concat([_valid_crystals(meta), meta[meta["split"] == POSES_SPLIT]])
    return rows.sort_values(["parent", "row"], kind="stable")


def aug_valid_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """Every PDBbind valid crystal row plus all its augmented rows (poses_valid and aug_valid
    labels), grouped by parent (crystal first)."""
    rows = pd.concat([_valid_crystals(meta), meta[meta["split"].isin([POSES_SPLIT, AUG_SPLIT])]])
    return rows.sort_values(["parent", "row"], kind="stable")


def split_rows(meta: pd.DataFrame, with_poses: bool, with_aug: bool = False,
               with_groups: bool = False) -> dict[str, pd.DataFrame]:
    """The rows of each .pt file, in file order."""
    out = {split: meta[meta["split"] == split] for split in SPLITS}
    if with_groups:
        out[GROUP_SPLIT] = meta[meta["split"] == GROUP_SPLIT]
    if with_poses:
        out[POSES_SPLIT] = poses_valid_rows(meta)
    if with_aug:
        out[AUG_SPLIT] = aug_valid_rows(meta)
    return out


def count_pairs(meta: pd.DataFrame) -> int:
    """Number of same-series, different-ligand pairs (what pair_index would emit over these rows)."""
    if meta.empty:
        return 0
    n = meta.groupby("series_id").size()
    same = meta.groupby(["series_id", "lig_id"]).size()
    return int((n * (n - 1) // 2).sum() - (same * (same - 1) // 2).sum())


def summarise(meta: pd.DataFrame, missing: dict[str, int], args: dict,
              parts: dict[str, pd.DataFrame] | None = None, poses: dict | None = None,
              aug: dict | None = None, groups: dict | None = None) -> dict:
    """Rows / series / pairs / kinds per split (parts = the rows of each .pt; default: by split).
    Kinds 0-2 are always listed (arm 4), other kinds when the meta has them."""
    if parts is None:
        parts = split_rows(meta, with_poses=False)
    kinds_listed = sorted(set(KINDS) | set(int(k) for k in meta["kind"].unique()))
    rows, series, pairs, kinds = {}, {}, {}, {}
    for split, part in parts.items():
        rows[split] = {s: int((part["source"] == s).sum()) for s in SOURCE_ID}
        series[split] = {s: int(part.loc[part["source"] == s, "series_id"].nunique()) for s in SOURCE_ID}
        pairs[split] = count_pairs(part)
        kinds[split] = {str(k): int((part["kind"] == k).sum()) for k in kinds_listed}
    out = {"n_rows": int(len(meta)), "n_series": int(meta["series_id"].nunique()),
           "n_ligands": int(meta["lig_id"].nunique()), "rows": rows, "series": series,
           "pairs": pairs, "kinds": kinds, "missing_graphs": missing, "args": args}
    if poses is not None:
        out["poses"] = poses
    if aug is not None:
        out["aug"] = aug
    if groups is not None:
        out["group_holdout"] = groups
    return out


def parse_aug(spec: str) -> tuple[str, str]:
    """'PICKLE:META' -> (pickle path, meta path)."""
    parts = spec.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"--aug {spec!r}: expected PICKLE:META (paths without ':')")
    return parts[0], parts[1]


def aug_inputs(args) -> list[tuple[str, str, str]]:
    """(name, pickle, meta) per augmentation input: the --poses-* pair first (name 'poses'), then
    each --aug in order (name = the pickle's file stem, made unique)."""
    out = []
    if args.poses_pickle:
        out.append(("poses", args.poses_pickle, args.poses_meta))
    for pickle_path, meta_path in args.aug or []:
        name = os.path.splitext(os.path.basename(pickle_path))[0]
        taken = {n for n, _, _ in out}
        base, i = name, 2
        while name in taken:
            name, i = f"{base}_{i}", i + 1
        out.append((name, pickle_path, meta_path))
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-data", required=True, help="dir with {source}.pickle and {source}_processed.csv")
    parser.add_argument("--pdbbind-index-tgz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--holdout-min-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the series holdout and of the group holdout")
    parser.add_argument("--subset-series", type=int, default=None)
    parser.add_argument("--aug", type=parse_aug, action="append", default=[], metavar="PICKLE:META",
                        help="augmented rows (pose-format pickle and its meta CSV); repeatable")
    parser.add_argument("--poses-pickle", default=None,
                        help="pdbbind_poses.pickle from featurise_poses.py (with --poses-meta); "
                             "alias for one --aug pair")
    parser.add_argument("--poses-meta", default=None, help="pdbbind_poses_meta.csv (with --poses-pickle)")
    parser.add_argument("--groups", default=None,
                        help="pocket_groups.csv (unique_id, group_id): adds group_id / group_leak")
    parser.add_argument("--holdout-groups", type=float, default=0.0,
                        help="fraction of eligible target groups moved to group_valid (needs --groups)")
    parser.add_argument("--holdout-group-min-rows", type=int, default=5,
                        help="eligible groups have at least this many train + series_valid rows")
    args = parser.parse_args(argv)
    if (args.poses_pickle is None) != (args.poses_meta is None):
        parser.error("--poses-pickle and --poses-meta go together")
    if args.holdout_groups and args.groups is None:
        parser.error("--holdout-groups needs --groups")
    if not 0.0 <= args.holdout_groups <= 1.0:
        parser.error("--holdout-groups must be in [0, 1]")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    name_index, data_index = read_pdbbind_index(args.pdbbind_index_tgz)
    print(f"index: {len(name_index)} name rows, {len(data_index)} data rows")

    frames, graphs, missing = {}, {}, {}
    for src in SOURCES:
        frame = pd.read_csv(os.path.join(args.parent_data, f"{src}_processed.csv"), index_col=0,
                            dtype={c: str for c in STR_COLUMNS[src]})
        with open(os.path.join(args.parent_data, f"{src}.pickle"), "rb") as fh:
            src_graphs = pickle.load(fh)
        clash = graphs.keys() & src_graphs.keys()
        if clash:
            raise ValueError(f"{src} graph keys collide with earlier sources: {sorted(clash)[:5]}")
        ids = frame[ID_COLUMN[src]]
        has_graph = np.fromiter((uid in src_graphs for uid in ids), dtype=bool, count=len(ids))
        lost = (frame["max_tanimoto_fep_benchmark"] < TANIMOTO_MAX).to_numpy() & ~has_graph
        missing[src] = int(lost.sum())
        if missing[src]:
            print(f"WARNING {src}: {missing[src]} kept rows have no graph and are dropped, "
                  f"e.g. {ids[lost].tolist()[:5]}")
        frames[src] = frame[has_graph]
        graphs.update(src_graphs)
        del src_graphs
        print(f"{src}: {len(frame)} csv rows, {len(graphs)} graphs so far ({time.time() - t0:.0f} s)")

    meta = build_meta(frames["pdbbind"], frames["bindingnet"], frames["bindingdb"], name_index,
                      data_index, args.holdout_frac, args.holdout_min_size, args.seed,
                      subset_series=args.subset_series)
    del frames
    group_info = None
    if args.groups:
        meta, group_info = group_holdout(meta, read_groups(args.groups), args.holdout_groups,
                                         args.holdout_group_min_rows, args.seed)
        print(f"group holdout: {group_info['n_held']} of {group_info['n_eligible']} eligible groups "
              f"({group_info['n_groups']} with rows) -> {group_info['group_valid_rows']} group_valid rows "
              f"{group_info['moved_rows']}; leak rows kept in place {group_info['leak_rows']}; "
              f"rows without group {group_info['rows_without_group']}")
    print(meta.groupby(["split", "source"]).size().to_string())
    empty = [s for s in SPLITS if not (meta["split"] == s).any()]
    if empty:
        raise ValueError(f"empty splits {empty}; check --holdout-frac / --subset-series / --holdout-groups")

    inputs = aug_inputs(args)
    aug_counts, pose_kinds_given = None, False
    if inputs:
        tables = []
        for name, pickle_path, meta_path in inputs:
            with open(pickle_path, "rb") as fh:
                aug_graphs = pickle.load(fh)
            clash = graphs.keys() & aug_graphs.keys()
            if clash:
                raise ValueError(f"{name} graph keys collide with earlier graphs: {sorted(clash)[:5]}")
            table = pd.read_csv(meta_path, dtype=POSE_STR_COLUMNS, keep_default_na=False)
            tables.append((name, table, set(aug_graphs)))
            graphs.update(aug_graphs)
            del aug_graphs
            print(f"{name}: {len(table)} meta rows from {meta_path} ({time.time() - t0:.0f} s)")
        meta, aug_counts = add_aug(meta, tables)
        del tables
        # arm 4's poses_valid file whenever an input carries near-native / decoy rows
        pose_kinds_given = any(str(k) in c["meta_kinds"] for c in aug_counts.values() for k in POSE_KINDS)
        for name, counts in aug_counts.items():
            print(f"{name}: {counts}")
            for key in ("dropped_no_graph", "dropped_no_parent", "split_mismatch", "pickle_only",
                        "dropped_twin_unknown", "dropped_twin_not_in_train"):
                if counts[key]:
                    print(f"WARNING {name}: {key} = {counts[key]}")
            if counts["twin_rule"].startswith("not checked"):
                print(f"WARNING {name}: twin matching rule {counts['twin_rule']}")
        print(meta.groupby(["split", "kind"]).size().to_string())
    if group_info is not None:
        shared = train_groups_in_group_valid(meta)
        group_info["train_rows_sharing_group_valid_group"] = shared
        print(f"group holdout disjoint: train rows sharing a group_valid group_id = {shared}")
        if shared:
            raise ValueError(f"{shared} train rows share a group_id with group_valid rows")

    parts = split_rows(meta, with_poses=pose_kinds_given, with_aug=bool(inputs),
                       with_groups=args.holdout_groups > 0)
    os.makedirs(args.out_dir, exist_ok=True)
    for split, rows in parts.items():
        PairedDataset.build(args.out_dir, split, rows, graphs)
        print(f"wrote paired_{split}.pt: {len(rows)} graphs ({time.time() - t0:.0f} s)")
    del graphs
    gc.collect()

    meta.to_csv(os.path.join(args.out_dir, "paired_meta.csv"), index=False)
    pose_counts = None
    if args.poses_pickle:
        pose_counts = {k: aug_counts["poses"][k] for k in POSE_COUNT_KEYS}
    summary = summarise(meta, missing, vars(args), parts, pose_counts, aug_counts, group_info)
    with open(os.path.join(args.out_dir, "paired_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
