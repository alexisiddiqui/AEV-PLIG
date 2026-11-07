"""Locate the dataset roots inside the extracted archives and check that every file the AEV-PLIG generate
scripts will open exists, so a missing file fails in seconds instead of crashing a generate script an hour in.

  python featurise_check.py locate <kind> <search_dir>   -> prints the directory to symlink (exit 1 if not found)
      kind: pdbbind-refined | pdbbind-general | bindingnet | bindingdb
  python featurise_check.py check                         -> run from $WORK after the symlinks exist; prints
      expected/missing counts per dataset, exit 1 if anything is missing

Anchors come from the repo's *_processed.csv files in $WORK/data, so the archives' top-level folder names
(refined-set, v2020-other-PL, general-set, from_chembl_client, surflex, ...) do not matter.
"""
import os
import sys

import pandas as pd


def csvs():
    pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
    bnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
    bdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
    return pdbbind, bnet, bdb


def expected_files():
    """(dataset, relative path under work/) for every file the generate scripts read."""
    pdbbind, bnet, bdb = csvs()
    out = []
    for pdb, refined in zip(pdbbind["PDB_code"], pdbbind["refined"]):
        d = "data/pdbbind/refined-set/" if refined else "data/pdbbind/general-set/"
        out.append(("pdbbind-" + ("refined" if refined else "general"), f"{d}{pdb}/{pdb}_ligand.mol2"))
        out.append(("pdbbind-" + ("refined" if refined else "general"), f"{d}{pdb}/{pdb}_protein.pdb"))
    f = "data/bindingnet/from_chembl_client/"
    for _, r in bnet.iterrows():
        p, t, c = r["pdb"], r["target"], r["compnd"]
        out.append(("bindingnet", f"{f}{p}/target_{t}/{c}/{p}_{t}_{c}.sdf"))
        out.append(("bindingnet", f"{f}{p}/rec_h_opt.pdb"))
    f = "data/bindingdb/surflex/"
    for _, r in bdb.iterrows():
        out.append(("bindingdb", f"{f}{r['folder']}/{r['mol2_file']}"))
        out.append(("bindingdb", f"{f}{r['folder']}/{r['pdb_file']}"))
    return out


def anchor(kind):
    """Relative path (under the dataset root) of a few files that must exist below the root."""
    pdbbind, bnet, bdb = csvs()
    if kind.startswith("pdbbind"):
        sub = pdbbind[pdbbind["refined"] == (kind == "pdbbind-refined")]
        return [f"{p}/{p}_protein.pdb" for p in sub["PDB_code"].head(5)]
    if kind == "bindingnet":
        return [f"{r.pdb}/target_{r.target}/{r.compnd}/{r.pdb}_{r.target}_{r.compnd}.sdf" for r in bnet.head(5).itertuples()]
    if kind == "bindingdb":
        return [f"{r.folder}/{r.pdb_file}" for r in bdb.head(5).itertuples()]
    raise SystemExit(f"unknown kind {kind}")


def locate(kind, search_dir, max_depth=5):
    anchors = anchor(kind)
    base_depth = search_dir.rstrip("/").count("/")
    for root, dirs, _ in os.walk(search_dir):
        if root.count("/") - base_depth >= max_depth:
            dirs[:] = []
        hits = sum(os.path.exists(os.path.join(root, a)) for a in anchors)
        if hits >= max(1, len(anchors) // 2):
            return root
    return None


def main():
    if sys.argv[1] == "locate":
        root = locate(sys.argv[2], sys.argv[3])
        if root is None:
            print(f"no {sys.argv[2]} root under {sys.argv[3]} (anchors {anchor(sys.argv[2])[:2]})", file=sys.stderr)
            sys.exit(1)
        print(root)
    elif sys.argv[1] == "check":
        exp = expected_files()
        df = pd.DataFrame(exp, columns=["dataset", "path"]).drop_duplicates()
        df["exists"] = [os.path.exists(p) for p in df["path"]]
        bad = 0
        for ds, g in df.groupby("dataset"):
            miss = g[~g["exists"]]
            print(f"{ds}: {len(g)} files expected, {len(miss)} missing")
            for p in miss["path"].head(10):
                print("   missing", p)
            bad += len(miss)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
