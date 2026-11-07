"""Generate perturbed ligand poses for every PDBbind complex of paired_meta.csv and featurise them
with the verbatim copies of the clone's functions (aevplig_featurise).

Inputs: --pdbbind-root (holds refined-set/ and general-set/ with <pdb>/<pdb>_{ligand.mol2,protein.pdb}),
--meta (arm 1 paired_meta.csv; rows with source == pdbbind give pdb = unique_id and split),
--pdbbind-csv (the parent's pdbbind_processed.csv; column `refined` picks the folder, as the clone).
Per complex (one Pool task): the ligand is read as the clone does (MolFromMol2File, AddHs with
coordinates), the protein table with LoadPDBasDF (its X, Y, Z = the typed heavy atoms the AEV sees,
used for the clash/contact rules), rng = default_rng(zlib.crc32(pdb)), poses from
pose_augment.generate_poses, and each pose is featurised with GetMolAEVs_extended + mol_to_graph.
Outputs in --out-dir:
  pdbbind_poses.pickle        {"<pdb>::pos<k>" | "<pdb>::neg<k>": (c_size, features, edge_index, edge_features)}
  pdbbind_poses_meta.csv      unique_id, pdb, kind, k, method, rmsd, min_dist, contact_frac, attempts, split,
                              relief_steps (relief Monte Carlo steps; 0 = none needed), n_within_3A
                              (ligand heavy atoms with a protein heavy atom closer than 3.0 A); the two
                              added columns are last so readers of the earlier columns are unaffected
  pdbbind_poses_summary.json  counts, quantiles per kind (rmsd, min_dist, contact_frac, attempts;
                              relief_steps and n_within_3A under their own *_quantiles keys), the
                              crystal pose's min_dist / n_within_3A quantiles over processed complexes,
                              the close-contact cue AUROCs on min_dist alone (cue_auroc_min_dist:
                              P(min_dist of group A > group B), ties 1/2; 0.5 = no cue, far below 0.5 =
                              reversed cue), failures (complexes with < K poses of a kind), errors,
                              relaxed-clash complexes, identity-check result
--check-identity N (with --parent-pickle) runs ONLY the identity check: it re-featurises the crystal
pose of the first N complexes (by pdb id) through the same worker path and requires np.array_equal on
all four tuple members against the parent's pickle. It prints "IDENTITY OK n=N" and writes
pdbbind_poses_identity.json (read back into the summary of a later full run), or prints the first
mismatch and exits 3. It first checks that the copied source blocks equal
$AEVPLIG_REPO/generate_pdbbind_graphs.py when that file exists (mismatch: exit 3).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import json
import pickle
import time
import zlib
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem

import aevplig_featurise as aevf
from scipy.spatial import cKDTree

from pose_augment import CLOSE_CUTOFF, generate_poses, heavy_xyz, min_dist, n_within

META_COLUMNS = ["unique_id", "pdb", "kind", "k", "method", "rmsd", "min_dist", "contact_frac",
                "attempts", "split", "relief_steps", "n_within_3A"]
KIND_TAG = {1: "pos", 2: "neg"}
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
CLASH = 2.2          # generate_poses default; complexes whose crystal pose is closer are "relaxed"
MAX_ATTEMPTS = 60    # generate_poses default; decoys with more attempts used the 3.0 A fallback
IDENTITY_FAIL = 3
POSES_PICKLE = "pdbbind_poses.pickle"
POSES_META = "pdbbind_poses_meta.csv"
POSES_SUMMARY = "pdbbind_poses_summary.json"
IDENTITY_JSON = "pdbbind_poses_identity.json"

_ATOM_KEYS = None
_ATOM_MAP = None


def _init_worker(atom_keys_path):
    global _ATOM_KEYS, _ATOM_MAP
    import torch
    torch.set_num_threads(1)
    _ATOM_KEYS, _ATOM_MAP = aevf.load_atom_keys(atom_keys_path)


def load_ligand(mol_path):
    """As the clone: Chem.MolFromMol2File then Chem.AddHs(addCoords=True); None if unreadable."""
    if not os.path.exists(mol_path):
        raise FileNotFoundError(mol_path)
    mol = Chem.MolFromMol2File(mol_path)
    if mol is None:
        return None
    return Chem.AddHs(mol, addCoords=True)


def featurise_mol(protein_path, mol):
    """The clone's per-complex featurisation (generate_pdbbind_graphs.py lines 264-265)."""
    mol_df, aevs = aevf.GetMolAEVs_extended(protein_path, mol, _ATOM_KEYS, aevf.radial_coefs, _ATOM_MAP)
    return aevf.mol_to_graph(mol, mol_df, aevs)


def crystal_complex(task):
    pdb, _split, mol_path, protein_path = task[:4]
    try:
        mol = load_ligand(mol_path)
        if mol is None:
            return pdb, None, "can't read molecule structure"
        if not os.path.exists(protein_path):
            raise FileNotFoundError(protein_path)
        return pdb, featurise_mol(protein_path, mol), ""
    except Exception as e:  # noqa: BLE001 - reported as an identity mismatch
        return pdb, None, f"{type(e).__name__}: {e}"


def pose_complex(task):
    pdb, split, mol_path, protein_path, k_pos, k_neg = task
    t0 = time.time()
    res = {"pdb": pdb, "split": split, "graphs": {}, "rows": [], "error": "", "crystal_min_dist": None,
           "crystal_n_within_3A": None}
    try:
        mol = load_ligand(mol_path)
        if mol is None:
            raise ValueError("can't read molecule structure")
        if not os.path.exists(protein_path):
            raise FileNotFoundError(protein_path)
        target = aevf.LoadPDBasDF(protein_path, _ATOM_KEYS)
        target_xyz = target[["X", "Y", "Z"]].to_numpy(dtype=float)
        cmd = min_dist(heavy_xyz(mol), target_xyz)
        res["crystal_min_dist"] = cmd if np.isfinite(cmd) else None
        tree = cKDTree(target_xyz.reshape(-1, 3))
        res["crystal_n_within_3A"] = n_within(heavy_xyz(mol), tree, CLOSE_CUTOFF)
        rng = np.random.default_rng(zlib.crc32(pdb.encode()))
        graphs, rows = {}, []
        for p in generate_poses(mol, target_xyz, k_pos, k_neg, rng):
            uid = f"{pdb}::{KIND_TAG[p.kind]}{p.k}"
            graphs[uid] = featurise_mol(protein_path, p.mol)
            rows.append({"unique_id": uid, "pdb": pdb, "kind": p.kind, "k": p.k, "method": p.method,
                         "rmsd": p.rmsd, "min_dist": p.min_dist, "contact_frac": p.contact_frac,
                         "attempts": p.attempts, "split": split, "relief_steps": p.relief_steps,
                         "n_within_3A": n_within(heavy_xyz(p.mol), tree, CLOSE_CUTOFF)})
        res["graphs"], res["rows"] = graphs, rows
    except Exception as e:  # noqa: BLE001 - one bad complex must not stop the run; it is recorded
        res["error"] = f"{type(e).__name__}: {e}"
    res["seconds"] = time.time() - t0
    return res


def _run(fn, tasks, workers, atom_keys, label):
    if not tasks:
        return []
    out = []
    t0 = time.time()
    step = max(1, len(tasks) // 20)

    def progress(i):
        if i % step == 0 or i == len(tasks):
            print(f"{label} {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)

    if workers <= 1:
        import torch
        n_threads = torch.get_num_threads()
        _init_worker(atom_keys)
        try:
            for i, t in enumerate(tasks, 1):
                out.append(fn(t))
                progress(i)
        finally:
            torch.set_num_threads(n_threads)
        return out
    with Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(atom_keys,)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            out.append(r)
            progress(i)
    return out


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def read_complexes(meta_csv, pdbbind_csv, root):
    """[(pdb, split, mol2 path, protein path)] sorted by pdb id, and pdb ids absent from the csv."""
    meta = pd.read_csv(meta_csv, usecols=["unique_id", "source", "split"], dtype=str)
    meta = meta[meta["source"] == "pdbbind"]
    pdbb = pd.read_csv(pdbbind_csv, index_col=0, dtype={"PDB_code": str})
    refined = {p: _as_bool(r) for p, r in zip(pdbb["PDB_code"], pdbb["refined"])}
    complexes, missing = [], []
    for pdb, split in sorted(zip(meta["unique_id"], meta["split"])):
        if pdb not in refined:
            missing.append(pdb)
            continue
        folder = os.path.join(root, "refined-set" if refined[pdb] else "general-set", pdb)
        complexes.append((pdb, split, os.path.join(folder, f"{pdb}_ligand.mol2"),
                          os.path.join(folder, f"{pdb}_protein.pdb")))
    return complexes, missing


def compare_graphs(ref, new):
    """'' when the two graph tuples are identical, else a description of the first difference."""
    if len(ref) != 4 or len(new) != 4:
        return f"tuple length {len(ref)} vs {len(new)}"
    if ref[0] != new[0]:
        return f"c_size {ref[0]} vs {new[0]}"
    for name, a, b in zip(("features", "edge_index", "edge_features"), ref[1:], new[1:]):
        a, b = np.asarray(a), np.asarray(b)
        if not np.array_equal(a, b):
            detail = f"shape {a.shape} vs {b.shape}"
            if a.shape == b.shape and a.size:
                detail += f", max abs diff {np.max(np.abs(a.astype(float) - b.astype(float))):.3g}"
            return f"{name} differs ({detail})"
    return ""


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def check_identity(args, complexes):
    path = os.path.join(args.out_dir, IDENTITY_JSON)
    result = {"ok": False, "n": 0, "n_requested": args.check_identity, "mismatch": "",
              "parent_pickle": args.parent_pickle, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def fail(msg):
        result["mismatch"] = msg
        _write_json(result, path)
        print(f"IDENTITY MISMATCH {msg}", flush=True)
        sys.exit(IDENTITY_FAIL)

    clone_script = os.path.join(os.environ["AEVPLIG_REPO"], "generate_pdbbind_graphs.py")
    if os.path.exists(clone_script):
        bad = aevf.source_mismatches(clone_script)
        result["source_check"] = "ok" if not bad else "mismatch: " + ",".join(bad)
        if bad:
            fail(f"source of {','.join(bad)} differs from {clone_script}")
        print(f"SOURCE OK {len(aevf.COPIED_FUNCTIONS)} functions + constants match {clone_script}")
    else:
        result["source_check"] = "skipped"
        print(f"SOURCE CHECK SKIPPED: {clone_script} not found")
    if not args.parent_pickle:
        fail("--check-identity needs --parent-pickle")
    chosen = complexes[:args.check_identity]
    if not chosen:
        fail("no complexes to check")
    # featurise first, then load the parent pickle, so the Pool forks from a small process
    results = sorted(_run(crystal_complex, chosen, args.workers, args.atom_keys, "identity"),
                     key=lambda r: r[0])
    with open(args.parent_pickle, "rb") as f:
        parent = pickle.load(f)
    for pdb, graph, err in results:
        if graph is None:
            fail(f"{pdb}: featurisation failed: {err}")
        if pdb not in parent:
            fail(f"{pdb}: not in {args.parent_pickle}")
        msg = compare_graphs(parent[pdb], graph)
        if msg:
            fail(f"{pdb}: {msg}")
    result.update(ok=True, n=len(results))
    _write_json(result, path)
    print(f"IDENTITY OK n={len(results)}", flush=True)


def _quantiles(values):
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return {}
    return {str(q): float(v.quantile(q)) for q in QUANTILES}


def _auroc(a, b):
    """P(a > b) + P(a == b) / 2 for samples a, b (rank form of the Mann-Whitney U); None if empty."""
    a = pd.Series(a, dtype=float).dropna().to_numpy()
    b = pd.Series(b, dtype=float).dropna().to_numpy()
    if not len(a) or not len(b):
        return None
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(u / (len(a) * len(b)))


def summarise(results, meta, missing, args, seconds):
    want = {1: args.k_pos, 2: args.k_neg}
    ok = [r for r in results if not r["error"]]
    errors = [[r["pdb"], r["error"]] for r in results if r["error"]]
    errors += [[p, "not in pdbbind csv"] for p in missing]
    splits = sorted({r["split"] for r in results})
    short = {}
    failures = {}
    for kind, tag in KIND_TAG.items():
        rows = []
        for r in ok:
            n = sum(row["kind"] == kind for row in r["rows"])
            if n < want[kind]:
                rows.append([r["pdb"], r["split"], n])
        short[tag] = rows
        failures[tag] = {"n": len(rows), "by_split": dict(Counter(x[1] for x in rows)),
                         "by_n_poses": {str(k): v for k, v in sorted(Counter(x[2] for x in rows).items())}}
    relaxed = [[r["pdb"], round(r["crystal_min_dist"], 3)] for r in ok
               if r["crystal_min_dist"] is not None and r["crystal_min_dist"] < CLASH]
    crystal_md = [r["crystal_min_dist"] for r in ok if r["crystal_min_dist"] is not None]
    identity_path = os.path.join(args.out_dir, IDENTITY_JSON)
    identity = None
    if os.path.exists(identity_path):
        with open(identity_path) as f:
            identity = json.load(f)
    return {
        "n_complexes": len(results) + len(missing),
        "n_complexes_ok": len(ok),
        "n_poses": int(len(meta)),
        "k_pos": args.k_pos,
        "k_neg": args.k_neg,
        "complexes_by_split": dict(Counter(r["split"] for r in results)),
        "counts_split_kind": {s: {str(k): int(((meta["split"] == s) & (meta["kind"] == k)).sum())
                                  for k in KIND_TAG} for s in splits},
        "counts_kind": {str(k): int((meta["kind"] == k).sum()) for k in KIND_TAG},
        "counts_method": {m: int((meta["method"] == m).sum()) for m in ("nn", "flip", "conf")},
        "decoys_fallback_3A": int(((meta["kind"] == 2) & (meta["attempts"] > MAX_ATTEMPTS)).sum()),
        "failures": failures,
        "n_errors": len(errors),
        "n_relaxed_clash": len(relaxed),
        "quantiles": {str(k): {c: _quantiles(meta.loc[meta["kind"] == k, c])
                               for c in ("rmsd", "min_dist", "contact_frac", "attempts")} for k in KIND_TAG},
        "relief_steps_quantiles": {str(k): _quantiles(meta.loc[meta["kind"] == k, "relief_steps"])
                                   for k in KIND_TAG},
        "decoys_relieved": int(((meta["kind"] == 2) & (meta["relief_steps"] > 0)).sum()),
        "n_within_3A_quantiles": {str(k): _quantiles(meta.loc[meta["kind"] == k, "n_within_3A"])
                                  for k in KIND_TAG},
        "crystal_min_dist_quantiles": _quantiles(crystal_md),
        "crystal_n_within_3A_quantiles": _quantiles([r["crystal_n_within_3A"] for r in ok
                                                     if r["crystal_n_within_3A"] is not None]),
        "cue_auroc_min_dist": {
            "crystal_vs_decoy": _auroc(crystal_md, meta.loc[meta["kind"] == 2, "min_dist"]),
            "nn_vs_decoy": _auroc(meta.loc[meta["kind"] == 1, "min_dist"],
                                  meta.loc[meta["kind"] == 2, "min_dist"]),
            "crystal_vs_nn": _auroc(crystal_md, meta.loc[meta["kind"] == 1, "min_dist"]),
        },
        "complex_seconds": _quantiles([r["seconds"] for r in results]),
        "identity_check": identity,
        "seconds": round(seconds, 1),
        "workers": args.workers,
        "limit": args.limit,
        # long lists last, so `head` of the file shows the counts
        "failure_complexes": short,
        "errors": errors,
        "relaxed_clash_complexes": relaxed,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pdbbind-root", required=True, help="dir holding refined-set/ and general-set/")
    p.add_argument("--meta", required=True, help="arm 1 paired_meta.csv")
    p.add_argument("--pdbbind-csv", required=True, help="parent pdbbind_processed.csv (column refined)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k-pos", type=int, default=2)
    p.add_argument("--k-neg", type=int, default=2)
    p.add_argument("--workers", type=int, default=72)
    p.add_argument("--limit", type=int, default=0, help="first N complexes by pdb id (0 = all)")
    p.add_argument("--check-identity", type=int, default=0, metavar="N",
                   help="only check N crystal complexes against --parent-pickle")
    p.add_argument("--parent-pickle", default=None)
    p.add_argument("--atom-keys", default=os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if not aevf.AEV_AVAILABLE:
        print(f"WARNING: AEV code not importable ({aevf.AEV_IMPORT_ERROR}); "
              "GetMolAEVs_extended fails unless stubbed", flush=True)
    complexes, missing = read_complexes(args.meta, args.pdbbind_csv, args.pdbbind_root)
    print(f"pdbbind complexes {len(complexes)} (+{len(missing)} absent from {args.pdbbind_csv})", flush=True)
    if args.check_identity:
        check_identity(args, complexes)
        return
    if args.limit:
        complexes = complexes[:args.limit]
        missing = []
    t0 = time.time()
    tasks = [c + (args.k_pos, args.k_neg) for c in complexes]
    results = sorted(_run(pose_complex, tasks, args.workers, args.atom_keys, "poses"),
                     key=lambda r: r["pdb"])
    graphs, rows = {}, []
    for r in results:
        graphs.update(r["graphs"])
        rows.extend(r["rows"])
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    pk = os.path.join(args.out_dir, POSES_PICKLE)
    with open(pk + ".tmp", "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pk + ".tmp", pk)
    meta_path = os.path.join(args.out_dir, POSES_META)
    meta.to_csv(meta_path + ".tmp", index=False)
    os.replace(meta_path + ".tmp", meta_path)
    summary = summarise(results, meta, missing, args, time.time() - t0)
    _write_json(summary, os.path.join(args.out_dir, POSES_SUMMARY))
    print(f"poses {len(graphs)} from {summary['n_complexes_ok']}/{summary['n_complexes']} complexes; "
          f"errors {summary['n_errors']}; short pos {summary['failures']['pos']['n']} "
          f"neg {summary['failures']['neg']['n']}; {summary['seconds']}s", flush=True)
    if results and not summary["n_complexes_ok"]:
        print("ERROR: every complex failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
