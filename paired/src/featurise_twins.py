"""Scheme A shape twins (docs/plans/2026-11-05-aevplig-soft-negatives-AC.md, Task A): select twins from the
3D Zernike descriptors (twin_select), overlay them and the regenerated cognate ligand into the anchor's
pocket (twin_place), and featurise every accepted pose with the clone's functions.

Inputs (read-only, untrusted):
  --zernike         zernike.npz (shell mode; twin_select.DESC_KEYS are read)
  --meta            paired_meta.csv (unique_id, source, split, lig_key, uniprot, anchor_pdb)
  --groups          pocket_groups.csv (unique_id, group_id)
  --parent-data     the parent's data dir with {pdbbind,bindingnet,bindingdb}_processed.csv
  --pdbbind-root    DIR/{refined-set|general-set}/<pdb>/<pdb>_{ligand.mol2,protein.pdb} (column `refined` of
                    pdbbind_processed.csv picks the folder, as the clone)
  --bindingnet-root DIR/<pdb>/target_<target>/<compnd>/<pdb>_<target>_<compnd>.sdf (bindingnet_processed.csv:
                    unique_identify -> pdb, target, compnd)
  --bindingdb-root  DIR/<folder>/<mol2_file> (bindingdb_processed.csv: unique_id -> folder, mol2_file)
Paths are built from validated components only (featurise_zernike.safe_join); ligand files are read with
featurise_zernike.read_ligand (size cap; mol2: MolFromMol2File + AddHs(addCoords=True); BindingNet sdf: first
entry of SDMolSupplier(removeHs=False)) followed by Chem.AddHs(addCoords=True), which adds only missing Hs.

Steps:
1. Anchors: meta rows with source pdbbind and split train or valid, sorted by pdb id (--limit N: first N).
2. Selection, in the parent process before any worker starts: twin_select.load_pool on the npz + meta +
   groups, then select_twins_batch(k=--n-candidates) gives each anchor a ranked candidate list (lowest
   complementarity first, distinct lig_keys; filters and shape_frac as the options below).
3. Per anchor (one Pool task; worker as featurise_poses: fork pool, 1 torch thread, atom keys loaded once):
   the anchor ligand read as the clone reads it, the protein's typed heavy atoms (LoadPDBasDF X, Y, Z: the
   atoms the AEV sees) as the clash/contact target, rng = default_rng(zlib.crc32(pdb)). Candidates are
   placed in rank order with twin_place.place_twin until --k (2) are accepted; an unresolvable, unreadable
   or rejected candidate is recorded and the next one is tried. When at least one twin is accepted, the
   anchor's own ligand goes through the identical path (twin_place.regenerate_positive) as the kind-3
   positive. An anchor with no accepted twin writes nothing and records why: writing its positive alone
   would add exactly the ligands no twin fits (very small, very large, highly charged) to kind 3 only,
   a size/charge cue between kinds 3 and 4. Twins are kept when the positive is rejected.
   --clash (2.2 A) and --relief-steps (300) are passed to twin_place for both kinds: --clash 3.0 accepts
   only poses at least as clash-free as the crystal pose (capped at 3.0 A), which removes most of the
   close-contact difference between overlaid positives and twins (local FEP check).
4. Each accepted pose: featurise_poses.featurise_mol (GetMolAEVs_extended + mol_to_graph, the verbatim
   copies in aevplig_featurise) against the anchor's protein file.

Outputs in --out-dir:
  twins.pickle        {"<pdb>::tpos0" (kind 3) | "<pdb>::twin<k>" (kind 4):
                       (c_size, features, edge_index, edge_features)}
  twins_meta.csv      META_COLUMNS: the plan's ten fixed columns, its ten scheme-A columns, then
                      logp (Crippen MolLogP of the posed ligand), cand_rank (rank in the candidate list;
                      -1 for kind 3), attempts, relief_steps, twin_lig_key, twin_group_id.
                      Kind 3 rows: twin_unique_id = the anchor, shape_dist 0, complementarity = the anchor's
                      own ligand vs its pocket, d* = 0. Kind 4 rows: rmsd_to_crystal NaN; dlogp, dcharge,
                      darom from the selection arrays (twin - anchor); n_heavy_* from the molecules.
  twins_summary.json  counts per split and kind, anchors by number of twins, failure reasons (selection,
                      per anchor, per candidate), quantiles per kind (shape_dist, complementarity,
                      o3a_score, min_dist, contact_frac, n_within_3A, attempts, relief_steps),
                      rmsd_to_crystal (kind 3), twin-vs-anchor deltas and marginals, twin sources,
                      crystal min_dist / n_within_3A, cue AUROCs kind 3 vs kind 4 (P(positive > twin), ties
                      1/2) on min_dist, n_within_3A, heavy-atom count and logP, timings, errors.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import gc
import pickle
import time
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

import aevplig_featurise as aevf
import featurise_poses as fp
import twin_place
import twin_select
from featurise_zernike import read_ligand, safe_join
from pose_augment import CLOSE_CUTOFF, heavy_xyz, min_dist, n_within

META_COLUMNS = ["unique_id", "pdb", "kind", "k", "method", "split", "rmsd_to_crystal", "min_dist",
                "contact_frac", "n_within_3A",
                "twin_unique_id", "twin_source", "shape_dist", "complementarity", "o3a_score",
                "n_heavy_anchor", "n_heavy_twin", "dlogp", "dcharge", "darom",
                "logp", "cand_rank", "attempts", "relief_steps", "twin_lig_key", "twin_group_id"]
KIND_POS, KIND_TWIN = 3, 4
METHOD = "o3a"
ANCHOR_SPLITS = ("train", "valid")
QUANTILES = fp.QUANTILES
TWINS_PICKLE = "twins.pickle"
TWINS_META = "twins_meta.csv"
TWINS_SUMMARY = "twins_summary.json"
CANDIDATE_OUTCOMES = ("accepted", "unresolved", "unreadable", "embed_failed", "overlay_failed", "clash",
                      "contacts", "error", "not_tried")


# ---------------------------------------------------------------- structures

def read_tables(parent_data):
    """Lookup tables of the parent's processed CSVs (last duplicate wins, as the clone's dict fill)."""
    def table(src, key, cols):
        df = pd.read_csv(os.path.join(parent_data, f"{src}_processed.csv"), index_col=0, dtype=str,
                         keep_default_na=False)
        return df.drop_duplicates(key, keep="last").set_index(key)[cols]
    return {"pdbbind": table("pdbbind", "PDB_code", ["refined"]),
            "bindingnet": table("bindingnet", "unique_identify", ["pdb", "target", "compnd"]),
            "bindingdb": table("bindingdb", "unique_id", ["folder", "mol2_file"])}


def pdbbind_paths(pdb, tables, root):
    """(ligand mol2, protein pdb) of a PDBbind complex; KeyError when absent from the csv."""
    if pdb not in tables["pdbbind"].index:
        raise KeyError(f"{pdb} not in pdbbind_processed.csv")
    refined = str(tables["pdbbind"].loc[pdb, "refined"]).strip().lower() in ("true", "1", "yes")
    folder = safe_join(root, "refined-set" if refined else "general-set", pdb)
    return safe_join(folder, f"{pdb}_ligand.mol2"), safe_join(folder, f"{pdb}_protein.pdb")


def ligand_spec(uid, source, tables, roots):
    """featurise_zernike.read_ligand spec of a pool row's ligand file (KeyError / ValueError when unresolvable)."""
    if source == "pdbbind":
        return ("mol2", pdbbind_paths(uid, tables, roots["pdbbind"])[0])
    if source == "bindingnet":
        if uid not in tables["bindingnet"].index:
            raise KeyError(f"{uid} not in bindingnet_processed.csv")
        rec = tables["bindingnet"].loc[uid]
        pdb, target, compnd = rec["pdb"], rec["target"], rec["compnd"]
        return ("sdf_first", safe_join(roots["bindingnet"], pdb, f"target_{target}", compnd,
                                       f"{pdb}_{target}_{compnd}.sdf"))
    if source == "bindingdb":
        if uid not in tables["bindingdb"].index:
            raise KeyError(f"{uid} not in bindingdb_processed.csv")
        rec = tables["bindingdb"].loc[uid]
        return ("mol2", safe_join(roots["bindingdb"], rec["folder"], rec["mol2_file"]))
    raise ValueError(f"unknown source {source!r}")


def load_structure(spec):
    """The ligand as the clone reads it, with every H explicit and placed (AddHs(addCoords=True))."""
    return Chem.AddHs(read_ligand(spec), addCoords=True)


def n_heavy(mol):
    return sum(a.GetSymbol() != "H" for a in mol.GetAtoms())


# ---------------------------------------------------------------- worker

def _row(pdb, split, kind, k, placement, mol, sel, cand_rank):
    return {"unique_id": f"{pdb}::tpos0" if kind == KIND_POS else f"{pdb}::twin{k}", "pdb": pdb, "kind": kind,
            "k": k, "method": METHOD, "split": split, "rmsd_to_crystal": placement.rmsd_to_crystal,
            "min_dist": placement.min_dist, "contact_frac": placement.contact_frac,
            "n_within_3A": placement.n_within_3A, "twin_unique_id": sel["twin_uid"],
            "twin_source": sel["twin_source"], "shape_dist": sel["shape_dist"],
            "complementarity": sel["complementarity"], "o3a_score": placement.o3a_score,
            "n_heavy_anchor": sel["n_heavy_anchor_mol"], "n_heavy_twin": n_heavy(mol),
            "dlogp": sel["dlogp"], "dcharge": sel["dcharge"], "darom": sel["darom"],
            "logp": float(Crippen.MolLogP(mol)), "cand_rank": cand_rank, "attempts": placement.attempts,
            "relief_steps": placement.relief_steps, "twin_lig_key": sel["twin_lig_key"],
            "twin_group_id": sel["twin_group_id"]}


def twin_anchor(task):
    """Place and featurise the twins and the positive of one anchor (see module docstring); never raises."""
    pdb, split = task["pdb"], task["split"]
    t0 = time.time()
    res = {"pdb": pdb, "split": split, "graphs": {}, "rows": [], "error": "", "status": "",
           "candidates": [], "crystal_min_dist": None, "crystal_n_within_3A": None, "n_twins": 0}
    try:
        if not task["candidates"]:
            res["status"] = f"selection:{task['selection_reason']}"
            return res
        anchor = load_structure(task["ligand"])
        if not os.path.exists(task["protein"]):
            raise FileNotFoundError(task["protein"])
        target = aevf.LoadPDBasDF(task["protein"], fp._ATOM_KEYS)
        target_xyz = target[["X", "Y", "Z"]].to_numpy(dtype=float).reshape(-1, 3)
        cryst = heavy_xyz(anchor)
        cmd = min_dist(cryst, target_xyz)
        res["crystal_min_dist"] = cmd if np.isfinite(cmd) else None
        res["crystal_n_within_3A"] = n_within(cryst, target_xyz, CLOSE_CUTOFF)
        n_heavy_anchor = n_heavy(anchor)
        rng = np.random.default_rng(zlib.crc32(pdb.encode()))
        kw = task.get("place_kw", {})
        graphs, rows, accepted = {}, [], 0
        for cand in task["candidates"]:
            if accepted == task["k"]:
                res["candidates"].append((cand["twin_uid"], "not_tried"))
                continue
            if cand.get("spec") is None:
                res["candidates"].append((cand["twin_uid"], "unresolved"))
                continue
            try:
                twin = load_structure(cand["spec"])
            except Exception:  # noqa: BLE001 - an unreadable twin is skipped and recorded
                res["candidates"].append((cand["twin_uid"], "unreadable"))
                continue
            try:
                p = twin_place.place_twin(anchor, twin, target_xyz, rng, **kw)
            except Exception:  # noqa: BLE001 - recorded; the next candidate is tried
                res["candidates"].append((cand["twin_uid"], "error"))
                continue
            if not p.ok:
                res["candidates"].append((cand["twin_uid"], p.reason))
                continue
            uid = f"{pdb}::twin{accepted}"
            graphs[uid] = fp.featurise_mol(task["protein"], p.mol)
            rows.append(_row(pdb, split, KIND_TWIN, accepted, p, p.mol,
                             dict(cand, n_heavy_anchor_mol=n_heavy_anchor), cand["rank"]))
            res["candidates"].append((cand["twin_uid"], "accepted"))
            accepted += 1
        res["n_twins"] = accepted
        if accepted == 0:
            res["status"] = "no_twin_placed"
        else:
            p = twin_place.regenerate_positive(anchor, rng, target_xyz, **kw)
            if p.ok:
                own = {"twin_uid": pdb, "twin_source": "pdbbind", "shape_dist": 0.0,
                       "complementarity": task["anchor_complementarity"], "n_heavy_anchor_mol": n_heavy_anchor,
                       "dlogp": 0.0, "dcharge": 0, "darom": 0.0, "twin_lig_key": task["lig_key"],
                       "twin_group_id": task["group_id"]}
                graphs[f"{pdb}::tpos0"] = fp.featurise_mol(task["protein"], p.mol)
                rows.insert(0, _row(pdb, split, KIND_POS, 0, p, p.mol, own, -1))
                res["status"] = "ok" if accepted == task["k"] else "short"
            else:
                res["status"] = f"positive_rejected:{p.reason}"
        res["graphs"], res["rows"] = graphs, rows
    except Exception as e:  # noqa: BLE001 - one bad anchor must not stop the run; it is recorded
        res["error"] = f"{type(e).__name__}: {e}"
        res["status"] = "error"
        res["graphs"], res["rows"], res["n_twins"] = {}, [], 0
    finally:
        res["seconds"] = time.time() - t0
    return res


# ---------------------------------------------------------------- tasks

def read_anchors(meta):
    """[(pdb, split)] of the PDBbind train/valid rows, sorted by pdb id."""
    m = meta[(meta["source"] == "pdbbind") & meta["split"].isin(ANCHOR_SPLITS)]
    if "kind" in m.columns:
        m = m[m["kind"].astype(str).str.strip() == "0"]
    return sorted(zip(m["unique_id"], m["split"]))


def select(args, anchor_uids):
    """{anchor: (list[TwinChoice], info)} from the descriptor pool (built and freed here)."""
    pool = twin_select.load_pool(args.zernike, args.meta, args.groups)
    stats = {k: v for k, v in pool.stats.items() if not isinstance(v, np.ndarray)}
    out = twin_select.select_twins_batch(anchor_uids, pool, k=args.n_candidates, shape_frac=args.shape_frac,
                                         max_dlogp=args.max_dlogp, max_darom=args.max_darom,
                                         same_charge_sign=not args.any_charge_sign, size_tol=args.size_tol,
                                         block=args.select_block)
    del pool
    gc.collect()
    return out, stats


def build_tasks(anchors, selection, meta, groups, tables, roots, k, place_kw=None):
    """One task per anchor (dicts, picklable). Anchors absent from pdbbind_processed.csv are returned
    separately as (pdb, error)."""
    lig_key = dict(zip(meta["unique_id"], meta["lig_key"]))
    group_id = dict(zip(groups["unique_id"], groups["group_id"])) if groups is not None else {}
    tasks, missing = [], []
    for pdb, split in anchors:
        try:
            lig_path, protein = pdbbind_paths(pdb, tables, roots["pdbbind"])
        except (KeyError, ValueError) as e:
            missing.append((pdb, split, f"{type(e).__name__}: {e}"))
            continue
        choices, info = selection.get(pdb, ([], {"reason": "anchor_no_descriptor"}))
        cands = []
        for c in choices:
            d = c.as_dict()
            try:
                d["spec"] = ligand_spec(c.twin_uid, c.twin_source, tables, roots)
                d["spec_error"] = ""
            except (KeyError, ValueError) as e:
                d["spec"], d["spec_error"] = None, f"{type(e).__name__}: {e}"
            cands.append(d)
        tasks.append({"pdb": pdb, "split": split, "ligand": ("mol2", lig_path), "protein": protein,
                      "candidates": cands, "k": k, "selection_reason": info.get("reason", ""),
                      "anchor_complementarity": info.get("anchor_complementarity", float("nan")),
                      "lig_key": lig_key.get(pdb, ""), "group_id": group_id.get(pdb, ""),
                      "place_kw": dict(place_kw or {})})
    return tasks, missing


# ---------------------------------------------------------------- summary

def _quantiles(values):
    return fp._quantiles(values)


def _counts(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def summarise(results, meta, selection, missing, args, seconds, selection_seconds=0.0, pool_stats=None):
    """The twins_summary.json dict (long lists last)."""
    ok = [r for r in results if not r["error"]]
    errors = [[r["pdb"], r["error"]] for r in results if r["error"]] + [[p, e] for p, _, e in missing]
    splits = sorted({r["split"] for r in results} | {s for _, s, _ in missing})
    kinds = (KIND_POS, KIND_TWIN)
    pos, tw = meta[meta["kind"] == KIND_POS], meta[meta["kind"] == KIND_TWIN]
    sel_infos = [selection[r["pdb"]][1] for r in results if r["pdb"] in selection]
    cand_outcomes = Counter(o for r in results for _, o in r["candidates"])
    per_kind = {str(k): {c: _quantiles(meta.loc[meta["kind"] == k, c])
                         for c in ("shape_dist", "complementarity", "o3a_score", "min_dist", "contact_frac",
                                   "n_within_3A", "attempts", "relief_steps")} for k in kinds}
    own = {r["pdb"]: r for r in pos.to_dict("records")}
    comp_gain = [row["complementarity"] - own[row["pdb"]]["complementarity"]
                 for row in tw.to_dict("records") if row["pdb"] in own]
    return {
        "n_anchors": len(results) + len(missing),
        "n_anchors_ok": len(ok),
        "n_poses": int(len(meta)),
        "k": args.k,
        "n_candidates": args.n_candidates,
        "selection_params": {"shape_frac": args.shape_frac, "max_dlogp": args.max_dlogp,
                             "max_darom": args.max_darom, "same_charge_sign": not args.any_charge_sign,
                             "size_tol": args.size_tol},
        "place_params": {**place_kw(args), "n_confs": twin_place.N_CONFS, "max_overlays": twin_place.MAX_OVERLAYS,
                         "contact_keep": twin_place.CONTACT_KEEP, "relief_target_A": twin_place.RELIEF_TARGET_A},
        "pool": pool_stats or {},
        "anchors_by_split": _counts([r["split"] for r in results] + [s for _, s, _ in missing]),
        "counts_split_kind": {s: {str(k): int(((meta["split"] == s) & (meta["kind"] == k)).sum())
                                  for k in kinds} for s in splits},
        "counts_kind": {str(k): int((meta["kind"] == k).sum()) for k in kinds},
        "anchors_by_n_twins": {str(n): sum(r["n_twins"] == n for r in results) + (len(missing) if n == 0 else 0)
                               for n in range(args.k, -1, -1)},
        "anchors_by_n_twins_split": {s: {str(n): sum(r["n_twins"] == n and r["split"] == s for r in results)
                                         for n in range(args.k, -1, -1)} for s in splits},
        "anchor_status": _counts([r["status"] for r in results] + ["missing"] * len(missing)),
        "selection_reasons": _counts([i.get("reason", "") for i in sel_infos]),
        "selection_no_candidate_empty_at": _counts([i.get("empty_at", "") for i in sel_infos
                                                    if i.get("reason") == "no_twin"]),
        "anchors_by_n_candidates": _counts([len(selection[r["pdb"]][0]) for r in results
                                            if r["pdb"] in selection]),
        "candidate_outcomes": {o: int(cand_outcomes.get(o, 0)) for o in CANDIDATE_OUTCOMES},
        "quantiles": per_kind,
        "rmsd_to_crystal_quantiles": _quantiles(pos["rmsd_to_crystal"]),
        "twins_vs_anchors": {
            "dlogp": _quantiles(tw["dlogp"]), "abs_dlogp": _quantiles(tw["dlogp"].abs()),
            "darom": _quantiles(tw["darom"]), "abs_darom": _quantiles(tw["darom"].abs()),
            "dcharge": _counts(tw["dcharge"].astype(int)),
            "complementarity_twin_minus_own": _quantiles(comp_gain),
            "logp": {str(k): _quantiles(meta.loc[meta["kind"] == k, "logp"]) for k in kinds},
            "n_heavy": {str(k): _quantiles(meta.loc[meta["kind"] == k, "n_heavy_twin"]) for k in kinds},
        },
        "twin_source_counts": _counts(tw["twin_source"]),
        "crystal_min_dist_quantiles": _quantiles([r["crystal_min_dist"] for r in ok
                                                  if r["crystal_min_dist"] is not None]),
        "crystal_n_within_3A_quantiles": _quantiles([r["crystal_n_within_3A"] for r in ok
                                                     if r["crystal_n_within_3A"] is not None]),
        "cue_auroc_pos_vs_twin": {
            "min_dist": fp._auroc(pos["min_dist"], tw["min_dist"]),
            "n_within_3A": fp._auroc(pos["n_within_3A"], tw["n_within_3A"]),
            "n_heavy": fp._auroc(pos["n_heavy_twin"], tw["n_heavy_twin"]),
            "logp": fp._auroc(pos["logp"], tw["logp"]),
        },
        "anchor_seconds": _quantiles([r["seconds"] for r in results]),
        "selection_seconds": round(selection_seconds, 2),
        "seconds": round(seconds, 1),
        "workers": args.workers,
        "limit": args.limit,
        # long lists last, so `head` of the file shows the counts
        "failure_anchors": [[r["pdb"], r["split"], r["status"], r["n_twins"]] for r in results
                            if r["status"] not in ("ok",)],
        "errors": errors,
    }


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zernike", required=True, help="zernike.npz (shell mode)")
    p.add_argument("--meta", required=True, help="paired_meta.csv")
    p.add_argument("--groups", required=True, help="pocket_groups.csv")
    p.add_argument("--pdbbind-root", required=True, help="dir holding refined-set/ and general-set/")
    p.add_argument("--bindingnet-root", required=True, help="BindingNetv1/ (ligand sdf files)")
    p.add_argument("--bindingdb-root", required=True, help="surflex/ (BindingDB folders)")
    p.add_argument("--parent-data", required=True, help="dir with {pdbbind,bindingnet,bindingdb}_processed.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=72)
    p.add_argument("--limit", type=int, default=0, help="first N anchors by pdb id (0 = all)")
    p.add_argument("--k", type=int, default=2, help="twins per anchor")
    p.add_argument("--n-candidates", type=int, default=6, help="ranked candidates per anchor (k + spares)")
    p.add_argument("--shape-frac", type=float, default=0.01)
    p.add_argument("--max-dlogp", type=float, default=1.5)
    p.add_argument("--max-darom", type=float, default=1.0)
    p.add_argument("--size-tol", type=float, default=0.2)
    p.add_argument("--any-charge-sign", action="store_true", help="drop the same-charge-sign filter")
    p.add_argument("--clash", type=float, default=twin_place.CLASH,
                   help="acceptance min_dist >= min(clash, crystal min_dist); the relief target drops with it")
    p.add_argument("--relief-steps", type=int, default=twin_place.RELIEF_STEPS_MAX, help="relief budget per overlay")
    p.add_argument("--select-block", type=int, default=twin_select.DEFAULT_BLOCK)
    p.add_argument("--atom-keys", default=os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv"))
    args = p.parse_args(argv)
    if args.k < 1 or args.n_candidates < args.k:
        p.error("need 1 <= --k <= --n-candidates")
    return args


def place_kw(args):
    """twin_place.place keyword arguments from the options (the same for kinds 3 and 4)."""
    return {"clash": args.clash, "relief_steps_max": args.relief_steps}


def _atomic_write(path, write):
    write(path + ".tmp")
    os.replace(path + ".tmp", path)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if not aevf.AEV_AVAILABLE:
        print(f"WARNING: AEV code not importable ({aevf.AEV_IMPORT_ERROR}); "
              "GetMolAEVs_extended fails unless stubbed", flush=True)
    t0 = time.time()
    meta = pd.read_csv(args.meta, dtype=str, keep_default_na=False)
    groups = pd.read_csv(args.groups, dtype=str, keep_default_na=False)
    anchors = read_anchors(meta)
    if args.limit:
        anchors = anchors[:args.limit]
    print(f"anchors {len(anchors)} (pdbbind {'/'.join(ANCHOR_SPLITS)})", flush=True)
    t_sel = time.time()
    selection, pool_stats = select(args, [p for p, _ in anchors])
    t_sel = time.time() - t_sel
    print(f"selection {t_sel:.1f}s: {_counts(i['reason'] for _, i in selection.values())}", flush=True)
    tables = read_tables(args.parent_data)
    roots = {"pdbbind": args.pdbbind_root, "bindingnet": args.bindingnet_root, "bindingdb": args.bindingdb_root}
    tasks, missing = build_tasks(anchors, selection, meta, groups, tables, roots, args.k, place_kw(args))
    del tables
    gc.collect()
    results = sorted(fp._run(twin_anchor, tasks, args.workers, args.atom_keys, "twins"), key=lambda r: r["pdb"])
    graphs, rows = {}, []
    for r in results:
        graphs.update(r["graphs"])
        rows.extend(r["rows"])
    out_meta = pd.DataFrame(rows, columns=META_COLUMNS)

    def dump_pickle(path):
        with open(path, "wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write(os.path.join(args.out_dir, TWINS_PICKLE), dump_pickle)
    _atomic_write(os.path.join(args.out_dir, TWINS_META), lambda p: out_meta.to_csv(p, index=False))
    summary = summarise(results, out_meta, selection, missing, args, time.time() - t0, t_sel, pool_stats)
    fp._write_json(summary, os.path.join(args.out_dir, TWINS_SUMMARY))
    print(f"twins: {len(graphs)} poses ({summary['counts_kind']}) from {summary['n_anchors_ok']}/"
          f"{summary['n_anchors']} anchors; by n twins {summary['anchors_by_n_twins']}; errors "
          f"{len(summary['errors'])}; cue AUROCs {summary['cue_auroc_pos_vs_twin']}; {summary['seconds']}s",
          flush=True)
    if results and not summary["n_anchors_ok"]:
        print("ERROR: every anchor failed", flush=True)
        sys.exit(1)
    return summary


if __name__ == "__main__":
    main()
