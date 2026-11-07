"""Generate in-place ligand edits (scheme C) for every PDBbind train/valid anchor of paired_meta.csv and
featurise them with the verbatim copies of the clone's functions (aevplig_featurise, via the arm-4
featurise_poses worker path).

Inputs: --pdbbind-root (refined-set/ and general-set/ with <pdb>/<pdb>_{ligand.mol2,protein.pdb}),
--meta (paired_meta.csv; rows with source == pdbbind and split in {train, valid} are the anchors),
--parent-data (the parent's data dir; its pdbbind_processed.csv column `refined` picks the folder, as the
clone; --pdbbind-csv overrides the file).
Per anchor (one Pool task): the ligand is read as the clone does (featurise_poses.load_ligand), the AEV's
typed protein heavy atoms (LoadPDBasDF X, Y, Z) are the relief/acceptance/min_dist target as in arm 4,
pocket_fields.protein_fields gives the channel-typed atoms for site classification, the non-water HETATM
heavy atoms (ligand_edits.load_hetatm_xyz) count for the exposure test, rng = default_rng(crc32(pdb)),
poses from ligand_edits.generate_edits, and each pose is featurised with GetMolAEVs_extended +
mol_to_graph (featurise_poses.featurise_mol).
Outputs in --out-dir:
  edits.pickle        {"<pdb>::epos0" (kind 5) | "<pdb>::cedit<k>" (kind 6) | "<pdb>::xedit<k>" (kind 7):
                       (c_size, features, edge_index, edge_features)}
  edits_meta.csv      META_COLUMNS: the ten fixed columns of the AC plan, the six scheme-C columns, then
                      relief_steps, n_heavy, formal_charge, dcharge (edited - anchor), place_method
                      (embedding mode + force field, ligand_edits docstring). method = edit rule
                      ("null" for kind 5); rmsd_to_crystal = RMSD of the unchanged heavy atoms vs the
                      crystal pose after MMFF + relief (no alignment).
  edits_summary.json  counts per kind x split, per rule x site class x split for the sites found
                      (candidate_counts) and for the poses written (pose_counts), failures (null-edit
                      reasons, edit placement reasons per rule, contact/exposed edits per anchor), quantiles
                      per kind, mmff_restrained_rmsd quantiles, cue AUROCs kind 5 vs 6 and 5 vs 7
                      (P(kind 5 value > other) + ties / 2) on min_dist, n_within_3A, heavy-atom count
                      (and rmsd_to_crystal), exposed_radius (--exposed-radius, default 3.5 A) and
                      exposed_by_radius (exposed-site counts and the kind-7 count before placement
                      failures had the exposure radius been 3.5, 4.0 or 4.5 A), errors.
No identity check here: the featuriser functions are the ones featurise_poses.py verifies.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import pickle
import time
import zlib
from collections import Counter

import numpy as np
import pandas as pd

import aevplig_featurise as aevf
import featurise_poses as fp
from ligand_edits import EXPOSED_RADIUS, RULE_NAMES, generate_edits, load_hetatm_xyz
from pocket_fields import protein_fields
from pose_augment import heavy_xyz

META_COLUMNS = ["unique_id", "pdb", "kind", "k", "method", "split", "rmsd_to_crystal", "min_dist",
                "contact_frac", "n_within_3A",
                "edit_rule", "site_class", "n_edited_atoms", "n_contacts_edited_group", "partner_type",
                "mmff_restrained_rmsd",
                "relief_steps", "n_heavy", "formal_charge", "dcharge", "place_method"]
KIND_TAG = {5: "epos", 6: "cedit", 7: "xedit"}
KINDS = tuple(KIND_TAG)
ANCHOR_SPLITS = ("train", "valid")
SITE_CLASSES = ("contact", "exposed", "ambiguous")
CUE_COLUMNS = ("min_dist", "n_within_3A", "n_heavy", "rmsd_to_crystal")
QUANTILE_COLUMNS = ("rmsd_to_crystal", "min_dist", "contact_frac", "n_within_3A", "relief_steps",
                    "mmff_restrained_rmsd")
EDITS_PICKLE = "edits.pickle"
EDITS_META = "edits_meta.csv"
EDITS_SUMMARY = "edits_summary.json"


def _plain(log):
    """JSON-able copy of a generate_edits log (Counters -> dicts)."""
    return {"n_sites": int(log.get("n_sites", 0)),
            "null_failure": log.get("null_failure", ""),
            "candidates": {r: dict(c) for r, c in log.get("candidates", {}).items()},
            "failures": {r: dict(c) for r, c in log.get("failures", {}).items()},
            "exposed_sites_by_radius": {r: int(n) for r, n in log.get("exposed_sites_by_radius", {}).items()}}


def edit_complex(task):
    pdb, split, mol_path, protein_path, k_contact, k_exposed, exposed_radius = task
    t0 = time.time()
    res = {"pdb": pdb, "split": split, "graphs": {}, "rows": [], "error": "", "log": _plain({})}
    try:
        mol = fp.load_ligand(mol_path)
        if mol is None:
            raise ValueError("can't read molecule structure")
        if not os.path.exists(protein_path):
            raise FileNotFoundError(protein_path)
        target = aevf.LoadPDBasDF(protein_path, fp._ATOM_KEYS)[["X", "Y", "Z"]].to_numpy(dtype=float)
        fields = protein_fields(protein_path)
        extra = load_hetatm_xyz(protein_path, exclude_xyz=heavy_xyz(mol))
        rng = np.random.default_rng(zlib.crc32(pdb.encode()))
        log = {}
        poses = generate_edits(mol, fields, rng, k_contact, k_exposed, target_xyz=target, extra_xyz=extra,
                               exposed_radius=exposed_radius, log=log)
        graphs, rows = {}, []
        for p in poses:
            uid = f"{pdb}::{KIND_TAG[p.kind]}{p.k}"
            graphs[uid] = fp.featurise_mol(protein_path, p.mol)
            rows.append({"unique_id": uid, "pdb": pdb, "kind": p.kind, "k": p.k, "method": p.rule,
                         "split": split, "rmsd_to_crystal": p.rmsd_to_crystal, "min_dist": p.min_dist,
                         "contact_frac": p.contact_frac, "n_within_3A": p.n_within_3A,
                         "edit_rule": p.rule, "site_class": p.site_class,
                         "n_edited_atoms": p.n_edited_atoms, "n_contacts_edited_group": p.n_contacts,
                         "partner_type": p.partner_type, "mmff_restrained_rmsd": p.mmff_restrained_rmsd,
                         "relief_steps": p.relief_steps, "n_heavy": p.n_heavy,
                         "formal_charge": p.formal_charge, "dcharge": p.dcharge,
                         "place_method": p.place_method})
        res["graphs"], res["rows"], res["log"] = graphs, rows, _plain(log)
    except Exception as e:  # noqa: BLE001 - one bad complex must not stop the run; it is recorded
        res["error"] = f"{type(e).__name__}: {e}"
    res["seconds"] = time.time() - t0
    return res


def read_anchors(meta_csv, pdbbind_csv, root):
    """featurise_poses.read_complexes restricted to split in ANCHOR_SPLITS (both lists)."""
    complexes, missing = fp.read_complexes(meta_csv, pdbbind_csv, root)
    meta = pd.read_csv(meta_csv, usecols=["unique_id", "source", "split"], dtype=str)
    split = dict(zip(meta.loc[meta["source"] == "pdbbind", "unique_id"],
                     meta.loc[meta["source"] == "pdbbind", "split"]))
    return ([c for c in complexes if c[1] in ANCHOR_SPLITS],
            [p for p in missing if split.get(p) in ANCHOR_SPLITS])


def _nested_counts(items, splits):
    """{rule: {class: {split: n}}} from (rule, class, split, n) items; zero entries dropped."""
    out = {}
    for rule, cls, split, n in items:
        if n:
            d = out.setdefault(rule, {}).setdefault(cls, {})
            d[split] = d.get(split, 0) + int(n)
    order = {r: i for i, r in enumerate(RULE_NAMES + ("null",))}
    return {r: {c: dict(sorted(out[r][c].items(), key=lambda kv: splits.index(kv[0])))
                for c in sorted(out[r])} for r in sorted(out, key=lambda r: order.get(r, len(order)))}


def summarise(results, meta, missing, args, seconds):
    ok = [r for r in results if not r["error"]]
    errors = [[r["pdb"], r["error"]] for r in results if r["error"]]
    errors += [[p, "not in pdbbind csv"] for p in missing]
    splits = sorted({r["split"] for r in results})
    candidates = _nested_counts(((rule, cls, r["split"], n) for r in ok
                                 for rule, cnt in r["log"]["candidates"].items() for cls, n in cnt.items()),
                                splits)
    poses = _nested_counts(((rule, cls, split, 1) for rule, cls, split in
                            zip(meta["edit_rule"], meta["site_class"], meta["split"])), splits)
    null_fail = Counter(r["log"]["null_failure"] for r in ok if r["log"]["null_failure"])
    edit_fail = {}
    for r in ok:
        for rule, cnt in r["log"]["failures"].items():
            for reason, n in cnt.items():
                d = edit_fail.setdefault(rule, {})
                d[reason] = d.get(reason, 0) + int(n)

    exposed_by_radius = {}
    for r in ok:
        for radius, n in r["log"]["exposed_sites_by_radius"].items():
            d = exposed_by_radius.setdefault(radius, {"exposed_sites": 0, "anchors_with_exposed_site": 0,
                                                      "kind7_if_all_placed": 0})
            d["exposed_sites"] += n
            d["anchors_with_exposed_site"] += int(n > 0)
            d["kind7_if_all_placed"] += min(n, args.k_exposed)
    exposed_by_radius = dict(sorted(exposed_by_radius.items(), key=lambda kv: float(kv[0])))

    def per_anchor(kind):
        n = Counter(int(((meta["pdb"] == r["pdb"]) & (meta["kind"] == kind)).sum()) for r in ok
                    if not r["log"]["null_failure"])
        return {str(k): v for k, v in sorted(n.items())}

    by_kind = {k: meta[meta["kind"] == k] for k in KINDS}
    return {
        "n_anchors": len(results) + len(missing),
        "n_anchors_ok": len(ok),
        "n_poses": int(len(meta)),
        "k_contact": args.k_contact,
        "k_exposed": args.k_exposed,
        "exposed_radius": args.exposed_radius,
        # exposed sites when classified at other radii (exposed_radius decides); kind7_if_all_placed =
        # sum over anchors of min(exposed sites, k_exposed), i.e. before placement failures
        "exposed_by_radius": exposed_by_radius,
        "anchors_by_split": dict(sorted(Counter(r["split"] for r in results).items())),
        "counts_split_kind": {s: {str(k): int(((meta["split"] == s) & (meta["kind"] == k)).sum())
                                  for k in KINDS} for s in splits},
        "counts_kind": {str(k): int(len(by_kind[k])) for k in KINDS},
        "candidate_counts": candidates,
        "pose_counts": poses,
        "n_sites_quantiles": fp._quantiles([r["log"]["n_sites"] for r in ok]),
        "failures": {
            "null": dict(null_fail),
            "anchors_without_null": int(sum(null_fail.values())),
            "edit": edit_fail,
            "contact_edits_per_anchor": per_anchor(6),
            "exposed_edits_per_anchor": per_anchor(7),
        },
        "n_errors": len(errors),
        "quantiles": {str(k): {c: fp._quantiles(by_kind[k][c]) for c in QUANTILE_COLUMNS} for k in KINDS},
        "mmff_restrained_rmsd_quantiles": {str(k): fp._quantiles(by_kind[k]["mmff_restrained_rmsd"])
                                           for k in KINDS},
        "cue_auroc": {f"5_vs_{k}": {c: fp._auroc(by_kind[5][c], by_kind[k][c]) for c in CUE_COLUMNS}
                      for k in (6, 7)},
        "dcharge_counts": {str(k): {str(q): int(n) for q, n in sorted(Counter(by_kind[k]["dcharge"]).items())}
                           for k in KINDS},
        "partner_type_counts": dict(Counter(by_kind[6]["partner_type"])),
        "place_method_counts": dict(Counter(meta["place_method"])),
        "anchor_seconds": fp._quantiles([r["seconds"] for r in results]),
        "seconds": round(seconds, 1),
        "workers": args.workers,
        "limit": args.limit,
        # long lists last, so `head` of the file shows the counts
        "null_failure_anchors": [[r["pdb"], r["split"], r["log"]["null_failure"]] for r in ok
                                 if r["log"]["null_failure"]],
        "errors": errors,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pdbbind-root", required=True, help="dir holding refined-set/ and general-set/")
    p.add_argument("--meta", required=True, help="paired_meta.csv")
    p.add_argument("--parent-data", default=None, help="parent data dir holding pdbbind_processed.csv")
    p.add_argument("--pdbbind-csv", default=None, help="overrides <parent-data>/pdbbind_processed.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k-contact", type=int, default=2)
    p.add_argument("--k-exposed", type=int, default=2)
    p.add_argument("--exposed-radius", type=float, default=EXPOSED_RADIUS,
                   help="exposed site: no protein/HETATM heavy atom within this distance (A)")
    p.add_argument("--workers", type=int, default=72)
    p.add_argument("--limit", type=int, default=0, help="first N anchors by pdb id (0 = all)")
    p.add_argument("--atom-keys", default=os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv"))
    args = p.parse_args(argv)
    if not args.pdbbind_csv:
        if not args.parent_data:
            p.error("--parent-data or --pdbbind-csv is required")
        args.pdbbind_csv = os.path.join(args.parent_data, "pdbbind_processed.csv")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if not aevf.AEV_AVAILABLE:
        print(f"WARNING: AEV code not importable ({aevf.AEV_IMPORT_ERROR}); "
              "GetMolAEVs_extended fails unless stubbed", flush=True)
    complexes, missing = read_anchors(args.meta, args.pdbbind_csv, args.pdbbind_root)
    print(f"pdbbind train/valid anchors {len(complexes)} (+{len(missing)} absent from {args.pdbbind_csv})",
          flush=True)
    if args.limit:
        complexes = complexes[:args.limit]
        missing = []
    t0 = time.time()
    tasks = [c + (args.k_contact, args.k_exposed, args.exposed_radius) for c in complexes]
    results = sorted(fp._run(edit_complex, tasks, args.workers, args.atom_keys, "edits"),
                     key=lambda r: r["pdb"])
    graphs, rows = {}, []
    for r in results:
        graphs.update(r["graphs"])
        rows.extend(r["rows"])
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    pk = os.path.join(args.out_dir, EDITS_PICKLE)
    with open(pk + ".tmp", "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pk + ".tmp", pk)
    meta_path = os.path.join(args.out_dir, EDITS_META)
    meta.to_csv(meta_path + ".tmp", index=False)
    os.replace(meta_path + ".tmp", meta_path)
    summary = summarise(results, meta, missing, args, time.time() - t0)
    fp._write_json(summary, os.path.join(args.out_dir, EDITS_SUMMARY))
    print(f"edits {len(graphs)} from {summary['n_anchors_ok']}/{summary['n_anchors']} anchors; "
          f"errors {summary['n_errors']}; no null edit {summary['failures']['anchors_without_null']}; "
          f"kinds {summary['counts_kind']}; {summary['seconds']}s", flush=True)
    if results and not summary["n_anchors_ok"]:
        print("ERROR: every anchor failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
