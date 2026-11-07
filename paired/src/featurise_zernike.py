"""3D Zernike descriptors (pocket and ligand, 7 property channels) for every complex of paired_meta.csv.

Inputs (all read-only, treated as untrusted data):
  --meta          arm 1 paired_meta.csv (unique_id, source, split; rows with kind != 0, i.e. arm-4 pose
                  rows, are skipped and counted)
  --parent-data   the parent's data dir with {pdbbind,bindingnet,bindingdb}_processed.csv
  --roots         pdbbind=DIR,bindingnet=DIR,bindingdb=DIR laid out as the clone reads them:
                    pdbbind:    DIR/{refined-set|general-set}/<pdb>/<pdb>_{ligand.mol2,protein.pdb}
                                (column `refined` of pdbbind_processed.csv picks the folder)
                    bindingnet: DIR/<pdb>/target_<target>/<compnd>/<pdb>_<target>_<compnd>.sdf and
                                DIR/<pdb>/rec_h_opt.pdb (unique_identify -> pdb, target, compnd)
                    bindingdb:  DIR/<folder>/<mol2_file> and DIR/<folder>/<pdb_file> (unique_id -> ...)
  --fep-dir       alternative input: the Schrodinger benchmark (structure_inputs/<subset>/<series>_{protein.pdb,
                  ligands.sdf}); one row per distinct ligand title (first SDF entry kept), unique_id
                  <subset>/<series>/<title>, source "fep"; also writes <out stem>_meta.csv (unique_id, source,
                  split, series_key, uniprot = the benchmark's Protein name, anchor_pdb) for pocket_clusters.py;
                  --fep-ids CSV (unique_id column, e.g. fep_benchmark.csv) keeps only the listed titles.
  --csv           alternative input (arm Z's evaluation sets): a CSV with columns unique_id, sdf_file, pdb_file
                  (fep_prepare.py's fep_benchmark.csv, pgdh_prepare.py's pgdh.csv); one row per CSV row in CSV order
                  (a duplicate unique_id is an error), source = the CSV's file stem; a relative path is taken under
                  --work-dir (default: the CWD, $WORK in the jobs), an absolute one as is. The single-molecule sdf is
                  read as the prep wrote it (first entry, SDMolSupplier(removeHs=False, sanitize=False)), then
                  sanitised (Chem.SanitizeMol, as the training readers' default sanitize=True does) and given its
                  missing hydrogens with AddHs(addCoords=True) (the mol2 path); the channel code reads heavy atoms
                  only, so the descriptor equals the training readers' for the same molecule.
Ligands are read as the clone reads them (mol2: MolFromMol2File + AddHs(addCoords=True); BindingNet sdf:
first entry of SDMolSupplier(removeHs=False)). Paths are built only from validated components (no absolute
paths, no "..", no NUL; --csv: absolute paths allowed, never ".." or NUL); file sizes and atom counts are capped;
a bad row is recorded, never fatal.

Per row, everything centred on the ligand heavy-atom centroid (pocket_fields.pocket_descriptor):
  --pocket-mode shell (default): pocket = protein heavy atoms within --cutoff (8 A) of any ligand heavy atom,
                scaled by --radius (12 A); pocket atoms beyond --radius are dropped and counted.
  --pocket-mode sphere: pocket = protein heavy atoms within --pocket-radius (10 A) of the centroid, and the
                descriptor radius is --pocket-radius, so nothing is cropped (--radius / --cutoff not allowed).
  ligand = its heavy atoms scaled by --ligand-radius (12 A); atoms beyond it are dropped and counted.

Output --out (one compressed npz, float32 descriptors):
  unique_id, source (str), pocket_F [N,7,K], pocket_G [N,21,K], ligand_F, ligand_G, pocket_total [N,7],
  ligand_total [N,7] (signed sum of each channel's weights over the atoms the descriptor keeps),
  n_pocket_atoms, n_ligand_atoms, n_outside_pocket, n_outside_ligand (int), ok (bool), error (str),
  seconds (per-row wall time including its share of the protein parse); K = 49 at order 12. A failed row
  (ok False) has NaN in every descriptor and total array and -1 in every count.
  Scalar keys (0-d arrays): pocket_mode, pocket_radius (NaN in shell mode), order, radius (the pocket
  descriptor radius), cutoff (NaN in sphere mode), ligand_radius, variant_note.
and <out stem>_summary.json: counts, failures, per-source timing quantiles, atom-count quantiles, the
parameters and --variant-note (free text labelling an A/B run).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse
import glob
import json
import time
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np
import pandas as pd

import pocket_fields as pf
import zernike3d

SOURCES = ("pdbbind", "bindingnet", "bindingdb")
MAX_PROTEIN_BYTES = 200 * 1024 ** 2
MAX_LIGAND_BYTES = 50 * 1024 ** 2
GROUP_ROWS = 32          # rows sharing one protein file are processed together (protein parsed once)
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
ERROR_CHARS = 300


# ---------------------------------------------------------------- inputs

def safe_join(root, *parts):
    """os.path.join(root, *parts) after rejecting absolute, parent-directory or NUL components."""
    for part in parts:
        part = str(part)
        if not part or "\x00" in part or os.path.isabs(part):
            raise ValueError(f"unsafe path component {part!r}")
        if any(seg in ("..", "") for seg in part.replace("\\", "/").split("/")):
            raise ValueError(f"unsafe path component {part!r}")
    return os.path.join(root, *[str(p) for p in parts])


def parse_roots(text):
    roots = {}
    for item in filter(None, (text or "").split(",")):
        if "=" not in item:
            raise ValueError(f"--roots item {item!r} is not source=DIR")
        key, val = item.split("=", 1)
        if key not in SOURCES:
            raise ValueError(f"unknown source {key!r} in --roots")
        roots[key] = val
    return roots


def _as_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _processed(parent_data, src):
    df = pd.read_csv(os.path.join(parent_data, f"{src}_processed.csv"), index_col=0, dtype=str,
                     keep_default_na=False)
    return df


def build_rows(meta_csv, parent_data, roots, limit=0, sources=SOURCES):
    """(rows, unresolved, info): rows are dicts uid/source/protein/ligand; unresolved (uid, source, error)."""
    meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    info = {"meta_rows": int(len(meta)), "skipped_pose_rows": 0, "duplicate_csv_ids": {}}
    if "kind" in meta.columns:
        poses = meta["kind"].str.strip() != "0"
        info["skipped_pose_rows"] = int(poses.sum())
        meta = meta[~poses]
    rows, unresolved = [], []
    for src in sources:
        uids = meta.loc[meta["source"] == src, "unique_id"].tolist()
        if limit:
            uids = uids[:limit]
        if not uids:
            continue
        if src not in roots:
            unresolved += [(u, src, f"no --roots entry for {src}") for u in uids]
            continue
        root = roots[src]
        csv = _processed(parent_data, src)
        if src == "pdbbind":
            key, cols = "PDB_code", ["refined"]
        elif src == "bindingnet":
            key, cols = "unique_identify", ["pdb", "target", "compnd"]
        else:
            key, cols = "unique_id", ["folder", "pdb_file", "mol2_file"]
        info["duplicate_csv_ids"][src] = int(csv[key].duplicated().sum())
        # the clone fills a dict row by row, so the last duplicate is the one in its pickle
        table = csv.drop_duplicates(key, keep="last").set_index(key)[cols]
        for uid in uids:
            try:
                if uid not in table.index:
                    raise KeyError(f"{uid} not in {src}_processed.csv")
                rec = table.loc[uid]
                if src == "pdbbind":
                    folder = "refined-set" if _as_bool(rec["refined"]) else "general-set"
                    base = safe_join(root, folder, uid)
                    protein = safe_join(base, f"{uid}_protein.pdb")
                    ligand = ("mol2", safe_join(base, f"{uid}_ligand.mol2"))
                elif src == "bindingnet":
                    pdb, target, compnd = rec["pdb"], rec["target"], rec["compnd"]
                    protein = safe_join(root, pdb, "rec_h_opt.pdb")
                    ligand = ("sdf_first", safe_join(root, pdb, f"target_{target}", compnd,
                                                     f"{pdb}_{target}_{compnd}.sdf"))
                else:
                    folder = rec["folder"]
                    protein = safe_join(root, folder, rec["pdb_file"])
                    ligand = ("mol2", safe_join(root, folder, rec["mol2_file"]))
            except (KeyError, ValueError) as e:
                unresolved.append((uid, src, f"{type(e).__name__}: {e}"[:ERROR_CHARS]))
                continue
            rows.append({"uid": uid, "source": src, "protein": protein, "ligand": ligand})
    return rows, unresolved, info


def _structure_dir(fep_dir):
    for cand in (os.path.join(fep_dir, "fep_benchmark_inputs", "structure_inputs"),
                 os.path.join(fep_dir, "structure_inputs"), fep_dir):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(fep_dir)


def build_fep_rows(fep_dir, limit=0, keep_ids=None):
    """Rows for the benchmark: one per distinct title of each <series>_ligands.sdf, plus a meta table.
    keep_ids (a set of unique_id) restricts the rows, e.g. to fep_benchmark.csv's scored titles."""
    from rdkit import Chem

    sdir = _structure_dir(fep_dir)
    meta_csv = os.path.join(os.path.dirname(sdir), "benchmark_metadata.csv")
    proteins = {}
    if os.path.exists(meta_csv):
        bm = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
        for _, r in bm.iterrows():
            proteins[(r["Group abbreviation"], r["Input file naming scheme"])] = (r["Protein"], r["Reference PDB"])
    rows, unresolved, meta = [], [], []
    for sdf in sorted(glob.glob(os.path.join(sdir, "*", "*_ligands.sdf"))):
        subset = os.path.basename(os.path.dirname(sdf))
        series = os.path.basename(sdf)[:-len("_ligands.sdf")]
        protein = os.path.join(os.path.dirname(sdf), f"{series}_protein.pdb")
        seen = set()
        supplier = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False)
        for idx in range(len(supplier)):
            mol = supplier[idx]
            title = mol.GetProp("_Name").strip() if mol is not None and mol.HasProp("_Name") else ""
            uid = f"{subset}/{series}/{title if title else f'entry{idx}'}"
            if uid in seen or (keep_ids is not None and uid not in keep_ids):
                continue
            seen.add(uid)
            name, ref = proteins.get((subset, series), ("", ""))
            meta.append({"unique_id": uid, "source": "fep", "split": "fep", "series_key": f"{subset}/{series}",
                         "uniprot": name, "anchor_pdb": ref.lower()})
            if mol is None:
                unresolved.append((uid, "fep", "RDKit could not parse the SDF entry"))
                continue
            rows.append({"uid": uid, "source": "fep", "protein": protein, "ligand": ("sdf_index", sdf, idx)})
    if limit:
        rows = rows[:limit]
        keep = {r["uid"] for r in rows} | {u for u, _, _ in unresolved}
        meta = [m for m in meta if m["unique_id"] in keep]
    return rows, unresolved, pd.DataFrame(meta)


def csv_path(path, work_dir):
    """A --csv path: an absolute path as is (normalised), a relative one under work_dir (safe_join); "..", NUL and
    empty paths are rejected."""
    path = str(path)
    if not path or "\x00" in path or path.lower() == "nan":
        raise ValueError(f"unsafe path {path!r}")
    if any(seg == ".." for seg in path.replace("\\", "/").split("/")):
        raise ValueError(f"unsafe path {path!r}")
    if os.path.isabs(path):
        return os.path.normpath(path)
    return safe_join(work_dir, os.path.normpath(path))


def build_csv_rows(csv_file, work_dir, limit=0):
    """(rows, unresolved, order_uids, info) for --csv: one row per CSV row (unique_id, sdf_file, pdb_file)."""
    table = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
    missing = [c for c in ("unique_id", "sdf_file", "pdb_file") if c not in table.columns]
    if missing:
        raise ValueError(f"{csv_file} lacks columns {missing}")
    dup = table["unique_id"][table["unique_id"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{csv_file}: duplicate unique_id {dup[:5]}")
    if limit:
        table = table.iloc[:limit]
    source = os.path.splitext(os.path.basename(csv_file))[0]
    rows, unresolved = [], []
    for uid, sdf, pdb in zip(table["unique_id"], table["sdf_file"], table["pdb_file"]):
        try:
            if not uid:
                raise ValueError("empty unique_id")
            rows.append({"uid": uid, "source": source, "protein": csv_path(pdb, work_dir),
                         "ligand": ("sdf_prep", csv_path(sdf, work_dir))})
        except ValueError as e:
            unresolved.append((uid, source, f"{type(e).__name__}: {e}"[:ERROR_CHARS]))
    info = {"input": "csv", "csv": os.path.abspath(csv_file), "work_dir": os.path.abspath(work_dir),
            "csv_rows": int(len(table))}
    return rows, unresolved, table["unique_id"].tolist(), info


# ---------------------------------------------------------------- worker

_SUPPLIERS = OrderedDict()


def _init_worker():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.info")


def _check_file(path, max_bytes):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"{path} is {size} bytes (> {max_bytes})")


def read_ligand(spec):
    from rdkit import Chem

    fmt, path = spec[0], spec[1]
    _check_file(path, MAX_LIGAND_BYTES)
    if fmt == "mol2":
        mol = Chem.MolFromMol2File(path)
        if mol is None:
            raise ValueError("RDKit could not read the mol2 ligand")
        return Chem.AddHs(mol, addCoords=True)
    if fmt == "sdf_first":
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        mol = supplier[0] if len(supplier) else None
    elif fmt == "sdf_prep":  # --csv: the single-molecule sdf of fep_prepare / pgdh_prepare
        supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mol = supplier[0] if len(supplier) else None
        if mol is None:
            raise ValueError("RDKit could not read the sdf ligand")
        Chem.SanitizeMol(mol)
        return Chem.AddHs(mol, addCoords=True)
    elif fmt == "sdf_index":
        supplier = _SUPPLIERS.get(path)
        if supplier is None:
            supplier = Chem.SDMolSupplier(path, removeHs=False)
            _SUPPLIERS[path] = supplier
            while len(_SUPPLIERS) > 4:
                _SUPPLIERS.popitem(last=False)
        mol = supplier[int(spec[2])]
    else:
        raise ValueError(f"unknown ligand format {fmt!r}")
    if mol is None:
        raise ValueError("RDKit could not read the sdf ligand")
    return mol


def _empty_result(row, k):
    nan_f = np.full((pf.N_CHANNELS, k), np.nan, dtype=np.float32)
    nan_g = np.full((len(zernike3d.channel_pairs(pf.N_CHANNELS)), k), np.nan, dtype=np.float32)
    nan_t = np.full(pf.N_CHANNELS, np.nan, dtype=np.float32)
    return {"uid": row["uid"], "source": row["source"], "ok": False, "error": "",
            "pocket_F": nan_f, "pocket_G": nan_g, "ligand_F": nan_f.copy(), "ligand_G": nan_g.copy(),
            "pocket_total": nan_t, "ligand_total": nan_t.copy(),
            "n_pocket_atoms": -1, "n_ligand_atoms": -1, "n_outside_pocket": -1, "n_outside_ligand": -1,
            "seconds": 0.0, "t_protein": 0.0}


def geometry_params(order=12, radius=12.0, cutoff=8.0, pocket_mode="shell", ligand_radius=12.0):
    """The per-row geometry parameters process_group reads (radius = the pocket descriptor radius; in
    sphere mode it is also the sphere radius)."""
    return {"order": order, "radius": radius, "cutoff": cutoff, "pocket_mode": pocket_mode,
            "ligand_radius": ligand_radius}


def process_group(task):
    protein_path, rows, params = task
    order, radius, cutoff = params["order"], params["radius"], params["cutoff"]
    mode, ligand_radius = params["pocket_mode"], params["ligand_radius"]
    k = zernike3d.n_invariants(order)
    t0 = time.perf_counter()
    prot, prot_error = None, ""
    try:
        _check_file(protein_path, MAX_PROTEIN_BYTES)
        prot = pf.protein_fields(protein_path)
        if len(prot[0]) == 0:
            raise ValueError("no standard-residue heavy atoms in the protein file")
    except Exception as e:  # noqa: BLE001 - recorded per row
        prot, prot_error = None, f"protein {type(e).__name__}: {e}"
    t_prot = time.perf_counter() - t0
    out = []
    for row in rows:
        t1 = time.perf_counter()
        res = _empty_result(row, k)
        try:
            mol = read_ligand(row["ligand"])
            lig_xyz, lig_w = pf.ligand_fields(mol)
            center = lig_xyz.mean(axis=0)
            lf, lg, _, l_out = zernike3d.descriptor(lig_xyz, lig_w, order=order, radius=ligand_radius,
                                                    center=center)
            l_tot = pf.channel_totals(lig_xyz, lig_w, center, ligand_radius)
            if prot is None:
                raise ValueError(prot_error)
            pfeat, pg, n_pocket, p_out, p_tot = pf.pocket_descriptor(
                prot, lig_xyz, order=order, radius=radius, cutoff=cutoff, mode=mode, return_total=True)
            if n_pocket == 0:
                where = (f"within {cutoff} A of the ligand" if mode == "shell"
                         else f"within {radius} A of the ligand centroid")
                raise ValueError(f"no protein heavy atom {where}")
            # stored only once every step succeeded: a failed row keeps NaN descriptors and -1 counts
            res.update(ligand_F=lf.astype(np.float32), ligand_G=lg.astype(np.float32),
                       ligand_total=l_tot.astype(np.float32), n_ligand_atoms=len(lig_xyz),
                       n_outside_ligand=l_out, pocket_F=pfeat.astype(np.float32),
                       pocket_G=pg.astype(np.float32), pocket_total=p_tot.astype(np.float32),
                       n_pocket_atoms=n_pocket, n_outside_pocket=p_out, ok=True)
        except Exception as e:  # noqa: BLE001 - one bad row must not stop the run
            msg = str(e) if str(e).startswith("protein ") else f"{type(e).__name__}: {e}"
            res["error"] = msg[:ERROR_CHARS]
        res["seconds"] = time.perf_counter() - t1 + t_prot / len(rows)
        res["t_protein"] = t_prot if not out else 0.0  # the group's parse time, on its first row only
        out.append(res)
    return out


def make_tasks(rows, params, group_rows=GROUP_ROWS):
    by_protein = OrderedDict()
    for r in rows:
        by_protein.setdefault(r["protein"], []).append(r)
    tasks = []
    for protein, members in by_protein.items():
        for i in range(0, len(members), group_rows):
            tasks.append((protein, members[i:i + group_rows], params))
    tasks.sort(key=lambda t: -len(t[1]))  # big groups first for load balance
    return tasks


def run_tasks(tasks, workers):
    results = []
    n_rows = sum(len(t[1]) for t in tasks)
    step = max(1, n_rows // 20)
    done, next_report, t0 = 0, step, time.time()

    def collect(batch):
        nonlocal done, next_report
        results.extend(batch)
        done += len(batch)
        if done >= next_report or done == n_rows:
            print(f"rows {done}/{n_rows} {time.time() - t0:.0f}s", flush=True)
            next_report = done + step

    if workers <= 1 or len(tasks) <= 1:
        _init_worker()
        for t in tasks:
            collect(process_group(t))
        return results
    with Pool(min(workers, len(tasks)), initializer=_init_worker) as pool:
        for batch in pool.imap_unordered(process_group, tasks, chunksize=1):
            collect(batch)
    return results


# ---------------------------------------------------------------- outputs

def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES} | \
        {"mean": float(values.mean()), "n": int(len(values))}


def assemble(order_uids, results, unresolved, order):
    """Arrays in input order; unresolved rows appear with ok False."""
    k = zernike3d.n_invariants(order)
    by_uid = {r["uid"]: r for r in results}
    for uid, src, err in unresolved:
        if uid not in by_uid:
            res = _empty_result({"uid": uid, "source": src}, k)
            res["error"] = err
            by_uid[uid] = res
    recs = [by_uid[u] for u in order_uids if u in by_uid]
    arrays = {
        "unique_id": np.array([r["uid"] for r in recs], dtype=str),
        "source": np.array([r["source"] for r in recs], dtype=str),
        "ok": np.array([r["ok"] for r in recs], dtype=bool),
        "error": np.array([r["error"] for r in recs], dtype=str),
        "seconds": np.array([r["seconds"] for r in recs], dtype=np.float32),
    }
    for name in ("pocket_F", "pocket_G", "ligand_F", "ligand_G"):
        arrays[name] = np.stack([r[name] for r in recs]).astype(np.float32) if recs else \
            np.zeros((0, pf.N_CHANNELS if name.endswith("F") else 21, k), dtype=np.float32)
    for name in ("pocket_total", "ligand_total"):
        arrays[name] = np.stack([r[name] for r in recs]).astype(np.float32) if recs else \
            np.zeros((0, pf.N_CHANNELS), dtype=np.float32)
    for name in ("n_pocket_atoms", "n_ligand_atoms", "n_outside_pocket", "n_outside_ligand"):
        arrays[name] = np.array([r[name] for r in recs], dtype=np.int64)
    return arrays, recs


def _error_class(message):
    """Error message with paths and numbers masked, for counting failure kinds."""
    words = ["<path>" if "/" in w else ("<n>" if w.strip(".,()").isdigit() else w) for w in message.split()]
    return " ".join(words)[:90]


def summarise(arrays, recs, info, args, wall):
    src = arrays["source"]
    per_source = {}
    for s in sorted(set(src.tolist())):
        sel = src == s
        ok = arrays["ok"][sel]
        okm = sel & arrays["ok"]
        protein_parse = [r["t_protein"] for r in recs if r["source"] == s and r["t_protein"] > 0]
        per_source[s] = {
            "rows": int(sel.sum()), "ok": int(ok.sum()), "failed": int((~ok).sum()),
            "seconds_per_row": _quantiles(arrays["seconds"][sel & (arrays["seconds"] > 0)]),
            "seconds_protein_parse_per_group": _quantiles(protein_parse),
            "n_pocket_atoms": _quantiles(arrays["n_pocket_atoms"][okm]),
            "n_ligand_atoms": _quantiles(arrays["n_ligand_atoms"][okm]),
            "n_outside_pocket": _quantiles(arrays["n_outside_pocket"][okm]),
            "n_outside_ligand": _quantiles(arrays["n_outside_ligand"][okm]),
            "frac_pocket_atoms_outside": _quantiles(
                arrays["n_outside_pocket"][okm] / np.maximum(arrays["n_pocket_atoms"][okm], 1)),
            "rows_with_ligand_atoms_outside": int((arrays["n_outside_ligand"][okm] > 0).sum()),
        }
    errors = Counter(_error_class(e) for e in arrays["error"].tolist() if e)
    failures = [{"unique_id": u, "source": s, "error": e} for u, s, e, ok in
                zip(arrays["unique_id"], arrays["source"], arrays["error"], arrays["ok"]) if not ok]
    return {
        "rows": int(len(arrays["ok"])), "ok": int(arrays["ok"].sum()),
        "failed": int((~arrays["ok"]).sum()), "per_source": per_source,
        "error_counts": dict(errors.most_common(30)), "failures_first": failures[:50],
        "wall_seconds": wall, "cpu_seconds_rows": float(arrays["seconds"].sum()),
        "workers": args.workers, "variant_note": args.variant_note,
        "params": {"order": args.order, "radius": args.radius, "cutoff": args.cutoff,
                   "pocket_mode": args.pocket_mode, "pocket_radius": args.pocket_radius,
                   "ligand_radius": args.ligand_radius,
                   "channels": list(pf.CHANNELS), "n_invariants": zernike3d.n_invariants(args.order),
                   "channel_pairs": [list(p) for p in zernike3d.channel_pairs(pf.N_CHANNELS)],
                   "nl_index": [list(p) for p in zernike3d.nl_index(args.order)]},
        **info,
    }


def scalar_arrays(args):
    """The run parameters as 0-d npz entries (NaN for a parameter the pocket mode does not use)."""
    nan_if_none = lambda v: np.float64(np.nan if v is None else v)
    return {"pocket_mode": np.array(args.pocket_mode), "pocket_radius": nan_if_none(args.pocket_radius),
            "order": np.int64(args.order), "radius": np.float64(args.radius),
            "cutoff": nan_if_none(args.cutoff), "ligand_radius": np.float64(args.ligand_radius),
            "variant_note": np.array(args.variant_note)}


def _stem(path):
    base = path[:-4] if path.endswith(".npz") else path
    return base


def write_outputs(arrays, summary, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out)
    summary_path = _stem(out) + "_summary.json"
    with open(summary_path + ".tmp", "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=False)
    os.replace(summary_path + ".tmp", summary_path)
    return summary_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--meta", help="paired_meta.csv (unique_id, source, split)")
    p.add_argument("--parent-data", help="dir with {pdbbind,bindingnet,bindingdb}_processed.csv")
    p.add_argument("--roots", default="", help="pdbbind=DIR,bindingnet=DIR,bindingdb=DIR")
    p.add_argument("--fep-dir", default=None, help="benchmark dir (alternative to --meta/--roots)")
    p.add_argument("--fep-ids", default=None,
                   help="optional CSV with a unique_id column (e.g. fep_benchmark.csv) restricting --fep-dir rows")
    p.add_argument("--csv", default=None,
                   help="CSV with unique_id, sdf_file, pdb_file (fep_benchmark.csv, pgdh.csv): alternative input")
    p.add_argument("--work-dir", default=None, help="--csv: base of relative sdf / pdb paths (default: the CWD)")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--workers", type=int, default=72)
    p.add_argument("--limit", type=int, default=0, help="first N rows of each source (0 = all)")
    p.add_argument("--sources", default=",".join(SOURCES), help="comma list of sources to featurise")
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--pocket-mode", choices=pf.POCKET_MODES, default="shell",
                   help="shell: atoms within --cutoff of any ligand atom (default); sphere: atoms within "
                        "--pocket-radius of the ligand centroid, descriptor radius = --pocket-radius")
    p.add_argument("--radius", type=float, default=None, help="shell mode: pocket descriptor radius (12)")
    p.add_argument("--cutoff", type=float, default=None, help="shell mode: pocket shell cutoff (8)")
    p.add_argument("--pocket-radius", type=float, default=None, help="sphere mode: sphere radius (10)")
    p.add_argument("--ligand-radius", type=float, default=12.0, help="ligand descriptor radius")
    p.add_argument("--variant-note", default="", help="free text stored in the summary (labels A/B runs)")
    args = p.parse_args(argv)
    if args.csv is not None and (args.fep_dir is not None or args.meta):
        p.error("--csv is an alternative input to --fep-dir / --meta: give one")
    if args.fep_dir is None and args.csv is None and (not args.meta or not args.parent_data):
        p.error("--meta and --parent-data are required unless --fep-dir or --csv is given")
    if args.work_dir is not None and args.csv is None:
        p.error("--work-dir applies to --csv only")
    if args.pocket_mode == "shell":
        if args.pocket_radius is not None:
            p.error("--pocket-radius applies to --pocket-mode sphere only")
        args.radius = 12.0 if args.radius is None else args.radius
        args.cutoff = 8.0 if args.cutoff is None else args.cutoff
    else:
        if args.radius is not None or args.cutoff is not None:
            p.error("--pocket-mode sphere takes --pocket-radius (the descriptor radius); "
                    "--radius and --cutoff are shell-mode options")
        args.pocket_radius = 10.0 if args.pocket_radius is None else args.pocket_radius
        args.radius = args.pocket_radius
    for name in ("radius", "ligand_radius") + (("cutoff",) if args.pocket_mode == "shell" else ()):
        if not getattr(args, name) > 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if args.csv:
        try:
            rows, unresolved, order_uids, info = build_csv_rows(args.csv, args.work_dir or os.getcwd(), args.limit)
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
    elif args.fep_dir:
        keep = None
        if args.fep_ids:
            keep = set(pd.read_csv(args.fep_ids, dtype=str, keep_default_na=False, usecols=["unique_id"])
                       ["unique_id"])
        rows, unresolved, fep_meta = build_fep_rows(args.fep_dir, args.limit, keep)
        info = {"input": "fep", "fep_dir": os.path.abspath(args.fep_dir)}
        order_uids = fep_meta["unique_id"].tolist()
        meta_path = _stem(args.out) + "_meta.csv"
        os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
        fep_meta.to_csv(meta_path, index=False)
        info["meta_written"] = meta_path
    else:
        sources = [s for s in args.sources.split(",") if s]
        rows, unresolved, info = build_rows(args.meta, args.parent_data, parse_roots(args.roots),
                                            args.limit, sources)
        info["input"] = "meta"
        order_uids = [r["uid"] for r in rows] + [u for u, _, _ in unresolved]
        meta = pd.read_csv(args.meta, dtype=str, keep_default_na=False, usecols=["unique_id"])
        position = {u: i for i, u in enumerate(meta["unique_id"])}
        order_uids.sort(key=lambda u: position.get(u, len(position)))
    info["unresolved"] = len(unresolved)
    print(f"{len(rows)} rows to featurise, {len(unresolved)} unresolved", flush=True)
    params = geometry_params(order=args.order, radius=args.radius, cutoff=args.cutoff,
                            pocket_mode=args.pocket_mode, ligand_radius=args.ligand_radius)
    tasks = make_tasks(rows, params)
    results = run_tasks(tasks, args.workers)
    arrays, recs = assemble(order_uids, results, unresolved, args.order)
    summary = summarise(arrays, recs, info, args, time.time() - t0)
    summary_path = write_outputs({**arrays, **scalar_arrays(args)}, summary, args.out)
    print(f"wrote {args.out} ({summary['ok']}/{summary['rows']} ok) and {summary_path} "
          f"in {summary['wall_seconds']:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    main()
