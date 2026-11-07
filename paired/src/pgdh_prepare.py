"""Prepare the 15-PGDH lead-optimisation set (Rich, Dodda & Price 2026) for AEV-PLIG evaluation.

Source: github.com/inductive-bio-misc-collab/ml-affinity-15-PGDH-case-study (no licence file:
evaluation use only), ChemMedChem doi 10.1002/cmdc.70531. 43 inhibitors in four series.

Inputs under --repo-dir (the extracted repository; nothing in it is executed):
  data/poses.sdf               canonical list and labels: title compound_N, SD tags ID, series,
                               pKi (experimental, = -r_fepplus_exp_dg / 1.364), FEP_pKi (FEP+), smiles
  scoring/vina_predictions.sdf the gnina/Vina-minimised poses the paper's AEV-PLIG run scored
                               (scoring/run_aevplig.py splits this file); heavy atoms only, almost no H
  data/reference.pdb           PDB 9PFL, Maestro-prepared receptor (chain A, NAD as HETATM, ACE/NMA caps)
  scoring/aevplig_original.csv, scoring/aevplig_finetuned.csv   the paper's predictions (unique_id, preds)

Series mapping (the `series` tag is build_poses_sdf.py's `table` with "/" replaced by "_"):
  "1" -> S1, "2" -> S2, "3" -> S3, "4_5" (table "4/5") -> S4, the paper's naming (11/7/13/12 compounds).

Outputs in --out-dir:
  pgdh.csv                     fep_benchmark.csv columns (fep_prepare.COLUMNS): unique_id =
                               pgdh/<S1..S4>/<compound_N>, pK = the pKi tag (experimental, never
                               FEP_pKi), sdf_file, pdb_file, subset = pgdh, series, ref_pdb = 9PFL,
                               n_entries (entries carrying that ID in the pose source, normally 1)
  pgdh_<S>_<compound_N>.sdf    one pose per compound, from vina_predictions.sdf, falling back to
                               poses.sdf (recorded) when a compound has no Vina pose
  pgdh_reference_9PFL.pdb      data/reference.pdb copied unchanged
  pgdh_paper.csv, pgdh_paper_finetuned.csv   the paper's preds keyed by our unique_id
  pgdh_fepplus.csv             FEP_pKi keyed by our unique_id (the paper's physics reference)
  pgdh_residue_census.csv      receptor atoms the clone's featuriser ignores (when PDB_Atom_Keys.csv
                               is found; pass it to eval_paired.py --residue-census)
  pgdh_summary.json            counts and pK ranges per series, pose source per compound, missing
                               labels, paper-CSV joins, the hydrogen handling and coordinate check

Hydrogens (--hydrogens): run_aevplig.py writes each Vina pose as Chem.AddHs(mol, addCoords=True)
of the sanitised RDKit molecule. AEV-PLIG's atom features count explicit H neighbours and use the
explicit valence, so a hydrogen-free copy changes the features of every ligand. `add` (default)
reproduces the paper's input: the source atoms (all heavy atoms and the few H present) keep their
coordinates byte-identical (checked: re-read with removeHs=False, sanitize=False and compared),
and the missing H are appended by RDKit. `asis` writes the entry exactly as read
(removeHs=False, sanitize=False, as fep_prepare.py does).

Scoring mode (--score-eval-dir DIR ... --data-dir DIR): reads DIR/predictions_fep.csv written by
eval_paired.py and writes DIR/pgdh_spearman.csv: Spearman of every predictor (ours, the paper's
original and fine-tuned AEV-PLIG, FEP+) on the same ligands, pooled and per series, including the
paper's pooled S1&S2 group (the paper's Table 2 groups: all, S1&S2, S3, S4).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import json
import math
import re
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))  # src/, also under python -I
from fep_prepare import COLUMNS, load_allowed_atoms, residue_census, safe_name, write_single_sdf

SUBSET = "pgdh"
REF_PDB = "9PFL"
ID_TAG, SERIES_TAG, PK_TAG, FEP_TAG = "ID", "series", "pKi", "FEP_pKi"
SERIES_MAP = {"1": "S1", "2": "S2", "3": "S3", "4_5": "S4", "4/5": "S4"}
SERIES_ORDER = ["S1", "S2", "S3", "S4"]
POSES_SDF = os.path.join("data", "poses.sdf")
VINA_SDF = os.path.join("scoring", "vina_predictions.sdf")
REFERENCE_PDB = os.path.join("data", "reference.pdb")
PAPER_CSVS = {"pgdh_paper.csv": os.path.join("scoring", "aevplig_original.csv"),
              "pgdh_paper_finetuned.csv": os.path.join("scoring", "aevplig_finetuned.csv")}
PDB_NAME = f"{SUBSET}_reference_{REF_PDB}.pdb"
SCORE_GROUPS = [("all", SERIES_ORDER), ("S1", ["S1"]), ("S2", ["S2"]), ("S3", ["S3"]), ("S4", ["S4"]),
                ("S1&S2", ["S1", "S2"])]
SCORE_PREDICTORS = [("paper_original", "pgdh_paper.csv"), ("paper_finetuned", "pgdh_paper_finetuned.csv"),
                    ("fepplus", "pgdh_fepplus.csv")]


# ----------------------------------------------------------------------------- reading

def compound_id(mol):
    """The ID tag, else the title (run_aevplig.py's rule)."""
    if mol.HasProp(ID_TAG) and mol.GetProp(ID_TAG).strip():
        return mol.GetProp(ID_TAG).strip()
    return mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""


def compound_number(cid):
    m = re.search(r"(\d+)$", cid)
    return int(m.group(1)) if m else math.inf


def tag(mol, name):
    return mol.GetProp(name).strip() if mol.HasProp(name) else ""


def float_tag(mol, name):
    try:
        value = float(tag(mol, name))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def index_sdf(path):
    """{compound id: (first record index, n_entries)}, the record count and unparsable indices.

    Records are read with removeHs=False, sanitize=False; the index is the supplier position, so
    the same record can be re-read with another supplier (the paper's sanitised read)."""
    first, counts, unparsed = {}, {}, []
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    for i in range(len(suppl)):
        mol = suppl[i]
        if mol is None:
            unparsed.append(i)
            continue
        cid = compound_id(mol)
        counts[cid] = counts.get(cid, 0) + 1
        first.setdefault(cid, i)
    return {c: (first[c], counts[c]) for c in first}, len(suppl), unparsed


# ----------------------------------------------------------------------------- writing

def write_pose(src_path, index, dst_path, hydrogens):
    """Write record `index` of src_path to dst_path; return (n_h_added, sanitised_ok).

    hydrogens == "add": the paper's run_aevplig.py (sanitised read, Chem.AddHs(addCoords=True),
    default SDWriter); a record RDKit cannot sanitise is written as is (sanitised_ok False).
    hydrogens == "asis": the record as read (removeHs=False, sanitize=False), as fep_prepare."""
    raw = Chem.SDMolSupplier(src_path, removeHs=False, sanitize=False)[index]
    if hydrogens == "add":
        san = Chem.SDMolSupplier(src_path, removeHs=False)[index]
        if san is not None:
            molh = Chem.AddHs(san, addCoords=True)
            with Chem.SDWriter(dst_path) as writer:
                writer.write(molh)
            return molh.GetNumAtoms() - raw.GetNumAtoms(), True
        write_single_sdf(raw, dst_path)
        return 0, False
    write_single_sdf(raw, dst_path)
    return 0, None


def coordinate_check(src_path, index, dst_path):
    """Max |difference| (A) between the source record's atoms and the first atoms of the written
    file, with element agreement; both read with removeHs=False, sanitize=False. 0.0 = identical."""
    src = Chem.SDMolSupplier(src_path, removeHs=False, sanitize=False)[index]
    out = Chem.SDMolSupplier(dst_path, removeHs=False, sanitize=False)[0]
    n = src.GetNumAtoms()
    if out is None or out.GetNumAtoms() < n:
        return math.inf
    if any(src.GetAtomWithIdx(i).GetSymbol() != out.GetAtomWithIdx(i).GetSymbol() for i in range(n)):
        return math.inf
    a = src.GetConformer().GetPositions()
    b = out.GetConformer().GetPositions()[:n]
    return float(np.abs(a - b).max()) if n else 0.0


def readable_like_process_data(path):
    """process_data's read: SDMolSupplier(removeHs=False) with sanitisation, exactly one record."""
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    return len(suppl) == 1 and suppl[0] is not None


# ----------------------------------------------------------------------------- preparation

def map_predictions(src_csv, cid_to_uid, order):
    """The paper CSV (unique_id = compound id, preds) re-keyed to our unique_ids, in `order`.

    Returns (DataFrame[unique_id, preds], bookkeeping dict)."""
    paper = pd.read_csv(src_csv, dtype={"unique_id": str})
    paper["unique_id"] = paper["unique_id"].str.strip()
    n_rows = len(paper)
    dup = sorted(paper.loc[paper["unique_id"].duplicated(), "unique_id"].unique())
    paper = paper.drop_duplicates("unique_id", keep="first")
    paper["uid"] = paper["unique_id"].map(cid_to_uid)
    mapped = paper.dropna(subset=["uid"]).set_index("uid")["preds"]
    out = pd.DataFrame({"unique_id": [u for u in order if u in mapped.index]})
    out["preds"] = out["unique_id"].map(mapped).astype(float)
    uid_to_cid = {u: c for c, u in cid_to_uid.items()}
    book = {"source": src_csv, "n_rows": int(n_rows), "n_mapped": int(len(out)),
            "paper_ids_not_in_pgdh": sorted(paper.loc[paper["uid"].isna(), "unique_id"], key=compound_number),
            "pgdh_ids_not_in_paper": [uid_to_cid[u] for u in order if u not in mapped.index],
            "duplicate_paper_ids": dup}
    return out, book


def prepare_pgdh(repo_dir, out_dir, hydrogens="add", atom_keys_csv=None):
    """Write pgdh.csv and its companions (see the module docstring); return pgdh.csv as a DataFrame."""
    if hydrogens not in ("add", "asis"):
        raise ValueError(f"hydrogens must be add or asis, not {hydrogens!r}")
    repo_dir = os.path.abspath(repo_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    poses_path = os.path.join(repo_dir, POSES_SDF)
    vina_path = os.path.join(repo_dir, VINA_SDF)
    pdb_src = os.path.join(repo_dir, REFERENCE_PDB)
    for path in (poses_path, pdb_src):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    poses_index, n_pose_records, poses_unparsed = index_sdf(poses_path)
    poses_suppl = Chem.SDMolSupplier(poses_path, removeHs=False, sanitize=False)
    if os.path.isfile(vina_path):
        vina_index, n_vina_records, vina_unparsed = index_sdf(vina_path)
        vina_suppl = Chem.SDMolSupplier(vina_path, removeHs=False, sanitize=False)
    else:
        vina_index, n_vina_records, vina_unparsed, vina_suppl = {}, 0, [], None

    pdb_dst = os.path.join(out_dir, PDB_NAME)
    shutil.copyfile(pdb_src, pdb_dst)

    rows, fepplus, per_compound = [], {}, {}
    no_series, unknown_series, no_pk, label_mismatch = [], {}, [], []
    for cid in sorted(poses_index, key=lambda c: (compound_number(c), c)):
        pmol = poses_suppl[poses_index[cid][0]]
        series_tag, pk = tag(pmol, SERIES_TAG), float_tag(pmol, PK_TAG)
        if not series_tag:
            no_series.append(cid)
            continue
        if series_tag not in SERIES_MAP:
            unknown_series[cid] = series_tag
            continue
        if pk is None:
            no_pk.append(cid)
            continue
        series = SERIES_MAP[series_tag]
        uid = f"{SUBSET}/{series}/{cid}"
        if cid in vina_index:
            src_path, (index, n_entries), source = vina_path, vina_index[cid], VINA_SDF
            vmol = vina_suppl[index]
            for name in (SERIES_TAG, PK_TAG):
                if vmol.HasProp(name) and tag(vmol, name) != tag(pmol, name):
                    label_mismatch.append({"compound": cid, "tag": name, "poses": tag(pmol, name),
                                           "vina": tag(vmol, name)})
        else:
            src_path, (index, n_entries), source = poses_path, poses_index[cid], POSES_SDF
        sdf_dst = os.path.join(out_dir, f"{SUBSET}_{series}_{safe_name(cid)}.sdf")
        n_h_added, sanitised = write_pose(src_path, index, sdf_dst, hydrogens)
        max_diff = coordinate_check(src_path, index, sdf_dst)
        per_compound[cid] = {"unique_id": uid, "pose_source": source, "n_entries": n_entries,
                             "n_h_added": n_h_added, "sanitised": sanitised,
                             "max_coord_diff": max_diff, "readable": readable_like_process_data(sdf_dst)}
        rows.append({"unique_id": uid, "pK": pk, "sdf_file": sdf_dst, "pdb_file": pdb_dst, "subset": SUBSET,
                     "series": series, "ref_pdb": REF_PDB, "n_entries": n_entries})
        fep = float_tag(pmol, FEP_TAG)
        if fep is not None:
            fepplus[uid] = fep

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["_order"] = df["series"].map(SERIES_ORDER.index)
    df["_num"] = [compound_number(u.rsplit("/", 1)[1]) for u in df["unique_id"]]
    df = df.sort_values(["_order", "_num", "unique_id"]).drop(columns=["_order", "_num"]).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "pgdh.csv"), index=False)
    order = df["unique_id"].tolist()
    cid_to_uid = {c: d["unique_id"] for c, d in per_compound.items()}

    paper_join = {}
    for out_name, rel in PAPER_CSVS.items():
        src = os.path.join(repo_dir, rel)
        if not os.path.isfile(src):
            paper_join[out_name] = {"source": src, "missing": True}
            continue
        mapped, book = map_predictions(src, cid_to_uid, order)
        mapped.to_csv(os.path.join(out_dir, out_name), index=False)
        paper_join[out_name] = book
    pd.DataFrame({"unique_id": [u for u in order if u in fepplus],
                  "preds": [fepplus[u] for u in order if u in fepplus]}).to_csv(
        os.path.join(out_dir, "pgdh_fepplus.csv"), index=False)

    census = None
    if atom_keys_csv and os.path.isfile(atom_keys_csv):
        census = pd.DataFrame([{"subset": SUBSET, "series": REF_PDB, **r}
                               for r in residue_census(pdb_src, load_allowed_atoms(atom_keys_csv))],
                              columns=["subset", "series", "record", "residue_name", "n_residues",
                                       "n_dropped_atoms", "dropped_atom_names"])
        census.to_csv(os.path.join(out_dir, "pgdh_residue_census.csv"), index=False)

    by_series = {}
    for s in SERIES_ORDER:
        sub = df[df["series"] == s]
        by_series[s] = {"n": int(len(sub)),
                        "pK_min": float(sub["pK"].min()) if len(sub) else None,
                        "pK_max": float(sub["pK"].max()) if len(sub) else None,
                        "pK_mean": float(sub["pK"].mean()) if len(sub) else None,
                        "series_tag": sorted(t for t, v in SERIES_MAP.items() if v == s)}
    pose_source = {c: per_compound[c]["pose_source"] for c in per_compound}
    summary = {
        "repo_dir": repo_dir,
        "n_rows": int(len(df)),
        "series_map": SERIES_MAP,
        "label_tags": {"id": ID_TAG, "series": SERIES_TAG, "pK": PK_TAG, "fepplus_not_used_as_label": FEP_TAG},
        "by_series": by_series,
        "pK_min": float(df["pK"].min()) if len(df) else None,
        "pK_max": float(df["pK"].max()) if len(df) else None,
        "hydrogens": hydrogens,
        "n_poses_records": n_pose_records,
        "n_vina_records": n_vina_records,
        "vina_sdf_present": vina_suppl is not None,
        "unparsable_records": {POSES_SDF: poses_unparsed, VINA_SDF: vina_unparsed},
        "pose_source_counts": {k: int(v) for k, v in pd.Series(pose_source, dtype=object).value_counts().items()},
        "compounds_without_vina_pose": sorted([c for c, s in pose_source.items() if s != VINA_SDF], key=compound_number),
        "vina_compounds_not_in_poses": sorted(set(vina_index) - set(poses_index), key=compound_number),
        "compounds_without_series": no_series,
        "compounds_with_unknown_series": unknown_series,
        "compounds_without_pK": no_pk,
        "label_mismatch_vina_vs_poses": label_mismatch,
        "max_coord_diff": max((d["max_coord_diff"] for d in per_compound.values()), default=None),
        "coords_identical_all": all(d["max_coord_diff"] == 0.0 for d in per_compound.values()),
        "not_sanitisable_written_as_is": [c for c, d in per_compound.items() if d["sanitised"] is False],
        "not_readable_by_process_data": [c for c, d in per_compound.items() if not d["readable"]],
        "multi_entry_compounds": {c: d["n_entries"] for c, d in per_compound.items() if d["n_entries"] != 1},
        "paper_join": paper_join,
        "n_fepplus": len(fepplus),
        "dropped_protein_atoms": census.to_dict("records") if census is not None else None,
        "compounds": per_compound,
    }
    with open(os.path.join(out_dir, "pgdh_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return df


# ----------------------------------------------------------------------------- scoring

def spearman(truth, pred):
    from scipy import stats

    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(pred) == 0:
        return math.nan
    return float(stats.spearmanr(truth, pred)[0])


def score_eval_dir(eval_dir, data_dir):
    """DIR/pgdh_spearman.csv (predictor, group, n, spearman) on the ligands in predictions_fep.csv."""
    ours = pd.read_csv(os.path.join(eval_dir, "predictions_fep.csv"), dtype={"unique_id": str})
    labels = pd.read_csv(os.path.join(data_dir, "pgdh.csv"), dtype={"unique_id": str, "series": str})
    base = ours[["unique_id", "pred"]].merge(labels[["unique_id", "series", "pK"]], on="unique_id", how="inner")
    if len(base) != len(ours):
        raise SystemExit(f"ERROR: {len(ours) - len(base)} predictions_fep.csv ids are not in pgdh.csv")
    if "truth" in ours and not np.allclose(ours.set_index("unique_id").loc[base["unique_id"], "truth"],
                                           base["pK"], atol=1e-6):
        raise SystemExit("ERROR: predictions_fep.csv truth differs from pgdh.csv pK")
    predictors = [("ours", base.rename(columns={"pred": "p"}))]
    for name, fname in SCORE_PREDICTORS:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            other = pd.read_csv(path, dtype={"unique_id": str}).rename(columns={"preds": "p"})
            predictors.append((name, base[["unique_id", "series", "pK"]].merge(other, on="unique_id")))
    rows = []
    for name, df in predictors:
        for group, members in SCORE_GROUPS:
            sub = df[df["series"].isin(members)]
            rows.append({"predictor": name, "group": group, "n": int(len(sub)),
                         "spearman": spearman(sub["pK"], sub["p"])})
    out = pd.DataFrame(rows, columns=["predictor", "group", "n", "spearman"])
    out.to_csv(os.path.join(eval_dir, "pgdh_spearman.csv"), index=False)
    return out


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-dir", help="extracted ml-affinity-15-PGDH-case-study repository")
    ap.add_argument("--out-dir", help="where pgdh.csv, the per-compound SDFs and the receptor go")
    ap.add_argument("--hydrogens", choices=("add", "asis"), default="add",
                    help="add: the paper's AddHs(addCoords=True) on the Vina poses (default); asis: copy as read")
    ap.add_argument("--atom-keys", default=os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv"),
                    help="the clone's PDB_Atom_Keys.csv, for pgdh_residue_census.csv (skipped if absent)")
    ap.add_argument("--score-eval-dir", nargs="+", default=None,
                    help="scoring mode: eval_paired.py output dirs (predictions_fep.csv) -> pgdh_spearman.csv")
    ap.add_argument("--data-dir", default=None, help="scoring mode: dir holding pgdh.csv and pgdh_paper*.csv")
    args = ap.parse_args(argv)

    if args.score_eval_dir:
        if not args.data_dir:
            ap.error("--score-eval-dir needs --data-dir")
        for d in args.score_eval_dir:
            out = score_eval_dir(d, args.data_dir)
            table = out.pivot(index="predictor", columns="group", values="spearman")
            table = table[[g for g, _ in SCORE_GROUPS]].reindex([p for p in ["ours"] + [n for n, _ in SCORE_PREDICTORS]
                                                                 if p in table.index])
            print(f"== {d}: Spearman (n = {int(out.loc[(out.predictor == 'ours') & (out.group == 'all'), 'n'].iloc[0])})")
            print(table.round(3).to_string())
        return
    if not (args.repo_dir and args.out_dir):
        ap.error("--repo-dir and --out-dir are required (or --score-eval-dir with --data-dir)")

    atom_keys = args.atom_keys if os.path.isfile(args.atom_keys) else None
    if atom_keys is None:
        print(f"WARNING: {args.atom_keys} not found; skipping the residue census")
    df = prepare_pgdh(args.repo_dir, args.out_dir, args.hydrogens, atom_keys)
    with open(os.path.join(os.path.abspath(args.out_dir), "pgdh_summary.json")) as fh:
        s = json.load(fh)
    print(f"rows {len(df)}  hydrogens {s['hydrogens']}  pose sources {s['pose_source_counts']}  "
          f"coords identical {s['coords_identical_all']} (max diff {s['max_coord_diff']})")
    for k, v in s["by_series"].items():
        if v["n"]:
            print(f"  {k} (tag {', '.join(v['series_tag'])}): n {v['n']}  pK {v['pK_min']:.3f}..{v['pK_max']:.3f}  "
                  f"mean {v['pK_mean']:.3f}")
    for key in ("compounds_without_vina_pose", "vina_compounds_not_in_poses", "compounds_without_series",
                "compounds_with_unknown_series", "compounds_without_pK", "label_mismatch_vina_vs_poses",
                "not_sanitisable_written_as_is", "not_readable_by_process_data", "multi_entry_compounds"):
        if s[key]:
            print(f"  {key}: {s[key]}")
    for name, book in s["paper_join"].items():
        if book.get("missing"):
            print(f"  {name}: source missing ({book['source']})")
        else:
            print(f"  {name}: mapped {book['n_mapped']} of {book['n_rows']} rows; paper-only "
                  f"{book['paper_ids_not_in_pgdh']}; ours-only {book['pgdh_ids_not_in_paper']}")
    for r in s["dropped_protein_atoms"] or []:
        print(f"  dropped by featuriser: {r['record']:6s} {r['residue_name']:4s} residues {r['n_residues']:4d} "
              f"heavy atoms {r['n_dropped_atoms']:4d}")


if __name__ == "__main__":
    main()
