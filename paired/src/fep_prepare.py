"""Prepare the Schrodinger public FEP benchmark for AEV-PLIG featurisation.

Reads <benchmark>/fep_benchmark_inputs/structure_inputs/<subset>/<series>_{ligands.sdf,protein.pdb},
keeps the first SDF entry of every ligand title (and, when 21_4_results/ is present, only the
titles the benchmark scores), converts r_exp_dg (kcal/mol) to pK, writes one
single-molecule SDF per ligand and one protein PDB per series (copied unchanged) under
<structures-dir>, and writes <out-dir>/fep_benchmark.csv with COLUMNS (unique_id =
<subset>/<series>/<title>, the paper's convention). It also writes fep_prepare_summary.json and,
when PDB_Atom_Keys.csv is available, fep_residue_census.csv: the protein atoms the clone's
featuriser (GetMolAEVs_extended -> LoadPDBasDF_old) never sees, i.e. every HETATM record and
every ATOM whose RES-ATOM key is absent from PDB_Atom_Keys.csv (ACE, NMA, NME caps, ...).
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import glob
import json
import math
import re
import shutil

import pandas as pd
from rdkit import Chem

R_KCAL = 0.0019872  # kcal / (mol K)
T_K = 298.15
COLUMNS = ["unique_id", "pK", "sdf_file", "pdb_file", "subset", "series", "ref_pdb", "n_entries"]
LIGAND_SUFFIX = "_ligands.sdf"
PROTEIN_SUFFIX = "_protein.pdb"
# charge_annhil/egfr carries its experimental value only as r_fepplus_exp_dg (same numbers as the
# 21_4_results "Exp. dG" column); every other series uses r_exp_dg.
EXP_DG_TAGS = ("r_exp_dg", "r_fepplus_exp_dg")


def dg_to_pk(dg):
    """Binding free energy (kcal/mol) to pK at 298.15 K."""
    return -float(dg) / (R_KCAL * T_K * math.log(10.0))


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._+-]", "_", text)


def find_inputs_dir(benchmark_dir):
    """Accept the repository root or its fep_benchmark_inputs/ directory."""
    for cand in (os.path.join(benchmark_dir, "fep_benchmark_inputs"), benchmark_dir):
        if os.path.isdir(os.path.join(cand, "structure_inputs")):
            return cand
    raise FileNotFoundError(f"no structure_inputs/ under {benchmark_dir}")


def load_ref_pdbs(inputs_dir):
    """(subset, series) -> Reference PDB, from benchmark_metadata.csv, then subset_metadata.csv."""
    refs = {}
    paths = [os.path.join(inputs_dir, "benchmark_metadata.csv")]
    paths += sorted(glob.glob(os.path.join(inputs_dir, "structure_inputs", "*", "subset_metadata.csv")))
    for path in paths:
        if not os.path.isfile(path):
            continue
        meta = pd.read_csv(path, dtype=str).fillna("")
        for _, r in meta.iterrows():
            key = (r["Group abbreviation"].strip(), r["Input file naming scheme"].strip())
            if key not in refs and r["Reference PDB"].strip():
                refs[key] = r["Reference PDB"].strip()
    return refs


def read_ligands(sdf_path):
    """Return ([(title, mol, n_entries)] in first-seen order, n_entries_total, n_unparsed)."""
    first, counts, order = {}, {}, []
    n_total, n_unparsed = 0, 0
    for mol in Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False):
        n_total += 1
        if mol is None:
            n_unparsed += 1
            continue
        title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        counts[title] = counts.get(title, 0) + 1
        if title not in first:
            first[title] = mol
            order.append(title)
    return [(t, first[t], counts[t]) for t in order], n_total, n_unparsed


def exp_dg(mol):
    """Experimental dG (kcal/mol) from the first numeric tag in EXP_DG_TAGS, else None."""
    for tag in EXP_DG_TAGS:
        if mol.HasProp(tag):
            try:
                value = float(mol.GetProp(tag))
            except ValueError:
                continue
            if math.isfinite(value):
                return value
    return None


def write_single_sdf(mol, path):
    writer = Chem.SDWriter(path)
    writer.SetKekulize(False)  # write the bonds as read; process_data sanitises on reading
    writer.write(mol)
    writer.close()


def _is_heavy(atom_name, element):
    if element in ("H", "D"):
        return False
    if element:
        return True
    # LoadPDBasDF_old's hydrogen rule when the element column is empty
    if len(atom_name) < 4:
        return atom_name[0] != "H"
    return atom_name[0] != "H" and atom_name[1] != "H"


def residue_census(pdb_path, allowed_atoms):
    """Heavy atoms the clone's featuriser drops: all HETATM, and ATOM with RES-ATOM not in keys.

    Returns one dict per (record, residue_name) with dropped atoms: record, residue_name,
    n_residues (residues with >= 1 dropped heavy atom), n_dropped_atoms, dropped_atom_names.
    """
    stats = {}
    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            name = line[12:16].replace(" ", "")
            if not name or not _is_heavy(name, line[76:78].strip().upper()):
                continue
            resname = line[17:20].strip()
            dropped = record == "HETATM" or f"{line[17:20]}-{name}" not in allowed_atoms
            if not dropped:
                continue
            s = stats.setdefault((record, resname), {"res": set(), "names": set(), "dropped": 0})
            s["res"].add((line[21], line[22:27]))
            s["names"].add(name)
            s["dropped"] += 1
    return [{"record": record, "residue_name": resname, "n_residues": len(s["res"]),
             "n_dropped_atoms": s["dropped"], "dropped_atom_names": " ".join(sorted(s["names"]))}
            for (record, resname), s in sorted(stats.items())]


def load_allowed_atoms(atom_keys_csv):
    return set(pd.read_csv(atom_keys_csv)["PDB_ATOM"])


def load_scored_ligands(inputs_dir):
    """unique_ids the benchmark itself scores, or None when 21_4_results/ is absent.

    21_4_results/benchmark_output_metadata.csv maps (Group abbreviation, Input file naming scheme)
    to ligand_predictions/<subset>/<Output file naming scheme>.csv, whose first column names the
    scored ligands. Alternate conformers ("flip", "pose 2", "in", ...) carry their own SDF titles
    but are not scored there; the AEV-PLIG paper's 1,184 ligands are exactly SDF titles that are
    scored.
    """
    results = os.path.join(os.path.dirname(os.path.abspath(inputs_dir)), "21_4_results")
    meta_path = os.path.join(results, "benchmark_output_metadata.csv")
    if not os.path.isfile(meta_path):
        return None
    scored = set()
    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    for _, r in meta.iterrows():
        subset, series = r["Group abbreviation"].strip(), r["Input file naming scheme"].strip()
        path = os.path.join(results, "ligand_predictions", subset, r["Output file naming scheme"].strip() + ".csv")
        if os.path.isfile(path):
            names = pd.read_csv(path, dtype=str).iloc[:, 0].fillna("")
            scored.update(f"{subset}/{series}/{n.strip()}" for n in names)
    return scored


def prepare_benchmark(benchmark_dir, out_dir, structures_dir=None, atom_keys_csv=None, scored_only=True):
    """Write fep_benchmark.csv (+ structures, summary, census) and return it as a DataFrame.

    With scored_only (default) and 21_4_results/ present, ligands the benchmark does not score
    (alternate conformers, series without FEP+ results) are left out and listed in the summary.
    """
    inputs_dir = find_inputs_dir(benchmark_dir)
    out_dir = os.path.abspath(out_dir)
    structures_dir = os.path.abspath(structures_dir or os.path.join(out_dir, "structures"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)
    refs = load_ref_pdbs(inputs_dir)
    allowed = load_allowed_atoms(atom_keys_csv) if atom_keys_csv else None
    scored = load_scored_ligands(inputs_dir) if scored_only else None

    rows, skipped, census, no_ref, unscored = [], [], [], [], []
    n_entries_total = n_unparsed_total = n_duplicate_entries = 0
    sdfs = sorted(glob.glob(os.path.join(inputs_dir, "structure_inputs", "*", "*" + LIGAND_SUFFIX)))
    for sdf in sdfs:
        subset = os.path.basename(os.path.dirname(sdf))
        series = os.path.basename(sdf)[: -len(LIGAND_SUFFIX)]
        pdb_src = sdf[: -len(LIGAND_SUFFIX)] + PROTEIN_SUFFIX
        if not os.path.isfile(pdb_src):
            skipped.append({"unique_id": f"{subset}/{series}/*", "reason": "no protein pdb"})
            continue
        ref_pdb = refs.get((subset, series), "")
        prefix = f"{safe_name(subset)}_{safe_name(series)}"
        pdb_dst = os.path.join(structures_dir, prefix + PROTEIN_SUFFIX)
        n_rows_before = len(rows)

        ligands, n_total, n_unparsed = read_ligands(sdf)
        n_entries_total += n_total
        n_unparsed_total += n_unparsed
        if n_unparsed:
            skipped.append({"unique_id": f"{subset}/{series}/*", "reason": f"{n_unparsed} unparsable entries"})
        used = set()
        for title, mol, n_entries in ligands:
            n_duplicate_entries += n_entries - 1
            uid = f"{subset}/{series}/{title}"
            if scored is not None and uid not in scored:
                unscored.append(uid)
                continue
            dg = exp_dg(mol)
            if dg is None:
                skipped.append({"unique_id": uid, "reason": "no numeric " + "/".join(EXP_DG_TAGS)})
                continue
            stem = f"{prefix}_{safe_name(title)}"
            name, k = stem, 1
            while name in used or name == prefix + "_protein":
                name, k = f"{stem}~{k}", k + 1
            used.add(name)
            sdf_dst = os.path.join(structures_dir, name + ".sdf")
            write_single_sdf(mol, sdf_dst)
            rows.append({"unique_id": uid, "pK": dg_to_pk(dg), "sdf_file": sdf_dst, "pdb_file": pdb_dst,
                         "subset": subset, "series": series, "ref_pdb": ref_pdb, "n_entries": n_entries})
        if len(rows) == n_rows_before:
            continue  # no ligand kept: no protein copy, not counted
        shutil.copyfile(pdb_src, pdb_dst)
        if not ref_pdb:
            no_ref.append(f"{subset}/{series}")
        if allowed is not None:
            census += [{"subset": subset, "series": series, **r} for r in residue_census(pdb_src, allowed)]

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(os.path.join(out_dir, "fep_benchmark.csv"), index=False)

    summary = {
        "benchmark_inputs": inputs_dir,
        "n_series": int(df.groupby(["subset", "series"]).ngroups) if len(df) else 0,
        "n_series_files": len(sdfs),
        "n_rows": len(df),
        "n_sdf_entries": n_entries_total,
        "n_unparsed_entries": n_unparsed_total,
        "n_duplicate_title_entries": n_duplicate_entries,
        "pK_min": float(df["pK"].min()) if len(df) else None,
        "pK_max": float(df["pK"].max()) if len(df) else None,
        "pK_mean": float(df["pK"].mean()) if len(df) else None,
        "rows_per_subset": {k: int(v) for k, v in df["subset"].value_counts().sort_index().items()},
        "series_without_ref_pdb": no_ref,
        "scored_filter": scored is not None,
        "n_unscored_dropped": len(unscored),
        "unscored_dropped": unscored,
        "skipped": skipped,
    }
    if allowed is not None:
        cdf = pd.DataFrame(census, columns=["subset", "series", "record", "residue_name", "n_residues",
                                            "n_dropped_atoms", "dropped_atom_names"])
        cdf.to_csv(os.path.join(out_dir, "fep_residue_census.csv"), index=False)
        tot = (cdf.groupby(["record", "residue_name"])
               .agg(n_series=("series", "size"), n_residues=("n_residues", "sum"),
                    n_dropped_atoms=("n_dropped_atoms", "sum"),
                    atom_names=("dropped_atom_names",
                                lambda x: " ".join(sorted(set(" ".join(x).split()))[:12])))
               .reset_index())
        summary["dropped_protein_atoms"] = tot.to_dict("records")
    with open(os.path.join(out_dir, "fep_prepare_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=int)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark-dir", required=True, help="extracted public_binding_free_energy_benchmark root")
    ap.add_argument("--out-dir", required=True, help="where fep_benchmark.csv and the summaries go")
    ap.add_argument("--structures-dir", default=None, help="per-ligand SDF / per-series PDB dir "
                    "(default <out-dir>/structures)")
    ap.add_argument("--keep-unscored", action="store_true",
                    help="keep SDF titles that 21_4_results/ligand_predictions does not score")
    ap.add_argument("--atom-keys", default=os.path.join(os.environ["AEVPLIG_REPO"], "data", "PDB_Atom_Keys.csv"),
                    help="the clone's PDB_Atom_Keys.csv, for the dropped-residue census (skipped if absent)")
    args = ap.parse_args(argv)
    atom_keys = args.atom_keys if os.path.isfile(args.atom_keys) else None
    if atom_keys is None:
        print(f"WARNING: {args.atom_keys} not found; skipping the residue census")
    df = prepare_benchmark(args.benchmark_dir, args.out_dir, args.structures_dir, atom_keys,
                           scored_only=not args.keep_unscored)
    with open(os.path.join(os.path.abspath(args.out_dir), "fep_prepare_summary.json")) as fh:
        summary = json.load(fh)
    print(f"rows {len(df)}  series {summary['n_series']}  sdf entries {summary['n_sdf_entries']}  "
          f"duplicate-title entries {summary['n_duplicate_title_entries']}  "
          f"pK {summary['pK_min']:.3f}..{summary['pK_max']:.3f}")
    print(f"scored filter {summary['scored_filter']}: dropped {summary['n_unscored_dropped']} unscored titles")
    print(f"skipped {len(summary['skipped'])}: {summary['skipped']}")
    print(f"series without ref_pdb: {summary['series_without_ref_pdb']}")
    for r in summary.get("dropped_protein_atoms", []):
        print(f"dropped by featuriser: {r['record']:6s} {r['residue_name']:4s} series {r['n_series']:3d} "
              f"residues {r['n_residues']:6d} heavy atoms {r['n_dropped_atoms']:6d}  [{r['atom_names']}]")


if __name__ == "__main__":
    main()
