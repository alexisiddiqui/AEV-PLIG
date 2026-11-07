"""Build a small stand-in for the repo's example set (data/example_dataset.csv lists the 285 CASF-2016 core
complexes as data/example_database/<pdb>.{sdf,pdb}, but those structure files are not in the repo).

Takes N entries of example_dataset.csv spread over the pK range, fetches each from RCSB (PDB-format coordinates
and the ligand instance as SDF from ModelServer), fixes bond orders against the CCD SMILES, adds hydrogens with
coordinates (as generate_pdbbind_graphs.py does for PDBbind mol2 ligands) and writes
  data/example_rcsb/<pdb>.{pdb,sdf}   and   data/example_rcsb.csv  (unique_id,pK,sdf_file,pdb_file)
Run from $WORK:  python $JOBS/fetch_example.py [N]
The ligand is the heaviest non-solvent, non-ion non-polymer entity, so it may differ from PDBbind's choice;
PDBbind's pocket files and protonation are not reproduced. This is an env check, not a CASF benchmark.
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
SKIP = {"HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "PEG", "MPD", "CL", "NA", "K", "MG", "ZN", "CA", "MN",
        "FE", "CU", "NI", "CO", "CD", "IOD", "BR", "NO3", "FMT", "TRS", "EPE", "MES", "BME", "PG4", "1PE", "P6G",
        "IMD", "NH4", "SCN", "CIT", "TLA", "MLI", "ACY", "IPA", "EOH", "MOH", "BU3", "HEZ", "PGE", "NAG",
        "BOG", "LDA", "C8E", "SOG", "LMT", "OLC", "PLM", "MYR", "HTO", "CPS", "DTT", "B3P", "MRD", "TAR", "SUC"}
ALLOWED = {"F", "N", "Cl", "O", "Br", "C", "B", "P", "I", "S", "H"}


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2)


def ligand_choice(pdb):
    entry = json.loads(get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}"))
    best = None
    for eid in entry["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids", []) or []:
        ne = json.loads(get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb}/{eid}"))
        comp = ne["pdbx_entity_nonpoly"]["comp_id"]
        if comp in SKIP:
            continue
        cc = json.loads(get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp}"))
        fw = cc["chem_comp"].get("formula_weight") or 0
        desc = {k.lower(): v for k, v in (cc.get("rcsb_chem_comp_descriptor") or {}).items()}
        smi = desc.get("smiles_stereo") or desc.get("smiles")
        asym = ne["rcsb_nonpolymer_entity_container_identifiers"]["asym_ids"][0]
        if fw > 150 and (best is None or fw > best[0]):
            best = (fw, comp, asym, smi)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    src = pd.read_csv("data/example_dataset.csv").sort_values("pK").reset_index(drop=True)
    # n entries evenly spread over the pK range first, then the midpoints between them as fallbacks
    idx = [round(i * (len(src) - 1) / (2 * n - 1)) for i in range(2 * n)]
    picks = src.iloc[idx[0::2] + idx[1::2]].drop_duplicates("unique_id")
    os.makedirs("data/example_rcsb", exist_ok=True)
    rows = []
    for _, r in picks.iterrows():
        if len(rows) >= n:
            break
        pdb = r["unique_id"]
        try:
            choice = ligand_choice(pdb)
            if choice is None:
                print(f"{pdb}: no small-molecule ligand, skipped")
                continue
            fw, comp, asym, smi = choice
            pdb_txt = get(f"https://files.rcsb.org/download/{pdb}.pdb").decode()
            sdf = get(f"https://models.rcsb.org/v1/{pdb}/ligand?label_asym_id={asym}&encoding=sdf").decode()
            mol = Chem.MolFromMolBlock(sdf, removeHs=True)  # sanitized, with ModelServer's bond orders
            if mol is None:
                print(f"{pdb}: ModelServer SDF for {comp} not readable by RDKit, skipped")
                continue
            try:  # bond orders and charges from the CCD SMILES where the heavy-atom graph matches
                mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smi), mol)
                how = "CCD template"
            except Exception as e:
                how = f"ModelServer bonds; CCD template failed: {str(e)[:60]}"
            mol = Chem.AddHs(mol, addCoords=True)
            if not {a.GetSymbol() for a in mol.GetAtoms()} <= ALLOWED:
                print(f"{pdb}: {comp} has rare elements, skipped")
                continue
            with open(f"data/example_rcsb/{pdb}.pdb", "w") as f:
                f.write(pdb_txt)
            w = Chem.SDWriter(f"data/example_rcsb/{pdb}.sdf")
            w.write(mol)
            w.close()
            rows.append(dict(unique_id=pdb, pK=r["pK"], sdf_file=f"data/example_rcsb/{pdb}.sdf",
                             pdb_file=f"data/example_rcsb/{pdb}.pdb", ligand=comp, asym=asym))
            print(f"{pdb}: pK {r['pK']} ligand {comp} chain {asym} ({mol.GetNumHeavyAtoms()} heavy atoms, {how})")
        except Exception as e:
            print(f"{pdb}: failed ({type(e).__name__}: {e}), skipped")
    pd.DataFrame(rows).to_csv("data/example_rcsb.csv", index=False)
    print(f"wrote data/example_rcsb.csv with {len(rows)} complexes")


if __name__ == "__main__":
    main()
