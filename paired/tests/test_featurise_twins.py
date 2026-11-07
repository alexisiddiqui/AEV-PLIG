"""Tests for src/featurise_twins.py on a synthetic parent layout (PDBbind mol2, BindingNet sdf, BindingDB
mol2), a synthetic zernike.npz and stubbed AEVs (torchani is not in the local venv)."""
import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen

import aevplig_featurise as aevf
import featurise_twins as ft
import twin_place
import twin_select as ts
from pose_augment import heavy_xyz, min_dist

AEV_WIDTH = 352
K = 49
ATOM_KEYS = ("PDB_ATOM,ATOM_TYPE\nALA-C,C;4;3;0;0;0\nALA-CA,C;4;3;1;0;0\nALA-CB,C;4;1;3;0;0\n"
             "ALA-N,N;3;2;1;0;0\nALA-O,O;2;1;0;0;0\n")
MOL2_TYPES = {"C": "C.3", "O": "O.3", "N": "N.3", "S": "S.3", "H": "H"}
FIXED_TEN = ["unique_id", "pdb", "kind", "k", "method", "split", "rmsd_to_crystal", "min_dist", "contact_frac",
             "n_within_3A"]
SCHEME_A = ["twin_unique_id", "twin_source", "shape_dist", "complementarity", "o3a_score", "n_heavy_anchor",
            "n_heavy_twin", "dlogp", "dcharge", "darom"]
COMPLEMENT = [1, 3, 2, 4, 6, 5]

# PDBbind complexes: pdb -> (smiles, split, refined, has protein)
PDBBIND = {"1abc": ("CCCCO", "train", True, True), "2def": ("OCCN(CC)CCO", "valid", False, True),
           "3ghi": ("CCOCC", "test", False, True), "4jkl": ("CCCCC", "train", True, False),
           "5mno": ("CC(C)(C)CCCCCCO", "train", True, True), "6pqr": ("CCCOC", "train", True, True)}
BN = {"CHEMBL1_7xyz_CHEMBL11": ("CCCCS", "7xyz", "CHEMBL1", "CHEMBL11"),   # file corrupted below
      "CHEMBL2_7xyz_CHEMBL22": ("CCCCN", "7xyz", "CHEMBL2", "CHEMBL22"),
      "CHEMBL3_8abc_CHEMBL33": ("OCCN(C)CCOC", "8abc", "CHEMBL3", "CHEMBL33"),
      "CHEMBL4_8abc_CHEMBL44": ("OCCC(O)CCCO", "8abc", "CHEMBL4", "CHEMBL44")}
BDB = {"FOLD_X_res_1": ("CCC(C)O", "FOLD_X", "FOLDX.pdb", "res_1.mol2")}
# designed selection: candidate uid -> (anchor whose shape it copies, cosine with that anchor's pocket complement)
DESIGN = {"CHEMBL1_7xyz_CHEMBL11": ("1abc", -1.0), "FOLD_X_res_1": ("1abc", -0.8), "6pqr": ("1abc", -0.6),
          "CHEMBL2_7xyz_CHEMBL22": ("1abc", -0.4),
          "CHEMBL3_8abc_CHEMBL33": ("2def", -0.9), "CHEMBL4_8abc_CHEMBL44": ("2def", -0.7)}
REJECT = {Chem.MolToSmiles(Chem.MolFromSmiles("OCCN(C)CCOC")): "clash"}  # fake placement rejects this twin
LAYOUT_LIMIT = ["--limit", "4"]  # anchors 1abc, 2def, 4jkl, 5mno (6pqr, a pool row, is a later anchor)


def zero_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map):
    mol_df = aevf.LoadMolasDF(mol)
    return mol_df, np.zeros((len(mol_df), AEV_WIDTH))


def coord_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map):
    mol_df = aevf.LoadMolasDF(mol)
    vals = mol_df[["X", "Y", "Z"]].to_numpy(dtype=float).sum(axis=1, keepdims=True)
    return mol_df, np.repeat(vals, AEV_WIDTH, axis=1) + np.arange(AEV_WIDTH)


def embedded(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def write_mol2(mol, path):
    conf = mol.GetConformer()
    lines = ["@<TRIPOS>MOLECULE", "LIG", f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0", "SMALL",
             "NO_CHARGES", "", "@<TRIPOS>ATOM"]
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetIdx() + 1:7d} {a.GetSymbol()}{a.GetIdx() + 1:<5d} {p.x:10.4f} {p.y:10.4f} "
                     f"{p.z:10.4f} {MOL2_TYPES[a.GetSymbol()]:<5s} 1 LIG1 0.0000")
    lines.append("@<TRIPOS>BOND")
    for b in mol.GetBonds():
        lines.append(f"{b.GetIdx() + 1:6d} {b.GetBeginAtomIdx() + 1:5d} {b.GetEndAtomIdx() + 1:5d} 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_sdf(mol, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()


def write_protein(path, center, n=60, seed=0):
    rng = np.random.default_rng(seed)
    names = ["N", "CA", "C", "O", "CB"]
    lines = []
    for i in range(n):
        v = rng.normal(size=3)
        x, y, z = center + v / np.linalg.norm(v) * rng.uniform(5.5, 9.0)
        name = names[i % 5]
        lines.append(f"ATOM  {i + 1:5d}  {name:<3s} ALA A{i // 5 + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00           {name[0]}")
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def chem_cos(comp, c, rng):
    v = comp.ravel() / np.linalg.norm(comp)
    o = rng.normal(size=v.size)
    o -= (o @ v) * v
    o /= np.linalg.norm(o)
    return ((c * v + np.sqrt(max(0.0, 1.0 - c * c)) * o) * np.linalg.norm(comp)).reshape(comp.shape)


def write_zernike(path, mols, meta, seed=0):
    """Synthetic descriptors: random background rows (too big to pass the size filter) set the
    standardisation; designed rows copy an anchor's shape with a set complementarity."""
    rng = np.random.default_rng(seed)
    uids = list(meta["unique_id"])
    n = len(uids)
    lf = rng.normal(size=(n, 7, K))
    pf = rng.normal(size=(n, 7, K))
    lt = np.zeros((n, 7))
    nh = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(uids):
        m = mols.get(u)
        if m is None:  # background row
            nh[i] = 400
            lt[i] = [400, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        else:
            nh[i] = sum(a.GetSymbol() != "H" for a in m.GetAtoms())
            lt[i] = [nh[i], Crippen.MolLogP(m), 1.0, 1.0, 0.0, 1.0, 1.0]
    for u, (anchor, c) in DESIGN.items():
        i, a = uids.index(u), uids.index(anchor)
        lf[i, 0] = lf[a, 0] + rng.normal(scale=0.02, size=K)
        lf[i, 1:] = chem_cos(pf[a][COMPLEMENT], c, rng)
    lf[uids.index("5mno"), 0] = lf[uids.index("1abc"), 0]  # same shape as 1abc but 11 heavy atoms: no twin
    lf[uids.index("4jkl"), 0] = lf[uids.index("1abc"), 0] + rng.normal(scale=0.02, size=K)  # has candidates
    np.savez_compressed(path, unique_id=np.array(uids), source=meta["source"].to_numpy(dtype=str),
                        ok=np.ones(n, dtype=bool), error=np.array([""] * n), ligand_F=lf.astype(np.float32),
                        pocket_F=pf.astype(np.float32), ligand_total=lt.astype(np.float32),
                        pocket_total=np.zeros((n, 7), np.float32), n_ligand_atoms=nh)


@pytest.fixture
def layout(tmp_path):
    mols = {}
    for i, (pdb, (smi, split, refined, has_protein)) in enumerate(PDBBIND.items()):
        mol = embedded(smi, i + 1)
        mols[pdb] = mol
        d = tmp_path / "pdbbind" / ("refined-set" if refined else "general-set") / pdb
        write_mol2(mol, d / f"{pdb}_ligand.mol2")
        if has_protein:
            write_protein(d / f"{pdb}_protein.pdb", heavy_xyz(mol).mean(axis=0), seed=i)
    for i, (uid, (smi, pdb, target, compnd)) in enumerate(BN.items()):
        mol = embedded(smi, 20 + i)
        mols[uid] = mol
        path = tmp_path / "bn" / pdb / f"target_{target}" / compnd / f"{pdb}_{target}_{compnd}.sdf"
        write_sdf(mol, path)
        if uid == "CHEMBL1_7xyz_CHEMBL11":
            path.write_text("not an sdf file\n")
    for i, (uid, (smi, folder, pdb_file, mol2_file)) in enumerate(BDB.items()):
        mol = embedded(smi, 40 + i)
        mols[uid] = mol
        write_mol2(mol, tmp_path / "surflex" / folder / mol2_file)
    rows = [{"unique_id": p, "source": "pdbbind", "split": s, "lig_key": f"pdbbind:L{p}", "uniprot": "",
             "anchor_pdb": p} for p, (_, s, _, _) in PDBBIND.items()]
    rows += [{"unique_id": u, "source": "bindingnet", "split": "train", "lig_key": f"bindingnet:{c}",
              "uniprot": "", "anchor_pdb": pdb} for u, (_, pdb, _, c) in BN.items()]
    rows += [{"unique_id": u, "source": "bindingdb", "split": "train", "lig_key": f"bindingdb:{u[-1]}",
              "uniprot": "", "anchor_pdb": "foldx"} for u in BDB]
    rows += [{"unique_id": f"bg{i}", "source": "bindingnet", "split": "train", "lig_key": f"bindingnet:bg{i}",
              "uniprot": "", "anchor_pdb": f"t{i}"} for i in range(80)]
    meta = pd.DataFrame(rows)
    meta.insert(0, "row", range(len(meta)))
    meta.to_csv(tmp_path / "paired_meta.csv", index=False)
    groups = pd.DataFrame({"unique_id": meta["unique_id"],
                           "group_id": [f"g_{u}" for u in meta["unique_id"]]})
    groups.to_csv(tmp_path / "pocket_groups.csv", index=False)
    write_zernike(tmp_path / "zernike.npz", mols, meta)
    parent = tmp_path / "parent"
    parent.mkdir()
    pd.DataFrame({"PDB_code": list(PDBBIND), "-logKd/Ki": 6.0,
                  "refined": [v[2] for v in PDBBIND.values()]}).to_csv(parent / "pdbbind_processed.csv")
    pd.DataFrame({"unique_identify": list(BN), "-logAffi": 6.0, "target": [v[2] for v in BN.values()],
                  "pdb": [v[1] for v in BN.values()], "compnd": [v[3] for v in BN.values()]}).to_csv(
        parent / "bindingnet_processed.csv")
    pd.DataFrame({"folder": [v[1] for v in BDB.values()], "pdb_file": [v[2] for v in BDB.values()],
                  "mol2_file": [v[3] for v in BDB.values()], "unique_id": list(BDB)}).to_csv(
        parent / "bindingdb_processed.csv")
    (tmp_path / "keys.csv").write_text(ATOM_KEYS)
    return tmp_path


def base_args(t):
    return ["--zernike", str(t / "zernike.npz"), "--meta", str(t / "paired_meta.csv"),
            "--groups", str(t / "pocket_groups.csv"), "--pdbbind-root", str(t / "pdbbind"),
            "--bindingnet-root", str(t / "bn"), "--bindingdb-root", str(t / "surflex"),
            "--parent-data", str(t / "parent"), "--out-dir", str(t / "out"), "--workers", "1",
            "--atom-keys", str(t / "keys.csv")]


def shifted(mol, dx):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, (p + [dx, 0.0, 0.0]).tolist())
    return out


def fake_placement(calls):
    def smiles(m):
        return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False)

    def place_twin(anchor, twin, target, rng, **kw):
        assert kw == {"clash": twin_place.CLASH, "relief_steps_max": twin_place.RELIEF_STEPS_MAX}
        calls.append(("twin", smiles(anchor), smiles(twin), rng.bit_generator.state["state"]["state"]))
        reason = REJECT.get(smiles(twin), "")
        if reason:
            return twin_place.Placement(None, False, reason, 50.0, 1.5, 1.0, 3, 3, 300, 0.4)
        rng.random()
        return twin_place.Placement(shifted(twin, 0.3), True, "", 55.0, 3.1, 0.9, 1, 1, 12, 0.2)

    def regenerate_positive(anchor, rng, target, **kw):
        assert kw == {"clash": twin_place.CLASH, "relief_steps_max": twin_place.RELIEF_STEPS_MAX}
        calls.append(("pos", smiles(anchor), smiles(anchor), rng.bit_generator.state["state"]["state"]))
        return twin_place.Placement(shifted(anchor, 0.2), True, "", 60.0, 3.3, 1.1, 0, 2, 0, 0.1, 0.42)
    return place_twin, regenerate_positive


def test_featurise_twins_layout(layout, monkeypatch):
    calls = []
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", coord_aevs)
    place, regen = fake_placement(calls)
    monkeypatch.setattr(twin_place, "place_twin", place)
    monkeypatch.setattr(twin_place, "regenerate_positive", regen)
    ft.main(base_args(layout) + LAYOUT_LIMIT)
    out = layout / "out"
    with open(out / "twins.pickle", "rb") as f:
        graphs = pickle.load(f)
    expected = {"1abc::tpos0", "1abc::twin0", "1abc::twin1", "2def::tpos0", "2def::twin0"}
    assert set(graphs) == expected
    assert all(len(g) == 4 and len(g[1][0]) == 15 + AEV_WIDTH for g in graphs.values())
    assert graphs["1abc::twin0"][0] == 5 and graphs["2def::tpos0"][0] == 9     # heavy atoms of the posed ligand
    assert not np.array_equal(graphs["1abc::tpos0"][1], graphs["1abc::twin0"][1])
    meta = pd.read_csv(out / "twins_meta.csv", dtype={"pdb": str})
    assert list(meta.columns[:10]) == FIXED_TEN and list(meta.columns[10:20]) == SCHEME_A
    assert list(meta.columns) == ft.META_COLUMNS and set(meta["unique_id"]) == expected
    m = meta.set_index("unique_id")
    # 1abc: the corrupted sdf (rank 0) is skipped, ranks 1 and 2 are accepted, rank 3 is never tried
    assert (m.loc["1abc::twin0", "twin_unique_id"], m.loc["1abc::twin0", "twin_source"]) == ("FOLD_X_res_1", "bindingdb")
    assert (m.loc["1abc::twin1", "twin_unique_id"], m.loc["1abc::twin1", "twin_source"]) == ("6pqr", "pdbbind")
    assert list(m.loc[["1abc::twin0", "1abc::twin1"], "cand_rank"]) == [1, 2]
    # 2def (valid): its first candidate is rejected by the placement, the second is the only twin
    assert m.loc["2def::twin0", "twin_unique_id"] == "CHEMBL4_8abc_CHEMBL44"
    assert m.loc["2def::twin0", "split"] == "valid" and m.loc["2def::twin0", "k"] == 0
    assert (meta["method"] == "o3a").all()
    pos = meta[meta["kind"] == 3]
    tw = meta[meta["kind"] == 4]
    assert set(pos["unique_id"]) == {"1abc::tpos0", "2def::tpos0"} and (pos["k"] == 0).all()
    assert (pos["rmsd_to_crystal"] == 0.42).all() and tw["rmsd_to_crystal"].isna().all()
    assert (pos["twin_unique_id"] == pos["pdb"]).all() and (pos["shape_dist"] == 0).all()
    assert (pos["dlogp"] == 0).all() and (pos["cand_rank"] == -1).all()
    assert (tw["min_dist"] == 3.1).all() and (tw["relief_steps"] == 12).all() and (pos["min_dist"] == 3.3).all()
    assert (m.loc["1abc::tpos0", "n_heavy_anchor"], m.loc["1abc::tpos0", "n_heavy_twin"]) == (5, 5)
    assert m.loc["1abc::twin0", "twin_group_id"] == "g_FOLD_X_res_1"
    assert m.loc["1abc::twin0", "twin_lig_key"] == "bindingdb:1"
    # selection fields agree with twin_select on the same inputs
    pool = ts.load_pool(str(layout / "zernike.npz"), str(layout / "paired_meta.csv"),
                        str(layout / "pocket_groups.csv"))
    sel = {c.twin_uid: c for c in ts.select_twins("1abc", pool, k=6)}
    for uid in ("1abc::twin0", "1abc::twin1"):
        c = sel[m.loc[uid, "twin_unique_id"]]
        assert m.loc[uid, "shape_dist"] == pytest.approx(c.shape_dist, abs=1e-6)
        assert m.loc[uid, "complementarity"] == pytest.approx(c.complementarity, abs=1e-6)
        assert m.loc[uid, "dlogp"] == pytest.approx(c.dlogp, abs=1e-6) and m.loc[uid, "dcharge"] == c.dcharge
    _, info = ts.select_twins_with_info("1abc", pool, k=6)
    assert m.loc["1abc::tpos0", "complementarity"] == pytest.approx(info["anchor_complementarity"], abs=1e-6)
    assert m.loc["1abc::tpos0", "complementarity"] > m.loc["1abc::twin0", "complementarity"]
    # one rng per anchor, crc32-seeded, shared by its twins and then its positive (twins first)
    first = {}
    for kind, anchor_smi, _, state in calls:
        first.setdefault(anchor_smi, (kind, state))
    for pdb in ("1abc", "2def"):
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(PDBBIND[pdb][0]), isomericSmiles=False)
        assert first[smi] == ("twin", np.random.default_rng(zlib.crc32(pdb.encode())).bit_generator.state["state"]["state"])
    assert [c[0] for c in calls].count("pos") == 2
    summary = json.loads((out / "twins_summary.json").read_text())
    assert summary["n_anchors"] == 4 and summary["n_anchors_ok"] == 3   # 1abc, 2def, 4jkl, 5mno; 3ghi is test
    assert summary["anchors_by_split"] == {"train": 3, "valid": 1}
    assert summary["counts_kind"] == {"3": 2, "4": 3}
    assert summary["counts_split_kind"]["valid"] == {"3": 1, "4": 1}
    assert summary["anchors_by_n_twins"] == {"2": 1, "1": 1, "0": 2}
    assert summary["anchor_status"] == {"error": 1, "ok": 1, "selection:no_twin": 1, "short": 1}
    co = summary["candidate_outcomes"]
    assert (co["accepted"], co["unreadable"], co["clash"]) == (3, 1, 1) and co["not_tried"] >= 1
    assert summary["selection_reasons"]["no_twin"] == 1
    assert summary["selection_no_candidate_empty_at"] == {"size": 1}
    assert summary["twin_source_counts"] == {"bindingdb": 1, "bindingnet": 1, "pdbbind": 1}
    assert summary["rmsd_to_crystal_quantiles"]["0.5"] == pytest.approx(0.42)
    cue = summary["cue_auroc_pos_vs_twin"]
    assert set(cue) == {"min_dist", "n_within_3A", "n_heavy", "logp"}
    assert cue["min_dist"] == 1.0 and cue["n_within_3A"] == 0.0   # fake poses: 3.3 vs 3.1, 0 vs 1
    assert set(summary["quantiles"]) == {"3", "4"}
    assert summary["quantiles"]["4"]["o3a_score"]["0.5"] == 55.0
    assert summary["twins_vs_anchors"]["dcharge"] == {"0": 3}
    assert summary["errors"][0][0] == "4jkl" and "FileNotFoundError" in summary["errors"][0][1]
    assert ["5mno", "train", "selection:no_twin", 0] in summary["failure_anchors"]


def test_limit_and_no_lonely_positive(layout, monkeypatch):
    calls = []
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", zero_aevs)
    place, regen = fake_placement(calls)
    monkeypatch.setattr(twin_place, "regenerate_positive", regen)
    # every twin rejected: no anchor writes anything, and its positive is never generated
    kws = []

    def reject(*a, **k):
        kws.append(k)
        return twin_place.Placement(None, False, "contacts", 40.0, 3.0, 0.1, 0, 3, 5, 0.3)

    monkeypatch.setattr(twin_place, "place_twin", reject)
    ft.main(base_args(layout) + ["--limit", "2", "--clash", "3.0", "--relief-steps", "1000"])
    assert kws and all(k == {"clash": 3.0, "relief_steps_max": 1000} for k in kws)
    out = layout / "out"
    summary = json.loads((out / "twins_summary.json").read_text())
    assert summary["n_anchors"] == 2 and summary["anchors_by_split"] == {"train": 1, "valid": 1}
    assert summary["n_poses"] == 0 and summary["anchors_by_n_twins"] == {"2": 0, "1": 0, "0": 2}
    assert summary["anchor_status"] == {"no_twin_placed": 2}
    assert summary["place_params"]["clash"] == 3.0 and summary["place_params"]["relief_steps_max"] == 1000
    assert summary["candidate_outcomes"]["contacts"] >= 2
    assert not calls  # no regenerated positive without a twin
    assert pickle.loads((out / "twins.pickle").read_bytes()) == {}
    assert list(pd.read_csv(out / "twins_meta.csv").columns) == ft.META_COLUMNS


def test_real_placement_end_to_end(layout, monkeypatch):
    monkeypatch.setattr(aevf, "GetMolAEVs_extended", zero_aevs)
    ft.main(base_args(layout) + ["--limit", "2"])
    out = layout / "out"
    meta = pd.read_csv(out / "twins_meta.csv", dtype={"pdb": str})
    graphs = pickle.loads((out / "twins.pickle").read_bytes())
    assert set(graphs) == set(meta["unique_id"])
    summary = json.loads((out / "twins_summary.json").read_text())
    assert summary["n_anchors_ok"] == 2 and not summary["errors"]
    assert (meta["contact_frac"] >= twin_place.CONTACT_KEEP).all()
    keys, _ = aevf.load_atom_keys(str(layout / "keys.csv"))
    for pdb, sub in meta.groupby("pdb"):
        folder = layout / "pdbbind" / ("refined-set" if PDBBIND[pdb][2] else "general-set") / pdb
        target = aevf.LoadPDBasDF(str(folder / f"{pdb}_protein.pdb"), keys)[["X", "Y", "Z"]].to_numpy(float)
        crystal = ft.load_structure(("mol2", str(folder / f"{pdb}_ligand.mol2")))
        clash_eff = min(twin_place.CLASH, min_dist(heavy_xyz(crystal), target))
        assert (sub["min_dist"] >= clash_eff - 1e-9).all()
        assert (sub["n_within_3A"] >= 0).all()
    pos = meta[meta["kind"] == 3]
    assert set(pos["pdb"]) <= set(meta.loc[meta["kind"] == 4, "pdb"])   # a positive only beside a twin
    assert (pos["rmsd_to_crystal"] < 2.0).all()
    assert (meta["o3a_score"] > 0).all()
    for _, r in meta.iterrows():
        assert graphs[r["unique_id"]][0] == r["n_heavy_twin"]


def test_parse_args_guards(layout):
    with pytest.raises(SystemExit):
        ft.parse_args(base_args(layout) + ["--k", "3", "--n-candidates", "2"])
    args = ft.parse_args(base_args(layout))
    assert (args.k, args.n_candidates, args.shape_frac, args.workers) == (2, 6, 0.01, 1)


def test_ligand_spec_paths(layout):
    tables = ft.read_tables(str(layout / "parent"))
    roots = {"pdbbind": str(layout / "pdbbind"), "bindingnet": str(layout / "bn"),
             "bindingdb": str(layout / "surflex")}
    assert ft.ligand_spec("2def", "pdbbind", tables, roots) == \
        ("mol2", os.path.join(roots["pdbbind"], "general-set", "2def", "2def_ligand.mol2"))
    assert ft.ligand_spec("CHEMBL2_7xyz_CHEMBL22", "bindingnet", tables, roots) == \
        ("sdf_first", os.path.join(roots["bindingnet"], "7xyz", "target_CHEMBL2", "CHEMBL22",
                                   "7xyz_CHEMBL2_CHEMBL22.sdf"))
    assert ft.ligand_spec("FOLD_X_res_1", "bindingdb", tables, roots) == \
        ("mol2", os.path.join(roots["bindingdb"], "FOLD_X", "res_1.mol2"))
    with pytest.raises(KeyError):
        ft.ligand_spec("nope", "bindingnet", tables, roots)
    sdf = ft.load_structure(ft.ligand_spec("CHEMBL2_7xyz_CHEMBL22", "bindingnet", tables, roots))
    assert all(a.GetNumImplicitHs() == 0 for a in sdf.GetAtoms())  # every H explicit, with coordinates
    assert sdf.GetNumAtoms() == Chem.AddHs(Chem.MolFromSmiles("CCCCN")).GetNumAtoms()
