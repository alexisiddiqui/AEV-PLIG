"""Tests for src/build_paired_dataset.py on small synthetic PDBbind/BindingNet/BindingDB inputs."""
import io
import json
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
import pytest

from build_paired_dataset import build_meta, main, read_pdbbind_index

META_COLUMNS = ["row", "unique_id", "source", "split", "pK", "series_key", "series_id",
                "lig_key", "lig_id", "label_type", "uniprot", "anchor_pdb", "kind", "parent"]

# pdb -> uniprot as in INDEX_general_PL_name.2020 (6pqr, 8vwx, 1hp5 absent on purpose)
NAME_INDEX = {"1abc": "P1", "2def": "------", "3ghi": "P1", "4jkl": "------", "5mno": "P2",
              "7stu": "P1", "1swr": "P3", "4dbu": "P4", "1xyz": "------"}
# pdb -> (affinity string, ligand name inside the parentheses) as in INDEX_general_PL_data.2020
DATA_INDEX = {"1abc": ("Kd=10uM", "ABC"), "2def": ("Ki=1uM", "XYZ"), "3ghi": ("IC50=100nM", "5-mer"),
              "4jkl": ("Kd>50mM", "FMN hq"), "5mno": ("Ki=100uM", "QQQ"), "6pqr": ("IC50=5uM", ""),
              "7stu": ("Ki~2nM", "5-mer")}


def make_frames():
    pdbbind = pd.DataFrame({
        "PDB_code": ["1abc", "2def", "3ghi", "4jkl", "5mno", "6pqr", "7stu", "8vwx"],
        "-logKd/Ki": [5.0, 6.0, 7.0, 8.0, 4.0, 5.3, 8.7, 6.6],
        "refined": [False] * 8,
        "core": [False] * 8,
        "split_core": ["train", "train", "valid", "test", "train", "train", "valid", "train"],
        "max_tanimoto_fep_benchmark": [0.3, 0.4, 0.5, 0.2, 0.95, 0.1, 0.1, 0.89],
    })
    bn_target = ["T1", "T2", "T1", "T1", "T1", "T1", "T1"] + ["T3"] * 6
    bn_pdb = ["1swr"] * 7 + ["1hp5"] * 6
    bn_cmp = ["C1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]
    bindingnet = pd.DataFrame({
        "unique_identify": [f"{t}_{p}_{c}" for t, p, c in zip(bn_target, bn_pdb, bn_cmp)],
        "-logAffi": np.linspace(4.0, 9.0, 13),
        "target": bn_target,
        "pdb": bn_pdb,
        "compnd": bn_cmp,
        "max_tanimoto_fep_benchmark": [0.2] * 12 + [0.92],
    })
    folders = ["4DBU_BT9"] * 5 + ["1XYZ_ABC"] * 2
    mol_ids = [100, 101, 102, 103, 104, 200, 201]
    bindingdb = pd.DataFrame({
        "folder": folders,
        "pdb_file": [f.split("_")[0] + ".pdb" for f in folders],
        "mol2_file": [f"{f.split('_')[0]}-results_{m}.mol2" for f, m in zip(folders, mol_ids)],
        "Ki(nM)": ["12", None, None, "3", None, None, None],
        "IC50(nM)": [None, "40", "50", "7", "1500;1520", None, "9"],
        "Kd(nM)": [None, None, "5", None, None, "8", None],
        "unique_id": [f"{f}_{f.split('_')[0]}-results_{m}" for f, m in zip(folders, mol_ids)],
        "pK": [7.9, 7.4, 8.3, 8.5, 5.8, 8.1, 8.0],
        "max_tanimoto_fep_benchmark": [0.3] * 6 + [0.9],
    })
    return pdbbind, bindingnet, bindingdb


def meta_for(holdout_frac=0.5, holdout_min_size=5, seed=0, **kw):
    pdbbind, bindingnet, bindingdb = make_frames()
    return build_meta(pdbbind, bindingnet, bindingdb, NAME_INDEX, DATA_INDEX,
                      holdout_frac=holdout_frac, holdout_min_size=holdout_min_size, seed=seed, **kw)


def by_id(meta):
    return meta.set_index("unique_id")


def test_keys_follow_rules():
    meta = meta_for(holdout_frac=0.0)
    assert list(meta.columns) == META_COLUMNS
    assert (meta.kind == 0).all() and (meta.parent == meta.row).all()
    m = by_id(meta)
    # PDBbind
    assert m.loc["1abc", ["series_key", "lig_key", "label_type", "uniprot", "anchor_pdb"]].tolist() == \
        ["pdbbind:P1", "pdbbind:ABC", "Kd", "P1", "1abc"]
    assert m.loc["3ghi", ["series_key", "lig_key", "label_type"]].tolist() == \
        ["pdbbind:P1", "pdbbind:single:3ghi", "IC50"]
    assert m.loc["4jkl", ["series_key", "lig_key", "label_type", "uniprot"]].tolist() == \
        ["pdbbind:single:4jkl", "pdbbind:FMN", "Kd", ""]
    assert m.loc["6pqr", ["series_key", "lig_key", "label_type", "uniprot"]].tolist() == \
        ["pdbbind:single:6pqr", "pdbbind:single:6pqr", "IC50", ""]
    assert m.loc["7stu", ["series_key", "lig_key", "label_type"]].tolist() == \
        ["pdbbind:P1", "pdbbind:single:7stu", "Ki"]
    assert m.loc["8vwx", ["series_key", "lig_key", "label_type"]].tolist() == \
        ["pdbbind:single:8vwx", "pdbbind:single:8vwx", "unknown"]
    assert (m.loc[m.source == "pdbbind", "pK"].tolist()) == [5.0, 6.0, 7.0, 8.0, 5.3, 8.7, 6.6]
    # BindingNet: template series, compound ligand, the same compound under two targets shares lig
    bn = m[m.source == "bindingnet"]
    assert set(bn.series_key) == {"bindingnet:1swr", "bindingnet:1hp5"}
    assert m.loc["T1_1swr_C1", ["series_key", "lig_key", "label_type", "uniprot", "anchor_pdb"]].tolist() == \
        ["bindingnet:1swr", "bindingnet:C1", "unknown", "P3", "1swr"]
    assert m.loc["T1_1swr_C1", "lig_id"] == m.loc["T2_1swr_C1", "lig_id"]
    assert (bn.loc[bn.anchor_pdb == "1hp5", "uniprot"] == "").all()
    # BindingDB: folder series, BindingDB monomer id ligand, first non-empty of Kd, Ki, IC50
    bdb = m[m.source == "bindingdb"]
    assert bdb.series_key.tolist() == ["bindingdb:4DBU_BT9"] * 5 + ["bindingdb:1XYZ_ABC"]
    assert bdb.lig_key.tolist() == [f"bindingdb:{i}" for i in (100, 101, 102, 103, 104, 200)]
    assert bdb.label_type.tolist() == ["Ki", "IC50", "Kd", "Ki", "IC50", "Kd"]
    assert bdb.anchor_pdb.tolist() == ["4dbu"] * 5 + ["1xyz"]
    assert bdb.uniprot.tolist() == ["P4"] * 5 + [""]


def test_pdbbind_unmapped_rows_are_singletons():
    meta = meta_for(holdout_frac=0.0)
    m = by_id(meta)
    unmapped = ["2def", "4jkl", "6pqr", "8vwx"]
    sids = m.loc[unmapped, "series_id"]
    assert sids.nunique() == len(unmapped)
    for s in sids:
        assert (meta.series_id == s).sum() == 1
    # peptide / unnamed ligands: own lig id each, even inside a shared series
    peptide = ["3ghi", "7stu", "6pqr", "8vwx"]
    lids = m.loc[peptide, "lig_id"]
    assert lids.nunique() == len(peptide)
    for lg in lids:
        assert (meta.lig_id == lg).sum() == 1
    assert m.loc["3ghi", "series_id"] == m.loc["1abc", "series_id"] == m.loc["7stu", "series_id"]


def test_filter_matches_original():
    """Same rows, order, pK and split as the clone's create_pytorch_data.py (no holdout)."""
    pdbbind, bindingnet, bindingdb = make_frames()
    p = pdbbind.rename(columns={"PDB_code": "unique_id", "split_core": "split", "-logKd/Ki": "pK"})
    p = p[p["max_tanimoto_fep_benchmark"] < 0.9][["unique_id", "pK", "split"]]
    n = bindingnet.rename(columns={"-logAffi": "pK", "unique_identify": "unique_id"})
    n = n[n["max_tanimoto_fep_benchmark"] < 0.9][["unique_id", "pK"]].assign(split="train")
    d = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9][["unique_id", "pK"]].assign(split="train")
    original = pd.concat([p, n, d], ignore_index=True)

    meta = meta_for(holdout_frac=0.0)
    assert meta.unique_id.tolist() == original.unique_id.tolist()
    assert meta.split.tolist() == original.split.tolist()
    assert np.allclose(meta.pK.to_numpy(), original.pK.to_numpy())
    assert meta.row.tolist() == list(range(len(meta)))
    for dropped in ("5mno", "T3_1hp5_C12", "1XYZ_ABC_1XYZ-results_201"):
        assert dropped not in set(meta.unique_id)
    assert "8vwx" in set(meta.unique_id)  # 0.89 < 0.9 is kept


def test_holdout_moves_whole_series():
    meta = meta_for(holdout_frac=0.5, holdout_min_size=5, seed=0)
    pair_src = meta.source.isin(["bindingnet", "bindingdb"])
    held = meta[meta.split == "series_valid"]
    assert set(held.source) <= {"bindingnet", "bindingdb"}
    held_series = set(held.series_key)
    # BindingNet: 2 eligible templates -> round(0.5 * 2) = 1; BindingDB: 1 eligible -> round(0.5) = 0
    assert len(held_series) == 1 and next(iter(held_series)).startswith("bindingnet:")
    in_held = meta.series_key.isin(held_series)
    assert (meta.loc[in_held, "split"] == "series_valid").all()
    assert (meta.loc[pair_src & ~in_held, "split"] == "train").all()
    assert meta.loc[meta.source == "pdbbind", "split"].tolist() == \
        ["train", "train", "valid", "test", "train", "valid", "train"]
    # deterministic in the seed
    assert meta.equals(meta_for(holdout_frac=0.5, holdout_min_size=5, seed=0))
    # every eligible series held out at frac 1; the 1-row folder stays in train
    full = meta_for(holdout_frac=1.0, holdout_min_size=5, seed=0)
    assert set(full.loc[full.split == "series_valid", "series_key"]) == \
        {"bindingnet:1swr", "bindingnet:1hp5", "bindingdb:4DBU_BT9"}
    assert full.loc[full.series_key == "bindingdb:1XYZ_ABC", "split"].tolist() == ["train"]


def test_series_ids_dense():
    meta = meta_for()
    for key, code in (("series_key", "series_id"), ("lig_key", "lig_id")):
        assert sorted(meta[code].unique()) == list(range(meta[key].nunique()))
        pairs = meta[[key, code]].drop_duplicates()
        assert pairs[key].is_unique and pairs[code].is_unique
        ordered = pairs.sort_values(code)[key].tolist()
        assert ordered == sorted(ordered)
        assert meta[code].dtype.kind == "i"


def test_subset_series_keeps_first_series_and_all_eval_rows():
    meta = meta_for(holdout_frac=0.0, subset_series=1)
    m = by_id(meta)
    assert set(m.index[m.source == "pdbbind"]) == {"1abc", "3ghi", "7stu", "4jkl"}
    assert set(m.loc[m.source == "bindingnet", "series_key"]) == {"bindingnet:1hp5"}
    assert set(m.loc[m.source == "bindingdb", "series_key"]) == {"bindingdb:1XYZ_ABC"}
    assert len(meta) == 4 + 5 + 1
    assert sorted(meta.series_id.unique()) == list(range(meta.series_key.nunique()))
    assert meta.row.tolist() == list(range(len(meta)))


def _index_text():
    name = ["# ====", "# PDB code, release year, Uniprot ID, protein name", "# ===="]
    name += [f"{pdb}  2010  {uni}  SOME PROTEIN NAME" for pdb, uni in NAME_INDEX.items()]
    data = ["# ====", "# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name"]
    data += [f"{pdb}  2.00  2010   5.00  {aff:<12}  // {pdb}.pdf ({lig})" for pdb, (aff, lig) in DATA_INDEX.items()]
    return "\n".join(name) + "\n", "\n".join(data) + "\n"


def write_index_tgz(path):
    name, data = _index_text()
    with tarfile.open(path, "w:gz") as tf:
        for member, text in (("index/INDEX_general_PL_name.2020", name),
                             ("index/INDEX_general_PL_data.2020", data),
                             ("index/INDEX_general_PL.2020", "# unrelated\n")):
            blob = text.encode()
            info = tarfile.TarInfo(member)
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))


def test_read_pdbbind_index_from_tgz(tmp_path):
    tgz = tmp_path / "PDBbind_v2020_plain_text_index.tar.gz"
    write_index_tgz(tgz)
    name_index, data_index = read_pdbbind_index(str(tgz))
    assert name_index == NAME_INDEX
    assert data_index == DATA_INDEX


def _graph(rng, n_atoms):
    feats = [rng.normal(size=5) for _ in range(n_atoms)]
    edges = [[k, k + 1] for k in range(n_atoms - 1)] + [[k + 1, k] for k in range(n_atoms - 1)]
    return (n_atoms, feats, edges, [[1.0, 0.0, 0.0, 0.0]] * len(edges))


def write_parent(parent, missing=()):
    """Write the three *_processed.csv files and graph pickles the way the parent tree has them."""
    os.makedirs(parent, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = dict(zip(("pdbbind", "bindingnet", "bindingdb"), make_frames()))
    id_cols = {"pdbbind": "PDB_code", "bindingnet": "unique_identify", "bindingdb": "unique_id"}
    for src, frame in frames.items():
        frame.to_csv(os.path.join(parent, f"{src}_processed.csv"))
        graphs = {uid: _graph(rng, 2 + k % 3) for k, uid in enumerate(frame[id_cols[src]])
                  if uid not in missing}
        with open(os.path.join(parent, f"{src}.pickle"), "wb") as fh:
            pickle.dump(graphs, fh)


def test_main_writes_outputs(tmp_path):
    pytest.importorskip("torch_geometric")
    from paired_data import SOURCE_ID, PairedDataset

    parent, out = str(tmp_path / "parent"), str(tmp_path / "out")
    write_parent(parent, missing=("8vwx",))
    tgz = str(tmp_path / "index.tar.gz")
    write_index_tgz(tgz)
    main(["--parent-data", parent, "--pdbbind-index-tgz", tgz, "--out-dir", out,
          "--holdout-frac", "0.5", "--holdout-min-size", "5", "--seed", "0"])

    meta = pd.read_csv(os.path.join(out, "paired_meta.csv"), keep_default_na=False)
    assert list(meta.columns) == META_COLUMNS
    assert "8vwx" not in set(meta.unique_id)  # no graph -> dropped before ids are assigned
    assert sorted(meta.series_id.unique()) == list(range(meta.series_key.nunique()))
    with open(os.path.join(out, "paired_summary.json")) as fh:
        summary = json.load(fh)
    assert summary["missing_graphs"] == {"pdbbind": 1, "bindingnet": 0, "bindingdb": 0}
    assert summary["n_rows"] == len(meta)
    for split in ("train", "valid", "test", "series_valid"):
        rows = meta[meta.split == split]
        ds = PairedDataset(out, split)
        assert len(ds) == len(rows) > 0
        assert ds._data.row.tolist() == rows.row.tolist()
        assert ds._data.series.tolist() == rows.series_id.tolist()
        assert ds._data.lig.tolist() == rows.lig_id.tolist()
        assert ds._data.src.tolist() == [SOURCE_ID[s] for s in rows.source]
        assert np.allclose(ds._data.y.numpy(), rows.pK.to_numpy(), atol=1e-5)
        assert ds._data.kind.tolist() == [0] * len(rows)
        assert ds._data.parent.tolist() == rows.row.tolist()
        assert summary["rows"][split] == {s: int((rows.source == s).sum()) for s in SOURCE_ID}
        assert summary["kinds"][split] == {"0": len(rows), "1": 0, "2": 0}
    assert (meta.kind == 0).all() and (meta.parent == meta.row).all()
    assert "poses" not in summary
    assert not os.path.exists(os.path.join(out, "processed", "paired_poses_valid.pt"))
    # train pairs: 1swr or 1hp5 (whichever stayed) plus PDBbind P1 (1abc only in train)
    train = meta[meta.split == "train"]
    expected = 0
    for _, g in train.groupby("series_id"):
        n, same = len(g), g.groupby("lig_id").size()
        expected += n * (n - 1) // 2 - int((same * (same - 1) // 2).sum())
    assert summary["pairs"]["train"] == expected > 0



# ----------------------------------------------------------------------------- pose rows (arm 4)

POSE_META_COLUMNS = ["unique_id", "pdb", "kind", "k", "method", "rmsd", "min_dist", "contact_frac",
                     "attempts", "split"]
# (pdb, kind, k, split as written by featurise_poses): 1abc/2def train, 3ghi/7stu valid, 4jkl test,
# 5mno has no parent (dropped by the Tanimoto filter), 6pqr::neg0 is in the meta but not the pickle.
POSES = [("1abc", 1, 0, "train"), ("1abc", 1, 1, "train"), ("1abc", 2, 0, "train"), ("1abc", 2, 1, "train"),
         ("2def", 1, 0, "train"), ("2def", 2, 0, "train"),
         ("3ghi", 1, 0, "valid"), ("3ghi", 1, 1, "valid"), ("3ghi", 2, 0, "valid"), ("3ghi", 2, 1, "valid"),
         ("7stu", 1, 0, "valid"),
         ("4jkl", 1, 0, "test"), ("4jkl", 2, 0, "test"),
         ("5mno", 1, 0, "train"),
         ("6pqr", 2, 0, "train")]
NO_GRAPH = {"6pqr::neg0"}


def _pose_uid(pdb, kind, k):
    return f"{pdb}::{'pos' if kind == 1 else 'neg'}{k}"


def write_poses(directory):
    """pdbbind_poses.pickle + pdbbind_poses_meta.csv in the featurise_poses format (rows shuffled)."""
    rng = np.random.default_rng(1)
    records, graphs = [], {}
    for pdb, kind, k, split in POSES:
        uid = _pose_uid(pdb, kind, k)
        records.append({"unique_id": uid, "pdb": pdb, "kind": kind, "k": k,
                        "method": "nn" if kind == 1 else ("flip" if k % 2 == 0 else "conf"),
                        "rmsd": 1.0 if kind == 1 else 5.0, "min_dist": 2.8, "contact_frac": 0.8,
                        "attempts": 3, "split": split})
        if uid not in NO_GRAPH:
            graphs[uid] = _graph(rng, 3)
    meta = pd.DataFrame(records, columns=POSE_META_COLUMNS).sample(frac=1.0, random_state=3)
    pickle_path, meta_path = os.path.join(directory, "pdbbind_poses.pickle"), \
        os.path.join(directory, "pdbbind_poses_meta.csv")
    with open(pickle_path, "wb") as fh:
        pickle.dump(graphs, fh)
    meta.to_csv(meta_path, index=False)
    return pickle_path, meta_path, graphs


def _build(tmp_path, name, poses=False):
    parent = str(tmp_path / "parent")
    if not os.path.isdir(parent):
        write_parent(parent)
        write_index_tgz(str(tmp_path / "index.tar.gz"))
    out = str(tmp_path / name)
    argv = ["--parent-data", parent, "--pdbbind-index-tgz", str(tmp_path / "index.tar.gz"),
            "--out-dir", out, "--holdout-frac", "0.5", "--holdout-min-size", "5", "--seed", "0"]
    graphs = None
    if poses:
        pickle_path, meta_path, graphs = write_poses(str(tmp_path))
        argv += ["--poses-pickle", pickle_path, "--poses-meta", meta_path]
    main(argv)
    meta = pd.read_csv(os.path.join(out, "paired_meta.csv"), keep_default_na=False)
    with open(os.path.join(out, "paired_summary.json")) as fh:
        summary = json.load(fh)
    return out, meta, summary, graphs


def test_build_with_poses_adds_rows_and_poses_valid(tmp_path):
    pytest.importorskip("torch_geometric")
    from paired_data import PairedDataset

    out, meta, summary, graphs = _build(tmp_path, "poses", poses=True)
    assert list(meta.columns) == META_COLUMNS
    assert meta.row.tolist() == list(range(len(meta)))
    base = meta[meta.kind == 0]
    poses = meta[meta.kind > 0]
    assert (base.parent == base.row).all()
    assert base.row.max() < poses.row.min()  # pose rows are appended after every original row
    m = by_id(base)

    # kept pose rows: train parents (1abc, 2def) -> train; valid parents (3ghi, 7stu) -> poses_valid
    expected = [_pose_uid(p, kd, k) for p, kd, k, _ in sorted(POSES, key=lambda t: (m.loc[t[0], "row"]
                if t[0] in m.index else 99, t[1], t[2]))
                if p in ("1abc", "2def", "3ghi", "7stu") and _pose_uid(p, kd, k) not in NO_GRAPH]
    assert poses.unique_id.tolist() == expected  # sorted by (parent row, kind, k), not by meta order
    for r in poses.itertuples():
        pdb, tail = r.unique_id.split("::")
        par = m.loc[pdb]
        assert r.parent == par.row and r.kind == (1 if tail.startswith("pos") else 2)
        assert r.source == "pdbbind" and r.pK == par.pK
        assert r.split == ("train" if par.split == "train" else "poses_valid")
        for col in ("series_key", "series_id", "lig_key", "lig_id", "label_type", "uniprot", "anchor_pdb"):
            assert getattr(r, col) == par[col], col
    assert set(meta.loc[meta.split.isin(["valid", "test", "series_valid"]), "kind"]) == {0}

    # .pt files: train holds kinds 0, 1, 2 in meta order; poses_valid = every valid crystal + its poses
    train = PairedDataset(out, "train")
    rows = meta[meta.split == "train"]
    assert train._data.row.tolist() == rows.row.tolist()
    assert train._data.kind.tolist() == rows.kind.tolist()
    assert train._data.parent.tolist() == rows.parent.tolist()
    assert np.allclose(train._data.y.numpy(), rows.pK.to_numpy(), atol=1e-5)
    pv = PairedDataset(out, "poses_valid")
    crystals = base[base.split == "valid"]
    pv_rows = pd.concat([crystals, meta[meta.split == "poses_valid"]]).sort_values(["parent", "row"])
    assert pv._data.row.tolist() == pv_rows.row.tolist()
    assert pv._data.kind.tolist() == pv_rows.kind.tolist() == [0, 1, 1, 2, 2, 0, 1]
    assert pv._data.parent.tolist() == pv_rows.parent.tolist()
    assert pv._data.series.tolist() == pv_rows.series_id.tolist()
    assert np.allclose(pv._data.y.numpy(), pv_rows.pK.to_numpy(), atol=1e-5)
    # the crystal copies carry the same graph and row id as in the valid split
    valid = PairedDataset(out, "valid")
    for k in range(len(pv)):
        d = pv[k]
        uid = pv_rows.unique_id.iloc[k]
        if int(d.kind) == 0:
            v = valid[valid._data.row.tolist().index(int(d.row))]
            assert torch_equal(d.x, v.x)
        else:
            assert np.allclose(d.x.numpy(), np.asarray(graphs[uid][1], dtype=np.float32))
    for split in ("valid", "test", "series_valid"):
        ds = PairedDataset(out, split)
        assert set(ds._data.kind.tolist()) == {0}

    # summary: kind counts per split, poses_valid rows, drop counts
    assert summary["kinds"]["train"] == {"0": int((rows.kind == 0).sum()), "1": 3, "2": 3}
    assert summary["kinds"]["poses_valid"] == {"0": 2, "1": 3, "2": 2}
    assert summary["rows"]["poses_valid"]["pdbbind"] == 7
    assert summary["poses"] == {"meta_rows": 15, "kept": 11, "train": 6, "poses_valid": 5,
                                "dropped_no_graph": 1, "dropped_no_parent": 1, "dropped_test": 2,
                                "dropped_series_valid": 0, "split_mismatch": 0,
                                "pickle_only": 0}


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def test_train_kinds_zero_matches_base_rows(tmp_path):
    pytest.importorskip("torch_geometric")
    from paired_data import PairedDataset

    out0, meta0, summary0, _ = _build(tmp_path, "base")
    out1, meta1, summary1, _ = _build(tmp_path, "poses", poses=True)
    t0 = meta0[meta0.split == "train"]
    t1 = meta1[(meta1.split == "train") & (meta1.kind == 0)]
    assert t1.unique_id.tolist() == t0.unique_id.tolist()
    # every original row is unchanged (ids, splits, labels, series and ligand codes)
    pd.testing.assert_frame_equal(meta1[meta1.kind == 0].reset_index(drop=True), meta0)
    # and the .pt kind-0 train rows are arm 1's train rows, in the same order
    d0, d1 = PairedDataset(out0, "train")._data, PairedDataset(out1, "train")._data
    sel = (d1.kind == 0)
    assert d1.row[sel].tolist() == d0.row.tolist()
    assert d1.series[sel].tolist() == d0.series.tolist()
    for split in ("valid", "test", "series_valid"):
        assert PairedDataset(out1, split)._data.row.tolist() == PairedDataset(out0, split)._data.row.tolist()
    assert summary1["rows"]["series_valid"] == summary0["rows"]["series_valid"]


def test_poses_flags_go_together(tmp_path, capsys):
    pickle_path, meta_path, _ = write_poses(str(tmp_path))
    for extra in (["--poses-pickle", pickle_path], ["--poses-meta", meta_path]):
        with pytest.raises(SystemExit) as exc:
            main(["--parent-data", str(tmp_path), "--pdbbind-index-tgz", "x.tgz",
                  "--out-dir", str(tmp_path / "o"), *extra])
        assert exc.value.code not in (0, None)
        err = capsys.readouterr().err
        assert "unrecognized" not in err and "--poses-pickle and --poses-meta" in err, err


# ----------------------------------------------------------------------------- soft negatives (kinds 3-7), group holdout

AUG_FIXED = ["unique_id", "pdb", "kind", "k", "method", "split", "rmsd_to_crystal", "min_dist", "contact_frac",
             "n_within_3A"]
TWIN_EXTRA = ["twin_unique_id", "twin_source", "shape_dist", "complementarity", "o3a_score", "n_heavy_anchor",
              "n_heavy_twin", "dlogp", "dcharge", "darom"]
EDIT_EXTRA = ["edit_rule", "site_class", "n_edited_atoms", "n_contacts_edited_group", "partner_type",
              "mmff_restrained_rmsd"]
AUG_TAGS = {3: "tpos", 4: "twin", 5: "epos", 6: "cedit", 7: "xedit"}
PARENT_SPLIT = {"1abc": "train", "2def": "train", "3ghi": "valid", "4jkl": "test", "6pqr": "train",
                "7stu": "valid"}
BDB_100 = "4DBU_BT9_4DBU-results_100"
# (pdb, kind, k, twin_unique_id): 3ghi::twin1 names an unknown row, 6pqr's twin ligand (4jkl, FMN) is no
# train positive, 4jkl is a test anchor (dropped)
TWINS = [("1abc", 3, 0, ""), ("1abc", 4, 0, "2def"), ("1abc", 4, 1, BDB_100),
         ("2def", 3, 0, ""), ("2def", 4, 0, "1abc"),
         ("3ghi", 3, 0, ""), ("3ghi", 4, 0, "2def"), ("3ghi", 4, 1, "zzzz"),
         ("4jkl", 3, 0, ""), ("4jkl", 4, 0, "2def"),
         ("6pqr", 4, 0, "4jkl")]
EDITS = [("1abc", 5, 0), ("1abc", 6, 0), ("1abc", 7, 0), ("1abc", 7, 1),
         ("3ghi", 5, 0), ("3ghi", 6, 0), ("3ghi", 6, 1), ("3ghi", 7, 0),
         ("7stu", 5, 0),                     # a C-positive without edits: orphan, dropped
         ("2def", 5, 0), ("2def", 6, 0)]
OWN_PK = 12.34  # a twin's "own" label in its own pocket: must never reach the dataset


def _aug_uid(pdb, kind, k):
    return f"{pdb}::{AUG_TAGS[kind]}{k}"


def write_aug_files(directory, name, records, extra_columns, seed):
    """<name>.pickle + <name>_meta.csv in the generator format (first ten columns fixed), rows shuffled."""
    rng = np.random.default_rng(seed)
    graphs = {r["unique_id"]: _graph(rng, 3) for r in records}
    meta = pd.DataFrame(records)
    meta = meta[AUG_FIXED + extra_columns + [c for c in meta.columns if c not in AUG_FIXED + extra_columns]]
    meta = meta.sample(frac=1.0, random_state=seed)
    pickle_path = os.path.join(directory, f"{name}.pickle")
    meta_path = os.path.join(directory, f"{name}_meta.csv")
    with open(pickle_path, "wb") as fh:
        pickle.dump(graphs, fh)
    meta.to_csv(meta_path, index=False)
    return pickle_path, meta_path


def _fixed(pdb, kind, k, method):
    return {"unique_id": _aug_uid(pdb, kind, k), "pdb": pdb, "kind": kind, "k": k, "method": method,
            "split": PARENT_SPLIT[pdb], "rmsd_to_crystal": 0.5, "min_dist": 2.9, "contact_frac": 0.8,
            "n_within_3A": 1}


def write_twins(directory):
    records = []
    for pdb, kind, k, twin in TWINS:
        r = _fixed(pdb, kind, k, "overlay")
        r.update(twin_unique_id=twin, twin_source="pdbbind" if twin in PARENT_SPLIT else "bindingdb",
                 shape_dist=0.01, complementarity=-0.2, o3a_score=40.0, n_heavy_anchor=20, n_heavy_twin=21,
                 dlogp=0.3, dcharge=0, darom=0,
                 pK=OWN_PK if kind == 4 else np.nan)  # not a format column: the builder must ignore it
        records.append(r)
    return write_aug_files(directory, "twins", records, TWIN_EXTRA, seed=11)


def write_edits(directory):
    records = []
    for pdb, kind, k in EDITS:
        r = _fixed(pdb, kind, k, "edit")
        r.update(edit_rule="" if kind == 5 else "cooh_to_me",
                 site_class={5: "null", 6: "contact", 7: "exposed"}[kind], n_edited_atoms=3,
                 n_contacts_edited_group=2 if kind == 6 else 0, partner_type="pos" if kind == 6 else "",
                 mmff_restrained_rmsd=0.1)
        records.append(r)
    return write_aug_files(directory, "edits", records, EDIT_EXTRA, seed=12)


GROUPS = {"1abc": "uniprot:P1", "3ghi": "uniprot:P1", "7stu": "uniprot:P1", "2def": "single:2def",
          "4jkl": "single:4jkl", "6pqr": "single:6pqr", "5mno": "single:5mno"}  # 8vwx not listed


def write_groups(path, overrides=None):
    """pocket_groups.csv: P1 = 1abc, 3ghi, 7stu and the 1swr BindingNet template (merged by name)."""
    pdbbind, bindingnet, bindingdb = make_frames()
    rows = [(uid, g) for uid, g in {**GROUPS, **(overrides or {})}.items()]
    for uid, pdb in zip(bindingnet.unique_identify, bindingnet.pdb):
        rows.append((uid, "uniprot:P1" if pdb == "1swr" else f"single:{pdb}"))
    for uid, folder in zip(bindingdb.unique_id, bindingdb.folder):
        rows.append((uid, "uniprot:P4" if folder.startswith("4DBU") else "single:1xyz"))
    pd.DataFrame(rows, columns=["unique_id", "group_id"]).assign(group_size_rows=1).to_csv(path, index=False)


def _build_ac(tmp_path, name, extra, holdout_frac="0.5"):
    parent = str(tmp_path / "parent")
    if not os.path.isdir(parent):
        write_parent(parent)
        write_index_tgz(str(tmp_path / "index.tar.gz"))
    out = str(tmp_path / name)
    main(["--parent-data", parent, "--pdbbind-index-tgz", str(tmp_path / "index.tar.gz"), "--out-dir", out,
          "--holdout-frac", holdout_frac, "--holdout-min-size", "5", "--seed", "0", *extra])
    meta = pd.read_csv(os.path.join(out, "paired_meta.csv"), keep_default_na=False)
    with open(os.path.join(out, "paired_summary.json")) as fh:
        summary = json.load(fh)
    return out, meta, summary


def test_multiple_aug_inputs(tmp_path):
    pytest.importorskip("torch_geometric")
    from paired_data import PairedDataset

    d = str(tmp_path)
    twins_pk, twins_meta = write_twins(d)
    edits_pk, edits_meta = write_edits(d)
    poses_pk, poses_meta, _ = write_poses(d)
    out, meta, summary = _build_ac(tmp_path, "ac", ["--aug", f"{twins_pk}:{twins_meta}",
                                                     "--aug", f"{edits_pk}:{edits_meta}",
                                                     "--poses-pickle", poses_pk, "--poses-meta", poses_meta])
    _, base, _ = _build_ac(tmp_path, "base", [])
    assert list(meta.columns) == META_COLUMNS  # no --groups: arm-4 columns
    pd.testing.assert_frame_equal(meta[meta.kind == 0].reset_index(drop=True), base)
    assert meta.row.tolist() == list(range(len(meta)))
    aug = meta[meta.kind > 0]
    assert aug.row.min() > meta.loc[meta.kind == 0, "row"].max()
    # one ordering over every input: (parent row, kind, k)
    keys = list(zip(aug.parent, aug.kind, [int(u.split("::")[1].lstrip("abcdefghijklmnopqrstuvwxyz")) for u in aug.unique_id]))
    assert keys == sorted(keys)
    # input order does not matter
    _, swapped, _ = _build_ac(tmp_path, "swapped", ["--poses-pickle", poses_pk, "--poses-meta", poses_meta,
                                                     "--aug", f"{edits_pk}:{edits_meta}",
                                                     "--aug", f"{twins_pk}:{twins_meta}"])
    pd.testing.assert_frame_equal(swapped, meta)

    m = by_id(base)
    kept_twins = {_aug_uid(p, kd, k) for p, kd, k, t in TWINS
                  if p in ("1abc", "2def", "3ghi") and t != "zzzz"}
    kept_edits = {_aug_uid(p, kd, k) for p, kd, k in EDITS if p != "7stu"}
    assert set(aug.loc[aug.kind.isin([3, 4]), "unique_id"]) == kept_twins
    assert set(aug.loc[aug.kind.isin([5, 6, 7]), "unique_id"]) == kept_edits
    for r in aug.itertuples():
        par = m.loc[r.unique_id.split("::")[0]]
        assert r.parent == par.row and r.source == "pdbbind" and r.pK == par.pK
        for col in ("series_key", "series_id", "lig_key", "lig_id", "label_type", "uniprot", "anchor_pdb"):
            assert getattr(r, col) == par[col], col
        if par.split == "train":
            assert r.split == "train"
        else:  # valid anchors: arm-4 label for kinds 1/2, aug_valid for the rest
            assert r.split == ("poses_valid" if r.kind in (1, 2) else "aug_valid")
    assert set(meta.loc[meta.split.isin(["valid", "test", "series_valid"]), "kind"]) == {0}

    # train .pt holds every kind; aug_valid = valid crystals + all their rows; poses_valid = kinds 0-2 only
    train = PairedDataset(out, "train")
    rows = meta[meta.split == "train"]
    assert train._data.row.tolist() == rows.row.tolist()
    assert sorted(set(train._data.kind.tolist())) == [0, 1, 2, 3, 4, 5, 6, 7]
    av = PairedDataset(out, "aug_valid")
    crystals = base[base.split == "valid"]
    av_rows = pd.concat([crystals, meta[meta.split.isin(["poses_valid", "aug_valid"])]]).sort_values(["parent", "row"])
    assert av._data.row.tolist() == av_rows.row.tolist()
    assert av._data.kind.tolist() == [0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 0, 1]
    assert av._data.parent.tolist() == av_rows.parent.tolist()
    assert np.allclose(av._data.y.numpy(), av_rows.pK.to_numpy(), atol=1e-5)
    pv = PairedDataset(out, "poses_valid")
    assert pv._data.kind.tolist() == [0, 1, 1, 2, 2, 0, 1]
    assert not os.path.exists(os.path.join(out, "processed", "paired_group_valid.pt"))

    # summary: kinds per split (0-7), per-input counts, arm-4 block for the --poses-* input
    assert summary["kinds"]["train"] == {"0": int((rows.kind == 0).sum()), "1": 3, "2": 3, "3": 2, "4": 3,
                                         "5": 2, "6": 2, "7": 2}
    assert summary["kinds"]["aug_valid"] == {"0": 2, "1": 3, "2": 2, "3": 1, "4": 1, "5": 1, "6": 2, "7": 1}
    assert set(summary["aug"]) == {"poses", "twins", "edits"}
    tw = summary["aug"]["twins"]
    assert (tw["meta_rows"], tw["kept"], tw["train"], tw["aug_valid"], tw["poses_valid"]) == (11, 7, 5, 2, 0)
    assert (tw["dropped_test"], tw["dropped_twin_unknown"], tw["dropped_twin_not_in_train"]) == (2, 1, 1)
    assert tw["twin_rule"] == "checked" and summary["aug"]["edits"]["twin_rule"] == "no twins"
    ed = summary["aug"]["edits"]
    assert (ed["meta_rows"], ed["kept"], ed["train"], ed["aug_valid"]) == (11, 10, 6, 4)
    assert ed["dropped_orphan_5"] == 1 and tw["dropped_orphan_3"] == tw["dropped_orphan_4"] == 0
    assert summary["poses"] == {"meta_rows": 15, "kept": 11, "train": 6, "poses_valid": 5,
                                "dropped_no_graph": 1, "dropped_no_parent": 1, "dropped_test": 2,
                                "dropped_series_valid": 0, "split_mismatch": 0, "pickle_only": 0}


def test_aug_only_kinds_3_to_7_skip_poses_valid(tmp_path):
    pytest.importorskip("torch_geometric")
    twins_pk, twins_meta = write_twins(str(tmp_path))
    out, meta, summary = _build_ac(tmp_path, "a", ["--aug", f"{twins_pk}:{twins_meta}"])
    assert os.path.exists(os.path.join(out, "processed", "paired_aug_valid.pt"))
    assert not os.path.exists(os.path.join(out, "processed", "paired_poses_valid.pt"))
    assert "poses" not in summary and set(summary["aug"]) == {"twins"}
    assert set(meta.loc[meta.kind > 0, "split"]) == {"train", "aug_valid"}


def test_aug_ids_must_match_kind(tmp_path):
    from build_paired_dataset import add_aug

    base = meta_for(holdout_frac=0.0)
    bad = pd.DataFrame({"unique_id": ["1abc::twin0"], "pdb": ["1abc"], "kind": [3], "k": [0]})
    with pytest.raises(ValueError, match="disagree"):
        add_aug(base, [("twins", bad, {"1abc::twin0"})])
    unknown = pd.DataFrame({"unique_id": ["1abc::x0"], "pdb": ["1abc"], "kind": [8], "k": [0]})
    with pytest.raises(ValueError, match="kinds"):
        add_aug(base, [("x", unknown, {"1abc::x0"})])
    ok = pd.DataFrame({"unique_id": ["1abc::tpos0", "1abc::twin0"], "pdb": ["1abc"] * 2, "kind": [3, 4],
                       "k": [0, 0]})
    with pytest.raises(ValueError, match="repeat"):
        add_aug(base, [("a", ok, set(ok.unique_id)), ("b", ok, set(ok.unique_id))])


def test_hinge_uses_parent_label(tmp_path):
    """A shape twin carries its own pocket's label in the generator meta (here pK = 12.34): the builder
    writes the PARENT's pK on the row, and the hinge reads that y."""
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.data import Batch

    import pair_losses
    import train_paired
    from paired_data import PairedDataset

    twins_pk, twins_meta = write_twins(str(tmp_path))
    assert (pd.read_csv(twins_meta).query("kind == 4")["pK"] == OWN_PK).all()
    out, meta, _ = _build_ac(tmp_path, "a", ["--aug", f"{twins_pk}:{twins_meta}"])
    m = by_id(meta[meta.kind == 0])
    twins = meta[meta.kind == 4]
    assert len(twins) == 4
    for r in twins.itertuples():
        assert r.pK == m.loc[r.unique_id.split("::")[0], "pK"] != OWN_PK

    train = PairedDataset(out, "train")
    idx = [i for i in range(len(train)) if int(train[i].kind) == 4]
    batch = Batch.from_data_list([train[i] for i in idx])
    parent_pk = torch.tensor([m.loc[u.split("::")[0], "pK"] for u in
                              meta.set_index("row").loc[batch.row.tolist(), "unique_id"]], dtype=torch.float)
    assert torch.allclose(batch.y, parent_pk)
    # pred 0.5 pK below the parent's label, margin 1.0: relu(0.5) per row with the parent's label;
    # with the twin's own label (12.34) every term would be 0
    pred = parent_pk - 0.5
    got = pair_losses.hinge_by_kind(pred, batch.y, batch.kind, {4: 1.0})
    assert got.item() == pytest.approx(0.5, abs=1e-5)
    assert pair_losses.hinge_by_kind(pred, torch.full_like(pred, OWN_PK), batch.kind, {4: 1.0}).item() == 0.0
    cfg = train_paired.LossConfig(kind="none", weight=0.0, tie=0.0, delta=1.0, weighting="none",
                                  hinge_margins={4: 1.0}, hinge_weights={4: 1.0})
    total, mse, pair, decoy, hinge = train_paired.compute_loss(pred.view(-1, 1), batch, cfg)
    assert mse.item() == 0.0 and decoy.item() == 0.0  # kind 4 never enters the MSE
    assert hinge.item() == pytest.approx(0.5, abs=1e-5) and total.item() == pytest.approx(0.5, abs=1e-5)


def test_group_holdout_disjoint(tmp_path, capsys):
    pytest.importorskip("torch_geometric")
    from paired_data import PairedDataset

    d = str(tmp_path)
    twins_pk, twins_meta = write_twins(d)
    edits_pk, edits_meta = write_edits(d)
    groups = str(tmp_path / "pocket_groups.csv")
    write_groups(groups)
    # series holdout of every eligible series (1swr, 1hp5, 4DBU -> series_valid); P1 is the only group with
    # >= 8 train/series_valid rows (1abc + the seven 1swr rows) and is held out
    extra = ["--groups", groups, "--holdout-groups", "1.0", "--holdout-group-min-rows", "8",
             "--aug", f"{twins_pk}:{twins_meta}", "--aug", f"{edits_pk}:{edits_meta}"]
    out, meta, summary = _build_ac(tmp_path, "g", extra, holdout_frac="1.0")
    printed = capsys.readouterr().out
    assert list(meta.columns) == META_COLUMNS + ["group_id", "group_leak"]
    _, plain, _ = _build_ac(tmp_path, "plain", [], holdout_frac="1.0")

    base = meta[meta.kind == 0]
    gv = meta[meta.split == "group_valid"]
    assert set(gv.group_id) == {"uniprot:P1"} and set(gv.kind) == {0}
    assert sorted(gv.unique_id) == sorted(["1abc"] + [u for u in base.unique_id if "_1swr_" in u])
    # every train / series_valid row of the held group moved (all sources); nothing else changed split
    old_split = plain.set_index("unique_id")["split"]
    for r in base.itertuples():
        if r.group_id == "uniprot:P1" and old_split[r.unique_id] in ("train", "series_valid"):
            assert r.split == "group_valid"
        else:
            assert r.split == old_split[r.unique_id]
    # valid / test rows of the held group stay where they are and are flagged
    assert base.set_index("unique_id").loc[["3ghi", "7stu"], ["split", "group_leak"]].values.tolist() == \
        [["valid", 1], ["valid", 1]]
    assert (base.loc[~base.unique_id.isin(["3ghi", "7stu"]), "group_leak"] == 0).all()
    assert base.set_index("unique_id").loc["8vwx", "group_id"] == ""  # not in the groups file
    # disjointness: no train row of any kind shares a group_id with group_valid
    held = set(gv.group_id)
    assert not meta[(meta.split == "train") & meta.group_id.isin(held)].shape[0]
    assert "train rows sharing a group_valid group_id = 0" in printed

    # augmented rows: none for the held-out train anchor (1abc), group columns copied from the parent
    aug = meta[meta.kind > 0]
    assert not (aug.parent == by_id(base).loc["1abc", "row"]).any()
    for r in aug.itertuples():
        par = base.set_index("row").loc[r.parent]
        assert (r.group_id, r.group_leak) == (par.group_id, par.group_leak)
    assert (aug.loc[aug.split == "aug_valid", "group_leak"] == 1).all()  # 3ghi / 7stu rows
    tw, ed = summary["aug"]["twins"], summary["aug"]["edits"]
    assert tw["dropped_group_valid"] == 3 and ed["dropped_group_valid"] == 4
    # 2def's twin ligand (1abc's ABC) left train with the group: the twin is dropped (matching rule) and
    # its A-positive, left without a twin, is dropped as an orphan
    assert "2def::twin0" not in set(meta.unique_id) and "2def::tpos0" not in set(meta.unique_id)
    assert tw["dropped_twin_not_in_train"] == 2  # 2def::twin0 and 6pqr::twin0
    assert tw["dropped_orphan_3"] == 1 and ed["dropped_orphan_5"] == 1  # 2def::tpos0, 7stu::epos0

    # .pt files and summary
    ds = PairedDataset(out, "group_valid")
    assert ds._data.row.tolist() == gv.row.tolist()
    assert set(ds._data.kind.tolist()) == {0}
    g = summary["group_holdout"]
    assert (g["n_held"], g["n_eligible"], g["held_groups"]) == (1, 1, ["uniprot:P1"])
    assert g["group_valid_rows"] == len(gv) == 8
    assert g["moved_rows"] == {"pdbbind:train": 1, "bindingnet:series_valid": 7}
    assert g["leak_rows"] == {"valid": 2, "test": 0}
    assert g["train_rows_sharing_group_valid_group"] == 0
    assert g["rows_without_group"]["pdbbind"] == 1
    assert summary["rows"]["group_valid"] == {"pdbbind": 1, "bindingnet": 7, "bindingdb": 0}


def test_group_holdout_draw_and_eligibility():
    from build_paired_dataset import group_holdout

    meta = meta_for(holdout_frac=0.0)
    # 4 groups of BindingNet/BindingDB rows: 1swr (7 train rows), 1hp5 (5), 4DBU (5), and P1 = 1abc (train)
    # plus 3ghi, 7stu (valid): only train/series_valid rows count, so P1 (1 row) is never eligible
    groups = {}
    for uid, src, key in zip(meta.unique_id, meta.source, meta.series_key):
        if src != "pdbbind":
            groups[uid] = key
    groups.update({"1abc": "P1", "3ghi": "P1", "7stu": "P1"})
    groups = pd.Series(groups)
    out, info = group_holdout(meta, groups, frac=0.67, min_rows=5, seed=0)
    eligible = ["bindingdb:4DBU_BT9", "bindingnet:1hp5", "bindingnet:1swr"]
    assert info["n_eligible"] == 3 and info["n_held"] == 2
    assert info["held_groups"] == sorted(np.random.default_rng(0).choice(eligible, 2, replace=False).tolist())
    moved = out[out.split == "group_valid"]
    assert set(moved.group_id) == set(info["held_groups"])
    assert (out.loc[out.group_id.isin(info["held_groups"]), "split"] == "group_valid").all()
    assert out.loc[out.unique_id == "1abc", "split"].item() == "train"
    assert (out.group_leak == 0).all()
    # deterministic, and a fraction that rounds to no group is an error
    pd.testing.assert_frame_equal(out, group_holdout(meta, groups, frac=0.67, min_rows=5, seed=0)[0])
    with pytest.raises(ValueError, match="no group held out"):
        group_holdout(meta, groups, frac=0.1, min_rows=5, seed=0)


def test_holdout_groups_needs_groups(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--parent-data", str(tmp_path), "--pdbbind-index-tgz", "x.tgz", "--out-dir", str(tmp_path / "o"),
              "--holdout-groups", "0.05"])
    assert exc.value.code not in (0, None)
    assert "--holdout-groups needs --groups" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--parent-data", str(tmp_path), "--pdbbind-index-tgz", "x.tgz", "--out-dir", str(tmp_path / "o"),
              "--aug", "no_colon_here"])


def test_end_to_end_build_train_eval(tmp_path):
    """The builder's files feed train_paired (kinds 0,3-7) and eval_paired (--aug-pt, --group-valid-pt)."""
    pytest.importorskip("torch_geometric")
    import eval_paired
    import train_paired

    d = str(tmp_path)
    twins_pk, twins_meta = write_twins(d)
    edits_pk, edits_meta = write_edits(d)
    groups = str(tmp_path / "pocket_groups.csv")
    write_groups(groups, {"1abc": "single:1abc"})  # P1 = 3ghi, 7stu (valid) + 1swr: 1abc's rows stay in train
    out, meta, _ = _build_ac(tmp_path, "data_ac",
                             ["--groups", groups, "--holdout-groups", "1.0", "--holdout-group-min-rows", "7",
                              "--aug", f"{twins_pk}:{twins_meta}", "--aug", f"{edits_pk}:{edits_meta}"],
                             holdout_frac="1.0")
    work = str(tmp_path / "output")
    side = train_paired.main(["--data-dir", out, "--out-dir", work, "--seed_index", "0", "--tag", "AC",
                              "--epochs", "2", "--hidden-dim", "8", "--batch-size", "4", "--chunk", "2",
                              "--train-kinds", "0,3,4,5,6,7"])
    train = meta[meta.split == "train"]
    assert side["n_train_rows"] == len(train) and side["hinge_kinds"] == [4, 6]
    ev = tmp_path / "eval"
    proc = os.path.join(out, "processed")
    eval_paired.main(["--models", os.path.join(work, "trained_models", "AC_0.model"),
                      "--scaler", os.path.join(work, "trained_models", "AC.pickle"),
                      "--meta", os.path.join(out, "paired_meta.csv"),
                      "--series-valid-pt", os.path.join(proc, "paired_series_valid.pt"),
                      "--test-pt", os.path.join(proc, "paired_test.pt"),
                      "--aug-pt", os.path.join(proc, "paired_aug_valid.pt"),
                      "--group-valid-pt", os.path.join(proc, "paired_group_valid.pt"),
                      "--out-dir", str(ev), "--hidden-dim", "8", "--device", "cpu"])
    aug = pd.read_csv(ev / "aug_summary.csv")
    valid_aug = meta[meta.split.isin(["aug_valid"])]
    assert aug["kind"].tolist() == [0] + sorted(set(valid_aug.kind))
    gv = pd.read_csv(ev / "group_valid_summary.csv")
    assert gv.iloc[0]["scope"] == "pointwise" and gv.iloc[0]["n_items"] == 7
    preds = pd.read_csv(ev / "predictions_group_valid.csv", keep_default_na=False)
    assert set(preds["group_id"]) == {"uniprot:P1"}


def test_orphans_dropped():
    from build_paired_dataset import add_aug

    base = meta_for(holdout_frac=0.0)  # PDBbind train: 1abc, 2def, 6pqr, 8vwx; valid: 3ghi, 7stu

    def table(specs, **extra):
        return pd.DataFrame([{"unique_id": _aug_uid(p, kd, k), "pdb": p, "kind": kd, "k": k, **extra.get(p, {})}
                             for p, kd, k in specs])

    twins = table([("1abc", 4, 0), ("2def", 3, 0), ("3ghi", 3, 0), ("3ghi", 4, 0), ("8vwx", 3, 0), ("8vwx", 4, 0)])
    twins["twin_unique_id"] = ["2def", "", "", "zzzz", "", "2def"]  # 3ghi's only twin names an unknown row
    edits = table([("1abc", 6, 0), ("2def", 7, 0), ("6pqr", 5, 0), ("8vwx", 5, 0), ("8vwx", 6, 0),
                   ("8vwx", 7, 0), ("7stu", 5, 0), ("7stu", 7, 0)])
    out, counts = add_aug(base, [("twins", twins, set(twins.unique_id)), ("edits", edits, set(edits.unique_id))])
    kept = set(out.loc[out.kind > 0, "unique_id"])
    assert kept == {"8vwx::tpos0", "8vwx::twin0", "8vwx::epos0", "8vwx::cedit0", "8vwx::xedit0",
                    "7stu::epos0", "7stu::xedit0"}
    tw, ed = counts["twins"], counts["edits"]
    assert tw["dropped_twin_unknown"] == 1
    assert (tw["dropped_orphan_4"], tw["dropped_orphan_3"]) == (1, 2)  # 1abc::twin0; 2def::tpos0, 3ghi::tpos0
    assert (ed["dropped_orphan_5"], ed["dropped_orphan_6"], ed["dropped_orphan_7"]) == (1, 1, 1)
    assert (tw["kept"], ed["kept"]) == (2, 5)
