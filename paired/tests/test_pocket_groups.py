"""Target groups on a synthetic meta + PDBbind name index + zernike npz: name merge, generic names,
template inheritance, single groups, nearest training pocket of another group, optional descriptor merge."""
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

import pocket_groups as pg

K = 49

NAME_INDEX = """# ==============================================================================
# List of protein-ligand complexes with known binding data in PDBbind v.2020
# PDB code, release year, Uniprot ID, protein name
# ==============================================================================
1aaa  2000  P10001  KINASE ALPHA
1aab  2000  P10002  Kinase-alpha
1bbb  2001  P20001  INTEGRASE
1bbc  2001  P20002  INTEGRASE
1bbd  2001  P20003  INTEGRASE
1bbe  2001  P20004  INTEGRASE
1bbf  2001  P20005  INTEGRASE
1ccc  2002  P30001  PROTEASE X
1zzz  2003  ------  ANTIBODY FAB
"""

# unique_id, source, split, uniprot, series_key, anchor_pdb, kind
META = [
    ("1aaa", "pdbbind", "train", "P10001", "pdbbind:P10001", "1aaa", "0"),
    ("1aab", "pdbbind", "test", "P10002", "pdbbind:P10002", "1aab", "0"),
    ("1bbb", "pdbbind", "train", "P20001", "pdbbind:P20001", "1bbb", "0"),
    ("1bbc", "pdbbind", "valid", "P20002", "pdbbind:P20002", "1bbc", "0"),
    ("1bbd", "pdbbind", "train", "P20003", "pdbbind:P20003", "1bbd", "0"),
    ("1bbe", "pdbbind", "train", "P20004", "pdbbind:P20004", "1bbe", "0"),
    ("1bbf", "pdbbind", "train", "P20005", "pdbbind:P20005", "1bbf", "0"),
    ("1ccc", "pdbbind", "train", "P30001", "pdbbind:P30001", "1ccc", "0"),
    ("1ddd", "pdbbind", "train", "P30001", "pdbbind:P30001", "1ddd", "0"),   # absent from the npz
    ("1zzz", "pdbbind", "train", "", "pdbbind:single:1zzz", "1zzz", "0"),
    ("bn1", "bindingnet", "train", "P10002", "bindingnet:1aab", "1aab", "0"),
    ("bn2", "bindingnet", "series_valid", "P10002", "bindingnet:1aab", "1aab", "0"),
    ("bd1", "bindingdb", "train", "", "bindingdb:1ZZZ_X", "1zzz", "0"),
    ("bd2", "bindingdb", "train", "P30001", "bindingdb:1CCC_Y", "1ccc", "0"),  # featurisation failed
    ("1aaa::pos0", "pdbbind", "train", "P10001", "pdbbind:P10001", "1aaa", "1"),  # pose row: skipped
]


def _meta():
    return pd.DataFrame(META, columns=["unique_id", "source", "split", "uniprot", "series_key", "anchor_pdb",
                                       "kind"])


def _descriptors(uids, seed=0):
    """Random pockets; bn1 is a near copy of 1aab (same group); 1zzz and bd1 near copies of 1ccc."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(len(uids), 7, K)).astype(np.float32)
    g = rng.normal(size=(len(uids), 21, K)).astype(np.float32)
    at = {u: i for i, u in enumerate(uids)}
    for copy, src in (("bn1", "1aab"), ("1zzz", "1ccc"), ("bd1", "1ccc")):
        f[at[copy]] = f[at[src]] + 1e-3 * rng.normal(size=(7, K))
        g[at[copy]] = g[at[src]] + 1e-3 * rng.normal(size=(21, K))
    return f, g


def _write_inputs(tmp_path):
    meta = _meta()
    meta.to_csv(tmp_path / "meta.csv", index=False)
    uids = [u for u in meta["unique_id"] if u not in ("1ddd", "1aaa::pos0")] + ["not_in_meta"]
    f, g = _descriptors(uids)
    ok = np.ones(len(uids), dtype=bool)
    i = uids.index("bd2")
    ok[i] = False
    f[i] = np.nan
    g[i] = np.nan
    np.savez_compressed(tmp_path / "zernike.npz", unique_id=np.array(uids), source=np.array(["x"] * len(uids)),
                        ok=ok, pocket_F=f, pocket_G=g, pocket_mode=np.array("sphere"), radius=np.float64(10.0))
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    (index_dir / pg.NAME_FILE).write_text(NAME_INDEX)
    tgz = tmp_path / "PDBbind_v2020_plain_text_index.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        data = NAME_INDEX.encode()
        info = tarfile.TarInfo(f"index/{pg.NAME_FILE}")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return tgz, index_dir


def _run(tmp_path, *extra):
    tgz, _ = _write_inputs(tmp_path)
    out = tmp_path / "groups" / "pocket_groups.csv"
    pg.main(["--meta", str(tmp_path / "meta.csv"), "--pdbbind-index-tgz", str(tgz),
             "--zernike", str(tmp_path / "zernike.npz"), "--out", str(out), *extra])
    rows = pd.read_csv(out, keep_default_na=False, na_values=[""])
    summary = json.loads((tmp_path / "groups" / "pocket_groups_summary.json").read_text())
    return rows.set_index("unique_id", drop=False), summary


def test_normalise_name_and_index_reader(tmp_path):
    assert pg.normalise_name("HIV-1 protease (mutant)") == "HIV 1 PROTEASE MUTANT"
    tgz, index_dir = _write_inputs(tmp_path)
    a = pg.read_name_index(tgz=str(tgz))
    b = pg.read_name_index(index_dir=str(index_dir))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 9 and a.set_index("pdb").loc["1zzz", "uniprot"] == ""
    assert a.set_index("pdb").loc["1aab", "name"] == "KINASE ALPHA"
    with pytest.raises(ValueError):
        pg.read_name_index()


def test_groups_name_merge_generic_skip_and_inheritance(tmp_path):
    rows, summary = _run(tmp_path)
    assert list(rows.columns) == ["unique_id", "source", "split", "uniprot", "group_id", "group_size_rows",
                                  "group_size_units", "nn_train_dist"]
    assert rows["unique_id"].tolist() == [m[0] for m in META if m[6] == "0"]     # meta order, pose row dropped
    gid = rows["group_id"]
    # "KINASE ALPHA" / "Kinase-alpha" merge P10001 and P10002; BindingNet rows join their template's group
    assert gid[["1aaa", "1aab", "bn1", "bn2"]].nunique() == 1
    assert gid["1aab"] == "uniprot:P10002"                           # the member UniProt with most rows
    assert rows.loc["1aab", "group_size_rows"] == 4 and rows.loc["1aab", "group_size_units"] == 3
    # INTEGRASE maps to 5 UniProts: generic, so its UniProts stay apart
    assert gid[["1bbb", "1bbc", "1bbd", "1bbe", "1bbf"]].nunique() == 5
    generic = summary["name_merge"]["generic_names"]
    assert [g["name"] for g in generic] == ["INTEGRASE"] and generic[0]["n_uniprots_index"] == 5
    assert generic[0]["meta_rows"] == 5
    # rows without a UniProt: "single:<anchor>"; the BindingDB row on the same structure joins the PDBbind row
    assert gid["1zzz"] == gid["bd1"] == "single:1zzz"
    assert rows.loc["1zzz", "group_size_rows"] == 2 and rows.loc["1zzz", "group_size_units"] == 2
    assert gid[["1ccc", "1ddd", "bd2"]].nunique() == 1 and gid["1ccc"] == "uniprot:P30001"
    assert summary["groups"] == 1 + 5 + 1 + 1 and summary["groups_single"] == 1
    assert summary["groups_with_ge2_uniprots"] == 1 and summary["uniprots_in_multi_uniprot_groups"] == 2
    assert summary["largest_groups"][0]["names_top"] == {"KINASE ALPHA": 2}   # distinct structures
    assert summary["npz_params"] == {"pocket_mode": "sphere", "radius": 10.0}
    # a higher generic threshold lets INTEGRASE merge its UniProts
    rows6, summary6 = _run(tmp_path, "--generic-min-uniprots", "6")
    assert rows6["group_id"][["1bbb", "1bbc", "1bbd", "1bbe", "1bbf"]].nunique() == 1
    assert summary6["name_merge"]["generic_names_skipped"] == 0 and summary6["groups"] == 4


def test_nn_train_dist_is_to_another_group(tmp_path):
    rows, summary = _run(tmp_path)
    dist = rows["nn_train_dist"]
    assert np.isnan(dist[["1ddd", "bd2"]]).all()            # absent from the npz / featurisation failed
    assert np.isfinite(dist.drop(["1ddd", "bd2"])).all()
    # brute force on the module's embedding
    meta = _meta()
    meta = meta[meta["kind"] == "0"].reset_index(drop=True)
    emb, _ = pg.load_embedding(str(tmp_path / "zernike.npz"), meta["unique_id"].to_numpy())
    group = pd.factorize(rows["group_id"])[0]
    train = (meta["split"] == "train").to_numpy()
    have = np.isfinite(emb).all(axis=1)
    sim = emb.astype(np.float64) @ np.nan_to_num(emb).astype(np.float64).T
    for i in np.flatnonzero(have):
        cand = np.flatnonzero(train & have & (group != group[i]))
        assert dist.iloc[i] == pytest.approx(1 - sim[i, cand].max(), abs=1e-5)
    # 1aab's nearest training pocket is its near copy bn1 (same group), so it does not count; its distance
    # is to the nearest other-group train pocket
    d_copy = 1 - sim[1, meta.index[meta["unique_id"] == "bn1"][0]]
    assert d_copy < 0.01 and dist["1aab"] > 0.5
    assert summary["per_split"]["test"]["frac_nearest_train_in_other_group"] == 0.0
    # 1ccc's same-group train pocket is 1ddd (no descriptor), and its near copies 1zzz/bd1 are another group
    assert dist["1ccc"] < 0.01
    assert summary["per_split"]["test"]["nn_train_dist"]["n"] == 1
    assert summary["per_split"]["valid"]["frac_rows_whose_group_has_other_train_rows"] == 0.0


def test_nearest_train_other_group_blocks_and_edge_cases():
    rng = np.random.default_rng(3)
    emb = pg.unit_rows(rng.normal(size=(50, 8))).astype(np.float32)
    emb[7] = np.nan
    group = rng.integers(0, 6, size=50)
    train = rng.uniform(size=50) < 0.6
    d1, o1 = pg.nearest_train_other_group(emb, group, train, block=3)
    d2, o2 = pg.nearest_train_other_group(emb, group, train, block=1000)
    np.testing.assert_allclose(d1, d2, equal_nan=True)
    np.testing.assert_array_equal(o1, o2)
    assert np.isnan(d1[7]) and np.isnan(o1[7])
    for i in np.flatnonzero(np.isfinite(d1)):
        cand = np.flatnonzero(train & (group != group[i]) & (np.arange(50) != 7))
        assert d1[i] == pytest.approx(1 - (emb[cand] @ emb[i]).max(), abs=1e-6)
        anyc = np.flatnonzero(train & (np.arange(50) != i) & (np.arange(50) != 7))
        assert o1[i] == float(group[anyc[np.argmax(emb[anyc] @ emb[i])]] != group[i])
    # one group only: no other-group train pocket
    d, o = pg.nearest_train_other_group(emb, np.zeros(50, dtype=int), train)
    assert np.isnan(d).all() and np.nanmax(o) == 0.0
    d, o = pg.nearest_train_other_group(emb, group, np.zeros(50, dtype=bool))
    assert np.isnan(d).all() and np.isnan(o).all()


def test_pocket_embedding_standardises_and_signed_sqrt():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(30, 7, K)) * 3 + 1
    g = rng.normal(size=(30, 21, K)) * 10
    f[:, 0, 0] = 2.0                                    # constant column keeps scale 1 (centred to 0)
    emb = pg.pocket_embedding(f, g)
    x = np.concatenate([f.reshape(30, -1), (np.sign(g) * np.sqrt(np.abs(g))).reshape(30, -1)], axis=1)
    sd = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1)
    np.testing.assert_allclose(emb, z / np.linalg.norm(z, axis=1, keepdims=True), atol=1e-6)
    assert emb.dtype == np.float32 and emb.shape == (30, 7 * K + 21 * K)


def test_upgma_merge_matches_scipy_average_linkage():
    rng = np.random.default_rng(5)
    vec = pg.unit_rows(rng.normal(size=(14, 6)))
    counts = rng.integers(1, 4, size=14)
    expanded = np.repeat(vec, counts, axis=0)           # a group of c identical units
    owner = np.repeat(np.arange(14), counts)
    z = linkage(expanded, method="average", metric="cosine")
    for t in (0.3, 0.8, 1.0, 1.2):
        ref = fcluster(z, t, criterion="distance")
        ref_groups = np.array([ref[owner == k][0] for k in range(14)])
        labels, merges = pg.upgma_merge(vec, counts, t)
        same = labels[:, None] == labels[None, :]
        assert (same == (ref_groups[:, None] == ref_groups[None, :])).all(), t
        assert merges == 14 - len(np.unique(labels))
    labels, merges = pg.upgma_merge(vec, np.array([0] + [1] * 13), 10.0)   # a count-0 group never merges
    assert merges == 12 and (labels == labels[0]).sum() == 1


def test_descriptor_merge_optional(tmp_path):
    rows, summary = _run(tmp_path)
    assert rows.loc["1ccc", "group_id"] != rows.loc["1zzz", "group_id"]
    assert summary["descriptor_merge"]["merge_cosine"] is None
    rows_m, summary_m = _run(tmp_path, "--merge-cosine", "0.1")
    # 1zzz/bd1 (no UniProt) sit on 1ccc's pocket: the groups merge and keep the UniProt group's id
    assert rows_m["group_id"][["1ccc", "1ddd", "bd2", "1zzz", "bd1"]].unique().tolist() == ["uniprot:P30001"]
    assert summary_m["descriptor_merge"]["merges"] == 1 and summary_m["groups"] == summary["groups"] - 1
    # random pockets stay apart
    assert rows_m["group_id"][["1bbb", "1bbc", "1bbd", "1bbe", "1bbf"]].nunique() == 5
    # after the merge 1ccc's near copies are in its own group, so its nearest other-group pocket is far
    assert rows_m.loc["1ccc", "nn_train_dist"] > 0.5


def test_cli_index_dir_matches_tgz(tmp_path):
    rows_tgz, _ = _run(tmp_path)
    _, index_dir = _write_inputs(tmp_path)
    out = tmp_path / "dir" / "groups.csv"
    pg.main(["--meta", str(tmp_path / "meta.csv"), "--pdbbind-index-dir", str(index_dir),
             "--zernike", str(tmp_path / "zernike.npz"), "--out", str(out)])
    rows_dir = pd.read_csv(out, keep_default_na=False, na_values=[""]).set_index("unique_id", drop=False)
    pd.testing.assert_frame_equal(rows_dir, rows_tgz)
    assert (tmp_path / "dir" / "groups_summary.json").exists()
    with pytest.raises(SystemExit):
        pg.main(["--meta", str(tmp_path / "meta.csv"), "--zernike", str(tmp_path / "zernike.npz"),
                 "--out", str(out)])
