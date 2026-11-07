"""Tests for src/build_zernike_side.py (arm P): descriptor sources per kind, masks, train-only standardisation,
the row lookup, the CLI, and (when the local copies exist) a run on the real zernike.npz + arm-1 meta."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import build_zernike_side as bzs

REAL_NPZ = ("/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/"
            "zernike_ab/raw/zernike.npz")
REAL_META = ("/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/"
             "zernike_analysis/data/paired_meta.csv")


def _fake_npz(path, uids, seed=0, failed=()):
    """zernike.npz layout: unique_id, source, ok, pocket_F / ligand_F [N, 7, 49] (failed rows NaN)."""
    rng = np.random.default_rng(seed)
    n = len(uids)
    pocket = rng.normal(5.0, 2.0, size=(n, 7, 49)).astype(np.float32)
    ligand = rng.normal(-1.0, 0.5, size=(n, 7, 49)).astype(np.float32)
    ok = np.ones(n, dtype=bool)
    for k in failed:
        ok[k] = False
        pocket[k] = np.nan
        ligand[k] = np.nan
    np.savez_compressed(path, unique_id=np.array(uids), source=np.array(["pdbbind"] * n), ok=ok,
                        pocket_F=pocket, ligand_F=ligand, pocket_G=np.zeros((n, 21, 49), np.float32))
    return pocket.reshape(n, -1), ligand.reshape(n, -1)


def _fake_meta():
    """Kind-0 rows: 4 PDBbind train crystals (p0..p3), one valid crystal (p4), a BindingNet train row (bn0) and a
    test row (t0); augmented rows of p0 (kinds 1-7) and of p4 (valid: kinds 3, 4); one row whose parent is
    missing."""
    base = [("p0", "train"), ("p1", "train"), ("p2", "train"), ("p3", "train"), ("p4", "valid"),
            ("bn0", "train"), ("t0", "test")]
    rows = [{"row": r, "unique_id": u, "split": s, "kind": 0, "parent": r, "series_id": r, "lig_id": 10 + r}
            for r, (u, s) in enumerate(base)]
    aug = [("p0::pos0", 1, 0, "train"), ("p0::neg0", 2, 0, "train"), ("p0::tpos0", 3, 0, "train"),
           ("p0::twin0", 4, 0, "train"), ("p0::twin1", 4, 0, "train"), ("p0::epos0", 5, 0, "train"),
           ("p0::cedit0", 6, 0, "train"), ("p0::xedit0", 7, 0, "train"),
           ("p4::tpos0", 3, 4, "aug_valid"), ("p4::twin0", 4, 4, "aug_valid"), ("p9::pos0", 1, 99, "train")]
    for k, (u, kind, parent, split) in enumerate(aug):
        r = len(base) + k
        rows.append({"row": r, "unique_id": u, "split": split, "kind": kind, "parent": parent,
                     "series_id": parent, "lig_id": 10 + parent})
    return pd.DataFrame(rows)


def _twins():
    # p0's twins carry the ligands of p2 and bn0; p4's twin a ligand missing from the npz
    return pd.Series({"p0::twin0": "p2", "p0::twin1": "bn0", "p4::twin0": "zz9", "p4::tpos0": "p4",
                      "p0::tpos0": "p0"})


def test_descriptor_sources_and_masks(tmp_path):
    uids = ["p0", "p1", "p2", "p3", "p4", "bn0", "t0"]
    pocket, ligand = _fake_npz(tmp_path / "z.npz", uids, failed=(3,))  # p3: featurisation failed
    meta = _fake_meta()
    pos, pd_, ld_ = bzs.read_descriptors(str(tmp_path / "z.npz"))
    arrays, summary = bzs.build(meta, pos, pd_, ld_, _twins())
    src = dict(zip(arrays["unique_id"], zip(arrays["pocket_src"], arrays["ligand_src"])))
    assert src["p0"] == ("p0", "p0") and src["bn0"] == ("bn0", "bn0")
    assert src["p0::pos0"] == src["p0::neg0"] == src["p0::tpos0"] == src["p0::epos0"] == ("p0", "p0")
    assert src["p0::twin0"] == ("p0", "p2") and src["p0::twin1"] == ("p0", "bn0")  # the twin's own ligand
    assert src["p0::cedit0"] == src["p0::xedit0"] == ("p0", "")                     # edited ligand: masked
    assert src["p4::twin0"] == ("p4", "")       # twin ligand not in the npz
    assert src["p9::pos0"] == ("", "")          # parent row missing
    assert src["p3"] == ("", "")                # failed row: NaN descriptor -> masked
    i = list(arrays["unique_id"]).index
    assert np.array_equal(arrays["pocket_F"][i("p0::twin0")], pocket[0])
    assert np.array_equal(arrays["ligand_F"][i("p0::twin0")], ligand[2])
    assert np.isnan(arrays["ligand_F"][i("p0::cedit0")]).all() and not arrays["mask_ligand"][i("p0::cedit0")]
    assert arrays["mask_pocket"].sum() == 16 and arrays["mask_ligand"].sum() == 13
    assert summary["source_not_in_npz"] == {"pocket": 0, "ligand": 1}
    assert summary["ligand_masked_edit"] == 2 and summary["no_parent_row"] == 1
    assert summary["coverage_by_kind"]["4"] == {"rows": 3, "pocket": 3, "ligand": 2}
    for col in ("row", "kind", "parent", "series_id", "lig_id"):
        assert np.array_equal(arrays[col], meta[col].to_numpy())
    # kind-4 rows need the twins table
    with pytest.raises(ValueError, match="twins-meta"):
        bzs.build(meta, pos, pd_, ld_, None)


def test_zernike_side_standardisation(tmp_path):
    """mean / sd from the kind-0 train rows with a descriptor only (not valid / test / augmented rows), saved in
    the side file; load_side standardises with them and zeroes masked rows."""
    uids = ["p0", "p1", "p2", "p3", "p4", "bn0", "t0"]
    pocket, ligand = _fake_npz(tmp_path / "z.npz", uids, failed=(3,))
    meta = _fake_meta()
    meta.to_csv(tmp_path / "meta.csv", index=False)
    twins = pd.DataFrame({"unique_id": _twins().index, "twin_unique_id": _twins().values, "kind": 4})
    twins.to_csv(tmp_path / "twins_meta.csv", index=False)
    out = tmp_path / "side" / "paired_zernike.npz"
    summary = bzs.main(["--meta", str(tmp_path / "meta.csv"), "--zernike", str(tmp_path / "z.npz"),
                        "--twins-meta", str(tmp_path / "twins_meta.csv"), "--out", str(out)])
    assert summary["train_rows_for_standardisation"] == {"pocket": 4, "ligand": 4}  # p0, p1, p2, bn0 (p3 failed)
    side = bzs.load_side(str(out))
    train = [0, 1, 2, 5]  # npz positions of p0, p1, p2, bn0 (p3 failed; p4 valid, t0 test excluded)
    assert np.allclose(side["pocket_mean"], pocket[train].mean(axis=0), atol=1e-5)
    assert np.allclose(side["pocket_sd"], pocket[train].std(axis=0), atol=1e-5)
    assert np.allclose(side["ligand_mean"], ligand[train].mean(axis=0), atol=1e-5)
    # augmented rows (several copies of p0's pocket) do not weigh in
    assert not np.allclose(side["pocket_mean"], np.nanmean(side["pocket_F"][side["mask_pocket"]], axis=0))
    i = list(side["unique_id"]).index
    expect = (pocket[4] - side["pocket_mean"]) / side["pocket_sd"]
    assert np.allclose(side["pocket_std"][i("p4")], expect, atol=1e-5)
    assert np.all(side["ligand_std"][i("p0::cedit0")] == 0.0) and np.isfinite(side["pocket_std"]).all()
    # standardised train rows: mean 0, sd 1 per column
    tr = (side["split"] == "train") & (side["kind"] == 0) & side["mask_pocket"]
    assert np.allclose(side["pocket_std"][tr].mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(side["pocket_std"][tr].std(axis=0), 1.0, atol=1e-4)
    # row lookup (row ids need not be positions)
    assert np.array_equal(bzs.side_index(side, [0, 17, 999, -1]), [0, 17, -1, -1])
    meta_json = json.loads(str(side["summary"]))
    assert meta_json["twins_meta"].endswith("twins_meta.csv") and str(side["meta_path"]).endswith("meta.csv")
    # a constant column keeps sd 1
    const = np.zeros((3, 343), np.float32)
    mean, sd = bzs.standardisation(const, np.ones(3, bool), np.ones(3, bool))
    assert np.all(sd == 1.0) and np.all(mean == 0.0)


def test_arm1_meta_without_kind(tmp_path):
    uids = ["a", "b", "c"]
    pocket, _ = _fake_npz(tmp_path / "z.npz", uids)
    pd.DataFrame({"row": [0, 1, 2, 3], "unique_id": ["a", "b", "c", "d"], "split": ["train", "train", "test", "train"],
                  "series_id": [0, 0, 1, 2], "lig_id": [0, 1, 2, 3]}).to_csv(tmp_path / "meta.csv", index=False)
    out = tmp_path / "side.npz"
    bzs.main(["--meta", str(tmp_path / "meta.csv"), "--zernike", str(tmp_path / "z.npz"), "--out", str(out)])
    side = bzs.load_side(str(out))
    assert side["kind"].tolist() == [0, 0, 0, 0] and side["parent"].tolist() == [0, 1, 2, 3]
    assert side["mask_pocket"].tolist() == [True, True, True, False]  # "d" is not in the npz
    assert np.allclose(side["pocket_mean"], pocket[[0, 1]].mean(axis=0), atol=1e-5)
    with pytest.raises(SystemExit):
        bzs.main(["--meta", str(tmp_path / "meta.csv"), "--zernike", str(tmp_path / "absent.npz"), "--out", str(out)])


@pytest.mark.skipif(not (os.path.isfile(REAL_NPZ) and os.path.isfile(REAL_META)),
                    reason="local copies of the real zernike.npz / paired_meta.csv not present")
def test_real_side_file(tmp_path):
    out = tmp_path / "paired_zernike.npz"
    summary = bzs.main(["--meta", REAL_META, "--zernike", REAL_NPZ, "--out", str(out)])
    side = bzs.load_side(str(out))
    meta = pd.read_csv(REAL_META, usecols=["row", "split"])
    n = len(meta)
    assert len(side["row"]) == n and np.array_equal(side["row"], meta["row"].to_numpy())
    with np.load(REAL_NPZ, allow_pickle=False) as z:
        n_ok = int(z["ok"].sum())
    assert side["mask_pocket"].sum() == side["mask_ligand"].sum() == n_ok  # failed rows masked, all others found
    assert summary["source_not_in_npz"] == {"pocket": 0, "ligand": 0}
    tr = (side["split"] == "train") & side["mask_pocket"]
    sel = tr & (side["pocket_sd"] > 0).all()
    assert np.allclose(side["pocket_std"][sel].mean(axis=0), 0.0, atol=1e-3)
    assert side["pocket_F"].shape == (n, 343) and side["pocket_F"].dtype == np.float32
