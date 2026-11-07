"""Pocket clustering on synthetic descriptors: two well-separated blobs, units, threshold rule, CLI."""
import json

import numpy as np
import pandas as pd
import pytest

import pocket_clusters as pc

K = 49


def _blobs(n_per=40, seed=0, sep=6.0, noise=0.3):
    """Two blobs in descriptor space; blob 0 holds UniProts P1/P2, blob 1 holds P3/P4."""
    rng = np.random.default_rng(seed)
    dim = 7 * K + 21 * K
    centres = rng.normal(size=(2, dim)) * sep
    x = np.concatenate([centres[0] + rng.normal(size=(n_per, dim)) * noise,
                        centres[1] + rng.normal(size=(n_per, dim)) * noise])
    f = x[:, :7 * K].reshape(-1, 7, K)
    g = x[:, 7 * K:].reshape(-1, 21, K)
    blob = np.repeat([0, 1], n_per)
    uniprot = np.where(blob == 0, np.where(np.arange(2 * n_per) % 2, "P1", "P2"),
                       np.where(np.arange(2 * n_per) % 2, "P3", "P4"))
    uid = np.array([f"c{i:03d}" for i in range(2 * n_per)])
    return uid, f, g, blob, uniprot


def test_pair_rate_toy():
    labels = np.array([1, 1, 2, 2, 2, 3])
    groups = np.array(["A", "A", "A", "B", "", "B"])
    rate, group_mean, n_pairs = pc.pair_rate(labels, groups)
    # A: 3 pairs, 1 co-clustered; B: 1 pair (rows 3, 5), 0 co-clustered; "" forms no pairs
    assert n_pairs == 4
    assert rate == pytest.approx(1 / 4)
    assert group_mean == pytest.approx((1 / 3 + 0) / 2)
    assert np.isnan(pc.pair_rate(labels, np.array([""] * 6))[0])


def test_medoid_picks_central_row():
    v = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.95, 0.05]])
    assert pc.medoid(v) in (1, 3)


def test_two_blobs_target_09_keeps_blobs_apart():
    """At the 0.9 rule the cut stops before the blobs merge; a few late-merging outliers may stay alone."""
    uid, f, g, blob, uniprot = _blobs()
    rows, summary = pc.cluster_pockets(uid, np.array(["pdbbind"] * len(uid)), uniprot, np.array([""] * len(uid)),
                                       f, g, pca_dims=50, target=0.9)
    assert summary["same_uniprot_rate_at_threshold"] >= 0.9
    assert 2 <= summary["n_clusters"] <= 6
    assert (rows.groupby("cluster_id").apply(lambda d: blob[d.index].min() == blob[d.index].max())).all()
    for b in (0, 1):
        assert rows.loc[blob == b, "cluster_id"].value_counts().iloc[0] >= 0.9 * 40
    assert np.isfinite(rows["nearest_other_cluster_dist"]).all()


def test_two_blobs_two_clusters():
    """With target 1.0 every same-UniProt pair must co-cluster: exactly the two blobs."""
    uid, f, g, blob, uniprot = _blobs()
    rows, summary = pc.cluster_pockets(uid, np.array(["pdbbind"] * len(uid)), uniprot, np.array([""] * len(uid)),
                                       f, g, pca_dims=50, target=1.0)
    assert summary["n_clusters"] == 2
    assert summary["same_uniprot_rate_at_threshold"] == 1.0
    # each blob is one cluster and the two blobs differ
    by_blob = rows.groupby(blob)["cluster_id"].nunique()
    assert (by_blob == 1).all()
    assert rows.loc[blob == 0, "cluster_id"].iloc[0] != rows.loc[blob == 1, "cluster_id"].iloc[0]
    assert set(rows["cluster_size"]) == {40}
    assert summary["clusters_mixing_ge2_uniprots_all_rows"] == 2
    # far from the other blob: cosine distance near 2 after centring, far above the cut
    assert rows["nearest_other_cluster_dist"].min() > summary["threshold"]
    assert rows["nearest_other_cluster_dist"].min() > 1.0
    # the reported curve is monotone in the threshold and contains the chosen cut
    rates = [c["same_uniprot_rate"] for c in summary["curve"]]
    assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))
    assert any(abs(c["threshold"] - summary["threshold"]) < 1e-12 for c in summary["curve"])


def test_threshold_is_smallest_reaching_target():
    uid, f, g, blob, uniprot = _blobs(seed=1)
    rows, summary = pc.cluster_pockets(uid, np.array(["pdbbind"] * len(uid)), uniprot, np.array([""] * len(uid)),
                                       f, g, target=0.9)
    below = [c for c in summary["curve"] if c["threshold"] < summary["threshold"] - 1e-9]
    assert all(c["same_uniprot_rate"] < 0.9 for c in below)


def test_series_units_stay_together():
    """Grouped-source rows sharing a series_key form one unit, hence one cluster, even when one of them
    sits in the other blob."""
    uid, f, g, blob, uniprot = _blobs()
    source = np.array(["pdbbind"] * len(uid), dtype=object)
    series = np.array([""] * len(uid), dtype=object)
    source[:6] = "bindingnet"
    series[:6] = "bindingnet:1abc"
    source[-2:] = "bindingnet"
    series[-2:] = "bindingnet:1abc"   # two blob-1 rows on the blob-0 template
    rows, summary = pc.cluster_pockets(uid, source, uniprot, series, f, g, target=0.9)
    assert rows.loc[series == "bindingnet:1abc", "cluster_id"].nunique() == 1
    assert summary["units"] == len(uid) - 8 + 1
    assert summary["units_by_source"] == {"bindingnet": 1, "pdbbind": len(uid) - 8}


def test_cli_roundtrip(tmp_path):
    uid, f, g, blob, uniprot = _blobs(n_per=25)
    ok = np.ones(len(uid), dtype=bool)
    ok[3] = False
    f = f.astype(np.float32)
    f[3] = np.nan
    np.savez_compressed(tmp_path / "z.npz", unique_id=uid, source=np.array(["pdbbind"] * len(uid)),
                        pocket_F=f, pocket_G=g.astype(np.float32), ok=ok)
    meta = pd.DataFrame({"unique_id": list(uid) + ["absent"], "source": "pdbbind",
                         "uniprot": list(uniprot) + ["------"], "series_key": "", "kind": "0"})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    out = tmp_path / "pocket_clusters.csv"
    pc.main(["--zernike", str(tmp_path / "z.npz"), "--meta", str(tmp_path / "meta.csv"), "--out", str(out)])
    rows = pd.read_csv(out, keep_default_na=False)
    assert list(rows.columns) == ["unique_id", "source", "uniprot", "cluster_id", "cluster_size",
                                  "nearest_other_cluster_dist", "unit_key"]
    assert rows["unique_id"].tolist() == meta["unique_id"].tolist()
    assert rows.set_index("unique_id").loc[["c003", "absent"], "cluster_id"].tolist() == [-1, -1]
    summary = json.loads((tmp_path / "pocket_clusters_summary.json").read_text())
    assert summary["rows_without_descriptor"] == 2 and summary["rows"] == 49
    assert summary["same_uniprot_rate_at_threshold"] >= 0.9
    assert summary["threshold_rule"] == "co-clustering target"
