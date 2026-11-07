"""Tests for src/paired_data.py: series-aware sampler, pair index, PairedDataset round trip."""
import numpy as np
import pytest
import torch

from paired_data import SOURCE_ID, SeriesBatchSampler, pair_index


def _series_array(n_items=1000, n_series=120, seed=0):
    """Series labels for n_items over n_series with sizes in 1..30 (both ends present), shuffled."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 31, n_series)
    sizes[0], sizes[1] = 1, 30
    while sizes.sum() > n_items:
        k = rng.integers(2, n_series)
        if sizes[k] > 1:
            sizes[k] -= 1
    while sizes.sum() < n_items:
        k = rng.integers(2, n_series)
        if sizes[k] < 30:
            sizes[k] += 1
    series = np.repeat(np.arange(n_series), sizes)
    return rng.permutation(series), sizes


def _epoch(sampler, epoch):
    sampler.set_epoch(epoch)
    return [list(b) for b in sampler]


def test_sampler_covers_every_index_once():
    series, sizes = _series_array()
    assert len(series) == 1000 and sizes.min() == 1 and sizes.max() == 30
    sampler = SeriesBatchSampler(series, batch_size=128, chunk=8, seed=0)
    batches = _epoch(sampler, 0)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1000))
    assert len(sampler) == len(batches)
    assert all(len(b) <= 128 for b in batches[:-1])
    assert len(batches[-1]) <= 129
    # same epoch is deterministic, a new epoch reshuffles
    assert _epoch(sampler, 0) == batches
    other = [i for b in _epoch(sampler, 1) for i in b]
    assert sorted(other) == list(range(1000))
    assert other != flat


def test_sampler_no_singleton_batch():
    series, _ = _series_array()
    for seed in (0, 1, 2):
        sampler = SeriesBatchSampler(series, batch_size=128, chunk=8, seed=seed)
        for epoch in range(20):
            assert all(len(b) > 1 for b in _epoch(sampler, epoch))
    # 129 singleton series: a trailing singleton must be appended to the previous batch
    sampler = SeriesBatchSampler(np.arange(129), batch_size=128, chunk=8, seed=0)
    batches = _epoch(sampler, 0)
    assert [len(b) for b in batches] == [129]
    # small batches with a chunk larger than the batch never produce a singleton
    sampler = SeriesBatchSampler(series[:101], batch_size=4, chunk=8, seed=3)
    for epoch in range(20):
        batches = _epoch(sampler, epoch)
        assert all(len(b) > 1 for b in batches)
        assert sorted(i for b in batches for i in b) == list(range(101))


def test_sampler_chunks_keep_series_together():
    series, sizes = _series_array()
    sampler = SeriesBatchSampler(series, batch_size=128, chunk=8, seed=0)
    for epoch in range(3):
        batches = _epoch(sampler, epoch)
        for s in np.flatnonzero(sizes >= 8):
            best = max(int(np.sum(series[b] == s)) for b in batches)
            assert best >= 8, (epoch, s, best)


def test_pair_index_skips_same_ligand():
    series = torch.tensor([0, 0, 0, 1, 1])
    lig = torch.tensor([5, 5, 6, 7, 8])
    i, j = pair_index(series, lig)
    assert i.dtype == torch.long and j.dtype == torch.long
    assert set(zip(i.tolist(), j.tolist())) == {(0, 2), (1, 2), (3, 4)}
    assert bool((i < j).all())
    # no pairs at all: one member per series
    i, j = pair_index(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    assert i.numel() == 0 and j.numel() == 0


def test_paired_dataset_roundtrip(tmp_path):
    pytest.importorskip("torch_geometric")
    import pandas as pd

    from paired_data import PairedDataset

    rng = np.random.default_rng(0)

    def graph(n_atoms, edges):
        feats = [rng.normal(size=5) for _ in range(n_atoms)]
        edge_index = [[a, b] for a, b in edges]
        edge_attr = [[1.0, 0.0, 0.0, 0.0] for _ in edges]
        return (n_atoms, feats, edge_index, edge_attr)

    graphs = {
        "g_a": graph(3, [(0, 1), (1, 0), (1, 2), (2, 1)]),
        "g_b": graph(1, []),  # single heavy atom, no bonds
        "g_c": graph(2, [(0, 1), (1, 0)]),
    }
    rows = pd.DataFrame({
        "row": [4, 7, 9],
        "unique_id": ["g_a", "g_b", "g_c"],
        "source": ["pdbbind", "bindingnet", "bindingdb"],
        "pK": [5.5, 6.25, 7.0],
        "series_id": [2, 2, 0],
        "lig_id": [10, 11, 3],
    })
    root = str(tmp_path / "data")
    built = PairedDataset.build(root, "train", rows, graphs)
    assert len(built) == 3

    ds = PairedDataset(root, "train")
    assert len(ds) == 3
    assert ds.processed_file_names == ["paired_train.pt"]
    for k, uid in enumerate(rows.unique_id):
        d = ds[k]
        n_atoms, feats, edge_index, edge_attr = graphs[uid]
        assert d.x.shape == (n_atoms, 5) and d.x.dtype == torch.float
        assert torch.allclose(d.x, torch.tensor(np.array(feats), dtype=torch.float))
        assert d.edge_index.shape == (2, len(edge_index)) and d.edge_index.dtype == torch.long
        assert d.edge_attr.shape == (len(edge_index), 4)
        assert d.y.tolist() == [rows.pK[k]]
        assert d.row.tolist() == [rows.row[k]]
        assert d.series.tolist() == [rows.series_id[k]]
        assert d.lig.tolist() == [rows.lig_id[k]]
        assert d.src.tolist() == [SOURCE_ID[rows.source[k]]]
        for name in ("row", "series", "lig", "src"):
            assert getattr(d, name).dtype == torch.long

    # loading never rebuilds: a missing split is an error, not an empty dataset
    with pytest.raises(FileNotFoundError):
        PairedDataset(root, "valid")
