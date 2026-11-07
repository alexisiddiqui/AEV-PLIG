"""Batch plans, capacities, host batch assembly and the store-backed batch source."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aevplig_jax import batching
from aevplig_jax.aev import radial_aev_jax, radial_aev_numpy
from aevplig_jax.batching import Batch, Capacities, StoreSource, assemble, capacities, plan_epoch
from aevplig_jax.loading import BatchSource, LoaderConfig, Prefetcher
from aevplig_jax.pdb_atoms import PocketArrays, typed_within
from aevplig_jax.rdkit_features import LigandArrays
from aevplig_jax.store import Record, Store
from test_store import LIG_A, PROT_A, SPECIES_A, make_record, random_record, write_store

SRC = Path(__file__).resolve().parents[1] / "src"
SINGLE, AROMATIC, DOUBLE, TRIPLE = np.eye(4, dtype=np.float32)  # bond one-hot over [1, 12, 2, 3]
BATCH_FIELDS = ("x_lig", "node_graph", "node_valid", "nbr", "nbr_mask", "nbr_feat", "lig_xyz",
                "env_xyz", "env_species", "aev_nbr", "aev_mask", "y", "valid", "n_real")
NODE_FIELDS = ("x_lig", "lig_xyz", "nbr", "nbr_mask", "nbr_feat", "aev_nbr", "aev_mask")


def ligand_record(rid, lig_xyz, bonds, prot_xyz=(), species=(), label=6.0) -> Record:
    """A record with undirected ``bonds`` ``(i, j, one_hot)`` stored in both directions, sorted by
    (src, dst) as ``ligand_arrays`` stores them, and node features that differ on every row."""
    lig_xyz = np.asarray(lig_xyz, dtype=np.float32).reshape(-1, 3)
    edges = sorted([(i, j, tuple(t)) for i, j, t in bonds] + [(j, i, tuple(t)) for i, j, t in bonds])
    n_lig, n_prot = len(lig_xyz), len(prot_xyz)
    lig = LigandArrays(
        xyz=lig_xyz,
        feat=(np.arange(n_lig * 15, dtype=np.float32).reshape(n_lig, 15) + 1000 * len(rid)),
        bond_src=np.array([e[0] for e in edges], dtype=np.int32),
        bond_dst=np.array([e[1] for e in edges], dtype=np.int32),
        bond_type=np.array([e[2] for e in edges], dtype=np.float32).reshape(-1, 4),
    )
    pocket = PocketArrays(
        xyz=np.asarray(prot_xyz, dtype=np.float32).reshape(n_prot, 3),
        species=np.asarray(species, dtype=np.int8), res=np.zeros(n_prot, np.int32),
        resname=np.zeros(n_prot, np.int8), atom=np.zeros(n_prot, np.int16),
        chain=np.zeros(n_prot, np.int8), bfactor=np.zeros(n_prot, np.float16),
    )
    return Record(id=rid, lig=lig, pocket=pocket, label=label, source="synthetic", split="train")


def random_records(rng, n) -> list[Record]:
    """``n`` random records plus one whose ligand is far from every pocket atom (A = 0)."""
    records = [random_record(rng, f"r{i}") for i in range(n)]
    records.append(make_record(rng, "far", [[60.0, 60.0, 60.0], [61.5, 60.0, 60.0]], PROT_A,
                               SPECIES_A, 2, label=4.0))
    return records


def test_plan_epoch_is_permutation_and_deterministic():
    ids = np.array([f"c{i}" for i in range(300)])
    plan = plan_epoch(ids, seed=100, epoch=3)
    assert [len(chunk) for chunk in plan] == [128, 128, 44]
    order = np.random.default_rng([100, 3]).permutation(300)
    np.testing.assert_array_equal(np.concatenate(plan), ids[order])
    assert sorted(np.concatenate(plan)) == sorted(ids)
    again = plan_epoch(ids, seed=100, epoch=3)
    assert all(np.array_equal(a, b) for a, b in zip(plan, again))
    assert not np.array_equal(np.concatenate(plan_epoch(ids, 100, 4)), np.concatenate(plan))
    assert not np.array_equal(np.concatenate(plan_epoch(ids, 123, 3)), np.concatenate(plan))
    assert [len(chunk) for chunk in plan_epoch(ids, 100, 3, batch=100)] == [100, 100, 100]


def test_self_loop_feature_is_mean_of_incoming():
    # Atom 0 has a single bond to atom 1 and a double bond to atom 2; atom 3 is isolated.
    first = ligand_record("first", np.arange(12).reshape(4, 3),
                          [(0, 1, SINGLE), (0, 2, DOUBLE)])
    # A second complex shifts node indices by 4: atom 1 (node 5) has bonds from atoms 0 and 2.
    second = ligand_record("second", np.arange(9).reshape(3, 3),
                           [(0, 1, AROMATIC), (1, 2, TRIPLE)])
    # A third complex is a single atom without bonds (node 7).
    third = ligand_record("third", [[0.0, 0.0, 0.0]], [])
    cap = Capacities(ncap=256, mcap=1024, kb=4, ka=32, batch=3)
    batch = assemble([first, second, third], cap, y_mean=0.0, y_sd=1.0)

    np.testing.assert_array_equal(batch.nbr_feat[0, 0], [0.5, 0.0, 0.5, 0.0])
    np.testing.assert_array_equal(batch.nbr[0], [0, 1, 2, 0])
    np.testing.assert_array_equal(batch.nbr_mask[0], [1, 1, 1, 0])
    np.testing.assert_array_equal(batch.nbr_feat[0, 1:], [SINGLE, DOUBLE, np.zeros(4)])
    np.testing.assert_array_equal(batch.nbr_feat[1, 0], SINGLE)
    np.testing.assert_array_equal(batch.nbr[1], [1, 0, 1, 1])
    np.testing.assert_array_equal(batch.nbr_mask[1], [1, 1, 0, 0])
    # Isolated node: self-loop only, zero feature, masked slots point at itself.
    np.testing.assert_array_equal(batch.nbr_feat[3], np.zeros((4, 4)))
    np.testing.assert_array_equal(batch.nbr[3], [3, 3, 3, 3])
    np.testing.assert_array_equal(batch.nbr_mask[3], [1, 0, 0, 0])
    # Second complex, offset by 4 nodes.
    np.testing.assert_array_equal(batch.nbr[5], [5, 4, 6, 5])
    np.testing.assert_array_equal(batch.nbr_mask[5], [1, 1, 1, 0])
    np.testing.assert_array_equal(batch.nbr_feat[5, 0], [0.0, 0.5, 0.0, 0.5])
    np.testing.assert_array_equal(batch.nbr[7], [7, 7, 7, 7])
    np.testing.assert_array_equal(batch.nbr_mask[7], [1, 0, 0, 0])
    np.testing.assert_array_equal(batch.nbr_feat[7], np.zeros((4, 4)))
    np.testing.assert_array_equal(batch.node_graph[:8], [0, 0, 0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(batch.node_valid[:9], [1, 1, 1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(batch.x_lig[:4], first.lig.feat)
    np.testing.assert_array_equal(batch.x_lig[4:7], second.lig.feat)
    # No complex has pocket atoms: environment rows stay zero (species 0) and every AEV slot masked.
    assert not batch.env_xyz.any() and not batch.env_species.any()
    assert not batch.aev_mask.any() and not batch.aev_nbr.any()


def test_padding_is_copies_with_zero_valid(rng):
    records = random_records(rng, 4)
    cap = Capacities(ncap=256, mcap=2048, kb=32, ka=320, batch=8)
    batch = assemble(records, cap, y_mean=6.0, y_sd=2.0)

    assert isinstance(batch, Batch) and Batch._fields == BATCH_FIELDS
    expected_shapes = {
        "x_lig": ((256, 15), np.float32), "node_graph": ((256,), np.int32),
        "node_valid": ((256,), np.float32), "nbr": ((256, 32), np.int32),
        "nbr_mask": ((256, 32), np.float32), "nbr_feat": ((256, 32, 4), np.float32),
        "lig_xyz": ((256, 3), np.float32), "env_xyz": ((2048, 3), np.float32),
        "env_species": ((2048,), np.int32), "aev_nbr": ((256, 320), np.int32),
        "aev_mask": ((256, 320), np.float32), "y": ((8,), np.float32), "valid": ((8,), np.float32),
    }
    for name, (shape, dtype) in expected_shapes.items():
        array = getattr(batch, name)
        assert array.shape == shape and array.dtype == dtype, name
    assert batch.n_real == 5

    sizes = [len(r.lig.xyz) for r in records]
    n = sum(sizes)
    assert n < 256
    np.testing.assert_array_equal(batch.node_graph[:n], np.repeat(np.arange(5), sizes))
    np.testing.assert_array_equal(batch.node_valid[:n], 1.0)
    np.testing.assert_array_equal(batch.node_graph[n:], 8)  # the dump segment
    np.testing.assert_array_equal(batch.node_valid[n:], 0.0)
    source = np.arange(n, 256) % n
    for name in NODE_FIELDS:
        array = getattr(batch, name)
        np.testing.assert_array_equal(array[n:], array[source], err_msg=name)

    env_xyz = np.concatenate([r.pocket.xyz[typed_within(r.pocket, r.lig.xyz)] for r in records])
    env_species = np.concatenate([r.pocket.species[typed_within(r.pocket, r.lig.xyz)]
                                  for r in records])
    m = len(env_xyz)
    assert 0 < m < 2048
    np.testing.assert_array_equal(batch.env_xyz[:m], env_xyz)
    np.testing.assert_array_equal(batch.env_species[:m], env_species)
    env_source = np.arange(m, 2048) % m
    np.testing.assert_array_equal(batch.env_xyz[m:], batch.env_xyz[env_source])
    np.testing.assert_array_equal(batch.env_species[m:], batch.env_species[env_source])

    np.testing.assert_array_equal(batch.valid, [1, 1, 1, 1, 1, 0, 0, 0])
    labels = np.array([r.label for r in records], dtype=np.float32)
    np.testing.assert_allclose(batch.y[:5], (labels - 6.0) / 2.0, rtol=1e-6)
    np.testing.assert_array_equal(batch.y[5:], 0.0)


def test_batch_aev_matches_per_complex_reference(rng):
    """AEV lists index the concatenated env arrays: the batch AEV of each complex's rows equals the
    brute-force AEV over that complex's whole pocket; the far ligand (A = 0) gets zero rows."""
    records = random_records(rng, 4)
    cap = Capacities(ncap=256, mcap=2048, kb=32, ka=320, batch=8)
    batch = assemble(records, cap, y_mean=0.0, y_sd=1.0)
    aev = np.asarray(radial_aev_jax(batch.lig_xyz, batch.env_xyz, batch.env_species,
                                    batch.aev_nbr, batch.aev_mask > 0))
    start = 0
    for record in records:
        rows = slice(start, start + len(record.lig.xyz))
        want = radial_aev_numpy(record.lig.xyz, record.pocket.xyz, record.pocket.species)
        np.testing.assert_allclose(aev[rows], want, atol=1e-5, err_msg=record.id)
        start = rows.stop
    far = slice(start - 2, start)
    assert records[-1].id == "far"
    np.testing.assert_array_equal(batch.aev_mask[far], 0.0)
    assert np.all((batch.aev_nbr >= 0) & (batch.aev_nbr < cap.mcap))


def _two_records():
    """"first": L 2, A 1, KA 1, in-degree 1. "second": L 3, A 3, KA 3, atom 0 has in-degree 2."""
    first = ligand_record("first", [[0, 0, 0], [1.5, 0, 0]], [(0, 1, SINGLE)],
                          prot_xyz=[[0, 0, 3.0]], species=[5])
    second = ligand_record("second", LIG_A, [(0, 1, SINGLE), (0, 2, DOUBLE)],
                           prot_xyz=PROT_A, species=SPECIES_A)
    return [first, second]


@pytest.mark.parametrize("field, value", [("ka", 2), ("kb", 2), ("ncap", 4), ("mcap", 3),
                                          ("batch", 1)])
def test_assemble_raises_when_capacity_exceeded(field, value):
    records = _two_records()
    cap = Capacities(ncap=8, mcap=8, kb=3, ka=3, batch=2)
    batch = assemble(records, cap, y_mean=0.0, y_sd=1.0)  # exactly at capacity in kb, ka, batch
    assert batch.valid.tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="second"):
        assemble(records, replace(cap, **{field: value}), y_mean=0.0, y_sd=1.0)


def test_capacities_cover_all_plans(tmp_path, rng, monkeypatch):
    records = random_records(rng, 49)
    store = Store(write_store(tmp_path / "store", records))
    ids = np.array(store.ids)
    rows = {rid: i for i, rid in enumerate(store.ids)}
    size = 16

    def plans_of(seeds, epochs):
        return [chunk for seed in seeds for epoch in range(epochs)
                for chunk in plan_epoch(ids, seed, epoch, batch=size)]

    def maxima(plans):  # largest per-batch total of L and of A
        return tuple(max(int(store.dims[[rows[i] for i in chunk], column].sum()) for chunk in plans)
                     for column in (0, 3))

    plans = plans_of([1, 2], 3)
    atoms, env = maxima(plans)
    degree = max(int(np.bincount(r.lig.bond_dst).max(initial=0)) for r in records)
    cap = capacities(store, ids, [1, 2], 3, batch=size)
    assert cap == Capacities(ncap=-(-atoms // 256) * 256, mcap=-(-env // 1024) * 1024,
                             kb=degree + 1, ka=-(-int(store.dims[:, 4].max()) // 32) * 32,
                             batch=size)
    assert cap.ncap > 256 and cap.mcap > 1024  # totals span several quanta
    assert Capacities.from_json(json.loads(json.dumps(cap.to_json()))) == cap

    # Without rounding the capacities are the plan maxima themselves, for any seeds and epochs.
    monkeypatch.setattr(batching, "NODE_QUANTUM", 1)
    monkeypatch.setattr(batching, "ENV_QUANTUM", 1)
    for seeds, epochs in (([1, 2], 3), ([2], 1)):
        exact = capacities(store, ids, seeds, epochs, batch=size)
        assert (exact.ncap, exact.mcap) == maxima(plans_of(seeds, epochs)), (seeds, epochs)

    for chunk in plans:  # full chunks of 16 and the short last chunk of 2
        batch = assemble([store.get(i) for i in chunk], cap, y_mean=0.0, y_sd=1.0)
        assert batch.n_real == len(chunk) and batch.valid.sum() == len(chunk)
    assert sorted({len(chunk) for chunk in plans}) == [2, 16]
    store.close()


def test_prefetcher_yields_in_order(tmp_path, rng):
    records = random_records(rng, 19)
    store = Store(write_store(tmp_path / "store", records))
    cap = Capacities(ncap=1024, mcap=8192, kb=32, ka=320, batch=4)
    specs = plan_epoch(np.array(store.ids), seed=7, epoch=0, batch=4)
    assert len(specs) == 5
    source = StoreSource(store, cap, y_mean=6.0, y_sd=2.0, workers=2)
    assert isinstance(source, BatchSource)
    seen = []
    for spec, batch in Prefetcher(source, specs, LoaderConfig(workers=2, prefetch=2)):
        seen.append(spec)
        want = assemble([store.get(i) for i in spec], cap, y_mean=6.0, y_sd=2.0)
        for name in Batch._fields:
            np.testing.assert_array_equal(getattr(batch, name), getattr(want, name), err_msg=name)
    assert [list(s) for s in seen] == [list(s) for s in specs]
    assert source.provenance() == {"store": str(store.path), "capacities": cap.to_json(),
                                   "y_mean": 6.0, "y_sd": 2.0, "workers": 2}
    source.close()
    store.close()


def test_batching_imports_no_jax_rdkit_or_pandas():
    code = ("import sys, aevplig_jax.batching, aevplig_jax.loading; "
            "print(sorted({'jax', 'rdkit', 'pandas', 'torch'} & set(sys.modules)))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"
