"""Tests for src/twin_select.py on synthetic descriptor pools.

Background rows have N(0, 1) descriptor columns, so the pool standardisation is close to the identity
and the raw vectors written here keep their geometry after it."""
import time

import numpy as np
import pandas as pd
import pytest

import twin_select as ts

K = 49
COMPLEMENT = [1, 3, 2, 4, 6, 5]  # pocket channel that faces each ligand chemistry channel 1..6


def comp_raw(pocket):
    """Raw complement chemistry vector [6, K] of a pocket_F [7, K] (pos<->neg, donor<->acceptor)."""
    return pocket[COMPLEMENT]


def chem_cos(comp, c, rng):
    """A chemistry block [6, K] whose cosine with comp is c and whose norm is |comp| (a random direction
    orthogonal to comp fills the rest), so rankings by complementarity are set by c alone."""
    v = comp.ravel() / np.linalg.norm(comp)
    o = rng.normal(size=v.size)
    o -= (o @ v) * v
    o /= np.linalg.norm(o)
    return ((c * v + np.sqrt(max(0.0, 1.0 - c * c)) * o) * np.linalg.norm(comp)).reshape(comp.shape)


class PoolBuilder:
    """Collects rows, then writes the zernike.npz-like dict and the meta / groups tables."""

    def __init__(self, n_background=300, seed=0):
        self.rng = np.random.default_rng(seed)
        self.rows = []
        for i in range(n_background):
            self.add(f"bg{i:04d}", source=["bindingnet", "bindingdb", "pdbbind"][i % 3], group=f"g_bg{i % 40}",
                     lig_key=f"lk_bg{i}", anchor_pdb=f"t{i % 50}")

    def add(self, uid, source="bindingnet", split="train", group="g_other", lig_key=None, anchor_pdb="",
            shape=None, chem=None, pocket=None, logp=2.0, charge=0, arom_atoms=12.0, n_heavy=30, ok=True):
        r = self.rng
        self.rows.append(dict(
            uid=uid, source=source, split=split, group=group, lig_key=lig_key or f"lk_{uid}",
            anchor_pdb=anchor_pdb, ok=ok,
            shape=r.normal(size=K) if shape is None else np.asarray(shape, float),
            chem=r.normal(size=(6, K)) if chem is None else np.asarray(chem, float),
            pocket=r.normal(size=(7, K)) if pocket is None else np.asarray(pocket, float),
            logp=logp, charge=charge, arom=arom_atoms, n_heavy=n_heavy))
        return self.rows[-1]

    def near(self, shape, scale=0.05):
        return np.asarray(shape) + self.rng.normal(scale=scale, size=K)

    def desc(self):
        n = len(self.rows)
        lf = np.zeros((n, 7, K), np.float32)
        pf = np.zeros((n, 7, K), np.float32)
        lt = np.zeros((n, 7), np.float32)
        for i, row in enumerate(self.rows):
            lf[i, 0] = row["shape"]
            lf[i, 1:] = row["chem"]
            pf[i] = row["pocket"]
            q = row["charge"]
            lt[i] = [row["n_heavy"], row["logp"], 1.5 + max(q, 0), 1.5 + max(-q, 0), row["arom"], 2, 3]
        ok = np.array([row["ok"] for row in self.rows])
        lf[~ok] = np.nan
        pf[~ok] = np.nan
        lt[~ok] = np.nan
        return {"unique_id": np.array([row["uid"] for row in self.rows]),
                "source": np.array([row["source"] for row in self.rows]),
                "ok": ok, "ligand_F": lf, "pocket_F": pf, "ligand_total": lt,
                "n_ligand_atoms": np.array([row["n_heavy"] if row["ok"] else -1 for row in self.rows])}

    def meta(self, extra=()):
        rows = [{"unique_id": r["uid"], "source": r["source"], "split": r["split"], "lig_key": r["lig_key"],
                 "uniprot": "", "anchor_pdb": r["anchor_pdb"]} for r in self.rows]
        return pd.DataFrame(rows + list(extra))

    def groups(self, extra=()):
        return pd.DataFrame([{"unique_id": r["uid"], "group_id": r["group"]} for r in self.rows] + list(extra))

    def pool(self, meta_extra=(), groups_extra=()):
        return ts.build_pool(self.desc(), self.meta(meta_extra), self.groups(groups_extra))


def make_anchor(b, uid="1abc", **kw):
    kw.setdefault("source", "pdbbind")
    kw.setdefault("group", "g_anchor")
    kw.setdefault("anchor_pdb", uid)
    return b.add(uid, **kw)


def test_twin_prefers_same_shape_worst_chemistry():
    b = PoolBuilder()
    a = make_anchor(b)
    comp = comp_raw(a["pocket"])
    coeffs = {"near_p10": 1.0, "near_p05": 0.5, "near_0": 0.0, "near_m05": -0.5, "near_m08": -0.8,
              "near_m10": -1.0}
    for uid, c in coeffs.items():
        b.add(uid, shape=b.near(a["shape"]), chem=chem_cos(comp, c, b.rng))
    # worst chemistry of all, but a different shape: never a twin
    b.add("far_m10", chem=-comp)
    pool = b.pool()
    n_pool = len(pool.pool_rows)
    shape_frac = 6 / n_pool  # exactly the 6 near rows (the anchor's own row is excluded first)
    twins, info = ts.select_twins_with_info("1abc", pool, k=2, shape_frac=shape_frac)
    assert [t.twin_uid for t in twins] == ["near_m10", "near_m08"]
    assert info["reason"] == "ok" and info["n_selected"] == 2
    assert twins[0].complementarity < twins[1].complementarity < -0.5
    assert all(t.shape_dist < 0.05 for t in twins)
    assert [t.rank for t in twins] == [0, 1]
    # complementarity = cosine of the standardised, swapped pocket vector and the standardised chemistry
    lf = pool_stats_z(b, "ligand_F", pool)
    pf = pool_stats_z(b, "pocket_F", pool)
    ia, it = b_index(b, "1abc"), b_index(b, "near_m10")
    pc = pf[ia][COMPLEMENT].ravel()
    ch = lf[it][1:].ravel()
    assert twins[0].complementarity == pytest.approx(pc @ ch / np.linalg.norm(pc) / np.linalg.norm(ch), abs=1e-5)
    sa, st = lf[ia][0], lf[it][0]
    assert twins[0].shape_dist == pytest.approx(1 - sa @ st / np.linalg.norm(sa) / np.linalg.norm(st), abs=1e-5)
    assert info["anchor_complementarity"] == pytest.approx(
        pc @ lf[ia][1:].ravel() / np.linalg.norm(pc) / np.linalg.norm(lf[ia][1:]), abs=1e-5)
    # the plain interface returns the same choices
    assert [t.twin_uid for t in ts.select_twins("1abc", pool, k=2, shape_frac=shape_frac)] == \
        ["near_m10", "near_m08"]
    # a wider neighbourhood reaches the far row's shape only if it is within shape_frac: with all rows it wins
    twins_all = ts.select_twins("1abc", pool, k=1, shape_frac=1.0)
    assert twins_all[0].twin_uid in {"far_m10", "near_m10"}


def b_index(b, uid):
    return [r["uid"] for r in b.rows].index(uid)


def pool_stats_z(b, key, pool):
    """Independent z-scoring over the pool rows (split train, ok) of the builder's raw arrays."""
    d = b.desc()[key].astype(float)
    train = np.array([r["split"] == "train" and r["ok"] for r in b.rows])
    mu = d[train].mean(axis=0)
    sd = d[train].std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (d - mu) / sd


def test_twin_excludes_own_group_and_ligand():
    b = PoolBuilder()
    a = make_anchor(b, lig_key="pdbbind:AAA")
    comp = comp_raw(a["pocket"])
    worst = lambda: chem_cos(comp, -1.0, b.rng)  # noqa: E731
    # all of these are same-shape and have the worst chemistry, and all must be excluded
    b.add("same_group_bn", source="bindingnet", group="g_anchor", anchor_pdb="9zzz",
          shape=b.near(a["shape"]), chem=worst())
    b.add("same_ligkey_other_group", source="pdbbind", group="g_far", lig_key="pdbbind:AAA",
          anchor_pdb="2xyz", shape=b.near(a["shape"]), chem=worst())
    b.add("binder_of_group", source="bindingdb", group="g_far2", lig_key="bindingdb:777",
          anchor_pdb="5abc", shape=b.near(a["shape"]), chem=worst())
    b.add("template_row", source="bindingnet", group="g_mislabelled", anchor_pdb="1abc",
          shape=b.near(a["shape"]), chem=worst())
    # bindingdb:777 is a known binder of the anchor's group: a valid-split row without a descriptor
    b.add("binder_in_group_valid", source="bindingdb", split="valid", group="g_anchor",
          lig_key="bindingdb:777", anchor_pdb="5abc", ok=False)
    # allowed twins: same shape, slightly better chemistry
    b.add("ok1", shape=b.near(a["shape"]), chem=chem_cos(comp, -0.7, b.rng))
    b.add("ok2", shape=b.near(a["shape"]), chem=chem_cos(comp, -0.6, b.rng))
    pool = b.pool()
    frac = 2 / len(pool.pool_rows)  # the two nearest rows left after the exclusions
    excluded = {"same_group_bn", "same_ligkey_other_group", "binder_of_group", "template_row", "1abc"}
    twins, info = ts.select_twins_with_info("1abc", pool, k=2, shape_frac=frac)
    assert [t.twin_uid for t in twins] == ["ok1", "ok2"]
    assert info["n_excluded"] >= len(excluded)
    # nothing excluded leaks in even when every other row is a candidate
    twins_all = ts.select_twins("1abc", pool, k=50, shape_frac=1.0, max_dlogp=99, max_darom=99,
                                same_charge_sign=False, size_tol=99)
    chosen = {t.twin_uid for t in twins_all}
    assert not chosen & excluded
    assert all(t.twin_group_id != "g_anchor" for t in twins_all)
    assert all(t.twin_lig_key not in {"pdbbind:AAA", "bindingdb:777"} for t in twins_all)
    # the batched path agrees
    batch = ts.select_twins_batch(["1abc"], pool, k=2, shape_frac=frac)
    assert [t.twin_uid for t in batch["1abc"][0]] == ["ok1", "ok2"]


def test_template_exclusion_is_case_blind():
    b = PoolBuilder()
    a = make_anchor(b, anchor_pdb="1ABC ")
    b.add("upper_template", group="g_far", anchor_pdb=" 1abc", shape=b.near(a["shape"]),
          chem=chem_cos(comp_raw(a["pocket"]), -1.0, b.rng))
    pool = b.pool()
    assert "upper_template" not in {t.twin_uid for t in ts.select_twins(
        "1abc", pool, k=50, shape_frac=1.0, max_dlogp=99, max_darom=99, same_charge_sign=False, size_tol=99)}


def test_twins_have_distinct_lig_keys():
    b = PoolBuilder()
    a = make_anchor(b)
    comp = comp_raw(a["pocket"])
    b.add("dup_a", lig_key="bindingnet:CHEMBL1", group="g1", shape=b.near(a["shape"]),
          chem=chem_cos(comp, -1.0, b.rng))
    b.add("dup_b", lig_key="bindingnet:CHEMBL1", group="g2", shape=b.near(a["shape"]),
          chem=chem_cos(comp, -1.0, b.rng))
    b.add("other", lig_key="bindingnet:CHEMBL2", group="g3", shape=b.near(a["shape"]),
          chem=chem_cos(comp, -0.5, b.rng))
    pool = b.pool()
    twins = ts.select_twins("1abc", pool, k=2, shape_frac=3 / len(pool.pool_rows))
    assert len(twins) == 2 and twins[1].twin_uid == "other"
    assert twins[0].twin_uid in {"dup_a", "dup_b"}


def test_marginal_filters():
    b = PoolBuilder()
    a = make_anchor(b, logp=2.0, charge=0, arom_atoms=12.0, n_heavy=30)
    comp = comp_raw(a["pocket"])
    near = lambda c: dict(shape=b.near(a["shape"]), chem=chem_cos(comp, c, b.rng))  # noqa: E731
    b.add("too_big", n_heavy=37, **near(-1.0))           # +23% > 20%
    b.add("too_greasy", logp=3.6, **near(-0.98))           # dlogP 1.6 > 1.5
    b.add("too_aromatic", arom_atoms=24.0, **near(-0.96))  # 2 more rings > 1
    b.add("charged", charge=1, **near(-0.94))              # sign differs
    b.add("pass_edge", n_heavy=36, logp=0.6, arom_atoms=18.0, charge=0, **near(-0.5))  # +20%, -1.4, +1 ring
    b.add("pass_mild", n_heavy=27, logp=2.5, arom_atoms=6.0, **near(-0.4))
    pool = b.pool()
    frac = 6 / len(pool.pool_rows)
    twins, info = ts.select_twins_with_info("1abc", pool, k=2, shape_frac=frac)
    assert [t.twin_uid for t in twins] == ["pass_edge", "pass_mild"]
    t = twins[0]
    assert (t.n_heavy_anchor, t.n_heavy_twin, t.dcharge) == (30, 36, 0)
    assert t.dlogp == pytest.approx(-1.4, abs=1e-5) and t.darom == pytest.approx(1.0, abs=1e-6)
    assert twins[1].darom == pytest.approx(-1.0, abs=1e-6)
    assert info["n_near"] == 6 and info["n_size"] == 5 and info["n_logp"] == 4
    assert info["n_arom"] == 3 and info["n_charge"] == 2
    # each filter, relaxed alone, lets its violator in first (it has the worst chemistry)
    relax = {"too_big": dict(size_tol=0.3), "too_greasy": dict(max_dlogp=2.0),
             "too_aromatic": dict(max_darom=2.0), "charged": dict(same_charge_sign=False)}
    for uid, kw in relax.items():
        got = ts.select_twins("1abc", pool, k=1, shape_frac=frac, **kw)
        assert got[0].twin_uid == uid, (uid, kw)
    charged = ts.select_twins("1abc", pool, k=1, shape_frac=frac, same_charge_sign=False)[0]
    assert charged.dcharge == 1


def test_charge_sign_classes():
    b = PoolBuilder()
    a = make_anchor(b, charge=-2)
    comp = comp_raw(a["pocket"])
    b.add("minus1", charge=-1, shape=b.near(a["shape"]), chem=chem_cos(comp, -1.0, b.rng))
    b.add("neutral", charge=0, shape=b.near(a["shape"]), chem=chem_cos(comp, -0.9, b.rng))
    b.add("plus1", charge=1, shape=b.near(a["shape"]), chem=chem_cos(comp, -0.8, b.rng))
    pool = b.pool()
    twins = ts.select_twins("1abc", pool, k=3, shape_frac=3 / len(pool.pool_rows))
    assert [t.twin_uid for t in twins] == ["minus1"] and twins[0].dcharge == 1


def test_no_twin_records_failure():
    b = PoolBuilder()
    big = make_anchor(b, "2big", n_heavy=200, logp=-12.0, charge=-6, arom_atoms=0.0)
    make_anchor(b, "3bad", ok=False)
    a = make_anchor(b, "4one")
    comp = comp_raw(a["pocket"])
    b.add("only_one", shape=b.near(a["shape"]), chem=chem_cos(comp, -1.0, b.rng))
    b.add("wrong_size", n_heavy=60, shape=b.near(a["shape"]), chem=chem_cos(comp, -1.0, b.rng))
    pool = b.pool()
    for uid, reason in (("2big", "no_twin"), ("3bad", "anchor_no_descriptor"),
                        ("not_in_pool", "anchor_no_descriptor")):
        twins, info = ts.select_twins_with_info(uid, pool, k=2)
        assert twins == [] and info["reason"] == reason
    _, info = ts.select_twins_with_info("2big", pool, k=2)
    assert info["empty_at"] == "size" and info["n_near"] > 0
    twins, info = ts.select_twins_with_info("4one", pool, k=2, shape_frac=2 / len(pool.pool_rows))
    assert [t.twin_uid for t in twins] == ["only_one"] and info["reason"] == "short"
    # the batch path records the same and never raises
    out = ts.select_twins_batch(["2big", "3bad", "4one", "not_in_pool"], pool, k=2,
                                shape_frac=2 / len(pool.pool_rows))
    assert [out[u][1]["reason"] for u in ("2big", "3bad", "4one", "not_in_pool")] == \
        ["no_twin", "anchor_no_descriptor", "short", "anchor_no_descriptor"]
    assert big["n_heavy"] == 200


def test_valid_anchor_and_pool_is_train_only():
    b = PoolBuilder()
    a = make_anchor(b, "5val", split="valid")
    comp = comp_raw(a["pocket"])
    b.add("valid_row", split="valid", shape=b.near(a["shape"]), chem=chem_cos(comp, -1.0, b.rng))
    b.add("sv_row", split="series_valid", shape=b.near(a["shape"]), chem=chem_cos(comp, -1.0, b.rng))
    b.add("train_row", shape=b.near(a["shape"]), chem=chem_cos(comp, -0.5, b.rng))
    pool = b.pool()
    assert "valid_row" not in set(pool.uid[pool.pool_rows])
    twins = ts.select_twins("5val", pool, k=1, shape_frac=2 / len(pool.pool_rows))
    assert twins[0].twin_uid == "train_row"


def test_batch_matches_single_and_block_size():
    b = PoolBuilder(n_background=400, seed=3)
    anchors = [make_anchor(b, f"{i}an{i}", group=f"g_bg{i % 40}")["uid"] for i in range(9)]
    pool = b.pool()
    single = {u: ts.select_twins_with_info(u, pool, k=2, shape_frac=0.05) for u in anchors}
    for block in (1, 4, 256):
        batch = ts.select_twins_batch(anchors, pool, k=2, shape_frac=0.05, block=block)
        for u in anchors:
            assert [t.twin_uid for t in batch[u][0]] == [t.twin_uid for t in single[u][0]]
            for x, y in zip(batch[u][0], single[u][0]):
                assert x.shape_dist == pytest.approx(y.shape_dist, abs=1e-5)
                assert x.complementarity == pytest.approx(y.complementarity, abs=1e-5)


def test_pool_standardisation_and_scalars():
    b = PoolBuilder(n_background=50)
    make_anchor(b, logp=1.25, charge=-1, arom_atoms=9.0, n_heavy=22)
    b.add("const_col_row")
    pool = b.pool()
    i = pool.index["1abc"]
    assert pool.logp[i] == pytest.approx(1.25) and pool.charge[i] == -1
    assert pool.arom[i] == pytest.approx(1.5) and pool.n_heavy[i] == 22
    assert np.allclose(np.linalg.norm(pool.shape, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pool.chem, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pool.pocket_comp, axis=1), 1.0, atol=1e-5)
    assert pool.shape.dtype == np.float32 and pool.chem.shape[1] == 6 * K
    # a constant column keeps scale 1 (no division by zero)
    d = b.desc()
    d["ligand_F"][:, 0, 5] = 3.0
    p2 = ts.build_pool(d, b.meta(), b.groups())
    assert np.isfinite(p2.shape).all()


def test_selection_speed_blocked():
    b = PoolBuilder(n_background=20000, seed=5)
    anchors = [make_anchor(b, f"{i:04d}", group=f"g_bg{i % 40}")["uid"] for i in range(200)]
    pool = b.pool()
    t0 = time.perf_counter()
    out = ts.select_twins_batch(anchors, pool, k=2)
    per_anchor = (time.perf_counter() - t0) / len(anchors)
    assert len(out) == 200
    assert per_anchor < 0.05  # 20k-row pool; the full table (~75k train rows) scales linearly
