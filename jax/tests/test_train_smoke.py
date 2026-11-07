"""Training loop, selection rule, resume, ensemble prediction and the precomputed-AEV path on small
synthetic stores (CPU)."""

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aevplig_jax import inference, model, precompute_aev, predict, train
from aevplig_jax.aev import radial_aev_numpy
from aevplig_jax.batching import Capacities, StoreSource, assemble, prepare, prepare_all
from aevplig_jax.store import Store
from test_store import make_record, random_record, write_store

SRC = Path(__file__).resolve().parents[1] / "src"
DATASET = "pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"
FAST = ["--workers", "2", "--prefetch", "2", "--prepare-workers", "1"]
HISTORY_KEYS = {"epoch", "loss", "terms", "valid_pearson", "selection_mean", "saved", "steps",
                "train_seconds", "valid_seconds", "wait_fraction"}


def trainside(root: Path, seed: int = 0, nan_label: str | None = None) -> Path:
    """A synthetic ``trainside`` store: 44 train, 8 valid, 6 test and 1 excluded record."""
    rng = np.random.default_rng(seed)
    records = [random_record(rng, f"tr{i:02d}", split="train") for i in range(44)]
    records += [random_record(rng, f"va{i}", split="valid") for i in range(8)]
    records += [random_record(rng, f"te{i}", split="test") for i in range(6)]
    records.append(random_record(rng, "ex0", split="excluded"))
    for record in records:
        if record.id == nan_label:
            record.label = math.nan
    write_store(root / "trainside", records)
    return root


def train_args(root: Path, out: Path, tag: str, epochs: int, *extra: str,
               verified: bool = False) -> list[str]:
    """train.py arguments; synthetic stores carry no verification.json, so they opt out unless
    ``verified``."""
    skip = [] if verified else ["--skip-verification"]
    return ["--store-root", str(root), "--out", str(out), "--tag", tag, "--seed-index", "0",
            "--epochs", str(epochs), "--limit", "40", *FAST, *skip, *extra]


def history(seed_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]


def read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != "__meta__"}


def test_train_smoke_two_epochs(tmp_path):
    root, out = trainside(tmp_path / "store"), tmp_path / "out"
    assert train.main(train_args(root, out, "smoke", 2)) == 0

    run, seed_dir = out / "smoke", out / "smoke" / "seed-0"
    rows = history(seed_dir)
    assert [row["epoch"] for row in rows] == [0, 1]
    for row in rows:
        assert set(row) == HISTORY_KEYS
        assert set(row["terms"]) == {"mse"} and math.isfinite(row["loss"])
        assert row["loss"] == pytest.approx(row["terms"]["mse"], rel=1e-6)  # one term, weight 1
        assert 0.0 <= row["wait_fraction"] <= 1.0
    assert [row["steps"] for row in rows] == [1, 2]  # 40 ids: one short batch per epoch
    assert rows[0]["saved"] and rows[0]["selection_mean"] == rows[0]["valid_pearson"]
    assert sorted(p.name for p in (seed_dir / "checkpoints").iterdir()) == ["best", "latest"]
    assert (seed_dir / "checkpoints" / "best" / "state.npz").is_file()
    assert (seed_dir / "checkpoints" / "latest" / "state.npz").is_file()

    with Store(root / "trainside") as store:
        train_labels = [store.records[i]["label"] for i in store.split_ids("train")[:40]]
        test_ids = store.split_ids("test")
        test_labels = [store.records[i]["label"] for i in test_ids]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 40, "valid": 8, "test": 6}
    assert manifest["y_mean"] == pytest.approx(np.mean(train_labels), rel=1e-12)
    assert manifest["y_sd"] == pytest.approx(np.std(train_labels), rel=1e-12)  # population sd
    assert set(manifest["capacities"]) == {"ncap", "mcap", "kb", "ka", "batch"}

    parent = read_csv(out / "predictions" / f"smoke_{DATASET}_test_seed0.csv")
    assert parent[0] == ["truth", "preds_0", "preds"]
    assert len(parent) == 1 + len(test_ids)
    np.testing.assert_allclose([float(r[0]) for r in parent[1:]], test_labels, rtol=1e-12)
    assert all(np.isfinite(float(r[1])) and r[1] == r[2] for r in parent[1:])
    ours = read_csv(seed_dir / "test_predictions.csv")
    assert ours[0] == ["unique_id", "truth", "pred"]
    assert [r[0] for r in ours[1:]] == test_ids
    assert [r[2] for r in ours[1:]] == [r[1] for r in parent[1:]]

    protocol = json.loads((seed_dir / "protocol.json").read_text())
    assert protocol["flags"]["limit"] == 40 and protocol["flags"]["aev"] == "onthefly"
    assert protocol["seed"] == 100 and protocol["constants"]["SELECTION_WINDOW"] == 8
    assert {"train.py", "model.py", "batching.py", "inference.py"} <= set(protocol["package_sha256"])
    assert protocol["capacities"] == manifest["capacities"]
    assert {"jax", "optax", "numpy"} <= set(protocol["versions"])
    selection = json.loads((seed_dir / "selection.json").read_text())
    saved = [row["epoch"] for row in rows if row["saved"]]
    assert selection["epoch"] == saved[-1]
    assert selection["valid_pearson"] == pytest.approx(rows[saved[-1]]["valid_pearson"])
    assert selection["selection_mean"] == pytest.approx(rows[saved[-1]]["selection_mean"])


def test_train_and_predict_refuse_unverified_stores(tmp_path, capsys):
    root, out = trainside(tmp_path / "store"), tmp_path / "out"
    with pytest.raises(RuntimeError, match="no verification.json"):
        train.main(train_args(root, out, "unverified", 1, verified=True))
    assert not (out / "unverified").exists()  # refused before anything is written
    with pytest.raises(RuntimeError, match="no verification.json"):
        predict.main(["--store", str(root / "trainside"), "--manifest", str(tmp_path / "m.json"),
                      "--out", str(tmp_path / "p.csv"), "--checkpoints", str(tmp_path / "none"),
                      *FAST])
    capsys.readouterr()
    assert train.main(train_args(root, out, "optout", 1)) == 0
    assert "--skip-verification" in capsys.readouterr().err
    protocol = json.loads((out / "optout" / "seed-0" / "protocol.json").read_text())
    assert protocol["flags"]["skip_verification"] is True


def test_selection_rule_window_of_eight():
    pcs = [0.5, 0.4, 0.55, 0.3, 0.7, 0.7, 0.7, 0.7, 0.9, 0.1, 0.1, 0.8, math.nan]
    best, saved, means = train.INITIAL_BEST, [], []
    for epoch in range(len(pcs)):
        mean, save = train.selection_step(pcs[:epoch + 1], best)
        means.append(mean)
        if save:
            best = mean
            saved.append(epoch)
    # e1-e4 never beat e0's 0.5; e8's window is e1..e8 (0.61875; with e0 it would be 0.6056); a NaN
    # Pearson makes the window mean NaN, which never saves.
    assert saved == [0, 5, 6, 7, 8]
    expected = [np.mean(pcs[max(e - 7, 0):e + 1]) for e in range(len(pcs))]
    np.testing.assert_allclose(means[:-1], expected[:-1], rtol=1e-12)
    assert means[8] == pytest.approx(0.61875) and math.isnan(means[-1])
    assert best == pytest.approx(0.61875)


def test_resume_continues_from_latest(tmp_path, monkeypatch):
    root, out = trainside(tmp_path / "store"), tmp_path / "out"
    assert train.main(train_args(root, out, "straight", 3)) == 0

    save_state = inference.save_state

    def interrupt(path, trees, meta):  # dies after epoch 2's validation, before its latest write
        if "opt_state" in trees and meta["epoch"] == 2:
            raise RuntimeError("interrupted")
        save_state(path, trees, meta)

    monkeypatch.setattr(inference, "save_state", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        train.main(train_args(root, out, "resumed", 3))
    monkeypatch.undo()
    seed_dir = out / "resumed" / "seed-0"
    assert [row["epoch"] for row in history(seed_dir)] == [0, 1]
    with pytest.raises(FileExistsError, match="--resume"):
        train.main(train_args(root, out, "resumed", 3))

    assert train.main(train_args(root, out, "resumed", 3, "--resume")) == 0
    straight_dir = out / "straight" / "seed-0"
    got, want = history(seed_dir), history(straight_dir)
    assert [row["epoch"] for row in got] == [0, 1, 2]
    assert [row["steps"] for row in got] == [row["steps"] for row in want] == [1, 2, 3]
    np.testing.assert_allclose([row["valid_pearson"] for row in got],
                               [row["valid_pearson"] for row in want], rtol=1e-6)
    for name in ("latest", "best"):
        got_arrays = npz_arrays(seed_dir / "checkpoints" / name / "state.npz")
        want_arrays = npz_arrays(straight_dir / "checkpoints" / name / "state.npz")
        assert got_arrays.keys() == want_arrays.keys()
        for key, value in want_arrays.items():
            np.testing.assert_allclose(got_arrays[key], value, rtol=1e-5, atol=1e-7, err_msg=key)
    assert (json.loads((seed_dir / "selection.json").read_text())["epoch"]
            == json.loads((straight_dir / "selection.json").read_text())["epoch"])


def test_nonfinite_loss_raises(tmp_path, monkeypatch):
    root, out = trainside(tmp_path / "store"), tmp_path / "out"
    monkeypatch.setitem(train.TERMS, "mse",
                        (1.0, lambda pred, batch: train.mse_term(pred, batch) * jnp.nan))
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train.main(train_args(root, out, "nan", 1))
    assert not (out / "nan" / "seed-0" / "checkpoints" / "latest").exists()


def test_nonfinite_label_is_rejected(tmp_path):
    root = trainside(tmp_path / "store", nan_label="va3")
    with pytest.raises(ValueError, match="va3"):
        train.main(train_args(root, tmp_path / "out", "badlabel", 1))


def test_predict_plans_capacities_from_target_store(tmp_path):
    rng = np.random.default_rng(3)
    # The training manifest's capacities: small ligands only.
    small = Capacities(ncap=256, mcap=1024, kb=3, ka=32, batch=128)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"capacities": small.to_json(), "y_mean": 6.0, "y_sd": 2.0}))
    members = []
    for k in range(2):
        params, state = model.init_params(jax.random.key(k)), model.init_bn_state()
        inference.save_state(tmp_path / "run" / f"seed-{k}" / "checkpoints" / "best" / "state.npz",
                             {"params": params, "bn_state": state}, {"epoch": 0})
        members.append((params, state))
    # A 300-atom ligand with a dense bond graph and pocket: over every training capacity.
    big = make_record(rng, "big", rng.uniform(0, 14, (300, 3)), rng.uniform(-3, 17, (400, 3)),
                      rng.integers(1, 23, 400), 900, label=7.0, split="predict",
                      meta={"subset": "sub", "series": "ser", "ref_pdb": "1ABC"})
    others = [random_record(rng, f"p{i}", split="predict") for i in range(5)]
    records = [others[0], big, *others[1:]]
    target = write_store(tmp_path / "target", records)
    with pytest.raises(ValueError, match="big"):
        assemble([big], small, 6.0, 2.0)

    out_csv = tmp_path / "pred.csv"
    assert predict.main(["--store", str(target), "--manifest", str(manifest), "--out", str(out_csv),
                         "--checkpoints", str(tmp_path / "run" / "seed-*" / "checkpoints" / "best"),
                         "--skip-verification", *FAST]) == 0
    rows = read_csv(out_csv)
    assert rows[0] == ["unique_id", "subset", "series", "ref_pdb", "truth", "pred", "pred_0", "pred_1"]
    assert [r[0] for r in rows[1:]] == [r.id for r in records]
    by_id = {r[0]: r for r in rows[1:]}
    assert by_id["big"][1:5] == ["sub", "ser", "1ABC", "7.0"]
    assert by_id["p0"][1:4] == ["", "", ""]
    for record in records:
        prepared = {record.id: prepare(record)}
        cap = inference.plan_capacities(prepared, [[record.id]])
        batch = assemble([prepared[record.id]], cap, 6.0, 2.0)._replace(n_real=None)
        row = by_id[record.id]
        direct = [float(model.predict(p, s, batch, False)[0][0]) * 2.0 + 6.0 for p, s in members]
        np.testing.assert_allclose([float(row[6]), float(row[7])], direct, rtol=1e-5, atol=1e-4)
        assert float(row[5]) == pytest.approx((float(row[6]) + float(row[7])) / 2, rel=1e-9)


def test_precomputed_aev_table_matches_onthefly(tmp_path):
    rng = np.random.default_rng(5)
    records = [random_record(rng, f"r{i}") for i in range(12)]
    path = write_store(tmp_path / "store", records)
    table = tmp_path / "aev.npz"
    assert precompute_aev.main(["--store", str(path), "--out", str(table), "--workers", "1"]) == 0
    with np.load(table) as data:
        assert list(data["ids"]) == [r.id for r in records]
        offsets, aev = data["offsets"], data["aev"]
    assert aev.dtype == np.float32 and aev.shape == (offsets[-1], 352)
    np.testing.assert_array_equal(np.diff(offsets), [len(r.lig.xyz) for r in records])
    for i in (0, 7):
        expected = radial_aev_numpy(records[i].lig.xyz, records[i].pocket.xyz, records[i].pocket.species)
        np.testing.assert_allclose(aev[offsets[i]:offsets[i + 1]], expected, rtol=1e-4, atol=1e-5)

    ids = [r.id for r in records]
    with Store(path) as store:
        prepared = prepare_all(store, ids)
        cap = inference.plan_capacities(prepared, [ids])
        plain = StoreSource(store, cap, 6.0, 2.0, 1, prepared=prepared).load(ids)
        rows = precompute_aev.load_table(table, store)
        pre = StoreSource(store, cap, 6.0, 2.0, 1, prepared=prepared, aev_table=rows).load(ids)
    assert plain.x_lig.shape[1] == 15 and pre.x_lig.shape[1] == 367
    params, state = model.init_params(jax.random.key(1)), model.init_bn_state()
    want = model.predict(params, state, plain._replace(n_real=None), False)[0]
    got = model.predict(params, state, pre._replace(n_real=None), False,
                        features=train.FEATURES["precomputed"])[0]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_train_runs_with_precomputed_aev(tmp_path):
    root, out = trainside(tmp_path / "store"), tmp_path / "out"
    table = tmp_path / "aev.npz"
    assert precompute_aev.main(["--store", str(root / "trainside"), "--out", str(table),
                                "--workers", "1"]) == 0
    assert train.main(train_args(root, out, "pre", 1, "--aev", "precomputed",
                                 "--aev-table", str(table))) == 0
    assert len(history(out / "pre" / "seed-0")) == 1
    protocol = json.loads((out / "pre" / "seed-0" / "protocol.json").read_text())
    assert protocol["flags"]["aev"] == "precomputed"
    with pytest.raises(ValueError, match="--aev-table"):
        train.main(train_args(root, out, "pre-missing", 1, "--aev", "precomputed"))


@pytest.mark.parametrize("module, allowed", [("train", set()), ("precompute_aev", set()),
                                             ("predict", {"pandas"}), ("evaluate", {"pandas"})])
def test_entry_points_import_no_torch_or_rdkit(module, allowed):
    """train.py and precompute_aev.py run without pandas; none of the four needs torch or RDKit."""
    code = (f"import sys, aevplig_jax.{module}; "
            "print(sorted({'torch', 'torch_geometric', 'rdkit', 'pandas'} & set(sys.modules)))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == str(sorted(allowed))
