"""CPU smoke test for train_paired.main on synthetic paired_<split>.pt files (Task 2)."""
import json
import os
import pickle
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

import train_paired
from model_defs import GATv2Net

N_FEAT, N_EDGE_FEAT = 367, 4
LOG_LINE = re.compile(
    r"^epoch (\d+) loss (\S+) mse (\S+) pair (\S+) decoy (\S+) hinge (\S+) siam (\S+) valid_pearson (\S+) "
    r"series_valid_spearman (\S+) series_valid_pearson (\S+)$")


def _graph(rng, row, series, lig, src, pk):
    n = int(rng.integers(3, 9))
    chain = [(a, a + 1) for a in range(n - 1)]
    edges = chain + [(b, a) for a, b in chain]
    return Data(x=torch.tensor(rng.normal(size=(n, N_FEAT)), dtype=torch.float),
                edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
                edge_attr=torch.tensor(rng.random((len(edges), N_EDGE_FEAT)), dtype=torch.float),
                y=torch.tensor([pk], dtype=torch.float), row=torch.tensor([row]),
                series=torch.tensor([series]), lig=torch.tensor([lig]), src=torch.tensor([src]))


def _write_splits(data_dir, seed=0):
    """Train: two 6-member series (one repeated ligand) + 10 singletons; series_valid: one
    6-row series; valid/test: singleton PDBbind-like rows. Returns {split: DataFrame}."""
    rng = np.random.default_rng(seed)
    layout = {
        "train": [(0, 1)] * 6 + [(1, 1)] * 6 + [(2 + k, 0) for k in range(10)],
        "valid": [(20 + k, 0) for k in range(8)],
        "test": [(30 + k, 0) for k in range(5)],
        "series_valid": [(40, 2)] * 6,
    }
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    meta, row, lig = {}, 0, 0
    for split, members in layout.items():
        records, graphs = [], []
        for k, (series, src) in enumerate(members):
            this_lig = lig - 1 if (split == "train" and series == 0 and k == 1) else lig
            pk = float(rng.uniform(4.0, 10.0))
            graphs.append(_graph(rng, row, series, this_lig, src, pk))
            records.append({"row": row, "series": series, "lig": this_lig, "pK": pk})
            row, lig = row + 1, lig + 1
        InMemoryDataset.save(graphs, os.path.join(data_dir, "processed", f"paired_{split}.pt"))
        meta[split] = pd.DataFrame(records)
    return meta


@pytest.mark.parametrize("extra", [
    ["--pair-loss", "delta_huber", "--pair-weight", "0.3"],
    ["--pair-loss", "listwise", "--pair-weight", "1.0", "--ligand-only",
     "--batch-size", "8", "--chunk", "4"],
], ids=["delta_huber", "listwise_ligand_only_small_batches"])
def test_train_paired_smoke(tmp_path, capsys, extra):
    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    meta = _write_splits(data_dir)
    tag = "smoke"
    summary = train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir,
                                 "--seed_index", "1", "--tag", tag, "--epochs", "2",
                                 "--hidden-dim", "8", *extra])
    out = capsys.readouterr().out

    # per-epoch log lines in the agreed format, with a non-zero pair term
    epochs = [LOG_LINE.match(line) for line in out.splitlines()]
    epochs = [m for m in epochs if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2]
    assert all(np.isfinite(float(m.group(2))) for m in epochs)
    assert max(float(m.group(4)) for m in epochs) > 0

    # model: a bare GATv2Net state_dict (no LigandOnly prefix) that loads with hidden 8
    model_path = os.path.join(out_dir, "trained_models", f"{tag}_1.model")
    state = torch.load(model_path)
    assert not any(k.startswith("model.") for k in state)
    config = SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=8)
    GATv2Net(N_FEAT, N_EDGE_FEAT, config).load_state_dict(state)

    # scaler: fit on train pK
    with open(os.path.join(out_dir, "trained_models", f"{tag}.pickle"), "rb") as fh:
        scaler = pickle.load(fh)
    assert scaler.mean_[0] == pytest.approx(meta["train"]["pK"].mean(), abs=1e-6)
    assert scaler.scale_[0] == pytest.approx(meta["train"]["pK"].std(ddof=0), abs=1e-6)
    assert summary["scaler_scale"] == pytest.approx(scaler.scale_[0])

    # four prediction CSVs: row, truth (raw pK, in split order), finite pred
    for split, df_meta in meta.items():
        path = os.path.join(out_dir, "predictions", f"{tag}_{split}_seed1.csv")
        df = pd.read_csv(path)
        assert list(df.columns) == ["row", "truth", "pred"]
        assert len(df) == len(df_meta)
        assert df["row"].tolist() == df_meta["row"].tolist()
        assert np.allclose(df["truth"], df_meta["pK"], atol=1e-4)
        assert np.isfinite(df["pred"]).all()
    assert os.path.isfile(os.path.join(out_dir, "trained_models", f"{tag}_1.json"))
    assert all(float(m.group(7)) == 0.0 for m in epochs)  # no Siamese head: siam term 0
    with open(os.path.join(out_dir, "trained_models", f"{tag}_1.json")) as fh:
        side = json.load(fh)
    assert side["siamese"] is False and side["model_format"] == "gatv2net"


@pytest.mark.parametrize("extra", [[], ["--ligand-only", "--batch-size", "8", "--chunk", "4"]],
                         ids=["siamese", "siamese_ligand_only_small_batches"])
def test_train_paired_smoke_siamese(tmp_path, capsys, extra):
    import heads

    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    meta = _write_splits(data_dir)
    tag = "siam"
    summary = train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0",
                                 "--tag", tag, "--epochs", "2", "--hidden-dim", "8", "--siamese",
                                 "--siamese-weight", "1.0", "--siamese-hidden", "16", "--siamese-tie", "0.5",
                                 *extra])
    out = capsys.readouterr().out
    epochs = [m for m in (LOG_LINE.match(line) for line in out.splitlines()) if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2]
    siam = [float(m.group(7)) for m in epochs]
    assert all(np.isfinite(siam)) and max(siam) > 0
    assert all(float(m.group(4)) == 0.0 for m in epochs)  # no --pair-loss
    # loss = mse + siam (no pair loss, no hinge rows)
    for m in epochs:
        assert float(m.group(2)) == pytest.approx(float(m.group(3)) + float(m.group(7)), abs=2e-6)

    # model: the wrapper's state dict (encoder / affinity / siamese), loadable as the wrapper and,
    # after renaming, as a plain GATv2Net
    model_path = os.path.join(out_dir, "trained_models", f"{tag}_0.model")
    state = torch.load(model_path, weights_only=True)
    assert {k.split(".")[0] for k in state} == {"encoder", "affinity", "siamese"}
    config = SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=8)
    gnn = GATv2Net(N_FEAT, N_EDGE_FEAT, config)
    wrapper = heads.AffinityWithSiamese(gnn, heads.SiameseHead(48, hidden=16))
    wrapper.load_state_dict(state)
    GATv2Net(N_FEAT, N_EDGE_FEAT, config).load_state_dict(heads.wrapped_to_plain(state))

    with open(os.path.join(out_dir, "trained_models", f"{tag}_0.json")) as fh:
        side = json.load(fh)
    assert side["siamese"] is True and side["model_format"] == "affinity_with_siamese"
    assert (side["siamese_hidden"], side["siamese_weight"], side["siamese_dim"]) == (16, 1.0, 48)
    scale = summary["scaler_scale"]
    assert side["siamese_tie"] == 0.5 and side["siamese_tie_std"] == pytest.approx(0.5 / scale)
    assert side["siamese_tau_std"] == pytest.approx(0.5 / scale)
    assert side["siamese_huber_delta_std"] == pytest.approx(1.0 / scale)
    assert side["args"]["ligand_only"] is ("--ligand-only" in extra)

    # pointwise predictions as before: four CSVs in split order, finite
    for split, df_meta in meta.items():
        df = pd.read_csv(os.path.join(out_dir, "predictions", f"{tag}_{split}_seed0.csv"))
        assert list(df.columns) == ["row", "truth", "pred"]
        assert df["row"].tolist() == df_meta["row"].tolist()
        assert np.isfinite(df["pred"]).all()


def test_siamese_flag_guards():
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t"]
    args = train_paired.parse_args(base)
    assert args.siamese is False and (args.siamese_weight, args.siamese_hidden, args.siamese_tie) == (1.0, 512, 0.5)
    for bad in (["--siamese-weight", "0"], ["--siamese-weight", "-1"], ["--siamese-hidden", "0"],
                ["--siamese-tie", "-0.1"]):
        with pytest.raises(SystemExit):
            train_paired.parse_args([*base, "--siamese", *bad])



# ----------------------------------------------------------------------------- pose kinds (arm 4)

def _graph_kind(rng, row, series, lig, src, pk, kind, parent):
    d = _graph(rng, row, series, lig, src, pk)
    d.kind = torch.tensor([kind])
    d.parent = torch.tensor([parent])
    return d


def _write_kind_splits(data_dir, seed=0):
    """Train: 12 crystal rows in 3 series, each with one pose (kind 1) and two decoys (kind 2)
    sharing the crystal's series / lig / label; valid / test / series_valid kind 0 only."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    meta, row = {}, 0
    crystals = []
    graphs, records = [], []
    for k in range(12):
        pk = float(rng.uniform(4.0, 10.0))
        graphs.append(_graph_kind(rng, row, k % 3, k, 0, pk, 0, row))
        records.append({"row": row, "kind": 0, "parent": row, "pK": pk})
        crystals.append((row, k % 3, k, pk))
        row += 1
    for parent, series, lig, pk in crystals:
        for kind in (1, 2, 2):
            graphs.append(_graph_kind(rng, row, series, lig, 0, pk, kind, parent))
            records.append({"row": row, "kind": kind, "parent": parent, "pK": pk})
            row += 1
    InMemoryDataset.save(graphs, os.path.join(data_dir, "processed", "paired_train.pt"))
    meta["train"] = pd.DataFrame(records)
    for split, members in (("valid", [(20 + k, 0) for k in range(8)]),
                           ("test", [(30 + k, 0) for k in range(5)]),
                           ("series_valid", [(40, 2)] * 6)):
        graphs, records = [], []
        for series, src in members:
            pk = float(rng.uniform(4.0, 10.0))
            graphs.append(_graph_kind(rng, row, series, 100 + row, src, pk, 0, row))
            records.append({"row": row, "kind": 0, "parent": row, "pK": pk})
            row += 1
        InMemoryDataset.save(graphs, os.path.join(data_dir, "processed", f"paired_{split}.pt"))
        meta[split] = pd.DataFrame(records)
    return meta


def _record_batches(monkeypatch):
    """Wrap compute_loss to record the (row, kind) of every training batch."""
    seen = []
    original = train_paired.compute_loss

    def wrapped(output, data, cfg):
        seen.append((data.row.view(-1).tolist(), data.kind.view(-1).tolist()))
        return original(output, data, cfg)

    monkeypatch.setattr(train_paired, "compute_loss", wrapped)
    return seen


@pytest.mark.parametrize("kinds,weight", [("0,1,2", "1.0"), ("0", "0"), ("0,1", "0")],
                         ids=["pdanda", "base", "pda"])
def test_train_paired_smoke_with_decoys(tmp_path, capsys, monkeypatch, kinds, weight):
    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    meta = _write_kind_splits(data_dir)
    seen = _record_batches(monkeypatch)
    tag = "kinds"
    summary = train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0",
                                 "--tag", tag, "--epochs", "2", "--hidden-dim", "8",
                                 "--batch-size", "8", "--chunk", "4",
                                 "--train-kinds", kinds, "--decoy-weight", weight,
                                 "--decoy-margin", "0.75"])
    out = capsys.readouterr().out
    selected = [int(k) for k in kinds.split(",")]
    train = meta["train"]
    used = train[train.kind.isin(selected)]

    # the sampler and the loader see exactly the selected rows, each once per epoch
    rows_seen = sorted(r for rows, _ in seen for r in rows)
    assert rows_seen == sorted(used.row.tolist() * 2)
    assert {k for _, ks in seen for k in ks} == set(selected)
    assert all(len(rows) >= 2 for rows, _ in seen)

    epochs = [m for m in (LOG_LINE.match(line) for line in out.splitlines()) if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2]
    decoys = [float(m.group(5)) for m in epochs]
    assert all(np.isfinite(float(m.group(2))) for m in epochs)
    if 2 in selected:
        assert max(decoys) > 0
    else:
        assert decoys == [0.0, 0.0]

    # scaler: fit on the kind-0 train rows whatever the selection (arm 1's base scaler)
    with open(os.path.join(out_dir, "trained_models", f"{tag}.pickle"), "rb") as fh:
        scaler = pickle.load(fh)
    crystal = train[train.kind == 0]
    assert scaler.mean_[0] == pytest.approx(crystal.pK.mean(), abs=1e-6)
    assert scaler.scale_[0] == pytest.approx(crystal.pK.std(ddof=0), abs=1e-6)

    # train predictions cover the rows used; sidecar records the selection
    pred = pd.read_csv(os.path.join(out_dir, "predictions", f"{tag}_train_seed0.csv"))
    assert pred.row.tolist() == used.row.tolist()
    with open(os.path.join(out_dir, "trained_models", f"{tag}_0.json")) as fh:
        side = json.load(fh)
    assert side["train_kinds"] == selected
    assert side["decoy_weight"] == float(weight) and side["decoy_margin"] == 0.75
    assert side["decoy_margin_std"] == pytest.approx(0.75 / scaler.scale_[0])
    assert side["n_train_rows"] == len(used) == summary["n_train_rows"]
    assert side["train_kind_counts"] == {str(k): int((used.kind == k).sum()) for k in selected}
    for split in ("valid", "test", "series_valid"):
        df = pd.read_csv(os.path.join(out_dir, "predictions", f"{tag}_{split}_seed0.csv"))
        assert df.row.tolist() == meta[split].row.tolist()


@pytest.mark.parametrize("kinds,weight,fmt", [
    ("0,2", "0", "kinds"),       # decoys selected with weight 0: never silently ignore decoys
    ("0,1", "1.0", "kinds"),     # decoy weight without decoy rows selected
    ("0,3", "0", "kinds"),       # unknown kind
    ("0,1", "0", "plain"),       # arm-1 .pt (no kind field): kind 1 has no rows
], ids=["decoys_weight0", "weight_without_decoys", "unknown_kind", "missing_kind_rows"])
def test_train_kinds_errors(tmp_path, capsys, kinds, weight, fmt):
    data_dir = str(tmp_path / "data")
    (_write_kind_splits if fmt == "kinds" else _write_splits)(data_dir)
    with pytest.raises(SystemExit) as exc:
        train_paired.main(["--data-dir", data_dir, "--out-dir", str(tmp_path / "output"),
                           "--seed_index", "0", "--tag", "bad", "--epochs", "1", "--hidden-dim", "8",
                           "--train-kinds", kinds, "--decoy-weight", weight])
    assert exc.value.code not in (0, None)
    msg = f"{exc.value.code} {capsys.readouterr().err}"
    assert "unrecognized arguments" not in msg and "kind" in msg, msg
    assert not os.path.exists(os.path.join(tmp_path, "output", "trained_models", "bad_0.model"))
    assert not os.path.exists(os.path.join(tmp_path, "output", "trained_models", "bad.pickle"))


def test_train_kinds_zero_equals_arm1_dataset(tmp_path):
    """--train-kinds 0 on a pose build trains exactly as arm 1 on the kind-0 rows alone
    (arm-1 format files: no kind / parent fields): same batches, same predictions."""
    import shutil

    from torch_geometric.data import Data

    pose_dir, plain_dir = str(tmp_path / "poses"), str(tmp_path / "plain")
    _write_kind_splits(pose_dir)
    os.makedirs(os.path.join(plain_dir, "processed"))
    torch.serialization.add_safe_globals([Data])
    for split in ("train", "valid", "test", "series_valid"):
        src = os.path.join(pose_dir, "processed", f"paired_{split}.pt")
        dst = os.path.join(plain_dir, "processed", f"paired_{split}.pt")
        if split != "train":
            shutil.copy(src, dst)
            continue

        class _Ds(InMemoryDataset):
            def __init__(self):
                super().__init__(None)
                self.load(src)

        ds = _Ds()
        keep = []
        for k in range(len(ds)):
            d = ds[k]
            if int(d.kind) == 0:
                del d.kind, d.parent
                keep.append(d)
        InMemoryDataset.save(keep, dst)

    preds = {}
    for name, data_dir, extra in (("poses", pose_dir, ["--train-kinds", "0"]), ("plain", plain_dir, [])):
        out_dir = str(tmp_path / f"out_{name}")
        train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "2",
                           "--tag", "t", "--epochs", "2", "--hidden-dim", "8", "--batch-size", "4",
                           "--chunk", "2", *extra])
        preds[name] = {split: pd.read_csv(os.path.join(out_dir, "predictions", f"t_{split}_seed2.csv"))
                       for split in ("train", "valid", "test", "series_valid")}
    for split in preds["plain"]:
        pd.testing.assert_frame_equal(preds["poses"][split], preds["plain"][split], check_exact=True)


# ----------------------------------------------------------------------------- soft negatives (kinds 3-7)

def _write_aug_splits(data_dir, aug_kinds, seed=0):
    """Train: 12 crystal rows in 3 series, each followed by one row per entry of aug_kinds sharing the
    crystal's series / lig / label (the parent's pK); valid / test / series_valid kind 0 only."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    meta, row, crystals, graphs, records = {}, 0, [], [], []
    for k in range(12):
        pk = float(rng.uniform(4.0, 10.0))
        graphs.append(_graph_kind(rng, row, k % 3, k, 0, pk, 0, row))
        records.append({"row": row, "kind": 0, "parent": row, "pK": pk})
        crystals.append((row, k % 3, k, pk))
        row += 1
    for parent, series, lig, pk in crystals:
        for kind in aug_kinds:
            graphs.append(_graph_kind(rng, row, series, lig, 0, pk, kind, parent))
            records.append({"row": row, "kind": kind, "parent": parent, "pK": pk})
            row += 1
    InMemoryDataset.save(graphs, os.path.join(data_dir, "processed", "paired_train.pt"))
    meta["train"] = pd.DataFrame(records)
    for split, members in (("valid", [(20 + k, 0) for k in range(8)]),
                           ("test", [(30 + k, 0) for k in range(5)]),
                           ("series_valid", [(40, 2)] * 6)):
        graphs, records = [], []
        for series, src in members:
            pk = float(rng.uniform(4.0, 10.0))
            graphs.append(_graph_kind(rng, row, series, 100 + row, src, pk, 0, row))
            records.append({"row": row, "kind": 0, "parent": row, "pK": pk})
            row += 1
        InMemoryDataset.save(graphs, os.path.join(data_dir, "processed", f"paired_{split}.pt"))
        meta[split] = pd.DataFrame(records)
    return meta


@pytest.mark.parametrize("kinds,hinge", [("0,3,4", [4]), ("0,5,6,7", [6]), ("0,3,4,5,6,7", [4, 6]),
                                          ("0", [])], ids=["A", "C", "AC", "base-g"])
def test_train_paired_smoke_soft_negatives(tmp_path, capsys, monkeypatch, kinds, hinge):
    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    meta = _write_aug_splits(data_dir, (3, 4, 4, 5, 6, 6, 7, 7))
    seen = _record_batches(monkeypatch)
    summary = train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0",
                                 "--tag", "soft", "--epochs", "2", "--hidden-dim", "8",
                                 "--batch-size", "8", "--chunk", "4", "--train-kinds", kinds,
                                 "--hinge-margins", "2:0.75,4:1.0,6:1.0", "--hinge-weight", "1.0"])
    out = capsys.readouterr().out
    selected = [int(k) for k in kinds.split(",")]
    train = meta["train"]
    used = train[train.kind.isin(selected)]
    rows_seen = sorted(r for rows, _ in seen for r in rows)
    assert rows_seen == sorted(used.row.tolist() * 2)
    assert {k for _, ks in seen for k in ks} == set(selected)

    epochs = [m for m in (LOG_LINE.match(line) for line in out.splitlines()) if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2]
    decoys = [float(m.group(5)) for m in epochs]
    hinges = [float(m.group(6)) for m in epochs]
    assert decoys == [0.0, 0.0]  # no kind 2 here
    if hinge:
        assert max(hinges) > 0
    else:
        assert hinges == [0.0, 0.0]

    with open(os.path.join(out_dir, "trained_models", "soft.pickle"), "rb") as fh:
        scaler = pickle.load(fh)
    crystal = train[train.kind == 0]
    assert scaler.mean_[0] == pytest.approx(crystal.pK.mean(), abs=1e-6)
    with open(os.path.join(out_dir, "trained_models", "soft_0.json")) as fh:
        side = json.load(fh)
    assert side["train_kinds"] == selected
    assert side["hinge_kinds"] == hinge
    assert side["mse_kinds"] == [k for k in selected if k in (0, 1, 3, 5, 7)]
    assert side["hinge_margins"] == {str(k): 1.0 for k in hinge}
    assert side["hinge_margins_std"] == pytest.approx({str(k): 1.0 / scaler.scale_[0] for k in hinge})
    assert side["hinge_weights"] == {str(k): 1.0 for k in hinge} and side["hinge_weight"] == 1.0
    assert side["decoy_weight"] == 0.0 and side["decoy_margin"] == 0.75
    assert side["n_train_rows"] == len(used) == summary["n_train_rows"]
    assert side["train_kind_counts"] == {str(k): int((used.kind == k).sum()) for k in selected}
    pred = pd.read_csv(os.path.join(out_dir, "predictions", "soft_train_seed0.csv"))
    assert pred.row.tolist() == used.row.tolist()


@pytest.mark.parametrize("extra,needle", [
    (["--train-kinds", "0,3,4", "--hinge-weight", "0"], "hinge kind 4"),
    (["--train-kinds", "0,5,6", "--hinge-margins", "2:0.75,4:1.0"], "hinge kind 6"),
    (["--train-kinds", "0,3,4", "--hinge-margins", "3:1.0"], "not a hinge kind"),
    (["--train-kinds", "0,4", "--hinge-margins", "4:-1"], "kind 4"),
    (["--train-kinds", "0,8"], "kinds must be among"),
], ids=["weight0", "no_margin", "margin_for_mse_kind", "negative_margin", "unknown_kind"])
def test_hinge_guards(tmp_path, capsys, extra, needle):
    data_dir = str(tmp_path / "data")
    _write_aug_splits(data_dir, (3, 4, 5, 6))
    with pytest.raises(SystemExit) as exc:
        train_paired.main(["--data-dir", data_dir, "--out-dir", str(tmp_path / "output"), "--seed_index", "0",
                           "--tag", "bad", "--epochs", "1", "--hidden-dim", "8", *extra])
    assert exc.value.code not in (0, None)
    msg = f"{exc.value.code} {capsys.readouterr().err}"
    assert needle in msg, msg
    assert not os.path.exists(os.path.join(tmp_path, "output", "trained_models", "bad.pickle"))


def test_decoy_aliases_set_kind_2_only():
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t"]
    args = train_paired.parse_args([*base, "--train-kinds", "0,2,4", "--decoy-weight", "2.0",
                                    "--decoy-margin", "0.5", "--hinge-weight", "0.5"])
    assert args.hinge_kinds == [2, 4]
    assert args.hinge_weights == {2: 2.0, 4: 0.5}
    assert args.hinge_margins_pk == {2: 0.5, 4: 1.0}
    # defaults: every selected hinge kind at --hinge-weight 1.0 and the plan's margins
    args = train_paired.parse_args([*base, "--train-kinds", "0,2,4,6"])
    assert args.hinge_weights == {2: 1.0, 4: 1.0, 6: 1.0}
    assert args.hinge_margins_pk == {2: 0.75, 4: 1.0, 6: 1.0}
    # arm-4 command lines keep their meaning
    args = train_paired.parse_args([*base, "--train-kinds", "0,1,2", "--decoy-weight", "1.0", "--decoy-margin", "0.75"])
    assert (args.hinge_kinds, args.hinge_weights, args.hinge_margins_pk) == ([2], {2: 1.0}, {2: 0.75})
    args = train_paired.parse_args([*base, "--train-kinds", "0,1", "--decoy-weight", "0"])
    assert args.hinge_kinds == [] and args.hinge_weights == {}


# ----------------------------------------------------------------------------- pair head + moments heads (arm P)

AUX_LINE = re.compile(r"^aux epoch (\d+) pairh (\S+) anchors (\d+) negatives (\d+) moments (\S+) pocket (\S+) "
                      r"ligand (\S+)$")


def _side_from_pt(data_dir, path, seed=0, series_shift=0, failed=()):
    """A side file for the .pt files of data_dir: unique ids u<row> (kind 0) / u<parent>::<kind>_<row>, random
    descriptors for the kind-0 rows, twins (kind 4) taking the ligand of the next crystal's row; the crystal rows in
    `failed` have NaN descriptors (a failed featurisation: masked)."""
    import build_zernike_side as bzs
    from paired_data import PairedDataset

    frames = []
    for split in ("train", "valid", "test", "series_valid"):
        d = PairedDataset(data_dir, split)._data
        rows = d.row.view(-1).numpy()
        kind = d.kind.view(-1).numpy() if getattr(d, "kind", None) is not None else np.zeros(len(rows), int)
        parent = d.parent.view(-1).numpy() if getattr(d, "parent", None) is not None else rows
        frames.append(pd.DataFrame({"row": rows, "split": split, "kind": kind, "parent": parent,
                                    "series_id": d.series.view(-1).numpy() + series_shift,
                                    "lig_id": d.lig.view(-1).numpy()}))
    meta = pd.concat(frames, ignore_index=True)
    meta["unique_id"] = [f"u{r}" if k == 0 else f"u{p}::{k}_{r}" for r, k, p in zip(meta.row, meta.kind, meta.parent)]
    crystals = meta.loc[meta.kind == 0, "unique_id"].tolist()
    twins = pd.Series({u: crystals[(crystals.index(f"u{p}") + 1) % len(crystals)]
                       for u, k, p in zip(meta.unique_id, meta.kind, meta.parent) if k == 4}, dtype=object)
    rng = np.random.default_rng(seed)
    pos = pd.Series(np.arange(len(crystals)), index=crystals)
    pocket = rng.normal(3.0, 1.0, (len(crystals), 343)).astype(np.float32)
    ligand = rng.normal(-1.0, 2.0, (len(crystals), 343)).astype(np.float32)
    for r in failed:
        pocket[pos[f"u{r}"]] = np.nan
        ligand[pos[f"u{r}"]] = np.nan
    arrays, _ = bzs.build(meta, pos, pocket, ligand, twins if len(twins) else None)
    bzs.write_npz(str(path), arrays)
    return arrays


@pytest.mark.parametrize("arm", ["P", "PMneg", "M-lig"])
def test_train_paired_smoke_pair_head_and_moments(tmp_path, capsys, arm):
    import eval_paired
    import heads

    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    side_path = tmp_path / "paired_zernike.npz"
    if arm == "P":  # arm-1 format files (no kind / parent fields): parent = row
        _write_splits(data_dir)
        extra = ["--pair-head", "--pair-head-weight", "0.3", "--pair-hidden", "16", "--sampler-parent-aware"]
    else:
        _write_aug_splits(data_dir, (3, 4, 4, 5, 6, 6, 7, 7))
        _side_from_pt(data_dir, side_path)
        if arm == "PMneg":
            extra = ["--train-kinds", "0,3,4,5,6,7", "--hinge-weight", "0", "--pair-head", "--pair-neg-kinds", "4,6",
                     "--pair-hidden", "16", "--moments", "both", "--moments-weight", "0.3",
                     "--zernike-side", str(side_path), "--batch-size", "64", "--chunk", "32",
                     "--sampler-parent-aware"]  # a chunk holds 3 families (9 rows each) of one series
        else:
            extra = ["--moments", "both", "--zernike-side", str(side_path), "--ligand-only",
                     "--batch-size", "8", "--chunk", "4"]
    tag = arm.lower()
    summary = train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0", "--tag", tag,
                                 "--epochs", "2", "--hidden-dim", "8", *extra])
    out = capsys.readouterr().out
    epochs = [m for m in (LOG_LINE.match(line) for line in out.splitlines()) if m]
    aux = [m for m in (AUX_LINE.match(line) for line in out.splitlines()) if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2] and [int(m.group(1)) for m in aux] == [1, 2]
    pairh = [float(m.group(2)) for m in aux]
    moments = [float(m.group(5)) for m in aux]
    if arm in ("P", "PMneg"):
        assert max(pairh) > 0 and sum(int(m.group(3)) for m in aux) > 0
    else:
        assert pairh == [0.0, 0.0]
    if arm == "PMneg":
        assert sum(int(m.group(4)) for m in aux) > 0       # negatives reached their anchors' lists
        assert all(float(m.group(6)) == 0.0 for m in epochs)  # hinge weight 0: kinds 4 / 6 listwise only
    if arm != "P":
        assert min(moments) > 0 and all(float(m.group(6)) > 0 and float(m.group(7)) > 0 for m in aux)
        for m in aux:  # moments = weight (0.3) * (pocket + ligand)
            assert float(m.group(5)) == pytest.approx(0.3 * (float(m.group(6)) + float(m.group(7))), rel=1e-4)
    for m, e in zip(aux, epochs):  # loss = mse + hinge + pair-head term + moments term
        assert float(e.group(2)) == pytest.approx(float(e.group(3)) + float(e.group(6)) + float(m.group(2))
                                                  + float(m.group(5)), abs=2e-5)

    state = torch.load(os.path.join(out_dir, "trained_models", f"{tag}_0.model"), weights_only=True)
    expected = {"encoder", "affinity"} | ({"pair"} if arm != "M-lig" else set()) | ({"moments"} if arm != "P" else set())
    assert {k.split(".")[0] for k in state} == expected
    with open(os.path.join(out_dir, "trained_models", f"{tag}_0.json")) as fh:
        side = json.load(fh)
    assert side["model_format"] == "affinity_with_siamese" and side["siamese"] is False
    if arm != "M-lig":
        assert side["pair_head"] is True and (side["pair_hidden"], side["pair_head_weight"]) == (16, 0.3)
        assert "pair_weight" not in side
        assert side["pair_neg_kinds"] == ([4, 6] if arm == "PMneg" else [])
        assert side["pair_tau"] == 1.0 and side["pair_tau_std"] == pytest.approx(1.0 / summary["scaler_scale"])
        assert side["args"]["pair_weight"] == 0.3  # --pair-loss's weight, untouched
    else:
        assert side["pair_head"] is False and side["pair_head_weight"] is None
    if arm == "P":
        assert side["moments"] == "none" and side["moments_weight"] is None and side["zernike_side"] is None
    else:
        assert (side["moments"], side["moments_weight"], side["moments_dim"]) == ("both", 0.3, 343)
        assert side["moments_kinds"] == [0]
        assert side["zernike_side"] == str(side_path)
        assert side["moments_rows"]["train"]["in_side"] == side["moments_rows"]["train"]["rows"]
    assert side["sampler_parent_aware"] is (arm in ("P", "PMneg"))
    if arm == "PMneg":
        assert side["hinge_kinds"] == [] and side["train_kinds"] == [0, 3, 4, 5, 6, 7]
    # the evaluator rebuilds the wrapper with the same heads, and its pointwise prediction is the model's
    model = eval_paired.load_models([os.path.join(out_dir, "trained_models", f"{tag}_0.model")], N_FEAT, N_EDGE_FEAT,
                                    8, 3, torch.device("cpu"))[0]
    assert eval_paired.has_pair_head(model) is (arm != "M-lig")
    assert eval_paired.moments_targets(model) == (() if arm == "P" else ("pocket", "ligand"))
    assert isinstance(model, heads.AffinityWithSiamese) and model.siamese is None


def test_plain_run_sidecar_has_arm_p_defaults(tmp_path):
    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    _write_splits(data_dir)
    train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0", "--tag", "plain",
                       "--epochs", "1", "--hidden-dim", "8"])
    with open(os.path.join(out_dir, "trained_models", "plain_0.json")) as fh:
        side = json.load(fh)
    assert side["model_format"] == "gatv2net" and side["pair_head"] is False and side["moments"] == "none"
    assert side["pair_head_weight"] is None and side["pair_neg_kinds"] is None and side["moments_rows"] is None
    assert side["moments_kinds"] is None and side["sampler_parent_aware"] is False
    state = torch.load(os.path.join(out_dir, "trained_models", "plain_0.model"), weights_only=True)
    GATv2Net(N_FEAT, N_EDGE_FEAT, SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=8)).load_state_dict(state)


@pytest.mark.parametrize("extra,needle", [
    (["--pair-neg-kinds", "4", "--train-kinds", "0,4"], "--pair-neg-kinds needs --pair-head"),
    (["--pair-head", "--pair-neg-kinds", "4"], "not in --train-kinds"),
    (["--pair-head", "--pair-neg-kinds", "3", "--train-kinds", "0,3"], "hinge kinds"),
    (["--pair-head", "--pair-head-weight", "0"], "--pair-head-weight must be > 0"),
    (["--pair-head", "--pair-hidden", "0"], "--pair-hidden"),
    (["--moments", "both"], "--moments needs --zernike-side"),
    (["--zernike-side", "x.npz"], "--moments none"),
    (["--moments", "pocket", "--zernike-side", "x.npz", "--moments-weight", "0"], "--moments-weight"),
    # a zero-weight hinge kind is allowed only as a listwise negative of the pair head
    (["--train-kinds", "0,3,4,5,6", "--hinge-weight", "0", "--pair-head", "--pair-neg-kinds", "4"], "hinge kind 6"),
], ids=["neg_without_head", "neg_not_selected", "neg_not_hinge", "weight0", "hidden0", "moments_no_side",
        "side_no_moments", "moments_weight0", "zero_weight_not_listwise"])
def test_arm_p_flag_guards(capsys, extra, needle):
    with pytest.raises(SystemExit) as exc:
        train_paired.parse_args(["--data-dir", "x", "--seed_index", "0", "--tag", "t", *extra])
    assert exc.value.code not in (0, None)
    assert needle in capsys.readouterr().err


def test_arm_p_flag_defaults_and_listwise_only_kinds():
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t"]
    args = train_paired.parse_args(base)
    assert (args.pair_head, args.pair_head_weight, args.pair_hidden, args.pair_neg_kinds) == (False, 0.3, 512, ())
    assert (args.moments, args.moments_weight, args.zernike_side, args.pair_weight) == ("none", 0.3, None, 0.3)
    args = train_paired.parse_args([*base, "--train-kinds", "0,3,4,5,6,7", "--hinge-weight", "0", "--pair-head",
                                    "--pair-neg-kinds", "6,4"])
    assert args.pair_neg_kinds == (4, 6) and args.hinge_kinds == [] and args.hinge_weights == {}
    # a positive hinge weight keeps the hinge on the negatives as well
    args = train_paired.parse_args([*base, "--train-kinds", "0,3,4", "--pair-head", "--pair-neg-kinds", "4"])
    assert args.hinge_kinds == [4] and args.hinge_weights == {4: 1.0}


def test_moments_side_file_of_another_build(tmp_path):
    data_dir = str(tmp_path / "data")
    _write_aug_splits(data_dir, (3, 4, 5, 6))
    side_path = tmp_path / "other.npz"
    _side_from_pt(data_dir, side_path, series_shift=1)  # same rows, other series ids
    with pytest.raises(SystemExit) as exc:
        train_paired.main(["--data-dir", data_dir, "--out-dir", str(tmp_path / "output"), "--seed_index", "0",
                           "--tag", "bad", "--epochs", "1", "--hidden-dim", "8", "--moments", "ligand",
                           "--zernike-side", str(side_path)])
    assert "does not belong" in str(exc.value.code) and "series_id" in str(exc.value.code)
    assert not os.path.exists(os.path.join(tmp_path, "output", "trained_models", "bad_0.model"))
    assert not os.path.exists(os.path.join(tmp_path, "output", "trained_models", "bad.pickle"))


def _side_without_rows(data_dir, path, drop):
    """_side_from_pt, then drop the rows `drop` from the side file."""
    import build_zernike_side as bzs

    arrays = _side_from_pt(data_dir, path)
    keep = ~np.isin(arrays["row"], list(drop))
    n = len(arrays["row"])
    bzs.write_npz(str(path), {k: (v[keep] if v.ndim and v.shape[0] == n else v) for k, v in arrays.items()})


def test_moments_side_missing_rows(tmp_path, capsys):
    """A data row missing from the side file is an error; --allow-partial-side masks it (counted in the sidecar)."""
    data_dir = str(tmp_path / "data")
    meta = _write_aug_splits(data_dir, (3, 4))
    side_path = tmp_path / "partial.npz"
    dropped = [int(meta["valid"].row.iloc[0]), int(meta["train"].row.iloc[0])]
    _side_without_rows(data_dir, side_path, dropped)
    base = ["--data-dir", data_dir, "--out-dir", str(tmp_path / "output"), "--seed_index", "0", "--epochs", "1",
            "--hidden-dim", "8", "--moments", "both", "--zernike-side", str(side_path)]
    with pytest.raises(SystemExit) as exc:
        train_paired.main([*base, "--tag", "bad"])
    assert "not in the side file" in str(exc.value.code) and "--allow-partial-side" in str(exc.value.code)
    assert not os.path.exists(tmp_path / "output" / "trained_models" / "bad.pickle")
    train_paired.main([*base, "--tag", "ok", "--allow-partial-side"])
    with open(tmp_path / "output" / "trained_models" / "ok_0.json") as fh:
        side = json.load(fh)
    rows = side["moments_rows"]
    assert rows["valid"]["in_side"] == rows["valid"]["rows"] - 1 and rows["train"]["in_side"] == rows["train"]["rows"] - 1
    # pocket / ligand count the kind-0 rows only (--moments-kinds 0): 12 crystals minus the dropped one
    assert rows["train"]["pocket"] == 11 and side["moments_kinds"] == [0]


def test_moments_kinds_flag():
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t"]
    args = train_paired.parse_args([*base, "--moments", "both", "--zernike-side", "x.npz"])
    assert args.moments_kinds == (0,) and args.allow_partial_side is False
    args = train_paired.parse_args([*base, "--train-kinds", "0,3,4", "--moments", "pocket", "--zernike-side", "x.npz",
                                    "--moments-kinds", "4,0"])
    assert args.moments_kinds == (0, 4)
    for bad in (["--moments-kinds", "0"], ["--allow-partial-side"],
                ["--moments", "both", "--zernike-side", "x.npz", "--moments-kinds", "3"],
                ["--moments", "both", "--zernike-side", "x.npz", "--moments-kinds", "9"]):
        with pytest.raises(SystemExit):
            train_paired.parse_args([*base, *bad])


@pytest.mark.parametrize("extra", [
    ["--train-kinds", "0,3,5,7", "--pair-head", "--moments", "both", "--zernike-side", "x.npz",
     "--sampler-parent-aware"],
    ["--train-kinds", "0,3,4,5,6,7", "--hinge-weight", "0", "--pair-head", "--pair-neg-kinds", "4,6", "--moments",
     "both", "--zernike-side", "x.npz", "--sampler-parent-aware"],
], ids=["PMpos", "PMneg"])
def test_control_arm_flags_accepted(extra):
    args = train_paired.parse_args(["--data-dir", "x", "--seed_index", "0", "--tag", "t", *extra])
    assert args.pair_head and args.sampler_parent_aware and args.moments == "both"
    assert args.hinge_kinds == []  # PMpos: no hinge kind selected; PMneg: kinds 4 / 6 listwise only


# ----------------------------------------------------------------------------- parent-aware sampler

def _family_layout(seed=0):
    """Series ids, row ids and parents of a shuffled index set: series 0 with 3 parents (2, 7 and 10 aug rows),
    series 1 with 4 parents without aug rows, series 2 with one parent and an orphan aug row (parent absent)."""
    rng = np.random.default_rng(seed)
    entries, row = [], 100
    for series, n_aug in ((0, 2), (0, 7), (0, 10), (1, 0), (1, 0), (1, 0), (1, 0), (2, 1)):
        parent = row
        entries.append((series, row, row))
        row += 1
        for _ in range(n_aug):
            entries.append((series, row, parent))
            row += 1
    entries.append((2, row, 999))  # orphan: its parent is not in the set
    entries = [entries[k] for k in rng.permutation(len(entries))]
    series, rows, parents = (np.array(v) for v in zip(*entries))
    return series, rows, parents


def test_parent_aware_sampler():
    from paired_data import SeriesBatchSampler

    series, rows, parents = _family_layout()
    n = len(series)
    kw = dict(batch_size=12, chunk=8, seed=5)
    sampler = SeriesBatchSampler(series, parent_aware=True, rows=rows, parents=parents, **kw)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(n))  # every index exactly once
        again = SeriesBatchSampler(series, parent_aware=True, rows=rows, parents=parents, **kw)
        again.set_epoch(epoch)
        assert list(again) == batches  # deterministic per (seed, epoch)
        batch_of = {i: k for k, b in enumerate(batches) for i in b}
        pos_of_row = {int(r): i for i, r in enumerate(rows)}
        for root in (i for i in range(n) if parents[i] == rows[i]):
            family = [i for i in range(n) if parents[i] == rows[root] and i != root]
            same = sum(batch_of[i] == batch_of[root] for i in family)
            assert same >= min(len(family), kw["chunk"] - 1)  # parent + up to chunk - 1 aug rows share a chunk
        assert pos_of_row  # the orphan (parent 999) is still sampled
    sampler.set_epoch(0)
    e0 = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != e0
    # chunks: families packed whole, a family larger than chunk cut with the parent in its first piece
    perm = np.arange(n)[series == 0]
    chunks = sampler._family_chunks(perm)
    assert all(len(c) <= 8 for c in chunks) and sorted(int(i) for c in chunks for i in c) == sorted(perm.tolist())
    for c in chunks:
        members = set(c.tolist())
        for r in (int(i) for i in c if parents[i] == rows[i]):
            fam = [int(i) for i in perm if parents[i] == rows[r] and i != r]
            if len(fam) + 1 <= 8:
                assert set(fam) <= members          # the whole family in the parent's chunk
            else:
                assert len(c) == 8 and members <= set(fam) | {r}  # the first piece: parent + 7 of its aug rows
    # a series without aug rows gets exactly the chunks (and batches) of the plain sampler
    plain_series = series[series == 1]
    pr = rows[series == 1]
    a = SeriesBatchSampler(plain_series, parent_aware=True, rows=pr, parents=pr, **kw)
    b = SeriesBatchSampler(plain_series, **kw)
    assert list(a) == list(b)
    with pytest.raises(ValueError):
        SeriesBatchSampler(series, parent_aware=True, **kw)
    with pytest.raises(ValueError):
        SeriesBatchSampler(series, parent_aware=True, rows=rows[:3], parents=parents[:3], **kw)


def test_parent_aware_training_batches(tmp_path, monkeypatch):
    """--sampler-parent-aware: every crystal shares its training batch with all of its augmented rows (families of 5
    within a chunk of 8), each row once per epoch."""
    data_dir, out_dir = str(tmp_path / "data"), str(tmp_path / "output")
    meta = _write_aug_splits(data_dir, (3, 4, 5, 6))
    seen = _record_batches(monkeypatch)
    train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0", "--tag", "pa",
                       "--epochs", "2", "--hidden-dim", "8", "--batch-size", "16", "--chunk", "8",
                       "--train-kinds", "0,3,4,5,6", "--sampler-parent-aware"])
    train = meta["train"]
    assert sorted(r for rows, _ in seen for r in rows) == sorted(train.row.tolist() * 2)
    parent_of = dict(zip(train.row, train.parent))
    for rows, _ in seen:
        here = set(rows)
        for r in rows:
            assert parent_of[r] in here  # an aug row's crystal is in its batch (and a crystal's aug rows)
        crystals = [r for r in rows if parent_of[r] == r]
        assert all(sum(parent_of[x] == c for x in rows) == 5 for c in crystals)


# ----------------------------------------------------------------------------- moments as input (arm Z)

ARM_Z = {
    "Zbx": ["--input-moments", "bothx", "--input-moments-proj", "16"],
    "Zlin": ["--model", "moments_only", "--input-moments", "bothx"],
    "Zb-lig": ["--input-moments", "both", "--ligand-only"],
}


def _paired_meta(data_dir):
    """paired_meta.csv-like frame over the four splits of data_dir (unique ids as _side_from_pt)."""
    from paired_data import PairedDataset

    frames = []
    for split in ("train", "valid", "test", "series_valid"):
        d = PairedDataset(data_dir, split)._data
        rows = d.row.view(-1).numpy()
        kind = d.kind.view(-1).numpy() if getattr(d, "kind", None) is not None else np.zeros(len(rows), int)
        parent = d.parent.view(-1).numpy() if getattr(d, "parent", None) is not None else rows
        series = d.series.view(-1).numpy()
        frames.append(pd.DataFrame({"row": rows, "split": split, "kind": kind, "parent": parent,
                                    "series_id": series, "lig_id": d.lig.view(-1).numpy(),
                                    "series_key": [f"src:s{v}" for v in series], "source": "bindingnet",
                                    "uniprot": "P1", "pK": d.y.view(-1).numpy().astype(float)}))
    meta = pd.concat(frames, ignore_index=True)
    meta["unique_id"] = [f"u{r}" if k == 0 else f"u{p}::{k}_{r}" for r, k, p in zip(meta.row, meta.kind, meta.parent)]
    meta["lig_key"] = [f"l{v}" for v in meta["lig_id"]]
    return meta


def _fabricate_fep(fep_dir, rng, n=12):
    """fep_benchmark.csv (2 series), a SHUFFLED fep_benchmark_processed.csv (one ligand dropped), the .pt in processed-
    CSV order (y = graph_id) and zernike_fep.npz in the featurise_zernike layout (another order; one ligand failed,
    one absent). Returns (fep frame, processed frame, npz path)."""
    os.makedirs(fep_dir / "processed", exist_ok=True)
    fep = pd.DataFrame({"unique_id": [f"sub/s{k % 2}/lig{k}" for k in range(n)], "pK": rng.normal(6, 1, n),
                        "sdf_file": "x.sdf", "pdb_file": "x.pdb", "subset": "sub", "series": [f"s{k % 2}" for k in range(n)],
                        "ref_pdb": "1ABC", "n_entries": 1})
    fep.to_csv(fep_dir / "fep_benchmark.csv", index=False)
    proc = fep.drop(index=[5]).sample(frac=1.0, random_state=3).reset_index(drop=True)
    proc.to_csv(fep_dir / "fep_benchmark_processed.csv", index=False)
    graphs = []
    for g in range(len(proc)):
        k = int(rng.integers(3, 7))
        chain = [(a, a + 1) for a in range(k - 1)]
        edges = chain + [(b, a) for a, b in chain]
        graphs.append(Data(x=torch.tensor(rng.normal(size=(k, N_FEAT)), dtype=torch.float),
                           edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
                           edge_attr=torch.tensor(rng.random((len(edges), N_EDGE_FEAT)), dtype=torch.float),
                           y=torch.tensor([g], dtype=torch.int32)))
    InMemoryDataset.save(graphs, str(fep_dir / "processed" / "fep_benchmark.pt"))
    uids = [u for u in fep["unique_id"] if u != "sub/s1/lig7"][::-1]  # lig7 absent, file order reversed
    m = len(uids)
    pocket = rng.normal(5.0, 2.0, (m, 7, 49)).astype(np.float32)
    ligand = rng.normal(-3.0, 1.0, (m, 7, 49)).astype(np.float32)
    ok = np.array([u != "sub/s0/lig2" for u in uids])
    pocket[~ok] = np.nan
    ligand[~ok] = np.nan
    path = fep_dir / "zernike_fep.npz"
    np.savez_compressed(path, unique_id=np.array(uids), pocket_F=pocket, ligand_F=ligand, ok=ok,
                        error=np.array(["" if o else "ValueError: x" for o in ok]),
                        pocket_total=np.zeros((m, 7), np.float32), ligand_total=np.zeros((m, 7), np.float32))
    return fep, proc, path


@pytest.mark.parametrize("arm", list(ARM_Z))
def test_train_eval_arm_z_smoke(tmp_path, capsys, arm):
    """Zbx / Zlin / Zb-lig on synthetic data with a fabricated side file (one failed crystal) and zernike_fep.npz:
    training runs with finite losses, the sidecar records the moments input, the evaluator rebuilds the model,
    reproduces the trainer's predictions on series_valid and test, and evaluates FEP through graph_id -> unique_id
    with the side file's statistics."""
    import build_zernike_side as bzs
    import eval_paired
    import heads

    data_dir, out_dir = tmp_path / "data", str(tmp_path / "output")
    meta = _write_aug_splits(str(data_dir), (3,))
    failed = int(meta["train"].row.iloc[1])
    side_path = data_dir / "paired_zernike.npz"
    _side_from_pt(str(data_dir), side_path, failed=(failed, int(meta["series_valid"].row.iloc[0])))
    tag = arm.lower()
    summary = train_paired.main(["--data-dir", str(data_dir), "--out-dir", out_dir, "--seed_index", "0", "--tag", tag,
                                 "--epochs", "2", "--hidden-dim", "8", "--batch-size", "8", "--chunk", "4",
                                 "--zernike-side", str(side_path), *ARM_Z[arm]])
    out = capsys.readouterr().out
    epochs = [m for m in (LOG_LINE.match(line) for line in out.splitlines()) if m]
    assert [int(m.group(1)) for m in epochs] == [1, 2] and all(np.isfinite(float(m.group(2))) for m in epochs)
    assert "input moments" in out
    mode = ARM_Z[arm][ARM_Z[arm].index("--input-moments") + 1]
    with open(os.path.join(out_dir, "trained_models", f"{tag}_0.json")) as fh:
        side = json.load(fh)
    assert side["input_moments"] == mode and side["input_dim"] == heads.input_moments_dim(mode)
    assert side["model_arch"] == ("moments_only" if arm == "Zlin" else "gatv2") == side["args"]["model"]
    assert side["model"].endswith(f"{tag}_0.model")  # the top-level "model" stays the checkpoint path
    assert side["input_moments_proj"] == {"Zbx": 16, "Zlin": None, "Zb-lig": 256}[arm]
    assert side["model_format"] == ("moments_only" if arm == "Zlin" else "affinity_with_siamese")
    assert side["zernike_side"] == str(side_path) and side["moments"] == "none" and side["moments_rows"] is None
    assert side["zernike_side_stats"] == heads.side_stats_digest(bzs.load_side(str(side_path)))
    n_train = int((meta["train"].kind == 0).sum())  # --train-kinds 0: the crystals; one failed
    assert side["input_moments_rows"]["train"] == {"rows": n_train, "missing": 1}
    assert side["input_moments_rows"]["series_valid"]["missing"] == 1
    assert side["args"]["ligand_only"] is (arm == "Zb-lig")
    state = torch.load(os.path.join(out_dir, "trained_models", f"{tag}_0.model"), weights_only=True)
    if arm == "Zlin":
        assert heads.moments_only_from_state_dict(state) == (1030, (512, 256))
    else:
        assert {k.split(".")[0] for k in state} == {"encoder", "affinity", "moments_in"}
        assert heads.fc1_input_width(state) == 48 + (16 if arm == "Zbx" else 256)
    for split in ("train", "valid", "test", "series_valid"):
        pred = pd.read_csv(os.path.join(out_dir, "predictions", f"{tag}_{split}_seed0.csv"))
        assert np.isfinite(pred["pred"]).all()

    # evaluation: FEP (fabricated zernike_fep.npz), series_valid and test from the side file
    paired = _paired_meta(str(data_dir))
    paired.to_csv(data_dir / "paired_meta.csv", index=False)
    fep, proc, fep_npz = _fabricate_fep(data_dir, np.random.default_rng(4))
    model_path = os.path.join(out_dir, "trained_models", f"{tag}_0.model")
    scaler_path = os.path.join(out_dir, "trained_models", f"{tag}.pickle")
    ev = tmp_path / "eval"
    common = ["--models", model_path, "--scaler", scaler_path, "--meta", str(data_dir / "paired_meta.csv"),
              "--fep-csv", str(data_dir / "fep_benchmark.csv"), "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt"),
              "--series-valid-pt", str(data_dir / "processed" / "paired_series_valid.pt"),
              "--test-pt", str(data_dir / "processed" / "paired_test.pt"), "--pdbbind-index-tgz", str(tmp_path / "x.tgz"),
              "--hidden-dim", "8", "--head", "3", "--device", "cpu", "--pairwise"]
    eval_paired.main([*common, "--zernike-side", str(side_path), "--zernike-eval", str(fep_npz), "--out-dir", str(ev)])
    for split in ("series_valid", "test"):  # the evaluator attaches the same inputs as the trainer
        got = pd.read_csv(ev / f"predictions_{split}.csv")
        ref = pd.read_csv(os.path.join(out_dir, "predictions", f"{tag}_{split}_seed0.csv"))
        assert got["row"].tolist() == ref["row"].tolist()
        np.testing.assert_allclose(got["pred"], ref["pred"], atol=1e-4)
    fp = pd.read_csv(ev / "predictions_fep.csv")
    assert len(fp) == len(proc) and np.isfinite(fp["pred"]).all()
    text = (ev / "SUMMARY.md").read_text()
    assert f"pocket descriptor for {len(proc) - 2} of {len(proc)} ligands" in text  # lig2 failed, lig7 absent
    if arm == "Zlin":  # FEP by hand: graph_id -> processed row -> unique_id -> descriptor, side-file statistics
        import pickle

        model = eval_paired.load_models([model_path], N_FEAT, N_EDGE_FEAT, 8, 3, torch.device("cpu"))[0].eval()
        sd = bzs.load_side(str(side_path))
        z = np.load(fep_npz)
        pos = {u: k for k, u in enumerate(z["unique_id"])}
        uids = fp["unique_id"].tolist()
        raw_p = np.stack([z["pocket_F"][pos[u]].reshape(-1) if u in pos else np.full(343, np.nan) for u in uids])
        raw_l = np.stack([z["ligand_F"][pos[u]].reshape(-1) if u in pos else np.full(343, np.nan) for u in uids])
        mp, ml = np.isfinite(raw_p).all(1), np.isfinite(raw_l).all(1)
        p_std = np.where(mp[:, None], (raw_p - sd["pocket_mean"]) / sd["pocket_sd"], 0.0)
        l_std = np.where(ml[:, None], (raw_l - sd["ligand_mean"]) / sd["ligand_sd"], 0.0)
        mat = heads.moments_input_matrix(torch.tensor(p_std), torch.tensor(l_std), torch.tensor(mp), torch.tensor(ml),
                                         "bothx")
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        with torch.no_grad():
            raw = model(SimpleNamespace(moments=mat)).view(-1).double().numpy()
        np.testing.assert_allclose(fp["pred"], scaler.inverse_transform(raw.reshape(-1, 1)).ravel(), atol=1e-4)
        assert (mat[:, -1] == 1).sum() == 2
    # without --zernike-eval the FEP section cannot run: an error, nothing guessed
    with pytest.raises(SystemExit) as exc:
        eval_paired.main([*common, "--zernike-side", str(side_path), "--out-dir", str(tmp_path / "e2")])
    assert "--zernike-eval" in str(exc.value.code)


def test_input_moments_none_byte_identical(tmp_path):
    """A run with --input-moments none saves exactly the bytes of a run without the flag (plain and Siamese), and
    the plain checkpoint is a GATv2Net state dict (re-saved from a GATv2Net: the same bytes)."""
    data_dir = str(tmp_path / "data")
    _write_splits(data_dir)
    for extra in ([], ["--siamese", "--siamese-hidden", "8"]):
        paths = []
        for name, flag in (("a", []), ("b", ["--input-moments", "none"])):
            out_dir = str(tmp_path / f"out_{name}_{len(extra)}")
            train_paired.main(["--data-dir", data_dir, "--out-dir", out_dir, "--seed_index", "0", "--tag", "t",
                               "--epochs", "2", "--hidden-dim", "8", *extra, *flag])
            paths.append(os.path.join(out_dir, "trained_models", "t_0.model"))
        a, b = (open(p, "rb").read() for p in paths)
        assert a == b, extra
        if not extra:
            gnn = GATv2Net(N_FEAT, N_EDGE_FEAT, SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=8))
            gnn.load_state_dict(torch.load(paths[0], weights_only=True))
            again = tmp_path / "resaved" / "t_0.model"
            again.parent.mkdir()
            torch.save(gnn.state_dict(), again)
            assert again.read_bytes() == a
        with open(paths[1][:-len(".model")] + ".json") as fh:
            side = json.load(fh)
        assert side["input_moments"] == "none" and side["input_dim"] is None and side["model_arch"] == "gatv2"
        assert side["zernike_side"] is None and side["input_moments_proj"] is None


def test_moments_only_no_graph_leakage(tmp_path):
    """Zlin trained on the same rows with every node / edge feature zeroed gives the same checkpoint bytes and
    predictions: nothing of the graph reaches the moments-only model through the collate."""
    import shutil

    data_dir = tmp_path / "data"
    _write_aug_splits(str(data_dir), (3,))
    side_path = tmp_path / "side.npz"
    _side_from_pt(str(data_dir), side_path)
    blank = tmp_path / "blank"
    shutil.copytree(data_dir, blank)
    from paired_data import PairedDataset
    for split in ("train", "valid", "test", "series_valid"):
        ds = PairedDataset(str(blank), split)
        graphs = []
        for k in range(len(ds)):
            d = ds[k]
            d.x = torch.zeros_like(d.x)
            d.edge_attr = torch.zeros_like(d.edge_attr)
            graphs.append(d)
        os.remove(os.path.join(blank, "processed", f"paired_{split}.pt"))
        InMemoryDataset.save(graphs, os.path.join(blank, "processed", f"paired_{split}.pt"))
        reread = PairedDataset(str(blank), split)._data
        assert float(reread.x.abs().sum()) == 0.0 and float(reread.edge_attr.abs().sum()) == 0.0
        assert float(PairedDataset(str(data_dir), split)._data.x.abs().sum()) > 0
    preds = {}
    for name, ddir in (("data", data_dir), ("blank", blank)):
        out_dir = tmp_path / f"out_{name}"
        train_paired.main(["--data-dir", str(ddir), "--out-dir", str(out_dir), "--seed_index", "0", "--tag", "zlin",
                           "--epochs", "2", "--batch-size", "8", "--chunk", "4", "--model", "moments_only",
                           "--input-moments", "bothx", "--zernike-side", str(side_path)])
        preds[name] = ((out_dir / "trained_models" / "zlin_0.model").read_bytes(),
                       {s: pd.read_csv(out_dir / "predictions" / f"zlin_{s}_seed0.csv")
                        for s in ("train", "valid", "test", "series_valid")})
    assert preds["data"][0] == preds["blank"][0]
    for split, df in preds["data"][1].items():
        pd.testing.assert_frame_equal(df, preds["blank"][1][split], check_exact=True)


@pytest.mark.parametrize("extra,needle", [
    (["--input-moments", "both"], "--input-moments needs --zernike-side"),
    (["--model", "moments_only", "--zernike-side", "x.npz"], "--model moments_only needs --input-moments"),
    (["--model", "moments_only", "--input-moments", "both", "--zernike-side", "x.npz", "--ligand-only"],
     "--ligand-only"),
    (["--model", "moments_only", "--input-moments", "both", "--zernike-side", "x.npz", "--siamese"], "--siamese"),
    (["--model", "moments_only", "--input-moments", "both", "--zernike-side", "x.npz", "--input-moments-proj", "8"],
     "--input-moments-proj"),
    (["--input-moments-proj", "8"], "--input-moments-proj given with --input-moments none"),
    (["--input-moments", "both", "--zernike-side", "x.npz", "--input-moments-proj", "-1"], "must be >= 0"),
    (["--allow-partial-side"], "--allow-partial-side"),
    (["--input-moments", "all", "--zernike-side", "x.npz"], "invalid choice"),
], ids=["no_side", "moments_only_no_input", "moments_only_lig", "moments_only_siamese", "moments_only_proj",
        "proj_without_input", "negative_proj", "partial_without_side", "bad_mode"])
def test_arm_z_flag_guards(capsys, extra, needle):
    with pytest.raises(SystemExit) as exc:
        train_paired.parse_args(["--data-dir", "x", "--seed_index", "0", "--tag", "t", *extra])
    assert exc.value.code not in (0, None)
    assert needle in capsys.readouterr().err


def test_arm_z_flag_defaults():
    base = ["--data-dir", "x", "--seed_index", "0", "--tag", "t"]
    args = train_paired.parse_args(base)
    assert (args.input_moments, args.input_moments_proj, args.model) == ("none", None, "gatv2")
    args = train_paired.parse_args([*base, "--input-moments", "bothx", "--zernike-side", "x.npz"])
    assert args.input_moments_proj == 256 and train_paired.wants_wrapper(args) and train_paired.uses_side(args)
    args = train_paired.parse_args([*base, "--input-moments", "pocket", "--zernike-side", "x.npz",
                                    "--input-moments-proj", "0", "--allow-partial-side"])
    assert args.input_moments_proj == 0 and args.allow_partial_side
    args = train_paired.parse_args([*base, "--model", "moments_only", "--input-moments", "bothx",
                                    "--zernike-side", "x.npz"])
    assert args.input_moments_proj is None and not train_paired.wants_wrapper(args)
    # an input-moments arm may carry the arm-P / arm-S heads as well (they read z only)
    args = train_paired.parse_args([*base, "--input-moments", "bothx", "--zernike-side", "x.npz", "--pair-head",
                                    "--moments", "both", "--siamese"])
    assert args.pair_head and args.moments == "both" and args.siamese
