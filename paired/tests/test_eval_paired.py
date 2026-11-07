"""Tests for src/eval_paired.py: metrics, the weighted-mean rule, the UniProt lookup,
ligand-only masking, sidecar flags, and an end-to-end CLI run on synthetic .pt files."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch_geometric.data import Data, InMemoryDataset  # noqa: E402
from torch_geometric.nn import global_add_pool  # noqa: E402

import eval_paired  # noqa: E402


def test_series_metrics_on_toy():
    df = pd.DataFrame({
        "series": ["A"] * 4 + ["B"] * 3 + ["C"] * 4,
        "truth": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4],
        "pred": [2, 3, 4, 5, 3, 2, 1, 1, 3, 2, 4],
    })
    per = eval_paired.per_series_metrics(df, ["series"]).set_index("series")
    assert per.loc["A", "spearman"] == pytest.approx(1.0)
    assert per.loc["A", "pearson"] == pytest.approx(1.0)
    assert per.loc["A", "rmse"] == pytest.approx(1.0)
    assert per.loc["B", "spearman"] == pytest.approx(-1.0)
    assert per.loc["B", "rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert per.loc["C", "spearman"] == pytest.approx(0.8)
    assert per.loc["C", "kendall"] == pytest.approx(4 / 6)
    assert per["n"].tolist() == [4, 3, 4]


def test_weighted_mean_rule():
    per = pd.DataFrame({"series": ["s5", "s10", "s20"], "n": [5, 10, 20],
                        "pearson": [0.0, 0.5, 0.8], "spearman": [0.0, 0.5, 0.8],
                        "kendall": [0.0, 0.5, 0.8], "rmse": [1.0, 1.0, 1.0]})
    summ = eval_paired.summarise(per, min_ns=(10,))
    row = summ.set_index(["scope", "weighting"])
    assert row.loc[("n>=10", "weighted"), "pearson"] == pytest.approx((10 * 0.5 + 20 * 0.8) / 30)
    assert row.loc[("n>=10", "unweighted"), "pearson"] == pytest.approx(0.65)
    assert row.loc[("n>=10", "weighted"), "n_series"] == 2
    assert row.loc[("n>=10", "weighted"), "n_items"] == 30
    assert row.loc[("all", "weighted"), "pearson"] == pytest.approx((0 + 5 + 16) / 35)
    assert row.loc[("all", "unweighted"), "pearson"] == pytest.approx(1.3 / 3)


def test_weighted_mean_skips_nan():
    assert eval_paired.weighted_mean([np.nan, 0.5], [10, 30]) == pytest.approx(0.5)
    assert np.isnan(eval_paired.weighted_mean([np.nan], [10]))


def test_train_rows_lookup():
    name_index = {"1abc": "P12345", "2xyz": "------"}
    meta = pd.DataFrame({
        "split": ["train", "train", "train", "valid", "series_valid", "train", "train", "train"],
        "uniprot": ["P12345", "P12345", "P12345", "P12345", "P12345", "Q99999", np.nan, ""],
    })
    uniprots, counts = eval_paired.train_rows_same_uniprot(["1ABC", "9zzz", "2XYZ"], meta, name_index)
    assert uniprots == ["P12345", "", ""]
    assert counts == [3, -1, -1]
    # arm 4 builds: pose rows (kind 1 / 2) of train complexes are not counted
    posed = pd.concat([meta.assign(kind=0), meta.iloc[:2].assign(kind=[1, 2])], ignore_index=True)
    assert eval_paired.train_rows_same_uniprot(["1ABC"], posed, name_index)[1] == [3]


class _AevSum(torch.nn.Module):
    """Fake model: per graph, the sum of |AEV columns|."""

    def forward(self, data):
        return global_add_pool(data.x[:, 15:].abs().sum(dim=1, keepdim=True), data.batch)


def _graphs(n, width=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = 3 + i % 3
        out.append(Data(x=torch.rand(k, width, generator=g) + 0.1,
                        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                        edge_attr=torch.ones(2, 4), y=torch.tensor([i], dtype=torch.int32)))
    return out


def test_eval_ligand_only_masks_aev():
    graphs = _graphs(7)
    x_before = [d.x.clone() for d in graphs]
    scaler = StandardScaler().fit(np.array([[-1.0], [1.0]]))  # identity transform
    preds, cols = eval_paired.predict_models([_AevSum()], scaler, graphs, torch.device("cpu"),
                                             ligand_only=True, collect=("y",), batch_size=3)
    assert preds.shape == (1, 7)
    assert np.all(preds == 0.0)
    assert cols["y"].tolist() == list(range(7))
    assert all(torch.equal(a, d.x) for a, d in zip(x_before, graphs))  # inputs untouched
    preds_full, _ = eval_paired.predict_models([_AevSum()], scaler, graphs, torch.device("cpu"),
                                               ligand_only=False, batch_size=3)
    assert np.all(preds_full > 0)


def test_sidecar_ligand_only_flags(tmp_path, capsys):
    paths = {}
    for stem, payload in [("lig_0", {"args": {"ligand_only": True}}), ("lig_1", {"ligand_only": True}),
                          ("full_0", {"args": {"ligand_only": False}}), ("bare_0", None)]:
        p = tmp_path / f"{stem}.model"
        p.write_bytes(b"")
        if payload is not None:
            (tmp_path / f"{stem}.json").write_text(json.dumps(payload))
        paths[stem] = str(p)
    assert eval_paired.resolve_ligand_only([paths["lig_0"], paths["lig_1"]], False) is True
    assert "ligand-only" in capsys.readouterr().out
    assert eval_paired.resolve_ligand_only([paths["full_0"]], False) is False
    assert eval_paired.resolve_ligand_only([paths["bare_0"]], False) is False
    assert eval_paired.resolve_ligand_only([paths["bare_0"]], True) is True
    with pytest.raises(SystemExit):
        eval_paired.resolve_ligand_only([paths["lig_0"], paths["full_0"]], False)


class _Pt(InMemoryDataset):
    def __init__(self):
        super().__init__(None)


def _save(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _Pt.save(data_list, path)


def test_cli_end_to_end(tmp_path):
    from types import SimpleNamespace

    from model_defs import GATv2Net

    rng = np.random.default_rng(0)
    width = 367

    def graph(**fields):
        k = int(rng.integers(3, 6))
        src = list(range(k - 1)) + list(range(1, k))
        dst = list(range(1, k)) + list(range(k - 1))
        return Data(x=torch.tensor(rng.random((k, width)), dtype=torch.float),
                    edge_index=torch.tensor([src, dst], dtype=torch.long),
                    edge_attr=torch.ones(len(src), 4), **fields)

    data_dir = tmp_path / "data"
    # FEP: 2 series (12 and 3 ligands) + processed CSV + .pt keyed by graph_id
    fep = pd.DataFrame({
        "unique_id": [f"subA/s1/l{i}" for i in range(12)] + [f"subB/s2/m{i}" for i in range(3)],
        "pK": rng.normal(6, 1, 15), "sdf_file": "x.sdf", "pdb_file": "x.pdb",
        "subset": ["subA"] * 12 + ["subB"] * 3, "series": ["s1"] * 12 + ["s2"] * 3,
        "ref_pdb": ["1ABC"] * 12 + ["3E64"] * 3, "n_entries": 1})
    os.makedirs(data_dir / "processed")
    fep.to_csv(data_dir / "fep_benchmark.csv", index=False)
    # process_data round-trips through pandas (3E64 -> 3e+64) and may drop rows
    processed = pd.read_csv(data_dir / "fep_benchmark.csv").drop(index=[13]).reset_index(drop=True)
    processed.to_csv(data_dir / "fep_benchmark_processed.csv", index=False)
    _save([graph(y=torch.tensor([i], dtype=torch.int32)) for i in range(len(processed))],
          str(data_dir / "processed" / "fep_benchmark.pt"))
    # paired meta + series_valid / test splits
    meta = pd.DataFrame({
        "row": range(20), "unique_id": [f"u{i}" for i in range(20)],
        "source": ["pdbbind"] * 6 + ["bindingnet"] * 8 + ["bindingdb"] * 6,
        "split": ["test"] * 6 + ["series_valid"] * 14, "pK": rng.normal(6, 1, 20),
        "series_key": [f"pdbbind:single:{i}" for i in range(6)] + ["bindingnet:1abc"] * 8 + ["bindingdb:2x_y"] * 6,
        "series_id": list(range(6)) + [6] * 8 + [7] * 6, "lig_key": [f"l{i}" for i in range(20)],
        "lig_id": range(20), "label_type": "unknown", "uniprot": ["P12345"] * 20, "anchor_pdb": "1abc"})
    meta.to_csv(data_dir / "paired_meta.csv", index=False)
    for split in ("test", "series_valid"):
        rows = meta[meta["split"] == split]
        _save([graph(y=torch.tensor([r.pK], dtype=torch.float), row=torch.tensor([r.row]),
                     series=torch.tensor([r.series_id]), lig=torch.tensor([r.lig_id]),
                     src=torch.tensor([0])) for r in rows.itertuples()],
              str(data_dir / "processed" / f"paired_{split}.pt"))
    # two tiny models + scaler + one ligand-only-free sidecar
    model_dir = tmp_path / "models"
    os.makedirs(model_dir)
    cfg = SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu")
    model_paths = []
    for i in range(2):
        torch.manual_seed(i)
        path = str(model_dir / f"tiny_{i}.model")
        torch.save(GATv2Net(node_feature_dim=width, edge_feature_dim=4, config=cfg).state_dict(), path)
        model_paths.append(path)
    (model_dir / "tiny_0.json").write_text(json.dumps({"args": {"ligand_only": False}}))
    with open(model_dir / "tiny.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)
    paper = fep[["unique_id"]].iloc[:10].assign(preds=rng.normal(6, 1, 10))
    paper.to_csv(tmp_path / "paper.csv")
    census = pd.DataFrame({"subset": ["subA", "subB"], "series": ["s1", "s2"], "record": ["HETATM", "ATOM"],
                           "residue_name": ["HOH", "ACE"], "n_residues": [5, 1], "n_dropped_atoms": [5, 3],
                           "dropped_atom_names": ["O", "C CH3 O"]})
    census.to_csv(data_dir / "fep_residue_census.csv", index=False)

    out = tmp_path / "eval"
    eval_paired.main([
        "--models", *model_paths, "--scaler", str(model_dir / "tiny.pickle"),
        "--meta", str(data_dir / "paired_meta.csv"), "--fep-csv", str(data_dir / "fep_benchmark.csv"),
        "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt"),
        "--series-valid-pt", str(data_dir / "processed" / "paired_series_valid.pt"),
        "--test-pt", str(data_dir / "processed" / "paired_test.pt"), "--out-dir", str(out),
        "--paper-csv", str(tmp_path / "paper.csv"), "--pdbbind-index-tgz", str(tmp_path / "absent.tgz"),
        "--hidden-dim", "8", "--head", "1", "--device", "cpu"])

    per = pd.read_csv(out / "fep_per_series.csv")
    assert list(per.columns) == ["subset", "series", "ref_pdb", "uniprot", "n", "train_rows_same_uniprot",
                                 "pearson", "spearman", "kendall", "rmse"]
    assert per["n"].tolist() == [12, 2]
    assert per["ref_pdb"].tolist() == ["1ABC", "3E64"]
    assert per["train_rows_same_uniprot"].isna().all()  # no PDBbind index given
    summ = pd.read_csv(out / "fep_summary.csv")
    assert set(summ["scope"]) == {"n>=10", "all"}
    preds = pd.read_csv(out / "predictions_fep.csv")
    assert len(preds) == 14 and {"unique_id", "truth", "pred", "pred_0", "pred_1"} <= set(preds.columns)
    assert np.allclose(preds["pred"], preds[["pred_0", "pred_1"]].mean(axis=1))
    truth = preds.merge(processed, on="unique_id")
    assert np.allclose(truth["truth"], truth["pK"])
    sv = pd.read_csv(out / "series_valid_per_series.csv")
    assert sorted(sv["n"].tolist()) == [6, 8]
    assert os.path.isfile(out / "series_valid_summary.csv")
    test = pd.read_csv(out / "test_summary.csv")
    assert test["n"].tolist() == [6]
    assert len(pd.read_csv(out / "predictions_test.csv")) == 6
    assert len(pd.read_csv(out / "predictions_series_valid.csv")) == 14
    text = (out / "SUMMARY.md").read_text()
    assert "joined 10" in text
    assert "HOH" in text and "ACE" in text

    # with a PDBbind index: 1ABC -> P12345 (no train rows in this meta: 0), 3E64 unknown -> -1
    import io
    import tarfile
    tgz = tmp_path / "index.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        for name, text in [("index/INDEX_general_PL_name.2020", "# c\n1abc  2000  P12345  PROT\n"),
                           ("index/INDEX_general_PL_data.2020",
                            "# c\n1abc  2.00  2000   6.00  Kd=1uM       // 1abc.pdf (ABC)\n")]:
            blob = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    out2 = tmp_path / "eval2"
    eval_paired.main([
        "--models", *model_paths, "--scaler", str(model_dir / "tiny.pickle"),
        "--meta", str(data_dir / "paired_meta.csv"), "--fep-csv", str(data_dir / "fep_benchmark.csv"),
        "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt"), "--out-dir", str(out2),
        "--pdbbind-index-tgz", str(tgz), "--hidden-dim", "8", "--head", "1", "--device", "cpu"])
    per2 = pd.read_csv(out2 / "fep_per_series.csv", keep_default_na=False)
    assert per2["uniprot"].tolist() == ["P12345", ""]
    assert per2["train_rows_same_uniprot"].tolist() == [0, -1]


# ----------------------------------------------------------------------------- pose sensitivity (arm 4)

POSE_COLUMNS = ["n_parents", "decoy_rank_acc", "decoy_gap_mean", "nn_spread", "pearson_crystal",
                "pearson_pos", "pearson_neg"]


def _toy_poses():
    # parent A (row 0): crystal 6.0; pos 5.5, 6.4; decoys 5.0, 6.5 -> max decoy beats the crystal
    # parent B (row 1): crystal 7.0; pos 7.2; decoys 4.0, 5.0
    # parent C (row 2): crystal only
    # parent D (row 3): crystal 8.5; decoy 8.0 (no pos)
    rows = [  # row, kind, parent, truth, pred
        (0, 0, 0, 5.0, 6.0), (10, 1, 0, 5.0, 5.5), (11, 1, 0, 5.0, 6.4), (12, 2, 0, 5.0, 5.0),
        (13, 2, 0, 5.0, 6.5),
        (1, 0, 1, 7.0, 7.0), (14, 1, 1, 7.0, 7.2), (15, 2, 1, 7.0, 4.0), (16, 2, 1, 7.0, 5.0),
        (2, 0, 2, 6.0, 5.0),
        (3, 0, 3, 8.0, 8.5), (17, 2, 3, 8.0, 8.0),
    ]
    return pd.DataFrame(rows, columns=["row", "kind", "parent", "truth", "pred"])


def test_pose_metrics_on_toy():
    from scipy.stats import pearsonr

    df = _toy_poses().sample(frac=1.0, random_state=0)  # order must not matter
    got = eval_paired.pose_metrics(df)
    assert list(got)[:7] == POSE_COLUMNS
    assert got["n_parents"] == 4
    assert got["decoy_rank_acc"] == pytest.approx(2 / 3)
    assert got["decoy_gap_mean"] == pytest.approx((0.25 + 2.5 + 0.5) / 3)
    assert got["nn_spread"] == pytest.approx((0.5 + 0.4 + 0.2) / 3)
    assert got["pearson_crystal"] == pytest.approx(pearsonr([5, 7, 6, 8], [6, 7, 5, 8.5])[0])
    assert got["pearson_pos"] == pytest.approx(pearsonr([5, 5, 7], [5.5, 6.4, 7.2])[0])
    assert got["pearson_neg"] == pytest.approx(pearsonr([5, 5, 7, 7, 8], [5.0, 6.5, 4.0, 5.0, 8.0])[0])
    assert got["n_pos"] == 3 and got["n_neg"] == 5 and got["n_parents_with_decoys"] == 3
    # a tie with the best decoy is not a win; no decoys / pos at all -> NaN, not an error
    tie = _toy_poses()
    tie.loc[tie.row == 16, "pred"] = 7.0
    assert eval_paired.pose_metrics(tie)["decoy_rank_acc"] == pytest.approx(1 / 3)
    crystals = _toy_poses().query("kind == 0")
    only = eval_paired.pose_metrics(crystals)
    assert only["n_parents"] == 4
    assert np.isnan(only["decoy_rank_acc"]) and np.isnan(only["nn_spread"]) and np.isnan(only["pearson_neg"])


def test_eval_pose_pt_cli(tmp_path):
    from types import SimpleNamespace

    from model_defs import GATv2Net

    rng = np.random.default_rng(1)
    width = 367
    graphs, records, row = [], [], 0
    crystal_rows = []
    for p in range(6):
        k = int(rng.integers(3, 6))
        x = torch.tensor(rng.random((k, width)), dtype=torch.float)
        ei = torch.tensor([list(range(k - 1)) + list(range(1, k)), list(range(1, k)) + list(range(k - 1))])
        pk = float(rng.normal(6, 1))
        crystal_rows.append((row, x, ei, pk))
        row += 1
    pose_specs = []
    for parent, x, ei, pk in crystal_rows:
        for kind in (1, 1, 2, 2):
            xp = x.clone()
            xp[:, 15:] = torch.tensor(rng.random((x.shape[0], width - 15)), dtype=torch.float)  # new AEVs only
            pose_specs.append((row, kind, parent, xp, ei, pk))
            row += 1
    items = [(r, 0, r, x, ei, pk) for r, x, ei, pk in crystal_rows] + pose_specs
    items.sort(key=lambda t: (t[2], t[0]))
    for r, kind, parent, x, ei, pk in items:
        graphs.append(Data(x=x, edge_index=ei, edge_attr=torch.ones(ei.shape[1], 4),
                           y=torch.tensor([pk], dtype=torch.float), row=torch.tensor([r]),
                           series=torch.tensor([parent]), lig=torch.tensor([parent]), src=torch.tensor([0]),
                           kind=torch.tensor([kind]), parent=torch.tensor([parent])))
        records.append({"row": r, "kind": kind, "parent": parent, "pK": pk})
    pt = str(tmp_path / "data" / "processed" / "paired_poses_valid.pt")
    _save(graphs, pt)

    model_dir = tmp_path / "models"
    os.makedirs(model_dir)
    cfg = SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu")
    paths = []
    for i in range(2):
        torch.manual_seed(10 + i)
        path = str(model_dir / f"tiny_{i}.model")
        torch.save(GATv2Net(node_feature_dim=width, edge_feature_dim=4, config=cfg).state_dict(), path)
        paths.append(path)
    with open(model_dir / "tiny.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)

    common = ["--models", *paths, "--scaler", str(model_dir / "tiny.pickle"), "--pose-pt", pt,
              "--hidden-dim", "8", "--head", "1", "--device", "cpu"]
    out = tmp_path / "eval"
    meta = pd.DataFrame({"row": [t[0] for t in items[:-1]], "unique_id": [f"u{t[0]}" for t in items[:-1]]})
    meta.to_csv(tmp_path / "meta.csv", index=False)  # last pose row absent: unique_id left empty
    eval_paired.main([*common, "--out-dir", str(out), "--meta", str(tmp_path / "meta.csv")])
    summ = pd.read_csv(out / "pose_summary.csv")
    assert len(summ) == 1 and list(summ.columns)[:7] == POSE_COLUMNS
    assert summ["n_parents"].iloc[0] == 6 and summ["n_pos"].iloc[0] == 12 and summ["n_neg"].iloc[0] == 12
    preds = pd.read_csv(out / "predictions_poses_valid.csv", keep_default_na=False)
    assert preds["row"].tolist() == [t[0] for t in items]
    assert preds["unique_id"].tolist() == [f"u{t[0]}" for t in items[:-1]] + [""]
    assert preds["kind"].tolist() == [t[1] for t in items]
    assert preds["parent"].tolist() == [t[2] for t in items]
    assert np.allclose(preds["truth"], [t[5] for t in items], atol=1e-5)
    assert np.allclose(preds["pred"], preds[["pred_0", "pred_1"]].mean(axis=1))
    recomputed = eval_paired.pose_metrics(preds)
    for col in POSE_COLUMNS:
        assert summ[col].iloc[0] == pytest.approx(recomputed[col], nan_ok=True)
    assert summ["nn_spread"].iloc[0] > 0  # AEV-dependent model: poses move the prediction
    text = (out / "SUMMARY.md").read_text()
    assert "## Pose sensitivity (PDBbind valid)" in text

    # ligand-only (here via the sidecar): poses differ only in the AEV block -> identical predictions
    (model_dir / "tiny_0.json").write_text(json.dumps({"args": {"ligand_only": True}}))
    out_lig = tmp_path / "eval_lig"
    eval_paired.main([*common, "--out-dir", str(out_lig)])
    lig = pd.read_csv(out_lig / "pose_summary.csv")
    assert lig["nn_spread"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert lig["decoy_gap_mean"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert lig["decoy_rank_acc"].iloc[0] == 0.0


# ----------------------------------------------------------------------------- augmented kinds 1-7, group_valid

AUG_COLUMNS = ["kind", "role", "ref_kind", "n_parents", "n_rows", "rank_acc", "gap_mean", "spread", "pearson"]


def _toy_aug():
    rows = [  # row, kind, parent, truth, pred
        # parent 0: crystal 6.0; near-native 6.3; decoy 4.0; A: tpos 5.8, twins 5.0 / 6.1 (twin wins);
        #           C: epos 6.2, contact edits 5.0 / 5.5, exposed edits 6.4 / 5.9
        (0, 0, 0, 5.0, 6.0), (10, 1, 0, 5.0, 6.3), (11, 2, 0, 5.0, 4.0), (12, 3, 0, 5.0, 5.8),
        (13, 4, 0, 5.0, 5.0), (14, 4, 0, 5.0, 6.1), (15, 5, 0, 5.0, 6.2), (16, 6, 0, 5.0, 5.0),
        (17, 6, 0, 5.0, 5.5), (18, 7, 0, 5.0, 6.4), (19, 7, 0, 5.0, 5.9),
        # parent 1: crystal 7.0; decoy 7.5 (beats the crystal); tpos 7.1, twin 6.0
        (1, 0, 1, 7.0, 7.0), (20, 2, 1, 7.0, 7.5), (21, 3, 1, 7.0, 7.1), (22, 4, 1, 7.0, 6.0),
        # parent 2: crystal 5.0; twin without its A-positive (not ranked); epos 5.5, edit 5.0, exposed 5.2
        (2, 0, 2, 6.0, 5.0), (23, 4, 2, 6.0, 4.0), (24, 5, 2, 6.0, 5.5), (25, 6, 2, 6.0, 5.0),
        (26, 7, 2, 6.0, 5.2),
        # parent 3: no crystal row: tpos 6.0 vs twin 5.0 still ranks; its controls have no spread reference
        (27, 3, 3, 8.0, 6.0), (28, 4, 3, 8.0, 5.0), (29, 7, 3, 8.0, 9.0),
    ]
    return pd.DataFrame(rows, columns=["row", "kind", "parent", "truth", "pred"])


def test_aug_metrics_on_toy():
    from scipy.stats import pearsonr

    df = _toy_aug()
    got = eval_paired.aug_metrics(df.sample(frac=1.0, random_state=1)).set_index("kind")
    assert list(got.reset_index().columns) == AUG_COLUMNS
    assert got.index.tolist() == list(range(8))
    assert got["role"].tolist() == ["crystal", "control", "negative", "control", "negative", "control",
                                    "negative", "control"]
    assert got["ref_kind"].tolist()[1:] == [0, 0, 0, 3, 0, 5, 5]
    # negatives against the positive of the same scheme
    assert (got.loc[2, "n_parents"], got.loc[2, "rank_acc"]) == (2, 0.5)
    assert got.loc[2, "gap_mean"] == pytest.approx((2.0 - 0.5) / 2)
    assert (got.loc[4, "n_parents"], got.loc[4, "rank_acc"]) == (3, pytest.approx(2 / 3))
    assert got.loc[4, "gap_mean"] == pytest.approx((0.25 + 1.1 + 1.0) / 3)
    assert (got.loc[6, "n_parents"], got.loc[6, "rank_acc"]) == (2, 1.0)
    assert got.loc[6, "gap_mean"] == pytest.approx((0.95 + 0.5) / 2)
    # controls: spread vs the crystal (kind 7: vs its C-positive, kind 5), rows without the reference skipped
    assert got.loc[1, "spread"] == pytest.approx(0.3)
    assert (got.loc[3, "n_parents"], got.loc[3, "spread"]) == (2, pytest.approx(0.15))
    assert got.loc[5, "spread"] == pytest.approx(0.35)
    # kind 7 vs kind 5: parent 0 |6.4 - 6.2|, |5.9 - 6.2|; parent 2 |5.2 - 5.5|; parent 3 has no kind 5
    assert (got.loc[7, "n_parents"], got.loc[7, "spread"]) == (2, pytest.approx(0.8 / 3))
    assert np.isnan(got.loc[4, "spread"]) and np.isnan(got.loc[7, "rank_acc"])
    # pearson: label (the parent's) vs pred over the kind's rows
    for k in (0, 4, 7):
        sub = df[df.kind == k]
        assert got.loc[k, "pearson"] == pytest.approx(pearsonr(sub.truth, sub.pred)[0])
    assert got.loc[0, "n_parents"] == 3 and got.loc[4, "n_rows"] == 5
    # kinds 0-2 agree with arm 4's pose metrics
    pose = eval_paired.pose_metrics(df[df.kind <= 2])
    assert got.loc[2, "rank_acc"] == pytest.approx(pose["decoy_rank_acc"])
    assert got.loc[2, "gap_mean"] == pytest.approx(pose["decoy_gap_mean"])
    assert got.loc[1, "spread"] == pytest.approx(pose["nn_spread"])
    # a tie with the best negative is not a win; a kind without its positive gives NaN, not an error
    tie = _toy_aug()
    tie.loc[tie.row == 22, "pred"] = 7.1
    assert eval_paired.aug_metrics(tie).set_index("kind").loc[4, "rank_acc"] == pytest.approx(1 / 3)
    lone = eval_paired.aug_metrics(_toy_aug().query("kind in (0, 6)")).set_index("kind")
    assert lone.loc[6, "n_parents"] == 0 and np.isnan(lone.loc[6, "rank_acc"])


def _tiny_models(model_dir, width, rng, seeds=(20, 21)):
    from types import SimpleNamespace

    from model_defs import GATv2Net

    os.makedirs(model_dir, exist_ok=True)
    cfg = SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu")
    paths = []
    for i, s in enumerate(seeds):
        torch.manual_seed(s)
        path = str(model_dir / f"tiny_{i}.model")
        torch.save(GATv2Net(node_feature_dim=width, edge_feature_dim=4, config=cfg).state_dict(), path)
        paths.append(path)
    with open(model_dir / "tiny.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)
    return paths


def test_eval_aug_and_group_valid_cli(tmp_path):
    rng = np.random.default_rng(2)
    width = 367

    def chain(k):
        return torch.tensor([list(range(k - 1)) + list(range(1, k)), list(range(1, k)) + list(range(k - 1))])

    # aug_valid: 5 parents, each crystal + one row of kinds 1, 2, 3, 4, 4, 5, 6, 7 (new AEVs only)
    graphs, items, row = [], [], 0
    parents = []
    for p in range(5):
        k = int(rng.integers(3, 6))
        parents.append((row, torch.tensor(rng.random((k, width)), dtype=torch.float), chain(k), float(rng.normal(6, 1))))
        row += 1
    for parent, x, ei, pk in parents:
        items.append((parent, 0, parent, x, ei, pk))
        for kind in (1, 2, 3, 4, 4, 5, 6, 7):
            xa = x.clone()
            xa[:, 15:] = torch.tensor(rng.random((x.shape[0], width - 15)), dtype=torch.float)
            items.append((row, kind, parent, xa, ei, pk))
            row += 1
    items.sort(key=lambda t: (t[2], t[0]))
    for r, kind, parent, x, ei, pk in items:
        graphs.append(Data(x=x, edge_index=ei, edge_attr=torch.ones(ei.shape[1], 4),
                           y=torch.tensor([pk], dtype=torch.float), row=torch.tensor([r]),
                           series=torch.tensor([parent]), lig=torch.tensor([parent]), src=torch.tensor([0]),
                           kind=torch.tensor([kind]), parent=torch.tensor([parent])))
    aug_pt = str(tmp_path / "data" / "processed" / "paired_aug_valid.pt")
    _save(graphs, aug_pt)

    # group_valid: 2 groups, series of 6 / 5 / 2 rows; meta covers every row of both files
    gv_rows = []
    for series_key, group, n, src in (("bindingnet:1abc", "uniprot:P1", 6, "bindingnet"),
                                      ("pdbbind:P2", "uniprot:P2", 5, "pdbbind"),
                                      ("bindingdb:2x_y", "uniprot:P2", 2, "bindingdb")):
        for _ in range(n):
            gv_rows.append((row, series_key, group, src, float(rng.normal(6, 1))))
            row += 1
    gv_graphs = []
    for r, series_key, group, src, pk in gv_rows:
        k = int(rng.integers(3, 6))
        gv_graphs.append(Data(x=torch.tensor(rng.random((k, width)), dtype=torch.float), edge_index=chain(k),
                              edge_attr=torch.ones(2 * (k - 1), 4), y=torch.tensor([pk], dtype=torch.float),
                              row=torch.tensor([r]), series=torch.tensor([0]), lig=torch.tensor([r]),
                              src=torch.tensor([0]), kind=torch.tensor([0]), parent=torch.tensor([r])))
    gv_pt = str(tmp_path / "data" / "processed" / "paired_group_valid.pt")
    _save(gv_graphs, gv_pt)
    meta = pd.DataFrame(
        [{"row": t[0], "unique_id": f"u{t[0]}", "source": "pdbbind", "split": "aug_valid" if t[1] else "valid",
          "pK": t[5], "series_key": f"pdbbind:single:{t[2]}", "group_id": "uniprot:P9", "group_leak": 0}
         for t in items]
        + [{"row": r, "unique_id": f"g{r}", "source": src, "split": "group_valid", "pK": pk, "series_key": sk,
            "group_id": g, "group_leak": 0} for r, sk, g, src, pk in gv_rows])
    meta.to_csv(tmp_path / "meta.csv", index=False)
    paths = _tiny_models(tmp_path / "models", width, rng)
    common = ["--models", *paths, "--scaler", str(tmp_path / "models" / "tiny.pickle"),
              "--hidden-dim", "8", "--head", "1", "--device", "cpu"]

    out = tmp_path / "eval"
    eval_paired.main([*common, "--meta", str(tmp_path / "meta.csv"), "--aug-pt", aug_pt,
                      "--group-valid-pt", gv_pt, "--out-dir", str(out)])
    summ = pd.read_csv(out / "aug_summary.csv")
    assert list(summ.columns) == AUG_COLUMNS
    assert summ["kind"].tolist() == list(range(8))
    assert summ.set_index("kind").loc[[2, 4, 6], "n_parents"].tolist() == [5, 5, 5]
    preds = pd.read_csv(out / "predictions_aug_valid.csv", keep_default_na=False)
    assert preds["row"].tolist() == [t[0] for t in items]
    assert preds["kind"].tolist() == [t[1] for t in items]
    assert preds["unique_id"].tolist() == [f"u{t[0]}" for t in items]
    assert np.allclose(preds["truth"], [t[5] for t in items], atol=1e-5)
    recomputed = eval_paired.aug_metrics(preds)
    for col in ("rank_acc", "gap_mean", "spread", "pearson"):
        assert np.allclose(summ[col], recomputed[col], equal_nan=True), col
    assert summ.set_index("kind").loc[7, "spread"] > 0

    gsum = pd.read_csv(out / "group_valid_summary.csv")
    assert gsum["scope"].tolist()[0] == "pointwise"
    point = gsum.iloc[0]
    assert point["n_items"] == 13 and point["n_series"] == 3
    gp = pd.read_csv(out / "predictions_group_valid.csv")
    assert gp["row"].tolist() == [t[0] for t in gv_rows]
    assert point["pearson"] == pytest.approx(eval_paired.series_metrics(gp["truth"], gp["pred"])["pearson"])
    n5 = gsum[(gsum["scope"] == "n>=5") & (gsum["weighting"] == "weighted")].iloc[0]
    assert n5["n_series"] == 2 and n5["n_items"] == 11
    per = pd.read_csv(out / "group_valid_per_series.csv")
    assert list(per.columns)[:3] == ["series_key", "source", "group_id"]
    assert sorted(per["n"].tolist()) == [2, 5, 6]
    text = (out / "SUMMARY.md").read_text()
    assert "## Augmented-row discrimination (PDBbind valid)" in text
    assert "## Group-disjoint holdout (group_valid)" in text and "target groups 2" in text

    # ligand-only: augmented rows differ only in the AEV block -> identical predictions
    (tmp_path / "models" / "tiny_0.json").write_text(json.dumps({"args": {"ligand_only": True}}))
    out_lig = tmp_path / "eval_lig"
    eval_paired.main([*common, "--aug-pt", aug_pt, "--out-dir", str(out_lig)])
    lig = pd.read_csv(out_lig / "aug_summary.csv").set_index("kind")
    assert np.allclose(lig.loc[[1, 3, 5, 7], "spread"], 0.0, atol=1e-6)
    assert (lig.loc[[2, 4, 6], "rank_acc"] == 0.0).all()

    with pytest.raises(SystemExit):  # group_valid needs the meta (series keys)
        eval_paired.main([*common, "--group-valid-pt", gv_pt, "--out-dir", str(tmp_path / "x")])


def test_eval_rejects_data_dir_mismatch(tmp_path, capsys):
    rng = np.random.default_rng(3)
    width = 367
    graphs = []
    for parent in range(3):
        for r, kind in ((parent, 0), (10 + 2 * parent, 3), (11 + 2 * parent, 4)):
            k = int(rng.integers(3, 6))
            ei = torch.tensor([list(range(k - 1)) + list(range(1, k)), list(range(1, k)) + list(range(k - 1))])
            graphs.append(Data(x=torch.tensor(rng.random((k, width)), dtype=torch.float), edge_index=ei,
                               edge_attr=torch.ones(ei.shape[1], 4), y=torch.tensor([6.0 + parent]),
                               row=torch.tensor([r]), kind=torch.tensor([kind]), parent=torch.tensor([parent])))
    data_dir = tmp_path / "data_ac"
    aug_pt = str(data_dir / "processed" / "paired_aug_valid.pt")
    _save(graphs, aug_pt)
    paths = _tiny_models(tmp_path / "models", width, rng)
    common = ["--models", *paths, "--scaler", str(tmp_path / "models" / "tiny.pickle"), "--aug-pt", aug_pt,
              "--hidden-dim", "8", "--head", "1", "--device", "cpu"]
    (tmp_path / "models" / "tiny_0.json").write_text(json.dumps({"args": {"data_dir": str(tmp_path / "data")}}))
    with pytest.raises(SystemExit) as exc:
        eval_paired.main([*common, "--out-dir", str(tmp_path / "e1")])
    assert "data-dir mismatch" in str(exc.value.code) and "tiny_0.model" in str(exc.value.code)
    assert not os.path.exists(tmp_path / "e1" / "aug_summary.csv")
    with pytest.raises(SystemExit):  # the series_valid / group_valid files are checked the same way
        eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "tiny.pickle"),
                          "--meta", "m.csv", "--group-valid-pt", aug_pt, "--out-dir", str(tmp_path / "e0")])
    eval_paired.main([*common, "--out-dir", str(tmp_path / "e2"), "--allow-data-dir-mismatch"])
    assert "WARNING: data-dir mismatch" in capsys.readouterr().out
    assert os.path.isfile(tmp_path / "e2" / "aug_summary.csv")
    # the training build itself passes (paths compared after normalisation: trailing slash ignored)
    (tmp_path / "models" / "tiny_0.json").write_text(json.dumps({"args": {"data_dir": str(data_dir) + "/"}}))
    eval_paired.main([*common, "--out-dir", str(tmp_path / "e3")])
    assert os.path.isfile(tmp_path / "e3" / "aug_summary.csv")


# ----------------------------------------------------------------------------- pairwise ranking (arm S)

def _toy_pairwise():
    # series A: 3 ligands; series B: 4 rows, rows 3 and 4 carry the same ligand; series C: a singleton
    return pd.DataFrame({
        "series": ["A", "A", "A", "B", "B", "B", "B", "C"],
        "lig": ["a0", "a1", "a2", "b0", "b0", "b1", "b2", "c0"],
        "truth": [5.0, 6.0, 6.2, 7.0, 7.4, 6.0, 8.0, 5.0],
        "pred": [5.5, 5.0, 7.0, 7.0, 7.0, 6.5, 7.5, 6.0],
    })


def test_pairwise_metrics_on_toy():
    from scipy.stats import kendalltau

    df = _toy_pairwise()
    i, j = eval_paired.series_pairs(df, ["series"], lig=df["lig"])
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1), (0, 2), (1, 2), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    # A: dy = -1.0, -1.2, -0.2; B: delta = dy exactly (dy = 1.0, -1.0, 1.4, -0.6, -2.0)
    delta = np.array([-0.5, -1.5, 0.3, 1.0, -1.0, 1.4, -0.6, -2.0])
    per = eval_paired.pairwise_per_series(df, ["series"], i, j, delta).set_index("series")
    assert list(per.reset_index().columns) == ["series"] + eval_paired.PAIRWISE_COLUMNS
    assert per.index.tolist() == ["A", "B"]  # the singleton series has no pair
    assert per["n"].tolist() == [3, 4] and per["n_pairs"].tolist() == [3, 5]
    assert per["n_dir_pairs"].tolist() == [2, 5]  # A: |dy| = 0.2 is below 0.5
    a, b = per.loc["A"], per.loc["B"]
    # pair head, series A: residuals 0.5, -0.3, 0.5; direction on (0,1), (0,2) only: both right
    assert a["pair_rmse"] == pytest.approx(np.sqrt(0.59 / 3))
    assert a["pair_dir_acc"] == pytest.approx(1.0)
    # scores s = mean_j delta_ij: s0 = (-0.5 - 1.5) / 2 = -1.0, s1 = (0.5 + 0.3) / 2 = 0.4,
    # s2 = (1.5 - 0.3) / 2 = 0.6 -> the order of y (5.0, 6.0, 6.2)
    assert a["kendall_pair"] == pytest.approx(1.0) and a["spearman_pair"] == pytest.approx(1.0)
    # point head, series A: f_i - f_j = 0.5, -1.5, -2.0; residuals 1.5, -0.3, -1.8
    assert a["pair_rmse_point"] == pytest.approx(np.sqrt(5.58 / 3))
    assert a["pair_dir_acc_point"] == pytest.approx(0.5)  # (0,1) wrong, (0,2) right
    assert a["kendall_point"] == pytest.approx(1 / 3)  # f = 5.5, 5.0, 7.0: one discordant pair of three
    # series B: the head is exact; s = (0, 0.4, -1.467, 1.2) follows y = (7.0, 7.4, 6.0, 8.0)
    assert b["pair_rmse"] == pytest.approx(0.0, abs=1e-12) and b["pair_dir_acc"] == 1.0
    assert b["kendall_pair"] == pytest.approx(1.0) and b["spearman_pair"] == pytest.approx(1.0)
    assert b["pair_rmse_point"] == pytest.approx(np.sqrt(2.32 / 5)) and b["pair_dir_acc_point"] == 1.0
    assert b["kendall_point"] == pytest.approx(kendalltau([7.0, 7.4, 6.0, 8.0], [7.0, 7.0, 6.5, 7.5])[0])
    # the pair order does not matter
    perm = np.random.default_rng(0).permutation(len(i))
    again = eval_paired.pairwise_per_series(df, ["series"], i[perm], j[perm], delta[perm])
    pd.testing.assert_frame_equal(again.set_index("series"), per)

    summ = eval_paired.pairwise_summary(per.reset_index(), "toy", min_ns=(4,))
    assert list(summ.columns) == eval_paired.PAIRWISE_SUMMARY_COLUMNS
    assert summ["head"].tolist() == ["pair"] * 4 + ["point"] * 4
    row = summ.set_index(["head", "scope", "weighting"])
    assert row.loc[("pair", "n>=4", "weighted"), "n_series"] == 1
    assert row.loc[("pair", "all", "weighted"), "n_items"] == 7
    assert row.loc[("pair", "all", "weighted"), "pair_rmse"] == pytest.approx(3 * np.sqrt(0.59 / 3) / 7)
    assert row.loc[("pair", "all", "unweighted"), "pair_rmse"] == pytest.approx(np.sqrt(0.59 / 3) / 2)
    assert row.loc[("point", "all", "weighted"), "pair_dir_acc"] == pytest.approx((3 * 0.5 + 4 * 1.0) / 7)
    assert row.loc[("point", "n>=4", "weighted"), "kendall"] == pytest.approx(b["kendall_point"])

    # no head: pair columns NaN, point rows only
    none = eval_paired.pairwise_per_series(df, ["series"], i, j, None)
    assert none[["pair_rmse", "pair_dir_acc", "kendall_pair", "spearman_pair"]].isna().all().all()
    assert np.allclose(none["kendall_point"], per["kendall_point"])
    only = eval_paired.pairwise_summary(none, "toy", min_ns=(4,), with_pair=False)
    assert set(only["head"]) == {"point"} and len(only) == 4
    # without ligand keys every pair of a series counts; missing keys never match each other
    i2, _ = eval_paired.series_pairs(df, ["series"])
    assert len(i2) == 3 + 6
    lig = df["lig"].where(df.index != 3, None).where(df.index != 4, "")
    assert len(eval_paired.series_pairs(df, ["series"], lig=lig)[0]) == 3 + 6


def _pairwise_inputs(tmp_path, rng, width=367):
    """FEP (12 + 3 ligands, one dropped), meta with uniprot and lig_key, and paired test /
    series_valid / group_valid .pt files. Returns (data_dir, meta)."""
    def graph(**fields):
        k = int(rng.integers(3, 6))
        src = list(range(k - 1)) + list(range(1, k))
        dst = list(range(1, k)) + list(range(k - 1))
        return Data(x=torch.tensor(rng.random((k, width)), dtype=torch.float),
                    edge_index=torch.tensor([src, dst], dtype=torch.long),
                    edge_attr=torch.ones(len(src), 4), **fields)

    data_dir = tmp_path / "data"
    os.makedirs(data_dir / "processed")
    fep = pd.DataFrame({
        "unique_id": [f"subA/s1/l{i}" for i in range(12)] + [f"subB/s2/m{i}" for i in range(3)],
        "pK": rng.normal(6, 1, 15), "sdf_file": "x.sdf", "pdb_file": "x.pdb",
        "subset": ["subA"] * 12 + ["subB"] * 3, "series": ["s1"] * 12 + ["s2"] * 3,
        "ref_pdb": ["1ABC"] * 12 + ["3E64"] * 3, "n_entries": 1})
    fep.to_csv(data_dir / "fep_benchmark.csv", index=False)
    processed = pd.read_csv(data_dir / "fep_benchmark.csv").drop(index=[13]).reset_index(drop=True)
    processed.to_csv(data_dir / "fep_benchmark_processed.csv", index=False)
    _save([graph(y=torch.tensor([i], dtype=torch.int32)) for i in range(len(processed))],
          str(data_dir / "processed" / "fep_benchmark.pt"))
    # test rows 0-5: UniProt P1 (rows 0-2; rows 1 and 2 share a ligand), P2 (3-4), unmapped (5)
    # series_valid rows 6-19: two series (rows 6 and 7 share a ligand); group_valid rows 20-30
    meta = pd.DataFrame({
        "row": range(31), "unique_id": [f"u{i}" for i in range(31)],
        "source": ["pdbbind"] * 6 + ["bindingnet"] * 8 + ["bindingdb"] * 6 + ["bindingnet"] * 6 + ["pdbbind"] * 5,
        "split": ["test"] * 6 + ["series_valid"] * 14 + ["group_valid"] * 11, "pK": rng.normal(6, 1, 31),
        "series_key": ([f"pdbbind:{u}" for u in ["P1", "P1", "P1", "P2", "P2"]] + ["pdbbind:single:5"]
                       + ["bindingnet:1abc"] * 8 + ["bindingdb:2x_y"] * 6 + ["bindingnet:g1"] * 6
                       + ["pdbbind:P3"] * 5),
        "series_id": [0, 0, 0, 1, 1, 2] + [3] * 8 + [4] * 6 + [5] * 6 + [6] * 5,
        "lig_key": ["l0", "lx", "lx"] + [f"l{i}" for i in range(3, 6)] + ["ly", "ly"]
                   + [f"l{i}" for i in range(8, 31)],
        "lig_id": range(31), "label_type": "unknown",
        "uniprot": ["P1", "P1", "P1", "P2", "P2", ""] + ["Q1"] * 14 + ["Q2"] * 6 + ["P3"] * 5,
        "anchor_pdb": "1abc", "group_id": ["uniprot:P1"] * 20 + ["uniprot:Q2"] * 6 + ["uniprot:P3"] * 5})
    meta.to_csv(data_dir / "paired_meta.csv", index=False)
    for split in ("test", "series_valid", "group_valid"):
        rows = meta[meta["split"] == split]
        _save([graph(y=torch.tensor([r.pK], dtype=torch.float), row=torch.tensor([r.row]),
                     series=torch.tensor([r.series_id]), lig=torch.tensor([r.lig_id]),
                     src=torch.tensor([0])) for r in rows.itertuples()],
              str(data_dir / "processed" / f"paired_{split}.pt"))
    return data_dir, meta


def _pairwise_cli(data_dir, model_paths, scaler_path, out, extra=()):
    eval_paired.main([
        "--models", *model_paths, "--scaler", str(scaler_path), "--meta", str(data_dir / "paired_meta.csv"),
        "--fep-csv", str(data_dir / "fep_benchmark.csv"), "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt"),
        "--series-valid-pt", str(data_dir / "processed" / "paired_series_valid.pt"),
        "--test-pt", str(data_dir / "processed" / "paired_test.pt"),
        "--group-valid-pt", str(data_dir / "processed" / "paired_group_valid.pt"),
        "--pdbbind-index-tgz", str(data_dir / "absent.tgz"), "--out-dir", str(out),
        "--hidden-dim", "8", "--head", "1", "--device", "cpu", "--pairwise", *extra])


def test_pairwise_eval_without_head(tmp_path):
    rng = np.random.default_rng(5)
    data_dir, meta = _pairwise_inputs(tmp_path, rng)
    paths = _tiny_models(tmp_path / "models", 367, rng)  # plain GATv2Net, no sidecar (released-ensemble style)
    out = tmp_path / "eval"
    _pairwise_cli(data_dir, paths, tmp_path / "models" / "tiny.pickle", out)

    summ = pd.read_csv(out / "pairwise_summary.csv")
    assert list(summ.columns) == eval_paired.PAIRWISE_SUMMARY_COLUMNS
    assert set(summ["head"]) == {"point"}
    assert set(summ["dataset"]) == {"fep", "series_valid", "group_valid", "test"}
    assert set(summ.loc[summ.dataset == "fep", "scope"]) == {"n>=10", "all"}
    assert set(summ.loc[summ.dataset == "test", "scope"]) == {"n>=5", "n>=10", "all"}
    fep = pd.read_csv(out / "pairwise_fep_per_series.csv")
    assert list(fep.columns) == ["subset", "series"] + eval_paired.PAIRWISE_COLUMNS
    assert fep["n"].tolist() == [12, 2] and fep["n_pairs"].tolist() == [66, 1]
    assert fep[["pair_rmse", "kendall_pair"]].isna().all().all()
    # the point head over all FEP ligands reproduces the pointwise FEP Kendall / Spearman
    fsum = pd.read_csv(out / "fep_summary.csv").set_index(["scope", "weighting"])
    prow = summ[(summ.dataset == "fep") & (summ.scope == "n>=10") & (summ.weighting == "weighted")].iloc[0]
    assert prow["kendall"] == pytest.approx(fsum.loc[("n>=10", "weighted"), "kendall"])
    assert prow["spearman"] == pytest.approx(fsum.loc[("n>=10", "weighted"), "spearman"])
    sv = pd.read_csv(out / "pairwise_series_valid_per_series.csv")
    assert sv["series_key"].tolist() == ["bindingdb:2x_y", "bindingnet:1abc"]
    assert sv["n"].tolist() == [6, 8] and sv["n_pairs"].tolist() == [15, 27]  # (6, 7) share a ligand
    test = pd.read_csv(out / "pairwise_test_per_series.csv")
    assert test["uniprot"].tolist() == ["P1", "P2"]  # the unmapped row is a singleton: no pair
    assert test["n"].tolist() == [3, 2] and test["n_pairs"].tolist() == [2, 1]
    gv = pd.read_csv(out / "pairwise_group_valid_per_series.csv")
    assert gv["n_pairs"].tolist() == [15, 10]
    text = (out / "SUMMARY.md").read_text()
    assert "## Pairwise ranking within series (--pairwise)" in text and "point rows only" in text

    # without --pairwise nothing pairwise is written (as before)
    out2 = tmp_path / "eval_plain"
    eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "tiny.pickle"),
                      "--meta", str(data_dir / "paired_meta.csv"),
                      "--series-valid-pt", str(data_dir / "processed" / "paired_series_valid.pt"),
                      "--out-dir", str(out2), "--hidden-dim", "8", "--head", "1", "--device", "cpu"])
    assert not any(p.name.startswith("pairwise") for p in out2.iterdir())


def _siamese_models(model_dir, width, rng, seeds=(30, 31), sidecar=True):
    """Tiny AffinityWithSiamese models (GATv2Net hidden 8, head 1 -> z dim 16; head hidden 8) with
    non-trivial head weights, their sidecars, the plain-GATv2Net copies and a scaler."""
    from types import SimpleNamespace

    import heads
    from model_defs import GATv2Net

    os.makedirs(model_dir, exist_ok=True)
    cfg = SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu")
    paths, plain_paths = [], []
    for k, s in enumerate(seeds):
        torch.manual_seed(s)
        gnn = GATv2Net(node_feature_dim=width, edge_feature_dim=4, config=cfg)
        head = heads.SiameseHead(heads.embed_dim(gnn), hidden=8)
        for p in head.parameters():
            torch.nn.init.normal_(p, std=0.5)
        wrapper = heads.AffinityWithSiamese(gnn, head)
        path = str(model_dir / f"siam_{k}.model")
        torch.save(wrapper.state_dict(), path)
        if sidecar:
            (model_dir / f"siam_{k}.json").write_text(json.dumps(
                {"args": {"ligand_only": False, "siamese": True}, "siamese": True, "siamese_hidden": 8}))
        plain = str(model_dir / f"plain_{k}.model")
        torch.save(wrapper.to_plain_state_dict(), plain)
        paths.append(path)
        plain_paths.append(plain)
    with open(model_dir / "siam.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)
    return paths, plain_paths


def test_pairwise_ensemble_averages_delta(tmp_path):
    rng = np.random.default_rng(6)
    data_dir, meta = _pairwise_inputs(tmp_path, rng)
    paths, plain_paths = _siamese_models(tmp_path / "models", 367, rng)
    scaler_path = tmp_path / "models" / "siam.pickle"
    out = tmp_path / "eval"
    _pairwise_cli(data_dir, paths, scaler_path, out)

    summ = pd.read_csv(out / "pairwise_summary.csv")
    assert set(summ["head"]) == {"pair", "point"}
    assert set(summ["dataset"]) == {"fep", "series_valid", "group_valid", "test"}
    assert "Siamese head on all 2 models" in (out / "SUMMARY.md").read_text()

    # recompute series_valid: per-model delta_hat (pK), averaged per pair, then the metrics
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    ds = eval_paired.load_pt(str(data_dir / "processed" / "paired_series_valid.pt"))
    models = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))
    assert all(eval_paired.has_siamese(m) for m in models)
    preds = pd.read_csv(out / "predictions_series_valid.csv")
    lig = meta.set_index("row").loc[preds["row"], "lig_key"].to_numpy()
    i, j = eval_paired.series_pairs(preds, ["series_key"], lig)
    per_model = eval_paired.pair_deltas(models, scaler, ds, i, j, torch.device("cpu"), batch_size=4)
    assert per_model.shape == (2, len(i)) and np.abs(per_model).max() > 1e-3
    # one pair by hand: standardised delta of model 0 times the scaler's scale
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([ds[k] for k in range(len(ds))])
    with torch.no_grad():
        models[0].eval()
        z = models[0].embed(batch)
        d0 = models[0].siamese(z[[int(i[0])]], z[[int(j[0])]]).item() * float(scaler.scale_[0])
        d0_rev = models[0].siamese(z[[int(j[0])]], z[[int(i[0])]]).item() * float(scaler.scale_[0])
    assert per_model[0, 0] == pytest.approx(d0, abs=1e-5) and d0_rev == pytest.approx(-d0, abs=1e-5)
    expected = eval_paired.pairwise_per_series(preds, ["series_key"], i, j, per_model.mean(axis=0))
    got = pd.read_csv(out / "pairwise_series_valid_per_series.csv")
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, atol=1e-6)
    single = eval_paired.pairwise_per_series(preds, ["series_key"], i, j, per_model[0])
    assert not np.allclose(single["pair_rmse"], got["pair_rmse"])  # the ensemble is not one model's head

    # the pointwise predictions of the Siamese models equal their plain GATv2Net copies
    out_plain = tmp_path / "eval_plain"
    _pairwise_cli(data_dir, plain_paths, scaler_path, out_plain)
    for name in ("fep", "series_valid", "test", "group_valid"):
        a = pd.read_csv(out / f"predictions_{name}.csv")
        b = pd.read_csv(out_plain / f"predictions_{name}.csv")
        pd.testing.assert_frame_equal(a, b)
    point = summ[summ["head"] == "point"].reset_index(drop=True)
    plain_summ = pd.read_csv(out_plain / "pairwise_summary.csv")
    pd.testing.assert_frame_equal(point, plain_summ)


def test_siamese_model_loading(tmp_path):
    rng = np.random.default_rng(7)
    paths, plain_paths = _siamese_models(tmp_path / "with", 367, rng, seeds=(40,))
    # wrapper keys without a sidecar: rebuilt from the keys (hidden width from the weights)
    nosc, _ = _siamese_models(tmp_path / "without", 367, rng, seeds=(40,), sidecar=False)
    cpu = torch.device("cpu")
    a = eval_paired.load_models(paths, 367, 4, 8, 1, cpu)[0]
    b = eval_paired.load_models(nosc, 367, 4, 8, 1, cpu)[0]
    assert eval_paired.has_siamese(a) and eval_paired.has_siamese(b) and b.siamese.hidden == 8
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    plain = eval_paired.load_models(plain_paths, 367, 4, 8, 1, cpu)[0]
    assert type(plain).__name__ == "GATv2Net" and not eval_paired.has_siamese(plain)
    # a sidecar that says siamese true next to a plain state dict is an error, as is a hidden mismatch
    (tmp_path / "with" / "plain_0.json").write_text(json.dumps({"siamese": True, "siamese_hidden": 8}))
    with pytest.raises(SystemExit):
        eval_paired.load_models(plain_paths, 367, 4, 8, 1, cpu)
    (tmp_path / "with" / "siam_0.json").write_text(json.dumps({"siamese": True, "siamese_hidden": 16}))
    with pytest.raises(SystemExit):
        eval_paired.load_models(paths, 367, 4, 8, 1, cpu)
    # siamese false in the sidecar: plain loading as before
    (tmp_path / "with" / "plain_0.json").write_text(json.dumps({"siamese": False, "args": {"siamese": False}}))
    assert type(eval_paired.load_models(plain_paths, 367, 4, 8, 1, cpu)[0]).__name__ == "GATv2Net"


# ----------------------------------------------------------------------------- pair head, moments heads (arm P)

def test_pairhead_metrics_on_toy():
    from scipy.stats import kendalltau

    df = _toy_pairwise()
    i, j = eval_paired.series_pairs(df, ["series"], lig=df["lig"])
    # series A: l_ij = l(z_i, z_j), l_ji = l(z_j, z_i) on (0,1), (0,2), (1,2); series B: l(a, c) = y_c - y_a (exact)
    y = df["truth"].to_numpy()
    l_ij = np.array([0.2, 0.0, 1.0] + [y[b] - y[a] for a, b in zip(i[3:], j[3:])])
    l_ji = np.array([1.0, 0.5, 0.3] + [y[a] - y[b] for a, b in zip(i[3:], j[3:])])
    per = eval_paired.pairwise_per_series(df, ["series"], i, j, None, pairhead=(l_ij, l_ji)).set_index("series")
    assert list(per.columns)[-3:] == eval_paired.PAIRHEAD_COLUMNS
    a, b = per.loc["A"], per.loc["B"]
    # series A, centred per anchor: anchor 0: l(0,1) = 0.2, l(0,2) = 0.0 -> mean 0.1 -> l' = 0.1, -0.1;
    # anchor 1: l(1,0) = l(1,2) = 1.0 -> l' = 0, 0; anchor 2: l(2,0) = 0.5, l(2,1) = 0.3 -> l' = 0.1, -0.1.
    # s0 = (l'(1,0) + l'(2,0)) / 2 = 0.05, s1 = (0.1 - 0.1) / 2 = 0, s2 = (-0.1 + 0) / 2 = -0.05
    assert a["kendall_pairhead"] == pytest.approx(kendalltau([5.0, 6.0, 6.2], [0.05, 0.0, -0.05])[0])
    assert a["kendall_pairhead"] == pytest.approx(-1.0)
    # direction from s_i - s_j = 0.05, 0.1 on the pairs with |dy| >= 0.5 (dy = -1.0, -1.2): both wrong
    assert a["pair_dir_acc_pairhead"] == 0.0
    # series B, l(a, c) = y_c - y_a: s_i = y_i - mean over i's partners a of (mean y of a's partners)
    # = (-0.133, 0.267, -0.933, 0.844) for y = (7.0, 7.4, 6.0, 8.0): same order, every direction right
    assert b["kendall_pairhead"] == pytest.approx(1.0) and b["pair_dir_acc_pairhead"] == 1.0
    assert per[["pair_rmse", "kendall_pair"]].isna().all().all()  # no Siamese head
    # the pair order does not matter
    perm = np.random.default_rng(1).permutation(len(i))
    again = eval_paired.pairwise_per_series(df, ["series"], i[perm], j[perm], None,
                                            pairhead=(l_ij[perm], l_ji[perm])).set_index("series")
    pd.testing.assert_frame_equal(again, per)
    summ = eval_paired.pairwise_summary(per.reset_index(), "toy", min_ns=(4,), with_pair=False, with_pairhead=True)
    assert summ["head"].tolist() == ["point"] * 4 + ["pairhead"] * 4
    row = summ.set_index(["head", "scope", "weighting"])
    assert np.isnan(row.loc[("pairhead", "all", "weighted"), "pair_rmse"])
    assert row.loc[("pairhead", "all", "weighted"), "kendall"] == pytest.approx((3 * (-1.0) + 4 * 1.0) / 7)
    assert row.loc[("pairhead", "n>=4", "weighted"), "pair_dir_acc"] == 1.0
    with pytest.raises(ValueError):
        eval_paired.pairwise_per_series(df, ["series"], i, j, None, pairhead=(l_ij[:2], l_ji))


def test_pairhead_metrics_invariant_to_anchor_offsets():
    """Adding a per-anchor constant b_a to every l(a, .) leaves every pair-head metric unchanged (pairwise and aug)."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"series": np.repeat(["A", "B", "C"], [6, 5, 4]), "lig": [f"l{k}" for k in range(15)],
                       "truth": rng.normal(6, 1, 15), "pred": rng.normal(6, 1, 15)})
    df.loc[3, "lig"] = "l2"  # a replicate pair in series A
    i, j = eval_paired.series_pairs(df, ["series"], lig=df["lig"])
    l_ij, l_ji = rng.normal(size=len(i)), rng.normal(size=len(i))
    offset = rng.normal(0, 5, len(df))
    base = eval_paired.pairwise_per_series(df, ["series"], i, j, None, pairhead=(l_ij, l_ji))
    moved = eval_paired.pairwise_per_series(df, ["series"], i, j, None, pairhead=(l_ij + offset[i], l_ji + offset[j]))
    pd.testing.assert_frame_equal(base, moved, atol=1e-12)
    # without the centring the offsets would change the scores: the test has teeth
    raw = lambda la, lb: np.bincount(np.concatenate([i, j]), weights=np.concatenate([lb, la]), minlength=len(df))  # noqa: E731
    assert not np.allclose(raw(l_ij, l_ji), raw(l_ij + offset[i], l_ji + offset[j]))
    # aug: anchor = the parent's crystal; one constant per anchor
    aug = _toy_aug()
    a, c = eval_paired.aug_pair_candidates(aug)
    score = rng.normal(size=len(c))
    anchor_offset = rng.normal(0, 5, len(aug))
    m0 = eval_paired.aug_pairhead_metrics(aug, c, score)
    m1 = eval_paired.aug_pairhead_metrics(aug, c, score + anchor_offset[a])
    pd.testing.assert_frame_equal(m0, m1, atol=1e-12)
    assert m0["kind"].tolist() == [2, 4, 6] and m0["n_parents"].tolist() == [2, 2, 2]


def _arm_p_models(model_dir, width, rng, seeds=(50, 51), pair=True, moments=("pocket", "ligand"), sidecar=True,
                  ligand_only=False):
    """Tiny AffinityWithSiamese models (z dim 16) with a pair head and / or moments heads (non-trivial weights),
    their sidecars, the plain-GATv2Net copies and a scaler."""
    from types import SimpleNamespace

    import heads
    from model_defs import GATv2Net

    os.makedirs(model_dir, exist_ok=True)
    cfg = SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu")
    paths, plain_paths = [], []
    for k, s in enumerate(seeds):
        torch.manual_seed(s)
        gnn = GATv2Net(node_feature_dim=width, edge_feature_dim=4, config=cfg)
        ph = heads.PairHead(16, hidden=8) if pair else None
        mh = {t: heads.MomentsHead(16, out=343, hidden=8) for t in moments}
        for head in [ph] + list(mh.values()):
            if head is not None:
                for p in head.parameters():
                    torch.nn.init.normal_(p, std=0.5)
        wrapper = heads.AffinityWithSiamese(gnn, None, ph, mh or None)
        path = str(model_dir / f"armp_{k}.model")
        torch.save(wrapper.state_dict(), path)
        if sidecar:
            name = {(): "none", ("pocket",): "pocket", ("ligand",): "ligand", ("pocket", "ligand"): "both"}[tuple(moments)]
            (model_dir / f"armp_{k}.json").write_text(json.dumps(
                {"args": {"ligand_only": ligand_only, "pair_head": pair, "moments": name}, "siamese": False,
                 "pair_head": pair, "pair_hidden": 8 if pair else None, "moments": name,
                 "moments_dim": 343 if moments else None}))
        plain = str(model_dir / f"plain_{k}.model")
        torch.save(wrapper.to_plain_state_dict(), plain)
        paths.append(path)
        plain_paths.append(plain)
    with open(model_dir / "armp.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)
    return paths, plain_paths


def _pairhead_l(models, ds, a, c):
    zs = [eval_paired.embed_dataset(m, ds, torch.device("cpu"), batch_size=4) for m in models]
    with torch.no_grad():
        return np.mean([m.pair(z[torch.as_tensor(a)], z[torch.as_tensor(c)]).double().numpy()
                        for m, z in zip(models, zs)], axis=0)


def test_pairwise_pairhead_eval(tmp_path):
    rng = np.random.default_rng(8)
    data_dir, meta = _pairwise_inputs(tmp_path, rng)
    paths, plain_paths = _arm_p_models(tmp_path / "models", 367, rng, moments=())
    scaler_path = tmp_path / "models" / "armp.pickle"
    out = tmp_path / "eval"
    _pairwise_cli(data_dir, paths, scaler_path, out)
    summ = pd.read_csv(out / "pairwise_summary.csv")
    assert set(summ["head"]) == {"point", "pairhead"}
    assert set(summ.loc[summ["head"] == "pairhead", "dataset"]) == {"fep", "series_valid", "group_valid", "test"}
    assert summ.loc[summ["head"] == "pairhead", "pair_rmse"].isna().all()
    assert "pair head on all 2 models" in (out / "SUMMARY.md").read_text()

    # series_valid by hand: l averaged over the models per ordered pair
    ds = eval_paired.load_pt(str(data_dir / "processed" / "paired_series_valid.pt"))
    models = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))
    assert all(eval_paired.has_pair_head(m) for m in models)
    preds = pd.read_csv(out / "predictions_series_valid.csv")
    lig = meta.set_index("row").loc[preds["row"], "lig_key"].to_numpy()
    i, j = eval_paired.series_pairs(preds, ["series_key"], lig)
    l_ij, l_ji = _pairhead_l(models, ds, i, j), _pairhead_l(models, ds, j, i)
    assert np.abs(l_ij - l_ji).max() > 1e-3
    expected = eval_paired.pairwise_per_series(preds, ["series_key"], i, j, None, pairhead=(l_ij, l_ji))
    got = pd.read_csv(out / "pairwise_series_valid_per_series.csv")
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, atol=1e-6)
    assert not np.allclose(_pairhead_l(models[:1], ds, i, j), l_ij)  # the ensemble is not one model's head

    # the point rows and the predictions equal those of the plain GATv2Net copies
    out_plain = tmp_path / "eval_plain"
    _pairwise_cli(data_dir, plain_paths, scaler_path, out_plain)
    for name in ("fep", "series_valid", "test", "group_valid"):
        pd.testing.assert_frame_equal(pd.read_csv(out / f"predictions_{name}.csv"),
                                      pd.read_csv(out_plain / f"predictions_{name}.csv"))
    point = summ[summ["head"] == "point"].reset_index(drop=True)
    pd.testing.assert_frame_equal(point, pd.read_csv(out_plain / "pairwise_summary.csv"))
    plain_per = pd.read_csv(out_plain / "pairwise_series_valid_per_series.csv")
    assert list(plain_per.columns) == ["series_key"] + eval_paired.PAIRWISE_COLUMNS  # no pair head: as before


def _aug_inputs(tmp_path, rng, width=367):
    """aug_valid .pt: parents 0-3 with a crystal and kinds 2, 3, 4, 4, 5, 6 (new AEVs each); parent 4 without its
    crystal (kinds 3, 4); returns (pt path, rows as (row, kind, parent))."""
    def chain(k):
        return torch.tensor([list(range(k - 1)) + list(range(1, k)), list(range(1, k)) + list(range(k - 1))])

    items, row = [], 0
    for p in range(5):
        k = int(rng.integers(3, 6))
        x = torch.tensor(rng.random((k, width)), dtype=torch.float)
        kinds = (0, 2, 3, 4, 4, 5, 6) if p < 4 else (3, 4)
        for kind in kinds:
            xa = x.clone()
            if kind:
                xa[:, 15:] = torch.tensor(rng.random((k, width - 15)), dtype=torch.float)
            items.append((100 + row, kind, 100 + 7 * p, xa, chain(k), float(rng.normal(6, 1))))
            row += 1
    graphs = [Data(x=x, edge_index=ei, edge_attr=torch.ones(ei.shape[1], 4), y=torch.tensor([pk]),
                   row=torch.tensor([r]), series=torch.tensor([par]), lig=torch.tensor([par]), src=torch.tensor([0]),
                   kind=torch.tensor([kd]), parent=torch.tensor([par])) for r, kd, par, x, ei, pk in items]
    pt = str(tmp_path / "data" / "processed" / "paired_aug_valid.pt")
    _save(graphs, pt)
    return pt, [(r, kd, par) for r, kd, par, *_ in items]


def test_aug_pairhead_rows(tmp_path):
    rng = np.random.default_rng(9)
    aug_pt, items = _aug_inputs(tmp_path, rng)
    paths, plain_paths = _arm_p_models(tmp_path / "models", 367, rng, moments=())
    common = ["--scaler", str(tmp_path / "models" / "armp.pickle"), "--aug-pt", aug_pt, "--hidden-dim", "8",
              "--head", "1", "--device", "cpu"]
    eval_paired.main(["--models", *paths, *common, "--out-dir", str(tmp_path / "eval")])
    eval_paired.main(["--models", *plain_paths, *common, "--out-dir", str(tmp_path / "eval_plain")])
    summ = pd.read_csv(tmp_path / "eval" / "aug_summary.csv")
    plain = pd.read_csv(tmp_path / "eval_plain" / "aug_summary.csv")
    assert "head" not in plain.columns  # no pair head: the file is written as before
    assert list(summ.columns) == list(plain.columns) + ["head"]
    pd.testing.assert_frame_equal(summ[summ["head"] == "point"].drop(columns="head").reset_index(drop=True), plain)
    ph = summ[summ["head"] == "pairhead"].set_index("kind")
    assert ph.index.tolist() == [2, 4, 6] and (ph["role"] == "negative").all()
    assert ph["ref_kind"].tolist() == [0, 3, 5] and ph["n_parents"].tolist() == [4, 4, 4]  # parent 4: no crystal
    assert ph["n_rows"].tolist() == [4, 8, 4] and ph[["spread", "pearson"]].isna().all().all()

    # by hand: anchor = the parent's crystal, l(z_crystal, z_row) averaged over the models
    ds = eval_paired.load_pt(aug_pt)
    models = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))
    df = pd.DataFrame(items, columns=["row", "kind", "parent"])
    crystal = {p: k for k, (kd, p) in enumerate(zip(df.kind, df.parent)) if kd == 0}
    for neg_kind, pos_kind in ((2, 0), (4, 3), (6, 5)):
        wins, gaps = [], []
        for p, a in crystal.items():
            pos = [k for k in range(len(df)) if df.parent[k] == p and df.kind[k] == pos_kind]
            neg = [k for k in range(len(df)) if df.parent[k] == p and df.kind[k] == neg_kind]
            lp = _pairhead_l(models, ds, [a] * len(pos), pos).mean()
            ln = _pairhead_l(models, ds, [a] * len(neg), neg)
            wins.append(lp > ln.max())
            gaps.append(lp - ln.mean())
        assert ph.loc[neg_kind, "rank_acc"] == pytest.approx(np.mean(wins))
        assert ph.loc[neg_kind, "gap_mean"] == pytest.approx(np.mean(gaps), abs=1e-6)
    assert "head pairhead" in (tmp_path / "eval" / "SUMMARY.md").read_text()


def _moments_side(tmp_path, meta, masked_ligand=(), seed=0):
    """A side file over the meta's rows (all kind 0) with random descriptors; ligand masked for masked_ligand."""
    import build_zernike_side as bzs

    m = meta[[c for c in ("row", "unique_id", "split", "kind", "parent") if c in meta]].copy()
    rng = np.random.default_rng(seed)
    n = len(m)
    pocket = rng.normal(2.0, 1.0, (n, 343)).astype(np.float32)
    ligand = rng.normal(-1.0, 3.0, (n, 343)).astype(np.float32)
    ligand[list(masked_ligand)] = np.nan
    m["split"] = np.where(np.arange(n) % 2 == 0, "train", m["split"])  # half the rows set the standardisation
    full = _meta_frame(m)
    kind0 = (full["kind"] == 0).to_numpy()  # the npz holds kind-0 rows only, as zernike.npz does
    uid_of = dict(zip(full["row"], full["unique_id"]))
    twins = pd.Series({u: uid_of.get(p, "") for u, k, p in zip(full["unique_id"], full["kind"], full["parent"]) if k == 4},
                      dtype=object)  # a twin carries its parent's ligand here
    arrays, _ = bzs.build(full, pd.Series(np.flatnonzero(kind0), index=full["unique_id"].to_numpy()[kind0]),
                          pocket, ligand, twins if len(twins) else None)
    path = tmp_path / "paired_zernike.npz"
    bzs.write_npz(str(path), arrays)
    return path


def _meta_frame(m):
    m = m.copy()
    if "kind" not in m:
        m["kind"] = 0
    if "parent" not in m:
        m["parent"] = m["row"]
    m["series_id"] = -1
    m["lig_id"] = -1
    return m


def _manual_r2(models, ds, side, rows, target, ligand_only=False):
    import build_zernike_side as bzs

    idx = bzs.side_index(side, rows)
    sel = idx >= 0
    sel[sel] = side[f"mask_{target}"][idx[sel]]
    preds = []
    for m in models:
        z = eval_paired.embed_dataset(m, ds, torch.device("cpu"), ligand_only=ligand_only, batch_size=4)
        with torch.no_grad():
            preds.append(m.moments_pred(z, target).double().numpy()[sel])
    return eval_paired.moments_r2(np.mean(preds, axis=0), side[f"{target}_std"][idx[sel]]), int(sel.sum())


def test_moments_eval(tmp_path, capsys):
    import build_zernike_side as bzs

    rng = np.random.default_rng(10)
    data_dir, meta = _pairwise_inputs(tmp_path, rng)
    side_path = _moments_side(tmp_path, meta, masked_ligand=(6, 7, 20))
    paths, _ = _arm_p_models(tmp_path / "models", 367, rng, pair=False)
    out = tmp_path / "eval"
    _pairwise_cli(data_dir, paths, tmp_path / "models" / "armp.pickle", out, extra=["--zernike-side", str(side_path)])
    summ = pd.read_csv(out / "moments_summary.csv")
    assert list(summ.columns) == eval_paired.MOMENTS_COLUMNS
    assert set(summ["dataset"]) == {"series_valid", "test", "group_valid"}  # FEP has no row ids
    assert set(summ["model"]) == {"ensemble", "0", "1"} and set(summ["target"]) == {"pocket", "ligand"}
    row = summ.set_index(["dataset", "target", "model"])
    side = bzs.load_side(str(side_path))
    ds = eval_paired.load_pt(str(data_dir / "processed" / "paired_series_valid.pt"))
    models = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))
    rows = pd.read_csv(out / "predictions_series_valid.csv")["row"].to_numpy()
    (r2, r2u, mse), n = _manual_r2(models, ds, side, rows, "ligand")
    got = row.loc[("series_valid", "ligand", "ensemble")]
    assert got["n_rows"] == n == 12 and got["n_models"] == 2  # rows 6, 7 have no ligand descriptor
    assert got["r2"] == pytest.approx(r2, abs=1e-6) and got["r2_uniform"] == pytest.approx(r2u, abs=1e-6)
    assert got["mse"] == pytest.approx(mse, abs=1e-6)
    assert row.loc[("series_valid", "pocket", "ensemble"), "n_rows"] == 14
    (r2_0, _, _), _ = _manual_r2(models[:1], ds, side, rows, "pocket")
    assert row.loc[("series_valid", "pocket", "0"), "r2"] == pytest.approx(r2_0, abs=1e-6)
    assert row.loc[("group_valid", "ligand", "ensemble"), "n_rows"] == 10  # row 20 masked
    assert "## Zernike moments heads" in (out / "SUMMARY.md").read_text()
    # moments_r2 by hand
    y = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 4.0]])
    p = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]])
    r2, r2u, mse = eval_paired.moments_r2(p, y)
    assert r2 == pytest.approx(1 - 3.0 / (8.0 + 6.0)) and r2u == pytest.approx(((1 - 2 / 8) + (1 - 1 / 6)) / 2)
    assert mse == pytest.approx(0.5)
    # plain models with --zernike-side: no moments file, a note
    plain = _tiny_models(tmp_path / "plain", 367, rng)
    _pairwise_cli(data_dir, plain, tmp_path / "plain" / "tiny.pickle", tmp_path / "eval_plain",
                  extra=["--zernike-side", str(side_path)])
    assert not (tmp_path / "eval_plain" / "moments_summary.csv").exists()
    assert "no model has a moments head" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _pairwise_cli(data_dir, paths, tmp_path / "models" / "armp.pickle", tmp_path / "x",
                      extra=["--zernike-side", str(tmp_path / "absent.npz")])


def test_moments_eval_ligand_only(tmp_path):
    """A ligand-only model with both moments heads: the pocket R^2 is reported (from the masked-graph embedding),
    not suppressed; augmented rows that differ only in the AEV block get identical predictions."""
    import build_zernike_side as bzs

    rng = np.random.default_rng(11)
    aug_pt, items = _aug_inputs(tmp_path, rng)
    meta = pd.DataFrame({"row": [r for r, _, _ in items],
                         "unique_id": [f"u{r}" if k == 0 else f"u{p}::x{r}" for r, k, p in items],
                         "split": "aug_valid", "kind": [k for _, k, _ in items], "parent": [p for _, _, p in items]})
    side_path = _moments_side(tmp_path, meta)
    paths, _ = _arm_p_models(tmp_path / "models", 367, rng, pair=False, ligand_only=True)
    out = tmp_path / "eval"
    eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "armp.pickle"), "--aug-pt", aug_pt,
                      "--zernike-side", str(side_path), "--hidden-dim", "8", "--head", "1", "--device", "cpu",
                      "--out-dir", str(out)])
    summ = pd.read_csv(out / "moments_summary.csv").set_index(["dataset", "target", "model"])
    assert set(summ.index.get_level_values("dataset")) == {"valid", "aug_valid_all"}
    # valid: the 4 crystals; aug_valid_all: every row with a descriptor (parent 4 has no crystal: its rows get
    # none; kind-6 rows have no ligand descriptor)
    n_pocket = sum(1 for _, _, p in items if p != 128)
    n_ligand = sum(1 for _, k, p in items if p != 128 and k != 6)
    for target, n_all in (("pocket", n_pocket), ("ligand", n_ligand)):
        assert summ.loc[("valid", target, "ensemble"), "n_rows"] == 4
        got = summ.loc[("aug_valid_all", target, "ensemble")]
        assert got["n_rows"] == n_all and np.isfinite(got["r2"])
    side = bzs.load_side(str(side_path))
    ds = eval_paired.load_pt(aug_pt)
    models = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))
    rows = np.array([r for r, _, _ in items])
    (r2, _, _), _ = _manual_r2(models, ds, side, rows, "pocket", ligand_only=True)
    assert summ.loc[("aug_valid_all", "pocket", "ensemble"), "r2"] == pytest.approx(r2, abs=1e-6)
    kind0 = np.array([k == 0 for _, k, _ in items])
    (r2_valid, _, _), _ = _manual_r2(models, eval_paired.load_pt(aug_pt)[torch.as_tensor(np.flatnonzero(kind0))],
                                     side, rows[kind0], "pocket", ligand_only=True)
    assert summ.loc[("valid", "pocket", "ensemble"), "r2"] == pytest.approx(r2_valid, abs=1e-6)
    (r2_full, _, _), _ = _manual_r2(models, ds, side, rows, "pocket", ligand_only=False)
    assert r2_full != pytest.approx(r2, abs=1e-6)  # the AEV block was zeroed


def test_arm_p_model_loading(tmp_path):
    rng = np.random.default_rng(12)
    cpu = torch.device("cpu")
    paths, plain_paths = _arm_p_models(tmp_path / "with", 367, rng, seeds=(60,))
    nosc, _ = _arm_p_models(tmp_path / "without", 367, rng, seeds=(60,), sidecar=False)
    a = eval_paired.load_models(paths, 367, 4, 8, 1, cpu)[0]
    b = eval_paired.load_models(nosc, 367, 4, 8, 1, cpu)[0]  # rebuilt from the keys alone
    for m in (a, b):
        assert eval_paired.has_pair_head(m) and eval_paired.moments_targets(m) == ("pocket", "ligand")
        assert not eval_paired.has_siamese(m) and m.pair.hidden == 8
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert type(eval_paired.load_models(plain_paths, 367, 4, 8, 1, cpu)[0]).__name__ == "GATv2Net"
    side_json = tmp_path / "with" / "armp_0.json"
    good = json.loads(side_json.read_text())
    for bad in ({"pair_hidden": 16}, {"moments_dim": 100}):
        side_json.write_text(json.dumps({**good, **bad}))
        with pytest.raises(SystemExit):
            eval_paired.load_models(paths, 367, 4, 8, 1, cpu)
    # the sidecar declares heads the weights lack
    only_pair, _ = _arm_p_models(tmp_path / "pair_only", 367, rng, seeds=(61,), moments=())
    (tmp_path / "pair_only" / "armp_0.json").write_text(json.dumps({**good, "moments": "both"}))
    with pytest.raises(SystemExit):
        eval_paired.load_models(only_pair, 367, 4, 8, 1, cpu)
    no_pair, _ = _arm_p_models(tmp_path / "moments_only", 367, rng, seeds=(62,), pair=False)
    (tmp_path / "moments_only" / "armp_0.json").write_text(json.dumps({**good, "pair_head": True}))
    with pytest.raises(SystemExit):
        eval_paired.load_models(no_pair, 367, 4, 8, 1, cpu)
    (tmp_path / "with" / "plain_0.json").write_text(json.dumps({"pair_head": True}))
    with pytest.raises(SystemExit):
        eval_paired.load_models(plain_paths, 367, 4, 8, 1, cpu)


# ----------------------------------------------------------------------------- moments as input (arm Z)

def _arm_z_models(model_dir, width, rng, arch="gatv2", mode="bothx", proj=8, seeds=(70,), side=None, sidecar=True):
    """Tiny arm-Z models (GATv2Net hidden 8, head 1 -> z dim 16, with a MomentsInput; or a MomentsOnlyNet) with
    non-trivial weights / BatchNorm statistics, their sidecars (zernike_side_stats of `side` when given), a scaler."""
    from types import SimpleNamespace

    import heads
    from model_defs import GATv2Net

    os.makedirs(model_dir, exist_ok=True)
    d_in = heads.input_moments_dim(mode)
    paths = []
    for k, s in enumerate(seeds):
        torch.manual_seed(s)
        if arch == "moments_only":
            model = heads.MomentsOnlyNet(d_in, mode=mode)
        else:
            gnn = GATv2Net(node_feature_dim=width, edge_feature_dim=4,
                           config=SimpleNamespace(hidden_dim=8, head=1, activation_function="leaky_relu"))
            model = heads.AffinityWithSiamese(gnn, None, None, None, heads.MomentsInput(d_in, proj=proj, mode=mode))
        with torch.no_grad():
            for name, p in model.named_parameters():
                torch.nn.init.normal_(p, std=0.3)
            for mod in model.modules():
                if isinstance(mod, torch.nn.BatchNorm1d):
                    mod.running_mean.normal_(0, 0.5)
                    mod.running_var.uniform_(0.5, 2.0)
        path = str(model_dir / f"z{arch}_{k}.model")
        torch.save(model.state_dict(), path)
        if sidecar:
            payload = {"args": {"ligand_only": False, "model": arch, "input_moments": mode},
                       "model": path, "model_arch": arch, "input_moments": mode,
                       "input_moments_proj": proj if arch == "gatv2" else None, "input_dim": d_in,
                       "zernike_side_stats": heads.side_stats_digest(side) if side is not None else None}
            (model_dir / f"z{arch}_{k}.json").write_text(json.dumps(payload))
        paths.append(path)
    with open(model_dir / "z.pickle", "wb") as fh:
        pickle.dump(StandardScaler().fit(rng.normal(6, 1.5, (50, 1))), fh)
    return paths


def _zernike_eval_npz(path, uids, rng, failed=(), loc=(5.0, -3.0)):
    """A zernike_fep.npz in featurise_zernike's layout ([n, 7, 49]); rows of `failed` ok False / NaN."""
    n = len(uids)
    pocket = rng.normal(loc[0], 2.0, (n, 7, 49)).astype(np.float32)
    ligand = rng.normal(loc[1], 1.0, (n, 7, 49)).astype(np.float32)
    ok = np.array([u not in failed for u in uids])
    pocket[~ok] = np.nan
    ligand[~ok] = np.nan
    np.savez_compressed(path, unique_id=np.array(uids), pocket_F=pocket, ligand_F=ligand, ok=ok,
                        error=np.array(["" if o else "ValueError: failed" for o in ok]))
    return pocket.reshape(n, -1), ligand.reshape(n, -1), ok


def _fep_toy(tmp_path, rng, width=367, n=12):
    """fep_benchmark.csv, a SHUFFLED processed CSV (row 4 dropped) and the .pt in processed order (y = graph_id)."""
    def graph(**fields):
        k = int(rng.integers(3, 6))
        src = list(range(k - 1)) + list(range(1, k))
        dst = list(range(1, k)) + list(range(k - 1))
        return Data(x=torch.tensor(rng.random((k, width)), dtype=torch.float),
                    edge_index=torch.tensor([src, dst], dtype=torch.long), edge_attr=torch.ones(len(src), 4), **fields)

    data_dir = tmp_path / "fep"
    os.makedirs(data_dir / "processed", exist_ok=True)
    fep = pd.DataFrame({"unique_id": [f"subA/s{k % 2}/l{k}" for k in range(n)], "pK": rng.normal(6, 1, n),
                        "sdf_file": "x.sdf", "pdb_file": "x.pdb", "subset": "subA",
                        "series": [f"s{k % 2}" for k in range(n)], "ref_pdb": "1ABC", "n_entries": 1})
    fep.to_csv(data_dir / "fep_benchmark.csv", index=False)
    proc = fep.drop(index=[4]).sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert proc["unique_id"].tolist() != sorted(proc["unique_id"].tolist(), key=lambda u: int(u.rsplit("l", 1)[1]))
    proc.to_csv(data_dir / "fep_benchmark_processed.csv", index=False)
    _save([graph(y=torch.tensor([g], dtype=torch.int32)) for g in range(len(proc))],
          str(data_dir / "processed" / "fep_benchmark.pt"))
    return data_dir, fep, proc


def _expected_fep_preds(model_path, uids, raw_p, raw_l, side, mode, scaler_path, ds=None):
    """The model's FEP predictions by hand: per unique_id its raw descriptor, standardised with the SIDE file's mean
    / sd, the moments input matrix, then the model (a MomentsOnlyNet, or a wrapper on the graphs of ds)."""
    import heads

    mp, ml = np.isfinite(raw_p).all(1), np.isfinite(raw_l).all(1)
    p_std = np.where(mp[:, None], (raw_p - side["pocket_mean"]) / side["pocket_sd"], 0.0)
    l_std = np.where(ml[:, None], (raw_l - side["ligand_mean"]) / side["ligand_sd"], 0.0)
    mat = heads.moments_input_matrix(torch.tensor(p_std, dtype=torch.float32), torch.tensor(l_std, dtype=torch.float32),
                                     torch.tensor(mp), torch.tensor(ml), mode)
    model = eval_paired.load_models([model_path], 367, 4, 8, 1, torch.device("cpu"))[0].eval()
    with torch.no_grad():
        if ds is None:
            from types import SimpleNamespace
            raw = model(SimpleNamespace(moments=mat)).view(-1).double().numpy()
        else:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([ds[k] for k in range(len(ds))])
            batch.moments = mat
            raw = model(batch).view(-1).double().numpy()
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    return scaler.inverse_transform(raw.reshape(-1, 1)).ravel(), mat


def _side_for_toy(tmp_path, rng):
    """A side file over 20 kind-0 rows (descriptors around 0, sd ~1) for the standardisation."""
    meta = pd.DataFrame({"row": range(20), "unique_id": [f"u{i}" for i in range(20)], "split": "train"})
    return _moments_side(tmp_path, meta)


@pytest.mark.parametrize("arch", ["moments_only", "gatv2"])
def test_fep_moments_follow_processed_csv_order(tmp_path, arch):
    """graph_id -> unique_id follows fep_benchmark_processed.csv's row order (shuffled here, as make_predictions
    numbers it) whatever the order of fep_benchmark.csv and of the zernike-eval file; a failed or absent ligand gets
    zeros and the missing flag."""
    import build_zernike_side as bzs

    rng = np.random.default_rng(20)
    data_dir, fep, proc = _fep_toy(tmp_path, rng)
    side_path = _side_for_toy(tmp_path, rng)
    side = bzs.load_side(str(side_path))
    eval_uids = [u for u in fep["unique_id"] if u != "subA/s1/l9"]
    eval_uids = [eval_uids[k] for k in rng.permutation(len(eval_uids))]
    raw_p, raw_l, ok = _zernike_eval_npz(tmp_path / "zernike_fep.npz", eval_uids, rng, failed=("subA/s0/l2",))
    paths = _arm_z_models(tmp_path / "models", 367, rng, arch=arch, side=side)
    out = tmp_path / "eval"
    eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "z.pickle"),
                      "--fep-csv", str(data_dir / "fep_benchmark.csv"), "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt"),
                      "--zernike-side", str(side_path), "--zernike-eval", str(tmp_path / "zernike_fep.npz"),
                      "--pdbbind-index-tgz", str(tmp_path / "absent.tgz"), "--hidden-dim", "8", "--head", "1",
                      "--device", "cpu", "--out-dir", str(out)])
    preds = pd.read_csv(out / "predictions_fep.csv")
    assert preds["unique_id"].tolist() == proc["unique_id"].tolist()  # graph_id order = processed CSV order
    pos = {u: k for k, u in enumerate(eval_uids)}
    uids = preds["unique_id"].tolist()
    nan = np.full(343, np.nan, dtype=np.float32)
    rp = np.stack([raw_p[pos[u]] if u in pos else nan for u in uids])
    rl = np.stack([raw_l[pos[u]] if u in pos else nan for u in uids])
    ds = eval_paired.load_pt(str(data_dir / "processed" / "fep_benchmark.pt")) if arch == "gatv2" else None
    expected, mat = _expected_fep_preds(paths[0], uids, rp, rl, side, "bothx", tmp_path / "models" / "z.pickle", ds)
    np.testing.assert_allclose(preds["pred"], expected, atol=1e-5)
    flag = dict(zip(uids, mat[:, -1].tolist()))
    assert flag["subA/s0/l2"] == 1 and flag["subA/s1/l9"] == 1 and sum(flag.values()) == 2
    # the source itself: graph_id g <-> processed row g
    src = eval_paired.fep_moments_source(side, eval_paired.read_eval_descriptors([str(tmp_path / "zernike_fep.npz")]),
                                         proc["unique_id"].to_numpy())
    g = int(np.flatnonzero(proc["unique_id"] == "subA/s1/l3")[0])
    m = src.matrix([g], "pocket")
    np.testing.assert_allclose(m[0, :343].numpy(), (raw_p[pos["subA/s1/l3"]] - side["pocket_mean"]) / side["pocket_sd"],
                               rtol=1e-5, atol=1e-5)
    assert "pocket descriptor for 9 of 11 ligands" in (out / "SUMMARY.md").read_text()


def test_eval_moments_use_train_stats(tmp_path):
    """FEP / 15-PGDH descriptors are standardised with the side file's TRAIN mean / sd, never with the eval set's own
    (here the eval descriptors sit far from the training distribution, so the two differ)."""
    import build_zernike_side as bzs

    rng = np.random.default_rng(21)
    side_path = _side_for_toy(tmp_path, rng)
    side = bzs.load_side(str(side_path))
    uids = [f"pgdh/S1/compound_{k}" for k in range(6)]
    raw_p, raw_l, _ = _zernike_eval_npz(tmp_path / "zernike_pgdh.npz", uids, rng, loc=(40.0, -25.0))
    p_std, l_std, mp, ml = eval_paired.standardise_eval(side, raw_p, raw_l)
    np.testing.assert_allclose(p_std, (raw_p - side["pocket_mean"]) / side["pocket_sd"], rtol=1e-5)
    np.testing.assert_allclose(l_std, (raw_l - side["ligand_mean"]) / side["ligand_sd"], rtol=1e-5)
    assert mp.all() and ml.all()
    own = (raw_p - raw_p.mean(axis=0)) / raw_p.std(axis=0)
    assert not np.allclose(p_std, own, atol=1.0) and abs(p_std.mean()) > 10  # far from the training distribution
    src = eval_paired.fep_moments_source(side, eval_paired.read_eval_descriptors([str(tmp_path / "zernike_pgdh.npz")]),
                                         np.array(uids[::-1]))
    m = src.matrix(np.arange(6), "both")
    np.testing.assert_allclose(m[:, :343].numpy(), p_std[::-1], rtol=1e-5)
    np.testing.assert_allclose(m[:, 343:686].numpy(), l_std[::-1], rtol=1e-5)
    # end to end, with a FEP file and a 15-PGDH file passed together (merged by unique_id)
    data_dir = tmp_path / "pg"
    os.makedirs(data_dir / "processed")
    pg = pd.DataFrame({"unique_id": uids, "pK": rng.normal(6, 1, 6), "sdf_file": "x.sdf", "pdb_file": "x.pdb",
                       "subset": "pgdh", "series": "S1", "ref_pdb": "9PFL", "n_entries": 1})
    pg.to_csv(data_dir / "pgdh.csv", index=False)
    pg.to_csv(data_dir / "pgdh_processed.csv", index=False)
    _save([Data(x=torch.rand(3, 367), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 4),
                y=torch.tensor([g], dtype=torch.int32)) for g in range(6)], str(data_dir / "processed" / "pgdh.pt"))
    _zernike_eval_npz(tmp_path / "zernike_fep.npz", ["subA/s0/l0", "subA/s0/l1"], rng)
    paths = _arm_z_models(tmp_path / "models", 367, rng, arch="moments_only", mode="both", side=side)
    out = tmp_path / "eval"
    eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "z.pickle"),
                      "--fep-csv", str(data_dir / "pgdh.csv"), "--fep-pt", str(data_dir / "processed" / "pgdh.pt"),
                      "--zernike-side", str(side_path),
                      "--zernike-eval", f"{tmp_path / 'zernike_fep.npz'},{tmp_path / 'zernike_pgdh.npz'}",
                      "--pdbbind-index-tgz", str(tmp_path / "absent.tgz"), "--device", "cpu", "--out-dir", str(out)])
    preds = pd.read_csv(out / "predictions_fep.csv")
    expected, _ = _expected_fep_preds(paths[0], uids, raw_p, raw_l, side, "both", tmp_path / "models" / "z.pickle")
    np.testing.assert_allclose(preds["pred"], expected, atol=1e-5)
    assert "pocket descriptor for 6 of 6 ligands" in (out / "SUMMARY.md").read_text()
    # the same unique_id in two files is an error
    with pytest.raises(SystemExit):
        eval_paired.main(["--models", *paths, "--scaler", str(tmp_path / "models" / "z.pickle"),
                          "--fep-csv", str(data_dir / "pgdh.csv"), "--fep-pt", str(data_dir / "processed" / "pgdh.pt"),
                          "--zernike-side", str(side_path),
                          "--zernike-eval", f"{tmp_path / 'zernike_pgdh.npz'},{tmp_path / 'zernike_pgdh.npz'}",
                          "--device", "cpu", "--out-dir", str(tmp_path / "dup")])


def test_eval_input_models_need_side_files(tmp_path, capsys):
    import build_zernike_side as bzs

    rng = np.random.default_rng(22)
    data_dir, meta = _pairwise_inputs(tmp_path, rng)
    side_path = _moments_side(tmp_path, meta, masked_ligand=(6,))
    side = bzs.load_side(str(side_path))
    _zernike_eval_npz(tmp_path / "zernike_fep.npz", [f"subA/s1/l{i}" for i in range(12)], rng)
    paths = _arm_z_models(tmp_path / "models", 367, rng, arch="gatv2", mode="both", side=side)
    scaler = tmp_path / "models" / "z.pickle"
    series = ["--series-valid-pt", str(data_dir / "processed" / "paired_series_valid.pt")]
    fep = ["--fep-csv", str(data_dir / "fep_benchmark.csv"), "--fep-pt", str(data_dir / "processed" / "fep_benchmark.pt")]
    base = ["--models", *paths, "--scaler", str(scaler), "--meta", str(data_dir / "paired_meta.csv"), "--hidden-dim", "8",
            "--head", "1", "--device", "cpu", "--pdbbind-index-tgz", str(tmp_path / "absent.tgz")]
    for extra, needle in (([*series], "--zernike-side"),
                          ([*fep, "--zernike-side", str(side_path)], "--zernike-eval"),
                          ([*fep, "--zernike-side", str(side_path), "--zernike-eval", str(tmp_path / "absent.npz")],
                           "missing --zernike-eval")):
        with pytest.raises(SystemExit) as exc:
            eval_paired.main([*base, *extra, "--out-dir", str(tmp_path / "x")])
        assert needle in str(exc.value.code), (extra, exc.value.code)
    # another side file's statistics: refused for a model with moments inputs
    os.makedirs(tmp_path / "o")
    other = _moments_side(tmp_path / "o", meta, seed=5)
    with pytest.raises(SystemExit) as exc:
        eval_paired.main([*base, *series, "--zernike-side", str(other), "--out-dir", str(tmp_path / "x")])
    assert "mean / sd" in str(exc.value.code)
    # the right files: series_valid rows looked up in the side file (row 6 has no ligand descriptor)
    out = tmp_path / "eval"
    eval_paired.main([*base, *series, *fep, "--zernike-side", str(side_path), "--zernike-eval",
                      str(tmp_path / "zernike_fep.npz"), "--out-dir", str(out), "--pairwise"])
    text = capsys.readouterr().out
    assert "14 rows, pocket descriptor 14, ligand descriptor 13" in text
    preds = pd.read_csv(out / "predictions_series_valid.csv")
    ds = eval_paired.load_pt(str(data_dir / "processed" / "paired_series_valid.pt"))
    model = eval_paired.load_models(paths, 367, 4, 8, 1, torch.device("cpu"))[0].eval()
    src = eval_paired.MomentsSource.from_side(side)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([ds[k] for k in range(len(ds))])
    batch.moments = src.matrix(preds["row"].to_numpy(), "both")
    assert batch.moments[preds["row"].tolist().index(6), -1] == 1
    with torch.no_grad():
        raw = model(batch).view(-1).double().numpy()
    with open(scaler, "rb") as fh:
        sc = pickle.load(fh)
    np.testing.assert_allclose(preds["pred"], sc.inverse_transform(raw.reshape(-1, 1)).ravel(), atol=1e-5)
    # models without moments inputs ignore --zernike-eval (and need no side file)
    plain = _tiny_models(tmp_path / "plain", 367, rng)
    eval_paired.main(["--models", *plain, "--scaler", str(tmp_path / "plain" / "tiny.pickle"), *fep,
                      "--zernike-eval", str(tmp_path / "zernike_fep.npz"), "--hidden-dim", "8", "--head", "1",
                      "--device", "cpu", "--pdbbind-index-tgz", str(tmp_path / "absent.tgz"),
                      "--out-dir", str(tmp_path / "plain_eval")])
    assert "no model takes moments inputs; ignored" in capsys.readouterr().out


def test_arm_z_model_loading(tmp_path):
    import heads

    rng = np.random.default_rng(23)
    cpu = torch.device("cpu")
    for arch, proj, kind in (("gatv2", 8, heads.AffinityWithSiamese), ("gatv2", 0, heads.AffinityWithSiamese),
                             ("moments_only", None, heads.MomentsOnlyNet)):
        d = tmp_path / f"{arch}{proj}"
        paths = _arm_z_models(d, 367, rng, arch=arch, mode="bothx", proj=proj)
        model = eval_paired.load_models(paths, 367, 4, 8, 1, cpu)[0]
        assert isinstance(model, kind) and eval_paired.input_mode(model) == "bothx"
        state = torch.load(paths[0], weights_only=True)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
        if arch == "gatv2":
            assert model.moments_in.proj == proj and model.affinity.fc1.in_features == 16 + (proj or 1030)
        side_json = d / f"z{arch}_0.json"
        good = json.loads(side_json.read_text())
        bad_sides = [{"input_moments": "both"}, {"input_dim": 344}]
        if arch == "gatv2":
            bad_sides.append({"input_moments_proj": 4 if proj else 8})
        for bad in bad_sides:
            side_json.write_text(json.dumps({**good, **bad}))
            with pytest.raises(SystemExit):
                eval_paired.load_models(paths, 367, 4, 8, 1, cpu)
        side_json.unlink()  # without the sidecar the moments input cannot be rebuilt: an error, not a guess
        with pytest.raises(SystemExit):
            eval_paired.load_models(paths, 367, 4, 8, 1, cpu)
    assert eval_paired.input_mode(eval_paired.load_models(_tiny_models(tmp_path / "p", 367, rng), 367, 4, 8, 1, cpu)[0]) == "none"
