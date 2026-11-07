import os, subprocess, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write(d, fep_p, sv_s, test_p):
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([{"scope": "n>=10", "weighting": "weighted", "n_series": 43, "n_items": 948, "pearson": fep_p,
                   "spearman": fep_p - 0.05, "kendall": fep_p - 0.2, "rmse": 1.0},
                  {"scope": "all", "weighting": "weighted", "n_series": 91, "n_items": 1184, "pearson": fep_p,
                   "spearman": fep_p, "kendall": fep_p, "rmse": 0.95}]).to_csv(os.path.join(d, "fep_summary.csv"), index=False)
    pd.DataFrame([{"source": "all", "scope": "n>=5", "weighting": "weighted", "n_series": 170, "n_items": 3900,
                   "pearson": sv_s, "spearman": sv_s, "kendall": sv_s - 0.1, "rmse": 1.1},
                  {"source": "all", "scope": "all", "weighting": "weighted", "n_series": 176, "n_items": 3917,
                   "pearson": sv_s, "spearman": sv_s, "kendall": sv_s, "rmse": 1.1}]).to_csv(os.path.join(d, "series_valid_summary.csv"), index=False)
    pd.DataFrame([{"n": 277, "pearson": test_p, "spearman": test_p, "kendall": 0.7, "rmse": 1.04}]).to_csv(os.path.join(d, "test_summary.csv"), index=False)


def test_compare_arms_report(tmp_path):
    root = tmp_path / "eval"
    vals = {"base": (0.55, 0.40, 0.88), "dh03": (0.60, 0.46, 0.88), "base-lig": (0.50, 0.30, 0.80), "dh03-lig": (0.51, 0.31, 0.80)}
    for tag, (f, s, t) in vals.items():
        _write(str(root / tag), f, s, t)
        for i in range(3):
            _write(str(root / tag / f"seed{i}"), f - 0.02 + 0.01 * i, s - 0.02 + 0.01 * i, t)
    out = tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base,dh03,base-lig,dh03-lig", "--ref", "base", "--lig-pairs", "base-lig:base,dh03-lig:dh03",
                        "--out", str(out)], capture_output=True, text=True, env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "| FEP wPCC (n>=10) | 0.550 | 0.600 | 0.500 | 0.510 |" in text
    # paired per-seed difference dh03 - base on FEP wPCC is exactly 0.05 at every seed
    assert "0.0500 [0.050, 0.050, 0.050]" in text
    # structure-specific gain: 0.05 - 0.01 = 0.04 with zero spread -> passes
    assert "passes" in text
    assert (tmp_path / "REPORT_values.csv").exists()


def _write_pose(d, acc):
    pd.DataFrame([{"n_parents": 993, "decoy_rank_acc": acc, "decoy_gap_mean": 2 * acc, "nn_spread": 0.1 + acc,
                   "pearson_crystal": 0.7, "pearson_pos": 0.6, "pearson_neg": 0.3}]).to_csv(
        os.path.join(d, "pose_summary.csv"), index=False)


def _run(root, out, tags):
    return subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                           "--tags", tags, "--ref", "base", "--out", str(out)], capture_output=True, text=True,
                          env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})


def test_compare_arms_includes_pose_rows(tmp_path):
    root = tmp_path / "eval"
    accs = {"base": 0.40, "pda": 0.50}
    for tag, acc in accs.items():
        _write(str(root / tag), 0.55, 0.40, 0.88)
        _write_pose(str(root / tag), acc)
        for i in range(3):
            _write(str(root / tag / f"seed{i}"), 0.55, 0.40, 0.88)
            _write_pose(str(root / tag / f"seed{i}"), acc + 0.01 * i)
    out = tmp_path / "REPORT.md"
    r = _run(root, out, "base,pda")
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "| Pose decoy-rank acc | 0.400 | 0.500 |" in text
    assert "| Pose decoy gap (pK) | 0.800 | 1.000 |" in text
    assert "| Pose NN spread (pK) | 0.500 | 0.600 |" in text
    assert "| Pose crystal Pearson | 0.700 | 0.700 |" in text
    assert "0.1000 [0.100, 0.100, 0.100]" in text  # per-seed paired difference pda - base
    values = pd.read_csv(tmp_path / "REPORT_values.csv")
    assert "decoy_rank_acc" in set(values["column"])

    # one tag without any pose_summary.csv: the table stays as it was (no pose rows)
    for d in [root / "base"] + [root / "base" / f"seed{i}" for i in range(3)]:
        os.remove(d / "pose_summary.csv")
    out2 = tmp_path / "REPORT2.md"
    r = _run(root, out2, "base,pda")
    assert r.returncode == 0, r.stderr
    text2 = out2.read_text()
    assert "Pose" not in text2
    assert "| FEP wPCC (n>=10) | 0.550 | 0.550 |" in text2

    # present in seed dirs only (not the ensemble dir) still counts
    for i in range(3):
        _write_pose(str(root / "base" / f"seed{i}"), 0.4)
    out3 = tmp_path / "REPORT3.md"
    r = _run(root, out3, "base,pda")
    assert r.returncode == 0, r.stderr
    assert "| Pose decoy-rank acc | nan | 0.500 |" in out3.read_text()


def test_pose_metric_direction():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import compare_arms

    assert compare_arms.HIGHER_IS_BETTER["decoy_rank_acc"] is True
    assert compare_arms.HIGHER_IS_BETTER["decoy_gap_mean"] is True
    assert compare_arms.HIGHER_IS_BETTER["pearson_crystal"] is True
    assert compare_arms.HIGHER_IS_BETTER["nn_spread"] is False


def _write_group(d, p):
    pd.DataFrame([{"scope": "pointwise", "weighting": "none", "n_series": 120, "n_items": 900, "pearson": p,
                   "spearman": p - 0.02, "kendall": p - 0.2, "rmse": 1.3},
                  {"scope": "n>=5", "weighting": "weighted", "n_series": 40, "n_items": 700, "pearson": p,
                   "spearman": p - 0.1, "kendall": p - 0.3, "rmse": 1.2}]).to_csv(
        os.path.join(d, "group_valid_summary.csv"), index=False)


def _write_aug(d, acc, kinds=(0, 3, 4, 5, 6, 7)):
    role = {0: "crystal", 1: "control", 2: "negative", 3: "control", 4: "negative", 5: "control",
            6: "negative", 7: "control"}
    ref = {1: 0, 2: 0, 3: 0, 4: 3, 5: 0, 6: 5, 7: 0}
    rows = []
    for k in kinds:
        neg = role[k] == "negative"
        rows.append({"kind": k, "role": role[k], "ref_kind": ref.get(k), "n_parents": 900, "n_rows": 1800,
                     "rank_acc": acc + 0.01 * k if neg else np.nan, "gap_mean": 2 * acc if neg else np.nan,
                     "spread": 0.1 * k if role[k] == "control" else np.nan, "pearson": 0.6})
    pd.DataFrame(rows).to_csv(os.path.join(d, "aug_summary.csv"), index=False)


def test_compare_arms_includes_group_and_aug_rows(tmp_path):
    root = tmp_path / "eval"
    spec = {"base-g": (0.50, 0.30), "A": (0.52, 0.40), "C": (0.51, 0.45)}
    for tag, (p, acc) in spec.items():
        kinds = (0, 3, 4) if tag == "A" else (0, 3, 4, 5, 6, 7)  # A has no C rows: C kinds omitted
        for d in [root / tag] + [root / tag / f"seed{i}" for i in range(3)]:
            _write(str(d), 0.55, 0.40, 0.88)
            _write_group(str(d), p)
            _write_aug(str(d), acc, kinds)
    def run(out):
        return subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root",
                               str(root), "--tags", "base-g,A,C", "--ref", "base-g", "--out", str(out)],
                              capture_output=True, text=True,
                              env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})

    out = tmp_path / "REPORT.md"
    r = run(out)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "| group_valid Pearson | 0.500 | 0.520 | 0.510 |" in text
    assert "| group_valid Spearman | 0.480 | 0.500 | 0.490 |" in text
    assert "| group_valid within-series wSpearman (n>=5) | 0.400 | 0.420 | 0.410 |" in text
    assert "| Aug kind 4 (A-twin) rank acc | 0.340 | 0.440 | 0.490 |" in text
    assert "| Aug kind 4 (A-twin) gap (pK) | 0.600 | 0.800 | 0.900 |" in text
    assert "| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.300 | 0.300 | 0.300 |" in text
    assert "Aug kind 6" not in text and "Aug kind 7" not in text and "Aug kind 0" not in text
    assert "0.0200 [0.020, 0.020, 0.020]" in text  # A - base-g on group_valid Pearson, every seed
    values = pd.read_csv(tmp_path / "REPORT_values.csv")
    assert {"rank_acc", "gap_mean", "spread"} <= set(values["column"])

    # one tag without group_valid_summary.csv: group rows omitted, with a note
    for d in [root / "C"] + [root / "C" / f"seed{i}" for i in range(3)]:
        os.remove(d / "group_valid_summary.csv")
    out2 = tmp_path / "REPORT2.md"
    r = run(out2)
    assert r.returncode == 0, r.stderr
    text2 = out2.read_text()
    assert "| group_valid Pearson" not in text2 and "group_valid_summary.csv missing for C" in text2


def test_aug_metric_direction():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import compare_arms

    assert compare_arms.HIGHER_IS_BETTER["rank_acc"] is True
    assert compare_arms.HIGHER_IS_BETTER["gap_mean"] is True
    assert compare_arms.HIGHER_IS_BETTER["spread"] is False


def test_aug_control_label_names_its_reference(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import compare_arms

    root = tmp_path / "eval"
    for tag in ("base-g", "C"):
        os.makedirs(root / tag)
        _write_aug(str(root / tag), 0.4)
    labels = [m[3] for m in compare_arms.aug_metrics_for(str(root), ["base-g", "C"], [0, 1, 2])]
    assert "Aug kind 7 (C-exposed) spread vs C-pos (pK)" in labels
    assert "Aug kind 5 (C-pos) spread vs crystal (pK)" in labels


def _write_pairwise(d, kendall, with_pair, datasets=("fep", "series_valid", "group_valid", "test")):
    """pairwise_summary.csv: point rows (Kendall `kendall`) and, with_pair, pair rows (Kendall + 0.1)."""
    scope = {"fep": "n>=10", "series_valid": "n>=5", "group_valid": "n>=5", "test": "n>=5"}
    rows = []
    for ds in datasets:
        for head in (("pair", "point") if with_pair else ("point",)):
            k = kendall + (0.1 if head == "pair" else 0.0)
            for sc in (scope[ds], "all"):
                for w in ("weighted", "unweighted"):
                    rows.append({"dataset": ds, "head": head, "scope": sc, "weighting": w, "n_series": 40,
                                 "n_items": 900, "pair_rmse": 1.2 - k, "pair_dir_acc": 0.5 + k / 2,
                                 "kendall": k, "spearman": k + 0.05})
    pd.DataFrame(rows).to_csv(os.path.join(d, "pairwise_summary.csv"), index=False)


def test_compare_arms_pairwise_rows(tmp_path):
    root = tmp_path / "eval"
    spec = {"base-g": (0.30, False), "S": (0.32, True)}
    for tag, (k, with_pair) in spec.items():
        for i, d in enumerate([root / tag] + [root / tag / f"seed{s}" for s in range(3)]):
            _write(str(d), 0.55, 0.40, 0.88)
            _write_pairwise(str(d), k + (0.01 * (i - 1) if i else 0.0), with_pair)
    out = tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,S", "--ref", "base-g", "--out", str(out)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    # pair-head rows NaN for the tag without a head; point rows line up for both
    assert "| FEP pair-head wKendall (n>=10) | nan | 0.420 |" in text
    assert "| FEP pair-head wSpearman (n>=10) | nan | 0.470 |" in text
    assert "| FEP pair RMSE (n>=10) | nan | 0.780 |" in text
    assert "| FEP pair dir acc (n>=10) | nan | 0.710 |" in text
    assert "| FEP point-head wKendall (n>=10) | 0.300 | 0.320 |" in text
    assert "| series_valid pair-head wKendall (n>=5) | nan | 0.420 |" in text
    assert "| group_valid point pair RMSE (n>=5) | 0.900 | 0.880 |" in text
    assert "test" not in {line.split(" ")[1] for line in text.splitlines() if line.startswith("| ")}
    # per-seed paired difference S - base-g on the point head: 0.02 at every seed
    assert "| FEP point-head wKendall (n>=10) |" in text and "0.0200 [0.020, 0.020, 0.020]" in text
    # pair head vs point heads: S pair - S point = 0.1, S pair - base-g point = 0.12, every seed
    assert "## Pair head vs pointwise head" in text
    line = next(l for l in text.splitlines() if l.startswith("| FEP pair-head wKendall (n>=10) | S |"))
    assert "| base-g | 0.420 | 0.320 | 0.300 |" in line
    assert "0.1000 [0.100, 0.100, 0.100]" in line and "0.1200 [0.120, 0.120, 0.120]" in line
    pvp_rows = [l.split(" | ") for l in text.split("## Pair head vs pointwise head")[1].split("##")[0].splitlines()
                if l.startswith("| FEP")]
    assert pvp_rows and {r[1] for r in pvp_rows} == {"S"}  # base-g has no pair head: no row of its own
    values = pd.read_csv(tmp_path / "REPORT_values.csv")
    assert {"pair_rmse", "pair_dir_acc"} <= set(values["column"])

    # datasets: only those every tag has; a tag without the file drops the pairwise rows with a note
    for d in [root / "S"] + [root / "S" / f"seed{s}" for s in range(3)]:
        _write_pairwise(str(d), 0.32, True, datasets=("fep",))
    out2 = tmp_path / "REPORT2.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,S", "--ref", "base-g", "--out", str(out2)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text2 = out2.read_text()
    assert "| FEP pair-head wKendall (n>=10) | nan | 0.420 |" in text2 and "series_valid pair" not in text2
    for d in [root / "base-g"] + [root / "base-g" / f"seed{s}" for s in range(3)]:
        os.remove(d / "pairwise_summary.csv")
    out3 = tmp_path / "REPORT3.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,S", "--ref", "base-g", "--out", str(out3)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text3 = out3.read_text()
    assert "pair-head" not in text3 and "pairwise_summary.csv missing for base-g" in text3
    assert "## Pair head vs pointwise head" not in text3


def test_pairwise_metric_direction():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import compare_arms

    assert compare_arms.HIGHER_IS_BETTER["pair_rmse"] is False
    assert compare_arms.HIGHER_IS_BETTER["pair_dir_acc"] is True


def test_compare_arms_pairwise_gate(tmp_path):
    """Shortcut gate on the pair-head rows: arm pair head vs reference point head, against the ligand-only pair."""
    root = tmp_path / "eval"
    spec = {"base-g": (0.30, False), "S": (0.32, True), "base-g-lig": (0.20, False), "S-lig": (0.21, True)}
    for tag, (k, with_pair) in spec.items():
        for d in [root / tag] + [root / tag / f"seed{s}" for s in range(3)]:
            _write(str(d), 0.55, 0.40, 0.88)
            _write_pairwise(str(d), k, with_pair, datasets=("fep",))
    out = tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,S,base-g-lig,S-lig", "--ref", "base-g",
                        "--lig-pairs", "base-g-lig:base-g,S-lig:S", "--out", str(out)], capture_output=True,
                       text=True, env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    gate = out.read_text().split("## Shortcut gate: S vs base-g, against S-lig vs base-g-lig")[1]
    # S pair (0.42) - base-g point (0.30) = 0.12; S-lig pair (0.31) - base-g-lig point (0.20) = 0.11
    line = next(l for l in gate.splitlines() if l.startswith("| FEP pair-head wKendall (n>=10) (ref: point head) |"))
    assert "| 0.1200 | 0.1100 | 0.0100 (sd 0.0000) |" in line
    # point rows keep the plain comparison: S point (0.32) - base-g point (0.30)
    line = next(l for l in gate.splitlines() if l.startswith("| FEP point-head wKendall (n>=10) |"))
    assert "| 0.0200 | 0.0100 |" in line
    pvp = out.read_text().split("## Pair head vs pointwise head")[1].split("##")[0]
    assert "| FEP pair-head wKendall (n>=10) | S-lig | base-g-lig | 0.310 | 0.210 | 0.200 |" in pvp


# ----------------------------------------------------------------------------- arm P: pair head, moments heads

def _write_pairwise_ph(d, kendall, with_pairhead, datasets=("fep", "series_valid", "group_valid")):
    """pairwise_summary.csv with point rows (Kendall `kendall`) and, with_pairhead, pairhead rows (Kendall + 0.05,
    no pair RMSE)."""
    scope = {"fep": "n>=10", "series_valid": "n>=5", "group_valid": "n>=5"}
    rows = []
    for ds in datasets:
        for head in (("point", "pairhead") if with_pairhead else ("point",)):
            k = kendall + (0.05 if head == "pairhead" else 0.0)
            for sc in (scope[ds], "all"):
                for w in ("weighted", "unweighted"):
                    rows.append({"dataset": ds, "head": head, "scope": sc, "weighting": w, "n_series": 40,
                                 "n_items": 900, "pair_rmse": np.nan if head == "pairhead" else 1.2 - k,
                                 "pair_dir_acc": 0.5 + k / 2, "kendall": k, "spearman": k + 0.05})
    pd.DataFrame(rows).to_csv(os.path.join(d, "pairwise_summary.csv"), index=False)


def _write_aug_ph(d, acc, ph_acc):
    """aug_summary.csv with a head column: the point rows of _write_aug and pairhead rows of kinds 4 and 6."""
    _write_aug(d, acc)
    df = pd.read_csv(os.path.join(d, "aug_summary.csv")).assign(head="point")
    ph = [{"kind": k, "role": "negative", "ref_kind": r, "n_parents": 900, "n_rows": 1800, "rank_acc": ph_acc + 0.01 * k,
           "gap_mean": 3 * ph_acc, "spread": np.nan, "pearson": np.nan, "head": "pairhead"} for k, r in ((4, 3), (6, 5))]
    pd.concat([df, pd.DataFrame(ph)], ignore_index=True).to_csv(os.path.join(d, "aug_summary.csv"), index=False)


def _write_moments(d, r2):
    rows = [{"dataset": ds, "target": t, "model": m, "n_models": 1, "n_rows": 100,
             "r2": r2 + (0.1 if t == "ligand" else 0.0) + (0.01 if m == "0" else 0.0), "r2_uniform": r2, "mse": 1 - r2}
            for ds in ("valid", "series_valid", "aug_valid_all") for t in ("pocket", "ligand") for m in ("ensemble", "0")]
    pd.DataFrame(rows).to_csv(os.path.join(d, "moments_summary.csv"), index=False)


def test_compare_arms_pairhead_and_moments_rows(tmp_path):
    root = tmp_path / "eval"
    for tag in ("base-g", "P", "M"):
        for i, d in enumerate([root / tag] + [root / tag / f"seed{s}" for s in range(3)]):
            os.makedirs(d, exist_ok=True)
            _write(str(d), 0.55, 0.40, 0.88)
            shift = 0.01 * (i - 1) if i else 0.0
            _write_pairwise_ph(str(d), 0.30 + shift, with_pairhead=(tag == "P"))
            if tag == "P":
                _write_aug_ph(str(d), 0.40, 0.60 + shift)
            else:
                _write_aug(str(d), 0.40)
            if tag == "M":
                _write_moments(str(d), 0.50 + shift)
    out = tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,P,M", "--ref", "base-g", "--out", str(out)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "| FEP pairhead wKendall (n>=10) | nan | 0.350 | nan |" in text
    assert "| FEP pairhead wSpearman (n>=10) | nan | 0.400 | nan |" in text
    assert "| FEP pairhead dir acc (n>=10) | nan | 0.675 | nan |" in text
    assert "| series_valid pairhead wKendall (n>=5) | nan | 0.350 | nan |" in text
    assert "| FEP point-head wKendall (n>=10) | 0.300 | 0.300 | 0.300 |" in text  # unchanged
    assert "pairhead pair RMSE" not in text
    # aug: the pointwise rows read P's file at head point; the pair head's rows per negative kind
    assert "| Aug kind 4 (A-twin) rank acc | 0.440 | 0.440 | 0.440 |" in text
    assert "| Aug kind 4 (A-twin) pairhead rank acc | nan | 0.640 | nan |" in text
    assert "| Aug kind 6 (C-contact) pairhead gap (logit) | nan | 1.800 | nan |" in text
    assert "Aug kind 3 (A-pos) pairhead" not in text
    # moments: ensemble r2 per dataset and target, NaN for tags without the heads
    assert "| Moments R2 pocket (valid) | nan | nan | 0.500 |" in text
    assert "| Moments R2 ligand (series_valid) | nan | nan | 0.600 |" in text
    assert "| Moments R2 pocket (aug_valid_all) |" in text and "Moments R2 pocket (test)" not in text
    # pair head vs point heads: P's pair head against its own and base-g's point head (0.05 at every seed)
    pvp = text.split("## Pair head vs pointwise head")[1].split("##")[0]
    line = next(l for l in pvp.splitlines() if l.startswith("| FEP pairhead wKendall (n>=10) | P |"))
    assert "| base-g | 0.350 | 0.300 | 0.300 |" in line and "0.0500 [0.050, 0.050, 0.050]" in line
    line = next(l for l in pvp.splitlines() if l.startswith("| Aug kind 4 (A-twin) pairhead rank acc | P |"))
    assert "| base-g | 0.640 | 0.440 | 0.440 |" in line
    assert "pairhead gap" not in pvp  # logit gaps are values only: not linked to the pointwise gap
    seeds_table = text.split("## Single seeds")[1].split("##")[0]
    gap = next(l for l in seeds_table.splitlines() if l.startswith("| Aug kind 4 (A-twin) pairhead gap (logit) |"))
    assert gap.rstrip(" |").endswith("| - | -")  # no paired differences
    rank = next(l for l in seeds_table.splitlines() if l.startswith("| Aug kind 4 (A-twin) pairhead rank acc |"))
    assert "[nan, nan, nan]" in rank
    values = pd.read_csv(tmp_path / "REPORT_values.csv")
    assert "r2" in set(values["column"])

    # shortcut gate: the pair head's rank acc against the reference's pointwise rank acc; no gap row
    for tag in ("base-g-lig", "P-lig"):
        for i, d in enumerate([root / tag] + [root / tag / f"seed{s}" for s in range(3)]):
            os.makedirs(d, exist_ok=True)
            _write(str(d), 0.55, 0.40, 0.88)
            _write_pairwise_ph(str(d), 0.20, with_pairhead=(tag == "P-lig"))
            if tag == "P-lig":
                _write_aug_ph(str(d), 0.40, 0.45)
            else:
                _write_aug(str(d), 0.40)
    out3 = tmp_path / "REPORT3.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g,P,base-g-lig,P-lig", "--ref", "base-g", "--lig-pairs",
                        "base-g-lig:base-g,P-lig:P", "--out", str(out3)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    gate = out3.read_text().split("## Shortcut gate: P vs base-g, against P-lig vs base-g-lig")[1]
    line = next(l for l in gate.splitlines() if l.startswith("| Aug kind 4 (A-twin) pairhead rank acc (ref: point head) |"))
    # P pair head 0.64 / 0.65 / 0.66 - base-g point 0.44 = 0.21 on average; P-lig 0.49 - base-g-lig point 0.44 = 0.05
    assert "| 0.2100 | 0.0500 |" in line
    assert "pairhead gap" not in gate

    # without arm-P outputs nothing of it appears
    out2 = tmp_path / "REPORT2.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "compare_arms.py"), "--eval-root", str(root),
                        "--tags", "base-g", "--ref", "base-g", "--out", str(out2)], capture_output=True, text=True,
                       env={**os.environ, "AEVPLIG_REPO": os.path.join(ROOT, "tests", "fake_repo")})
    assert r.returncode == 0, r.stderr
    text2 = out2.read_text()
    assert "pairhead" not in text2 and "Moments" not in text2 and "| Aug kind 4 (A-twin) rank acc | 0.440 |" in text2


def test_moments_metric_direction():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import compare_arms

    assert compare_arms.HIGHER_IS_BETTER["r2"] is True
    # a selector key the file lacks: only its default value matches (aug_summary.csv before arm P)
    assert compare_arms.SELECTOR_DEFAULTS == {"head": "point"}
