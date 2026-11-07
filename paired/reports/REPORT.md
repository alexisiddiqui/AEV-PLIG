# AEV-PLIG arm 1 (within-series pairwise loss): result

Date 2026-11-05. Tree `/scratch/u6xq/hussain.u6xq/aevplig-paired` (local mirror `~/isambard-ssh/aevplig-paired`).
Plan: `docs/plans/2026-11-05-aevplig-paired-arm1.md`. Comparison tables: `work/output/eval/REPORT_arms.md` (from `src/compare_arms.py`).

## Verdict

**Negative.** A within-series Δ-Huber term on top of AEV-PLIG's pointwise MSE gives no gain on the FEP benchmark at either weight, and its small gain on held-out series is matched by the same loss applied to a ligand-only model, so it is not evidence of better use of the protein. By the simplicity rule the vanilla recipe stays.

## Setup (what was compared)

- Train = the paper's training set (PDBbind v2020 + BindingNet + BindingDB-DCS, ligsim90) minus 176 held-out series (142 BindingNet templates, 34 BindingDB folders; 3,917 rows), 91,763 rows, 2.48M same-receptor different-ligand pairs.
- Batches: 16 series × 8 ligands (series-aware sampler), identical across arms; only the loss differs. Checkpoint rule, optimiser, seeds (100, 123, 15) as the paper.
- Arms (3 seeds each, 200 epochs, ~65 min per seed on a GH200): `base` (MSE), `dh03` (MSE + 0.3 × Δ-Huber on same-series pairs, ties |ΔpK| < 0.5 ignored), `dh10` (weight 1.0), `base-lig` and `dh03-lig` (same, with the AEV columns zeroed: ligand-only shortcut gate).
- Evaluation: Ross et al. 2023 FEP benchmark (92 series; 1,184 ligands; weighted means over the 43 series with ≥ 10 ligands, as the paper), the held-out series (`series_valid`, Spearman per series with ≥ 5 rows, weighted by n), and CASF-2016. The FEP featurisation reproduces the paper's own predictions exactly (Pearson 1.0000, max |Δ| 0.0000 pK over 1,184 ligands).

## Reference points (same FEP pipeline)

| model | members | FEP wPCC | FEP wKendall | FEP wSpearman | CASF PCC |
|---|---|---|---|---|---|
| released ensemble (paper) | 10 | 0.590 | 0.416 | 0.551 | 0.893 |
| parent retrain, random shuffle, full train | 10 | 0.590 | 0.428 | 0.560 | 0.892 |
| parent retrain, first 3 members | 3 | 0.549 | 0.381 | 0.510 | 0.884 |
| parent retrain, single members 0/1/2 | 1 | 0.461 / 0.464 / 0.510 | | | |
| `base` (this tree: series sampler, minus holdout) | 3 | 0.575 | 0.415 | 0.558 | 0.888 |
| `base` single seeds (mean, sd) | 1 | 0.491 (0.029) | 0.340 (0.013) | 0.463 (0.019) | 0.873 (0.005) |

The paper's numbers are reproduced; the gap between a 3-member and a 10-member ensemble is about +0.04 wPCC. The series sampler and the 3.9k-row holdout cost nothing (0.575 vs 0.549 at 3 members).

## Arm results

3-seed ensembles:

| metric | base | dh03 | dh10 | base-lig | dh03-lig |
|---|---|---|---|---|---|
| FEP wPCC (n ≥ 10) | 0.575 | 0.572 | 0.537 | 0.443 | 0.462 |
| FEP wKendall | 0.415 | 0.394 | 0.379 | 0.315 | 0.324 |
| FEP wRMSE (all) | 0.975 | 0.948 | 0.979 | 1.000 | 0.956 |
| series_valid wSpearman (n ≥ 5) | 0.370 | 0.381 | 0.388 | 0.383 | 0.394 |
| series_valid wRMSE | 0.864 | 0.863 | 0.852 | 0.860 | 0.866 |
| CASF Pearson | 0.888 | 0.881 | 0.881 | 0.828 | 0.814 |

Per-seed paired differences to `base` (mean over seeds; the three seeds in brackets):

| metric | dh03 − base | dh10 − base | dh03-lig − base-lig |
|---|---|---|---|
| FEP wPCC | −0.001 [+0.045, −0.028, −0.020] | −0.026 [−0.008, −0.031, −0.041] | +0.013 |
| FEP wKendall | −0.001 [+0.023, −0.032, +0.005] | −0.011 | +0.010 |
| series_valid wSpearman | +0.011 [+0.013, +0.038, −0.019] | +0.015 [0.000, +0.069, −0.024] | +0.006 |
| CASF Pearson | −0.005 | −0.008 | −0.019 |

Shortcut gate (structure-specific gain = (dh03 − base) − (dh03-lig − base-lig)): negative on every FEP metric (wPCC −0.014, Kendall −0.011, Spearman −0.023), and within seed noise on series_valid (+0.005, sd 0.026) and CASF. Weight 1.0 is worse than 0.3 on FEP, so there is no dose response to chase.

## What else the run showed

1. **Held-out series are ligand-dominated.** A ligand-only model (AEVs zeroed) ranks held-out series as well as the full model (series_valid wSpearman 0.383 vs 0.370) while being far worse on FEP (0.443 vs 0.575) and CASF (0.828 vs 0.888). The template-level holdout keeps the target in training 91% of the time, so within-series ranking there is mostly a ligand QSAR task; it is not a useful selection metric for structure-aware changes. A target-level holdout would be the next hygiene step if series ranking is to be used for selection.
2. **The protein matters on FEP.** Zeroing the AEVs costs 0.13 wPCC on the FEP benchmark and 0.06 Pearson on CASF, so AEV-PLIG is reading the structure there; the pair loss just does not improve how.
3. **Seed noise is large at a single seed.** FEP wPCC sd 0.03 per seed; the ensemble of 3 moves it by +0.08. Any future arm needs ≥ 3 seeds and the ensemble-level comparison, as here.

## Interpretation and recommendation

The literature gains for pairwise training came from pairwise *architectures* (PBCNet, DeltaDelta) or from assay-grouped data with large cross-assay offsets (Boltz-2, LigUnity). AEV-PLIG with its pointwise MSE already fits the within-series differences from the same batches, and the BindingNet series labels are not assay-grouped, so a Δ term adds nothing the MSE did not already use. Recommendation: close arm 1; do not adopt the pair loss. Of the remaining arms from the write-up, the ones that could still move FEP are those that change what the model sees (pose decoys with near-native positives, PDBbind-only) rather than how the same examples are weighted; the listwise variant and |Δ| weighting are one-flag runs but the dose response here argues against them.

## Files

- Models and per-seed predictions: `work/output/trained_models/<tag>_<i>.model`, `work/output/predictions/<tag>_<split>_seed<i>.csv`.
- Evaluations: `work/output/eval/<tag>/` (ensemble) and `<tag>/seed<i>/`; `released-check/`, `retrain10/`, `retrain3/`.
- Jobs: training 8075953 (dh03), 8075954 (dh10), 8075956 (dh03-lig), 8075982 (base), 8075983 (base-lig); evals 8075958, 8075959, 8075961, 8075997, 8075998, 8077925; dataset build 8075417; FEP prep 8075751.
