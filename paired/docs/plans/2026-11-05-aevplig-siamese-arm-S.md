# AEV-PLIG arm S: Siamese difference head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Siamese pairwise head on AEV-PLIG's shared encoder that predicts ΔpK for two ligands of the same series (PBCNet/DeltaDelta style), trained alongside the pointwise MSE anchor, and evaluate it as a *pairwise ranker* (within-series Kendall/Spearman and pairwise RMSE) on the FEP benchmark, the held-out series and the group-disjoint holdout, against the pointwise head and `base-g`.

**Architecture:** `src/heads.py` holds `SiameseHead`; `train_paired.py` wraps the GATv2 encoder so the pooled embedding `z` (dim = `hidden_dim*head*2` = 1536 from the cat of max and mean pooling, taken BEFORE the affinity MLP) is available to both heads; the model file stores the encoder + affinity MLP + Siamese head state dict with a sidecar flag; `eval_paired.py` gains a pairwise section. Pairs come from the existing series sampler (`pair_index`: same series, different ligand, MSE-kind rows only). Everything else (data, sampler, seeds, checkpoint rule on pointwise valid Pearson) unchanged, so `base-g` (data_ac, kinds 0) remains the reference.

**Tech Stack:** as before. **Spec:** user instruction "do siamese first"; literature: PBCNet (MSE on ΔpIC50 + BCE on direction, pointwise anchor; pointwise 0.559/0.372 → pairwise 0.636/0.513 FEP1/FEP2), McNutt & Koes (ΔΔG head collapses without the ΔG anchor), DeltaDelta (antisymmetric linear layer on the latent difference).

## Global Constraints (in addition to the earlier plans)

- **Head:** `SiameseHead(dim=1536, hidden=512)`: `m(z_i, z_j) = MLP([z_i - z_j, z_i * z_j]) -> scalar` (two hidden layers 512, leaky_relu, BatchNorm1d as the clone's MLP) and `delta_hat(z_i, z_j) = (m(z_i, z_j) - m(z_j, z_i)) / 2` so antisymmetry is exact. Standardised units (same scaler as the pointwise head).
- **Loss:** `L = mse_pointwise + w_s * (huber(delta_hat_ij - (y_i - y_j), delta=1.0) + bce(sign_logit_ij, 1[y_i > y_j]))` over same-series different-ligand pairs in the batch (both orders not needed; i<j from `pair_index`), where `sign_logit_ij = delta_hat_ij / tau` with `tau = 0.5 pK / scale`; BCE only on pairs with `|y_i - y_j| >= 0.5 pK`; `--siamese-weight w_s` (default 1.0); both terms averaged over their pairs; zero when no pairs. The pointwise head and loss are unchanged (anchor). Flags `--siamese` (on/off) and `--siamese-weight`, `--siamese-hidden 512`, `--siamese-tie 0.5`.
- **Pairs:** MSE-kind rows only (kind in {0,1,3,5,7}); with `--train-kinds 0` that is all rows. No hinge kinds in arm S.
- **Model file:** state dict of the wrapper (`encoder` = GATv2 layers + BNs, `affinity` = fc1..out, `siamese` = head); sidecar json records `siamese: true`, `siamese_hidden`, `siamese_weight`. The evaluator reconstructs the wrapper from the sidecar; models without the flag load as before (plain GATv2Net), so all earlier models still evaluate. Implement the wrapper so that a plain `GATv2Net` state dict maps onto `encoder`+`affinity` keys deterministically (a key-renaming function with a test both ways).
- **Pairwise evaluation** (`eval_paired.py --pairwise`), for each dataset with series ids (FEP: series = (subset, series); series_valid and group_valid: `series_key` from meta; CASF test: UniProt groups): compute `z` for every row, then for each series all pairs (i<j, different ligand key where available), `delta_hat_ij`; metrics per series: `pair_rmse` (pK, PBCNet metric: RMSE of delta_hat vs y_i - y_j), `pair_dir_acc` (sign agreement on pairs with |Δy| ≥ 0.5), and a per-ligand score `s_i = mean_j delta_hat_ij` giving `kendall_pair`, `spearman_pair`; the same per-series metrics for the pointwise head (`kendall_point`, `spearman_point`, and pointwise differences f_i - f_j for `pair_rmse_point`). Weighted means over series with n ≥ 10 (weights n) and over all, as the FEP summary. Output `pairwise_<dataset>_per_series.csv`, `pairwise_summary.csv` (dataset, head ∈ {pair, point}, scope, weighting, n_series, n_items, pair_rmse, pair_dir_acc, kendall, spearman), and a SUMMARY section. The ensemble of 3 seeds averages `delta_hat` per pair (and `f` per row) before metrics.
- `compare_arms.py` gains rows `FEP pair-head wKendall`, `FEP pair-head wSpearman`, `FEP pair RMSE`, `FEP pair dir acc`, and the same for series_valid and group_valid, when every tag has `pairwise_summary.csv`; for tags without a Siamese head (base-g) the evaluator still writes the `point` rows so comparisons line up (pair-head rows NaN).
- `jobs/train.sbatch`: env `SIAMESE=1`, `SIAMESE_WEIGHT`, `SIAMESE_HIDDEN`; `jobs/eval.sbatch`: always passes `--pairwise`.
- Ligand-only gate (`--ligand-only`) applies to the encoder input as before; `S-lig` = Siamese + ligand-only.

## Review Focus

- A batch with no same-series pair: the Siamese terms must be exact zeros with gradient flow unaffected; test `test_siamese_no_pairs`.
- Antisymmetry: `delta_hat(z_i, z_j) == -delta_hat(z_j, z_i)` to 1e-6 in eval mode; test `test_siamese_antisymmetric`.
- Loading an old plain `GATv2Net` state dict into the wrapper and back must be lossless; test `test_state_dict_roundtrip`.
- Pairwise eval with a tag that has no Siamese head writes `point` rows only and no error; test `test_pairwise_eval_without_head`.
- BatchNorm inside the head in eval mode with a single pair: must not raise (use `BatchNorm1d` only on batches ≥ 2 or switch to LayerNorm; decide and test `test_siamese_single_pair_eval`).

---

### Task S1: head, loss, training wiring

**Files:** create `src/heads.py`; modify `src/train_paired.py`, `src/pair_losses.py`, `jobs/train.sbatch`; tests `tests/test_heads.py`, extend `tests/test_pair_losses.py`, `tests/test_train_paired_smoke.py`.

**Interfaces:** `heads.SiameseHead(dim, hidden=512, act="leaky_relu")` with `forward(zi, zj) -> delta_hat [n]`; `heads.AffinityWithSiamese(gnn: GATv2Net, siamese: SiameseHead | None)` exposing `embed(data) -> z [B, 1536]`, `affinity(z) -> [B,1]`, `forward(data) -> [B,1]` (so the existing training/eval code paths keep working), `from_plain_state_dict(sd)` / `to_plain_state_dict()`; `pair_losses.siamese_loss(delta_hat, y, i, j, tie_std, tau_std) -> (huber_term, bce_term)`; `train_paired.py` flags above; the log line gains ` siam S` (weighted term); sidecar fields.

- [x] Steps: failing tests (`test_siamese_antisymmetric`, `test_siamese_no_pairs`, `test_state_dict_roundtrip`, `test_siamese_single_pair_eval`, `test_siamese_loss_manual`, smoke training with `--siamese`), implement, pass, `py_compile`, `bash -n`, tick.
  Done 2026-11-05. Decisions: BatchNorm1d kept (both orders of every pair run as one batch, so one pair is a batch of 2 and eval uses running stats); Huber delta 1.0 pK / scale and tau 0.5 pK / scale are constants (`SIAMESE_HUBER_PK`, `SIAMESE_TAU_PK`, recorded in the sidecar); head initialised after the GATv2Net (encoder init identical to the plain arm); head not called and terms exact zeros in a batch without pairs; log line `... hinge H siam S valid_pearson ...` (always printed, 0 without `--siamese`); `--siamese-weight` must be > 0.

### Task S2: pairwise evaluation and comparison

**Files:** modify `src/eval_paired.py`, `src/compare_arms.py`, `jobs/eval.sbatch`; tests `tests/test_eval_paired.py`, `tests/test_compare_arms.py`.

- [x] Steps: failing tests (`test_pairwise_metrics_on_toy` with hand-computed pair RMSE/direction accuracy/Kendall from mean scores, `test_pairwise_eval_without_head`, `test_pairwise_ensemble_averages_delta`, compare rows), implement, pass, tick.
  Done 2026-11-05. Decisions: scopes per dataset FEP n>=10 / all, series_valid, group_valid and test n>=5 / n>=10 / all (CASF UniProt groups have ~5 members); compare rows at FEP n>=10 and series_valid / group_valid n>=5 (labels carry the scope), pair-head and point-head rows; a `Pair head vs pointwise head` table and the shortcut gate's pair-head rows set the arm's pair head against the reference's point head; per-series n = ligands in at least one pair; models without a head on every model -> point rows only (warning if mixed).

### Task S3: runs (coordinating session + Sonnet monitor)

- [ ] `TAG=S TRAIN_KINDS=0 SIAMESE=1 SIAMESE_WEIGHT=1.0 DATA_DIR=$WORK/data_ac`, `TAG=S-lig ... LIGAND_ONLY=1`, 3 seeds each; evals with `--pairwise` for S, S-lig, and a re-eval of `base-g` and `base-g-lig` (pointwise rows only) into `eval/<tag>-pw/`; `compare_arms --tags base-g,S,base-g-lig,S-lig --ref base-g --lig-pairs base-g-lig:base-g,S-lig:S`; `REPORT_siamese.md`.
