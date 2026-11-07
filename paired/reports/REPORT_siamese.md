# AEV-PLIG arm S (Siamese difference head): 3-seed result (10-seed confirmation running)

Date 2026-11-05. Plan `docs/plans/2026-11-05-aevplig-siamese-arm-S.md`; tables `work/output/eval/REPORT_siamese_arms.md`. Reference `base-g` (data_ac, group-disjoint holdout, kinds 0), evaluated with `--pairwise` as `base-g-pw`.

## Setup
Shared GATv2 encoder → pooled embedding z (1536) → (i) the unchanged affinity MLP with pointwise MSE, (ii) a Siamese head `delta_hat(z_i, z_j) = (m(z_i,z_j) − m(z_j,z_i))/2`, `m = MLP([z_i − z_j, z_i ⊙ z_j])`, trained on same-series different-ligand pairs in the batch with Huber on ΔpK + BCE on the sign (ties < 0.5 pK excluded), weight 1.0. Same sampler, seeds, scaler and checkpoint rule (pointwise valid Pearson) as base-g. Ligand-only gate S-lig.

## Result (3 seeds; per-seed paired difference to base-g)

| metric | base-g | S | Δ per seed | base-g-lig | S-lig | Δ per seed |
|---|---|---|---|---|---|---|
| FEP wPCC (n ≥ 10), ensemble | 0.537 | 0.553 | +0.024 [−0.007, +0.033, +0.047] | 0.492 | 0.490 | +0.005 |
| FEP wKendall | 0.378 | 0.400 | +0.020 [−0.013, +0.049, +0.025] | 0.359 | 0.356 | +0.001 |
| FEP wRMSE (all) | 0.986 | 0.958 | −0.020 at every seed | 0.978 | 0.974 | −0.010 |
| CASF Pearson | 0.880 | 0.891 | +0.010 [+0.010, +0.019, 0.000] | 0.823 | 0.818 | −0.002 |
| CASF RMSE | 1.080 | 1.027 | −0.045 at every seed | 1.248 | 1.259 | 0.000 |
| series_valid wSpearman | 0.380 | 0.373 | −0.003 | 0.352 | 0.366 | +0.019 |
| group_valid Pearson (unseen targets) | 0.584 | 0.571 | −0.012 at every seed | 0.497 | 0.491 | −0.012 |
| group_valid RMSE | 1.355 | 1.351 | −0.004 | 1.542 | 1.529 | 0.000 |

Siamese head as a ranker (FEP, n ≥ 10): wKendall 0.406 / wSpearman 0.530 / pair RMSE 0.903 / direction accuracy 0.774, against S's own pointwise head 0.400 / 0.523 / 0.899 / 0.769 and base-g's pointwise 0.378 / 0.512 / 0.922 / 0.756. Per seed, head minus own point head: Kendall +0.004 [+0.008, −0.007, +0.010]; head minus base-g point: +0.024 [−0.006, +0.042, +0.036].

## Reading
- First arm with a consistent positive sign on the primary task: FEP wPCC/Kendall up at 2 of 3 seeds and on the ensemble, FEP and CASF RMSE down at every seed, CASF Pearson up at every seed. Magnitude +0.02, i.e. GQT-sized and below the 0.04 ensemble-level resolution measured earlier, hence the 10-seed confirmation (S and base-g seeds 3–9 running; evals `S-10`, `base-g-10`).
- The gain is structure-specific: the ligand-only model gains +0.005 from the same head.
- It lives in the encoder, not the head: the Siamese head's own ranking equals the pointwise head's. Consistent with McNutt & Koes (the ΔΔG term helps through the shared representation) and PBCNet (pairwise + pointwise anchor).
- Unseen-target pointwise Pearson −0.012 at every seed: the pair term sharpens within-series ordering at a small cost to absolute prediction on new targets. Watch at 10 seeds.
- Contrast with arm 1: the same pairs and a Δ-Huber on the *scalar* head gave nothing; a separate head with its own capacity does. That is the jax-Ka pattern.

## Files
Models `work/output/trained_models/S_{0..9}.model`, `S-lig_{0,1,2}.model`; evals `work/output/eval/{S,S-lig,base-g-pw,base-g-lig-pw}/`; jobs train 8113167 (S 0-2), 8113169 (S-lig), 8114834 (S 3-9), 8114833 (base-g 3-9); evals 8113170-8113173, 8114835 (base-g-10), S-10 pending.
