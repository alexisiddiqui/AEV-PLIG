# AEV-PLIG arm Z: Zernike pocket/ligand invariants as model input — 3-seed screen

Date 2026-11-07. Plan `docs/plans/2026-11-07-aevplig-moments-input-arm-Z.md`; tables `work/output/eval/REPORT_momentsin_arms.md`. Reference `base-g` / `base-g-lig` (data_ac, kinds 0). Descriptors: shell-mode 3DZD, order 12, 7 channels, standardised with the training rows' mean/sd; FEP and 15-PGDH descriptors computed from the prepared structures with the same code (1,184/1,184 and 43/43).

## Verdict

**No full-model configuration passes the screen; the gate result is the finding.** Adding the pocket and/or ligand invariants (projected to 256 and concatenated to the pooled embedding) to the full graph+AEV model leaves unseen-target and held-out-series metrics unchanged or lower and costs 0.02 on FEP; the cross-channel block makes it worse. But adding the same inputs to a *ligand-only* model raises its unseen-target Pearson by +0.063 ± 0.008 and its CASF Pearson by +0.077 ± 0.011, to within 0.01–0.02 of the full model. The descriptors therefore carry essentially the pocket information the radial AEVs provide for absolute affinity, and that information is already saturated in the full model. Neither representation moves within-series ranking on FEP.

## Setup
Arms (3 seeds, data_ac, kinds 0): `Zp` pocket (343), `Zl` ligand (343), `Zb` both (686), `Zbx` both + cross block (1029), `Zlin` moments-only MLP (no graph), `Zb-lig` / `Zbx-lig` ligand-only encoder + moments. Projection 256 → `fc1` input 1792; encoder and all other weights identical to base-g at the same seed (verified).

## Results (3-seed ensembles; per-seed paired differences mean ± se)

| metric | base-g | Zp | Zl | Zb | Zbx | Zlin | base-g-lig | Zb-lig | Zbx-lig |
|---|---|---|---|---|---|---|---|---|---|
| FEP wPCC (n ≥ 10) | 0.537 | 0.521 | 0.510 | 0.515 | 0.471 | 0.306 | 0.492 | 0.479 | 0.455 |
| FEP wKendall | 0.378 | 0.367 | 0.356 | 0.354 | 0.333 | 0.223 | 0.359 | 0.337 | 0.318 |
| CASF Pearson | 0.880 | 0.887 | 0.887 | 0.898 | 0.880 | 0.855 | 0.823 | **0.893** | 0.869 |
| series_valid wSpearman | 0.380 | 0.352 | 0.379 | 0.347 | 0.316 | 0.224 | 0.351 | 0.339 | 0.315 |
| group_valid Pearson (unseen targets) | 0.584 | 0.569 | 0.590 | 0.589 | 0.542 | 0.465 | 0.497 | **0.569** | 0.557 |
| group_valid RMSE | 1.355 | 1.338 | 1.316 | 1.341 | 1.387 | 1.449 | 1.542 | **1.332** | 1.370 |

| paired Δ (per seed) | Zp − base-g | Zl − base-g | Zb − base-g | Zbx − base-g | Zlin − base-g | Zb-lig − base-g-lig | Zbx-lig − base-g-lig |
|---|---|---|---|---|---|---|---|
| group_valid Pearson | −0.020 ± 0.013 | +0.005 ± 0.015 | +0.004 ± 0.013 | −0.054 ± 0.015 | −0.133 ± 0.008 | **+0.063 ± 0.008** | +0.053 ± 0.019 |
| group_valid RMSE | −0.011 | −0.037 | −0.011 | +0.047 | +0.111 | **−0.208 ± 0.008** | −0.173 |
| series_valid wSpearman | −0.031 ± 0.017 | +0.001 ± 0.020 | −0.029 ± 0.012 | −0.057 ± 0.014 | −0.146 | −0.025 | −0.050 |
| FEP wPCC | −0.026 ± 0.013 | −0.019 ± 0.001 | −0.020 ± 0.012 | −0.046 ± 0.003 | −0.210 | −0.019 ± 0.036 | −0.029 |
| CASF Pearson | +0.003 | +0.008 | +0.016 ± 0.004 | −0.004 | −0.033 | **+0.077 ± 0.011** | +0.054 |

Screen rule (3-seed paired mean > 2 se on group_valid Pearson or series_valid wSpearman, and above the ligand-only gate): no full-model arm qualifies; no 10-seed run.

15-PGDH (pooled Spearman, 43 ligands): Zb −0.02, Zbx −0.05, Zlin −0.08 (base-g +0.21, released 0.09, Vina 0.45): noise level, as for every AEV-PLIG variant on this set.

## Reading
1. **The invariants are a sufficient pocket summary for absolute affinity.** Ligand graph + pocket/ligand moments reaches 0.893 CASF and 0.569 unseen-target Pearson against the full model's 0.880–0.898 and 0.584–0.590. Added to the full model they are redundant.
2. **Descriptors alone are weak** (Zlin: CASF 0.855, unseen targets 0.465, FEP 0.306): the graph encoder does most of the work; the pocket contribution to absolute affinity is a modest, low-dimensional signal that both representations capture.
3. **FEP ranking is untouched by either representation.** Every arm with pocket inputs loses 0.02–0.05 on FEP (the extra input adds variance to the series ordering without adding series-level signal), and the ligand-only+moments model gains on CASF and unseen targets but not on FEP. Within-series ranking depends on something neither the AEVs nor the moments encode: differences between near-identical ligands in the same pocket, where the noise floor of the labels (0.47 pK between replicates) is of the order of the differences being ranked.
4. **The cross block hurts** (−0.054 unseen-target Pearson, −0.046 FEP): 343 product features on a 1792-wide first layer overfit the training pockets.

## What this closes and leaves
- Closed: "the pocket side is thin and a richer pocket representation will lift FEP" in the form tested. The pocket information AEV-PLIG uses is recoverable from a 686-dim invariant summary; richer pocket inputs of this kind do not add to it.
- Left: the ligand-only+moments result is a cheap, pose-free model family at 0.89 CASF / 0.57 unseen-target Pearson that needs no protein graph; worth noting for screening uses, not for FEP-style ranking.
- The programme's consistent picture across arms 1, 4, A, C, S, P, Z: AEV-PLIG's FEP ranking is at the level the data support for this architecture; every lever that changes training signal, heads or pocket inputs leaves it within ±0.02, while the clean effects all land on CASF (absolute, seen targets).

## Files
Models `work/output/trained_models/{Zp,Zl,Zb,Zbx,Zlin,Zb-lig,Zbx-lig}_{0,1,2}.model`; evals `work/output/eval/<tag>/`; descriptors `work/data/zernike_fep.npz`, `zernike_pgdh.npz`, side file `work/data_ac/paired_zernike.npz`. Jobs 8161348 (eval-set descriptors), 8161546–8161552 (train), 8161553–8161559 (evals), 8162924 (15-PGDH).
