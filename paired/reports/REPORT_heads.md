# AEV-PLIG head arms: Siamese (S) at 10 seeds, pair head + Zernike moment heads (arm P) at 3 seeds

Date 2026-11-06. Plans `docs/plans/2026-11-05-aevplig-siamese-arm-S.md`, `docs/plans/2026-11-05-aevplig-pairhead-moments-arm-P.md`. Reference `base-g` (data_ac: group-disjoint holdout, kinds 0). Numbers below are from the monitor's collection (`scratchpad/aevplig_out/*.txt`); `PMneg` is still uncollected (certificate lapsed at 00:35 UTC while it ran; its eval 8114893 is chained and will have run).

## 1. Siamese head at 10 seeds: null on FEP

| | base-g (10 seeds) | S (10 seeds) |
|---|---|---|
| FEP wPCC per seed, mean | 0.478 | 0.480 |
| paired S − base-g per seed | | −0.007, +0.032, +0.047, −0.018, −0.001, +0.021, +0.023, −0.034, −0.006, −0.035 (mean +0.002, sd 0.028, se 0.009) |
| FEP wPCC / wKendall, 10-member ensemble | 0.589 / 0.429 | 0.581 / 0.416 |
| Siamese head as ranker, FEP wKendall (ensemble) | — | 0.426 (its own point head 0.416) |
| CASF Pearson, ensemble | 0.890 | 0.898 |
| series_valid wSpearman | 0.390 | 0.384 |
| group_valid Pearson / RMSE | 0.589 / 1.327 | 0.586 / 1.359 |

The 3-seed +0.024 was seeds 1 and 2 (+0.032, +0.047); over ten seeds the paired mean is +0.002 ± 0.009. A small CASF gain (+0.008 at the ensemble) persists; FEP ranking does not. The Siamese head is a ranker exactly as good as the pointwise head it shares an encoder with. The 10-member base-g ensemble reproduces the paper's FEP number (0.589 / 0.429), so the group holdout cost nothing at 10 members either.

**Calibration from this pair of runs:** the per-seed paired sd on FEP wPCC is 0.028, so three seeds resolve effects of ~0.05 and ten seeds ~0.02. The arm-P numbers below are at three seeds and must be read with that.

## 2. Arm P at 3 seeds (vs base-g seeds 0–2: 0.485 / 0.462 / 0.456)

| tag | what | FEP wPCC per seed | paired Δ vs base-g | ensemble FEP wPCC / wKendall | CASF | group_valid Pearson | pair-head FEP wKendall (ens) |
|---|---|---|---|---|---|---|---|
| base-g | — | 0.485 / 0.462 / 0.456 | — | 0.537 / 0.378 | 0.880 | 0.584 | — |
| M | moment heads (pocket + ligand), 0.3 | 0.490 / 0.451 / 0.456 | −0.002 | 0.541 / 0.385 | 0.894 | 0.580 | — |
| P | pair head (within-listwise), 0.3 | 0.486 / 0.484 / 0.444 | +0.004 | 0.533 / 0.380 | 0.885 | 0.585 | 0.390 |
| **PM** | pair head + moments | 0.526 / 0.507 / 0.466 | **+0.032 [+0.041, +0.045, +0.010]** | **0.568 / 0.406** | 0.892 | 0.577 | **0.423** |
| P1 | pair head, 1.0 | 0.483 / 0.500 / 0.485 | +0.022 [−0.002, +0.038, +0.029] | 0.562 / 0.394 | 0.887 | 0.591 | 0.403 |
| PMpos | PM on kinds 0,3,5,7, parent-aware sampler, chunk 32 (control for PMneg) | 0.467 / 0.460 / 0.498 | +0.007 | 0.541 / 0.384 | 0.887 | 0.578 | 0.412 |
| PMneg | PMpos + twins/edits as zero-target candidates | 0.447 / 0.373 / 0.455 | −0.043 vs base-g; −0.050 vs PMpos [−0.020, −0.087, −0.043] | 0.494 / 0.351 | 0.889 | 0.560 | — (twin rank acc 0.705, contact-edit 0.952 vs PMpos 0.48 / 0.47) |
| M-lig | moments, ligand-only | 0.462 / 0.434 / 0.458 | +0.025 vs base-g-lig (0.404/0.444/0.431) | 0.503 / 0.352 | 0.827 | 0.505 | — |
| PM-lig | pair head + moments, ligand-only | 0.456 / 0.450 / 0.428 | +0.018 vs base-g-lig | 0.503 / 0.354 | 0.837 | 0.500 | 0.375 |

Moment heads (R², ensemble): pocket on unseen targets 0.26 (M) vs 0.10 (M-lig); on series_valid 0.45 vs 0.36; ligand ~0.60–0.81 everywhere. The encoder carries pocket information the ligand alone does not give, but the ligand-centred 8 Å shell lets a ligand-only model recover a third of it.

Negative discrimination (aug_valid, rank accuracy, pair head): PM 0.55 on twins / 0.46 on contact edits, near chance, as expected without negatives. PMneg learns them (0.705 / 0.952) and loses 0.050 wPCC against its matched control PMpos at every seed, with unseen-target Pearson 0.560 vs 0.578. Fourth time the same pattern (arm 4, A, C, PMneg): construction-matched negatives are learnable and cost affinity ranking, now shown with a control that differs only by the negatives in the listwise term.

## 3. Reading
- **PM is the best 3-seed arm in the programme** (+0.032 per seed, positive at all three, ensemble +0.031 wPCC and +0.028 Kendall, pair head 0.423 Kendall vs base-g 0.378), but the Siamese lesson applies with full force: a 3-seed +0.024 just vanished at 10 seeds. PM's ligand-only counterpart also gains (+0.018), so the structure-specific part is ~+0.014.
- Neither component alone does it: moments alone −0.002, pair head alone +0.004 (0.3) / +0.022 (1.0). The combination is what moves, which is either a real interaction or two noisy positives adding.
- The moments aux improves the ligand-only model as much as the full model, so part of PM's effect is ligand-side representation shaping (ligand R² 0.6–0.8 is easy to fit), not pocket reading.
- Decision rule stated in advance: PM is adopted only if, at 10 seeds, PM − base-g has a paired mean above 2 se (≈ +0.02) AND exceeds PM-lig − base-g-lig by more than its se. Otherwise it joins arms 1, 4, A, C, S as null at the FEP resolution.

## 4. Next (needs the Isambard certificate)
1. PMneg collected (above).
2. Launched 03:05 UTC: PM seeds 3–9 (8120835), PM-lig 3–9 (8120836), base-g-lig 3–9 (8120837), evals PM-10 / PM-lig-10 / base-g-lig-10 (8120838–8120840). Then: launch PM seeds 3–9 and PM-lig seeds 3–9 (14 jobs, ~1.3 h) and evaluate PM-10 / PM-lig-10 / base-g-lig-10 (seeds 3–9 of base-g-lig are also needed: 7 more jobs).
3. `compare_arms` over the 10-seed sets; final verdict.


## 5. PM at 10 seeds (2026-11-06): not adopted under the pre-registered rule

| | base-g-10 | PM-10 | S-10 | base-g-lig-10 | PM-lig-10 |
|---|---|---|---|---|---|
| FEP wPCC / wKendall, 10-member ensemble | 0.588 / 0.429 | 0.591 / 0.425 | 0.581 / 0.416 | 0.526 / 0.376 | 0.530 / 0.375 |
| pair head as ranker, FEP wKendall | — | 0.429 | 0.426 | — | 0.389 |
| CASF Pearson | 0.890 | 0.899 | 0.898 | 0.826 | 0.842 |
| group_valid Pearson / RMSE | 0.589 / 1.327 | 0.586 / 1.358 | 0.586 / 1.359 | 0.506 / 1.521 | 0.518 / 1.489 |
| series_valid wSpearman | 0.390 | 0.389 | 0.384 | 0.370 | 0.398 |

Per-seed paired differences over 10 seeds:

| metric | PM − base-g (mean ± se) | per seed | PM-lig − base-g-lig | structure-specific |
|---|---|---|---|---|
| FEP wPCC | +0.009 ± 0.008 | +0.041, +0.045, +0.010, +0.003, −0.025, +0.022, +0.018, +0.004, −0.023, −0.010 | +0.016 ± 0.012 | −0.007 ± 0.012 |
| FEP wKendall | +0.007 ± 0.007 | | +0.009 ± 0.009 | −0.002 ± 0.013 |
| CASF Pearson | **+0.010 ± 0.003** (9 of 10 seeds) | | **+0.019 ± 0.009** | −0.010 ± 0.009 |
| group_valid Pearson | +0.001 ± 0.004 | | +0.008 ± 0.006 | −0.007 ± 0.007 |

Rule: adopt only if the 10-seed paired FEP mean exceeds 2 se (≈ +0.02) and exceeds the ligand-only gate's gain. PM fails both: +0.009 (1.1 se), and the ligand-only model gains more (+0.016). The 3-seed +0.032 was seeds 0 and 1 (+0.041, +0.045), as the Siamese +0.024 was seeds 1 and 2.

The one robust effect is CASF Pearson +0.010 at 4 se (and the same for S), but PM-lig gains +0.019 on CASF from the same heads, so it is a ligand-side representation effect of the moment heads, not pocket reading. The pair head ranks the FEP series exactly as well as the pointwise head (0.429 vs 0.425 Kendall), as the Siamese head did.

## 6. What the head arms establish

1. Heads over the shared encoder learn their tasks (Siamese Δ, within-listwise ordering, pocket/ligand moments R² 0.26–0.81, negatives when given) and the encoder absorbs them without changing FEP ranking. The jax-Ka pattern transplanted: the gains stay in the heads' own metrics.
2. Three-seed screens on the FEP benchmark are not usable: the per-seed paired sd is 0.024–0.028, so +0.03 at 3 seeds arises by chance regularly (two of two arms screened positive at 3 seeds and regressed to +0.00–0.01 at 10). Any further arm needs 10 seeds, or a screen with more pairs (series_valid 3,591 rows / group_valid 5,124 rows within-series ranking, where the paired sd is ~0.01–0.015).
3. Negatives in any form (pose decoys, twins, edits, listwise zero-targets) are learned and cost affinity, now with a matched control (PMneg vs PMpos −0.050 per seed).
4. The pocket side is the weak link, not the losses: pocket-moment R² on unseen targets is 0.26 for the full model vs 0.10 ligand-only, the ligand-only model matches the full model on held-out series, and 15-PGDH sits below Vina. The encoder's protein input (radial AEVs within 5.1 Å, 22 atom types, no angular terms, no pocket-level features) has not been changed by any arm. The untested Zernike use is as an *input* (pocket and ligand invariants concatenated to the pooled embedding), and that is the next lever if the line continues.
