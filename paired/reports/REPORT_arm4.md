# AEV-PLIG arm 4 (pose decoys with near-native positives): result

Date 2026-11-05. Tree `/scratch/u6xq/hussain.u6xq/aevplig-paired` (local mirror `~/isambard-ssh/aevplig-paired`). Plan `docs/plans/2026-11-05-aevplig-pose-decoys-arm4.md`; tables `work/output/eval/REPORT_arm4_arms.md` (from `src/compare_arms.py`).

## Verdict

**Null for affinity ranking, positive for pose sensitivity.** Training AEV-PLIG with far pose decoys (one-sided hinge) and near-native positives (experimental label) makes it strongly pose-sensitive but does not improve FEP-benchmark ranking; decoys alone make it worse. This reproduces PIGNet2's ablation pattern (negatives alone hurt, positives repair) without PIGNet2's net gain. Together with arm 1, the evidence says AEV-PLIG's FEP ceiling is not set by how its own examples are weighted or contrasted.

## Setup

- Poses: for every PDBbind complex (19,174; train/valid/test), 2 near-native (RMSD 0.3-2.0 Å) and 2 decoys (RMSD ≥ 4 Å, ≥ 3 Å fallback) from one generator (rigid/torsion perturbation or flip/ETKDG re-embedding, then rigid-body soft-repulsion relief to 3.0 Å applied identically to both kinds), no clash below min(2.2 Å, crystal), decoys keep ≥ 50% of crystal contacts. 74,653 poses; identity check: the copied featuriser reproduces the parent's crystal graphs bit-for-bit (200/200).
- Contact cue after relief (min_dist alone): crystal vs decoy AUROC 0.50, near-native vs decoy 0.74, crystal vs near-native 0.21 (near-natives are slightly more relaxed than crystals).
- Train rows: base 91,763 (kind 0); `nda` adds 34,974 decoys; `pdanda` adds 34,675 near-natives and 34,974 decoys. BindingNet/BindingDB rows unchanged. Loss: MSE over kinds 0/1 + 1.0 × mean relu(pred − (y − 0.75 pK)) over decoys. Same series sampler, seeds, checkpoint rule as arm 1; reference = arm 1's `base` (identical kind-0 rows, verified).
- Per-epoch time 26 s (nda) / 33 s (pdanda); ~1.5 / ~1.9 h per seed.

## Results (3-seed ensembles; per-seed paired differences in brackets)

| metric | base | nda | pdanda | nda − base (per seed) | pdanda − base (per seed) |
|---|---|---|---|---|---|
| FEP wPCC (n ≥ 10) | 0.575 | 0.523 | 0.537 | −0.049 [−0.019, −0.096, −0.032] | −0.017 [+0.044, −0.047, −0.049] |
| FEP wKendall | 0.415 | 0.374 | 0.377 | −0.040 | −0.006 |
| FEP wRMSE (all) | 0.975 | 1.054 | 0.958 | +0.085 | −0.017 |
| series_valid wSpearman | 0.370 | 0.352 | 0.370 | −0.017 | +0.004 |
| CASF Pearson | 0.888 | 0.882 | 0.887 | −0.015 | −0.001 |
| Pose: decoy rank acc (PDBbind valid) | 0.735 | 0.991 | 0.966 | +0.30 | +0.26 |
| Pose: crystal − decoy gap (pK) | 0.68 | 5.37 | 5.09 | +4.7 | +4.4 |
| Pose: near-native spread (pK) | 0.50 | 1.76 | 0.36 | +1.22 | −0.13 |
| Pose: crystal Pearson | 0.814 | 0.780 | 0.799 | −0.040 | −0.017 |

Reading: the vanilla model already ranks the crystal above both decoys 73% of the time with a 0.7 pK gap (it is not pose-blind); the hinge pushes decoys ~5 pK down. Decoys alone also make near-native poses unstable (spread 0.5 → 1.8 pK) and cost 0.05 wPCC on FEP and 0.04 Pearson on crystal poses; adding near-natives restores stability (0.36 pK, better than base) and recovers most of the FEP loss, but not past the baseline.

## What this closes and what it leaves open

- Closed: "AEV-PLIG would rank congeneric series better if it depended more on the interactions" is not supported in its direct form. Pose-sensitive training changes what the model does with poses, not how it orders ligands on FEP-prepared poses.
- Consistent with Rich et al. 2026 (15-PGDH): their evidence pointed at generalisation rather than pose discrimination, and arm 4 agrees.
- Still open (user direction): construction-matched soft negatives in a property-descriptor space (Zernike shape/property twins, design in `docs/zernike_soft_negatives_design.md`), and a pocket-group-disjoint split so that gains cannot come from target memory.

## Files

Models `work/output/trained_models/{nda,pdanda}_<i>.model`; evals `work/output/eval/{nda,pdanda,base-poses}/` (+ `seed<i>/`); poses `work/data/pdbbind_poses.pickle` (7.2 GB), `pdbbind_poses_meta.csv`, `pdbbind_poses_summary.json`; datasets `work/data_poses/processed/paired_{train,valid,test,series_valid,poses_valid}.pt`. Jobs: featurise 8079261, build 8080092, train 8080229 (nda) / 8080230 (pdanda), evals 8080238 / 8080241 / 8080242.
