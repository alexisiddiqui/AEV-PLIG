# AEV-PLIG paired-data programme: results and verdict

7 November 2026. Living copy (comments, edits): https://claude.ai/artifact/VVpi6tnQJ3xmyVPNqR7tcB

## Summary

Seven paired-data arms were built on the released AEV-PLIG recipe and none moves FEP-benchmark ranking beyond seed noise: within-series ranking loss, pose decoys, Zernike-selected shape twins, in-place property edits, a Siamese difference head, a pair head with pocket and ligand moment heads, and the moments as model inputs. Every arm learned the task it was given (decoys, twins, edits and series order are all discriminated at 0.8 to 0.99 where the vanilla model sits at chance), and the shared encoder absorbed each without changing how it orders congeneric ligands. The seed-robust effects all land on CASF-2016, that is absolute affinity for seen targets, and are matched by ligand-only controls.

Two calibration results frame everything else. The vanilla recipe reproduces the paper exactly (10-member ensemble FEP weighted PCC 0.589, Kendall 0.429; CASF 0.892) and loses nothing on a target-group-disjoint split. The per-seed paired standard deviation on FEP weighted PCC is 0.028, so three seeds resolve about 0.05 and ten about 0.02; two arms that screened at +0.02 to +0.03 on three seeds regressed to +0.002 and +0.009 at ten.

The pocket information AEV-PLIG uses for absolute affinity is recoverable from a 686-dimensional rotation-invariant summary: a ligand-only graph given the pocket and ligand Zernike invariants reaches 0.893 CASF and 0.569 unseen-target Pearson against the full model's 0.880 to 0.898 and 0.584 to 0.590. Adding the same inputs to the full model is redundant. On the one genuinely unseen target tested (15-PGDH, Rich et al. 2026), every AEV-PLIG variant sits below Vina. The programme leaves a group-disjoint split, a ten-seed protocol, pose and augmentation evaluators, an exact reproduction of the 15-PGDH test, and descriptors for all 96,947 training complexes.

## Data, splits and evaluation protocol

The training set is the paper's: PDBbind v2020 crystals, BindingNet v1 (ChEMBL ligands modelled onto PDBbind templates) and BindingDB-DCS (Surflex-docked), filtered at Tanimoto 0.9 to the FEP ligands, 95,680 rows. Its pairing structure, measured from the tables: 91,596 rows fall in 2,098 same-UniProt groups (36M pairs, 9.4M with |ΔpK| ≥ 2); BindingNet alone gives 4,352 same-receptor congeneric series sharing one receptor file and one modelling pipeline (2.2M pairs); 63,054 BindingNet rows sit on 5,384 PDBbind-train crystals, the FEP benchmark's shape; 5,080 BindingDB pairs are one measurement docked into two crystals of the same protein with identical labels; 1,714 PDBbind replicate pairs give a label-noise floor of 0.47 pK median; measured non-binders are almost absent (202 PDBbind ">" rows, none in the other sources).

| Split or test | What it holds | Why it exists |
| --- | --- | --- |
| train (data_ac) | 86,965 crystal/modelled rows after the two holdouts | reference for every arm |
| valid (paper) | 993 PDBbind rows, random by complex; 89% share a target with train | the paper's checkpoint selection, kept unchanged |
| test (CASF-2016) | 277 PDBbind core rows; 109 of them are BindingNet templates for 1,044 train rows | absolute affinity, seen targets |
| series_valid | 176 whole BindingNet/BindingDB series held out (3,591 rows) | within-series ranking; ligand-dominated, since 91% keep the target in train |
| group_valid | 55 whole target groups (UniProt merged over non-generic PDBbind names), 5,124 rows, all sources | unseen-target prediction |
| FEP benchmark | Ross et al. 2023 Schrödinger set, 92 series, 1,184 ligands, weighted means over the 43 series with ≥ 10 ligands | the paper's headline metric |
| 15-PGDH | Rich, Dodda & Price 2026: 43 ligands, 4 series, one crystal (9PFL), Vina-minimised poses; zero training rows on the target | external, genuinely unseen |

Protocol fixed after the first two head arms: paired per-seed differences against the matched baseline, a ligand-only control trained with the same change, and adoption only when the ten-seed paired mean exceeds two standard errors and the ligand-only control's gain. Three-seed runs serve as screens on series_valid and group_valid only. The FEP featurisation reproduces the paper's released-model predictions on all 1,184 ligands with zero difference, and the 15-PGDH featurisation reproduces the paper's default-model predictions on all 43.

## Baseline: the vanilla recipe reproduced and characterised

The released ensemble and a full retrain agree with the paper. Retrained on the paper's split (10 seeds, 56 minutes per seed on a GH200), the ensemble scores CASF Pearson 0.892 / RMSE 1.041 against the released 0.893 / 1.039, and on the FEP benchmark weighted PCC 0.590 / Kendall 0.428 against the released 0.590 / 0.416. Retrained on the group-disjoint split (base-g, 10 seeds) it scores 0.589 / 0.429 on FEP and 0.890 on CASF, so the holdout costs nothing.

| Baseline measurement | Value |
| --- | --- |
| FEP weighted PCC, single seed, mean (sd) over 10 seeds | 0.478 (0.018) |
| FEP weighted PCC, 3-member ensemble | 0.537 to 0.575 |
| FEP weighted PCC, 10-member ensemble | 0.589 |
| Per-seed paired sd of FEP weighted PCC between matched arms | 0.024 to 0.028 |
| Two 3-member ensembles on near-identical data | differ by 0.04 on FEP; per-series Pearson swings up to ±0.5 |
| Unseen-target Pearson / RMSE (group_valid) | 0.589 / 1.33 |
| Ligand-only model (AEVs zeroed): CASF / FEP / unseen targets | 0.823 / 0.492 / 0.497 |
| Ligand-only vs full on held-out series (within-series Spearman) | 0.383 vs 0.370 |
| 15-PGDH pooled Spearman: released / retrain / base-g; Vina; FEP+ | 0.09 / 0.15 / 0.21; 0.45; 0.70 |
| Pose sensitivity of the vanilla model: crystal ranked above both far decoys | 74%, gap 0.7 pK |

Three of these shape the reading of every arm. The ligand-only model loses 0.06 on CASF and 0.09 on unseen targets but matches the full model on held-out series, so within-series ranking on seen targets is mostly a ligand task. The FEP benchmark's seed noise is of the same size as the effects the literature reports for pairwise training. And on 15-PGDH, two retrains of one recipe move per-series Spearman by ±0.4 on 7 to 13 compounds, so that set is a sanity check and nothing more.

## The seven arms and their results

Every arm keeps the paper's architecture, optimiser, batch size, 200 epochs, seeds and checkpoint rule (running mean of validation Pearson), and uses the same series-aware batch sampler as its baseline, so the paired difference isolates the change. Differences are per-seed paired FEP weighted PCC against the matched baseline, mean ± standard error; "gate" is the same change applied to a ligand-only model (AEV columns zeroed).

| Arm | What it adds | Seeds | FEP wPCC Δ vs baseline | Ligand-only gate Δ | What the model learned | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Within-series Δ loss | Δ-Huber or listwise term on the affinity scalar over same-receptor pairs (weights 0.3, 1.0) | 3 | −0.001 [+0.045, −0.028, −0.020]; weight 1.0: −0.026 | +0.013 | nothing beyond the MSE | null |
| 4 Pose decoys | far decoys (RMSD ≥ 4 Å) with a one-sided hinge, near-native poses with the label, one generator and matched relief | 3 | decoys only −0.049; with near-natives −0.017 [+0.044, −0.047, −0.049] | not applicable | decoy rank accuracy 0.74 → 0.97, gap 0.7 → 5 pK | null for affinity; pose-sensitive |
| A Shape twins | Zernike-selected wrong-chemistry ligands of the same shape, overlaid into the pocket; cognate ligand regenerated by the same path | 3 | −0.015 [+0.028, −0.067, −0.005]; unseen targets −0.055 at every seed | −0.135 (the negatives are unsolvable from the ligand) | twin rank accuracy 0.48 → 0.83 | null; hurts unseen targets |
| C In-place edits | MMP-style property edits at contact sites (hinge) and exposed sites (label kept) | 3 | −0.028; unseen targets −0.075 | −0.099, but the ligand-only model still learns the edits | contact-edit rank accuracy 0.43 → 0.99 | null; hurts unseen targets |
| S Siamese head | antisymmetric ΔpK head over the shared embedding, Huber + sign BCE, with the pointwise anchor | 10 | +0.002 ± 0.009 (3-seed screen had read +0.024) | +0.005 | the head ranks exactly as the pointwise head | null |
| P Pair head + moment heads | GQT-style within-listwise pair head and pocket/ligand moment regression heads | 10 | +0.009 ± 0.008 (3-seed screen had read +0.032) | +0.016 | pair head Kendall 0.429 = pointwise; pocket R² 0.26 on unseen targets | not adopted |
| P-neg Negatives in the pair head | twins and edits as zero-target listwise candidates, matched control | 3 | −0.050 vs its control at every seed | — | twin / edit rank accuracy 0.70 / 0.95 | negative |
| Z Moments as input | pocket and/or ligand invariants projected and concatenated to the embedding; moments-only baseline | 3 | pocket −0.026, ligand −0.019, both −0.020, with cross block −0.046; moments-only model 0.306 absolute | ligand-only + moments: CASF +0.077, unseen targets +0.063 | the invariants carry the pocket information the AEVs carry | redundant in the full model |

The within-series losses (arms 1, S, P) share one pattern: the pair signal is learned, the pointwise head is unchanged, and the pair heads themselves rank the FEP series no better than the pointwise head they share an encoder with. The negatives (arms 4, A, C, P-neg) share another: construction-matched negatives are learned to near-perfect discrimination and cost absolute prediction on unseen targets by 0.05 to 0.07 Pearson, now shown with a control that differs only by the negatives in the listwise term. Arm Z closes the pocket-representation question in the form tested: a richer invariant description of the pocket adds nothing the AEVs do not already give.

## What the arms establish

The gates held in every arm, so the nulls are informative rather than artefacts. Each construction was matched on source and pipeline (positives and negatives through the same overlay, edit or relief path), contact-distance cues were measured and driven to chance before featurisation (crystal-versus-decoy AUROC 0.50 after relief), the twin exclusions removed every same-ligand, same-group and same-template candidate, and a ligand-only model trained with the same change accompanied every structure-based arm.

Four mechanisms are now documented with numbers.

1. Within-series signal is already fully used by the pointwise loss. The same-receptor pairs that drive FEP-style ranking are in every batch of the vanilla recipe, and a ranking term, a Siamese head or a listwise pair head adds information the MSE did not already extract: the heads rank as well as the pointwise head and no better (Kendall 0.426 to 0.429 on FEP in both cases).
2. Negatives trade absolute accuracy for discrimination. Decoys, twins, edits and listwise zero-targets are learned to 0.83 to 0.99 rank accuracy and cost 0.05 to 0.07 Pearson on unseen targets, with the loss shown against a control that differs only by the negatives. The jax-Ka lesson that cross-example negatives need construction matching was followed; matched negatives still do not help affinity.
3. The pocket contribution to absolute affinity is low-dimensional and saturated. Pocket and ligand 3D Zernike invariants (686 numbers) lift a ligand-only model to the full model's CASF and unseen-target scores, a moments-only MLP reaches 0.855 CASF, and the invariants add nothing to the full model. Pocket-moment regression from the full model's embedding reaches R² 0.26 on unseen targets against 0.10 from a ligand-only embedding, so the encoder does carry pocket information the ligand cannot give, but a richer invariant summary of it is redundant.
4. The FEP benchmark cannot resolve what the arms were looking for at the budgets used. With a per-seed paired sd of 0.028 and 43 scored series, a three-seed screen produces +0.03 by chance regularly; the two arms that screened positive regressed to +0.002 and +0.009 at ten seeds. The held-out series and groups (3,591 and 5,124 rows, paired sd about 0.01) are the usable screens, and on those every arm is flat or negative.

What this does not test: changes to the protein input itself (angular AEV terms, more atom types, pocket-level graph features), data quality and pose filtering of the modelled complexes, which Hsu et al. 2025 found to matter, and architectures that are pairwise by design rather than by a head. The ligand-only parity on held-out series and the 15-PGDH result point at generalisation across targets, and the arms here changed training signal and heads, not that.

## Infrastructure left in place

Everything below runs on Isambard-AI from `/scratch/u6xq/hussain.u6xq/aevplig-paired` (code mirrored at `~/isambard-ssh/aevplig-paired` and committed under `paired/` on the `paired-data` branch of the fork), reads the parent tree's archives and graph pickles read-only, and keeps datasets as single files.

| Piece | What it gives | Where |
| --- | --- | --- |
| Dataset builder and sampler | row ids, series and ligand keys, series × ligand batches, held-out series, target-group-disjoint holdout, augmented rows by kind | `src/build_paired_dataset.py`, `src/paired_data.py`, `src/pocket_groups.py` |
| Training script | the paper's loop with per-kind loss roles, pair losses, hinge by kind, Siamese / pair / moment heads, moments as input, ligand-only switch, parent-aware sampler | `src/train_paired.py`, `src/heads.py`, `src/pair_losses.py` |
| Evaluator | FEP benchmark (paper-identical featurisation), CASF, held-out series and groups, pose and augmentation discrimination, pairwise ranking by head, moments R², data-dir and scaler guards | `src/eval_paired.py`, `src/compare_arms.py` |
| 3D Zernike descriptors | order-12 invariants of 7 property channels for pocket and ligand, 96,947 training complexes plus FEP and 15-PGDH, validated against ZMPY3D to 1e-11; target recall@1 0.75 | `src/zernike3d.py`, `src/pocket_fields.py`, `src/featurise_zernike.py`, `work/data/zernike*.npz` |
| Generators | pose decoys and near-natives with soft-repulsion relief; shape twins by descriptor selection and overlay; MMP-style in-place edits with site classification; all with cue AUROCs in their summaries | `src/pose_augment.py`, `src/twin_select.py`, `src/twin_place.py`, `src/ligand_edits.py`, `src/featurise_*.py` |
| External test | Rich et al. 2026 15-PGDH set prepared from the released repository, reproducing the paper's predictions exactly, with per-series Spearman beside Vina and FEP+ | `src/pgdh_prepare.py`, `work/output/eval/pgdh-*` |
| Trained models | base-g, S, PM and their ligand-only gates at 10 seeds; every other arm at 3 seeds; the paper reproduction at 10 seeds in the parent tree | `work/output/trained_models/` |

Two results are reusable on their own. The ligand-graph-plus-invariants model (`Zb-lig`) is a pose-free family at 0.893 CASF and 0.569 unseen-target Pearson that needs no protein graph at inference beyond the 343 pocket invariants. And the target-group assignment (4,014 groups; BACE1 the largest at 4.4% of rows) with a per-row distance to the nearest training pocket gives the reporting axis the 15-PGDH paper asks for.

## Recommendations and open questions

1. Keep the vanilla recipe as the production model; nothing tested beats it on FEP ranking, and the 10-member ensemble reproduces the paper.
2. Report AEV-PLIG results on a target-group-disjoint split and with the nearest-training-pocket distance per target. Both are built; the paper's own test split shares a target with training for 89% of rows and 109 of the 277 CASF crystals are BindingNet templates.
3. Screen future changes on the held-out series and groups (paired sd about 0.01) and confirm on FEP at ten seeds only. Three-seed FEP screens produced two false positives in this programme.
4. Treat negatives of any construction as a cost to absolute prediction unless an arm shows otherwise with a matched control; four constructions here did not.
5. The levers not yet pulled are on the protein side and the data side: angular AEV terms and more protein atom types; pocket-level graph features instead of per-atom radial sums; quality filtering of the modelled complexes (Hsu et al. 2025 found the low-confidence BindingNet fraction scales negatively); and a pairwise-by-design architecture evaluated as a ranker. The ligand-only parity on held-out series and the 15-PGDH result both point at generalisation across targets rather than at training signal.

Open questions: whether assay-matched series (ChEMBL assay ids for the BindingNet rows; the ChEMBL 37 dump is on scratch) change the within-series noise floor enough for a Siamese or pair head to show a gain; whether the ligand-plus-invariants family holds its 0.89 CASF on the group-disjoint split at ten seeds (3-seed: 0.569 unseen-target Pearson); and whether the 15-PGDH gap to Vina closes with any protein-side change, which the external test can now answer in minutes per model.

## Appendix: paths, jobs and per-arm reports

| Item | Location |
| --- | --- |
| Pinned AEV-PLIG clone (oxpig release 3ccac4b, untouched) | `/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b` |
| Environment (uv, Python 3.11, torch 2.6.0+cu126, PyG 2.5.3, torchani 2.2.4) | `/projects/u6xq/hussain/opig-harness/envs/aevplig` |
| Vanilla retrain tree (parent session): archives, pickles, 10-seed retrain | `/scratch/u6xq/hussain.u6xq/aevplig` (`raw/SOURCES.md` lists every archive with checksums) |
| Paired-data tree: code, datasets, models, evaluations | `/scratch/u6xq/hussain.u6xq/aevplig-paired` |
| Local mirror and git branch | `~/isambard-ssh/aevplig-paired`; fork branch `paired-data`, directory `paired/` |
| Data census and design notes | `paired/docs/aevplig_paired_data_opportunities.md`, `paired/docs/zernike_soft_negatives_design.md`, `paired/docs/pocket_descriptor_analysis.md` |
| Per-arm reports | `paired/reports/REPORT.md` (arm 1), `REPORT_arm4.md`, `REPORT_softneg.md` (A, C), `REPORT_siamese.md`, `REPORT_heads.md` (S at 10 seeds, P, P-neg, PM at 10 seeds), `REPORT_moments_input.md` (Z); tables in the matching `*_arms.md` files |
| External sources | Ross et al. 2023 benchmark repository; Rich, Dodda & Price 2026 repository; BindingNet v1 (Zenodo 11069450); BindingDB-DCS; PDBbind v2020; ChEMBL 37 SQLite (all as single archives under `aevplig/raw`) |

| Run | Jobs |
| --- | --- |
| Vanilla retrain, 10 seeds + eval | 8073760, 8073761 |
| Arm 1 (base, dh03, dh10, ligand-only gates) | 8075982, 8075953, 8075954, 8075983, 8075956; evals 8075997, 8075958, 8075959, 8075998, 8075961 |
| Arm 4 (featurise, build, nda, pdanda) | 8079261, 8080092, 8080229, 8080230; evals 8080238, 8080241, 8080242 |
| Arms A and C (featurise, build, base-g, A, C, AC, gates) | 8097433, 8097329, 8104500, 8104554–8104560; evals 8104578–8104584 |
| Arm S (3 + 7 seeds), base-g seeds 3–9 | 8113167, 8113169, 8114834, 8114833; evals 8113170–8113173, 8116390, 8116412 |
| Arm P (M, P, PM, P1, PMpos, PMneg, gates), PM/PM-lig/base-g-lig seeds 3–9 | 8114863–8114870; 8120835–8120837; evals 8114888–8114895, 8120838–8120840 |
| Arm Z (descriptors for eval sets, seven configurations) | 8161348, 8161546–8161552; evals 8161553–8161559, 8162924 |
| Descriptors, groups, side file | 8079736, 8091845, 8091846, 8114832 |
| 15-PGDH preparation and reproduction check | 8079028, 8110456 |
