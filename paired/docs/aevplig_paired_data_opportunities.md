# Paired-data opportunities for AEV-PLIG from the data we already have

Date: 2026-11-04. Inputs: the GQT pairing code in the jax-Ka `_submission` snapshots on Isambard, a census of the AEV-PLIG training tables (oxpig/AEV-PLIG @ 3ccac4b plus the PDBbind v2020 index), and a literature pass. Supporting reports (scratchpad, this session):
`.../scratchpad/gqt_pairing_inventory.md`, `.../scratchpad/census/aevplig_pairing_census.md` (+ `results/census_tables.md`, scripts), `.../scratchpad/pl_pairing_literature.md`.
Numbers marked ✓ were recomputed independently from the raw CSVs; the rest are from the census scripts.

## 0. Bottom line

1. The strongest paired signal is already in the training set and needs no new data: **same-receptor congeneric series**. BindingNet puts 67.7k rows into 4,352 series that share one receptor file and one modelling pipeline (2.2M pairs, 301k with |ΔpK| ≥ 2); 63,054 of those rows ✓ sit on 5,384 PDBbind-train crystals ✓, which is exactly the shape of the FEP benchmarks (one crystal, congeners modelled onto it). This is the AEV-PLIG analogue of GQT's within-complex listwise negatives: the partners come from the same example, so family, source and construction are matched by construction.
2. The literature is consistent on one point: a **pairwise/listwise term on top of the pointwise MSE** is the one paired loss with repeatable structure-based gains (PBCNet 0.559→0.636 / 0.372→0.513 Spearman on FEP1/FEP2; McNutt +0.05 Pearson, collapsing to −0.06 without the pointwise anchor; Boltz-2 weights the intra-assay difference term above the absolute term). Everything else (decoys, soft negatives, geometric invariance) is either flat or harmful for congeneric ranking unless paired with a matched positive construction.
3. **Hard negatives we can use now are measured, not synthetic**: the weak members of each series, and 12k same-ligand/different-protein (selectivity) pairs that a ligand-only model cannot order. **Soft negatives need a docking generator** we do not have in the pipeline, and both the GQT in-batch-decoy result and PIGNet2 (negative-only augmentation: FEP mean R 0.50→0.39, CDK2 −0.29) say they would hurt the objective we care about. True measured non-binders are nearly absent (202 PDBbind ">" rows ✓-class; none in BindingNet/BindingDB because the authors dropped them), so a Negatome-style line is parked for the same reason it was parked in GQT.
4. **Label-free consistency pairs exist for free**: 5,080 ✓ BindingDB pairs where one measurement was docked into two different crystals of the same protein (5,036 ✓ identical labels). That is the ANM/minimised-structure analogue, with the same prior: GQT ANM consistency was a wash, McNutt's rotation-consistency term did nothing, and PIGNet2's near-native positives only mattered as the counterpart to decoys. Cheap to test, low expected yield on its own.
5. **Hygiene first.** The valid split is random by complex (89% of valid rows share a UniProt with train; only 25 groups with ≥ 5 ligands), 109 of the 277 CASF test crystals are BindingNet templates for 1,044 training rows ✓, and the FEP targets keep 27k training rows (28.5% of train) on the same proteins. Any pairwise gain must be read against a ligand-only control and on a series-held-out validation, or it will be target memorisation.

## 1. What the data already pairs (train rows after the ligsim90 filter: 95,680)

| Pair type | Where | Groups ≥ 2 / rows | Pairs (|Δ| ≥ 1 / ≥ 2 / ≥ 3) | Construction matched? | Use |
|---|---|---|---|---|---|
| Same receptor file, different ligands | BindingNet by template (`rec_h_opt.pdb`) | 4,352 / 67,674 | 2.22M (909k / 301k / 78k) | yes: one receptor, one modelling pipeline | ranking / Δ loss (core) |
| | BindingDB by folder (one crystal + ≤ 19 docked ligands) | 969 / 8,237 | 41k (14k / 4.4k / 1.2k) | yes: one crystal, Surflex both sides | ranking / Δ loss |
| | PDBbind by UniProt (different crystals, mutants merged) | 1,941 / 15,744 | 375k (229k / 125k / 63k) | source-matched (crystal), receptor differs | ranking / Δ loss, lower weight |
| Crystal-anchored series | PDBbind-train crystal + its BindingNet congeners | 5,384 ✓ anchors / 63,054 ✓ rows | 25.8k / 9.0k / 2.6k vs the anchor | mixed: crystal pose vs modelled pose | anchor as reference; gate the mixed pairs |
| Cross-source same protein | PDBbind crystal vs BindingNet/BindingDB modelled | — | 4.5M + 0.8M + 4.4M | **no** (crystal vs modelled, different H treatment) | exclude from losses; construction-cue gate |
| Same ligand, different proteins (selectivity) | PDBbind HET × UniProt (peptides, cofactors removed) | 1,002 ligands / 3,346 | 6.7k (3.6k / 1.8k / 0.8k) | crystal both sides | hard pairs, ligand-only cannot solve |
| | BindingNet compound × target | 3,197 / 7,064 | 5.4k (2.3k / 0.7k / 0.2k) | modelled both sides, **945 pairs are the same file under two ChEMBL targets** | hard pairs after dropping the 945 |
| | BindingDB monomer × UniProt | 120 / 452 | 455 | docked both sides | small |
| Same ligand + same protein, different structure | BindingDB monomer × UniProt across folders | 1,319 / 3,760 | 5,080 ✓, 5,036 ✓ identical label | yes | label-free consistency |
| | PDBbind replicate entries (same UniProt + HET) | 727 / 1,758 | 1,714, median |Δ| 0.47, 313 identical | crystal both sides | **noise floor**; replicate-spread metric |
| | Crystal pose vs modelled pose of the same compound | suspected (label coincidence 3.8× permutation baseline) | not countable from CSVs | mixed | needs ligand identity from the structure files (compute job) |
| Measured non-binders | PDBbind qualifiers in train rows | ">" 202, "<" 137, "~" 123 | — | — | too few; BindingNet/BindingDB carry none |

Notes. BindingDB `pK` is the median of pKd → pKi → pIC50 across measurements (1,225 rows are medians, 422 with spread ≥ 1). BindingNet has no assay or qualifier column; the template is the only series key available offline. Grouping by UniProt merges mutants and constructs, which is part of why the replicate noise floor is as high as 0.47.

## 2. GQT mechanism → AEV-PLIG analogue

| GQT (snapshot) | What it pairs | AEV-PLIG analogue | Verdict |
|---|---|---|---|
| Within-complex listwise negatives (`--within-weight 0.3`, soft distance targets, same-complex candidates only) | anchor residue vs candidates of the same complex | **within-series ranking of ligands on one receptor file**: series = BindingNet template / BindingDB folder / PDBbind UniProt, never mixing sources inside a series; soft targets from pK; ties below the 0.5 noise band ignored (GQT drops anchors with near-zero targets) | first arm |
| pKa Siamese difference loss (bound − free supervised on the label difference) | two views of one example | **ΔpK regression on same-receptor pairs**: Huber((f_i − f_j) − (y_i − y_j)) with the pointwise term kept (McNutt, PBCNet, Boltz-2). Same sampler as above; listwise vs Δ-Huber is one flag | first arm (variant) |
| Free branch distilled to bound (stop-grad teacher) | partner-removed counterfactual | **ligand-only pass (AEVs zeroed)**. No label exists for it, so it is a gate, not a loss: within-series ranking of the full model must beat the ligand-only model trained with the same losses | gate, mandatory |
| dRSA-weighted targets (`--within-target drsa`) | relevance weighted by a physical signal | **|ΔpK|-weighted pairs** above the noise band (cliff emphasis defined by measured Δ, not by Tanimoto; ACANet mines triplets by activity alone for the same reason) | variant of arm 1 |
| Hotspot mask (one side, p 0.5, PeSTo oracle, masked pass feeds label-free heads) | partial information + invariance | AEV species-channel dropout (16 of the 352 channels per species) or pocket-residue dropout, with distillation from the unmasked pass. Cheap, on the fly, no refeaturisation | speculative, later |
| ANM / minimised consistency (`alt-cons`, alt geometry feeds label-free heads only) | same example, alternate geometry | **BindingDB cross-crystal pairs** (5,080, identical label, same docking program): MSE(f_a − f_b). Later: refeaturised near-native perturbations (< 2 Å) with the same label (PIGNet2 "PDA"). Affinity labels are not computed on the structure, so the label-structure rule is not violated, but wrong poses are unknown, so consistency only inside the near-native basin | arm 3, low expected yield alone |
| Negatome true negatives (matched construction, parked at ≤ 1.7 SE) | database non-partners | censored ">" rows with a one-sided hinge (Tobit). 202 rows. ChEMBL inactives for the same targets would be new data | park |
| In-batch cross-complex decoys (failed: learned family similarity) | batch-mate partner | **ligand-swap / cross-target docked soft negatives**. Needs a docking generator and both sides through it; PIGNet2 shows negative-only augmentation hurts FEP ranking | avoid for the FEP objective |
| Re-typed decoys with margin loss (overfit) | perturbed partner identity | **pose decoys of the same ligand** (> 4 Å, hinge: prediction ≤ label − margin), PDBbind rows only (crystal reference), always with near-native positives from the same generator | arm 4, only after arms 1–3 |
| Construction-matching rule + gate suite (`score_negatives`, `score_matched`, `check_similarity`) | — | same-source/same-pipeline grouping; gates listed in §5 | before arm 1 |

Model-side facts that matter for the mapping (census §F): protein atoms enter only through radial AEVs (Rc 5.1 Å, 16 shifts × 22 species = 352 features per ligand heavy atom, angular terms discarded, hydrogens and HETATM dropped, atoms prefiltered to the ligand box ± 5.2 Å); the ligand graph has no coordinates; pooling is global mean + max. So the Siamese difference is simply f(a) − f(b) through the shared network (no architecture change), the ligand-only counterfactual is "AEV block zeroed", and geometric sensitivity is only to neighbour densities inside 5.1 Å.

## 3. Hard and soft negatives: what is actually available

Define usefulness relative to the known shortcut: on CASF a ligand-only model reaches Pearson 0.75 against 0.81 with the protein (Volkov 2022), a ligand + pocket-composition baseline reaches 0.85 (Durant 2025), and scoring functions keep their correlation when the ligand is docked into the wrong protein. A negative earns its place only if a ligand-only model cannot solve it.

Hard negatives available now (all measured):
- **Weak members of the same series.** 301k same-template pairs with |ΔpK| ≥ 2 (78k ≥ 3), same receptor and pipeline. These are the activity-cliff pairs, but defined by the construction (shared template, MCS-anchored modelling), not by fingerprint similarity. They are ranking targets rather than "negatives", and they are what the FEP metric rewards.
- **Same ligand, different protein with a measurement** (selectivity pairs): ~6.7k PDBbind + ~4.5k BindingNet (after dropping the 945 same-file pairs) + 455 BindingDB. Ligand-only models cannot order them, so they force the AEV path. Small, but construction-clean within each source.
- **Censored ">" rows**: 202. Not a line.

Hard negatives that need generation:
- **Pose decoys of the same ligand** (> 4 Å, one-sided hinge). Evidence that they raise receptor attribution (Scantlebury: ≤ 1.7% → 28–38%) and pose sensitivity (gnina) without affinity cost, provided positives are augmented with near-native poses from the same generator (PIGNet2: 0.39 → 0.64). Only for PDBbind rows, where the reference pose is a crystal; BindingNet/BindingDB rows are already modelled, so "worse than the given pose" is undefined there. Cost: a perturbation/redocking generator plus refeaturisation (AEVs are cheap; k extra graphs per PDBbind row).

Soft negatives:
- **Random or in-batch protein–ligand mismatches, cross-target docking.** Need a docking generator for the negative and, to avoid a construction cue, the same generator for the positives (as in GQT's matched Negatome). PIGNet traded scoring R 0.79 → 0.75 for screening gains; EquiScore found augmentation mattered least for analogue ranking; ConPLex/DUD-E gains are construction-matched. GQT's in-batch decoys learned family similarity. Not for the FEP objective.
- **"Soft" in the label sense**: pairs with |ΔpK| below ~0.5 are ties within replicate noise; treat them as ignored, as GQT ignores near-zero-target anchors.

## 4. Consistency across alternate structures

- **Free pairs**: 5,080 BindingDB cross-crystal pairs (different receptor conformation and docked pose, same label, same docking program). Loss MSE(f_a − f_b) with gradient into both sides (GQT `alt-cons` style, weight ~0.3). A zero-training diagnostic first: run the released 10-model ensemble on both members and compare prediction spread with pointwise error. If the spread is already small, a consistency loss has nothing to fix.
- **PDBbind replicates** (1,714 pairs, median |Δ| 0.47): not a loss (the pointwise MSE already pulls both to their labels); use as a metric (prediction spread vs label spread on replicates).
- **Minimised / predicted structures** (your suggestion): Hsu et al. 2025 (same group) found AEV-PLIG trained on Boltz-1x reproductions indistinguishable from crystal training, and that low-confidence modelled data has a negative scaling trend. So predicted structures are usable as alternates, but the evidence says the value is quality-gated positives, not an invariance term. Place them as the near-native positive side of arm 4 rather than as a standalone consistency arm.
- **Crystal vs modelled pose of the same compound**: likely present (label coincidence 3.8× the permutation baseline on train templates) but unconfirmed. Confirming it needs ligand identity from the structure files (an InChIKey pass inside the featurise job). If confirmed, those pairs are the cleanest "same label, crystal vs modelled" consistency set, and also the cleanest construction-cue gate.

## 5. Hygiene before any arm

1. **Group ids in the dataset.** `GraphDataset` stores only x, edge_index, edge_attr, y. Add unique_id, source, series key (template / folder / UniProt) and label type.
2. **Series-aware batch sampler.** Batch = S series × K ligands (e.g. 16 × 8 = 128), sources mixed across series but never inside one. A random batch of 128 contains only ~46 same-group pairs by accident, so a pair loss on the current loader is mostly empty.
3. **A validation set for within-series ranking.** Current valid: 993 PDBbind rows, 25 groups with ≥ 5 ligands. Hold out whole series (≈ 5% of BindingNet templates and BindingDB folders) for a within-series Spearman/Kendall used for checkpoint selection and arm comparison; keep the PDBbind valid for the pointwise metric. Do not select on the FEP sets.
4. **Leakage reporting.** 109/277 CASF test crystals are BindingNet templates (1,044 train rows) and the FEP targets keep 27k train rows on the same proteins (CDK2 1,697, CHK1 1,256, PTP1B 1,023, renin 1,281, FXa 1,667). McNutt & Koes found > 95% of the CDK2/PTP1B FEP ligands in BindingDB. Report every FEP result per target with the count of train rows on that UniProt.
5. **Gates** (the AEV-PLIG version of `score_negatives` / `check_similarity`): ligand-only model trained with the same losses; pocket-composition baseline; replicate-spread metric; the BindingDB cross-crystal pairs as a "right protein, other conformation" invariance check; the 0 Ligand Bias set; 3 seeds and the simplicity rule.
6. **Noise band.** Ties at |ΔpK| < 0.5 for pair terms. Label-type mixing (IC50 vs Ki/Kd) cancels within a series only when both come from the same assay; the BindingNet CSV has no assay id, so the template is the best key offline. ChEMBL assay ids for BindingNet rows would be a small, high-value addition (Boltz-2 and LigUnity group by assay; Landrum & Riniker: cross-assay repeats differ by > 0.3 log in 48% of cases).

## 6. Proposed arm order

| Arm | Change | Data | Extra cost | Expectation / evidence |
|---|---|---|---|---|
| 0 | Vanilla retrain, 10 seeds (running) + hygiene items 1–5 | — | sampler + ids + held-out series | baseline and per-seed noise on the FEP sets |
| 1 | Within-series Δ-Huber (and/or listwise) at weights 0.3 / 1.0, pointwise MSE kept, ties < 0.5 ignored | BindingNet-by-template, BindingDB-by-folder, PDBbind-by-UniProt, same source per series | none (pairs form inside the batch) | FEP Spearman +0.05 to +0.14 in the literature; must beat the ligand-only arm by more than seed noise |
| 2 | |Δ|-weighting and the selectivity pairs added to the pair set | +12k same-ligand/different-protein pairs | none | shortcut-resistant; small |
| 3 | Cross-crystal consistency on the 5,080 BindingDB pairs | existing rows | none | probably a wash (ANM, McNutt); one seed, drop if flat |
| 4 | PDBbind-only pose decoys (> 4 Å, hinge) + near-native positives (< 2 Å, same label), one generator for both | generated | featurisation of k decoys per PDBbind row | pose sensitivity and receptor attribution; FEP effect unclear |
| parked | censored negatives, cross-target docked soft negatives, similarity-defined cliff pairs, AEV channel dropout | | | |

## 7. Questions and what would help

- **FEP benchmark structures.** The Schindler/Merck and Wang sets prepared for AEV-PLIG are not in the repo; the `test` split that `create_pytorch_data.py` builds is CASF-2016. Without them the only evaluation is the leaky CASF set. Do you have the prepared FEP complexes (the paper's Zenodo or OPIG copy)?
- **ChEMBL assay ids for BindingNet rows** (or the BindingNet v1 index inside the tarball, to be checked in the featurise job) would turn template series into assay-matched series.
- Papers that would sharpen this: the Boltz-2 appendix (exact affinity loss and censored handling, unverified), PBCNet SI Table 5 (metric for the 0.559 → 0.636 ablation), LigUnity SI (the ranking-term ablation numbers), and Hsu et al. 2025 in full (predicted structures as training data, same group).
