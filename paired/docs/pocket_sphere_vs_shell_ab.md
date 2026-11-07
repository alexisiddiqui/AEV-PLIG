# Pocket 3DZD A/B: 10 Å sphere vs 8 Å shell

## Verdict

**The 10 Å sphere (R 10) is worse than the 8 Å shell (R 12). Keep the shell.**

- **Target identification drops for every pocket variant.**
  - F+G std cosine: R@1 0.746 → 0.694. Paired difference −0.052 (UniProt-bootstrap CI −0.060 to −0.046).
  - Signed-sqrt variant: 0.756 → 0.704.
  - The ligand control is unchanged (0.456).
- **Cropping was not the shell's problem.** The sphere wins only for ligands under 16 heavy atoms (+0.009); on units the shell barely cropped, R@1 still falls (0.773 → 0.746).
- **The ligand-size dependence gets steeper.** For ligands of 70 or more atoms, R@1 falls 0.543 → 0.438. A big ligand fills the sphere, so the protein atom count still encodes ligand size, inverted (Spearman −0.57 against heavy atoms).
- **The ligand imprint falls, but target signal falls faster.**
  - Within a UniProt, the correlation of pocket distance with ligand-F distance falls 0.61 → 0.47, and with heavy-atom difference 0.54 → 0.43.
  - Same-target distances get noisier (median same-UniProt pair 0.66 → 0.72).
  - R@1 with the 20 ligand-nearest units excluded falls 0.587 → 0.524.
- **Separation is worse.** An other-group pocket is nearer than any same-group pocket for 26.9% of queries, against 21.5% (signed-sqrt). Other/same median distance ratio: 1.50 against 1.76.
- **More failures.** The sphere fails 4 more rows and gives 348 PDBbind pockets under 50 atoms (shell: 12).

**Defaults:** use the shell with F std + signed-sqrt G std, cosine, no PCA (the `pocket_groups.py` descriptor) for both:
- (a) nearest-other-target negatives;
- (b) the `nn_train_dist` report.

A 12 Å sphere was not tested.

**Checks:**
- The shell rerun is bit-identical to the earlier `zernike.npz` on all 96,947 ok rows (`pocket_F` max abs difference 0).
- The 3 failed rows are NaN in every descriptor array, with −1 in every count.

---

## 1. Jobs

Both jobs reached COMPLETED (exit 0):
- **8091845 `ap-zern-shell`:** 2 min 44 s.
- **8091846 `ap-zern-sphere`:** 2 min 36 s.

Extraction took about 1 minute and featurisation about 27 s. The logs have no errors.

Local copies are in `/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/zernike_ab/raw/`:
- **Descriptors:** `zernike.npz` (978,228,774 B) and `zernike_sphere10.npz` (959,902,784 B). Sizes match Isambard and the zip CRCs pass.
- **Summaries and groups:** both `*_summary.json`, both pocket-groups csv/json pairs.
- **Logs:** both job logs.

The meta file is the same one in both runs (Isambard `paired_meta.csv`, 12,545,936 B, the same size as the local copy).

## 2. `*_summary.json` top level

| | shell (`zernike_summary.json`) | sphere (`zernike_sphere10_summary.json`) |
|---|---|---|
| rows / ok / failed | 96,950 / 96,947 / 3 | 96,950 / 96,943 / 7 |
| pocket_mode, pocket_radius | shell, null | sphere, 10.0 |
| radius / cutoff / ligand_radius | 12 / 8 / 12 | 10 / null / 12 |
| variant_note | "shell8-r12 rerun with NaN-on-failure and signed totals" | "sphere r10 A/B" |
| failures | 2r1w, 2IJ7_TPF_bak_…, 2BUC_008_… (no atom within 8 Å of the ligand) | same 3, plus 6e83, 2w9r, 3kl8, 2OZ5_7XY_2OZ5-results_50341981 (no atom within 10 Å of the centroid) |

**n_pocket_atoms** (q00 / q05 / q25 / q50 / q75 / q95 / q100):

| Source | shell | sphere |
|---|---|---|
| PDBbind | 21 / 143 / 194 / 230 / 269 / 338 / 660 | 2 / 69 / 124 / 152 / 173 / 199 / 240 |
| BindingNet | 60 / 179 / 222 / 246 / 276 / 328 / 493 | 24 / 103 / 133 / 152 / 169 / 189 / 236 |
| BindingDB | 11 / 158 / 208 / 235 / 264 / 324 / 417 | 41 / 105 / 141 / 161 / 175 / 193 / 222 |

**n_outside_pocket** (q50 / q95 / q100, mean):

| Source | shell | sphere |
|---|---|---|
| PDBbind | 26 / 133 / 501, mean 40.3 | 0 everywhere |
| BindingNet | 36 / 98 / 282, mean 41.0 | 0 everywhere |
| BindingDB | 23 / 94 / 218, mean 31.1 | 0 everywhere |

**Ligand counts are identical in both runs.**
- PDBbind ligand heavy atoms: median 28, q95 81.
- Ligands with atoms beyond 12 Å: mean outside 1.06.

**The new sphere failures are very large ligands, or a ligand barely touching the protein.**

| Row | Shell pocket atoms | Of which beyond R | Ligand heavy atoms |
|---|---|---|---|
| 6e83 | 133 | 120 | 91 |
| 2w9r | 146 | 143 | 87 |
| 3kl8 | 79 | 74 | 136 |
| 2OZ5 | 11 | — | 39 |

## 3. Pocket-groups summaries

**The groups are identical in both runs.** Every row has the same `group_id`:
- 4,014 groups: 3,314 with a UniProt, 700 single.
- 382 groups with two or more UniProts, covering 931 UniProts.
- 49 generic names skipped.
- Largest group: uniprot:P56817 (BACE1), 4,263 rows. That is 4.4% of rows and 2.8% of units.

Only the distances differ. The descriptor is `pocket_groups.py`'s: standardised [F | signed-sqrt G], cosine.

| | shell | sphere |
|---|---|---|
| rows with a descriptor | 96,947 | 96,943 |
| fraction of rows whose nearest train pocket is in another group | **0.067** | **0.082** |
| nn_train_dist q05 / q25 / q50 / q75 / q95 (all rows) | 0.237 / 0.421 / 0.529 / 0.631 / 0.731 | 0.288 / 0.455 / 0.552 / 0.640 / 0.730 |
| valid: other-group fraction, median nn_train_dist | 0.238, 0.501 | 0.283, 0.545 |
| test: other-group fraction, median nn_train_dist | 0.069, 0.468 | 0.087, 0.553 |
| series_valid: other-group fraction, median nn_train_dist | 0.137, 0.531 | 0.182, 0.551 |
| train: other-group fraction | 0.062 | 0.076 |

Per-row `nn_train_dist` Spearman, shell vs sphere: train 0.74, valid 0.70, test 0.76, series_valid 0.83.

## 4. Same-UniProt nearest-neighbour recall (PDBbind units with a UniProt)

**Setup.** The existing scripts (`prep.py`, `nn_recall.py`, `nn_recall2.py extra`, `pocket_def.py`) ran unedited on each run's own usable set:
- **shell:** N 18,828, 16,976 queries.
- **sphere:** N 18,825, 16,972 queries (2w9r, 3kl8 and 6e83 dropped out).

**Chance:** R@1 0.0027, R@5 0.0134, R@1 diff-lig 0.0028, R@1 macro 0.0004 (the same in both runs).

**The shell rerun reproduces the earlier `nn_recall.csv` exactly.**

| Variant | Metric | R@1 shell | R@1 sphere | R@5 shell | R@5 sphere | R@1 diff-lig shell | R@1 diff-lig sphere | R@1 macro shell | R@1 macro sphere |
|---|---|---|---|---|---|---|---|---|---|
| F std | euc | 0.732 | 0.666 | 0.827 | 0.778 | 0.715 | 0.649 | 0.566 | 0.496 |
| F+G std, no PCA | cos | 0.746 | 0.694 | 0.839 | 0.797 | 0.732 | 0.678 | 0.588 | 0.526 |
| F std + signed-sqrt G std | cos | **0.756** | **0.704** | 0.845 | 0.805 | 0.743 | 0.687 | 0.599 | 0.536 |
| occupancy only (raw) | cos | 0.631 | 0.559 | 0.739 | 0.676 | 0.608 | 0.533 | 0.461 | 0.391 |
| chemistry only (std) | euc | 0.721 | 0.650 | 0.820 | 0.763 | 0.704 | 0.633 | 0.554 | 0.480 |
| ligand F std (control) | cos | 0.456 | 0.456 | 0.590 | 0.590 | 0.421 | 0.421 | 0.320 | 0.321 |

Metrics follow the earlier report's choice. For the sphere, occupancy works marginally better under euclidean (0.560). The full grids are in `{shell,sphere}/nn_recall*.csv`.

**Paired sphere − shell R@1** over the 16,972 common queries (UniProt bootstrap, 1,000 draws):

| Variant | Difference | 95% CI | Gained / lost |
|---|---|---|---|
| F std | −0.066 | −0.075 to −0.058 | 691 / 1,812 |
| F+G std | −0.052 | −0.060 to −0.046 | 617 / 1,508 |
| signed-sqrt | −0.052 | −0.059 to −0.046 | 574 / 1,462 |
| occupancy | −0.072 | −0.080 to −0.064 | — |
| chemistry | −0.071 | −0.081 to −0.063 | — |
| ligand | 0 | — | — |

### Breakdowns (paired common queries; F+G std cosine / signed-sqrt cosine)

**By ligand heavy atoms:**

| Heavy atoms | Queries | F+G shell | F+G sphere | ssqrt shell | ssqrt sphere |
|---|---|---|---|---|---|
| < 16 | 2,097 | 0.754 | 0.763 | 0.765 | 0.773 |
| 16–24 | 3,844 | 0.784 | 0.736 | 0.790 | 0.744 |
| 24–32 | 4,704 | 0.786 | 0.743 | 0.795 | 0.752 |
| 32–40 | 2,932 | 0.759 | 0.685 | 0.769 | 0.699 |
| 40–50 | 1,461 | 0.693 | 0.619 | 0.702 | 0.628 |
| 50–70 | 892 | 0.646 | 0.548 | 0.666 | 0.554 |
| ≥ 70 | 1,042 | 0.543 | 0.438 | 0.556 | 0.450 |

**By the shell's fraction of pocket atoms beyond R = 12 Å** (the same units in both columns):

| Fraction beyond R | Queries | F+G shell | F+G sphere | ssqrt shell | ssqrt sphere |
|---|---|---|---|---|---|
| < 0.05 | 5,278 | 0.773 | 0.746 | 0.781 | 0.753 |
| 0.05–0.1 | 2,546 | 0.771 | 0.724 | 0.780 | 0.734 |
| 0.1–0.2 | 4,567 | 0.766 | 0.704 | 0.775 | 0.713 |
| 0.2–0.3 | 2,293 | 0.748 | 0.689 | 0.756 | 0.701 |
| 0.3–0.5 | 1,654 | 0.641 | 0.555 | 0.661 | 0.569 |
| ≥ 0.5 | 634 | 0.557 | 0.454 | 0.566 | 0.470 |

**The sphere's own fraction beyond R = 10 Å is 0 for every unit** (max 0). It falls in one stratum: 16,972 queries, R@1 0.694 (F+G) and 0.704 (ssqrt).

**By the sphere's pocket atom count** (sparse spheres hurt both runs, the sphere more):

| Sphere pocket atoms | Queries | F+G shell | F+G sphere |
|---|---|---|---|
| < 50 | 274 | 0.401 | 0.292 |
| 50–100 | 1,730 | 0.579 | 0.468 |
| 100–125 | 2,017 | 0.736 | 0.658 |
| 125–150 | 3,896 | 0.789 | 0.738 |
| 150–175 | 5,110 | 0.751 | 0.704 |
| ≥ 175 | 3,945 | 0.802 | 0.784 |

**By UniProt size** (F+G, sphere, each run's own set): R@1 is lower in every stratum.
- Fewer than 3 units: 0.441 → 0.378.
- 6–20 units: 0.729 → 0.672.
- 21–100 units: 0.833 → 0.783.
- 301 or more units: 0.968 → 0.961.

## 5. Ligand dependence (`pocket_def.py`, F+G std cosine, as in the earlier analysis)

| | shell | sphere |
|---|---|---|
| Spearman(n_pocket_atoms, n_ligand_atoms) | 0.536 | **−0.573** |
| Same-UniProt pairs: Spearman(pocket dist, ligand-F dist) | 0.607 | 0.474 |
| Same-UniProt pairs: Spearman(pocket dist, \|Δ heavy atoms\|) | 0.543 | 0.432 |
| Different-UniProt pairs: the same two correlations | 0.117 / 0.137 | 0.040 / 0.052 |
| Median pocket distance, same / different UniProt | 0.661 / 1.000 | 0.719 / 1.004 |
| Linear R² of pocket vector on ligand F | 0.134 | 0.092 |
| R@1 after regressing out ligand F | 0.746 | 0.689 |
| R@1 excluding same `lig_key` and the 20 ligand-nearest units | 0.587 | 0.524 |
| Rank-1 misses: median ligand-F distance to the wrong neighbour (random 24.8) | 16.9 | 19.7 |
| Rank-1 misses: median \|Δ heavy atoms\| (random 13) | 4 | 6 |
| Ligand atoms beyond 12 Å: R@1 with / without | 0.557 / 0.761 | 0.462 / 0.712 |

**The ligand imprint is smaller, but the descriptor does not depend less on the ligand.**
- The sphere's misses are less ligand-matched.
- Its within-target correlations with ligand distance are lower.
- But within-target distances are larger, and every ligand-controlled recall is lower.
- Sphere protein atom count by ligand size (median, q05):

| Ligand heavy atoms | Median | q05 |
|---|---|---|
| < 16 | 181 | 118 |
| 16–32 | 159 | 107 |
| 32–50 | 140 | 82 |
| 50–70 | 101 | 57 |
| ≥ 70 | 73 | 32 |

  The ligand displaces protein from the fixed sphere, so ligand size enters through the hole.
- The size-only recall check is undefined for the sphere: its constant "fraction outside" column gives NaN.

## 6. Nearest same-group vs other-group pocket

**Setup:**
- **Units:** the PDBbind cache set.
- **Groups:** `group_id` from each run's `pocket_groups.csv`.
- **Queries:** units in a group with at least 2 units in the set (17,331 shell, 17,328 sphere; 3,314 / 3,313 groups).

These are `pocket_groups.py` groups (UniProt plus non-generic name merge), not the earlier ad hoc name merge. So 0.225 here is not comparable with the earlier 0.198.

| | shell F+G | sphere F+G | shell ssqrt | sphere ssqrt |
|---|---|---|---|---|
| Nearest other-group pocket nearer than any same-group | **0.225** | **0.278** | **0.215** | **0.269** |
| NN of every unit in another group | 0.287 | 0.336 | 0.278 | 0.327 |
| Nearest same-group distance q10 / 25 / 50 / 75 / 90 | 0.070 / 0.145 / 0.273 / 0.451 / 0.639 | 0.123 / 0.216 / 0.351 / 0.530 / 0.696 | 0.091 / 0.175 / 0.310 / 0.488 / 0.674 | 0.150 / 0.249 / 0.388 / 0.558 / 0.717 |
| Nearest other-group distance q10 / 25 / 50 / 75 / 90 | 0.227 / 0.356 / 0.501 / 0.620 / 0.691 | 0.263 / 0.423 / 0.551 / 0.643 / 0.700 | 0.278 / 0.416 / 0.547 / 0.656 / 0.721 | 0.321 / 0.467 / 0.583 / 0.673 / 0.728 |
| Median other / median same | 1.84 | 1.57 | 1.76 | 1.50 |

Here signed-sqrt G is standardised over the PDBbind cache (`zk_variants.build_extra`). `pocket_groups.py` standardises over all usable rows; the difference is small.

## 7. Reproduction and failure checks (`repro_check.log`)

**The shell rerun reproduces the earlier run exactly.**
- Row order is identical across the old, shell and sphere files, and `ok` status is identical between old and shell.
- On the 96,947 rows ok in both, `pocket_F` max abs difference is 0.000. The same holds for `pocket_G`, `ligand_F` and `ligand_G`, with every row bit-identical.
- All four count columns match.
- The new files add `pocket_total` / `ligand_total` (the signed totals) and 0-d parameter keys.

**Failed rows:**
- **Earlier copy:** the 3 failures held zeros and `n_pocket_atoms` 0.
- **Shell rerun:** the same 3 rows are all-NaN in `pocket_F`, `pocket_G`, `ligand_F` and `ligand_G`, with −1 in all four counts (`n_ligand_atoms` as well).
- **Sphere:** the 7 failures are encoded the same way.
- No ok row has a non-finite value in either run.

**Shell vs sphere:**
- `ligand_F` and `ligand_G` max abs difference is 0, and the ligand counts are identical.
- Sphere `n_outside_pocket` is 0 on every row.
- Occupancy F00/C00 matches the inside-atom count to 3e-5 (shell) and 2e-5 (sphere).

## Files

Everything is under `/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/zernike_ab/`:
- **This report:** `sphere_vs_shell.md`.
- **Copied outputs:** `raw/` (npz, json, csv and logs).
- **Per-run analysis:** `shell/` and `sphere/`, each holding:
  - `data/`: symlinks to the npz and `paired_meta.csv`;
  - `cache/`;
  - `knn/`;
  - `nn_recall.csv`, `nn_recall_extra.csv`;
  - `prep.log`, `nn_recall*.log`, `pocket_def.log`.
- **New scripts:** `scripts/ab_compare.py` (paired recall, breakdowns, group distances) and `scripts/repro_check.py`.
- **Their output:** `ab_compare.log`, `repro_check.log`, `failures_and_counts.log`.

The original scripts in `zernike_analysis/scripts/` were run unedited. Nothing in the project tree was changed.
