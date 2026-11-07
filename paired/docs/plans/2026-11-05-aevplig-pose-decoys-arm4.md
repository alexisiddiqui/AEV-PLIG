# AEV-PLIG arm 4: pose decoys with near-native positives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether PDBbind-only pose augmentation (near-native positives trained with the experimental label, far decoys trained with a one-sided hinge, both from one generator) improves AEV-PLIG's FEP-benchmark ranking and makes it pose-sensitive, following the PIGNet2 ablation (positives only, decoys only, both).

**Architecture:** Extends the `aevplig-paired` tree (plan `2026-11-05-aevplig-paired-arm1.md`, whose Global Constraints, file formats and conventions all still apply). A new generator makes perturbed ligand poses for every PDBbind complex (train, valid, test), a new featuriser turns them into AEV-PLIG graphs with verbatim copies of the clone's functions (proved identical on crystal poses), the dataset builder appends them as rows carrying `kind` and `parent`, the training script adds a hinge term for decoys and a `--train-kinds` filter, and the evaluator adds pose-sensitivity metrics on the PDBbind valid complexes.

**Tech Stack:** as arm 1 (torch 2.6.0, torch_geometric 2.5.3, rdkit 2024.3.3, numpy 1.26.4, pandas 2.0.3); on Isambard additionally torchani 2.2.4, qcelemental and the clone's `torchani_mod` (NOT in the local venv: AEV code runs only on Isambard).

**Spec:** `/Users/alexi/isambard-ssh/aevplig_paired_data_opportunities.md` §3 ("Hard negatives that need generation") and §6 arm 4; literature: PIGNet2 (near-native < 2 Å positives with MSE, decoys > 4 Å with hinge; negatives alone hurt FEP ranking 0.50 → 0.39, both together 0.64), Gnina/CrossDocked (hinge for poses > 2 Å).

## Global Constraints (in addition to arm 1's)

- Only PDBbind rows are augmented (crystal reference poses). BindingNet/BindingDB rows keep `kind = 0`.
- Augmentation kinds: `0` original row, `1` near-native positive (heavy-atom symmetry-corrected RMSD to the crystal in [0.3, 2.0] Å), `2` decoy (RMSD ≥ 4.0 Å, falling back to ≥ 3.0 Å after 60 failed attempts; recorded). `K_POS = 2`, `K_NEG = 2` per complex. Every generated pose has no clash (minimum ligand-heavy-atom to protein-heavy-atom distance ≥ 2.2 Å, protein atoms = the typed `Target` table the AEV sees) and decoys keep ≥ 50% of the crystal pose's protein contacts (protein heavy atoms within 4.0 Å of any ligand heavy atom).
- Determinism: `rng = numpy.random.default_rng(zlib.crc32(pdb.encode()))` per complex; the pose set is fully reproducible from the archives.
- The clone's featurisation functions (`LoadPDBasDF`, `LoadMolasDF`, `GetMolAEVs_extended`, `atom_features`, `one_of_k_encoding`, `mol_to_graph`, `elements_to_atomicnums`, and the radial constants) are copied VERBATIM from `generate_pdbbind_graphs.py` (lines 13-219 and 228-238) into `src/aevplig_featurise.py` with a header saying so. They are not imported from the clone, whose script has no main guard. Identity on crystal poses is proved on Isambard before any pose graph is used.
- Labels: near-native positives inherit the parent's pK and enter the MSE; decoys inherit the parent's pK only as the hinge reference and never enter the MSE.
- Hinge (standardised units): `relu(pred - (y - margin))` per decoy row, `margin = --decoy-margin (pK, default 0.75) / scaler.scale_[0]`; loss = MSE(kind in train_kinds and kind != 2) + decoy_weight × mean hinge(kind == 2). `--decoy-weight 0` with decoy rows selected is an error (never silently ignore decoys).
- Training walltime 05:00:00 (up to 163k rows per epoch).

## File formats (new)

**`work/data/pdbbind_poses.pickle`**: dict `unique_id -> (c_size, features, edge_index, edge_features)` in the clone's graph-tuple format, with `unique_id = f"{pdb}::pos{k}"` or `f"{pdb}::neg{k}"`, `k` in `0..K-1`.

**`work/data/pdbbind_poses_meta.csv`**: columns `unique_id, pdb, kind (1|2), k, method (nn|flip|conf), rmsd, min_dist, contact_frac, attempts, split (train|valid|test as the parent), relief_steps (relief Monte Carlo steps used, 0 = none needed), n_within_3A (ligand heavy atoms with a protein heavy atom closer than 3.0 Å)`; the last two columns were added after the first build, at the end so readers of the earlier columns are unaffected. One row per generated pose; complexes with fewer than K poses of a kind are listed in **`work/data/pdbbind_poses_summary.json`** (counts per split, per kind, per method, per-kind RMSD/min_dist/contact_frac/attempts quantiles, `relief_steps_quantiles` and `n_within_3A_quantiles` per kind, `decoys_relieved`, `crystal_min_dist_quantiles` and `crystal_n_within_3A_quantiles` of the crystal poses of the processed complexes, `cue_auroc_min_dist` = {`crystal_vs_decoy`, `nn_vs_decoy`, `crystal_vs_nn`} with AUROC = P(min_dist of the first group > the second), ties ½ (0.5 = no close-contact cue, far below 0.5 = reversed cue), failures, identity-check result).

**`paired_meta.csv`** gains columns `kind` (0|1|2) and `parent` (the `row` of the original PDBbind row; `parent == row` for kind 0). **`paired_<split>.pt`** `Data` gains `kind[long,1]` and `parent[long,1]`. New split **`poses_valid`** = every PDBbind `valid` crystal row plus its pos/neg rows (kinds 0,1,2; ≈ 993 × 5 rows). The `train` split holds kinds 0, 1 and 2 (the training script selects with `--train-kinds`). `valid`, `test`, `series_valid` keep kind 0 only.

**Pose-sensitivity metrics** (evaluator, grouped by `parent` over `poses_valid`): `n_parents`, `decoy_rank_acc` (fraction with pred(crystal) > max pred(decoys)), `decoy_gap_mean` (mean of pred(crystal) − mean pred(decoys), pK), `nn_spread` (mean |pred(pos) − pred(crystal)|, pK), `pearson_crystal`, `pearson_pos` (label vs pred over pos rows), `pearson_neg` (label vs pred over decoy rows), written to `pose_summary.csv` (one row) and a `## Pose sensitivity (PDBbind valid)` section in `SUMMARY.md`.

## Review Focus

- A ligand whose rotatable bonds are all in rings or that has ≤ 3 heavy atoms: the near-native generator must still return rigid-body perturbations (no torsion step) and the decoy generator must fall back to the 3.0 Å rule or record failure, never raise. Test in Task A (`test_generator_handles_rigid_small_ligand`).
- A crystal pose that already clashes by the 2.2 Å rule (some PDBbind entries do): the clash threshold for its poses becomes `min(2.2, crystal_min_dist)` so the complex is not silently skipped. Test in Task A (`test_clash_threshold_relaxes_to_crystal`).
- Symmetry: a decoy made by a 180° flip of a symmetric ligand can have RMSD ≈ 0; the RMSD must be symmetry-corrected (`rdMolAlign.CalcRMS`, no alignment). Test in Task A (`test_rmsd_is_symmetry_corrected` on benzene).
- `--train-kinds 0` on the new build must reproduce arm 1's `base` row set exactly (same unique_ids), so arm 1's base results stay valid as the reference. Test in Task B (`test_train_kinds_zero_matches_base_rows`) and the Isambard check in Task C.
- A batch that contains decoys only (possible with the series sampler when a complex's poses chunk together): MSE over zero rows must not be NaN; the loss must fall back to the hinge term alone. Test in Task B (`test_loss_decoy_only_batch`).

---

### Task A: Pose generator and featuriser

**Files:**
- Create: `src/pose_augment.py`, `src/aevplig_featurise.py` (verbatim copies, see constraints), `src/featurise_poses.py`, `jobs/featurise_poses.sbatch`
- Test: `tests/test_pose_augment.py`, `tests/test_featurise_poses.py` (graph assembly with a stub AEV function), `tests/data/` is NOT used (tests build tiny molecules with RDKit in `tmp_path`)

**Interfaces:**
- Consumes: PDBbind archives (read-only) `/scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_refined.tar.gz` (`refined-set/<pdb>/<pdb>_{ligand.mol2,protein.pdb}`) and `PDBbind_v2020_other_PL.tar.gz` (`v2020-other-PL/<pdb>/...`), extracted to `/local` by the job; `work/data/paired_meta.csv` from arm 1 (rows with `source == pdbbind` → pdb id = `unique_id`, `split`, and `refined` from `$PARENT_WORK/data/pdbbind_processed.csv` column `refined`); the parent's `$PARENT_WORK/data/pdbbind.pickle` for the identity check.
- Produces:
  - `pose_augment.generate_poses(mol_h: Chem.Mol, target_xyz: np.ndarray[n,3], k_pos: int, k_neg: int, rng: np.random.Generator, rmsd_pos=(0.3, 2.0), rmsd_neg=4.0, rmsd_neg_fallback=3.0, clash=2.2, contact_keep=0.5, max_attempts=60, *, relief_steps_max=300, relief_step_deg=10.0, relief_step_A=0.5, relief_target_A=3.0) -> list[Pose]` where `Pose = dataclass(kind: int, k: int, method: str, mol: Chem.Mol (new conformer, Hs kept), rmsd: float, min_dist: float, contact_frac: float, attempts: int, relief_steps: int = 0)`.
  - Helpers: `symm_rmsd(ref: Chem.Mol, probe: Chem.Mol) -> float` (heavy atoms, `rdMolAlign.CalcRMS`), `min_dist(lig_xyz, target_xyz) -> float`, `contacts(lig_xyz, target_xyz, cutoff=4.0) -> int`, `rigid_perturb(mol, rng, max_angle_deg, trans_sd) -> Chem.Mol`, `torsion_perturb(mol, rng, sd_deg) -> Chem.Mol`, `random_conformer(mol, rng) -> Chem.Mol` (ETKDG, `randomSeed` from rng, then centroid-aligned to the input), `n_within(lig_xyz, target_xyz, cutoff=3.0) -> int`, `clash_score(lig_xyz, target, r0) -> float` (`sum(max(0, r0 - d_ij)^2)` over each ligand atom's 24 nearest protein atoms; 0 exactly when `min_dist >= r0`), `relieve_clash(mol, target, rng, r0, anchor, max_steps=300, step_deg=10.0, step_A=0.5, max_shift=2.5) -> (Chem.Mol, steps, reached_r0)`.
  - `featurise_poses.main(argv)`: `--pdbbind-root DIR` (holds `refined-set/` and `general-set/`), `--meta work/data/paired_meta.csv --pdbbind-csv $PARENT_WORK/data/pdbbind_processed.csv --out-dir work/data --k-pos 2 --k-neg 2 --workers 72 [--limit N] [--check-identity N --parent-pickle FILE]`; writes the three files in "File formats"; `--check-identity N` re-featurises N crystal complexes (first N by pdb id) through `aevplig_featurise` and asserts `np.array_equal` on all four tuple members against the parent's pickle, printing `IDENTITY OK n=N` or exiting 3 with the first mismatch.
  - Multiprocessing: `multiprocessing.Pool(workers)` over complexes; each worker loads its own protein table; torch threads set to 1 per worker (`torch.set_num_threads(1)`) because `GetMolAEVs_extended` uses torch.

Generator rules (exact): near-native = `rigid_perturb(max_angle_deg=20, trans_sd=0.5)` then `torsion_perturb(sd_deg=15)` on every rotatable bond (`Lipinski.RotatableBondSmarts` pattern, bonds not in rings), then relief (below), accepted when `0.3 <= rmsd <= 2.0`, `min_dist >= clash_eff`, where `clash_eff = min(clash, crystal_min_dist)` (a near-native whose relief pushes RMSD above 2.0 Å is a failed attempt); attempts up to `max_attempts` per pose. Decoys alternate `method = "flip"` (rigid rotation angle `U(90, 180)` degrees about a random axis through the heavy-atom centroid, translation `N(0, 1.0)` per axis) and `"conf"` (random ETKDG conformer, random rotation, centroid at the crystal centroid plus `N(0, 1.0)` translation), then relief, accepted when `rmsd >= 4.0`, `min_dist >= clash_eff`, `contact_frac >= 0.5`; after 60 failures the threshold drops to 3.0 for the remaining attempts (another 60); still failing → no pose, recorded. Poses must be distinct (`symm_rmsd` between accepted poses of the same kind ≥ 0.3 Å).

Relief (soft repulsion, one construction path for kinds 1 and 2; the crystal pose, kind 0, is never moved): every candidate goes through `relieve_clash`, a greedy rigid-body Monte Carlo of up to `relief_steps_max = 300` steps drawn from the complex's `rng` (rotation `U(-10, +10)` degrees about a random axis through the heavy-atom centroid plus a translation of `U(0, 0.5)` Å in a random direction). A step is kept when it lowers `sum(max(0, r0_eff - d_ij)^2)` over ligand-heavy/protein-heavy pairs and keeps the heavy-atom centroid within 2.5 Å of the crystal centroid (a candidate placed further out may not move further out); it stops when `min_dist >= r0_eff` or the budget is spent, and the candidate it ends with then meets the acceptance rules above (unchanged; failing to reach `r0_eff` does not by itself reject). `r0_eff = relief_target_A - (clash - clash_eff)`, `relief_target_A = 3.0` Å: equal to 3.0 Å whenever the crystal pose is clash-free, and lowered by the same amount as the clash threshold when the crystal pose already clashes, so such complexes still get near-natives (Review Focus). Purpose: without relief only ~24% of complexes got 2 decoys (buried pockets); relief to the 2.2 Å limit alone gave 100% but decoys piled up at min_dist ≈ 2.23 Å (crystal-vs-decoy AUROC on min_dist 0.93); the 3.0 Å soft target brings that to ≈ 0.59 on the 184 FEP-benchmark ligands checked locally (`Pose.relief_steps` and the meta/summary cue statistics record it).

- [x] **Step 1: Write failing tests in `tests/test_pose_augment.py`** (RDKit-built ligands: ethanol-like flexible molecule with a 3D conformer, benzene, methane; a synthetic "protein" of ~200 random points on a shell 4-8 Å from the ligand centroid): `test_near_native_rmsd_and_no_clash`, `test_decoy_rmsd_and_contacts`, `test_rmsd_is_symmetry_corrected` (benzene rotated 60° in-plane → RMSD < 0.05), `test_generator_handles_rigid_small_ligand` (methane: pos poses exist, decoys fall back or fail without raising), `test_clash_threshold_relaxes_to_crystal`, `test_deterministic_given_rng` (same crc seed → identical coordinates).
- [x] **Step 2: Run; FAIL. Implement `src/pose_augment.py`. Run; PASS.**
- [x] **Step 3: Create `src/aevplig_featurise.py`** by copying the functions and constants verbatim from `/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/census/data/AEV-PLIG/generate_pdbbind_graphs.py` (the imports `torchani`, `torchani_mod`, `qcelemental` are kept; wrap them in a try/except that sets a flag `AEV_AVAILABLE` so the module imports locally for graph-assembly tests).
- [x] **Step 4: Write failing tests in `tests/test_featurise_poses.py`**: `test_graph_assembly_matches_clone_format` (with a stub `GetMolAEVs_extended` returning zeros of width 352, `mol_to_graph` output has `len(features[0]) == 367` and the clone's tuple shape), `test_unique_ids_and_meta_rows` (a fake `generate_poses` returning two pos and two neg → keys `<pdb>::pos0..1`, `::neg0..1`, meta rows with `split` from paired_meta), `test_limit_and_summary_json`.
- [x] **Step 5: Run; FAIL. Implement `src/featurise_poses.py`. Run; PASS.**
- [x] **Step 6: Write `jobs/featurise_poses.sbatch`** (`--time=03:00:00`; extracts the two PDBbind archives to `/local/user/$(id -u)/aevplig-paired/poses-$SLURM_JOB_ID/` with `tar -I pigz`, links `refined-set` and `v2020-other-PL` → `general-set` under one root, runs `featurise_poses.py --check-identity 200 --parent-pickle $PARENT_WORK/data/pdbbind.pickle` FIRST and aborts on failure, then the full run with `--workers 72`, prints the summary json, EXIT trap cleans `/local`).
- [x] **Step 7: `py_compile`, tick boxes, report decisions left open.**

---

### Task B: Dataset, training and evaluation support for poses

**Files:**
- Modify: `src/build_paired_dataset.py`, `src/paired_data.py` (only to add `kind`/`parent` to `PairedDataset.build`), `src/train_paired.py`, `src/pair_losses.py` (add `decoy_hinge`), `src/eval_paired.py`, `src/compare_arms.py` (add the pose metrics to `METRICS` when `pose_summary.csv` exists), `jobs/train.sbatch` (env `TRAIN_KINDS=0`, `DECOY_WEIGHT=0`, `DECOY_MARGIN=0.75`; `--time=05:00:00`), `jobs/build_dataset.sbatch` (env `POSES=1` adds `--poses-pickle $WORK/data/pdbbind_poses.pickle --poses-meta $WORK/data/pdbbind_poses_meta.csv --out-dir $WORK/data_poses`), `jobs/eval.sbatch` (passes `--pose-pt $DATA_DIR/processed/paired_poses_valid.pt` when the file exists)
- Test: extend `tests/test_build_paired_dataset.py`, `tests/test_pair_losses.py`, `tests/test_train_paired_smoke.py`, `tests/test_eval_paired.py`, `tests/test_compare_arms.py`

**Interfaces:**
- Consumes: the pose pickle/meta formats above (tests fabricate them); arm 1's modules.
- Produces:
  - `build_paired_dataset.main` flags `--poses-pickle FILE --poses-meta FILE` (both or neither). With poses: `paired_meta.csv` gains `kind`, `parent`; pose rows get `unique_id` = pickle key, `source = pdbbind`, the parent's `split` for train rows (`train` → `train`), the parent's `pK`, `series_key/series_id/lig_key/lig_id/label_type/uniprot/anchor_pdb` copied from the parent; pose rows whose parent is `valid` go ONLY to the new split `poses_valid` together with a copy of the parent's crystal row (same `row` id as in `valid`); pose rows whose parent is `test` or `series_valid` are dropped (counted). Without poses: output identical to arm 1 (`kind = 0`, `parent = row`).
  - `pair_losses.decoy_hinge(pred: Tensor[n], y: Tensor[n], kind: LongTensor[n], margin: float) -> Tensor` (mean over `kind == 2` rows of `relu(pred - (y - margin))`; `0.0` when none).
  - `train_paired.py` flags `--train-kinds 0` (comma list; rows of other kinds are excluded from the train split by index selection before the sampler is built), `--decoy-weight 0.0`, `--decoy-margin 0.75`; MSE over `kind != 2` rows of the batch (mean over those rows; if none, `0.0`); error exit when decoys are selected and `--decoy-weight == 0`, or when `--decoy-weight > 0` and kind 2 is not selected. Log line gains `decoy D` (the hinge term). Sidecar json records `train_kinds`, `decoy_weight`, `decoy_margin`, and the number of train rows used.
  - `eval_paired.py --pose-pt FILE` → `pose_summary.csv` + SUMMARY section (metrics above).
  - `compare_arms.py`: adds `("pose_summary.csv", {}, col, label)` rows for `decoy_rank_acc`, `decoy_gap_mean`, `nn_spread`, `pearson_crystal` when the file exists for every tag (NaN otherwise).

- [x] **Step 1: Tests first** (`test_build_with_poses_adds_rows_and_poses_valid`, `test_train_kinds_zero_matches_base_rows` (unique_ids of kind-0 train rows equal those of a build without poses), `test_decoy_hinge_manual`, `test_loss_decoy_only_batch`, `test_train_paired_smoke_with_decoys` (synthetic `.pt` with kinds; `--train-kinds 0,1,2 --decoy-weight 1.0` runs 2 epochs; `--train-kinds 0,2 --decoy-weight 0` exits non-zero), `test_pose_metrics_on_toy`, `test_compare_arms_includes_pose_rows`). Run; FAIL.
- [x] **Step 2: Implement; run the whole suite; PASS** (`.venv/bin/python -m pytest tests -q`).
- [x] **Step 3: Update the three sbatch files; `bash -n`; `py_compile`; tick boxes; report decisions left open.**

---

### Task C: Integration on Isambard (coordinating session + Sonnet monitor)

- [ ] **Step 1:** tar-over-ssh the tree; `sbatch jobs/featurise_poses.sbatch`; require `IDENTITY OK n=200` in its log, then inspect `pdbbind_poses_summary.json` (expect ≈ 19.3k complexes × 2 pos + 2 neg, RMSD quantiles in range, fallback and failure counts).
- [ ] **Step 2:** `POSES=1 sbatch jobs/build_dataset.sbatch` → `work/data_poses/`; check `paired_summary.json` (train kind counts; `poses_valid` ≈ 4,965 rows) and that kind-0 train unique_ids equal arm 1's (`sort | md5sum` of the two meta subsets, in-job or locally).
- [ ] **Step 3:** Arms, 3 seeds each, `DATA_DIR=$WORK/data_poses`: `TAG=pda TRAIN_KINDS=0,1 DECOY_WEIGHT=0`; `TAG=nda TRAIN_KINDS=0,2 DECOY_WEIGHT=1.0`; `TAG=pdanda TRAIN_KINDS=0,1,2 DECOY_WEIGHT=1.0`. Reference = arm 1's `base` (kinds 0; verified identical rows). Evals chained with `--dependency=afterok:<array>` (not on the finished featurise job), `DATA_DIR=$WORK/data_poses`, plus a pose eval of arm 1's `base` and of the released ensemble on `poses_valid` for reference.
- [ ] **Step 4:** `compare_arms.py --tags base,pda,nda,pdanda --ref base` (base's eval re-run with `--pose-pt` so the pose rows exist); write `work/output/eval/REPORT_arm4.md`; update the memory note.


## Outcome (2026-11-05)

Null for FEP ranking (nda −0.049, pdanda −0.017 wPCC per seed), strong pose sensitivity gained; see `work/output/eval/REPORT_arm4.md`. Task C steps 1-4 done (featurise 8079261, build 8080092, arms 8080229/8080230, evals 8080238/8080241/8080242).
