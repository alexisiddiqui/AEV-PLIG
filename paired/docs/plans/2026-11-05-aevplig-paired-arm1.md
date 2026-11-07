# AEV-PLIG paired-data arm 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain AEV-PLIG with a within-series pairwise affinity loss on top of its pointwise MSE, with the data hygiene (series ids, series-aware batches, a held-out-series validation set, a ligand-only shortcut gate) and an FEP-benchmark evaluation that make the comparison against the vanilla model trustworthy.

**Architecture:** A new tree `aevplig-paired/` (local mirror `/Users/alexi/isambard-ssh/aevplig-paired/`, Isambard copy `/scratch/u6xq/hussain.u6xq/aevplig-paired/`) holds `src/` (new Python), `jobs/` (sbatch), `tests/` (pytest, CPU) and `work/` (outputs, Isambard only). It reads the parent tree's graph pickles and CSVs from `/scratch/u6xq/hussain.u6xq/aevplig/work/data/` **read-only** and imports the unmodified AEV-PLIG modules (`utils`, `model_defs`, `helpers`, `process_and_predict`) from the pinned clone `/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b` (`$AEVPLIG_REPO`). Three independent pieces (data layer, training, FEP evaluation) meet only through the file formats fixed below.

**Tech Stack:** Python 3.11, torch 2.6.0, torch_geometric 2.5.3, numpy 1.26.4, pandas 2.0.3, scipy 1.10.1, scikit-learn 1.3.2, rdkit 2024.3.3, biopandas 0.4.1, pytest. Local CPU venv: `/Users/alexi/isambard-ssh/aevplig-paired/.venv` (same versions). Slurm on Isambard-AI (GH200).

**Spec:** `/Users/alexi/isambard-ssh/aevplig_paired_data_opportunities.md` (sections 1, 2, 5, 6) and the census `.../scratchpad/census/aevplig_pairing_census.md`.

## Global Constraints

- Every sbatch job: `--partition=workq --gres=gpu:1 --cpus-per-task=72 --mem-per-cpu=2G --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]`, output to `/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out` (arrays: `%x-%A_%a.out`).
- No Python on the Isambard login node except `python -m py_compile`. Functional checks run in sbatch jobs. Archives are extracted only under `/local/user/$(id -u)/aevplig-paired/<job>/` and removed by an EXIT trap.
- Never write under `/scratch/u6xq/hussain.u6xq/aevplig/` (the parent session owns it). Inputs from there are opened read-only by absolute path.
- The tree stays under 300 files on Isambard (no venv, no `__pycache__`: `PYTHONDONTWRITEBYTECODE=1`).
- `jobs/env.sh` exports `HARNESS_HOME=/projects/u6xq/hussain/opig-harness`, `AEVPLIG_REPO=$HARNESS_HOME/src/AEV-PLIG-3ccac4b`, `PARENT_WORK=/scratch/u6xq/hussain.u6xq/aevplig/work`, `PAIRED_ROOT=/scratch/u6xq/hussain.u6xq/aevplig-paired`, `WORK=$PAIRED_ROOT/work`, `SRC=$PAIRED_ROOT/src`, `PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1`, activates `$HARNESS_HOME/envs/aevplig`, and does `cd $WORK`. Every `src/*.py` starts with `sys.path.insert(0, os.environ["AEVPLIG_REPO"])` so `from utils import ...` resolves to the clone; tests set `AEVPLIG_REPO` to `tests/fake_repo/` which holds a copy of the clone's `utils.py`, `helpers.py`, `model_defs.py` (copy them from `/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/census/data/AEV-PLIG/`).
- Model hyper-parameters are the paper's and are not flags of this work: GATv2Net, hidden 256, 3 heads, leaky_relu, Adam lr 0.00012291937615434127, batch 128, 200 epochs, seeds `[100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]` indexed by `--seed_index`, checkpoint = highest running mean of the last 8 validation Pearson values (as `training.py`).
- Node features: columns `0:15` are RDKit atom features, `15:367` are AEVs (`generate_pdbbind_graphs.py:190`, `np.append(atom_features(atom), aevs[...])`). Edge features: 4.
- Labels are pK. Standardisation: `StandardScaler` fit on the train split's pK (as `GraphDataset`), applied to every split; "tie" and "huber delta" are given in pK and divided by `scaler.scale_[0]` before use.
- No git commits (user rule: commit only when asked). Record progress by ticking the checkboxes in this file.

## File formats (the contract between tasks)

**`work/data/paired_meta.csv`** (one row per complex kept by the original filter `max_tanimoto_fep_benchmark < 0.9`): columns `row` (int, 0-based, unique), `unique_id` (the pickle key), `source` (`pdbbind|bindingnet|bindingdb`), `split` (`train|valid|test|series_valid`), `pK` (float), `series_key` (str), `series_id` (int, dense 0..S-1 over the whole table), `lig_key` (str), `lig_id` (int, dense), `label_type` (`Kd|Ki|IC50|unknown`), `uniprot` (str or empty), `anchor_pdb` (PDBbind code; BindingNet template; BindingDB folder PDB).

**`work/data/processed/paired_<split>.pt`** for `split in {train, valid, test, series_valid}`: a `torch_geometric` InMemoryDataset collate of `Data(x[float, n×367], edge_index[long, 2×e], edge_attr[float, e×4], y[float, 1] (raw pK), row[long, 1], series[long, 1], lig[long, 1], src[long, 1])` with `src` = `{pdbbind:0, bindingnet:1, bindingdb:2}`. Rows appear in `paired_meta.csv` order within each split.

**`work/data/fep_benchmark.csv`**: columns `unique_id` (`<subset>/<series>/<title>`, matching the paper's CSVs), `pK`, `sdf_file`, `pdb_file`, `subset`, `series`, `ref_pdb`, `n_entries` (SDF entries carrying that title). **`work/data/fep_benchmark_graphs.pickle`** (process_and_predict format, keyed by `unique_id`), **`work/data/processed/fep_benchmark.pt`** (`GraphDatasetPredict` with `graph_id` = row order of `work/data/fep_benchmark_processed.csv`).

**Predictions** `work/output/predictions/<tag>_<split>_seed<i>.csv`: columns `row` (or `unique_id` for fep), `truth`, `pred` (pK). **Models** `work/output/trained_models/<tag>_<i>.model` (state_dict), **scaler** `work/output/trained_models/<tag>.pickle`.

## Review Focus

- A series whose members all share one `lig_id` (e.g. a BindingNet template carrying the same compound under two ChEMBL targets): the pair loss must see zero pairs from it and not divide by zero. Test in Task 2 (`test_pair_index_skips_same_ligand`, `test_delta_huber_no_pairs_is_zero`).
- A batch in which no series has two members (possible in the last batch): the total loss must equal the MSE. Test in Task 2 (`test_delta_huber_no_pairs_is_zero`).
- The sampler must emit every train index exactly once per epoch and never a batch of size 1 (BatchNorm1d raises on a single sample in train mode). Test in Task 1 (`test_sampler_covers_every_index_once`, `test_sampler_no_singleton_batch`).
- A PDBbind row without a UniProt (`------`) or with a peptide ligand (`5-mer`) must become a singleton series / singleton ligand, never be merged with other unmapped rows. Test in Task 1 (`test_pdbbind_unmapped_rows_are_singletons`).
- FEP ligand titles that repeat inside one SDF (rotamers/protomers): keep the first entry, record `n_entries`, and the row count must equal the number of distinct titles. Test in Task 3 (`test_fep_prepare_dedupes_titles`).

---

### Task 1: Data layer (ids, series split, dataset, sampler)

**Files:**
- Create: `src/paired_data.py`, `src/build_paired_dataset.py`, `jobs/env.sh`, `jobs/build_dataset.sbatch`
- Test: `tests/test_paired_data.py`, `tests/test_build_paired_dataset.py`, `tests/fake_repo/{utils.py,helpers.py,model_defs.py}`, `tests/data/` (tiny synthetic CSVs/index/pickles written by the tests themselves)

**Interfaces:**
- Consumes: parent inputs `$PARENT_WORK/data/{pdbbind,bindingnet,bindingdb}.pickle` (dict `unique_id -> (c_size, features, edge_index, edge_features)`), `$PARENT_WORK/data/{pdbbind,bindingnet,bindingdb}_processed.csv` (columns as read by the clone's `create_pytorch_data.py`: PDBbind `PDB_code, -logKd/Ki, split_core, max_tanimoto_fep_benchmark`; BindingNet `unique_identify, -logAffi, target, pdb, compnd, max_tanimoto_fep_benchmark`; BindingDB `folder, pdb_file, mol2_file, Ki(nM), IC50(nM), Kd(nM), unique_id, pK, max_tanimoto_fep_benchmark`), and the PDBbind index tarball `/scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_plain_text_index.tar.gz` (members `index/INDEX_general_PL_name.2020`: `pdb year uniprot name`; `index/INDEX_general_PL_data.2020`: `pdb resolution year -logKd/Ki affinity_string // ligand_name_in_parentheses`), read with `tarfile` (no extraction).
- Produces:
  - `paired_data.PairedDataset(root: str, split: str)` (InMemoryDataset; `processed_file_names = [f"paired_{split}.pt"]`; constructed by `build_paired_dataset` via `PairedDataset.build(root, split, rows: pd.DataFrame, graphs: dict)` which writes the file; loading an existing file never rebuilds).
  - `paired_data.SeriesBatchSampler(series: np.ndarray, batch_size: int = 128, chunk: int = 8, seed: int = 0)` with `set_epoch(epoch: int) -> None`, `__iter__ -> Iterator[list[int]]`, `__len__`.
  - `paired_data.pair_index(series: torch.LongTensor, lig: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]` (i < j, same series, different lig).
  - `paired_data.SOURCE_ID = {"pdbbind": 0, "bindingnet": 1, "bindingdb": 2}`.
  - `build_paired_dataset.build_meta(pdbbind: pd.DataFrame, bindingnet: pd.DataFrame, bindingdb: pd.DataFrame, name_index: dict[str, str], data_index: dict[str, tuple[str, str]], holdout_frac: float, holdout_min_size: int, seed: int) -> pd.DataFrame` (the `paired_meta.csv` table) and the CLI `python src/build_paired_dataset.py --parent-data DIR --pdbbind-index-tgz FILE --out-dir DIR [--holdout-frac 0.05] [--holdout-min-size 5] [--seed 0] [--subset-series N]`.

Series and ligand keys (exact rules):
- PDBbind: `series_key = "pdbbind:" + uniprot` when the name index gives a UniProt other than `------`, else `"pdbbind:single:" + pdb`. `lig_key = "pdbbind:" + first whitespace token inside the parentheses of the data-index line`, except tokens matching `^\d+-mer$` or an empty name, which give `"pdbbind:single:" + pdb`. `label_type` from the affinity string prefix (`Kd`, `Ki`, `IC50`). `anchor_pdb = pdb`, `uniprot` as mapped.
- BindingNet: `series_key = "bindingnet:" + pdb` (template), `lig_key = "bindingnet:" + compnd`, `label_type = "unknown"`, `anchor_pdb = pdb`, `uniprot` = the template's UniProt from the name index (empty if absent).
- BindingDB: `series_key = "bindingdb:" + folder`, `lig_key = "bindingdb:" + re.search(r"results_(\d+)\.mol2", mol2_file).group(1)`, `label_type` = first non-empty of `Kd(nM)`, `Ki(nM)`, `IC50(nM)` → `Kd|Ki|IC50`, `anchor_pdb = folder.split("_")[0].lower()`, `uniprot` via the name index.
- `series_id` / `lig_id`: dense integer codes over the whole table (`pd.factorize`, sorted keys).
- Splits: PDBbind rows keep `split_core`; BindingNet/BindingDB rows are `train`; then the holdout: for each of `bindingnet`, `bindingdb` separately, among train series with ≥ `holdout_min_size` rows, draw `round(holdout_frac × n_series)` series with `numpy.random.default_rng(seed).choice(sorted_keys, size, replace=False)` and set all their rows to `series_valid`. `--subset-series N` keeps only the first N series (sorted key) of each source (train) plus all valid/test rows, for the selftest.

- [x] **Step 1: Copy the clone's `utils.py`, `helpers.py`, `model_defs.py` into `tests/fake_repo/`; write `tests/conftest.py` that sets `os.environ["AEVPLIG_REPO"]` to that dir and inserts `src/` on `sys.path`.**
- [x] **Step 2: Write failing tests in `tests/test_paired_data.py`:**
  - `test_sampler_covers_every_index_once`: series array of 1,000 items over 120 series with sizes 1..30; one epoch's batches concatenated == every index once; no batch > 128 except by exactly one appended singleton; `set_epoch(1)` gives a different order.
  - `test_sampler_no_singleton_batch`: no batch has length 1 (a trailing singleton is appended to the previous batch).
  - `test_sampler_chunks_keep_series_together`: with chunk=8, every series with ≥ 8 members has at least one batch containing ≥ 8 of its members.
  - `test_pair_index_skips_same_ligand`: series `[0,0,0,1,1]`, lig `[5,5,6,7,8]` → pairs `{(0,2),(1,2),(3,4)}`.
  - `test_paired_dataset_roundtrip` (skip if `torch_geometric` import fails): build from 3 synthetic graphs, reload, check `x.shape[1] == 367` is NOT required (synthetic width 5 is fine) and `row/series/lig/src/y` round-trip.
- [x] **Step 3: Run the tests; expected: FAIL (ImportError).** Run: `cd /Users/alexi/isambard-ssh/aevplig-paired && .venv/bin/python -m pytest tests/test_paired_data.py -v`
- [x] **Step 4: Implement `src/paired_data.py`.** Sampler algorithm: per epoch `rng = np.random.default_rng(seed * 1_000_003 + epoch)`; for each series, permute its indices and split into consecutive chunks of ≤ `chunk`; shuffle the chunk list; greedily pack chunks into batches of ≤ `batch_size` (a chunk that does not fit closes the batch); if the last batch has one index, append it to the previous batch; `__len__` returns the batch count for the current epoch.
- [x] **Step 5: Run the tests; expected: PASS.**
- [x] **Step 6: Write failing tests in `tests/test_build_paired_dataset.py`** using small synthetic inputs written by the test (3 PDBbind rows: one with UniProt P1 and ligand `ABC`, one `------`, one peptide `5-mer`; 12 BindingNet rows on 2 templates; 6 BindingDB rows in 2 folders): `test_keys_follow_rules`, `test_pdbbind_unmapped_rows_are_singletons`, `test_filter_matches_original` (rows with `max_tanimoto_fep_benchmark >= 0.9` dropped), `test_holdout_moves_whole_series` (with `holdout_frac=0.5, holdout_min_size=5`, every row of a held-out series is `series_valid` and no other BindingNet/BindingDB row is), `test_series_ids_dense`.
- [x] **Step 7: Run; expected: FAIL.**
- [x] **Step 8: Implement `src/build_paired_dataset.py`** (`build_meta`, `read_pdbbind_index(tgz_path) -> (name_index, data_index)`, `main()` that loads the three pickles once, merges, calls `PairedDataset.build` per split, writes `paired_meta.csv` and `paired_summary.json` with rows per split×source, series counts, and the number of same-series different-ligand pairs per split).
- [x] **Step 9: Run all tests; expected: PASS.** Also `.venv/bin/python -m py_compile src/*.py`.
- [x] **Step 10: Write `jobs/env.sh` (see Global Constraints) and `jobs/build_dataset.sbatch`** (`--time=02:00:00`; runs `python $SRC/build_paired_dataset.py --parent-data $PARENT_WORK/data --pdbbind-index-tgz /scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_plain_text_index.tar.gz --out-dir $WORK/data`, then prints `paired_summary.json` and `ls -la $WORK/data/processed`). Also `jobs/build_small.sbatch` doing the same with `--subset-series 40 --out-dir $WORK/data_small`.
- [x] **Step 11: Tick the boxes here and report: file list, test output, and the three decisions you had to make that the plan left open.**

---

### Task 2: Paired training script and gates

**Files:**
- Create: `src/train_paired.py`, `src/pair_losses.py`, `jobs/train.sbatch`, `jobs/selftest.sbatch`
- Test: `tests/test_pair_losses.py`, `tests/test_train_paired_smoke.py`

**Interfaces:**
- Consumes: `paired_data.PairedDataset`, `paired_data.SeriesBatchSampler`, `paired_data.pair_index` (Task 1 signatures above; until Task 1 lands, code against those signatures and build synthetic `.pt` files in the test with the same `Data` fields). `GATv2Net` from the clone's `model_defs`; `pearson`, `spearman` from `helpers`; `init_weights` from `utils`.
- Produces: CLI `python $SRC/train_paired.py --data-dir DIR --seed_index I --tag TAG [--epochs 200] [--pair-loss none|delta_huber|listwise] [--pair-weight 0.3] [--tie 0.5] [--huber-delta 1.0] [--pair-weighting none|absdelta] [--ligand-only] [--batch-size 128] [--chunk 8]` and the prediction/model/scaler files in the formats above; per-epoch log lines `epoch E loss TOTAL mse MSE pair PAIR valid_pearson V series_valid_spearman S series_valid_pearson P` (`S` = mean Spearman over `series_valid` series with ≥ 5 rows, computed on raw pK).
- `pair_losses.delta_huber(pred: Tensor[n], y: Tensor[n], series: LongTensor[n], lig: LongTensor[n], tie: float, delta: float, weighting: str) -> Tensor[scalar]` (standardised units; pairs from `pair_index`; keep pairs with `|y_i - y_j| >= tie`; `F.huber_loss((p_i-p_j) - (y_i-y_j), 0, delta=delta, reduction="none")`, multiplied by `|y_i-y_j|` when `weighting == "absdelta"`; mean over kept pairs; `0.0` tensor when none).
- `pair_losses.listwise(pred, y, series, lig, tie, weighting) -> Tensor` (ListNet top-one: for each series with ≥ 2 distinct `lig` in the batch, keep one member per `lig` (the first), `q = softmax(y_members)`, `loss = -(q * log_softmax(p_members)).sum()`; mean over series; `tie` unused except to skip series whose `max(y)-min(y) < tie`).
- `train_paired.LigandOnly(model)` wrapper: `forward(data)` clones `data.x`, zeroes columns `15:` and calls the model.

- [x] **Step 1: Write failing tests in `tests/test_pair_losses.py`:** `test_delta_huber_matches_manual` (3 items in one series, hand-computed value), `test_delta_huber_tie_excludes_pair`, `test_delta_huber_no_pairs_is_zero` (all different series → `0.0` with `requires_grad` false allowed), `test_absdelta_weighting`, `test_listwise_matches_manual`, `test_ligand_only_zeroes_aev` (fake model that returns `x[:, 15:].abs().sum()`; wrapped output is 0).
- [x] **Step 2: Run; expected: FAIL.** Run: `.venv/bin/python -m pytest tests/test_pair_losses.py -v`
- [x] **Step 3: Implement `src/pair_losses.py`.**
- [x] **Step 4: Run; expected: PASS.**
- [x] **Step 5: Write `tests/test_train_paired_smoke.py`:** builds synthetic `paired_{train,valid,test,series_valid}.pt` (random graphs, x width 367, 2 series of 6 and 10 singletons in train; 6-row series in series_valid) in `tmp_path`, runs `train_paired.main([...,"--epochs","2","--pair-loss","delta_huber","--pair-weight","0.3","--hidden-dim","8"])` on CPU and asserts the model, scaler and four prediction CSVs exist with the right columns and row counts. (`--hidden-dim` and `--head` exist as hidden flags defaulting to 256/3 for this test only.)
- [x] **Step 6: Implement `src/train_paired.py`**: load the four splits; fit `StandardScaler` on train `y`; transform `y` in place for all splits; `DataLoader(train, batch_sampler=sampler)` with `sampler.set_epoch(e)` each epoch; train loop as `training.py` but `loss = mse + pair_weight * pair_loss(...)` with the pair loss skipped when `--pair-loss none`; validation/checkpoint rule unchanged; after training, reload the best checkpoint and write predictions for `valid`, `test`, `series_valid` (`row, truth, pred` in pK); scaler written atomically before training as `train_seed.py` does; `torch.manual_seed(seed)`, `random.seed(seed)`, `numpy` seed = seed.
- [x] **Step 7: Run both test files; expected: PASS.**
- [x] **Step 8: Write `jobs/train.sbatch`** (`--array=0-2` default, overridable; `--time=03:00:00`; env vars `PAIR_LOSS=none`, `PAIR_WEIGHT=0.3`, `TIE=0.5`, `WEIGHTING=none`, `LIGAND_ONLY=0`, `TAG`, `EPOCHS=200`, `DATA_DIR=$WORK/data`; builds the flag list and runs `train_paired.py` with `--seed_index $SLURM_ARRAY_TASK_ID`) and `jobs/selftest.sbatch` (`EPOCHS=2`, `DATA_DIR=$WORK/data_small`, `--array=0`, tags `self-none` and `self-dh` run sequentially in one job, then greps the two log lines and asserts both prediction CSVs exist).
- [x] **Step 9: `.venv/bin/python -m py_compile src/*.py`; tick the boxes; report decisions left open.**

---

### Task 3: FEP benchmark preparation, featurisation and evaluation

**Files:**
- Create: `src/fep_prepare.py`, `src/fep_featurise.py`, `src/eval_paired.py`, `jobs/fep_prepare.sbatch`, `jobs/eval.sbatch`
- Test: `tests/test_fep_prepare.py`, `tests/test_eval_paired.py`

**Interfaces:**
- Consumes: the Schrödinger archive `/scratch/u6xq/hussain.u6xq/aevplig/raw/schrodinger_public_binding_free_energy_benchmark-main.tar.gz` (local copy extracted at `/private/tmp/claude-501/-Users-alexi-isambard-ssh/a842a9ce-2325-4e67-9749-0f51bf88e121/scratchpad/fetched/x/public_binding_free_energy_benchmark-main/`): `fep_benchmark_inputs/structure_inputs/<subset>/<series>_ligands.sdf` (multi-molecule; title line = ligand name; tag `r_exp_dg` in kcal/mol), `<series>_protein.pdb`, `<series>_edges.csv`, `<subset>/subset_metadata.csv`, and `fep_benchmark_inputs/benchmark_metadata.csv` (`Group name, Group abbreviation, Protein, Input file naming scheme, Reference PDB, Number of nodes, Dynamic range (kcal/mol), DOI of source`). The paper's predictions for the featurisation check: `aev-plig-paper-main.tar.gz` member `aev-plig-paper-main/data/bindingnet_bindingdb/AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim90_predictions.csv` (columns `group_id, unique_id, sdf_file, pdb_file, graph_id, preds_0..preds_9, preds`; `unique_id` like `bayer_macrocycles/wagner_brd4/61a`; 1,184 rows). The clone's `process_and_predict.process_data(config)`, `generate_graphs(config)` (config fields: `dataset_csv, data_name, num_workers, skip_validation, device`), `GraphDatasetPredict` from `utils`, `GATv2Net`.
- Produces: CLI `python $SRC/fep_prepare.py --benchmark-dir DIR --out-dir DIR` → `fep_benchmark.csv` + per-ligand `sdf` and per-series `pdb` under `out-dir/structures/`; `python $SRC/fep_featurise.py --csv FILE --work-dir DIR [--num-workers 72]` → `data/fep_benchmark_graphs.pickle`, `data/fep_benchmark_processed.csv`, `data/processed/fep_benchmark.pt`; `python $SRC/eval_paired.py --models M1 [M2 ...] --scaler S --meta work/data/paired_meta.csv --fep-csv work/data/fep_benchmark.csv --fep-pt work/data/processed/fep_benchmark.pt --series-valid-pt work/data/processed/paired_series_valid.pt --test-pt work/data/processed/paired_test.pt --out-dir work/output/eval/<name> [--paper-csv FILE] [--hidden-dim 256 --head 3]` → `fep_per_series.csv` (`subset, series, ref_pdb, uniprot, n, train_rows_same_uniprot, pearson, spearman, kendall, rmse`), `fep_summary.csv` (weighted and unweighted means over series with `n >= 10`, weights = `n`; plus over all series), `series_valid_per_series.csv`, `series_valid_summary.csv`, `test_summary.csv` (Pearson, RMSE, Spearman, Kendall on CASF), `predictions_<dataset>.csv`, and `SUMMARY.md`.
- pK conversion: `pK = -dG / (R * T * ln(10))` with `R = 0.0019872 kcal/(mol K)`, `T = 298.15` (so −7.04 kcal/mol → 5.160).
- `train_rows_same_uniprot`: count of `paired_meta` rows with `split == "train"` and `uniprot == uniprot(ref_pdb)` where `uniprot(ref_pdb)` comes from the PDBbind name index (reuse `build_paired_dataset.read_pdbbind_index`); `-1` when `ref_pdb` is not in the index.

- [x] **Step 1: Write failing tests `tests/test_fep_prepare.py`:** `test_pk_conversion` (−7.04 → 5.160 ± 0.001), `test_fep_prepare_dedupes_titles` (a synthetic SDF with titles `a, b, a` → 2 rows, `n_entries` 2 and 1, first `a` kept), `test_unique_id_format` (`<subset>/<series>/<title>` with `series` = the file stem without `_ligands`).
- [x] **Step 2: Run; FAIL. Implement `src/fep_prepare.py`** (RDKit `SDMolSupplier(removeHs=False, sanitize=False)` for reading, `SDWriter` for the per-ligand files; copy the protein PDB unchanged). **Run; PASS.**
- [x] **Step 3: Implement `src/fep_featurise.py`** (a `types.SimpleNamespace` config; `device = "cpu"`; run `process_data` then `generate_graphs` from the clone, both of which use the CWD, so `os.chdir(work_dir)` first; then build the `.pt` exactly as `make_predictions` does, without predicting). No unit test beyond `py_compile`; it is exercised by `jobs/fep_prepare.sbatch`.
- [x] **Step 4: Write failing tests `tests/test_eval_paired.py`:** `test_series_metrics_on_toy` (3 series with known Spearman), `test_weighted_mean_rule` (only `n >= 10` series enter the weighted mean; weights = n), `test_train_rows_lookup` (`-1` for an unknown ref_pdb). **Run; FAIL. Implement `src/eval_paired.py`** (predict with every model on each `.pt`, average → `pred`; metrics per group). **Run; PASS.**
- [x] **Step 5: Write `jobs/fep_prepare.sbatch`** (`--time=02:00:00`; extracts the Schrödinger tarball to `/local/user/$(id -u)/aevplig-paired/fep-$SLURM_JOB_ID/`, runs `fep_prepare.py` into `$WORK/data/fep_structures` — about 1,300 small files, acceptable — then `fep_featurise.py --csv $WORK/data/fep_benchmark.csv --work-dir $WORK`, then a featurisation check: `eval_paired.py` with the released ensemble `$PARENT_WORK/output/trained_models/model_GATv2Net_ligsim90_fep_benchmark_{0..9}.model` and scaler `..._fep_benchmark.pickle`, `--paper-csv` pointing at the paper CSV extracted from `aev-plig-paper-main.tar.gz` on `/local`, writing `work/output/eval/released-check/`; `SUMMARY.md` must report the Pearson between our ensemble prediction and the paper's `preds` column over the joined `unique_id`s and the number joined) and `jobs/eval.sbatch` (`--time=01:00:00`; env `TAG`, `SEEDS="0,1,2"`, `NAME`; builds the model list from `$WORK/output/trained_models/${TAG}_<i>.model`).
- [x] **Step 6: `py_compile`, tick boxes, report decisions left open (in particular how unsupported residues `TPO, ACE, NMA, HOH` in the benchmark PDBs are handled: they are dropped by the clone's loader with a warning; count them in the SUMMARY).**

---

### Task 4: Integration on Isambard (run by the coordinating session with a Sonnet monitor)

- [x] **Step 1:** `rsync` (used tar-over-ssh: the local rsync 3.4 and remote 3.2.7 disagree on rule exchange) the tree (excluding `.venv`, `tests/data`, `__pycache__`) to `$PAIRED_ROOT`; `python -m py_compile` on the login node; `mkdir -p work/data/processed work/output/{trained_models,predictions,eval} logs`.
- [x] **Step 2:** `sbatch jobs/build_dataset.sbatch` and `sbatch jobs/build_small.sbatch` and `sbatch jobs/fep_prepare.sbatch` (independent). Check `paired_summary.json` against the census (train 95,680 minus the held-out series; series_valid ≈ 5% of BindingNet templates with ≥ 5 rows ≈ 140 series and ≈ 5% of BindingDB folders with ≥ 5 rows ≈ 34 folders).
- [x] **Step 3:** `sbatch jobs/selftest.sbatch`; then the featurisation check in `work/output/eval/released-check/SUMMARY.md` (expect Pearson > 0.99 between our released-ensemble predictions and the paper's over ≥ 1,100 joined ligands; if lower, stop and diagnose the preparation).
- [x] **Step 4:** Launch the arms, 3 seeds each (array 0-2): `TAG=base PAIR_LOSS=none`; `TAG=dh03 PAIR_LOSS=delta_huber PAIR_WEIGHT=0.3`; `TAG=dh10 PAIR_LOSS=delta_huber PAIR_WEIGHT=1.0`; `TAG=base-lig LIGAND_ONLY=1`; `TAG=dh03-lig PAIR_LOSS=delta_huber PAIR_WEIGHT=0.3 LIGAND_ONLY=1`. 15 jobs, ~1 h each.
- [x] **Step 5:** `jobs/eval.sbatch` per tag; compare `fep_summary.csv` and `series_valid_summary.csv` across tags per seed and as 3-seed ensembles; the gate: the dh arms must beat `base` by more than the seed spread AND by more than `dh03-lig` beats `base-lig`.
- [x] **Step 6:** Write `work/output/eval/REPORT.md` and copy it to the local tree; update the memory note.


## Outcome (2026-11-05)

Arm 1 is negative: see `work/output/eval/REPORT.md` (narrative) and `REPORT_arms.md` (tables). The gate failed on every FEP metric; the vanilla recipe stays.
