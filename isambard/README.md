# AEV-PLIG retraining on Isambard-AI

AEV-PLIG (oxpig/AEV-PLIG, commit `3ccac4b84d6fc1fb372cb623e705837fbf280f15`), retrained on
PDBbind v2020 + BindingNet v1 + BindingDB-DCS (surflex), ligand similarity < 0.9 to the FEP benchmark.

## Layout

| Path | What |
|---|---|
| `/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b` | Pinned clone (`$REPO`). Never edited; the scripts run from it. |
| `/projects/u6xq/hussain/opig-harness/envs/aevplig` | uv venv on CPython 3.11.17. Torch 2.6.0+cu126, PyG 2.5.3, torchani 2.2.4, rdkit 2024.3.3. No torch-scatter: nothing imports it. |
| `/projects/u6xq/hussain/opig-harness/build-logs/aevplig.log` | Env build log, import test and `uv pip freeze` (from `jobs/build_env.sbatch`). |
| `raw/` | Dataset archives, kept as tarballs. Never extracted here. |
| `work/` | Run directory (`$WORK`); every script runs with this as its CWD. |
| `work/data/*.csv` | Repo metadata CSVs (`pdbbind/bindingnet/bindingdb_processed.csv`, `PDB_Atom_Keys.csv`, `example_dataset.csv`). |
| `work/data/{pdbbind,bindingnet,bindingdb}.pickle` | Graph pickles (from featurise). |
| `work/data/processed/<DATASET>_{train,valid,test}.pt` | PyG datasets (from featurise). |
| `work/data/example_rcsb*` | Small RCSB stand-in for the example set (12 complexes). |
| `work/output/trained_models/` | Released ensemble (symlinks into the clone) and retrained models. |
| `work/output/predictions/` | Predictions and evaluation CSVs. |
| `jobs/` | Job scripts, listed below. |
| `logs/` | Slurm logs. |

`DATASET=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark`. The README's `--dataset` value lacks the
`_fep_benchmark` suffix and is wrong.

## Jobs (`jobs/`)

Every job asks for 1 GPU, 72 CPUs and 2G/CPU on `workq`, with the standard `--exclude` list.

| File | What |
|---|---|
| `env.sh` | Sourced by every job. Activates the venv, exports `REPO WORK JOBS DATASET LOCAL_ROOT`, sets `PYTHONDONTWRITEBYTECODE=1`, then `cd $WORK`. |
| `build_env.sbatch` | Rebuilds the env (`rm -rf envs/aevplig/.complete` to force it). |
| `featurise.sbatch` (+ `featurise_check.py`) | Archives to pickles to `.pt` files. |
| `train_seed.py` (+ `train_seed.diff`) | Patched copy of `training.py`: `--seed_index`, `--tag`, fixed output names. The diff against the clone's `training.py` is in `train_seed.diff`. |
| `train.sbatch` | Array 0-9, one seed per task. |
| `ensemble_eval.py` / `eval.sbatch` | Test-split ensemble evaluation of the retrained and released ensembles: mean of the members, then Pearson, RMSE, Spearman and Kendall. |
| `predict_example.sbatch` (+ `fetch_example.py`) | Env check: `process_and_predict.py` with the released weights. |
| `selftest.sbatch` | Toy 2-epoch run of `train_seed.py` and `ensemble_eval.py` on the example graphs, in `/local`. |

Scripts run as `cd $WORK && python $REPO/<script>.py`. Python puts the script's own directory (the clone) on
`sys.path`, so `utils`, `helpers`, `model_defs` and `torchani_mod` import from the clone. `train_seed.py` and
`ensemble_eval.py` live in `jobs/`, so they add `$AEVPLIG_REPO` to `sys.path` themselves.

## Featurisation (`jobs/featurise.sbatch`)

Archives in `raw/` (md5s in `raw/SOURCES.md`), each extracted to `/local/user/$UID/aevplig/feat-<job>/<dir>`.
The archives are extracted in parallel (`tar -I pigz` for .tar.gz).

| Variable (default) | Layout inside the archive | Linked as |
|---|---|---|
| `PDBBIND_REFINED` (`PDBbind_v2020_refined.tar.gz`) | `refined-set/<pdb>/<pdb>_{protein.pdb,ligand.mol2,ligand.sdf,pocket.pdb}` | `work/data/pdbbind/refined-set` |
| `PDBBIND_GENERAL` (`PDBbind_v2020_other_PL.tar.gz`) | `v2020-other-PL/<pdb>/<pdb>_{protein.pdb,ligand.mol2,...}` | `work/data/pdbbind/general-set` |
| `BINDINGNET` (`BindingNetv1.tar.gz`) | `./BindingNetv1/<pdb>/target_<CHEMBL>/<compnd>/<pdb>_<CHEMBL>_<compnd>.sdf`, no `rec_h_opt.pdb` | `work/data/bindingnet/from_chembl_client` (after the merge) |
| `BINDINGNET_TEMPLATES` (`Crystal_Templates_for_BindingNet1.tar.gz`) | `./Crystal_Templates_for_BindingNet1/<pdb>/{rec_h_opt.pdb,cry_lig_opt_converted.sdf}` | merged into BindingNetv1 |
| `BINDINGDB` (`surflex.tar`) | 1,598 folders at the tar root, e.g. `11GS_EAA/11GS.pdb`, `11GS_EAA/11GS-results_2.mol2` | extracted into `surflex/`, linked as `work/data/bindingdb/surflex` |

BindingNet merge: `generate_bindingnet_graphs.py` reads `from_chembl_client/<pdb>/target_<t>/<c>/<pdb>_<t>_<c>.sdf`
and `from_chembl_client/<pdb>/rec_h_opt.pdb`. Both archives are extracted into one directory. For each
`BindingNetv1/<pdb>/`, the job symlinks `BindingNetv1/<pdb>/rec_h_opt.pdb` to
`Crystal_Templates_for_BindingNet1/<pdb>/rec_h_opt.pdb`. `from_chembl_client` then points at `BindingNetv1/`.
In job 8072251, 7,146 of 7,150 pdb dirs got a template. The 4 without one are not in `bindingnet_processed.csv`.

Steps: extract, merge, link, `featurise_check.py check` (every mol2/sdf/pdb named by the three CSVs must
exist; pdbbind rows use the `refined` column to choose the set), then the three `generate_*_graphs.py` in
parallel (16 threads each, one log each), then `create_pytorch_data.py`. The job ends with a summary of sizes
and counts, and the EXIT trap removes the `/local` tree and the links. If `jobs/generate_<name>_graphs.py`
exists, the job runs it in place of the clone's script. No such copy exists: the clone's scripts ran unmodified.

Measured (job 8072251, nid011001, 2026-11-04): **28 min 22 s wall** (sacct).

| Phase | Time |
|---|---|
| extract all five archives (59 GB on /local) | 16 s |
| merge + links + check | 11 s |
| `generate_pdbbind_graphs.py` (19,308 complexes, 0 failed) | 450 s |
| `generate_bindingnet_graphs.py` (69,727) | 1,592 s (the critical path, about 44 it/s) |
| `generate_bindingdb_graphs.py` (8,822) | 179 s |
| `create_pytorch_data.py` | 52 s |
| cleanup of /local | 5 s |

Outputs:

| File | Size | Count |
|---|---|---|
| `data/pdbbind.pickle` | 1.95 GB | 19,308 keys |
| `data/bindingnet.pickle` | 6.84 GB | 69,727 keys |
| `data/bindingdb.pickle` | 0.73 GB | 8,822 keys |
| `data/processed/<DATASET>_train.pt` | 4.68 GB | 95,680 graphs |
| `data/processed/<DATASET>_valid.pt` | 48.7 MB | 993 graphs |
| `data/processed/<DATASET>_test.pt` | 10.4 MB | 277 graphs |

No complexes were skipped. Every key count matches its CSV row count. The scratch file count under
`aevplig/` went from 62 to 72: 3 pickles, 3 `.pt` files and 4 logs.

## Commands

```bash
J=/scratch/u6xq/hussain.u6xq/aevplig/jobs

# 1. featurise (done: job 8072251; the archive names in raw/ are the defaults, so no --export is needed)
sbatch $J/featurise.sbatch
#    reuse existing pickles for some datasets: --export=ALL,SKIP_GENERATE=pdbbind,bindingdb

# 2. smoke test: seed index 0, 2 epochs, tag "smoke" (outputs smoke_model_*; delete afterwards)
sbatch --array=0 --export=ALL,EPOCHS=2,TAG=smoke $J/train.sbatch

# 3. full ensemble: 10 array tasks, 200 epochs each, tag "retrain"
sbatch $J/train.sbatch

# 4. ensemble evaluation (after the array finishes)
sbatch --dependency=afterok:<array job id> $J/eval.sbatch
#    smoke run instead: sbatch --export=ALL,TAG=smoke $J/eval.sbatch

# env check / prediction on the example stand-in
sbatch $J/predict_example.sbatch
```

What each train task runs (task `i`, from `$WORK`):

```bash
python $JOBS/train_seed.py --seed_index=$i --tag=retrain --epochs=200 --model=GATv2Net \
  --dataset=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark --batch_size=128 --hidden_dim=256 \
  --head=3 --lr=0.00012291937615434127 --activation_function=leaky_relu
```

Seeds by index: `[100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]`. These are the same seeds and the same
per-seed code path as the original sequential loop.

## Where outputs land

- `work/output/trained_models/retrain_model_GATv2Net_<DATASET>_<i>.model`: the best checkpoint by the
  8-epoch running mean of validation Pearson, as in `training.py`.
- `work/output/trained_models/retrain_model_GATv2Net_<DATASET>.pickle`: the y-scaler. It is identical for
  every seed and is written atomically at the start of each task.
- `work/output/predictions/retrain_<DATASET>_test_seed<i>.csv`: per-seed test predictions.
- `work/output/predictions/ensemble_eval_<DATASET>_test.csv`: truth, every member and both ensemble means.
  The summary table is in the eval log.
- `work/output/predictions/example_rcsb_predictions.csv`: example predictions from the released ensemble.
- Featurise logs: `logs/aevplig-feat-<job>.out`, plus one log per generate script,
  `logs/featurise-<job>-{pdbbind,bindingnet,bindingdb}.log`.

## Inode policy

- Scratch shares a quota of about 1.02M files with the user's other projects, and it is nearly full. On
  scratch, the datasets exist only as the archives in `raw/`. The extracted trees (hundreds of thousands of
  files) never touch scratch.
- `featurise.sbatch` extracts to `/local/user/$UID/aevplig/feat-<jobid>` (node-local, 3.5 TB). It symlinks
  the trees into `work/data/{pdbbind/refined-set, pdbbind/general-set, bindingnet/from_chembl_client,
  bindingdb/surflex}` and removes the extraction and the links at exit, including on failure.
- The scripts keep only a few large files on scratch: three pickles, three `.pt` files, about 21 model files
  and the CSVs.
- The env and clone live in `/projects`, which has a 51M-file quota. `PYTHONDONTWRITEBYTECODE=1` keeps
  `__pycache__` out of the clone and the env. The build's temporary directories are on `/local`.
- Delete smoke or toy outputs (`smoke_*`) when done.

## Training run

Smoke test (job 8073327, array task 0, EPOCHS=2, TAG=smoke, nid010911): queue wait 7 min 9 s (submit 23:31:23, start 23:38:32), 62 s wall,
python total 41.3 s for 2 epochs including dataset load (about 1-2 s from the .pt files) and one test pass. The log has no per-epoch timestamps,
so epoch time is total elapsed / 2, about 20 s per epoch (train 95,680 graphs + validation). No GPU memory use is printed (only the 97,871 MiB
total); MaxRSS 6.0 GB. Smoke outputs landed in work/output/ with the smoke_ prefix, scaler pickle included; smoke model, scaler and
prediction CSV deleted afterwards, log kept.

Per-seed ETA at 200 epochs: 200 x 20 s + load = about 67 min (about 1.1 h), far under the 24 h limit in train.sbatch, so --time was not changed.

Launched 2026-11-04T23:42:00Z: train array job 8073760 (tasks 0-9, TAG=retrain, 200 epochs), eval job 8073761 (afterok:8073760).

## Result (2026-11-05)
Array 8073760 (10 seeds, 200 epochs, 1 GPU each) all COMPLETED in 54-58 min per seed; launched 23:42 UTC, eval 8073761 done 00:41 UTC (59 min wall end to end).
Test split (277 complexes), ensemble of 10 (logs/aevplig-eval-8073761*, output/predictions/ensemble_eval_*_test.csv):
  retrained  Pearson 0.8919  RMSE 1.0407  Spearman 0.8923  Kendall 0.7247
  released   Pearson 0.8929  RMSE 1.0392  Spearman 0.8947  Kendall 0.7295
Member Pearson range: retrained 0.862-0.889, released 0.854-0.889. The retrain reproduces the release.
Models: work/output/trained_models/retrain_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_{0..9}.model + .pickle (use --trained_model_name=retrain_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark with process_and_predict.py).
