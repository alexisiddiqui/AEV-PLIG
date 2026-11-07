# AEV-PLIG in JAX

A JAX rewrite of AEV-PLIG: a structure store built from the archives, radial AEVs computed in
JAX on the fly, the vanilla model and training loop, and profiles of the PyTorch and JAX loops.
Design: `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md`; plan:
`docs/superpowers/plans/2026-11-07-aevplig-jax-rewrite.md` (both at the repository root).
Results: `docs/REPORT_reproduction.md`.

## Results

- Store: 97,857 training-side, 1,184 FEP and 43 15-PGDH complexes; on-the-fly AEVs match the pickled features to 2.34e-5 on every record. Schema 2 adds each pocket atom's element and raw chain, resSeq and insertion code; training, prediction and the parity check refuse a store whose verification is missing, failed, partial or stale.
- Weight port: the parent's PyTorch seed 0 run in JAX matches its PyTorch predictions to 3.08e-5 pK (CASF and FEP).
- Ten seeds, strict float32: CASF 0.8921 / 1.043, FEP weighted PCC 0.5833, Kendall 0.4150 (design targets 0.892 / 1.041, 0.589 / 0.429), 22 min per seed. TF32: 0.8874 / 1.056, 0.5659 / 0.4013 (FEP outside the band), 14 min per seed.
- Speed: 4.1x the PyTorch loop (13.8 vs 57.0 min per seed) in TF32, 2.6x in float32; the 10-min target is missed because the step is GPU-bound (kernels busy about 93% of it): by inference from the kernel kinds, mostly memory-bound fusions in the GAT layers, whose neighbour slots are 64% padding, with matrix multiplies about 29%. Launch gaps (about 7%), the loader (0.1% wait) and the AEV (4-7%) are not the limit.

## Layout

| Path | Contents |
|---|---|
| `src/aevplig_jax/` | the package (modules below) |
| `tests/` | CPU pytest suite (153 tests) and fixtures in `tests/data/`; tests marked `torch` skip when torch or torch_geometric is missing, RDKit-dependent tests skip without RDKit |
| `jobs/` | Isambard sbatch jobs and their helpers; `env.sh` exports the shared paths and flags |
| `docs/` | `REPORT_reproduction.md`; `results/` holds the outputs fetched from Isambard |
| `work/` | runtime outputs (store, output, cache, tmp); not tracked |

Modules of `src/aevplig_jax/`:

| Module | Purpose |
|---|---|
| `constants.py` | fixed values of the reproduction (AEV cutoff and shifts, model sizes, learning rate, batch, seeds) and the 22-type protein atom table |
| `aev.py` | radial AEV: numpy reference, cKDTree neighbour lists, JAX implementation |
| `rdkit_features.py` | ligand reading per source and the 15 RDKit node and 4 bond features |
| `pdb_atoms.py` | protein pocket atoms of a PDB file, filtered exactly as the clone's `LoadPDBasDF` |
| `store.py` | zstd record store (`records.bin`, `index.npz`, `metadata.json`, `records.json`), reader, writer, merge and side-table join |
| `build_store.py` | store builder from a processed CSV and the extracted archives, or from a generic CSV (FEP, PGDH) |
| `verify_store.py` | store verification against the pickled graphs (AEV, RDKit columns, bonds) |
| `store_census.py` | species-0 pocket atoms of a store per source, by element and by atom-name code |
| `batching.py` | per-epoch batch plans, static capacities, one-time record preparation in a fork pool, padded batch assembly |
| `loading.py` | jax-Ka's batch-loading protocol (`BatchSource`, `Prefetcher`, `LoaderTelemetry`, `DeferredScalars`), copied with a provenance header |
| `model.py` | GATv2Net in JAX: initialisation, on-device AEV features, neighbour-list GATv2 layers, masked BatchNorm, pooling, readout |
| `torch_port.py` | PyTorch `GATv2Net` state dict to `model.py` parameter and BatchNorm trees |
| `inference.py` | checkpoints, prediction batches, evaluation-mode prediction, Pearson and RMSE shared by the CLIs |
| `train.py` | one seed of the paper's loop: loss, Adam, per-epoch validation, eight-epoch selection, test prediction; `--precision` default (TF32) or highest, `--aev` onthefly or precomputed |
| `predict.py` | ensemble prediction over a store from `best` checkpoints |
| `evaluate.py` | CASF, FEP (weighted, n >= 10 series) and 15-PGDH metrics and the `SUMMARY.md` table |
| `precompute_aev.py` | the precomputed AEV table for the overhead A/B |
| `profile.py` | JAX step profile and the combined PyTorch / JAX profile report |
| `port_parity.py` | weight-port parity of a ported PyTorch model against its PyTorch predictions |

## Environments

- Local: `jax/.venv`, created with uv (not tracked; Python 3.11, jax 0.10.2, numpy 2.4.6, optax 0.2.8):

  ```bash
  uv venv jax/.venv --python 3.11
  # rdkit pinned: the Isambard build env has 2024.3.3 (numpy<2); 2024.3.4 is the nearest release
  # that takes numpy 2, and both share the GetExplicitValence() path in rdkit_features.py
  uv pip install --python jax/.venv/bin/python jax optax numpy scipy zstandard rdkit==2024.3.4 pytest pandas
  # parity tests only; on Linux add --index-url https://download.pytorch.org/whl/cpu to the torch line
  uv pip install --python jax/.venv/bin/python torch
  uv pip install --python jax/.venv/bin/python torch_geometric==2.5.3
  ```

- Isambard (`source jobs/env.sh`):
  - `$JAX_PY`, the jax-Ka venv `/scratch/u6xq/hussain.u6xq/jax-Ka/.venv` (Python 3.12, jax 0.11.2 with the
    CUDA 13 plugin, optax 0.2.8, numpy 2.5.3, scipy 1.18.1, zstandard 0.25.0; no RDKit, no torch): store
    verification, the self-test, training, prediction, evaluation, the parity comparison and the JAX profile.
  - `$AEVPLIG_PY`, the AEV-PLIG env `/projects/u6xq/hussain/opig-harness/envs/aevplig` (rdkit 2024.3.3,
    numpy 1.26.4, torch 2.6.0, PyG 2.5.3, plus zstandard 0.25.0 added for this work): the store build, the
    PyTorch model export for the parity job and the PyTorch profile.
  - `env.sh` also sets `$TREE` (`/scratch/u6xq/hussain.u6xq/aevplig-jax`), `$PARENT` and `$PAIRED`
    (read-only inputs under `$S/aevplig/` and `$S/aevplig-paired/`), `$SBATCH_FLAGS` (one GPU and 72 CPUs
    per job), `XLA_PYTHON_CLIENT_PREALLOCATE=false`, the compilation cache `$TREE/work/cache` and
    `TMPDIR=$TREE/work/tmp` (ptxas cannot write the node-local `/local/user/<uid>` on every node).
    No compute on login nodes.

## Running on Isambard

The jobs source `/scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh` and log to fixed paths, so the code
tree must live at `$TREE`. Mirror `jax/` there by tar over ssh (rsync fails on a version mismatch);
`COPYFILE_DISABLE=1` keeps macOS `._*` metadata files out of the tree:

```bash
COPYFILE_DISABLE=1 tar -C jax -cf - src jobs tests pytest.ini \
  | ssh u6xq.aip2.isambard 'tar -C /scratch/u6xq/hussain.u6xq/aevplig-jax -xf -'
```

Then, on Isambard:

```bash
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p $TREE/work/store/logs $TREE/work/output/logs
J=$TREE/jobs
b=$(sbatch --parsable $J/build_store.sbatch)        # refuses to overwrite existing stores
v=$(sbatch --parsable --dependency=afterok:$b $J/verify_store.sbatch)
t=$(sbatch --parsable --dependency=afterok:$v $J/selftest.sbatch)
a=$(sbatch --parsable --dependency=afterok:$t $J/train.sbatch)                     # array 0-9, TAG=jax, TF32
sbatch --dependency=afterok:$a --export=ALL,TAG=jax $J/eval.sbatch
a=$(sbatch --parsable --export=ALL,TAG=jax-fp32,PRECISION=highest $J/train.sbatch)  # strict float32
sbatch --dependency=afterok:$a --export=ALL,TAG=jax-fp32 $J/eval.sbatch
sbatch $J/port_parity.sbatch
sbatch $J/profile_torch.sbatch
sbatch $J/profile_jax.sbatch                        # profiles, then one timed seed (tag timing)
sbatch --export=ALL,SKIP_SEED=1 $J/profile_jax.sbatch   # profiles only; re-renders REPORT.md
```

`build_store.sbatch` refuses existing stores: move them aside before a rebuild (stores older than schema 2
are refused by `Store`), then rerun `verify_store.sbatch`. `python -m aevplig_jax.store_census --store DIR`
prints a store's species-0 census (untyped pocket atoms per source by element and atom-name code); the
verify job runs it on trainside. Every `profile_jax.sbatch` run, `SKIP_SEED=1` included, summarises its
traces per kernel into `profile.json`, `REPORT.md` and `trace/<mode>/kernels.csv`;
`python -m aevplig_jax.profile --trace-summary TRACE_DIR --out DIR` does it for an existing trace.

`train.sbatch` takes `TAG` (default `jax`), `EPOCHS` (default 200) and `PRECISION=highest` for strict
float32 (default TF32); `--array=0` runs one seed. A seed directory that already holds
`checkpoints/latest` is refused; continue it with `train.py --resume`. `train.py`, `predict.py` and
`port_parity.py` refuse a store whose `verification.json` is missing, failed, partial (fewer records
compared than stored) or older than the store's files (its `store_fingerprint`: `records.bin` size and
`index.npz` sha256); rerun `verify_store.sbatch` after any rebuild. `--skip-verification` (train and
predict only) opts out with a warning, for synthetic stores and smoke runs.

| Job | Env | Writes (under `$TREE/work`) | Time |
|---|---|---|---|
| `build_store.sbatch` | AEV-PLIG | `store/{pdbbind,bindingnet,bindingdb,trainside,fep,pgdh}/`, logs in `store/logs/` | 1.5-2.5 min wall (92 s of building) |
| `verify_store.sbatch` | jax-Ka | `store/{trainside,fep,pgdh}/verification.json`; the trainside species-0 census in the log and `store/logs/aj-census-<job>-trainside.json` | 2 min |
| `selftest.sbatch` | jax-Ka | pytest on the CPU, then a 2-epoch GPU smoke in `output/selftest/` | 1 min |
| `train.sbatch` | jax-Ka | `output/$TAG/manifest.json`; `output/$TAG/seed-<i>/{protocol.json, history.jsonl, selection.json, test_predictions.csv, checkpoints/{best,latest}/state.npz}`; `output/predictions/${TAG}_..._test_seed<i>.csv` | 14 min per seed TF32, 22 min float32 |
| `eval.sbatch` | jax-Ka | predictions at the tag's training precision; `output/eval/$TAG/`: `predictions_{test,fep,pgdh}.csv`, `casf_summary.csv`, `fep_summary.csv`, `fep_per_series.csv`, `pgdh_summary.csv`, `SUMMARY.md` | 35 s |
| `port_parity.sbatch` | both | `output/parity/`: `torch_seed0.npz`, `parity.csv`, `parity_rows.csv`, `REPORT.md` | |
| `profile_torch.sbatch` | AEV-PLIG | `output/profile/torch/`: `profile_torch.json`, `key_averages.txt`, `gpu_util.txt` | |
| `profile_jax.sbatch` | jax-Ka | `output/profile/`: `profile.json`, `REPORT.md`, `gpu_util.txt`, `trace/` (with `<mode>/kernels.csv`); `output/aev_trainside.npz`; the timing seed in `output/timing/` | profiles plus a 14-min seed |

Logs: `store/logs/` for the store jobs, `output/logs/` for the rest. A seed's `best` checkpoint is
30 MB and `latest` 91 MB.

## Tests

```bash
cd jax
.venv/bin/pytest -p no:cacheprovider                # whole suite: 153 passed in about 45 s with torch
.venv/bin/pytest -p no:cacheprovider tests/test_aev.py -v
```

On Isambard the suite runs inside `selftest.sbatch` (`JAX_PLATFORMS=cpu`, `-m "not torch"`, from `$TREE`):
the jax-Ka venv has neither torch nor RDKit, so those tests skip or are deselected. The species-table test
reads `data/PDB_Atom_Keys.csv` at the repository root and skips when it is absent (code-only mirror).
