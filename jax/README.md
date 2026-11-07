# AEV-PLIG in JAX

A JAX rewrite of AEV-PLIG: a structure store built from the archives, radial AEVs computed in
JAX on the fly, the vanilla model and training loop, and profiles of the PyTorch and JAX loops.
Design: `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md`; plan:
`docs/superpowers/plans/2026-11-07-aevplig-jax-rewrite.md` (both at the repository root).

## Layout

| Path | Contents |
|---|---|
| `src/aevplig_jax/` | the package; `constants.py` (fixed values, species table), `aev.py` (radial AEV: numpy reference, cKDTree neighbour lists, JAX) |
| `tests/` | CPU pytest suite; tests marked `torch` skip when torch or torch_geometric is missing |
| `jobs/` | Isambard sbatch jobs; `env.sh` exports the shared paths and flags |
| `docs/` | reports |
| `work/` | runtime outputs (store, runs, cache); not tracked |

## Environments

- Local: `jax/.venv`, created with uv (not tracked):

  ```bash
  uv venv jax/.venv --python 3.11
  # rdkit pinned: the Isambard build env has 2024.3.3 (numpy<2); 2024.3.4 is the nearest release
  # that takes numpy 2, and both share the GetExplicitValence() path in rdkit_features.py
  uv pip install --python jax/.venv/bin/python jax numpy scipy zstandard rdkit==2024.3.4 pytest pandas
  # parity tests only (on Linux add --index-url https://download.pytorch.org/whl/cpu for torch)
  uv pip install --python jax/.venv/bin/python torch torch_geometric==2.5.3
  ```

- Isambard (`source jobs/env.sh`): `$JAX_PY` (jax-Ka venv) for training, verification,
  profiling and prediction; `$AEVPLIG_PY` (AEV-PLIG env) for the store build, the PyTorch
  profile and reading the pickles. The code tree is `$TREE` (`/scratch/u6xq/hussain.u6xq/aevplig-jax`),
  a copy of `jax/` mirrored by tar over ssh; `$S/aevplig/` and `$S/aevplig-paired/` are read-only inputs.
  Every job requests one GPU and 72 CPUs (`$SBATCH_FLAGS`); no compute on login nodes.

## Tests

```bash
cd jax
.venv/bin/pytest            # whole suite
.venv/bin/pytest tests/test_aev.py -v
```

The species-table test reads `data/PDB_Atom_Keys.csv` at the repository root and skips when it
is absent (code-only mirror).
