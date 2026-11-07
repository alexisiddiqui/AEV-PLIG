# AEV-PLIG JAX Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the structure store, the on-the-fly radial AEVs, the vanilla AEV-PLIG model and training loop in JAX, and the profiles that show where the time goes, reproducing the released model at ten seeds.

**Architecture:** A zstd record store (jax-Ka style) holds ligand atoms, bonds and typed pocket atoms per complex; a host-side assembler builds static-shape batches (concatenated nodes with padded neighbour lists, pocket atoms with per-ligand-atom neighbour lists) that a jitted step turns into AEVs, five GATv2 layers with masked BatchNorm, max+mean pooling and the dense head. Plain JAX parameter dicts and optax, with jax-Ka's prefetcher, telemetry, run directory and selection conventions.

**Tech Stack:** JAX 0.11.2 (CUDA 13 plugin) + optax 0.2.8 + numpy 2.5 + scipy + zstandard in the jax-Ka venv for training; rdkit 2024.3.3 + numpy 1.26 + torch 2.6 (reading pickles and models) + zstandard (to add) in the AEV-PLIG env for the store build; pytest locally with jax CPU, torch CPU and torch_geometric 2.5.3 for parity tests.

**Spec:** `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md` (read it first; this plan argues from it).

## Global Constraints

- Code under `jax/` on branch `jax-rewrite` of the fork (`jax/src/aevplig_jax/` package on `PYTHONPATH`, `jax/jobs/`, `jax/tests/`, `jax/docs/`); Isambard tree `/scratch/u6xq/hussain.u6xq/aevplig-jax/` (`$TREE`), mirrored by tar over ssh from `~/isambard-ssh/aevplig-jax/` (a copy of `jax/`).
- Every Isambard job: `--partition=workq --gres=gpu:1 --cpus-per-task=72 --mem-per-cpu=2G --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]`; archives extracted only to node-local disk inside jobs; no compute on login nodes; nothing written under `$S/aevplig/` (read-only inputs) or `$S/aevplig-paired/`.
- Environments: `JAX_PY=/scratch/u6xq/hussain.u6xq/jax-Ka/.venv/bin/python` (training, verification, profiling, prediction); `AEVPLIG_PY=/projects/u6xq/hussain/opig-harness/envs/aevplig/bin/python` (store build, PyTorch profile, pickle reading); `zstandard` is added to the AEV-PLIG env with `uv pip install --python $AEVPLIG_PY zstandard` (the env is this project's). No other new environment or package.
- Module import boundaries: `rdkit_features.py`, `pdb_atoms.py`, `build_store.py` may import rdkit and pandas; nothing else imports them. `aev.py` imports jax lazily inside `radial_aev_jax`; `store.py`, `batching.py`, `loading.py`, `verify_store.py` import only numpy, scipy, zstandard, json.
- Constants copied from the spec: `RC = 5.1`, `ETA = 19.7`, `SHIFTS = [0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68, 2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83]`, 22 species, AEV term `0.25 * exp(-ETA * (r - Rs)^2) * (0.5 * cos(pi * r / RC) + 0.5)` for `r < RC`, protein neighbours only; `ATOM_SYMBOLS = ['F','N','Cl','O','Br','C','B','P','I','S']`; bond one-hot over RDKit bond types `[1, 12, 2, 3]`; `HIDDEN = 256`, `HEADS = 3`, `LAYERS = 5`, dense 1024/512/256/1; `LR = 1.2291937615434127e-4`; `BATCH = 128`; `EPOCHS = 200`; `SELECTION_WINDOW = 8`; `SEEDS = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]`; leaky-relu slopes 0.2 (attention) and 0.01 (activation); BatchNorm momentum 0.1, eps 1e-5; attention mask fill `-1e9`; float32 everywhere; matmul precision default (TF32) for training, `highest` for the port-parity check.
- Padding convention: padded node rows, pocket rows and batch slots are copies of real rows with their valid flags 0 (spread padding); every loss and statistic multiplies by the valid flags.
- Commit after every task with a message that has no attribution lines; never push.

## Review Focus

- A complex whose ligand has a single heavy atom or no bonds (isolated node): the self-loop feature is zeros, the layer still runs, pooling is that one node. Test in Task 7.
- A ligand atom with no protein atom within 5.1 A (solvent-exposed): its AEV row is all zeros, as in the pickles. Test in Task 1.
- A pocket with more typed atoms within 5.1 A of one ligand atom than `Ka`: capacities come from the data so this cannot happen in training, but `assemble` must raise rather than silently truncate. Test in Task 6.
- A PDB file with insertion codes, alternate locations or chain breaks: residue serials stay unique per (chain, resSeq, iCode) and atom filtering matches `LoadPDBasDF` (ATOM records, hydrogens dropped by name, unmapped names dropped). Test in Task 3.
- A batch where all 128 graphs are real but one is larger than any seen in planning (prediction stores such as FEP): prediction batches are planned from their own store's capacities, never the training manifest's. Test in Task 8.

---

### Task 1: Package scaffold, constants, radial AEV (numpy reference and JAX)

**Files:**
- Create: `jax/README.md` (layout, environments, how to run tests), `jax/pytest.ini` (`pythonpath = src`, `testpaths = tests`, `markers = torch: needs torch and torch_geometric`), `jax/.venv` via `uv venv jax/.venv --python 3.11`, `uv pip install jax numpy scipy zstandard rdkit pytest pandas` and, for the parity tests only, `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` then `uv pip install torch_geometric==2.5.3`.
- Create: `jax/src/aevplig_jax/__init__.py`, `jax/src/aevplig_jax/constants.py`, `jax/src/aevplig_jax/aev.py`
- Create: `jax/tests/conftest.py`, `jax/tests/test_aev.py`
- Create: `jax/jobs/env.sh` (exports `S`, `TREE`, `JAX_PY`, `AEVPLIG_PY`, `PARENT=$S/aevplig/work`, `PAIRED=$S/aevplig-paired/work`, `SBATCH_FLAGS`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_COMPILATION_CACHE_DIR=$TREE/work/cache`, `PYTHONPATH=$TREE/src`)

**Interfaces:**
- Produces: `constants.py`: the Global Constraints values as module constants plus `species_table(keys_csv: Path) -> dict[str, int]` mapping `PDB_ATOM` keys (e.g. `ALA-CA`) to type index 1..22 in order of first appearance of `ATOM_TYPE` in `data/PDB_Atom_Keys.csv`, and `N_SPECIES = 22`, `N_SHIFTS = 16`, `AEV_DIM = 352`, `RDKIT_DIM = 15`, `NODE_DIM = 367`, `EDGE_DIM = 4`.
- Produces: `aev.py`: `radial_aev_numpy(lig_xyz: np.ndarray, env_xyz: np.ndarray, env_species: np.ndarray) -> np.ndarray` (brute force, float64, shape `(L, 352)`, species-major columns, species 0 contributes nothing); `neighbour_lists(lig_xyz, env_xyz, cutoff: float = RC, k: int | None = None) -> tuple[np.ndarray, np.ndarray]` returning `nbr (L, K) int32` and `mask (L, K) bool` from `scipy.spatial.cKDTree` (K = max count when `k` is None; raises `ValueError` when a row has more than `k` neighbours); `radial_aev_jax(lig_xyz, env_xyz, env_species, nbr, mask) -> jnp.ndarray` of shape `(L, 352)` float32 (gather by `nbr`, one-hot over species, masked terms zero).
- `conftest.py` fixtures: `rng` (`np.random.default_rng(0)`), `point_set` (30 ligand points, 200 pocket points in a 12 A box, species uniform in 0..22 with 10% zeros), `torch_available` skip marker.

- [ ] **Step 1: Write the failing tests** in `jax/tests/test_aev.py`: `test_numpy_matches_brute_force_single_pair` (one ligand atom, one pocket atom of species 7 at distance 2.0: column `(7-1)*16 + k` equals `0.25*exp(-19.7*(2.0-Rs_k)^2)*(0.5*cos(pi*2/5.1)+0.5)`, all other columns 0); `test_beyond_cutoff_is_zero` (distance 5.1 and 6.0 give zeros); `test_species_zero_contributes_nothing`; `test_isolated_ligand_atom_row_is_zero` (no pocket atom within 5.1 A); `test_jax_matches_numpy` (point_set: `radial_aev_jax` vs `radial_aev_numpy`, max abs diff < 1e-5); `test_neighbour_lists_raise_when_k_too_small`.
- [ ] **Step 2: Run** `cd jax && .venv/bin/pytest tests/test_aev.py -v` — expected: FAIL (module not found).
- [ ] **Step 3: Implement** `constants.py` and `aev.py` with the signatures above.
- [ ] **Step 4: Run** the tests — expected: PASS.
- [ ] **Step 5: Commit** `git add jax/README.md jax/pytest.ini jax/src/aevplig_jax/{__init__,constants,aev}.py jax/tests/{conftest,test_aev}.py jax/jobs/env.sh` with message "Add JAX AEV-PLIG scaffold with radial AEV reference and JAX implementation" (never add `jax/.venv`; add it to `.gitignore`).

### Task 2: Ligand reader and RDKit features

**Files:**
- Create: `jax/src/aevplig_jax/rdkit_features.py`, `jax/tests/test_rdkit_features.py`, `jax/tests/data/benzamide.sdf` and `jax/tests/data/benzamide.mol2` (written by the test setup from SMILES `NC(=O)c1ccccc1` with RDKit, embedded 3D, hydrogens explicit).

**Interfaces:**
- Produces: `read_ligand(path: Path, kind: str) -> Chem.Mol` with `kind in {"sdf", "mol2"}`: sdf = `Chem.SDMolSupplier(str(path), removeHs=False)[0]`; mol2 = `Chem.MolFromMol2File(str(path))` then `Chem.AddHs(mol, addCoords=True)`; returns `None` when RDKit cannot read (the caller records the failure as the clone does).
- Produces: `LigandArrays` (dataclass): `xyz (L,3) float32`, `feat (L,15) float32`, `bond_src (E,) int32`, `bond_dst (E,) int32`, `bond_type (E,4) float32`; and `ligand_arrays(mol) -> LigandArrays` replicating `atom_features` and `mol_to_graph` in `generate_bindingnet_graphs.py` (heavy atoms in RDKit index order; features: one-hot symbol over `ATOM_SYMBOLS` (raise on others, as the clone does), heavy-neighbour count, H-neighbour count, explicit valence, aromatic, in-ring; both bond directions; edges sorted by (src, dst); bond one-hot over `[1, 12, 2, 3]`).

- [ ] **Step 1: Write the failing tests**: `test_benzamide_features_sdf` (10 heavy atoms; ring carbons aromatic=1, in_ring=1; the amide N has H count 2, heavy count 1; explicit valences as RDKit reports; symbol one-hot columns correct), `test_bonds_sorted_and_symmetric` (E = 20, every (i,j) has (j,i), rows sorted), `test_mol2_adds_hydrogen_counts` (reading the mol2 of the same molecule gives the same `feat` as the sdf path), `test_unknown_element_raises` (a molecule with Si).
- [ ] **Step 2: Run** `pytest tests/test_rdkit_features.py -v` — FAIL.
- [ ] **Step 3: Implement** `rdkit_features.py`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** "Add ligand reader and RDKit node and bond features".

### Task 3: PDB pocket reader

**Files:**
- Create: `jax/src/aevplig_jax/pdb_atoms.py`, `jax/tests/test_pdb_atoms.py`, `jax/tests/data/pocket.pdb` (a 60-line hand-written PDB fragment with two chains, one insertion code, one altloc pair, hydrogens, a HETATM water and one atom whose name is not in the key table).

**Interfaces:**
- Produces: `PocketArrays` (dataclass): `xyz (P,3) float32`, `species (P,) int8` (1..22, 0 when the `RES-ATOM` key is not in the species table), `res (P,) int32` (serial over distinct (chain, resSeq, iCode) in file order, 0-based), `resname (P,) int8` (index into the 20 standard residues in alphabetical three-letter order, 20 = other), `atom (P,) int16` (index into a sorted table of atom names seen in `PDB_Atom_Keys.csv`, -1 = other), `chain (P,) int8` (serial over chains in file order), `bfactor (P,) float16`.
- Produces: `read_pocket(pdb_path: Path, lig_xyz: np.ndarray, species: dict[str, int], radius: float = 10.0) -> PocketArrays`: keeps exactly the atoms `LoadPDBasDF` keeps (`ATOM` records only; hydrogen test on the atom-name field as in `generate_pdbbind_graphs.py:48-56`), then only those within `radius` of any ligand heavy atom; unmapped names are kept with species 0 (the AEV ignores them; the pocket encoder may use them later).
- Produces: `typed_within(pocket: PocketArrays, lig_xyz, cutoff: float = RC) -> np.ndarray` boolean over pocket atoms (species > 0 and within `cutoff` of some ligand atom): this subset is what goes to the device.

- [ ] **Step 1: Write the failing tests**: `test_filters_like_loadpdbasdf` (HETATM and hydrogens dropped; the unmapped atom kept with species 0; both altloc atoms kept, as the clone keeps them), `test_residue_serials_unique_with_insertion_code`, `test_radius_crop` (an atom 11 A from every ligand atom is dropped), `test_species_lookup_matches_key_table` (`ALA-CA` -> 2).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `pdb_atoms.py`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** "Add PDB pocket reader matching the clone's atom filtering".

### Task 4: Record store and the store builder

**Files:**
- Create: `jax/src/aevplig_jax/store.py`, `jax/src/aevplig_jax/build_store.py`, `jax/jobs/build_store.sbatch`, `jax/tests/test_store.py`, `jax/tests/test_build_store.py`.

**Interfaces:**
- Produces: `Record` (dataclass): `id: str`, `lig: LigandArrays`, `pocket: PocketArrays`, `label: float` (NaN when absent), `source: str`, `split: str`, `meta: dict`.
- Produces: `FIELDS`: ordered list of `(name, dtype, trailing_shape)` for the twelve arrays (`lig_xyz, lig_feat, bond_src, bond_dst, bond_type, prot_xyz, prot_species, prot_res, prot_resname, prot_atom, prot_chain, prot_bfactor`); a record frame is the concatenation of these arrays' bytes in that order, zstd-compressed (level 3).
- Produces: `StoreWriter(path: Path, metadata: dict)` with `add(record: Record) -> None` and `close() -> None` writing `records.bin`, `index.npz` (`ids` (N,) str, `offset` (N,) int64, `length` (N,) int64, `dims` (N,5) int32 = `[L, E, P, A, KA]` with `A = typed_within(...).sum()` and `KA` = the largest number of typed protein atoms within 5.1 A of any one ligand atom). Store order = the order of `create_pytorch_data.py`'s concatenated frame (PDBbind rows, then BindingNet, then BindingDB, each in its processed-CSV order), so `split_ids(split)` returns ids in the same order as the clone's `.pt` datasets and the parent's `ensemble_eval.py` lines up `truth` rows., `records.json` (id -> `{source, split, label, meta}`), `metadata.json` (`schema: 1`, fields, `cutoff: 5.1`, `pocket_radius: 10.0`, species table, package versions).
- Produces: `Store(path: Path)` with `ids: list[str]`, `dims: np.ndarray`, `records: dict`, `metadata: dict`, `get(id: str) -> Record` (pread + decompress, thread-safe), `labels() -> np.ndarray`, `split_ids(split: str) -> list[str]`.
- Produces: `merge_stores(parts: list[Path], out: Path) -> None` (concatenates part files, rewrites offsets, keeps the source order above).
- Produces: `join_side(store: Store, ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: aligns a side table (`values[i]` belongs to `ids[i]`) to store order, returning the aligned array (zeros where missing) and a boolean `missing` mask; this is how the Zernike npz files and later RSA tables attach.
- Produces: `build_store.py` CLI: `--source {pdbbind, bindingnet, bindingdb}` with `--processed-csv`, `--root` (extracted archive directory) and `--keys`; or `--generic --csv FILE` (`unique_id, sdf_file, pdb_file`, optional `label` column); `--out DIR`, `--workers N`, `--limit N`. Path templates, the mol2-vs-sdf choice and the split rule are copied from `create_pytorch_data.py:26-57` and `generate_{pdbbind,bindingnet,bindingdb}_graphs.py` (PDBbind: `split_core` -> split; BindingNet and BindingDB: `max_tanimoto_fep_benchmark` rule -> `train` or `excluded`; valid/test come from PDBbind only). Failures are listed in `build_summary.json` (id, reason) and must match the clone's failed lists (none expected for the three sources after the parent's run).
- Produces: `jax/jobs/build_store.sbatch`: extraction to `/local` copied from `isambard/jobs/featurise.sbatch`, then the three sources built in parallel (72 workers split 24/24/24) into `$TREE/work/store/{pdbbind,bindingnet,bindingdb}` and merged into `$TREE/work/store/trainside` (one store with split labels), then generic builds for FEP (`$PAIRED/data/fep_benchmark.csv`) and 15-PGDH (`$PAIRED/data/pgdh.csv`) into `$TREE/work/store/{fep,pgdh}`; runs under `$AEVPLIG_PY`.

- [ ] **Step 1: Write the failing tests**: `test_store_round_trip` (two synthetic records in, same arrays out, dims correct, `A` counts typed atoms within 5.1 A), `test_store_get_is_thread_safe` (8 threads x 50 gets), `test_merge_stores_preserves_records`, `test_build_generic_csv` (benzamide sdf + `pocket.pdb` -> store with one record, label NaN when the column is absent), `test_split_rule_matches_create_pytorch_data` (a tiny processed CSV with `max_tanimoto_fep_benchmark` 0.85 and 0.95 -> `train` and `excluded`; a PDBbind row with `split_core = test` -> `test`), `test_store_order_follows_source_order`, `test_join_side_aligns_and_flags_missing`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `store.py`, `build_store.py`, `build_store.sbatch`.
- [ ] **Step 4: Run** — PASS; `bash -n jax/jobs/build_store.sbatch`.
- [ ] **Step 5: Commit** "Add the zstd record store and the store builder".

### Task 5: Store verification against the pickles, and the Isambard build

**Files:**
- Create: `jax/src/aevplig_jax/verify_store.py`, `jax/jobs/verify_store.sbatch`, `jax/tests/test_verify_store.py`.

**Interfaces:**
- Produces: `verify_store.py` CLI: `--store DIR --pickle FILE [--pickle FILE ...] --out verification.json [--sample N]`: for every store id present in a pickle (`graphs_dict[id] = (c_size, features, edge_index, edge_features)`), compares `lig_feat` to `features[:, :15]` (exact), `radial_aev_jax` from the record (`typed_within` subset, `neighbour_lists`) to `features[:, 15:]` (max abs < 1e-4), bonds to `edge_index`/`edge_features` (exact), `L` to `c_size`; writes counts, the worst id and difference, and `passed: bool`; exits non-zero on failure. Reads pickles with numpy only (no torch import).
- Produces: `verify_store.sbatch`: runs under `$JAX_PY` on a GPU node (every job does; JAX uses the GPU) over `trainside` against the three parent pickles (`$PARENT/data/{pdbbind,bindingnet,bindingdb}.pickle`), `fep` against `$PAIRED/data/fep_benchmark_graphs.pickle`, `pgdh` against `$PAIRED/data/pgdh_graphs.pickle`.

- [ ] **Step 1: Write the failing tests**: `test_verify_passes_on_consistent_pickle` (build a store from the benzamide fixture, write a pickle whose features come from `radial_aev_numpy` + `ligand_arrays`, verify passes), `test_verify_fails_on_perturbed_feature` (one AEV value changed by 1e-3 -> `passed` false, worst id reported), `test_verify_handles_missing_id` (counted, not an error).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** "Add store verification against the pickled features".
- [ ] **Step 6: Isambard**: mirror `jax/` to `$TREE` (tar over ssh), `uv pip install --python $AEVPLIG_PY zstandard`, submit `build_store.sbatch` (job name `aj-build`), then `verify_store.sbatch` chained with `--dependency=afterok`. A Sonnet monitor collects `build_summary.json` and `verification.json`. Expected: 97,857 + 1,184 + 43 records, `passed: true` for all three verifications, worst AEV difference below 1e-4. If verification fails, the fix goes into Tasks 2 to 4 with a regression test, and the build reruns.
- [ ] **Step 7: Record** the counts, worst differences and job ids in `jax/docs/REPORT_reproduction.md` (section "Store"); commit "Record the store build and verification results".

### Task 6: Loader, plans, capacities and batch assembly

**Files:**
- Create: `jax/src/aevplig_jax/loading.py` (copy of `pkatrain/loading.py` classes `LoaderConfig`, `BatchSource`, `LoaderTelemetry`, `Prefetcher`, `DeferredScalars` from the jax-Ka snapshot, with a header line naming the source and commit), `jax/src/aevplig_jax/batching.py`, `jax/tests/test_batching.py`.

**Interfaces:**
- Produces: `Capacities` (frozen dataclass): `ncap: int`, `mcap: int`, `kb: int`, `ka: int`, `batch: int = 128`; `to_json()/from_json()`.
- Produces: `plan_epoch(ids: np.ndarray, seed: int, epoch: int, batch: int = 128) -> list[np.ndarray]`: permutation from `np.random.default_rng([seed, epoch])`, chunks of `batch`, last chunk short.
- Produces: `capacities(store: Store, ids: np.ndarray, seeds: list[int], epochs: int, batch: int = 128) -> Capacities`: `ncap` = max over all planned batches of `sum(L)` rounded up to 256; `mcap` = max of `sum(A)` rounded up to 1024; `kb` = max bonded degree + 1 (self-loop slot); `ka` = max of `dims[:, 4]` (KA) rounded up to 32. Written to `manifest.json` next to the run by `train.py`.
- Produces: `Batch` (NamedTuple of numpy arrays, converted to device arrays by `Prefetcher.to_device`): `x_lig (ncap,15) f32`, `node_graph (ncap,) i32` (padded rows point at segment `batch`), `node_valid (ncap,) f32`, `nbr (ncap,kb) i32`, `nbr_mask (ncap,kb) f32`, `nbr_feat (ncap,kb,4) f32` (slot 0 = self with the mean of the node's incoming bond features, zeros when isolated), `lig_xyz (ncap,3) f32`, `env_xyz (mcap,3) f32`, `env_species (mcap,) i32`, `aev_nbr (ncap,ka) i32`, `aev_mask (ncap,ka) f32`, `y (batch,) f32` (standardised), `valid (batch,) f32`, `n_real: int` (host-only metadata, not passed to jit).
- Produces: `assemble(records: list[Record], cap: Capacities, y_mean: float, y_sd: float) -> Batch` (raises `ValueError` naming the id when any capacity is exceeded; padding by copies as in Global Constraints).
- Produces: `StoreSource(store: Store, cap: Capacities, y_mean: float, y_sd: float, workers: int)` implementing `BatchSource` (`load(spec: np.ndarray) -> Batch` fetching records in a thread pool, `close`, `provenance`).

- [ ] **Step 1: Write the failing tests**: `test_plan_epoch_is_permutation_and_deterministic` (same seed/epoch -> same plan; all ids once; chunk sizes 128 except the last), `test_self_loop_feature_is_mean_of_incoming` (a node with single+double bonds gets `[0.5, 0, 0.5, 0]` in slot 0; isolated node gets zeros), `test_padding_is_copies_with_zero_valid` (padded node rows equal some real row, `node_graph == batch`, valid 0; padded batch slots valid 0), `test_assemble_raises_when_capacity_exceeded` (ka too small -> ValueError with the id), `test_capacities_cover_all_plans` (every planned batch of a 50-record synthetic store fits), `test_prefetcher_yields_in_order` (StoreSource over 5 specs with 2 workers).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `loading.py` (copy) and `batching.py`. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** "Add batch planning, capacities and host assembly with the jax-Ka prefetcher".

### Task 7: Model, masked BatchNorm, readout, PyTorch weight port

**Files:**
- Create: `jax/src/aevplig_jax/model.py`, `jax/src/aevplig_jax/torch_port.py`, `jax/tests/test_model.py`, `jax/tests/test_torch_port.py`.

**Interfaces:**
- Produces: `init_params(key, node_dim: int = 367, edge_dim: int = 4, hidden: int = 256, heads: int = 3, layers: int = 5) -> dict`: `params["gat"][i] = {"w_l": (in, heads*hidden), "b_l": (heads*hidden,), "w_r", "b_r", "w_e": (edge_dim, heads*hidden), "att": (heads, hidden), "bias": (heads*hidden,), "bn": {"scale", "shift"}}`; `params["fc1"|"fc2"|"fc3"|"out"] = {"w", "b"}`; `params["bn1"|"bn2"|"bn3"] = {"scale", "shift"}`. Initialisers: xavier normal for `w_l`, `w_r`, `w_e`, `fc*.w`, `out.w`; `att` glorot uniform with bound `sqrt(6 / (heads + hidden))`; `b_l`, `b_r`, `fc*.b`, `out.b` uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`; `bias` zeros; BatchNorm scale 1, shift 0. Heads other than the affinity head are not created here; `params["heads"] = {}` is reserved and `init_head(key, name, salt, shape)` adds one from `fold_in(key, salt)`.
- Produces: `init_bn_state(layers: int = 5) -> dict` of running mean/var (zeros/ones) for the five layer norms and the three dense norms.
- Produces: `gatv2_layer(p: dict, x, nbr, nbr_mask, nbr_feat, heads: int, hidden: int) -> jnp.ndarray` of shape `(ncap, heads*hidden)`: `hl = x @ w_l + b_l`, `hr = x @ w_r + b_r`, `he = nbr_feat @ w_e`; per head `e = leaky_relu_0.2(hl[nbr] + hr[:, None] + he)`, logits `(e * att).sum(-1)` masked to `-1e9`, softmax over the K axis, message `sum_k alpha * hl[nbr]`, concat heads, `+ bias`.
- Produces: `masked_batchnorm(p, state, x, valid, train: bool, momentum: float = 0.1, eps: float = 1e-5) -> tuple[jnp.ndarray, dict]` (statistics over rows with `valid > 0`; biased variance for the batch, unbiased for the running update as torch does; running statistics used when `train` is False).
- Produces: `encode(params, state, batch, train: bool) -> tuple[node_emb (ncap,768), graph_emb (batch,1536), state]` (five layers each `gatv2_layer` -> leaky_relu 0.01 -> masked BatchNorm -> `* node_valid`; pooling by `jax.ops.segment_max` and `segment_sum / count` with `num_segments = batch + 1` and the dump segment dropped); `readout(params, state, graph_emb, valid, train) -> tuple[pred (batch,), state]`; `predict(params, state, batch, train) -> tuple[pred, state]`; `aev_features(batch) -> jnp.ndarray (ncap, 367)` = concat of `x_lig` and `radial_aev_jax(...)`.
- Produces: `torch_port.py`: `port_state_dict(sd: dict[str, np.ndarray]) -> tuple[params, bn_state]` mapping `GNN_layers.{i}.lin_l.weight` (transpose), `lin_l.bias`, `lin_r.*`, `lin_edge.weight`, `att` (reshape `(1,3,256)` -> `(3,256)`), `bias`, `BN_layers.{i}.module.{weight,bias,running_mean,running_var}`, `fc{1,2,3}.*`, `bn_connect{1,2,3}.*`, `out.*`; `load_torch_model(path: Path) -> dict[str, np.ndarray]` via `torch.load(map_location="cpu")` (torch imported lazily).

- [ ] **Step 1: Write the failing tests** (torch-dependent ones under the `torch_available` marker): `test_gatv2_matches_pyg` (random graph with 12 nodes and 4-dim edge features; PyG `GATv2Conv(367, 256, heads=3, edge_dim=4)` with its weights copied into `p`; the neighbour-list form from Task 6's self-loop convention; max abs diff < 1e-5), `test_masked_batchnorm_matches_torch_train_and_eval` (padded rows excluded; running stats after one step match `BatchNorm1d` on the real rows), `test_isolated_node_graph_runs` (single-atom ligand gives a finite prediction), `test_padding_does_not_change_prediction` (same graphs assembled with ncap 256 and 512 give identical predictions to 1e-6), `test_param_count_matches_pytorch` (count equals `GATv2Net`'s parameter count from `model_defs.py`, computed as a number in the test), `test_port_round_trip_predictions` (a freshly initialised `GATv2Net` saved and ported; predictions on three random graphs match the torch model in eval mode to 1e-4 with `jax.default_matmul_precision("highest")`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `model.py` and `torch_port.py`. **Step 4: Run** — PASS with torch installed, and PASS under `pytest -m "not torch"` (what `selftest.sbatch` runs in the jax-Ka venv).
- [ ] **Step 5: Commit** "Add the GATv2 model, masked BatchNorm, readout and the PyTorch weight port".

### Task 8: Training loop, prediction and evaluation

**Files:**
- Create: `jax/src/aevplig_jax/train.py`, `jax/src/aevplig_jax/predict.py`, `jax/src/aevplig_jax/evaluate.py`, `jax/src/aevplig_jax/precompute_aev.py`, `jax/jobs/train.sbatch`, `jax/jobs/eval.sbatch`, `jax/jobs/selftest.sbatch`, `jax/tests/test_train_smoke.py`, `jax/tests/test_evaluate.py`.

**Interfaces:**
- Produces: `train.py` CLI `python -m aevplig_jax.train --store-root DIR --tag T --seed-index I --epochs 200 --out DIR [--aev onthefly|precomputed] [--precision default|highest] [--workers 15] [--prefetch 4] [--limit N]`. Behaviour: ids and labels from `trainside` splits `train`/`valid`/`test`; `y_mean`, `y_sd` from the training labels (population sd, as `StandardScaler`); `Capacities` from Task 6 over the ten seeds and the epoch count, written to `<out>/<tag>/manifest.json` once and reused; `optax.adam(LR)`; the loss is built from `TERMS: dict[str, tuple[float, Callable]]` (`{"mse": (1.0, mse_term)}` here; later heads add entries) and every term's value is logged in `history.jsonl` while selection reads the validation Pearson only; three jitted functions `loss_and_grad(params, bn_state, batch) -> ((loss, aux_terms, bn_state), grads)`, `apply(params, opt_state, grads, lr) -> (params, opt_state)`, `predict_fn(params, bn_state, batch) -> pred`; per epoch: all training batches via `Prefetcher`, then validation Pearson in evaluation mode, `history.jsonl` row (epoch, mean loss, valid Pearson, seconds, loader wait fraction), checkpoint `checkpoints/best/state.npz` replaced when the mean of the last `SELECTION_WINDOW` Pearsons exceeds the best, `checkpoints/latest` every epoch for resume; at the end, best weights predict the test split and write `<out>/predictions/<tag>_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_test_seed<I>.csv` with columns `truth, preds_<I>` in the test split's id order (the parent's `ensemble_eval.py` format) plus `unique_id`; `protocol.json` records every flag, constants and the sha256 of the package files; `selection.json` records the chosen epoch.
- Produces: `precompute_aev.py --store DIR --out FILE.npy`: the 352 columns for every record in store order via `radial_aev_jax`, float32, with an offsets array; `StoreSource` reads it when `--aev precomputed`.
- Produces: `predict.py --store DIR --checkpoints GLOB --out CSV [--manifest FILE]`: ensemble over checkpoints; capacities planned from the target store itself; columns `unique_id, truth, preds_0.., preds` (mean).
- Produces: `evaluate.py casf --predictions CSV` (Pearson, RMSE, as `helpers.py`), `evaluate.py fep --predictions CSV --fep-csv FILE` (weighted PCC, weighted Kendall and RMSE over series with >= 10 ligands, the definitions in `paired/src/eval_paired.py`; import that module's function when its interface fits, otherwise reimplement and keep a test that both agree on a fixture), `evaluate.py pgdh --predictions CSV --pgdh-csv FILE` (pooled and per-series Spearman).
- Produces: `train.sbatch` (array `0-9`, `$JAX_PY`, `--tag jax`, runs from `$TREE`), `eval.sbatch` (predict on `test`, `fep`, `pgdh` with the ten best checkpoints; evaluate; write `work/output/eval/<tag>/SUMMARY.md`), `selftest.sbatch` (pytest under `$JAX_PY` with torch tests skipped, then `train.py --limit 512 --epochs 2` on the real store as a GPU smoke).

- [ ] **Step 1: Write the failing tests**: `test_train_smoke_two_epochs` (synthetic store of 40 records with random labels, `--epochs 2 --limit 40`, CPU: history has 2 rows, `checkpoints/best` exists, prediction CSV has the test ids), `test_selection_rule_window_of_eight` (feed a Pearson sequence through the selection function and check which epochs save), `test_resume_continues_from_latest`, `test_evaluate_casf_matches_helpers` (fixture arrays, Pearson/RMSE equal to the formulas in `helpers.py`), `test_evaluate_fep_weighted_metrics` (two series of 10 and 12 ligands with known correlations), `test_predict_plans_capacities_from_target_store` (a prediction store with a ligand larger than any training ligand still predicts).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** the four modules and the three job scripts. **Step 4: Run** — PASS; `bash -n` on the job scripts.
- [ ] **Step 5: Commit** "Add the JAX training loop, prediction, evaluation and job scripts".

### Task 9: Port parity and profiling on Isambard

**Files:**
- Create: `jax/src/aevplig_jax/profile.py`, `jax/jobs/port_parity.sbatch`, `jax/jobs/profile_torch.py`, `jax/jobs/profile_torch.sbatch`, `jax/jobs/profile_jax.sbatch`, `jax/tests/test_profile.py`.

**Interfaces:**
- Produces: `profile.py --store-root DIR --manifest FILE --steps 300 --trace-steps 20 --out DIR [--aev onthefly|precomputed]`: compiles, then times `loss_and_grad` + `apply` per step with `block_until_ready` (mean, p50, p95), loader wait fraction from `LoaderTelemetry`, padding fractions (`node_valid.mean()`, `aev_mask` fill, env fill), a `jax.profiler` trace of the trace steps, and a text table of the top 15 ops from the trace (`xprof`/`tensorboard_plugin_profile` are not in the venv: parse the trace's `.xplane.pb` only if `xprof` imports; otherwise report per-step timings and the compiled HLO op counts from `jitted.lower(...).compile().as_text()`), written to `<out>/profile.json` and `<out>/REPORT.md`.
- Produces: `profile_torch.py` (runs under `$AEVPLIG_PY` from the clone's work directory, read-only): the paper's loop for 3 epochs on seed 0 with per-epoch timing and `torch.profiler` over 200 steps (CPU and CUDA activities), once with the paper's `DataLoader` and once with `num_workers=8, pin_memory=True`; writes `profile_torch.json` (seconds per epoch, data vs compute split) into `$TREE/work/output/profile/torch/`.
- Produces: `port_parity.sbatch`: `torch_port.load_torch_model` on `$PARENT/output/trained_models/retrain_model_GATv2Net_..._0.model` (the parent's retrained seed 0; name pattern from `isambard/README.md`), JAX predictions on `test` and `fep` stores under `--precision highest`, PyTorch predictions from the parent's prediction CSVs (`$PARENT/output/predictions/`) or recomputed with the clone's `predict` if the CSVs lack FEP; writes `work/output/parity/REPORT.md` with max abs differences (acceptance 1e-3 pK).
- Produces: `profile_jax.sbatch`: `profile.py` for `onthefly` and `precomputed` (after `precompute_aev.py` on `trainside`), `nvidia-smi dmon -s u -d 5` sampled in the background to a file, then one full seed (`train.py --seed-index 0 --tag timing`) timed end to end; `work/output/profile/REPORT.md` with the table from the spec (PyTorch, PyTorch workers, JAX precomputed, JAX on-the-fly: seconds per epoch, minutes per seed, GPU utilisation, loader wait, padding fractions) and the op summary.

- [ ] **Step 1: Write the failing test** `test_profile_report_table` (given a `profile.json` fixture, `profile.render_report` produces the table with the four rows and the acceptance line "on-the-fly overhead: x%").
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `profile.py`, `profile_torch.py` and the three job scripts. **Step 4: Run** — PASS; `bash -n`.
- [ ] **Step 5: Commit** "Add profiling for the PyTorch and JAX loops and the weight-port parity job".
- [ ] **Step 6: Isambard**: mirror; submit `selftest.sbatch`, then `port_parity.sbatch`, `profile_torch.sbatch` and `profile_jax.sbatch` (the two profiles may run in parallel). Sonnet monitor collects `parity/REPORT.md` and `profile/REPORT.md`. Expected: parity max abs < 1e-3 pK on CASF and FEP; a seed under 10 minutes; loader wait < 5%; padding waste < 25%; on-the-fly overhead < 10%. A miss is written into the report with the measured cause and the next lever; the user decides before Task 10.
- [ ] **Step 7: Commit** the reports under `jax/docs/` ("Record port parity and profiling results").

### Task 10: Ten-seed reproduction, write-up

**Files:**
- Modify: `jax/docs/REPORT_reproduction.md`, `jax/README.md`; memory note `aevplig-jax-rewrite.md` (new) in the user's memory directory with paths, versions, numbers and job ids.

- [ ] **Step 1: Isambard**: submit `train.sbatch` (array 0-9, job name `aj-train`) and `eval.sbatch` with `--dependency=afterok`. Sonnet monitor collects `work/output/eval/jax/SUMMARY.md`.
- [ ] **Step 2: Compare** against acceptance 3 of the spec: ensemble CASF Pearson within 0.005 of 0.892 and RMSE within 0.02 of 1.041; FEP weighted PCC within 0.02 of 0.589 and Kendall within 0.02 of 0.429; per-seed validation Pearsons against the PyTorch seeds' values from the parent's logs (`$S/aevplig/logs/`, the array 8073760 task logs, "validation set Pearson" lines). Write the table and the verdict into `REPORT_reproduction.md`; a miss is reported as a miss with the strict-float32 rerun as the first follow-up.
- [ ] **Step 3: Write** `jax/README.md` (layout, environments, commands, timings) and the memory note; mirror the reports back to the fork tree.
- [ ] **Step 4: Commit** "Record the ten-seed JAX reproduction" and report the commit list to the user for message verification; no push.
