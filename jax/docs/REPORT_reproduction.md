# AEV-PLIG in JAX: reproduction report

Design: `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md`. Plan: `docs/superpowers/plans/2026-11-07-aevplig-jax-rewrite.md`. Isambard tree `/scratch/u6xq/hussain.u6xq/aevplig-jax` (`$TREE`). The result files quoted below are copied to `jax/docs/results/`.

## Summary

The JAX rewrite consists of a single-file record store of the 97,857 training-side, 1,184 FEP-benchmark and 43 15-PGDH complexes, radial AEVs computed on the GPU inside the training step, the vanilla GATv2Net and its training loop, a PyTorch weight port, and profiles of both loops. Strict float32 (`PRECISION=highest`) reproduces the paper on every ensemble acceptance number (CASF 0.8921 / 1.043, FEP weighted PCC 0.5833, Kendall 0.4150) at 22 min per seed; TF32, JAX's float32 default on Hopper, trains in 14 min per seed but its FEP ensemble lands just outside the band (0.5659 / 0.4013). Neither meets the 10-minute target. The per-kernel traces show a GPU-bound step, not a launch-bound one: kernels keep the GPU busy about 93% of a 5.3 ms TF32 training step. 68% of that kernel time is elementwise, reduction, transpose and scatter fusions and 29% matrix multiplies; at least half runs in kernels launched once per GAT layer, whose neighbour slots are 64% padding. Loader wait is 0.1% and the on-the-fly AEV adds 4-7%. The store is now at schema 2 (per-atom element and raw residue ids) and training, prediction and the parity check refuse a store whose verification is missing, failed, partial or stale.

## Store (Task 5, 2026-11-07)

Build job 8180200 (`jax/jobs/build_store.sbatch`, code at a820f1c): 2 min 23 s wall (92 s of building after a 16 s extraction of 59 GB to node-local disk), peak host memory 2.3 GB, zero read failures.

| store | records | splits | records.bin | max dims [L, E, P, A, KA] |
| --- | --- | --- | --- | --- |
| pdbbind | 19,308 | train 17,904 / valid 993 / test 277 / excluded 134 | 126 MB | |
| bindingnet | 69,727 | train 69,194 / excluded 533 | 577 MB | |
| bindingdb | 8,822 | train 8,582 / excluded 240 | 60 MB | |
| trainside (merged) | 97,857 | train 95,680 / valid 993 / test 277 / excluded 907 | 769 MB | [224, 494, 6348, 1826, 731] |
| fep | 1,184 | predict | 7.6 MB | [53, 116, 561, 140, 32] |
| pgdh | 43 | predict | 0.3 MB | [34, 76, 433, 101, 33] |

The split counts equal the paper's dataset files (95,680 / 993 / 277). `KA = 731` comes from multi-model NMR entries: the original `LoadPDBasDF` sums ATOM lines over every MODEL, and the store keeps the same atoms so that the AEVs match. BindingNet template merge: 7,146 directories linked, 4 without a template (none of them in the processed table). RDKit warned about missing explicit hydrogens on about 50 PDBbind and BindingDB mol2 ligands, as it does in the original pipeline.

Verification job 8180508 (`jax/jobs/verify_store.sbatch`, code at d6de8e1; JAX on the GPU, pickles loaded with numpy 2.5): 2 min 17 s wall, peak host memory 14 GB (the three training pickles).

| store | compared | failed | missing in store | missing in pickle | worst id | max abs AEV diff | padded capacities L / K / A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trainside | 97,857 | 0 | 0 | 0 | 5ief | 2.34e-5 | 224 / 736 / 1856 |
| fep | 1,184 | 0 | 0 | 0 | merck/pfkfb3_automap/47 | 8.22e-6 | 64 / 32 / 160 |
| pgdh | 43 | 0 | 0 | 0 | pgdh/S4/compound_46 | 6.57e-7 | 64 / 64 / 128 |

Every record also matched the pickled RDKit columns, bond lists and heavy-atom counts exactly. Acceptance 1 of the design (AEV parity below 1e-4, no missing complexes) is met. One deprecation warning per run: the pickles reference `numpy.core.numeric`, which numpy 2.5 maps to `numpy._core` with a warning.

Rebuild (Task 8 fix, code at f711fe5): the generic FEP/PGDH builds now read the CSV metadata as text (`ref_pdb` values such as "3E63" had become floats). Build 8183208 rebuilt all six stores (1 min 30 s wall, same counts) and verification 8183209 passed every record with the same worst differences; training, parity and evaluation used these stores.

Schema 2 (fix wave, code at 4e9ce8e). Build 8186473 rebuilt all six stores with four per-atom fields appended: `prot_element` (int8 code into `metadata.json`'s `elements`: other, C, N, O, S, P, H, SE, four halogens and twelve metals; from PDB columns 76:78, else from the atom name), `prot_chain_id` (the raw chain character), `prot_resseq` (int32) and `prot_icode` (0 when blank). Residues can now be matched across records of one receptor and joined to per-residue tables computed on the full structure; the first-appearance serials `prot_res` and `prot_chain` could do neither. The twelve schema-1 fields are unchanged and `Store` refuses schema-1 stores. Counts and dims are as above; trainside `records.bin` is 804 MB (769 MB at schema 1), FEP 8.0 MB, PGDH 0.3 MB. Verification 8186476 passed every record with the same worst differences (2.34e-5, 8.22e-6, 6.57e-7) and now records each store's `store_fingerprint`: the size of `records.bin` and the sha256 of `index.npz` (`results/store/{trainside,fep,pgdh}/verification.json`, `results/store/trainside/metadata.json`). `train.py`, `predict.py` and `port_parity.py` refuse a store whose `verification.json` is missing, failed, partial (fewer records compared than stored) or older than its files; `--skip-verification` (train and predict only) is for synthetic stores and smoke runs.

Species-0 census of trainside (8186476, `store_census.py`; `results/store/census-trainside.txt`). These are the pocket atoms kept with species 0 because their residue-atom key is not in the 22-type table:

| source | pocket atoms | species 0 | hydrogens (atom code -1) | heavy atoms |
| --- | --- | --- | --- | --- |
| PDBbind | 7,532,837 | 0 | 0 | 0 |
| BindingNet | 38,251,163 | 9,616,283 (25.1%) | 9,495,945 | 120,338 with known atom names: 72,335 C, 36,161 N, 11,784 O, 58 S |
| BindingDB | 3,527,445 | 2,410 (0.07%) | 1,382 | 1,028 with unknown atom names: 505 C, 247 N, 248 O, 28 P |

The BindingNet hydrogens are old-style names with H second (such as `1HA`), which the original hydrogen rule keeps; the BindingNet receptors are the hydrogen-optimised `rec_h_opt.pdb` templates. Its heavy atoms carry known atom names under residue names the key table lacks; protonation-state residue names from the receptor preparation are the likely source, but the census does not split by residue. (`results/store/census-trainside.txt` is the full table from the verify log.) None of these atoms entered the original AEV: `LoadPDBasDF`'s inner merge with the key table dropped every unmapped key, and the AEV ignores species 0, so parity holds. A pocket encoder has to filter by `prot_element` (drop hydrogens) and decide what to do with the unmapped heavy atoms.

## Weight-port parity (Task 9, job 8183723)

The parent's retrained seed-0 PyTorch model (`retrain_model_GATv2Net_..._0.model`; scaler y_mean 6.971045, y_sd 1.549093) was exported under the AEV-PLIG env, ported with `torch_port.port_state_dict` and run in evaluation mode at matmul precision `highest` on the GPU, against column `pred_0` of `$PAIRED/output/eval/retrain10/predictions_{test,fep}.csv`. Thresholds: max abs difference below 1e-3 pK on every set, every row compared, truths equal to 1e-5 (`results/parity/REPORT.md`, `parity.csv`).

| set | rows ours / reference / compared | truth equal | truth by row | max abs | mean abs | worst | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CASF | 277 / 277 / 277 | yes | yes | 1.19e-5 | 1.76e-6 | 2wnc | yes |
| FEP | 1184 / 1184 / 1184 | yes | yes | 3.08e-5 | 1.40e-6 | merck/pfkfb3_automap/39 | yes |

"Truth by row" compares labels in file order: the store's test-split order is the reference's, which the per-seed prediction CSVs (no ids) rely on. Acceptance 2 is met with a margin above 30x.

## Profile (Task 9, jobs 8183724, 8183725, 8184629; fix wave 8186480)

PyTorch: the paper's loop run read-only from `$PARENT` (8183724, AEV-PLIG env, torch 2.6.0, PyG 2.5.3). JAX: `profile.py` on the on-the-fly and precomputed AEV paths over the same batches, then one full TF32 seed timed end to end (8183725); 8184629 reran the profiles only (`SKIP_SEED=1`) with the corrected HLO op summary. `results/profile/REPORT.md` is the rerun's render, `REPORT-8183725.md` the first. 8186480 (fix wave, code at 4e9ce8e) reran the profiles only, now with per-kernel summaries of the traces and the GAT-slot padding; its render, `profile.json` and the per-kernel tables are in `results/output/profile/` (`REPORT.md`, `trace/{onthefly,precomputed}/kernels.csv`). The table from 8184629's `REPORT.md`:

| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding | env padding | AEV-slot padding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | 17.1 | 57.0 | 44.2% | 24.7% | n/a | n/a | n/a |
| PyTorch workers | 14.6 | 48.8 | 46.7% | 2.4% | n/a | n/a | n/a |
| JAX precomputed | 5.1 | 17.0 | 0.0% | 0.1% | 20.7% | 24.4% | 98.9% |
| JAX on-the-fly | 5.4 | 17.9 | 1.0% | 0.1% | 20.7% | 24.4% | 98.9% |

The JAX rows are estimates from the timed profile segment (300 steps, each blocked with `block_until_ready`). The timing seed, which does not block per step, ran 5.29 ms per step (790.9 s of training over 200 x 748 steps): 13.8 min measured against 17.9 min estimated. The difference is the per-step block: in 8186480 the blocked segment takes 7.1 ms per step (2.12 s over 300 steps, 6.1 ms of it inside the step call) against 5.29 ms per step in the training loop, for 5.0 ms of kernels (below). 8186480 gives 5.2 s / 17.2 min precomputed and 5.3 s / 17.8 min on the fly, with the same node, environment and AEV-slot padding, and adds GAT-slot padding 64.0% for both modes.

| target (design G4/G5) | measured | met |
| --- | --- | --- |
| seed at most 10 min (200 epochs with validation) | 13.8 min (measured) | no |
| at least 5x the PyTorch loop | 4.1x | no |
| loader wait under 5.0% | 0.1% | yes |
| node padding under 25.0% | 20.7% | yes |
| on-the-fly overhead under 10.0% | 6.7% | yes |

The 4.1x divides the PyTorch estimate (57.0 min from three timed epochs) by the JAX wall time measured end to end, start-up and test prediction included; the parent's array 8073760 took 54-58 min per seed. Strict float32 takes 22.0-22.8 min per seed (2.6x).

PyTorch profiler, 200 steps with the paper's loader:

| phase | seconds | share |
| --- | --- | --- |
| data | 1.48 | 21.9% |
| forward | 3.03 | 44.8% |
| backward | 1.94 | 28.7% |
| optimiser | 0.31 | 4.6% |

In the steady epochs the paper's loader waits 4.2 s per epoch; eight workers with pinned memory cut that to 0.35 s and the epoch from 17.1 to 14.6 s (`torch/profile_torch.json`), so most of the PyTorch epoch is the GPU loop at 44-47% utilisation.

JAX step statistics (`loss_and_grad` + `apply`, `block_until_ready`, 300 steps after 5 warm-up steps, same batches in both modes; capacities ncap 5120, mcap 14336, kb 7, ka 736, batch 128; 748 steps per epoch; validation pass 0.05 s):

| AEV path | job | mean | p50 | p95 | first step (compile) |
| --- | --- | --- | --- | --- | --- |
| on the fly | 8184629 | 6.15 ms | 6.15 ms | 6.27 ms | 0.9 s |
| precomputed | 8184629 | 5.77 ms | 5.73 ms | 6.01 ms | 2.0 s |
| on the fly | 8183725 | 6.09 ms | 6.09 ms | 6.24 ms | 1.1 s |
| precomputed | 8183725 | 5.86 ms | 5.82 ms | 6.08 ms | 1.9 s |
| on the fly | 8186480 | 6.12 ms | 6.12 ms | 6.21 ms | 1.0 s |
| precomputed | 8186480 | 5.83 ms | 5.80 ms | 6.04 ms | 1.0 s |

On-the-fly overhead: 4.0% in the first run (6.09 vs 5.86 ms), 6.7% in the rerun (6.15 vs 5.77 ms), 5.0% in 8186480 (6.12 vs 5.83 ms). The spread across the three runs is measurement noise; all are under 10%. The precomputed path still assembles and transfers the environment arrays, so the A/B measures device compute only.

Static view, op counts of the compiled HLO of `loss_and_grad` (launch-level ops only, fusion bodies not counted; the separate `apply` step not included; counts only, no times, because xprof is not installed; these five kinds are the whole list). Counting the command-buffer call bodies (c2f9709) left the numbers unchanged from the first render.

| op | on the fly | precomputed |
| --- | --- | --- |
| fusion:kInput | 104 | 103 |
| fusion:kLoop | 103 | 100 |
| fusion:kCustom:__triton | 21 | 21 |
| fusion:kCustom:__triton_nested_gemm_fusion | 20 | 19 |
| custom-call:__cublas$lt$matmul | 18 | 18 |
| total | 266 | 261 |

GPU utilisation (`nvidia-smi dmon` SM %, 5 s samples): PyTorch 44.2%, PyTorch with workers 46.7%, JAX timing seed 88.5% over its 13.8 min. The JAX profile rows (0.0% and 1.0% here, 65.0% and 0.0% in the first render, 0.0% and none in 8186480) average 5 s samples over 2 s windows and carry no information.

Per-kernel time (8186480). `profile.py` sums the GPU events of each mode's 20-step perfetto trace (kernels and host-to-device copies on the device's stream lines); `results/output/profile/trace/<mode>/kernels.csv` lists every kernel. The profiler runs XLA op by op (it switches command buffers off while a trace is active) and each traced step is blocked, so the traced window and its idle share describe the profiled step, not the training loop. The kernel durations are the training loop's.

| | on the fly | precomputed |
| --- | --- | --- |
| GPU events per step: `loss_and_grad` / `apply` / host-to-device copies | 348: 270 / 65 / 13 | 343: 265 / 65 / 13 |
| summed kernel time per step | 4.99 ms | 4.51 ms |
| traced window per step | 7.56 ms | 7.40 ms |
| GPU busy, share of the traced window (union of kernel intervals) | 65.0% | 59.9% |
| kernel time in `jit_loss_and_grad` / `jit_apply` / rest (copies) | 94.6% / 3.1% / 2.3% | 93.5% / 3.4% / 3.1% |
| GAT slot fill on the real rows (of kb = 7) | 45.4% | 45.4% |
| GAT-slot padding (share of ncap x kb) | 64.0% | 64.0% |

The 270 `loss_and_grad` kernels per step match the 266 launch-level ops of its HLO. Kernel time by class, grouped by kernel name from `kernels.csv`:

| class (kernel names) | on the fly | precomputed |
| --- | --- | --- |
| matrix multiplies (`sm90_xmma`, `nvjet`, cutlass, XLA Triton `gemm_fusion_dot*`) | 28.6% | 26.1% |
| elementwise loop fusions (`loop_*`) | 21.6% | 19.9% |
| reductions (`*reduce*`) | 17.3% | 18.6% |
| transposes (`*transpose*`) | 14.9% | 15.6% |
| unnamed fusions (`fusion_N`) | 7.7% | 8.2% |
| scatters (`*scatter*`) | 6.7% | 7.9% |
| copies (`Memcpy*`, `memcpy*`) | 2.4% | 3.1% |
| other (concatenates, a divide) | 0.8% | 0.6% |

By launches per step: names launched five times per step (once per GAT layer) take 52.0% of the kernel time on the fly and 57.0% precomputed; names launched four times take another 16.9% and 18.6%.

Top 10 kernels on the fly (20 traced steps):

| kernel | launches per step | ms per step | share |
| --- | --- | --- | --- |
| `input_add_transpose_fusion_4` | 5 | 0.513 | 10.3% |
| `loop_add_fusion_4` | 5 | 0.367 | 7.4% |
| `input_scatter_fusion` | 5 | 0.281 | 5.6% |
| `gemm_fusion_dot_general_51` | 1 | 0.265 | 5.3% |
| `fusion_472` | 5 | 0.255 | 5.1% |
| `sm90_xmma_gemm_..._nt_n_tilesize128x256x32` (cuBLAS, TF32) | 4 | 0.255 | 5.1% |
| `sm90_xmma_gemm_..._nn_n_tilesize64x128x32` (cuBLAS, TF32) | 4 | 0.227 | 4.6% |
| `loop_add_fusion_9` | 6 | 0.214 | 4.3% |
| `input_transpose_fusion_5` | 5 | 0.210 | 4.2% |
| `gemm_fusion_dot_47` | 5 | 0.173 | 3.5% |

Top 10 kernels precomputed:

| kernel | launches per step | ms per step | share |
| --- | --- | --- | --- |
| `input_add_transpose_fusion_4` | 5 | 0.511 | 11.3% |
| `loop_add_fusion_4` | 5 | 0.367 | 8.1% |
| `input_scatter_fusion` | 5 | 0.302 | 6.7% |
| `sm90_xmma_gemm_..._nt_n_tilesize128x256x32` (cuBLAS, TF32) | 4 | 0.256 | 5.7% |
| `fusion_451` | 5 | 0.255 | 5.7% |
| `sm90_xmma_gemm_..._nn_n_tilesize64x128x32` (cuBLAS, TF32) | 4 | 0.232 | 5.2% |
| `loop_add_fusion_9` | 6 | 0.216 | 4.8% |
| `gemm_fusion_dot_55` | 5 | 0.178 | 3.9% |
| `input_transpose_fusion_5` | 5 | 0.171 | 3.8% |
| `input_reduce_fusion_1` | 5 | 0.170 | 3.8% |

Where the time goes. The step is GPU-bound. On the fly the traced kernels keep the GPU busy 4.91 ms per step (union of intervals), against 5.29 ms per step in the unblocked timing seed: the GPU works about 93% of a training step, and host launch and dispatch gaps are at most about 7% (0.4 ms). The seed's 88.5% `nvidia-smi` utilisation agrees once its start-up and test prediction (831 s wall, 791 s of training) count as idle. The blocked profile step (6.1 ms in the step call, 7.1 ms of segment wall, for 5.0 ms of kernels) and the traced window (65% busy) carry host round trips and op-by-op dispatch that the training loop does not have; this is also why the profile estimates 17.8 min per seed where the seed measures 13.8. Of the kernel time, 68% is elementwise, reduction, transpose, scatter and unnamed fusions, kinds that are memory-bound, and 29% is matrix multiplies. At least half runs in kernels launched once per GAT layer (52% on the fly), led by `input_add_transpose_fusion_4` (10.3%), `loop_add_fusion_4` (7.4%) and `input_scatter_fusion` (5.6%; by name and count most likely the backward of the neighbour gather). The GAT layers work on `(ncap, kb, heads, hidden)` = (5120, 7, 3, 256) slot tensors of which 64% are padding (45.4% fill of kb = 7 on the real rows, about 3.2 slots, plus the 20.7% padded rows). Which HLO op each fusion implements is not mapped yet: the kernel names are XLA fusion names, and an `XLA_FLAGS=--xla_dump_to` dump would map them. The share of slot-proportional work is therefore an inference from launch counts and names. The on-the-fly AEV shows as `gemm_fusion_dot_general_51`, launched once per step and only on the fly (0.27 ms per step against the measured 0.29 ms overhead; likely the species one-hot contraction). AEV-list slots are 98.9% padding because ka = 736 is sized by multi-model NMR entries (KA max 731 in trainside), but that gather costs at most the measured 4-7%. The loader is not the limit (wait 0.1%). The earlier reading of this section, a step bound by kernel launch and fusion granularity, was inferred from op counts and step times; the per-kernel times do not support it.

## Ten-seed reproduction (Task 10)

TF32: array 8183375 and eval 8183376 (tag `jax`, code f711fe5 with the TMPDIR job fix later committed as d41ab2e). Strict float32: array 8184085 and eval 8184086 (tag `jax-fp32`, `PRECISION=highest`, `train.sbatch` at 1cff018). Evaluation precision: 8183376 and 8184086 predicted both tags at the default matmul precision (TF32), so the first strict-float32 ensemble was scored with TF32 matmuls. The fix-wave evals predict at each tag's training precision, read from the seeds' `protocol.json`: 8186478 (`jax`, default, TF32) and 8186479 (`jax-fp32`, highest), in `results/output/eval/`. The float32 row below is unchanged at the digits shown (CASF r 0.892093 against 0.892090, FEP wPCC 0.58329 against 0.58325, wKendall 0.41498 against 0.41499; per-seed CASF r within 2.8e-5), and the TF32 re-evaluation reproduces 8183376 within 1e-6. Both use the trainside store (train 95,680 / valid 993 / test 277), the paper's seeds [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621], batch 128, 200 epochs and the capacities above (`results/{jax,jax-fp32}/manifest.json`). FEP metrics are weighted over the 43 series with at least 10 ligands (948 of 1,184); `evaluate.py` reproduces the paired tree's reference summary exactly from the reference predictions (0.5904 / 0.5599 / 0.4277 / 0.986, checked locally on `retrain10/predictions_fep.csv`).

| run | CASF r | CASF RMSE | FEP wPCC | FEP wSpearman | FEP wKendall | FEP wRMSE | per-seed CASF r, mean (range) | per-seed valid r, mean (range) | min / seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch (paper release) (a) | 0.8929 | 1.039 | 0.5898 | 0.5512 | 0.4158 | 0.981 | n/a | n/a | n/a |
| PyTorch retrain (parent, 10 seeds) | 0.8919 | 1.041 | 0.5904 | 0.5599 | 0.4277 | 0.986 | 0.8769 (0.8615-0.8889) | 0.8127 (0.8081-0.8177) (b) | 54-58 |
| JAX TF32 (`jax`) | 0.8874 | 1.056 | 0.5659 | 0.5274 | 0.4013 | 0.996 | 0.8729 (0.8644-0.8809) | 0.8094 (0.8050-0.8166) | 13.9-15.3 |
| JAX strict float32 (`jax-fp32`) | 0.8921 | 1.043 | 0.5833 | 0.5481 | 0.4150 | 0.991 | 0.8766 (0.8701-0.8856) | 0.8102 (0.8034-0.8194) | 22.0-22.8 |

(a) The released models scored by the paired tree's code (`$PAIRED/output/eval/released-check/`, whose FEP predictions match the paper's published ones within 2.1e-6 pK). The design's targets (CASF 0.892 / 1.041, FEP 0.589 / 0.429) come from two other sources: CASF from the parent retrain (0.8919 / 1.0407), FEP from the paired tree's base-g 10-member ensemble (0.5885 / 0.4285). The acceptance verdict below uses the design's targets. (b) PyTorch: the maximum of each seed's 200 per-epoch validation Pearsons (parent logs `aevplig-train-8073760_<i>.out`). JAX: the validation Pearson at the selected epoch, which is at most that seed's maximum. The two are not like-for-like; the JAX per-seed maxima, read afterwards from `history.jsonl` on Isambard, are: TF32 0.8089-0.8188 (mean 0.8119), strict float32 0.8077-0.8194 (mean 0.8129), against PyTorch 0.8081-0.8177 (mean 0.8127). PyTorch per-seed CASF RMSE 1.054-1.132 (mean 1.085). PGDH pooled Spearman: TF32 0.095, float32 0.123 (noise level for every variant).

Per seed (`results/{jax,jax-fp32}/seed-<i>/selection.json`; selection = best running mean of the last eight validation Pearsons; CASF from the selected checkpoint, agreeing with the eval job's re-prediction to 3e-5):

| i | seed | TF32 epoch | valid r | 8-epoch mean | CASF r | CASF RMSE | fp32 epoch | valid r | 8-epoch mean | CASF r | CASF RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100 | 151 | 0.8166 | 0.8146 | 0.8697 | 1.109 | 199 | 0.8075 | 0.8045 | 0.8749 | 1.088 |
| 1 | 123 | 183 | 0.8155 | 0.8130 | 0.8801 | 1.059 | 193 | 0.8106 | 0.8096 | 0.8849 | 1.058 |
| 2 | 15 | 199 | 0.8063 | 0.8059 | 0.8688 | 1.111 | 176 | 0.8123 | 0.8120 | 0.8717 | 1.106 |
| 3 | 257 | 174 | 0.8066 | 0.8058 | 0.8761 | 1.083 | 196 | 0.8112 | 0.8097 | 0.8856 | 1.063 |
| 4 | 2 | 188 | 0.8050 | 0.8046 | 0.8644 | 1.126 | 167 | 0.8039 | 0.8044 | 0.8701 | 1.113 |
| 5 | 2012 | 199 | 0.8149 | 0.8119 | 0.8803 | 1.067 | 197 | 0.8134 | 0.8141 | 0.8766 | 1.083 |
| 6 | 3752 | 199 | 0.8085 | 0.8081 | 0.8809 | 1.080 | 199 | 0.8115 | 0.8111 | 0.8769 | 1.092 |
| 7 | 350 | 199 | 0.8050 | 0.8062 | 0.8646 | 1.129 | 181 | 0.8034 | 0.8064 | 0.8782 | 1.087 |
| 8 | 843 | 184 | 0.8073 | 0.8059 | 0.8720 | 1.100 | 197 | 0.8087 | 0.8052 | 0.8755 | 1.085 |
| 9 | 621 | 196 | 0.8084 | 0.8095 | 0.8722 | 1.120 | 119 | 0.8194 | 0.8145 | 0.8719 | 1.097 |
| mean | | | 0.8094 | 0.8085 | 0.8729 | 1.098 | | 0.8102 | 0.8091 | 0.8766 | 1.087 |

The GPU loop is not bit-reproducible: the timing run (8183725, same training code, seed index 0, TF32) selected epoch 191 (valid 0.8104, CASF 0.8640 / 1.134) where `jax` seed 0 selected epoch 151 (0.8166, 0.8697 / 1.109). Seed-level comparisons are statistical.

Acceptance (design section 8):

1. Store: met. AEV max abs difference 2.34e-5 (trainside), 8.22e-6 (FEP), 6.57e-7 (PGDH) against 1e-4; RDKit columns and bonds identical; no missing complexes. The schema-2 rebuild (8186473) verified with the same differences (8186476), and the design's verification gate is now enforced: `train.py`, `predict.py` and `port_parity.py` refuse a store whose `verification.json` is missing, failed, partial or older than the store's files (`store_fingerprint`).
2. Weight-port parity: met. Max abs 1.19e-5 pK (CASF) and 3.08e-5 pK (FEP) against 1e-3.
3. Reproduction: met in strict float32 (CASF r 0.8921 for 0.892 ± 0.005, RMSE 1.043 for 1.041 ± 0.02, FEP wPCC 0.5833 for 0.589 ± 0.02, Kendall 0.4150 for 0.429 ± 0.02); not met in TF32 (CASF 0.8874 / 1.056 inside the band, FEP wPCC 0.5659 at -0.023 and Kendall 0.4013 at -0.028 outside; against the paper release's Kendall 0.4158 the gap is 0.015). The per-seed validation clause is read here as agreement of the per-seed distributions, not as every JAX seed falling inside the PyTorch spread. Comparing per-seed maximum validation Pearsons, the means agree within 0.0002 in float32 (0.8129 vs PyTorch 0.8127) and 0.0008 in TF32 (0.8119), and the spreads overlap: float32 0.8077-0.8194 and TF32 0.8089-0.8188 against PyTorch 0.8081-0.8177. The float32 range extends past PyTorch's at both ends and TF32's at the top, so the literal clause (all ten within the PyTorch spread) is not met by either. The per-seed CASF mean matches in float32 (0.8766 vs PyTorch 0.8769).
4. Speed: not met. A seed takes 13.8 min (TF32 timing seed; 13.9-15.3 in the array) or 22.0-22.8 min (float32) against 10 min, and 4.1x against 5x. Loader wait 0.1% (target 5%), node padding 20.7% (25%) and on-the-fly overhead 4.0-6.7% (5.0% in 8186480; 10%) are met. Cause under Profile: a GPU-bound step, most of it memory-bound fusions in the GAT layers over 64%-padded neighbour slots. Levers below.
5. Tests: met. 121 passed locally at c2f9709 (`jax/.venv` with torch 2.15.0 CPU and torch_geometric 2.5.3, 42 s). Isambard selftest 8183210 at f711fe5 (jax-Ka venv, CPU, `-m "not torch"`): 67 passed, 31 skipped, 5 deselected, then a 2-epoch GPU smoke on 512 records. The 18 tests added after f711fe5 (profile, parity) have run locally only. Fix wave: 153 passed locally at 4e9ce8e (same venv, 42 s); Isambard selftest 8186477 at 4e9ce8e: 101 passed, 32 skipped, 5 deselected, then the GPU smoke on the verified schema-2 store.

## Precision

| | TF32 (`jax`) | strict float32 (`jax-fp32`) | TF32 - float32 |
| --- | --- | --- | --- |
| ensemble CASF r | 0.8874 | 0.8921 | -0.0047 |
| ensemble CASF RMSE | 1.056 | 1.043 | +0.013 |
| FEP wPCC | 0.5659 | 0.5833 | -0.0174 |
| FEP wKendall | 0.4013 | 0.4150 | -0.0137 |
| per-seed CASF r, mean | 0.8729 | 0.8766 | -0.0037 |
| per-seed valid r at selection, mean | 0.8094 | 0.8102 | -0.0008 |
| training s / epoch | 3.8-4.3 | 6.11-6.16 | |
| min / seed | 13.9-15.3 | 22.0-22.8 | |

All four TF32 ensemble numbers are worse than float32's, by 0.005 to 0.017 (FEP wSpearman by 0.021). That sits at the edge of what 10-member ensembles resolve. The design takes ensembles of ten to agree to about 0.01-0.02 (paired-seed sd of FEP wPCC 0.028), while the PyTorch ensembles at hand agree more closely on FEP wPCC: parent retrain 0.5904, released models 0.5898, the paired tree's base-g-10 0.5885 (near-identical data). The per-seed CASF means differ by 0.0037 with per-seed sd 0.0062 (TF32) and 0.0052 (float32), about 1.4 standard errors of the difference. A second ten-seed ensemble per precision would settle it. The user decides the default. The cost of float32: 6.1 s instead of 3.8-4.3 s per training epoch (timing seed 3.95 s), 22 instead of 14 min per seed.

## Next levers

Re-ranked from the per-kernel traces (8186480). Every gain below is an estimate, not a measurement. Reaching 10 min from 13.8 needs about 28% less wall time. The GPU is busy about 93% of a step, so the time has to come out of kernel work, not launch overhead.

1. Cut the GAT slot padding. 64% of the `(ncap, kb)` slots are padding: kb = 7 is set by the largest in-degree plus the self-loop, while real rows fill 45.4% of it, and 20.7% of the rows are padding. Sort or bucket ligand atoms by in-degree (two kb tiers, for example 4 and 7), or size kb from the 99th percentile of in-degree with an overflow path for the rest, together with tighter node capacities. The maths does not change. Map the top fusions to HLO ops first (`XLA_FLAGS=--xla_dump_to`) to size it. Estimate: up to about half of the slot-proportional gather, scatter and transpose work, 10-25% of step time.
2. A fused attention kernel over the K axis (jax-Ka's Triton pattern): logits, masked softmax and the weighted sum per target in one kernel, and a backward kernel in place of the scatter, so the `(5120, 7, 3, 256)` intermediates never go to memory. The existing layer test against PyG checks it. Estimate: 15-30% of step time; the most work on this list. It removes the same slot work as lever 1, so the two gains do not add; 1 is the cheaper first step and sizes 2.
3. bf16 for the elementwise fusions and the stored activations (68% of the kernel time is of memory-bound kinds). Estimate: 10-25%. It changes the numerics: run it as a ten-seed arm judged like TF32 (TF32 already costs 0.017 FEP wPCC).
4. A larger batch (256) halves the steps per epoch, but in a GPU-bound step most kernel work scales with the batch; only the gaps (about 7%), the `apply` step (3% of kernel time) and small-kernel inefficiency amortise. Estimate: 5-15% per epoch. A recipe change, not a reproduction: it needs its own ten-seed check against this report.
5. One jit for `loss_and_grad` + `apply`, with buffer donation of parameters and optimiser state. It removes one dispatch and one parameter-sized copy per step without touching the maths. Ceiling: the launch and dispatch gaps of the unblocked step, about 7%. Estimate: 3-7%.
6. A pair-list `segment_sum` AEV instead of the ka = 736 padded gather. It removes the 98.9% slot padding and the NMR-driven capacity. Gain bounded by the measured AEV overhead, 4.0-6.7% of step time (5.0% in 8186480).

## Files and jobs

| job | name | purpose | outcome |
| --- | --- | --- | --- |
| 8180200 | aj-build | store build (a820f1c) | 2 min 23 s, zero failures |
| 8180508 | aj-verify | store verification (d6de8e1) | 2 min 17 s, all passed |
| 8183054 | aj-selftest | pytest + GPU smoke (2ef0b4f) | failed at collection in 9 s: rdkit absent from the jax-Ka venv |
| 8183208 | aj-build | rebuild with text CSV metadata (f711fe5) | 1 min 30 s, same counts |
| 8183209 | aj-verify | re-verification | all passed |
| 8183210 | aj-selftest | pytest + GPU smoke (f711fe5) | passed in 57 s |
| 8183339 | aj-train | TF32 array, tag `jax` (f711fe5) | failed on all 10 tasks: ptxas could not open its temp file under `/local/user/<uid>` |
| 8183342 | aj-eval | eval of 8183339 | cancelled |
| 8183375 | aj-train | TF32 array, tag `jax` (TMPDIR on scratch) | 13.9-15.3 min per seed |
| 8183376 | aj-eval | ensemble eval, tag `jax` (default precision) | 35 s; CASF 0.8874, FEP 0.5659 / 0.4013 |
| 8183723 | aj-parity | weight-port parity (a6fc1e3) | passed |
| 8183724 | aj-prof-torch | PyTorch profile | 17.1 s / epoch |
| 8183725 | aj-prof-jax | JAX profiles + timing seed | 13.8 min seed; `REPORT-8183725.md` |
| 8184085 | aj-train | strict float32 array, tag `jax-fp32` (1cff018) | 22.0-22.8 min per seed |
| 8184086 | aj-eval | ensemble eval, tag `jax-fp32` (predicted at default precision) | CASF 0.8921, FEP 0.5833 / 0.4150 |
| 8184629 | aj-prof-jax | profiles only (`SKIP_SEED=1`, c2f9709) | `REPORT.md` |
| 8186473 | aj-build | schema-2 rebuild (fix wave, 4e9ce8e) | same counts; trainside `records.bin` 804 MB |
| 8186476 | aj-verify | re-verification with store fingerprints, trainside species-0 census | all passed, same worst differences |
| 8186477 | aj-selftest | pytest + GPU smoke (4e9ce8e) | 101 passed, 32 skipped, 5 deselected; smoke passed |
| 8186478 | aj-eval | ensemble eval, tag `jax`, default precision (TF32) | CASF 0.887 / 1.056, FEP 0.566 / 0.401 |
| 8186479 | aj-eval | ensemble eval, tag `jax-fp32`, precision highest | CASF 0.892 / 1.043, FEP 0.583 / 0.415 |
| 8186480 | aj-prof-jax | profiles only (`SKIP_SEED=1`) with per-kernel tables (4e9ce8e) | `output/profile/REPORT.md` |

On Isambard: stores in `$TREE/work/store/{pdbbind,bindingnet,bindingdb,trainside,fep,pgdh}` (logs in `store/logs`). Under `$TREE/work/output`: `{jax,jax-fp32,timing}/seed-<i>/` (protocol.json, history.jsonl, selection.json, test_predictions.csv, checkpoints/{best,latest}), `{jax,jax-fp32,timing}/manifest.json`, `predictions/`, `eval/{jax,jax-fp32}/`, `parity/`, `profile/` (profile.json, REPORT.md, gpu_util.txt, `torch/`, `trace/`), `aev_trainside.npz`, `logs/`. Copied to `jax/docs/results/`: `parity/`, `profile/`, `eval/{jax,jax-fp32}/` (SUMMARY.md and the summary CSVs), `{jax,jax-fp32}/manifest.json` and `seed-<i>/selection.json`, `timing/seed-0/selection.json`. Fix wave: `store/{trainside,fep,pgdh}/verification.json`, `store/trainside/metadata.json`, `store/census-trainside.txt` (the census also sits in `$TREE/work/store/logs/aj-census-8186476-trainside.json`), `output/eval/{jax,jax-fp32}/` (SUMMARY.md, CASF and FEP summaries) and `output/profile/` (REPORT.md, profile.json, `trace/{onthefly,precomputed}/kernels.csv`).
