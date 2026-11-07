# AEV-PLIG in JAX: reproduction report

Design: `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md`. Plan: `docs/superpowers/plans/2026-11-07-aevplig-jax-rewrite.md`. Isambard tree `/scratch/u6xq/hussain.u6xq/aevplig-jax` (`$TREE`). The result files quoted below are copied to `jax/docs/results/`.

## Summary

The JAX rewrite consists of a single-file record store of the 97,857 training-side, 1,184 FEP-benchmark and 43 15-PGDH complexes, radial AEVs computed on the GPU inside the training step, the vanilla GATv2Net and its training loop, a PyTorch weight port, and profiles of both loops. Strict float32 (`PRECISION=highest`) reproduces the paper on every ensemble acceptance number (CASF 0.8921 / 1.043, FEP weighted PCC 0.5833, Kendall 0.4150) at 22 min per seed; TF32, JAX's float32 default on Hopper, trains in 14 min per seed but its FEP ensemble lands just outside the band (0.5659 / 0.4013). Neither meets the 10-minute target: a step runs about 270 small fused kernels in about 6 ms, while loader wait is 0.1% and the on-the-fly AEV adds under 7%.

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

## Weight-port parity (Task 9, job 8183723)

The parent's retrained seed-0 PyTorch model (`retrain_model_GATv2Net_..._0.model`; scaler y_mean 6.971045, y_sd 1.549093) was exported under the AEV-PLIG env, ported with `torch_port.port_state_dict` and run in evaluation mode at matmul precision `highest` on the GPU, against column `pred_0` of `$PAIRED/output/eval/retrain10/predictions_{test,fep}.csv`. Thresholds: max abs difference below 1e-3 pK on every set, every row compared, truths equal to 1e-5 (`results/parity/REPORT.md`, `parity.csv`).

| set | rows ours / reference / compared | truth equal | truth by row | max abs | mean abs | worst | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CASF | 277 / 277 / 277 | yes | yes | 1.19e-5 | 1.76e-6 | 2wnc | yes |
| FEP | 1184 / 1184 / 1184 | yes | yes | 3.08e-5 | 1.40e-6 | merck/pfkfb3_automap/39 | yes |

"Truth by row" compares labels in file order: the store's test-split order is the reference's, which the per-seed prediction CSVs (no ids) rely on. Acceptance 2 is met with a margin above 30x.

## Profile (Task 9, jobs 8183724, 8183725, 8184629)

PyTorch: the paper's loop run read-only from `$PARENT` (8183724, AEV-PLIG env, torch 2.6.0, PyG 2.5.3). JAX: `profile.py` on the on-the-fly and precomputed AEV paths over the same batches, then one full TF32 seed timed end to end (8183725); 8184629 reran the profiles only (`SKIP_SEED=1`) with the corrected HLO op summary. `results/profile/REPORT.md` is the rerun's render, `REPORT-8183725.md` the first. The table from `REPORT.md`:

| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding | env padding | AEV-slot padding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | 17.1 | 57.0 | 44.2% | 24.7% | n/a | n/a | n/a |
| PyTorch workers | 14.6 | 48.8 | 46.7% | 2.4% | n/a | n/a | n/a |
| JAX precomputed | 5.1 | 17.0 | 0.0% | 0.1% | 20.7% | 24.4% | 98.9% |
| JAX on-the-fly | 5.4 | 17.9 | 1.0% | 0.1% | 20.7% | 24.4% | 98.9% |

The JAX rows are estimates from the timed profile segment (300 steps, each blocked with `block_until_ready`). The timing seed, which does not block per step, ran 5.29 ms per step (790.9 s of training over 200 x 748 steps): 13.8 min measured against 17.9 min estimated.

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

On-the-fly overhead: 4.0% in the first run (6.09 vs 5.86 ms), 6.7% in the rerun (6.15 vs 5.77 ms). The difference between the two runs is measurement noise; both are under 10%. The precomputed path still assembles and transfers the environment arrays, so the A/B measures device compute only.

Op counts of the compiled HLO of `loss_and_grad` (launch-level ops only, fusion bodies not counted; the separate `apply` step not included; counts only, no times, because xprof is not installed; these five kinds are the whole list). Counting the command-buffer call bodies (c2f9709) left the numbers unchanged from the first render.

| op | on the fly | precomputed |
| --- | --- | --- |
| fusion:kInput | 104 | 103 |
| fusion:kLoop | 103 | 100 |
| fusion:kCustom:__triton | 21 | 21 |
| fusion:kCustom:__triton_nested_gemm_fusion | 20 | 19 |
| custom-call:__cublas$lt$matmul | 18 | 18 |
| total | 266 | 261 |

GPU utilisation (`nvidia-smi dmon` SM %, 5 s samples): PyTorch 44.2%, PyTorch with workers 46.7%, JAX timing seed 88.5% over its 13.8 min. The JAX profile rows (0.0% and 1.0% here, 65.0% and 0.0% in the first render) average 5 s samples over 2 s windows and carry no information.

Where the time goes. One step runs about 270 kernels from `loss_and_grad` alone (104 kInput and 103 kLoop fusions, 21 Triton and 20 Triton nested-GEMM fusions, 18 cuBLAS matmuls) in about 6 ms, about 22 µs per kernel on average. The loader is not the limit (wait 0.1%), nor is the AEV (overhead under 7%). Node padding is 20.7% and environment padding 24.4%. AEV-list slots are 98.9% padding because ka = 736 is sized by multi-model NMR entries (KA max 731 in trainside), but that gather costs at most the measured 4-7%. The loop is bound by kernel launch and fusion granularity, not by the loader or the AEV. This is inferred from op counts and step times, not from per-kernel times; the 20-step traces (`$TREE/work/output/profile/trace/{onthefly,precomputed}/perfetto_trace.json.gz`) hold those and open in Perfetto.

## Ten-seed reproduction (Task 10)

TF32: array 8183375 and eval 8183376 (tag `jax`, code f711fe5 with the TMPDIR job fix later committed as d41ab2e). Strict float32: array 8184085 and eval 8184086 (tag `jax-fp32`, `PRECISION=highest`, `train.sbatch` at 1cff018). Both use the trainside store (train 95,680 / valid 993 / test 277), the paper's seeds [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621], batch 128, 200 epochs and the capacities above (`results/{jax,jax-fp32}/manifest.json`). FEP metrics are weighted over the 43 series with at least 10 ligands (948 of 1,184); `evaluate.py` reproduces the paired tree's reference summary exactly from the reference predictions (0.5904 / 0.5599 / 0.4277 / 0.986, checked locally on `retrain10/predictions_fep.csv`).

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

1. Store: met. AEV max abs difference 2.34e-5 (trainside), 8.22e-6 (FEP), 6.57e-7 (PGDH) against 1e-4; RDKit columns and bonds identical; no missing complexes.
2. Weight-port parity: met. Max abs 1.19e-5 pK (CASF) and 3.08e-5 pK (FEP) against 1e-3.
3. Reproduction: met in strict float32 (CASF r 0.8921 for 0.892 ± 0.005, RMSE 1.043 for 1.041 ± 0.02, FEP wPCC 0.5833 for 0.589 ± 0.02, Kendall 0.4150 for 0.429 ± 0.02); not met in TF32 (CASF 0.8874 / 1.056 inside the band, FEP wPCC 0.5659 at -0.023 and Kendall 0.4013 at -0.028 outside; against the paper release's Kendall 0.4158 the gap is 0.015). The per-seed validation clause is met like-for-like: the per-seed maximum validation Pearson spans 0.8077-0.8194 in float32 (mean 0.8129) and 0.8089-0.8188 in TF32 (mean 0.8119) against PyTorch's 0.8081-0.8177 (mean 0.8127), and the per-seed CASF mean matches in float32 (0.8766 vs PyTorch 0.8769).
4. Speed: not met. A seed takes 13.8 min (TF32 timing seed; 13.9-15.3 in the array) or 22.0-22.8 min (float32) against 10 min, and 4.1x against 5x. Loader wait 0.1% (target 5%), node padding 20.7% (25%) and on-the-fly overhead 4.0-6.7% (10%) are met. Cause under Profile, levers below.
5. Tests: met. 121 passed locally at c2f9709 (`jax/.venv` with torch 2.15.0 CPU and torch_geometric 2.5.3, 42 s). Isambard selftest 8183210 at f711fe5 (jax-Ka venv, CPU, `-m "not torch"`): 67 passed, 31 skipped, 5 deselected, then a 2-epoch GPU smoke on 512 records. The 18 tests added after f711fe5 (profile, parity) have run locally only.

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

Ranked by expected gain for effort. Every gain below is an estimate, not a measurement. Reaching 10 min from 13.8 needs about 28% less wall time.

1. One jit for `loss_and_grad` + `apply`, with buffer donation of parameters and optimiser state and fewer host round trips. This removes one dispatch and one parameter-sized copy per step without touching the maths. Estimate: 5-10% of step time.
2. XLA flags and CUDA graphs (wider command-buffer coverage, fusion settings) for the ~270 small kernels. Estimate: 10-30% if launch overhead dominates; size it first from the per-kernel times in the existing Perfetto traces.
3. A larger batch (256) halves the steps per epoch, and a launch-bound step should grow less than 2x. Estimate: 1.5-1.8x per epoch. This is a recipe change, not a reproduction, and needs its own ten-seed check against this report.
4. bf16 matmuls. Estimate: under 15%, since matmul-type ops (18 cuBLAS calls, at most 41 Triton fusions) are a minority of the ~270 kernels. Run as a ten-seed arm judged like TF32 (TF32 already costs 0.017 FEP wPCC).
5. A pair-list `segment_sum` AEV instead of the ka = 736 padded gather. It removes the 98.9% slot padding and the NMR-driven capacity. Gain bounded by the measured AEV overhead, 4-7% of step time.
6. Two-tier capacities (node padding 20.7%, env padding 24.4%). Estimate: a few % at most, since padded rows add work but not kernels.

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
| 8183376 | aj-eval | ensemble eval, tag `jax` | 35 s; CASF 0.8874, FEP 0.5659 / 0.4013 |
| 8183723 | aj-parity | weight-port parity (a6fc1e3) | passed |
| 8183724 | aj-prof-torch | PyTorch profile | 17.1 s / epoch |
| 8183725 | aj-prof-jax | JAX profiles + timing seed | 13.8 min seed; `REPORT-8183725.md` |
| 8184085 | aj-train | strict float32 array, tag `jax-fp32` (1cff018) | 22.0-22.8 min per seed |
| 8184086 | aj-eval | ensemble eval, tag `jax-fp32` | CASF 0.8921, FEP 0.5833 / 0.4150 |
| 8184629 | aj-prof-jax | profiles only (`SKIP_SEED=1`, c2f9709) | `REPORT.md` |

On Isambard: stores in `$TREE/work/store/{pdbbind,bindingnet,bindingdb,trainside,fep,pgdh}` (logs in `store/logs`). Under `$TREE/work/output`: `{jax,jax-fp32,timing}/seed-<i>/` (protocol.json, history.jsonl, selection.json, test_predictions.csv, checkpoints/{best,latest}), `{jax,jax-fp32,timing}/manifest.json`, `predictions/`, `eval/{jax,jax-fp32}/`, `parity/`, `profile/` (profile.json, REPORT.md, gpu_util.txt, `torch/`, `trace/`), `aev_trainside.npz`, `logs/`. Copied to `jax/docs/results/`: `parity/`, `profile/`, `eval/{jax,jax-fp32}/` (SUMMARY.md and the summary CSVs), `{jax,jax-fp32}/manifest.json` and `seed-<i>/selection.json`, `timing/seed-0/selection.json`.
