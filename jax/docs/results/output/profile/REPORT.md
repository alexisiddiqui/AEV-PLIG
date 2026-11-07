# AEV-PLIG loop profile

PyTorch: the paper's loop (`jobs/profile_torch.py`), seconds per epoch from the timed epochs (train + validation) after the first; workers: `num_workers=8, pin_memory=True, persistent_workers=True`. JAX: `profile.py`, seconds per epoch estimated from the timed steps (steps per epoch x wall per step + one validation pass, start-up excluded); the full seed below is measured end to end. GPU utilisation: mean `nvidia-smi dmon` SM % over each row's timed window (5 s samples). Padding: share of padded node rows, environment rows, AEV-list slots and GAT-list slots (the masked share of each layer's `(ncap, kb, heads, hidden)` gather).

| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding | env padding | AEV-slot padding | GAT-slot padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | 17.1 | 57.0 | 44.2% | 24.7% | n/a | n/a | n/a | n/a |
| PyTorch workers | 14.6 | 48.8 | 46.7% | 2.4% | n/a | n/a | n/a | n/a |
| JAX precomputed | 5.2 | 17.2 | 0.0% | 0.1% | 20.7% | 24.4% | 98.9% | 64.0% |
| JAX on-the-fly | 5.3 | 17.8 | n/a | 0.1% | 20.7% | 24.4% | 98.9% | 64.0% |

on-the-fly overhead: 5.0% (mean step 6.12 ms on the fly vs 5.83 ms precomputed, same batches)

The precomputed A/B measures compute only: the environment arrays are still assembled and transferred in both modes, and the precomputed batches also carry the 352 AEV columns.

## Targets

| target | measured | met |
| --- | --- | --- |
| seed at most 10 min (200 epochs with validation) | 13.8 min (measured) | no |
| at least 5x the PyTorch loop | 4.1x | no |
| loader wait under 5.0% | 0.1% | yes |
| node padding under 25.0% | 20.7% | yes |
| on-the-fly overhead under 10.0% | 5.0% | yes |

## Where the time goes

Measured: the per-kernel tables of each JAX row below (time by kernel, family and HLO module from the perfetto trace, and the GPU busy share of the traced window); the HLO op counts are static and carry no times. JAX on-the-fly: GPU busy 65.0% of the traced window, 348.0 kernels per step; JAX precomputed: GPU busy 59.9% of the traced window, 343.0 kernels per step.

Conclusion: pending the per-kernel table.

## Full seed (measured)

`/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/timing/seed-0`: 13.8 min wall end to end (start-up, 200 epochs with validation, test prediction); history: train 791 s + validation 9 s (4.00 s per epoch); mean loader wait 0.1%; GPU utilisation 88.5%.

## JAX on-the-fly

Step (`loss_and_grad` + `apply`, `block_until_ready`): mean 6.12 ms, p50 6.12 ms, p95 6.21 ms over 300 steps after 5 warm-up steps (first step 1.0 s with compilation); timed segment 2.12 s; validation pass 0.05 s; 748 steps per epoch; capacities ncap 5120, mcap 14336, kb 7, ka 736; GAT slot fill 45.4% of kb on the real rows; trace of 20 steps in `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/onthefly` (xplane.pb and perfetto_trace.json.gz).

Kernel time, measured (GPU kernel events of the perfetto trace (ph X on the device stream lines; dur in us), summed over the traced steps; `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/onthefly/plugins/profile/2026_11_07_23_15_42/perfetto_trace.json.gz`, lines /device:GPU:0 Stream #14(Memset,Compute), /device:GPU:0 Stream #15(MemcpyH2D)): 6960 kernel events over 20 traced steps (348.0 per step); trace window 151.12 ms (7.556 ms per step), GPU busy 65.0% of it (union of kernel intervals), idle 52.83 ms; summed kernel time 99.81 ms (4.991 ms per step). The traced steps block on the parameters after each step, as the timed ones do, and XLA runs them op by op unless `XLA_FLAGS` has `--xla_enable_command_buffers_during_profiling=true` (by default profiling turns command buffers off), so the idle share bounds the untraced step's launch gaps from above. Every kernel: `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/onthefly/kernels.csv`.

Top 15 kernels by summed time:

| kernel | count | total ms | share |
| --- | --- | --- | --- |
| input_add_transpose_fusion_4 | 100 | 10.251 | 10.3% |
| loop_add_fusion_4 | 100 | 7.342 | 7.4% |
| input_scatter_fusion | 100 | 5.612 | 5.6% |
| gemm_fusion_dot_general_51 | 20 | 5.306 | 5.3% |
| fusion_472 | 100 | 5.098 | 5.1% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_of... | 80 | 5.091 | 5.1% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x128x32_warpgroupsize1x1x1_execute_segment_k_off... | 80 | 4.547 | 4.6% |
| loop_add_fusion_9 | 120 | 4.287 | 4.3% |
| input_transpose_fusion_5 | 100 | 4.204 | 4.2% |
| gemm_fusion_dot_47 | 100 | 3.462 | 3.5% |
| input_reduce_fusion_1 | 100 | 3.412 | 3.4% |
| gemm_fusion_dot_21 | 100 | 3.253 | 3.3% |
| nvjet_sm90_sss_tf32_128x128_32x6_4x2_h_bz_TNT | 160 | 3.221 | 3.2% |
| input_reduce_fusion_19 | 100 | 2.874 | 2.9% |
| input_reduce_fusion_8 | 100 | 2.862 | 2.9% |

By family (trailing instance numbers stripped):

| family | count | total ms | share |
| --- | --- | --- | --- |
| loop_add_fusion | 1540 | 14.655 | 14.7% |
| input_reduce_fusion | 1480 | 12.207 | 12.2% |
| input_add_transpose_fusion | 100 | 10.251 | 10.3% |
| gemm_fusion_dot | 380 | 9.039 | 9.1% |
| fusion | 340 | 7.720 | 7.7% |
| input_scatter_fusion | 180 | 6.696 | 6.7% |
| gemm_fusion_dot_general | 20 | 5.306 | 5.3% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_of... | 80 | 5.091 | 5.1% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x128x32_warpgroupsize1x1x1_execute_segment_k_off... | 80 | 4.547 | 4.6% |
| input_transpose_fusion | 120 | 4.352 | 4.4% |
| nvjet_sm90_sss_tf32_128x128_32x6_4x2_h_bz_TNT | 160 | 3.221 | 3.2% |
| loop_multiply_fusion | 260 | 2.734 | 2.7% |
| MemcpyH2D | 260 | 2.346 | 2.4% |
| loop_compare_fusion | 20 | 1.974 | 2.0% |
| input_add_reduce_fusion | 100 | 1.940 | 1.9% |

By HLO module: `jit_loss_and_grad` 5400 kernels, 94.41 ms (94.6%); `jit_apply` 1300 kernels, 3.05 ms (3.1%).

HLO op counts (static fallback, no kernel times): op counts of the compiled HLO of loss_and_grad (no times: xprof is not installed).

| op | value |
| --- | --- |
| fusion:kInput | 104 |
| fusion:kLoop | 103 |
| fusion:kCustom:__triton | 21 |
| fusion:kCustom:__triton_nested_gemm_fusion | 20 |
| custom-call:__cublas$lt$matmul | 18 |

## JAX precomputed

Step (`loss_and_grad` + `apply`, `block_until_ready`): mean 5.83 ms, p50 5.80 ms, p95 6.04 ms over 300 steps after 5 warm-up steps (first step 1.0 s with compilation); timed segment 2.05 s; validation pass 0.05 s; 748 steps per epoch; capacities ncap 5120, mcap 14336, kb 7, ka 736; GAT slot fill 45.4% of kb on the real rows; trace of 20 steps in `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/precomputed` (xplane.pb and perfetto_trace.json.gz).

Kernel time, measured (GPU kernel events of the perfetto trace (ph X on the device stream lines; dur in us), summed over the traced steps; `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/precomputed/plugins/profile/2026_11_07_23_15_58/perfetto_trace.json.gz`, lines /device:GPU:0 Stream #14(Memset,Compute), /device:GPU:0 Stream #15(MemcpyH2D)): 6860 kernel events over 20 traced steps (343.0 per step); trace window 148.00 ms (7.400 ms per step), GPU busy 59.9% of it (union of kernel intervals), idle 59.28 ms; summed kernel time 90.22 ms (4.511 ms per step). The traced steps block on the parameters after each step, as the timed ones do, and XLA runs them op by op unless `XLA_FLAGS` has `--xla_enable_command_buffers_during_profiling=true` (by default profiling turns command buffers off), so the idle share bounds the untraced step's launch gaps from above. Every kernel: `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/precomputed/kernels.csv`.

Top 15 kernels by summed time:

| kernel | count | total ms | share |
| --- | --- | --- | --- |
| input_add_transpose_fusion_4 | 100 | 10.221 | 11.3% |
| loop_add_fusion_4 | 100 | 7.338 | 8.1% |
| input_scatter_fusion | 100 | 6.047 | 6.7% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_of... | 80 | 5.121 | 5.7% |
| fusion_451 | 100 | 5.098 | 5.7% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x128x32_warpgroupsize1x1x1_execute_segment_k_off... | 80 | 4.648 | 5.2% |
| loop_add_fusion_9 | 120 | 4.321 | 4.8% |
| gemm_fusion_dot_55 | 100 | 3.557 | 3.9% |
| input_transpose_fusion_5 | 100 | 3.420 | 3.8% |
| input_reduce_fusion_1 | 100 | 3.394 | 3.8% |
| nvjet_sm90_sss_tf32_128x128_32x6_4x2_h_bz_TNT | 160 | 3.302 | 3.7% |
| gemm_fusion_dot_21 | 100 | 3.265 | 3.6% |
| input_reduce_fusion_19 | 100 | 2.871 | 3.2% |
| input_reduce_fusion | 100 | 2.863 | 3.2% |
| MemcpyH2D | 260 | 2.752 | 3.0% |

By family (trailing instance numbers stripped):

| family | count | total ms | share |
| --- | --- | --- | --- |
| loop_add_fusion | 1540 | 14.682 | 16.3% |
| input_reduce_fusion | 1480 | 12.192 | 13.5% |
| input_add_transpose_fusion | 100 | 10.221 | 11.3% |
| gemm_fusion_dot | 380 | 9.092 | 10.1% |
| fusion | 340 | 7.415 | 8.2% |
| input_scatter_fusion | 180 | 7.146 | 7.9% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_of... | 80 | 5.121 | 5.7% |
| sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x128x32_warpgroupsize1x1x1_execute_segment_k_off... | 80 | 4.648 | 5.2% |
| input_transpose_fusion | 120 | 3.565 | 4.0% |
| nvjet_sm90_sss_tf32_128x128_32x6_4x2_h_bz_TNT | 160 | 3.302 | 3.7% |
| MemcpyH2D | 260 | 2.752 | 3.0% |
| input_add_reduce_fusion | 100 | 1.942 | 2.2% |
| input_multiply_reduce_fusion | 140 | 1.695 | 1.9% |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_tt_align1>(cutlass_80_tensorop_s... | 20 | 1.271 | 1.4% |
| loop_multiply_fusion | 240 | 1.038 | 1.2% |

By HLO module: `jit_loss_and_grad` 5300 kernels, 84.40 ms (93.5%); `jit_apply` 1300 kernels, 3.07 ms (3.4%).

HLO op counts (static fallback, no kernel times): op counts of the compiled HLO of loss_and_grad (no times: xprof is not installed).

| op | value |
| --- | --- |
| fusion:kInput | 103 |
| fusion:kLoop | 100 |
| fusion:kCustom:__triton | 21 |
| fusion:kCustom:__triton_nested_gemm_fusion | 19 |
| custom-call:__cublas$lt$matmul | 18 |

## PyTorch profiler (200 steps, the paper's loader)

| phase | seconds | share |
| --- | --- | --- |
| data | 1.48 | 21.9% |
| forward | 3.03 | 44.8% |
| backward | 1.94 | 28.7% |
| optimiser | 0.31 | 4.6% |

`torch.profiler` table: `torch/key_averages.txt`.
