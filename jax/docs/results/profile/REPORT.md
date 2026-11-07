# AEV-PLIG loop profile

PyTorch: the paper's loop (`jobs/profile_torch.py`), seconds per epoch from the timed epochs (train + validation) after the first; workers: `num_workers=8, pin_memory=True, persistent_workers=True`. JAX: `profile.py`, seconds per epoch estimated from the timed steps (steps per epoch x wall per step + one validation pass, start-up excluded); the full seed below is measured end to end. GPU utilisation: mean `nvidia-smi dmon` SM % over each row's timed window (5 s samples). Padding: share of padded node rows, environment rows and AEV-list slots.

| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding | env padding | AEV-slot padding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | 17.1 | 57.0 | 44.2% | 24.7% | n/a | n/a | n/a |
| PyTorch workers | 14.6 | 48.8 | 46.7% | 2.4% | n/a | n/a | n/a |
| JAX precomputed | 5.1 | 17.0 | 0.0% | 0.1% | 20.7% | 24.4% | 98.9% |
| JAX on-the-fly | 5.4 | 17.9 | 1.0% | 0.1% | 20.7% | 24.4% | 98.9% |

on-the-fly overhead: 6.7% (mean step 6.15 ms on the fly vs 5.77 ms precomputed, same batches)

The precomputed A/B measures compute only: the environment arrays are still assembled and transferred in both modes, and the precomputed batches also carry the 352 AEV columns.

## Targets

| target | measured | met |
| --- | --- | --- |
| seed at most 10 min (200 epochs with validation) | 13.8 min (measured) | no |
| at least 5x the PyTorch loop | 4.1x | no |
| loader wait under 5.0% | 0.1% | yes |
| node padding under 25.0% | 20.7% | yes |
| on-the-fly overhead under 10.0% | 6.7% | yes |

## Full seed (measured)

`/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/timing/seed-0`: 13.8 min wall end to end (start-up, 200 epochs with validation, test prediction); history: train 791 s + validation 9 s (4.00 s per epoch); mean loader wait 0.1%; GPU utilisation 88.5%.

## JAX on-the-fly

Step (`loss_and_grad` + `apply`, `block_until_ready`): mean 6.15 ms, p50 6.15 ms, p95 6.27 ms over 300 steps after 5 warm-up steps (first step 0.9 s with compilation); timed segment 2.13 s; validation pass 0.05 s; 748 steps per epoch; capacities ncap 5120, mcap 14336, kb 7, ka 736; trace of 20 steps in `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/onthefly` (xplane.pb and perfetto_trace.json.gz).

Top ops: op counts of the compiled HLO of loss_and_grad (no times: xprof is not installed).

| op | value |
| --- | --- |
| fusion:kInput | 104 |
| fusion:kLoop | 103 |
| fusion:kCustom:__triton | 21 |
| fusion:kCustom:__triton_nested_gemm_fusion | 20 |
| custom-call:__cublas$lt$matmul | 18 |

## JAX precomputed

Step (`loss_and_grad` + `apply`, `block_until_ready`): mean 5.77 ms, p50 5.73 ms, p95 6.01 ms over 300 steps after 5 warm-up steps (first step 2.0 s with compilation); timed segment 2.03 s; validation pass 0.05 s; 748 steps per epoch; capacities ncap 5120, mcap 14336, kb 7, ka 736; trace of 20 steps in `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/profile/trace/precomputed` (xplane.pb and perfetto_trace.json.gz).

Top ops: op counts of the compiled HLO of loss_and_grad (no times: xprof is not installed).

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
