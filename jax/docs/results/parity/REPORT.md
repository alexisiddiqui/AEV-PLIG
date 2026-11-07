# Weight-port parity

Ported `/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/parity/torch_seed0.npz` (state dict and scaler: y_mean 6.971045, y_sd 1.549093) with `torch_port.port_state_dict`; JAX predictions in evaluation mode at matmul precision highest on cuda:0, against column `pred_0` of `/scratch/u6xq/hussain.u6xq/aevplig-paired/work/output/eval/retrain10/predictions_*.csv`. Acceptance: max abs difference below 0.001 pK on every set, every row compared, truths equal (to 1e-05).

| set | join | rows ours / reference / compared | truth equal | truth by row | max abs | mean abs | worst | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casf | unique_id | 277 / 277 / 277 | yes | yes | 1.19e-05 | 1.76e-06 | 2wnc | yes |
| fep | unique_id | 1184 / 1184 / 1184 | yes | yes | 3.08e-05 | 1.4e-06 | merck/pfkfb3_automap/39 | yes |

Verdict: PASSED.

`truth by row` compares the labels in file order: yes shows the store's split order is the reference's (needed for the per-seed CSVs, which carry no ids).
