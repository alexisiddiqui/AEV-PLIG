# JAX AEV-PLIG ensemble: jax-fp32 (10 members, matmul precision highest (strict float32))

| dataset | scope | n | pearson | spearman | kendall | rmse |
|---|---|---|---|---|---|---|
| CASF-2016 | ensemble | 277 | 0.892 |  |  | 1.043 |
| FEP | n>=10 weighted | 948 | 0.583 | 0.548 | 0.415 | 0.991 |
| FEP | n>=10 unweighted | 948 | 0.590 | 0.542 | 0.413 | 1.090 |
| PGDH | pooled | 43 |  | 0.123 |  |  |
| PGDH | S1 | 11 |  | 0.155 |  |  |
| PGDH | S2 | 7 |  | 0.607 |  |  |
| PGDH | S3 | 13 |  | 0.313 |  |  |
| PGDH | S4 | 12 |  | 0.000 |  |  |

- members: 10; CASF member Pearson mean 0.877 (min 0.870, max 0.886)
- FEP: 43 of 91 series have n >= 10 (948 of 1184 ligands)
