# JAX AEV-PLIG ensemble: jax (10 members)

| dataset | scope | n | pearson | spearman | kendall | rmse |
|---|---|---|---|---|---|---|
| CASF-2016 | ensemble | 277 | 0.887 |  |  | 1.056 |
| FEP | n>=10 weighted | 948 | 0.566 | 0.527 | 0.401 | 0.996 |
| FEP | n>=10 unweighted | 948 | 0.577 | 0.530 | 0.408 | 1.094 |
| PGDH | pooled | 43 |  | 0.095 |  |  |
| PGDH | S1 | 11 |  | 0.137 |  |  |
| PGDH | S2 | 7 |  | 0.429 |  |  |
| PGDH | S3 | 13 |  | 0.324 |  |  |
| PGDH | S4 | 12 |  | -0.315 |  |  |

- members: 10; CASF member Pearson mean 0.873 (min 0.864, max 0.881)
- FEP: 43 of 91 series have n >= 10 (948 of 1184 ligands)
