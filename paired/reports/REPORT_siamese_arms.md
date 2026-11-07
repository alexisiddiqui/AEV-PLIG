# Arm comparison (base-g-pw, S, base-g-lig-pw, S-lig; reference base-g-pw; seeds 0,1,2)

## 3-seed ensembles

| metric | base-g-pw | S | base-g-lig-pw | S-lig |
|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.537 | 0.553 | 0.492 | 0.490 |
| FEP wKendall (n>=10) | 0.378 | 0.400 | 0.359 | 0.356 |
| FEP wSpearman (n>=10) | 0.512 | 0.523 | 0.470 | 0.464 |
| FEP wRMSE (all) | 0.986 | 0.958 | 0.978 | 0.974 |
| series_valid wSpearman (n>=5) | 0.380 | 0.373 | 0.352 | 0.366 |
| series_valid wKendall (n>=5) | 0.287 | 0.283 | 0.271 | 0.286 |
| series_valid wRMSE (all) | 0.863 | 0.875 | 0.875 | 0.875 |
| CASF Pearson | 0.880 | 0.891 | 0.823 | 0.818 |
| CASF RMSE | 1.080 | 1.027 | 1.248 | 1.259 |
| group_valid Pearson | 0.584 | 0.571 | 0.497 | 0.491 |
| group_valid Spearman | 0.537 | 0.527 | 0.473 | 0.465 |
| group_valid RMSE | 1.355 | 1.351 | 1.542 | 1.529 |
| group_valid within-series wSpearman (n>=5) | 0.280 | 0.289 | 0.238 | 0.244 |
| FEP pair-head wKendall (n>=10) | nan | 0.406 | nan | 0.367 |
| FEP pair-head wSpearman (n>=10) | nan | 0.530 | nan | 0.480 |
| FEP pair RMSE (n>=10) | nan | 0.903 | nan | 0.929 |
| FEP pair dir acc (n>=10) | nan | 0.774 | nan | 0.743 |
| FEP point-head wKendall (n>=10) | 0.378 | 0.400 | 0.359 | 0.356 |
| FEP point-head wSpearman (n>=10) | 0.512 | 0.523 | 0.470 | 0.464 |
| FEP point pair RMSE (n>=10) | 0.922 | 0.899 | 0.957 | 0.947 |
| FEP point pair dir acc (n>=10) | 0.756 | 0.769 | 0.740 | 0.736 |
| series_valid pair-head wKendall (n>=5) | nan | 0.296 | nan | 0.305 |
| series_valid pair-head wSpearman (n>=5) | nan | 0.391 | nan | 0.392 |
| series_valid pair RMSE (n>=5) | nan | 1.100 | nan | 1.099 |
| series_valid pair dir acc (n>=5) | nan | 0.703 | nan | 0.704 |
| series_valid point-head wKendall (n>=5) | 0.287 | 0.283 | 0.271 | 0.286 |
| series_valid point-head wSpearman (n>=5) | 0.380 | 0.373 | 0.352 | 0.366 |
| series_valid point pair RMSE (n>=5) | 1.061 | 1.066 | 1.097 | 1.087 |
| series_valid point pair dir acc (n>=5) | 0.703 | 0.695 | 0.685 | 0.691 |
| group_valid pair-head wKendall (n>=5) | nan | 0.214 | nan | 0.185 |
| group_valid pair-head wSpearman (n>=5) | nan | 0.295 | nan | 0.253 |
| group_valid pair RMSE (n>=5) | nan | 1.279 | nan | 1.324 |
| group_valid pair dir acc (n>=5) | nan | 0.647 | nan | 0.623 |
| group_valid point-head wKendall (n>=5) | 0.203 | 0.207 | 0.169 | 0.176 |
| group_valid point-head wSpearman (n>=5) | 0.280 | 0.289 | 0.238 | 0.244 |
| group_valid point pair RMSE (n>=5) | 1.239 | 1.237 | 1.355 | 1.370 |
| group_valid point pair dir acc (n>=5) | 0.643 | 0.644 | 0.621 | 0.622 |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.458 | 0.479 | 0.000 | 0.000 |
| Aug kind 4 (A-twin) rank acc | 0.479 | 0.487 | 0.403 | 0.390 |
| Aug kind 4 (A-twin) gap (pK) | 0.254 | 0.290 | 0.065 | 0.081 |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.384 | 0.408 | 0.000 | 0.000 |
| Aug kind 6 (C-contact) rank acc | 0.425 | 0.435 | 0.412 | 0.429 |
| Aug kind 6 (C-contact) gap (pK) | 0.051 | 0.072 | 0.020 | 0.046 |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.330 | 0.338 | 0.346 | 0.397 |

## Single seeds: mean (sd) and per-seed paired difference to the reference

| metric | base-g-pw mean (sd) | S mean (sd) | base-g-lig-pw mean (sd) | S-lig mean (sd) | S - base-g-pw (paired) | base-g-lig-pw - base-g-pw (paired) | S-lig - base-g-pw (paired) |
|---|---|---|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.468 (0.016) | 0.492 (0.013) | 0.426 (0.020) | 0.431 (0.041) | 0.0243 [-0.007, 0.033, 0.047] | -0.0416 [-0.082, -0.018, -0.025] | -0.0364 [-0.092, 0.014, -0.031] |
| FEP wKendall (n>=10) | 0.328 (0.025) | 0.348 (0.007) | 0.302 (0.024) | 0.302 (0.025) | 0.0204 [-0.013, 0.049, 0.025] | -0.0259 [-0.079, 0.019, -0.018] | -0.0251 [-0.073, 0.026, -0.028] |
| FEP wSpearman (n>=10) | 0.441 (0.023) | 0.462 (0.004) | 0.405 (0.034) | 0.403 (0.032) | 0.0207 [-0.008, 0.043, 0.027] | -0.0362 [-0.099, 0.011, -0.021] | -0.0379 [-0.087, 0.017, -0.044] |
| FEP wRMSE (all) | 1.023 (0.011) | 1.003 (0.008) | 1.045 (0.025) | 1.035 (0.019) | -0.0202 [-0.025, -0.017, -0.019] | 0.0216 [0.039, 0.005, 0.021] | 0.0119 [0.020, 0.012, 0.003] |
| series_valid wSpearman (n>=5) | 0.339 (0.024) | 0.336 (0.020) | 0.325 (0.010) | 0.344 (0.025) | -0.0030 [-0.043, 0.017, 0.017] | -0.0147 [-0.049, -0.014, 0.018] | 0.0043 [-0.036, -0.009, 0.058] |
| series_valid wKendall (n>=5) | 0.254 (0.016) | 0.253 (0.015) | 0.249 (0.011) | 0.263 (0.018) | -0.0006 [-0.031, 0.017, 0.011] | -0.0052 [-0.034, -0.001, 0.020] | 0.0093 [-0.020, 0.002, 0.045] |
| series_valid wRMSE (all) | 0.904 (0.003) | 0.915 (0.002) | 0.929 (0.008) | 0.928 (0.021) | 0.0109 [0.007, 0.013, 0.012] | 0.0251 [0.016, 0.022, 0.038] | 0.0243 [0.037, 0.033, 0.003] |
| CASF Pearson | 0.868 (0.004) | 0.878 (0.007) | 0.798 (0.022) | 0.796 (0.009) | 0.0098 [0.010, 0.019, 0.000] | -0.0700 [-0.092, -0.049, -0.069] | -0.0717 [-0.068, -0.080, -0.067] |
| CASF RMSE | 1.113 (0.015) | 1.068 (0.031) | 1.324 (0.074) | 1.324 (0.028) | -0.0450 [-0.034, -0.091, -0.011] | 0.2111 [0.285, 0.141, 0.207] | 0.2111 [0.201, 0.231, 0.201] |
| group_valid Pearson | 0.555 (0.012) | 0.543 (0.016) | 0.463 (0.019) | 0.450 (0.042) | -0.0118 [-0.012, -0.022, -0.002] | -0.0918 [-0.077, -0.117, -0.081] | -0.1040 [-0.044, -0.143, -0.124] |
| group_valid Spearman | 0.505 (0.015) | 0.499 (0.029) | 0.441 (0.018) | 0.432 (0.046) | -0.0057 [-0.018, -0.009, 0.010] | -0.0637 [-0.043, -0.084, -0.064] | -0.0729 [-0.005, -0.097, -0.117] |
| group_valid RMSE | 1.399 (0.016) | 1.395 (0.025) | 1.604 (0.021) | 1.603 (0.108) | -0.0041 [0.007, 0.027, -0.046] | 0.2049 [0.217, 0.231, 0.167] | 0.2040 [0.097, 0.333, 0.182] |
| group_valid within-series wSpearman (n>=5) | 0.238 (0.011) | 0.246 (0.021) | 0.210 (0.015) | 0.208 (0.026) | 0.0089 [-0.017, 0.015, 0.029] | -0.0273 [-0.026, -0.008, -0.048] | -0.0297 [-0.019, -0.046, -0.024] |
| FEP pair-head wKendall (n>=10) | nan (nan) | 0.352 (0.007) | nan (nan) | 0.319 (0.011) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| FEP pair-head wSpearman (n>=10) | nan (nan) | 0.468 (0.008) | nan (nan) | 0.418 (0.016) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| FEP pair RMSE (n>=10) | nan (nan) | 0.989 (0.012) | nan (nan) | 1.012 (0.014) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| FEP pair dir acc (n>=10) | nan (nan) | 0.740 (0.007) | nan (nan) | 0.710 (0.008) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| FEP point-head wKendall (n>=10) | 0.328 (0.025) | 0.348 (0.007) | 0.302 (0.024) | 0.302 (0.025) | 0.0204 [-0.013, 0.049, 0.025] | -0.0259 [-0.079, 0.019, -0.018] | -0.0251 [-0.073, 0.026, -0.028] |
| FEP point-head wSpearman (n>=10) | 0.441 (0.023) | 0.462 (0.004) | 0.405 (0.034) | 0.403 (0.032) | 0.0207 [-0.008, 0.043, 0.027] | -0.0362 [-0.099, 0.011, -0.021] | -0.0379 [-0.087, 0.017, -0.044] |
| FEP point pair RMSE (n>=10) | 0.973 (0.009) | 0.949 (0.008) | 1.041 (0.037) | 1.027 (0.037) | -0.0246 [-0.009, -0.024, -0.042] | 0.0672 [0.117, 0.036, 0.049] | 0.0531 [0.100, 0.019, 0.041] |
| FEP point pair dir acc (n>=10) | 0.720 (0.009) | 0.737 (0.006) | 0.702 (0.010) | 0.701 (0.018) | 0.0165 [0.001, 0.023, 0.026] | -0.0189 [-0.037, -0.002, -0.017] | -0.0197 [-0.042, 0.007, -0.025] |
| series_valid pair-head wKendall (n>=5) | nan (nan) | 0.266 (0.013) | nan (nan) | 0.281 (0.014) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| series_valid pair-head wSpearman (n>=5) | nan (nan) | 0.354 (0.017) | nan (nan) | 0.363 (0.020) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| series_valid pair RMSE (n>=5) | nan (nan) | 1.177 (0.012) | nan (nan) | 1.171 (0.033) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| series_valid pair dir acc (n>=5) | nan (nan) | 0.684 (0.009) | nan (nan) | 0.687 (0.010) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| series_valid point-head wKendall (n>=5) | 0.254 (0.016) | 0.253 (0.015) | 0.249 (0.011) | 0.263 (0.018) | -0.0006 [-0.031, 0.017, 0.011] | -0.0052 [-0.034, -0.001, 0.020] | 0.0093 [-0.020, 0.002, 0.045] |
| series_valid point-head wSpearman (n>=5) | 0.339 (0.024) | 0.336 (0.020) | 0.325 (0.010) | 0.344 (0.025) | -0.0030 [-0.043, 0.017, 0.017] | -0.0147 [-0.049, -0.014, 0.018] | 0.0043 [-0.036, -0.009, 0.058] |
| series_valid point pair RMSE (n>=5) | 1.112 (0.015) | 1.111 (0.013) | 1.160 (0.002) | 1.142 (0.035) | -0.0009 [0.011, 0.006, -0.019] | 0.0478 [0.042, 0.066, 0.035] | 0.0301 [0.065, 0.039, -0.013] |
| series_valid point pair dir acc (n>=5) | 0.678 (0.012) | 0.675 (0.010) | 0.669 (0.007) | 0.679 (0.013) | -0.0032 [-0.025, 0.010, 0.006] | -0.0084 [-0.028, -0.006, 0.009] | 0.0016 [-0.022, 0.000, 0.026] |
| group_valid pair-head wKendall (n>=5) | nan (nan) | 0.185 (0.012) | nan (nan) | 0.155 (0.011) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| group_valid pair-head wSpearman (n>=5) | nan (nan) | 0.256 (0.018) | nan (nan) | 0.215 (0.012) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| group_valid pair RMSE (n>=5) | nan (nan) | 1.376 (0.020) | nan (nan) | 1.424 (0.016) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| group_valid pair dir acc (n>=5) | nan (nan) | 0.627 (0.010) | nan (nan) | 0.601 (0.003) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| group_valid point-head wKendall (n>=5) | 0.171 (0.007) | 0.178 (0.015) | 0.151 (0.009) | 0.149 (0.018) | 0.0063 [-0.010, 0.010, 0.019] | -0.0207 [-0.019, -0.008, -0.034] | -0.0227 [-0.016, -0.035, -0.018] |
| group_valid point-head wSpearman (n>=5) | 0.238 (0.011) | 0.246 (0.021) | 0.210 (0.015) | 0.208 (0.026) | 0.0089 [-0.017, 0.015, 0.029] | -0.0273 [-0.026, -0.008, -0.048] | -0.0297 [-0.019, -0.046, -0.024] |
| group_valid point pair RMSE (n>=5) | 1.304 (0.016) | 1.301 (0.012) | 1.443 (0.001) | 1.462 (0.014) | -0.0034 [-0.012, 0.013, -0.011] | 0.1387 [0.121, 0.151, 0.143] | 0.1572 [0.136, 0.185, 0.151] |
| group_valid point pair dir acc (n>=5) | 0.620 (0.004) | 0.622 (0.012) | 0.604 (0.005) | 0.603 (0.009) | 0.0026 [-0.008, 0.000, 0.015] | -0.0161 [-0.018, -0.008, -0.023] | -0.0164 [-0.010, -0.021, -0.018] |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.522 (0.011) | 0.548 (0.020) | 0.000 (0.000) | 0.000 (0.000) | 0.0265 [0.011, 0.061, 0.007] | -0.5218 [-0.525, -0.510, -0.531] | -0.5218 [-0.525, -0.510, -0.531] |
| Aug kind 4 (A-twin) rank acc | 0.470 (0.012) | 0.472 (0.006) | 0.388 (0.013) | 0.378 (0.008) | 0.0025 [0.021, -0.004, -0.010] | -0.0821 [-0.085, -0.077, -0.084] | -0.0917 [-0.089, -0.089, -0.098] |
| Aug kind 4 (A-twin) gap (pK) | 0.254 (0.005) | 0.290 (0.018) | 0.065 (0.021) | 0.081 (0.053) | 0.0362 [0.025, 0.055, 0.029] | -0.1891 [-0.210, -0.198, -0.160] | -0.1727 [-0.228, -0.121, -0.169] |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.443 (0.019) | 0.466 (0.007) | 0.000 (0.000) | 0.000 (0.000) | 0.0237 [-0.000, 0.048, 0.023] | -0.4427 [-0.464, -0.426, -0.438] | -0.4427 [-0.464, -0.426, -0.438] |
| Aug kind 6 (C-contact) rank acc | 0.415 (0.017) | 0.434 (0.027) | 0.420 (0.004) | 0.427 (0.012) | 0.0188 [-0.031, 0.038, 0.049] | 0.0052 [-0.011, 0.011, 0.015] | 0.0122 [-0.021, 0.028, 0.030] |
| Aug kind 6 (C-contact) gap (pK) | 0.051 (0.011) | 0.072 (0.052) | 0.020 (0.020) | 0.046 (0.017) | 0.0216 [-0.032, 0.067, 0.030] | -0.0305 [-0.050, -0.046, 0.004] | -0.0046 [0.012, -0.021, -0.004] |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.407 (0.035) | 0.423 (0.016) | 0.417 (0.005) | 0.462 (0.013) | 0.0165 [-0.021, 0.038, 0.033] | 0.0104 [-0.029, 0.044, 0.016] | 0.0553 [0.016, 0.081, 0.069] |

## Pair head vs pointwise head

| metric | tag | reference | pair head (ens) | own point head (ens) | reference point head (ens) | pair - own point (per seed) | pair - reference point (per seed) |
|---|---|---|---|---|---|---|---|
| FEP pair-head wKendall (n>=10) | S | base-g-pw | 0.406 | 0.400 | 0.378 | 0.0037 [0.008, -0.007, 0.010] | 0.0241 [-0.006, 0.042, 0.036] |
| FEP pair-head wSpearman (n>=10) | S | base-g-pw | 0.530 | 0.523 | 0.512 | 0.0068 [0.014, -0.007, 0.013] | 0.0275 [0.007, 0.036, 0.040] |
| FEP pair RMSE (n>=10) | S | base-g-pw | 0.903 | 0.899 | 0.922 | 0.0405 [0.041, 0.045, 0.035] | 0.0159 [0.033, 0.021, -0.006] |
| FEP pair dir acc (n>=10) | S | base-g-pw | 0.774 | 0.769 | 0.756 | 0.0028 [0.006, -0.002, 0.005] | 0.0194 [0.006, 0.020, 0.031] |
| series_valid pair-head wKendall (n>=5) | S | base-g-pw | 0.296 | 0.283 | 0.287 | 0.0128 [0.019, 0.011, 0.008] | 0.0122 [-0.011, 0.028, 0.020] |
| series_valid pair-head wSpearman (n>=5) | S | base-g-pw | 0.391 | 0.373 | 0.380 | 0.0176 [0.025, 0.015, 0.013] | 0.0146 [-0.018, 0.033, 0.029] |
| series_valid pair RMSE (n>=5) | S | base-g-pw | 1.100 | 1.066 | 1.061 | 0.0654 [0.055, 0.062, 0.080] | 0.0645 [0.065, 0.067, 0.061] |
| series_valid pair dir acc (n>=5) | S | base-g-pw | 0.703 | 0.695 | 0.703 | 0.0094 [0.015, 0.010, 0.004] | 0.0062 [-0.011, 0.019, 0.010] |
| group_valid pair-head wKendall (n>=5) | S | base-g-pw | 0.214 | 0.207 | 0.203 | 0.0074 [0.016, 0.002, 0.004] | 0.0137 [0.005, 0.012, 0.023] |
| group_valid pair-head wSpearman (n>=5) | S | base-g-pw | 0.295 | 0.289 | 0.280 | 0.0097 [0.020, 0.003, 0.006] | 0.0186 [0.003, 0.017, 0.035] |
| group_valid pair RMSE (n>=5) | S | base-g-pw | 1.279 | 1.237 | 1.239 | 0.0746 [0.067, 0.089, 0.068] | 0.0712 [0.055, 0.102, 0.057] |
| group_valid pair dir acc (n>=5) | S | base-g-pw | 0.647 | 0.644 | 0.643 | 0.0049 [0.010, 0.003, 0.002] | 0.0075 [0.003, 0.003, 0.017] |
| FEP pair-head wKendall (n>=10) | S-lig | base-g-lig-pw | 0.367 | 0.356 | 0.359 | 0.0167 [0.032, 0.002, 0.015] | 0.0175 [0.038, 0.009, 0.006] |
| FEP pair-head wSpearman (n>=10) | S-lig | base-g-lig-pw | 0.480 | 0.464 | 0.470 | 0.0151 [0.031, -0.004, 0.018] | 0.0135 [0.043, 0.002, -0.005] |
| FEP pair RMSE (n>=10) | S-lig | base-g-lig-pw | 0.929 | 0.947 | 0.957 | -0.0146 [-0.051, 0.005, 0.002] | -0.0286 [-0.068, -0.012, -0.006] |
| FEP pair dir acc (n>=10) | S-lig | base-g-lig-pw | 0.743 | 0.736 | 0.740 | 0.0096 [0.019, -0.002, 0.012] | 0.0088 [0.014, 0.007, 0.005] |
| series_valid pair-head wKendall (n>=5) | S-lig | base-g-lig-pw | 0.305 | 0.286 | 0.271 | 0.0181 [0.018, 0.023, 0.013] | 0.0326 [0.033, 0.026, 0.039] |
| series_valid pair-head wSpearman (n>=5) | S-lig | base-g-lig-pw | 0.392 | 0.366 | 0.352 | 0.0194 [0.021, 0.024, 0.013] | 0.0385 [0.034, 0.029, 0.052] |
| series_valid pair RMSE (n>=5) | S-lig | base-g-lig-pw | 1.099 | 1.087 | 1.097 | 0.0291 [0.028, 0.028, 0.032] | 0.0114 [0.051, -0.000, -0.017] |
| series_valid pair dir acc (n>=5) | S-lig | base-g-lig-pw | 0.704 | 0.691 | 0.685 | 0.0079 [0.010, 0.010, 0.004] | 0.0179 [0.015, 0.017, 0.022] |
| group_valid pair-head wKendall (n>=5) | S-lig | base-g-lig-pw | 0.185 | 0.176 | 0.169 | 0.0066 [-0.000, 0.014, 0.006] | 0.0046 [0.004, -0.012, 0.023] |
| group_valid pair-head wSpearman (n>=5) | S-lig | base-g-lig-pw | 0.253 | 0.244 | 0.238 | 0.0070 [-0.006, 0.023, 0.005] | 0.0045 [0.001, -0.016, 0.029] |
| group_valid pair RMSE (n>=5) | S-lig | base-g-lig-pw | 1.324 | 1.370 | 1.355 | -0.0379 [-0.024, -0.046, -0.044] | -0.0195 [-0.009, -0.012, -0.037] |
| group_valid pair dir acc (n>=5) | S-lig | base-g-lig-pw | 0.623 | 0.622 | 0.621 | -0.0019 [-0.009, 0.003, 0.000] | -0.0022 [-0.002, -0.010, 0.005] |

## Shortcut gate: S vs base-g-pw, against S-lig vs base-g-lig-pw

| metric | S - base-g-pw | S-lig - base-g-lig-pw | structure-specific gain | verdict |
|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.0243 | 0.0052 | 0.0192 (sd 0.0294) | within seed noise |
| FEP wKendall (n>=10) | 0.0204 | 0.0008 | 0.0196 (sd 0.0340) | within seed noise |
| FEP wSpearman (n>=10) | 0.0207 | -0.0016 | 0.0224 (sd 0.0369) | within seed noise |
| FEP wRMSE (all) | -0.0202 | -0.0097 | 0.0105 (sd 0.0122) | within seed noise |
| series_valid wSpearman (n>=5) | -0.0030 | 0.0191 | -0.0221 (sd 0.0343) | no structure-specific gain |
| series_valid wKendall (n>=5) | -0.0006 | 0.0144 | -0.0150 (sd 0.0297) | no structure-specific gain |
| series_valid wRMSE (all) | 0.0109 | -0.0007 | -0.0116 (sd 0.0319) | no structure-specific gain |
| CASF Pearson | 0.0098 | -0.0017 | 0.0115 (sd 0.0336) | within seed noise |
| CASF RMSE | -0.0450 | -0.0000 | 0.0450 (sd 0.1204) | within seed noise |
| group_valid Pearson | -0.0118 | -0.0123 | 0.0004 (sd 0.0428) | within seed noise |
| group_valid Spearman | -0.0057 | -0.0092 | 0.0036 (sd 0.0596) | within seed noise |
| group_valid RMSE | -0.0041 | -0.0009 | 0.0033 (sd 0.1123) | within seed noise |
| group_valid within-series wSpearman (n>=5) | 0.0089 | -0.0024 | 0.0113 (sd 0.0392) | within seed noise |
| FEP pair-head wKendall (n>=10) (ref: point head) | 0.0241 | 0.0175 | 0.0066 (sd 0.0440) | within seed noise |
| FEP pair-head wSpearman (n>=10) (ref: point head) | 0.0275 | 0.0135 | 0.0140 (sd 0.0443) | within seed noise |
| FEP pair RMSE (n>=10) (ref: point head) | 0.0159 | -0.0286 | -0.0445 (sd 0.0514) | no structure-specific gain |
| FEP pair dir acc (n>=10) (ref: point head) | 0.0194 | 0.0088 | 0.0106 (sd 0.0175) | within seed noise |
| FEP point-head wKendall (n>=10) | 0.0204 | 0.0008 | 0.0196 (sd 0.0340) | within seed noise |
| FEP point-head wSpearman (n>=10) | 0.0207 | -0.0016 | 0.0224 (sd 0.0369) | within seed noise |
| FEP point pair RMSE (n>=10) | -0.0246 | -0.0140 | 0.0106 (sd 0.0211) | within seed noise |
| FEP point pair dir acc (n>=10) | 0.0165 | -0.0008 | 0.0174 (sd 0.0148) | passes (gain beyond ligand-only, > 1 sd) |
| series_valid pair-head wKendall (n>=5) (ref: point head) | 0.0122 | 0.0326 | -0.0203 (sd 0.0231) | no structure-specific gain |
| series_valid pair-head wSpearman (n>=5) (ref: point head) | 0.0146 | 0.0385 | -0.0239 (sd 0.0282) | no structure-specific gain |
| series_valid pair RMSE (n>=5) (ref: point head) | 0.0645 | 0.0114 | -0.0531 (sd 0.0342) | no structure-specific gain |
| series_valid pair dir acc (n>=5) (ref: point head) | 0.0062 | 0.0179 | -0.0117 (sd 0.0143) | no structure-specific gain |
| series_valid point-head wKendall (n>=5) | -0.0006 | 0.0144 | -0.0150 (sd 0.0297) | no structure-specific gain |
| series_valid point-head wSpearman (n>=5) | -0.0030 | 0.0191 | -0.0221 (sd 0.0343) | no structure-specific gain |
| series_valid point pair RMSE (n>=5) | -0.0009 | -0.0177 | -0.0169 (sd 0.0255) | no structure-specific gain |
| series_valid point pair dir acc (n>=5) | -0.0032 | 0.0100 | -0.0132 (sd 0.0170) | no structure-specific gain |
| group_valid pair-head wKendall (n>=5) (ref: point head) | 0.0137 | 0.0046 | 0.0090 (sd 0.0135) | within seed noise |
| group_valid pair-head wSpearman (n>=5) (ref: point head) | 0.0186 | 0.0045 | 0.0141 (sd 0.0169) | within seed noise |
| group_valid pair RMSE (n>=5) (ref: point head) | 0.0712 | -0.0195 | -0.0907 (sd 0.0252) | no structure-specific gain |
| group_valid pair dir acc (n>=5) (ref: point head) | 0.0075 | -0.0022 | 0.0097 (sd 0.0048) | passes (gain beyond ligand-only, > 1 sd) |
| group_valid point-head wKendall (n>=5) | 0.0063 | -0.0020 | 0.0083 (sd 0.0262) | within seed noise |
| group_valid point-head wSpearman (n>=5) | 0.0089 | -0.0024 | 0.0113 (sd 0.0392) | within seed noise |
| group_valid point pair RMSE (n>=5) | -0.0034 | 0.0185 | 0.0218 (sd 0.0042) | passes (gain beyond ligand-only, > 1 sd) |
| group_valid point pair dir acc (n>=5) | 0.0026 | -0.0002 | 0.0029 (sd 0.0159) | within seed noise |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.0265 | 0.0000 | -0.0265 (sd 0.0304) | no structure-specific gain |
| Aug kind 4 (A-twin) rank acc | 0.0025 | -0.0096 | 0.0121 (sd 0.0113) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) gap (pK) | 0.0362 | 0.0164 | 0.0198 (sd 0.0362) | within seed noise |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.0237 | 0.0000 | -0.0237 (sd 0.0243) | no structure-specific gain |
| Aug kind 6 (C-contact) rank acc | 0.0188 | 0.0070 | 0.0117 (sd 0.0293) | within seed noise |
| Aug kind 6 (C-contact) gap (pK) | 0.0216 | 0.0259 | -0.0043 (sd 0.0777) | no structure-specific gain |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.0165 | 0.0449 | 0.0283 (sd 0.0341) | within seed noise |
