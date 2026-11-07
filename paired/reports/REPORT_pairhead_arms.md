# Arm comparison (base-g, M, P, PM, P1, PMpos, PMneg, base-g-lig, M-lig, PM-lig; reference base-g; seeds 0,1,2)

pairwise_summary.csv missing for base-g, base-g-lig: pairwise rows omitted.

## 3-seed ensembles

| metric | base-g | M | P | PM | P1 | PMpos | PMneg | base-g-lig | M-lig | PM-lig |
|---|---|---|---|---|---|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.537 | 0.541 | 0.533 | 0.567 | 0.562 | 0.541 | 0.494 | 0.492 | 0.503 | 0.503 |
| FEP wKendall (n>=10) | 0.378 | 0.385 | 0.380 | 0.406 | 0.394 | 0.384 | 0.351 | 0.359 | 0.352 | 0.354 |
| FEP wSpearman (n>=10) | 0.512 | 0.511 | 0.498 | 0.537 | 0.523 | 0.510 | 0.470 | 0.471 | 0.465 | 0.458 |
| FEP wRMSE (all) | 0.986 | 0.978 | 0.991 | 0.986 | 0.960 | 0.953 | 1.004 | 0.978 | 0.950 | 0.942 |
| series_valid wSpearman (n>=5) | 0.380 | 0.383 | 0.348 | 0.380 | 0.368 | 0.364 | 0.370 | 0.351 | 0.385 | 0.384 |
| series_valid wKendall (n>=5) | 0.287 | 0.292 | 0.267 | 0.291 | 0.280 | 0.279 | 0.279 | 0.270 | 0.296 | 0.294 |
| series_valid wRMSE (all) | 0.863 | 0.852 | 0.870 | 0.871 | 0.866 | 0.857 | 0.855 | 0.875 | 0.861 | 0.871 |
| CASF Pearson | 0.880 | 0.894 | 0.885 | 0.892 | 0.887 | 0.887 | 0.889 | 0.823 | 0.827 | 0.837 |
| CASF RMSE | 1.080 | 1.024 | 1.061 | 1.028 | 1.054 | 1.050 | 1.061 | 1.248 | 1.237 | 1.204 |
| group_valid Pearson | 0.584 | 0.580 | 0.585 | 0.577 | 0.591 | 0.578 | 0.560 | 0.497 | 0.505 | 0.500 |
| group_valid Spearman | 0.537 | 0.526 | 0.540 | 0.521 | 0.544 | 0.537 | 0.511 | 0.473 | 0.485 | 0.461 |
| group_valid RMSE | 1.355 | 1.377 | 1.338 | 1.368 | 1.361 | 1.338 | 1.370 | 1.542 | 1.524 | 1.497 |
| group_valid within-series wSpearman (n>=5) | 0.281 | 0.294 | 0.285 | 0.291 | 0.291 | 0.281 | 0.268 | 0.238 | 0.244 | 0.261 |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.458 | 0.454 | 0.457 | 0.461 | 0.482 | 0.274 | 0.312 | 0.000 | 0.000 | 0.000 |
| Aug kind 4 (A-twin) rank acc | 0.479 | 0.484 | 0.492 | 0.525 | 0.451 | 0.472 | 0.302 | 0.401 | 0.380 | 0.393 |
| Aug kind 4 (A-twin) gap (pK) | 0.254 | 0.289 | 0.259 | 0.343 | 0.249 | 0.276 | -0.817 | 0.065 | 0.049 | 0.061 |
| Aug kind 4 (A-twin) pairhead rank acc | nan | nan | 0.507 | 0.549 | 0.500 | 0.480 | 0.705 | nan | nan | 0.482 |
| Aug kind 4 (A-twin) pairhead gap (logit) | nan | nan | 0.281 | 0.397 | 0.319 | 0.283 | 10.911 | nan | nan | 0.289 |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.384 | 0.352 | 0.390 | 0.370 | 0.398 | 0.187 | 0.196 | 0.000 | 0.000 | 0.000 |
| Aug kind 6 (C-contact) rank acc | 0.425 | 0.421 | 0.421 | 0.449 | 0.430 | 0.466 | 0.286 | 0.412 | 0.408 | 0.416 |
| Aug kind 6 (C-contact) gap (pK) | 0.051 | 0.068 | 0.038 | 0.074 | 0.063 | 0.116 | -1.328 | 0.020 | 0.017 | 0.036 |
| Aug kind 6 (C-contact) pairhead rank acc | nan | nan | 0.415 | 0.459 | 0.439 | 0.471 | 0.952 | nan | nan | 0.496 |
| Aug kind 6 (C-contact) pairhead gap (logit) | nan | nan | 0.040 | 0.098 | 0.068 | 0.122 | 21.651 | nan | nan | 0.150 |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.330 | 0.312 | 0.341 | 0.306 | 0.322 | 0.233 | 0.277 | 0.346 | 0.388 | 0.346 |
| Moments R2 pocket (valid) | nan | 0.504 | nan | 0.504 | nan | 0.498 | 0.493 | nan | 0.370 | 0.365 |
| Moments R2 ligand (valid) | nan | 0.785 | nan | 0.785 | nan | 0.781 | 0.782 | nan | 0.769 | 0.767 |
| Moments R2 pocket (series_valid) | nan | 0.451 | nan | 0.450 | nan | 0.442 | 0.441 | nan | 0.357 | 0.348 |
| Moments R2 ligand (series_valid) | nan | 0.683 | nan | 0.685 | nan | 0.681 | 0.672 | nan | 0.659 | 0.650 |
| Moments R2 pocket (group_valid) | nan | 0.259 | nan | 0.245 | nan | 0.260 | 0.262 | nan | 0.102 | 0.102 |
| Moments R2 ligand (group_valid) | nan | 0.604 | nan | 0.600 | nan | 0.599 | 0.613 | nan | 0.597 | 0.601 |
| Moments R2 pocket (test) | nan | 0.529 | nan | 0.532 | nan | 0.526 | 0.524 | nan | 0.413 | 0.402 |
| Moments R2 ligand (test) | nan | 0.813 | nan | 0.812 | nan | 0.800 | 0.796 | nan | 0.796 | 0.785 |
| Moments R2 pocket (aug_valid_all) | nan | 0.393 | nan | 0.393 | nan | 0.400 | 0.369 | nan | 0.222 | 0.223 |
| Moments R2 ligand (aug_valid_all) | nan | 0.761 | nan | 0.762 | nan | 0.759 | 0.749 | nan | 0.785 | 0.779 |

## Single seeds: mean (sd) and per-seed paired difference to the reference

| metric | base-g mean (sd) | M mean (sd) | P mean (sd) | PM mean (sd) | P1 mean (sd) | PMpos mean (sd) | PMneg mean (sd) | base-g-lig mean (sd) | M-lig mean (sd) | PM-lig mean (sd) | M - base-g (paired) | P - base-g (paired) | PM - base-g (paired) | P1 - base-g (paired) | PMpos - base-g (paired) | PMneg - base-g (paired) | base-g-lig - base-g (paired) | M-lig - base-g (paired) | PM-lig - base-g (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.468 (0.016) | 0.465 (0.021) | 0.471 (0.024) | 0.500 (0.031) | 0.489 (0.009) | 0.475 (0.020) | 0.425 (0.045) | 0.426 (0.020) | 0.451 (0.015) | 0.445 (0.015) | -0.0022 [0.005, -0.011, -0.000] | 0.0037 [0.001, 0.023, -0.012] | 0.0319 [0.041, 0.045, 0.010] | 0.0217 [-0.002, 0.039, 0.029] | 0.0075 [-0.018, -0.001, 0.042] | -0.0424 [-0.038, -0.088, -0.001] | -0.0416 [-0.082, -0.018, -0.025] | -0.0165 [-0.023, -0.028, 0.002] | -0.0228 [-0.029, -0.012, -0.028] |
| FEP wKendall (n>=10) | 0.328 (0.025) | 0.326 (0.024) | 0.316 (0.017) | 0.356 (0.024) | 0.343 (0.022) | 0.336 (0.021) | 0.293 (0.035) | 0.301 (0.024) | 0.314 (0.018) | 0.307 (0.009) | -0.0015 [-0.000, 0.011, -0.014] | -0.0114 [-0.029, 0.022, -0.027] | 0.0286 [0.026, 0.051, 0.008] | 0.0159 [-0.019, 0.064, 0.003] | 0.0081 [-0.024, 0.014, 0.035] | -0.0346 [-0.035, -0.052, -0.017] | -0.0263 [-0.079, 0.019, -0.019] | -0.0139 [-0.030, -0.011, -0.001] | -0.0202 [-0.038, 0.003, -0.026] |
| FEP wSpearman (n>=10) | 0.441 (0.023) | 0.437 (0.028) | 0.429 (0.017) | 0.478 (0.036) | 0.461 (0.027) | 0.448 (0.025) | 0.394 (0.045) | 0.404 (0.033) | 0.417 (0.022) | 0.405 (0.012) | -0.0040 [0.002, -0.004, -0.009] | -0.0124 [-0.031, 0.019, -0.024] | 0.0371 [0.050, 0.051, 0.011] | 0.0203 [-0.017, 0.070, 0.008] | 0.0068 [-0.026, 0.004, 0.042] | -0.0467 [-0.044, -0.081, -0.015] | -0.0365 [-0.099, 0.011, -0.021] | -0.0238 [-0.035, -0.031, -0.006] | -0.0355 [-0.050, -0.016, -0.040] |
| FEP wRMSE (all) | 1.023 (0.011) | 1.014 (0.005) | 1.034 (0.048) | 1.029 (0.012) | 1.008 (0.028) | 0.999 (0.046) | 1.063 (0.046) | 1.045 (0.025) | 1.006 (0.035) | 1.003 (0.017) | -0.0089 [-0.020, -0.013, 0.006] | 0.0111 [-0.030, -0.014, 0.077] | 0.0061 [0.007, 0.005, 0.006] | -0.0152 [-0.054, 0.015, -0.007] | -0.0246 [-0.074, 0.028, -0.027] | 0.0401 [0.064, 0.058, -0.002] | 0.0216 [0.039, 0.005, 0.021] | -0.0169 [-0.043, 0.025, -0.033] | -0.0198 [-0.013, -0.022, -0.025] |
| series_valid wSpearman (n>=5) | 0.340 (0.024) | 0.351 (0.015) | 0.321 (0.006) | 0.346 (0.011) | 0.332 (0.009) | 0.333 (0.020) | 0.328 (0.024) | 0.325 (0.010) | 0.348 (0.006) | 0.354 (0.017) | 0.0111 [-0.002, -0.008, 0.043] | -0.0184 [-0.046, -0.022, 0.013] | 0.0068 [-0.025, 0.001, 0.044] | -0.0078 [-0.026, -0.003, 0.006] | -0.0070 [-0.013, -0.005, -0.004] | -0.0114 [-0.037, 0.012, -0.009] | -0.0150 [-0.049, -0.013, 0.018] | 0.0086 [-0.016, 0.013, 0.029] | 0.0145 [0.001, 0.024, 0.019] |
| series_valid wKendall (n>=5) | 0.254 (0.016) | 0.267 (0.011) | 0.244 (0.002) | 0.261 (0.005) | 0.250 (0.006) | 0.252 (0.012) | 0.246 (0.018) | 0.248 (0.010) | 0.267 (0.006) | 0.270 (0.012) | 0.0127 [0.003, 0.002, 0.033] | -0.0101 [-0.029, -0.009, 0.008] | 0.0066 [-0.015, 0.007, 0.028] | -0.0040 [-0.017, 0.001, 0.004] | -0.0021 [-0.008, 0.001, 0.001] | -0.0086 [-0.027, 0.011, -0.010] | -0.0058 [-0.034, -0.001, 0.018] | 0.0123 [-0.007, 0.020, 0.024] | 0.0160 [0.001, 0.028, 0.019] |
| series_valid wRMSE (all) | 0.904 (0.003) | 0.892 (0.008) | 0.910 (0.007) | 0.917 (0.011) | 0.905 (0.017) | 0.906 (0.003) | 0.902 (0.004) | 0.929 (0.008) | 0.913 (0.011) | 0.919 (0.013) | -0.0119 [-0.007, -0.019, -0.010] | 0.0062 [0.011, 0.004, 0.003] | 0.0131 [0.021, 0.002, 0.016] | 0.0015 [-0.012, 0.021, -0.005] | 0.0022 [-0.002, 0.005, 0.003] | -0.0019 [-0.001, -0.003, -0.002] | 0.0251 [0.016, 0.022, 0.038] | 0.0093 [0.000, 0.022, 0.006] | 0.0151 [0.026, 0.004, 0.015] |
| CASF Pearson | 0.868 (0.004) | 0.879 (0.009) | 0.873 (0.011) | 0.880 (0.004) | 0.874 (0.008) | 0.874 (0.011) | 0.872 (0.003) | 0.798 (0.022) | 0.804 (0.025) | 0.814 (0.015) | 0.0111 [0.022, 0.015, -0.003] | 0.0048 [-0.005, 0.010, 0.010] | 0.0119 [0.010, 0.015, 0.011] | 0.0064 [0.006, 0.016, -0.003] | 0.0056 [0.005, 0.019, -0.007] | 0.0041 [0.011, 0.002, -0.001] | -0.0700 [-0.092, -0.049, -0.069] | -0.0636 [-0.083, -0.067, -0.041] | -0.0543 [-0.067, -0.039, -0.057] |
| CASF RMSE | 1.113 (0.015) | 1.069 (0.039) | 1.100 (0.039) | 1.073 (0.020) | 1.093 (0.028) | 1.098 (0.028) | 1.108 (0.014) | 1.324 (0.074) | 1.310 (0.079) | 1.276 (0.039) | -0.0438 [-0.089, -0.056, 0.015] | -0.0137 [0.022, -0.036, -0.026] | -0.0400 [-0.026, -0.055, -0.039] | -0.0198 [-0.010, -0.061, 0.012] | -0.0153 [-0.007, -0.056, 0.018] | -0.0055 [-0.023, 0.001, 0.005] | 0.2111 [0.285, 0.141, 0.207] | 0.1967 [0.254, 0.210, 0.126] | 0.1631 [0.198, 0.119, 0.172] |
| group_valid Pearson | 0.555 (0.012) | 0.554 (0.013) | 0.553 (0.001) | 0.549 (0.022) | 0.561 (0.013) | 0.547 (0.016) | 0.521 (0.040) | 0.463 (0.019) | 0.469 (0.031) | 0.464 (0.032) | -0.0004 [0.001, 0.007, -0.009] | -0.0015 [0.013, -0.008, -0.009] | -0.0058 [-0.016, 0.010, -0.012] | 0.0069 [0.034, -0.012, -0.002] | -0.0078 [0.017, -0.008, -0.033] | -0.0337 [0.003, -0.017, -0.087] | -0.0918 [-0.077, -0.117, -0.081] | -0.0855 [-0.100, -0.058, -0.099] | -0.0910 [-0.108, -0.100, -0.065] |
| group_valid Spearman | 0.505 (0.015) | 0.499 (0.018) | 0.505 (0.012) | 0.491 (0.041) | 0.513 (0.018) | 0.502 (0.020) | 0.476 (0.042) | 0.441 (0.018) | 0.454 (0.015) | 0.434 (0.026) | -0.0055 [-0.009, 0.012, -0.020] | 0.0004 [0.001, 0.005, -0.005] | -0.0135 [-0.046, 0.011, -0.005] | 0.0085 [0.038, -0.012, -0.000] | -0.0028 [0.019, 0.012, -0.040] | -0.0286 [0.010, -0.004, -0.092] | -0.0637 [-0.043, -0.084, -0.064] | -0.0508 [-0.046, -0.033, -0.073] | -0.0712 [-0.082, -0.073, -0.059] |
| group_valid RMSE | 1.399 (0.016) | 1.416 (0.023) | 1.388 (0.044) | 1.411 (0.063) | 1.404 (0.060) | 1.410 (0.095) | 1.432 (0.057) | 1.604 (0.021) | 1.594 (0.070) | 1.568 (0.099) | 0.0176 [0.039, 0.027, -0.014] | -0.0111 [-0.054, -0.003, 0.024] | 0.0117 [0.081, -0.011, -0.035] | 0.0047 [-0.055, 0.085, -0.016] | 0.0111 [-0.086, 0.125, -0.005] | 0.0329 [0.004, 0.011, 0.084] | 0.2049 [0.217, 0.231, 0.167] | 0.1951 [0.271, 0.158, 0.157] | 0.1690 [0.244, 0.219, 0.044] |
| group_valid within-series wSpearman (n>=5) | 0.237 (0.011) | 0.260 (0.015) | 0.244 (0.011) | 0.252 (0.020) | 0.249 (0.005) | 0.245 (0.018) | 0.226 (0.027) | 0.211 (0.014) | 0.214 (0.015) | 0.231 (0.004) | 0.0229 [0.003, 0.053, 0.013] | 0.0072 [-0.010, 0.032, -0.000] | 0.0149 [-0.001, 0.050, -0.004] | 0.0125 [0.008, 0.020, 0.010] | 0.0077 [-0.006, 0.040, -0.012] | -0.0114 [-0.014, -0.028, 0.008] | -0.0260 [-0.027, -0.005, -0.046] | -0.0225 [-0.041, 0.008, -0.034] | -0.0064 [-0.019, 0.007, -0.007] |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.522 (0.011) | 0.507 (0.006) | 0.520 (0.037) | 0.514 (0.011) | 0.549 (0.048) | 0.341 (0.002) | 0.424 (0.022) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | -0.0148 [-0.011, -0.007, -0.027] | -0.0015 [-0.032, -0.004, 0.031] | -0.0082 [0.001, -0.004, -0.022] | 0.0269 [0.019, 0.089, -0.028] | -0.1803 [-0.183, -0.167, -0.191] | -0.0979 [-0.125, -0.080, -0.088] | -0.5218 [-0.525, -0.510, -0.531] | -0.5218 [-0.525, -0.510, -0.531] | -0.5218 [-0.525, -0.510, -0.531] |
| Aug kind 4 (A-twin) rank acc | 0.470 (0.012) | 0.471 (0.012) | 0.478 (0.003) | 0.509 (0.012) | 0.468 (0.014) | 0.469 (0.013) | 0.330 (0.091) | 0.387 (0.012) | 0.376 (0.005) | 0.387 (0.012) | 0.0017 [0.015, 0.013, -0.023] | 0.0083 [0.019, 0.011, -0.005] | 0.0392 [0.042, 0.052, 0.023] | -0.0017 [-0.004, 0.013, -0.014] | -0.0008 [0.001, 0.014, -0.018] | -0.1396 [-0.023, -0.199, -0.198] | -0.0829 [-0.084, -0.081, -0.084] | -0.0938 [-0.086, -0.089, -0.106] | -0.0825 [-0.056, -0.089, -0.103] |
| Aug kind 4 (A-twin) gap (pK) | 0.254 (0.005) | 0.289 (0.031) | 0.259 (0.025) | 0.343 (0.024) | 0.249 (0.029) | 0.276 (0.027) | -0.817 (0.883) | 0.065 (0.021) | 0.049 (0.037) | 0.061 (0.053) | 0.0353 [0.065, 0.009, 0.031] | 0.0051 [-0.016, -0.008, 0.039] | 0.0898 [0.070, 0.076, 0.124] | -0.0045 [-0.034, 0.026, -0.005] | 0.0227 [-0.008, 0.049, 0.027] | -1.0704 [-0.073, -1.738, -1.400] | -0.1891 [-0.210, -0.198, -0.160] | -0.2049 [-0.192, -0.250, -0.174] | -0.1922 [-0.138, -0.207, -0.232] |
| Aug kind 4 (A-twin) pairhead rank acc | nan (nan) | nan (nan) | 0.482 (0.033) | 0.527 (0.017) | 0.492 (0.021) | 0.478 (0.013) | 0.682 (0.034) | nan (nan) | nan (nan) | 0.469 (0.015) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Aug kind 4 (A-twin) pairhead gap (logit) | nan (nan) | nan (nan) | 0.281 (0.040) | 0.397 (0.028) | 0.319 (0.043) | 0.283 (0.023) | 10.911 (2.968) | nan (nan) | nan (nan) | 0.289 (0.048) | - | - | - | - | - | - | - | - | - |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.443 (0.019) | 0.402 (0.007) | 0.448 (0.014) | 0.417 (0.007) | 0.460 (0.029) | 0.244 (0.007) | 0.271 (0.007) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | -0.0412 [-0.055, -0.026, -0.043] | 0.0048 [-0.028, 0.018, 0.025] | -0.0259 [-0.050, -0.014, -0.014] | 0.0171 [-0.009, 0.065, -0.004] | -0.1989 [-0.221, -0.176, -0.201] | -0.1713 [-0.193, -0.162, -0.159] | -0.4427 [-0.464, -0.426, -0.438] | -0.4427 [-0.464, -0.426, -0.438] | -0.4427 [-0.464, -0.426, -0.438] |
| Aug kind 6 (C-contact) rank acc | 0.415 (0.017) | 0.428 (0.013) | 0.417 (0.031) | 0.430 (0.005) | 0.425 (0.013) | 0.462 (0.031) | 0.326 (0.047) | 0.420 (0.004) | 0.401 (0.015) | 0.417 (0.023) | 0.0131 [-0.021, 0.030, 0.031] | 0.0023 [0.018, -0.006, -0.006] | 0.0150 [0.000, 0.025, 0.020] | 0.0098 [-0.020, 0.034, 0.015] | 0.0469 [0.063, 0.038, 0.039] | -0.0891 [-0.056, -0.093, -0.118] | 0.0052 [-0.011, 0.011, 0.015] | -0.0141 [-0.030, 0.008, -0.021] | 0.0023 [0.008, -0.004, 0.003] |
| Aug kind 6 (C-contact) gap (pK) | 0.051 (0.011) | 0.068 (0.004) | 0.038 (0.041) | 0.074 (0.014) | 0.063 (0.037) | 0.116 (0.007) | -1.328 (0.527) | 0.020 (0.020) | 0.017 (0.008) | 0.036 (0.040) | 0.0176 [0.010, 0.014, 0.029] | -0.0127 [0.030, -0.041, -0.028] | 0.0233 [0.036, 0.008, 0.026] | 0.0123 [-0.022, 0.044, 0.015] | 0.0648 [0.069, 0.049, 0.076] | -1.3783 [-0.776, -1.737, -1.622] | -0.0305 [-0.050, -0.046, 0.004] | -0.0336 [-0.036, -0.035, -0.030] | -0.0146 [0.028, -0.044, -0.028] |
| Aug kind 6 (C-contact) pairhead rank acc | nan (nan) | nan (nan) | 0.409 (0.051) | 0.440 (0.013) | 0.431 (0.015) | 0.463 (0.028) | 0.928 (0.006) | nan (nan) | nan (nan) | 0.486 (0.029) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Aug kind 6 (C-contact) pairhead gap (logit) | nan (nan) | nan (nan) | 0.040 (0.054) | 0.098 (0.014) | 0.068 (0.021) | 0.122 (0.009) | 21.651 (3.153) | nan (nan) | nan (nan) | 0.150 (0.015) | - | - | - | - | - | - | - | - | - |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.407 (0.035) | 0.387 (0.020) | 0.419 (0.028) | 0.379 (0.024) | 0.415 (0.027) | 0.302 (0.022) | 0.370 (0.014) | 0.417 (0.005) | 0.456 (0.015) | 0.406 (0.020) | -0.0201 [-0.032, 0.004, -0.033] | 0.0126 [-0.049, 0.046, 0.041] | -0.0281 [-0.090, 0.017, -0.011] | 0.0082 [-0.057, 0.064, 0.017] | -0.1045 [-0.126, -0.094, -0.093] | -0.0366 [-0.066, -0.017, -0.026] | 0.0104 [-0.029, 0.044, 0.016] | 0.0489 [0.022, 0.094, 0.031] | -0.0004 [-0.051, 0.058, -0.008] |
| Moments R2 pocket (valid) | nan (nan) | 0.449 (0.000) | nan (nan) | 0.448 (0.003) | nan (nan) | 0.450 (0.005) | 0.447 (0.006) | nan (nan) | 0.325 (0.004) | 0.322 (0.008) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 ligand (valid) | nan (nan) | 0.752 (0.004) | nan (nan) | 0.750 (0.006) | nan (nan) | 0.749 (0.004) | 0.747 (0.001) | nan (nan) | 0.745 (0.006) | 0.742 (0.003) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 pocket (series_valid) | nan (nan) | 0.391 (0.005) | nan (nan) | 0.388 (0.009) | nan (nan) | 0.396 (0.005) | 0.397 (0.002) | nan (nan) | 0.306 (0.005) | 0.300 (0.014) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 ligand (series_valid) | nan (nan) | 0.622 (0.008) | nan (nan) | 0.628 (0.019) | nan (nan) | 0.629 (0.001) | 0.617 (0.004) | nan (nan) | 0.611 (0.008) | 0.603 (0.009) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 pocket (group_valid) | nan (nan) | 0.176 (0.008) | nan (nan) | 0.160 (0.012) | nan (nan) | 0.197 (0.007) | 0.193 (0.015) | nan (nan) | 0.030 (0.010) | 0.031 (0.019) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 ligand (group_valid) | nan (nan) | 0.548 (0.006) | nan (nan) | 0.542 (0.019) | nan (nan) | 0.548 (0.007) | 0.557 (0.009) | nan (nan) | 0.549 (0.003) | 0.552 (0.007) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 pocket (test) | nan (nan) | 0.470 (0.004) | nan (nan) | 0.472 (0.009) | nan (nan) | 0.476 (0.004) | 0.475 (0.004) | nan (nan) | 0.361 (0.014) | 0.352 (0.014) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 ligand (test) | nan (nan) | 0.771 (0.010) | nan (nan) | 0.767 (0.005) | nan (nan) | 0.760 (0.003) | 0.753 (0.011) | nan (nan) | 0.759 (0.009) | 0.746 (0.010) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 pocket (aug_valid_all) | nan (nan) | 0.328 (0.008) | nan (nan) | 0.325 (0.009) | nan (nan) | 0.343 (0.005) | 0.310 (0.003) | nan (nan) | 0.175 (0.002) | 0.178 (0.002) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |
| Moments R2 ligand (aug_valid_all) | nan (nan) | 0.719 (0.001) | nan (nan) | 0.719 (0.005) | nan (nan) | 0.718 (0.006) | 0.706 (0.008) | nan (nan) | 0.759 (0.006) | 0.754 (0.007) | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] | nan [nan, nan, nan] |

## Pair head vs pointwise head

| metric | tag | reference | pair head (ens) | own point head (ens) | reference point head (ens) | pair - own point (per seed) | pair - reference point (per seed) |
|---|---|---|---|---|---|---|---|
| Aug kind 4 (A-twin) pairhead rank acc | P | base-g | 0.507 | 0.492 | 0.479 | 0.0037 [0.044, -0.020, -0.013] | 0.0121 [0.062, -0.009, -0.018] |
| Aug kind 6 (C-contact) pairhead rank acc | P | base-g | 0.415 | 0.421 | 0.425 | -0.0084 [0.013, -0.004, -0.034] | -0.0061 [0.031, -0.010, -0.039] |
| Aug kind 4 (A-twin) pairhead rank acc | PM | base-g | 0.549 | 0.525 | 0.479 | 0.0188 [0.011, 0.023, 0.022] | 0.0579 [0.054, 0.075, 0.045] |
| Aug kind 6 (C-contact) pairhead rank acc | PM | base-g | 0.459 | 0.449 | 0.425 | 0.0098 [0.017, 0.011, 0.001] | 0.0248 [0.017, 0.037, 0.021] |
| Aug kind 4 (A-twin) pairhead rank acc | P1 | base-g | 0.500 | 0.451 | 0.479 | 0.0238 [0.015, 0.026, 0.030] | 0.0221 [0.011, 0.039, 0.016] |
| Aug kind 6 (C-contact) pairhead rank acc | P1 | base-g | 0.439 | 0.430 | 0.425 | 0.0066 [0.000, 0.004, 0.015] | 0.0164 [-0.020, 0.038, 0.031] |
| Aug kind 4 (A-twin) pairhead rank acc | PMpos | base-g | 0.480 | 0.472 | 0.479 | 0.0096 [0.009, 0.009, 0.011] | 0.0087 [0.010, 0.023, -0.006] |
| Aug kind 6 (C-contact) pairhead rank acc | PMpos | base-g | 0.471 | 0.466 | 0.425 | 0.0009 [-0.004, -0.004, 0.011] | 0.0478 [0.059, 0.034, 0.051] |
| Aug kind 4 (A-twin) pairhead rank acc | PMneg | base-g | 0.705 | 0.302 | 0.479 | 0.3521 [0.229, 0.450, 0.378] | 0.2125 [0.206, 0.251, 0.180] |
| Aug kind 6 (C-contact) pairhead rank acc | PMneg | base-g | 0.952 | 0.286 | 0.425 | 0.6024 [0.556, 0.617, 0.634] | 0.5134 [0.499, 0.525, 0.516] |
| Aug kind 4 (A-twin) pairhead rank acc | PM-lig | base-g-lig | 0.482 | 0.393 | 0.401 | 0.0817 [0.054, 0.104, 0.088] | 0.0821 [0.081, 0.096, 0.069] |
| Aug kind 6 (C-contact) pairhead rank acc | PM-lig | base-g-lig | 0.496 | 0.416 | 0.412 | 0.0684 [0.049, 0.053, 0.103] | 0.0656 [0.069, 0.038, 0.090] |

## Shortcut gate: M vs base-g, against M-lig vs base-g-lig

| metric | M - base-g | M-lig - base-g-lig | structure-specific gain | verdict |
|---|---|---|---|---|
| FEP wPCC (n>=10) | -0.0022 | 0.0251 | -0.0272 (sd 0.0266) | no structure-specific gain |
| FEP wKendall (n>=10) | -0.0015 | 0.0123 | -0.0138 (sd 0.0474) | no structure-specific gain |
| FEP wSpearman (n>=10) | -0.0040 | 0.0127 | -0.0166 (sd 0.0503) | no structure-specific gain |
| FEP wRMSE (all) | -0.0089 | -0.0386 | -0.0296 (sd 0.0540) | no structure-specific gain |
| series_valid wSpearman (n>=5) | 0.0111 | 0.0237 | -0.0125 (sd 0.0388) | no structure-specific gain |
| series_valid wKendall (n>=5) | 0.0127 | 0.0181 | -0.0054 (sd 0.0284) | no structure-specific gain |
| series_valid wRMSE (all) | -0.0119 | -0.0157 | -0.0038 (sd 0.0209) | no structure-specific gain |
| CASF Pearson | 0.0111 | 0.0065 | 0.0046 (sd 0.0330) | within seed noise |
| CASF RMSE | -0.0438 | -0.0144 | 0.0294 (sd 0.1131) | within seed noise |
| group_valid Pearson | -0.0004 | 0.0062 | -0.0066 (sd 0.0399) | no structure-specific gain |
| group_valid Spearman | -0.0055 | 0.0129 | -0.0184 (sd 0.0179) | no structure-specific gain |
| group_valid RMSE | 0.0176 | -0.0098 | -0.0274 (sd 0.0635) | no structure-specific gain |
| group_valid within-series wSpearman (n>=5) | 0.0229 | 0.0034 | 0.0195 (sd 0.0196) | within seed noise |
| Aug kind 3 (A-pos) spread vs crystal (pK) | -0.0148 | -0.0000 | 0.0148 (sd 0.0103) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) rank acc | 0.0017 | -0.0108 | 0.0125 (sd 0.0109) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) gap (pK) | 0.0353 | -0.0159 | 0.0512 (sd 0.0087) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) pairhead rank acc (ref: point head) | nan | nan | nan (sd nan) | n/a |
| Aug kind 5 (C-pos) spread vs crystal (pK) | -0.0412 | -0.0000 | 0.0412 (sd 0.0144) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 6 (C-contact) rank acc | 0.0131 | -0.0192 | 0.0323 (sd 0.0352) | within seed noise |
| Aug kind 6 (C-contact) gap (pK) | 0.0176 | -0.0031 | 0.0207 (sd 0.0375) | within seed noise |
| Aug kind 6 (C-contact) pairhead rank acc (ref: point head) | nan | nan | nan (sd nan) | n/a |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | -0.0201 | 0.0385 | 0.0586 (sd 0.0205) | passes (gain beyond ligand-only, > 1 sd) |
| Moments R2 pocket (valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (series_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (series_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (group_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (group_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (test) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (test) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (aug_valid_all) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (aug_valid_all) | nan | nan | nan (sd nan) | n/a |

## Shortcut gate: PM vs base-g, against PM-lig vs base-g-lig

| metric | PM - base-g | PM-lig - base-g-lig | structure-specific gain | verdict |
|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.0319 | 0.0188 | 0.0131 (sd 0.0257) | within seed noise |
| FEP wKendall (n>=10) | 0.0286 | 0.0061 | 0.0225 (sd 0.0410) | within seed noise |
| FEP wSpearman (n>=10) | 0.0371 | 0.0010 | 0.0361 (sd 0.0389) | within seed noise |
| FEP wRMSE (all) | 0.0061 | -0.0414 | -0.0475 (sd 0.0139) | no structure-specific gain |
| series_valid wSpearman (n>=5) | 0.0068 | 0.0295 | -0.0228 (sd 0.0599) | no structure-specific gain |
| series_valid wKendall (n>=5) | 0.0066 | 0.0218 | -0.0152 (sd 0.0393) | no structure-specific gain |
| series_valid wRMSE (all) | 0.0131 | -0.0099 | -0.0231 (sd 0.0144) | no structure-specific gain |
| CASF Pearson | 0.0119 | 0.0157 | -0.0038 (sd 0.0100) | no structure-specific gain |
| CASF RMSE | -0.0400 | -0.0480 | -0.0080 (sd 0.0480) | no structure-specific gain |
| group_valid Pearson | -0.0058 | 0.0008 | -0.0065 (sd 0.0216) | no structure-specific gain |
| group_valid Spearman | -0.0135 | -0.0076 | -0.0059 (sd 0.0054) | no structure-specific gain |
| group_valid RMSE | 0.0117 | -0.0359 | -0.0476 (sd 0.0431) | no structure-specific gain |
| group_valid within-series wSpearman (n>=5) | 0.0149 | 0.0195 | -0.0046 (sd 0.0403) | no structure-specific gain |
| Aug kind 3 (A-pos) spread vs crystal (pK) | -0.0082 | -0.0000 | 0.0082 (sd 0.0125) | within seed noise |
| Aug kind 4 (A-twin) rank acc | 0.0392 | 0.0004 | 0.0388 (sd 0.0226) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) gap (pK) | 0.0898 | -0.0032 | 0.0930 (sd 0.0991) | within seed noise |
| Aug kind 4 (A-twin) pairhead rank acc (ref: point head) | 0.0579 | 0.0821 | -0.0242 (sd 0.0031) | no structure-specific gain |
| Aug kind 5 (C-pos) spread vs crystal (pK) | -0.0259 | 0.0000 | 0.0259 (sd 0.0211) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 6 (C-contact) rank acc | 0.0150 | -0.0028 | 0.0178 (sd 0.0328) | within seed noise |
| Aug kind 6 (C-contact) gap (pK) | 0.0233 | 0.0159 | 0.0074 (sd 0.0502) | within seed noise |
| Aug kind 6 (C-contact) pairhead rank acc (ref: point head) | 0.0248 | 0.0656 | -0.0408 (sd 0.0351) | no structure-specific gain |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | -0.0281 | -0.0109 | 0.0173 (sd 0.0445) | within seed noise |
| Moments R2 pocket (valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (series_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (series_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (group_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (group_valid) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (test) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (test) | nan | nan | nan (sd nan) | n/a |
| Moments R2 pocket (aug_valid_all) | nan | nan | nan (sd nan) | n/a |
| Moments R2 ligand (aug_valid_all) | nan | nan | nan (sd nan) | n/a |
