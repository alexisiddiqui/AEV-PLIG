# Arm comparison (base-g-10, PM-10, S-10, base-g-lig-10, PM-lig-10; reference base-g-10; seeds 0,1,2,3,4,5,6,7,8,9)

## 3-seed ensembles

| metric | base-g-10 | PM-10 | S-10 | base-g-lig-10 | PM-lig-10 |
|---|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.588 | 0.591 | 0.581 | 0.526 | 0.530 |
| FEP wKendall (n>=10) | 0.429 | 0.425 | 0.416 | 0.376 | 0.375 |
| FEP wSpearman (n>=10) | 0.562 | 0.557 | 0.551 | 0.494 | 0.495 |
| FEP wRMSE (all) | 0.950 | 0.956 | 0.956 | 0.948 | 0.932 |
| series_valid wSpearman (n>=5) | 0.390 | 0.389 | 0.384 | 0.370 | 0.398 |
| series_valid wKendall (n>=5) | 0.298 | 0.299 | 0.291 | 0.289 | 0.306 |
| series_valid wRMSE (all) | 0.857 | 0.854 | 0.858 | 0.855 | 0.850 |
| CASF Pearson | 0.890 | 0.899 | 0.898 | 0.826 | 0.842 |
| CASF RMSE | 1.043 | 1.008 | 1.020 | 1.244 | 1.193 |
| group_valid Pearson | 0.589 | 0.586 | 0.586 | 0.506 | 0.518 |
| group_valid Spearman | 0.542 | 0.530 | 0.544 | 0.482 | 0.482 |
| group_valid RMSE | 1.327 | 1.358 | 1.359 | 1.521 | 1.489 |
| group_valid within-series wSpearman (n>=5) | 0.317 | 0.322 | 0.315 | 0.258 | 0.276 |
| FEP pair-head wKendall (n>=10) | nan | nan | 0.426 | nan | nan |
| FEP pair-head wSpearman (n>=10) | nan | nan | 0.562 | nan | nan |
| FEP pair RMSE (n>=10) | nan | nan | 0.883 | nan | nan |
| FEP pair dir acc (n>=10) | nan | nan | 0.787 | nan | nan |
| FEP point-head wKendall (n>=10) | 0.429 | 0.425 | 0.416 | 0.376 | 0.375 |
| FEP point-head wSpearman (n>=10) | 0.562 | 0.557 | 0.551 | 0.494 | 0.495 |
| FEP point pair RMSE (n>=10) | 0.890 | 0.891 | 0.893 | 0.914 | 0.910 |
| FEP point pair dir acc (n>=10) | 0.784 | 0.784 | 0.783 | 0.748 | 0.748 |
| FEP pairhead wKendall (n>=10) | nan | 0.429 | nan | nan | 0.389 |
| FEP pairhead wSpearman (n>=10) | nan | 0.567 | nan | nan | 0.517 |
| FEP pairhead dir acc (n>=10) | nan | 0.789 | nan | nan | 0.759 |
| series_valid pair-head wKendall (n>=5) | nan | nan | 0.300 | nan | nan |
| series_valid pair-head wSpearman (n>=5) | nan | nan | 0.398 | nan | nan |
| series_valid pair RMSE (n>=5) | nan | nan | 1.071 | nan | nan |
| series_valid pair dir acc (n>=5) | nan | nan | 0.710 | nan | nan |
| series_valid point-head wKendall (n>=5) | 0.298 | 0.299 | 0.291 | 0.289 | 0.306 |
| series_valid point-head wSpearman (n>=5) | 0.390 | 0.389 | 0.384 | 0.370 | 0.398 |
| series_valid point pair RMSE (n>=5) | 1.043 | 1.041 | 1.049 | 1.072 | 1.056 |
| series_valid point pair dir acc (n>=5) | 0.708 | 0.706 | 0.703 | 0.693 | 0.708 |
| series_valid pairhead wKendall (n>=5) | nan | 0.318 | nan | nan | 0.323 |
| series_valid pairhead wSpearman (n>=5) | nan | 0.413 | nan | nan | 0.419 |
| series_valid pairhead dir acc (n>=5) | nan | 0.717 | nan | nan | 0.717 |
| group_valid pair-head wKendall (n>=5) | nan | nan | 0.226 | nan | nan |
| group_valid pair-head wSpearman (n>=5) | nan | nan | 0.311 | nan | nan |
| group_valid pair RMSE (n>=5) | nan | nan | 1.241 | nan | nan |
| group_valid pair dir acc (n>=5) | nan | nan | 0.653 | nan | nan |
| group_valid point-head wKendall (n>=5) | 0.230 | 0.235 | 0.229 | 0.185 | 0.199 |
| group_valid point-head wSpearman (n>=5) | 0.317 | 0.322 | 0.315 | 0.258 | 0.276 |
| group_valid point pair RMSE (n>=5) | 1.202 | 1.200 | 1.207 | 1.316 | 1.300 |
| group_valid point pair dir acc (n>=5) | 0.657 | 0.662 | 0.658 | 0.624 | 0.637 |
| group_valid pairhead wKendall (n>=5) | nan | 0.238 | nan | nan | 0.209 |
| group_valid pairhead wSpearman (n>=5) | nan | 0.326 | nan | nan | 0.287 |
| group_valid pairhead dir acc (n>=5) | nan | 0.660 | nan | nan | 0.643 |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.438 | 0.442 | 0.450 | 0.000 | 0.000 |
| Aug kind 4 (A-twin) rank acc | 0.481 | 0.516 | 0.489 | 0.379 | 0.385 |
| Aug kind 4 (A-twin) gap (pK) | 0.240 | 0.327 | 0.270 | 0.066 | 0.092 |
| Aug kind 4 (A-twin) pairhead rank acc | nan | 0.561 | nan | nan | 0.485 |
| Aug kind 4 (A-twin) pairhead gap (logit) | nan | 0.386 | nan | nan | 0.311 |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.366 | 0.349 | 0.381 | 0.000 | 0.000 |
| Aug kind 6 (C-contact) rank acc | 0.411 | 0.443 | 0.422 | 0.416 | 0.429 |
| Aug kind 6 (C-contact) gap (pK) | 0.039 | 0.075 | 0.062 | 0.022 | 0.043 |
| Aug kind 6 (C-contact) pairhead rank acc | nan | 0.467 | nan | nan | 0.516 |
| Aug kind 6 (C-contact) pairhead gap (logit) | nan | 0.106 | nan | nan | 0.160 |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.290 | 0.272 | 0.288 | 0.323 | 0.329 |
| Moments R2 pocket (valid) | nan | 0.522 | nan | nan | 0.373 |
| Moments R2 ligand (valid) | nan | 0.794 | nan | nan | 0.769 |
| Moments R2 pocket (series_valid) | nan | 0.471 | nan | nan | 0.359 |
| Moments R2 ligand (series_valid) | nan | 0.707 | nan | nan | 0.662 |
| Moments R2 pocket (group_valid) | nan | 0.279 | nan | nan | 0.128 |
| Moments R2 ligand (group_valid) | nan | 0.627 | nan | nan | 0.610 |
| Moments R2 pocket (test) | nan | 0.553 | nan | nan | 0.412 |
| Moments R2 ligand (test) | nan | 0.826 | nan | nan | 0.791 |
| Moments R2 pocket (aug_valid_all) | nan | 0.415 | nan | nan | 0.236 |
| Moments R2 ligand (aug_valid_all) | nan | 0.773 | nan | nan | 0.782 |

## Single seeds: mean (sd) and per-seed paired difference to the reference

| metric | base-g-10 mean (sd) | PM-10 mean (sd) | S-10 mean (sd) | base-g-lig-10 mean (sd) | PM-lig-10 mean (sd) | PM-10 - base-g-10 (paired) | S-10 - base-g-10 (paired) | base-g-lig-10 - base-g-10 (paired) | PM-lig-10 - base-g-10 (paired) |
|---|---|---|---|---|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.478 (0.018) | 0.487 (0.021) | 0.480 (0.017) | 0.424 (0.029) | 0.440 (0.020) | 0.0087 [0.041, 0.045, 0.010, 0.003, -0.025, 0.022, 0.018, 0.004, -0.023, -0.010] | 0.0022 [-0.007, 0.033, 0.047, -0.018, -0.001, 0.021, 0.023, -0.034, -0.007, -0.035] | -0.0537 [-0.082, -0.018, -0.025, -0.074, -0.033, -0.080, -0.054, -0.018, -0.081, -0.071] | -0.0376 [-0.029, -0.012, -0.028, -0.029, -0.052, -0.040, 0.010, -0.079, -0.049, -0.067] |
| FEP wKendall (n>=10) | 0.339 (0.015) | 0.345 (0.018) | 0.336 (0.016) | 0.299 (0.024) | 0.307 (0.015) | 0.0067 [0.026, 0.051, 0.008, -0.005, -0.012, -0.003, -0.019, 0.007, -0.016, 0.028] | -0.0028 [-0.013, 0.049, 0.025, -0.010, 0.005, -0.008, -0.013, -0.047, -0.017, 0.002] | -0.0403 [-0.078, 0.019, -0.019, -0.039, -0.011, -0.077, -0.065, -0.027, -0.075, -0.030] | -0.0314 [-0.037, 0.003, -0.026, -0.026, -0.041, -0.044, -0.017, -0.063, -0.050, -0.012] |
| FEP wSpearman (n>=10) | 0.456 (0.017) | 0.465 (0.024) | 0.451 (0.016) | 0.399 (0.030) | 0.410 (0.021) | 0.0091 [0.050, 0.051, 0.011, -0.002, -0.015, 0.006, -0.016, 0.002, -0.024, 0.029] | -0.0047 [-0.008, 0.043, 0.027, -0.014, 0.007, -0.006, -0.013, -0.051, -0.030, -0.003] | -0.0574 [-0.099, 0.011, -0.021, -0.054, -0.021, -0.100, -0.094, -0.040, -0.103, -0.053] | -0.0462 [-0.050, -0.016, -0.041, -0.027, -0.062, -0.056, -0.019, -0.091, -0.072, -0.028] |
| FEP wRMSE (all) | 1.014 (0.016) | 1.011 (0.028) | 1.016 (0.019) | 1.043 (0.026) | 1.024 (0.032) | -0.0036 [0.007, 0.005, 0.006, 0.006, 0.008, -0.044, -0.027, -0.033, 0.059, -0.024] | 0.0013 [-0.025, -0.017, -0.019, 0.032, 0.032, -0.005, -0.013, 0.013, -0.011, 0.024] | 0.0289 [0.039, 0.005, 0.021, 0.067, 0.064, 0.003, 0.036, -0.031, 0.067, 0.019] | 0.0094 [-0.013, -0.022, -0.025, 0.098, 0.022, -0.041, -0.004, -0.010, 0.034, 0.054] |
| series_valid wSpearman (n>=5) | 0.331 (0.018) | 0.347 (0.026) | 0.336 (0.016) | 0.323 (0.015) | 0.347 (0.019) | 0.0154 [-0.025, 0.001, 0.044, -0.019, 0.076, 0.043, 0.013, 0.020, -0.014, 0.015] | 0.0042 [-0.043, 0.017, 0.016, 0.011, 0.005, 0.013, 0.024, 0.017, -0.041, 0.023] | -0.0086 [-0.048, -0.013, 0.018, -0.011, 0.001, -0.023, 0.039, -0.030, -0.018, -0.000] | 0.0156 [-0.000, 0.023, 0.019, 0.022, 0.021, 0.029, 0.031, 0.045, -0.038, 0.003] |
| series_valid wKendall (n>=5) | 0.250 (0.011) | 0.262 (0.019) | 0.253 (0.011) | 0.247 (0.012) | 0.263 (0.014) | 0.0126 [-0.014, 0.006, 0.028, -0.014, 0.059, 0.031, 0.006, 0.016, -0.006, 0.016] | 0.0032 [-0.030, 0.017, 0.011, 0.006, 0.003, 0.009, 0.015, 0.009, -0.026, 0.018] | -0.0029 [-0.034, -0.001, 0.019, -0.008, 0.008, -0.019, 0.030, -0.019, -0.012, 0.008] | 0.0129 [0.000, 0.028, 0.019, 0.014, 0.018, 0.019, 0.021, 0.033, -0.024, -0.000] |
| series_valid wRMSE (all) | 0.916 (0.018) | 0.911 (0.016) | 0.911 (0.009) | 0.929 (0.015) | 0.924 (0.014) | -0.0055 [0.021, 0.002, 0.016, 0.017, -0.021, -0.026, -0.030, -0.034, 0.029, -0.029] | -0.0050 [0.007, 0.013, 0.012, -0.003, 0.025, -0.012, -0.034, -0.037, 0.021, -0.043] | 0.0129 [0.016, 0.022, 0.038, 0.010, 0.010, 0.027, -0.020, 0.026, 0.040, -0.039] | 0.0075 [0.026, 0.004, 0.015, 0.014, 0.026, 0.003, -0.037, -0.022, 0.052, -0.006] |
| CASF Pearson | 0.871 (0.007) | 0.880 (0.008) | 0.879 (0.004) | 0.794 (0.021) | 0.813 (0.020) | 0.0098 [0.010, 0.015, 0.011, -0.004, 0.009, 0.004, 0.003, 0.009, 0.025, 0.016] | 0.0083 [0.010, 0.019, 0.000, 0.001, 0.006, 0.004, 0.020, -0.002, 0.016, 0.008] | -0.0768 [-0.092, -0.049, -0.069, -0.061, -0.062, -0.121, -0.074, -0.089, -0.093, -0.059] | -0.0574 [-0.067, -0.039, -0.057, -0.113, -0.039, -0.056, -0.051, -0.055, -0.053, -0.045] |
| CASF RMSE | 1.102 (0.025) | 1.066 (0.027) | 1.077 (0.017) | 1.338 (0.065) | 1.281 (0.063) | -0.0353 [-0.026, -0.055, -0.039, 0.010, -0.023, -0.034, -0.021, -0.023, -0.065, -0.077] | -0.0244 [-0.034, -0.091, -0.011, 0.009, -0.003, -0.008, -0.060, 0.033, -0.052, -0.029] | 0.2359 [0.285, 0.141, 0.207, 0.201, 0.205, 0.366, 0.213, 0.279, 0.285, 0.177] | 0.1792 [0.198, 0.119, 0.172, 0.370, 0.131, 0.167, 0.147, 0.187, 0.161, 0.142] |
| group_valid Pearson | 0.547 (0.017) | 0.548 (0.017) | 0.545 (0.021) | 0.455 (0.017) | 0.463 (0.026) | 0.0009 [-0.016, 0.010, -0.012, 0.019, 0.009, -0.003, 0.006, -0.009, 0.010, -0.006] | -0.0023 [-0.012, -0.022, -0.002, -0.018, 0.046, -0.037, -0.024, 0.022, 0.039, -0.015] | -0.0918 [-0.077, -0.117, -0.081, -0.104, -0.099, -0.122, -0.077, -0.077, -0.050, -0.114] | -0.0838 [-0.108, -0.100, -0.065, -0.108, -0.103, -0.075, -0.058, -0.071, -0.037, -0.114] |
| group_valid Spearman | 0.497 (0.017) | 0.492 (0.028) | 0.502 (0.025) | 0.431 (0.022) | 0.438 (0.023) | -0.0054 [-0.046, 0.011, -0.005, 0.039, 0.000, -0.009, 0.024, -0.028, -0.024, -0.016] | 0.0047 [-0.018, -0.009, 0.010, -0.001, 0.042, -0.010, -0.009, 0.017, 0.023, 0.003] | -0.0659 [-0.043, -0.084, -0.064, -0.076, -0.062, -0.091, -0.011, -0.094, -0.039, -0.095] | -0.0585 [-0.082, -0.073, -0.059, -0.080, -0.045, -0.047, -0.040, -0.068, -0.012, -0.080] |
| group_valid RMSE | 1.391 (0.046) | 1.415 (0.046) | 1.419 (0.044) | 1.610 (0.059) | 1.591 (0.106) | 0.0242 [0.081, -0.011, -0.035, 0.037, -0.001, 0.022, 0.002, 0.075, -0.006, 0.078] | 0.0280 [0.007, 0.027, -0.046, 0.103, -0.082, 0.127, 0.101, -0.013, -0.065, 0.122] | 0.2194 [0.217, 0.231, 0.167, 0.300, 0.252, 0.294, 0.198, 0.092, 0.109, 0.335] | 0.2004 [0.244, 0.219, 0.044, 0.375, 0.275, 0.200, 0.099, 0.086, 0.068, 0.394] |
| group_valid within-series wSpearman (n>=5) | 0.256 (0.015) | 0.263 (0.017) | 0.249 (0.013) | 0.212 (0.017) | 0.227 (0.013) | 0.0065 [-0.002, 0.051, -0.005, 0.004, -0.004, 0.009, 0.006, 0.000, -0.023, 0.029] | -0.0077 [-0.016, 0.016, 0.029, -0.012, -0.027, -0.015, -0.019, -0.017, -0.016, 0.000] | -0.0443 [-0.027, -0.007, -0.047, -0.061, -0.038, -0.063, -0.022, -0.045, -0.081, -0.052] | -0.0290 [-0.022, 0.006, -0.008, -0.067, -0.006, -0.051, -0.039, -0.043, -0.037, -0.023] |
| FEP pair-head wKendall (n>=10) | nan (nan) | nan (nan) | 0.345 (0.013) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP pair-head wSpearman (n>=10) | nan (nan) | nan (nan) | 0.460 (0.016) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP pair RMSE (n>=10) | nan (nan) | nan (nan) | 1.002 (0.017) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP pair dir acc (n>=10) | nan (nan) | nan (nan) | 0.734 (0.010) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP point-head wKendall (n>=10) | 0.339 (0.015) | 0.345 (0.018) | 0.336 (0.016) | 0.299 (0.024) | 0.307 (0.015) | 0.0067 [0.026, 0.051, 0.008, -0.005, -0.012, -0.003, -0.019, 0.007, -0.016, 0.028] | -0.0028 [-0.013, 0.049, 0.025, -0.010, 0.005, -0.008, -0.013, -0.047, -0.017, 0.002] | -0.0403 [-0.078, 0.019, -0.019, -0.039, -0.011, -0.077, -0.065, -0.027, -0.075, -0.030] | -0.0314 [-0.037, 0.003, -0.026, -0.026, -0.041, -0.044, -0.017, -0.063, -0.050, -0.012] |
| FEP point-head wSpearman (n>=10) | 0.456 (0.017) | 0.465 (0.024) | 0.451 (0.016) | 0.399 (0.030) | 0.410 (0.021) | 0.0091 [0.050, 0.051, 0.011, -0.002, -0.015, 0.006, -0.016, 0.002, -0.024, 0.029] | -0.0047 [-0.008, 0.043, 0.027, -0.014, 0.007, -0.006, -0.013, -0.051, -0.030, -0.003] | -0.0574 [-0.099, 0.011, -0.021, -0.054, -0.021, -0.100, -0.094, -0.040, -0.103, -0.053] | -0.0462 [-0.050, -0.016, -0.041, -0.027, -0.062, -0.056, -0.019, -0.091, -0.072, -0.028] |
| FEP point pair RMSE (n>=10) | 0.970 (0.012) | 0.957 (0.015) | 0.966 (0.018) | 1.028 (0.026) | 1.011 (0.019) | -0.0123 [-0.037, -0.032, -0.011, 0.002, -0.000, -0.012, -0.008, -0.026, -0.007, 0.010] | -0.0038 [-0.009, -0.024, -0.042, 0.015, 0.005, -0.012, -0.014, 0.028, -0.017, 0.030] | 0.0587 [0.117, 0.036, 0.049, 0.081, 0.053, 0.064, 0.051, 0.020, 0.055, 0.062] | 0.0415 [0.038, 0.037, 0.016, 0.072, 0.028, 0.029, 0.013, 0.046, 0.053, 0.082] |
| FEP point pair dir acc (n>=10) | 0.729 (0.010) | 0.733 (0.009) | 0.730 (0.010) | 0.698 (0.015) | 0.706 (0.012) | 0.0042 [0.021, 0.023, 0.003, 0.017, -0.009, 0.003, -0.007, 0.000, -0.017, 0.008] | 0.0007 [0.001, 0.023, 0.026, 0.001, -0.008, -0.005, 0.003, -0.021, -0.015, 0.001] | -0.0306 [-0.037, -0.002, -0.017, -0.024, -0.015, -0.054, -0.044, -0.028, -0.059, -0.025] | -0.0235 [-0.021, -0.010, -0.020, -0.003, -0.044, -0.030, -0.009, -0.049, -0.035, -0.014] |
| FEP pairhead wKendall (n>=10) | nan (nan) | 0.357 (0.011) | nan (nan) | nan (nan) | 0.318 (0.025) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP pairhead wSpearman (n>=10) | nan (nan) | 0.480 (0.016) | nan (nan) | nan (nan) | 0.427 (0.034) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| FEP pairhead dir acc (n>=10) | nan (nan) | 0.743 (0.006) | nan (nan) | nan (nan) | 0.711 (0.017) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pair-head wKendall (n>=5) | nan (nan) | nan (nan) | 0.261 (0.010) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pair-head wSpearman (n>=5) | nan (nan) | nan (nan) | 0.347 (0.014) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pair RMSE (n>=5) | nan (nan) | nan (nan) | 1.178 (0.012) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pair dir acc (n>=5) | nan (nan) | nan (nan) | 0.682 (0.007) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid point-head wKendall (n>=5) | 0.250 (0.011) | 0.262 (0.019) | 0.253 (0.011) | 0.247 (0.012) | 0.263 (0.014) | 0.0126 [-0.014, 0.006, 0.028, -0.014, 0.059, 0.031, 0.006, 0.016, -0.006, 0.016] | 0.0032 [-0.030, 0.017, 0.011, 0.006, 0.003, 0.009, 0.015, 0.009, -0.026, 0.018] | -0.0029 [-0.034, -0.001, 0.019, -0.008, 0.008, -0.019, 0.030, -0.019, -0.012, 0.008] | 0.0129 [0.000, 0.028, 0.019, 0.014, 0.018, 0.019, 0.021, 0.033, -0.024, -0.000] |
| series_valid point-head wSpearman (n>=5) | 0.331 (0.018) | 0.347 (0.026) | 0.336 (0.016) | 0.323 (0.015) | 0.347 (0.019) | 0.0154 [-0.025, 0.001, 0.044, -0.019, 0.076, 0.043, 0.013, 0.020, -0.014, 0.015] | 0.0042 [-0.043, 0.017, 0.016, 0.011, 0.005, 0.013, 0.024, 0.017, -0.041, 0.023] | -0.0086 [-0.048, -0.013, 0.018, -0.011, 0.001, -0.023, 0.039, -0.030, -0.018, -0.000] | 0.0156 [-0.000, 0.023, 0.019, 0.022, 0.021, 0.029, 0.031, 0.045, -0.038, 0.003] |
| series_valid point pair RMSE (n>=5) | 1.113 (0.012) | 1.103 (0.016) | 1.111 (0.011) | 1.153 (0.013) | 1.135 (0.015) | -0.0102 [-0.004, 0.011, -0.015, 0.004, -0.031, -0.037, -0.016, -0.029, 0.032, -0.017] | -0.0021 [0.011, 0.006, -0.019, -0.013, 0.023, -0.019, 0.007, -0.024, 0.027, -0.019] | 0.0405 [0.042, 0.066, 0.035, 0.023, 0.046, 0.031, 0.032, 0.053, 0.057, 0.019] | 0.0223 [0.001, 0.014, 0.018, 0.038, 0.039, 0.011, 0.016, -0.003, 0.058, 0.032] |
| series_valid point pair dir acc (n>=5) | 0.674 (0.008) | 0.682 (0.013) | 0.676 (0.008) | 0.667 (0.008) | 0.680 (0.008) | 0.0079 [-0.009, 0.003, 0.019, -0.012, 0.033, 0.024, 0.002, 0.011, -0.004, 0.011] | 0.0024 [-0.025, 0.010, 0.006, 0.004, 0.003, 0.010, 0.009, 0.009, -0.017, 0.015] | -0.0066 [-0.028, -0.006, 0.008, -0.017, -0.006, -0.014, 0.014, -0.015, -0.009, 0.006] | 0.0056 [-0.004, 0.012, 0.006, 0.004, 0.006, 0.014, 0.013, 0.017, -0.016, 0.003] |
| series_valid pairhead wKendall (n>=5) | nan (nan) | 0.277 (0.016) | nan (nan) | nan (nan) | 0.280 (0.009) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pairhead wSpearman (n>=5) | nan (nan) | 0.365 (0.021) | nan (nan) | nan (nan) | 0.366 (0.012) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| series_valid pairhead dir acc (n>=5) | nan (nan) | 0.692 (0.011) | nan (nan) | nan (nan) | 0.689 (0.007) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pair-head wKendall (n>=5) | nan (nan) | nan (nan) | 0.186 (0.013) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pair-head wSpearman (n>=5) | nan (nan) | nan (nan) | 0.257 (0.017) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pair RMSE (n>=5) | nan (nan) | nan (nan) | 1.366 (0.016) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pair dir acc (n>=5) | nan (nan) | nan (nan) | 0.626 (0.008) | nan (nan) | nan (nan) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid point-head wKendall (n>=5) | 0.185 (0.011) | 0.190 (0.013) | 0.180 (0.009) | 0.152 (0.012) | 0.163 (0.010) | 0.0051 [0.000, 0.038, -0.005, 0.002, 0.001, 0.006, 0.003, 0.002, -0.018, 0.023] | -0.0052 [-0.010, 0.011, 0.019, -0.006, -0.015, -0.008, -0.015, -0.014, -0.014, -0.001] | -0.0330 [-0.020, -0.008, -0.033, -0.045, -0.028, -0.041, -0.018, -0.032, -0.062, -0.043] | -0.0215 [-0.017, 0.003, -0.007, -0.047, -0.001, -0.038, -0.032, -0.029, -0.028, -0.021] |
| group_valid point-head wSpearman (n>=5) | 0.256 (0.015) | 0.263 (0.017) | 0.249 (0.013) | 0.212 (0.017) | 0.227 (0.013) | 0.0065 [-0.002, 0.051, -0.005, 0.004, -0.004, 0.009, 0.006, 0.000, -0.023, 0.029] | -0.0077 [-0.016, 0.016, 0.029, -0.012, -0.027, -0.015, -0.019, -0.017, -0.016, 0.000] | -0.0443 [-0.027, -0.007, -0.047, -0.061, -0.038, -0.063, -0.022, -0.045, -0.081, -0.052] | -0.0290 [-0.022, 0.006, -0.008, -0.067, -0.006, -0.051, -0.039, -0.043, -0.037, -0.023] |
| group_valid point pair RMSE (n>=5) | 1.289 (0.020) | 1.279 (0.010) | 1.293 (0.012) | 1.444 (0.027) | 1.424 (0.042) | -0.0101 [-0.026, -0.020, -0.010, 0.013, -0.028, 0.015, 0.006, -0.011, -0.020, -0.019] | 0.0042 [-0.012, 0.013, -0.011, 0.015, -0.024, 0.043, 0.033, -0.008, -0.003, -0.005] | 0.1549 [0.121, 0.151, 0.143, 0.216, 0.167, 0.166, 0.155, 0.115, 0.178, 0.137] | 0.1350 [0.153, 0.121, 0.079, 0.223, 0.163, 0.131, 0.098, 0.121, 0.121, 0.140] |
| group_valid point pair dir acc (n>=5) | 0.626 (0.006) | 0.629 (0.010) | 0.623 (0.007) | 0.602 (0.009) | 0.612 (0.007) | 0.0038 [-0.006, 0.025, -0.003, 0.004, -0.003, 0.002, 0.001, 0.002, -0.004, 0.020] | -0.0023 [-0.007, 0.001, 0.015, -0.005, -0.011, -0.013, -0.013, -0.001, 0.005, 0.007] | -0.0234 [-0.018, -0.008, -0.023, -0.031, -0.027, -0.036, -0.013, -0.024, -0.038, -0.017] | -0.0140 [-0.014, -0.001, -0.002, -0.032, -0.008, -0.025, -0.018, -0.026, -0.011, -0.003] |
| group_valid pairhead wKendall (n>=5) | nan (nan) | 0.193 (0.014) | nan (nan) | nan (nan) | 0.174 (0.011) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pairhead wSpearman (n>=5) | nan (nan) | 0.265 (0.019) | nan (nan) | nan (nan) | 0.238 (0.014) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| group_valid pairhead dir acc (n>=5) | nan (nan) | 0.630 (0.008) | nan (nan) | nan (nan) | 0.619 (0.007) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Aug kind 3 (A-pos) spread vs crystal (pK) | 0.528 (0.021) | 0.516 (0.010) | 0.543 (0.013) | 0.000 (0.000) | 0.000 (0.000) | -0.0121 [0.001, -0.004, -0.022, -0.015, 0.016, -0.041, -0.046, -0.011, -0.034, 0.034] | 0.0151 [0.011, 0.061, 0.007, 0.000, 0.046, -0.015, -0.006, 0.022, -0.024, 0.047] | -0.5283 [-0.525, -0.510, -0.531, -0.536, -0.493, -0.546, -0.555, -0.532, -0.553, -0.503] | -0.5283 [-0.525, -0.510, -0.531, -0.536, -0.493, -0.546, -0.555, -0.532, -0.553, -0.503] |
| Aug kind 4 (A-twin) rank acc | 0.462 (0.012) | 0.499 (0.013) | 0.470 (0.012) | 0.387 (0.007) | 0.383 (0.011) | 0.0376 [0.042, 0.052, 0.023, 0.050, 0.037, 0.065, 0.040, 0.006, 0.038, 0.023] | 0.0079 [0.021, -0.004, -0.010, 0.026, 0.031, 0.029, 0.003, -0.024, 0.004, 0.003] | -0.0750 [-0.083, -0.081, -0.085, -0.060, -0.078, -0.070, -0.064, -0.083, -0.076, -0.071] | -0.0785 [-0.054, -0.091, -0.101, -0.060, -0.078, -0.075, -0.077, -0.095, -0.068, -0.086] |
| Aug kind 4 (A-twin) gap (pK) | 0.240 (0.018) | 0.327 (0.022) | 0.270 (0.021) | 0.066 (0.020) | 0.092 (0.055) | 0.0872 [0.070, 0.076, 0.124, 0.122, 0.105, 0.115, 0.059, 0.047, 0.087, 0.067] | 0.0296 [0.025, 0.055, 0.029, 0.016, 0.081, 0.053, 0.032, -0.007, 0.021, -0.007] | -0.1743 [-0.210, -0.198, -0.160, -0.125, -0.172, -0.163, -0.172, -0.196, -0.159, -0.188] | -0.1479 [-0.138, -0.207, -0.232, -0.039, -0.113, -0.147, -0.188, -0.176, -0.062, -0.176] |
| Aug kind 4 (A-twin) pairhead rank acc | nan (nan) | 0.524 (0.020) | nan (nan) | nan (nan) | 0.475 (0.024) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Aug kind 4 (A-twin) pairhead gap (logit) | nan (nan) | 0.386 (0.044) | nan (nan) | nan (nan) | 0.311 (0.061) | - | - | - | - |
| Aug kind 5 (C-pos) spread vs crystal (pK) | 0.449 (0.021) | 0.419 (0.008) | 0.465 (0.007) | 0.000 (0.000) | 0.000 (0.000) | -0.0297 [-0.050, -0.014, -0.014, -0.024, -0.004, -0.033, -0.055, -0.043, -0.062, 0.002] | 0.0163 [-0.000, 0.048, 0.023, 0.019, 0.045, 0.022, -0.010, -0.012, 0.001, 0.026] | -0.4488 [-0.464, -0.426, -0.438, -0.452, -0.414, -0.446, -0.471, -0.471, -0.474, -0.431] | -0.4488 [-0.464, -0.426, -0.438, -0.452, -0.414, -0.446, -0.471, -0.471, -0.474, -0.431] |
| Aug kind 6 (C-contact) rank acc | 0.406 (0.021) | 0.424 (0.015) | 0.426 (0.020) | 0.413 (0.015) | 0.421 (0.022) | 0.0176 [0.000, 0.025, 0.020, 0.031, 0.000, 0.027, 0.065, 0.025, -0.003, -0.014] | 0.0198 [-0.031, 0.038, 0.049, 0.003, 0.038, 0.028, 0.089, 0.024, -0.006, -0.034] | 0.0065 [-0.011, 0.011, 0.015, -0.010, 0.007, 0.010, 0.068, 0.025, -0.001, -0.049] | 0.0143 [0.008, -0.004, 0.003, 0.000, -0.001, 0.055, 0.039, 0.014, 0.025, 0.004] |
| Aug kind 6 (C-contact) gap (pK) | 0.039 (0.021) | 0.075 (0.017) | 0.062 (0.032) | 0.022 (0.025) | 0.043 (0.031) | 0.0357 [0.036, 0.008, 0.026, 0.069, 0.014, 0.025, 0.098, 0.050, 0.019, 0.013] | 0.0225 [-0.032, 0.067, 0.030, 0.022, 0.022, 0.005, 0.109, 0.042, -0.027, -0.014] | -0.0170 [-0.050, -0.046, 0.004, -0.008, -0.055, 0.016, 0.039, 0.012, -0.015, -0.068] | 0.0036 [0.028, -0.044, -0.028, 0.007, -0.032, 0.059, 0.047, -0.000, 0.013, -0.013] |
| Aug kind 6 (C-contact) pairhead rank acc | nan (nan) | 0.445 (0.012) | nan (nan) | nan (nan) | 0.492 (0.020) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Aug kind 6 (C-contact) pairhead gap (logit) | nan (nan) | 0.106 (0.014) | nan (nan) | nan (nan) | 0.160 (0.032) | - | - | - | - |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | 0.413 (0.025) | 0.382 (0.019) | 0.413 (0.019) | 0.435 (0.034) | 0.427 (0.039) | -0.0311 [-0.090, 0.017, -0.011, -0.030, -0.043, -0.011, -0.041, -0.008, -0.043, -0.051] | -0.0001 [-0.021, 0.038, 0.033, -0.029, 0.021, -0.028, -0.027, 0.041, 0.003, -0.031] | 0.0220 [-0.029, 0.044, 0.016, -0.050, 0.019, 0.010, 0.081, 0.059, 0.051, 0.019] | 0.0139 [-0.051, 0.058, -0.008, 0.036, -0.003, -0.010, 0.023, 0.116, -0.000, -0.021] |
| Moments R2 pocket (valid) | nan (nan) | 0.447 (0.003) | nan (nan) | nan (nan) | 0.314 (0.010) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 ligand (valid) | nan (nan) | 0.748 (0.005) | nan (nan) | nan (nan) | 0.735 (0.007) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 pocket (series_valid) | nan (nan) | 0.390 (0.006) | nan (nan) | nan (nan) | 0.292 (0.010) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 ligand (series_valid) | nan (nan) | 0.631 (0.011) | nan (nan) | nan (nan) | 0.595 (0.013) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 pocket (group_valid) | nan (nan) | 0.168 (0.012) | nan (nan) | nan (nan) | 0.037 (0.017) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 ligand (group_valid) | nan (nan) | 0.550 (0.012) | nan (nan) | nan (nan) | 0.541 (0.015) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 pocket (test) | nan (nan) | 0.473 (0.005) | nan (nan) | nan (nan) | 0.344 (0.014) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 ligand (test) | nan (nan) | 0.769 (0.005) | nan (nan) | nan (nan) | 0.738 (0.011) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 pocket (aug_valid_all) | nan (nan) | 0.325 (0.009) | nan (nan) | nan (nan) | 0.176 (0.002) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |
| Moments R2 ligand (aug_valid_all) | nan (nan) | 0.716 (0.006) | nan (nan) | nan (nan) | 0.746 (0.010) | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] | nan [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] |

## Pair head vs pointwise head

| metric | tag | reference | pair head (ens) | own point head (ens) | reference point head (ens) | pair - own point (per seed) | pair - reference point (per seed) |
|---|---|---|---|---|---|---|---|
| FEP pairhead wKendall (n>=10) | PM-10 | base-g-10 | 0.429 | 0.425 | 0.429 | 0.0111 [-0.007, -0.000, 0.041, 0.020, 0.013, 0.027, 0.009, 0.001, 0.022, -0.015] | 0.0178 [0.019, 0.051, 0.049, 0.015, 0.001, 0.024, -0.009, 0.008, 0.006, 0.013] |
| FEP pairhead wSpearman (n>=10) | PM-10 | base-g-10 | 0.567 | 0.557 | 0.562 | 0.0153 [-0.008, 0.004, 0.056, 0.026, 0.017, 0.033, 0.011, 0.008, 0.028, -0.022] | 0.0244 [0.042, 0.055, 0.067, 0.024, 0.002, 0.039, -0.005, 0.010, 0.004, 0.007] |
| FEP pairhead dir acc (n>=10) | PM-10 | base-g-10 | 0.789 | 0.784 | 0.784 | 0.0099 [0.000, 0.003, 0.032, 0.013, 0.008, 0.017, 0.006, 0.003, 0.022, -0.005] | 0.0141 [0.021, 0.026, 0.035, 0.030, -0.001, 0.020, -0.001, 0.003, 0.004, 0.003] |
| series_valid pairhead wKendall (n>=5) | PM-10 | base-g-10 | 0.318 | 0.299 | 0.298 | 0.0145 [0.013, 0.009, 0.014, 0.012, 0.003, 0.007, 0.027, 0.031, 0.009, 0.020] | 0.0272 [-0.002, 0.016, 0.042, -0.002, 0.062, 0.037, 0.033, 0.047, 0.003, 0.036] |
| series_valid pairhead wSpearman (n>=5) | PM-10 | base-g-10 | 0.413 | 0.389 | 0.390 | 0.0184 [0.018, 0.014, 0.016, 0.013, 0.001, 0.006, 0.038, 0.041, 0.009, 0.029] | 0.0338 [-0.007, 0.015, 0.059, -0.005, 0.077, 0.048, 0.050, 0.061, -0.005, 0.044] |
| series_valid pairhead dir acc (n>=5) | PM-10 | base-g-10 | 0.717 | 0.706 | 0.708 | 0.0104 [0.013, 0.005, 0.011, 0.006, 0.003, 0.005, 0.021, 0.020, 0.006, 0.015] | 0.0183 [0.003, 0.008, 0.029, -0.006, 0.036, 0.029, 0.023, 0.031, 0.002, 0.026] |
| group_valid pairhead wKendall (n>=5) | PM-10 | base-g-10 | 0.238 | 0.235 | 0.230 | 0.0027 [0.004, 0.006, 0.003, -0.007, 0.002, 0.000, 0.002, 0.006, 0.002, 0.010] | 0.0079 [0.004, 0.043, -0.003, -0.005, 0.003, 0.006, 0.004, 0.008, -0.016, 0.033] |
| group_valid pairhead wSpearman (n>=5) | PM-10 | base-g-10 | 0.326 | 0.322 | 0.317 | 0.0024 [0.005, 0.005, 0.002, -0.007, 0.001, -0.004, -0.001, 0.008, 0.001, 0.013] | 0.0088 [0.004, 0.055, -0.003, -0.003, -0.003, 0.004, 0.005, 0.008, -0.022, 0.042] |
| group_valid pairhead dir acc (n>=5) | PM-10 | base-g-10 | 0.660 | 0.662 | 0.657 | 0.0009 [0.002, -0.001, 0.004, -0.004, 0.004, 0.002, -0.004, 0.004, 0.002, -0.000] | 0.0047 [-0.004, 0.024, 0.001, 0.000, 0.002, 0.004, -0.003, 0.006, -0.002, 0.019] |
| Aug kind 4 (A-twin) pairhead rank acc | PM-10 | base-g-10 | 0.561 | 0.516 | 0.481 | 0.0251 [0.011, 0.023, 0.022, 0.030, 0.021, 0.047, 0.009, 0.051, -0.006, 0.042] | 0.0628 [0.054, 0.075, 0.045, 0.080, 0.059, 0.112, 0.049, 0.057, 0.031, 0.065] |
| Aug kind 6 (C-contact) pairhead rank acc | PM-10 | base-g-10 | 0.467 | 0.443 | 0.411 | 0.0211 [0.017, 0.011, 0.001, 0.003, 0.042, 0.048, 0.014, 0.014, 0.025, 0.035] | 0.0387 [0.017, 0.037, 0.021, 0.034, 0.042, 0.075, 0.079, 0.039, 0.023, 0.021] |
| FEP pair-head wKendall (n>=10) | S-10 | base-g-10 | 0.426 | 0.416 | 0.429 | 0.0085 [0.008, -0.007, 0.010, 0.010, 0.007, 0.004, 0.010, 0.013, 0.017, 0.012] | 0.0058 [-0.006, 0.042, 0.036, -0.000, 0.012, -0.004, -0.003, -0.034, -0.000, 0.014] |
| FEP pair-head wSpearman (n>=10) | S-10 | base-g-10 | 0.562 | 0.551 | 0.562 | 0.0090 [0.014, -0.007, 0.013, 0.014, 0.012, 0.001, 0.006, 0.009, 0.018, 0.010] | 0.0043 [0.007, 0.036, 0.040, -0.000, 0.018, -0.005, -0.006, -0.042, -0.012, 0.007] |
| FEP pair RMSE (n>=10) | S-10 | base-g-10 | 0.883 | 0.893 | 0.890 | 0.0360 [0.041, 0.045, 0.035, 0.033, 0.034, 0.044, 0.026, 0.030, 0.042, 0.030] | 0.0323 [0.033, 0.021, -0.006, 0.048, 0.039, 0.033, 0.012, 0.058, 0.025, 0.061] |
| FEP pair dir acc (n>=10) | S-10 | base-g-10 | 0.787 | 0.783 | 0.784 | 0.0039 [0.006, -0.002, 0.005, 0.006, 0.010, 0.002, 0.003, 0.002, 0.004, 0.004] | 0.0046 [0.006, 0.020, 0.031, 0.008, 0.002, -0.003, 0.005, -0.019, -0.011, 0.005] |
| series_valid pair-head wKendall (n>=5) | S-10 | base-g-10 | 0.300 | 0.291 | 0.298 | 0.0084 [0.019, 0.011, 0.008, 0.003, -0.000, 0.001, 0.015, 0.005, 0.014, 0.008] | 0.0116 [-0.011, 0.028, 0.019, 0.009, 0.003, 0.010, 0.029, 0.014, -0.012, 0.025] |
| series_valid pair-head wSpearman (n>=5) | S-10 | base-g-10 | 0.398 | 0.384 | 0.390 | 0.0115 [0.025, 0.016, 0.013, 0.001, -0.002, 0.001, 0.022, 0.008, 0.020, 0.011] | 0.0157 [-0.018, 0.033, 0.029, 0.012, 0.003, 0.014, 0.046, 0.026, -0.022, 0.034] |
| series_valid pair RMSE (n>=5) | S-10 | base-g-10 | 1.071 | 1.049 | 1.043 | 0.0676 [0.055, 0.062, 0.080, 0.074, 0.064, 0.081, 0.061, 0.065, 0.074, 0.059] | 0.0655 [0.065, 0.067, 0.061, 0.061, 0.087, 0.062, 0.068, 0.041, 0.101, 0.041] |
| series_valid pair dir acc (n>=5) | S-10 | base-g-10 | 0.710 | 0.703 | 0.708 | 0.0054 [0.014, 0.010, 0.004, 0.000, -0.003, -0.001, 0.011, 0.005, 0.009, 0.005] | 0.0078 [-0.011, 0.019, 0.010, 0.004, -0.000, 0.009, 0.020, 0.014, -0.008, 0.020] |
| group_valid pair-head wKendall (n>=5) | S-10 | base-g-10 | 0.226 | 0.229 | 0.230 | 0.0063 [0.015, 0.002, 0.004, 0.002, 0.009, 0.001, 0.008, 0.002, -0.008, 0.028] | 0.0011 [0.006, 0.013, 0.022, -0.004, -0.006, -0.007, -0.007, -0.012, -0.022, 0.027] |
| group_valid pair-head wSpearman (n>=5) | S-10 | base-g-10 | 0.311 | 0.315 | 0.317 | 0.0085 [0.020, 0.003, 0.005, 0.003, 0.013, 0.001, 0.009, 0.002, -0.009, 0.037] | 0.0008 [0.004, 0.019, 0.034, -0.009, -0.014, -0.014, -0.009, -0.016, -0.025, 0.038] |
| group_valid pair RMSE (n>=5) | S-10 | base-g-10 | 1.241 | 1.207 | 1.202 | 0.0730 [0.067, 0.089, 0.068, 0.081, 0.080, 0.071, 0.060, 0.087, 0.073, 0.054] | 0.0771 [0.055, 0.102, 0.057, 0.096, 0.056, 0.114, 0.094, 0.079, 0.070, 0.049] |
| group_valid pair dir acc (n>=5) | S-10 | base-g-10 | 0.653 | 0.658 | 0.657 | 0.0030 [0.010, 0.003, 0.001, -0.002, 0.004, -0.001, 0.003, 0.003, -0.004, 0.013] | 0.0007 [0.003, 0.003, 0.016, -0.007, -0.008, -0.014, -0.009, 0.002, 0.001, 0.020] |
| FEP pairhead wKendall (n>=10) | PM-lig-10 | base-g-lig-10 | 0.389 | 0.375 | 0.376 | 0.0106 [0.030, -0.001, 0.035, 0.014, -0.008, 0.009, -0.009, -0.019, 0.047, 0.009] | 0.0195 [0.071, -0.017, 0.027, 0.027, -0.039, 0.042, 0.039, -0.055, 0.073, 0.026] |
| FEP pairhead wSpearman (n>=10) | PM-lig-10 | base-g-lig-10 | 0.517 | 0.495 | 0.494 | 0.0168 [0.042, 0.006, 0.055, 0.020, -0.003, 0.008, -0.008, -0.025, 0.062, 0.010] | 0.0280 [0.091, -0.021, 0.036, 0.047, -0.044, 0.052, 0.067, -0.076, 0.093, 0.035] |
| FEP pairhead dir acc (n>=10) | PM-lig-10 | base-g-lig-10 | 0.759 | 0.748 | 0.748 | 0.0059 [0.015, 0.003, 0.024, 0.007, 0.001, 0.006, -0.004, -0.015, 0.020, 0.003] | 0.0130 [0.032, -0.004, 0.021, 0.028, -0.028, 0.030, 0.030, -0.037, 0.044, 0.014] |
| series_valid pairhead wKendall (n>=5) | PM-lig-10 | base-g-lig-10 | 0.323 | 0.306 | 0.289 | 0.0175 [0.002, 0.016, 0.029, 0.007, 0.020, 0.023, 0.014, -0.005, 0.044, 0.024] | 0.0332 [0.036, 0.045, 0.028, 0.029, 0.031, 0.061, 0.005, 0.048, 0.032, 0.016] |
| series_valid pairhead wSpearman (n>=5) | PM-lig-10 | base-g-lig-10 | 0.419 | 0.398 | 0.370 | 0.0188 [-0.003, 0.021, 0.037, 0.001, 0.025, 0.026, 0.013, -0.015, 0.055, 0.027] | 0.0429 [0.045, 0.057, 0.039, 0.034, 0.045, 0.078, 0.005, 0.060, 0.035, 0.031] |
| series_valid pairhead dir acc (n>=5) | PM-lig-10 | base-g-lig-10 | 0.717 | 0.708 | 0.693 | 0.0097 [-0.003, 0.014, 0.018, 0.002, 0.015, 0.014, 0.004, -0.005, 0.027, 0.012] | 0.0219 [0.021, 0.032, 0.016, 0.023, 0.026, 0.043, 0.003, 0.027, 0.020, 0.009] |
| group_valid pairhead wKendall (n>=5) | PM-lig-10 | base-g-lig-10 | 0.209 | 0.199 | 0.185 | 0.0112 [0.020, 0.022, 0.015, 0.012, 0.008, 0.012, 0.014, 0.010, 0.003, -0.004] | 0.0228 [0.023, 0.033, 0.041, 0.010, 0.035, 0.015, 0.001, 0.013, 0.037, 0.018] |
| group_valid pairhead wSpearman (n>=5) | PM-lig-10 | base-g-lig-10 | 0.287 | 0.276 | 0.258 | 0.0110 [0.017, 0.025, 0.013, 0.017, 0.012, 0.012, 0.007, 0.014, 0.006, -0.012] | 0.0263 [0.022, 0.038, 0.052, 0.012, 0.044, 0.024, -0.010, 0.016, 0.050, 0.016] |
| group_valid pairhead dir acc (n>=5) | PM-lig-10 | base-g-lig-10 | 0.643 | 0.637 | 0.624 | 0.0069 [0.009, 0.013, 0.008, 0.010, 0.005, 0.011, 0.011, 0.013, -0.005, -0.005] | 0.0163 [0.013, 0.020, 0.029, 0.008, 0.024, 0.021, 0.006, 0.011, 0.023, 0.009] |
| Aug kind 4 (A-twin) pairhead rank acc | PM-lig-10 | base-g-lig-10 | 0.485 | 0.385 | 0.379 | 0.0914 [0.052, 0.106, 0.085, 0.086, 0.116, 0.090, 0.055, 0.094, 0.114, 0.115] | 0.0879 [0.081, 0.096, 0.069, 0.086, 0.116, 0.085, 0.041, 0.081, 0.122, 0.100] |
| Aug kind 6 (C-contact) pairhead rank acc | PM-lig-10 | base-g-lig-10 | 0.516 | 0.429 | 0.416 | 0.0713 [0.049, 0.053, 0.103, 0.097, 0.098, 0.052, 0.070, 0.065, 0.073, 0.052] | 0.0792 [0.069, 0.038, 0.090, 0.107, 0.090, 0.097, 0.042, 0.053, 0.100, 0.105] |

## Shortcut gate: PM-10 vs base-g-10, against PM-lig-10 vs base-g-lig-10

| metric | PM-10 - base-g-10 | PM-lig-10 - base-g-lig-10 | structure-specific gain | verdict |
|---|---|---|---|---|
| FEP wPCC (n>=10) | 0.0087 | 0.0161 | -0.0074 (sd 0.0378) | no structure-specific gain |
| FEP wKendall (n>=10) | 0.0067 | 0.0088 | -0.0022 (sd 0.0408) | no structure-specific gain |
| FEP wSpearman (n>=10) | 0.0091 | 0.0112 | -0.0021 (sd 0.0516) | no structure-specific gain |
| FEP wRMSE (all) | -0.0036 | -0.0195 | -0.0159 (sd 0.0503) | no structure-specific gain |
| series_valid wSpearman (n>=5) | 0.0154 | 0.0241 | -0.0088 (sd 0.0436) | no structure-specific gain |
| series_valid wKendall (n>=5) | 0.0126 | 0.0158 | -0.0031 (sd 0.0323) | no structure-specific gain |
| series_valid wRMSE (all) | -0.0055 | -0.0054 | 0.0001 (sd 0.0302) | within seed noise |
| CASF Pearson | 0.0098 | 0.0194 | -0.0096 (sd 0.0273) | no structure-specific gain |
| CASF RMSE | -0.0353 | -0.0567 | -0.0214 (sd 0.0869) | no structure-specific gain |
| group_valid Pearson | 0.0009 | 0.0080 | -0.0071 (sd 0.0214) | no structure-specific gain |
| group_valid Spearman | -0.0054 | 0.0074 | -0.0128 (sd 0.0375) | no structure-specific gain |
| group_valid RMSE | 0.0242 | -0.0190 | -0.0432 (sd 0.0533) | no structure-specific gain |
| group_valid within-series wSpearman (n>=5) | 0.0065 | 0.0153 | -0.0088 (sd 0.0318) | no structure-specific gain |
| FEP pair-head wKendall (n>=10) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| FEP pair-head wSpearman (n>=10) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| FEP pair RMSE (n>=10) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| FEP pair dir acc (n>=10) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| FEP point-head wKendall (n>=10) | 0.0067 | 0.0088 | -0.0022 (sd 0.0408) | no structure-specific gain |
| FEP point-head wSpearman (n>=10) | 0.0091 | 0.0112 | -0.0021 (sd 0.0516) | no structure-specific gain |
| FEP point pair RMSE (n>=10) | -0.0123 | -0.0173 | -0.0050 (sd 0.0299) | no structure-specific gain |
| FEP point pair dir acc (n>=10) | 0.0042 | 0.0071 | -0.0030 (sd 0.0250) | no structure-specific gain |
| FEP pairhead wKendall (n>=10) (ref: point head) | 0.0178 | 0.0195 | -0.0017 (sd 0.0480) | no structure-specific gain |
| FEP pairhead wSpearman (n>=10) (ref: point head) | 0.0244 | 0.0280 | -0.0036 (sd 0.0607) | no structure-specific gain |
| FEP pairhead dir acc (n>=10) (ref: point head) | 0.0141 | 0.0130 | 0.0011 (sd 0.0263) | within seed noise |
| series_valid pair-head wKendall (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| series_valid pair-head wSpearman (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| series_valid pair RMSE (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| series_valid pair dir acc (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| series_valid point-head wKendall (n>=5) | 0.0126 | 0.0158 | -0.0031 (sd 0.0323) | no structure-specific gain |
| series_valid point-head wSpearman (n>=5) | 0.0154 | 0.0241 | -0.0088 (sd 0.0436) | no structure-specific gain |
| series_valid point pair RMSE (n>=5) | -0.0102 | -0.0182 | -0.0080 (sd 0.0305) | no structure-specific gain |
| series_valid point pair dir acc (n>=5) | 0.0079 | 0.0123 | -0.0044 (sd 0.0207) | no structure-specific gain |
| series_valid pairhead wKendall (n>=5) (ref: point head) | 0.0272 | 0.0332 | -0.0061 (sd 0.0273) | no structure-specific gain |
| series_valid pairhead wSpearman (n>=5) (ref: point head) | 0.0338 | 0.0429 | -0.0091 (sd 0.0356) | no structure-specific gain |
| series_valid pairhead dir acc (n>=5) (ref: point head) | 0.0183 | 0.0219 | -0.0037 (sd 0.0184) | no structure-specific gain |
| group_valid pair-head wKendall (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| group_valid pair-head wSpearman (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| group_valid pair RMSE (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| group_valid pair dir acc (n>=5) (ref: point head) | nan | nan | nan (sd nan) | n/a |
| group_valid point-head wKendall (n>=5) | 0.0051 | 0.0115 | -0.0064 (sd 0.0237) | no structure-specific gain |
| group_valid point-head wSpearman (n>=5) | 0.0065 | 0.0153 | -0.0088 (sd 0.0318) | no structure-specific gain |
| group_valid point pair RMSE (n>=5) | -0.0101 | -0.0199 | -0.0098 (sd 0.0401) | no structure-specific gain |
| group_valid point pair dir acc (n>=5) | 0.0038 | 0.0094 | -0.0056 (sd 0.0162) | no structure-specific gain |
| group_valid pairhead wKendall (n>=5) (ref: point head) | 0.0079 | 0.0228 | -0.0149 (sd 0.0227) | no structure-specific gain |
| group_valid pairhead wSpearman (n>=5) (ref: point head) | 0.0088 | 0.0263 | -0.0175 (sd 0.0324) | no structure-specific gain |
| group_valid pairhead dir acc (n>=5) (ref: point head) | 0.0047 | 0.0163 | -0.0117 (sd 0.0126) | no structure-specific gain |
| Aug kind 3 (A-pos) spread vs crystal (pK) | -0.0121 | 0.0000 | 0.0121 (sd 0.0252) | within seed noise |
| Aug kind 4 (A-twin) rank acc | 0.0376 | -0.0035 | 0.0411 (sd 0.0181) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) gap (pK) | 0.0872 | 0.0264 | 0.0608 (sd 0.0590) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 4 (A-twin) pairhead rank acc (ref: point head) | 0.0628 | 0.0879 | -0.0251 (sd 0.0329) | no structure-specific gain |
| Aug kind 5 (C-pos) spread vs crystal (pK) | -0.0297 | 0.0000 | 0.0297 (sd 0.0223) | passes (gain beyond ligand-only, > 1 sd) |
| Aug kind 6 (C-contact) rank acc | 0.0176 | 0.0079 | 0.0097 (sd 0.0452) | within seed noise |
| Aug kind 6 (C-contact) gap (pK) | 0.0357 | 0.0206 | 0.0151 (sd 0.0473) | within seed noise |
| Aug kind 6 (C-contact) pairhead rank acc (ref: point head) | 0.0387 | 0.0792 | -0.0405 (sd 0.0392) | no structure-specific gain |
| Aug kind 7 (C-exposed) spread vs C-pos (pK) | -0.0311 | -0.0081 | 0.0230 (sd 0.0451) | within seed noise |
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
