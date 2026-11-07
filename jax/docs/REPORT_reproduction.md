# AEV-PLIG in JAX: reproduction report

Design: `docs/superpowers/specs/2026-11-07-aevplig-jax-rewrite-design.md`. Plan: `docs/superpowers/plans/2026-11-07-aevplig-jax-rewrite.md`. Isambard tree `/scratch/u6xq/hussain.u6xq/aevplig-jax` (`$TREE`).

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
