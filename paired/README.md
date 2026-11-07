# paired/ — paired-data experiments on AEV-PLIG (Isambard-AI, Nov 2026)

Code, job scripts, tests and reports for the paired-data programme run on the oxpig release
(commit 3ccac4b): within-series ranking losses, pose decoys, Zernike-selected shape twins,
in-place property edits, a Siamese head, a pair head with pocket/ligand moment heads, and the
moments as model inputs — each against a matched baseline with a ligand-only control.

* `src/` — dataset builder (series ids, group-disjoint holdout), series-aware sampler, training
  script with the extra heads/losses, evaluator (FEP benchmark, CASF, held-out series and
  target groups, pose/augmentation discrimination, pairwise ranking), 3D Zernike descriptors
  (`zernike3d.py`, `pocket_fields.py`, `featurise_zernike.py`), pose/twin/edit generators,
  15-PGDH external test, arm comparison.
* `jobs/` — Slurm scripts for Isambard-AI (1 GPU + 72 CPUs per job; archives extracted to
  node-local disk only).
* `tests/` — pytest suite (CPU; 317 tests).
* `docs/` — plans per arm, the data census and design notes.
* `reports/` — results per arm. `REPORT_heads.md` and `REPORT_moments_input.md` carry the
  10-seed verdicts.

The scripts import the unmodified AEV-PLIG modules from this repository via `AEVPLIG_REPO`
and never edit them. Trained models, descriptors and evaluation outputs live on Isambard
scratch and are not in git.
