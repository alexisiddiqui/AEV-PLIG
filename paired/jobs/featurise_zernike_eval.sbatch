#!/bin/bash
#SBATCH --job-name=ap-featurise-zernike-eval
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Arm Z: 3D Zernike descriptors (pocket + ligand, 7 property channels) of the evaluation sets, for models with
# moments inputs (eval_paired.py --zernike-eval):
#   $WORK/data/fep_benchmark.csv (jobs/fep_prepare.sbatch)  -> $WORK/data/zernike_fep.npz
#   $WORK/data/pgdh.csv          (jobs/pgdh_prepare.sbatch) -> $WORK/data/zernike_pgdh.npz
# featurise_zernike.py --csv reads each row's sdf_file / pdb_file (the structures already under
# $WORK/data/fep_structures/ and $WORK/data/pgdh_structures/; relative paths are taken under $WORK, absolute ones as
# they are), with the training run's geometry: shell pocket (8 A cutoff), descriptor radius 12 A, ligand radius 12 A,
# order 12, the same channel code. No archive is extracted. Each output gets <stem>_summary.json; the ok / failed
# counts and the first failures are printed. A missing input CSV is reported and the job exits 1 at the end (the
# other set is still computed).
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

NCPU=${SLURM_CPUS_PER_TASK:-72}
GEOM=(--pocket-mode shell --order 12 --radius 12.0 --cutoff 8.0 --ligand-radius 12.0)
FAILED=()

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is) (cwd $(pwd))"
for item in fep_benchmark:zernike_fep:fep_prepare pgdh:zernike_pgdh:pgdh_prepare; do
    IFS=: read -r STEM OUT_STEM PREP <<< "$item"
    CSV=$WORK/data/$STEM.csv
    OUT=$WORK/data/$OUT_STEM.npz
    if [ ! -s "$CSV" ]; then
        echo "ERROR: missing $CSV (jobs/$PREP.sbatch writes it)"
        FAILED+=("$CSV")
        continue
    fi
    echo "---- featurise_zernike --csv $CSV -> $OUT $(date -Is): ${GEOM[*]}"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$SRC/featurise_zernike.py" --csv "$CSV" --work-dir "$WORK" --out "$OUT" --workers "$NCPU" "${GEOM[@]}" \
        --variant-note "arm Z evaluation set $(basename "$CSV")"
    python - "$OUT" <<'PY'
import sys

import numpy as np

path = sys.argv[1]
with np.load(path, allow_pickle=False) as z:
    ok, uid, err = z["ok"], z["unique_id"], z["error"]
print(f"{path}: rows {len(ok)} ok {int(ok.sum())} failed {int((~ok).sum())}")
for u, e in list(zip(uid[~ok], err[~ok]))[:20]:
    print(f"  failed {u}: {e}")
PY
done

echo "---- outputs"
ls -la "$WORK"/data/zernike_fep.npz "$WORK"/data/zernike_pgdh.npz 2>/dev/null || true
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
echo "done $(date -Is)"
