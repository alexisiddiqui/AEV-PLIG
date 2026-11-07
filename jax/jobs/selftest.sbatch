#!/bin/bash
# Self-test on Isambard: the CPU pytest suite under $JAX_PY (torch-marked tests deselected), then a GPU smoke
# of train.py on the real trainside store (first 512 training ids, 2 epochs, full valid/test splits) under the
# tag selftest in $TREE/work/output/selftest (removed first). Checks history.jsonl has 2 rows and the test
# prediction CSV exists. Run from $TREE.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/selftest.sbatch
#SBATCH --job-name=aj-selftest
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=$TREE/work/output/selftest
TAG=selftest
cd "$TREE"
t0=$(date +%s)
stamp() { echo "[selftest] $(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s $*"; }
stamp "job ${SLURM_JOB_ID:-$$} on $(hostname)"
"$JAX_PY" -c "import jax; print('[selftest] jax', jax.__version__, jax.devices())"

# 1. the test suite on the CPU, as locally (torch-marked tests need torch_geometric, absent from the jax-Ka
#    venv; GPU TF32 matmuls would miss the parity tolerances); no .pytest_cache (inode budget)
JAX_PLATFORMS=cpu "$JAX_PY" -m pytest -m "not torch" -q -p no:cacheprovider
stamp "pytest passed"

# 2. GPU smoke: train.py on the real store
rm -rf "$OUT"
"$JAX_PY" -m aevplig_jax.train --store-root "$TREE/work/store" --tag "$TAG" --seed-index 0 --epochs 2 \
  --limit 512 --out "$OUT" --workers 15 --prefetch 4 --prepare-workers 12
rows=$(wc -l < "$OUT/$TAG/seed-0/history.jsonl")
pred=$OUT/predictions/${TAG}_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_test_seed0.csv
[ "$rows" -eq 2 ] || { echo "[selftest] history.jsonl has $rows rows, expected 2" >&2; exit 1; }
[ -s "$pred" ] || { echo "[selftest] missing $pred" >&2; exit 1; }
stamp "smoke passed: $rows history rows, $(( $(wc -l < "$pred") - 1 )) test predictions"
cat "$OUT/$TAG/seed-0/history.jsonl" "$OUT/$TAG/seed-0/selection.json"
