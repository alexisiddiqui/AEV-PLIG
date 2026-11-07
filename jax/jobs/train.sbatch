#!/bin/bash
# JAX AEV-PLIG training, one seed per array task (the paper's 10 seeds in parallel), from the trainside
# store: python -m aevplig_jax.train (src/aevplig_jax/train.py). Run from $TREE with $JAX_PY.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   full ensemble:  sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/train.sbatch
#   other tag/len:  sbatch --array=0 --export=ALL,TAG=jax-short,EPOCHS=20 .../jobs/train.sbatch
#   strict float32: sbatch --export=ALL,TAG=jax-fp32,PRECISION=highest .../jobs/train.sbatch
# Outputs under $TREE/work/output: $TAG/manifest.json (shared by the seeds), $TAG/seed-<i>/{protocol.json,
#   history.jsonl, selection.json, test_predictions.csv, checkpoints/{best,latest}/state.npz} and
#   predictions/${TAG}_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_test_seed<i>.csv.
# A seed directory that already holds checkpoints/latest is refused; continue it by running train.py with --resume.
#SBATCH --job-name=aj-train
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs/%x-%A_%a.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TAG=${TAG:-jax}
EPOCHS=${EPOCHS:-200}
cd "$TREE"
echo "[train] $(date -Is) task ${SLURM_ARRAY_TASK_ID} tag $TAG epochs $EPOCHS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$JAX_PY" -m aevplig_jax.train --store-root "$TREE/work/store" --tag "$TAG" \
  --seed-index "$SLURM_ARRAY_TASK_ID" --epochs "$EPOCHS" --out "$TREE/work/output" \
  --workers 15 --prefetch 4 --prepare-workers 12 ${PRECISION:+--precision "$PRECISION"}
echo "[train] done $(date -Is)"
