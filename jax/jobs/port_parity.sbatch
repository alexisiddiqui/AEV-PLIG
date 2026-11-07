#!/bin/bash
# Weight-port parity (design acceptance 2): the parent's retrained PyTorch seed 0 ported to JAX against its PyTorch
# predictions. Two environments: (1) jobs/export_torch_model.py under $AEVPLIG_PY (torch, sklearn) writes the state
# dict and the label scaler to one .npz; (2) python -m aevplig_jax.port_parity under $JAX_PY ports it, predicts the
# trainside test split (CASF) and the fep store at matmul precision highest, and compares with pred_0 of the paired
# tree's PyTorch predictions ($PAIRED/output/eval/retrain10/predictions_{test,fep}.csv). Run from $TREE.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/port_parity.sbatch
# Outputs in $TREE/work/output/parity: torch_seed0.npz, parity.csv (per set: max/mean abs difference, truth checks,
#   passed), parity_rows.csv (per complex), REPORT.md. The job fails when a set misses max abs < 1e-3 pK.
#SBATCH --job-name=aj-parity
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATASET=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark
MODEL=$PARENT/output/trained_models/retrain_model_GATv2Net_${DATASET}_0.model
SCALER=$PARENT/output/trained_models/retrain_model_GATv2Net_${DATASET}.pickle
REFERENCE=$PAIRED/output/eval/retrain10
OUT=$TREE/work/output/parity
cd "$TREE"
echo "[parity] $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"

# 1. state dict + scaler -> npz (AEV-PLIG env: torch, sklearn; aevplig_jax.torch_port imports numpy only)
"$AEVPLIG_PY" "$TREE/jobs/export_torch_model.py" --model "$MODEL" --scaler "$SCALER" --out "$OUT/torch_seed0.npz"

# 2. port, predict and compare (jax-Ka venv)
status=0
"$JAX_PY" -m aevplig_jax.port_parity --npz "$OUT/torch_seed0.npz" --store-root "$TREE/work/store" \
  --reference-dir "$REFERENCE" --out "$OUT" --workers 15 --prefetch 4 --prepare-workers 12 || status=$?
[ -f "$OUT/REPORT.md" ] && cat "$OUT/REPORT.md"
echo "[parity] done $(date -Is), exit $status"
exit "$status"
