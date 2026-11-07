#!/bin/bash
# AEV-PLIG retraining, one seed per array task (the 10 seeds of training.py run in parallel).
# Needs data/processed/${DATASET}_{train,valid,test}.pt in the work dir (featurise.sbatch).
#   full ensemble:  sbatch /scratch/u6xq/hussain.u6xq/aevplig/jobs/train.sbatch
#   smoke test:     sbatch --array=0 --export=ALL,EPOCHS=2,TAG=smoke /scratch/u6xq/hussain.u6xq/aevplig/jobs/train.sbatch
# Outputs: output/trained_models/${TAG}_model_GATv2Net_${DATASET}_<i>.model, ${TAG}_model_GATv2Net_${DATASET}.pickle,
#          output/predictions/${TAG}_${DATASET}_test_seed<i>.csv
#SBATCH --job-name=aevplig-train
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --array=0-9
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%A_%a.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig/jobs/env.sh
EPOCHS=${EPOCHS:-200}
TAG=${TAG:-retrain}
echo "[train] $(date -Is) task ${SLURM_ARRAY_TASK_ID} epochs $EPOCHS tag $TAG on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$JOBS/train_seed.py" --seed_index="${SLURM_ARRAY_TASK_ID}" --tag="$TAG" --epochs="$EPOCHS" \
  --model=GATv2Net --dataset="$DATASET" --batch_size=128 --hidden_dim=256 --head=3 \
  --lr=0.00012291937615434127 --activation_function=leaky_relu
echo "[train] done $(date -Is)"
