#!/bin/bash
# Ensemble evaluation on the test split: retrained (${TAG:-retrain}) vs released AEV-PLIG ensembles.
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig/jobs/eval.sbatch          (after the train array; or --dependency=afterok:<array id>)
#SBATCH --job-name=aevplig-eval
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig/jobs/env.sh
python "$JOBS/ensemble_eval.py" --dataset="$DATASET" --tag="${TAG:-retrain}" ${EVAL_ARGS:-}
