#!/bin/bash
# End-to-end env check: process_and_predict.py (CPU featurisation + GPU inference) with the released ensemble.
# The repo's data/example_dataset.csv points at data/example_database/*, which is not shipped, so the first run
# builds a small RCSB stand-in (jobs/fetch_example.py -> data/example_rcsb.csv, ~12 complexes; needs internet).
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig/jobs/predict_example.sbatch
# Output: output/predictions/example_rcsb_predictions.csv
#SBATCH --job-name=aevplig-predict
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig/jobs/env.sh
CSV=${CSV:-data/example_rcsb.csv}
NAME=$(basename "$CSV" .csv)
[ -f "$CSV" ] || python "$JOBS/fetch_example.py" "${N_EXAMPLE:-12}"
rm -f "data/processed/$NAME.pt"
t=$(date +%s)
python "$REPO/process_and_predict.py" --dataset_csv="$CSV" --data_name="$NAME" \
  --trained_model_name=model_GATv2Net_ligsim90_fep_benchmark --num_workers=16
echo "[predict] process_and_predict.py $(( $(date +%s) - t ))s"
python - "$NAME" <<'PY'
import sys, pandas as pd, numpy as np
d = pd.read_csv(f"output/predictions/{sys.argv[1]}_predictions.csv")
print(d[["unique_id", "pK", "preds"] + [c for c in d.columns if c.startswith("preds_")][:3]].round(3).to_string(index=False))
print(f"n={len(d)}  Pearson(pK, preds)={np.corrcoef(d.pK, d.preds)[0, 1]:.3f}  RMSE={np.sqrt(((d.pK - d.preds) ** 2).mean()):.3f}"
      f"  member spread (mean sd)={d[[c for c in d.columns if c.startswith('preds_')]].std(axis=1).mean():.3f}")
PY
