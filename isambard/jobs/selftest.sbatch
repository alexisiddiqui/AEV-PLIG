#!/bin/bash
# Self-test of the training wrapper and ensemble evaluation before the real data exist: builds a toy
# train/valid/test split from the example graphs (data/example_rcsb_graphs.pickle, written by
# predict_example.sbatch) in a throwaway dir on /local, trains seeds 0 and 1 for 2 epochs with jobs/train_seed.py,
# runs jobs/ensemble_eval.py (2 toy members + the released 10), and checks the released-ensemble predictions
# against process_and_predict.py's. Nothing is written to $WORK.
#   sbatch --dependency=afterok:<predict job> /scratch/u6xq/hussain.u6xq/aevplig/jobs/selftest.sbatch
#SBATCH --job-name=aevplig-selftest
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig/jobs/env.sh
T=$LOCAL_ROOT/selftest-${SLURM_JOB_ID:-$$}
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/data/processed" "$T/output/trained_models" "$T/output/predictions"
ln -s "$WORK"/output/trained_models/model_GATv2Net_ligsim90_fep_benchmark* "$T/output/trained_models/"
cd "$T"
python - "$WORK" <<'PY'
import os, sys, pickle, pandas as pd
sys.path.insert(0, os.environ["REPO"])
from utils import GraphDataset
work = sys.argv[1]
with open(f"{work}/data/example_rcsb_graphs.pickle", "rb") as f:
    g = pickle.load(f)
df = pd.read_csv(f"{work}/data/example_rcsb.csv")
df = df[df.unique_id.isin(g)]
gd = {f"{u}_{k}": g[u] for u in df.unique_id for k in range(10)}
tr = [(f"{u}_{k}", y) for k in range(10) for u, y in zip(df.unique_id, df.pK)]
te = [(f"{u}_0", y) for u, y in zip(df.unique_id, df.pK)]
for split, rows in (("train", tr), ("valid", te), ("test", te)):
    GraphDataset(root="data", dataset=f"selftest_{split}", ids=[r[0] for r in rows], y=[r[1] for r in rows], graphs_dict=gd)
print(f"toy split: train {len(tr)}, valid/test {len(te)}")
PY
for i in 0 1; do
  python "$JOBS/train_seed.py" --seed_index=$i --tag=selftest --epochs=2 --dataset=selftest --model=GATv2Net \
    --batch_size=128 --hidden_dim=256 --head=3 --lr=0.00012291937615434127 --activation_function=leaky_relu
done
ls -la output/trained_models/ output/predictions/
python "$JOBS/ensemble_eval.py" --dataset=selftest --tag=selftest
python - "$WORK" <<'PY'
import sys, numpy as np, pandas as pd
work = sys.argv[1]
ev = pd.read_csv("output/predictions/ensemble_eval_selftest_test.csv")
pp = pd.read_csv(f"{work}/output/predictions/example_rcsb_predictions.csv")
ids = pd.read_csv(f"{work}/data/example_rcsb.csv").unique_id
pp = pp.set_index("unique_id").loc[[u for u in ids if u in set(pp.unique_id)]]
d = np.abs(ev["released_mean"].values - pp["preds"].values).max()
print(f"released ensemble via ensemble_eval.py vs process_and_predict.py: max |diff| = {d:.2e}")
sys.exit(0 if d < 1e-3 else 1)
PY
echo "[selftest] ok"
