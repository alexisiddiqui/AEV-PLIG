#!/bin/bash
# PyTorch baseline profile: jobs/profile_torch.py (the paper's loop, seed index 0, 3 timed epochs with the paper's
# DataLoader, torch.profiler over 200 steps, 3 timed epochs with num_workers=8 + pin_memory) under $AEVPLIG_PY,
# from the parent's work dir $PARENT (GraphDataset(root='data') reads data/processed there; nothing is written
# there), with nvidia-smi dmon sampling the GPU in the background. Run alongside profile_jax.sbatch if wanted.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/profile_torch.sbatch
# Outputs in $TREE/work/output/profile/torch: profile_torch.json, key_averages.txt, gpu_util.txt. The JAX
#   profile's REPORT.md picks them up; if this job finishes last, re-render it with
#   $JAX_PY -m aevplig_jax.profile --out $TREE/work/output/profile   (in a job: no login-node compute).
#SBATCH --job-name=aj-prof-torch
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
# the parent's training environment (isambard/jobs/env.sh): no thread limits, no .pyc in the clone or the env
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export AEVPLIG_REPO=/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b
OUT=$TREE/work/output/profile/torch
mkdir -p "$OUT"
echo "[profile-torch] $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# GPU utilisation every 5 s with date and time (profile.py matches samples to each loader's window)
stdbuf -oL nvidia-smi dmon -s u -d 5 -o DT > "$OUT/gpu_util.txt" &
DMON=$!
trap 'kill "$DMON" 2>/dev/null || true' EXIT
cd "$PARENT"
"$AEVPLIG_PY" "$TREE/jobs/profile_torch.py" --out "$OUT" --epochs 3 --profile-steps 200 --workers 8
kill "$DMON" 2>/dev/null || true
echo "[profile-torch] done $(date -Is)"
cat "$OUT/profile_torch.json"
