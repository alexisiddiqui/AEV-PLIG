#!/bin/bash
# JAX loop profile (design section 7): precompute_aev.py on trainside (once), profile.py for the on-the-fly and the
# precomputed AEV path (300 timed steps, 20 traced, same batches), then one full seed (train.py --tag timing
# --seed-index 0 --epochs 200) timed end to end, with nvidia-smi dmon sampling the GPU in the background. Run from
# $TREE with $JAX_PY.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/profile_jax.sbatch
#   profiles only:  sbatch --export=ALL,SKIP_SEED=1 .../jobs/profile_jax.sbatch   (no full seed; profile.json keeps
#                   the earlier seed row and gpu_util.txt is appended to, so its utilisation still renders)
#   Each profile mode summarises its new trace's GPU kernels (time by kernel, family and HLO module, GPU busy share of
#   the traced window) into trace/<mode>/kernels.csv, its profile.json row and REPORT.md, and prints the tables here.
#   Profiling runs XLA op by op; to trace with command buffers on, add
#   XLA_FLAGS=--xla_enable_command_buffers_during_profiling=true to --export.
# Outputs in $TREE/work/output/profile: profile.json, REPORT.md (the PyTorch rows come from torch/profile_torch.json
#   when profile_torch.sbatch has finished), gpu_util.txt, trace/{onthefly,precomputed}/ (with kernels.csv); the AEV table
#   $TREE/work/output/aev_trainside.npz; the timing run $TREE/work/output/timing/ (manifest.json written by the first
#   profile and reused by the seed; seed-0/ is removed and rerun on every submission without SKIP_SEED=1).
#SBATCH --job-name=aj-prof-jax
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STORE=$TREE/work/store
OUT=$TREE/work/output/profile
TABLE=$TREE/work/output/aev_trainside.npz
TAG=timing
RUN=$TREE/work/output/$TAG
LOADER=(--workers 15 --prefetch 4 --prepare-workers 12)
SKIP_SEED=${SKIP_SEED:-0}
cd "$TREE"
mkdir -p "$OUT"
t_job=$(date +%s)
stamp() { echo "[profile-jax] $(date +%H:%M:%S) +$(( $(date +%s) - t_job ))s $*"; }
stamp "job ${SLURM_JOB_ID:-$$} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# GPU utilisation every 5 s with date and time (profile.py matches samples to each phase's window); with
# SKIP_SEED=1 appended, so the kept seed row's window still finds the earlier job's samples
if [ "$SKIP_SEED" = 1 ]; then
  stdbuf -oL nvidia-smi dmon -s u -d 5 -o DT >> "$OUT/gpu_util.txt" &
else
  stdbuf -oL nvidia-smi dmon -s u -d 5 -o DT > "$OUT/gpu_util.txt" &
fi
DMON=$!
trap 'kill "$DMON" 2>/dev/null || true' EXIT

# 1. precomputed AEV table for the A/B (kept between submissions)
if [ ! -s "$TABLE" ]; then
  "$JAX_PY" -m aevplig_jax.precompute_aev --store "$STORE/trainside" --out "$TABLE" --workers 12 --prefetch 4
  stamp "AEV table written"
fi

# 2. step profiles; both modes write $OUT/profile.json and $OUT/REPORT.md
for aev in onthefly precomputed; do
  extra=()
  [ "$aev" = precomputed ] && extra=(--aev-table "$TABLE")
  "$JAX_PY" -m aevplig_jax.profile --store-root "$STORE" --manifest "$RUN/manifest.json" --steps 300 \
    --trace-steps 20 --out "$OUT" --aev "$aev" ${extra[@]+"${extra[@]}"} "${LOADER[@]}"
  stamp "profile $aev done"
done

# 3. one full seed, timed end to end (start-up, 200 epochs with validation, test prediction), unless SKIP_SEED=1
if [ "$SKIP_SEED" = 1 ]; then
  stamp "SKIP_SEED=1: no full seed; re-rendering with the seed row already in profile.json (if any)"
  kill "$DMON" 2>/dev/null || true
  wait "$DMON" 2>/dev/null || true
  "$JAX_PY" -m aevplig_jax.profile --out "$OUT"
else
  rm -rf "$RUN/seed-0"
  t0=$(date +%s)
  "$JAX_PY" -m aevplig_jax.train --store-root "$STORE" --tag "$TAG" --seed-index 0 --epochs 200 \
    --out "$TREE/work/output" "${LOADER[@]}"
  t1=$(date +%s)
  stamp "full seed: $(( t1 - t0 )) s"
  kill "$DMON" 2>/dev/null || true
  wait "$DMON" 2>/dev/null || true
  "$JAX_PY" -m aevplig_jax.profile --out "$OUT" --seed-run "$RUN/seed-0" --seed-window "$t0" "$t1"
fi
stamp "done"
cat "$OUT/REPORT.md"
