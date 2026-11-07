#!/bin/bash
# Ensemble evaluation of a JAX training tag: predict.py with every seed's best checkpoint on the CASF test split
# (trainside, split test), the FEP benchmark store and the 15-PGDH store, then evaluate.py for each and a
# SUMMARY.md table. Predictions use the matmul precision the seeds trained with (flags.precision of every
# $RUN/seed-*/protocol.json, train.run_precision; the job stops if they disagree), and the SUMMARY.md header
# states it. Run from $TREE with $JAX_PY.
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/output/logs
#   sbatch --export=ALL,TAG=jax /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/eval.sbatch
#   (after training: --dependency=afterok:<aj-train array job id>)
# Outputs in $TREE/work/output/eval/$TAG: predictions_{test,fep,pgdh}.csv (unique_id, subset, series, ref_pdb,
#   truth, pred, pred_0..), casf_summary.csv, fep_summary.csv, fep_per_series.csv, pgdh_summary.csv, SUMMARY.md.
#SBATCH --job-name=aj-eval
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
TAG=${TAG:-jax}
STORE=$TREE/work/store
RUN=$TREE/work/output/$TAG
OUT=$TREE/work/output/eval/$TAG
CHECKPOINTS="$RUN/seed-*/checkpoints/best"
cd "$TREE"
[ -f "$RUN/manifest.json" ] || { echo "missing $RUN/manifest.json (train tag $TAG first)" >&2; exit 1; }
n=$( (compgen -G "$CHECKPOINTS" || true) | wc -l)
[ "$n" -gt 0 ] || { echo "no best checkpoints under $RUN" >&2; exit 1; }
PRECISION=$("$JAX_PY" -c 'import sys; from aevplig_jax.train import run_precision; print(run_precision(sys.argv[1]))' "$RUN")
case $PRECISION in
  highest) PRECISION_TEXT="highest (strict float32)" ;;
  default) PRECISION_TEXT="default (TF32 on the GPU)" ;;
  *) PRECISION_TEXT=$PRECISION ;;
esac
echo "[eval] $(date -Is) tag $TAG: $n best checkpoints on $(hostname); matmul precision $PRECISION_TEXT"
mkdir -p "$OUT"

predict() {  # predict <name> <store> [args...]
  local name=$1 store=$2; shift 2
  "$JAX_PY" -m aevplig_jax.predict --store "$STORE/$store" --checkpoints "$CHECKPOINTS" \
    --manifest "$RUN/manifest.json" --out "$OUT/predictions_$name.csv" \
    --precision "$PRECISION" --workers 15 --prefetch 4 --prepare-workers 12 "$@"
}
predict test trainside --split test
predict fep fep
predict pgdh pgdh

"$JAX_PY" -m aevplig_jax.evaluate casf --predictions "$OUT/predictions_test.csv" --out-dir "$OUT"
"$JAX_PY" -m aevplig_jax.evaluate fep --predictions "$OUT/predictions_fep.csv" --out-dir "$OUT"
"$JAX_PY" -m aevplig_jax.evaluate pgdh --predictions "$OUT/predictions_pgdh.csv" --out-dir "$OUT"
"$JAX_PY" -m aevplig_jax.evaluate summary --eval-dir "$OUT" --title "JAX AEV-PLIG ensemble: $TAG ($n members, matmul precision $PRECISION_TEXT)"
echo "[eval] done $(date -Is): $OUT/SUMMARY.md"
