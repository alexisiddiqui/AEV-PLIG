#!/bin/bash
#SBATCH --job-name=ap-selftest
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=00:30:00
#SBATCH --array=0
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%A_%a.out
#
# GPU self-test of train_paired.py on the small dataset (jobs/build_small.sbatch):
# tags self-none (MSE only) and self-dh (MSE + 0.3 delta-Huber) run back to back, then the
# last epoch log line of each is checked (self-dh must have a non-zero pair term) and every
# model, scaler and prediction CSV must exist. Exit status 1 on any failure.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

EPOCHS=${EPOCHS:-2}
DATA_DIR=${DATA_DIR:-$WORK/data_small}
SEED_INDEX=${SLURM_ARRAY_TASK_ID:-0}
TMP=/local/user/$(id -u)/aevplig-paired/selftest-${SLURM_JOB_ID}
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is) data $DATA_DIR epochs $EPOCHS"
nvidia-smi -L || true
touch "$TMP/start"  # outputs must be newer than this (no stale files from an earlier run)

run() {
    local tag=$1
    shift
    python "$SRC/train_paired.py" --data-dir "$DATA_DIR" --seed_index "$SEED_INDEX" \
        --tag "$tag" --epochs "$EPOCHS" "$@" 2>&1 | tee "$TMP/$tag.log"
}
run self-none --pair-loss none
run self-dh --pair-loss delta_huber --pair-weight 0.3

fail=0
for tag in self-none self-dh; do
    line=$(grep -E "^epoch ${EPOCHS} loss " "$TMP/$tag.log" || true)
    if [ -z "$line" ]; then
        echo "SELFTEST FAIL $tag: no 'epoch $EPOCHS' log line"
        fail=1
    else
        echo "SELFTEST $tag: $line"
    fi
    for f in output/trained_models/${tag}_${SEED_INDEX}.model output/trained_models/${tag}.pickle \
             output/predictions/${tag}_{train,valid,test,series_valid}_seed${SEED_INDEX}.csv; do
        if [ -s "$f" ] && [ "$f" -nt "$TMP/start" ]; then
            echo "SELFTEST ok $f"
        else
            echo "SELFTEST FAIL missing or stale $f"
            fail=1
        fi
    done
done
pair=$(grep -E "^epoch ${EPOCHS} loss " "$TMP/self-dh.log" \
       | awk '{for (i = 1; i < NF; i++) if ($i == "pair") print $(i + 1)}' || true)
if awk -v p="${pair:-0}" 'BEGIN { exit !(p > 0) }'; then
    echo "SELFTEST self-dh pair term $pair > 0"
else
    echo "SELFTEST FAIL self-dh pair term '${pair}' is not > 0"
    fail=1
fi
if [ "$fail" != 0 ]; then
    echo "SELFTEST FAILED"
    exit 1
fi
echo "SELFTEST PASS"
