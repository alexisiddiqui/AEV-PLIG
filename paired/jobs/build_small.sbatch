#!/bin/bash
#SBATCH --job-name=ap-build-small
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Small paired dataset for the selftest: first 40 train series per source plus all valid/test rows.
# Reads the parent pickles/CSVs read-only; writes paired_meta.csv, paired_summary.json and
# processed/paired_{train,valid,test,series_valid}.pt under the output dir.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

OUT=$WORK/data_small
INDEX_TGZ=/scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_plain_text_index.tar.gz

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
python "$SRC/build_paired_dataset.py" --parent-data "$PARENT_WORK/data" \
    --pdbbind-index-tgz "$INDEX_TGZ" --out-dir "$OUT" --subset-series 40
cat "$OUT/paired_summary.json"
ls -la "$OUT/processed"
echo "done $(date -Is)"
