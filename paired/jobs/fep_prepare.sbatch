#!/bin/bash
#SBATCH --job-name=ap-fep-prepare
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# FEP benchmark: prepare (fep_prepare.py), featurise with the clone (fep_featurise.py), then the
# featurisation check: the released ensemble (model_GATv2Net_ligsim90_fep_benchmark_{0..9})
# evaluated with eval_paired.py and compared with the paper's predictions
# -> $WORK/output/eval/released-check/SUMMARY.md (expect Pearson > 0.99 over >= 1,100 ligands).
# Archives are extracted under /local and removed on exit; parent inputs are read-only.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

RAW=/scratch/u6xq/hussain.u6xq/aevplig/raw
BENCH_TGZ=$RAW/schrodinger_public_binding_free_energy_benchmark-main.tar.gz
PAPER_TGZ=$RAW/aev-plig-paper-main.tar.gz
PAPER_MEMBER=AEV-PLIG_fep_benchmark_pdbbind_U_bindingnet_U_bindingdb_ligsim90_predictions.csv
RELEASED=$PARENT_WORK/output/trained_models/model_GATv2Net_ligsim90_fep_benchmark
TMPD=/local/user/$(id -u)/aevplig-paired/fep-${SLURM_JOB_ID}
mkdir -p "$TMPD/bench" "$TMPD/paper"
trap 'rm -rf "$TMPD"' EXIT

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
tar -xzf "$BENCH_TGZ" -C "$TMPD/bench" --no-same-owner
tar -xzf "$PAPER_TGZ" -C "$TMPD/paper" --no-same-owner --wildcards "*/$PAPER_MEMBER"
INPUTS=$(find "$TMPD/bench" -maxdepth 3 -type d -name fep_benchmark_inputs -print -quit)
PAPER_CSV=$(find "$TMPD/paper" -type f -name "$PAPER_MEMBER" -print -quit)
[ -n "$INPUTS" ] || { echo "ERROR: no fep_benchmark_inputs/ in $BENCH_TGZ"; exit 1; }
[ -n "$PAPER_CSV" ] || { echo "ERROR: no $PAPER_MEMBER in $PAPER_TGZ"; exit 1; }
BENCH_ROOT=$(dirname "$INPUTS")
echo "benchmark root $BENCH_ROOT; paper csv $PAPER_CSV ($(wc -l < "$PAPER_CSV") lines)"

mkdir -p "$WORK/data/processed" "$WORK/output/eval"
python "$SRC/fep_prepare.py" --benchmark-dir "$BENCH_ROOT" --out-dir "$WORK/data" \
    --structures-dir "$WORK/data/fep_structures"
python "$SRC/fep_featurise.py" --csv "$WORK/data/fep_benchmark.csv" --work-dir "$WORK" \
    --num-workers "${SLURM_CPUS_PER_TASK:-72}"

MODELS=()
for i in 0 1 2 3 4 5 6 7 8 9; do
    MODELS+=("${RELEASED}_${i}.model")
done
EXTRA=()
if [ -f "$WORK/data/paired_meta.csv" ]; then
    EXTRA+=(--meta "$WORK/data/paired_meta.csv")
fi
if [ -f "$WORK/data/processed/paired_test.pt" ]; then
    EXTRA+=(--test-pt "$WORK/data/processed/paired_test.pt")
fi
python "$SRC/eval_paired.py" --models "${MODELS[@]}" --scaler "${RELEASED}.pickle" \
    --fep-csv "$WORK/data/fep_benchmark.csv" --fep-pt "$WORK/data/processed/fep_benchmark.pt" \
    --paper-csv "$PAPER_CSV" --out-dir "$WORK/output/eval/released-check" ${EXTRA[@]+"${EXTRA[@]}"}

echo "---- fep_prepare_summary.json (head)"
head -c 2000 "$WORK/data/fep_prepare_summary.json"; echo
echo "---- released-check/SUMMARY.md"
cat "$WORK/output/eval/released-check/SUMMARY.md"
echo "done $(date -Is)"
