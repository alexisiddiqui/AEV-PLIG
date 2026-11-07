#!/bin/bash
#SBATCH --job-name=ap-pgdh-prepare
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# External set: 15-PGDH, Rich, Dodda & Price 2026 (ChemMedChem, doi 10.1002/cmdc.70531), 43
# inhibitors in series S1..S4, from github.com/inductive-bio-misc-collab/ml-affinity-15-PGDH-case-study
# (no licence file: evaluation use only).
#  1. the repo tarball: reused from PGDH_TGZ, else from the parent's raw/ (read-only) if a copy is
#     already there, else downloaded with curl into PGDH_TGZ (default $PAIRED_ROOT/raw/: the
#     plan's Global Constraints forbid writing under /scratch/u6xq/hussain.u6xq/aevplig/; set
#     PGDH_TGZ=/scratch/u6xq/hussain.u6xq/aevplig/raw/ml-affinity-15-PGDH-case-study-main.tar.gz
#     to download there instead);
#  2. extracted under /local (removed on exit); pgdh_prepare.py -> $WORK/data/pgdh_structures/,
#     CSVs + summary copied to $WORK/data/ (pgdh.csv, pgdh_paper.csv, pgdh_paper_finetuned.csv,
#     pgdh_fepplus.csv, pgdh_summary.json, pgdh_residue_census.csv);
#  3. fep_featurise.py hard-codes the data name fep_benchmark (no flag) and stages its CSV at
#     <work-dir>/data/fep_benchmark.csv, so it runs with --work-dir on /local (never on $WORK,
#     where it would overwrite the FEP benchmark's csv/pickle/.pt); its outputs are copied to
#     $WORK/data/{pgdh_processed.csv, pgdh_graphs.pickle, processed/pgdh.pt, pgdh_featurise_summary.json};
#  4. eval_paired.py on processed/pgdh.pt for three model sets -> $WORK/output/eval/pgdh-<name>/:
#     released (the AEV-PLIG release, = the paper's "original" weights: the reproduction test is
#     SUMMARY.md's Pearson against pgdh_paper.csv, expect ~1), retrain10 (the parent's 10-seed
#     retrain), base (arm-1 base, seeds 0-2). Then pgdh_prepare.py --score-eval-dir writes
#     pgdh_spearman.csv: Spearman pooled / S1..S4 / S1&S2 (the paper's Table 2 groups) for ours, the
#     paper's original and fine-tuned AEV-PLIG and FEP+, on the same ligands.
# A model set whose files are missing is skipped with a WARNING; the job exits 1 if any set failed.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

URL=https://codeload.github.com/inductive-bio-misc-collab/ml-affinity-15-PGDH-case-study/tar.gz/refs/heads/main
TGZ_NAME=ml-affinity-15-PGDH-case-study-main.tar.gz
PARENT_TGZ=/scratch/u6xq/hussain.u6xq/aevplig/raw/$TGZ_NAME
PGDH_TGZ=${PGDH_TGZ:-$PAIRED_ROOT/raw/$TGZ_NAME}
STRUCT=$WORK/data/pgdh_structures
PARENT_MODELS=$PARENT_WORK/output/trained_models
TMPD=/local/user/$(id -u)/aevplig-paired/pgdh-${SLURM_JOB_ID}
mkdir -p "$TMPD/repo" "$TMPD/feat"
trap 'rm -rf "$TMPD"' EXIT

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"

# ---- 1. tarball
if [ -f "$PGDH_TGZ" ]; then
    TGZ=$PGDH_TGZ
elif [ -f "$PARENT_TGZ" ]; then
    TGZ=$PARENT_TGZ
else
    mkdir -p "$(dirname "$PGDH_TGZ")"
    PART="$PGDH_TGZ.part-${SLURM_JOB_ID}"
    curl -sSL --fail --retry 3 -o "$PART" "$URL"
    tar -tzf "$PART" > /dev/null
    mv "$PART" "$PGDH_TGZ"
    TGZ=$PGDH_TGZ
    echo "downloaded $URL"
fi
echo "tarball $TGZ ($(stat -c %s "$TGZ") bytes, sha256 $(sha256sum "$TGZ" | cut -c1-16))"

# ---- 2. prepare
tar -xzf "$TGZ" -C "$TMPD/repo" --no-same-owner
POSES=$(find "$TMPD/repo" -maxdepth 3 -type f -path "*/data/poses.sdf" -print -quit)
[ -n "$POSES" ] || { echo "ERROR: no data/poses.sdf in $TGZ"; exit 1; }
REPO=$(dirname "$(dirname "$POSES")")
for f in data/reference.pdb scoring/vina_predictions.sdf scoring/aevplig_original.csv; do
    [ -f "$REPO/$f" ] || echo "WARNING: $f missing from the repo"
done

mkdir -p "$WORK/data/processed" "$WORK/output/eval"
rm -rf "$STRUCT"
python "$SRC/pgdh_prepare.py" --repo-dir "$REPO" --out-dir "$STRUCT"
for f in pgdh.csv pgdh_paper.csv pgdh_paper_finetuned.csv pgdh_fepplus.csv pgdh_summary.json \
         pgdh_residue_census.csv; do
    if [ -f "$STRUCT/$f" ]; then
        cp -f "$STRUCT/$f" "$WORK/data/$f"
    else
        echo "WARNING: $STRUCT/$f not written"
    fi
done
CENSUS=$WORK/data/pgdh_residue_census.csv
[ -f "$CENSUS" ] || CENSUS=/dev/null   # never fall back to the FEP benchmark's census

# ---- 3. featurise (isolated work dir: fep_featurise.py writes data/fep_benchmark* under it)
python "$SRC/fep_featurise.py" --csv "$WORK/data/pgdh.csv" --work-dir "$TMPD/feat" \
    --num-workers "${SLURM_CPUS_PER_TASK:-72}"
cp -f "$TMPD/feat/data/fep_benchmark_processed.csv" "$WORK/data/pgdh_processed.csv"
cp -f "$TMPD/feat/data/fep_benchmark_graphs.pickle" "$WORK/data/pgdh_graphs.pickle"
cp -f "$TMPD/feat/data/processed/fep_benchmark.pt" "$WORK/data/processed/pgdh.pt"
cp -f "$TMPD/feat/data/fep_featurise_summary.json" "$WORK/data/pgdh_featurise_summary.json"
echo "---- pgdh_featurise_summary.json"
cat "$WORK/data/pgdh_featurise_summary.json"; echo

# ---- 4. evaluate
META=()
if [ -f "$WORK/data/paired_meta.csv" ]; then
    META+=(--meta "$WORK/data/paired_meta.csv")
fi
FAILED=()
run_eval() {  # run_eval NAME SCALER MODEL...
    local name=$1 scaler=$2
    shift 2
    local f out=$WORK/output/eval/pgdh-$name
    for f in "$@" "$scaler"; do
        if [ ! -f "$f" ]; then
            echo "WARNING: pgdh-$name skipped: missing $f"
            FAILED+=("$name(missing)")
            return 0
        fi
    done
    echo "==== pgdh-$name: $# models, scaler $scaler"
    if python "$SRC/eval_paired.py" --models "$@" --scaler "$scaler" \
        --fep-csv "$WORK/data/pgdh.csv" --fep-pt "$WORK/data/processed/pgdh.pt" ${META[@]+"${META[@]}"} \
        --paper-csv "$WORK/data/pgdh_paper.csv" --residue-census "$CENSUS" --out-dir "$out"; then
        grep -h "Pearson(ours, paper preds)" "$out/SUMMARY.md" || true
        cat "$out/fep_per_series.csv"
        python "$SRC/pgdh_prepare.py" --score-eval-dir "$out" --data-dir "$WORK/data" || FAILED+=("$name(score)")
    else
        FAILED+=("$name")
    fi
}

M=()
for i in 0 1 2 3 4 5 6 7 8 9; do M+=("$PARENT_MODELS/model_GATv2Net_ligsim90_fep_benchmark_${i}.model"); done
run_eval released "$PARENT_MODELS/model_GATv2Net_ligsim90_fep_benchmark.pickle" "${M[@]}"

RETRAIN=$PARENT_MODELS/retrain_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark
M=()
for i in 0 1 2 3 4 5 6 7 8 9; do M+=("${RETRAIN}_${i}.model"); done
run_eval retrain10 "${RETRAIN}.pickle" "${M[@]}"

M=()
for i in 0 1 2; do M+=("$WORK/output/trained_models/base_${i}.model"); done
run_eval base "$WORK/output/trained_models/base.pickle" "${M[@]}"

echo "---- pgdh_summary.json (head)"
head -c 1500 "$WORK/data/pgdh_summary.json"; echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
echo "done $(date -Is)"
