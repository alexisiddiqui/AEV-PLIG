#!/bin/bash
#SBATCH --job-name=ap-featurise-poses
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Arm 4 pose decoys: extract the PDBbind v2020 refined and other-PL archives to /local (ligand mol2
# and protein pdb members only), link them as refined-set/ and general-set/ under one root, then
#   1. featurise_poses.py --check-identity $IDENTITY_N: the copied featuriser must reproduce the
#      parent's pdbbind.pickle bit-for-bit on crystal poses ("IDENTITY OK n=N"), else abort;
#   2. the full run: K_POS near-native positives + K_NEG decoys per PDBbind complex of
#      paired_meta.csv -> $OUT_DIR/pdbbind_poses.pickle, pdbbind_poses_meta.csv,
#      pdbbind_poses_summary.json.
# Env overrides: OUT_DIR (default $WORK/data), LIMIT (0 = all complexes), K_POS=2, K_NEG=2,
# IDENTITY_N=200. Parent inputs are read-only; /local is removed on exit.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

RAW=/scratch/u6xq/hussain.u6xq/aevplig/raw
REFINED_TGZ=$RAW/PDBbind_v2020_refined.tar.gz
OTHER_TGZ=$RAW/PDBbind_v2020_other_PL.tar.gz
OUT_DIR=${OUT_DIR:-$WORK/data}
LIMIT=${LIMIT:-0}
K_POS=${K_POS:-2}
K_NEG=${K_NEG:-2}
IDENTITY_N=${IDENTITY_N:-200}
NCPU=${SLURM_CPUS_PER_TASK:-72}
TMPD=/local/user/$(id -u)/aevplig-paired/poses-${SLURM_JOB_ID}
mkdir -p "$TMPD/x_refined" "$TMPD/x_other" "$TMPD/root" "$OUT_DIR"
trap 'rm -rf "$TMPD"' EXIT

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
if command -v pigz >/dev/null 2>&1; then UNZ=(-I pigz); else echo "pigz not found, using gzip"; UNZ=(-z); fi
MEMBERS=(--wildcards "*_ligand.mol2" "*_protein.pdb")
tar "${UNZ[@]}" -xf "$REFINED_TGZ" -C "$TMPD/x_refined" --no-same-owner "${MEMBERS[@]}" &
PID_R=$!
tar "${UNZ[@]}" -xf "$OTHER_TGZ" -C "$TMPD/x_other" --no-same-owner "${MEMBERS[@]}" &
PID_O=$!
wait "$PID_R"
wait "$PID_O"
REFINED_DIR=$(find "$TMPD/x_refined" -maxdepth 2 -type d -name refined-set -print -quit)
OTHER_DIR=$(find "$TMPD/x_other" -maxdepth 2 -type d -name v2020-other-PL -print -quit)
[ -n "$REFINED_DIR" ] || { echo "ERROR: no refined-set/ in $REFINED_TGZ"; exit 1; }
[ -n "$OTHER_DIR" ] || { echo "ERROR: no v2020-other-PL/ in $OTHER_TGZ"; exit 1; }
ln -s "$REFINED_DIR" "$TMPD/root/refined-set"
ln -s "$OTHER_DIR" "$TMPD/root/general-set"
echo "extracted $(date -Is): refined $(ls "$REFINED_DIR" | wc -l) dirs, other $(ls "$OTHER_DIR" | wc -l) dirs"
df -h "$TMPD" | tail -1

COMMON=(--pdbbind-root "$TMPD/root" --meta "$WORK/data/paired_meta.csv"
        --pdbbind-csv "$PARENT_WORK/data/pdbbind_processed.csv" --out-dir "$OUT_DIR" --workers "$NCPU")

echo "---- identity check ($IDENTITY_N crystal complexes vs the parent's pdbbind.pickle)"
if ! python "$SRC/featurise_poses.py" "${COMMON[@]}" --check-identity "$IDENTITY_N" \
        --parent-pickle "$PARENT_WORK/data/pdbbind.pickle"; then
    echo "ABORT: identity check failed; no poses generated"
    exit 3
fi

echo "---- pose generation $(date -Is)"
python "$SRC/featurise_poses.py" "${COMMON[@]}" --k-pos "$K_POS" --k-neg "$K_NEG" --limit "$LIMIT"

echo "---- outputs"
ls -la "$OUT_DIR"/pdbbind_poses*
echo "---- pdbbind_poses_summary.json (head)"
head -c 8000 "$OUT_DIR/pdbbind_poses_summary.json"; echo
echo "done $(date -Is)"
