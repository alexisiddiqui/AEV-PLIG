#!/bin/bash
#SBATCH --job-name=ap-featurise-twins
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Soft-negative scheme A, shape twins (docs/plans/2026-11-05-aevplig-soft-negatives-AC.md, Task A).
# Extracts to /local (removed on exit), with the layout of jobs/featurise_zernike.sbatch:
#   PDBbind refined + other-PL (ligand mol2 / protein pdb only) -> root/refined-set, root/general-set
#                                                                  (anchors, and PDBbind twin ligands)
#   BindingNetv1.tar.gz (ligand sdf only)                       -> BindingNetv1/<pdb>/target_<t>/<c>/*.sdf
#   surflex.tar (folders at the tar root)                       -> surflex/<folder>/<mol2_file>
# Crystal_Templates_for_BindingNet1.tar.gz (the BindingNet receptors) is the fifth archive of the
# Zernike job; it is not extracted here because every pose is placed and featurised in the anchor's
# PDBbind pocket, so no BindingNet receptor is read.
# then featurise_twins.py: twin selection from $ZERNIKE (shell mode) + $META + $GROUPS_CSV, overlay
# placement, AEV featurisation with the copied clone functions (the same featurise_poses worker path
# whose identity check passed in arm 4; no identity check here) ->
#   $OUT_DIR/twins.pickle, twins_meta.csv, twins_summary.json
# Env overrides: OUT_DIR (default $WORK/data), LIMIT (first N anchors by pdb id; 0 = all), K (twins per
# anchor, 2), N_CANDIDATES (ranked candidates per anchor, 6), ZERNIKE (default $WORK/data/zernike.npz),
# GROUPS_CSV (default $WORK/data/pocket_groups.csv), META (default $WORK/data/paired_meta.csv),
# CLASH (acceptance min_dist >= min(CLASH, crystal min_dist), both kinds; 2.2 = arm 4, 3.0 = as clash-free
# as the crystal pose) and RELIEF_STEPS (relief budget per overlay, 300).
# Parent inputs are read-only.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

RAW=/scratch/u6xq/hussain.u6xq/aevplig/raw
REFINED_TGZ=$RAW/PDBbind_v2020_refined.tar.gz
OTHER_TGZ=$RAW/PDBbind_v2020_other_PL.tar.gz
BN_TGZ=$RAW/BindingNetv1.tar.gz
SURFLEX_TAR=$RAW/surflex.tar
OUT_DIR=${OUT_DIR:-$WORK/data}
LIMIT=${LIMIT:-0}
K=${K:-2}
N_CANDIDATES=${N_CANDIDATES:-6}
CLASH=${CLASH:-2.2}
RELIEF_STEPS=${RELIEF_STEPS:-300}
ZERNIKE=${ZERNIKE:-$WORK/data/zernike.npz}
GROUPS_CSV=${GROUPS_CSV:-$WORK/data/pocket_groups.csv}
META=${META:-$WORK/data/paired_meta.csv}
NCPU=${SLURM_CPUS_PER_TASK:-72}
TMPD=/local/user/$(id -u)/aevplig-paired/twins-${SLURM_JOB_ID}
mkdir -p "$TMPD/x_refined" "$TMPD/x_other" "$TMPD/x_bn" "$TMPD/surflex" "$TMPD/root" "$OUT_DIR"
trap 'rm -rf "$TMPD"' EXIT

for f in "$REFINED_TGZ" "$OTHER_TGZ" "$BN_TGZ" "$SURFLEX_TAR" "$ZERNIKE" "$GROUPS_CSV" "$META" \
         "$PARENT_WORK/data/pdbbind_processed.csv" "$PARENT_WORK/data/bindingnet_processed.csv" \
         "$PARENT_WORK/data/bindingdb_processed.csv"; do
    [ -f "$f" ] || { echo "ERROR: missing input $f"; exit 1; }
done

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
if command -v pigz >/dev/null 2>&1; then UNZ=(-I pigz); else echo "pigz not found, using gzip"; UNZ=(-z); fi
tar "${UNZ[@]}" -xf "$REFINED_TGZ" -C "$TMPD/x_refined" --no-same-owner --wildcards "*_ligand.mol2" "*_protein.pdb" &
PID_R=$!
tar "${UNZ[@]}" -xf "$OTHER_TGZ" -C "$TMPD/x_other" --no-same-owner --wildcards "*_ligand.mol2" "*_protein.pdb" &
PID_O=$!
tar "${UNZ[@]}" -xf "$BN_TGZ" -C "$TMPD/x_bn" --no-same-owner --wildcards "*.sdf" &
PID_B=$!
tar -xf "$SURFLEX_TAR" -C "$TMPD/surflex" --no-same-owner &
PID_S=$!
for pid in "$PID_R" "$PID_O" "$PID_B" "$PID_S"; do wait "$pid"; done

REFINED_DIR=$(find "$TMPD/x_refined" -maxdepth 2 -type d -name refined-set -print -quit)
OTHER_DIR=$(find "$TMPD/x_other" -maxdepth 2 -type d -name v2020-other-PL -print -quit)
BN_DIR=$(find "$TMPD/x_bn" -maxdepth 2 -type d -name BindingNetv1 -print -quit)
[ -n "$REFINED_DIR" ] || { echo "ERROR: no refined-set/ in $REFINED_TGZ"; exit 1; }
[ -n "$OTHER_DIR" ] || { echo "ERROR: no v2020-other-PL/ in $OTHER_TGZ"; exit 1; }
[ -n "$BN_DIR" ] || { echo "ERROR: no BindingNetv1/ in $BN_TGZ"; exit 1; }
ln -s "$REFINED_DIR" "$TMPD/root/refined-set"
ln -s "$OTHER_DIR" "$TMPD/root/general-set"
# surflex.tar holds <folder>/ at its root; accept a single wrapping surflex/ as well
if [ -d "$TMPD/surflex/surflex" ]; then BDB_ROOT=$TMPD/surflex/surflex; else BDB_ROOT=$TMPD/surflex; fi
echo "extracted $(date -Is): refined $(ls "$REFINED_DIR" | wc -l), other $(ls "$OTHER_DIR" | wc -l)," \
     "BindingNet templates $(ls "$BN_DIR" | wc -l), BindingDB folders $(ls "$BDB_ROOT" | wc -l)"
df -h "$TMPD" | tail -1

echo "---- featurise_twins $(date -Is): K=$K N_CANDIDATES=$N_CANDIDATES CLASH=$CLASH RELIEF_STEPS=$RELIEF_STEPS" \
     "LIMIT=$LIMIT -> $OUT_DIR"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python "$SRC/featurise_twins.py" --zernike "$ZERNIKE" --meta "$META" --groups "$GROUPS_CSV" \
    --pdbbind-root "$TMPD/root" --bindingnet-root "$BN_DIR" --bindingdb-root "$BDB_ROOT" \
    --parent-data "$PARENT_WORK/data" --out-dir "$OUT_DIR" --workers "$NCPU" --limit "$LIMIT" \
    --k "$K" --n-candidates "$N_CANDIDATES" --clash "$CLASH" --relief-steps "$RELIEF_STEPS"

echo "---- outputs"
ls -la "$OUT_DIR"/twins.pickle "$OUT_DIR"/twins_meta.csv "$OUT_DIR"/twins_summary.json
echo "---- twins_summary.json (head)"
head -c 12000 "$OUT_DIR/twins_summary.json"; echo
echo "done $(date -Is)"
