#!/bin/bash
#SBATCH --job-name=ap-featurise-zernike
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# 3D Zernike descriptors (pocket + ligand, 7 property channels) for every complex of paired_meta.csv,
# then target groups. Extracts to /local (removed on exit), with the parent's layout:
#   PDBbind refined + other-PL (ligand mol2 / protein pdb only) -> root/refined-set, root/general-set
#   BindingNetv1.tar.gz (ligand sdf only)                       -> BindingNetv1/<pdb>/target_<t>/<c>/*.sdf
#   Crystal_Templates_for_BindingNet1.tar.gz (rec_h_opt.pdb)    -> linked as BindingNetv1/<pdb>/rec_h_opt.pdb
#   surflex.tar (folders at the tar root)                       -> surflex/<folder>/{pdb_file,mol2_file}
# then
#   1. featurise_zernike.py -> $OUT_DIR/$OUT_NAME.npz + ${OUT_NAME}_summary.json
#   2. pocket_groups.py     -> $OUT_DIR/pocket_groups.csv + pocket_groups_summary.json
#                              (${OUT_NAME}_pocket_groups.* when OUT_NAME is not "zernike", so an A/B run never
#                              overwrites the main groups file)
# Env overrides: OUT_DIR (default $WORK/data), OUT_NAME (default zernike), POCKET_MODE (shell | sphere;
# default shell), POCKET_RADIUS (sphere radius = descriptor radius; featurise_zernike's default 10 when
# empty; sphere mode only), VARIANT_NOTE (free text stored in the summary), MERGE_COSINE (pocket_groups
# descriptor merge threshold; empty = off), LIMIT (first N rows per source; 0 = all), META (default
# $WORK/data/paired_meta.csv). Parent inputs are read-only.
# A/B example: POCKET_MODE=sphere POCKET_RADIUS=10 OUT_NAME=zernike_sphere10 VARIANT_NOTE="sphere 10 A" \
#              sbatch jobs/featurise_zernike.sbatch
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

RAW=/scratch/u6xq/hussain.u6xq/aevplig/raw
REFINED_TGZ=$RAW/PDBbind_v2020_refined.tar.gz
OTHER_TGZ=$RAW/PDBbind_v2020_other_PL.tar.gz
BN_TGZ=$RAW/BindingNetv1.tar.gz
TPL_TGZ=$RAW/Crystal_Templates_for_BindingNet1.tar.gz
SURFLEX_TAR=$RAW/surflex.tar
INDEX_TGZ=$RAW/PDBbind_v2020_plain_text_index.tar.gz
OUT_DIR=${OUT_DIR:-$WORK/data}
OUT_NAME=${OUT_NAME:-zernike}
POCKET_MODE=${POCKET_MODE:-shell}
POCKET_RADIUS=${POCKET_RADIUS:-}
VARIANT_NOTE=${VARIANT_NOTE:-}
MERGE_COSINE=${MERGE_COSINE:-}
LIMIT=${LIMIT:-0}
META=${META:-$WORK/data/paired_meta.csv}
case "$OUT_NAME" in ""|*/*) echo "ERROR: OUT_NAME must be a plain file stem, got '$OUT_NAME'"; exit 1 ;; esac
case "$POCKET_MODE" in shell|sphere) ;; *) echo "ERROR: POCKET_MODE must be shell or sphere"; exit 1 ;; esac
if [ "$POCKET_MODE" = shell ] && [ -n "$POCKET_RADIUS" ]; then
    echo "ERROR: POCKET_RADIUS applies to POCKET_MODE=sphere only"; exit 1
fi
if [ "$OUT_NAME" = zernike ]; then GROUPS_NAME=pocket_groups; else GROUPS_NAME=${OUT_NAME}_pocket_groups; fi
NCPU=${SLURM_CPUS_PER_TASK:-72}
TMPD=/local/user/$(id -u)/aevplig-paired/zernike-${SLURM_JOB_ID}
mkdir -p "$TMPD/x_refined" "$TMPD/x_other" "$TMPD/x_bn" "$TMPD/x_tpl" "$TMPD/surflex" "$TMPD/root" "$OUT_DIR"
trap 'rm -rf "$TMPD"' EXIT

for f in "$REFINED_TGZ" "$OTHER_TGZ" "$BN_TGZ" "$TPL_TGZ" "$SURFLEX_TAR" "$INDEX_TGZ" "$META"; do
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
tar "${UNZ[@]}" -xf "$TPL_TGZ" -C "$TMPD/x_tpl" --no-same-owner --wildcards "*rec_h_opt.pdb" &
PID_T=$!
tar -xf "$SURFLEX_TAR" -C "$TMPD/surflex" --no-same-owner &
PID_S=$!
for pid in "$PID_R" "$PID_O" "$PID_B" "$PID_T" "$PID_S"; do wait "$pid"; done

REFINED_DIR=$(find "$TMPD/x_refined" -maxdepth 2 -type d -name refined-set -print -quit)
OTHER_DIR=$(find "$TMPD/x_other" -maxdepth 2 -type d -name v2020-other-PL -print -quit)
BN_DIR=$(find "$TMPD/x_bn" -maxdepth 2 -type d -name BindingNetv1 -print -quit)
TPL_DIR=$(find "$TMPD/x_tpl" -maxdepth 2 -type d -name Crystal_Templates_for_BindingNet1 -print -quit)
[ -n "$REFINED_DIR" ] || { echo "ERROR: no refined-set/ in $REFINED_TGZ"; exit 1; }
[ -n "$OTHER_DIR" ] || { echo "ERROR: no v2020-other-PL/ in $OTHER_TGZ"; exit 1; }
[ -n "$BN_DIR" ] || { echo "ERROR: no BindingNetv1/ in $BN_TGZ"; exit 1; }
[ -n "$TPL_DIR" ] || { echo "ERROR: no Crystal_Templates_for_BindingNet1/ in $TPL_TGZ"; exit 1; }
ln -s "$REFINED_DIR" "$TMPD/root/refined-set"
ln -s "$OTHER_DIR" "$TMPD/root/general-set"
# BindingNet receptors: the clone reads <BindingNet root>/<pdb>/rec_h_opt.pdb
N_LINK=0
for rec in "$TPL_DIR"/*/rec_h_opt.pdb; do
    [ -f "$rec" ] || continue
    pdb=$(basename "$(dirname "$rec")")
    if [ -d "$BN_DIR/$pdb" ] && [ ! -e "$BN_DIR/$pdb/rec_h_opt.pdb" ]; then
        ln -s "$rec" "$BN_DIR/$pdb/rec_h_opt.pdb"
        N_LINK=$((N_LINK + 1))
    fi
done
# surflex.tar holds <folder>/ at its root; accept a single wrapping surflex/ as well
if [ -d "$TMPD/surflex/surflex" ]; then BDB_ROOT=$TMPD/surflex/surflex; else BDB_ROOT=$TMPD/surflex; fi
echo "extracted $(date -Is): refined $(ls "$REFINED_DIR" | wc -l), other $(ls "$OTHER_DIR" | wc -l)," \
     "BindingNet templates $(ls "$BN_DIR" | wc -l) ($N_LINK receptors linked), BindingDB folders $(ls "$BDB_ROOT" | wc -l)"
df -h "$TMPD" | tail -1

# geometry: shell = 8 A shell scaled by R = 12 A (the original run); sphere = POCKET_RADIUS sphere, R = POCKET_RADIUS
GEOM=(--pocket-mode "$POCKET_MODE" --order 12 --ligand-radius 12.0)
if [ "$POCKET_MODE" = shell ]; then GEOM+=(--radius 12.0 --cutoff 8.0); fi
if [ -n "$POCKET_RADIUS" ]; then GEOM+=(--pocket-radius "$POCKET_RADIUS"); fi
echo "---- featurise_zernike $(date -Is): ${GEOM[*]} note='$VARIANT_NOTE' -> $OUT_DIR/$OUT_NAME.npz"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python "$SRC/featurise_zernike.py" --meta "$META" --parent-data "$PARENT_WORK/data" \
    --roots "pdbbind=$TMPD/root,bindingnet=$BN_DIR,bindingdb=$BDB_ROOT" \
    --out "$OUT_DIR/$OUT_NAME.npz" --workers "$NCPU" --limit "$LIMIT" "${GEOM[@]}" \
    --variant-note "$VARIANT_NOTE"

GROUP_ARGS=(--meta "$META" --pdbbind-index-tgz "$INDEX_TGZ" --zernike "$OUT_DIR/$OUT_NAME.npz"
            --out "$OUT_DIR/$GROUPS_NAME.csv")
if [ -n "$MERGE_COSINE" ]; then GROUP_ARGS+=(--merge-cosine "$MERGE_COSINE"); fi
echo "---- pocket_groups $(date -Is)"
python "$SRC/pocket_groups.py" "${GROUP_ARGS[@]}"

echo "---- outputs"
ls -la "$OUT_DIR/$OUT_NAME.npz" "$OUT_DIR/${OUT_NAME}_summary.json" "$OUT_DIR/$GROUPS_NAME.csv" \
    "$OUT_DIR/${GROUPS_NAME}_summary.json"
echo "---- ${OUT_NAME}_summary.json"
cat "$OUT_DIR/${OUT_NAME}_summary.json"; echo
echo "---- ${GROUPS_NAME}_summary.json"
cat "$OUT_DIR/${GROUPS_NAME}_summary.json"; echo
echo "done $(date -Is)"
