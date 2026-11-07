#!/bin/bash
#SBATCH --job-name=ap-featurise-edits
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Scheme C in-place ligand edits: extract the PDBbind v2020 refined and other-PL archives to /local (ligand
# mol2 and protein pdb members only), link them as refined-set/ and general-set/ under one root, then run
# featurise_edits.py over the PDBbind train/valid anchors of paired_meta.csv: per anchor 1 null-edit
# positive (kind 5), up to K_CONTACT contact-site edits (kind 6) and up to K_EXPOSED solvent-exposed edits
# (kind 7) -> $OUT_DIR/edits.pickle, edits_meta.csv, edits_summary.json.
# No identity check: featurise_edits.py featurises through the same functions featurise_poses.py verifies
# (its last identity result is $WORK/data/pdbbind_poses_identity.json, printed below when present).
# Env overrides: OUT_DIR (default $WORK/data), LIMIT (0 = all anchors), K_CONTACT=2, K_EXPOSED=2,
# EXPOSED_RADIUS=3.5 (A; exposed site = no protein ATOM or non-water HETATM heavy atom within this distance
# of any edited atom; the summary also reports the exposed counts at 4.0 and 4.5 A).
# Parent inputs are read-only; /local is removed on exit.
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

RAW=/scratch/u6xq/hussain.u6xq/aevplig/raw
REFINED_TGZ=$RAW/PDBbind_v2020_refined.tar.gz
OTHER_TGZ=$RAW/PDBbind_v2020_other_PL.tar.gz
OUT_DIR=${OUT_DIR:-$WORK/data}
LIMIT=${LIMIT:-0}
K_CONTACT=${K_CONTACT:-2}
K_EXPOSED=${K_EXPOSED:-2}
EXPOSED_RADIUS=${EXPOSED_RADIUS:-3.5}
NCPU=${SLURM_CPUS_PER_TASK:-72}
TMPD=/local/user/$(id -u)/aevplig-paired/edits-${SLURM_JOB_ID}
mkdir -p "$TMPD/x_refined" "$TMPD/x_other" "$TMPD/root" "$OUT_DIR"
trap 'rm -rf "$TMPD"' EXIT

echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
if [ -f "$WORK/data/pdbbind_poses_identity.json" ]; then
    echo "featuriser identity (arm 4): $(tr -d '\n' < "$WORK/data/pdbbind_poses_identity.json" | head -c 400)"
fi
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

echo "---- edit generation $(date -Is)"
python "$SRC/featurise_edits.py" --pdbbind-root "$TMPD/root" --meta "$WORK/data/paired_meta.csv" \
    --parent-data "$PARENT_WORK/data" --out-dir "$OUT_DIR" --workers "$NCPU" \
    --k-contact "$K_CONTACT" --k-exposed "$K_EXPOSED" --exposed-radius "$EXPOSED_RADIUS" --limit "$LIMIT"

echo "---- outputs"
ls -la "$OUT_DIR"/edits.pickle "$OUT_DIR"/edits_meta.csv "$OUT_DIR"/edits_summary.json
echo "---- edits_summary.json (head)"
head -c 8000 "$OUT_DIR/edits_summary.json"; echo
echo "done $(date -Is)"
