#!/bin/bash
#SBATCH --job-name=ap-build
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Full paired dataset (all series, 5% BindingNet/BindingDB series holdout).
# Reads the parent pickles/CSVs read-only; writes paired_meta.csv, paired_summary.json and
# processed/paired_{train,valid,test,series_valid}.pt under the output dir ($WORK/data).
# POSES=1 (arm 4) adds the PDBbind pose rows from jobs/featurise_poses.sbatch
# ($WORK/data/pdbbind_poses.{pickle,_meta.csv}) and writes to $WORK/data_poses instead,
# with the extra split processed/paired_poses_valid.pt.
#   POSES=1 sbatch jobs/build_dataset.sbatch
# Soft negatives (A twins, C edits) and the group-disjoint holdout:
#   AUGS="twins edits poses"   space-separated names; <name> -> $WORK/data/<name>.pickle +
#                              $WORK/data/<name>_meta.csv (poses -> pdbbind_poses.*, passed with the
#                              arm-4 --poses-* flags); adds processed/paired_aug_valid.pt
#   HOLDOUT_GROUPS=0.05        move that fraction of target groups (pocket groups with
#                              >= HOLDOUT_GROUP_MIN_ROWS=5 train/series_valid rows) to group_valid
#                              (processed/paired_group_valid.pt); implies the groups file
#   POCKET_GROUPS=1            add group_id / group_leak without a holdout
#   GROUPS_CSV                 default $WORK/data/pocket_groups.csv
#   OUT_DIR                    required with AUGS / HOLDOUT_GROUPS / POCKET_GROUPS (never
#                              $WORK/data, arm 1's build); default $WORK/data or $WORK/data_poses
# (The plan's GROUPS=1 is not usable: GROUPS is a bash special variable holding the user's group
# ids, so an assignment to it is ignored and the job would read the gid; HOLDOUT_GROUPS replaces it.)
#   AUGS="twins edits" HOLDOUT_GROUPS=0.05 OUT_DIR=$WORK/data_ac sbatch jobs/build_dataset.sbatch
# Zernike side file (arm P): ZERNIKE=1 runs build_zernike_side.py after the build -> $OUT_DIR/paired_zernike.npz
# from ZERNIKE_NPZ (default $WORK/data/zernike.npz) and, when AUGS has twins, $WORK/data/twins_meta.csv.
# ZERNIKE_ONLY=1 (with ZERNIKE=1) skips the dataset build and only writes the side file of the existing
# $OUT_DIR/paired_meta.csv (AUGS still decides whether the twins table is passed).
#   ZERNIKE=1 ZERNIKE_ONLY=1 AUGS="twins edits" HOLDOUT_GROUPS=0.05 OUT_DIR=$WORK/data_ac sbatch jobs/build_dataset.sbatch
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

POSES=${POSES:-0}
AUGS=${AUGS:-}
HOLDOUT_GROUPS=${HOLDOUT_GROUPS:-0}
HOLDOUT_GROUP_MIN_ROWS=${HOLDOUT_GROUP_MIN_ROWS:-5}
POCKET_GROUPS=${POCKET_GROUPS:-0}
GROUPS_CSV=${GROUPS_CSV:-$WORK/data/pocket_groups.csv}
ZERNIKE=${ZERNIKE:-0}
ZERNIKE_ONLY=${ZERNIKE_ONLY:-0}
ZERNIKE_NPZ=${ZERNIKE_NPZ:-$WORK/data/zernike.npz}
case "$ZERNIKE" in 0|1) ;; *) echo "ZERNIKE must be 0 or 1 (got $ZERNIKE)" >&2; exit 2 ;; esac
case "$ZERNIKE_ONLY" in 0|1) ;; *) echo "ZERNIKE_ONLY must be 0 or 1 (got $ZERNIKE_ONLY)" >&2; exit 2 ;; esac
if [ "$ZERNIKE_ONLY" = 1 ] && [ "$ZERNIKE" != 1 ]; then
  echo "ZERNIKE_ONLY=1 needs ZERNIKE=1" >&2; exit 2
fi
if [ "$ZERNIKE" = 1 ] && [ ! -s "$ZERNIKE_NPZ" ]; then
  echo "ZERNIKE=1 but $ZERNIKE_NPZ is missing (jobs/featurise_zernike.sbatch)" >&2; exit 2
fi
INDEX_TGZ=/scratch/u6xq/hussain.u6xq/aevplig/raw/PDBbind_v2020_plain_text_index.tar.gz
FLAGS=(--parent-data "$PARENT_WORK/data" --pdbbind-index-tgz "$INDEX_TGZ")

case "$POSES" in
  0) ;;
  1) case " $AUGS " in *" poses "*) ;; *) AUGS="$AUGS poses" ;; esac ;;
  *) echo "POSES must be 0 or 1 (got $POSES)" >&2; exit 2 ;;
esac
case "$POCKET_GROUPS" in 0|1) ;; *) echo "POCKET_GROUPS must be 0 or 1 (got $POCKET_GROUPS)" >&2; exit 2 ;; esac
use_groups=$POCKET_GROUPS
if awk -v f="$HOLDOUT_GROUPS" 'BEGIN { exit !(f + 0 > 0) }'; then use_groups=1; fi
case "$AUGS" in *[!A-Za-z0-9_\ ]*) echo "AUGS='$AUGS': names must be [A-Za-z0-9_], space-separated" >&2; exit 2 ;; esac
aug_names=$(echo $AUGS)  # trimmed, single spaces (no glob characters: checked above)

if [ -z "${OUT_DIR:-}" ]; then
  if [ -z "$aug_names" ] && [ "$use_groups" = 0 ]; then
    OUT_DIR=$WORK/data
  elif [ "$POSES" = 1 ] && [ "$aug_names" = poses ] && [ "$use_groups" = 0 ]; then
    OUT_DIR=$WORK/data_poses
  else
    echo "set OUT_DIR for AUGS / HOLDOUT_GROUPS / POCKET_GROUPS builds (e.g. OUT_DIR=\$WORK/data_ac)" >&2; exit 2
  fi
fi
if [ "${OUT_DIR%/}" = "$WORK/data" ] && { [ -n "$aug_names" ] || [ "$use_groups" = 1 ]; }; then
  echo "OUT_DIR=$OUT_DIR is arm 1's build: choose another dir for augmented / grouped builds" >&2; exit 2
fi

for name in $aug_names; do
  base=$name
  [ "$name" = poses ] && base=pdbbind_poses
  pk=$WORK/data/$base.pickle
  mt=$WORK/data/${base}_meta.csv
  for f in "$pk" "$mt"; do
    if [ ! -s "$f" ]; then
      echo "AUGS has $name but $f is missing (run its featurise job first)" >&2; exit 2
    fi
  done
  if [ "$name" = poses ]; then
    FLAGS+=(--poses-pickle "$pk" --poses-meta "$mt")
  else
    FLAGS+=(--aug "$pk:$mt")
  fi
done
if [ "$use_groups" = 1 ]; then
  if [ ! -s "$GROUPS_CSV" ]; then
    echo "groups requested but $GROUPS_CSV is missing (pocket_groups.py)" >&2; exit 2
  fi
  FLAGS+=(--groups "$GROUPS_CSV" --holdout-groups "$HOLDOUT_GROUPS"
          --holdout-group-min-rows "$HOLDOUT_GROUP_MIN_ROWS")
fi
FLAGS+=(--out-dir "$OUT_DIR")
ZFLAGS=(--meta "$OUT_DIR/paired_meta.csv" --zernike "$ZERNIKE_NPZ" --out "$OUT_DIR/paired_zernike.npz")
case " $aug_names " in *" twins "*) ZFLAGS+=(--twins-meta "$WORK/data/twins_meta.csv") ;; esac

echo "host $(hostname) job ${SLURM_JOB_ID} augs '${aug_names}' groups $use_groups holdout_groups $HOLDOUT_GROUPS out $OUT_DIR zernike $ZERNIKE start $(date -Is)"
if [ "$ZERNIKE_ONLY" = 1 ]; then
  if [ ! -s "$OUT_DIR/paired_meta.csv" ]; then
    echo "ZERNIKE_ONLY=1 but $OUT_DIR/paired_meta.csv is missing" >&2; exit 2
  fi
  echo "ZERNIKE_ONLY=1: dataset build skipped"
else
  echo "python $SRC/build_paired_dataset.py ${FLAGS[*]}"
  python "$SRC/build_paired_dataset.py" "${FLAGS[@]}"
  cat "$OUT_DIR/paired_summary.json"
  ls -la "$OUT_DIR/processed"
fi
if [ "$ZERNIKE" = 1 ]; then
  echo "python $SRC/build_zernike_side.py ${ZFLAGS[*]}"
  python "$SRC/build_zernike_side.py" "${ZFLAGS[@]}"
  ls -la "$OUT_DIR/paired_zernike.npz"
fi
echo "done $(date -Is)"
