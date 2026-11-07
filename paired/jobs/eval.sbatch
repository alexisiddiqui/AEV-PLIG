#!/bin/bash
#SBATCH --job-name=ap-eval
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%j.out
#
# Evaluate one training arm: the ensemble of $WORK/output/trained_models/${TAG}_<i>.model over
# SEEDS -> $WORK/output/eval/$NAME/, and (PER_SEED=1, default) each seed alone ->
# $WORK/output/eval/$NAME/seed<i>/. LIGAND_ONLY=1 passes --ligand-only (eval_paired.py also
# switches it on from the models' .json sidecars).
# DATA_DIR holds paired_meta.csv and processed/paired_{series_valid,test}.pt; the FEP inputs
# come from FEP_DIR (default $WORK/data, where jobs/fep_prepare.sbatch writes them). When
# DATA_DIR has processed/paired_poses_valid.pt (a POSES=1 build) the pose-sensitivity
# section runs too (--pose-pt -> pose_summary.csv); with processed/paired_aug_valid.pt (an AUGS
# build) the per-kind discrimination section (--aug-pt -> aug_summary.csv), and with
# processed/paired_group_valid.pt (a HOLDOUT_GROUPS build) the group-disjoint holdout
# (--group-valid-pt -> group_valid_summary.csv). eval_paired.py exits when these files are not under
# the models' training DATA_DIR (sidecar args.data_dir); ALLOW_DATA_DIR_MISMATCH=1 makes it a warning.
# --pairwise is always passed: within-series pairwise ranking -> pairwise_summary.csv and
# pairwise_<dataset>_per_series.csv (Siamese-head rows for arm-S models, pair-head rows for arm-P models,
# pointwise rows for every model). Moments heads (arm P): ZERNIKE_SIDE (default $DATA_DIR/paired_zernike.npz
# when that file exists) is passed as --zernike-side -> moments_summary.csv (written only for models with moments
# heads); an explicit ZERNIKE_SIDE that is missing is an error. Moments inputs (arm Z): ZERNIKE_EVAL (a comma list;
# default: whichever of $FEP_DIR/zernike_fep.npz and $WORK/data/zernike_pgdh.npz exist, written by
# jobs/featurise_zernike_eval.sbatch) is passed as --zernike-eval: the FEP / 15-PGDH ligands' descriptors (by
# unique_id), standardised with ZERNIKE_SIDE's train mean / sd; eval_paired.py reads them only for models with
# moments inputs (and then requires ZERNIKE_SIDE, and --zernike-eval for the FEP section); an explicit ZERNIKE_EVAL
# with a missing file is an error.
#   TAG=Zbx DATA_DIR=$WORK/data_ac FEP_DIR=$WORK/data sbatch jobs/eval.sbatch
#   TAG=PM DATA_DIR=$WORK/data_ac FEP_DIR=$WORK/data sbatch jobs/eval.sbatch
#   TAG=S DATA_DIR=$WORK/data_ac FEP_DIR=$WORK/data sbatch jobs/eval.sbatch
#   TAG=base-g NAME=base-g-pw DATA_DIR=$WORK/data_ac FEP_DIR=$WORK/data sbatch jobs/eval.sbatch
#   TAG=dh03 sbatch jobs/eval.sbatch
#   TAG=base-lig LIGAND_ONLY=1 SEEDS=0,1,2 NAME=base-lig sbatch jobs/eval.sbatch
#   TAG=pdanda DATA_DIR=$WORK/data_poses sbatch jobs/eval.sbatch
#   TAG=A DATA_DIR=$WORK/data_ac FEP_DIR=$WORK/data sbatch jobs/eval.sbatch
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

TAG=${TAG:?set TAG}
SEEDS=${SEEDS:-0,1,2}
NAME=${NAME:-$TAG}
LIGAND_ONLY=${LIGAND_ONLY:-0}
PER_SEED=${PER_SEED:-1}
DATA_DIR=${DATA_DIR:-$WORK/data}
FEP_DIR=${FEP_DIR:-$WORK/data}
MODEL_DIR=$WORK/output/trained_models
OUT=$WORK/output/eval/$NAME
POSE_PT=$DATA_DIR/processed/paired_poses_valid.pt
AUG_PT=$DATA_DIR/processed/paired_aug_valid.pt
GROUP_PT=$DATA_DIR/processed/paired_group_valid.pt
if [ -n "${ZERNIKE_SIDE:-}" ]; then
    if [ ! -s "$ZERNIKE_SIDE" ]; then
        echo "ZERNIKE_SIDE=$ZERNIKE_SIDE is missing" >&2; exit 2
    fi
elif [ -s "$DATA_DIR/paired_zernike.npz" ]; then
    ZERNIKE_SIDE=$DATA_DIR/paired_zernike.npz
else
    ZERNIKE_SIDE=
fi
if [ -n "${ZERNIKE_EVAL:-}" ]; then
    IFS=, read -r -a ZE_LIST <<< "$ZERNIKE_EVAL"
    for f in "${ZE_LIST[@]}"; do
        if [ ! -s "$f" ]; then
            echo "ZERNIKE_EVAL file $f is missing" >&2; exit 2
        fi
    done
else
    ZERNIKE_EVAL=
    for f in "$FEP_DIR/zernike_fep.npz" "$WORK/data/zernike_pgdh.npz"; do
        if [ -s "$f" ]; then
            ZERNIKE_EVAL=${ZERNIKE_EVAL:+$ZERNIKE_EVAL,}$f
        fi
    done
fi

COMMON=(--scaler "$MODEL_DIR/${TAG}.pickle" --meta "$DATA_DIR/paired_meta.csv"
        --fep-csv "$FEP_DIR/fep_benchmark.csv" --fep-pt "$FEP_DIR/processed/fep_benchmark.pt"
        --series-valid-pt "$DATA_DIR/processed/paired_series_valid.pt"
        --test-pt "$DATA_DIR/processed/paired_test.pt" --pairwise)
if [ "$LIGAND_ONLY" = 1 ]; then
    COMMON+=(--ligand-only)
fi
if [ "${ALLOW_DATA_DIR_MISMATCH:-0}" = 1 ]; then  # e.g. a reference model trained on another build
    COMMON+=(--allow-data-dir-mismatch)
fi
if [ -f "$POSE_PT" ]; then
    COMMON+=(--pose-pt "$POSE_PT")
    echo "pose sensitivity: $POSE_PT"
else
    echo "pose sensitivity: skipped ($POSE_PT not found)"
fi
if [ -f "$AUG_PT" ]; then
    COMMON+=(--aug-pt "$AUG_PT")
    echo "augmented-row discrimination: $AUG_PT"
else
    echo "augmented-row discrimination: skipped ($AUG_PT not found)"
fi
if [ -f "$GROUP_PT" ]; then
    COMMON+=(--group-valid-pt "$GROUP_PT")
    echo "group-disjoint holdout: $GROUP_PT"
else
    echo "group-disjoint holdout: skipped ($GROUP_PT not found)"
fi
if [ -n "$ZERNIKE_SIDE" ]; then
    COMMON+=(--zernike-side "$ZERNIKE_SIDE")
    echo "moments heads: $ZERNIKE_SIDE"
else
    echo "moments heads: skipped (no ZERNIKE_SIDE, no $DATA_DIR/paired_zernike.npz)"
fi
if [ -n "$ZERNIKE_EVAL" ]; then
    COMMON+=(--zernike-eval "$ZERNIKE_EVAL")
    echo "moments inputs (FEP / 15-PGDH descriptors): $ZERNIKE_EVAL"
else
    echo "moments inputs: no ZERNIKE_EVAL ($FEP_DIR/zernike_fep.npz not found; jobs/featurise_zernike_eval.sbatch)"
fi

MODELS=()
IFS=, read -r -a SEED_LIST <<< "$SEEDS"
for i in "${SEED_LIST[@]}"; do
    MODELS+=("$MODEL_DIR/${TAG}_${i}.model")
done

echo "host $(hostname) job ${SLURM_JOB_ID} tag $TAG seeds $SEEDS name $NAME data $DATA_DIR start $(date -Is)"
python "$SRC/eval_paired.py" --models "${MODELS[@]}" "${COMMON[@]}" --out-dir "$OUT"
if [ "$PER_SEED" = 1 ]; then
    for i in "${SEED_LIST[@]}"; do
        python "$SRC/eval_paired.py" --models "$MODEL_DIR/${TAG}_${i}.model" "${COMMON[@]}" \
            --out-dir "$OUT/seed$i"
    done
fi
echo "---- $OUT/SUMMARY.md"
cat "$OUT/SUMMARY.md"
echo "done $(date -Is)"
