#!/bin/bash
#SBATCH --job-name=ap-train
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --time=05:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-paired/logs/%x-%A_%a.out
#
# One seed (SLURM_ARRAY_TASK_ID = --seed_index) of one paired-training arm.
# Configure with environment variables at submission, e.g.
#   TAG=base PAIR_LOSS=none sbatch --job-name=ap-train-base jobs/train.sbatch
#   TAG=dh03 PAIR_LOSS=delta_huber PAIR_WEIGHT=0.3 sbatch jobs/train.sbatch
#   TAG=dh03-lig PAIR_LOSS=delta_huber PAIR_WEIGHT=0.3 LIGAND_ONLY=1 sbatch jobs/train.sbatch
#   TAG=base-lig LIGAND_ONLY=1 sbatch --array=0-9 jobs/train.sbatch
# Augmented arms: TRAIN_KINDS selects train row kinds (0 original, 1 near-native pose, 2 decoy,
# 3 A-positive, 4 shape twin, 5 C-positive, 6 contact edit, 7 exposed edit). Kinds 2/4/6 enter a
# one-sided hinge below the parent's pK with HINGE_MARGINS (pK, default 2:0.75,4:1.0,6:1.0) and
# HINGE_WEIGHT (default 1.0); DECOY_WEIGHT / DECOY_MARGIN (arm 4) set kind 2's weight / margin and
# are passed only when set. train_paired.py refuses a selected hinge kind with weight 0 or no margin.
#   TAG=pda TRAIN_KINDS=0,1 DATA_DIR=$WORK/data_poses sbatch jobs/train.sbatch
#   TAG=nda TRAIN_KINDS=0,2 DECOY_WEIGHT=1.0 DATA_DIR=$WORK/data_poses sbatch jobs/train.sbatch
#   TAG=pdanda TRAIN_KINDS=0,1,2 DECOY_WEIGHT=1.0 DECOY_MARGIN=0.75 DATA_DIR=$WORK/data_poses sbatch jobs/train.sbatch
#   TAG=A TRAIN_KINDS=0,3,4 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=C TRAIN_KINDS=0,5,6,7 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=A-lig TRAIN_KINDS=0,3,4 LIGAND_ONLY=1 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
# Siamese difference head (arm S): SIAMESE=1 adds --siamese with SIAMESE_WEIGHT (default 1.0),
# SIAMESE_HIDDEN (default 512) and SIAMESE_TIE (pK, default 0.5); these settings without SIAMESE=1
# are an error.
#   TAG=S TRAIN_KINDS=0 SIAMESE=1 SIAMESE_WEIGHT=1.0 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=S-lig TRAIN_KINDS=0 SIAMESE=1 SIAMESE_WEIGHT=1.0 LIGAND_ONLY=1 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
# Pair head + Zernike moments heads (arm P): PAIR_HEAD=1 adds --pair-head with PAIR_HEAD_WEIGHT (default 0.3; the
# plan's PAIR_WEIGHT is taken: it is --pair-loss's weight), PAIR_HIDDEN (default 512) and PAIR_NEG_KINDS (comma
# list of hinge kinds, default none; a hinge kind listed there may run with HINGE_WEIGHT=0, listwise only);
# these settings without PAIR_HEAD=1 are an error. MOMENTS=pocket|ligand|both (default none) adds --moments with
# MOMENTS_WEIGHT (default 0.3) and needs ZERNIKE_SIDE (paired_zernike.npz from build_zernike_side.py, e.g.
# $WORK/data_ac/paired_zernike.npz); MOMENTS_KINDS (row kinds in the moments loss, default 0) and
# ALLOW_PARTIAL_SIDE=1 (mask rows missing from the side file instead of exiting) are passed only when set;
# MOMENTS_WEIGHT / ZERNIKE_SIDE / MOMENTS_KINDS / ALLOW_PARTIAL_SIDE with MOMENTS=none are an error.
# SAMPLER_PARENT_AWARE=1 passes --sampler-parent-aware (series chunks built from an original row followed by its
# augmented rows, so a parent and up to CHUNK-1 aug rows share a chunk). CHUNK (passed as --chunk only when set;
# train_paired's default 8) sets the chunk size: with families of ~8-9 rows (data_ac kinds 0,3,4,5,6,7) a chunk of 8
# holds one crystal, so a kind-0 pair-head anchor rarely meets two same-series crystals in its batch; CHUNK=32 packs
# ~3-4 families of one series per chunk.
#   TAG=M TRAIN_KINDS=0 MOMENTS=both ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=P TRAIN_KINDS=0 PAIR_HEAD=1 PAIR_HEAD_WEIGHT=0.3 DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=PMpos TRAIN_KINDS=0,3,5,7 PAIR_HEAD=1 MOMENTS=both SAMPLER_PARENT_AWARE=1 \
#     ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=PMneg TRAIN_KINDS=0,3,4,5,6,7 HINGE_WEIGHT=0 PAIR_HEAD=1 PAIR_NEG_KINDS=4,6 MOMENTS=both SAMPLER_PARENT_AWARE=1 \
#     ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
# Zernike moments as INPUT (arm Z): INPUT_MOMENTS=pocket|ligand|both|bothx (default none) adds --input-moments (the
# standardised descriptors + a missing flag fed to the pointwise head; heads still read z only) with
# INPUT_MOMENTS_PROJ (passed only when set; train_paired's default 256, 0 = raw concatenation); MODEL=moments_only
# (default gatv2) passes --model moments_only: an MLP on the moments input alone (needs INPUT_MOMENTS; INPUT_MOMENTS_PROJ,
# LIGAND_ONLY, SIAMESE, PAIR_HEAD and MOMENTS are errors with it). INPUT_MOMENTS other than none (or
# MODEL=moments_only) needs ZERNIKE_SIDE; ALLOW_PARTIAL_SIDE applies as with MOMENTS.
#   TAG=Zbx TRAIN_KINDS=0 INPUT_MOMENTS=bothx ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=Zlin TRAIN_KINDS=0 MODEL=moments_only INPUT_MOMENTS=bothx ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz \
#     DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
#   TAG=Zb-lig TRAIN_KINDS=0 INPUT_MOMENTS=both LIGAND_ONLY=1 ZERNIKE_SIDE=$WORK/data_ac/paired_zernike.npz \
#     DATA_DIR=$WORK/data_ac sbatch jobs/train.sbatch
set -eo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
set -u

TAG=${TAG:?set TAG}
if [ "${PAIR_LOSS:-none}" = none ] && [ -n "${PAIR_WEIGHT:-}${TIE:-}${HUBER_DELTA:-}${WEIGHTING:-}" ]; then
  echo "pair settings given with PAIR_LOSS=none (did you forget PAIR_LOSS=delta_huber? the pair HEAD's weight" \
       "is PAIR_HEAD_WEIGHT)" >&2; exit 2
fi
SIAMESE=${SIAMESE:-0}
if [ "$SIAMESE" != 1 ] && [ -n "${SIAMESE_WEIGHT:-}${SIAMESE_HIDDEN:-}${SIAMESE_TIE:-}" ]; then
  echo "Siamese settings given without SIAMESE=1" >&2; exit 2
fi
PAIR_HEAD=${PAIR_HEAD:-0}
case "$PAIR_HEAD" in 0|1) ;; *) echo "PAIR_HEAD must be 0 or 1 (got $PAIR_HEAD)" >&2; exit 2 ;; esac
if [ "$PAIR_HEAD" != 1 ] && [ -n "${PAIR_HEAD_WEIGHT:-}${PAIR_HIDDEN:-}${PAIR_NEG_KINDS:-}" ]; then
  echo "pair-head settings given without PAIR_HEAD=1" >&2; exit 2
fi
MOMENTS=${MOMENTS:-none}
case "$MOMENTS" in none|pocket|ligand|both) ;; *) echo "MOMENTS must be none, pocket, ligand or both (got $MOMENTS)" >&2; exit 2 ;; esac
INPUT_MOMENTS=${INPUT_MOMENTS:-none}
case "$INPUT_MOMENTS" in none|pocket|ligand|both|bothx) ;;
  *) echo "INPUT_MOMENTS must be none, pocket, ligand, both or bothx (got $INPUT_MOMENTS)" >&2; exit 2 ;; esac
MODEL=${MODEL:-gatv2}
case "$MODEL" in gatv2|moments_only) ;; *) echo "MODEL must be gatv2 or moments_only (got $MODEL)" >&2; exit 2 ;; esac
INPUT_MOMENTS_PROJ=${INPUT_MOMENTS_PROJ:-}
case "$INPUT_MOMENTS_PROJ" in ""|*[!0-9]*) [ -z "$INPUT_MOMENTS_PROJ" ] || {
  echo "INPUT_MOMENTS_PROJ must be a non-negative integer (got $INPUT_MOMENTS_PROJ)" >&2; exit 2; } ;; esac
if [ "$INPUT_MOMENTS" = none ] && [ -n "$INPUT_MOMENTS_PROJ" ]; then
  echo "INPUT_MOMENTS_PROJ given with INPUT_MOMENTS=none" >&2; exit 2
fi
if [ "$MODEL" = moments_only ]; then
  if [ "$INPUT_MOMENTS" = none ]; then
    echo "MODEL=moments_only needs INPUT_MOMENTS (pocket, ligand, both or bothx)" >&2; exit 2
  fi
  if [ -n "$INPUT_MOMENTS_PROJ" ] || [ "${LIGAND_ONLY:-0}" = 1 ] || [ "${SIAMESE:-0}" = 1 ] || [ "$PAIR_HEAD" = 1 ] \
     || [ "$MOMENTS" != none ]; then
    echo "MODEL=moments_only has no graph encoder: INPUT_MOMENTS_PROJ, LIGAND_ONLY, SIAMESE, PAIR_HEAD and MOMENTS" \
         "are not allowed" >&2; exit 2
  fi
fi
if [ "$MOMENTS" != none ] || [ "$INPUT_MOMENTS" != none ]; then SIDE_USED=1; else SIDE_USED=0; fi
if [ "$MOMENTS" = none ] && [ -n "${MOMENTS_WEIGHT:-}${MOMENTS_KINDS:-}" ]; then
  echo "moments settings given with MOMENTS=none (did you forget MOMENTS=both?)" >&2; exit 2
fi
if [ "$SIDE_USED" = 0 ] && { [ -n "${ZERNIKE_SIDE:-}" ] || [ "${ALLOW_PARTIAL_SIDE:-0}" = 1 ]; }; then
  echo "ZERNIKE_SIDE / ALLOW_PARTIAL_SIDE given with MOMENTS=none and INPUT_MOMENTS=none (did you forget" \
       "MOMENTS=both or INPUT_MOMENTS=...?)" >&2; exit 2
fi
if [ "$SIDE_USED" = 1 ]; then
  ZERNIKE_SIDE=${ZERNIKE_SIDE:?MOMENTS=$MOMENTS / INPUT_MOMENTS=$INPUT_MOMENTS / MODEL=$MODEL needs ZERNIKE_SIDE (paired_zernike.npz from build_zernike_side.py)}
  if [ ! -s "$ZERNIKE_SIDE" ]; then
    echo "ZERNIKE_SIDE=$ZERNIKE_SIDE is missing (run build_dataset.sbatch with ZERNIKE=1)" >&2; exit 2
  fi
fi
PAIR_HEAD_WEIGHT=${PAIR_HEAD_WEIGHT:-0.3}
PAIR_HIDDEN=${PAIR_HIDDEN:-512}
PAIR_NEG_KINDS=${PAIR_NEG_KINDS:-}
PAIR_NEG_KINDS=${PAIR_NEG_KINDS// /}
MOMENTS_WEIGHT=${MOMENTS_WEIGHT:-0.3}
MOMENTS_KINDS=${MOMENTS_KINDS:-}
MOMENTS_KINDS=${MOMENTS_KINDS// /}
ALLOW_PARTIAL_SIDE=${ALLOW_PARTIAL_SIDE:-0}
case "$ALLOW_PARTIAL_SIDE" in 0|1) ;; *) echo "ALLOW_PARTIAL_SIDE must be 0 or 1 (got $ALLOW_PARTIAL_SIDE)" >&2; exit 2 ;; esac
CHUNK=${CHUNK:-}
case "$CHUNK" in ""|*[!0-9]*) [ -z "$CHUNK" ] || { echo "CHUNK must be a positive integer (got $CHUNK)" >&2; exit 2; } ;; esac
SAMPLER_PARENT_AWARE=${SAMPLER_PARENT_AWARE:-0}
case "$SAMPLER_PARENT_AWARE" in 0|1) ;; *) echo "SAMPLER_PARENT_AWARE must be 0 or 1 (got $SAMPLER_PARENT_AWARE)" >&2; exit 2 ;; esac
SIAMESE_WEIGHT=${SIAMESE_WEIGHT:-1.0}
SIAMESE_HIDDEN=${SIAMESE_HIDDEN:-512}
SIAMESE_TIE=${SIAMESE_TIE:-0.5}
TRAIN_KINDS=${TRAIN_KINDS:-0}
TRAIN_KINDS=${TRAIN_KINDS// /}
HINGE_MARGINS=${HINGE_MARGINS:-2:0.75,4:1.0,6:1.0}
HINGE_MARGINS=${HINGE_MARGINS// /}
HINGE_WEIGHT=${HINGE_WEIGHT:-1.0}
DECOY_WEIGHT=${DECOY_WEIGHT:-}
DECOY_MARGIN=${DECOY_MARGIN:-}
PAIR_LOSS=${PAIR_LOSS:-none}
PAIR_WEIGHT=${PAIR_WEIGHT:-0.3}
TIE=${TIE:-0.5}
HUBER_DELTA=${HUBER_DELTA:-1.0}
WEIGHTING=${WEIGHTING:-none}
LIGAND_ONLY=${LIGAND_ONLY:-0}
EPOCHS=${EPOCHS:-200}
DATA_DIR=${DATA_DIR:-$WORK/data}
SEED_INDEX=${SLURM_ARRAY_TASK_ID:?submit as an array job}

FLAGS=(--data-dir "$DATA_DIR" --seed_index "$SEED_INDEX" --tag "$TAG" --epochs "$EPOCHS"
       --pair-loss "$PAIR_LOSS" --pair-weight "$PAIR_WEIGHT" --tie "$TIE"
       --huber-delta "$HUBER_DELTA" --pair-weighting "$WEIGHTING"
       --train-kinds "$TRAIN_KINDS" --hinge-margins "$HINGE_MARGINS" --hinge-weight "$HINGE_WEIGHT")
if [ -n "$DECOY_WEIGHT" ]; then
    FLAGS+=(--decoy-weight "$DECOY_WEIGHT")
fi
if [ -n "$DECOY_MARGIN" ]; then
    FLAGS+=(--decoy-margin "$DECOY_MARGIN")
fi
if [ "$LIGAND_ONLY" = 1 ]; then
    FLAGS+=(--ligand-only)
fi
if [ "$SIAMESE" = 1 ]; then
    FLAGS+=(--siamese --siamese-weight "$SIAMESE_WEIGHT" --siamese-hidden "$SIAMESE_HIDDEN"
            --siamese-tie "$SIAMESE_TIE")
fi
if [ "$PAIR_HEAD" = 1 ]; then
    FLAGS+=(--pair-head --pair-head-weight "$PAIR_HEAD_WEIGHT" --pair-hidden "$PAIR_HIDDEN")
    if [ -n "$PAIR_NEG_KINDS" ]; then
        FLAGS+=(--pair-neg-kinds "$PAIR_NEG_KINDS")
    fi
fi
if [ "$MOMENTS" != none ]; then
    FLAGS+=(--moments "$MOMENTS" --moments-weight "$MOMENTS_WEIGHT")
    if [ -n "$MOMENTS_KINDS" ]; then
        FLAGS+=(--moments-kinds "$MOMENTS_KINDS")
    fi
fi
if [ "$INPUT_MOMENTS" != none ]; then
    FLAGS+=(--input-moments "$INPUT_MOMENTS")
    if [ -n "$INPUT_MOMENTS_PROJ" ]; then
        FLAGS+=(--input-moments-proj "$INPUT_MOMENTS_PROJ")
    fi
fi
if [ "$MODEL" != gatv2 ]; then
    FLAGS+=(--model "$MODEL")
fi
if [ "$SIDE_USED" = 1 ]; then
    FLAGS+=(--zernike-side "$ZERNIKE_SIDE")
    if [ "$ALLOW_PARTIAL_SIDE" = 1 ]; then
        FLAGS+=(--allow-partial-side)
    fi
fi
if [ "$SAMPLER_PARENT_AWARE" = 1 ]; then
    FLAGS+=(--sampler-parent-aware)
fi
if [ -n "$CHUNK" ]; then
    FLAGS+=(--chunk "$CHUNK")
fi

echo "host $(hostname) job ${SLURM_JOB_ID} array ${SLURM_ARRAY_JOB_ID:-}_${SEED_INDEX} start $(date -Is)"
nvidia-smi -L || true
echo "python $SRC/train_paired.py ${FLAGS[*]}"
python "$SRC/train_paired.py" "${FLAGS[@]}"
echo "done $(date -Is)"
