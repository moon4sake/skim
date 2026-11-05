#!/usr/bin/env bash
set -euo pipefail

usage="usage: ./run.sh <prune|kd|eval|dense> <model> <sparsity> [seed] [extra flags for the stage's command]"
STAGE=${1:?$usage}
MODEL=${2:?$usage}
SPARSITY=${3:?$usage}
shift 3
SEED=0
if [ $# -gt 0 ] && [[ $1 != -* ]]; then SEED=$1; shift; fi

OUT=${OUT:-out}
DEVICE=${DEVICE:-cuda:0}
DATASETS=${DATASETS:-bfcl apibank}
EVAL_TARGET=${EVAL_TARGET:-kd}
SP=$(awk -v s="$SPARSITY" 'BEGIN { printf "%02d", s * 100 + 0.5 }')
EXP=$OUT/${MODEL}_sp$SP
RUN=$EXP/seed$SEED
KD=kd${TAG:+_$TAG}

TEACHER=()
if [ -n "${TEACHER_DEVICE:-}" ]; then TEACHER=(--teacher_device "$TEACHER_DEVICE"); fi

case $STAGE in
  prune)
    skim-prune --model "$MODEL" --sparsity "$SPARSITY" --output_dir "$EXP/prune" \
      --device "$DEVICE" "$@"
    ;;
  kd)
    skim-distill --model "$MODEL" --mask_path "$EXP/prune/pruning_masks.pt" \
      --output_dir "$RUN/$KD" --device "$DEVICE" --seed "$SEED" ${TEACHER[@]+"${TEACHER[@]}"} "$@"
    ;;
  eval)
    for DATASET in $DATASETS; do
      if [ "$EVAL_TARGET" = prune ]; then
        SOURCE=(--model "$MODEL" --mask_path "$EXP/prune/pruning_masks.pt")
        EVAL_DIR=$EXP/eval_prune
      else
        SOURCE=(--model "$MODEL" --model_path "$RUN/$KD/final")
        EVAL_DIR=$RUN/eval_$KD
      fi
      skim-eval "${SOURCE[@]}" --dataset "$DATASET" --output_dir "$EVAL_DIR" \
        --device "$DEVICE" "$@"
    done
    ;;
  dense)
    for DATASET in $DATASETS; do
      skim-eval --model "$MODEL" --dataset "$DATASET" --output_dir "$OUT/${MODEL}_dense" \
        --device "$DEVICE" "$@"
    done
    ;;
  *)
    echo "$usage" >&2
    exit 2
    ;;
esac
