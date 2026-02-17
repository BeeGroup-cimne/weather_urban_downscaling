#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ml_m4/bin/python}"

MODELS="${MODELS:-unet lstm transformer mamba}"
PATCH_SIZE="${PATCH_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-6}"
EPOCHS="${EPOCHS:-35}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-3000}"
VAL_PATCHES="${VAL_PATCHES:-1600}"
PREVIEW_TIME="${PREVIEW_TIME:-2017-08-15T15:00:00}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-4}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
LR_PATIENCE="${LR_PATIENCE:-3}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_MIN="${LR_MIN:-1e-6}"

TRAIN_START="${TRAIN_START:-2017-05-01}"
TRAIN_END="${TRAIN_END:-2017-08-01}"
VAL_START="${VAL_START:-2017-08-01}"
VAL_END="${VAL_END:-2017-09-01}"
TEST_START="${TEST_START:-2017-09-01}"
TEST_END="${TEST_END:-2017-10-01}"

read -r -a MODELS_ARR <<< "$MODELS"

caffeinate -dimsu "${PYTHON_BIN}" scripts/run_ablation_tiles.py \
  --models "${MODELS_ARR[@]}" \
  --seq-len "${SEQ_LEN}" \
  --patch-size "${PATCH_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --patches-per-epoch "${PATCHES_PER_EPOCH}" \
  --val-patches "${VAL_PATCHES}" \
  --sampler uhi_proxy \
  --temporal-sampler p95 \
  --no-temporal-season-balance \
  --epochs "${EPOCHS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --lr-patience "${LR_PATIENCE}" \
  --lr-factor "${LR_FACTOR}" \
  --lr-min "${LR_MIN}" \
  --split-mode time \
  --train-start "${TRAIN_START}" \
  --train-end "${TRAIN_END}" \
  --val-start "${VAL_START}" \
  --val-end "${VAL_END}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --post-inference \
  --preview-time "${PREVIEW_TIME}"

echo "Training finished."
echo "Split used (independent from config/config.py):"
echo "  Train: ${TRAIN_START} -> ${TRAIN_END}"
echo "  Val:   ${VAL_START} -> ${VAL_END}"
