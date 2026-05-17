#!/usr/bin/env bash
set -euo pipefail

# Train only baseline models (UNet, UNet+LSTM, Transformer) with the same
# tile setup used for fair comparison against an already-trained Mamba.

cd "$(dirname "$0")/../.."

export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ml_m4/bin/python}"

PATCH_SIZE="${PATCH_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-6}"
EPOCHS="${EPOCHS:-30}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-4000}"
VAL_PATCHES="${VAL_PATCHES:-400}"
PREVIEW_TIME="${PREVIEW_TIME:-2017-08-15T15:00:00}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-8}"
LR_PATIENCE="${LR_PATIENCE:-4}"

caffeinate -dimsu "${PYTHON_BIN}" scripts/ablation/run_ablation_tiles.py \
  --models unet lstm transformer \
  --seq-len "${SEQ_LEN}" \
  --patch-size "${PATCH_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --patches-per-epoch "${PATCHES_PER_EPOCH}" \
  --val-patches "${VAL_PATCHES}" \
  --sampler uhi_proxy \
  --temporal-sampler p95 \
  --temporal-season-balance \
  --epochs "${EPOCHS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
  --lr-patience "${LR_PATIENCE}" \
  --post-inference \
  --preview-time "${PREVIEW_TIME}"

echo "Training finished."
echo "Models:"
echo "  experiments/models/Tiles_UNET_best.h5"
echo "  experiments/models/Tiles_LSTM_best.h5"
echo "  experiments/models/Tiles_TRANSFORMER_best.h5"
