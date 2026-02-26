#!/bin/bash
set -e

echo "=========================================================="
echo "📊 Evaluando SSIM: UNET Base (SEQ=6)"
echo "=========================================================="
python scripts/evaluation/evaluate_test_set.py \
    --model-type unet \
    --model-path experiments/models/Tiles_UNET_S42_best.h5 \
    --split val \
    --seq-len 6 \
    --ssim-samples 200

echo "=========================================================="
echo "📊 Evaluando SSIM: ConvLSTM (SEQ=6)"
echo "=========================================================="
python scripts/evaluation/evaluate_test_set.py \
    --model-type convlstm \
    --model-path experiments/models/Tiles_LSTM_S42_best.h5 \
    --split val \
    --seq-len 6 \
    --ssim-samples 200

echo "=========================================================="
echo "📊 Evaluando SSIM: Transformer (SEQ=6)"
echo "=========================================================="
python scripts/evaluation/evaluate_test_set.py \
    --model-type transformer \
    --model-path experiments/models/Tiles_TRANSFORMER_S42_best.h5 \
    --split val \
    --seq-len 6 \
    --ssim-samples 200

echo "✅ Evaluaciones de Baselines finalizadas."
