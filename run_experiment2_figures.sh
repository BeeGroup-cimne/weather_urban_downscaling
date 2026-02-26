#!/bin/bash
set -e

echo "=========================================================="
echo "🎯 Generando Figuras del Experimento 2 (Ablación Temporal)"
echo "=========================================================="

# 1. Regenerate evolution for UNet (Baseline target)
echo "----------------------------------------------------------"
echo "⏳ Regenerando fig07 para UNET..."
python scripts/figures/fig07_hourly_field_evolution.py \
    --model-type unet \
    --model-path experiments/models/Tiles_UNET_S42_best.h5 \
    --day 2017-08-15 \
    --out-dir experiments/presentation_figures/ \
    --tag "PUB"

# 2. Regenerate evolution for Mamba (Top Architecture)
echo "----------------------------------------------------------"
echo "⏳ Regenerando fig07 para MAMBA..."
python scripts/figures/fig07_hourly_field_evolution.py \
    --model-type mamba \
    --model-path experiments/models/Tiles_MAMBA_S42_best.h5 \
    --day 2017-08-15 \
    --out-dir experiments/presentation_figures/ \
    --tag "PUB"

# 3. Regenerate Comparative Side-by-Side Panel
echo "----------------------------------------------------------"
echo "⏳ Regenerando panel comparativo fig08 (MAMBA vs UNET)..."
python scripts/figures/fig08_hourly_top2_side_by_side.py \
    --model-a-type mamba \
    --model-a-path experiments/models/Tiles_MAMBA_S42_best.h5 \
    --model-b-type unet \
    --model-b-path experiments/models/Tiles_UNET_S42_best.h5 \
    --day 2017-08-15 \
    --out-dir experiments/presentation_figures/ \
    --tag "PUB"

# 4. Generate the new Training vs Validation Loss Curve
echo "----------------------------------------------------------"
echo "📉 Generando Curvas de Loss (Entrenamiento vs Validación)..."
python scripts/figures/fig09_train_val_curves.py

echo "=========================================================="
echo "✅ Generación completa. Revisa experiments/presentation_figures/"
echo "=========================================================="
