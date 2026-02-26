#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Server full-frame run (run_ablation: 4 modelos)"
echo "   Models: unet lstm transformer mamba"

export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export FULLFRAME="${FULLFRAME:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Por defecto no generamos imágenes/figuras en servidor.
export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"

python -m scripts.print_active_config
python scripts/run_ablation.py --models unet lstm transformer mamba --min-seq-len 6

