#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Full-frame local fast run (UNet -> LSTM -> Mamba)"
echo "Using Config.MAX_STEPS_PER_EPOCH and reduced SHUFFLE_BUFFER_SIZE"

python -m scripts.tools.print_active_config

echo ""
echo "▶️ UNet"
python scripts/ablation/run_ablation.py --models unet

echo ""
echo "▶️ LSTM"
python scripts/ablation/run_ablation.py --models lstm

echo ""
echo "▶️ Mamba"
python scripts/ablation/run_ablation.py --models mamba

echo "✅ Full-frame local fast run complete."
