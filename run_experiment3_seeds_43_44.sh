#!/bin/bash
set -e

echo "=========================================================="
echo "🎯 Entrenando y Evaluando Semillas 43 y 44 (Exp 3)"
echo "=========================================================="

export MODELS="unet lstm transformer mamba"
export SEEDS="43 44"
export SKIP_TRAINING="0"

bash scripts/ablation/run_experiment3_fullframe_replica.sh

echo ""
echo "=========================================================="
echo "✅ Entrenamiento y evaluación de semillas 43 y 44 completado."
echo "=========================================================="
