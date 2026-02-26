#!/bin/bash
set -e

echo "=========================================================="
echo "🎯 Entrenando MAMBA con Configuración SEQ_LEN=12"
echo "=========================================================="

# Forzamos solo correr Mamba, y le damos un tag especial para no sobreescribir el exp1
python scripts/ablation/run_ablation.py \
    --models mamba \
    --experiment-suffix S42_SEQ12 \
    --epochs 200 

echo "=========================================================="
echo "✅ Entrenamiento finalizado (ver experiments/logs/)"
echo "=========================================================="
