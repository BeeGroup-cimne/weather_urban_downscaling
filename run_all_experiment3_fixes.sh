#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🎯 Parte A: Ejecutar Evaluación Nativa Full-Frame (Exp 3)"
echo "=========================================================="
# Ejecutamos el script desde su propio directorio para que encuentre el path relativo bien
export MODELS="unet convlstm transformer mamba" # We can use convlstm or lstm depending on the scripts logic... actually run_ablation evaluates mapping: lstm -> convlstm? Let's check. Wait, in run_experimen3_fullframe, it's 'lstm' that converts to 'convlstm' for eval. So 'lstm' is the right model name.
export MODELS="unet lstm transformer mamba"
export SEEDS="42"
export SKIP_TRAINING="1"

bash scripts/ablation/run_experiment3_fullframe_replica.sh

echo ""
echo "=========================================================="
echo "🎯 Parte B.1: Generar predicciones (.npy) para Fig G"
echo "=========================================================="
python3 scripts/figures/generate_fullframe_preds.py
# Generamos el target vs baseline
python3 scripts/figures/generate_fullframe_preds.py --baseline

echo ""
echo "=========================================================="
echo "🎯 Parte B.2: Actualizar las 7 Figuras de la Presentación"
echo "=========================================================="
python3 scripts/figures/generate_presentation_figures.py

echo ""
echo "✅ Todo completado! Puedes revisar las métricas en experiments/fullframe/ y las figuras en experiments/presentation_figures/fig_g_fullframe_comparison.png"
