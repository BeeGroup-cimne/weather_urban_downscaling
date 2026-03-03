#!/usr/bin/env bash
# =============================================================================
# Phase 1: Train unet, lstm, transformer, mamba for seeds 43 & 44
#
# Configuration MATCHED to seed 42 original training:
#   - EPOCHS=50 (via --epochs 50)
#   - SEQ_LEN=12 (from GPUServerConfig for 23GB GPU)
#   - BATCH_SIZE=2 (from GPUServerConfig)
#   - LR=5e-05 (from GPUServerConfig)
#   - EARLY_STOPPING_PATIENCE=10
#   - FULLFRAME=1
#
# NOTE: --min-seq-len 6 does NOT lower SEQ_LEN when GPUServerConfig already
#       sets it to 12. All seed 42 models were trained with SEQ_LEN=12.
#       We replicate the same behavior here for consistency.
#
# Usage:
#   bash scripts/ablation/run_phase1_seq6.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-43 44}"
MODELS="unet lstm transformer mamba"
EPOCHS=50

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 1: All models · EPOCHS=${EPOCHS}              ║"
echo "║  Config: GPUServerConfig (SEQ=12, BS=2)      ║"
echo "║  Seeds: ${SEEDS}"
echo "║  Models: ${MODELS}"
echo "╚══════════════════════════════════════════════╝"

for seed in ${SEEDS}; do
  suffix="S${seed}"

  # Skip check
  all_done=true
  for model in ${MODELS}; do
    ckpt="experiments/models/Ablation_$(echo "${model}" | tr '[:lower:]' '[:upper:]')_Legacy_${suffix}_best.h5"
    if [[ ! -f "${ckpt}" ]]; then
      all_done=false
      break
    fi
  done

  if $all_done; then
    echo "⏭️  Seed=${seed}: all models already trained. Skipping."
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎲 Seed=${seed}  📛 ${suffix}  🕐 Epochs=${EPOCHS}"
  echo "   Models: ${MODELS}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  docker run --rm \
    --gpus all \
    -v "$(pwd)":/app \
    -w /app \
    -e TF_USE_LEGACY_KERAS=1 \
    -e PYTHONPATH=/app \
    -e PYTHONUNBUFFERED=1 \
    -e FULLFRAME=1 \
    -e MPLBACKEND=Agg \
    weather_thesis:tf \
    python scripts/ablation/run_ablation.py \
      --seed "${seed}" \
      --experiment-suffix "${suffix}" \
      --min-seq-len 6 \
      --epochs ${EPOCHS} \
      --models ${MODELS}

  echo "✅ Seed=${seed} complete."
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Phase 1 complete                         ║"
echo "╚══════════════════════════════════════════════╝"
