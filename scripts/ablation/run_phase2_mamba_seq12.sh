#!/usr/bin/env bash
# =============================================================================
# Phase 2: Train mamba with SEQ=12 for seeds 43 & 44
#
# Configuration MATCHED to seed 42 mamba_seq12 (Ablation_MAMBA_Legacy_S42_SEQ12):
#   - EPOCHS=100 (GPUServerConfig default; seed 42 early-stopped at 65)
#   - SEQ_LEN=12 (from GPUServerConfig; --min-seq-len 12 is a no-op here)
#   - BATCH_SIZE=2
#   - LR=5e-05
#   - EARLY_STOPPING_PATIENCE=10
#
# Usage:
#   bash scripts/ablation/run_phase2_mamba_seq12.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-43 44}"
EPOCHS="${EPOCHS:-100}"

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 2: Mamba SEQ=12 · EPOCHS=${EPOCHS}           ║"
echo "║  Config: GPUServerConfig (SEQ=12, BS=2)      ║"
echo "║  Seeds: ${SEEDS}"
echo "╚══════════════════════════════════════════════╝"

for seed in ${SEEDS}; do
  suffix="S${seed}_SEQ12"
  ckpt="experiments/models/Ablation_MAMBA_Legacy_${suffix}_best.h5"

  if [[ -f "${ckpt}" ]]; then
    echo "⏭️  Seed=${seed} mamba SEQ=12: already trained. Skipping."
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎲 Seed=${seed}  📛 ${suffix}  🕐 Epochs=${EPOCHS}"
  echo "   Model: mamba (only)"
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
      --min-seq-len 12 \
      --epochs ${EPOCHS} \
      --models mamba

  echo "✅ Seed=${seed} mamba SEQ=12 complete."
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Phase 2 complete                         ║"
echo "╚══════════════════════════════════════════════╝"
