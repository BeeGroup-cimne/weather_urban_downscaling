#!/usr/bin/env bash
# =============================================================================
# Multi-seed ablation training for publication confidence intervals.
#
# Experimental design (per seed):
#   Phase 1 — SEQ=6:  unet, lstm, transformer, mamba   (4 models)
#   Phase 2 — SEQ=12: mamba only                        (1 model, mamba_seq12)
#
# Each phase is a separate run_ablation.py invocation because --min-seq-len
# sets Config.SEQ_LEN globally for the data pipeline.
#
# Checkpoint naming convention:
#   SEQ=6:  Ablation_{MODEL}_Legacy_S{seed}_best.h5
#   SEQ=12: Ablation_MAMBA_Legacy_S{seed}_SEQ12_best.h5
#
# The script auto-skips any seed+model combo whose checkpoint already exists.
#
# Usage (GPU server, via Docker):
#   docker compose -f docker/compose.yml run --rm tf-trainer \
#       bash scripts/ablation/run_ablation_multiseed.sh
#
# Environment variables:
#   SEEDS      - Space-separated list of seeds (default: "42 43 44")
#   MODELS     - Models for SEQ=6 phase (default: "unet lstm transformer mamba")
#   EPOCHS     - Override training epochs (optional)
#   FULLFRAME  - Set to 1 for full-frame training (default: 1)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${MODELS:-unet lstm transformer mamba}"
EPOCH_FLAG=""
if [[ -n "${EPOCHS:-}" ]]; then
  EPOCH_FLAG="--epochs ${EPOCHS}"
fi

export FULLFRAME="${FULLFRAME:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "╔══════════════════════════════════════════╗"
echo "║   Multi-seed Ablation Training           ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Seeds:      ${SEEDS}"
echo "║  Phase 1:    ${MODELS}  (SEQ=6)"
echo "║  Phase 2:    mamba               (SEQ=12)"
echo "║  Fullframe:  ${FULLFRAME}"
echo "╚══════════════════════════════════════════╝"

runs_done=0
runs_skipped=0

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: All models with SEQ=6
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  PHASE 1: All models · SEQ_LEN=6"
echo "═══════════════════════════════════════════"

for seed in ${SEEDS}; do
  suffix="S${seed}"

  # Check if ALL models for this seed are already trained
  all_done=true
  for model in ${MODELS}; do
    ckpt="experiments/models/Ablation_$(echo "${model}" | tr '[:lower:]' '[:upper:]')_Legacy_${suffix}_best.h5"
    if [[ ! -f "${ckpt}" ]]; then
      all_done=false
      break
    fi
  done

  if $all_done; then
    echo "⏭️  Seed=${seed} SEQ=6: all models already trained. Skipping."
    runs_skipped=$((runs_skipped + 1))
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎲 Seed=${seed}  📏 SEQ_LEN=6  📛 Suffix=${suffix}"
  echo "   Models: ${MODELS}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  ${PYTHON_BIN} scripts/ablation/run_ablation.py \
    --seed "${seed}" \
    --experiment-suffix "${suffix}" \
    --min-seq-len 6 \
    --models ${MODELS} \
    ${EPOCH_FLAG}

  runs_done=$((runs_done + 1))
  echo "✅ Phase 1 done for seed=${seed}"
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Mamba only with SEQ=12
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  PHASE 2: Mamba only · SEQ_LEN=12"
echo "═══════════════════════════════════════════"

for seed in ${SEEDS}; do
  suffix_seq12="S${seed}_SEQ12"
  ckpt_mamba12="experiments/models/Ablation_MAMBA_Legacy_${suffix_seq12}_best.h5"

  if [[ -f "${ckpt_mamba12}" ]]; then
    echo "⏭️  Seed=${seed} mamba SEQ=12: already trained. Skipping."
    runs_skipped=$((runs_skipped + 1))
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎲 Seed=${seed}  📏 SEQ_LEN=12  📛 Suffix=${suffix_seq12}"
  echo "   Models: mamba (only)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  ${PYTHON_BIN} scripts/ablation/run_ablation.py \
    --seed "${seed}" \
    --experiment-suffix "${suffix_seq12}" \
    --min-seq-len 12 \
    --models mamba \
    ${EPOCH_FLAG}

  runs_done=$((runs_done + 1))
  echo "✅ Phase 2 done for seed=${seed}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅ Multi-seed training complete         ║"
echo "║  Runs completed: ${runs_done}"
echo "║  Runs skipped:   ${runs_skipped}"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Run consolidated evaluation (Exp1, Exp2, Exp3)"
echo "  2. Regenerate figures and reports"
