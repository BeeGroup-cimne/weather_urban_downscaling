#!/usr/bin/env bash
# =============================================================================
# Multi-seed ablation training for publication confidence intervals.
#
# Usage (GPU server):
#   bash scripts/ablation/run_ablation_multiseed.sh
#
# Environment variables:
#   SEEDS      - Space-separated list of seeds (default: "42 43 44")
#   MODELS     - Models to train (default: "unet lstm transformer mamba")
#   SEQ_LENS   - Sequence lengths to train (default: "6 12")
#   EPOCHS     - Override training epochs (optional)
#   FULLFRAME  - Set to 1 for full-frame training (default: 1)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${MODELS:-unet lstm transformer mamba}"
SEQ_LENS="${SEQ_LENS:-6 12}"
EPOCH_FLAG=""
if [[ -n "${EPOCHS:-}" ]]; then
  EPOCH_FLAG="--epochs ${EPOCHS}"
fi

export FULLFRAME="${FULLFRAME:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "========================================"
echo " Multi-seed Ablation Training"
echo "========================================"
echo " Seeds:    ${SEEDS}"
echo " Models:   ${MODELS}"
echo " Seq lens: ${SEQ_LENS}"
echo " Fullframe: ${FULLFRAME}"
echo "========================================"

total=0
done=0

for seed in ${SEEDS}; do
  for seq_len in ${SEQ_LENS}; do
    suffix="S${seed}"
    if [[ "${seq_len}" != "6" ]]; then
      suffix="${suffix}_SEQ${seq_len}"
    fi

    # Check if ALL models for this seed+seq_len combo are already trained
    all_done=true
    for model in ${MODELS}; do
      ckpt="experiments/models/Ablation_$(echo "${model}" | tr '[:lower:]' '[:upper:]')_Legacy_${suffix}_best.h5"
      if [[ ! -f "${ckpt}" ]]; then
        all_done=false
        break
      fi
    done

    if $all_done; then
      echo "⏭️  All models for seed=${seed} seq_len=${seq_len} already trained. Skipping."
      continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎲 Seed=${seed}  📏 SEQ_LEN=${seq_len}  📛 Suffix=${suffix}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    model_args=""
    for model in ${MODELS}; do
      model_args="${model_args} ${model}"
    done

    ${PYTHON_BIN} scripts/ablation/run_ablation.py \
      --seed "${seed}" \
      --experiment-suffix "${suffix}" \
      --min-seq-len "${seq_len}" \
      --models ${model_args} \
      ${EPOCH_FLAG}

    done=$((done + 1))
    total=$((total + 1))
    echo "✅ Completed: seed=${seed} seq_len=${seq_len} (${done}/${total})"
  done
done

echo ""
echo "========================================"
echo " ✅ Multi-seed training complete"
echo "   Total runs: ${done}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Re-run Exp1 with Ablation_ checkpoints"
echo "  2. Re-run Exp2 with MAX_SAMPLES=1000 + BIAS_CORRECTION=1"
echo "  3. Re-run Exp3 with all seeds"
echo "  4. Regenerate report"
