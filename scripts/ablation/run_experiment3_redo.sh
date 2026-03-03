#!/usr/bin/env bash
# =============================================================================
# 🔬 Experiment 3 — Full Redo with SEQ=6
#
# Retrains ALL models from scratch with forced SEQ_LEN=6 across seeds 42, 43, 44.
# Optimized for NVIDIA L4 (24GB): BATCH_SIZE=4, SEQ_LEN=6.
#
# Training plan (15 runs):
#   Phase 1-3: unet, lstm, transformer, mamba  (SEQ=6, 50 epochs)  × 3 seeds
#   Phase 4:   mamba_seq12                      (SEQ=12, 100 epochs) × 3 seeds
#
# Usage:
#   # Foreground:
#   bash scripts/ablation/run_experiment3_redo.sh
#
#   # Background (recommended):
#   nohup bash scripts/ablation/run_experiment3_redo.sh > experiments/logs/exp3_redo.log 2>&1 &
#   tail -f experiments/logs/exp3_redo.log
#
# Environment overrides:
#   SEEDS="43 44"      - Only specific seeds
#   SKIP_BACKUP=1      - Skip checkpoint backup step
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEEDS="${SEEDS:-42 43 44}"
MODELS_SEQ6="unet lstm transformer mamba"
EPOCHS_SEQ6=50
BATCH_SIZE_SEQ6=4
SEQ_LEN_SEQ6=6

EPOCHS_SEQ12=100
BATCH_SIZE_SEQ12=2
SEQ_LEN_SEQ12=12

DOCKER_IMAGE="weather_thesis:tf"
SKIP_BACKUP="${SKIP_BACKUP:-0}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  🔬 Experiment 3 — Full Redo                         ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Started: $(timestamp)"
echo "║  Seeds:   ${SEEDS}"
echo "║"
echo "║  Phase 1-3: ${MODELS_SEQ6}"
echo "║    SEQ=${SEQ_LEN_SEQ6}  BS=${BATCH_SIZE_SEQ6}  EPOCHS=${EPOCHS_SEQ6}"
echo "║"
echo "║  Phase 4: mamba (SEQ=12)"
echo "║    SEQ=${SEQ_LEN_SEQ12}  BS=${BATCH_SIZE_SEQ12}  EPOCHS=${EPOCHS_SEQ12}"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Backup existing seed 42 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
BACKUP_DIR="experiments/models/backup_seq12_$(date +%Y%m%d_%H%M%S)"

if [[ "${SKIP_BACKUP}" != "1" ]]; then
  existing_s42=$(ls experiments/models/Ablation_*_S42_best.h5 2>/dev/null || true)
  existing_s42_seq12=$(ls experiments/models/Ablation_*_S42_SEQ12_best.h5 2>/dev/null || true)
  existing_last=$(ls experiments/models/Ablation_*_S42_last.h5 2>/dev/null || true)
  existing_last_seq12=$(ls experiments/models/Ablation_*_S42_SEQ12_last.h5 2>/dev/null || true)

  if [[ -n "${existing_s42}${existing_s42_seq12}" ]]; then
    echo "📦 Backing up old seed 42 checkpoints to ${BACKUP_DIR}/"
    mkdir -p "${BACKUP_DIR}"
    for f in ${existing_s42} ${existing_s42_seq12} ${existing_last} ${existing_last_seq12}; do
      if [[ -f "$f" ]]; then
        cp "$f" "${BACKUP_DIR}/"
        rm "$f"
        echo "   Moved: $(basename $f)"
      fi
    done
    # Also backup and clear logs
    for f in experiments/logs/Ablation_*S42*.csv; do
      if [[ -f "$f" ]]; then
        cp "$f" "${BACKUP_DIR}/"
        rm "$f"
        echo "   Moved: $(basename $f)"
      fi
    done
    echo "✅ Backup complete."
  else
    echo "ℹ️  No existing seed 42 checkpoints found. Nothing to backup."
  fi
else
  echo "⏭️  Skipping backup (SKIP_BACKUP=1)."
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one training invocation
# ─────────────────────────────────────────────────────────────────────────────
run_training() {
  local seed="$1"
  local suffix="$2"
  local seq_len="$3"
  local batch_size="$4"
  local epochs="$5"
  local models="$6"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎲 Seed=${seed}  📏 SEQ=${seq_len}  📦 BS=${batch_size}  🕐 Epochs=${epochs}"
  echo "   Suffix: ${suffix}"
  echo "   Models: ${models}"
  echo "   Started: $(timestamp)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  docker run --rm \
    --gpus '"device=0"' \
    -v "$(pwd)":/app \
    -w /app \
    -e TF_USE_LEGACY_KERAS=1 \
    -e PYTHONPATH=/app \
    -e PYTHONUNBUFFERED=1 \
    -e FULLFRAME=1 \
    -e MPLBACKEND=Agg \
    -e SAVE_MODEL_DIAGRAM=0 \
    -e SAVE_VISUALIZATIONS=0 \
    -e SAVE_COMPARATIVE_HISTORY=0 \
    "${DOCKER_IMAGE}" \
    python scripts/ablation/run_ablation.py \
      --seed "${seed}" \
      --experiment-suffix "${suffix}" \
      --seq-len "${seq_len}" \
      --batch-size "${batch_size}" \
      --epochs "${epochs}" \
      --models ${models}

  echo "✅ Completed: seed=${seed} suffix=${suffix} ($(timestamp))"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-3: All models with SEQ=6
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  PHASES 1-3: All models · SEQ=${SEQ_LEN_SEQ6} · EPOCHS=${EPOCHS_SEQ6}"
echo "═══════════════════════════════════════════════════════"

for seed in ${SEEDS}; do
  suffix="S${seed}"

  # Skip check
  all_done=true
  for model in ${MODELS_SEQ6}; do
    ckpt="experiments/models/Ablation_$(echo "${model}" | tr '[:lower:]' '[:upper:]')_Legacy_${suffix}_best.h5"
    if [[ ! -f "${ckpt}" ]]; then
      all_done=false
      break
    fi
  done

  if $all_done; then
    echo "⏭️  Seed=${seed} SEQ=${SEQ_LEN_SEQ6}: all models already trained. Skipping."
    continue
  fi

  run_training "${seed}" "${suffix}" "${SEQ_LEN_SEQ6}" "${BATCH_SIZE_SEQ6}" "${EPOCHS_SEQ6}" "${MODELS_SEQ6}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Mamba only with SEQ=12
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  PHASE 4: Mamba only · SEQ=${SEQ_LEN_SEQ12} · EPOCHS=${EPOCHS_SEQ12}"
echo "═══════════════════════════════════════════════════════"

for seed in ${SEEDS}; do
  suffix="S${seed}_SEQ12"
  ckpt="experiments/models/Ablation_MAMBA_Legacy_${suffix}_best.h5"

  if [[ -f "${ckpt}" ]]; then
    echo "⏭️  Seed=${seed} mamba SEQ=${SEQ_LEN_SEQ12}: already trained. Skipping."
    continue
  fi

  run_training "${seed}" "${suffix}" "${SEQ_LEN_SEQ12}" "${BATCH_SIZE_SEQ12}" "${EPOCHS_SEQ12}" "mamba"
done

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  ✅ Experiment 3 Redo — Complete                      ║"
echo "║  Finished: $(timestamp)"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Checkpoints:                                        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "📁 Checkpoint inventory:"
ls -lh experiments/models/Ablation_*_best.h5 2>/dev/null | awk '{print "   " $NF " (" $5 ")"}'
echo ""
total_ckpts=$(ls experiments/models/Ablation_*_best.h5 2>/dev/null | wc -l)
echo "   Total: ${total_ckpts} / 15 expected"
echo ""
echo "Next steps:"
echo "  1. bash scripts/evaluation/run_casestudy1_eval.sh"
echo "  2. bash scripts/evaluation/run_casestudy2_eval.sh"
echo "  3. Regenerate figures"
