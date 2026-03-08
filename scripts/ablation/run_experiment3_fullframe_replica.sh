#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export FULLFRAME="${FULLFRAME:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"
# For this legacy ablation stack, default to tf_keras compatibility on TF 2.16+/Keras 3.
export FORCE_TF_LEGACY_KERAS="${FORCE_TF_LEGACY_KERAS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  elif [[ -x "/opt/miniconda3/envs/ml_m4/bin/python" ]]; then
    PYTHON_BIN="/opt/miniconda3/envs/ml_m4/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

# Experiment 3 defaults: full-frame replica on top models
MODELS="${MODELS:-unet lstm transformer mamba mamba_seq12}"
SEEDS="${SEEDS:-42 43 44}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-6}"
EPOCHS="${EPOCHS:-35}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
CKPT_PREFIX="${CKPT_PREFIX:-Ablation_@MODEL@_Legacy_}"
export CKPT_PREFIX

TRAIN_START="${TRAIN_START:-2017-05-01}"
TRAIN_END="${TRAIN_END:-2017-08-01}"
VAL_START="${VAL_START:-2017-08-01}"
VAL_END="${VAL_END:-2017-09-01}"
TEST_START="${TEST_START:-2017-09-01}"
TEST_END="${TEST_END:-2017-10-01}"

EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_SSIM_SAMPLES="${EVAL_SSIM_SAMPLES:-128}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-25}"

OUTDIR="${OUTDIR:-experiments/fullframe/experiment3_$(date +%Y%m%d_%H%M%S)}"
EXP1_AGG_CSV="${EXP1_AGG_CSV:-}"

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/evals"

read -r -a MODELS_ARR <<< "${MODELS}"
read -r -a SEEDS_ARR <<< "${SEEDS}"

RAW_CSV="${OUTDIR}/fullframe_eval_raw.csv"
echo "model,seed,split,mae,rmse,mse,ssim,ssim_samples,model_type,model_path" > "${RAW_CSV}"

RUNS_CSV="${OUTDIR}/runs_status.csv"
echo "model,seed,checkpoint,train_status,eval_status" > "${RUNS_CSV}"

to_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

model_type_for_eval() {
  case "$1" in
    lstm) echo "convlstm" ;;
    mamba_seq12) echo "mamba" ;;
    *) echo "$1" ;;
  esac
}

checkpoint_for_model_seed() {
  local model="$1"
  local seed="$2"
  local model_u
  model_u="$(to_upper "${model}")"
  local suffix="S${seed}"
  local ckpt_prefix="${CKPT_PREFIX//@MODEL@/${model_u}}"
  if [[ "${model}" == "mamba_seq12" ]]; then
    echo "experiments/models/Ablation_MAMBA_Legacy_${suffix}_SEQ12_best.h5"
    return
  fi
  echo "experiments/models/${ckpt_prefix}${suffix}_best.h5"
}

train_model_key() {
  case "$1" in
    mamba_seq12) echo "mamba" ;;
    *) echo "$1" ;;
  esac
}

eval_seq_len_for_model() {
  case "$1" in
    mamba_seq12) echo "12" ;;
    *) echo "" ;;
  esac
}

echo "🚀 Experiment 3 - full-frame replica"
echo "   Models: ${MODELS}"
echo "   Seeds: ${SEEDS}"
echo "   Split: ${TRAIN_START}..${TRAIN_END} | ${VAL_START}..${VAL_END} | ${TEST_START}..${TEST_END}"
echo "   Eval split: ${EVAL_SPLIT}"
echo "   Output dir: ${OUTDIR}"

{
  echo "MODELS=${MODELS}"
  echo "SEEDS=${SEEDS}"
  echo "MIN_SEQ_LEN=${MIN_SEQ_LEN}"
  echo "EPOCHS=${EPOCHS}"
  echo "TRAIN_START=${TRAIN_START}"
  echo "TRAIN_END=${TRAIN_END}"
  echo "VAL_START=${VAL_START}"
  echo "VAL_END=${VAL_END}"
  echo "TEST_START=${TEST_START}"
  echo "TEST_END=${TEST_END}"
  echo "EVAL_SPLIT=${EVAL_SPLIT}"
  echo "EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES}"
  echo "EVAL_SSIM_SAMPLES=${EVAL_SSIM_SAMPLES}"
  echo "EVAL_LOG_EVERY=${EVAL_LOG_EVERY}"
  echo "EXP1_AGG_CSV=${EXP1_AGG_CSV}"
  echo "SKIP_TRAINING=${SKIP_TRAINING}"
  echo "CKPT_PREFIX=${CKPT_PREFIX}"
} > "${OUTDIR}/run_config.env"

for seed in "${SEEDS_ARR[@]}"; do
  for model in "${MODELS_ARR[@]}"; do
    suffix="S${seed}"
    ckpt="$(checkpoint_for_model_seed "${model}" "${seed}")"
    eval_type="$(model_type_for_eval "${model}")"
    eval_seq_len="$(eval_seq_len_for_model "${model}")"
    model_train_key="$(train_model_key "${model}")"
    eval_csv="${OUTDIR}/evals/${model}_${suffix}.csv"

    if [[ "${SKIP_TRAINING}" == "1" ]]; then
      echo "=== SKIP TRAIN (eval-only): model=${model} seed=${seed} ckpt=${ckpt} ==="
      train_status="skipped"
    else
      echo "=== TRAIN full-frame: model=${model} seed=${seed} ==="
      train_status="ok"
      if [[ "${model}" == "mamba_seq12" ]]; then
        echo "⚠️ Training mamba_seq12 is not supported from this script. Use SKIP_TRAINING=1 with pre-trained checkpoint."
        train_status="unsupported"
      elif ! "${PYTHON_BIN}" scripts/ablation/run_ablation.py \
        --models "${model_train_key}" \
        --min-seq-len "${MIN_SEQ_LEN}" \
        --seed "${seed}" \
        --experiment-suffix "${suffix}" \
        --epochs "${EPOCHS}" \
        --split-mode time \
        --train-start "${TRAIN_START}" \
        --train-end "${TRAIN_END}" \
        --val-start "${VAL_START}" \
        --val-end "${VAL_END}" \
        --test-start "${TEST_START}" \
        --test-end "${TEST_END}"; then
        train_status="failed"
      fi
      if [[ ! -f "${ckpt}" ]]; then
        train_status="failed"
      fi
    fi

    eval_status="skipped"
    if [[ -f "${ckpt}" ]]; then
      echo "=== EVAL full-frame: model=${model} seed=${seed} split=${EVAL_SPLIT} ==="
      eval_cmd=(
        "${PYTHON_BIN}" scripts/evaluation/evaluate_test_set.py
        --model-type "${eval_type}"
        --model-path "${ckpt}"
        --split "${EVAL_SPLIT}"
        --max-batches "${EVAL_MAX_BATCHES}"
        --ssim-samples "${EVAL_SSIM_SAMPLES}"
        --log-every "${EVAL_LOG_EVERY}"
        --out-csv "${eval_csv}"
        --split-mode time
        --train-start "${TRAIN_START}"
        --train-end "${TRAIN_END}"
        --val-start "${VAL_START}"
        --val-end "${VAL_END}"
        --test-start "${TEST_START}"
        --test-end "${TEST_END}"
      )
      if [[ -n "${eval_seq_len}" ]]; then
        eval_cmd+=(--seq-len "${eval_seq_len}")
      fi
      if "${eval_cmd[@]}"; then
        eval_status="ok"
        tail -n +2 "${eval_csv}" | while IFS=',' read -r model_type model_path split mae rmse mse ssim ssim_samples; do
          echo "${model},${seed},${split},${mae},${rmse},${mse},${ssim},${ssim_samples},${model_type},${model_path}" >> "${RAW_CSV}"
        done
      else
        eval_status="failed"
      fi
    else
      echo "⚠️ Missing checkpoint: ${ckpt}"
    fi

    echo "${model},${seed},${ckpt},${train_status},${eval_status}" >> "${RUNS_CSV}"
  done
done

"${PYTHON_BIN}" - <<'PY' "${OUTDIR}" "${MODELS}" "${SEEDS}"
import csv
import os
import sys

out_dir = sys.argv[1]
models = sys.argv[2].split()
seeds = sys.argv[3].split()
train_path = os.path.join(out_dir, "fullframe_training_summary.csv")

with open(train_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "model", "seed", "epochs_logged",
        "best_val_loss", "best_val_loss_epoch",
        "best_val_mae", "best_val_mae_epoch",
        "final_val_loss", "final_val_mae", "final_lr"
    ])
    for model in models:
        model_u = model.upper()
        for seed in seeds:
            ckpt_prefix = os.environ.get("CKPT_PREFIX", "Ablation_@MODEL@_Legacy_").replace("@MODEL@", model_u)
            log_path = os.path.join("experiments", "logs", f"{ckpt_prefix}S{seed}_log.csv")
            if not os.path.exists(log_path):
                w.writerow([model, seed, 0, "", "", "", "", "", "", ""])
                continue
            with open(log_path, newline="", encoding="utf-8") as lf:
                rows = list(csv.DictReader(lf))
            if not rows:
                w.writerow([model, seed, 0, "", "", "", "", "", "", ""])
                continue
            for r in rows:
                r["epoch"] = int(float(r["epoch"]))
                for k in ["val_loss", "val_mae", "lr"]:
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        r[k] = float("nan")

            def _best(metric):
                valid = [r for r in rows if str(r.get(metric)) != "nan"]
                return min(valid, key=lambda x: x[metric]) if valid else None

            bvl = _best("val_loss")
            bmae = _best("val_mae")
            last = rows[-1]
            w.writerow([
                model,
                seed,
                len(rows),
                "" if bvl is None else f"{bvl['val_loss']:.6f}",
                "" if bvl is None else bvl["epoch"],
                "" if bmae is None else f"{bmae['val_mae']:.6f}",
                "" if bmae is None else bmae["epoch"],
                "" if str(last.get("val_loss")) == "nan" else f"{last['val_loss']:.6f}",
                "" if str(last.get("val_mae")) == "nan" else f"{last['val_mae']:.6f}",
                "" if str(last.get("lr")) == "nan" else f"{last['lr']:.8f}",
            ])

print(f"Training summary: {train_path}")
PY

consolidate_cmd=(
  "${PYTHON_BIN}" scripts/evaluation/consolidate_experiment3.py
  --out-dir "${OUTDIR}"
  --eval-raw-csv "${RAW_CSV}"
  --training-summary-csv "${OUTDIR}/fullframe_training_summary.csv"
)
if [[ -n "${EXP1_AGG_CSV}" && -f "${EXP1_AGG_CSV}" ]]; then
  consolidate_cmd+=(--exp1-agg-csv "${EXP1_AGG_CSV}")
fi
"${consolidate_cmd[@]}"

echo "✅ Experiment 3 completed."
echo "   Runs: ${RUNS_CSV}"
echo "   Eval raw: ${RAW_CSV}"
echo "   Training summary: ${OUTDIR}/fullframe_training_summary.csv"
echo "   Aggregate: ${OUTDIR}/fullframe_eval_aggregate.csv"
echo "   Ranking stability: ${OUTDIR}/ranking_stability_vs_exp1.csv"
echo "   Report: ${OUTDIR}/report_experiment3.md"
