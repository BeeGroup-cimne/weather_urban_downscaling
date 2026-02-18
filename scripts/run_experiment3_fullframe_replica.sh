#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export FULLFRAME="${FULLFRAME:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/miniconda3/envs/ml_m4/bin/python" ]]; then
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
MODELS="${MODELS:-transformer mamba}"
SEEDS="${SEEDS:-42 43 44}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-6}"
EPOCHS="${EPOCHS:-35}"

TRAIN_START="${TRAIN_START:-2017-05-01}"
TRAIN_END="${TRAIN_END:-2017-08-01}"
VAL_START="${VAL_START:-2017-08-01}"
VAL_END="${VAL_END:-2017-09-01}"
TEST_START="${TEST_START:-2017-09-01}"
TEST_END="${TEST_END:-2017-10-01}"

EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_SSIM_SAMPLES="${EVAL_SSIM_SAMPLES:-128}"

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
    *) echo "$1" ;;
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
  echo "EXP1_AGG_CSV=${EXP1_AGG_CSV}"
} > "${OUTDIR}/run_config.env"

for seed in "${SEEDS_ARR[@]}"; do
  for model in "${MODELS_ARR[@]}"; do
    model_u="$(to_upper "${model}")"
    suffix="S${seed}"
    ckpt="experiments/models/Ablation_${model_u}_Legacy_${suffix}_best.h5"
    eval_type="$(model_type_for_eval "${model}")"
    eval_csv="${OUTDIR}/evals/${model}_${suffix}.csv"

    echo "=== TRAIN full-frame: model=${model} seed=${seed} ==="
    train_status="ok"
    if ! "${PYTHON_BIN}" scripts/run_ablation.py \
      --models "${model}" \
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

    eval_status="skipped"
    if [[ -f "${ckpt}" ]]; then
      echo "=== EVAL full-frame: model=${model} seed=${seed} split=${EVAL_SPLIT} ==="
      if "${PYTHON_BIN}" scripts/evaluate_test_set.py \
        --model-type "${eval_type}" \
        --model-path "${ckpt}" \
        --split "${EVAL_SPLIT}" \
        --max-batches "${EVAL_MAX_BATCHES}" \
        --ssim-samples "${EVAL_SSIM_SAMPLES}" \
        --out-csv "${eval_csv}" \
        --split-mode time \
        --train-start "${TRAIN_START}" \
        --train-end "${TRAIN_END}" \
        --val-start "${VAL_START}" \
        --val-end "${VAL_END}" \
        --test-start "${TEST_START}" \
        --test-end "${TEST_END}"; then
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
            log_path = os.path.join("experiments", "logs", f"Ablation_{model_u}_Legacy_S{seed}_log.csv")
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

if [[ -z "${EXP1_AGG_CSV}" ]]; then
  if [[ -f "experiments/heatwaves/latest/metrics_aggregate_ci.csv" ]]; then
    EXP1_AGG_CSV="experiments/heatwaves/latest/metrics_aggregate_ci.csv"
  fi
fi

consolidate_cmd=(
  "${PYTHON_BIN}" scripts/consolidate_experiment3.py
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
