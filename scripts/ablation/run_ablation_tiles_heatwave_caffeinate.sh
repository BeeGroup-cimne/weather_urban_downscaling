#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENABLE_CAFFEINATE="${ENABLE_CAFFEINATE:-auto}"
if [[ "${UNDER_CAFFEINATE:-0}" != "1" ]]; then
  if [[ "${ENABLE_CAFFEINATE}" == "1" ]] || { [[ "${ENABLE_CAFFEINATE}" == "auto" ]] && command -v caffeinate >/dev/null 2>&1; }; then
    exec caffeinate -dimsu env UNDER_CAFFEINATE=1 ENABLE_CAFFEINATE="${ENABLE_CAFFEINATE}" bash "$0" "$@"
  fi
fi

export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

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

# Publish-ready defaults:
# - homogeneous train budget for all models
# - multi-seed training
# - multi-event full-frame evaluation with fixed color scale source (hr_lr)
MODELS="${MODELS:-unet lstm transformer mamba}"
SEEDS="${SEEDS:-42 43 44}"
RUN_BASELINES="${RUN_BASELINES:-1}"
PATCH_SIZE="${PATCH_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-6}"
EPOCHS="${EPOCHS:-35}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-3000}"
VAL_PATCHES="${VAL_PATCHES:-1600}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-4}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
LR_PATIENCE="${LR_PATIENCE:-3}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_MIN="${LR_MIN:-1e-6}"
INFER_BATCH="${INFER_BATCH:-8}"
STRIDE="${STRIDE:-48}"
OUTDIR="${OUTDIR:-experiments/heatwaves/publish_run_$(date +%Y%m%d_%H%M%S)}"
TIMES_FILE="${TIMES_FILE:-}"

TRAIN_START="${TRAIN_START:-2017-05-01}"
TRAIN_END="${TRAIN_END:-2017-08-01}"
VAL_START="${VAL_START:-2017-08-01}"
VAL_END="${VAL_END:-2017-09-01}"
TEST_START="${TEST_START:-2017-09-01}"
TEST_END="${TEST_END:-2017-10-01}"

read -r -a MODELS_ARR <<< "$MODELS"
read -r -a SEEDS_ARR <<< "$SEEDS"

mkdir -p "${OUTDIR}"
FIG_DIR="${OUTDIR}/figures"
mkdir -p "${FIG_DIR}"

# Default heatwave timestamps (can be overridden via TIMES_FILE).
TIMES=(
  "2017-06-28T15:00:00"
  "2017-07-13T16:00:00"
  "2017-08-15T15:00:00"
  "2017-06-28T01:00:00"
  "2017-07-13T02:00:00"
  "2017-08-15T03:00:00"
)

if [[ -n "${TIMES_FILE}" ]]; then
  if [[ ! -f "${TIMES_FILE}" ]]; then
    echo "TIMES_FILE not found: ${TIMES_FILE}" >&2
    exit 1
  fi
  TIMES=()
  while IFS= read -r line; do
    TIMES+=("${line}")
  done < <(rg -v '^(\\s*$|\\s*#)' "${TIMES_FILE}" || true)
  if [[ "${#TIMES[@]}" -eq 0 ]]; then
    echo "TIMES_FILE provided but no timestamps were parsed: ${TIMES_FILE}" >&2
    exit 1
  fi
fi

echo "🚀 Publish-ready heatwave ablation"
echo "   Models: ${MODELS}"
echo "   Seeds: ${SEEDS}"
echo "   Events: ${#TIMES[@]} timestamps"
echo "   Output dir: ${OUTDIR}"
echo "   Python: ${PYTHON_BIN}"
if [[ "${UNDER_CAFFEINATE:-0}" == "1" ]]; then
  echo "   Caffeinate: enabled"
else
  echo "   Caffeinate: disabled"
fi

{
  echo "MODELS=${MODELS}"
  echo "SEEDS=${SEEDS}"
  echo "RUN_BASELINES=${RUN_BASELINES}"
  echo "PATCH_SIZE=${PATCH_SIZE}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "EPOCHS=${EPOCHS}"
  echo "PATCHES_PER_EPOCH=${PATCHES_PER_EPOCH}"
  echo "VAL_PATCHES=${VAL_PATCHES}"
  echo "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
  echo "EARLY_STOPPING_START_EPOCH=${EARLY_STOPPING_START_EPOCH}"
  echo "EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA}"
  echo "LR_PATIENCE=${LR_PATIENCE}"
  echo "LR_FACTOR=${LR_FACTOR}"
  echo "LR_MIN=${LR_MIN}"
  echo "INFER_BATCH=${INFER_BATCH}"
  echo "STRIDE=${STRIDE}"
  echo "TRAIN_START=${TRAIN_START}"
  echo "TRAIN_END=${TRAIN_END}"
  echo "VAL_START=${VAL_START}"
  echo "VAL_END=${VAL_END}"
  echo "TEST_START=${TEST_START}"
  echo "TEST_END=${TEST_END}"
  echo "TIMES_FILE=${TIMES_FILE}"
  printf "TIMES=%s\n" "${TIMES[*]}"
} > "${OUTDIR}/run_config.env"

train_one() {
  local model="$1"
  local seed="$2"
  local suffix="S${seed}"
  echo "=== TRAIN model=${model} seed=${seed} ==="
  "${PYTHON_BIN}" scripts/run_ablation_tiles.py \
    --models "${model}" \
    --seed "${seed}" \
    --experiment-suffix "${suffix}" \
    --seq-len "${SEQ_LEN}" \
    --patch-size "${PATCH_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --patches-per-epoch "${PATCHES_PER_EPOCH}" \
    --val-patches "${VAL_PATCHES}" \
    --sampler uhi_proxy \
    --temporal-sampler p95 \
    --no-temporal-season-balance \
    --epochs "${EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --lr-patience "${LR_PATIENCE}" \
    --lr-factor "${LR_FACTOR}" \
    --lr-min "${LR_MIN}" \
    --split-mode time \
    --train-start "${TRAIN_START}" \
    --train-end "${TRAIN_END}" \
    --val-start "${VAL_START}" \
    --val-end "${VAL_END}" \
    --test-start "${TEST_START}" \
    --test-end "${TEST_END}" \
    --no-post-inference
}

to_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

infer_one_trained() {
  local model="$1"
  local seed="$2"
  local t="$3"
  local model_type=""
  local ckpt=""
  local model_u
  model_u="$(to_upper "${model}")"
  local suffix="S${seed}"
  case "${model}" in
    unet)
      model_type="unet"
      ;;
    lstm)
      model_type="convlstm"
      ;;
    transformer)
      model_type="transformer"
      ;;
    mamba)
      model_type="mamba"
      ;;
    *)
      echo "Unknown model key: ${model}" >&2
      return 1
      ;;
  esac
  ckpt="experiments/models/Tiles_${model_u}_${suffix}_best.h5"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint (skip): ${ckpt}" >&2
    return 0
  fi
  local tag
  tag="$(echo "${t}" | tr ':-T' '_')"
  local exp="PUB_${model_u}_${suffix}_${tag}"
  "${PYTHON_BIN}" scripts/run_inference_tiles_fullframe.py \
    --model-type "${model_type}" \
    --model-path "${ckpt}" \
    --patch-size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    --batch-size "${INFER_BATCH}" \
    --time "${t}" \
    --use-last \
    --lr-resample nearest \
    --scale-from hr_lr \
    --experiment-name "${exp}" \
    --out "${FIG_DIR}/tiles_publish.png"
}

infer_one_baseline() {
  local baseline="$1"
  local t="$2"
  local tag
  tag="$(echo "${t}" | tr ':-T' '_')"
  local baseline_u
  baseline_u="$(to_upper "${baseline}")"
  local exp="PUB_${baseline_u}_${tag}"
  "${PYTHON_BIN}" scripts/run_inference_tiles_fullframe.py \
    --model-type "${baseline}" \
    --patch-size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    --batch-size "${INFER_BATCH}" \
    --time "${t}" \
    --use-last \
    --lr-resample nearest \
    --scale-from hr_lr \
    --experiment-name "${exp}" \
    --out "${FIG_DIR}/tiles_publish.png"
}

for seed in "${SEEDS_ARR[@]}"; do
  for model in "${MODELS_ARR[@]}"; do
    train_one "${model}" "${seed}"
  done
done

for seed in "${SEEDS_ARR[@]}"; do
  for model in "${MODELS_ARR[@]}"; do
    for t in "${TIMES[@]}"; do
      infer_one_trained "${model}" "${seed}" "${t}"
    done
  done
done

if [[ "${RUN_BASELINES}" == "1" ]]; then
  for t in "${TIMES[@]}"; do
    infer_one_baseline baseline_nearest "${t}"
    infer_one_baseline baseline_bilinear "${t}"
  done
fi

RAW_SUMMARY="${OUTDIR}/metrics_raw.csv"
echo "experiment,model,mae,rmse,ssim,patch_size,stride,seq_len,temporal_stride,time_index,time,scale_from,pct_low,pct_high,vmin,vmax" > "${RAW_SUMMARY}"
while IFS= read -r metrics_file; do
  tail -n +2 "${metrics_file}" >> "${RAW_SUMMARY}"
done < <(find "${FIG_DIR}" -name "tiles_publish_PUB_*_metrics.csv" | sort)

"${PYTHON_BIN}" - <<'PY' "${RAW_SUMMARY}" "${OUTDIR}"
import csv
import os
import statistics
import sys

raw_path = sys.argv[1]
out_dir = sys.argv[2]
agg_path = os.path.join(out_dir, "metrics_aggregate.csv")

def _label(model):
    mapping = {
        "convlstm": "lstm",
        "baseline_nearest": "baseline_nearest",
        "baseline_bilinear": "baseline_bilinear",
        "unet": "unet",
        "transformer": "transformer",
        "mamba": "mamba",
    }
    return mapping.get(model, model)

rows = []
with open(raw_path, newline="", encoding="utf-8") as f:
    for rec in csv.DictReader(f):
        try:
            rows.append(
                {
                    "model": _label(rec["model"]),
                    "mae": float(rec["mae"]),
                    "rmse": float(rec["rmse"]),
                    "ssim": float(rec["ssim"]) if rec.get("ssim", "") else float("nan"),
                }
            )
        except Exception:
            continue

groups = {}
for rec in rows:
    groups.setdefault(rec["model"], []).append(rec)

with open(agg_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "n", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "ssim_mean", "ssim_std"])
    out = []
    for model, vals in groups.items():
        maes = [v["mae"] for v in vals]
        rmses = [v["rmse"] for v in vals]
        ssims = [v["ssim"] for v in vals]
        n = len(vals)
        mae_mean = statistics.fmean(maes)
        rmse_mean = statistics.fmean(rmses)
        ssim_mean = statistics.fmean(ssims)
        mae_std = statistics.stdev(maes) if n > 1 else 0.0
        rmse_std = statistics.stdev(rmses) if n > 1 else 0.0
        ssim_std = statistics.stdev(ssims) if n > 1 else 0.0
        out.append((model, n, mae_mean, mae_std, rmse_mean, rmse_std, ssim_mean, ssim_std))
    out.sort(key=lambda x: x[4])  # by rmse_mean asc
    for row in out:
        w.writerow(row)

print(f"Aggregate metrics: {agg_path}")
PY

"${PYTHON_BIN}" - <<'PY' "${OUTDIR}" "${MODELS}" "${SEEDS}"
import csv
import os
import sys

out_dir = sys.argv[1]
models = sys.argv[2].split()
seeds = sys.argv[3].split()

train_path = os.path.join(out_dir, "training_summary.csv")
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
            log_path = os.path.join("experiments", "logs", f"Tiles_{model_u}_S{seed}_log.csv")
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

agg_path = os.path.join(out_dir, "metrics_aggregate.csv")
report_path = os.path.join(out_dir, "report_publish.md")
if os.path.exists(agg_path):
    with open(agg_path, newline="", encoding="utf-8") as af:
        agg = list(csv.DictReader(af))
else:
    agg = []

def _to_float(v, default=float("nan")):
    try:
        return float(v)
    except Exception:
        return default

agg.sort(key=lambda x: _to_float(x.get("rmse_mean")))
base = next((r for r in agg if r.get("model") == "baseline_nearest"), None)

with open(report_path, "w", encoding="utf-8") as rf:
    rf.write("# Heatwave Ablation (Publish Run)\n\n")
    rf.write(f"- Models: {' '.join(models)}\n")
    rf.write(f"- Seeds: {' '.join(seeds)}\n")
    rf.write("- Evaluation: full-frame, fixed color scale source `hr_lr`\n")
    rf.write("\n## Aggregate Metrics\n\n")
    rf.write("| model | n | mae_mean | mae_std | rmse_mean | rmse_std | ssim_mean | ssim_std |\n")
    rf.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in agg:
        rf.write(
            f"| {r.get('model','')} | {r.get('n','')} | {r.get('mae_mean','')} | {r.get('mae_std','')} | "
            f"{r.get('rmse_mean','')} | {r.get('rmse_std','')} | {r.get('ssim_mean','')} | {r.get('ssim_std','')} |\n"
        )

    if base is not None:
        b_mae = _to_float(base.get("mae_mean"))
        b_rmse = _to_float(base.get("rmse_mean"))
        b_ssim = _to_float(base.get("ssim_mean"))
        rf.write("\n## Delta vs baseline_nearest\n\n")
        rf.write("| model | delta_mae | delta_rmse | delta_ssim |\n")
        rf.write("|---|---:|---:|---:|\n")
        for r in agg:
            mae = _to_float(r.get("mae_mean"))
            rmse = _to_float(r.get("rmse_mean"))
            ssim = _to_float(r.get("ssim_mean"))
            rf.write(f"| {r.get('model','')} | {b_mae - mae:.6f} | {b_rmse - rmse:.6f} | {ssim - b_ssim:.6f} |\n")

print(f"Training summary: {train_path}")
print(f"Publish report: {report_path}")
PY

"${PYTHON_BIN}" scripts/consolidate_experiment1.py \
  --out-dir "${OUTDIR}" \
  --bootstrap-samples 2000 \
  --bootstrap-seed 42 \
  --alpha 0.05

echo "Training finished."
echo "Raw metrics: ${RAW_SUMMARY}"
echo "Aggregate: ${OUTDIR}/metrics_aggregate.csv"
echo "Aggregate CI: ${OUTDIR}/metrics_aggregate_ci.csv"
echo "By seed: ${OUTDIR}/metrics_by_model_seed.csv"
echo "Delta CI: ${OUTDIR}/metrics_delta_vs_baseline_ci.csv"
echo "Training summary: ${OUTDIR}/training_summary.csv"
echo "Publish report: ${OUTDIR}/report_publish.md"
echo "Experiment 1 report: ${OUTDIR}/report_experiment1.md"
echo "Figures dir: ${FIG_DIR}"
