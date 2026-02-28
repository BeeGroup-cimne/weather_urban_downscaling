#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Avoid slow/noisy Matplotlib font cache creation in non-writable $HOME locations.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" >/dev/null 2>&1 || true
export TF_USE_LEGACY_KERAS="${TF_USE_LEGACY_KERAS:-1}"

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

STATIONS_GRIB="${STATIONS_GRIB:-}"
STATIONS_OBS_CSV="${STATIONS_OBS_CSV:-}"
STATIONS_CSV="${STATIONS_CSV:-}"
HEATWAVE_TIMES_FILE="${HEATWAVE_TIMES_FILE:-}"
MODELS="${MODELS:-unet lstm transformer mamba mamba_seq12 baseline_bilinear baseline_nearest}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
STRIDE="${STRIDE:-1}"
DAY_START_HOUR="${DAY_START_HOUR:-8}"
DAY_END_HOUR="${DAY_END_HOUR:-19}"
TIME_OFFSET_HOURS="${TIME_OFFSET_HOURS:-0.0}"
BIAS_CORRECTION="${BIAS_CORRECTION:-}"
OUTDIR="${OUTDIR:-experiments/stations_eval/ablation_$(date +%Y%m%d_%H%M%S)}"

CKPT_UNET="${CKPT_UNET:-}"
CKPT_LSTM="${CKPT_LSTM:-}"
CKPT_TRANSFORMER="${CKPT_TRANSFORMER:-}"
CKPT_MAMBA="${CKPT_MAMBA:-}"
CKPT_MAMBA_SEQ12="${CKPT_MAMBA_SEQ12:-}"

if [[ $# -gt 0 && -z "${STATIONS_GRIB}" && -z "${STATIONS_OBS_CSV}" ]]; then
  case "$1" in
    *.csv) STATIONS_OBS_CSV="$1" ;;
    *) STATIONS_GRIB="$1" ;;
  esac
  shift
fi

if [[ -z "${STATIONS_GRIB}" && -z "${STATIONS_OBS_CSV}" ]]; then
  echo "Usage (GRIB): STATIONS_GRIB=/path/stations.grib scripts/evaluation/run_stations_eval_ablation.sh" >&2
  echo "Usage (CSV):  STATIONS_OBS_CSV=/path/stations_obs.csv scripts/evaluation/run_stations_eval_ablation.sh" >&2
  echo "   or: scripts/evaluation/run_stations_eval_ablation.sh /path/stations.grib" >&2
  echo "   or: scripts/evaluation/run_stations_eval_ablation.sh /path/stations_obs.csv" >&2
  exit 1
fi

if [[ -n "${STATIONS_GRIB}" && ! -f "${STATIONS_GRIB}" ]]; then
  echo "Stations GRIB not found: ${STATIONS_GRIB}" >&2
  exit 1
fi

if [[ -n "${STATIONS_OBS_CSV}" && ! -f "${STATIONS_OBS_CSV}" ]]; then
  echo "Stations observations CSV not found: ${STATIONS_OBS_CSV}" >&2
  exit 1
fi

if [[ -n "${STATIONS_CSV}" && ! -f "${STATIONS_CSV}" ]]; then
  echo "stations CSV not found: ${STATIONS_CSV}" >&2
  exit 1
fi

if [[ -n "${STATIONS_OBS_CSV}" && -n "${STATIONS_CSV}" ]]; then
  echo "ℹ️ STATIONS_CSV ignored when STATIONS_OBS_CSV is provided."
fi

if [[ -n "${HEATWAVE_TIMES_FILE}" && ! -f "${HEATWAVE_TIMES_FILE}" ]]; then
  echo "heatwave times file not found: ${HEATWAVE_TIMES_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"
CHECKPOINTS_CSV="${OUTDIR}/checkpoints_used.csv"
echo "model,model_type,checkpoint,status" > "${CHECKPOINTS_CSV}"

read -r -a MODELS_ARR <<< "${MODELS}"

resolve_ckpt() {
  local model="$1"
  local configured="$2"
  shift 2
  if [[ -n "${configured}" ]]; then
    echo "${configured}"
    return
  fi
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done
  echo "$1"
}

run_one() {
  local model="$1"
  local model_type=""
  local ckpt=""
  local seq_len="6"

  case "${model}" in
    unet)
      model_type="unet"
      ckpt="$(resolve_ckpt "${model}" "${CKPT_UNET}" \
        "experiments/models/Ablation_UNET_Legacy_S${SEED}_best.h5" \
        "experiments/models/Ablation_UNET_Legacy_best.h5" \
        "experiments/models/Ablation_UNet_Legacy_best.h5" \
        "experiments/models/Tiles_UNET_S${SEED}_best.h5" \
        "experiments/models/UNet_best.h5")"
      ;;
    lstm)
      model_type="convlstm"
      ckpt="$(resolve_ckpt "${model}" "${CKPT_LSTM}" \
        "experiments/models/Ablation_LSTM_Legacy_S${SEED}_best.h5" \
        "experiments/models/Ablation_LSTM_Legacy_best.h5" \
        "experiments/models/Tiles_LSTM_S${SEED}_best.h5" \
        "experiments/models/ConvLSTM_best.h5")"
      ;;
    transformer)
      model_type="transformer"
      ckpt="$(resolve_ckpt "${model}" "${CKPT_TRANSFORMER}" \
        "experiments/models/Ablation_TRANSFORMER_Legacy_S${SEED}_best.h5" \
        "experiments/models/Ablation_TRANSFORMER_Legacy_best.h5" \
        "experiments/models/Tiles_TRANSFORMER_S${SEED}_best.h5" \
        "experiments/models/Transformer_best.h5")"
      ;;
    mamba)
      model_type="mamba"
      ckpt="$(resolve_ckpt "${model}" "${CKPT_MAMBA}" \
        "experiments/models/Ablation_MAMBA_Legacy_S${SEED}_best.h5" \
        "experiments/models/Ablation_MAMBA_Legacy_best.h5" \
        "experiments/models/Tiles_MAMBA_S${SEED}_best.h5" \
        "experiments/models/Mamba_best.h5")"
      ;;
    mamba_seq12)
      model_type="mamba"
      seq_len="12"
      ckpt="$(resolve_ckpt "${model}" "${CKPT_MAMBA_SEQ12}" \
        "experiments/models/Ablation_MAMBA_Legacy_S${SEED}_SEQ12_best.h5")"
      ;;
    baseline_bilinear)
      model_type="baseline_bilinear"
      ckpt="__baseline__"
      ;;
    baseline_nearest)
      model_type="baseline_nearest"
      ckpt="__baseline__"
      ;;
    *)
      echo "Unknown model key: ${model}" >&2
      return 1
      ;;
  esac

  if [[ "${ckpt}" != "__baseline__" && ! -f "${ckpt}" ]]; then
    echo "⚠️ Missing checkpoint for ${model}: ${ckpt}"
    echo "${model},${model_type},${ckpt},missing" >> "${CHECKPOINTS_CSV}"
    return 0
  fi

  local model_out="${OUTDIR}/${model}"
  mkdir -p "${model_out}"

  echo "=== Stations eval: model=${model} type=${model_type} ==="
  echo "   checkpoint=${ckpt}"
  echo "   seq_len=${seq_len}"

  local cmd=(
    "${PYTHON_BIN}" scripts/evaluation/evaluate_stations_grib.py
    --model-type "${model_type}"
    --seq-len "${seq_len}"
    --split "${SPLIT}"
    --max-samples "${MAX_SAMPLES}"
    --stride "${STRIDE}"
    --out-dir "${model_out}"
  )
  if [[ "${ckpt}" == "__baseline__" ]]; then
    cmd+=(--baseline)
  else
    cmd+=(--model-path "${ckpt}")
  fi
  if [[ -n "${STATIONS_GRIB}" ]]; then
    cmd+=(--stations-grib "${STATIONS_GRIB}")
  fi
  if [[ -n "${STATIONS_OBS_CSV}" ]]; then
    cmd+=(--stations-obs-csv "${STATIONS_OBS_CSV}")
  fi
  if [[ -n "${STATIONS_CSV}" ]]; then
    cmd+=(--stations-csv "${STATIONS_CSV}")
  fi
  if [[ -n "${HEATWAVE_TIMES_FILE}" ]]; then
    cmd+=(--heatwave-times-file "${HEATWAVE_TIMES_FILE}")
  fi
  cmd+=(
    --day-start-hour "${DAY_START_HOUR}"
    --day-end-hour "${DAY_END_HOUR}"
    --time-offset-hours "${TIME_OFFSET_HOURS}"
  )
  if [[ -n "${BIAS_CORRECTION}" ]]; then
    cmd+=(--bias-correction)
  fi

  if "${cmd[@]}"; then
    echo "${model},${model_type},${ckpt},ok" >> "${CHECKPOINTS_CSV}"
    return 0
  fi

  echo "${model},${model_type},${ckpt},failed" >> "${CHECKPOINTS_CSV}"
  return 0
}

ok_runs=0
for model in "${MODELS_ARR[@]}"; do
  run_one "${model}"
done

while IFS= read -r line; do
  [[ "${line}" == "model,model_type,checkpoint,status" ]] && continue
  status="${line##*,}"
  if [[ "${status}" == "ok" ]]; then
    ok_runs=$((ok_runs + 1))
  fi
done < "${CHECKPOINTS_CSV}"

if [[ "${ok_runs}" -eq 0 ]]; then
  echo "❌ No model was evaluated successfully." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY' "${OUTDIR}" "${CHECKPOINTS_CSV}" "${MODELS}"
import csv
import os
import sys

out_dir = sys.argv[1]
checkpoints_csv = sys.argv[2]
models = sys.argv[3].split()

ckpt_map = {}
with open(checkpoints_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ckpt_map[row["model"]] = row

summary_out = os.path.join(out_dir, "stations_eval_models_summary.csv")
per_station_out = os.path.join(out_dir, "stations_eval_per_station_all_models.csv")
rank_out = os.path.join(out_dir, "stations_eval_rank_by_segment.csv")
report_out = os.path.join(out_dir, "report_experiment2.md")

summary_rows = []
per_station_rows = []

for model in models:
    summary_path = os.path.join(out_dir, model, "stations_eval_summary_by_segment.csv")
    fallback_summary = os.path.join(out_dir, model, "stations_eval_summary.csv")
    per_station_path = os.path.join(out_dir, model, "stations_eval_per_station.csv")
    per_station_seg_path = os.path.join(out_dir, model, "stations_eval_per_station_by_segment.csv")

    meta = ckpt_map.get(model, {})
    model_type = meta.get("model_type", "")
    checkpoint = meta.get("checkpoint", "")
    status = meta.get("status", "missing")

    if not os.path.exists(summary_path):
        summary_path = fallback_summary

    if os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as sf:
            rows = list(csv.DictReader(sf))
        for rec in rows:
            rec = dict(rec)
            rec["model"] = model
            rec["model_type"] = model_type
            rec["checkpoint"] = checkpoint
            rec["status"] = status
            rec["segment"] = rec.get("segment", "all")
            summary_rows.append(rec)

    if os.path.exists(per_station_seg_path):
        per_station_path = per_station_seg_path

    if os.path.exists(per_station_path):
        with open(per_station_path, newline="", encoding="utf-8") as pf:
            for rec in csv.DictReader(pf):
                rec = dict(rec)
                rec["model"] = model
                rec["model_type"] = model_type
                rec["checkpoint"] = checkpoint
                rec["segment"] = rec.get("segment", "all")
                per_station_rows.append(rec)

if summary_rows:
    summary_fields = [
        "model", "model_type", "status", "checkpoint",
        "segment",
        "split", "MAE_model", "RMSE_model", "Bias_model", "Corr_model",
        "MAE_urbclim", "RMSE_urbclim", "Bias_urbclim", "Corr_urbclim",
        "N", "samples_used"
    ]
    with open(summary_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in summary_fields})

if per_station_rows:
    station_fields = [
        "model", "model_type", "checkpoint", "segment", "station_id",
        "MAE_model", "RMSE_model", "Bias_model", "Corr_model",
        "MAE_urbclim", "RMSE_urbclim", "Bias_urbclim", "Corr_urbclim", "N"
    ]
    with open(per_station_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=station_fields)
        w.writeheader()
        for row in per_station_rows:
            w.writerow({k: row.get(k, "") for k in station_fields})

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

rank_rows = []
for row in summary_rows:
    mae_m = _to_float(row.get("MAE_model"))
    rmse_m = _to_float(row.get("RMSE_model"))
    ssim_like = _to_float(row.get("Corr_model"))
    mae_u = _to_float(row.get("MAE_urbclim"))
    rmse_u = _to_float(row.get("RMSE_urbclim"))
    corr_u = _to_float(row.get("Corr_urbclim"))
    rank_rows.append({
        "segment": row.get("segment", "all"),
        "model": row.get("model", ""),
        "N": row.get("N", ""),
        "MAE_model": mae_m,
        "RMSE_model": rmse_m,
        "Corr_model": ssim_like,
        "MAE_urbclim": mae_u,
        "RMSE_urbclim": rmse_u,
        "Corr_urbclim": corr_u,
        "delta_MAE_vs_urbclim": mae_u - mae_m,
        "delta_RMSE_vs_urbclim": rmse_u - rmse_m,
        "delta_Corr_vs_urbclim": ssim_like - corr_u,
    })

rank_rows.sort(key=lambda x: (x["segment"], x["MAE_model"]))
if rank_rows:
    with open(rank_out, "w", newline="", encoding="utf-8") as f:
        fields = [
            "segment", "model", "N",
            "MAE_model", "RMSE_model", "Corr_model",
            "MAE_urbclim", "RMSE_urbclim", "Corr_urbclim",
            "delta_MAE_vs_urbclim", "delta_RMSE_vs_urbclim", "delta_Corr_vs_urbclim",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rank_rows:
            w.writerow(row)

with open(report_out, "w", encoding="utf-8") as rf:
    rf.write("# Experimento 2: validación externa en estaciones\n\n")
    rf.write(f"- Modelos evaluados: {' '.join(models)}\n")
    rf.write("- Métrica de comparación externa: estaciones reales vs modelo y UrbClim\n\n")
    rf.write("## Ranking por segmento (menor MAE es mejor)\n\n")
    if rank_rows:
        rf.write("| segment | model | N | MAE_model | RMSE_model | Corr_model | ΔMAE vs UrbClim | ΔRMSE vs UrbClim | ΔCorr vs UrbClim |\n")
        rf.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rank_rows:
            rf.write(
                f"| {row['segment']} | {row['model']} | {row['N']} | "
                f"{row['MAE_model']:.6f} | {row['RMSE_model']:.6f} | {row['Corr_model']:.6f} | "
                f"{row['delta_MAE_vs_urbclim']:.6f} | {row['delta_RMSE_vs_urbclim']:.6f} | {row['delta_Corr_vs_urbclim']:.6f} |\n"
            )
    else:
        rf.write("No hay filas válidas para ranking.\n")

print(f"Combined summary: {summary_out}")
print(f"Combined per-station: {per_station_out}")
print(f"Rank by segment: {rank_out}")
print(f"Experiment 2 report: {report_out}")
PY

echo "✅ Stations evaluation completed."
echo "   Output dir: ${OUTDIR}"
echo "   Checkpoints: ${CHECKPOINTS_CSV}"
echo "   Summary: ${OUTDIR}/stations_eval_models_summary.csv"
echo "   Per-station: ${OUTDIR}/stations_eval_per_station_all_models.csv"
echo "   Rank by segment: ${OUTDIR}/stations_eval_rank_by_segment.csv"
echo "   Experiment 2 report: ${OUTDIR}/report_experiment2.md"
