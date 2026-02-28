#!/usr/bin/env bash
set -euo pipefail

# Evaluate full-frame reconstructions from tile-trained models on a fixed
# list of P95 timestamps. Keeps LR panel pixelated (nearest).

cd "$(dirname "$0")/.."

if [[ "${UNDER_CAFFEINATE:-0}" != "1" ]]; then
  exec caffeinate -dimsu env UNDER_CAFFEINATE=1 bash "$0" "$@"
fi

export SAVE_MODEL_DIAGRAM="${SAVE_MODEL_DIAGRAM:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
export SAVE_COMPARATIVE_HISTORY="${SAVE_COMPARATIVE_HISTORY:-0}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ml_m4/bin/python}"

PATCH_SIZE="${PATCH_SIZE:-96}"
STRIDE="${STRIDE:-48}"
INFER_BATCH="${INFER_BATCH:-8}"

OUTDIR="${OUTDIR:-experiments/figures/p95_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTDIR}"

# Optional: provide a newline-separated file with ISO timestamps to evaluate.
# Example: TIMES_FILE="experiments/heatwaves/aemet/event_times_2017.txt" ./scripts/run_p95_eval_caffeinate.sh
TIMES_FILE="${TIMES_FILE:-}"

# Default timestamps (fallback)
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
  while IFS= read -r line || [[ -n "${line}" ]]; do
    # trim leading/trailing spaces (bash 3 compatible)
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    TIMES+=("${line}")
  done < "${TIMES_FILE}"
  if [[ "${#TIMES[@]}" -eq 0 ]]; then
    echo "TIMES_FILE provided but no timestamps were parsed: ${TIMES_FILE}" >&2
    exit 1
  fi
  echo "Loaded ${#TIMES[@]} timestamps from ${TIMES_FILE}"
fi

eval_one() {
  local model_key="$1"
  local model_type model_path
  case "${model_key}" in
    UNET)
      model_type="unet"
      model_path="experiments/models/Tiles_UNET_best.h5"
      ;;
    LSTM)
      model_type="convlstm"
      model_path="experiments/models/Tiles_LSTM_best.h5"
      ;;
    TRANSFORMER)
      model_type="transformer"
      model_path="experiments/models/Tiles_TRANSFORMER_best.h5"
      ;;
    MAMBA)
      model_type="mamba"
      model_path="experiments/models/Tiles_MAMBA_best.h5"
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${model_path}" ]]; then
    echo "Missing model checkpoint: ${model_path}" >&2
    return 1
  fi

  for t in "${TIMES[@]}"; do
    local tag exp
    tag="$(echo "${t}" | tr ":-T" "_")"
    exp="Tiles_${model_key}_${tag}"
    "${PYTHON_BIN}" scripts/inference/run_inference_tiles_fullframe.py \
      --model-type "${model_type}" \
      --model-path "${model_path}" \
      --patch-size "${PATCH_SIZE}" \
      --stride "${STRIDE}" \
      --batch-size "${INFER_BATCH}" \
      --time "${t}" \
      --use-last \
      --lr-resample nearest \
      --experiment-name "${exp}" \
      --out "${OUTDIR}/${exp}.png"
  done
}

eval_one UNET
eval_one LSTM
eval_one TRANSFORMER
eval_one MAMBA

echo "experiment,model,mae,rmse,ssim,patch_size,stride,seq_len,temporal_stride,time_index,time" > "${OUTDIR}/summary.csv"
find "${OUTDIR}" -name "*_metrics.csv" -print0 | xargs -0 -I{} tail -n +2 "{}" >> "${OUTDIR}/summary.csv"

echo "Evaluation complete."
echo "Summary: ${OUTDIR}/summary.csv"
