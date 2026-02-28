#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Needed on TF 2.16+/Keras 3 for legacy models in this repo.
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

# Checkpoint family:
# - tiles_s42:       Tiles_*_S42_best.h5 (plus optional baselines)
# - ablation_s42:    Ablation_*_Legacy_S42_best.h5
CKPT_SET="${CKPT_SET:-tiles_s42}"

# Default case-study models (5): baseline + 4 learned models.
# Alternative fullframe-like top-5: "unet lstm transformer mamba mamba_seq12"
CASE_MODELS="${CASE_MODELS:-baseline_nearest unet lstm transformer mamba}"
SEED="${SEED:-42}"
SEQ_LEN="${SEQ_LEN:-6}"
PATCH_SIZE="${PATCH_SIZE:-96}"
STRIDE="${STRIDE:-48}"
INFER_BATCH="${INFER_BATCH:-8}"
LR_RESAMPLE="${LR_RESAMPLE:-nearest}"
SCALE_FROM="${SCALE_FROM:-hr_lr}"
OUTDIR="${OUTDIR:-experiments/heatwaves/casestudy1_$(date +%Y%m%d_%H%M%S)}"
TIMES_FILE="${TIMES_FILE:-}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
RANK_BY="${RANK_BY:-rmse}"

mkdir -p "${OUTDIR}/figures"
FIG_DIR="${OUTDIR}/figures"

TIMES=(
  "2017-06-15T15:00:00"
  "2017-06-15T03:00:00"
  "2017-06-16T15:00:00"
  "2017-08-03T15:00:00"
  "2017-08-03T03:00:00"
  "2017-08-05T15:00:00"
)

if [[ -n "${TIMES_FILE}" ]]; then
  if [[ ! -f "${TIMES_FILE}" ]]; then
    echo "TIMES_FILE not found: ${TIMES_FILE}" >&2
    exit 1
  fi
  TIMES=()
  while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    TIMES+=("${line}")
  done < "${TIMES_FILE}"
fi

read -r -a MODELS_ARR <<< "${CASE_MODELS}"

{
  echo "CASE_MODELS=${CASE_MODELS}"
  echo "SEED=${SEED}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "PATCH_SIZE=${PATCH_SIZE}"
  echo "STRIDE=${STRIDE}"
  echo "INFER_BATCH=${INFER_BATCH}"
  echo "LR_RESAMPLE=${LR_RESAMPLE}"
  echo "SCALE_FROM=${SCALE_FROM}"
  echo "POSTPROCESS_ONLY=${POSTPROCESS_ONLY}"
  echo "RANK_BY=${RANK_BY}"
  echo "TIMES_FILE=${TIMES_FILE}"
  printf "TIMES=%s\n" "${TIMES[*]}"
} > "${OUTDIR}/run_config.env"

echo "🚀 Case Study 1 (Heatwave + Urban Adaptation)"
echo "   Models: ${CASE_MODELS}"
echo "   Seed: ${SEED}"
echo "   Times: ${#TIMES[@]}"
echo "   Outdir: ${OUTDIR}"
echo "   Python: ${PYTHON_BIN}"

model_type_for_key() {
  case "$1" in
    unet) echo "unet" ;;
    lstm) echo "convlstm" ;;
    transformer) echo "transformer" ;;
    mamba) echo "mamba" ;;
    mamba_seq12) echo "mamba" ;;
    baseline_nearest) echo "baseline_nearest" ;;
    baseline_bilinear) echo "baseline_bilinear" ;;
    *)
      echo "unknown"
      ;;
  esac
}

checkpoint_for_key() {
  local key="$1"
  if [[ "${CKPT_SET}" == "ablation_s42" ]]; then
    case "${key}" in
      unet) echo "experiments/models/Ablation_UNET_Legacy_S${SEED}_best.h5" ;;
      lstm) echo "experiments/models/Ablation_LSTM_Legacy_S${SEED}_best.h5" ;;
      transformer) echo "experiments/models/Ablation_TRANSFORMER_Legacy_S${SEED}_best.h5" ;;
      mamba) echo "experiments/models/Ablation_MAMBA_Legacy_S${SEED}_best.h5" ;;
      mamba_seq12) echo "experiments/models/Ablation_MAMBA_Legacy_S${SEED}_SEQ12_best.h5" ;;
      *) echo "" ;;
    esac
  else
    case "${key}" in
      unet) echo "experiments/models/Tiles_UNET_S${SEED}_best.h5" ;;
      lstm) echo "experiments/models/Tiles_LSTM_S${SEED}_best.h5" ;;
      transformer) echo "experiments/models/Tiles_TRANSFORMER_S${SEED}_best.h5" ;;
      mamba) echo "experiments/models/Tiles_MAMBA_S${SEED}_best.h5" ;;
      *) echo "" ;;
    esac
  fi
}

run_one() {
  local key="$1"
  local model_type
  local ckpt
  local seq_for_model
  model_type="$(model_type_for_key "${key}")"
  ckpt="$(checkpoint_for_key "${key}")"
  seq_for_model="${SEQ_LEN}"
  if [[ "${key}" == "mamba_seq12" ]]; then
    seq_for_model="${SEQ_LEN_SEQ12:-12}"
  fi

  if [[ "${model_type}" == "unknown" ]]; then
    echo "Unknown model key: ${key}" >&2
    return 1
  fi
  if [[ -n "${ckpt}" && ! -f "${ckpt}" ]]; then
    echo "⚠️ Missing checkpoint, skip: ${ckpt}"
    return 0
  fi

  local key_u
  key_u="$(echo "${key}" | tr '[:lower:]' '[:upper:]')"
  for t in "${TIMES[@]}"; do
    local tag
    local exp
    tag="$(echo "${t}" | tr ':-T' '_')"
    if [[ -n "${ckpt}" ]]; then
      exp="CS1_${key_u}_S${SEED}_${tag}"
    else
      exp="CS1_${key_u}_${tag}"
    fi

    cmd=(
      "${PYTHON_BIN}" scripts/inference/run_inference_tiles_fullframe.py
      --model-type "${model_type}"
      --seq-len "${seq_for_model}"
      --patch-size "${PATCH_SIZE}"
      --stride "${STRIDE}"
      --batch-size "${INFER_BATCH}"
      --time "${t}"
      --use-last
      --lr-resample "${LR_RESAMPLE}"
      --scale-from "${SCALE_FROM}"
      --experiment-name "${exp}"
      --out "${FIG_DIR}/cs1_infer.png"
    )
    if [[ -n "${ckpt}" ]]; then
      cmd+=(--model-path "${ckpt}")
    fi
    "${cmd[@]}"
  done
}

if [[ "${POSTPROCESS_ONLY}" != "1" ]]; then
  for key in "${MODELS_ARR[@]}"; do
    run_one "${key}"
  done
else
  echo "ℹ️ POSTPROCESS_ONLY=1, skipping inference and rebuilding reports from existing metrics."
fi

RAW_CSV="${OUTDIR}/metrics_raw.csv"
echo "experiment,model,mae,rmse,ssim,patch_size,stride,seq_len,temporal_stride,time_index,time,scale_from,pct_low,pct_high,vmin,vmax" > "${RAW_CSV}"
while IFS= read -r f; do
  tail -n +2 "${f}" >> "${RAW_CSV}"
done < <(find "${FIG_DIR}" -name "cs1_infer_CS1_*_metrics.csv" | sort)

"${PYTHON_BIN}" - <<'PY' "${RAW_CSV}" "${OUTDIR}" "${MAMBA_SEQ_COMPARE_CSV:-}" "${RANK_BY}"
import csv
import glob
import os
import statistics
import sys

raw_csv = sys.argv[1]
out_dir = sys.argv[2]
mamba_seq_csv = sys.argv[3] if len(sys.argv) > 3 else ""
rank_by = (sys.argv[4] if len(sys.argv) > 4 else "rmse").strip().lower()
if rank_by not in {"rmse", "mae", "ssim"}:
    rank_by = "rmse"
agg_csv = os.path.join(out_dir, "metrics_aggregate.csv")
report_md = os.path.join(out_dir, "report_casestudy1.md")


def _safe_float(v, default=float("nan")):
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _variant(rec):
    exp = str(rec.get("experiment", "")).upper()
    model = str(rec.get("model", "")).lower()
    seq_len = _safe_int(rec.get("seq_len", "0"), 0)
    if "BASELINE_NEAREST" in exp or model == "baseline_nearest":
        return "baseline_nearest"
    if "BASELINE_BILINEAR" in exp or model == "baseline_bilinear":
        return "baseline_bilinear"
    if "MAMBA_SEQ12" in exp or (model == "mamba" and seq_len >= 12):
        return "mamba_seq12"
    if model == "mamba":
        return "mamba_seq6"
    if model == "convlstm":
        return "lstm"
    if model in {"unet", "transformer"}:
        return model
    return model

rows = []
with open(raw_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            rows.append({
                "variant": _variant(r),
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "ssim": float(r["ssim"]) if r.get("ssim", "") else float("nan"),
            })
        except Exception:
            continue

groups = {}
for r in rows:
    groups.setdefault(r["variant"], []).append(r)

with open(agg_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        f"rank_{rank_by}",
        "variant",
        "n",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "ssim_mean",
        "ssim_std",
    ])
    out = []
    for variant, vals in groups.items():
        n = len(vals)
        maes = [v["mae"] for v in vals]
        rmses = [v["rmse"] for v in vals]
        ssims = [v["ssim"] for v in vals]
        out.append((
            variant,
            n,
            statistics.fmean(maes),
            statistics.stdev(maes) if n > 1 else 0.0,
            statistics.fmean(rmses),
            statistics.stdev(rmses) if n > 1 else 0.0,
            statistics.fmean(ssims),
            statistics.stdev(ssims) if n > 1 else 0.0,
        ))
    if rank_by == "ssim":
        out.sort(key=lambda x: (-x[6], x[4], x[2]))
    elif rank_by == "mae":
        out.sort(key=lambda x: (x[2], x[4], -x[6]))
    else:
        out.sort(key=lambda x: (x[4], x[2], -x[6]))
    for rank, row in enumerate(out, start=1):
        w.writerow([rank, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]])

base = next((r for r in out if r[0] == "baseline_nearest"), None)

if not mamba_seq_csv:
    candidates = sorted(glob.glob(os.path.join("experiments", "figures", "p95_mamba_seq_compare_*", "paper_summary_mamba_seq.csv")))
    if candidates:
        mamba_seq_csv = candidates[-1]

mamba_rows = []
if mamba_seq_csv and os.path.exists(mamba_seq_csv):
    with open(mamba_seq_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mamba_rows.append(r)

with open(report_md, "w", encoding="utf-8") as rf:
    rf.write("# Case Study 1 - Heatwave + Urban Adaptation (S42)\n\n")
    rf.write(f"- Source metrics: `{raw_csv}`\n")
    rf.write(f"- Aggregate metrics: `{agg_csv}`\n")
    if mamba_seq_csv and os.path.exists(mamba_seq_csv):
        rf.write(f"- Mamba seq comparison: `{mamba_seq_csv}`\n")
    rf.write("\n## Aggregate Metrics\n\n")
    if rank_by == "ssim":
        order_msg = "Ordered by `ssim_mean` descending (tie-breakers: `rmse_mean`, then `mae_mean`)."
    elif rank_by == "mae":
        order_msg = "Ordered by `mae_mean` ascending (tie-breakers: `rmse_mean`, then `ssim_mean` descending)."
    else:
        order_msg = "Ordered by `rmse_mean` ascending (tie-breakers: `mae_mean`, then `ssim_mean` descending)."
    rf.write(order_msg + "\n\n")
    rf.write(f"| rank_{rank_by} | variant | n | mae_mean | mae_std | rmse_mean | rmse_std | ssim_mean | ssim_std |\n")
    rf.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for rank, row in enumerate(out, start=1):
        rf.write(
            f"| {rank} | {row[0]} | {row[1]} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.6f} | {row[5]:.6f} | {row[6]:.6f} | {row[7]:.6f} |\n"
        )
    if base is not None:
        b_mae, b_rmse, b_ssim = base[2], base[4], base[6]
        rf.write("\n## Delta vs Baseline Nearest\n\n")
        rf.write("| variant | delta_mae | delta_rmse | delta_ssim |\n")
        rf.write("|---|---:|---:|---:|\n")
        for row in out:
            rf.write(f"| {row[0]} | {b_mae - row[2]:.6f} | {b_rmse - row[4]:.6f} | {row[6] - b_ssim:.6f} |\n")
    if mamba_rows:
        rf.write("\n## Mamba Seq6 vs Seq12 (P95 Comparison)\n\n")
        rf.write("| variant | rmse | mae | ssim | rmse_day | rmse_night |\n")
        rf.write("|---|---:|---:|---:|---:|---:|\n")
        for r in mamba_rows:
            rf.write(
                f"| {r.get('variant','')} | {_safe_float(r.get('rmse','nan')):.6f} | {_safe_float(r.get('mae','nan')):.6f} | "
                f"{_safe_float(r.get('ssim','nan')):.6f} | {_safe_float(r.get('rmse_day','nan')):.6f} | {_safe_float(r.get('rmse_night','nan')):.6f} |\n"
            )

print(f"Saved aggregate: {agg_csv}")
print(f"Saved report: {report_md}")
PY

echo "✅ Case Study 1 updated."
echo "   Raw: ${RAW_CSV}"
echo "   Aggregate: ${OUTDIR}/metrics_aggregate.csv"
echo "   Report: ${OUTDIR}/report_casestudy1.md"
echo "   Figures: ${FIG_DIR}"
