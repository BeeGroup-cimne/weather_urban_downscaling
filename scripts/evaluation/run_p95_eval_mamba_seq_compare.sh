#!/usr/bin/env bash
set -euo pipefail

# Compare MAMBA performance for SEQ_LEN=6 vs SEQ_LEN=12 on the same fixed P95 timestamps.
# Produces per-timestamp figures + a CSV summary + an aggregate table (mean/day/night).

cd "$(dirname "$0")/../.."

export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ml_m4/bin/python}"

PATCH_SIZE="${PATCH_SIZE:-96}"
STRIDE="${STRIDE:-48}"
INFER_BATCH="${INFER_BATCH:-8}"

OUTDIR="${OUTDIR:-experiments/figures/p95_mamba_seq_compare_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTDIR}"
export OUTDIR

# Optional: provide a newline-separated file with ISO timestamps to evaluate.
TIMES_FILE="${TIMES_FILE:-}"

# Default timestamps (fallback) – same as run_p95_eval_caffeinate.sh
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

resolve_ckpt() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    echo "${p}"
    return 0
  fi
  return 1
}

SEQ6_CKPT="${SEQ6_CKPT:-}"
SEQ12_CKPT="${SEQ12_CKPT:-}"

if [[ -z "${SEQ6_CKPT}" ]]; then
  SEQ6_CKPT="experiments/models/Ablation_MAMBA_Legacy_S42_best.h5"
fi

if [[ -z "${SEQ12_CKPT}" ]]; then
  if [[ -f "experiments/models/Ablation_MAMBA_Legacy_S42_SEQ12_best.h5" ]]; then
    SEQ12_CKPT="experiments/models/Ablation_MAMBA_Legacy_S42_SEQ12_best.h5"
  else
    SEQ12_CKPT="experiments/experiments/models/Ablation_MAMBA_Legacy_S42_SEQ12_best.h5"
  fi
fi

if ! resolve_ckpt "${SEQ6_CKPT}" >/dev/null; then
  echo "Missing SEQ6 checkpoint: ${SEQ6_CKPT}" >&2
  exit 1
fi
if ! resolve_ckpt "${SEQ12_CKPT}" >/dev/null; then
  echo "Missing SEQ12 checkpoint: ${SEQ12_CKPT}" >&2
  exit 1
fi

run_one() {
  local seq_len="$1"
  local ckpt="$2"
  local label="$3"

  echo "=== P95 eval: ${label} (SEQ_LEN=${seq_len}) ==="
  echo "checkpoint=${ckpt}"

  for t in "${TIMES[@]}"; do
    local tag exp
    tag="$(echo "${t}" | tr ":-T" "_")"
    exp="P95_${label}_${tag}"
    "${PYTHON_BIN}" scripts/inference/run_inference_tiles_fullframe.py \
      --model-type "mamba" \
      --model-path "${ckpt}" \
      --seq-len "${seq_len}" \
      --patch-size "${PATCH_SIZE}" \
      --stride "${STRIDE}" \
      --batch-size "${INFER_BATCH}" \
      --time "${t}" \
      --use-last \
      --lr-resample nearest \
      --scale-from hr_lr \
      --experiment-name "${exp}" \
      --out "${OUTDIR}/${exp}.png"
  done
}

run_one 6  "${SEQ6_CKPT}"  "MAMBA_S42_SEQ6"
run_one 12 "${SEQ12_CKPT}" "MAMBA_S42_SEQ12"

echo "experiment,model,mae,rmse,ssim,patch_size,stride,seq_len,temporal_stride,time_index,time,scale_from,pct_low,pct_high,vmin,vmax" > "${OUTDIR}/summary.csv"
find "${OUTDIR}" -name "*_metrics.csv" -print0 | xargs -0 -I{} bash -c 'tail -n +2 "$1"' _ {} >> "${OUTDIR}/summary.csv"

{
  echo "variant,seq_len,checkpoint,sha256"
  echo "mamba_seq6,6,${SEQ6_CKPT},$(shasum -a 256 "${SEQ6_CKPT}" | awk '{print $1}')"
  echo "mamba_seq12,12,${SEQ12_CKPT},$(shasum -a 256 "${SEQ12_CKPT}" | awk '{print $1}')"
} > "${OUTDIR}/checkpoints_used.csv"

"${PYTHON_BIN}" - <<'PY'
import csv
import os
from collections import defaultdict
from datetime import datetime

outdir = os.environ.get("OUTDIR")
summary = os.path.join(outdir, "summary.csv")
out_csv = os.path.join(outdir, "paper_summary_mamba_seq.csv")

def variant(exp: str) -> str:
    # exp like: P95_MAMBA_S42_SEQ6_2017_...
    if "SEQ12" in exp:
        return "mamba_seq12"
    return "mamba_seq6"

def is_day(iso: str) -> bool:
    # Simple rule for these eval timestamps
    # day = 08:00–19:59, night otherwise
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return 8 <= dt.hour <= 19

agg = defaultdict(lambda: {"n": 0, "mae": 0.0, "rmse": 0.0, "ssim": 0.0, "ssim_n": 0})
agg_day = defaultdict(lambda: {"n": 0, "mae": 0.0, "rmse": 0.0, "ssim": 0.0, "ssim_n": 0})
agg_night = defaultdict(lambda: {"n": 0, "mae": 0.0, "rmse": 0.0, "ssim": 0.0, "ssim_n": 0})

with open(summary, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        v = variant(row["experiment"])
        mae = float(row["mae"])
        rmse = float(row["rmse"])
        ssim = row.get("ssim", "")
        ssim_val = float(ssim) if ssim not in ("", None) else None
        isd = is_day(row["time"])

        for bucket in (agg[v], agg_day[v] if isd else agg_night[v]):
            bucket["n"] += 1
            bucket["mae"] += mae
            bucket["rmse"] += rmse
            if ssim_val is not None:
                bucket["ssim"] += ssim_val
                bucket["ssim_n"] += 1

def row_out(name: str, bucket: dict) -> dict:
    n = max(1, bucket["n"])
    ssim_n = max(1, bucket["ssim_n"])
    return {
        "variant": name,
        "n": bucket["n"],
        "rmse": bucket["rmse"] / n,
        "mae": bucket["mae"] / n,
        "ssim": bucket["ssim"] / ssim_n if bucket["ssim_n"] else "",
    }

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "variant",
            "n",
            "rmse",
            "mae",
            "ssim",
            "rmse_day",
            "mae_day",
            "ssim_day",
            "rmse_night",
            "mae_night",
            "ssim_night",
        ],
    )
    w.writeheader()
    for v in sorted(agg.keys()):
        base = row_out(v, agg[v])
        day = row_out(v, agg_day[v])
        night = row_out(v, agg_night[v])
        w.writerow(
            {
                "variant": base["variant"],
                "n": base["n"],
                "rmse": f'{base["rmse"]:.6f}',
                "mae": f'{base["mae"]:.6f}',
                "ssim": "" if base["ssim"] == "" else f'{base["ssim"]:.6f}',
                "rmse_day": f'{day["rmse"]:.6f}',
                "mae_day": f'{day["mae"]:.6f}',
                "ssim_day": "" if day["ssim"] == "" else f'{day["ssim"]:.6f}',
                "rmse_night": f'{night["rmse"]:.6f}',
                "mae_night": f'{night["mae"]:.6f}',
                "ssim_night": "" if night["ssim"] == "" else f'{night["ssim"]:.6f}',
            }
        )

print(f"✅ Wrote {out_csv}")
PY

echo "✅ Done."
echo "Figures + metrics: ${OUTDIR}"
echo "Summary: ${OUTDIR}/summary.csv"
echo "Aggregate: ${OUTDIR}/paper_summary_mamba_seq.csv"
