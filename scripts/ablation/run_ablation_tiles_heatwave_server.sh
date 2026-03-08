#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export ENABLE_CAFFEINATE="${ENABLE_CAFFEINATE:-0}"
export USE_GPU_CONFIG="${USE_GPU_CONFIG:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

exec bash scripts/ablation/run_ablation_tiles_heatwave_caffeinate.sh "$@"
