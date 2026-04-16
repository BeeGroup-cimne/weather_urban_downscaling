#!/usr/bin/env bash
set -euo pipefail

# Sync Experiment 1 artifacts from server and rebuild the consolidated report.
#
# Required:
#   SERVER_HOST=ubuntu@kerin
#
# Optional:
#   REMOTE_ROOT=/hdd/weather_urban_downscaling
#   PUBLISH_RUN=publish_run_20260220_220458
#   LOCAL_ROOT=/path/to/weather_urban_downscaling
#   EXP2_DIR=experiments/stations_eval/experiment2_YYYYMMDD_HHMMSS
#   CS1_DIR=experiments/heatwaves/casestudy1_YYYYMMDD_HHMMSS
#   EXP3_DIR=experiments/fullframe/experiment3_YYYYMMDD_HHMMSS

SERVER_HOST="${SERVER_HOST:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/hdd/weather_urban_downscaling}"
PUBLISH_RUN="${PUBLISH_RUN:-publish_run_20260220_220458}"
LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"
EXP2_DIR="${EXP2_DIR:-}"
CS1_DIR="${CS1_DIR:-}"
EXP3_DIR="${EXP3_DIR:-}"

if [[ -z "${SERVER_HOST}" ]]; then
  echo "Missing SERVER_HOST. Example:"
  echo "  SERVER_HOST=ubuntu@kerin bash scripts/evaluation/sync_exp1_and_rebuild_consolidated.sh"
  exit 1
fi
if [[ -z "${EXP2_DIR}" || -z "${CS1_DIR}" || -z "${EXP3_DIR}" ]]; then
  echo "Missing explicit dirs. Set EXP2_DIR, CS1_DIR, EXP3_DIR." >&2
  exit 1
fi

cd "${LOCAL_ROOT}"

REMOTE_DIR="${REMOTE_ROOT}/experiments/heatwaves/${PUBLISH_RUN}/"
LOCAL_DIR="experiments/heatwaves/${PUBLISH_RUN}/"
EXP1_DIR="experiments/heatwaves/${PUBLISH_RUN}"

mkdir -p "${LOCAL_DIR}"

echo "Syncing from ${SERVER_HOST}:${REMOTE_DIR}"
rsync -avz --progress \
  --include='*/' \
  --include='*.csv' \
  --include='*.md' \
  --include='*.env' \
  --include='*.txt' \
  --exclude='*' \
  "${SERVER_HOST}:${REMOTE_DIR}" "${LOCAL_DIR}"

echo "Rebuilding consolidated report..."
python3 scripts/evaluation/consolidate_all_results.py \
  --project-root . \
  --exp2-dir "${EXP2_DIR}" \
  --exp1-dir "${EXP1_DIR}" \
  --cs1-dir "${CS1_DIR}" \
  --exp3-dir "${EXP3_DIR}"

echo "Done."
