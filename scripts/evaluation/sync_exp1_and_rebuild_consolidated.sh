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
#   LOCAL_ROOT=/Users/kerincardona/weather_urban_downscaling

SERVER_HOST="${SERVER_HOST:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/hdd/weather_urban_downscaling}"
PUBLISH_RUN="${PUBLISH_RUN:-publish_run_20260220_220458}"
LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"

if [[ -z "${SERVER_HOST}" ]]; then
  echo "Missing SERVER_HOST. Example:"
  echo "  SERVER_HOST=ubuntu@kerin bash scripts/evaluation/sync_exp1_and_rebuild_consolidated.sh"
  exit 1
fi

cd "${LOCAL_ROOT}"

REMOTE_DIR="${REMOTE_ROOT}/experiments/heatwaves/${PUBLISH_RUN}/"
LOCAL_DIR="experiments/heatwaves/${PUBLISH_RUN}/"

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
python3 scripts/evaluation/consolidate_all_results.py --project-root .

echo "Done."
