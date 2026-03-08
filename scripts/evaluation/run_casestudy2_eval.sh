#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

CASE2_DIR="${CASE2_DIR:-}"
EXP3_DIR="${EXP3_DIR:-}"
CS1_AGG_CSV="${CS1_AGG_CSV:-}"

if [[ -z "${CASE2_DIR}" ]]; then
  echo "Missing CASE2_DIR (explicit case study 2 directory)." >&2
  exit 1
fi
if [[ -z "${EXP3_DIR}" ]]; then
  echo "Missing EXP3_DIR (explicit experiment 3 directory)." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/evaluation/consolidate_casestudy2.py \
  --project-root . \
  --case2-dir "${CASE2_DIR}" \
  --exp3-dir "${EXP3_DIR}" \
  ${CS1_AGG_CSV:+--cs1-agg-csv "${CS1_AGG_CSV}"}
