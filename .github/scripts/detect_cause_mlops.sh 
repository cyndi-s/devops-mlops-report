#!/usr/bin/env bash
set -euo pipefail

# Optional overrides
ROOT_DIR="${ROOT_DIR:-POC-Image-Classification}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
SCRIPT_PAT="${SCRIPT_PAT:-$ROOT_DIR/(modelSteps/|MLproject$|conda\.yaml$|requirements\.txt$|[^/]+\.py$)}"

BASE_SHA="${GITHUB_EVENT_BEFORE:-${1:-}}"
if [[ -z "$BASE_SHA" ]]; then
  BASE_SHA="$(git rev-parse HEAD^ || true)"
fi

if [[ -z "$BASE_SHA" ]]; then
  echo "MLOPS_CHANGED=true"  >> "$GITHUB_ENV"
  echo "CAUSE_MLOPS=None"    >> "$GITHUB_ENV"
  echo "No base SHA; defaulting to CAUSE_MLOPS=None"
  exit 0
fi

CHANGED="$(git diff --name-only "$BASE_SHA"...HEAD || true)"

DATA_CHANGED=false
SCRIPT_CHANGED=false

# Data change: anything under data/
if echo "$CHANGED" | grep -E "^${DATA_DIR}(/|$)" >/dev/null 2>&1; then
  DATA_CHANGED=true
fi

# Script change: training scripts, MLproject/conda/requirements, or any .py directly under ROOT
if echo "$CHANGED" | grep -E "${SCRIPT_PAT}" >/dev/null 2>&1; then
  SCRIPT_CHANGED=true
fi

if $DATA_CHANGED && $SCRIPT_CHANGED; then
  CAUSE="Both"
elif $DATA_CHANGED; then
  CAUSE="Data"
elif $SCRIPT_CHANGED; then
  CAUSE="Script"
else
  CAUSE="None"
fi

echo "MLOPS_CHANGED=true"   >> "$GITHUB_ENV"
echo "CAUSE_MLOPS=${CAUSE}" >> "$GITHUB_ENV"

echo "Detected MLOps changes:"
echo "$CHANGED" | sed 's/^/ - /'
echo "CAUSE_MLOPS=${CAUSE}"
