#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
WATCHDOG=${WATCHDOG:-true}
OCCUPANCY=${OCCUPANCY:-true}
PYTHON=${PYTHON:-python}
RESTORED=0

restore_occupancy() {
  local code=$?
  trap - EXIT INT TERM HUP
  if [[ "$RESTORED" -eq 0 ]]; then
    bash "$OCCUPANCY" start || true
    bash "$WATCHDOG" start || true
    RESTORED=1
  fi
  exit "$code"
}

trap restore_occupancy EXIT INT TERM HUP

mkdir -p "$OUTPUT_ROOT/_operations"
bash "$WATCHDOG" stop
bash "$OCCUPANCY" stop

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PYTHON" -m experiments.sc_acil_v1.scripts.run_smoke \
  --output-root "$OUTPUT_ROOT" --device cuda

"$PYTHON" -m experiments.sc_acil_v1.gpu_queue \
  --stage fit_acil --output-root "$OUTPUT_ROOT" \
  --gpus 0,1,2,3 --workers-per-gpu 2

"$PYTHON" -m experiments.sc_acil_v1.gpu_queue \
  --stage formal_gate --output-root "$OUTPUT_ROOT" \
  --gpus 0,1,2,3 --workers-per-gpu 3

"$PYTHON" -m experiments.sc_acil_v1.scripts.adjudicate \
  --output-root "$OUTPUT_ROOT"
