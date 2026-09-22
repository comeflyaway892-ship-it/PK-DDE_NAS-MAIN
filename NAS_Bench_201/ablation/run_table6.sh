#!/usr/bin/env bash
# Run from any working directory; optional arguments go to the Python entry point.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="$(mktemp -d "$SCRIPT_DIR/table6_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf 'Log directory: %s\n' "$LOG_DIR"
set +e
/usr/bin/time -v -o "$LOG_DIR/resources.txt" \
  "$PYTHON_BIN" -u -X faulthandler "$SCRIPT_DIR/run_component_ablation.py" \
  --steps 15 --dynamic-steps 10 --population 20 --runs 100 --temperature 7.0 \
  --prior-strength 0.99 \
  --no-dynamic-mode uniform --no-posterior-mode direct_x0 \
  --output "$LOG_DIR/results" "$@" \
  2>&1 | tee "$LOG_DIR/run.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "$RUN_STATUS" | tee "$LOG_DIR/status.txt"
printf 'Logs: %s\n' "$LOG_DIR"
exit "$RUN_STATUS"
