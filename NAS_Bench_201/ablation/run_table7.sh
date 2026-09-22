#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="$(mktemp -d "$SCRIPT_DIR/table7_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf 'Log directory: %s\n' "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG_DIR/matplotlib_cache}"
set +e
/usr/bin/time -v -o "$LOG_DIR/resources.txt" \
  "$PYTHON_BIN" -u -X faulthandler "$SCRIPT_DIR/run_kernel_comparison.py" \
  --steps 15 --dynamic-steps 10 --population 20 --runs 100 --temperature 7.0 \
  --knowledge-topk 6 --output "$LOG_DIR/results" "$@" \
  2>&1 | tee "$LOG_DIR/run.log"
PIPE_CODES=("${PIPESTATUS[@]}")
RUN_STATUS=${PIPE_CODES[0]}
if [ "$RUN_STATUS" -eq 0 ] && [ "${PIPE_CODES[1]}" -ne 0 ]; then
  RUN_STATUS=${PIPE_CODES[1]}
fi
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "$RUN_STATUS" | tee "$LOG_DIR/status.txt"
printf 'Logs: %s\n' "$LOG_DIR"
exit "$RUN_STATUS"
