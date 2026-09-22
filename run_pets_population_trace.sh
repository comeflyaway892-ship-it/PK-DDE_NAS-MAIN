#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/pets_population_trace_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
printf 'Output directory: %s\n' "$RUN_DIR"
set +e
/usr/bin/time -v -o "$RUN_DIR/resources.txt" \
  "$PYTHON_BIN" -u "$ROOT_DIR/run_pets_population_trace.py" \
  --runs 10 --output "$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "$status" | tee "$RUN_DIR/status.txt"
exit "$status"
