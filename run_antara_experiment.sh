#!/usr/bin/env bash
# Run NPU (Antara) experiments with 4 models across 3 adaptive modes.
# Usage: ./run_antara_experiment.sh [DURATION]
#   DURATION: execution duration per combination in seconds (default: 30)
#
# Must be run inside the Docker container (./docker_run.sh).
# Outputs:
#   results/adaptive_metrics_mode{0,1,2}_antara_<TIMESTAMP>.csv
#   results/performance_antara_<TIMESTAMP>.json (per mode)

set -euo pipefail

DURATION=${1:-30}
SCHEDULE="gen_schedules_antara/model_schedules_rbrsybys.yaml"
RESULTS_DIR="results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$SCHEDULE" ]]; then
  echo "Error: Schedule file not found: $SCHEDULE" >&2
  exit 1
fi

PY=python

echo "============================================"
echo "NPU (Antara) Experiment — 4 models, 3 modes"
echo "Schedule: $SCHEDULE"
echo "Duration per combination: ${DURATION}s"
echo "Timestamp: $TIMESTAMP"
echo "============================================"
echo

for MODE in 0 1 2; do
  CSV_PATH="$RESULTS_DIR/adaptive_metrics_mode${MODE}_antara_${TIMESTAMP}.csv"
  echo "----------------------------------------"
  echo "Mode $MODE: Starting (CSV -> $CSV_PATH)"
  echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

  $PY schedule_executor_main.py \
    --schedule "$SCHEDULE" \
    --duration "$DURATION" \
    --adaptive-mode "$MODE" \
    --metrics-csv "$CSV_PATH" \
    --auto_start_all 2>&1 | tee "$RESULTS_DIR/log_mode${MODE}_antara_${TIMESTAMP}.txt"

  STATUS=$?
  echo "End: $(date '+%Y-%m-%d %H:%M:%S') (exit=$STATUS)"

  # Kill any lingering processes
  pkill -f "schedule_executor_main.py" 2>/dev/null || true
  sleep 3

  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Warning: CSV not generated for mode $MODE"
  else
    LINES=$(wc -l < "$CSV_PATH")
    echo "CSV generated: $CSV_PATH ($LINES lines)"
  fi
  echo
done

echo "============================================"
echo "All 3 modes completed."
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Next step: generate graphs with:"
echo "  python plot_antara_results.py $TIMESTAMP"
echo "============================================"
