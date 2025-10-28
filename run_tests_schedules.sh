#!/usr/bin/env bash

# Runs schedule_executor_main.py sequentially for each YAML file under the given directory (default: tests/) using --schedule option.
# After each run completes, forcibly ensures the application is fully terminated before proceeding.
# Usage: ./run_tests_schedules.sh [--timeout SECONDS] [SCHEDULE_DIR]
# Notes:
# - Iterates over SCHEDULE_DIR/*.yaml in sorted order (defaults to tests/ under project root).
# - Uses schedule_executor_main.sh if present (to keep consistent invocation), otherwise calls python directly.
# - After each run, kills any lingering schedule_executor_main.py processes and waits a bit to ensure clean shutdown.

set -euo pipefail

TIMEOUT_SECS=0
if [[ ${1:-} == "--timeout" ]]; then
  if [[ -n ${2:-} && ${2:-} =~ ^[0-9]+$ ]]; then
    TIMEOUT_SECS=$2
    shift 2
  else
    echo "Error: --timeout requires an integer argument (seconds)" >&2
    exit 1
  fi
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Positional arg: optional schedules directory (relative to project root or absolute). Default: tests
if [[ $# -gt 1 ]]; then
  echo "Error: too many arguments. Usage: $0 [--timeout SECONDS] [SCHEDULE_DIR]" >&2
  exit 1
fi

if [[ -n ${1:-} ]]; then
  if [[ "$1" = /* ]]; then
    TEST_DIR="$1"
  else
    TEST_DIR="$ROOT_DIR/$1"
  fi
else
  TEST_DIR="$ROOT_DIR/tests"
fi

if [[ ! -d "$TEST_DIR" ]]; then
  echo "Schedules directory not found at $TEST_DIR" >&2
  exit 1
fi

mapfile -t FILES < <(find "$TEST_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No YAML files found under $TEST_DIR" >&2
  exit 1
fi

# Determine Python executable from virtual environment
PY=python3
if [[ -d "$ROOT_DIR/.venv" ]]; then
  # Use virtual environment Python if available
  if [[ -f "$ROOT_DIR/.venv/bin/python3" ]]; then
    PY="$ROOT_DIR/.venv/bin/python3"
    echo "Using virtual environment Python: $PY"
  elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
    PY="$ROOT_DIR/.venv/bin/python"
    echo "Using virtual environment Python: $PY"
  fi
else
  # Fallback to system python or pyenv
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif [[ -f /opt/.pyenv/shims/python3 ]]; then
    PY=/opt/.pyenv/shims/python3
  fi
fi

# Verify Python has PyQt5
if ! "$PY" -c "import PyQt5" 2>/dev/null; then
  echo "Error: PyQt5 not found in Python environment ($PY)" >&2
  echo "Please install PyQt5: pip install pyqt5" >&2
  exit 1
fi

invoke_one() {
  local schedule_file="$1"
  echo "========================================"
  echo "Running schedule with file: $schedule_file"
  echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

  if [[ -x "$ROOT_DIR/schedule_executor_main.sh" ]]; then
    if (( TIMEOUT_SECS > 0 )); then
      timeout "$TIMEOUT_SECS" bash -c '"$0" -schedule "$1" --duration 30 --auto_start_all' "$ROOT_DIR/schedule_executor_main.sh" "$schedule_file"
    else
      "$ROOT_DIR/schedule_executor_main.sh" -schedule "$schedule_file" --duration 30 --auto_start_all
    fi
  else
    # Direct python invocation without sudo
    if (( TIMEOUT_SECS > 0 )); then
      timeout "$TIMEOUT_SECS" "$PY" "$ROOT_DIR/schedule_executor_main.py" --schedule "$schedule_file" --duration 30 --auto_start_all
    else
      "$PY" "$ROOT_DIR/schedule_executor_main.py" --schedule "$schedule_file" --duration 30 --auto_start_all
    fi
  fi

  local status=$?
  echo "End: $(date '+%Y-%m-%d %H:%M:%S') (status=$status)"
  return $status
}

force_cleanup() {
  echo "Ensuring application fully terminated..."
  # Kill python processes running schedule_executor_main.py (current user)
  pkill -f -u "$USER" "schedule_executor_main.py" >/dev/null 2>&1 || true
  # If the helper shell script remains, nothing to kill explicitly beyond above.
  # Give the system a moment to release resources
  sleep 2
}

OVERALL_OK=0

for f in "${FILES[@]}"; do
  if ! invoke_one "$f"; then
    echo "Run failed for $f" >&2
    OVERALL_OK=1
  fi
  force_cleanup
  echo "Proceeding to next file..."
  echo
  # Small delay to make sure everything is clean
  sleep 1
done

if [[ $OVERALL_OK -ne 0 ]]; then
  echo "One or more runs failed." >&2
  exit $OVERALL_OK
fi

echo "All schedules completed successfully."