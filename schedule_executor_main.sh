#!/bin/bash

# Default schedule file
SCHEDULE_FILE="./model_schedules.yaml"
SCHEDULE_NAME=""

# Simple arg parsing for -schedule/--schedule/-s <path>, --schedule_name <name>, and passthrough of other args
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -schedule|--schedule|-s)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        SCHEDULE_FILE="$2"
        shift 2
      else
        echo "Error: -schedule/--schedule/-s requires a file path argument" >&2
        exit 1
      fi
      ;;
    --schedule_name|--schedule-name)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        SCHEDULE_NAME="$2"
        shift 2
      else
        echo "Error: --schedule_name requires a combination name argument" >&2
        exit 1
      fi
      ;;
    --duration)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        EXTRA_ARGS+=(--duration "$2")
        shift 2
      else
        echo "Error: --duration requires a value" >&2
        exit 1
      fi
      ;;
    --auto_start_all)
      EXTRA_ARGS+=(--auto_start_all)
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [-schedule <path_to_yaml>] [--schedule_name <combination_name>] [--duration <sec>] [--auto_start_all]";
      echo "  Defaults: -schedule ./model_schedules.yaml";
      exit 0
      ;;
    *)
      # passthrough any other args
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

# If the specified schedule file doesn't exist and looks like a basename, try tests/<basename>
if [[ ! -f "$SCHEDULE_FILE" && "$SCHEDULE_FILE" != */* ]]; then
  if [[ -f "./tests/$SCHEDULE_FILE" ]]; then
    SCHEDULE_FILE="./tests/$SCHEDULE_FILE"
  fi
fi

# Build command: always pass --schedule; append --schedule_name only when provided
CMD=(sudo /opt/.pyenv/shims/python3 schedule_executor_main.py --schedule "$SCHEDULE_FILE")
if [[ -n "$SCHEDULE_NAME" ]]; then
  CMD+=(--schedule_name "$SCHEDULE_NAME")
fi
# Append any extra args collected (e.g., --duration, --auto_start_all, etc.)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

# Execute
# shellcheck disable=SC2068
${CMD[@]}
