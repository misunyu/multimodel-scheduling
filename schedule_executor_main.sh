#!/bin/bash

# Default schedule file
SCHEDULE_FILE="./model_schedules.yaml"
SCHEDULE_NAME=""

# Simple arg parsing for -schedule <path> and --schedule_name <name>
while [[ $# -gt 0 ]]; do
  case "$1" in
    -schedule)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        SCHEDULE_FILE="$2"
        shift 2
      else
        echo "Error: -schedule requires a file path argument" >&2
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
    -h|--help)
      echo "Usage: $0 [-schedule <path_to_yaml>] [--schedule_name <combination_name>]";
      echo "  Defaults: -schedule ./model_schedules.yaml";
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use -h for help" >&2
      exit 1
      ;;
  esac
done

# Build command: always pass --schedule; append --schedule_name only when provided
CMD=(sudo /opt/.pyenv/shims/python3 schedule_executor_main.py --schedule "$SCHEDULE_FILE")
if [[ -n "$SCHEDULE_NAME" ]]; then
  CMD+=(--schedule_name "$SCHEDULE_NAME")
fi

# Execute
# shellcheck disable=SC2068
${CMD[@]}
