#!/bin/bash

# Default schedule file
SCHEDULE_FILE="./model_schedules.yaml"

# Simple arg parsing for -schedule <path>
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
    -h|--help)
      echo "Usage: $0 [-schedule <path_to_yaml>]";
      echo "  Default: ./model_schedules.yaml";
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use -h for help" >&2
      exit 1
      ;;
  esac
done

sudo /opt/.pyenv/shims/python3 schedule_executor_main.py --schedule "$SCHEDULE_FILE"
