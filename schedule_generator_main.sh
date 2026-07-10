#!/bin/bash
# Deprecated: the Neubla profiler GUI has been retired. Use profile_models.py
# and generate_schedules.py instead (see schedule_generator_main.py / README.md).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_env.sh"
"$PYTHON_BIN" schedule_generator_main.py "$@"
