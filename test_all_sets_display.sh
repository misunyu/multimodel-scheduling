#!/bin/bash
# Run the 15-set sequential display test with the runtime interpreter on the real display.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_env.sh"
export DISPLAY="${DISPLAY:-:1}"
unset QT_QPA_PLATFORM            # render on the real display, not off-screen
export DISABLE_VISUALIZATION=0   # rendering ON
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/test_all_sets_display.py"
