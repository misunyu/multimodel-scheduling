#!/usr/bin/env bash
# Shared runtime environment for the Mobilint + GPU stack.
# Source this before running any executor/profiler/training script:
#     source runtime_env.sh
#
# PYTHON_BIN: interpreter that has the Mobilint SDK + onnxruntime-gpu + torch(cuda)
#             + transformers installed. Override by exporting PYTHON_BIN yourself.
# HF_MODULES_CACHE: writable dir for transformers "remote code" (trust_remote_code)
#             — avoids root-owned entries under the default ~/.cache path.

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
  elif [ -x "/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python" ]; then
    # Fallback: reuse the reference env's interpreter (READ-ONLY; never pip-install into it).
    PYTHON_BIN="/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHON_BIN

# Keep OpenCV headless-only. ultralytics / mblt-model-zoo / qbcompiler depend on
# the non-headless `opencv-python`, which bundles its own Qt xcb plugin and makes
# PyQt5 crash ("Could not load the Qt platform plugin xcb"). Any `pip install` can
# silently re-pull it, so we detect and remove it here. Takes a python interpreter
# path; only acts when the bad package's Qt plugins are actually present.
enforce_headless_opencv() {
  local py="${1:?enforce_headless_opencv: interpreter path required}"
  local sp
  sp="$("$py" -c 'import site,sys; print((site.getsitepackages() or [""])[0])' 2>/dev/null)" || return 0
  # Fast path: act only on the true non-headless marker (its bundled xcb plugin),
  # not on an empty cv2/qt/plugins dir a prior uninstall may have left behind.
  [ -f "$sp/cv2/qt/plugins/platforms/libqxcb.so" ] || return 0
  echo "[runtime_env] non-headless opencv detected -> repairing to headless-only"
  "$py" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
  "$py" -m pip install --force-reinstall --no-deps opencv-python-headless==4.13.0.92 >/dev/null 2>&1 || true
}

# Auto-repair only the project's OWN .venv (never the read-only video fallback).
case "$PYTHON_BIN" in
  "$(pwd)/.venv/bin/python") enforce_headless_opencv "$PYTHON_BIN" ;;
esac

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HOME/.cache/hf_modules_msyu}"
mkdir -p "$HF_MODULES_CACHE"

# All runtime model assets are pre-cached; force offline so transformers does not
# stall on Hub HEAD/rate-limit checks during profiling/execution. Set to 0 only
# when intentionally downloading a new checkpoint.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Mobilint NPU runs as the current user (device is accessible without sudo).
