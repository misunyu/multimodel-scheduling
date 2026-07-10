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

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HOME/.cache/hf_modules_msyu}"
mkdir -p "$HF_MODULES_CACHE"

# All runtime model assets are pre-cached; force offline so transformers does not
# stall on Hub HEAD/rate-limit checks during profiling/execution. Set to 0 only
# when intentionally downloading a new checkpoint.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Mobilint NPU runs as the current user (device is accessible without sudo).
