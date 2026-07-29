#!/usr/bin/env bash
# Reproducible build of the project's own ./.venv for the Mobilint + GPU stack.
# Usage:  ./setup_venv.sh
#
# Rebuilds .venv from scratch with the exact working configuration:
#   - Python 3.10, torch/torchvision CUDA-13 build (RTX 5090 / Blackwell)
#   - requirements.txt
#   - Mobilint SDK from local sources (not on PyPI)
# Ends by enforcing headless-only OpenCV (see enforce_headless_opencv below).
set -euo pipefail
cd "$(dirname "$0")"

# Local Mobilint SDK sources (qbcompiler wheel + editable model-zoo / tracker).
MOBILINT_SRC="/home/msyu/PycharmProjects/MobilintTest"
QBCOMPILER_WHL="$MOBILINT_SRC/qbcompiler-1.1.2+aries2-py3-none-any.whl"

echo "=== [1/6] create .venv (python3.10) ==="
rm -rf .venv
python3.10 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/6] torch/torchvision (CUDA 13.0 build for RTX 5090) ==="
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

echo "=== [3/6] requirements.txt ==="
pip install -r requirements.txt

echo "=== [4/6] Mobilint SDK: qb-runtime + qbcompiler wheel ==="
pip install mobilint-qb-runtime==1.2.0
pip install "$QBCOMPILER_WHL"

echo "=== [5/6] Mobilint editable: mblt-model-zoo (+transformers extra) + mblt-tracker ==="
pip install -e "$MOBILINT_SRC/mblt-model-zoo[transformers]"
pip install -e "$MOBILINT_SRC/mblt-tracker"

echo "=== [6/6] enforce headless-only OpenCV ==="
# ultralytics / mblt-model-zoo / qbcompiler pull in non-headless opencv-python,
# which bundles its own Qt xcb plugin and crashes PyQt5 ("Could not load the Qt
# platform plugin xcb"). Keep only opencv-python-headless.
source ./runtime_env.sh
enforce_headless_opencv "$(pwd)/.venv/bin/python"

echo "=== verify ==="
"$(pwd)/.venv/bin/python" - <<'PY' 2>&1 | grep -vE "reshape_check|Failed to initialize MMC|undefined symbol"
import cv2
from PyQt5 import QtCore
import torch, onnxruntime as ort
import qbruntime, mblt_model_zoo, mblt_tracker
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
print("ORT CUDA:", "CUDAExecutionProvider" in ort.get_available_providers())
print("cv2:", cv2.__version__, "| PyQt5:", QtCore.QT_VERSION_STR, "-- coexist OK")
print("Mobilint SDK (qbruntime/mblt_model_zoo/mblt_tracker): OK")
PY
echo "=== DONE ==="
