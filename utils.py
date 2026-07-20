"""
Utility functions for the multimodel scheduling application.
"""
import os
import json
import threading
import psutil
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import time
from contextlib import ContextDecorator

# Constants
LOG_DIR = "./logs"
MAX_LOG_ENTRIES = 500
# JSON Lines. Overridable per run so a convergence sweep can keep each run's trace
# separate instead of appending every run into one file.
RESULT_TIME_FILE = os.environ.get("RESULT_TIME_FILE", "result_pre_post_time.json")

# -------------------------------------------------------------
# Lightweight async performance log for UI display (per-view)
# -------------------------------------------------------------

def async_log(model_name, infer_time_ms, avg_fps, log_enabled=0):
    """
    Asynchronously log model performance data to a JSON file (per-model rolling log).
    Keep behavior/backward compatibility with existing callers.
    """
    if not log_enabled:
        return

    def write_log():
        os.makedirs(LOG_DIR, exist_ok=True)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "inference_time_ms": round(infer_time_ms, 2),
            "average_fps": round(avg_fps, 2)
        }
        log_file = os.path.join(LOG_DIR, f"{model_name}_log.json")

        need_trim = False
        line_count = 0

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for _ in f:
                    line_count += 1
            need_trim = line_count >= MAX_LOG_ENTRIES

        if need_trim:
            logs = []
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            logs.append(log_data)
            logs = logs[-MAX_LOG_ENTRIES:]
            with open(log_file, "w") as f:
                for entry in logs:
                    json.dump(entry, f)
                    f.write("\n")
        else:
            with open(log_file, "a+") as f:
                json.dump(log_data, f)
                f.write("\n")

    threading.Thread(target=write_log, daemon=True).start()

# -------------------------------------------------------------
# Timing and performance logging utilities (modularized)
# -------------------------------------------------------------

# Timing/logging lives in timing_utils.py -- the module the workers actually import.
# utils.py used to carry a second, byte-for-byte copy of these functions; nothing
# imported them, so instrumentation added here silently did nothing while the real
# path went unchanged. Re-export instead of duplicating.
from timing_utils import (  # noqa: F401
    should_record_time,
    get_run_id,
    append_timing_record,
    PerfTimer,
    log_model_load,
    log_inference,
    log_visualize,
    RESULT_TIME_FILE,
)


# -------------------------------------------------------------
# Model asset path resolution (Mobilint NPU .mxq + GPU/CPU ONNX)
# -------------------------------------------------------------
# Layout:
#   models/onnx/<name>.onnx       -> CPU / GPU (ONNX Runtime) vision models
#   models/mobilint/<name>.mxq    -> Mobilint Aries NPU vision models
# LLM / VLM assets are resolved from the HuggingFace hub via model_registry.


def _canonical_name(logical_name: str) -> str:
    """Map a logical/legacy name to a canonical registry name.

    A name the registry already knows is returned untouched. It used to go through
    `_normalize` unconditionally, and `_normalize` maps ANY name containing "yolo"
    to "yolo11s" -- so yolo11n/m/l/x all resolved to yolo11s.onnx and yolo11s.mxq.
    Every CPU/GPU run of a yolo variant was silently running yolo11s instead, which
    is why their profiled CPU/GPU latencies were identical regardless of model size.
    `_normalize` is for legacy aliases ("yolov3", "tiny") only; it must never be
    allowed to rewrite a real model name.
    """
    try:
        import model_registry as reg
        if logical_name in reg.MODELS:
            return logical_name
        return reg._normalize(logical_name)
    except Exception:
        return logical_name


def resolve_onnx_path(model_name: str) -> str:
    """Resolve the ONNX file path (CPU/GPU) for a vision model name."""
    name = _canonical_name(model_name)
    p = os.path.join("models", "onnx", f"{name}.onnx")
    if not os.path.exists(p):
        raise FileNotFoundError(f"ONNX file not found: {p} (model '{model_name}')")
    return p


def resolve_mxq_path(model_name: str) -> str:
    """Resolve the Mobilint .mxq file path for a vision model name."""
    name = _canonical_name(model_name)
    p = os.path.join("models", "mobilint", f"{name}.mxq")
    if not os.path.exists(p):
        raise FileNotFoundError(f".mxq file not found: {p} (model '{model_name}')")
    return p


# Backward-compatible alias used by a few older call sites.
def resolve_cpu_model_onnx(logical_name: str) -> str:
    return resolve_onnx_path(logical_name)

# -------------------------------------------------------------
# Image/Qt and system utilities (existing)
# -------------------------------------------------------------

def create_x_image(width=640, height=480, label="No model specified"):
    """
    Create an image with a black background and a white X across it.

    Args:
        width: Width of the image
        height: Height of the image
        label: Caption drawn across the centre (e.g. "No display" for an inactive slot)

    Returns:
        A numpy array representing the image
    """
    # Create a black image
    img = np.zeros((height, width, 3), np.uint8)

    # Draw a white X
    cv2.line(img, (0, 0), (width, height), (255, 255, 255), 5)
    cv2.line(img, (0, height), (width, 0), (255, 255, 255), 5)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = label
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    return img

def convert_cv_to_qt(cv_img):
    """
    Convert OpenCV image to Qt pixmap.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QPixmap object
    """
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    try:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    except Exception as e:
        print(f"[convert_cv_to_qt ERROR] {e}")
        return QPixmap()

def get_cpu_metrics(interval=0):
    """
    Get CPU performance metrics.
    
    Args:
        interval: Time interval for CPU percent calculation
        
    Returns:
        Dictionary containing CPU metrics
    """
    cpu_percent = psutil.cpu_percent(interval=interval)
    load1, load5, load15 = os.getloadavg()
    cpu_stats = psutil.cpu_stats()
    ctx_switches = cpu_stats.ctx_switches
    interrupts = cpu_stats.interrupts
    return {
        "CPU_Usage_percent": cpu_percent,
        "Load_Average": (load1, load5, load15),
        "Context_Switches": ctx_switches,
        "Interrupts": interrupts
    }