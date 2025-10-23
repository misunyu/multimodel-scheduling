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
RESULT_TIME_FILE = "result_pre_post_time.json"  # JSON Lines

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

def should_record_time() -> bool:
    """Return True if RECORD_TIME=1 in environment."""
    try:
        return int(os.environ.get("RECORD_TIME", "0")) == 1
    except Exception:
        return False


def get_run_id() -> str:
    """Return current RUN_ID from environment (may be empty)."""
    return os.environ.get("RUN_ID", "")


def append_timing_record(record: dict):
    """
    Append a single timing record as a JSON line to RESULT_TIME_FILE.
    Using JSON Lines format to avoid concurrency issues with multiple processes.
    Safe: will not raise.
    """
    try:
        rec = dict(record)
        rec.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rid = get_run_id()
        if rid:
            rec.setdefault("run_id", rid)
        with open(RESULT_TIME_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Timing LOGGING ERROR] {e}")


class PerfTimer(ContextDecorator):
    """
    Simple context manager/decorator to measure elapsed time in ms.
    Usage:
        with PerfTimer() as t:
            ...
        elapsed_ms = t.ms
    or as decorator for a function to get (result, elapsed_ms) if needed.
    """
    def __enter__(self):
        self._start = time.time()
        self.ms = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.time() - self._start) * 1000.0
        return False

    def reset(self):
        self._start = time.time()
        self.ms = 0.0


def log_model_load(pipeline: str, device: str, view: str, model: str,
                   model_load_time_ms: float = None, npu_memory_load_time_ms: float = None):
    """Convenience logger for model load events, respects RECORD_TIME."""
    if not should_record_time():
        return
    rec = {
        "kind": "model_load",
        "pipeline": pipeline,
        "device": device,
        "view": view,
        "model": model,
    }
    if model_load_time_ms is not None:
        rec["model_load_time_ms"] = model_load_time_ms
    if npu_memory_load_time_ms is not None:
        rec["npu_memory_load_time_ms"] = npu_memory_load_time_ms
    append_timing_record(rec)


def log_inference(pipeline: str, device: str, view: str, model: str,
                  preprocess_time_ms: float, inference_time_ms: float,
                  postprocess_time_ms: float, wait_to_preprocess_ms: float = 0.0):
    """Convenience logger for per-frame inference timing, respects RECORD_TIME."""
    if not should_record_time():
        return
    append_timing_record({
        "kind": "inference",
        "pipeline": pipeline,
        "device": device,
        "view": view,
        "model": model,
        "preprocess_time_ms": preprocess_time_ms,
        "inference_time_ms": inference_time_ms,
        "postprocess_time_ms": postprocess_time_ms,
        "wait_to_preprocess_ms": wait_to_preprocess_ms,
    })

# -------------------------------------------------------------
# Model path resolution (legacy -> current)
# -------------------------------------------------------------

# Map legacy logical model names to current base names
#   resnet50_small -> resnet50_1, resnet50_big -> resnet50_2
#   yolov3_small -> yolov3_1,   yolov3_big   -> yolov3_2
_LEGACY_TO_BASE = {
    "resnet50_small": "resnet50_1",
    "resnet50_big": "resnet50_2",
    "yolov3_small": "yolov3_1",
    "yolov3_big": "yolov3_2",
}


def _normalize_base(name: str) -> str:
    """Return best-guess base folder name for a given logical or base name."""
    if not name:
        return name
    # If already matches existing folder, keep
    candidate = name
    if os.path.isdir(os.path.join("models", candidate)):
        return candidate
    # Try legacy mapping
    mapped = _LEGACY_TO_BASE.get(name, name)
    if os.path.isdir(os.path.join("models", mapped)):
        return mapped
    # Heuristics: if contains 'small'/'big'
    low = name.lower()
    if "yolov3" in low:
        return "yolov3_1" if "small" in low or "_1" in low else "yolov3_2"
    if "resnet" in low:
        return "resnet50_1" if "small" in low or "_1" in low else "resnet50_2"
    return name


def resolve_cpu_model_onnx(logical_name: str) -> str:
    """
    Resolve CPU ONNX model path for a logical model name.
    Returns a path string (may be non-existent if nothing matched).
    """
    base = _normalize_base(logical_name)
    candidates = [
        os.path.join("models", base, "model", f"{base}.onnx"),
        os.path.join("models", base, "model", f"{logical_name}.onnx"),
        os.path.join("models", logical_name, "model", f"{logical_name}.onnx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback to the first candidate even if missing
    return candidates[0]


def resolve_npu_object_o(logical_name: str, part: int = 1) -> str:
    """
    Resolve NPU object (.o) path for a logical model name and partition id.
    Default part=1 to match typical middle partition binaries.
    """
    base = _normalize_base(logical_name)
    suffix = f"{base}_neubla_p{part}.o"
    candidates = [
        os.path.join("models", base, "npu_code", suffix),
        os.path.join("models", logical_name, "npu_code", f"{logical_name}_neubla_p{part}.o"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback
    return candidates[0]

# -------------------------------------------------------------
# Image/Qt and system utilities (existing)
# -------------------------------------------------------------

def create_x_image(width=640, height=480):
    """
    Create an image with a black background and a white X across it.
    
    Args:
        width: Width of the image
        height: Height of the image
        
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
    text = "No model specified"
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