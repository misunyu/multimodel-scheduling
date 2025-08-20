"""
Timing and performance logging utilities for the multimodel scheduling application.
Separated from utils.py to keep timing concerns modular.
"""
import os
import json
from datetime import datetime
import time
from contextlib import ContextDecorator

RESULT_TIME_FILE = "result_pre_post_time.json"  # JSON Lines


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
