"""
Timing and performance logging utilities for the multimodel scheduling application.
Separated from utils.py to keep timing concerns modular.
"""
import os
import json
from datetime import datetime
import time
from contextlib import ContextDecorator

# JSON Lines. Overridable per run so a sweep can keep each run's trace separate
# instead of appending every run into one file.
RESULT_TIME_FILE = os.environ.get("RESULT_TIME_FILE", "result_pre_post_time.json")


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
        # Sub-second wall clock: the string timestamp has 1s resolution, too coarse to
        # bin a convergence trace or to place an event inside the warmup window.
        rec.setdefault("t_wall", time.time())
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
                  postprocess_time_ms: float, wait_to_preprocess_ms: float = 0.0,
                  tokens_per_s: float = None, n_out: int = None, n_boxes: int = None):
    """Convenience logger for per-frame inference timing, respects RECORD_TIME.

    `tokens_per_s`/`n_out` are the LLM-VLM equivalent of a completed request; without
    them a trace cannot show y3 (token throughput) converging. `n_boxes` lets a
    detection trace explain its own postprocess cost (NMS scales with box count).
    """
    if not should_record_time():
        return
    rec = {
        "kind": "inference",
        "pipeline": pipeline,
        "device": device,
        "view": view,
        "model": model,
        "preprocess_time_ms": preprocess_time_ms,
        "inference_time_ms": inference_time_ms,
        "postprocess_time_ms": postprocess_time_ms,
        "wait_to_preprocess_ms": wait_to_preprocess_ms,
    }
    if tokens_per_s is not None:
        rec["tokens_per_s"] = tokens_per_s
    if n_out is not None:
        rec["n_out"] = n_out
    if n_boxes is not None:
        rec["n_boxes"] = n_boxes
    append_timing_record(rec)


def log_visualize(view: str, model: str, device: str, visualize_time_ms: float):
    """Time spent turning a result into pixels (the pipeline's 4th stage).

    It runs on the CPU in the GUI process no matter which device the model was placed
    on, so it is a cost the placement decision cannot remove; it has to be measured
    separately to be reasoned about.
    """
    if not should_record_time():
        return
    append_timing_record({
        "kind": "visualize",
        "view": view,
        "model": model,
        "device": device,
        "visualize_time_ms": visualize_time_ms,
    })
