"""Central model registry for the Mobilint + GPU multimodel scheduling app.

Single source of truth for the four models used across profiling, schedule
generation, execution and the XGBoost placement predictor. Every model must be
runnable on BOTH the GPU and the Mobilint Aries NPU (CPU is also available).

Model "kind" drives which runtime path and which throughput metric applies:
  - vision (detection / classification): metric = frames/sec (FPS)
  - llm   (text generation)            : metric = tokens/sec (decode)
  - vlm   (image-text-to-text)         : metric = tokens/sec (decode)

Device tokens used everywhere: "cpu", "gpu", "npu".
"""

from __future__ import annotations

import os
from typing import Dict, List

# Canonical device set for the placement problem.
DEVICES: List[str] = ["cpu", "gpu", "npu"]

# Deadline for a request = (1000 / infps) * DEADLINE_FACTOR  [ms] (period-based).
# Factor > 1 gives headroom for the end-to-end pipeline (frame IPC across
# processes, pre/post-processing) so that a well-placed model can actually meet
# its deadline while overloaded placements miss it.
DEADLINE_FACTOR: float = 3.0

# Input-rate scaling levels (multiples of each model's baseline rate).
# Training uses 1x/2x/4x; 3x is reserved as a rate hold-out test (paper protocol).
RATE_FACTORS: List[float] = [1.0, 2.0, 3.0, 4.0]
TRAIN_RATE_FACTORS: List[float] = [1.0, 2.0, 4.0]
HOLDOUT_RATE_FACTORS: List[float] = [3.0]

# kind -> throughput metric label
METRIC_BY_KIND = {
    "vision": "fps",
    "llm": "tokens_per_s",
    "vlm": "tokens_per_s",
}

# Per-model allowed devices. The VLM is impractically slow on CPU (~13s prefill,
# and it stalls contention windows), so restrict it to accelerators. Models not
# listed may use cpu + any accelerator.
DEVICE_CONSTRAINTS = {
    "qwen2_vl": ["gpu", "npu"],
}


def allowed_devices(model_name: str, accelerators=None):
    """Devices a model may be placed on. `accelerators` limits to available ones."""
    allowed = DEVICE_CONSTRAINTS.get(model_name, list(DEVICES))
    if accelerators is not None:
        avail = set(["cpu"]) | set(accelerators)
        allowed = [d for d in allowed if d in avail]
    return allowed


# name -> spec
def _yolo11_entry(size: str) -> dict:
    return {
        "kind": "vision",
        "task": "detection",
        "pipeline": "yolo",
        "npu_class": f"YOLO11{size}",
        "mxq": f"models/mobilint/yolo11{size}.mxq",  # local if present, else zoo auto-downloads
        "onnx": f"models/onnx/yolo11{size}.onnx",
        "input_size": 640,
    }


MODELS: Dict[str, dict] = {
    "yolo11n": _yolo11_entry("n"),
    "yolo11s": _yolo11_entry("s"),
    "yolo11m": _yolo11_entry("m"),
    "yolo11l": _yolo11_entry("l"),
    "yolo11x": _yolo11_entry("x"),
    "resnet50": {
        "kind": "vision",
        "task": "classification",
        "pipeline": "resnet",
        "npu_class": "ResNet50",
        "mxq": "models/mobilint/resnet50.mxq",
        "onnx": "models/onnx/resnet50.onnx",
        "input_size": 224,
    },
    "mobilenet_v2": {
        "kind": "vision",
        "task": "classification",
        # Same runtime path as resnet50: 224x224 ImageNet preprocessing, top-k output.
        "pipeline": "resnet",
        "npu_class": "MobileNet_V2",
        "mxq": "models/mobilint/mobilenet_v2.mxq",
        "onnx": "models/onnx/mobilenet_v2.onnx",
        "input_size": 224,
    },
    "llama1b": {
        "kind": "llm",
        "task": "text-generation",
        "pipeline": "llm",
        # GPU/CPU: open Llama-3.2-1B checkpoint. NPU: Mobilint pre-quantized W8 .mxq.
        "gpu_hf": "unsloth/Llama-3.2-1B-Instruct",
        "npu_hf": "mobilint/Llama-3.2-1B-Instruct",
        "revision": "W8",
    },
    "qwen2_vl": {
        "kind": "vlm",
        "task": "image-text-to-text",
        "pipeline": "vlm",
        "gpu_hf": "Qwen/Qwen2-VL-2B-Instruct",
        "npu_hf": "mobilint/Qwen2-VL-2B-Instruct",
        "revision": "main",
    },
}

# Convenience groupings
VISION_MODELS = [n for n, s in MODELS.items() if s["kind"] == "vision"]
LLM_MODELS = [n for n, s in MODELS.items() if s["kind"] in ("llm", "vlm")]
DETECTION_MODELS = [n for n, s in MODELS.items() if s.get("task") == "detection"]
CLASSIFICATION_MODELS = [n for n, s in MODELS.items() if s.get("task") == "classification"]


def get(model_name: str) -> dict:
    """Return the spec for a model name, tolerating a few legacy aliases."""
    if model_name in MODELS:
        return MODELS[model_name]
    alias = _normalize(model_name)
    if alias in MODELS:
        return MODELS[alias]
    raise KeyError(f"unknown model '{model_name}'. Known: {list(MODELS)}")


def kind_of(model_name: str) -> str:
    return get(model_name)["kind"]


def metric_of(model_name: str) -> str:
    return METRIC_BY_KIND[kind_of(model_name)]


def is_vision(model_name: str) -> bool:
    return kind_of(model_name) == "vision"


def is_detection(model_name: str) -> bool:
    return get(model_name).get("task") == "detection"


def is_classification(model_name: str) -> bool:
    return get(model_name).get("task") == "classification"


def is_llm_like(model_name: str) -> bool:
    return kind_of(model_name) in ("llm", "vlm")


# Explicit spelling aliases ONLY. There is deliberately no substring matching
# here. The old `_normalize` mapped any name containing "yolo" -> yolo11s and any
# name containing "resnet" -> resnet50, which silently collapsed distinct models
# (yolov4, yolov3_big, yolov3_small all became yolo11s) and let schedules that
# named unimplemented models "partially succeed" instead of failing. That turned
# a loud, first-run failure into silent phantom views (see docs/phantom_model_audit.md).
# Add an entry ONLY for a genuine spelling variant of a model already in MODELS.
_ALIASES = {
    # e.g. "llama-1b": "llama1b",  # spelling variant only
}


def _normalize(name: str) -> str:
    return _ALIASES.get((name or "").strip().lower(), name)


def resolution(model_name: str):
    """Return (resolved_id, via_alias) for a name, or raise KeyError.

    resolved_id is a key in MODELS; via_alias is True when an explicit spelling
    alias was applied. Use this (not a bare get()) when you need to log what a
    schedule name actually resolved to.
    """
    if model_name in MODELS:
        return model_name, False
    alias = _normalize(model_name)
    if alias in MODELS:
        return alias, True
    raise KeyError(f"unknown model '{model_name}'. Known: {list(MODELS)}")


def resolves(model_name: str) -> bool:
    """True iff the name resolves to a known model (no exception)."""
    try:
        resolution(model_name)
        return True
    except KeyError:
        return False


def unresolved_models(names):
    """Return the sorted, de-duplicated subset of *names* the registry cannot
    resolve. Empty/falsy names are ignored (an unused view slot is not an error)."""
    return sorted({n for n in names if n and not resolves(n)})


def norm_device(dev: str) -> str:
    """Normalize an execution token to one of DEVICES."""
    d = str(dev).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if d in ("cpu",):
        return "cpu"
    if d in ("gpu", "cuda"):
        return "gpu"
    # Any npu / npu0 / npu1 collapses to the single Mobilint NPU.
    if d.startswith("npu"):
        return "npu"
    return d


def model_names() -> List[str]:
    return list(MODELS.keys())
