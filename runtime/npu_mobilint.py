"""Mobilint MLA100 (ARIES) NPU runtime wrapper.

Replaces the legacy `from npu import ...` (Neubla) integration in
model_processors.py with a thin facade around `mblt_model_zoo`.

Public surface:
    list_supported_models(task)         -> dict[str, type]
    build_model(name, infer_mode, ...)  -> MbltModel  (preprocess / __call__ / postprocess / dispose)
    run_yolo_npu_process(...)           -> multiprocess.Process target (drop-in for the
                                           old run_yolo_npu_process signature)
    run_resnet_npu_process(...)         -> ditto for ResNet
"""

from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional


def list_supported_models(task: str = "detection") -> Dict[str, type]:
    """Return a {name: class} map of pre-quantized Mobilint vision models for `task`.

    `task` is one of: detection, segmentation, pose, classification.
    Only object detection is enumerated here; extend as needed.
    """
    from mblt_model_zoo import vision

    if task == "detection":
        # Subset most relevant to streaming perception baselines
        return {
            "yolov3":   vision.YOLOv3,
            "yolov5n":  vision.YOLOv5n,
            "yolov5s":  vision.YOLOv5s,
            "yolov5m":  vision.YOLOv5m,
            "yolov8n":  vision.YOLOv8n,
            "yolov8s":  vision.YOLOv8s,
            "yolov8m":  vision.YOLOv8m,
            "yolov8l":  vision.YOLOv8l,
            "yolov8x":  vision.YOLOv8x,
            "yolo11n":  vision.YOLO11n,
            "yolo11s":  vision.YOLO11s,
            "yolo11m":  vision.YOLO11m,
        }
    if task == "classification":
        return {
            "resnet18":  vision.ResNet18,
            "resnet34":  vision.ResNet34,
            "resnet50":  vision.ResNet50,
            "resnet101": vision.ResNet101,
            "resnet152": vision.ResNet152,
        }
    raise ValueError(f"unsupported task: {task}")


def build_model(name: str,
                task: str = "detection",
                infer_mode: str = "global8",
                product: str = "aries",
                local_path: Optional[str] = None) -> Any:
    """Instantiate a Mobilint NPU-backed model.

    Resolves `local_path` from models/mobilint/<name>.mxq if not given.
    `infer_mode` ∈ {single, multi, global4, global8}. global8 = full chip.
    `product`    ∈ {aries, regulus}.
    """
    cls = list_supported_models(task).get(name)
    if cls is None:
        raise KeyError(f"model '{name}' not in Mobilint zoo task='{task}'")
    if local_path is None:
        from utils import resolve_mxq_path
        try:
            local_path = resolve_mxq_path(name)
        except FileNotFoundError:
            local_path = None  # fall back to HF download
    return cls(local_path=local_path, infer_mode=infer_mode, product=product)


def run_yolo_npu_process(input_queue,
                         output_queue,
                         shutdown_event,
                         npu_id: int = 0,
                         view_name: Optional[str] = None,
                         model_name: str = "yolov8n",
                         infer_mode: str = "global8",
                         conf_thres: float = 0.25,
                         iou_thres: float = 0.45):
    """Drop-in replacement for legacy run_yolo_npu_process (Neubla).

    `npu_id` is currently advisory — mblt_model_zoo picks the device per `infer_mode`;
    multi-device routing will be wired in once we exercise multiple Aries cards.
    """
    try:
        host_load_s = time.time()
        model = build_model(model_name, task="detection", infer_mode=infer_mode)
        host_load_ms = (time.time() - host_load_s) * 1000.0
        print(f"[YOLO NPU{npu_id} view={view_name}] {model_name} loaded in {host_load_ms:.1f} ms")

        try:
            while not shutdown_event.is_set():
                try:
                    item = input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if isinstance(item, tuple) and len(item) == 2:
                    frame, enqueue_ts = item
                else:
                    frame, enqueue_ts = item, None

                t_pre0 = time.time()
                input_img = model.preprocess(frame)
                t_infer0 = time.time()
                raw = model(input_img)
                t_post0 = time.time()
                result = model.postprocess(raw, conf_thres=conf_thres, iou_thres=iou_thres)
                t_end = time.time()

                output_queue.put({
                    "view": view_name,
                    "model": model_name,
                    "device": f"NPU{npu_id}",
                    "result": result,
                    "timing_ms": {
                        "wait":   ((t_pre0 - enqueue_ts) * 1000.0) if enqueue_ts else 0.0,
                        "pre":    (t_infer0 - t_pre0) * 1000.0,
                        "infer":  (t_post0 - t_infer0) * 1000.0,
                        "post":   (t_end - t_post0) * 1000.0,
                    },
                })
        finally:
            model.dispose()
    except Exception as e:
        print(f"[YOLO NPU{npu_id} view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise


def run_resnet_npu_process(input_queue,
                           output_queue,
                           shutdown_event,
                           npu_id: int = 0,
                           view_name: Optional[str] = None,
                           model_name: str = "resnet50",
                           infer_mode: str = "global8",
                           topk: int = 5):
    """Drop-in replacement for legacy run_resnet_npu_process (Neubla)."""
    try:
        host_load_s = time.time()
        model = build_model(model_name, task="classification", infer_mode=infer_mode)
        host_load_ms = (time.time() - host_load_s) * 1000.0
        print(f"[ResNet NPU{npu_id} view={view_name}] {model_name} loaded in {host_load_ms:.1f} ms")

        try:
            while not shutdown_event.is_set():
                try:
                    item = input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if isinstance(item, tuple) and len(item) == 2:
                    frame, enqueue_ts = item
                else:
                    frame, enqueue_ts = item, None

                t_pre0 = time.time()
                input_img = model.preprocess(frame)
                t_infer0 = time.time()
                raw = model(input_img)
                t_post0 = time.time()
                result = model.postprocess(raw)
                t_end = time.time()

                output_queue.put({
                    "view": view_name,
                    "model": model_name,
                    "device": f"NPU{npu_id}",
                    "result": result,
                    "topk": topk,
                    "timing_ms": {
                        "wait":   ((t_pre0 - enqueue_ts) * 1000.0) if enqueue_ts else 0.0,
                        "pre":    (t_infer0 - t_pre0) * 1000.0,
                        "infer":  (t_post0 - t_infer0) * 1000.0,
                        "post":   (t_end - t_post0) * 1000.0,
                    },
                })
        finally:
            model.dispose()
    except Exception as e:
        print(f"[ResNet NPU{npu_id} view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise
