"""Model processing functions for the multimodel scheduling application.

Rewritten for the ACCV 2026 (Mobilint MLA100) stack:
  - NPU path uses mblt_model_zoo (Mobilint Aries .mxq), replacing the previous
    Neubla `npu` module integration.
  - CPU/GPU paths use ONNX Runtime against Ultralytics-exported YOLOv8 models
    and a torchvision-exported ResNet50 (under models/onnx/).
  - Function signatures are kept backward-compatible with unified_viewer.py,
    so wiring changes are limited to model-name defaults.
"""

import os
import queue
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from timing_utils import log_inference, log_model_load
from utils import resolve_mxq_path, resolve_onnx_path

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _drain_item(input_queue, timeout=1.0):
    """Pop one item from `input_queue`, normalizing (frame, ts) vs frame."""
    try:
        item = input_queue.get(timeout=timeout)
    except queue.Empty:
        return None, None
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
        return item[0], item[1]
    return item, None


def _ort_session(onnx_path: str, gpu: bool):
    """Build an ONNX Runtime session. `gpu=True` requires CUDAExecutionProvider."""
    so = ort.SessionOptions()
    try:
        so.log_severity_level = 3
    except Exception:
        pass
    providers = ["CUDAExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    if gpu and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"CUDA EP unavailable for {onnx_path}")
    return sess


def _yolov8_letterbox(frame_bgr, size=640):
    """Resize+pad BGR frame to (size, size) NCHW float32 [0,1] tensor."""
    h0, w0 = frame_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    arr = canvas[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(arr[None, ...]), (r, top, left)


def _yolov8_postprocess(raw_out, meta, conf_thres=0.25, iou_thres=0.45):
    """Decode YOLOv8 ONNX output (1, 84, N) → list[(x1,y1,x2,y2,score,cls)] in input coords."""
    # raw_out[0] shape: (1, 4 + nc, N)
    out = raw_out[0]
    if out.ndim == 3 and out.shape[1] < out.shape[2]:
        pred = out[0].transpose(1, 0)  # (N, 4+nc)
    else:
        pred = out[0]
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    cls_scores = scores_all.max(axis=1)
    keep = cls_scores >= conf_thres
    boxes_xywh = boxes_xywh[keep]
    cls_scores = cls_scores[keep]
    cls_ids = cls_ids[keep]
    if boxes_xywh.shape[0] == 0:
        return []
    # xywh -> xyxy
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:]
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    xyxy = np.concatenate([x1y1, x2y2], axis=1)
    # un-letterbox to original frame coords
    r, top, left = meta
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy /= r
    # NMS (torchvision)
    try:
        import torch
        from torchvision.ops import nms
        idx = nms(torch.from_numpy(xyxy).float(),
                  torch.from_numpy(cls_scores).float(),
                  iou_thres).cpu().numpy()
    except Exception:
        # naive fallback: keep top-300 by score
        idx = np.argsort(-cls_scores)[:300]
    return [(*xyxy[i].tolist(), float(cls_scores[i]), int(cls_ids[i])) for i in idx]


# ---------------------------------------------------------------------------
# Video reader (unchanged)
# ---------------------------------------------------------------------------

def video_reader_process(video_path, frame_queue, shutdown_event, max_queue_size=10):
    """Read frames from a video file and enqueue them with timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_reader] Failed to open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    period = 1.0 / fps
    next_t = time.time()
    try:
        while not shutdown_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# YOLO — NPU (Mobilint mblt_model_zoo)
# ---------------------------------------------------------------------------

def run_yolo_npu_process(input_queue, output_queue, shutdown_event,
                         npu_id: int = 0,
                         view_name: Optional[str] = None,
                         model_name: str = "yolov8n",
                         infer_mode: str = "global8",
                         conf_thres: float = 0.25,
                         iou_thres: float = 0.45):
    """Run a YOLO detection model on the Mobilint Aries NPU."""
    from runtime.npu_mobilint import build_model

    try:
        t0 = time.time()
        model = build_model(model_name, task="detection", infer_mode=infer_mode)
        load_ms = (time.time() - t0) * 1000.0
        log_model_load(pipeline="yolo", device=f"NPU{npu_id}", view=view_name,
                       model=model_name, model_load_time_ms=load_ms)
        try:
            while not shutdown_event.is_set():
                frame, enq_ts = _drain_item(input_queue)
                if frame is None:
                    continue
                t_pre0 = time.time()
                input_img = model.preprocess(frame)
                t_inf0 = time.time()
                raw = model(input_img)
                t_post0 = time.time()
                result = model.postprocess(raw, conf_thres=conf_thres, iou_thres=iou_thres)
                t_end = time.time()
                pre_ms = (t_inf0 - t_pre0) * 1000.0
                inf_ms = (t_post0 - t_inf0) * 1000.0
                post_ms = (t_end - t_post0) * 1000.0
                wait_ms = ((t_pre0 - enq_ts) * 1000.0) if enq_ts else 0.0
                log_inference(pipeline="yolo", device=f"NPU{npu_id}", view=view_name,
                              model=model_name,
                              preprocess_time_ms=pre_ms, inference_time_ms=inf_ms,
                              postprocess_time_ms=post_ms, wait_to_preprocess_ms=wait_ms)
                output_queue.put({"view": view_name, "model": model_name,
                                  "device": f"NPU{npu_id}", "frame": frame, "result": result,
                                  "timing_ms": {"wait": wait_ms, "pre": pre_ms,
                                                "infer": inf_ms, "post": post_ms}})
        finally:
            model.dispose()
    except Exception as e:
        print(f"[YOLO NPU{npu_id} view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# ResNet — NPU (Mobilint)
# ---------------------------------------------------------------------------

def run_resnet_npu_process(input_queue, output_queue, shutdown_event,
                           npu_id: int = 0,
                           view_name: Optional[str] = None,
                           model_name: str = "resnet50",
                           infer_mode: str = "global8",
                           topk: int = 5):
    """Run a ResNet classification model on the Mobilint Aries NPU."""
    from runtime.npu_mobilint import build_model

    try:
        t0 = time.time()
        model = build_model(model_name, task="classification", infer_mode=infer_mode)
        load_ms = (time.time() - t0) * 1000.0
        log_model_load(pipeline="resnet", device=f"NPU{npu_id}", view=view_name,
                       model=model_name, model_load_time_ms=load_ms)
        try:
            while not shutdown_event.is_set():
                frame, enq_ts = _drain_item(input_queue)
                if frame is None:
                    continue
                t_pre0 = time.time()
                input_img = model.preprocess(frame)
                t_inf0 = time.time()
                raw = model(input_img)
                t_post0 = time.time()
                result = model.postprocess(raw)
                t_end = time.time()
                pre_ms = (t_inf0 - t_pre0) * 1000.0
                inf_ms = (t_post0 - t_inf0) * 1000.0
                post_ms = (t_end - t_post0) * 1000.0
                wait_ms = ((t_pre0 - enq_ts) * 1000.0) if enq_ts else 0.0
                log_inference(pipeline="resnet", device=f"NPU{npu_id}", view=view_name,
                              model=model_name,
                              preprocess_time_ms=pre_ms, inference_time_ms=inf_ms,
                              postprocess_time_ms=post_ms, wait_to_preprocess_ms=wait_ms)
                output_queue.put({"view": view_name, "model": model_name,
                                  "device": f"NPU{npu_id}", "frame": frame,
                                  "result": result, "topk": topk,
                                  "timing_ms": {"wait": wait_ms, "pre": pre_ms,
                                                "infer": inf_ms, "post": post_ms}})
        finally:
            model.dispose()
    except Exception as e:
        print(f"[ResNet NPU{npu_id} view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# YOLO — CPU / GPU (ONNX Runtime)
# ---------------------------------------------------------------------------

def _run_yolo_ort(input_queue, output_queue, shutdown_event,
                  view_name, model_name, *, gpu: bool,
                  conf_thres=0.25, iou_thres=0.45):
    device = "GPU" if gpu else "CPU"
    onnx_path = resolve_onnx_path(model_name)
    t0 = time.time()
    sess = _ort_session(onnx_path, gpu=gpu)
    inp_name = sess.get_inputs()[0].name
    img_size = int(sess.get_inputs()[0].shape[2]) or 640
    load_ms = (time.time() - t0) * 1000.0
    log_model_load(pipeline="yolo", device=device, view=view_name,
                   model=model_name, model_load_time_ms=load_ms)

    while not shutdown_event.is_set():
        frame, enq_ts = _drain_item(input_queue)
        if frame is None:
            continue
        t_pre0 = time.time()
        input_arr, meta = _yolov8_letterbox(frame, size=img_size)
        t_inf0 = time.time()
        raw = sess.run(None, {inp_name: input_arr})
        t_post0 = time.time()
        result = _yolov8_postprocess(raw, meta, conf_thres=conf_thres, iou_thres=iou_thres)
        t_end = time.time()
        pre_ms = (t_inf0 - t_pre0) * 1000.0
        inf_ms = (t_post0 - t_inf0) * 1000.0
        post_ms = (t_end - t_post0) * 1000.0
        wait_ms = ((t_pre0 - enq_ts) * 1000.0) if enq_ts else 0.0
        log_inference(pipeline="yolo", device=device, view=view_name,
                      model=model_name,
                      preprocess_time_ms=pre_ms, inference_time_ms=inf_ms,
                      postprocess_time_ms=post_ms, wait_to_preprocess_ms=wait_ms)
        output_queue.put({"view": view_name, "model": model_name,
                          "device": device, "frame": frame, "result": result,
                          "timing_ms": {"wait": wait_ms, "pre": pre_ms,
                                        "infer": inf_ms, "post": post_ms}})


def run_yolo_cpu_process(input_queue, output_queue, shutdown_event,
                         view_name: Optional[str] = None,
                         model_name: str = "yolov8n",
                         conf_thres: float = 0.25,
                         iou_thres: float = 0.45):
    try:
        _run_yolo_ort(input_queue, output_queue, shutdown_event,
                      view_name, model_name, gpu=False,
                      conf_thres=conf_thres, iou_thres=iou_thres)
    except Exception as e:
        print(f"[YOLO CPU view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise


def run_yolo_gpu_process(input_queue, output_queue, shutdown_event,
                         view_name: Optional[str] = None,
                         model_name: str = "yolov8n",
                         conf_thres: float = 0.25,
                         iou_thres: float = 0.45):
    try:
        _run_yolo_ort(input_queue, output_queue, shutdown_event,
                      view_name, model_name, gpu=True,
                      conf_thres=conf_thres, iou_thres=iou_thres)
    except Exception as e:
        print(f"[YOLO GPU view={view_name}] FATAL: {type(e).__name__}: {e}")
        # Hard exit when GPU is requested but unavailable; matches old behavior.
        shutdown_event.set()
        os._exit(1)


# ---------------------------------------------------------------------------
# ResNet — CPU / GPU (ONNX Runtime)
# ---------------------------------------------------------------------------

def _resnet_preprocess(frame_bgr, size=224):
    h, w = frame_bgr.shape[:2]
    r = min(h, w)
    top, left = (h - r) // 2, (w - r) // 2
    crop = frame_bgr[top:top + r, left:left + r]
    resized = cv2.resize(crop, (size, size))
    rgb = resized[..., ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    arr = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return np.ascontiguousarray(arr)


def _run_resnet_ort(input_queue, output_queue, shutdown_event,
                    view_name, model_name, *, gpu: bool, topk=5):
    device = "GPU" if gpu else "CPU"
    onnx_path = resolve_onnx_path(model_name)
    t0 = time.time()
    sess = _ort_session(onnx_path, gpu=gpu)
    inp_name = sess.get_inputs()[0].name
    load_ms = (time.time() - t0) * 1000.0
    log_model_load(pipeline="resnet", device=device, view=view_name,
                   model=model_name, model_load_time_ms=load_ms)

    while not shutdown_event.is_set():
        frame, enq_ts = _drain_item(input_queue)
        if frame is None:
            continue
        t_pre0 = time.time()
        arr = _resnet_preprocess(frame)
        t_inf0 = time.time()
        logits = sess.run(None, {inp_name: arr})[0]
        t_post0 = time.time()
        probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
        top_idx = np.argsort(-probs[0])[:topk]
        result = [(imagenet_classes[i] if i < len(imagenet_classes) else str(i),
                   float(probs[0, i])) for i in top_idx]
        t_end = time.time()
        pre_ms = (t_inf0 - t_pre0) * 1000.0
        inf_ms = (t_post0 - t_inf0) * 1000.0
        post_ms = (t_end - t_post0) * 1000.0
        wait_ms = ((t_pre0 - enq_ts) * 1000.0) if enq_ts else 0.0
        log_inference(pipeline="resnet", device=device, view=view_name,
                      model=model_name,
                      preprocess_time_ms=pre_ms, inference_time_ms=inf_ms,
                      postprocess_time_ms=post_ms, wait_to_preprocess_ms=wait_ms)
        output_queue.put({"view": view_name, "model": model_name,
                          "device": device, "frame": frame, "result": result,
                          "timing_ms": {"wait": wait_ms, "pre": pre_ms,
                                        "infer": inf_ms, "post": post_ms}})


def run_resnet_cpu_process(input_queue, output_queue, shutdown_event,
                           view_name: Optional[str] = None,
                           model_name: str = "resnet50",
                           topk: int = 5):
    try:
        _run_resnet_ort(input_queue, output_queue, shutdown_event,
                        view_name, model_name, gpu=False, topk=topk)
    except Exception as e:
        print(f"[ResNet CPU view={view_name}] FATAL: {type(e).__name__}: {e}")
        raise


def run_resnet_gpu_process(input_queue, output_queue, shutdown_event,
                           view_name: Optional[str] = None,
                           model_name: str = "resnet50",
                           topk: int = 5):
    try:
        _run_resnet_ort(input_queue, output_queue, shutdown_event,
                        view_name, model_name, gpu=True, topk=topk)
    except Exception as e:
        print(f"[ResNet GPU view={view_name}] FATAL: {type(e).__name__}: {e}")
        shutdown_event.set()
        os._exit(1)
