"""Per-view inference workers (vision) for the new model set + MLA100 NPU.

Ported from the mobilint runtime backend but kept behind FSRR's existing worker
contract so the four-strategy dispatch (Static / Stop-and-restart / Adaptive
hot-swap / BoundGuard), the QoS V(t) loop, and the view handlers are unchanged:

  - workers run as threading.Thread and take a `ready_event` set AFTER the model
    is loaded (the adaptive hot-swap in adaptive_deploy._hot_swap_view blocks on it);
  - detection puts a 3-tuple  (frame, infer_ms, wait_ms);
  - classification puts a 4-tuple (frame, class_name:str, infer_ms, wait_ms).

Model choice is driven by `model_registry` (task/pipeline/onnx/mxq), not by a
hardcoded name: detection -> YOLO worker, classification -> ResNet/MobileNet worker.
CPU/GPU run ONNX (onnxruntime); NPU runs the Mobilint .mxq via runtime.mobilint_vision.

LLM/VLM (llama1b, qwen2_vl) workers are intentionally out of scope here (deferred):
routing skips generative views rather than starting a worker for them.
"""
import time
import os
import queue
from threading import Thread  # noqa: F401  (kept for callers importing from here)

import cv2
import numpy as np
import onnxruntime as ort

import model_registry as reg
from image_processing import draw_detection_boxes
from timing_utils import log_model_load, log_inference

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]


# ---------------------------------------------------------------------------
# Video reader (unchanged from FSRR: full-resolution frames, bare put)
# ---------------------------------------------------------------------------
def video_reader_process(video_path, frame_queue, shutdown_event, max_queue_size=10):
    """Read video frames and enqueue them (loops at EOF)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video Reader ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if shutdown_event.is_set():
            break
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        except (BrokenPipeError, EOFError, OSError) as e:
            print(f"[Video Reader] Output queue closed: {e}. Stopping video reader.")
            break
        except Exception as e:
            print(f"[Video Reader] Unexpected put error: {e}. Stopping video reader.")
            break
        time.sleep(frame_delay)

    cap.release()


# ---------------------------------------------------------------------------
# Shared helpers (ported from the mobilint runtime)
# ---------------------------------------------------------------------------
def _to_rgb(frame_bgr):
    """The Mobilint zoo preprocessor expects RGB; OpenCV hands us BGR."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _drain_item(input_queue, timeout=1.0):
    """Pop one item, normalizing (frame, ts) vs a bare frame."""
    try:
        item = input_queue.get(timeout=timeout)
    except queue.Empty:
        return None, None
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
        return item[0], item[1]
    return item, None


def _ort_session(onnx_path: str, gpu: bool):
    so = ort.SessionOptions()
    try:
        so.log_severity_level = 3
    except Exception:
        pass
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    if gpu and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"CUDA EP unavailable for {onnx_path}")
    return sess


def _limit_cpu_threads():
    """One torch thread per vision worker (NMS only); concurrency comes from
    running the views in parallel, not from threading inside one worker."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception as e:
        print(f"[worker] could not pin torch threads: {e}")


def _onnx_path(model_name: str) -> str:
    """CPU/GPU ONNX path for a model, from the registry (falls back to models/onnx)."""
    spec = reg.get(model_name) or {}
    return spec.get("onnx") or os.path.join("models", "onnx", f"{model_name}.onnx")


# ---------------------------------------------------------------------------
# Detection (YOLOv8/v11) — CPU / GPU (ONNX) and NPU (Mobilint .mxq)
# ---------------------------------------------------------------------------
def _yolo_letterbox(frame_bgr, size):
    h0, w0 = frame_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    arr = canvas[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(arr[None, ...]), (r, top, left)


def _letterbox_meta(frame_bgr, size):
    h0, w0 = frame_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    return (r, (size - nh) // 2, (size - nw) // 2)


def _npu_yolo_raw_to_pred(raw, reg_max=16, nc=80):
    """Mobilint YOLO head outputs -> (1, 4+nc, N) tensor, the same shape the ONNX
    export gives, so both devices run the identical decoder below."""
    dets = [np.asarray(t) for t in raw if np.asarray(t).shape[-1] == reg_max * 4]
    clss = [np.asarray(t) for t in raw if np.asarray(t).shape[-1] == nc]
    dets.sort(key=lambda a: a.size, reverse=True)
    clss.sort(key=lambda a: a.size, reverse=True)
    if len(dets) != len(clss):
        raise ValueError(f"NPU head mismatch: {len(dets)} box vs {len(clss)} cls tensors")

    bins = np.arange(reg_max, dtype=np.float32)
    boxes, scores = [], []
    for d, c in zip(dets, clss):
        h, w = d.shape[0], d.shape[1]
        stride = 640.0 / h
        b = d.reshape(-1, 4, reg_max).astype(np.float32)
        b -= b.max(axis=2, keepdims=True)
        np.exp(b, out=b)
        b /= b.sum(axis=2, keepdims=True)
        ltrb = b @ bins
        ys, xs = np.mgrid[0:h, 0:w]
        ax = (xs.reshape(-1) + 0.5).astype(np.float32)
        ay = (ys.reshape(-1) + 0.5).astype(np.float32)
        x1 = (ax - ltrb[:, 0]) * stride
        y1 = (ay - ltrb[:, 1]) * stride
        x2 = (ax + ltrb[:, 2]) * stride
        y2 = (ay + ltrb[:, 3]) * stride
        boxes.append(np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1))
        s = c.reshape(-1, nc).astype(np.float32)
        scores.append(1.0 / (1.0 + np.exp(-s)))

    pred = np.concatenate([np.concatenate(boxes, axis=0),
                           np.concatenate(scores, axis=0)], axis=1)
    return pred.T[None, ...]


def _yolo_decode(raw_out, meta, conf_thres, iou_thres):
    """Decode YOLOv8/v11 (1, 4+nc, N) -> [(x1,y1,x2,y2,score,cls)] in original coords."""
    out = raw_out[0]
    if out.ndim == 3 and out.shape[1] < out.shape[2]:
        pred = out[0].transpose(1, 0)
    else:
        pred = out[0]
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    cls_scores = scores_all.max(axis=1)
    keep = cls_scores >= conf_thres
    boxes_xywh, cls_scores, cls_ids = boxes_xywh[keep], cls_scores[keep], cls_ids[keep]
    if boxes_xywh.shape[0] == 0:
        return []
    xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:]
    xyxy = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1)
    r, top, left = meta
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy /= r
    try:
        import torch
        from torchvision.ops import nms
        idx = nms(torch.from_numpy(xyxy).float(),
                  torch.from_numpy(cls_scores).float(), iou_thres).cpu().numpy()
    except Exception:
        idx = np.argsort(-cls_scores)[:300]
    return [(*xyxy[i].tolist(), float(cls_scores[i]), int(cls_ids[i])) for i in idx]


def _draw_dets(frame, dets):
    for (x1, y1, x2, y2, score, cls) in dets:
        draw_detection_boxes(frame, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                             float(score), int(cls))
    return frame


def _detection_worker(device, input_queue, output_queue, shutdown_event,
                      view_name=None, model_name="yolo11s", ready_event=None,
                      conf_thres=0.25, iou_thres=0.45):
    """Shared detection loop. Emits FSRR's (frame, infer_ms, wait_ms) 3-tuple."""
    _limit_cpu_threads()
    device = reg.norm_device(device)
    spec = reg.get(model_name) or {}
    size = int(spec.get("input_size", 640))
    model = None
    try:
        if device == "npu":
            from runtime.mobilint_vision import build_vision_npu
            t0 = time.time()
            model = build_vision_npu(model_name, infer_mode="global8")
            log_model_load(pipeline="yolo", device="NPU", view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)
        else:
            t0 = time.time()
            sess = _ort_session(_onnx_path(model_name), gpu=(device == "gpu"))
            inp_name = sess.get_inputs()[0].name
            log_model_load(pipeline="yolo", device=device.upper(), view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)

        if ready_event is not None:
            ready_event.set()

        while not shutdown_event.is_set():
            frame, enq_ts = _drain_item(input_queue)
            if frame is None:
                continue
            wait_ms = ((time.time() - enq_ts) * 1000.0) if enq_ts else 0.0
            t_pre = time.time()
            if device == "npu":
                x = model.preprocess(_to_rgb(frame))
                t_inf = time.time()
                raw = model(x)
                t_post = time.time()
                dets = _yolo_decode([_npu_yolo_raw_to_pred(raw)],
                                    _letterbox_meta(frame, size), conf_thres, iou_thres)
            else:
                x, meta = _yolo_letterbox(frame, size)
                t_inf = time.time()
                raw = sess.run(None, {inp_name: x})
                t_post = time.time()
                dets = _yolo_decode(raw, meta, conf_thres, iou_thres)
            out_frame = _draw_dets(frame.copy(), dets)
            t_end = time.time()

            infer_ms = (t_post - t_inf) * 1000.0
            log_inference(pipeline="yolo",
                          device=("NPU" if device == "npu" else device.upper()),
                          view=view_name, model=model_name,
                          preprocess_time_ms=(t_inf - t_pre) * 1000.0,
                          inference_time_ms=infer_ms,
                          postprocess_time_ms=(t_end - t_post) * 1000.0,
                          wait_to_preprocess_ms=wait_ms)
            try:
                output_queue.put((out_frame, infer_ms, wait_ms))
            except (BrokenPipeError, EOFError, OSError):
                break
    except Exception as e:
        print(f"[Detection {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        if ready_event is not None:
            ready_event.set()  # unblock hot-swap watcher even on failure
    finally:
        try:
            if model is not None:
                model.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Classification (ResNet50 / MobileNet_V2) — CPU / GPU (ONNX) and NPU (Mobilint)
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
    arr = ((rgb - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
    return np.ascontiguousarray(arr)


def _label(class_id):
    if 0 <= class_id < len(imagenet_classes):
        return imagenet_classes[class_id]
    return f"Class ID: {class_id}"


def _classification_worker(device, input_queue, output_queue, shutdown_event,
                           view_name=None, model_name="resnet50", ready_event=None):
    """Shared classification loop. Emits FSRR's (frame, class_name, infer_ms, wait_ms)."""
    _limit_cpu_threads()
    device = reg.norm_device(device)
    spec = reg.get(model_name) or {}
    size = int(spec.get("input_size", 224))
    model = None
    try:
        if device == "npu":
            from runtime.mobilint_vision import build_vision_npu
            t0 = time.time()
            model = build_vision_npu(model_name, infer_mode="global8")
            log_model_load(pipeline="resnet", device="NPU", view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)
        else:
            t0 = time.time()
            sess = _ort_session(_onnx_path(model_name), gpu=(device == "gpu"))
            inp_name = sess.get_inputs()[0].name
            log_model_load(pipeline="resnet", device=device.upper(), view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)

        if ready_event is not None:
            ready_event.set()

        while not shutdown_event.is_set():
            frame, enq_ts = _drain_item(input_queue)
            if frame is None:
                continue
            wait_ms = ((time.time() - enq_ts) * 1000.0) if enq_ts else 0.0
            t_pre = time.time()
            if device == "npu":
                from runtime.mobilint_vision import npu_top1
                x = model.preprocess(_to_rgb(frame))
                t_inf = time.time()
                raw = model(x)
                t_post = time.time()
                res = model.postprocess(raw)
                class_id, _ = npu_top1(res)
            else:
                x = _resnet_preprocess(frame, size)
                t_inf = time.time()
                logits = sess.run(None, {inp_name: x})[0]
                t_post = time.time()
                class_id = int(np.argmax(np.squeeze(logits)))
            class_name = _label(class_id)
            out = frame.copy()
            t_end = time.time()

            infer_ms = (t_post - t_inf) * 1000.0
            log_inference(pipeline="resnet",
                          device=("NPU" if device == "npu" else device.upper()),
                          view=view_name, model=model_name,
                          preprocess_time_ms=(t_inf - t_pre) * 1000.0,
                          inference_time_ms=infer_ms,
                          postprocess_time_ms=(t_end - t_post) * 1000.0,
                          wait_to_preprocess_ms=wait_ms)
            try:
                output_queue.put((out, class_name, infer_ms, wait_ms))
            except (BrokenPipeError, EOFError, OSError):
                break
    except Exception as e:
        print(f"[Classification {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        if ready_event is not None:
            ready_event.set()
    finally:
        try:
            if model is not None:
                model.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FSRR-signature wrappers (names/args the dispatch layer calls; device fixed here)
# ---------------------------------------------------------------------------
def run_yolo_cpu_process(input_queue, output_queue, shutdown_event,
                         view_name=None, ready_event=None, model_name="yolo11s"):
    _detection_worker("cpu", input_queue, output_queue, shutdown_event,
                      view_name=view_name, model_name=model_name, ready_event=ready_event)


def run_yolo_gpu_process(input_queue, output_queue, shutdown_event,
                         view_name=None, model_name="yolo11s", ready_event=None):
    _detection_worker("gpu", input_queue, output_queue, shutdown_event,
                      view_name=view_name, model_name=model_name, ready_event=ready_event)


def run_yolo_npu_process(input_queue, output_queue, shutdown_event,
                         npu_id=0, view_name=None, model_name="yolo11s", ready_event=None):
    _detection_worker("npu", input_queue, output_queue, shutdown_event,
                      view_name=view_name, model_name=model_name, ready_event=ready_event)


def run_resnet_cpu_process(input_queue, output_queue, shutdown_event,
                           view_name=None, ready_event=None, model_name="resnet50"):
    _classification_worker("cpu", input_queue, output_queue, shutdown_event,
                           view_name=view_name, model_name=model_name, ready_event=ready_event)


def run_resnet_gpu_process(input_queue, output_queue, shutdown_event,
                           view_name=None, ready_event=None, model_name="resnet50"):
    _classification_worker("gpu", input_queue, output_queue, shutdown_event,
                           view_name=view_name, model_name=model_name, ready_event=ready_event)


def run_resnet_npu_process(input_queue, output_queue, shutdown_event,
                           npu_id=1, view_name=None, model_name="resnet50", ready_event=None):
    _classification_worker("npu", input_queue, output_queue, shutdown_event,
                           view_name=view_name, model_name=model_name, ready_event=ready_event)


# ---------------------------------------------------------------------------
# Routing helpers (registry-driven; replace the old "yolov4" substring checks)
# ---------------------------------------------------------------------------
def classify_view(model_name):
    """'yolo' (detection) or 'resnet' (classification) for a vision model, or None
    for generative (llm/vlm — deferred) / unknown models. Replaces `"yolov4" in model`."""
    try:
        if reg.kind_of(model_name) != "vision":
            return None
        spec = reg.get(model_name) or {}
        return "yolo" if spec.get("pipeline") == "yolo" else "resnet"
    except Exception:
        return None


_WORKER_TABLE = {
    ("yolo", "cpu"): run_yolo_cpu_process,
    ("yolo", "gpu"): run_yolo_gpu_process,
    ("yolo", "npu"): run_yolo_npu_process,
    ("resnet", "cpu"): run_resnet_cpu_process,
    ("resnet", "gpu"): run_resnet_gpu_process,
    ("resnet", "npu"): run_resnet_npu_process,
}


def worker_target(model_name, execution):
    """Return (worker_fn, kind, device) for a (model, execution) pair.

    worker_fn is None when the view is generative (llm/vlm — deferred here) or the
    model is unknown; kind is 'yolo'/'resnet'/None; device is cpu/gpu/npu. Call the
    returned fn as fn(iq, oq, se, view_name=..., model_name=..., ready_event=...).
    """
    try:
        device = reg.norm_device(execution)
    except Exception:
        device = "cpu"
    kind = classify_view(model_name)
    if kind is None:
        return None, None, device
    return _WORKER_TABLE.get((kind, device)), kind, device
