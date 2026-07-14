"""Model processing functions for the multimodel scheduling application.

Rewritten for the Mobilint Aries NPU + NVIDIA GPU + CPU stack (replaces the old
Neubla `npu` integration). Each model runs on one of three devices:

    device ∈ {"cpu", "gpu", "npu"}

Process entry points (spawned by unified_viewer.start_view_process):
  - run_detection_process     -> emits (frame_with_boxes, infer_ms, wait_ms)
  - run_classification_process-> emits (frame_with_label, class_name, infer_ms)
  - run_llm_process           -> headless; emits (None, gen_ms, tokens_per_s)
  - run_vlm_process           -> headless; emits (None, gen_ms, tokens_per_s)

Heavy backends (torch, onnxruntime, mblt_model_zoo) are imported lazily inside
each process so the parent GUI process stays light and multiprocessing 'spawn'
does not double-load device libraries.
"""

import os
import queue
import time
from typing import Optional

import cv2
import numpy as np

import model_registry as reg
from timing_utils import log_inference, log_model_load

# ImageNet labels for classification display
try:
    with open("imagenet_classes.txt", "r") as f:
        imagenet_classes = [line.strip() for line in f.readlines()]
except Exception:
    imagenet_classes = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _report_view_error(output_queue, message):
    """Push a failure to the view so it renders a labelled placeholder.

    A worker that dies silently leaves a black tile, which in a demo is
    indistinguishable from the whole app being broken.
    """
    try:
        output_queue.put_nowait(("error", str(message)))
    except Exception:
        pass


def _drain_item(input_queue, timeout=1.0):
    """Pop one item, normalizing (frame, ts) vs frame."""
    try:
        item = input_queue.get(timeout=timeout)
    except queue.Empty:
        return None, None
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
        return item[0], item[1]
    return item, None


def _ort_session(onnx_path: str, gpu: bool):
    import onnxruntime as ort
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


# ---------------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------------

# Frames are downscaled before crossing the multiprocessing queue so that high
# input rates are not bottlenecked by serializing full-resolution frames. Each
# worker re-resizes to its own input size (224/640), so accuracy is unaffected.
FEED_MAX_WIDTH = 640


def _downscale(frame, max_width=FEED_MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def video_reader_process(video_path, frame_queue, shutdown_event, max_queue_size=10):
    """Read frames from a video file, downscale, and enqueue them (loops at EOF)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_reader] Failed to open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0
    try:
        while not shutdown_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                frame_queue.put_nowait(_downscale(frame))
            except queue.Full:
                pass
            time.sleep(period)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Detection (YOLOv8/v11) — CPU / GPU (ONNX) and NPU (Mobilint)
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


def _yolo_decode(raw_out, meta, conf_thres, iou_thres):
    """Decode YOLOv8/v11 ONNX output (1, 4+nc, N) -> [(x1,y1,x2,y2,score,cls)] in original coords."""
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
        idx = nms(torch.from_numpy(xyxy).float(), torch.from_numpy(cls_scores).float(), iou_thres).cpu().numpy()
    except Exception:
        idx = np.argsort(-cls_scores)[:300]
    return [(*xyxy[i].tolist(), float(cls_scores[i]), int(cls_ids[i])) for i in idx]


def _draw_dets(frame, dets):
    from image_processing import draw_detection_boxes
    for (x1, y1, x2, y2, score, cls) in dets:
        draw_detection_boxes(frame, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(score), int(cls))
    return frame


def run_detection_process(input_queue, output_queue, shutdown_event,
                          device="cpu", view_name=None, model_name="yolo11s",
                          conf_thres=0.25, iou_thres=0.45):
    device = reg.norm_device(device)
    spec = reg.get(model_name)
    size = int(spec.get("input_size", 640))
    try:
        if device == "npu":
            from runtime.mobilint_vision import build_vision_npu, npu_detections
            t0 = time.time()
            model = build_vision_npu(model_name, infer_mode="global8")
            log_model_load(pipeline="yolo", device="NPU", view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)
        else:
            from utils import resolve_onnx_path
            t0 = time.time()
            sess = _ort_session(resolve_onnx_path(model_name), gpu=(device == "gpu"))
            inp_name = sess.get_inputs()[0].name
            log_model_load(pipeline="yolo", device=device.upper(), view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)

        while not shutdown_event.is_set():
            frame, enq_ts = _drain_item(input_queue)
            if frame is None:
                continue
            wait_ms = ((time.time() - enq_ts) * 1000.0) if enq_ts else 0.0
            t_pre = time.time()
            if device == "npu":
                x = model.preprocess(frame)
                t_inf = time.time()
                raw = model(x)
                t_post = time.time()
                res = model.postprocess(raw, conf_thres=conf_thres, iou_thres=iou_thres)
                dets = npu_detections(res, frame)
            else:
                x, meta = _yolo_letterbox(frame, size)
                t_inf = time.time()
                raw = sess.run(None, {inp_name: x})
                t_post = time.time()
                dets = _yolo_decode(raw, meta, conf_thres, iou_thres)
            out_frame = _draw_dets(frame.copy(), dets)
            t_end = time.time()

            infer_ms = (t_post - t_inf) * 1000.0
            latency_ms = ((t_end - enq_ts) * 1000.0) if enq_ts else (
                wait_ms + (t_end - t_pre) * 1000.0)
            log_inference(pipeline="yolo", device=("NPU" if device == "npu" else device.upper()),
                          view=view_name, model=model_name,
                          preprocess_time_ms=(t_inf - t_pre) * 1000.0,
                          inference_time_ms=infer_ms,
                          postprocess_time_ms=(t_end - t_post) * 1000.0,
                          wait_to_preprocess_ms=wait_ms)
            try:
                output_queue.put_nowait((out_frame, infer_ms, wait_ms, latency_ms))
            except queue.Full:
                pass
    except Exception as e:
        print(f"[Detection {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        _report_view_error(output_queue, f"{type(e).__name__}: {e}")
        if device == "gpu":
            shutdown_event.set()
    finally:
        try:
            if device == "npu":
                model.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Classification (ResNet50) — CPU / GPU (ONNX) and NPU (Mobilint)
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


def run_classification_process(input_queue, output_queue, shutdown_event,
                               device="cpu", view_name=None, model_name="resnet50"):
    device = reg.norm_device(device)
    try:
        if device == "npu":
            from runtime.mobilint_vision import build_vision_npu, npu_top1
            t0 = time.time()
            model = build_vision_npu(model_name, infer_mode="global8")
            log_model_load(pipeline="resnet", device="NPU", view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)
        else:
            from utils import resolve_onnx_path
            t0 = time.time()
            sess = _ort_session(resolve_onnx_path(model_name), gpu=(device == "gpu"))
            inp_name = sess.get_inputs()[0].name
            log_model_load(pipeline="resnet", device=device.upper(), view=view_name,
                           model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)

        while not shutdown_event.is_set():
            frame, enq_ts = _drain_item(input_queue)
            if frame is None:
                continue
            wait_ms = ((time.time() - enq_ts) * 1000.0) if enq_ts else 0.0
            t_pre = time.time()
            if device == "npu":
                x = model.preprocess(frame)
                t_inf = time.time()
                raw = model(x)
                t_post = time.time()
                res = model.postprocess(raw)
                class_id, _ = npu_top1(res)
            else:
                x = _resnet_preprocess(frame)
                t_inf = time.time()
                logits = sess.run(None, {inp_name: x})[0]
                t_post = time.time()
                class_id = int(np.argmax(np.squeeze(logits)))
            class_name = _label(class_id)
            # top-5 for the demo's bar chart; top-1 stays authoritative for the stats.
            try:
                if device == "npu":
                    from runtime.mobilint_vision import npu_topk
                    top5 = [(_label(cid), p) for cid, p in npu_topk(res, 5)]
                else:
                    s = np.squeeze(logits).astype(np.float64)
                    e = np.exp(s - s.max())
                    probs = e / max(e.sum(), 1e-9)
                    top5 = [(_label(int(i)), float(probs[i]))
                            for i in np.argsort(probs)[::-1][:5]]
            except Exception as e:
                print(f"[Classification view={view_name}] top-5 unavailable: {e}")
                top5 = [(class_name, 1.0)]
            out = frame.copy()
            t_end = time.time()

            infer_ms = (t_post - t_inf) * 1000.0
            latency_ms = ((t_end - enq_ts) * 1000.0) if enq_ts else (
                wait_ms + (t_end - t_pre) * 1000.0)
            log_inference(pipeline="resnet", device=("NPU" if device == "npu" else device.upper()),
                          view=view_name, model=model_name,
                          preprocess_time_ms=(t_inf - t_pre) * 1000.0,
                          inference_time_ms=infer_ms,
                          postprocess_time_ms=(t_end - t_post) * 1000.0,
                          wait_to_preprocess_ms=wait_ms)
            try:
                output_queue.put_nowait((out, top5, infer_ms, latency_ms))
            except queue.Full:
                pass
    except Exception as e:
        print(f"[Classification {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        # Tell the view so it can show "Load failed" instead of staying black.
        _report_view_error(output_queue, f"{type(e).__name__}: {e}")
        if device == "gpu":
            shutdown_event.set()
    finally:
        try:
            if device == "npu":
                model.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LLM / VLM — token-streaming workers
# ---------------------------------------------------------------------------

def _run_generative(input_queue, output_queue, shutdown_event, device, view_name,
                    model_name, infps, max_new_tokens, is_vlm, prompt_override=None):
    """Generate, streaming each decoded piece to the view as it arrives.

    Emits tagged messages so the view can type the answer out rather than receive it
    all at once (a completed paragraph appearing in one frame reads as a frozen view):
        ("start", frame_or_None, prompt)  -- new generation begins
        ("token", piece)                  -- one decoded piece
        ("done", gen_ms, tokens_per_s, latency_ms)
    Stats are unchanged: ("done",) carries what the old (None, gen_ms, tok_s, lat)
    tuple carried.
    """
    device = reg.norm_device(device)
    from runtime.llm_engine import LLMEngine
    t0 = time.time()
    engine = LLMEngine(model_name, device, max_new_tokens=max_new_tokens)
    log_model_load(pipeline=("vlm" if is_vlm else "llm"), device=("NPU" if device == "npu" else device.upper()),
                   view=view_name, model=model_name, model_load_time_ms=(time.time() - t0) * 1000.0)

    interval = (1.0 / infps) if (infps and infps > 0) else 0.0
    from runtime.llm_engine import LLM_PROMPTS, DEFAULT_PROMPT
    prompts = [prompt_override] if prompt_override else LLM_PROMPTS
    idx = 0
    last = 0.0

    def _emit(msg):
        try:
            output_queue.put_nowait(msg)
        except queue.Full:
            pass  # a view that fell behind gets the next token, not a stall

    while not shutdown_event.is_set():
        now = time.time()
        if interval and (now - last) < interval:
            time.sleep(min(0.02, interval))
            continue
        frame = None
        enq_ts = None
        if is_vlm:
            frame, enq_ts = _drain_item(input_queue, timeout=0.5)
            if frame is None:
                continue
        if is_vlm:
            prompt = prompt_override or DEFAULT_PROMPT
        else:
            prompt = prompts[idx % len(prompts)]
        idx += 1
        t_start = time.time()
        wait_ms = ((t_start - enq_ts) * 1000.0) if enq_ts else 0.0

        _emit(("start", frame, prompt))
        try:
            r = engine.infer_stream(lambda piece: _emit(("token", piece)),
                                    prompt=(None if is_vlm and not prompt_override else prompt),
                                    frame=frame, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"[{'VLM' if is_vlm else 'LLM'} {device} view={view_name}] infer error: {e}")
            _emit(("error", f"generation failed: {e}"))
            continue
        last = time.time()
        gen_ms = r["total_ms"]
        tok_s = r["tokens_per_s"]
        latency_ms = wait_ms + gen_ms
        log_inference(pipeline=("vlm" if is_vlm else "llm"),
                      device=("NPU" if device == "npu" else device.upper()),
                      view=view_name, model=model_name,
                      preprocess_time_ms=0.0, inference_time_ms=gen_ms,
                      postprocess_time_ms=0.0, wait_to_preprocess_ms=wait_ms)
        _emit(("done", gen_ms, tok_s, latency_ms))
    engine.dispose()


def run_llm_process(input_queue, output_queue, shutdown_event,
                    device="gpu", view_name=None, model_name="llama1b",
                    infps=1.0, max_new_tokens=64, prompt=None):
    try:
        _run_generative(input_queue, output_queue, shutdown_event, device, view_name,
                        model_name, infps, max_new_tokens, is_vlm=False,
                        prompt_override=prompt)
    except Exception as e:
        print(f"[LLM {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        _report_view_error(output_queue, f"{type(e).__name__}: {e}")
        if reg.norm_device(device) == "gpu":
            shutdown_event.set()


def run_vlm_process(input_queue, output_queue, shutdown_event,
                    device="gpu", view_name=None, model_name="qwen2_vl",
                    infps=1.0, max_new_tokens=32, prompt=None):
    try:
        _run_generative(input_queue, output_queue, shutdown_event, device, view_name,
                        model_name, infps, max_new_tokens, is_vlm=True,
                        prompt_override=prompt)
    except Exception as e:
        print(f"[VLM {device} view={view_name}] FATAL: {type(e).__name__}: {e}")
        _report_view_error(output_queue, f"{type(e).__name__}: {e}")
        if reg.norm_device(device) == "gpu":
            shutdown_event.set()
