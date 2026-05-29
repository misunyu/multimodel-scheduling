"""Shared infrastructure for Step D1 (capacity) and Step D2 (placement).

Implements:
  - per-stream foreground worker thread (GPU/CPU/NPU) that runs the streaming
    simulation and records (timestamps, input_fidx, parsed detections, infer_ms)
  - background load workers (L0/L1/L2/L3) running on GPU
  - per-stream sAP eval (with warmup-frame skip)

Threading throughout. Each foreground stream gets its own ultralytics YOLO
instance for GPU/CPU (load is small and avoids cross-thread races); the NPU
runner is shared (single Aries chip).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if torch.cuda.is_available():
    torch.cuda.init()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
from step0_compare_devices import ANNOT, DATA, FPS, IMG_SIZE, CONF, IOU, N_AHD  # noqa: E402

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO  # noqa: E402

WARMUP_FRAMES = 30


# ---------- annotation slicing ----------------------------------------------

def load_val():
    with open(ANNOT) as f:
        return json.load(f)


def load_split_for_sid(val, sid):
    imgs = sorted([i for i in val["images"] if i["sid"] == sid], key=lambda x: x["fid"])
    img_ids = {i["id"] for i in imgs}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    coco_mapping = np.asarray(val["coco_mapping"])
    seq_dir = val["seq_dirs"][sid]
    log_name = val["sequences"][sid]
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": val.get("info", {}), "licenses": val.get("licenses", []),
        "categories": val["categories"], "images": imgs, "annotations": anns,
    }
    coco_gt.createIndex()
    return {"imgs": imgs, "img_ids": img_ids, "coco_mapping": coco_mapping,
            "seq_dir": seq_dir, "log_name": log_name, "coco_gt": coco_gt}


# ---------- foreground inference adapters -----------------------------------

class FGModelGPU:
    """Per-stream Ultralytics YOLO instance on CUDA."""
    def __init__(self):
        self.m = YOLO("yolo11s.pt")
        # warmup forces device init
        self.m.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                       imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False, device="cuda")
    def predict(self, img_path):
        return self.m.predict(img_path, imgsz=IMG_SIZE, conf=CONF, iou=IOU,
                              verbose=False, device="cuda")[0]

class FGModelCPU:
    def __init__(self):
        self.m = YOLO("yolo11s.pt")
        self.m.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                       imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False, device="cpu")
    def predict(self, img_path):
        return self.m.predict(img_path, imgsz=IMG_SIZE, conf=CONF, iou=IOU,
                              verbose=False, device="cpu")[0]


_NPU_LOAD_LOCK = threading.Lock()
_NPU_INSTANCES: list = []  # per-stream independent Mobilint YOLO11s engines

def _find_yolo11s_mxq(infer_mode="multi"):
    """Locate yolo11s.mxq for the requested infer_mode. The mblt HF layout puts
    mode-specific compilations under `aries/<mode>/yolo11s.mxq`; falls back to
    the default un-suffixed file and lastly to models/mobilint/."""
    import glob
    cands = [
        *glob.glob(str(Path.home() / f".cache/huggingface/hub/models--mobilint--YOLO11s/snapshots/*/aries/{infer_mode}/yolo11s.mxq")),
        *glob.glob(str(Path.home() / ".cache/huggingface/hub/models--mobilint--YOLO11s/snapshots/*/aries/yolo11s.mxq")),
        "models/mobilint/yolo11s.mxq",
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def preload_npu_instances(n, infer_mode="multi"):
    """Pre-load `n` independent Mobilint YOLO11s engines so each stream can run
    truly concurrently on the NPU. Mobilint Aries supports multiple models
    in-flight; `infer_mode='multi'` partitions across two clusters which is
    well-suited to 2+ concurrent streams."""
    global _NPU_INSTANCES
    with _NPU_LOAD_LOCK:
        while len(_NPU_INSTANCES) < n:
            from mblt_model_zoo.vision import YOLO11s
            mxq = _find_yolo11s_mxq(infer_mode)
            m = YOLO11s(local_path=mxq, infer_mode=infer_mode, product="aries")
            # warmup
            dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
            x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
            _NPU_INSTANCES.append(m)
    return _NPU_INSTANCES[:n]


def get_npu_model(slot=0):
    """Get the NPU engine for a given stream slot (lazily creates if needed)."""
    if slot >= len(_NPU_INSTANCES):
        preload_npu_instances(slot + 1)
    return _NPU_INSTANCES[slot]


def npu_infer(img_path, frame_shape, npu_model):
    """NPU inference returning (xyxy, scores, coco_class_ids). Each caller
    passes its own dedicated mblt engine instance — Mobilint Aries handles
    multiple concurrent inference contexts."""
    import cv2
    img = cv2.imread(str(img_path))
    m = npu_model
    x = m.preprocess(img)
    out = m(x)
    res = m.postprocess(out, conf_thres=CONF, iou_thres=IOU)
    box_cls = getattr(res, "box_cls", None)
    if box_cls is None or box_cls.shape[0] == 0:
        return np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,int)
    arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
    h0, w0 = frame_shape[:2]
    gain = min(IMG_SIZE/h0, IMG_SIZE/w0)
    pad_x = (IMG_SIZE - w0*gain)/2.0; pad_y = (IMG_SIZE - h0*gain)/2.0
    xyxy = arr[:, :4].astype(np.float32, copy=True)
    xyxy[:, [0,2]] -= pad_x; xyxy[:, [1,3]] -= pad_y; xyxy /= gain
    np.clip(xyxy[:, [0,2]], 0, w0, out=xyxy[:, [0,2]])
    np.clip(xyxy[:, [1,3]], 0, h0, out=xyxy[:, [1,3]])
    return xyxy, arr[:, 4].astype(np.float32), arr[:, 5].astype(int)


# ---------- foreground streaming worker -------------------------------------

def fg_worker(stream_id, device, split, model_or_none, result, stop_event):
    """One foreground stream's streaming simulation. Records all the data
    needed to compute per-stream sAP afterwards."""
    imgs = split["imgs"]
    coco_mapping = split["coco_mapping"]
    seq_dir = split["seq_dir"]
    n_frame = len(imgs)
    t_total = n_frame / FPS
    t_elapsed = 0.0
    last_fidx = -1
    # cache frame shape from first image meta
    frame_shape = (imgs[0]["height"], imgs[0]["width"])

    while t_elapsed < t_total and not stop_event.is_set():
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last_fidx:
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS
        if fidx >= n_frame:
            break
        last_fidx = fidx
        img_path = DATA / seq_dir / imgs[fidx]["name"]
        wait_start = time.time()  # also-measure "wait" implicitly via streaming
        t0 = time.time()
        if device in ("GPU", "CPU"):
            r = model_or_none.predict(str(img_path))
            if len(r.boxes):
                coco_ids = r.boxes.cls.cpu().numpy().astype(int)
                ahd = coco_mapping[coco_ids]
                sel = ahd < N_AHD
                bb = r.boxes.xyxy.cpu().numpy()[sel].astype(np.float32)
                sc = r.boxes.conf.cpu().numpy()[sel].astype(np.float32)
                lb = ahd[sel].astype(np.int32)
            else:
                bb = np.zeros((0,4),np.float32); sc=np.zeros(0,np.float32); lb=np.zeros(0,np.int32)
        elif device == "NPU":
            xyxy_coco, scores, coco_cls = npu_infer(img_path, frame_shape, model_or_none)
            if len(xyxy_coco):
                ahd = coco_mapping[coco_cls]
                sel = ahd < N_AHD
                bb = xyxy_coco[sel]; sc = scores[sel]; lb = ahd[sel].astype(np.int32)
            else:
                bb = np.zeros((0,4),np.float32); sc=np.zeros(0,np.float32); lb=np.zeros(0,np.int32)
        else:
            raise ValueError(f"unknown device {device}")
        rt = time.time() - t0
        eff_rt = time.time() - wait_start  # same as rt in our flow but kept for clarity
        t_elapsed += rt
        result["infer_ms"].append(rt * 1000)
        result["eff_ms"].append(eff_rt * 1000)
        result["timestamps"].append(t_elapsed)
        result["input_fidx"].append(fidx)
        result["results"].append((bb, sc, lb))


# ---------- per-stream sAP eval (warmup-skipped) ----------------------------

def per_stream_sap(split, result, warmup_frames=WARMUP_FRAMES):
    imgs = split["imgs"]
    coco_gt = split["coco_gt"]
    timestamps = result["timestamps"]
    input_fidx = result["input_fidx"]
    results_parsed = result["results"]

    # Pair detections; only count GT frames AFTER warmup
    ccf = []
    miss = 0
    in_time = 0
    tidx_p1 = 0
    n_eval = 0
    for ii, img in enumerate(imgs):
        t_gt = ii / FPS
        while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t_gt:
            tidx_p1 += 1
        if ii < warmup_frames:
            continue
        n_eval += 1
        if tidx_p1 == 0:
            miss += 1
            continue
        tidx = tidx_p1 - 1
        bb, sc, lb = results_parsed[tidx]
        if input_fidx[tidx] == ii: in_time += 1
        for k in range(len(bb)):
            x1,y1,x2,y2 = bb[k]
            ccf.append({"image_id": int(img["id"]),
                        "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)],
                        "score": float(sc[k]), "category_id": int(lb[k])})

    # Frame skip = inferences whose effective latency exceeds frame period (33.3ms @30FPS)
    # — only counted after warmup
    eff_post = result["eff_ms"][WARMUP_FRAMES:] if len(result["eff_ms"]) > WARMUP_FRAMES else []
    infer_post = result["infer_ms"][WARMUP_FRAMES:] if len(result["infer_ms"]) > WARMUP_FRAMES else []
    period_ms = 1000.0 / FPS
    n_over = sum(1 for x in eff_post if x > period_ms) if eff_post else 0
    out = {"miss": miss, "in_time": in_time, "n_eval": n_eval,
           "frame_skip_pct": 100.0 * n_over / len(eff_post) if eff_post else 0.0,
           "infer_mean_ms": float(np.mean(infer_post)) if infer_post else 0.0,
           "infer_p95_ms": float(np.percentile(infer_post, 95)) if infer_post else 0.0,
           "eff_mean_ms": float(np.mean(eff_post)) if eff_post else 0.0,
           "eff_p95_ms": float(np.percentile(eff_post, 95)) if eff_post else 0.0,
           "n_processed": len(result["results"])}
    if not ccf:
        out.update({"sap_5095": 0.0, "sap_50": 0.0, "sap_small": 0.0, "sap_medium": 0.0, "sap_large": 0.0})
        return out

    # Eval only over warmup-skipped image_ids
    img_ids_eval = sorted({img["id"] for img in imgs[warmup_frames:]})
    coco_dt = coco_gt.loadRes(ccf)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = img_ids_eval
    e.evaluate(); e.accumulate(); e.summarize()
    out.update({"sap_5095": float(e.stats[0]), "sap_50": float(e.stats[1]),
                "sap_small": float(e.stats[3]), "sap_medium": float(e.stats[4]),
                "sap_large": float(e.stats[5])})
    return out


# ---------- background load workers -----------------------------------------

# Pre-loaded background models (must be loaded in this order: ORT CUDA sessions
# BEFORE any torch CUDA model — otherwise ORT 1.20 fails to init CUDA EP because
# torch+cu13 changes the cuDNN library state). preload_background_models() is the
# single entry point that respects this order.

_BG_PRELOADED = {}  # {"resnet": (sess, feeds), "tinyllama": (sess, feeds), "qwen2vl": (model, inputs)}


def preload_background_models(max_level="L3"):
    """Pre-load all bg models in a CUDA-safe order. Call once at startup."""
    levels = {"L0": [], "L1": ["resnet"], "L2": ["resnet","tinyllama"],
              "L3": ["resnet","tinyllama","qwen2vl"]}
    needed = levels[max_level]

    if "resnet" in needed and "resnet" not in _BG_PRELOADED:
        sess = ort.InferenceSession("models/onnx/resnet50.onnx",
                                    providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        feed = {name: np.random.randn(1, 3, 224, 224).astype(np.float32)}
        sess.run(None, feed)  # warmup
        _BG_PRELOADED["resnet"] = (sess, feed)
        print("[bg-preload] ResNet50 ORT-CUDA ready")

    if "tinyllama" in needed and "tinyllama" not in _BG_PRELOADED:
        sess = ort.InferenceSession("models/onnx/tiny-llama-chat-onnx/model.onnx",
                                    providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        seq_len = 32
        feeds = {
            "input_ids": np.random.randint(1, 30000, size=(1, seq_len), dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }
        for inp in sess.get_inputs():
            if inp.name.startswith("past_key_values."):
                feeds[inp.name] = np.zeros((1, 4, 0, 64), dtype=np.float32)
        sess.run(None, feeds)  # warmup
        _BG_PRELOADED["tinyllama"] = (sess, feeds)
        print("[bg-preload] TinyLLaMA ORT-CUDA ready")

    # Torch CUDA model goes LAST (after ORT)
    if "qwen2vl" in needed and "qwen2vl" not in _BG_PRELOADED:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image
        proc = AutoProcessor.from_pretrained("models/onnx/vlm/Qwen2-VL-2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained(
            "models/onnx/vlm/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
        img = Image.open("assets/sample_images/tabby.jpg").convert("RGB")
        messages = [{"role":"user", "content":[
            {"type":"image", "image": img},
            {"type":"text", "text":"Describe this in one word."}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)  # warmup
        _BG_PRELOADED["qwen2vl"] = (model, inputs)
        print("[bg-preload] Qwen2-VL torch-CUDA ready")


def _bg_resnet50_loop(stop_event):
    sess, feed = _BG_PRELOADED["resnet"]
    while not stop_event.is_set():
        sess.run(None, feed)


def _bg_tinyllama_loop(stop_event):
    sess, feeds = _BG_PRELOADED["tinyllama"]
    while not stop_event.is_set():
        sess.run(None, feeds)


def _bg_qwen2vl_loop(stop_event):
    model, inputs = _BG_PRELOADED["qwen2vl"]
    with torch.no_grad():
        while not stop_event.is_set():
            model(**inputs)


BG_REGISTRY = {
    "L0": [],
    "L1": [_bg_resnet50_loop],
    "L2": [_bg_resnet50_loop, _bg_tinyllama_loop],
    "L3": [_bg_resnet50_loop, _bg_tinyllama_loop, _bg_qwen2vl_loop],
}


def start_background(level):
    if level == "L0":
        return [], []
    stops = []; threads = []
    for fn in BG_REGISTRY[level]:
        ev = threading.Event()
        t = threading.Thread(target=fn, args=(ev,), daemon=True, name=f"bg_{fn.__name__}")
        stops.append(ev); threads.append(t)
        t.start()
    time.sleep(0.3)  # let bg workers enter the loop
    return stops, threads


def stop_background(stops, threads):
    for ev in stops: ev.set()
    for t in threads:
        t.join(timeout=10)


# ---------- main per-experiment driver --------------------------------------

def run_experiment(fg_devices, splits, bg_level, fg_models_by_dev):
    """Run one experiment: N foreground streams (per fg_devices list), one
    bg_level, with each stream mapped to one split."""
    assert len(fg_devices) == len(splits)
    stop_evt = threading.Event()
    results = [defaultdict(list) for _ in range(len(fg_devices))]

    bg_stops, bg_threads = start_background(bg_level)

    fg_threads = []
    for i, (dev, split) in enumerate(zip(fg_devices, splits)):
        model = fg_models_by_dev.get(dev) if dev != "NPU" else None
        if dev != "NPU" and model is None:
            raise RuntimeError(f"no preloaded model for device {dev}")
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, split, model, results[i], stop_evt),
                             daemon=True, name=f"fg_s{i}_{dev}")
        fg_threads.append(t)
    t_start = time.time()
    for t in fg_threads: t.start()
    for t in fg_threads: t.join()
    wall = time.time() - t_start

    stop_background(bg_stops, bg_threads)

    # Per-stream eval
    metrics = []
    for i, (split, r) in enumerate(zip(splits, results)):
        m = per_stream_sap(split, r)
        m["stream_id"] = i
        metrics.append(m)
    return metrics, wall
