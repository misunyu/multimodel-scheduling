from __future__ import annotations
import os
import sys
import json
import time
import argparse
import shutil
import zipfile
import urllib.request
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import onnxruntime as ort
# Import helpers from the CPU reference implementation for updated YOLOv3 outputs
try:
    import yolo_model_trans as ymt
    HAS_YMT = True
except Exception:
    HAS_YMT = False

# Optional: COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except Exception:
    HAS_COCO = False

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "datasets", "coco")
YOLO_MODEL_PATH = os.path.join(REPO_ROOT, "models", "yolov3_small", "model", "yolov3_small.onnx")

# COCO 2017 val URLs
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category mapping for 80-class models (index -> category_id)
# Source: standard mapping used by YOLO for COCO
COCO80_TO_COCO91_CLASS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write_log_line(log_path: Optional[str], text: str) -> None:
    print(text)
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def download_file(url: str, dst_path: str, log_path: Optional[str]) -> None:
    if os.path.exists(dst_path):
        write_log_line(log_path, f"[INFO] File exists, skipping download: {dst_path}")
        return
    ensure_dir(os.path.dirname(dst_path))
    write_log_line(log_path, f"[INFO] Downloading {url} -> {dst_path}")
    with urllib.request.urlopen(url) as resp, open(dst_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    write_log_line(log_path, f"[INFO] Downloaded: {dst_path}")


def unzip(zip_path: str, extract_dir: str, log_path: Optional[str]) -> None:
    write_log_line(log_path, f"[INFO] Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    write_log_line(log_path, f"[INFO] Extracted: {extract_dir}")


def prepare_coco_val(data_dir: str, log_path: Optional[str]) -> Tuple[str, str]:
    images_dir = os.path.join(data_dir, "val2017")
    ann_dir = os.path.join(data_dir, "annotations")
    ann_json = os.path.join(ann_dir, "instances_val2017.json")

    ensure_dir(data_dir)

    # Images
    if not os.path.isdir(images_dir) or len(os.listdir(images_dir)) == 0:
        zip_path = os.path.join(data_dir, "val2017.zip")
        download_file(COCO_VAL_IMAGES_URL, zip_path, log_path)
        unzip(zip_path, data_dir, log_path)
    else:
        write_log_line(log_path, f"[INFO] Using existing images at {images_dir}")

    # Annotations
    if not os.path.isfile(ann_json):
        zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
        download_file(COCO_VAL_ANN_URL, zip_path, log_path)
        unzip(zip_path, data_dir, log_path)
    else:
        write_log_line(log_path, f"[INFO] Using existing annotations at {ann_json}")

    if not os.path.isfile(ann_json):
        raise FileNotFoundError("COCO instances_val2017.json not found after download/extract.")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("COCO val2017 images directory not found after download/extract.")

    return images_dir, ann_json


def letterbox(image: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)


def preprocess(image_bgr: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    img, r, (dw, dh) = letterbox(image_bgr, input_size)
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    img = np.expand_dims(img, 0)  # add batch
    meta = {"ratio": r, "pad": (dw, dh), "orig_shape": image_bgr.shape[:2]}
    return img, meta


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def detect_from_model_outputs(outputs: List[np.ndarray], input_size: Tuple[int, int], meta: Dict[str, Any], conf_thr: float, iou_thr: float) -> List[Tuple[float, float, float, float, float, int]]:
    """Return list of detections per image: [x1,y1,x2,y2,score,cls] in original image coordinates.
    This tries to handle a few common YOLOv3 ONNX output formats.
    """
    # Many models output a single [1, N, 85] tensor (x,y,w,h, obj, 80 class probs)
    dets = []
    h0, w0 = meta["orig_shape"]
    r = meta["ratio"]
    dw, dh = meta["pad"]
    in_w, in_h = input_size

    def scale_coords(x1, y1, x2, y2):
        # reverse letterbox
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r
        x1 = np.clip(x1, 0, w0)
        y1 = np.clip(y1, 0, h0)
        x2 = np.clip(x2, 0, w0)
        y2 = np.clip(y2, 0, h0)
        return x1, y1, x2, y2

    # Flatten outputs if needed
    if len(outputs) == 1:
        out = outputs[0]
    else:
        # Some models output multiple tensors; try to concat last dim if shapes align
        try:
            out = np.concatenate([o.reshape((o.shape[0], -1, o.shape[-1])) for o in outputs], axis=1)
        except Exception:
            # fallback: take the largest by N dimension
            out = max(outputs, key=lambda a: (a.ndim, a.size))

    if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] >= 6:
        # out: [1, N, M]
        preds = out[0]
        if out.shape[2] == 6:
            # [x1,y1,x2,y2,score,cls] already
            boxes = preds[:, 0:4]
            scores = preds[:, 4]
            clses = preds[:, 5].astype(int)
        else:
            # assume YOLO: [x,y,w,h,obj, p1..p80]
            boxes_xywh = preds[:, 0:4]
            obj = preds[:, 4]
            cls_probs = preds[:, 5:]
            if cls_probs.size == 0:
                return []
            cls_ids = np.argmax(cls_probs, axis=1)
            cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]
            scores = obj * cls_scores
            boxes = xywh_to_xyxy(boxes_xywh)
            clses = cls_ids
        # filter conf
        mask = scores >= conf_thr
        boxes = boxes[mask]
        scores = scores[mask]
        clses = clses[mask]
        # NMS per class
        dets = []
        for c in np.unique(clses):
            inds = np.where(clses == c)[0]
            keep = nms_boxes(boxes[inds], scores[inds], iou_thr)
            for j in keep:
                x1, y1, x2, y2 = boxes[inds][j]
                # scale back to original
                x1, y1, x2, y2 = scale_coords(x1, y1, x2, y2)
                dets.append((float(x1), float(y1), float(x2), float(y2), float(scores[inds][j]), int(c)))
        return dets

    # Unknown format
    return []


def build_session(model_path: str, intra_op: int = 1, inter_op: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op
    so.inter_op_num_threads = inter_op
    # Reduce ORT log verbosity to suppress benign shape merge warnings from YOLO NMS graph
    # 0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def infer_input_shape(session: ort.InferenceSession) -> Tuple[int, int]:
    inp = session.get_inputs()[0]
    shape = list(inp.shape)
    # Expect [N,3,H,W]
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return int(shape[3]), int(shape[2])  # (W, H)
    # default to 608x608
    return 608, 608


def evaluate_coco(dets: List[Dict[str, Any]], ann_json: str, img_ids_subset: Optional[List[int]], log_path: Optional[str], max_dets: int = 100) -> Dict[str, float]:
    if not HAS_COCO:
        write_log_line(log_path, "[WARN] pycocotools not installed; skipping mAP evaluation. Install with: pip install pycocotools")
        return {}
    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(dets) if dets else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    if img_ids_subset:
        coco_eval.params.imgIds = img_ids_subset
    # Ensure maxDets uses explicit [1, 10, max_dets]
    try:
        md = sorted(set([d for d in [1, 10, int(max_dets)] if isinstance(d, int) and d > 0]))
        coco_eval.params.maxDets = md
        write_log_line(log_path, f"[INFO] COCOeval maxDets set to {coco_eval.params.maxDets}")
    except Exception:
        pass
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # Extract metrics
    stats = coco_eval.stats  # 12-value array (assumes standard summarize with [1,10,100] or similar)
    keys = [
        "mAP@[.5:.95]", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
    ]
    metrics = {k: float(v) for k, v in zip(keys, stats)}
    for k, v in metrics.items():
        write_log_line(log_path, f"[METRIC] {k}: {v:.4f}")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv3 model on COCO 2017 val using CPU.")
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH, help="Path to YOLOv3 ONNX model (CPU path)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="COCO data root directory")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to store logs/results")
    parser.add_argument("--max-images", type=int, default=100, help="Max number of images to evaluate (use 0 for all)")
    parser.add_argument("--conf-thr", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--threads", type=int, default=1, help="Number of intra/inter op threads for ORT (CPU only)")
    parser.add_argument("--save-dets", action="store_true", help="Save detection JSON for COCO eval")
    parser.add_argument("--max-dets", type=int, default=100, help="Max detections per image for COCOeval (uses [1,10,max])")
    # Device selection (CPU, NPU, or BOTH)
    parser.add_argument("--device", type=str, choices=["cpu", "npu", "both"], default="cpu", help="Device to run inference on: cpu, npu, or both")
    parser.add_argument("--npu-id", type=int, default=0, help="NPU device ID to use (for NPU modes)")
    parser.add_argument("--npu-onnx", type=str, default=None, help="Path to NPU-specific ONNX model (optional; if not set, use built-in default)")

    args = parser.parse_args(argv)

    ensure_dir(args.results_dir)
    log_path = os.path.join(args.results_dir, f"yolov3_coco_{args.device}_{_timestamp()}.log")

    write_log_line(log_path, f"[INFO] Starting YOLOv3 COCO eval on {args.device.upper()}")
    write_log_line(log_path, f"[INFO] Model: {args.model}")
    write_log_line(log_path, f"[INFO] Data dir: {args.data_dir}")
    write_log_line(log_path, f"[INFO] Max images: {args.max_images}")
    write_log_line(log_path, f"[INFO] Conf thr: {args.conf_thr}, IoU thr: {args.iou_thr}")
    write_log_line(log_path, f"[INFO] COCO maxDets (max) set to: {args.max_dets}")

    # Prepare dataset
    images_dir, ann_json = prepare_coco_val(args.data_dir, log_path)

    # Collect image list and optionally subset
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')])
    if args.max_images and args.max_images > 0:
        img_files = img_files[:args.max_images]

    write_log_line(log_path, f"[INFO] Number of images to process: {len(img_files)}")

    # If COCO available, map filename to image id
    img_id_map: Dict[str, int] = {}
    if HAS_COCO:
        coco = COCO(ann_json)
        for img in coco.imgs.values():
            img_id_map[img['file_name']] = img['id']
        img_ids_subset = [img_id_map[f] for f in img_files if f in img_id_map]
    else:
        img_ids_subset = None

    # Special mode: run both CPU and NPU sequentially and write combined results
    # NPU/BOTH execution disabled per request; keeping code for reference but not reachable
    if args.device == "both":
        run_ts = _timestamp()

        def run_cpu_eval() -> Dict[str, Any]:
            log_cpu = os.path.join(args.results_dir, f"yolov3_coco_cpu_{run_ts}.log")
            write_log_line(log_cpu, f"[INFO] Starting YOLOv3 COCO eval on CPU (both mode)")
            if not os.path.isfile(args.model):
                write_log_line(log_cpu, f"[ERROR] Model not found at {args.model}")
                return {"error": f"Model not found: {args.model}"}
            # Build CPU session
            t0 = time.time()
            session = build_session(args.model, intra_op=args.threads, inter_op=args.threads)
            input_name = session.get_inputs()[0].name
            input_w, input_h = infer_input_shape(session)
            write_log_line(log_cpu, f"[INFO] Inference input size: {input_w}x{input_h}")
            output_names = [o.name for o in session.get_outputs()]
            write_log_line(log_cpu, f"[INFO] Output names: {output_names}")
            t1 = time.time()
            write_log_line(log_cpu, f"[TIME] Session init: {(t1 - t0)*1000:.1f} ms")

            det_json: List[Dict[str, Any]] = []
            pre_times: List[float] = []
            inf_times: List[float] = []
            post_times: List[float] = []

            for i, fname in enumerate(img_files, 1):
                img_path = os.path.join(images_dir, fname)
                img0 = cv2.imread(img_path)
                if img0 is None:
                    write_log_line(log_cpu, f"[WARN] Failed to read image: {img_path}")
                    continue
                t_pre0 = time.time()
                inp, meta = preprocess(img0, (input_w, input_h))
                t_pre1 = time.time()
                pre_times.append((t_pre1 - t_pre0) * 1000)

                # Build input feed; include auxiliary inputs like 'image_shape' if required
                feeds = {input_name: inp}
                try:
                    # If yolo_model_trans is available, let it build the feed
                    if HAS_YMT:
                        # Build meta2 consistent with cpu path
                        meta2 = {
                            "orig_w": int(meta["orig_shape"][1]),
                            "orig_h": int(meta["orig_shape"][0]),
                            "ratio": float(meta["ratio"]),
                            "pad_w": int(meta["pad"][0]),
                            "pad_h": int(meta["pad"][1]),
                        }
                        feeds = ymt.build_input_feed(session, inp, meta2)
                except Exception:
                    pass
                if feeds is None or not isinstance(feeds, dict) or input_name not in feeds:
                    feeds = {input_name: inp}
                # Auto-add 'image_shape' if the session requires it
                try:
                    input_meta = {i.name: i for i in session.get_inputs()}
                    if "image_shape" in input_meta and "image_shape" not in feeds:
                        im_inp = input_meta["image_shape"]
                        h0, w0 = int(meta["orig_shape"][0]), int(meta["orig_shape"][1])
                        onnx_type = getattr(im_inp, "type", "tensor(float)")
                        dtype = np.int64 if "int64" in onnx_type else np.float32
                        shape = list(getattr(im_inp, "shape", []))
                        val = np.array([h0, w0], dtype=dtype)
                        if len(shape) == 2:
                            val = val.reshape(1, 2)
                        feeds["image_shape"] = val
                except Exception:
                    # best-effort; ignore if anything unexpected
                    pass

                t_inf0 = time.time()
                out_names = [o.name for o in session.get_outputs()]
                outputs = session.run(out_names, feeds)
                t_inf1 = time.time()
                inf_times.append((t_inf1 - t_inf0) * 1000)

                t_post0 = time.time()
                dets: List[Tuple[float, float, float, float, float, int]] = []
                if HAS_YMT:
                    try:
                        # Ensure thresholds are in sync with args
                        try:
                            ymt.CONF_THRES = float(args.conf_thr)
                            ymt.IOU_THRES = float(args.iou_thr)
                        except Exception:
                            pass
                        boxes, scores, cls_ids, already_scaled = ymt.parse_outputs(session, outputs)
                        if boxes.shape[0] > 0 and not already_scaled:
                            boxes = ymt.deletterbox_boxes(boxes, meta2)
                        for bi in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[bi].tolist()
                            dets.append((float(x1), float(y1), float(x2), float(y2), float(scores[bi]), int(cls_ids[bi])))
                    except Exception:
                        # Fallback to legacy parser
                        if 'meta' in locals() and meta is not None:
                            dets = detect_from_model_outputs(outputs, (input_w, input_h), meta, args.conf_thr, args.iou_thr)
                        else:
                            dets = []
                else:
                    dets = detect_from_model_outputs(outputs, (input_w, input_h), meta, args.conf_thr, args.iou_thr)
                t_post1 = time.time()
                post_times.append((t_post1 - t_post0) * 1000)

                if HAS_COCO and fname in img_id_map:
                    image_id = img_id_map[fname]
                    for x1, y1, x2, y2, score, cls_idx in dets:
                        w = x2 - x1
                        h = y2 - y1
                        cat_id = COCO80_TO_COCO91_CLASS[cls_idx] if 0 <= cls_idx < len(COCO80_TO_COCO91_CLASS) else 1
                        det_json.append({
                            "image_id": image_id,
                            "category_id": int(cat_id),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score)
                        })

                if i % 10 == 0 or i == len(img_files):
                    write_log_line(log_cpu, f"[PROGRESS] {i}/{len(img_files)} images processed")

            pre_avg = float(np.mean(pre_times)) if pre_times else None
            pre_p95 = float(np.percentile(pre_times, 95)) if pre_times else None
            inf_avg = float(np.mean(inf_times)) if inf_times else None
            inf_p95 = float(np.percentile(inf_times, 95)) if inf_times else None
            post_avg = float(np.mean(post_times)) if post_times else None
            post_p95 = float(np.percentile(post_times, 95)) if post_times else None

            if pre_avg is not None:
                write_log_line(log_cpu, f"[TIME] Preprocess avg: {pre_avg:.1f} ms, p95: {pre_p95:.1f} ms")
            if inf_avg is not None:
                write_log_line(log_cpu, f"[TIME] Inference avg: {inf_avg:.1f} ms, p95: {inf_p95:.1f} ms")
            if post_avg is not None:
                write_log_line(log_cpu, f"[TIME] Postprocess avg: {post_avg:.1f} ms, p95: {post_p95:.1f} ms")

            det_json_path = None
            if (HAS_COCO and det_json) or args.save_dets:
                det_json_path = os.path.join(args.results_dir, f"yolov3_coco_dets_cpu_{run_ts}.json")
                with open(det_json_path, "w", encoding="utf-8") as f:
                    json.dump(det_json, f)
                write_log_line(log_cpu, f"[INFO] Saved detections to {det_json_path}")

            metrics = {}
            if HAS_COCO and det_json:
                metrics = evaluate_coco(det_json, ann_json, img_ids_subset, log_cpu, max_dets=args.max_dets)
                mAP = metrics.get("mAP@[.5:.95]")
                mAP50 = metrics.get("mAP@0.5")
                if mAP is not None or mAP50 is not None:
                    parts = []
                    if mAP is not None:
                        parts.append(f"mAP@[.5:.95]={mAP:.4f}")
                    if mAP50 is not None:
                        parts.append(f"mAP@0.5={mAP50:.4f}")
                    write_log_line(log_cpu, f"[RESULT] Accuracy: " + ", ".join(parts))
            else:
                if not HAS_COCO:
                    write_log_line(log_cpu, "[WARN] Skipping evaluation: pycocotools not installed.")
                else:
                    write_log_line(log_cpu, "[WARN] Skipping evaluation: no detections were produced.")

            results_summary = {
                "timestamp": run_ts,
                "device": "cpu",
                "model": args.model,
                "data_dir": args.data_dir,
                "num_images": len(img_files),
                "conf_thr": args.conf_thr,
                "iou_thr": args.iou_thr,
                "threads": args.threads,
                "timing_ms": {
                    "preprocess_avg": pre_avg,
                    "preprocess_p95": pre_p95,
                    "inference_avg": inf_avg,
                    "inference_p95": inf_p95,
                    "postprocess_avg": post_avg,
                    "postprocess_p95": post_p95,
                },
                "metrics": metrics,
                "detections_json": det_json_path,
            }
            metrics_path = os.path.join(args.results_dir, f"yolov3_coco_metrics_cpu_{run_ts}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=2)
            write_log_line(log_cpu, f"[INFO] Saved metrics/results to {metrics_path}")
            write_log_line(log_cpu, "[INFO] CPU evaluation finished.")
            return results_summary

        def run_npu_eval() -> Dict[str, Any]:
            log_npu = os.path.join(args.results_dir, f"yolov3_coco_npu_{run_ts}.log")
            write_log_line(log_npu, f"[INFO] Starting YOLOv3 COCO eval on NPU (both mode)")
            try:
                from npu import initialize_driver, close_driver, send_receive_data_npu, yolo_prepare_onnx_model
            except Exception as e:
                write_log_line(log_npu, f"[ERROR] Failed to import NPU utilities: {e}")
                return {"error": f"NPU import failed: {e}"}

            host_t0 = time.time()
            try:
                npu_onnx_path = args.npu_onnx or "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
                write_log_line(log_npu, f"[INFO] Using NPU ONNX: {npu_onnx_path}")
                front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
                    npu_onnx_path
                )
                input_w, input_h = 608, 608
            except Exception as e:
                write_log_line(log_npu, f"[ERROR] NPU host model preparation failed: {e}")
                return {"error": f"NPU host prepare failed: {e}"}
            host_t1 = time.time()
            write_log_line(log_npu, f"[TIME] NPU host session prep: {(host_t1 - host_t0)*1000:.1f} ms")

            drv_t0 = time.time()
            try:
                driver = initialize_driver(args.npu_id, "./models/yolov3_small/npu_code/yolov3_small_neubla_p1.o")
            except Exception as e:
                write_log_line(log_npu, f"[ERROR] NPU driver init/load failed: {e}")
                return {"error": f"NPU driver failed: {e}"}
            drv_t1 = time.time()
            write_log_line(log_npu, f"[TIME] NPU memory load: {(drv_t1 - drv_t0)*1000:.1f} ms")

            def npu_preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
                # Use the same letterbox-based preprocess as CPU path to obtain meta
                inp, meta = preprocess(img_bgr, (input_w, input_h))
                meta2 = {
                    "orig_w": int(meta["orig_shape"][1]),
                    "orig_h": int(meta["orig_shape"][0]),
                    "ratio": float(meta["ratio"]),
                    "pad": (int(meta["pad"][0]), int(meta["pad"][1])),
                }
                return inp, meta2

            def npu_postprocess_to_dets(output: List[np.ndarray], meta2: Dict[str, Any], conf_thr: float, iou_thr: float) -> List[Tuple[float,float,float,float,float,int]]:
                # Apply filtering logic equivalent to image_processing.yolo_postprocess_npu
                output_box = np.squeeze(output[0])
                output_label = np.squeeze(output[1])
                rows = output_box.shape[0]
                boxes_xyxy = []
                scores = []
                class_ids = []
                for i in range(rows):
                    conf = float(output_box[i][4])
                    if conf >= conf_thr:
                        left, top, right, bottom = output_box[i][:4]
                        boxes_xyxy.append([left, top, right, bottom])
                        scores.append(conf)
                        class_ids.append(int(output_label[i]))
                dets: List[Tuple[float,float,float,float,float,int]] = []
                if boxes_xyxy:
                    b = np.array(boxes_xyxy, dtype=np.float32)
                    s = np.array(scores, dtype=np.float32)
                    # OpenCV NMS uses [x,y,w,h]
                    b_xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
                    keep = cv2.dnn.NMSBoxes(b_xywh.tolist(), s.tolist(), conf_thr, iou_thr)
                    if keep is not None and len(keep) > 0:
                        keep = [int(k) if isinstance(k, (int, np.integer)) else int(k[0]) for k in keep]
                        dw, dh = meta2["pad"]
                        r = meta2["ratio"]
                        w0, h0 = meta2["orig_w"], meta2["orig_h"]
                        for j in keep:
                            x1, y1, x2, y2 = b[j]
                            # de-letterbox
                            x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
                            x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
                            x1 = np.clip(x1, 0, w0); y1 = np.clip(y1, 0, h0)
                            x2 = np.clip(x2, 0, w0); y2 = np.clip(y2, 0, h0)
                            # Additional filters mirroring yolo_postprocess_npu
                            w = float(x2 - x1)
                            h = float(y2 - y1)
                            min_sz = 4.0
                            if w < min_sz or h < min_sz:
                                continue
                            if w > 0.98 * w0 or h > 0.98 * h0:
                                continue
                            ar = w / (h + 1e-6)
                            if ar > 25.0 or ar < 1.0/25.0:
                                continue
                            dets.append((x1, y1, x2, y2, float(s[j]), int(class_ids[j])))
                return dets

            det_json: List[Dict[str, Any]] = []
            pre_times: List[float] = []
            inf_times: List[float] = []
            post_times: List[float] = []
            try:
                for i, fname in enumerate(img_files, 1):
                    img_path = os.path.join(images_dir, fname)
                    img0 = cv2.imread(img_path)
                    if img0 is None:
                        write_log_line(log_npu, f"[WARN] Failed to read image: {img_path}")
                        continue
                    t_pre0 = time.time()
                    inp, meta2 = npu_preprocess(img0)
                    t_pre1 = time.time()
                    pre_times.append((t_pre1 - t_pre0) * 1000)

                    t_inf0 = time.time()
                    front_output = front_sess.run(None, {"input": inp})[0]
                    input_data = front_output.tobytes()
                    raw_outputs = send_receive_data_npu(driver, input_data, 3 * input_w * input_h)
                    output_data = [np.frombuffer(buf, dtype=np.uint8) for buf in raw_outputs]
                    output_dequant_data = [
                        (data.astype(np.float32) - zero_point[name]) * scale[name]
                        for name, data in zip(
                            [
                                "onnx::Transpose_684_DequantizeLinear",
                                "onnx::Transpose_688_DequantizeLinear",
                                "onnx::Transpose_692_DequantizeLinear",
                            ],
                            output_data,
                        )
                    ]
                    shape_dict = {
                        "onnx::Transpose_684": (1, 255, 19, 19),
                        "onnx::Transpose_688": (1, 255, 38, 38),
                        "onnx::Transpose_692": (1, 255, 76, 76),
                    }
                    back_feeds = {}
                    for name, data in zip(shape_dict.keys(), output_dequant_data):
                        needed_size = int(np.prod(shape_dict[name]))
                        back_feeds[name] = data[:needed_size].reshape(shape_dict[name])
                    outputs = back_sess.run(None, back_feeds)
                    t_inf1 = time.time()
                    inf_times.append((t_inf1 - t_inf0) * 1000)

                    t_post0 = time.time()
                    dets = npu_postprocess_to_dets(outputs, meta2, args.conf_thr, args.iou_thr)
                    t_post1 = time.time()
                    post_times.append((t_post1 - t_post0) * 1000)

                    if HAS_COCO and fname in img_id_map:
                        image_id = img_id_map[fname]
                        for x1, y1, x2, y2, score, cls_idx in dets:
                            w = x2 - x1
                            h = y2 - y1
                            cat_id = COCO80_TO_COCO91_CLASS[cls_idx] if 0 <= cls_idx < len(COCO80_TO_COCO91_CLASS) else 1
                            det_json.append({
                                "image_id": image_id,
                                "category_id": int(cat_id),
                                "bbox": [float(x1), float(y1), float(w), float(h)],
                                "score": float(score)
                            })

                    if i % 10 == 0 or i == len(img_files):
                        write_log_line(log_npu, f"[PROGRESS] {i}/{len(img_files)} images processed")
            finally:
                try:
                    close_driver(driver)
                except Exception:
                    pass

            pre_avg = float(np.mean(pre_times)) if pre_times else None
            pre_p95 = float(np.percentile(pre_times, 95)) if pre_times else None
            inf_avg = float(np.mean(inf_times)) if inf_times else None
            inf_p95 = float(np.percentile(inf_times, 95)) if inf_times else None
            post_avg = float(np.mean(post_times)) if post_times else None
            post_p95 = float(np.percentile(post_times, 95)) if post_times else None

            if pre_avg is not None:
                write_log_line(log_npu, f"[TIME] Preprocess avg: {pre_avg:.1f} ms, p95: {pre_p95:.1f} ms")
            if inf_avg is not None:
                write_log_line(log_npu, f"[TIME] Inference avg: {inf_avg:.1f} ms, p95: {inf_p95:.1f} ms")
            if post_avg is not None:
                write_log_line(log_npu, f"[TIME] Postprocess avg: {post_avg:.1f} ms, p95: {post_p95:.1f} ms")

            det_json_path = None
            if (HAS_COCO and det_json) or args.save_dets:
                det_json_path = os.path.join(args.results_dir, f"yolov3_coco_dets_npu_{run_ts}.json")
                with open(det_json_path, "w", encoding="utf-8") as f:
                    json.dump(det_json, f)
                write_log_line(log_npu, f"[INFO] Saved detections to {det_json_path}")

            metrics = {}
            if HAS_COCO and det_json:
                metrics = evaluate_coco(det_json, ann_json, img_ids_subset, log_npu, max_dets=args.max_dets)
                mAP = metrics.get("mAP@[.5:.95]")
                mAP50 = metrics.get("mAP@0.5")
                if mAP is not None or mAP50 is not None:
                    parts = []
                    if mAP is not None:
                        parts.append(f"mAP@[.5:.95]={mAP:.4f}")
                    if mAP50 is not None:
                        parts.append(f"mAP@0.5={mAP50:.4f}")
                    write_log_line(log_npu, f"[RESULT] Accuracy: " + ", ".join(parts))
            else:
                if not HAS_COCO:
                    write_log_line(log_npu, "[WARN] Skipping evaluation: pycocotools not installed.")
                else:
                    write_log_line(log_npu, "[WARN] Skipping evaluation: no detections were produced.")

            results_summary = {
                "timestamp": run_ts,
                "device": "npu",
                "model": args.model,
                "data_dir": args.data_dir,
                "num_images": len(img_files),
                "conf_thr": args.conf_thr,
                "iou_thr": args.iou_thr,
                "threads": None,
                "timing_ms": {
                    "preprocess_avg": pre_avg,
                    "preprocess_p95": pre_p95,
                    "inference_avg": inf_avg,
                    "inference_p95": inf_p95,
                    "postprocess_avg": post_avg,
                    "postprocess_p95": post_p95,
                },
                "metrics": metrics,
                "detections_json": det_json_path,
            }
            metrics_path = os.path.join(args.results_dir, f"yolov3_coco_metrics_npu_{run_ts}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=2)
            write_log_line(log_npu, f"[INFO] Saved metrics/results to {metrics_path}")
            write_log_line(log_npu, "[INFO] NPU evaluation finished.")
            return results_summary

        npu_summary = run_npu_eval()
        cpu_summary = run_cpu_eval()
        combined = {"timestamp": run_ts, "cpu": cpu_summary, "npu": npu_summary}
        combined_path = os.path.join(args.results_dir, f"yolov3_coco_combined_{run_ts}.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        write_log_line(os.path.join(args.results_dir, f"yolov3_coco_both_{run_ts}.log"), f"[INFO] Saved combined results to {combined_path}")

        # Also write a concise accuracy summary file
        cpu_metrics = cpu_summary.get("metrics", {}) if isinstance(cpu_summary, dict) else {}
        npu_metrics = npu_summary.get("metrics", {}) if isinstance(npu_summary, dict) else {}
        acc_record = {
            "timestamp": run_ts,
            "images": len(img_files),
            "cpu": {
                "mAP": cpu_metrics.get("mAP@[.5:.95]"),
                "mAP50": cpu_metrics.get("mAP@0.5"),
            },
            "npu": {
                "mAP": npu_metrics.get("mAP@[.5:.95]"),
                "mAP50": npu_metrics.get("mAP@0.5"),
            },
        }
        acc_path = os.path.join(args.results_dir, f"yolov3_coco_accuracy_{run_ts}.json")
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(acc_record, f, ensure_ascii=False, indent=2)
        write_log_line(os.path.join(args.results_dir, f"yolov3_coco_both_{run_ts}.log"), f"[INFO] Saved accuracy summary to {acc_path}")
        return 0

    # Inference loop common accumulators
    det_json: List[Dict[str, Any]] = []
    inf_times: List[float] = []
    pre_times: List[float] = []
    post_times: List[float] = []

    if args.device == "cpu":
        if not os.path.isfile(args.model):
            write_log_line(log_path, f"[ERROR] Model not found at {args.model}")
            return 1
        # Build CPU session
        t0 = time.time()
        session = build_session(args.model, intra_op=args.threads, inter_op=args.threads)
        input_name = session.get_inputs()[0].name
        input_w, input_h = infer_input_shape(session)
        write_log_line(log_path, f"[INFO] Inference input size: {input_w}x{input_h}")
        output_names = [o.name for o in session.get_outputs()]
        write_log_line(log_path, f"[INFO] Output names: {output_names}")
        t1 = time.time()
        write_log_line(log_path, f"[TIME] Session init: {(t1 - t0)*1000:.1f} ms")

        for i, fname in enumerate(img_files, 1):
            img_path = os.path.join(images_dir, fname)
            img0 = cv2.imread(img_path)
            if img0 is None:
                write_log_line(log_path, f"[WARN] Failed to read image: {img_path}")
                continue
            t_pre0 = time.time()
            if HAS_YMT:
                # Use unified preprocessing from yolo_model_trans (includes letterbox and meta fields)
                try:
                    inp, meta2 = ymt.preprocess(img_path)
                    meta = None  # not used in HAS_YMT branch
                except Exception:
                    # Fallback to local preprocess if anything goes wrong
                    inp, meta = preprocess(img0, (input_w, input_h))
                    meta2 = {
                        "orig_w": int(meta["orig_shape"][1]),
                        "orig_h": int(meta["orig_shape"][0]),
                        "ratio": float(meta["ratio"]),
                        "pad_w": int(meta["pad"][0]),
                        "pad_h": int(meta["pad"][1]),
                    }
            else:
                inp, meta = preprocess(img0, (input_w, input_h))
                meta2 = {
                    "orig_w": int(meta["orig_shape"][1]),
                    "orig_h": int(meta["orig_shape"][0]),
                    "ratio": float(meta["ratio"]),
                    "pad_w": int(meta["pad"][0]),
                    "pad_h": int(meta["pad"][1]),
                }
            t_pre1 = time.time()
            pre_times.append((t_pre1 - t_pre0) * 1000)

            # Set thresholds in yolo_model_trans if available
            if HAS_YMT:
                try:
                    ymt.CONF_THRES = float(args.conf_thr)
                    ymt.IOU_THRES = float(args.iou_thr)
                except Exception:
                    pass

            # Build input feed using yolo_model_trans if available
            feeds = {input_name: inp}
            if HAS_YMT:
                try:
                    feeds = ymt.build_input_feed(session, inp, meta2)
                except Exception:
                    feeds = {input_name: inp}
            # Auto-add 'image_shape' if the session requires it and it's missing
            try:
                input_meta = {i.name: i for i in session.get_inputs()}
                if "image_shape" in input_meta and "image_shape" not in feeds:
                    im_inp = input_meta["image_shape"]
                    if "orig_w" in meta2 and "orig_h" in meta2:
                        h0, w0 = int(meta2["orig_h"]), int(meta2["orig_w"])
                    elif 'meta' in locals() and meta is not None:
                        h0, w0 = int(meta["orig_shape"][0]), int(meta["orig_shape"][1])
                    else:
                        # fallback to input tensor size
                        if inp.ndim == 4:
                            h0, w0 = int(inp.shape[2]), int(inp.shape[3])
                        else:
                            h0, w0 = int(input_h), int(input_w)
                    # onnxruntime input info may have a type string like 'tensor(int64)' or 'tensor(float)'
                    onnx_type = getattr(im_inp, "type", "tensor(float)")
                    dtype = np.int64 if "int64" in onnx_type else np.float32
                    shape = list(getattr(im_inp, "shape", []))
                    val = np.array([h0, w0], dtype=dtype)
                    if len(shape) == 2:
                        val = val.reshape(1, 2)
                    feeds["image_shape"] = val
            except Exception:
                # best-effort; ignore if anything unexpected
                pass

            t_inf0 = time.time()
            out_names = [o.name for o in session.get_outputs()]
            outputs = session.run(out_names, feeds)
            t_inf1 = time.time()
            inf_times.append((t_inf1 - t_inf0) * 1000)

            t_post0 = time.time()
            dets_list = []
            if HAS_YMT:
                try:
                    boxes, scores, cls_ids, already_scaled = ymt.parse_outputs(session, outputs)
                    if boxes.shape[0] > 0 and not already_scaled:
                        boxes = ymt.deletterbox_boxes(boxes, meta2)
                    for bi in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[bi].tolist()
                        dets_list.append((x1, y1, x2, y2, float(scores[bi]), int(cls_ids[bi])))
                except Exception:
                    # Fallback to legacy detector if parsing fails
                    if 'meta' in locals() and meta is not None:
                        dets_list = detect_from_model_outputs(outputs, (input_w, input_h), meta, args.conf_thr, args.iou_thr)
                    else:
                        dets_list = []
            else:
                dets_list = detect_from_model_outputs(outputs, (input_w, input_h), meta, args.conf_thr, args.iou_thr)
            t_post1 = time.time()
            post_times.append((t_post1 - t_post0) * 1000)

            if HAS_COCO and fname in img_id_map:
                image_id = img_id_map[fname]
                for x1, y1, x2, y2, score, cls_idx in dets_list:
                    w = x2 - x1
                    h = y2 - y1
                    cat_id = COCO80_TO_COCO91_CLASS[cls_idx] if 0 <= cls_idx < len(COCO80_TO_COCO91_CLASS) else 1
                    det_json.append({
                        "image_id": image_id,
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

            if i % 10 == 0 or i == len(img_files):
                write_log_line(log_path, f"[PROGRESS] {i}/{len(img_files)} images processed")
    else:
        # NPU path
        try:
            from npu import initialize_driver, close_driver, send_receive_data_npu, yolo_prepare_onnx_model
        except Exception as e:
            write_log_line(log_path, f"[ERROR] Failed to import NPU utilities: {e}")
            return 1

        # Prepare NPU host/back sessions and driver
        host_t0 = time.time()
        try:
            # NOTE: allow overriding the NPU ONNX path to avoid any partitions usage
            npu_onnx_path = args.npu_onnx or "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
            write_log_line(log_path, f"[INFO] Using NPU ONNX: {npu_onnx_path}")
            front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
                npu_onnx_path
            )
            input_w, input_h = 608, 608
        except Exception as e:
            write_log_line(log_path, f"[ERROR] NPU host model preparation failed: {e}")
            return 1
        host_t1 = time.time()
        write_log_line(log_path, f"[TIME] NPU host session prep: {(host_t1 - host_t0)*1000:.1f} ms")

        drv_t0 = time.time()
        try:
            driver = initialize_driver(args.npu_id, "./models/yolov3_small/npu_code/yolov3_small_neubla_p1.o")
        except Exception as e:
            write_log_line(log_path, f"[ERROR] NPU driver init/load failed: {e}")
            return 1
        drv_t1 = time.time()
        write_log_line(log_path, f"[TIME] NPU memory load: {(drv_t1 - drv_t0)*1000:.1f} ms")

        def npu_preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
            # Use the same letterbox-based preprocess as CPU path to obtain meta
            inp, meta = preprocess(img_bgr, (input_w, input_h))
            meta2 = {
                "orig_w": int(meta["orig_shape"][1]),
                "orig_h": int(meta["orig_shape"][0]),
                "ratio": float(meta["ratio"]),
                "pad": (int(meta["pad"][0]), int(meta["pad"][1])),
            }
            return inp, meta2

        def npu_postprocess_to_dets(output: List[np.ndarray], meta2: Dict[str, Any], conf_thr: float, iou_thr: float) -> List[Tuple[float,float,float,float,float,int]]:
            # Apply filtering logic equivalent to image_processing.yolo_postprocess_npu
            output_box = np.squeeze(output[0])
            output_label = np.squeeze(output[1])
            rows = output_box.shape[0]
            boxes_xyxy = []
            scores = []
            class_ids = []
            for i in range(rows):
                conf = float(output_box[i][4])
                if conf >= conf_thr:
                    left, top, right, bottom = output_box[i][:4]
                    boxes_xyxy.append([left, top, right, bottom])
                    scores.append(conf)
                    class_ids.append(int(output_label[i]))
            dets: List[Tuple[float,float,float,float,float,int]] = []
            if boxes_xyxy:
                b = np.array(boxes_xyxy, dtype=np.float32)
                s = np.array(scores, dtype=np.float32)
                # OpenCV NMS uses [x,y,w,h]
                b_xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
                keep = cv2.dnn.NMSBoxes(b_xywh.tolist(), s.tolist(), conf_thr, iou_thr)
                if keep is not None and len(keep) > 0:
                    keep = [int(k) if isinstance(k, (int, np.integer)) else int(k[0]) for k in keep]
                    dw, dh = meta2["pad"]
                    r = meta2["ratio"]
                    w0, h0 = meta2["orig_w"], meta2["orig_h"]
                    for j in keep:
                        x1, y1, x2, y2 = b[j]
                        # de-letterbox
                        x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
                        x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
                        x1 = float(np.clip(x1, 0, w0)); y1 = float(np.clip(y1, 0, h0))
                        x2 = float(np.clip(x2, 0, w0)); y2 = float(np.clip(y2, 0, h0))
                        # Additional filters mirroring yolo_postprocess_npu
                        w = x2 - x1
                        h = y2 - y1
                        min_sz = 4.0
                        if w < min_sz or h < min_sz:
                            continue
                        if w > 0.98 * w0 or h > 0.98 * h0:
                            continue
                        ar = w / (h + 1e-6)
                        if ar > 25.0 or ar < 1.0/25.0:
                            continue
                        dets.append((x1, y1, x2, y2, float(s[j]), int(class_ids[j])))
            return dets

        try:
            for i, fname in enumerate(img_files, 1):
                img_path = os.path.join(images_dir, fname)
                img0 = cv2.imread(img_path)
                if img0 is None:
                    write_log_line(log_path, f"[WARN] Failed to read image: {img_path}")
                    continue
                t_pre0 = time.time()
                inp, meta2 = npu_preprocess(img0)
                t_pre1 = time.time()
                pre_times.append((t_pre1 - t_pre0) * 1000)

                t_inf0 = time.time()
                # Front (host) quantize
                front_output = front_sess.run(None, {"input": inp})[0]
                input_data = front_output.tobytes()
                # NPU core
                raw_outputs = send_receive_data_npu(driver, input_data, 3 * input_w * input_h)
                output_data = [np.frombuffer(buf, dtype=np.uint8) for buf in raw_outputs]
                # Dequantize and back (host) post layers
                output_dequant_data = [
                    (data.astype(np.float32) - zero_point[name]) * scale[name]
                    for name, data in zip(
                        [
                            "onnx::Transpose_684_DequantizeLinear",
                            "onnx::Transpose_688_DequantizeLinear",
                            "onnx::Transpose_692_DequantizeLinear",
                        ],
                        output_data,
                    )
                ]
                shape_dict = {
                    "onnx::Transpose_684": (1, 255, 19, 19),
                    "onnx::Transpose_688": (1, 255, 38, 38),
                    "onnx::Transpose_692": (1, 255, 76, 76),
                }
                back_feeds = {}
                for name, data in zip(shape_dict.keys(), output_dequant_data):
                    needed_size = int(np.prod(shape_dict[name]))
                    back_feeds[name] = data[:needed_size].reshape(shape_dict[name])
                outputs = back_sess.run(None, back_feeds)
                t_inf1 = time.time()
                inf_times.append((t_inf1 - t_inf0) * 1000)

                t_post0 = time.time()
                dets = npu_postprocess_to_dets(outputs, meta2, args.conf_thr, args.iou_thr)
                t_post1 = time.time()
                post_times.append((t_post1 - t_post0) * 1000)

                if HAS_COCO and fname in img_id_map:
                    image_id = img_id_map[fname]
                    for x1, y1, x2, y2, score, cls_idx in dets:
                        w = x2 - x1
                        h = y2 - y1
                        cat_id = COCO80_TO_COCO91_CLASS[cls_idx] if 0 <= cls_idx < len(COCO80_TO_COCO91_CLASS) else 1
                        det_json.append({
                            "image_id": image_id,
                            "category_id": int(cat_id),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score)
                        })

                if i % 10 == 0 or i == len(img_files):
                    write_log_line(log_path, f"[PROGRESS] {i}/{len(img_files)} images processed")
        finally:
            try:
                close_driver(driver)
            except Exception:
                pass

    # Timing summary (compute and log)
    pre_avg = float(np.mean(pre_times)) if pre_times else None
    pre_p95 = float(np.percentile(pre_times, 95)) if pre_times else None
    inf_avg = float(np.mean(inf_times)) if inf_times else None
    inf_p95 = float(np.percentile(inf_times, 95)) if inf_times else None
    post_avg = float(np.mean(post_times)) if post_times else None
    post_p95 = float(np.percentile(post_times, 95)) if post_times else None

    if pre_avg is not None:
        write_log_line(log_path, f"[TIME] Preprocess avg: {pre_avg:.1f} ms, p95: {pre_p95:.1f} ms")
    if inf_avg is not None:
        write_log_line(log_path, f"[TIME] Inference avg: {inf_avg:.1f} ms, p95: {inf_p95:.1f} ms")
    if post_avg is not None:
        write_log_line(log_path, f"[TIME] Postprocess avg: {post_avg:.1f} ms, p95: {post_p95:.1f} ms")

    # Save detections JSON if requested or needed for eval
    det_json_path = None
    if (HAS_COCO and det_json) or args.save_dets:
        det_json_path = os.path.join(args.results_dir, f"yolov3_coco_dets_{_timestamp()}.json")
        with open(det_json_path, "w", encoding="utf-8") as f:
            json.dump(det_json, f)
        write_log_line(log_path, f"[INFO] Saved detections to {det_json_path}")

    # Evaluate and save metrics
    metrics = {}
    if HAS_COCO and det_json:
        metrics = evaluate_coco(det_json, ann_json, img_ids_subset, log_path, max_dets=args.max_dets)
        mAP = metrics.get("mAP@[.5:.95]")
        mAP50 = metrics.get("mAP@0.5")
        if mAP is not None or mAP50 is not None:
            parts = []
            if mAP is not None:
                parts.append(f"mAP@[.5:.95]={mAP:.4f}")
            if mAP50 is not None:
                parts.append(f"mAP@0.5={mAP50:.4f}")
            write_log_line(log_path, f"[RESULT] Accuracy: " + ", ".join(parts))
    else:
        if not HAS_COCO:
            write_log_line(log_path, "[WARN] Skipping evaluation: pycocotools not installed.")
        else:
            write_log_line(log_path, "[WARN] Skipping evaluation: no detections were produced.")

    results_summary = {
        "timestamp": _timestamp(),
        "device": args.device,
        "model": args.model,
        "data_dir": args.data_dir,
        "num_images": len(img_files),
        "conf_thr": args.conf_thr,
        "iou_thr": args.iou_thr,
        "threads": args.threads if args.device == "cpu" else None,
        "timing_ms": {
            "preprocess_avg": pre_avg,
            "preprocess_p95": pre_p95,
            "inference_avg": inf_avg,
            "inference_p95": inf_p95,
            "postprocess_avg": post_avg,
            "postprocess_p95": post_p95,
        },
        "metrics": metrics,
        "detections_json": det_json_path,
    }
    metrics_path = os.path.join(args.results_dir, f"yolov3_coco_metrics_{_timestamp()}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    write_log_line(log_path, f"[INFO] Saved metrics/results to {metrics_path}")

    # Also write a concise accuracy summary for this device
    acc_record = {
        "timestamp": _timestamp(),
        "device": args.device,
        "images": len(img_files),
        "mAP": metrics.get("mAP@[.5:.95]") if metrics else None,
        "mAP50": metrics.get("mAP@0.5") if metrics else None,
    }
    acc_path = os.path.join(args.results_dir, f"yolov3_coco_accuracy_{_timestamp()}.json")
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(acc_record, f, ensure_ascii=False, indent=2)
    write_log_line(log_path, f"[INFO] Saved accuracy summary to {acc_path}")

    # Ensure a final accuracy line is printed at the very end for CPU runs, like on NPU
    if args.device == "cpu":
        mAP = metrics.get("mAP@[.5:.95]") if metrics else None
        mAP50 = metrics.get("mAP@0.5") if metrics else None
        if (mAP is not None) or (mAP50 is not None):
            parts = []
            if mAP is not None:
                parts.append(f"mAP@[.5:.95]={mAP:.4f}")
            if mAP50 is not None:
                parts.append(f"mAP@0.5={mAP50:.4f}")
            write_log_line(log_path, "[RESULT] Accuracy: " + ", ".join(parts))

    write_log_line(log_path, "[INFO] Evaluation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
