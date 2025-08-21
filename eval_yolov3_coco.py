from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
import shutil
import zipfile
import urllib.request
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import onnxruntime as ort

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
YOLO_MODEL_PATH = os.path.join(REPO_ROOT, "models", "yolov3_big", "model", "yolov3_big.onnx")

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


def evaluate_coco(dets: List[Dict[str, Any]], ann_json: str, img_ids_subset: Optional[List[int]], log_path: Optional[str]) -> Dict[str, float]:
    if not HAS_COCO:
        write_log_line(log_path, "[WARN] pycocotools not installed; skipping mAP evaluation. Install with: pip install pycocotools")
        return {}
    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(dets) if dets else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    if img_ids_subset:
        coco_eval.params.imgIds = img_ids_subset
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # Extract metrics
    stats = coco_eval.stats  # 12-value array
    keys = [
        "mAP@[.5:.95]", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
    ]
    metrics = {k: float(v) for k, v in zip(keys, stats)}
    for k, v in metrics.items():
        write_log_line(log_path, f"[METRIC] {k}: {v:.4f}")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv3 ONNX model on COCO 2017 val using CPU.")
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH, help="Path to YOLOv3 ONNX model")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="COCO data root directory")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to store logs/results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (currently images processed one by one)")
    parser.add_argument("--max-images", type=int, default=100, help="Max number of images to evaluate (use 0 for all)")
    parser.add_argument("--conf-thr", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--threads", type=int, default=1, help="Number of intra/inter op threads for ORT")
    parser.add_argument("--save-dets", action="store_true", help="Save detection JSON for COCO eval")

    args = parser.parse_args(argv)

    ensure_dir(args.results_dir)
    log_path = os.path.join(args.results_dir, f"yolov3_coco_cpu_{_timestamp()}.log")

    write_log_line(log_path, f"[INFO] Starting YOLOv3 COCO eval on CPU")
    write_log_line(log_path, f"[INFO] Model: {args.model}")
    write_log_line(log_path, f"[INFO] Data dir: {args.data_dir}")
    write_log_line(log_path, f"[INFO] Max images: {args.max_images}")
    write_log_line(log_path, f"[INFO] Conf thr: {args.conf_thr}, IoU thr: {args.iou_thr}")

    if not os.path.isfile(args.model):
        write_log_line(log_path, f"[ERROR] Model not found at {args.model}")
        return 1

    # Prepare dataset
    images_dir, ann_json = prepare_coco_val(args.data_dir, log_path)

    # Build session
    t0 = time.time()
    session = build_session(args.model, intra_op=args.threads, inter_op=args.threads)
    input_name = session.get_inputs()[0].name
    input_w, input_h = infer_input_shape(session)
    write_log_line(log_path, f"[INFO] Inference input size: {input_w}x{input_h}")

    output_names = [o.name for o in session.get_outputs()]
    write_log_line(log_path, f"[INFO] Output names: {output_names}")
    t1 = time.time()
    write_log_line(log_path, f"[TIME] Session init: {(t1 - t0)*1000:.1f} ms")

    # Collect image list and optionally subset
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')])
    if args.max_images and args.max_images > 0:
        img_files = img_files[:args.max_images]

    write_log_line(log_path, f"[INFO] Number of images to process: {len(img_files)}")

    # If COCO available, map filename to image id
    img_id_map: Dict[str, int] = {}
    if HAS_COCO:
        coco = COCO(ann_json)
        # Build map filename -> id
        for img in coco.imgs.values():
            img_id_map[img['file_name']] = img['id']
        # restrict to files present
        img_ids_subset = [img_id_map[f] for f in img_files if f in img_id_map]
    else:
        img_ids_subset = None

    # Inference loop
    det_json: List[Dict[str, Any]] = []
    inf_times: List[float] = []
    pre_times: List[float] = []
    post_times: List[float] = []

    for i, fname in enumerate(img_files, 1):
        img_path = os.path.join(images_dir, fname)
        img0 = cv2.imread(img_path)
        if img0 is None:
            write_log_line(log_path, f"[WARN] Failed to read image: {img_path}")
            continue
        t_pre0 = time.time()
        inp, meta = preprocess(img0, (input_w, input_h))
        t_pre1 = time.time()
        pre_times.append((t_pre1 - t_pre0) * 1000)

        t_inf0 = time.time()
        outputs = session.run(None, {input_name: inp})
        t_inf1 = time.time()
        inf_times.append((t_inf1 - t_inf0) * 1000)

        t_post0 = time.time()
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
            write_log_line(log_path, f"[PROGRESS] {i}/{len(img_files)} images processed")

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
        metrics = evaluate_coco(det_json, ann_json, img_ids_subset, log_path)
        # concise accuracy summary line
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

    # Save consolidated results JSON (timing + metrics)
    results_summary = {
        "timestamp": _timestamp(),
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
    metrics_path = os.path.join(args.results_dir, f"yolov3_coco_metrics_{_timestamp()}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    write_log_line(log_path, f"[INFO] Saved metrics/results to {metrics_path}")

    write_log_line(log_path, "[INFO] Evaluation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
