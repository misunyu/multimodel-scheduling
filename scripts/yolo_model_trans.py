# -*- coding: utf-8 -*-
# Fix YOLO ONNX input to 1x3x608x608, run inference on a test image,
# and optionally evaluate accuracy on one image.

from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import onnxruntime as ort

# ======= Inference config =======
TARGET_SHAPE = (608, 608)   # H, W
CONF_THRES   = 0.05
IOU_THRES    = 0.5


# ======= 2) Pre/Post =======
def letterbox(im: Image.Image, new_shape: Tuple[int,int]) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    w0, h0 = im.size
    nw, nh = new_shape[1], new_shape[0]
    r = min(nw / w0, nh / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    im_resized = im.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (nw, nh), (114, 114, 114))
    pad_w = (nw - new_w) // 2
    pad_h = (nh - new_h) // 2
    canvas.paste(im_resized, (pad_w, pad_h))
    arr = np.asarray(canvas)
    return arr, r, (pad_w, pad_h)

def preprocess(image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    im = Image.open(image_path).convert("RGB")
    arr, ratio, (pad_w, pad_h) = letterbox(im, TARGET_SHAPE)
    arr = arr.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    nchw = np.expand_dims(chw, 0)
    meta = {"orig_w": im.width, "orig_h": im.height, "ratio": ratio, "pad_w": pad_w, "pad_h": pad_h}
    return nchw, meta

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)

def iou(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box1[0], boxes[:,0]); y1 = np.maximum(box1[1], boxes[:,1])
    x2 = np.minimum(box1[2], boxes[:,2]); y2 = np.minimum(box1[3], boxes[:,3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

# ======= 3) Session I/O helpers =======
def build_input_feed(sess: ort.InferenceSession, inp: np.ndarray, meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    feeds: Dict[str, np.ndarray] = {}
    inputs = sess.get_inputs()

    img_in = next((i for i in inputs if isinstance(i.shape, list) and len(i.shape) == 4), inputs[0])
    feeds[img_in.name] = inp

    shape_in = next((i for i in inputs if i.name.lower().endswith("image_shape")), None)
    if shape_in is not None:
        h, w = meta["orig_h"], meta["orig_w"]
        if "float" in shape_in.type:
            arr = np.array([h, w], dtype=np.float32)
        else:
            arr = np.array([h, w], dtype=np.int64)
        if isinstance(shape_in.shape, list) and len(shape_in.shape) == 2:
            arr = arr[None, :]
        feeds[shape_in.name] = arr

    return feeds

def parse_outputs(session: ort.InferenceSession, outputs: List[np.ndarray]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,bool]:
    """
    Returns (boxes_xyxy, scores, cls_ids, already_on_original_scale).

    지원 레이아웃
      A) 인덱스 기반 NMS: boxes [1,N,4]/[N,4], scores [1,C,N]/[C,N], indices [M,3] (int)
      B) per-class scores: scores [1,C,N]/[C,N] + boxes [1,N,4]/[N,4]
      C) flat: [N, 5+num_classes] (xywh+obj+cls)
    """
    out_meta = session.get_outputs()
    arrs = {out_meta[i].name: np.asarray(outputs[i]) for i in range(len(out_meta))}

    boxes_arr = None   # (N,4) float
    scores_2d = None   # (C,N) float
    indices_2d = None  # (M,3) int

    # boxes 후보
    for a in arrs.values():
        if a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == 4 and np.issubdtype(a.dtype, np.floating):
            boxes_arr = a.reshape(-1, 4).astype(np.float32)
            break
    if boxes_arr is None:
        for a in arrs.values():
            a2 = np.squeeze(a)
            if a2.ndim == 2 and a2.shape[1] == 4 and np.issubdtype(a2.dtype, np.floating):
                boxes_arr = a2.astype(np.float32)
                break

    # scores 후보
    plausible_C = {80, 81, 90}
    for a in arrs.values():
        if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] in plausible_C and np.issubdtype(a.dtype, np.floating):
            scores_2d = a[0].astype(np.float32)  # (C,N)
            break
    if scores_2d is None:
        for a in arrs.values():
            a2 = np.squeeze(a)
            if a2.ndim == 2 and a2.shape[0] in plausible_C and np.issubdtype(a2.dtype, np.floating):
                scores_2d = a2.astype(np.float32)
                break

    # indices 후보
    for a in arrs.values():
        a2 = np.squeeze(a)
        if a2.ndim == 2 and a2.shape[1] == 3 and np.issubdtype(a2.dtype, np.integer):
            indices_2d = a2.astype(np.int32)
            break
        if a.ndim == 3 and a.shape[-1] == 3 and np.issubdtype(a.dtype, np.integer):
            indices_2d = a.reshape(-1, 3).astype(np.int32)
            break

    # ---- Case A: indices ----
    if boxes_arr is not None and scores_2d is not None and indices_2d is not None:
        C, N = scores_2d.shape
        picked_boxes, picked_scores, picked_cls = [], [], []
        for b, c, j in indices_2d:
            if 0 <= c < C and 0 <= j < N:
                picked_boxes.append(boxes_arr[j])
                picked_scores.append(scores_2d[c, j])
                picked_cls.append(c)
        if not picked_boxes:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), False
        boxes = np.stack(picked_boxes).astype(np.float32)
        # TF 스타일 (y1,x1,y2,x2) → (x1,y1,x2,y2)
        boxes = boxes[:, [1,0,3,2]]
        scores = np.array(picked_scores, dtype=np.float32)
        cls_ids = np.array(picked_cls, dtype=np.int32)
        m = scores >= CONF_THRES
        return boxes[m], scores[m], cls_ids[m], True  # 레터박스 스케일로 간주

    # ---- Case B: per-class scores + boxes ----
    if boxes_arr is not None and scores_2d is not None:
        S = scores_2d
        if S.max() > 1.0 or S.min() < 0.0:  # logits -> sigmoid
            S = 1.0 / (1.0 + np.exp(-S))
        best_cls = np.argmax(S, axis=0)                 # (N,)
        best_score = S[best_cls, np.arange(S.shape[1])] # (N,)
        m = best_score >= CONF_THRES
        if not np.any(m):
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), False
        boxes = boxes_arr[m]
        # TF 스타일 (y1,x1,y2,x2) → (x1,y1,x2,y2)
        boxes = boxes[:, [1,0,3,2]]
        scores = best_score[m].astype(np.float32)
        cls_ids = best_cls[m].astype(np.int32)
        # per-class NMS
        keep = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            kidx = nms(boxes[idx], scores[idx], IOU_THRES)
            keep.extend(idx[kidx])
        keep = np.array(keep, dtype=np.int32)
        return boxes[keep], scores[keep], cls_ids[keep], True  # 레터박스 스케일로 간주

    # ---- Case C: flat [N, 5+num_classes] ----
    for a in arrs.values():
        a2 = np.squeeze(a)
        if a2.ndim == 2 and a2.shape[1] >= 6 and np.issubdtype(a2.dtype, np.floating):
            flat = a2.astype(np.float32)
            xywh = flat[:, :4]
            obj = flat[:, 4]
            cls_part = flat[:, 5:]
            cls_scores = 1.0 / (1.0 + np.exp(-cls_part)) if (cls_part.max() > 1.0 or cls_part.min() < 0.0) else cls_part
            best_cls = cls_scores.argmax(axis=1)
            best_cls_score = cls_scores.max(axis=1)
            conf = obj * best_cls_score
            mask = conf >= CONF_THRES
            if not np.any(mask):
                return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), False
            xyxy = xywh_to_xyxy(xywh[mask])
            scores = conf[mask].astype(np.float32)
            cls_ids = best_cls[mask].astype(np.int32)
            keep_all = []
            for c in np.unique(cls_ids):
                mm = np.where(cls_ids == c)[0]
                kidx = nms(xyxy[mm], scores[mm], IOU_THRES)
                keep_all.extend(mm[kidx])
            keep_all = np.array(keep_all, dtype=np.int32)
            return xyxy[keep_all], scores[keep_all], cls_ids[keep_all], False

    # Fallback
    return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), False


def deletterbox_boxes(xyxy: np.ndarray, meta: Dict[str,Any]) -> np.ndarray:
    pad_w, pad_h = meta["pad_w"], meta["pad_h"]; r = meta["ratio"]
    x1 = (xyxy[:,0] - pad_w) / r; y1 = (xyxy[:,1] - pad_h) / r
    x2 = (xyxy[:,2] - pad_w) / r; y2 = (xyxy[:,3] - pad_h) / r
    w0, h0 = meta["orig_w"], meta["orig_h"]
    x1 = np.clip(x1, 0, w0-1); y1 = np.clip(y1, 0, h0-1)
    x2 = np.clip(x2, 0, w0-1); y2 = np.clip(y2, 0, h0-1)
    return np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
