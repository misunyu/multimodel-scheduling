"""Mobilint Aries NPU runtime for vision models (detection + classification).

Thin wrapper around `mblt_model_zoo.vision`, replacing the old Neubla `npu`
module. Exposes:

    build_vision_npu(model_name, infer_mode="global8") -> zoo model
    npu_detections(result, frame)                      -> [(x1,y1,x2,y2,score,cls), ...]
    npu_top1(result)                                   -> (class_id, prob)

Detections are returned in ORIGINAL frame coordinates.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

import model_registry as reg


def build_vision_npu(model_name: str, infer_mode: str = "global8", product: str = "aries"):
    """Instantiate a Mobilint NPU vision model from its local .mxq."""
    from mblt_model_zoo import vision

    spec = reg.get(model_name)
    cls_name = spec.get("npu_class")
    cls = getattr(vision, cls_name, None)
    if cls is None:
        raise KeyError(f"mblt_model_zoo.vision has no class '{cls_name}' for {model_name}")
    mxq = spec.get("mxq")
    # Use the local .mxq if present; otherwise let the zoo fetch it from the Hub.
    local_path = mxq if (mxq and os.path.exists(mxq)) else None
    return cls(local_path=local_path, infer_mode=infer_mode, product=product)


def npu_detections(result, frame) -> List[Tuple[float, float, float, float, float, int]]:
    """Convert a Mobilint detection Results object to original-frame boxes.

    The zoo runs its own letterbox during preprocess; `box_cls` (N, 6+) holds
    xyxy, score, cls in letterbox/input coordinates, which we rescale back.
    """
    box_cls = getattr(result, "box_cls", None)
    if box_cls is None:
        return []
    arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
    if arr.shape[0] == 0:
        return []

    pre_cfg = getattr(result, "pre_cfg", {}) or {}
    letterbox_cfg = pre_cfg.get("LetterBox", {}) if isinstance(pre_cfg, dict) else {}
    img_size = letterbox_cfg.get("img_size", 640)
    if isinstance(img_size, (list, tuple)):
        in_h, in_w = int(img_size[0]), int(img_size[1] if len(img_size) > 1 else img_size[0])
    else:
        in_h = in_w = int(img_size)

    h0, w0 = frame.shape[:2]
    gain = min(in_h / h0, in_w / w0)
    pad_x = (in_w - w0 * gain) / 2.0
    pad_y = (in_h - h0 * gain) / 2.0

    xyxy = arr[:, :4].astype(np.float32, copy=True)
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= gain
    np.clip(xyxy[:, [0, 2]], 0, w0, out=xyxy[:, [0, 2]])
    np.clip(xyxy[:, [1, 3]], 0, h0, out=xyxy[:, [1, 3]])
    scores = arr[:, 4].astype(np.float32)
    classes = arr[:, 5].astype(np.int32)
    return [(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]),
             float(scores[i]), int(classes[i])) for i in range(arr.shape[0])]


def npu_top1(result) -> Tuple[int, float]:
    """Extract (class_id, prob) from a Mobilint classification Results object."""
    # The zoo classification result exposes logits/probabilities in a few shapes;
    # be defensive about the attribute name.
    for attr in ("top1", "class_id", "pred"):
        v = getattr(result, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v), 1.0
    for attr in ("probs", "logits", "scores", "output", "cls"):
        v = getattr(result, attr, None)
        if v is not None:
            a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            a = np.squeeze(a)
            if a.ndim >= 1 and a.size > 0:
                cid = int(np.argmax(a))
                p = float(a.reshape(-1)[cid])
                return cid, p
    # last resort: result itself may be array-like
    try:
        a = np.squeeze(np.asarray(result))
        cid = int(np.argmax(a))
        return cid, float(a.reshape(-1)[cid])
    except Exception:
        return -1, 0.0
