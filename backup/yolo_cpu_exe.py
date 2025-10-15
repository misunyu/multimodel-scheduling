# -*- coding: utf-8 -*-
# Fix YOLO ONNX input to 1x3x608x608, run inference on a test image,
# and optionally evaluate accuracy on one image.

import os, json
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import onnxruntime as ort


# ======= Inference config =======
TARGET_SHAPE = (608, 608)   # H, W
CONF_THRES   = 0.05
IOU_THRES    = 0.5

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]



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

# ======= 4) Simple AP50 for single image =======
def iou_single(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    areaA = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    areaB = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union

def eval_ap50(dets: List[Dict], gts: List[Dict]) -> Dict[str, float]:
    if len(gts) == 0 and len(dets) == 0: return {"AP50":1.0,"precision":1.0,"recall":1.0}
    if len(dets) == 0: return {"AP50":0.0,"precision":0.0,"recall":0.0}
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    gt_used = [False]*len(gts)
    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)
    for i, d in enumerate(dets):
        cand = [(j, iou_single(np.array(d["bbox"]), np.array(gts[j]["bbox"])))
                for j in range(len(gts)) if gts[j]["cls"] == d["cls"]]
        if not cand: fp[i]=1.0; continue
        j_best, iou_best = max(cand, key=lambda x: x[1])
        if iou_best >= 0.5 and not gt_used[j_best]:
            tp[i]=1.0; gt_used[j_best]=True
        else:
            fp[i]=1.0
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = tp_cum / max(len(gts), 1e-9)
    ap = 0.0
    for r_th in [i/10 for i in range(11)]:
        p = np.max(precisions[recalls >= r_th]) if np.any(recalls >= r_th) else 0.0
        ap += p
    ap /= 11.0
    precision = precisions[-1] if precisions.size else 0.0
    recall = recalls[-1] if recalls.size else 0.0
    return {"AP50": float(ap), "precision": float(precision), "recall": float(recall)}

def verify_converted_vs_original(inp: np.ndarray,
                                 meta: Dict[str, Any],
                                 base_model_path: str,
                                 conv_model_path: str,
                                 atol: float = 1e-5,
                                 rtol: float = 1e-4) -> None:
    def run_model(path: str):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = build_input_feed(sess, inp, meta)
        names = [o.name for o in sess.get_outputs()]
        arrs = sess.run(names, feeds)
        return sess, names, arrs

    sessA, namesA, arrsA = run_model(base_model_path)
    sessB, namesB, arrsB = run_model(conv_model_path)

    n = min(len(arrsA), len(arrsB))
    print(f"[VERIFY] base outputs={len(arrsA)}, converted outputs={len(arrsB)}, compare first {n}")

    name_map = {}
    for i, na in enumerate(namesA):
        cand = [j for j, nb in enumerate(namesB) if nb.startswith(na)]
        name_map[i] = cand[0] if cand else (i if i < len(namesB) else None)

    all_pass = True
    for i in range(n):
        j = name_map.get(i, i)
        if j is None:
            print(f"[FAIL] cannot pair output {i}: {namesA[i]}")
            all_pass = False
            continue

        a = np.asarray(arrsA[i])
        b = np.asarray(arrsB[j])

        if a.size == b.size and a.shape != b.shape:
            try: b = b.reshape(a.shape)
            except Exception: pass

        same_dtype_class = (np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)) or \
                           (np.issubdtype(a.dtype, np.bool_) and np.issubdtype(b.dtype, np.bool_))
        if same_dtype_class:
            ok = np.array_equal(a, b)
            status = "PASS" if ok else "FAIL"
            if not ok: all_pass = False
            print(f"[CMP#{i}] {namesA[i]} vs {namesB[j]} "
                  f"shapeA={a.shape} shapeB={b.shape} dtypeA={a.dtype} dtypeB={b.dtype} -> {status} (exact)")
        else:
            af = a.astype(np.float64, copy=False)
            bf = b.astype(np.float64, copy=False)
            if af.size == bf.size:
                diff = np.abs(af - bf)
                max_abs = float(diff.max()) if diff.size else 0.0
                denom = np.maximum(np.abs(af), np.abs(bf))
                max_rel = float((diff / (denom + 1e-12)).max()) if diff.size else 0.0
                ok = (max_abs <= atol) or (max_rel <= rtol)
                status = "PASS" if ok else "FAIL"
                if not ok: all_pass = False
                print(f"[CMP#{i}] {namesA[i]} vs {namesB[j]} "
                      f"shapeA={a.shape} shapeB={b.shape} dtypeA={a.dtype} dtypeB={b.dtype} "
                      f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} -> {status}")
            else:
                all_pass = False
                print(f"[CMP#{i}] {namesA[i]} vs {namesB[j]} size mismatch: {af.size} vs {bf.size} -> FAIL")

    print(f"[VERIFY RESULT] {'PASS' if all_pass else 'FAIL'}")

# ======= 5) Main =======
def run(image_path: str, gt_json: str = None) -> None:
    # 1) Prepare model (use pre-converted model)
    # NOTE: Per request, disable conversion steps and use the already converted model directly.
    # if not os.path.exists(OUTPUT_MODEL):
    #     fix_model_input(INPUT_MODEL, OUTPUT_MODEL)
    # convert_outputs_like_reference(OUTPUT_MODEL, REF_MODEL, CONVERTED_MODEL)
    model_for_infer = "./models/yolov3_big/model/yolov3_big.onnx"

    # 2) Preprocess
    inp, meta = preprocess(image_path)
    # verify_converted_vs_original(inp, meta, OUTPUT_MODEL, CONVERTED_MODEL, atol=1e-5, rtol=1e-4)  # disabled

    # 3) Inference
    sess = ort.InferenceSession(model_for_infer, providers=["CPUExecutionProvider"])
    feeds = build_input_feed(sess, inp, meta)
    out_names = [o.name for o in sess.get_outputs()]
    pred = sess.run(out_names, feeds)

    # Debug: print raw outputs
    for i, o in enumerate(sess.get_outputs()):
        arr = np.asarray(pred[i])
        mn = float(arr.min()) if arr.size else 0.0
        mx = float(arr.max()) if arr.size else 0.0
        print(f"[OUT] {o.name}: shape={arr.shape}, dtype={arr.dtype}, min={mn:.4f}, max={mx:.4f}")

    # 4) Parse detections
    boxes, scores, cls_ids, already_scaled = parse_outputs(sess, pred)

    # De-letterbox ONLY if outputs are not on the original image scale
    if boxes.shape[0] > 0 and (not already_scaled):
        boxes = deletterbox_boxes(boxes, meta)

    print(f"Detections: {len(boxes)} (conf >= {CONF_THRES})")
    for i in range(min(20, len(boxes))):
        x1, y1, x2, y2 = boxes[i].tolist()
        c = int(cls_ids[i])
        name = COCO80[c] if 0 <= c < len(COCO80) else str(c)
        print(f"[{i:02d}] {name:>12s}  conf={scores[i]:.3f}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # 5) Load GT (custom JSON → sidecar → COCO)
    gt_loaded = False
    gt_objs: List[Dict[str, Any]] = []

    # (a) custom JSON path if provided
    if gt_json and isinstance(gt_json, str) and os.path.isfile(gt_json):
        try:
            with open(gt_json, "r") as f:
                gt = json.load(f)
            gt_objs = [
                {"cls": int(o["cls"]),
                 "bbox": [float(o["bbox"][0]), float(o["bbox"][1]),
                          float(o["bbox"][2]), float(o["bbox"][3])]}
                for o in gt.get("objects", []) if len(o.get("bbox", [])) == 4
            ]
            print(f"Loaded GT from file: {gt_json}")
            gt_loaded = True
        except Exception as e:
            print(f"Failed to load GT JSON '{gt_json}': {e}")

    # (b) sidecar next to image (same basename with .json)
    if not gt_loaded:
        sidecar = os.path.splitext(image_path)[0] + ".json"
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r") as f:
                    gt = json.load(f)
                gt_objs = [
                    {"cls": int(o["cls"]),
                     "bbox": [float(o["bbox"][0]), float(o["bbox"][1]),
                              float(o["bbox"][2]), float(o["bbox"][3])]}
                    for o in gt.get("objects", []) if len(o.get("bbox", [])) == 4
                ]
                print(f"Loaded GT from sidecar: {sidecar}")
                gt_loaded = True
            except Exception as e:
                print(f"Failed to load sidecar GT '{sidecar}': {e}")

    # (c) COCO annotation auto-discovery
    if not gt_loaded:
        try:
            base = os.path.basename(image_path)
            split = "val2017" if "val2017" in image_path else ("train2017" if "train2017" in image_path else None)
            if split is not None:
                ann_path = os.path.join("../datasets/coco/annotations", f"instances_{split}.json")
                if os.path.isfile(ann_path):
                    with open(ann_path, "r") as f:
                        coco = json.load(f)
                    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
                    name_to_idx = {name: i for i, name in enumerate(COCO80)}
                    img_rec = next((im for im in coco.get("images", []) if im.get("file_name") == base), None)
                    if img_rec is not None:
                        img_id = img_rec.get("id")
                        anns = [a for a in coco.get("annotations", [])
                                if a.get("image_id") == img_id and not a.get("iscrowd", 0)]
                        gts = []
                        for a in anns:
                            cname = cat_id_to_name.get(a.get("category_id"))
                            if cname is None or cname not in name_to_idx:
                                continue
                            cls_idx = name_to_idx[cname]
                            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
                            gts.append({"cls": int(cls_idx),
                                        "bbox": [float(x), float(y), float(x + w), float(y + h)]})
                        gt_objs = gts
                        print(f"Loaded GT from COCO annotations: {ann_path} (image: {base}, objects: {len(gt_objs)})")
                        gt_loaded = True
        except Exception as e:
            print(f"COCO GT auto-discovery failed: {e}")

    # 6) Evaluate if GT available
    if gt_loaded:
        det_list = [{"cls": int(cls_ids[i]), "bbox": boxes[i].tolist(), "score": float(scores[i])}
                    for i in range(len(boxes))]
        metrics = eval_ap50(det_list, gt_objs)
        print(f"AP@0.5={metrics['AP50']:.3f}  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}")
    else:
        print("GT not provided. Skipping accuracy. Provide JSON: {'objects':[{'cls':int,'bbox':[x1,y1,x2,y2]}]}")

def debug_iou_per_det(dets, gts, top_k=10):
    # dets: [{'cls':int,'bbox':[x1,y1,x2,y2],'score':float}]
    # gts : [{'cls':int,'bbox':[x1,y1,x2,y2]}]
    def iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1])
        x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter=max(0.0,x2-x1)*max(0.0,y2-y1)
        ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
        return inter/max(ua+ub-inter,1e-9)

    for i,d in enumerate(sorted(dets, key=lambda x: x['score'], reverse=True)[:top_k]):
        same_cls = [g for g in gts if g['cls']==d['cls']]
        ious = [iou(d['bbox'], g['bbox']) for g in same_cls] if same_cls else []
        best_iou = max(ious) if ious else 0.0
        print(f"[DBG] det#{i:02d} cls={d['cls']} score={d['score']:.3f} best_IoU_same_cls={best_iou:.3f}  "
              f"bbox={d['bbox']}")

def debug_matches(boxes, scores, cls_ids, gt_objs, top_k=15):
    def iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1])
        x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter=max(0.0,x2-x1)*max(0.0,y2-y1)
        ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
        return inter/max(ua+ub-inter,1e-9)

    order = np.argsort(-scores)
    for k,i in enumerate(order[:top_k]):
        bb = boxes[i].tolist(); pc = int(cls_ids[i])
        # best IoU ignoring class
        j_best, best_iou = -1, 0.0
        for j, g in enumerate(gt_objs):
            ii = iou(bb, g["bbox"])
            if ii > best_iou: j_best, best_iou = j, ii
        gt_cls = gt_objs[j_best]["cls"] if j_best>=0 else -1
        same = (gt_cls == pc)
        pred_name = COCO80[pc] if 0 <= pc < len(COCO80) else str(pc)
        gt_name   = COCO80[gt_cls] if 0 <= gt_cls < len(COCO80) else "-"
        print(f"[MATCH] #{k:02d} pred={pred_name:12s} vs gt={gt_name:12s}  IoU={best_iou:.3f}  same_cls={same}")

if __name__ == "__main__":
    TEST_IMAGE = "./datasets/coco/val2017/000000000139.jpg"
    GT_JSON    = None  # or custom GT json path
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError("Set TEST_IMAGE to a valid path.")
    run(TEST_IMAGE, GT_JSON)
