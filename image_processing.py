"""
Image processing functions for the multimodel scheduling application.
"""
import cv2
import numpy as np
import os

# Constants (default). Will be overridden by model input if provided at call time
input_width = input_height = 608

# COCO class labels
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def letterbox(image, new_shape, color=(114, 114, 114)):
    # new_shape = (W, H)
    h, w = image.shape[:2]
    new_w, new_h = int(new_shape[0]), int(new_shape[1])
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)

def image_preprocess(raw_input_img, target_width, target_height, use_letterbox=True):
    """
    Preprocess an image for neural network input.

    Args:
        raw_input_img (np.ndarray): BGR image
        target_width (int)
        target_height (int)
        use_letterbox (bool): if True, keep aspect ratio with padding (letterbox).
                              if False, plain resize.

    Returns:
        np.ndarray: [1,3,H,W] float32, RGB, 0~1
    """
    if use_letterbox:
        lb_img, _, _ = letterbox(raw_input_img, (target_width, target_height))
        img = lb_img
    else:
        img = cv2.resize(raw_input_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return img

def yolo_preprocess_local(raw_input_img, input_size=None):
    """
    Returns: (tensor[N,3,H,W], meta)
    meta: {"orig_w","orig_h","ratio","pad":(dw,dh),"input_size":(W,H)}
    """
    H = input_height
    W = input_width
    if input_size and isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        W, H = int(input_size[0]), int(input_size[1])

    img_l, r, (dw, dh) = letterbox(raw_input_img, (W, H))
    img = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    meta = {
        "orig_w": int(raw_input_img.shape[1]),
        "orig_h": int(raw_input_img.shape[0]),
        "ratio": float(r),
        "pad": (int(dw), int(dh)),
        "input_size": (int(W), int(H)),
    }
    return img, meta

def resnet50_preprocess_local(raw_input_img):
    """
    Preprocess an image for ResNet50 model input.
    
    Args:
        raw_input_img: Raw input image (numpy array)
        
    Returns:
        Preprocessed image data as numpy array
    """
    return image_preprocess(raw_input_img, 224, 224)

# --- ResNet50 postprocess (local) ---
# Load ImageNet class labels once
try:
    _IMAGENET_CLASSES = None
    classes_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as _f:
            _IMAGENET_CLASSES = [line.strip() for line in _f]
except Exception:
    _IMAGENET_CLASSES = None


def resnet50_postprocess_local(logits, original_img=None, draw=True):
    """
    Simple post-processing for ResNet50 classification logits.
    - Computes argmax over class dimension.
    - Optionally draws the class name on provided image.

    Args:
        logits (np.ndarray): Model output logits or probabilities. Shapes like (N,1000), (1,1000), or (1000,).
        original_img (np.ndarray, optional): BGR image to draw on.
        draw (bool): If True and image provided, draw top-1 class text.

    Returns:
        tuple: (class_id, class_name, drawn_image_or_none)
    """
    arr = np.asarray(logits)
    if arr.ndim == 2:
        arr_use = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    else:
        arr_use = np.squeeze(arr)
    class_id = int(np.argmax(arr_use)) if arr_use.size > 0 else -1

    if _IMAGENET_CLASSES is not None and 0 <= class_id < len(_IMAGENET_CLASSES):
        class_name = _IMAGENET_CLASSES[class_id]
    else:
        class_name = f"Class ID: {class_id}"

    img_out = None
    if draw and original_img is not None:
        img_out = original_img
        try:
            cv2.putText(img_out, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        except Exception:
            pass

    return class_id, class_name, img_out

def draw_detection_boxes(img, box, score, class_id):
    """Draw a single XYWH box on img (clamped to image)."""
    H, W = img.shape[:2]
    class_name = coco_classes[class_id] if 0 <= class_id < len(coco_classes) else f"ID:{class_id}"
    label = f"{class_name} {score:.2f}"

    x, y, w, h = map(int, box)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))

    color = (0, 255, 0)
    thickness = max(2, int(0.002 * max(H, W)))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    yb = y - 2
    yt = yb - th - 2
    if yt < 0:
        yt = y + 2
        yb = yt + th + 2
    cv2.rectangle(img, (x, yt), (x + tw, yb), color, -1)
    cv2.putText(img, label, (x, yb - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



def yolo_postprocess_cpu(outputs, original_img, meta, confidence_thres=0.5, iou_thres=0.5):
    """
    YOLOv3 ONNX postprocess (CPU).
      A) (boxes[1,N,4]/[N,4], scores[1,C,N]/[C,N], indices[M,3])
      B) [1,N,85] (xywh+obj+cls)
      C) [1,N,6]  (좌표 4개 + score + class)  ← 좌표 형식 자동 추정
    최종 드로잉은 현재 캔버스(original_img) 크기 기준.
    """
    # using module-level imports (np, cv2, os)

    # ---------- helpers ----------
    def ltwh_to_xyxy(ltwh):
        x, y, w, h = ltwh.T
        return np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)

    def yxhw_to_xyxy(yxhw):
        y, x, h, w = yxhw.T
        return np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)

    def xywh_to_xyxy(xywh):
        x, y, w, h = xywh.T
        return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1).astype(np.float32)

    def deletterbox_and_to_canvas(boxes_xyxy):
        """레터박스 해제 → 원본 좌표 → 현재 캔버스 좌표."""
        if boxes_xyxy.size == 0: return boxes_xyxy
        W_in, H_in = meta.get("input_size", (608, 608))
        w0, h0 = meta["orig_w"], meta["orig_h"]
        Hc, Wc = original_img.shape[:2]
        b = boxes_xyxy.astype(np.float32).copy()

        mn, mx = float(np.min(b)), float(np.max(b))
        # 정규화(0~1)면 원본 스케일로
        if 0.0 <= mn and mx <= 1.5:
            b[:, [0, 2]] *= w0
            b[:, [1, 3]] *= h0
        # 입력크기 좌표(레터박스)면 원본 스케일로
        elif mx <= max(W_in, H_in) + 1.5:
            dw, dh = meta["pad"]; r = meta["ratio"]
            b[:, [0, 2]] = (b[:, [0, 2]] - dw) / r
            b[:, [1, 3]] = (b[:, [1, 3]] - dh) / r
        # 그 외: 이미 원본 좌표라고 가정

        # 원본 경계 클램프
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w0)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h0)

        # 현재 캔버스로 스케일 (보통 동일 프레임이므로 변화 없음)
        if (Wc, Hc) != (w0, h0):
            sx, sy = Wc / float(w0), Hc / float(h0)
            b *= np.array([sx, sy, sx, sy], dtype=np.float32)
        return b

    # 좌표 포맷 추정: 0=XYXY, 1=YXYX, 2=LTWH, 3=YXHW
    def infer_box_format(sample_box4):
        if sample_box4.size == 0: return 0
        Hc, Wc = original_img.shape[:2]
        # 후보 4종을 XYXY로 변환
        cand = [
            sample_box4.astype(np.float32),                  # XYXY
            sample_box4[:, [1, 0, 3, 2]].astype(np.float32),# YXYX -> XYXY
            ltwh_to_xyxy(sample_box4.astype(np.float32)),   # LTWH -> XYXY
            yxhw_to_xyxy(sample_box4.astype(np.float32)),   # YXHW -> XYXY
        ]

        def score(xyxy):
            b = deletterbox_and_to_canvas(xyxy)
            if b.size == 0: return -1e9
            w = np.clip(b[:, 2] - b[:, 0], 0, None)
            h = np.clip(b[:, 3] - b[:, 1], 0, None)
            # 조건 점수
            valid = np.mean((w > 2) & (h > 2) & (b[:,0] >= 0) & (b[:,1] >= 0) &
                            (b[:,2] <= Wc+1) & (b[:,3] <= Hc+1) & (w < 0.98*Wc) & (h < 0.98*Hc))
            aspect = w / (h + 1e-6)
            aspect_good = np.mean((aspect > 0.1) & (aspect < 10.0))
            thin_penalty = np.mean((w < 4) | (h < 4))
            return 3.0*valid + 1.0*aspect_good - 0.5*thin_penalty

        scores = [score(c) for c in cand]
        return int(np.argmax(scores))

    def apply_format(box4, fmt):
        if fmt == 0:   # XYXY
            return box4.astype(np.float32)
        if fmt == 1:   # YXYX
            return box4[:, [1,0,3,2]].astype(np.float32)
        if fmt == 2:   # LTWH
            return ltwh_to_xyxy(box4.astype(np.float32))
        # fmt == 3: YXHW
        return yxhw_to_xyxy(box4.astype(np.float32))

    # ---------- outputs -> list ----------
    arrs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

    # ---------- try Case A (boxes, scores, indices) ----------
    boxes_arr = scores_2d = idx_2d = None
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[-1] == 4 and np.issubdtype(a.dtype, np.floating):
            boxes_arr = a.reshape(-1, 4).astype(np.float32)
        elif (a.ndim in (2,3)) and np.issubdtype(a.dtype, np.floating):
            a2 = a[0] if a.ndim == 3 else a
            if a2.ndim == 2 and a2.shape[0] in (80,81,90):
                scores_2d = a2.astype(np.float32)  # (C,N)
        elif (a.ndim in (2,3)) and np.issubdtype(a.dtype, np.integer) and a.shape[-1] == 3:
            idx_2d = a.reshape(-1, 3).astype(np.int32)

    boxes_draw, scores_draw, cls_draw = [], [], []

    if boxes_arr is not None and scores_2d is not None and idx_2d is not None:
        C, N = scores_2d.shape
        picked_boxes, picked_scores, picked_cls = [], [], []
        for b_i, c_i, j_i in idx_2d:
            if 0 <= c_i < C and 0 <= j_i < N:
                picked_boxes.append(boxes_arr[j_i])
                picked_scores.append(scores_2d[c_i, j_i])
                picked_cls.append(c_i)

        if picked_boxes:
            box4 = np.stack(picked_boxes).astype(np.float32)
            scores = np.asarray(picked_scores, dtype=np.float32)
            clses  = np.asarray(picked_cls, dtype=np.int32)
            m = scores >= confidence_thres
            box4 = box4[m]; scores = scores[m]; clses = clses[m]

            # ---- 좌표 형식 결정(한 번만) ----
            force_env = os.environ.get("YOLO_FORCE_FMT")  # 디버그용 강제 포맷(0~3)
            if force_env is not None and force_env.isdigit():
                fmt = int(force_env)
                meta["box_fmt"] = fmt
            else:
                fmt = meta.get("box_fmt", None)
                if fmt is None:
                    fmt = infer_box_format(box4[:64])
                    meta["box_fmt"] = fmt

            boxes_xyxy = apply_format(box4, fmt)
            boxes = deletterbox_and_to_canvas(boxes_xyxy)

            # per-class NMS & draw
            for c in np.unique(clses):
                inds = np.where(clses == c)[0]
                if inds.size == 0: continue
                b = boxes[inds]
                b_xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
                keep = cv2.dnn.NMSBoxes(b_xywh.tolist(),
                                        scores[inds].astype(np.float32).tolist(),
                                        confidence_thres, iou_thres)
                if keep is None or len(keep) == 0: continue
                keep = [int(k) if isinstance(k,(int,np.integer)) else int(k[0]) for k in keep]
                for j in keep:
                    x1,y1,x2,y2 = b[j]
                    boxes_draw.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                    scores_draw.append(float(scores[inds][j]))
                    cls_draw.append(int(c))

    else:
        # ---------- Case B/C ----------
        if len(arrs) == 1:
            out = np.asarray(arrs[0])
        else:
            try:
                out = np.concatenate([np.asarray(o).reshape((o.shape[0], -1, o.shape[-1])) for o in arrs], axis=1)
            except Exception:
                out = np.asarray(max(arrs, key=lambda a: np.asarray(a).size))

        preds2d = out[0] if (out.ndim == 3 and out.shape[0] == 1 and out.shape[2] >= 6) \
                        else (out if out.ndim == 2 and out.shape[1] >= 6 else None)

        if preds2d is not None:
            if preds2d.shape[1] == 6:
                box4 = preds2d[:, 0:4].astype(np.float32)
                scores = preds2d[:, 4].astype(np.float32)
                clses  = preds2d[:, 5].astype(np.int32)
                fmt = meta.get("box_fmt", None)
                if fmt is None:
                    fmt = infer_box_format(box4[:64])
                    meta["box_fmt"] = fmt
                boxes_xyxy = apply_format(box4, fmt)
            else:
                # 전형적 YOLO (xywh + obj + cls)
                xywh = preds2d[:, 0:4].astype(np.float32)
                obj  = preds2d[:, 4].astype(np.float32)
                cls_probs = preds2d[:, 5:] if preds2d.shape[1] > 5 else np.zeros((preds2d.shape[0],1), np.float32)
                cls_ids = np.argmax(cls_probs, axis=1)
                cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids].astype(np.float32)
                scores = (obj * cls_scores).astype(np.float32)
                clses  = cls_ids.astype(np.int32)
                boxes_xyxy = xywh_to_xyxy(xywh)

            m = scores >= confidence_thres
            boxes_xyxy = boxes_xyxy[m]; scores = scores[m]; clses = clses[m]
            boxes = deletterbox_and_to_canvas(boxes_xyxy)

            for c in np.unique(clses):
                inds = np.where(clses == c)[0]
                if inds.size == 0: continue
                b = boxes[inds]
                b_xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
                keep = cv2.dnn.NMSBoxes(b_xywh.tolist(),
                                        scores[inds].astype(np.float32).tolist(),
                                        confidence_thres, iou_thres)
                if keep is None or len(keep) == 0: continue
                keep = [int(k) if isinstance(k,(int,np.integer)) else int(k[0]) for k in keep]
                for j in keep:
                    x1,y1,x2,y2 = b[j]
                    boxes_draw.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                    scores_draw.append(float(scores[inds][j]))
                    cls_draw.append(int(c))

    # ---------- draw ----------
    for i in range(len(boxes_draw)):
        draw_detection_boxes(original_img, boxes_draw[i], scores_draw[i], cls_draw[i])

    # (옵션) 포맷 디버그용 표시
    fmt_txt = str(meta.get("box_fmt", ""))
    if fmt_txt != "":
        cv2.putText(original_img, f"fmt={fmt_txt}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return original_img



def yolo_postprocess_npu(output, original_img, meta, confidence_thres=0.5, iou_thres=0.5):
    """
    Post-process YOLO model output (NPU version) given pre-letterboxed meta.
    Expects output[0]: boxes (left,top,right,bottom,conf), output[1]: class ids
    """
    output_box = np.squeeze(output[0])
    output_label = np.squeeze(output[1])

    rows = output_box.shape[0]
    boxes_xyxy = []
    scores = []
    class_ids = []

    for i in range(rows):
        conf = float(output_box[i][4])
        if conf >= confidence_thres:
            left, top, right, bottom = output_box[i][:4]
            boxes_xyxy.append([left, top, right, bottom])
            scores.append(conf)
            class_ids.append(int(output_label[i]))

    # NMS
    drawn_boxes = []
    if boxes_xyxy:
        b = np.array(boxes_xyxy, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        # OpenCV NMS uses [x,y,w,h]
        b_xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
        keep = cv2.dnn.NMSBoxes(b_xywh.tolist(), s.tolist(), confidence_thres, iou_thres)
        if keep is not None and len(keep) > 0:
            keep = [int(k) if isinstance(k, (int, np.integer)) else int(k[0]) for k in keep]
            dw, dh = meta["pad"]
            r = meta["ratio"]
            w0, h0 = meta["orig_w"], meta["orig_h"]
            for j in keep:
                x1, y1, x2, y2 = b[j]
                # de-letterbox
                x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
                x1 = np.clip(x1, 0, w0); y1 = np.clip(y1, 0, h0)
                x2 = np.clip(x2, 0, w0); y2 = np.clip(y2, 0, h0)

                # Validate box to avoid drawing abnormal long thin lines
                w = float(x2 - x1)
                h = float(y2 - y1)
                # minimum size in pixels (post de-letterbox)
                min_sz = 4.0
                # reject if too small in either dimension
                if w < min_sz or h < min_sz:
                    continue
                # reject boxes that span (almost) the whole image in one dimension
                if w > 0.98 * w0 or h > 0.98 * h0:
                    continue
                # reject extreme aspect ratios (e.g., long horizontal/vertical lines)
                ar = w / (h + 1e-6)
                if ar > 25.0 or ar < 1.0/25.0:
                    continue

                box = [int(x1), int(y1), int(w), int(h)]
                draw_detection_boxes(original_img, box, float(s[j]), int(class_ids[j]))
                drawn_boxes.append(box)

    return original_img, drawn_boxes