"""Build figures/vis_failure_examples.pdf — qualitative NPU vs GPU failures.

Selects frames where the GPU (FP32) detects a safety-critical small-class object
(person, bicycle, motorcycle, traffic_light, stop_sign) that the INT8 NPU misses,
and saves a panel image annotated with:
  GREEN = GPU detection (FP32, the reference)
  RED   = object the GPU caught but the NPU did NOT detect

No background contention (L0), no sweep — the figure is a single-camera FP32 vs
INT8 qualitative diagnostic that supports the Sec. 5 narrative about the
quantization loss falling on the safety-critical classes.

Output: paper/figures/vis_failure_examples.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))
import _step_d_common as cm
from _step_d_common import (load_val, load_split_for_sid, npu_infer,
                            preload_npu_instances)
from step0_compare_devices import DATA, CONF, IOU, IMG_SIZE, N_AHD

PIN = json.loads((SCRIPT_DIR.parent / "results" / "rev7_pin.json").read_text())
MXQ = PIN["mxq_path"]

# Safety-critical small classes (AHD ids):
#   person=0, bicycle=1, motorcycle=3, traffic_light=6, stop_sign=7
SAFETY_SMALL = {0, 1, 3, 6, 7}
CLASS_NAMES = {0:"person",1:"bicycle",2:"car",3:"motorcycle",
               4:"bus",5:"truck",6:"traffic_light",7:"stop_sign"}

OUT = SCRIPT_DIR.parent.parent / "paper" / "figures" / "vis_failure_examples.pdf"

# We scan multiple sids known to be small-dominant or medium-mixed for variety.
SCAN_SIDS = [2, 13, 22, 16, 8, 17]
FRAMES_PER_SID_LIMIT = 200  # scan only first 200 frames per sid to keep wall short
N_PANELS = 4
MIN_AREA_DIFF = 32 * 32     # only consider objects above small-object threshold
IOU_MATCH = 0.4              # match GPU detection to NPU detection


def load_gpu():
    import os
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO
    m = YOLO("yolo11s.pt")
    m.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
              imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False, device="cuda")
    return m


def gpu_predict(m, img_path, coco_mapping):
    r = m.predict(str(img_path), imgsz=IMG_SIZE, conf=CONF, iou=IOU,
                  verbose=False, device="cuda")[0]
    if not len(r.boxes):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    coco_ids = r.boxes.cls.cpu().numpy().astype(int)
    ahd = coco_mapping[coco_ids]
    sel = ahd < N_AHD
    bb = r.boxes.xyxy.cpu().numpy()[sel].astype(np.float32)
    sc = r.boxes.conf.cpu().numpy()[sel].astype(np.float32)
    lb = ahd[sel].astype(np.int32)
    return bb, sc, lb


def npu_predict(npu_model, img_path, frame_shape, coco_mapping):
    xyxy, sc, coco_cls = npu_infer(img_path, frame_shape, npu_model)
    if len(xyxy) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    ahd = coco_mapping[coco_cls]
    sel = ahd < N_AHD
    return xyxy[sel], sc[sel], ahd[sel].astype(np.int32)


def iou_xyxy(a, b):
    """Vectorized IoU between one box and many."""
    if len(b) == 0:
        return np.zeros(0, np.float32)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def find_failure_frames():
    """Scan a few sids and return up to N candidate (path, gpu_boxes, npu_boxes,
    missed_boxes) where missed_boxes are GPU detections of safety-critical small
    classes that the NPU did not match."""
    val = load_val()
    print("[vis] loading GPU + NPU (global8, legacy mxq)…")
    gpu = load_gpu()
    preload_npu_instances(1, infer_mode="global8")
    npu = cm._NPU_INSTANCES[0]
    candidates = []
    for sid in SCAN_SIDS:
        split = load_split_for_sid(val, sid)
        imgs = split["imgs"][:FRAMES_PER_SID_LIMIT]
        coco_mapping = split["coco_mapping"]
        seq_dir = split["seq_dir"]
        frame_shape = (imgs[0]["height"], imgs[0]["width"])
        print(f"[vis] sid={sid}  scanning {len(imgs)} frames…")
        for ii, img in enumerate(imgs):
            img_path = DATA / seq_dir / img["name"]
            try:
                gpu_bb, gpu_sc, gpu_lb = gpu_predict(gpu, img_path, coco_mapping)
                npu_bb, npu_sc, npu_lb = npu_predict(npu, img_path, frame_shape, coco_mapping)
            except Exception as e:
                continue
            if not len(gpu_bb):
                continue
            # Find safety-critical small-class GPU detections that NPU missed
            missed = []
            for k in range(len(gpu_bb)):
                if gpu_lb[k] not in SAFETY_SMALL:
                    continue
                # IoU with same-class NPU detections
                same_cls = npu_lb == gpu_lb[k]
                if same_cls.any():
                    iou_vals = iou_xyxy(gpu_bb[k], npu_bb[same_cls])
                    if (iou_vals > IOU_MATCH).any():
                        continue
                missed.append({"bbox": gpu_bb[k], "score": float(gpu_sc[k]),
                               "class": int(gpu_lb[k])})
            if not missed:
                continue
            # Score: # safety-critical missed (more = better example)
            score = len(missed)
            # Bonus: pedestrian or stop_sign / traffic_light are headline
            if any(m["class"] in (0, 6, 7) for m in missed):
                score += 1
            candidates.append({
                "sid": sid, "frame_idx": ii, "img_path": str(img_path),
                "gpu_bb": gpu_bb, "gpu_sc": gpu_sc, "gpu_lb": gpu_lb,
                "npu_bb": npu_bb, "npu_sc": npu_sc, "npu_lb": npu_lb,
                "missed": missed, "score": score,
                "frame_shape": frame_shape,
            })
            # cap PER SID so all sids get scanned for diversity
            if sum(1 for c in candidates if c["sid"] == sid) >= 12:
                break
    # Sort by score (more interesting first) and keep diverse sids
    candidates.sort(key=lambda c: -c["score"])
    chosen = []; sids_seen = set()
    # First pass: one frame per sid for variety
    for c in candidates:
        if c["sid"] in sids_seen: continue
        chosen.append(c); sids_seen.add(c["sid"])
        if len(chosen) >= N_PANELS: break
    # Backfill if we don't have enough
    if len(chosen) < N_PANELS:
        for c in candidates:
            if c not in chosen:
                chosen.append(c)
                if len(chosen) >= N_PANELS: break
    return chosen[:N_PANELS]


def annotate_frame(img_bgr, frame_info):
    """Draw GPU detections in green, missed (GPU detected, NPU missed safety-critical) in red."""
    img = img_bgr.copy()
    H, W = img.shape[:2]
    # Draw all GPU dets in green (thin), then highlight missed in red (thick)
    for bb, sc, lb in zip(frame_info["gpu_bb"], frame_info["gpu_sc"], frame_info["gpu_lb"]):
        x1, y1, x2, y2 = bb.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 60), 2)
    for m in frame_info["missed"]:
        bb = m["bbox"]
        x1, y1, x2, y2 = bb.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 230), 4)
        label = f"{CLASS_NAMES[m['class']]} {m['score']:.2f} (NPU miss)"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(y1 - lh - 6, 0)), (x1 + lw + 4, y1), (40, 40, 230), -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, lh)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def render(chosen):
    if not chosen:
        print("[vis] no failure-mode frames found; cannot render figure.")
        return False
    n = len(chosen)
    cols = 2 if n >= 2 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.7 * rows),
                              squeeze=False)
    for k, c in enumerate(chosen):
        r, col = k // cols, k % cols
        ax = axes[r][col]
        img_bgr = cv2.imread(c["img_path"])
        if img_bgr is None:
            ax.set_visible(False); continue
        # Crop to top 75% to give boxes room
        annotated = annotate_frame(img_bgr, c)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.set_xticks([]); ax.set_yticks([])
        missed_summary = ", ".join(
            f"{CLASS_NAMES[m['class']]}" for m in c["missed"][:5])
        ax.set_title(f"sid {c['sid']}, frame {c['frame_idx']} — "
                     f"NPU misses: {missed_summary}", fontsize=9)
    # Hide extras
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    # Legend
    legend_handles = [
        mpatches.Patch(color=(60/255, 200/255, 60/255), label="GPU (FP32) detection"),
        mpatches.Patch(color=(230/255, 40/255, 40/255),
                       label="object NPU (INT8) misses but GPU detects"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")
    return True


def main():
    chosen = find_failure_frames()
    print(f"[vis] selected {len(chosen)} panel frames:")
    for c in chosen:
        miss = [CLASS_NAMES[m["class"]] for m in c["missed"]]
        print(f"  sid={c['sid']:>2d} frame={c['frame_idx']:>3d}  misses={miss}")
    render(chosen)


if __name__ == "__main__":
    main()
