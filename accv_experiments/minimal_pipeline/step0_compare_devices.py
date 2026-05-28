"""Compare single-stream sAP across CPU / CUDA / Mobilint NPU for the same
YOLO11s model and one Argoverse-HD val log (sid=5).

All three devices run the model at the same input size (640x640 — the Mobilint
mxq's native quantization size) for an apples-to-apples comparison.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Force CUDA lazy-init up front. Without this, ultralytics' CPU code path can
# leave torch in a state where torch.cuda.device_count() returns 0 by the time
# we want to spin up the GPU runner, even though CUDA is available.
if torch.cuda.is_available():
    torch.cuda.init()


ROOT = Path(__file__).resolve().parent.parent.parent
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
DATA = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-1.1/argoverse-tracking"
SID = 5
FPS = 30.0
IMG_SIZE = 640
CONF = 0.25
IOU = 0.45
N_AHD = 8


# ---------- shared dataset prep ----------------------------------------------

def load_split():
    with open(ANNOT) as f:
        val = json.load(f)
    imgs = sorted([i for i in val["images"] if i["sid"] == SID], key=lambda x: x["fid"])
    img_ids = {i["id"] for i in imgs}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    coco_mapping = np.asarray(val["coco_mapping"])
    seq_dir = val["seq_dirs"][SID]

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": val.get("info", {}), "licenses": val.get("licenses", []),
        "categories": val["categories"], "images": imgs, "annotations": anns,
    }
    coco_gt.createIndex()
    return imgs, img_ids, coco_mapping, seq_dir, coco_gt


# ---------- per-device inference adapters ------------------------------------

class UltralyticsRunner:
    """Wraps ultralytics YOLO for cpu/cuda runs (.pt weights)."""

    def __init__(self, device):
        import os
        os.environ["YOLO_VERBOSE"] = "False"
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO("yolo11s.pt")
        # warmup
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=IMG_SIZE, conf=CONF, iou=IOU,
                           verbose=False, device=device)

    def infer(self, img_path, frame_shape):
        r = self.model.predict(str(img_path), imgsz=IMG_SIZE,
                               conf=CONF, iou=IOU, verbose=False,
                               device=self.device)[0]
        # ultralytics returns coords in ORIGINAL frame coords
        if not len(r.boxes):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)
        return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
                r.boxes.conf.cpu().numpy().astype(np.float32),
                r.boxes.cls.cpu().numpy().astype(int))


class NpuRunner:
    """Wraps Mobilint mblt_model_zoo YOLO11s for NPU runs (.mxq)."""

    def __init__(self):
        from mblt_model_zoo.vision import YOLO11s
        self.model = YOLO11s(local_path="models/mobilint/yolo11s.mxq",
                             infer_mode="global8", product="aries")
        # warmup
        dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
        x = self.model.preprocess(dummy)
        out = self.model(x)
        self.model.postprocess(out, conf_thres=CONF, iou_thres=IOU)

    def infer(self, img_path, frame_shape):
        img = cv2.imread(str(img_path))  # BGR
        x = self.model.preprocess(img)
        out = self.model(x)
        res = self.model.postprocess(out, conf_thres=CONF, iou_thres=IOU)
        box_cls = getattr(res, "box_cls", None)
        if box_cls is None or box_cls.shape[0] == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), int)
        arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
        # box_cls is in letterboxed input coords (IMG_SIZE×IMG_SIZE). Un-letterbox.
        h0, w0 = frame_shape[:2]
        gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
        pad_x = (IMG_SIZE - w0 * gain) / 2.0
        pad_y = (IMG_SIZE - h0 * gain) / 2.0
        xyxy = arr[:, :4].astype(np.float32, copy=True)
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy /= gain
        np.clip(xyxy[:, [0, 2]], 0, w0, out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0, h0, out=xyxy[:, [1, 3]])
        return xyxy, arr[:, 4].astype(np.float32), arr[:, 5].astype(int)

    def close(self):
        self.model.dispose()


# ---------- streaming sim + sAP ----------------------------------------------

def run_one_device(label, runner, imgs, coco_mapping, seq_dir, img_ids, coco_gt):
    print(f"\n========== {label} ==========")
    n_frame = len(imgs)
    t_total = n_frame / FPS
    timestamps, input_fidx, results_parsed = [], [], []
    t_elapsed = 0.0
    last_fidx = -1
    frame_shape_cache = None

    infer_ms_list = []  # ACTUAL per-call inference latency, decoupled from stream cap
    t_wall_0 = time.time()
    while t_elapsed < t_total:
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last_fidx:
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS
        if fidx >= n_frame:
            break
        last_fidx = fidx

        img_meta = imgs[fidx]
        img_path = DATA / seq_dir / img_meta["name"]
        if frame_shape_cache is None:
            frame_shape_cache = (img_meta["height"], img_meta["width"])

        t0 = time.time()
        xyxy_coco, scores, coco_cls = runner.infer(img_path, frame_shape_cache)
        rt = time.time() - t0
        infer_ms_list.append(rt * 1000.0)

        if len(xyxy_coco):
            ahd = coco_mapping[coco_cls]
            sel = ahd < N_AHD
            xyxy = xyxy_coco[sel]
            scs = scores[sel]
            lbls = ahd[sel].astype(np.int32)
        else:
            xyxy = np.zeros((0, 4), np.float32)
            scs = np.zeros((0,), np.float32)
            lbls = np.zeros((0,), np.int32)

        t_elapsed += rt
        if t_elapsed >= t_total:
            break
        timestamps.append(t_elapsed)
        input_fidx.append(fidx)
        results_parsed.append((xyxy, scs, lbls))

    wall = time.time() - t_wall_0
    n_proc = len(results_parsed)
    infer_arr = np.asarray(infer_ms_list)
    infer_mean = float(infer_arr.mean()) if len(infer_arr) else 0.0
    infer_p50 = float(np.percentile(infer_arr, 50)) if len(infer_arr) else 0.0
    infer_p95 = float(np.percentile(infer_arr, 95)) if len(infer_arr) else 0.0
    print(f"  processed={n_proc}/{n_frame} ({100*n_proc/n_frame:.1f}%)  "
          f"wall={wall:.1f}s  infer_mean={infer_mean:.1f}ms  p95={infer_p95:.1f}ms")

    # pairing
    ccf = []
    tidx_p1 = 0
    miss = 0
    in_time = 0
    for ii, img in enumerate(imgs):
        t_gt = ii / FPS
        while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t_gt:
            tidx_p1 += 1
        if tidx_p1 == 0:
            miss += 1
            continue
        tidx = tidx_p1 - 1
        bb, sc, lb = results_parsed[tidx]
        if input_fidx[tidx] == ii:
            in_time += 1
        for k in range(len(bb)):
            x1, y1, x2, y2 = bb[k]
            ccf.append({
                "image_id": int(img["id"]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc[k]),
                "category_id": int(lb[k]),
            })
    print(f"  miss={miss}  in_time={in_time}/{n_frame}  ccf={len(ccf)}")

    if not ccf:
        return None
    coco_dt = coco_gt.loadRes(ccf)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = sorted(img_ids)
    e.evaluate(); e.accumulate(); e.summarize()
    return {
        "device": label,
        "processed_pct": 100 * n_proc / n_frame,
        "infer_mean_ms": infer_mean,
        "infer_p50_ms": infer_p50,
        "infer_p95_ms": infer_p95,
        "miss": miss,
        "in_time": in_time,
        "sAP_50_95": e.stats[0],
        "sAP_50": e.stats[1],
        "sAP_75": e.stats[2],
        "AP_s": e.stats[3],
        "AP_m": e.stats[4],
        "AP_l": e.stats[5],
    }


def main():
    imgs, img_ids, coco_mapping, seq_dir, coco_gt = load_split()
    print(f"Log sid={SID}  frames={len(imgs)}  GT={sum(1 for a in coco_gt.anns.values())}\n")

    results = []
    # CPU
    r = run_one_device("CPU (ultralytics yolo11s.pt, 640)",
                       UltralyticsRunner("cpu"), imgs, coco_mapping, seq_dir, img_ids, coco_gt)
    if r: results.append(r)
    # GPU
    r = run_one_device("GPU (ultralytics yolo11s.pt, 640, cu13)",
                       UltralyticsRunner("cuda"), imgs, coco_mapping, seq_dir, img_ids, coco_gt)
    if r: results.append(r)
    # NPU
    npu = NpuRunner()
    try:
        r = run_one_device("NPU (Mobilint Aries yolo11s.mxq, 640, global8)",
                           npu, imgs, coco_mapping, seq_dir, img_ids, coco_gt)
        if r: results.append(r)
    finally:
        npu.close()

    print("\n========== COMPARISON ==========")
    hdr = (f"{'device':45s}  {'infer_mean':>10s}  {'p95':>6s}  {'proc%':>6s}  "
           f"{'miss':>5s}  {'sAP':>6s}  {'sAP50':>6s}  {'AP_l':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['device']:45s}  {r['infer_mean_ms']:8.1f}ms  "
              f"{r['infer_p95_ms']:4.1f}ms  {r['processed_pct']:6.1f}  "
              f"{r['miss']:5d}  "
              f"{r['sAP_50_95']:6.3f}  {r['sAP_50']:6.3f}  {r['AP_l']:6.3f}")


if __name__ == "__main__":
    main()
