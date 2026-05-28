"""Minimal single-stream sAP measurement for one Argoverse-HD val log.

End-to-end: load val.json → filter to sid=5 → run YOLOv11s frame-by-frame
in simulated real-time → pair detections to GT by time → COCOeval → print sAP.

Designed to verify the full pipeline works once, not to be production-grade.
"""

import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent.parent
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
DATA = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-1.1/argoverse-tracking"
WEIGHTS = "yolo11s.pt"
SID = 5            # shortest val log
FPS = 30.0
IMG_SIZE = 1280
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = "cuda"    # torch GPU restored via cuDNN 9.20 → 9.12 downgrade (cu13 stack)
N_AHD = 8


def main():
    t_script = time.time()

    # 1. Load + filter annotations to sid=SID only
    with open(ANNOT) as f:
        val = json.load(f)

    imgs = sorted([i for i in val["images"] if i["sid"] == SID], key=lambda x: x["fid"])
    img_ids = {i["id"] for i in imgs}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    seq = val["sequences"][SID]
    seq_dir = val["seq_dirs"][SID]
    coco_mapping = np.asarray(val["coco_mapping"])  # COCO80 idx → AHD id (80=skip)
    print(f"Selected log: sid={SID} '{seq}'")
    print(f"  frames={len(imgs)}  GT_annotations={len(anns)}")

    # Build a COCO object holding only this log
    filt = {
        "info": val.get("info", {}),
        "licenses": val.get("licenses", []),
        "categories": val["categories"],
        "images": imgs,
        "annotations": anns,
    }
    coco_gt = COCO()
    coco_gt.dataset = filt
    coco_gt.createIndex()

    # 2. Streaming-style detection
    yolo = YOLO(WEIGHTS)
    print(f"Loaded {WEIGHTS} on {DEVICE}")

    timestamps = []
    input_fidx = []
    results_parsed = []  # list of (xyxy ndarray, scores ndarray, ahd_labels ndarray)
    t_elapsed = 0.0
    t_total = len(imgs) / FPS
    last_fidx = -1
    n_frame = len(imgs)

    t_stream0 = time.time()
    while t_elapsed < t_total:
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last_fidx:
            # detector was faster than the stream — idle until next frame arrives
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS  # advance simulated clock to next frame
        if fidx >= n_frame:
            break
        last_fidx = fidx

        img_path = DATA / seq_dir / imgs[fidx]["name"]
        t0 = time.time()
        r = yolo.predict(
            str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
            verbose=False, device=DEVICE,
        )[0]
        rt = time.time() - t0  # measured wall-clock latency

        if len(r.boxes):
            coco_ids = r.boxes.cls.cpu().numpy().astype(int)
            ahd_ids = coco_mapping[coco_ids]
            sel = ahd_ids < N_AHD
            bboxes = r.boxes.xyxy.cpu().numpy()[sel].astype(np.float32)
            scores = r.boxes.conf.cpu().numpy()[sel].astype(np.float32)
            labels = ahd_ids[sel].astype(np.int32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int32)

        t_elapsed += rt
        if t_elapsed >= t_total:
            break

        timestamps.append(t_elapsed)
        input_fidx.append(fidx)
        results_parsed.append((bboxes, scores, labels))

        if len(results_parsed) % 25 == 0:
            print(f"  processed={len(results_parsed):3d}  fidx={fidx:3d}  "
                  f"dets={len(bboxes):2d}  rt={rt*1000:6.1f}ms  "
                  f"t={t_elapsed:5.2f}s/{t_total:.2f}s")

    t_stream = time.time() - t_stream0
    n_processed = len(results_parsed)
    print(f"\nStreaming done: processed={n_processed}/{n_frame} "
          f"({100*n_processed/n_frame:.1f}%)  wall={t_stream:.1f}s "
          f"(simulated_window={t_total:.1f}s)")

    # 3. Streaming-eval pairing — for each GT frame ii, latest detection with timestamp <= ii/FPS
    results_ccf = []
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
        bboxes, scores, labels = results_parsed[tidx]
        if input_fidx[tidx] == ii:
            in_time += 1
        for k in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[k]
            results_ccf.append({
                "image_id": int(img["id"]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[k]),
                "category_id": int(labels[k]),
            })
    print(f"Pairing: miss={miss}  in_time={in_time}/{n_frame}  ccf_records={len(results_ccf)}")

    # 4. COCO-style evaluation → sAP
    if not results_ccf:
        print("\nNo detections to evaluate — sAP undefined.")
        return

    coco_dt = coco_gt.loadRes(results_ccf)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = sorted(img_ids)
    e.evaluate()
    e.accumulate()
    e.summarize()

    print(f"\n=== Streaming AP (single log sid={SID}) ===")
    print(f"  sAP @ IoU=0.50:0.95 (primary)  = {e.stats[0]:.4f}")
    print(f"  sAP @ IoU=0.50                 = {e.stats[1]:.4f}")
    print(f"  sAP @ IoU=0.75                 = {e.stats[2]:.4f}")
    print(f"\nTotal script time: {time.time()-t_script:.1f}s")


if __name__ == "__main__":
    main()
