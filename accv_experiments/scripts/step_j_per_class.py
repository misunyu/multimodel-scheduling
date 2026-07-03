"""Step J — Per-class AP breakdown for YOLOv11s (single-camera, L0).

Fills TODO-FILL tab:per-class in main_vision.tex.

For each device in {GPU FP32, NPU INT8}:
  - Run all 24 Argoverse-HD val logs as single foreground streams (no bg).
  - Collect every post-warmup detection into a single COCO results array.
  - Build one global COCO GT (the union of the 24 per-log GTs already used
    by per-camera evaluation) and run pycocotools COCOeval with catIds set
    to each class in turn to get per-class AP at IoU=[.50:.95] and IoU=0.50.

Outputs:
  results/per_class.csv      (per device, per class: ap_5095, ap_50, n_gt)
  paper/tables/per_class.tex (tab:per-class scaffolded with measured values)
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, fg_worker, get_npu_model,
                            load_split_for_sid, load_val, WARMUP_FRAMES,
                            preload_npu_instances)
from step0_compare_devices import FPS, N_AHD  # noqa: F401

RES = SCRIPT_DIR.parent / "results"
OUT_CSV = RES / "per_class.csv"
OUT_TEX = SCRIPT_DIR.parent.parent / "paper" / "tables" / "per_class.tex"

# Size-class labels (from main_vision.tex tab:per-class)
SIZE_CLASS = {
    "car":            "medium--large",
    "truck":          "large",
    "bus":            "large",
    "pedestrian":     "small",
    "bicycle":        "small",
    "motorcycle":     "small--medium",
    "traffic\\_light": "small",
    "stop\\_sign":     "small",
}
# Map (val.json name -> display name in paper)
DISPLAY = {
    "person":        "pedestrian",
    "bicycle":       "bicycle",
    "car":           "car",
    "motorcycle":    "motorcycle",
    "bus":           "bus",
    "truck":         "truck",
    "traffic_light": "traffic\\_light",
    "stop_sign":     "stop\\_sign",
}
PAPER_ORDER = ["car", "truck", "bus", "pedestrian", "bicycle",
               "motorcycle", "traffic\\_light", "stop\\_sign"]


def run_log_single_stream(sid, split, device, model):
    """Run one foreground stream at L0, return per-frame detections."""
    result = defaultdict(list)
    stop = threading.Event()
    fg_worker(0, device, split, model, result, stop)
    return result


def per_log_detections_in_coco(split, result):
    """Convert post-warmup detections of a stream into COCO results format.
    Uses the same warmup-skip rule as per_stream_sap in _step_d_common."""
    imgs = split["imgs"]
    timestamps = result["timestamps"]
    input_fidx = result["input_fidx"]
    results_parsed = result["results"]

    ccf = []
    tidx_p1 = 0
    for ii, img in enumerate(imgs):
        t_gt = ii / FPS
        while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t_gt:
            tidx_p1 += 1
        if ii < WARMUP_FRAMES:
            continue
        if tidx_p1 == 0:
            continue
        tidx = tidx_p1 - 1
        bb, sc, lb = results_parsed[tidx]
        for k in range(len(bb)):
            x1, y1, x2, y2 = bb[k]
            ccf.append({"image_id": int(img["id"]),
                        "bbox": [float(x1), float(y1),
                                 float(x2 - x1), float(y2 - y1)],
                        "score": float(sc[k]),
                        "category_id": int(lb[k])})
    return ccf


def build_global_gt(val, sids):
    """Build a COCO GT object containing the post-warmup images and
    annotations of all sids combined."""
    imgs_keep = []
    for sid in sids:
        sub = sorted([i for i in val["images"] if i["sid"] == sid],
                     key=lambda x: x["fid"])
        imgs_keep.extend(sub[WARMUP_FRAMES:])
    img_ids = {i["id"] for i in imgs_keep}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": val.get("info", {}),
        "licenses": val.get("licenses", []),
        "categories": val["categories"],
        "images": imgs_keep,
        "annotations": anns,
    }
    coco_gt.createIndex()
    return coco_gt, sorted(img_ids)


def per_class_ap(coco_gt, ccf, img_ids, cat_id):
    """Per-class AP at IoU=[.50:.95] and IoU=0.50, plus GT count."""
    n_gt = sum(1 for a in coco_gt.dataset["annotations"]
               if a["category_id"] == cat_id)
    sub_dets = [d for d in ccf if d["category_id"] == cat_id]
    if not sub_dets:
        return {"ap_5095": 0.0, "ap_50": 0.0, "n_gt": n_gt, "n_dets": 0}
    coco_dt = coco_gt.loadRes(sub_dets)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = img_ids
    e.params.catIds = [cat_id]
    e.evaluate(); e.accumulate(); e.summarize()
    return {"ap_5095": float(e.stats[0]),
            "ap_50":   float(e.stats[1]),
            "n_gt":    n_gt,
            "n_dets":  len(sub_dets)}


def main():
    val = load_val()
    n_logs = len(val["sequences"])
    sids = list(range(n_logs))
    print(f"[stepJ] {n_logs} Argoverse-HD val logs detected")

    cats = sorted(val["categories"], key=lambda c: c["id"])
    cat_names = {c["id"]: c["name"] for c in cats}
    print(f"[stepJ] {len(cats)} categories: {[c['name'] for c in cats]}")

    print("[stepJ] preloading GPU + NPU YOLO11s (single-mode)…")
    t0 = time.time()
    gpu = FGModelGPU()
    preload_npu_instances(1, infer_mode="single")
    npu = get_npu_model(0)
    print(f"[stepJ] preload {time.time()-t0:.1f}s")

    # Pre-load all splits so per-log eval is fast
    splits = {sid: load_split_for_sid(val, sid) for sid in sids}
    coco_gt_global, img_ids_global = build_global_gt(val, sids)
    print(f"[stepJ] global eval images: {len(img_ids_global)}")

    rows = []
    for device, model in [("GPU", gpu), ("NPU", npu)]:
        print(f"\n========== device={device} ==========")
        all_ccf = []
        t_dev = time.time()
        for sid in sids:
            t_log = time.time()
            res = run_log_single_stream(sid, splits[sid], device, model)
            ccf = per_log_detections_in_coco(splits[sid], res)
            all_ccf.extend(ccf)
            print(f"  sid={sid:>2d}  dets={len(ccf):>5d}  ({time.time()-t_log:.1f}s)")
        print(f"  device wall = {time.time()-t_dev:.1f}s  total dets = {len(all_ccf)}")

        print(f"[stepJ] per-class eval on device={device}…")
        for cat in cats:
            stats = per_class_ap(coco_gt_global, all_ccf,
                                 img_ids_global, cat["id"])
            stats.update({"device": device, "cat_id": cat["id"],
                          "cat_name": cat["name"]})
            rows.append(stats)
            print(f"   cat={cat['name']:<14s}  AP[.5:.95]={stats['ap_5095']:.4f}  "
                  f"AP50={stats['ap_50']:.4f}  n_gt={stats['n_gt']}  "
                  f"n_dets={stats['n_dets']}")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "cat_id", "cat_name",
                                          "ap_5095", "ap_50", "n_gt", "n_dets"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved {OUT_CSV}")

    # Build LaTeX table matching tab:per-class
    df = pd.DataFrame(rows)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("% Auto-generated by accv_experiments/scripts/step_j_per_class.py\n")
        f.write("% Per-class AP at IoU=[.50:.95] over 24 Argoverse-HD val logs,\n")
        f.write("% single-camera L0 (no background), warmup-skipped.\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n")
        f.write("Class & Size class & GPU AP & NPU AP & Gap \\\\\n\\midrule\n")
        for paper_name in PAPER_ORDER:
            val_name = {v: k for k, v in DISPLAY.items()}[paper_name]
            gpu_ap = float(df[(df.device == "GPU") &
                              (df.cat_name == val_name)]["ap_5095"].iloc[0])
            npu_ap = float(df[(df.device == "NPU") &
                              (df.cat_name == val_name)]["ap_5095"].iloc[0])
            gap = npu_ap - gpu_ap
            sz = SIZE_CLASS[paper_name]
            f.write(f"{paper_name} & {sz} & {gpu_ap:.3f} & {npu_ap:.3f} & "
                    f"${gap:+.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {OUT_TEX}")


if __name__ == "__main__":
    main()
