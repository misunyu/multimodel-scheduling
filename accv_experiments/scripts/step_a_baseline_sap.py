"""Step A — 24-log baseline sAP across CPU / GPU / NPU.

Outer loop over every Argoverse-HD val log; inner triple over CPU/GPU/NPU.
Per (log, device) measurement reuses the streaming-sim + COCOeval pipeline
defined in minimal_pipeline/step0_compare_devices.py (load_split_for_sid
is the only thing we add since the original load_split is hardcoded to sid=5).

The CSV is appended incrementally — if interrupted, re-running skips
already-completed (log_id, device) pairs.

Prerequisites:
  - accv_experiments/data/argoverse_hd/Argoverse-1.1/argoverse-tracking/val/
    must contain `<log_hash>/ring_front_center/` for every log we want measured.
  - logs without local images are skipped (logged to console).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

# Pre-init CUDA before ultralytics' CPU path can leave device_count stale.
if torch.cuda.is_available():
    torch.cuda.init()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # multimodel-scheduling-video/
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

# Reuse the inference adapters + streaming sim. We import after sys.path setup.
from step0_compare_devices import (  # noqa: E402
    ANNOT, DATA, NpuRunner, UltralyticsRunner, run_one_device,
)

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_a_baseline.csv"
COLUMNS = [
    "log_id", "device",
    "infer_mean_ms", "infer_p95_ms", "proc_percent",
    "sap_5095", "sap_50", "sap_small", "sap_medium", "sap_large",
    "miss_count", "frame_count",
]


# ---------- annotation slicing -----------------------------------------------

def load_val_json():
    with open(ANNOT) as f:
        return json.load(f)


def load_split_for_sid(val: dict, sid: int):
    """Build the per-log split + COCO-GT object expected by run_one_device."""
    imgs = sorted([i for i in val["images"] if i["sid"] == sid], key=lambda x: x["fid"])
    img_ids = {i["id"] for i in imgs}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    coco_mapping = np.asarray(val["coco_mapping"])
    seq_dir = val["seq_dirs"][sid]
    log_name = val["sequences"][sid]
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": val.get("info", {}),
        "licenses": val.get("licenses", []),
        "categories": val["categories"],
        "images": imgs,
        "annotations": anns,
    }
    coco_gt.createIndex()
    return imgs, img_ids, coco_mapping, seq_dir, coco_gt, log_name


def discover_logs(val: dict):
    """Return list of (sid, log_name, n_frames) for every val log."""
    ctr = Counter(img["sid"] for img in val["images"])
    out = []
    for sid in sorted(ctr):
        out.append((sid, val["sequences"][sid], ctr[sid]))
    return out


def images_present(seq_dir: str, n_expected: int) -> tuple[bool, int]:
    """(have_enough, n_found)."""
    p = DATA / seq_dir
    if not p.exists():
        return False, 0
    n = sum(1 for _ in p.glob("*.jpg"))
    return n >= n_expected, n


# ---------- CSV append + resume ----------------------------------------------

def load_done() -> set[tuple[str, str]]:
    if not OUT_CSV.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with open(OUT_CSV) as f:
        for row in csv.DictReader(f):
            done.add((row["log_id"], row["device"]))
    return done


def append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------- one (log, device) measurement ------------------------------------

def measure(label: str, dev_short: str, sid: int, log_name: str, n_frames: int,
            runner, val: dict) -> dict | None:
    imgs, img_ids, coco_mapping, seq_dir, coco_gt, _ = load_split_for_sid(val, sid)
    r = run_one_device(label, runner, imgs, coco_mapping, seq_dir, img_ids, coco_gt)
    if r is None:
        return None
    return {
        "log_id": log_name,
        "device": dev_short,
        "infer_mean_ms": round(r["infer_mean_ms"], 2),
        "infer_p95_ms": round(r["infer_p95_ms"], 2),
        "proc_percent": round(r["processed_pct"], 2),
        "sap_5095": round(r["sAP_50_95"], 4),
        "sap_50": round(r["sAP_50"], 4),
        "sap_small": round(r["AP_s"], 4),
        "sap_medium": round(r["AP_m"], 4),
        "sap_large": round(r["AP_l"], 4),
        "miss_count": int(r["miss"]),
        "frame_count": int(n_frames),
    }


# ---------- main loop --------------------------------------------------------

def main():
    val = load_val_json()
    logs = discover_logs(val)
    done = load_done()
    print(f"[step_a] {len(logs)} val logs, {sum(n for _, _, n in logs)} total frames")
    print(f"[step_a] CSV: {OUT_CSV}  ({len(done)} pairs already done)")

    skipped = []
    runnable = []
    for sid, name, n in logs:
        ok, found = images_present(val["seq_dirs"][sid], n)
        if ok:
            runnable.append((sid, name, n))
        else:
            skipped.append((sid, name, n, found))
    if skipped:
        print(f"[step_a] {len(skipped)} log(s) missing local images — will be skipped:")
        for sid, name, n, found in skipped:
            print(f"   sid={sid:2d} {name[:18]}…  expected={n}  found={found}")

    print(f"[step_a] runnable logs: {len(runnable)} / {len(logs)}")
    if not runnable:
        print("[step_a] nothing to measure. Extract images first then re-run.")
        return

    # Load runners once and reuse across all logs (model load is non-trivial).
    print("\n[step_a] loading runners…")
    t_load0 = time.time()
    cpu_runner = UltralyticsRunner("cpu")
    gpu_runner = UltralyticsRunner("cuda")
    npu_runner = NpuRunner()
    print(f"[step_a] runners ready in {time.time()-t_load0:.1f}s")

    t_overall0 = time.time()
    try:
        for i, (sid, log_name, n_frames) in enumerate(runnable, 1):
            print(f"\n=== [{i:2d}/{len(runnable)}] sid={sid:2d}  log={log_name[:18]}…  frames={n_frames} ===")
            for dev_short, runner, label in [
                ("CPU", cpu_runner, "CPU (yolo11s.pt, 640)"),
                ("GPU", gpu_runner, "GPU (yolo11s.pt, 640, cu13)"),
                ("NPU", npu_runner, "NPU (yolo11s.mxq, 640, global8)"),
            ]:
                if (log_name, dev_short) in done:
                    print(f"  [{dev_short}] already in CSV — skip")
                    continue
                t0 = time.time()
                row = measure(label, dev_short, sid, log_name, n_frames, runner, val)
                if row is None:
                    print(f"  [{dev_short}] no detections produced — skipping row")
                    continue
                append_row(row)
                done.add((log_name, dev_short))
                print(f"  [{dev_short}] sAP={row['sap_5095']:.3f}  "
                      f"infer={row['infer_mean_ms']:.1f}ms  "
                      f"proc={row['proc_percent']:.1f}%  "
                      f"(took {time.time()-t0:.1f}s)")
    finally:
        npu_runner.close()

    elapsed = time.time() - t_overall0
    print(f"\n[step_a] all done. wall={elapsed:.1f}s  csv={OUT_CSV}")


if __name__ == "__main__":
    main()
