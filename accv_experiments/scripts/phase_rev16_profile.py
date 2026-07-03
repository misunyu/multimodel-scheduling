"""rev16 — profile the normal-state NPU single-stream L0 latency breakdown.

Goal: decompose the ~35.5ms NPU per-frame time into:
  1. cv2.imread        (disk read + JPEG decode of full-res frame)
  2. preprocess        (resize / normalize / input-quantize)
  3. model() on-chip   (H2D + MLA100 compute + D2H — opaque single call)
  4. postprocess       (dequantize / decode / NMS)

This is a TIMING HARNESS — it does NOT modify any core eval script. It reuses
the same engine + same npu_infer sub-calls so the sum reproduces the measured
~35.5ms. Runs >=3 reps; reports mean±std per stage.

Comparison: GPU per-frame (ultralytics predict) total, to contextualize.

Output: results/rev16_latency_breakdown.csv + console summary.
Pinned state: mxq b2441f9d, global8, yolo11s. No bg load (L0).
"""

from __future__ import annotations

import csv, sys, time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))

import cv2
from _step_d_common import load_val, load_split_for_sid
from step0_compare_devices import DATA, CONF, IOU, IMG_SIZE, FPS
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                dispose_npu_for, DETECTORS)

RES = Path("accv_experiments/results")
OUT = RES / "rev16_latency_breakdown.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD_MS = 1000.0 / FPS

# Profiling params
TEST_SID = 2
N_FRAMES = 120         # frames per rep (post warmup)
WARMUP = 30
N_REPS = 3


def profile_npu_once(npu_model, img_paths):
    """Time each sub-stage of npu_infer over img_paths. Returns dict of lists."""
    m = npu_model
    timings = {"imread": [], "preprocess": [], "model": [], "postprocess": [],
                "total": []}
    for p in img_paths:
        t_all = time.time()
        t0 = time.time(); img = cv2.imread(str(p)); t1 = time.time()
        x = m.preprocess(img); t2 = time.time()
        out = m(x); t3 = time.time()
        m.postprocess(out, conf_thres=CONF, iou_thres=IOU); t4 = time.time()
        timings["imread"].append((t1-t0)*1000)
        timings["preprocess"].append((t2-t1)*1000)
        timings["model"].append((t3-t2)*1000)
        timings["postprocess"].append((t4-t3)*1000)
        timings["total"].append((t4-t_all)*1000)
    return timings


def profile_gpu_once(gpu_model, img_paths):
    """Time GPU per-frame (ultralytics predict, includes its own imread)."""
    tot = []
    for p in img_paths:
        t0 = time.time()
        gpu_model.predict(str(p))
        tot.append((time.time()-t0)*1000)
    return tot


def main():
    val = load_val()
    split = load_split_for_sid(val, TEST_SID)
    imgs = split["imgs"]
    seq_dir = split["seq_dir"]
    paths = [DATA / seq_dir / im["name"] for im in imgs]
    # post-warmup window
    work = paths[WARMUP:WARMUP + N_FRAMES]
    print(f"period_ms (1000/FPS) = {PERIOD_MS:.2f}ms")
    print(f"profiling sid={TEST_SID}, {len(work)} frames/rep, {N_REPS} reps")

    # Load NPU
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)[0]
    # warmup
    for p in paths[:WARMUP]:
        img = cv2.imread(str(p)); x = npu.preprocess(img); o = npu(x)
        npu.postprocess(o, conf_thres=CONF, iou_thres=IOU)

    rows = []
    agg = {k: [] for k in ["imread","preprocess","model","postprocess","total"]}
    for rep in range(N_REPS):
        t = profile_npu_once(npu, work)
        for k in agg:
            agg[k].append(float(np.mean(t[k])))
        print(f"  rep{rep}: imread={np.mean(t['imread']):.2f}  "
                f"pre={np.mean(t['preprocess']):.2f}  model={np.mean(t['model']):.2f}  "
                f"post={np.mean(t['postprocess']):.2f}  total={np.mean(t['total']):.2f}ms")
    dispose_npu_for("yolo11s")

    # GPU comparison
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    for p in paths[:WARMUP]: gpu.predict(str(p))
    gpu_tot = []
    for rep in range(N_REPS):
        g = profile_gpu_once(gpu, work)
        gpu_tot.append(float(np.mean(g)))
    print(f"  GPU total per-frame: {np.mean(gpu_tot):.2f}±{np.std(gpu_tot):.2f}ms")

    # write CSV
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage","mean_ms","std_ms","pct_of_total","n_reps"])
        npu_total = np.mean(agg["total"])
        for k in ["imread","preprocess","model","postprocess"]:
            mean = np.mean(agg[k]); std = np.std(agg[k])
            w.writerow([f"NPU_{k}", round(mean,3), round(std,3),
                         round(100*mean/npu_total,1), N_REPS])
        w.writerow(["NPU_total", round(npu_total,3), round(np.std(agg['total']),3), 100.0, N_REPS])
        w.writerow(["GPU_total", round(np.mean(gpu_tot),3), round(np.std(gpu_tot),3), "", N_REPS])
        w.writerow(["period_ms", round(PERIOD_MS,3), 0, "", ""])
    print(f"saved {OUT}")

    # Summary
    npu_total = np.mean(agg["total"])
    onchip = np.mean(agg["model"])
    host = npu_total - onchip
    print()
    print("=== breakdown ===")
    for k in ["imread","preprocess","model","postprocess"]:
        mean = np.mean(agg[k])
        print(f"  {k:12s}: {mean:6.2f}ms ({100*mean/npu_total:4.1f}%)")
    print(f"  {'TOTAL':12s}: {npu_total:6.2f}ms")
    print(f"  on-chip (model): {onchip:.2f}ms;  host (imread+pre+post): {host:.2f}ms")
    print(f"  period: {PERIOD_MS:.2f}ms")
    verdict = "DEVICE-bound" if onchip > PERIOD_MS else (
              "HOST-bound (on-chip < period — overhead is the skip cause)"
              if onchip < PERIOD_MS else "borderline")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
