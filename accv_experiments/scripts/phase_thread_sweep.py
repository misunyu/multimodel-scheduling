"""NPU post-processing thread-count sweep (single-stream, uncontended).

Goal: measure how the NPU *host-side post-processing* thread count
(torch.set_num_threads) affects (a) end-to-end single-stream latency and
(b) frame skip %, by sweeping thread count over {1,2,3,4,6,8,12,16,24}.
Decides whether end-to-end latency is monotonically increasing in thread
count (min at 1) or U-shaped (interior min). Ties back to Table tab:decomp,
which anchors 4 threads (staleness-free) vs 24 threads (staleness-on).

This script REUSES the existing single-stream harness components without
editing any core measurement script:
  - load_val / load_split_for_sid          (_step_d_common.py)
  - per_stream_sap                          (_step_d_common.py)  -> sAP + frame-skip
  - load_npu_engines / set_active_npu_engines / dispose_npu_for (phase_rev6_sweep.py)
  - DETECTORS[yolo11s] baseline mxq/mode    (phase_rev6_sweep.py)  -> global8, same as rev18

The only added instrumentation is a local streaming worker that mirrors the
core fg_worker timing EXACTLY (it advances the streaming clock by the full
preprocess+infer+postprocess wall time, identical to fg_worker) but
additionally splits the on-chip inference time (m(x)) from the host-side
post-processing time (m.postprocess). Detector outputs are unchanged: it
calls the same m.preprocess / m(x) / m.postprocess and the same letterbox
un-scaling math as npu_infer. The default thread count consumed by other
experiments is NOT touched (we only call torch.set_num_threads inside this
sweep's own process).

Fixed conditions (match the paper's controlled probe):
  - YOLOv11s, vendor INT8 global8 mxq (models/mobilint_backup/yolo11s.mxq)
  - Argoverse-HD val, all 24 logs, 640x640 letterbox, conf/iou = step0 defaults
  - Single foreground stream (N=1), NO co-tenant (bg=L0)
  - 30-frame warmup (WARMUP_FRAMES), full-log replay, 3 independent repeats

Outputs (results/thread_sweep/):
  thread_sweep.csv          one row per (threads, rep)
  thread_sweep_summary.csv  aggregated mean/std per threads
  bitident.csv              raw-detection bit-identity check vs threads=4
  raw stdout log via tee
"""

from __future__ import annotations

import csv
import sys
import time
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))

import _step_d_common as cm
from _step_d_common import load_val, load_split_for_sid, per_stream_sap, WARMUP_FRAMES
from phase_rev6_sweep import (load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, DETECTORS)
from step0_compare_devices import FPS, IMG_SIZE, CONF, IOU, N_AHD, DATA

import cv2

RES = Path("accv_experiments/results/thread_sweep")
RES.mkdir(parents=True, exist_ok=True)

DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD_MS = 1000.0 / FPS
N_REPS = 3
THREAD_SETTINGS = [1, 2, 3, 4, 6, 8, 12, 16, 24]


# ---------------------------------------------------------------------------
# Instrumented single-stream worker — mirrors _step_d_common.fg_worker timing
# exactly (clock advances by full pre+infer+post wall time) but splits the
# on-chip inference time from the host post-processing time, and optionally
# captures the raw detector output for the bit-identity check.
# ---------------------------------------------------------------------------

def fg_worker_split(split, npu_model, result, capture_raw=False):
    imgs = split["imgs"]
    coco_mapping = split["coco_mapping"]
    seq_dir = split["seq_dir"]
    n_frame = len(imgs)
    t_total = n_frame / FPS
    t_elapsed = 0.0
    last_fidx = -1
    frame_shape = (imgs[0]["height"], imgs[0]["width"])
    raw_frames = []  # list of (fidx, raw box_cls float array) when capture_raw

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
        img_path = DATA / seq_dir / imgs[fidx]["name"]

        img = cv2.imread(str(img_path))
        t0 = time.time()
        x = npu_model.preprocess(img)
        t1 = time.time()
        out = npu_model(x)                       # on-chip inference
        t2 = time.time()
        res = npu_model.postprocess(out, conf_thres=CONF, iou_thres=IOU)  # host post
        t3 = time.time()

        box_cls = getattr(res, "box_cls", None)
        if box_cls is None or (hasattr(box_cls, "shape") and box_cls.shape[0] == 0):
            arr = np.zeros((0, 6), np.float32)
        else:
            arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
        if capture_raw:
            raw_frames.append((fidx, arr.astype(np.float32, copy=True)))

        # Letterbox un-scale -> original image coords (identical to npu_infer)
        h0, w0 = frame_shape[:2]
        gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
        pad_x = (IMG_SIZE - w0 * gain) / 2.0
        pad_y = (IMG_SIZE - h0 * gain) / 2.0
        if arr.shape[0]:
            xyxy = arr[:, :4].astype(np.float32, copy=True)
            xyxy[:, [0, 2]] -= pad_x
            xyxy[:, [1, 3]] -= pad_y
            xyxy /= gain
            np.clip(xyxy[:, [0, 2]], 0, w0, out=xyxy[:, [0, 2]])
            np.clip(xyxy[:, [1, 3]], 0, h0, out=xyxy[:, [1, 3]])
            scores = arr[:, 4].astype(np.float32)
            coco_cls = arr[:, 5].astype(int)
            ahd = coco_mapping[coco_cls]
            sel = ahd < N_AHD
            bb = xyxy[sel]
            sc = scores[sel]
            lb = ahd[sel].astype(np.int32)
        else:
            bb = np.zeros((0, 4), np.float32)
            sc = np.zeros(0, np.float32)
            lb = np.zeros(0, np.int32)

        rt = t3 - t0                # full end-to-end (pre+infer+post) — drives clock
        t_elapsed += rt
        result["infer_ms"].append(rt * 1000)      # combined (kept for per_stream_sap)
        result["eff_ms"].append(rt * 1000)        # e2e — frame-skip uses this
        result["pre_ms"].append((t1 - t0) * 1000)
        result["onchip_ms"].append((t2 - t1) * 1000)  # on-chip inference
        result["post_ms"].append((t3 - t2) * 1000)    # host post-processing
        result["timestamps"].append(t_elapsed)
        result["input_fidx"].append(fidx)
        result["results"].append((bb, sc, lb))

    return raw_frames


def measure_one_log(split, npu_model, capture_raw=False):
    res = defaultdict(list)
    raw_frames = fg_worker_split(split, npu_model, res, capture_raw=capture_raw)
    sap = per_stream_sap(split, res)  # reuse core sAP + frame-skip computation
    eff_post = res["eff_ms"][WARMUP_FRAMES:]
    onchip_post = res["onchip_ms"][WARMUP_FRAMES:]
    post_post = res["post_ms"][WARMUP_FRAMES:]
    return {
        "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
        "frame_skip_pct": sap["frame_skip_pct"],
        "eff_ms": eff_post,        # raw per-frame e2e (post-warmup) for pooling
        "onchip_ms": onchip_post,
        "post_ms": post_post,
    }, raw_frames


def raw_signature(raw_frames):
    """Stable hash + flat array of the raw detector outputs over a fixed log."""
    h = hashlib.sha256()
    flat = []
    for fidx, arr in raw_frames:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        flat.append(arr.reshape(-1))
    cat = np.concatenate(flat) if flat else np.zeros(0, np.float32)
    return h.hexdigest(), cat


def main():
    val = load_val()
    n_sids = len(val["sequences"])
    print(f"period={PERIOD_MS:.2f}ms  logs={n_sids}  reps={N_REPS}  "
          f"threads={THREAD_SETTINGS}  mxq={DET['baseline_mxq']} ({DET['baseline_mode']})")

    # Load the single NPU engine once (thread count is a process-global torch
    # setting applied at call time; it does not depend on engine load).
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)[0]
    set_active_npu_engines([npu])

    rows = []
    bitident_ref = None      # (hash, flat) at threads=4 for sid=0
    bitident_rows = []

    out_csv = RES / "thread_sweep.csv"
    fields = ["threads", "rep",
              "eff_mean_ms", "eff_p50_ms", "eff_p99_ms",
              "onchip_mean_ms", "post_mean_ms",
              "frame_skip_pct", "sap_s", "sap_m", "sap_l", "n_frames"]

    for nt in THREAD_SETTINGS:
        print(f"\n=== torch threads = {nt} ===")
        torch.set_num_threads(nt)
        # warmup at this thread setting
        s0 = load_split_for_sid(val, 0)
        measure_one_log(s0, npu)

        for rep in range(N_REPS):
            torch.set_num_threads(nt)  # ensure
            all_eff, all_onchip, all_post = [], [], []
            skips, ss, sm, sl = [], [], [], []
            cap_raw = (rep == 0)
            for sid in range(n_sids):
                split = load_split_for_sid(val, sid)
                capture = cap_raw and (sid == 0)
                m, raw_frames = measure_one_log(split, npu, capture_raw=capture)
                all_eff.extend(m["eff_ms"])
                all_onchip.extend(m["onchip_ms"])
                all_post.extend(m["post_ms"])
                skips.append(m["frame_skip_pct"])
                ss.append(m["sap_s"]); sm.append(m["sap_m"]); sl.append(m["sap_l"])
                if capture:
                    sig = raw_signature(raw_frames)
                    if nt == 4:
                        bitident_ref = sig
                    bitident_rows.append((nt, sig))

            eff = np.asarray(all_eff)
            row = {
                "threads": nt, "rep": rep,
                "eff_mean_ms": round(float(eff.mean()), 4),
                "eff_p50_ms": round(float(np.percentile(eff, 50)), 4),
                "eff_p99_ms": round(float(np.percentile(eff, 99)), 4),
                "onchip_mean_ms": round(float(np.mean(all_onchip)), 4),
                "post_mean_ms": round(float(np.mean(all_post)), 4),
                "frame_skip_pct": round(float(np.mean(skips)), 4),
                "sap_s": round(float(np.mean(ss)), 4),
                "sap_m": round(float(np.mean(sm)), 4),
                "sap_l": round(float(np.mean(sl)), 4),
                "n_frames": int(eff.size),
            }
            rows.append(row)
            print(f"  rep{rep}: e2e mean={row['eff_mean_ms']:.2f} "
                  f"p50={row['eff_p50_ms']:.2f} p99={row['eff_p99_ms']:.2f}ms | "
                  f"onchip={row['onchip_mean_ms']:.2f} post={row['post_mean_ms']:.2f}ms | "
                  f"skip={row['frame_skip_pct']:.1f}% | "
                  f"sAP s/m/l={row['sap_s']:.3f}/{row['sap_m']:.3f}/{row['sap_l']:.3f}")

            # incremental save
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    dispose_npu_for("yolo11s")

    # ---- summary (mean/std per thread count) ----
    summ_csv = RES / "thread_sweep_summary.csv"
    metrics = ["eff_mean_ms", "eff_p50_ms", "eff_p99_ms", "onchip_mean_ms",
               "post_mean_ms", "frame_skip_pct", "sap_s", "sap_m", "sap_l"]
    with open(summ_csv, "w", newline="") as f:
        cols = ["threads"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for nt in THREAD_SETTINGS:
            sub = [r for r in rows if r["threads"] == nt]
            rec = {"threads": nt}
            for m in metrics:
                vals = np.asarray([r[m] for r in sub], float)
                rec[f"{m}_mean"] = round(float(vals.mean()), 4)
                rec[f"{m}_std"] = round(float(vals.std()), 4)
            w.writerow(rec)
    print(f"\nsaved {out_csv}\nsaved {summ_csv}")

    # ---- bit-identity check (sid=0, rep0) vs threads=4 reference ----
    bi_csv = RES / "bitident.csv"
    ref_hash, ref_flat = bitident_ref
    with open(bi_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "raw_sha256", "exact_equal_to_t4", "max_abs_diff_to_t4", "n_detections"])
        all_equal = True
        for nt, (hh, flat) in bitident_rows:
            if flat.shape == ref_flat.shape:
                maxdiff = float(np.max(np.abs(flat - ref_flat))) if flat.size else 0.0
            else:
                maxdiff = float("nan")
            exact = (hh == ref_hash)
            all_equal = all_equal and exact
            w.writerow([nt, hh[:16], exact, maxdiff, int(flat.size // 6)])
            print(f"  threads={nt:2d}: sha256={hh[:16]} exact_equal_to_t4={exact} "
                  f"max_abs_diff={maxdiff}")
    print(f"saved {bi_csv}")
    print(f"\nBIT-IDENTITY ACROSS ALL THREAD COUNTS (sid=0): "
          f"{'PASS (identical)' if all_equal else 'FAIL (NOT identical)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
