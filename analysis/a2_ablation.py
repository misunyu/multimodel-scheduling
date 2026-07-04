"""A2 Step-final A — imread/preprocess ablation (causal stage attribution).

Instead of inline timing (which perturbs and collapses the L2LM effect), we
CAUSALLY remove the read_pre stage: every frame is decoded + letterboxed ONCE
into a uint8 tensor held in RAM (~2.3 GB for PANEL4), so the runtime NPU path is
just model(x_pre) -> postprocess (no cv2.imread, no preprocess). No inline
timers. If removing read_pre collapses the L2LM NPU deadline-miss rate from ~68%
to single digits, the stall is causally attributed to the shared host
imread/preprocess stage. The none cell is the control (should stay DM~0); synth
tests whether the same shared-preprocess bottleneck generalizes.

All-NPU N=4, PANEL4, threads=4, configs {none, synth_c24, L2LM} x 3 reps.
Compares to the stock (non-ablated) reference: L2LM ~68% (a2_repro_check).
Output: analysis/a2_ablation.csv, console verdict.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "accv_experiments/scripts"
sys.path.insert(0, str(SD))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

import cv2
import torch
from _step_d_common import (load_val, load_split_for_sid, DATA, N_AHD,
                            per_stream_sap, stop_background, preload_background_models)
from phase_rev6_sweep import load_npu_engines, set_active_npu_engines, dispose_npu_for, DETECTORS
import step_h2_robustness as h2
from step0_compare_devices import FPS, IMG_SIZE, CONF, IOU

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD = 1000.0 / FPS
REPS = 3
STRESS = ROOT / "analysis/a2_cpu_stress_worker.py"
PY = ROOT / ".venv/bin/python"
OUT = ROOT / "analysis/a2_ablation.csv"


def preload_frames(splits, models):
    """Decode+letterbox every frame once -> list per stream of uint8 tensors."""
    x_all = []
    tot = 0
    for i, sp in enumerate(splits):
        m = models[i]
        xs = []
        for im in sp["imgs"]:
            img = cv2.imread(str(DATA / sp["seq_dir"] / im["name"]))
            xs.append(m.preprocess(img))
        x_all.append(xs)
        tot += len(xs)
    print(f"  preloaded {tot} frames ({tot*640*640*3/1e9:.1f} GB)", flush=True)
    return x_all


def ablated_worker(split, model, x_list, result, stop):
    """fg_worker NPU branch with imread+preprocess removed (x preloaded)."""
    imgs = split["imgs"]
    cm = split["coco_mapping"]
    n_frame = len(imgs)
    t_total = n_frame / FPS
    h0, w0 = imgs[0]["height"], imgs[0]["width"]
    gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
    pad_x = (IMG_SIZE - w0 * gain) / 2.0
    pad_y = (IMG_SIZE - h0 * gain) / 2.0
    t_elapsed = 0.0
    last = -1
    while t_elapsed < t_total and not stop.is_set():
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last:
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS
        if fidx >= n_frame:
            break
        last = fidx
        x = x_list[fidx]
        wait = time.time()
        t0 = time.time()
        out = model(x)
        res = model.postprocess(out, conf_thres=CONF, iou_thres=IOU)
        box_cls = getattr(res, "box_cls", None)
        if box_cls is None or box_cls.shape[0] == 0:
            bb = np.zeros((0, 4), np.float32); sc = np.zeros(0, np.float32); lb = np.zeros(0, np.int32)
        else:
            arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
            xyxy = arr[:, :4].astype(np.float32, copy=True)
            xyxy[:, [0, 2]] -= pad_x
            xyxy[:, [1, 3]] -= pad_y
            xyxy /= gain
            np.clip(xyxy[:, [0, 2]], 0, w0, out=xyxy[:, [0, 2]])
            np.clip(xyxy[:, [1, 3]], 0, h0, out=xyxy[:, [1, 3]])
            ahd = cm[arr[:, 5].astype(int)]
            sel = ahd < N_AHD
            bb = xyxy[sel]; sc = arr[:, 4].astype(np.float32)[sel]; lb = ahd[sel].astype(np.int32)
        rt = time.time() - t0
        eff = time.time() - wait
        t_elapsed += rt
        result["infer_ms"].append(rt * 1000)
        result["eff_ms"].append(eff * 1000)
        result["timestamps"].append(t_elapsed)
        result["input_fidx"].append(fidx)
        result["results"].append((bb, sc, lb))


def launch_stress(c):
    procs = [subprocess.Popen([str(PY), str(STRESS), str(core)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for core in range(c)]
    if c:
        time.sleep(1.0)
    return procs


def kill_stress(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()


def run_cfg(cfg, splits, models, x_all, rep):
    stress, bg_stops, bg_threads = [], [], []
    if cfg == "synth_c24":
        stress = launch_stress(24)
    elif cfg == "L2LM":
        bg_stops, bg_threads = h2.start_bg_custom("L2_lm")
    torch.set_num_threads(4)
    stop = threading.Event()
    results = [defaultdict(list) for _ in range(4)]
    threads = [threading.Thread(target=ablated_worker,
                                args=(splits[i], models[i], x_all[i], results[i], stop),
                                daemon=True) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if cfg == "synth_c24":
        kill_stress(stress)
    elif cfg == "L2LM":
        stop_background(bg_stops, bg_threads)
    saps, skips = [], []
    for i in range(4):
        sap = per_stream_sap(splits[i], results[i])
        saps.append(sap["sap_5095"]); skips.append(sap["frame_skip_pct"])
    return float(np.min(saps)), float(np.mean(saps)), float(np.mean(skips))


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 NPU engines + L2 co-tenants ...", flush=True)
    npu = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu)
    preload_background_models("L2")
    print("preloading decoded+letterboxed frames (ablate read_pre) ...", flush=True)
    x_all = preload_frames(splits, npu)

    rows = []
    for cfg in ["none", "synth_c24", "L2LM"]:
        print(f"\n=== ablated (no read_pre): {cfg} ===", flush=True)
        for rep in range(REPS):
            worst, mean, skip = run_cfg(cfg, splits, npu, x_all, rep)
            rows.append({"variant": "ablated_no_readpre", "config": cfg, "rep": rep,
                         "npu_skip": round(skip, 2), "worst_sap": round(worst, 4),
                         "mean_sap": round(mean, 4)})
            print(f"  [{cfg}/rep{rep}] npu_skip={skip:5.1f}% worst={worst:.4f} mean={mean:.4f}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT}")

    # verdict
    def m(cfg):
        v = [r["npu_skip"] for r in rows if r["config"] == cfg]
        return float(np.mean(v))
    print("\n=== ABLATION VERDICT (imread+preprocess removed) ===")
    print(f"  none  DM = {m('none'):.1f}%  (stock ~0%)")
    print(f"  synth DM = {m('synth_c24'):.1f}%  (stock ~27%)")
    print(f"  L2LM  DM = {m('L2LM'):.1f}%  (stock ~68%)")
    if m("L2LM") <= 15:
        print("  -> L2LM DM COLLAPSES without read_pre => host imread/preprocess is the "
              "CAUSAL stalled stage.")
        if m("synth_c24") <= 15:
            print("  -> synth also collapses => generalizes to a SHARED host-preprocess bottleneck.")
        else:
            print("  -> synth does NOT collapse => synth stalls a different stage than L2LM.")
    else:
        print("  -> L2LM DM persists without read_pre => stall is NOT (only) imread; "
              "attribute elsewhere (on-chip dispatch / postproc / scheduling).")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
