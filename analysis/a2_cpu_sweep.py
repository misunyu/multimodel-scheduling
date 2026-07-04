"""A2 Task 1 — host-CPU-localized contention sweep (mirror image of Fig 3).

Independent variable = number of CPU-pinned stress workers c (24-core host).
Stress is EXTERNAL subprocesses (a2_cpu_stress_worker.py), each pinned to a
distinct core, so they occupy real cores despite the GIL and touch NO GPU
(verified). For each c we measure All-GPU and All-NPU placements on PANEL4
(N=4), >=3 reps, exactly as rev30 does (same measure_multistream_full +
summarize + metrics). A realistic anchor point runs the L2LM language component
on CPU (TinyLLaMA ORT-CPU) instead of synthetic spin, to counter the "synthetic
stress is artificial" objection.

Pre-registered expectation: All-NPU worst-stream sAP decreases monotonically with
c (host post-processing / preprocess / decode stall -> NPU deadline miss), while
All-GPU stays comparatively flat (GPU inference off-host) until, possibly, an
extreme-c host bottleneck also hits the GPU preprocess path -- if observed, it is
reported, not hidden.

Reuses the frozen harness unchanged (imports only). New code stays in analysis/.
GPU/NPU OCCUPANCY: this script runs YOLOv11s on GPU and NPU. Time is reported by
--plan before any measurement.

Outputs: analysis/a2_cpu_sweep.csv (+ a2_cpu_sweep_util.csv sampler trace).
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "accv_experiments/scripts"
sys.path.insert(0, str(SD))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

import torch
from _step_d_common import load_val, load_split_for_sid
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                              set_active_npu_engines, dispose_npu_for, DETECTORS)
from phase_rev30_clean import (measure_multistream_full, summarize, PANEL4,
                               PERIOD, FPS, DATASET, THREADS)
import step_h2_robustness as h2
import a2_cotenants as a2c

OUT = ROOT / "analysis/a2_cpu_sweep.csv"
UTIL = ROOT / "analysis/a2_cpu_sweep_util.csv"
STRESS = ROOT / "analysis/a2_cpu_stress_worker.py"
SAMPLER = ROOT / "analysis/a2_util_sampler.py"
PY = ROOT / ".venv/bin/python"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]

SWEEP_C = [0, 2, 4, 8, 16, 24]
REPS = 3
PLACEMENTS = {"AllGPU": ["GPU"] * 4, "AllNPU": ["NPU"] * 4}
COLS = ["point", "cotenant", "c", "placement", "rep", "t_start", "t_end",
        "worst_sap", "mean_sap", "median_sap", "p10_sap",
        "sap_small", "sap_medium", "sap_large",
        "gpu_skip_pct", "npu_skip_pct", "deadline_margin_ms_mean",
        "cpu_pct_mean", "cpu_nbusy_mean", "gpu_util_mean", "power_w_mean",
        "per_stream_json"]


def launch_stress(c):
    procs = []
    for core in range(c):
        p = subprocess.Popen([str(PY), str(STRESS), str(core)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    if c:
        time.sleep(1.0)  # let them spin up
    return procs


def kill_stress(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()


def start_sampler():
    p = subprocess.Popen([str(PY), str(SAMPLER), str(UTIL), "0.2"])
    time.sleep(1.5)
    return p


def stop_sampler(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()


def slice_util(t0, t1):
    """Mean host CPU%, busy-core count, GPU util, power over [t0,t1]."""
    if not UTIL.exists():
        return {}
    rows = []
    with open(UTIL) as f:
        for r in csv.DictReader(f):
            e = float(r["epoch"])
            if t0 <= e <= t1:
                rows.append(r)
    if not rows:
        return {}
    def m(col):
        return float(np.mean([float(r[col]) for r in rows]))
    return {"cpu_pct_mean": round(m("cpu_pct"), 1),
            "cpu_nbusy_mean": round(m("cpu_nbusy"), 1),
            "gpu_util_mean": round(m("util_gpu"), 1),
            "power_w_mean": round(m("power_w"), 1)}


def append(row):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_point(point, cotenant, c, bg_level, splits, gpu_models, npu_models):
    """Run both placements x REPS for one sweep point."""
    procs = launch_stress(c) if cotenant == "cpu_stress" else []
    try:
        for pl_name, placement in PLACEMENTS.items():
            ng = sum(d == "GPU" for d in placement)
            nn = sum(d == "NPU" for d in placement)
            for rep in range(REPS):
                torch.set_num_threads(THREADS)
                t0 = time.time()
                agg = measure_multistream_full(PANEL4, splits, placement,
                                               gpu_models[:ng], npu_models[:nn], bg_level)
                t1 = time.time()
                s = summarize(agg, placement)
                row = {"point": point, "cotenant": cotenant, "c": c,
                       "placement": pl_name, "rep": rep,
                       "t_start": round(t0, 3), "t_end": round(t1, 3),
                       "worst_sap": s["worst_sap"], "mean_sap": s["mean_sap"],
                       "median_sap": s["median_sap"], "p10_sap": s["p10_sap"],
                       "sap_small": s["sap_small"], "sap_medium": s["sap_medium"],
                       "sap_large": s["sap_large"],
                       "gpu_skip_pct": s["gpu_skip_pct"], "npu_skip_pct": s["npu_skip_pct"],
                       "deadline_margin_ms_mean": s["deadline_margin_ms_mean"],
                       "per_stream_json": json.dumps(agg["per_stream"])}
                row.update(slice_util(t0, t1))
                append(row)
                print(f"  [{point}/{pl_name}/rep{rep}] c={c} worst={s['worst_sap']:.4f} "
                      f"gpu_skip={s['gpu_skip_pct']} npu_skip={s['npu_skip_pct']} "
                      f"cpu%={row.get('cpu_pct_mean','?')} ({t1-t0:.1f}s)", flush=True)
    finally:
        kill_stress(procs)


def plan():
    n_cells = (len(SWEEP_C) + 1) * len(PLACEMENTS) * REPS  # +1 anchor
    per_cell = 17  # ~s per 14s window incl. overhead (empirical rev30 ~15-20s)
    secs = n_cells * per_cell
    print(f"A2 Task 1 CPU-sweep plan:")
    print(f"  sweep c in {SWEEP_C} + 1 realistic anchor (tinyllama_cpu)")
    print(f"  x {list(PLACEMENTS)} x {REPS} reps = {n_cells} cells")
    print(f"  est ~{per_cell}s/cell -> ~{secs/60:.0f} min GPU+NPU occupancy "
          f"(+ ~2 min model load)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--anchor-only", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan(); return

    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]

    print("loading 4 GPU + 4 NPU yolo11s engines ...", flush=True)
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)

    # register CPU anchor co-tenant (thread; ORT-CPU releases GIL)
    a2c.preload_tinyllama_cpu(intra_threads=8)
    h2.BG_VARIANTS["A2_CPUANCHOR"] = [a2c._bg_tinyllama_cpu_loop]

    samp = start_sampler()
    try:
        if not args.anchor_only:
            for c in SWEEP_C:
                print(f"\n=== CPU stress c={c} ===", flush=True)
                run_point(f"c{c}", "cpu_stress", c, "L0", splits, gpu_models, npu_models)
        print(f"\n=== realistic anchor: TinyLLaMA CPU inference ===", flush=True)
        run_point("anchor_llmcpu", "tinyllama_cpu", -1, "A2_CPUANCHOR",
                  splits, gpu_models, npu_models)
    finally:
        stop_sampler(samp)
        dispose_npu_for("yolo11s")
    print("\n=== A2 Task 1 done ===", flush=True)


if __name__ == "__main__":
    main()
