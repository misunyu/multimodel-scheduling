"""T-D0 — driver-state reproducibility measurement.

Goal: measure the same Table 1 cell (mean NPU AP_large - mean GPU AP_large)
N>=8 times under varied process / driver / load conditions to characterize
the rev6 vs rev7 divergence (paper §5–6 P0-D in HUMAN_QUEUE).

Per run: 24 Argoverse-HD val logs × NPU L0 single-stream (global8 mode) on
the pinned legacy mxq (b2441f9d). GPU baseline is reused from rev7 (FP32,
deterministic). Metadata recorded per run.

Run modes:
  cold     : spawned as a fresh subprocess (this script's worker mode)
  warm     : same Python process as the previous run
  loaded   : cold but preceded by a brief NPU multistream load to mimic the
             rev6 conditions where v11s multistream had just disposed

Output: results/repro_driverstate.csv (1 row per run + 1 baseline GPU row).
"""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

RES = Path("accv_experiments/results")
OUT = RES / "repro_driverstate.csv"
PIN = json.loads((RES / "rev7_pin.json").read_text())
MXQ = PIN["mxq_path"]
MXQ_SHA = PIN["mxq_sha256"][:12]

N_RUNS = 8  # >= 8 per spec
MODES = ["cold", "cold", "cold", "warm", "warm", "loaded", "loaded", "cold"]


# ---------------- worker (invoked by subprocess) ----------------

def _worker_measure_npu():
    """Single-process worker: load NPU once in global8, sweep 24 logs L0,
    print JSON to stdout."""
    import _step_d_common as cm
    from _step_d_common import (fg_worker, load_split_for_sid, load_val,
                                per_stream_sap, start_background, stop_background,
                                preload_background_models)
    from mblt_model_zoo import vision as mv
    from step0_compare_devices import CONF, IOU

    t_total = time.time()
    preload_background_models(max_level="L0")  # L0 = nothing, just no-op
    val = load_val()
    n_logs = len(val["sequences"])

    # Construct NPU engine: same binary, same mode as rev6/rev7 baseline.
    t_load = time.time()
    m = mv.YOLO11s(local_path=MXQ, infer_mode="global8", product="aries")
    dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
    x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
    cm._NPU_INSTANCES.clear()
    cm._NPU_INSTANCES.append(m)
    load_wall = time.time() - t_load

    sap_s = []; sap_m = []; sap_l = []; sap_5095 = []
    t_meas = time.time()
    for sid in range(n_logs):
        split = load_split_for_sid(val, sid)
        res = defaultdict(list)
        stop = threading.Event()
        bg_stops, bg_threads = start_background("L0")
        fg_worker(0, "NPU", split, m, res, stop)
        stop_background(bg_stops, bg_threads)
        s = per_stream_sap(split, res)
        sap_s.append(s["sap_small"]); sap_m.append(s["sap_medium"])
        sap_l.append(s["sap_large"]); sap_5095.append(s["sap_5095"])
    meas_wall = time.time() - t_meas
    try: m.dispose()
    except Exception: pass

    # Capture nvidia-smi snapshot for GPU metadata (RTX 5090 idle state)
    smi = ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,driver_version,clocks.gr",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        smi = r.stdout.strip()
    except Exception as e:
        smi = f"(nvidia-smi err: {e})"

    out = {
        "load_wall_sec": round(load_wall, 2),
        "meas_wall_sec": round(meas_wall, 2),
        "total_wall_sec": round(time.time() - t_total, 2),
        "n_logs": n_logs,
        "npu_sap_5095_mean": float(np.mean(sap_5095)),
        "npu_sap_small_mean": float(np.mean(sap_s)),
        "npu_sap_medium_mean": float(np.mean(sap_m)),
        "npu_sap_large_mean": float(np.mean(sap_l)),
        "npu_sap_large_std":  float(np.std(sap_l, ddof=0)),
        "nvidia_smi": smi,
    }
    print("=JSONOUT=" + json.dumps(out) + "=JSONOUT=")


def _worker_load_npu_then_dispose_multi():
    """Mimic the rev6 v11s multistream load that preceded the single-stream
    baseline drift: load 4 instances in single mode briefly, then dispose."""
    import _step_d_common as cm
    from mblt_model_zoo import vision as mv
    from step0_compare_devices import CONF, IOU
    t0 = time.time()
    insts = []
    try:
        for _ in range(4):
            m = mv.YOLO11s(local_path=MXQ, infer_mode="single", product="aries")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
            insts.append(m)
    except Exception as e:
        print(f"loaded-warmup load fail: {e}")
    # Run a short hot loop to make sure the cores are exercised
    if insts:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        end = time.time() + 5.0
        while time.time() < end:
            for inst in insts:
                try:
                    x = inst.preprocess(dummy); o = inst(x)
                    inst.postprocess(o, conf_thres=CONF, iou_thres=IOU)
                except Exception:
                    break
    for inst in insts:
        try: inst.dispose()
        except Exception: pass
    print(f"=PRELOAD_DONE= duration={time.time()-t0:.1f}s n_loaded={len(insts)}")


# ---------------- driver (main process) ----------------

def driver():
    # Re-use rev7 GPU large baseline (deterministic FP32) — saves ~2 min/run.
    gpu_baseline_csv = RES / "rev7_single_stream.csv"
    if gpu_baseline_csv.exists():
        df = pd.read_csv(gpu_baseline_csv)
        g = df[df.device == "GPU"]
        gpu_large_mean = float(g["sap_l"].mean())
        gpu_large_std = float(g["sap_l"].std(ddof=0))
        gpu_medium_mean = float(g["sap_m"].mean())
        gpu_small_mean = float(g["sap_s"].mean())
    else:
        gpu_large_mean = gpu_medium_mean = gpu_small_mean = gpu_large_std = float("nan")

    # initialise CSV
    cols = ["run_idx", "ts_iso", "mode", "elapsed_since_prev_sec",
            "npu_sap_small_mean", "npu_sap_medium_mean", "npu_sap_large_mean",
            "npu_sap_large_std", "npu_sap_5095_mean",
            "gpu_sap_small_mean", "gpu_sap_medium_mean", "gpu_sap_large_mean",
            "large_gap_NPU_minus_GPU", "medium_gap_NPU_minus_GPU", "small_gap_NPU_minus_GPU",
            "load_wall_sec", "meas_wall_sec", "total_wall_sec",
            "mxq_sha", "host", "nvidia_smi", "notes"]
    if OUT.exists(): OUT.unlink()
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    host = socket.gethostname()
    prev_ts = None
    print(f"[T-D0] driver start. mxq sha[:12]={MXQ_SHA}, host={host}")
    print(f"[T-D0] GPU FP32 baseline (from rev7): large={gpu_large_mean:.4f}  medium={gpu_medium_mean:.4f}  small={gpu_small_mean:.4f}")

    for i, mode in enumerate(MODES):
        run_idx = i + 1
        ts0 = time.time()
        elapsed_since = (ts0 - prev_ts) if prev_ts else 0.0
        print(f"\n=== run {run_idx}/{N_RUNS}  mode={mode}  elapsed_since_prev={elapsed_since:.1f}s ===")
        notes = []

        if mode == "loaded":
            # Pre-warm with brief multistream single-mode load
            print(f"  pre-warm: spawning multi-mode preload (single binary forced via local_path)…")
            p = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--preload"],
                capture_output=True, text=True, timeout=180)
            notes.append("preload_stdout=" + p.stdout.strip().split("\n")[-1][:120])
            notes.append("preload_rc=" + str(p.returncode))

        if mode == "warm":
            # warm uses the SAME process; we run the worker inline rather than
            # spawning a fresh subprocess. (Not truly cold; uses cached SDK state.)
            print("  WARM: running worker in-process (no subprocess restart)…")
            # Avoid module re-import: capture stdout from a Python -c
            out = subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path.insert(0,'{SCRIPT_DIR}'); "
                 "import phase_TD0_repro as p; p._worker_measure_npu()"],
                capture_output=True, text=True, timeout=600)
            stdout = out.stdout
        else:
            # cold or loaded: spawn fresh process for true cold-start
            print("  COLD subprocess…")
            out = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--measure"],
                capture_output=True, text=True, timeout=600)
            stdout = out.stdout

        # parse JSON
        result = None
        for line in stdout.splitlines():
            if line.startswith("=JSONOUT=") and line.endswith("=JSONOUT="):
                try:
                    result = json.loads(line[len("=JSONOUT="):-len("=JSONOUT=")])
                    break
                except Exception:
                    pass
        if result is None:
            print(f"  FAIL: could not parse worker JSON output. last stdout 30 lines:")
            for line in stdout.splitlines()[-30:]:
                print(f"    {line}")
            notes.append("worker_json_missing")
            row = {"run_idx": run_idx, "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "mode": mode, "elapsed_since_prev_sec": round(elapsed_since, 1),
                   "notes": ";".join(notes), "host": host, "mxq_sha": MXQ_SHA}
        else:
            row = {
                "run_idx": run_idx,
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": mode,
                "elapsed_since_prev_sec": round(elapsed_since, 1),
                "npu_sap_small_mean":  round(result["npu_sap_small_mean"], 4),
                "npu_sap_medium_mean": round(result["npu_sap_medium_mean"], 4),
                "npu_sap_large_mean":  round(result["npu_sap_large_mean"], 4),
                "npu_sap_large_std":   round(result["npu_sap_large_std"], 4),
                "npu_sap_5095_mean":   round(result["npu_sap_5095_mean"], 4),
                "gpu_sap_small_mean":  round(gpu_small_mean, 4),
                "gpu_sap_medium_mean": round(gpu_medium_mean, 4),
                "gpu_sap_large_mean":  round(gpu_large_mean, 4),
                "large_gap_NPU_minus_GPU":  round(result["npu_sap_large_mean"] - gpu_large_mean, 4),
                "medium_gap_NPU_minus_GPU": round(result["npu_sap_medium_mean"] - gpu_medium_mean, 4),
                "small_gap_NPU_minus_GPU":  round(result["npu_sap_small_mean"] - gpu_small_mean, 4),
                "load_wall_sec":  result["load_wall_sec"],
                "meas_wall_sec":  result["meas_wall_sec"],
                "total_wall_sec": result["total_wall_sec"],
                "mxq_sha": MXQ_SHA, "host": host,
                "nvidia_smi": result["nvidia_smi"],
                "notes": ";".join(notes),
            }
            print(f"  run {run_idx} {mode:<6s}  NPU large={row['npu_sap_large_mean']:.4f}  "
                  f"gap_large={row['large_gap_NPU_minus_GPU']:+.4f}  "
                  f"(GPU large {gpu_large_mean:.4f})  wall {row['total_wall_sec']:.1f}s")
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)
        prev_ts = time.time()

    # Summary
    df = pd.read_csv(OUT)
    if "large_gap_NPU_minus_GPU" in df.columns and df["large_gap_NPU_minus_GPU"].notna().any():
        gaps = df["large_gap_NPU_minus_GPU"].dropna()
        print(f"\n=== summary over {len(gaps)} runs ===")
        print(f"  large_gap range:  [{gaps.min():+.4f}, {gaps.max():+.4f}]")
        print(f"  large_gap mean:   {gaps.mean():+.4f}")
        print(f"  large_gap stddev: {gaps.std(ddof=0):.4f}")
        # rev6 reproduction?
        rev6_gap = -0.10
        worst = gaps.min()
        print(f"  worst NPU-GPU large gap: {worst:+.4f}")
        if worst <= rev6_gap + 0.02:
            print(f"  rev6 −0.10 REPRODUCED at run with mode={df.loc[gaps.idxmin(),'mode']}")
        else:
            print(f"  rev6 −0.10 NOT reproduced (closest {worst:+.4f}); divergence likely "
                  f"required a specific driver state not captured in {len(gaps)} cold/warm/loaded runs.")
    print(f"\nsaved {OUT}")


# ---------------- CLI dispatch ----------------

if __name__ == "__main__":
    if "--measure" in sys.argv:
        _worker_measure_npu()
    elif "--preload" in sys.argv:
        _worker_load_npu_then_dispose_multi()
    else:
        driver()
