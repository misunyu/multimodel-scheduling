"""Step D2 — Multi-stream placement infrastructure + 4 sanity tests.

Reads a `placement` (list of devices, one per stream) and a `bg_level`,
runs N foreground streams (each can be on a different device), and reports
per-stream sAP plus aggregate (mean / worst / variance).

Includes 4 built-in sanity tests:
  1. 4× GPU, L0  — should match Step D1 cell {bg=L0, n=4}
  2. 2× GPU + 2× NPU, L0  — verifies independent per-device measurement
  3. 1× NPU only (Mobilint device check)
  4. 4× GPU, L2  — sAP should drop vs test 1
"""

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from pathlib import Path

from _step_d_common import (FGModelCPU, FGModelGPU, fg_worker, get_npu_model,
                            load_split_for_sid, load_val, per_stream_sap,
                            preload_background_models, preload_npu_instances,
                            start_background, stop_background)

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
OUT_CSV = RES / "step_d2_multistream.csv"

# Use the 4 shortest logs (~470 frames each) for stream diversity
STREAM_SIDS = [5, 12, 13, 14]  # we'll dedup if fewer streams needed


def run_placement(placement, bg_level, gpu_models, cpu_models, sids=None):
    """One placement experiment. Returns list of per-stream metric dicts + wall."""
    val = load_val()
    n = len(placement)
    sids = (sids or STREAM_SIDS)[:n]
    splits = [load_split_for_sid(val, sid) for sid in sids]

    # Preload one NPU instance per NPU stream — Mobilint supports concurrent
    # multi-model execution, so each stream gets its own engine.
    n_npu = sum(1 for d in placement if d == "NPU")
    if n_npu > 0:
        preload_npu_instances(n_npu, infer_mode="multi")

    bg_stops, bg_threads = start_background(bg_level)

    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    threads = []
    # device-slot index trackers
    gpu_i = 0; cpu_i = 0; npu_i = 0
    for i, dev in enumerate(placement):
        if dev == "GPU":
            model = gpu_models[gpu_i]; gpu_i += 1
        elif dev == "CPU":
            model = cpu_models[cpu_i]; cpu_i += 1
        else:  # NPU — give this stream its own dedicated engine
            model = get_npu_model(slot=npu_i); npu_i += 1
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, splits[i], model, results[i], stop),
                             daemon=True, name=f"fg_s{i}_{dev}")
        threads.append(t)
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)

    metrics = []
    for i, (sid, split) in enumerate(zip(sids, splits)):
        m = per_stream_sap(split, results[i])
        m.update({"stream_id": i, "device": placement[i], "sid": sid,
                  "log_id": split["log_name"]})
        metrics.append(m)
    return metrics, wall


def aggregate(metrics):
    saps = [m["sap_5095"] for m in metrics]
    import numpy as np
    return {"mean": float(np.mean(saps)),
            "worst": float(np.min(saps)),
            "std": float(np.std(saps)),
            "n_streams": len(saps)}


def fmt_row(stream_metrics, prefix=""):
    out = []
    for m in stream_metrics:
        out.append(f"{prefix}  s{m['stream_id']} dev={m['device']} sid={m['sid']:>2}  "
                   f"sAP={m['sap_5095']:.3f}  infer={m['infer_mean_ms']:5.1f}ms  "
                   f"skip(>33ms)={m['frame_skip_pct']:5.1f}%  small={m['sap_small']:.3f}")
    return "\n".join(out)


def main():
    # Preload models once. ORDER MATTERS: ORT bg models first, then torch CUDA.
    print("[d2] preloading background models (ORT-first, then torch)…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[d2] bg preload {time.time()-t0:.1f}s")

    print("[d2] preloading 4 GPU + 1 CPU YOLO instances…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(4)]
    cpu_models = [FGModelCPU() for _ in range(1)]
    print(f"[d2] yolo preload {time.time()-t0:.1f}s")

    rows_for_csv = []

    # ---- Test 1: 4 GPU, L0 (should match Step D1 cell bg=L0, n=4) ----
    print("\n=== Test 1: 4×GPU, L0 (compare with Step D1 L0/N=4) ===")
    m1, w1 = run_placement(["GPU"]*4, "L0", gpu_models, cpu_models, sids=[5]*4)
    agg1 = aggregate(m1)
    print(fmt_row(m1, "  "))
    print(f"  → mean={agg1['mean']:.3f}  worst={agg1['worst']:.3f}  std={agg1['std']:.3f}  wall={w1:.1f}s")
    for m in m1:
        rows_for_csv.append({"test": "T1_4GPU_L0", "bg_level": "L0", **{k: m[k] for k in
            ["stream_id","device","sid","log_id","infer_mean_ms","infer_p95_ms","eff_mean_ms",
             "frame_skip_pct","n_processed","sap_5095","sap_50","sap_small","sap_medium","sap_large"]}})

    # ---- Test 2: 2 GPU + 2 NPU, L0 (independent per-device measurement) ----
    print("\n=== Test 2: 2×GPU + 2×NPU, L0  (per-device independence check) ===")
    m2, w2 = run_placement(["GPU","GPU","NPU","NPU"], "L0", gpu_models, cpu_models, sids=[5,12,13,14])
    agg2 = aggregate(m2)
    print(fmt_row(m2, "  "))
    print(f"  → mean={agg2['mean']:.3f}  worst={agg2['worst']:.3f}  std={agg2['std']:.3f}  wall={w2:.1f}s")
    for m in m2:
        rows_for_csv.append({"test": "T2_2GPU_2NPU_L0", "bg_level": "L0", **{k: m[k] for k in
            ["stream_id","device","sid","log_id","infer_mean_ms","infer_p95_ms","eff_mean_ms",
             "frame_skip_pct","n_processed","sap_5095","sap_50","sap_small","sap_medium","sap_large"]}})

    # ---- Test 3: 1 NPU only — verify it actually runs on Mobilint device ----
    print("\n=== Test 3: 1×NPU only  (Mobilint device check) ===")
    m3, w3 = run_placement(["NPU"], "L0", gpu_models, cpu_models, sids=[5])
    print(fmt_row(m3, "  "))
    print(f"  → wall={w3:.1f}s (NPU latency ~10ms per frame expected)")
    # device verification — check Aries device is actively used
    try:
        import qbruntime as qb
        devs = qb.get_available_device_numbers()
        print(f"  qbruntime devices visible: {devs}  (expected [0] for Mobilint Aries)")
    except Exception as e:
        print(f"  qbruntime check failed: {e}")
    for m in m3:
        rows_for_csv.append({"test": "T3_1NPU_L0", "bg_level": "L0", **{k: m[k] for k in
            ["stream_id","device","sid","log_id","infer_mean_ms","infer_p95_ms","eff_mean_ms",
             "frame_skip_pct","n_processed","sap_5095","sap_50","sap_small","sap_medium","sap_large"]}})

    # ---- Test 4: 4 GPU, L2 (background should hurt foreground sAP vs test 1) ----
    print("\n=== Test 4: 4×GPU, L2 (background contention check) ===")
    m4, w4 = run_placement(["GPU"]*4, "L2", gpu_models, cpu_models, sids=[5]*4)
    agg4 = aggregate(m4)
    print(fmt_row(m4, "  "))
    print(f"  → mean={agg4['mean']:.3f}  worst={agg4['worst']:.3f}  std={agg4['std']:.3f}  wall={w4:.1f}s")
    delta = agg4["mean"] - agg1["mean"]
    print(f"  Δ mean_sAP (T4 − T1) = {delta:+.3f}  (negative = background hurt)")
    for m in m4:
        rows_for_csv.append({"test": "T4_4GPU_L2", "bg_level": "L2", **{k: m[k] for k in
            ["stream_id","device","sid","log_id","infer_mean_ms","infer_p95_ms","eff_mean_ms",
             "frame_skip_pct","n_processed","sap_5095","sap_50","sap_small","sap_medium","sap_large"]}})

    # CSV
    RES.mkdir(parents=True, exist_ok=True)
    cols = ["test","bg_level","stream_id","device","sid","log_id",
            "infer_mean_ms","infer_p95_ms","eff_mean_ms","frame_skip_pct","n_processed",
            "sap_5095","sap_50","sap_small","sap_medium","sap_large"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_for_csv:
            r2 = {k: r[k] for k in cols}
            # round numerics
            for kf in ["infer_mean_ms","infer_p95_ms","eff_mean_ms","frame_skip_pct"]:
                r2[kf] = round(r2[kf], 2)
            for kf in ["sap_5095","sap_50","sap_small","sap_medium","sap_large"]:
                r2[kf] = round(r2[kf], 4)
            w.writerow(r2)
    print(f"\n[d2] csv: {OUT_CSV} ({len(rows_for_csv)} rows)")

    # Summary
    print("\n=== Test summary ===")
    print(f"  T1 4xGPU L0    : mean sAP {agg1['mean']:.3f}  worst {agg1['worst']:.3f}")
    print(f"  T2 2xGPU+2xNPU : mean sAP {agg2['mean']:.3f}  worst {agg2['worst']:.3f}")
    print(f"  T3 1xNPU       : sAP {m3[0]['sap_5095']:.3f}  infer={m3[0]['infer_mean_ms']:.1f}ms")
    print(f"  T4 4xGPU L2    : mean sAP {agg4['mean']:.3f}  worst {agg4['worst']:.3f}  ΔvsT1={agg4['mean']-agg1['mean']:+.3f}")


if __name__ == "__main__":
    main()
