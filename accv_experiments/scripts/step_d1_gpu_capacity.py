"""Step D1 — GPU capacity under foreground × background contention.

For each (background_level, n_streams), spawn n_streams YOLO11s foreground
streams on GPU + background workers on GPU. Measure per-stream sAP,
inference latency, frame skip percentage.

CSV out: results/step_d1_gpu_capacity.csv
Figure : results/figures/step_d1_capacity.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _step_d_common import (
    BG_REGISTRY, FGModelGPU, load_split_for_sid, load_val,
    preload_background_models, run_experiment,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"
OUT_CSV = RES / "step_d1_gpu_capacity.csv"

N_STREAMS_LIST = [1, 2, 3, 4, 5, 6]
BG_LEVELS = ["L0", "L1", "L2", "L3"]
TEST_SID = 5  # shortest log (442 frames, ~14.7s simulated stream window)

COLS = ["bg_level", "n_streams", "stream_id",
        "eff_latency_mean", "eff_latency_p95",
        "infer_latency_mean", "infer_latency_p95",
        "frame_skip_pct", "n_processed",
        "sap_5095", "sap_50", "sap_small", "sap_medium", "sap_large"]


def main():
    val = load_val()
    split = load_split_for_sid(val, TEST_SID)
    print(f"[d1] using sid={TEST_SID} log={split['log_name'][:18]}…  frames={len(split['imgs'])}")

    # ORDER MATTERS: pre-load ORT bg models BEFORE torch+CUDA Qwen2-VL, BEFORE
    # any ultralytics YOLO. Otherwise ORT 1.20 fails its CUDA EP init.
    print("[d1] preloading background models (ORT first, then torch)…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[d1] bg preload done in {time.time()-t0:.1f}s")

    # Pre-create up to max-N GPU foreground models, reuse across cells.
    print("[d1] preloading 6 GPU YOLO instances…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(max(N_STREAMS_LIST))]
    print(f"[d1] yolo preload done in {time.time()-t0:.1f}s")

    rows = []
    cell_idx = 0
    n_cells = len(BG_LEVELS) * len(N_STREAMS_LIST)
    for bg in BG_LEVELS:
        for n in N_STREAMS_LIST:
            cell_idx += 1
            print(f"\n[d1 {cell_idx:2d}/{n_cells}] bg={bg}  n_streams={n}")
            # All streams point at one shared GPU model each (one per slot)
            # We map each fg slot to one GPU model — single dev key shared.
            class _Wrap:
                def __init__(self, models): self.models = models
                def predict(self, p):
                    # round-robin not needed: each thread already has its own slot,
                    # but our fg_worker calls model.predict directly. We use one
                    # model per stream by passing _PerStream below.
                    raise RuntimeError("should not be called")
            # We need ONE model per stream. We pass via a custom fg_models map keyed
            # by a unique device label per stream so run_experiment dispatches to
            # the slot's model. Easiest: instead inline a tiny loop here matching
            # run_experiment's contract.
            from _step_d_common import fg_worker, per_stream_sap, start_background, stop_background
            from collections import defaultdict
            import threading
            results = [defaultdict(list) for _ in range(n)]
            stop = threading.Event()
            splits = [split for _ in range(n)]
            bg_stops, bg_threads = start_background(bg)
            ts = []
            for i in range(n):
                t = threading.Thread(target=fg_worker,
                                     args=(i, "GPU", splits[i], gpu_models[i], results[i], stop),
                                     daemon=True, name=f"fg_s{i}")
                ts.append(t)
            t_start = time.time()
            for t in ts: t.start()
            for t in ts: t.join()
            wall = time.time() - t_start
            stop_background(bg_stops, bg_threads)

            for i in range(n):
                m = per_stream_sap(splits[i], results[i])
                rows.append({
                    "bg_level": bg,
                    "n_streams": n,
                    "stream_id": i,
                    "eff_latency_mean": round(m["eff_mean_ms"], 2),
                    "eff_latency_p95":  round(m["eff_p95_ms"], 2),
                    "infer_latency_mean": round(m["infer_mean_ms"], 2),
                    "infer_latency_p95":  round(m["infer_p95_ms"], 2),
                    "frame_skip_pct": round(m["frame_skip_pct"], 1),
                    "n_processed": m["n_processed"],
                    "sap_5095": round(m["sap_5095"], 4),
                    "sap_50":   round(m["sap_50"], 4),
                    "sap_small":  round(m["sap_small"], 4),
                    "sap_medium": round(m["sap_medium"], 4),
                    "sap_large":  round(m["sap_large"], 4),
                })
            sap_mean = sum(r["sap_5095"] for r in rows[-n:]) / n
            inf_mean = sum(r["infer_latency_mean"] for r in rows[-n:]) / n
            sk_mean  = sum(r["frame_skip_pct"] for r in rows[-n:]) / n
            print(f"   wall={wall:5.1f}s  avg_infer={inf_mean:5.1f}ms  avg_skip={sk_mean:5.1f}%  avg_sAP={sap_mean:.3f}")

    # Write CSV
    RES.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[d1] csv: {OUT_CSV} ({len(rows)} rows)")

    # Figure: per-bg_level lines, X=n_streams, two panels (latency, sAP) + 33ms line
    FIG.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["bg_level", "n_streams"]).agg(
        infer_mean=("infer_latency_mean", "mean"),
        sap_mean=("sap_5095", "mean"),
        skip_mean=("frame_skip_pct", "mean"),
    ).reset_index()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"L0": "#444444", "L1": "#3680c4", "L2": "#e9b341", "L3": "#c43b3b"}
    for bg in BG_LEVELS:
        sub = agg[agg.bg_level == bg].sort_values("n_streams")
        ax1.plot(sub["n_streams"], sub["infer_mean"], "-o", label=bg, color=colors[bg])
        ax2.plot(sub["n_streams"], sub["skip_mean"],  "-o", label=bg, color=colors[bg])
        ax3.plot(sub["n_streams"], sub["sap_mean"],   "-o", label=bg, color=colors[bg])
    ax1.axhline(1000/30, color="red", linestyle="--", linewidth=1, label="33ms budget")
    ax1.set_xlabel("# foreground GPU streams"); ax1.set_ylabel("infer latency (ms)")
    ax1.set_title("Inference latency"); ax1.grid(alpha=0.3); ax1.legend(title="bg")
    ax2.set_xlabel("# foreground GPU streams"); ax2.set_ylabel("frame skip %")
    ax2.set_title("Frame skip rate"); ax2.grid(alpha=0.3); ax2.legend(title="bg")
    ax3.set_xlabel("# foreground GPU streams"); ax3.set_ylabel("per-stream sAP[0.50:0.95] mean")
    ax3.set_title("Per-stream sAP (mean across streams)"); ax3.grid(alpha=0.3); ax3.legend(title="bg")
    plt.suptitle("Step D1 — GPU capacity (foreground YOLO11s ×N + background load)", fontsize=12)
    plt.tight_layout()
    out_fig = FIG / "step_d1_capacity.png"
    fig.savefig(out_fig, dpi=120)
    print(f"[d1] figure: {out_fig}")

    # Find N where frame skip starts (>5%) per bg_level
    print("\n=== capacity threshold (first N with avg frame-skip > 5%) ===")
    for bg in BG_LEVELS:
        sub = agg[agg.bg_level == bg].sort_values("n_streams")
        thr = sub[sub.skip_mean > 5.0]
        if len(thr) > 0:
            n_thr = int(thr.iloc[0]["n_streams"])
            print(f"  {bg}: N={n_thr}  (skip={thr.iloc[0]['skip_mean']:.1f}%)")
        else:
            print(f"  {bg}: no frame skip up to N={N_STREAMS_LIST[-1]}")


if __name__ == "__main__":
    main()
