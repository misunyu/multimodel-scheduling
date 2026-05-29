"""Step D1b — NPU capacity sweep (concurrent foreground YOLO11s on NPU).

Mobilint Aries supports multiple concurrent model instances. This sweep
measures per-stream sAP / latency / frame-skip as N foreground YOLO11s
streams are loaded onto independent NPU engines (infer_mode="multi", which
partitions the chip across clusters).

Background load is NOT applied here — the bg workers in step_d1 run on GPU
and have no effect on NPU. NPU capacity is decoupled from GPU contention.

CSV out: results/step_d1b_npu_capacity.csv
"""

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _step_d_common import (fg_worker, load_split_for_sid, load_val,
                            per_stream_sap, preload_npu_instances)

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"
OUT_CSV = RES / "step_d1b_npu_capacity.csv"

N_STREAMS_LIST = [1, 2, 3, 4]  # Mobilint multi-mode practical concurrent count
TEST_SID = 5
INFER_MODE = "multi"  # cluster-partitioned for concurrent execution

COLS = ["infer_mode", "n_streams", "stream_id",
        "eff_latency_mean", "eff_latency_p95",
        "infer_latency_mean", "infer_latency_p95",
        "frame_skip_pct", "n_processed",
        "sap_5095", "sap_50", "sap_small", "sap_medium", "sap_large"]


def main():
    val = load_val()
    split = load_split_for_sid(val, TEST_SID)
    print(f"[d1b] sid={TEST_SID} log={split['log_name'][:18]}…  frames={len(split['imgs'])}")
    print(f"[d1b] preloading {max(N_STREAMS_LIST)} NPU engines (infer_mode='{INFER_MODE}')…")
    t0 = time.time()
    npu_instances = preload_npu_instances(max(N_STREAMS_LIST), infer_mode=INFER_MODE)
    print(f"[d1b] NPU preload {time.time()-t0:.1f}s ({len(npu_instances)} instances)")

    rows = []
    for n in N_STREAMS_LIST:
        print(f"\n[d1b] N={n} concurrent NPU streams")
        results = [defaultdict(list) for _ in range(n)]
        stop = threading.Event()
        ts = []
        for i in range(n):
            t = threading.Thread(target=fg_worker,
                                 args=(i, "NPU", split, npu_instances[i], results[i], stop),
                                 daemon=True, name=f"fg_s{i}_NPU")
            ts.append(t)
        t_start = time.time()
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.time() - t_start

        # Per-stream metrics
        for i in range(n):
            m = per_stream_sap(split, results[i])
            rows.append({
                "infer_mode": INFER_MODE,
                "n_streams": n,
                "stream_id": i,
                "eff_latency_mean":   round(m["eff_mean_ms"], 2),
                "eff_latency_p95":    round(m["eff_p95_ms"], 2),
                "infer_latency_mean": round(m["infer_mean_ms"], 2),
                "infer_latency_p95":  round(m["infer_p95_ms"], 2),
                "frame_skip_pct":     round(m["frame_skip_pct"], 1),
                "n_processed":        m["n_processed"],
                "sap_5095":   round(m["sap_5095"], 4),
                "sap_50":     round(m["sap_50"], 4),
                "sap_small":  round(m["sap_small"], 4),
                "sap_medium": round(m["sap_medium"], 4),
                "sap_large":  round(m["sap_large"], 4),
            })
        sap_mean = sum(r["sap_5095"] for r in rows[-n:]) / n
        inf_mean = sum(r["infer_latency_mean"] for r in rows[-n:]) / n
        sk_mean  = sum(r["frame_skip_pct"] for r in rows[-n:]) / n
        print(f"   wall={wall:5.1f}s  avg_infer={inf_mean:5.1f}ms  avg_skip={sk_mean:5.1f}%  avg_sAP={sap_mean:.3f}")

    RES.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[d1b] csv: {OUT_CSV} ({len(rows)} rows)")

    # Figure: per-stream latency growth vs N
    FIG.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("n_streams").agg(
        infer_mean=("infer_latency_mean", "mean"),
        infer_p95=("infer_latency_p95", "mean"),
        sap_mean=("sap_5095", "mean"),
        skip_mean=("frame_skip_pct", "mean"),
    ).reset_index()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.3))
    ax1.plot(agg["n_streams"], agg["infer_mean"], "-o", label="mean", color="#c4a236")
    ax1.plot(agg["n_streams"], agg["infer_p95"], "--^", label="p95", color="#c4a236", alpha=0.6)
    ax1.axhline(1000/30, color="red", linestyle="--", linewidth=1, label="33ms budget")
    ax1.set_xlabel("# concurrent NPU streams"); ax1.set_ylabel("infer latency (ms)")
    ax1.set_title("NPU infer latency vs concurrency"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(agg["n_streams"], agg["skip_mean"], "-o", color="#c4a236")
    ax2.set_xlabel("# concurrent NPU streams"); ax2.set_ylabel("frame skip %")
    ax2.set_title("Frame skip"); ax2.grid(alpha=0.3)
    ax3.plot(agg["n_streams"], agg["sap_mean"], "-o", color="#c4a236")
    ax3.set_xlabel("# concurrent NPU streams"); ax3.set_ylabel("per-stream sAP[0.50:0.95] mean")
    ax3.set_title("Per-stream sAP"); ax3.grid(alpha=0.3)
    plt.suptitle(f"Step D1b — Mobilint NPU capacity (concurrent YOLO11s, infer_mode='{INFER_MODE}')",
                 fontsize=12)
    plt.tight_layout()
    out_fig = FIG / "step_d1b_npu_capacity.png"
    fig.savefig(out_fig, dpi=120)
    print(f"[d1b] figure: {out_fig}")

    # Capacity threshold
    print("\n=== NPU capacity threshold ===")
    over = agg[agg.skip_mean > 5.0]
    if len(over):
        n_thr = int(over.iloc[0]["n_streams"])
        print(f"  first N with avg frame skip > 5%: N={n_thr}  (skip={over.iloc[0]['skip_mean']:.1f}%)")
    else:
        print(f"  no significant frame skip up to N={N_STREAMS_LIST[-1]}")
    print(f"  per-stream infer latency: N=1 {agg.iloc[0]['infer_mean']:.1f}ms  →  N={N_STREAMS_LIST[-1]} {agg.iloc[-1]['infer_mean']:.1f}ms  "
          f"(scaling factor {agg.iloc[-1]['infer_mean']/agg.iloc[0]['infer_mean']:.2f}x)")


if __name__ == "__main__":
    main()
