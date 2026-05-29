"""Step H — Mobilint NPU concurrent stream capacity.

For N = 1..8 and each infer_mode in {multi, single}, attempt to load N
independent YOLO11s engines on the NPU, then run all N as concurrent
foreground streams against the shortest val log (sid=5) and measure per-stream
latency / frame skip / sAP. Background L0 (none) is the primary capacity
measurement; L1 (light CNN co-tenant on GPU) is recorded as a bonus.

Stops a given infer_mode sweep at the first N where loading or measurement
fails. CSV appends per measurement so partial data survives crashes.

CSV:    results/step_h_npu_capacity.csv
Figure: results/figures/step_h_npu_scaling.png
"""

from __future__ import annotations

import csv
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _step_d_common as cm
from _step_d_common import (fg_worker, load_split_for_sid, load_val,
                            per_stream_sap, preload_background_models,
                            preload_npu_instances, start_background,
                            stop_background)

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_h_npu_capacity.csv"
FIG = SCRIPT_DIR.parent / "results" / "figures" / "step_h_npu_scaling.png"

N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
INFER_MODES = ["multi", "single"]
BG_LEVELS = ["L0", "L1"]
TEST_SID = 5

COLS = ["n_streams", "infer_mode", "bg_level", "stream_id",
        "init_success",
        "latency_mean", "latency_p95",
        "frame_skip_pct", "sap_5095",
        "wall_sec", "init_err"]


def reset_npu_cache():
    """Dispose any cached NPU instances and clear the global cache."""
    with cm._NPU_LOAD_LOCK:
        for m in cm._NPU_INSTANCES:
            try:
                m.dispose()
            except Exception:
                pass
        cm._NPU_INSTANCES.clear()


def fresh_load_npu(n, infer_mode):
    """Reset cache and load N fresh NPU YOLO11s engines for the given mode."""
    reset_npu_cache()
    try:
        preload_npu_instances(n, infer_mode=infer_mode)
        return True, cm._NPU_INSTANCES[:n], None
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
        return False, [], err


def run_measurement(N, splits, models, bg):
    results = [defaultdict(list) for _ in range(N)]
    stop = threading.Event()
    bg_stops, bg_threads = start_background(bg)
    threads = []
    for i in range(N):
        t = threading.Thread(target=fg_worker,
                             args=(i, "NPU", splits[i], models[i], results[i], stop),
                             daemon=True, name=f"h_n{N}_s{i}")
        threads.append(t)
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    metrics = []
    for i in range(N):
        m = per_stream_sap(splits[i], results[i])
        metrics.append({
            "stream_id": i,
            "latency_mean": m["infer_mean_ms"],
            "latency_p95": m["infer_p95_ms"],
            "frame_skip_pct": m["frame_skip_pct"],
            "sap_5095": m["sap_5095"],
        })
    return metrics, wall


def append_row(row):
    new_file = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    val = load_val()
    base_split = load_split_for_sid(val, TEST_SID)
    print(f"[stepH] test log sid={TEST_SID}  frames={len(base_split['imgs'])}")

    print("[stepH] preloading bg L0/L1 (resnet only — L0 is no-op)")
    preload_background_models(max_level="L1")

    t_all = time.time()
    for mode in INFER_MODES:
        print(f"\n========== infer_mode = {mode} ==========")
        mode_aborted = False
        for N in N_VALUES:
            if mode_aborted:
                break
            print(f"\n[H mode={mode}] N={N}: loading {N} NPU engines…")
            t0 = time.time()
            ok, models, err = fresh_load_npu(N, mode)
            load_wall = time.time() - t0
            if not ok:
                print(f"   INIT FAIL ({load_wall:.1f}s): {err}")
                append_row({"n_streams": N, "infer_mode": mode, "bg_level": "-",
                            "stream_id": -1, "init_success": False, "init_err": err,
                            "wall_sec": round(load_wall, 1)})
                mode_aborted = True
                break
            print(f"   loaded {N} engines in {load_wall:.1f}s")

            # Per-stream identical log (capacity measurement)
            splits = [base_split for _ in range(N)]
            for bg in BG_LEVELS:
                try:
                    per_stream, wall = run_measurement(N, splits, models, bg)
                except Exception as e:
                    print(f"   RUN FAIL bg={bg}: {type(e).__name__}: {e}")
                    append_row({"n_streams": N, "infer_mode": mode, "bg_level": bg,
                                "stream_id": -1, "init_success": True,
                                "init_err": f"run: {type(e).__name__}: {str(e)[:150]}",
                                "wall_sec": 0})
                    mode_aborted = True
                    break

                avg_inf = sum(s["latency_mean"] for s in per_stream) / N
                avg_p95 = sum(s["latency_p95"] for s in per_stream) / N
                avg_skip = sum(s["frame_skip_pct"] for s in per_stream) / N
                avg_sap = sum(s["sap_5095"] for s in per_stream) / N
                print(f"   bg={bg}  avg_infer={avg_inf:5.1f}ms  p95={avg_p95:5.1f}ms  "
                      f"avg_skip={avg_skip:5.1f}%  avg_sAP={avg_sap:.3f}  ({wall:.1f}s)")
                for s in per_stream:
                    append_row({
                        "n_streams": N, "infer_mode": mode, "bg_level": bg,
                        "stream_id": s["stream_id"], "init_success": True,
                        "latency_mean": round(s["latency_mean"], 2),
                        "latency_p95": round(s["latency_p95"], 2),
                        "frame_skip_pct": round(s["frame_skip_pct"], 2),
                        "sap_5095": round(s["sap_5095"], 4),
                        "wall_sec": round(wall, 1),
                    })

    # Cleanup
    reset_npu_cache()
    print(f"\n[stepH] all done. wall_total={time.time()-t_all:.1f}s  csv={OUT_CSV}")

    # ---- Analysis & figure ----
    df = pd.read_csv(OUT_CSV)
    # Only consider successful per-stream rows
    df_ok = df[df.init_success == True].copy()
    if not len(df_ok):
        print("no successful measurements; skipping analysis.")
        return

    # Per (mode, N, bg) aggregate
    agg = df_ok.groupby(["infer_mode", "bg_level", "n_streams"]).agg(
        latency_mean=("latency_mean", "mean"),
        latency_p95=("latency_p95", "mean"),
        skip_mean=("frame_skip_pct", "mean"),
        sap_mean=("sap_5095", "mean"),
    ).reset_index()
    print("\n=== aggregated per-N stats ===")
    print(agg.round(2).to_string(index=False))

    # Init success table
    print("\n=== init success matrix ===")
    init_tab = df.groupby(["infer_mode", "n_streams"])["init_success"].agg(lambda x: x.iloc[0]).reset_index()
    for mode in INFER_MODES:
        sub = init_tab[init_tab.infer_mode == mode].sort_values("n_streams")
        marks = ["✓" if v else "✗" for v in sub["init_success"]]
        Ns = sub["n_streams"].tolist()
        print(f"  {mode:6s}: " + "  ".join(f"N={n}:{m}" for n, m in zip(Ns, marks)))

    # Capacity threshold (frame skip < 5%) per mode + bg
    print("\n=== NPU capacity threshold (avg frame skip < 5%) ===")
    for mode in INFER_MODES:
        for bg in BG_LEVELS:
            sub = agg[(agg.infer_mode == mode) & (agg.bg_level == bg)].sort_values("n_streams")
            if not len(sub): continue
            over = sub[sub.skip_mean >= 5]
            if len(over):
                last_ok = sub[sub.skip_mean < 5]["n_streams"].max() if len(sub[sub.skip_mean < 5]) else 0
                print(f"  {mode:6s} bg={bg}: capacity N≤{last_ok}  "
                      f"(N={over.iloc[0]['n_streams']} hits {over.iloc[0]['skip_mean']:.1f}% skip)")
            else:
                print(f"  {mode:6s} bg={bg}: no frame skip up to N={sub['n_streams'].max()}")

    # ---- Figure: latency vs N + frame skip ----
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(BG_LEVELS), figsize=(7 * len(BG_LEVELS), 5), squeeze=False)
    colors = {"multi": "#3680c4", "single": "#c4a236"}
    for i, bg in enumerate(BG_LEVELS):
        ax = axes[0][i]
        ax2 = ax.twinx()
        for mode in INFER_MODES:
            sub = agg[(agg.infer_mode == mode) & (agg.bg_level == bg)].sort_values("n_streams")
            if not len(sub): continue
            ax.plot(sub["n_streams"], sub["latency_mean"], "-o", color=colors[mode],
                    label=f"{mode} latency mean", linewidth=2, markersize=8)
            ax.plot(sub["n_streams"], sub["latency_p95"], "--^", color=colors[mode],
                    label=f"{mode} latency p95", alpha=0.6, markersize=7)
            ax2.plot(sub["n_streams"], sub["skip_mean"], ":s", color=colors[mode],
                     label=f"{mode} frame skip", alpha=0.7)
        ax.axhline(1000 / 30, color="red", linestyle="--", linewidth=1, alpha=0.6,
                   label="33ms budget")
        ax.set_xlabel("# concurrent NPU streams (N)")
        ax.set_ylabel("latency (ms)")
        ax2.set_ylabel("frame skip %")
        ax.set_title(f"bg = {bg}")
        ax.set_xticks(N_VALUES)
        ax.grid(alpha=0.3)
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="upper left")
    plt.suptitle("Step H — NPU concurrent capacity (latency + frame skip vs N)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG, dpi=120)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
