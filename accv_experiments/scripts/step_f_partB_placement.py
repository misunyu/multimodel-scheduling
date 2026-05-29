"""Step F Part B — 4-stream placement strategy comparison.

Reads strategy definitions from results/step_f_partA_strategy.json
(produced automatically by Part A's NPU-GPU difference matrix), then runs
each strategy + an Oracle sweep under both background L1 and L2.

Oracle = all 2^4 = 16 GPU/NPU placements. CPU is excluded from the oracle
sweep because Part A showed CPU is dominated (>98% frame skip under L1, sAP
0.07-0.20) and cannot improve worst-stream sAP. This decision is logged.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, FPS, WARMUP_FRAMES, fg_worker,
                            get_npu_model, load_split_for_sid, load_val,
                            per_stream_sap, preload_background_models,
                            preload_npu_instances, start_background,
                            stop_background)
from step_f_partA_matrix import per_stream_map_offline  # reuse offline mAP

BG_LEVELS = ["L1", "L2"]
STRATEGY_JSON = SCRIPT_DIR.parent / "results" / "step_f_partA_strategy.json"
OUT_CSV = SCRIPT_DIR.parent / "results" / "step_f_partB_placement.csv"
FIG_WORST = SCRIPT_DIR.parent / "results" / "figures" / "step_f_partB_worst_sap.png"
FIG_SAPMAP = SCRIPT_DIR.parent / "results" / "figures" / "step_f_partB_sap_vs_map.png"


def measure_placement(placement, sids, splits, bg_level, gpu_models, npu_models):
    """Run one (placement, bg_level) experiment. Returns per-stream metrics."""
    n = len(placement)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_background(bg_level)

    gpu_i = 0; npu_i = 0
    threads = []
    for i, dev in enumerate(placement):
        if dev == "GPU":
            model = gpu_models[gpu_i]; gpu_i += 1
        elif dev == "NPU":
            model = npu_models[npu_i]; npu_i += 1
        else:
            raise ValueError(f"unsupported device in oracle: {dev}")
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, splits[i], model, results[i], stop),
                             daemon=True, name=f"fg_s{i}_{dev}")
        threads.append(t)

    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)

    per_stream = []
    for i in range(n):
        sap = per_stream_sap(splits[i], results[i])
        mp = per_stream_map_offline(splits[i], results[i])
        per_stream.append({
            "stream_id": i, "sid": sids[i], "device": placement[i],
            "sap_5095": sap["sap_5095"], "sap_50": sap["sap_50"],
            "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
            "map_5095": mp["map_5095"], "map_50": mp["map_50"],
            "map_s": mp["map_s"], "map_m": mp["map_m"], "map_l": mp["map_l"],
            "latency_mean": sap["infer_mean_ms"], "frame_skip_pct": sap["frame_skip_pct"],
        })
    return per_stream, wall


def aggregate(per_stream):
    saps = [s["sap_5095"] for s in per_stream]
    maps = [s["map_5095"] for s in per_stream]
    lats = [s["latency_mean"] for s in per_stream]
    skips = [s["frame_skip_pct"] for s in per_stream]
    return {
        "mean_sap": float(np.mean(saps)), "worst_sap": float(np.min(saps)),
        "sap_var": float(np.var(saps)),
        "mean_map": float(np.mean(maps)), "worst_map": float(np.min(maps)),
        "map_var": float(np.var(maps)),
        "mean_latency": float(np.mean(lats)), "frame_skip_total": float(np.sum(skips)),
    }


def main():
    strat = json.loads(STRATEGY_JSON.read_text())
    PARTB_SIDS = strat["partb_sids"]
    PARTB_LOG_IDS = strat["partb_log_ids"]
    PARTB_GROUPS  = strat["partb_size_groups"]
    named_strategies = strat["strategies"]  # Naive, SizeAware, SizeBlindRev

    print("=== Part B inputs ===")
    print(f"  PARTB sids: {PARTB_SIDS}")
    print(f"  groups:     {PARTB_GROUPS}")
    print(f"  named strategies:")
    for name, plc in named_strategies.items():
        print(f"    {name:14s}: {plc}")

    # Oracle: all 2^4 GPU/NPU combos. CPU excluded (Part A showed CPU is dominated)
    oracle_placements = [list(p) for p in product(["GPU","NPU"], repeat=4)]
    print(f"\n  oracle pool: {len(oracle_placements)} GPU/NPU combos "
          f"(CPU excluded — Part A showed CPU 38ms / >98% skip under L1)")

    # All placements to measure (de-duplicated; named strategies may overlap with oracle)
    all_specs = {}  # spec_str -> {"name": str, "placement": list}
    def spec_str(p): return ",".join(p)
    for name, plc in named_strategies.items():
        if all(d in ("GPU","NPU") for d in plc):  # any CPU? add separately
            all_specs[spec_str(plc)] = {"name": name, "placement": plc}
        else:
            all_specs[spec_str(plc) + f"|{name}"] = {"name": name, "placement": plc}
    for plc in oracle_placements:
        s = spec_str(plc)
        if s in all_specs:
            all_specs[s]["name"] = all_specs[s]["name"] + "+Oracle"
        else:
            all_specs[s] = {"name": "Oracle", "placement": plc}

    print(f"\n  total unique placements to measure: {len(all_specs)}")

    # Preload
    print("\n[partB] preloading background L3-max + 4 GPU + 4 NPU…")
    t0 = time.time()
    preload_background_models(max_level="L3")  # we'll use L1 and L2
    gpu_models = [FGModelGPU() for _ in range(4)]
    preload_npu_instances(4, infer_mode="multi")
    npu_models = [get_npu_model(i) for i in range(4)]
    print(f"[partB] preload {time.time()-t0:.1f}s")

    # Load splits once
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PARTB_SIDS]

    # Run
    rows = []
    n_total = len(all_specs) * len(BG_LEVELS)
    cell = 0
    t_all = time.time()
    for bg in BG_LEVELS:
        for s_key, item in all_specs.items():
            cell += 1
            placement = item["placement"]
            name = item["name"]
            print(f"\n[B {cell:3d}/{n_total}] bg={bg}  name={name:30s}  placement={placement}")
            per_stream, wall = measure_placement(placement, PARTB_SIDS, splits, bg,
                                                 gpu_models, npu_models)
            agg = aggregate(per_stream)
            row = {
                "bg_level": bg, "placement_name": name,
                "placement_spec": json.dumps(placement),
                "mean_sap": round(agg["mean_sap"], 4),
                "worst_sap": round(agg["worst_sap"], 4),
                "sap_var": round(agg["sap_var"], 6),
                "mean_map": round(agg["mean_map"], 4),
                "worst_map": round(agg["worst_map"], 4),
                "map_var": round(agg["map_var"], 6),
                "mean_latency": round(agg["mean_latency"], 2),
                "frame_skip_total": round(agg["frame_skip_total"], 1),
                "wall_sec": round(wall, 1),
            }
            for s in per_stream:
                row[f"s{s['stream_id']}_dev"]   = s["device"]
                row[f"s{s['stream_id']}_sap"]   = round(s["sap_5095"], 4)
                row[f"s{s['stream_id']}_map"]   = round(s["map_5095"], 4)
                row[f"s{s['stream_id']}_sap_s"] = round(s["sap_s"], 4)
                row[f"s{s['stream_id']}_sap_m"] = round(s["sap_m"], 4)
                row[f"s{s['stream_id']}_sap_l"] = round(s["sap_l"], 4)
                row[f"s{s['stream_id']}_lat"]   = round(s["latency_mean"], 2)
                row[f"s{s['stream_id']}_skip"]  = round(s["frame_skip_pct"], 1)
            rows.append(row)
            print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
                  f"mean_mAP={agg['mean_map']:.3f}  worst_mAP={agg['worst_map']:.3f}  "
                  f"({wall:.1f}s)")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[partB] csv: {OUT_CSV}  ({len(df)} rows, wall_total={time.time()-t_all:.1f}s)")

    # --- Analysis ---
    print("\n=== summary table per bg ===")
    main_strategies = ["Naive_allGPU", "SizeAware", "SizeBlindRev"]
    for bg in BG_LEVELS:
        sub = df[df.bg_level == bg].copy()
        print(f"\n[{bg}]")
        # named strategies
        for name in main_strategies:
            r = sub[sub.placement_name.str.startswith(name)].iloc[0] if not sub[sub.placement_name.str.startswith(name)].empty else None
            if r is not None:
                print(f"  {name:14s} placement={r['placement_spec']:30s} "
                      f"mean_sAP={r['mean_sap']:.3f}  worst_sAP={r['worst_sap']:.3f}  "
                      f"mean_mAP={r['mean_map']:.3f}  worst_mAP={r['worst_map']:.3f}")
        # Oracle = best worst_sap among ALL placements in this bg
        oracle_row = sub.sort_values("worst_sap", ascending=False).iloc[0]
        print(f"  {'Oracle (best worst_sAP)':30s} placement={oracle_row['placement_spec']:30s} "
              f"mean_sAP={oracle_row['mean_sap']:.3f}  worst_sAP={oracle_row['worst_sap']:.3f}  "
              f"mean_mAP={oracle_row['mean_map']:.3f}  worst_mAP={oracle_row['worst_map']:.3f}")

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    x = np.arange(len(BG_LEVELS))
    for i, name in enumerate(["Naive_allGPU", "SizeAware", "SizeBlindRev"]):
        worst = []
        for bg in BG_LEVELS:
            sub = df[(df.bg_level == bg) & (df.placement_name.str.startswith(name))]
            worst.append(sub.iloc[0]["worst_sap"] if len(sub) else 0)
        axes[0].bar(x + (i-1)*0.25, worst, 0.22, label=name)
    # Oracle worst sap per bg
    oracle_worst = []
    for bg in BG_LEVELS:
        sub = df[df.bg_level == bg]
        oracle_worst.append(sub["worst_sap"].max())
    axes[0].plot(x, oracle_worst, "k*--", markersize=15, label="Oracle (best worst_sAP)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(BG_LEVELS)
    axes[0].set_ylabel("worst-stream sAP")
    axes[0].set_title("Worst-stream sAP by strategy × bg level")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    # sAP vs mAP scatter per strategy
    colors = {"Naive_allGPU": "#888888", "SizeAware": "#3680c4", "SizeBlindRev": "#c43b3b"}
    for name in ["Naive_allGPU", "SizeAware", "SizeBlindRev"]:
        for bg in BG_LEVELS:
            sub = df[(df.bg_level == bg) & (df.placement_name.str.startswith(name))]
            if not len(sub): continue
            r = sub.iloc[0]
            marker = "o" if bg == "L1" else "s"
            axes[1].scatter(r["worst_map"], r["worst_sap"], color=colors[name], s=140,
                            edgecolors="black", marker=marker, label=f"{name}/{bg}")
    axes[1].plot([0, 0.4], [0, 0.4], "k--", linewidth=0.5)
    axes[1].set_xlabel("worst-stream mAP (offline)")
    axes[1].set_ylabel("worst-stream sAP (streaming)")
    axes[1].set_title("worst-stream sAP vs mAP\n(distance from y=x = latency penalty)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)

    plt.suptitle("Step F Part B — 4-stream placement comparison", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_WORST, dpi=120)
    print(f"saved {FIG_WORST}")

    # Combined sAP vs mAP for ALL placements
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    for bg, m in [("L1", "o"), ("L2", "s")]:
        sub = df[df.bg_level == bg]
        sc = ax.scatter(sub["worst_map"], sub["worst_sap"], s=30, marker=m, alpha=0.4,
                       label=f"all placements ({bg})")
    # Highlight named strategies + oracle
    for bg, m in [("L1", "o"), ("L2", "s")]:
        sub = df[df.bg_level == bg]
        for name, color in [("Naive_allGPU", "#888888"), ("SizeAware", "#3680c4"),
                            ("SizeBlindRev", "#c43b3b")]:
            r = sub[sub.placement_name.str.startswith(name)]
            if len(r):
                r = r.iloc[0]
                ax.scatter(r["worst_map"], r["worst_sap"], color=color, s=180, edgecolors="black",
                          marker=m, label=f"{name} ({bg})", zorder=5)
        oracle = sub.sort_values("worst_sap", ascending=False).iloc[0]
        ax.scatter(oracle["worst_map"], oracle["worst_sap"], color="gold", s=280, edgecolors="black",
                  marker="*", label=f"Oracle ({bg})", zorder=6)
    ax.plot([0, 0.4], [0, 0.4], "k--", linewidth=0.5)
    ax.set_xlabel("worst-stream mAP (offline)")
    ax.set_ylabel("worst-stream sAP (streaming)")
    ax.set_title("All placements: worst-stream sAP vs mAP")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncols=2)
    plt.tight_layout()
    fig2.savefig(FIG_SAPMAP, dpi=120)
    print(f"saved {FIG_SAPMAP}")


if __name__ == "__main__":
    main()
