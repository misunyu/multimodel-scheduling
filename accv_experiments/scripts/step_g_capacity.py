"""Step G — sAP-constrained capacity sweep.

For each stream count N in {2,3,4,5,6,8} and each bg level in {L1, L2}, measure
worst-stream sAP under four placement strategies (Naive_allGPU, SizeAware,
SizeBlindRev, Oracle). Stream composition keeps a controlled mix of small-rich
and large-rich logs.

CSV: results/step_g_capacity.csv (append-on-the-fly so partial results survive
interruption).
"""

from __future__ import annotations

import csv
import json
import random
import sys
import threading
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, fg_worker, get_npu_model,
                            load_split_for_sid, load_val, per_stream_sap,
                            preload_background_models, preload_npu_instances,
                            start_background, stop_background)
from step_f_partA_matrix import per_stream_map_offline

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_g_capacity.csv"
FIG_CURVE = SCRIPT_DIR.parent / "results" / "figures" / "step_g_capacity_curve.png"
FIG_ADV = SCRIPT_DIR.parent / "results" / "figures" / "step_g_advantage.png"

NPU_CAP = 4
N_VALUES = [2, 3, 4, 5, 6, 8]
BG_LEVELS = ["L1", "L2"]
ORACLE_MAX_PER_K = 12  # cap for large N (k = # NPU streams)

# Stream compositions: small_rich sids + large_rich sids per N
COMP = {
    2: ([2],            [3]),
    3: ([2, 22],        [3]),
    4: ([2, 22],        [3, 21]),
    5: ([2, 22, 13],    [3, 21]),
    6: ([2, 22, 13],    [3, 21, 14]),
    8: ([2, 22, 13, 16],[3, 21, 14, 4]),
}


def get_stream_sids(N):
    small, large = COMP[N]
    return small + large


def make_naive(N):
    return ["GPU"] * N


def make_size_aware(N):
    """Small-rich → NPU first (Part A reversal), then GPU. NPU cap = 4."""
    sids = get_stream_sids(N)
    small_set = set(COMP[N][0])
    placement, n_npu = [], 0
    for sid in sids:
        if sid in small_set and n_npu < NPU_CAP:
            placement.append("NPU"); n_npu += 1
        else:
            placement.append("GPU")
    return placement


def make_size_blind_rev(N):
    """Large-rich → NPU first (single-stream intuition, expected to be worse)."""
    sids = get_stream_sids(N)
    large_set = set(COMP[N][1])
    placement, n_npu = [], 0
    for sid in sids:
        if sid in large_set and n_npu < NPU_CAP:
            placement.append("NPU"); n_npu += 1
        else:
            placement.append("GPU")
    return placement


def oracle_candidates(N, max_per_k=ORACLE_MAX_PER_K, seed=42):
    """Enumerate placements with NPU count k=0..NPU_CAP; sample down per-k when
    C(N,k) exceeds max_per_k. Total candidates printed per N for the report."""
    rng = random.Random(seed)
    cands, sampled_info = [], []
    positions = list(range(N))
    for k in range(NPU_CAP + 1):
        all_combos = list(combinations(positions, k))
        if len(all_combos) > max_per_k:
            chosen = rng.sample(all_combos, max_per_k)
            sampled_info.append(f"k={k}: {len(chosen)}/{len(all_combos)}")
        else:
            chosen = all_combos
            sampled_info.append(f"k={k}: {len(chosen)}/{len(all_combos)}")
        for npu_pos in chosen:
            s = set(npu_pos)
            cands.append(["NPU" if i in s else "GPU" for i in range(N)])
    return cands, sampled_info


def measure(placement, sids, splits, bg_level, gpu_models, npu_models):
    n = len(placement)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_background(bg_level)
    gpu_i = npu_i = 0
    threads = []
    for i, dev in enumerate(placement):
        if dev == "GPU":
            model = gpu_models[gpu_i]; gpu_i += 1
        elif dev == "NPU":
            model = npu_models[npu_i]; npu_i += 1
        else:
            raise ValueError(f"unknown device {dev}")
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, splits[i], model, results[i], stop),
                             daemon=True, name=f"g_n{n}_s{i}_{dev}")
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
            "sap_5095": sap["sap_5095"],
            "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
            "map_5095": mp["map_5095"],
            "latency_mean": sap["infer_mean_ms"],
            "frame_skip_pct": sap["frame_skip_pct"],
        })
    return per_stream, wall


def aggregate(per_stream):
    saps = [s["sap_5095"] for s in per_stream]
    maps = [s["map_5095"] for s in per_stream]
    skips = [s["frame_skip_pct"] for s in per_stream]
    lats = [s["latency_mean"] for s in per_stream]
    return {
        "mean_sap": float(np.mean(saps)), "worst_sap": float(np.min(saps)),
        "sap_var": float(np.var(saps)),
        "mean_map": float(np.mean(maps)), "worst_map": float(np.min(maps)),
        "mean_latency": float(np.mean(lats)), "frame_skip_total": float(np.sum(skips)),
        "n_active_streams": int(sum(1 for s in skips if s < 50)),
    }


def append_row(row, cols):
    new_file = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        # only write registered columns; missing keys become ""
        w.writerow({k: row.get(k, "") for k in cols})


def main():
    val = load_val()

    # Pre-build column list (max N = 8 → s0..s7 fields)
    base_cols = ["bg_level", "n_streams", "placement_name", "placement_spec",
                 "mean_sap", "worst_sap", "sap_var",
                 "mean_map", "worst_map",
                 "mean_latency", "frame_skip_total", "n_active_streams",
                 "wall_sec"]
    per_stream_cols = []
    for i in range(max(N_VALUES)):
        for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "map", "lat", "skip"]:
            per_stream_cols.append(f"s{i}_{k}")
    COLS = base_cols + per_stream_cols

    # Clear previous CSV (start fresh)
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    # Preload models — once
    print("[stepG] preloading bg L1+L2 (ORT first then torch)…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[stepG] bg preload {time.time()-t0:.1f}s")

    N_MAX = max(N_VALUES)
    print(f"[stepG] preloading {N_MAX} GPU YOLO instances…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(N_MAX)]
    print(f"[stepG] gpu preload {time.time()-t0:.1f}s")

    print(f"[stepG] preloading {NPU_CAP} NPU YOLO instances (infer_mode='multi')…")
    t0 = time.time()
    preload_npu_instances(NPU_CAP, infer_mode="multi")
    npu_models = [get_npu_model(i) for i in range(NPU_CAP)]
    print(f"[stepG] npu preload {time.time()-t0:.1f}s")

    # Strategy spec build per N, with dedup against oracle
    plan = []  # list of (bg, N, placement_name, placement)
    oracle_info_log = {}
    for bg in BG_LEVELS:
        for N in N_VALUES:
            named = {
                "Naive_allGPU":  make_naive(N),
                "SizeAware":     make_size_aware(N),
                "SizeBlindRev":  make_size_blind_rev(N),
            }
            oracle_pool, sampled_info = oracle_candidates(N)
            if N not in oracle_info_log:
                oracle_info_log[N] = (len(oracle_pool), sampled_info)
            uniq = {}
            def key(p): return ",".join(p)
            for name, plc in named.items():
                uniq[key(plc)] = name
            for plc in oracle_pool:
                k = key(plc)
                if k in uniq:
                    if "Oracle" not in uniq[k]:
                        uniq[k] = uniq[k] + "+Oracle"
                else:
                    uniq[k] = "Oracle"
            for k_str, name in uniq.items():
                plc = k_str.split(",")
                plan.append((bg, N, name, plc))

    print(f"\n[stepG] oracle cell counts per N:")
    for N, (total, info) in oracle_info_log.items():
        print(f"  N={N}: oracle={total} (per-k: {info})")
    print(f"[stepG] total measurements: {len(plan)} (across {len(BG_LEVELS)} bg × {len(N_VALUES)} N)")

    # Run plan
    t_all = time.time()
    for idx, (bg, N, name, placement) in enumerate(plan, 1):
        sids = get_stream_sids(N)
        splits = [load_split_for_sid(val, s) for s in sids]
        print(f"\n[G {idx:3d}/{len(plan)}] bg={bg}  N={N}  name={name[:30]:<30s}  plc={placement}")
        try:
            per_stream, wall = measure(placement, sids, splits, bg,
                                       gpu_models, npu_models)
            agg = aggregate(per_stream)
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}")
            continue
        row = {
            "bg_level": bg, "n_streams": N, "placement_name": name,
            "placement_spec": json.dumps(placement),
            "mean_sap": round(agg["mean_sap"], 4),
            "worst_sap": round(agg["worst_sap"], 4),
            "sap_var": round(agg["sap_var"], 6),
            "mean_map": round(agg["mean_map"], 4),
            "worst_map": round(agg["worst_map"], 4),
            "mean_latency": round(agg["mean_latency"], 2),
            "frame_skip_total": round(agg["frame_skip_total"], 1),
            "n_active_streams": agg["n_active_streams"],
            "wall_sec": round(wall, 1),
        }
        for s in per_stream:
            i = s["stream_id"]
            row[f"s{i}_dev"] = s["device"]
            row[f"s{i}_sid"] = s["sid"]
            row[f"s{i}_sap"] = round(s["sap_5095"], 4)
            row[f"s{i}_sap_s"] = round(s["sap_s"], 4)
            row[f"s{i}_sap_m"] = round(s["sap_m"], 4)
            row[f"s{i}_sap_l"] = round(s["sap_l"], 4)
            row[f"s{i}_map"] = round(s["map_5095"], 4)
            row[f"s{i}_lat"] = round(s["latency_mean"], 2)
            row[f"s{i}_skip"] = round(s["frame_skip_pct"], 1)
        append_row(row, COLS)
        print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
              f"n_active={agg['n_active_streams']}/{N}  ({wall:.1f}s)")

    print(f"\n[stepG] all done. wall_total={time.time()-t_all:.1f}s  csv={OUT_CSV}")

    # ---- Analysis & figures ----
    df = pd.read_csv(OUT_CSV)

    # Per-(bg, N, strategy) "best" row: for named ones it's unique; for Oracle pick max worst_sap
    rows = []
    for bg in BG_LEVELS:
        for N in N_VALUES:
            for name in ["Naive_allGPU", "SizeAware", "SizeBlindRev"]:
                sub = df[(df.bg_level == bg) & (df.n_streams == N) &
                         (df.placement_name.str.startswith(name))]
                if len(sub):
                    rows.append({"bg": bg, "N": N, "strategy": name,
                                 "worst_sap": float(sub.iloc[0]["worst_sap"]),
                                 "mean_sap": float(sub.iloc[0]["mean_sap"]),
                                 "spec": sub.iloc[0]["placement_spec"]})
            sub_o = df[(df.bg_level == bg) & (df.n_streams == N)]
            if len(sub_o):
                best = sub_o.sort_values("worst_sap", ascending=False).iloc[0]
                rows.append({"bg": bg, "N": N, "strategy": "Oracle",
                             "worst_sap": float(best["worst_sap"]),
                             "mean_sap": float(best["mean_sap"]),
                             "spec": best["placement_spec"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV.parent / "step_g_summary.csv", index=False)

    print("\n=== summary (worst-stream sAP) per (bg, N, strategy) ===")
    print(summary.pivot_table(index=["bg", "N"], columns="strategy",
                              values="worst_sap").round(3).to_string())

    # sAP-threshold table: max N achieving worst_sAP ≥ threshold
    print("\n=== max N achieving worst-sAP ≥ threshold ===")
    THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]
    threshold_rows = []
    for bg in BG_LEVELS:
        for strat in ["Naive_allGPU", "SizeAware", "SizeBlindRev", "Oracle"]:
            row = {"bg": bg, "strategy": strat}
            for th in THRESHOLDS:
                sub = summary[(summary.bg == bg) & (summary.strategy == strat) &
                              (summary.worst_sap >= th)]
                row[f"th_{th:.2f}"] = int(sub["N"].max()) if len(sub) else 0
            threshold_rows.append(row)
            print(f"  {bg}  {strat:<14s}  " +
                  "  ".join([f"≥{th:.2f}: N={row[f'th_{th:.2f}']}" for th in THRESHOLDS]))
    pd.DataFrame(threshold_rows).to_csv(OUT_CSV.parent / "step_g_thresholds.csv", index=False)

    # Figure 1: capacity curve
    FIG_CURVE.parent.mkdir(parents=True, exist_ok=True)
    n_bg = len(BG_LEVELS)
    fig, axes = plt.subplots(1, n_bg, figsize=(7 * n_bg, 5), squeeze=False)
    colors = {"Naive_allGPU": "#888888", "SizeAware": "#3680c4",
              "SizeBlindRev": "#c43b3b", "Oracle": "#c4a236"}
    markers = {"Naive_allGPU": "o", "SizeAware": "s",
               "SizeBlindRev": "^", "Oracle": "*"}
    for i, bg in enumerate(BG_LEVELS):
        ax = axes[0][i]
        for strat in ["Naive_allGPU", "SizeAware", "SizeBlindRev", "Oracle"]:
            sub = summary[(summary.bg == bg) & (summary.strategy == strat)].sort_values("N")
            ax.plot(sub["N"], sub["worst_sap"], "-", marker=markers[strat],
                    color=colors[strat], markersize=11 if strat == "Oracle" else 8,
                    label=strat, linewidth=2)
        for th in [0.05, 0.10]:
            ax.axhline(th, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.text(N_VALUES[-1] + 0.05, th, f"sAP={th:.2f}", fontsize=7,
                    color="gray", verticalalignment="center")
        ax.set_xlabel("# foreground streams (N)")
        ax.set_ylabel("worst-stream sAP")
        ax.set_title(f"bg = {bg}")
        ax.set_xticks(N_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    plt.suptitle("Step G — worst-stream sAP vs N under different placements", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_CURVE, dpi=120)
    print(f"saved {FIG_CURVE}")

    # Figure 2: SizeAware advantage (vs Naive) per N, per bg
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(N_VALUES))
    for i, bg in enumerate(BG_LEVELS):
        adv = []
        for N in N_VALUES:
            naive = summary[(summary.bg == bg) & (summary.N == N) &
                            (summary.strategy == "Naive_allGPU")]
            sa = summary[(summary.bg == bg) & (summary.N == N) &
                         (summary.strategy == "SizeAware")]
            if len(naive) and len(sa):
                adv.append(float(sa.iloc[0]["worst_sap"] - naive.iloc[0]["worst_sap"]))
            else:
                adv.append(0)
        bars = ax.bar(x + (i - 0.5) * width, adv, width, label=f"bg = {bg}")
        for rect, v in zip(bars, adv):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.001 if v >= 0 else v - 0.005,
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(N_VALUES)
    ax.set_xlabel("# foreground streams (N)")
    ax.set_ylabel("SizeAware $-$ Naive worst-stream sAP")
    ax.set_title("SizeAware advantage over Naive  vs  N")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_ADV, dpi=120)
    print(f"saved {FIG_ADV}")


if __name__ == "__main__":
    main()
