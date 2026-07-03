"""Step H2 — Robustness, bg-schedule, and loss-decomposition for AllNPU.

All measurements use NPU infer_mode="single" (Step H confirmed N=8 capacity).

Part A — Robustness of AllNPU at N=8 bg L1 across 3 different 8-log
compositions (verifies the Step G2 result is not an artifact of one specific
log mix).

Part B — Schedule shape across 5 background variants (no-bg, light CNN,
heavy CNN×3, CNN+LM, CNN+VLM) at N=4 and N=8 on Composition A.

Part C — Loss decomposition for AllNPU N=8 bg L1:
  pure quantization (N=1 NPU bg L0) → + bg contention (N=1 NPU bg L1)
  → + concurrent contention (N=8 AllNPU bg L0)
  → + interaction (N=8 AllNPU bg L1, reused from Step G2)

CSVs:
  results/step_h2_robustness.csv      (Part A)
  results/step_h2_bg_ablation.csv     (Part B)
  results/step_h2_decomposition.csv   (Part C)
Figures (PDF):
  results/figures/step_h2_robustness.pdf
  results/figures/step_h2_bg_schedule.pdf
  results/figures/step_h2_decomposition.pdf
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
import _step_d_common as cm
from _step_d_common import (FGModelGPU, fg_worker, get_npu_model,
                            load_split_for_sid, load_val, per_stream_sap,
                            preload_background_models, preload_npu_instances,
                            stop_background, _bg_resnet50_loop,
                            _bg_tinyllama_loop, _bg_qwen2vl_loop)
from step_f_partA_matrix import per_stream_map_offline

RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
OUT_A = RESULTS_DIR / "step_h2_robustness.csv"
OUT_B = RESULTS_DIR / "step_h2_bg_ablation.csv"
OUT_C = RESULTS_DIR / "step_h2_decomposition.csv"
FIG_A = FIG_DIR / "step_h2_robustness.pdf"
FIG_B = FIG_DIR / "step_h2_bg_schedule.pdf"
FIG_C = FIG_DIR / "step_h2_decomposition.pdf"
STEP_G2_CSV = RESULTS_DIR / "step_g2_single_mode.csv"

INFER_MODE = "single"
NPU_CAP = 8

# -------- compositions ------------------------------------------------------
# Composition A: matches Step G2 (verification baseline)
#   4 small-dominant (Part A NPU-friendly) + 4 large-dominant
COMP_A = [2, 22, 13, 16, 3, 21, 14, 4]
# Composition B: 4 medium-mixed + 2 small + 2 large
COMP_B = [17, 8, 10, 11, 2, 13, 3, 14]
# Composition C: 3 small + 3 medium + 2 large (size-diverse)
COMP_C = [15, 19, 23, 0, 1, 20, 5, 7]

COMPOSITIONS = {
    "A_baseline":     COMP_A,
    "B_medium_mixed": COMP_B,
    "C_diverse":      COMP_C,
}

# NPU-friendly priority within each composition (top→bottom).
# Derived from step_e pct_small_count (proxy for NPU friendliness under
# contention — Part A finding: small-rich benefits from NPU offload).
NPU_PRIORITY = {
    "A_baseline":     [2, 13, 22, 16, 21, 14, 4, 3],   # small first, then NPU-friendlier large (21 has highest pct_small among large)
    "B_medium_mixed": [2, 13, 8, 17, 10, 11, 14, 3],   # by pct_small_count desc
    "C_diverse":      [0, 15, 23, 19, 1, 20, 5, 7],    # by pct_small_count desc
}

# -------- background variants -----------------------------------------------
BG_VARIANTS = {
    "L0":       [],
    "L1_light": [_bg_resnet50_loop],
    "L1_heavy": [_bg_resnet50_loop, _bg_resnet50_loop, _bg_resnet50_loop],
    "L2_lm":    [_bg_resnet50_loop, _bg_tinyllama_loop],
    "L3_vlm":   [_bg_resnet50_loop, _bg_qwen2vl_loop],
}


def start_bg_custom(level):
    if level == "L0" or not BG_VARIANTS.get(level):
        return [], []
    stops, threads = [], []
    for fn in BG_VARIANTS[level]:
        ev = threading.Event()
        t = threading.Thread(target=fn, args=(ev,), daemon=True,
                             name=f"bg_{fn.__name__}")
        stops.append(ev); threads.append(t); t.start()
    time.sleep(0.3)
    return stops, threads


# -------- placement builders ------------------------------------------------

def make_naive(N): return ["GPU"] * N
def make_all_npu(N): return ["NPU"] * N

def make_size_aware_top_k(comp_name, sids_ordered, k_npu):
    """Top k_npu NPU-friendly sids (per NPU_PRIORITY) → NPU; rest → GPU.
    Returns placement matching sids_ordered's positional order."""
    priority = NPU_PRIORITY[comp_name]
    npu_set = set(priority[:k_npu])
    return ["NPU" if sid in npu_set else "GPU" for sid in sids_ordered]

def make_size_blind_rev(comp_name, sids_ordered, k_npu):
    """Bottom k_npu (NPU-unfriendly = large-rich) → NPU."""
    priority = NPU_PRIORITY[comp_name]
    npu_set = set(priority[-k_npu:])
    return ["NPU" if sid in npu_set else "GPU" for sid in sids_ordered]


# -------- measurement core --------------------------------------------------

def measure_cell(placement, sids, splits, bg_level, gpu_models, npu_models):
    """Run one (placement × bg) cell and return per-stream metrics + wall."""
    n = len(placement)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg_level)
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
                             daemon=True, name=f"h2_n{n}_s{i}_{dev}")
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


def append_row(out_path, row, cols):
    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def base_cols_for(n_max):
    base = ["part", "composition", "n_streams", "bg_level",
            "placement_name", "placement_spec",
            "mean_sap", "worst_sap", "sap_var",
            "mean_map", "worst_map",
            "mean_latency", "frame_skip_total", "n_active_streams", "wall_sec"]
    ps = []
    for i in range(n_max):
        for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "map", "lat", "skip"]:
            ps.append(f"s{i}_{k}")
    return base + ps


def row_from_measurement(part, comp, N, bg, name, placement, per_stream, agg, wall):
    row = {
        "part": part, "composition": comp, "n_streams": N, "bg_level": bg,
        "placement_name": name, "placement_spec": json.dumps(placement),
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
    return row


# ============================== PART A ======================================

def run_part_a(val, gpu_models, npu_models):
    """3 compositions × ~5 placements at N=8 bg L1."""
    print("\n" + "="*78)
    print("PART A — AllNPU robustness across log compositions (N=8, bg=L1_light)")
    print("="*78)
    if OUT_A.exists(): OUT_A.unlink()
    COLS = base_cols_for(8)

    bg = "L1_light"
    N = 8
    t_a = time.time()
    rows = []
    for comp_name, sids in COMPOSITIONS.items():
        splits = [load_split_for_sid(val, s) for s in sids]
        # 5 strategic placements
        placements = [
            ("Naive_allGPU",     make_naive(N)),
            ("SizeAware_NPU4",   make_size_aware_top_k(comp_name, sids, 4)),
            ("SizeAware_NPU6",   make_size_aware_top_k(comp_name, sids, 6)),
            ("AllNPU",           make_all_npu(N)),
            ("SizeBlindRev_NPU4",make_size_blind_rev(comp_name, sids, 4)),
        ]
        # Also a small "mini-oracle" — sample a few extra placements
        rng = random.Random(hash(comp_name) & 0xFFFFFFFF)
        extra = set()
        for k in [3, 5, 7]:
            for _ in range(2):
                idxs = rng.sample(range(N), k)
                spec = tuple("NPU" if i in idxs else "GPU" for i in range(N))
                extra.add(spec)
        for spec in extra:
            placements.append(("MiniOracle", list(spec)))

        # dedup by spec
        seen = {}
        for name, plc in placements:
            key = tuple(plc)
            if key not in seen:
                seen[key] = name
        for spec_tuple, name in seen.items():
            placement = list(spec_tuple)
            print(f"\n[A] comp={comp_name}  name={name:<20s}  plc={placement}")
            try:
                per_stream, wall = measure_cell(placement, sids, splits, bg, gpu_models, npu_models)
                agg = aggregate(per_stream)
            except Exception as e:
                print(f"   FAIL: {type(e).__name__}: {e}")
                continue
            row = row_from_measurement("A", comp_name, N, bg, name, placement, per_stream, agg, wall)
            rows.append(row)
            append_row(OUT_A, row, COLS)
            print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
                  f"n_active={agg['n_active_streams']}/{N}  ({wall:.1f}s)")
    print(f"\n[Part A] done. wall={time.time()-t_a:.1f}s   rows={len(rows)}")
    return rows


# ============================== PART B ======================================

def run_part_b(val, gpu_models, npu_models):
    """5 bg variants × N∈{4,8} × 4 placements on Composition A."""
    print("\n" + "="*78)
    print("PART B — Schedule shape across 5 bg variants (Composition A, N=4 and N=8)")
    print("="*78)
    if OUT_B.exists(): OUT_B.unlink()
    COLS = base_cols_for(8)

    comp_name = "A_baseline"
    sids_full = COMPOSITIONS[comp_name]
    bg_order = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
    t_b = time.time()
    rows = []
    for N in [4, 8]:
        # For N=4, use first 2 small + first 2 large = same shape as Step G's N=4
        if N == 4:
            sids = [sids_full[i] for i in [0, 1, 4, 5]]  # [2,22,3,21]
        else:
            sids = sids_full
        splits = [load_split_for_sid(val, s) for s in sids]

        # Strategies: Naive, SizeAware (top-N/2 NPU), AllNPU
        # plus SizeAware NPU-heavy for N=8 to track schedule mid-point
        if N == 4:
            strategies = [
                ("Naive_allGPU", make_naive(N)),
                ("SizeAware",    make_size_aware_top_k(comp_name, sids, N // 2)),  # 2 NPU
                ("AllNPU",       make_all_npu(N)),
            ]
        else:  # N=8
            strategies = [
                ("Naive_allGPU",   make_naive(N)),
                ("SizeAware_NPU4", make_size_aware_top_k(comp_name, sids, 4)),
                ("SizeAware_NPU6", make_size_aware_top_k(comp_name, sids, 6)),
                ("AllNPU",         make_all_npu(N)),
            ]
        for bg in bg_order:
            for name, placement in strategies:
                print(f"\n[B] N={N}  bg={bg:<10s}  name={name:<18s}  plc={placement}")
                try:
                    per_stream, wall = measure_cell(placement, sids, splits, bg, gpu_models, npu_models)
                    agg = aggregate(per_stream)
                except Exception as e:
                    print(f"   FAIL: {type(e).__name__}: {e}")
                    continue
                row = row_from_measurement("B", comp_name, N, bg, name, placement, per_stream, agg, wall)
                rows.append(row)
                append_row(OUT_B, row, COLS)
                print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
                      f"n_active={agg['n_active_streams']}/{N}  ({wall:.1f}s)")
    print(f"\n[Part B] done. wall={time.time()-t_b:.1f}s   rows={len(rows)}")
    return rows


# ============================== PART C ======================================

def run_part_c(val, gpu_models, npu_models):
    """Decomposition measurements.
    - N=1 NPU bg L0 on each Comp A sid (×8)
    - N=1 NPU bg L1_light on each Comp A sid (×8)
    - N=8 AllNPU bg L0 on Comp A (×1)
    """
    print("\n" + "="*78)
    print("PART C — Loss decomposition for AllNPU N=8 bg L1")
    print("="*78)
    if OUT_C.exists(): OUT_C.unlink()
    COLS = base_cols_for(8)

    comp_name = "A_baseline"
    sids_full = COMPOSITIONS[comp_name]
    t_c = time.time()
    rows = []

    # Per-sid N=1 NPU measurements at L0 and L1
    for bg in ["L0", "L1_light"]:
        for sid in sids_full:
            placement = ["NPU"]
            splits = [load_split_for_sid(val, sid)]
            print(f"\n[C] N=1  sid={sid}  bg={bg}")
            try:
                per_stream, wall = measure_cell(placement, [sid], splits, bg, gpu_models, npu_models)
                agg = aggregate(per_stream)
            except Exception as e:
                print(f"   FAIL: {type(e).__name__}: {e}")
                continue
            row = row_from_measurement("C", comp_name, 1, bg, "N1_NPU_solo", placement, per_stream, agg, wall)
            rows.append(row)
            append_row(OUT_C, row, COLS)
            print(f"   sap={agg['mean_sap']:.3f}  ({wall:.1f}s)")

    # N=8 AllNPU bg L0 on Composition A
    N = 8
    sids = sids_full
    splits = [load_split_for_sid(val, s) for s in sids]
    placement = ["NPU"] * N
    print(f"\n[C] N=8 AllNPU  bg=L0")
    try:
        per_stream, wall = measure_cell(placement, sids, splits, "L0", gpu_models, npu_models)
        agg = aggregate(per_stream)
        row = row_from_measurement("C", comp_name, N, "L0", "AllNPU_N8_L0", placement, per_stream, agg, wall)
        rows.append(row)
        append_row(OUT_C, row, COLS)
        print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  ({wall:.1f}s)")
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")

    print(f"\n[Part C] done. wall={time.time()-t_c:.1f}s   rows={len(rows)}")
    return rows


# ============================== FIGURES =====================================

def fig_part_a(rows):
    """Bar chart: per composition, 4-5 strategies side-by-side, worst sAP."""
    df = pd.DataFrame(rows)
    main_strats = ["Naive_allGPU", "SizeAware_NPU4", "SizeAware_NPU6",
                   "AllNPU", "SizeBlindRev_NPU4"]
    comps = list(COMPOSITIONS.keys())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_w = 0.16
    x = np.arange(len(comps))
    colors = {"Naive_allGPU":"#888888", "SizeAware_NPU4":"#3680c4",
              "SizeAware_NPU6":"#7bbef0", "AllNPU":"#3b9c4d",
              "SizeBlindRev_NPU4":"#c43b3b"}
    for i, strat in enumerate(main_strats):
        vals = []
        for comp in comps:
            sub = df[(df.composition==comp) & (df.placement_name==strat)]
            vals.append(float(sub.iloc[0]["worst_sap"]) if len(sub) else 0)
        bars = ax.bar(x + (i - 2)*bar_w, vals, bar_w,
                      label=strat, color=colors[strat], edgecolor="black", linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x()+rect.get_width()/2, v+0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    # also overlay the mini-oracle MAX per comp
    for j, comp in enumerate(comps):
        sub = df[(df.composition==comp)]
        if len(sub):
            best = float(sub["worst_sap"].max())
            ax.axhline(best, xmin=(j-0.4)/len(comps), xmax=(j+0.4)/len(comps),
                       color="black", linewidth=0)
            ax.scatter([x[j]], [best], marker="*", color="#c4a236", s=120,
                       edgecolor="black", linewidth=0.5, zorder=5,
                       label="Best-of-pool" if j==0 else None)
    ax.set_xticks(x); ax.set_xticklabels(comps)
    ax.set_ylabel("worst-stream sAP")
    ax.set_title("Part A — AllNPU robustness across log compositions (N=8, bg L1_light)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_A, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_A}")


def fig_part_b(rows):
    """Two panels: N=4 and N=8. X = bg level (severity order), Y = worst sAP.
    Lines: Naive, SizeAware, AllNPU (+ SizeAware_NPU6 for N=8)."""
    df = pd.DataFrame(rows)
    bg_order = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"Naive_allGPU":"#888888", "SizeAware":"#3680c4",
              "SizeAware_NPU4":"#3680c4", "SizeAware_NPU6":"#7bbef0",
              "AllNPU":"#3b9c4d"}
    markers = {"Naive_allGPU":"o", "SizeAware":"s", "SizeAware_NPU4":"s",
               "SizeAware_NPU6":"D", "AllNPU":"^"}
    for k, N in enumerate([4, 8]):
        ax = axes[k]
        sub_N = df[df.n_streams==N]
        strats = sorted(sub_N.placement_name.unique())
        for strat in strats:
            ys = []
            for bg in bg_order:
                r = sub_N[(sub_N.bg_level==bg) & (sub_N.placement_name==strat)]
                ys.append(float(r.iloc[0]["worst_sap"]) if len(r) else np.nan)
            ax.plot(bg_order, ys, "-", marker=markers.get(strat,"o"),
                    color=colors.get(strat,"#444"), linewidth=2, markersize=9,
                    label=strat)
        # Annotate optimal strategy per bg
        for j, bg in enumerate(bg_order):
            sub_b = sub_N[sub_N.bg_level==bg]
            if len(sub_b):
                best_row = sub_b.sort_values("worst_sap", ascending=False).iloc[0]
                ax.annotate(f"opt:\n{best_row['placement_name']}",
                            xy=(j, float(best_row["worst_sap"])),
                            xytext=(0, 14), textcoords="offset points",
                            fontsize=6, ha="center", color="#444",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#fffae6",
                                      ec="#cc9900", lw=0.5))
        ax.set_xlabel("background contention level")
        if k == 0:
            ax.set_ylabel("worst-stream sAP")
        ax.set_title(f"N = {N}  (Composition A)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    plt.suptitle("Part B — Schedule shape across background contention",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_B, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_B}")


def fig_part_c(rows_c, allnpu_l1_mean):
    """Stacked bar showing loss attribution.
    Reference is GPU N=1 bg L0 sAP (~ 0.265 from Step F Part A, computed per
    composition A). The four conditions step down through quantization, bg,
    concurrent, interaction."""
    df = pd.DataFrame(rows_c)

    # Compute means
    n1_l0 = df[(df.n_streams==1) & (df.bg_level=="L0")]["mean_sap"].mean()
    n1_l1 = df[(df.n_streams==1) & (df.bg_level=="L1_light")]["mean_sap"].mean()
    n8_l0 = df[(df.n_streams==8) & (df.bg_level=="L0")]["mean_sap"].iloc[0]
    n8_l1 = allnpu_l1_mean

    # GPU N=1 bg L0 reference: Part A averaged over comp A sids that exist there
    df_pa = pd.read_csv(RESULTS_DIR / "step_f_partA_matrix.csv")
    comp_a_sids = COMPOSITIONS["A_baseline"]
    gpu_baseline_rows = df_pa[(df_pa.device=="GPU") & (df_pa.sid.isin(comp_a_sids))]
    gpu_baseline = float(gpu_baseline_rows["sap_5095"].mean()) if len(gpu_baseline_rows) else np.nan

    deltas = {
        "Quantization":      gpu_baseline - n1_l0,
        "Bg contention":     n1_l0 - n1_l1,
        "Concurrent ×N=8":   n1_l1 - n8_l0,
        "Interaction":       n8_l0 - n8_l1,
    }
    print("\n[Part C] mean-sAP decomposition:")
    print(f"  GPU N=1 bg L0 (ref):     {gpu_baseline:.4f}")
    print(f"  NPU N=1 bg L0:           {n1_l0:.4f}   (Δquant = {deltas['Quantization']:.4f})")
    print(f"  NPU N=1 bg L1_light:     {n1_l1:.4f}   (Δbg = {deltas['Bg contention']:.4f})")
    print(f"  NPU N=8 bg L0:           {n8_l0:.4f}   (Δconcurrent = {deltas['Concurrent ×N=8']:.4f})")
    print(f"  NPU N=8 bg L1 (Step G2): {n8_l1:.4f}   (Δinteraction = {deltas['Interaction']:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: stacked bar of cumulative losses
    ax = axes[0]
    labels = ["GPU N=1 L0\n(reference)",
              "NPU N=1 L0\n(+ quant)",
              "NPU N=1 L1_light\n(+ bg)",
              "NPU N=8 L0\n(+ concurrent)",
              "NPU N=8 L1_light\n(+ interaction)"]
    values = [gpu_baseline, n1_l0, n1_l1, n8_l0, n8_l1]
    bars = ax.bar(labels, values, color=["#3680c4","#7bbef0","#a6c8e8","#e8a87b","#c43b3b"])
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean sAP (8-stream avg, Composition A)")
    ax.set_title("Cumulative loss progression")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)

    # Panel 2: loss contribution bars
    ax = axes[1]
    contrib_names = list(deltas.keys())
    contrib_vals = list(deltas.values())
    colors_d = ["#7bbef0","#a6c8e8","#e8a87b","#c43b3b"]
    bars = ax.barh(contrib_names, contrib_vals, color=colors_d, edgecolor="black", linewidth=0.5)
    total = gpu_baseline - n8_l1
    for b, v in zip(bars, contrib_vals):
        ax.text(v + 0.001, b.get_y()+b.get_height()/2,
                f"{v:.3f}  ({100*v/total:+.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel(f"sAP lost (total = {total:.3f} from GPU N=1 L0 → NPU N=8 L1)")
    ax.set_title("Per-component loss attribution")
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Part C — Loss decomposition for AllNPU at N=8 bg L1 (Composition A)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_C, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_C}")


# ============================== MAIN ========================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    val = load_val()

    print("[stepH2] preloading bg models max=L3 (ORT first, torch after)")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[stepH2] bg preload {time.time()-t0:.1f}s")

    print(f"[stepH2] preloading {NPU_CAP} GPU + {NPU_CAP} NPU (single mode)…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(NPU_CAP)]
    preload_npu_instances(NPU_CAP, infer_mode=INFER_MODE)
    npu_models = [get_npu_model(i) for i in range(NPU_CAP)]
    print(f"[stepH2] gpu+npu preload {time.time()-t0:.1f}s")

    t_total = time.time()

    rows_a = run_part_a(val, gpu_models, npu_models)
    rows_b = run_part_b(val, gpu_models, npu_models)
    rows_c = run_part_c(val, gpu_models, npu_models)

    print(f"\n[stepH2] all measurements done. wall={time.time()-t_total:.1f}s")

    # Get N=8 AllNPU bg L1 mean sAP from Step G2 for Part C
    df_g2 = pd.read_csv(STEP_G2_CSV)
    allnpu_l1 = df_g2[(df_g2.n_streams==8) &
                      (df_g2.placement_spec=='["NPU", "NPU", "NPU", "NPU", "NPU", "NPU", "NPU", "NPU"]')]
    allnpu_l1_mean = float(allnpu_l1.iloc[0]["mean_sap"]) if len(allnpu_l1) else float("nan")
    print(f"\n[stepH2] N=8 AllNPU bg L1 from Step G2: mean_sap={allnpu_l1_mean:.4f}")

    # Generate figures
    print("\n[stepH2] generating figures…")
    fig_part_a(rows_a)
    fig_part_b(rows_b)
    fig_part_c(rows_c, allnpu_l1_mean)

    # Summaries
    print("\n" + "="*78)
    print("=== PART A SUMMARY: worst sAP per (composition, strategy) ===")
    print("="*78)
    df_a = pd.read_csv(OUT_A)
    pivot_a = df_a.pivot_table(index="composition", columns="placement_name",
                                values="worst_sap").round(3)
    print(pivot_a.to_string())

    print("\n" + "="*78)
    print("=== PART B SUMMARY: worst sAP per (bg, strategy) ===")
    print("="*78)
    df_b = pd.read_csv(OUT_B)
    for N in [4, 8]:
        print(f"\nN = {N}:")
        sub = df_b[df_b.n_streams==N]
        pivot_b = sub.pivot_table(index="bg_level", columns="placement_name",
                                   values="worst_sap").round(3)
        print(pivot_b.to_string())

    print("\n" + "="*78)
    print("=== PART C SUMMARY ===")
    print("="*78)
    df_c = pd.read_csv(OUT_C)
    print("\nPer-sid N=1 NPU sAP:")
    sub = df_c[df_c.n_streams==1].pivot_table(index="s0_sid", columns="bg_level",
                                                values="mean_sap").round(3)
    print(sub.to_string())
    print("\nN=8 AllNPU bg L0 (new):")
    sub = df_c[df_c.n_streams==8]
    if len(sub):
        print(f"  mean_sap={float(sub.iloc[0]['mean_sap']):.3f}  worst_sap={float(sub.iloc[0]['worst_sap']):.3f}  "
              f"n_active={int(sub.iloc[0]['n_active_streams'])}/8")


if __name__ == "__main__":
    main()
