"""Step G2 — Step G re-measurement with NPU single mode.

Key changes vs Step G (step_g_capacity.py):
  1. NPU infer_mode = "single" (Step H confirmed N=8 capacity, multi caps at 6)
  2. NPU_CAP raised from 4 to 8.
  3. N ∈ {4, 5, 6, 8} only (N=2,3 are confirmed low-contention regime in Step G).
  4. bg = L1 only (L2 was saturated in Step G).
  5. For N=8, extra SizeAware variants test how much NPU offload helps:
       SizeAware_NPU4: 4 small-rich → NPU (= original Step G SizeAware)
       SizeAware_NPU6: 4 small-rich + 2 large-rich → NPU
       SizeAware_NPU8: all 8 → NPU (single mode's extreme)
  6. Adds AllNPU as a named strategy for N≤6 as well.

Outputs (NEW filenames, original step_g files untouched):
  results/step_g2_single_mode.csv
  results/figures/step_g2_comparison.png         (vs Step G multi)
  results/figures/step_g2_capacity_recheck.png   (curve with single)
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

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_g2_single_mode.csv"
FIG_CMP = SCRIPT_DIR.parent / "results" / "figures" / "step_g2_comparison.png"
FIG_CURVE = SCRIPT_DIR.parent / "results" / "figures" / "step_g2_capacity_recheck.png"
STEP_G_CSV = SCRIPT_DIR.parent / "results" / "step_g_capacity.csv"

INFER_MODE = "single"
NPU_CAP = 8
N_VALUES = [4, 5, 6, 8]
BG_LEVELS = ["L1"]
ORACLE_MAX_PER_K_LOW = 12
ORACLE_MAX_PER_K_HIGH = 8

COMP = {
    4: ([2, 22],         [3, 21]),
    5: ([2, 22, 13],     [3, 21]),
    6: ([2, 22, 13],     [3, 21, 14]),
    8: ([2, 22, 13, 16], [3, 21, 14, 4]),
}


def get_stream_sids(N):
    small, large = COMP[N]
    return small + large


def make_naive(N):
    return ["GPU"] * N


def make_all_npu(N):
    return ["NPU"] * N


def make_size_aware(N):
    """Original SizeAware: small-rich → NPU first (Part A reversal)."""
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
    """Single-stream-intuition: large-rich → NPU."""
    sids = get_stream_sids(N)
    large_set = set(COMP[N][1])
    placement, n_npu = [], 0
    for sid in sids:
        if sid in large_set and n_npu < NPU_CAP:
            placement.append("NPU"); n_npu += 1
        else:
            placement.append("GPU")
    return placement


def make_size_aware_with_target(N, target_npu):
    """SizeAware with explicit NPU target count: small-rich first (Part A order
    within small group: lowest absolute Δ NPU-GPU first), then large-rich in
    Part A's NPU-friendliness order (sid 21 < 3) until target_npu is reached."""
    sids = get_stream_sids(N)
    # Within-group ordering by Part A NPU-friendliness (less loss first)
    # small-rich: sids 2 (-0.015), 13 (-0.017), 22 (-0.022), 16 (unmeasured → place last)
    # large-rich: sids 21 (-0.034), 3 (-0.043), 14 (unmeasured → middle), 4 (unmeasured → last)
    small_priority = [2, 13, 22, 16]
    large_priority = [21, 3, 14, 4]
    npu_order = [s for s in small_priority if s in sids] + \
                [s for s in large_priority if s in sids]
    npu_set = set(npu_order[:target_npu])
    return ["NPU" if sid in npu_set else "GPU" for sid in sids]


def oracle_candidates(N, seed=42):
    rng = random.Random(seed)
    cands, info = [], []
    positions = list(range(N))
    for k in range(min(N, NPU_CAP) + 1):
        all_combos = list(combinations(positions, k))
        cap = ORACLE_MAX_PER_K_LOW if k <= 4 else ORACLE_MAX_PER_K_HIGH
        if len(all_combos) > cap:
            chosen = rng.sample(all_combos, cap)
        else:
            chosen = all_combos
        info.append(f"k={k}: {len(chosen)}/{len(all_combos)}")
        for npu_pos in chosen:
            s = set(npu_pos)
            cands.append(["NPU" if i in s else "GPU" for i in range(N)])
    return cands, info


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
                             daemon=True, name=f"g2_n{n}_s{i}_{dev}")
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
        w.writerow({k: row.get(k, "") for k in cols})


def main():
    val = load_val()

    base_cols = ["bg_level", "n_streams", "npu_mode", "placement_name", "placement_spec",
                 "mean_sap", "worst_sap", "sap_var",
                 "mean_map", "worst_map",
                 "mean_latency", "frame_skip_total", "n_active_streams", "wall_sec"]
    per_stream_cols = []
    for i in range(max(N_VALUES)):
        for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "map", "lat", "skip"]:
            per_stream_cols.append(f"s{i}_{k}")
    COLS = base_cols + per_stream_cols

    if OUT_CSV.exists():
        OUT_CSV.unlink()

    print(f"[stepG2] infer_mode={INFER_MODE}  NPU_CAP={NPU_CAP}")
    print("[stepG2] preloading bg (ORT first then torch)…")
    t0 = time.time()
    preload_background_models(max_level="L2")  # only L1 needed but preload L2 for safety
    print(f"[stepG2] bg preload {time.time()-t0:.1f}s")

    N_MAX = max(N_VALUES)
    print(f"[stepG2] preloading {N_MAX} GPU + {NPU_CAP} NPU (single mode)…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(N_MAX)]
    preload_npu_instances(NPU_CAP, infer_mode=INFER_MODE)
    npu_models = [get_npu_model(i) for i in range(NPU_CAP)]
    print(f"[stepG2] gpu+npu preload {time.time()-t0:.1f}s")

    # Build measurement plan
    plan = []
    oracle_info = {}
    for bg in BG_LEVELS:
        for N in N_VALUES:
            named = {
                "Naive_allGPU":  make_naive(N),
                "AllNPU":        make_all_npu(N),
                "SizeAware":     make_size_aware(N),
                "SizeBlindRev":  make_size_blind_rev(N),
            }
            # N=8 extras (SizeAware variants test how much NPU offload helps)
            if N == 8:
                named["SizeAware_NPU4"] = make_size_aware_with_target(N, 4)
                named["SizeAware_NPU6"] = make_size_aware_with_target(N, 6)
                named["SizeAware_NPU8"] = make_size_aware_with_target(N, 8)

            oracle_pool, sampled = oracle_candidates(N)
            if N not in oracle_info:
                oracle_info[N] = (len(oracle_pool), sampled)

            uniq = {}
            def k(p): return ",".join(p)
            for name, plc in named.items():
                uniq[k(plc)] = name
            for plc in oracle_pool:
                kk = k(plc)
                if kk in uniq:
                    if "Oracle" not in uniq[kk]:
                        uniq[kk] = uniq[kk] + "+Oracle"
                else:
                    uniq[kk] = "Oracle"
            for spec, name in uniq.items():
                plc = spec.split(",")
                plan.append((bg, N, name, plc))

    print(f"\n[stepG2] oracle pool sizes per N:")
    for N, (total, info) in oracle_info.items():
        print(f"  N={N}: oracle={total} (per-k: {info})")
    print(f"[stepG2] total measurements: {len(plan)} ({len(BG_LEVELS)} bg × {len(N_VALUES)} N)")

    t_all = time.time()
    for idx, (bg, N, name, placement) in enumerate(plan, 1):
        sids = get_stream_sids(N)
        splits = [load_split_for_sid(val, s) for s in sids]
        print(f"\n[G2 {idx:3d}/{len(plan)}] bg={bg}  N={N}  name={name[:25]:<25s}  plc={placement}")
        try:
            per_stream, wall = measure(placement, sids, splits, bg, gpu_models, npu_models)
            agg = aggregate(per_stream)
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}")
            continue
        row = {
            "bg_level": bg, "n_streams": N, "npu_mode": INFER_MODE,
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
            row[f"s{i}_dev"]   = s["device"]
            row[f"s{i}_sid"]   = s["sid"]
            row[f"s{i}_sap"]   = round(s["sap_5095"], 4)
            row[f"s{i}_sap_s"] = round(s["sap_s"], 4)
            row[f"s{i}_sap_m"] = round(s["sap_m"], 4)
            row[f"s{i}_sap_l"] = round(s["sap_l"], 4)
            row[f"s{i}_map"]   = round(s["map_5095"], 4)
            row[f"s{i}_lat"]   = round(s["latency_mean"], 2)
            row[f"s{i}_skip"]  = round(s["frame_skip_pct"], 1)
        append_row(row, COLS)
        print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
              f"n_active={agg['n_active_streams']}/{N}  ({wall:.1f}s)")

    print(f"\n[stepG2] all done. wall_total={time.time()-t_all:.1f}s")

    # ---- Analysis: build best-per-strategy summary ----
    df = pd.read_csv(OUT_CSV)
    strategies_main = ["Naive_allGPU", "AllNPU", "SizeAware", "SizeBlindRev"]
    summary_rows = []
    for bg in BG_LEVELS:
        for N in N_VALUES:
            for name in strategies_main:
                sub = df[(df.bg_level == bg) & (df.n_streams == N) &
                         (df.placement_name.str.startswith(name))]
                if len(sub):
                    summary_rows.append({"bg": bg, "N": N, "strategy": name,
                                         "worst_sap": float(sub.iloc[0]["worst_sap"]),
                                         "mean_sap": float(sub.iloc[0]["mean_sap"]),
                                         "spec": sub.iloc[0]["placement_spec"]})
            # N=8 extras
            if N == 8:
                for ex in ["SizeAware_NPU4", "SizeAware_NPU6", "SizeAware_NPU8"]:
                    sub = df[(df.bg_level == bg) & (df.n_streams == N) &
                             (df.placement_name.str.startswith(ex))]
                    if len(sub):
                        summary_rows.append({"bg": bg, "N": N, "strategy": ex,
                                             "worst_sap": float(sub.iloc[0]["worst_sap"]),
                                             "mean_sap": float(sub.iloc[0]["mean_sap"]),
                                             "spec": sub.iloc[0]["placement_spec"]})
            sub_o = df[(df.bg_level == bg) & (df.n_streams == N)]
            if len(sub_o):
                best = sub_o.sort_values("worst_sap", ascending=False).iloc[0]
                summary_rows.append({"bg": bg, "N": N, "strategy": "Oracle",
                                     "worst_sap": float(best["worst_sap"]),
                                     "mean_sap": float(best["mean_sap"]),
                                     "spec": best["placement_spec"]})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_CSV.parent / "step_g2_summary.csv", index=False)

    print("\n=== Step G2 (single mode) summary — worst-stream sAP ===")
    print(summary.pivot_table(index=["bg", "N"], columns="strategy",
                              values="worst_sap").round(3).to_string())

    # Compare with Step G (multi mode)
    if STEP_G_CSV.exists():
        df_g = pd.read_csv(STEP_G_CSV)
        print("\n=== Step G (multi) vs Step G2 (single) — worst-stream sAP, bg L1 ===")
        print(f"{'N':>3}  {'strategy':<14s}  {'multi':>7s}  {'single':>7s}  {'diff':>7s}")
        for N in N_VALUES:
            for strat in strategies_main:
                m = df_g[(df_g.bg_level == "L1") & (df_g.n_streams == N) &
                        (df_g.placement_name.str.startswith(strat))]
                s = summary[(summary.bg == "L1") & (summary.N == N) &
                            (summary.strategy == strat)]
                m_v = float(m.iloc[0]["worst_sap"]) if len(m) else float("nan")
                s_v = float(s.iloc[0]["worst_sap"]) if len(s) else float("nan")
                diff = s_v - m_v if not (np.isnan(m_v) or np.isnan(s_v)) else float("nan")
                print(f"  {N}  {strat:<14s}  {m_v:7.3f}  {s_v:7.3f}  {diff:+7.3f}")
            # Oracle from each
            m_o = df_g[(df_g.bg_level == "L1") & (df_g.n_streams == N)].sort_values("worst_sap", ascending=False)
            s_o = summary[(summary.bg == "L1") & (summary.N == N) & (summary.strategy == "Oracle")]
            if len(m_o) and len(s_o):
                m_v = float(m_o.iloc[0]["worst_sap"]); s_v = float(s_o.iloc[0]["worst_sap"])
                print(f"  {N}  {'Oracle':<14s}  {m_v:7.3f}  {s_v:7.3f}  {s_v - m_v:+7.3f}")
        print()
        for N in [8]:
            ex_rows = summary[(summary.bg == "L1") & (summary.N == N) &
                              (summary.strategy.str.startswith("SizeAware_NPU"))]
            if len(ex_rows):
                print(f"  N=8 extras (single mode only):")
                for _, r in ex_rows.iterrows():
                    print(f"    {r['strategy']:<18s}  worst_sAP={r['worst_sap']:.3f}  "
                          f"spec={r['spec']}")

    # ---- Figure 1: multi vs single comparison ----
    FIG_CMP.parent.mkdir(parents=True, exist_ok=True)
    if STEP_G_CSV.exists():
        df_g = pd.read_csv(STEP_G_CSV)
        fig, axes = plt.subplots(1, len(N_VALUES), figsize=(5 * len(N_VALUES), 5), squeeze=False)
        bar_w = 0.36
        x = np.arange(len(strategies_main))
        for i, N in enumerate(N_VALUES):
            ax = axes[0][i]
            multi_vals = []
            single_vals = []
            for strat in strategies_main:
                m = df_g[(df_g.bg_level == "L1") & (df_g.n_streams == N) &
                        (df_g.placement_name.str.startswith(strat))]
                s = summary[(summary.bg == "L1") & (summary.N == N) &
                            (summary.strategy == strat)]
                multi_vals.append(float(m.iloc[0]["worst_sap"]) if len(m) else 0)
                single_vals.append(float(s.iloc[0]["worst_sap"]) if len(s) else 0)
            ax.bar(x - bar_w / 2, multi_vals, bar_w, color="#999999",
                   edgecolor="black", label="multi (Step G)")
            ax.bar(x + bar_w / 2, single_vals, bar_w, color="#3680c4",
                   edgecolor="black", label="single (Step G2)")
            for j, (mv, sv) in enumerate(zip(multi_vals, single_vals)):
                ax.text(j - bar_w / 2, mv + 0.001, f"{mv:.3f}", ha="center",
                        va="bottom", fontsize=7)
                ax.text(j + bar_w / 2, sv + 0.001, f"{sv:.3f}", ha="center",
                        va="bottom", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(strategies_main, rotation=30, fontsize=7, ha="right")
            ax.set_ylabel("worst-stream sAP")
            ax.set_title(f"N = {N}  (bg L1)")
            ax.grid(axis="y", alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        plt.suptitle("Step G2 — multi mode vs single mode (per N, bg L1)", fontsize=12)
        plt.tight_layout()
        fig.savefig(FIG_CMP, dpi=120)
        print(f"saved {FIG_CMP}")

    # ---- Figure 2: capacity curve (single, with Step G N=2,3 borrowed) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # X = all N (2..8) using Step G for 2,3 and Step G2 for 4..8
    X_VALUES = [2, 3, 4, 5, 6, 8]
    colors = {"Naive_allGPU": "#888888", "AllNPU": "#3b9c4d",
              "SizeAware": "#3680c4", "SizeBlindRev": "#c43b3b", "Oracle": "#c4a236"}
    markers = {"Naive_allGPU": "o", "AllNPU": "D", "SizeAware": "s",
               "SizeBlindRev": "^", "Oracle": "*"}
    df_g = pd.read_csv(STEP_G_CSV) if STEP_G_CSV.exists() else None
    for strat in strategies_main + ["Oracle"]:
        ys = []
        for N in X_VALUES:
            if N in [2, 3] and df_g is not None:
                sub = df_g[(df_g.bg_level == "L1") & (df_g.n_streams == N)]
                if strat == "Oracle":
                    if len(sub): ys.append(float(sub["worst_sap"].max()))
                    else: ys.append(np.nan)
                elif strat == "AllNPU":
                    # AllNPU not in Step G; skip or get from oracle (all-NPU = Oracle k=N)
                    spec = json.dumps(["NPU"] * N)
                    s2 = sub[sub.placement_spec == spec]
                    ys.append(float(s2.iloc[0]["worst_sap"]) if len(s2) else np.nan)
                else:
                    s2 = sub[sub.placement_name.str.startswith(strat)]
                    ys.append(float(s2.iloc[0]["worst_sap"]) if len(s2) else np.nan)
            else:
                s2 = summary[(summary.bg == "L1") & (summary.N == N) &
                             (summary.strategy == strat)]
                ys.append(float(s2.iloc[0]["worst_sap"]) if len(s2) else np.nan)
        valid_mask = ~np.isnan(ys)
        ax.plot(np.array(X_VALUES)[valid_mask], np.array(ys)[valid_mask], "-",
                marker=markers[strat], color=colors[strat],
                markersize=11 if strat == "Oracle" else 8,
                label=strat, linewidth=2)
    for th in [0.05, 0.08, 0.10]:
        ax.axhline(th, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.text(X_VALUES[-1] + 0.05, th, f"sAP={th:.2f}", fontsize=7,
                color="gray", verticalalignment="center")
    ax.set_xlabel("# foreground streams (N)")
    ax.set_ylabel("worst-stream sAP")
    ax.set_title("Capacity recheck — N=2,3 from Step G (multi); N=4-8 from Step G2 (single)")
    ax.set_xticks(X_VALUES)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_CURVE, dpi=120)
    print(f"saved {FIG_CURVE}")


if __name__ == "__main__":
    main()
