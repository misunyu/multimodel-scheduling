"""Step I — Natural multi-camera composition measurement.

Goal: verify that the contention-driven NPU offload schedule, established on
size-diversity-maximizing Composition A (4 small-rich + 4 large-rich), also
applies to natural Argoverse-HD camera mixes (medium-mixed and size-diverse).

The figure-1 grid is 3 compositions × 2 stream counts × 2 background levels.
6 of the 12 cells are reused from Step H2 (already measured); 6 are new
measurements written here.

CSV:     results/step_i_natural_composition.csv  (new measurements only)
Figures: results/figures/step_i_natural_comparison.png  (12-cell grid)
         results/figures/step_i_schedule_natural.png    (offload-fraction curves)

All NPU instances use infer_mode="single".
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
                            stop_background)
from step_h2_robustness import (start_bg_custom, base_cols_for, append_row,
                                aggregate, row_from_measurement)
from step_f_partA_matrix import per_stream_map_offline

RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"
OUT = RES / "step_i_natural_composition.csv"
FIG_GRID = FIG / "step_i_natural_comparison.png"
FIG_SCHED = FIG / "step_i_schedule_natural.png"
H2_PARTA = RES / "step_h2_robustness.csv"
H2_PARTB = RES / "step_h2_bg_ablation.csv"

INFER_MODE = "single"
NPU_CAP = 8

# Compositions (N=8 use full list; N=4 uses subset)
COMPS_N8 = {
    "A_baseline":     [2, 22, 13, 16, 3, 21, 14, 4],     # 4 small + 4 large (extreme)
    "B_medium_mixed": [17, 8, 10, 11, 2, 13, 3, 14],     # 4 medium + 2 small + 2 large
    "C_diverse":      [15, 19, 23, 0, 1, 20, 5, 7],      # 3 small + 3 medium + 2 large
}
COMPS_N4 = {
    "A_baseline":     [2, 22, 3, 21],            # matches Step H2 Part B
    "B_medium_mixed": [17, 8, 10, 11],           # all medium-mixed (= Natural-Med)
    "C_diverse":      [15, 0, 5, 7],             # 1 small + 1 medium + 2 large
}

# NPU-friendliness priority per composition (small-rich first per Part A finding;
# medium next; large last; tie-breaks by pct_small_count desc).
NPU_PRIORITY = {
    ("A_baseline", 8):     [2, 13, 22, 16, 21, 14, 4, 3],
    ("B_medium_mixed", 8): [2, 13, 8, 17, 10, 11, 14, 3],
    ("C_diverse", 8):      [0, 15, 23, 19, 1, 20, 5, 7],
    ("A_baseline", 4):     [2, 22, 21, 3],   # 2 small, then 2 large by pct_small_count
    ("B_medium_mixed", 4): [8, 17, 10, 11],  # all medium, by pct_small_count desc
    ("C_diverse", 4):      [0, 15, 5, 7],    # medium 0 highest pct_small, then small 15, then large
}

BG_LEVELS = ["L1_light", "L1_heavy"]


# -------- placement builders ------------------------------------------------

def make_naive(N): return ["GPU"] * N
def make_all_npu(N): return ["NPU"] * N


def make_size_aware(comp_name, N, sids, k_npu):
    """Top k_npu NPU-friendly sids (per NPU_PRIORITY) → NPU."""
    pri = NPU_PRIORITY[(comp_name, N)]
    npu_set = set(pri[:k_npu])
    return ["NPU" if sid in npu_set else "GPU" for sid in sids]


def make_size_blind_rev(comp_name, N, sids, k_npu):
    """Bottom k_npu (NPU-unfriendly = large-rich) → NPU."""
    pri = NPU_PRIORITY[(comp_name, N)]
    npu_set = set(pri[-k_npu:])
    return ["NPU" if sid in npu_set else "GPU" for sid in sids]


def oracle_candidates(N, seed=42):
    """For N=4: all 16. For N=8: stratified sample by NPU count (3-6 per k)."""
    rng = random.Random(seed)
    cands = []
    for k in range(N + 1):
        all_combos = list(combinations(range(N), k))
        if N == 4 or len(all_combos) <= 6:
            chosen = all_combos
        elif k <= 4:
            chosen = rng.sample(all_combos, min(6, len(all_combos)))
        else:
            chosen = rng.sample(all_combos, min(4, len(all_combos)))
        for npu_pos in chosen:
            s = set(npu_pos)
            cands.append(["NPU" if i in s else "GPU" for i in range(N)])
    return cands


# -------- measurement core --------------------------------------------------

def measure_cell(placement, sids, splits, bg_level, gpu_models, npu_models):
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
            raise ValueError(dev)
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, splits[i], model, results[i], stop),
                             daemon=True, name=f"I_n{n}_s{i}_{dev}")
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


# -------- main measurement plan ---------------------------------------------

def main():
    val = load_val()

    print("[stepI] preloading bg L2 max (avoid Qwen since not used in this experiment)")
    t0 = time.time()
    preload_background_models(max_level="L2")
    print(f"[stepI] bg preload {time.time()-t0:.1f}s")

    print(f"[stepI] preloading {NPU_CAP} GPU + {NPU_CAP} NPU (single mode)…")
    t0 = time.time()
    gpu_models = [FGModelGPU() for _ in range(NPU_CAP)]
    preload_npu_instances(NPU_CAP, infer_mode=INFER_MODE)
    npu_models = [get_npu_model(i) for i in range(NPU_CAP)]
    print(f"[stepI] gpu+npu preload {time.time()-t0:.1f}s")

    if OUT.exists(): OUT.unlink()
    COLS = base_cols_for(8)

    # Only the 6 missing cells (others reused from H2)
    NEW_CELLS = [
        # (composition, N, bg)
        ("B_medium_mixed", 4, "L1_light"),
        ("B_medium_mixed", 4, "L1_heavy"),
        ("C_diverse",      4, "L1_light"),
        ("C_diverse",      4, "L1_heavy"),
        ("B_medium_mixed", 8, "L1_heavy"),
        ("C_diverse",      8, "L1_heavy"),
    ]

    # Build plan
    plan = []
    for comp, N, bg in NEW_CELLS:
        sids = COMPS_N4[comp] if N == 4 else COMPS_N8[comp]
        # named strategies
        if N == 4:
            named = {
                "Naive_allGPU":     make_naive(N),
                "SizeAware_NPU2":   make_size_aware(comp, N, sids, 2),
                "AllNPU":           make_all_npu(N),
                "SizeBlindRev_NPU2":make_size_blind_rev(comp, N, sids, 2),
            }
        else:
            named = {
                "Naive_allGPU":     make_naive(N),
                "SizeAware_NPU4":   make_size_aware(comp, N, sids, 4),
                "SizeAware_NPU6":   make_size_aware(comp, N, sids, 6),
                "AllNPU":           make_all_npu(N),
                "SizeBlindRev_NPU4":make_size_blind_rev(comp, N, sids, 4),
            }
        oracle = oracle_candidates(N)
        # dedup: later names overwrite earlier; Oracle only if spec new
        uniq = {}
        def kf(p): return ",".join(p)
        for name, plc in named.items():
            uniq[kf(plc)] = name
        for plc in oracle:
            kk = kf(plc)
            if kk in uniq:
                if "Oracle" not in uniq[kk]:
                    uniq[kk] = uniq[kk] + "+Oracle"
            else:
                uniq[kk] = "Oracle"
        for spec_str, name in uniq.items():
            plc = spec_str.split(",")
            plan.append((comp, N, bg, name, plc))

    print(f"\n[stepI] total measurements: {len(plan)} across {len(NEW_CELLS)} new cells")

    t_all = time.time()
    for idx, (comp, N, bg, name, placement) in enumerate(plan, 1):
        sids = COMPS_N4[comp] if N == 4 else COMPS_N8[comp]
        splits = [load_split_for_sid(val, s) for s in sids]
        print(f"\n[I {idx:3d}/{len(plan)}] comp={comp:<16s}  N={N}  bg={bg:<9s}  "
              f"name={name[:22]:<22s}  plc={placement}")
        try:
            per_stream, wall = measure_cell(placement, sids, splits, bg,
                                             gpu_models, npu_models)
            agg = aggregate(per_stream)
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}")
            continue
        row = row_from_measurement("I", comp, N, bg, name, placement,
                                    per_stream, agg, wall)
        append_row(OUT, row, COLS)
        print(f"   mean_sAP={agg['mean_sap']:.3f}  worst_sAP={agg['worst_sap']:.3f}  "
              f"n_active={agg['n_active_streams']}/{N}  ({wall:.1f}s)")
    print(f"\n[stepI] all done. wall_total={time.time()-t_all:.1f}s")

    build_figures()


# -------- analysis & figures ------------------------------------------------

def _lookup_step_h2_cell(comp, N, bg, strategy_filter=None):
    """Find best cell for (comp, N, bg) from Step H2 by strategy or Oracle.
    Returns dict {strategy_name: (worst_sap, mean_sap, placement_spec)}."""
    out = {}
    # H2 Part B covers Composition A × all bg × {4,8}
    if comp == "A_baseline":
        df = pd.read_csv(H2_PARTB)
        sub = df[(df.n_streams == N) & (df.bg_level == bg)]
        for _, r in sub.iterrows():
            name = r["placement_name"]
            if strategy_filter is None or name in strategy_filter:
                out[name] = (float(r["worst_sap"]), float(r["mean_sap"]),
                             r["placement_spec"])
    # H2 Part A covers A/B/C × N=8 × L1_light
    if N == 8 and bg == "L1_light":
        df = pd.read_csv(H2_PARTA)
        sub = df[df.composition == comp]
        for _, r in sub.iterrows():
            name = r["placement_name"]
            if strategy_filter is None or name in strategy_filter:
                out.setdefault(name, (float(r["worst_sap"]), float(r["mean_sap"]),
                                       r["placement_spec"]))
    return out


def _gather_grid(comp, N, bg, strategies_wanted):
    """Get a dict {strategy: (worst, mean, spec)} from either Step I or Step H2."""
    out = {}
    # Try Step I new measurements first
    if OUT.exists():
        df_i = pd.read_csv(OUT)
        sub = df_i[(df_i.composition == comp) & (df_i.n_streams == N) &
                   (df_i.bg_level == bg)]
        for _, r in sub.iterrows():
            name = r["placement_name"]
            for s in strategies_wanted:
                if name.startswith(s):
                    out[s] = (float(r["worst_sap"]), float(r["mean_sap"]),
                              r["placement_spec"])
                    break
        # Oracle = best worst_sap from any row
        if len(sub):
            best = sub.sort_values("worst_sap", ascending=False).iloc[0]
            out.setdefault("Oracle", (float(best["worst_sap"]),
                                       float(best["mean_sap"]), best["placement_spec"]))
    # Fill from Step H2
    h2 = _lookup_step_h2_cell(comp, N, bg)
    for name, val in h2.items():
        for s in strategies_wanted:
            if name.startswith(s):
                out.setdefault(s, val)
                break
    # Oracle from H2: best worst_sap among that cell
    if "Oracle" not in out:
        if comp == "A_baseline":
            df = pd.read_csv(H2_PARTB)
            sub = df[(df.n_streams == N) & (df.bg_level == bg)]
            if len(sub):
                best = sub.sort_values("worst_sap", ascending=False).iloc[0]
                out["Oracle"] = (float(best["worst_sap"]), float(best["mean_sap"]),
                                  best["placement_spec"])
        if N == 8 and bg == "L1_light":
            df = pd.read_csv(H2_PARTA)
            sub = df[df.composition == comp]
            if len(sub):
                best = sub.sort_values("worst_sap", ascending=False).iloc[0]
                out.setdefault("Oracle", (float(best["worst_sap"]),
                                            float(best["mean_sap"]), best["placement_spec"]))
    return out


def build_figures():
    print("\n[stepI] building grid summary + figures…")
    FIG.mkdir(parents=True, exist_ok=True)

    comps = ["A_baseline", "B_medium_mixed", "C_diverse"]
    Ns = [4, 8]
    bgs = BG_LEVELS

    # Pick a consistent strategy palette
    STRATS = ["Naive_allGPU", "SizeAware", "AllNPU", "SizeBlindRev", "Oracle"]
    STRAT_DISPLAY = {  # display label
        "Naive_allGPU": "Naive",
        "SizeAware":    "SizeAware",
        "AllNPU":       "AllNPU",
        "SizeBlindRev": "SizeBlindRev",
        "Oracle":       "Oracle",
    }
    # When merging different runs we map by prefix. "SizeAware" matches
    # SizeAware/SizeAware_NPU2/SizeAware_NPU4/SizeAware_NPU6 (take the one we
    # consider the canonical SizeAware: top-k matching N/2 NPU).
    # Strategy-prefix → which canonical
    def canon(name):
        if name.startswith("Naive_allGPU"): return "Naive_allGPU"
        if name.startswith("AllNPU"): return "AllNPU"
        if name.startswith("SizeBlindRev"): return "SizeBlindRev"
        if name.startswith("SizeAware"):
            # prefer SizeAware (no suffix) over numbered variants
            return "SizeAware"
        return None

    # Build a unified grid table
    grid_rows = []
    for comp in comps:
        for N in Ns:
            for bg in bgs:
                vals = _gather_grid(comp, N, bg, STRATS)
                # extract canonical strategies
                row = {"composition": comp, "N": N, "bg": bg}
                for s in STRATS:
                    if s in vals:
                        row[s] = vals[s][0]   # worst_sap
                grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RES / "step_i_grid_summary.csv", index=False)

    print("\n=== Grid summary (worst sAP) ===")
    print(grid.to_string(index=False))

    # ---- Figure 1: 12-cell grid (rows = comp × bg, cols = N) -----------------
    fig, axes = plt.subplots(len(comps) * len(bgs), len(Ns),
                              figsize=(5 * len(Ns), 2.6 * len(comps) * len(bgs)),
                              squeeze=False)
    colors = {"Naive_allGPU":"#888888", "SizeAware":"#3680c4",
              "AllNPU":"#3b9c4d", "SizeBlindRev":"#c43b3b", "Oracle":"#c4a236"}
    for r_i, (comp, bg) in enumerate([(c, b) for c in comps for b in bgs]):
        for c_i, N in enumerate(Ns):
            ax = axes[r_i][c_i]
            r = grid[(grid.composition == comp) & (grid.N == N) & (grid.bg == bg)]
            if not len(r):
                ax.set_visible(False); continue
            r = r.iloc[0]
            vals, labels, cols = [], [], []
            for s in STRATS:
                v = r.get(s, np.nan)
                if isinstance(v, float) and not np.isnan(v):
                    vals.append(v); labels.append(STRAT_DISPLAY[s]); cols.append(colors[s])
            bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v + 0.001, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
            # Annotate best strategy
            best_i = int(np.argmax(vals))
            bars[best_i].set_edgecolor("#000000")
            bars[best_i].set_linewidth(2.5)
            ax.set_title(f"{comp} | N={N} | bg={bg}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)
            if c_i == 0:
                ax.set_ylabel("worst sAP", fontsize=8)
    plt.suptitle("Step I — Natural composition grid (3 comp × 2 N × 2 bg)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_GRID, dpi=120)
    plt.close(fig)
    print(f"saved {FIG_GRID}")

    # ---- Figure 2: optimal offload fraction vs bg, per composition × N ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bgs_ordered = ["L1_light", "L1_heavy"]
    comp_color = {"A_baseline":"#c43b3b", "B_medium_mixed":"#3680c4", "C_diverse":"#3b9c4d"}
    comp_marker = {"A_baseline":"o", "B_medium_mixed":"s", "C_diverse":"^"}
    for c_i, N in enumerate(Ns):
        ax = axes[c_i]
        for comp in comps:
            opt_frac, opt_sap, opt_name = [], [], []
            for bg in bgs_ordered:
                vals = _gather_grid(comp, N, bg, STRATS)
                # Find best by worst_sap among all known placements (Step H2 oracle pool too)
                if "Oracle" in vals:
                    spec = vals["Oracle"][2]
                    try: spec_list = json.loads(spec)
                    except Exception: spec_list = spec.split(",")
                    npu_count = sum(1 for d in spec_list if d == "NPU")
                    opt_frac.append(npu_count / N)
                    opt_sap.append(vals["Oracle"][0])
                    # find what named strategy this corresponds to, if any
                    # default: just label by NPU count
                    opt_name.append(f"{npu_count}/{N}")
                else:
                    opt_frac.append(np.nan); opt_sap.append(np.nan); opt_name.append("")
            ax.plot(bgs_ordered, opt_frac, "-", marker=comp_marker[comp],
                    color=comp_color[comp], linewidth=2.2, markersize=11,
                    label=comp)
            for j, (frac, sap, nm) in enumerate(zip(opt_frac, opt_sap, opt_name)):
                if not np.isnan(frac):
                    ax.annotate(f"k={nm}\nsAP={sap:.3f}",
                                xy=(j, frac), xytext=(0, 11),
                                textcoords="offset points",
                                fontsize=7, ha="center", color="#444",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="#fffae6", ec="#cc9900", lw=0.4))
        ax.set_xlabel("background contention")
        if c_i == 0: ax.set_ylabel("optimal NPU offload fraction  k/N")
        ax.set_title(f"N = {N}")
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    plt.suptitle("Step I — Optimal offload schedule across natural compositions",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_SCHED, dpi=120)
    plt.close(fig)
    print(f"saved {FIG_SCHED}")


if __name__ == "__main__":
    main()
