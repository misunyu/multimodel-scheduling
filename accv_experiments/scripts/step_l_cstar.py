"""Step L — TASK C: Inversion threshold C* vs detector capacity (YOLOv11s).

Implements §3 and TASK C of claude_code_experiments.md. For YOLOv11s, sweep
the background ladder (L0..L3_VLM) on GPU and measure the per-size
streaming sAP under each level. The scalar contention proxy C is the GPU
foreground effective end-to-end latency averaged per ladder level. For each
size group g in {small-rich, medium-mixed, large-rich}, compute:

  Q_g    = sum_s p_g(s) * Q(s)                      [from TASK B's gen_decomp]
  L_g(C) = sum_s p_g(s) * L(s, C)                   [TASK B + this script's measurements]
  C*_g   = C such that L_g(C*) = Q_g (linear interpolation)

Outputs:
  results/cstar.csv             per-(ladder, size, group) aggregates + C* per group
  paper/figures/gen_cstar.pdf   C* plot. Only YOLOv11s capacity (9.4M) is filled;
                                YOLOv11n/m are TBD per Q1(c).

Reuses TASK B's gen_decomp.csv for L0 and L1_light. Measures L1_heavy, L2_lm,
L3_vlm fresh. ~25 min wall-clock.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, fg_worker, load_split_for_sid,
                            load_val, per_stream_sap,
                            preload_background_models, stop_background)
from step_f_partA_matrix import per_stream_map_offline
# Bring in the custom 5-level bg registry from Step H2
from step_h2_robustness import start_bg_custom, BG_VARIANTS

RES = SCRIPT_DIR.parent / "results"
OUT_CSV = RES / "cstar.csv"
OUT_SUM = RES / "cstar_summary.csv"
FIG_DIR = SCRIPT_DIR.parent.parent / "paper" / "figures"
FIG_OUT = FIG_DIR / "gen_cstar.pdf"

DETECTOR = "YOLOv11s"
PARAMS_M = 9.4

# Size-group → sids (from step_e size classification)
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}
# Mean p(size) per group (from step_e CSV, manual aggregation)
GROUP_P = {
    "small-rich":   {"small": 0.462, "medium": 0.409, "large": 0.129},
    "medium-mixed": {"small": 0.383, "medium": 0.409, "large": 0.208},
    "large-rich":   {"small": 0.283, "medium": 0.401, "large": 0.316},
}

# Background ladder (in increasing intensity order). For L0/L1_light we reuse
# TASK B (step_k_gen_decomp); only the heavier levels are measured here.
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
NEW_LADDER_FOR_K = ["L1_heavy", "L2_lm", "L3_vlm"]


def measure_gpu_cell(sid, split, bg, gpu_model):
    res = defaultdict(list)
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg)
    t0 = time.time()
    fg_worker(0, "GPU", split, gpu_model, res, stop)
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    sap = per_stream_sap(split, res)
    mp = per_stream_map_offline(split, res)
    return {
        "sap_5095": sap["sap_5095"], "sap_50": sap["sap_50"],
        "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
        "map_5095": mp["map_5095"], "map_s": mp["map_s"],
        "map_m": mp["map_m"], "map_l": mp["map_l"],
        "latency_mean": sap["infer_mean_ms"],
        "eff_e2e_mean": float(np.mean(res["eff_ms"])) if res["eff_ms"] else 0.0,
        "frame_skip_pct": sap["frame_skip_pct"],
        "wall_sec": wall,
    }


def main():
    val = load_val()
    n_logs = len(val["sequences"])
    sids = list(range(n_logs))
    print(f"[stepL] {n_logs} logs, detector={DETECTOR}")

    # Preload all bg models (need L3 = ResNet+Qwen2-VL)
    print("[stepL] preloading bg up to L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[stepL] bg preload {time.time()-t0:.1f}s")

    print("[stepL] preloading GPU YOLO11s…")
    t0 = time.time()
    gpu = FGModelGPU()
    print(f"[stepL] gpu preload {time.time()-t0:.1f}s")

    # ----- fresh measurements for L1_heavy, L2_lm, L3_vlm -----
    if OUT_CSV.exists(): OUT_CSV.unlink()
    cols = ["detector", "sid", "log_id", "device", "bg_level",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l",
            "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    t_all = time.time()
    for sid in sids:
        split = load_split_for_sid(val, sid)
        for bg in NEW_LADDER_FOR_K:
            print(f"[L sid={sid:>2d}  bg={bg:<9s}]", end=" ", flush=True)
            try:
                m = measure_gpu_cell(sid, split, bg, gpu)
            except Exception as e:
                print(f"FAIL: {type(e).__name__}: {e}")
                continue
            row = {
                "detector": DETECTOR, "sid": sid,
                "log_id": val["sequences"][sid],
                "device": "GPU", "bg_level": bg,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            }
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            print(f"sap={m['sap_5095']:.3f}  sap_s={m['sap_s']:.3f}/m={m['sap_m']:.3f}/l={m['sap_l']:.3f}  "
                  f"eff={m['eff_e2e_mean']:.1f}ms  skip={m['frame_skip_pct']:.0f}%  ({m['wall_sec']:.1f}s)")
    print(f"\n[stepL] new measurements done. wall={time.time()-t_all:.1f}s")

    # ----- merge with TASK B's L0/L1_light data -----
    df_new = pd.read_csv(OUT_CSV)
    df_b = pd.read_csv(RES / "gen_decomp.csv")
    df_b_gpu = df_b[df_b.device == "GPU"].copy()
    # gen_decomp.csv uses "L1_light" already; "L0" is also "L0". Same vocabulary.
    # gen_decomp.csv has the cols we need; add eff_e2e_mean=latency_mean placeholder
    # if missing (TASK B did not record E2E explicitly, but for L0 it equals
    # per-call latency since no skip). Use latency_mean as approximate eff.
    if "eff_e2e_mean" not in df_b_gpu.columns:
        df_b_gpu["eff_e2e_mean"] = df_b_gpu["latency_mean"]
    keep = ["detector", "sid", "log_id", "device", "bg_level",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l",
            "latency_mean", "eff_e2e_mean", "frame_skip_pct"]
    df_b_gpu = df_b_gpu[[c for c in keep if c in df_b_gpu.columns]].copy()
    df_full = pd.concat([df_b_gpu, df_new[keep]], ignore_index=True, sort=False)

    # Sanity: only ladder levels we care about
    df_full = df_full[df_full.bg_level.isin(LADDER)].copy()

    # ----- aggregate per (bg, size group) -----
    rows_sum = []
    # Per-size L(s,C) is computed first
    L_by_bg_size = {}
    # baseline: GPU L0 mAP per size (the AP_GPU_offline reference)
    gpu_L0 = df_full[df_full.bg_level == "L0"]
    map_L0 = {"small": float(gpu_L0["map_s"].mean()),
              "medium": float(gpu_L0["map_m"].mean()),
              "large": float(gpu_L0["map_l"].mean())}
    # ALSO use mean offline mAP at every ladder level (same detector,
    # offline mAP is unaffected by streaming clock; differences come from
    # frame-skip → fewer detections in the eval) — but we'll use map_L0 as
    # the offline ceiling per the paper definition.
    for bg in LADDER:
        sub = df_full[df_full.bg_level == bg]
        if not len(sub): continue
        sap_s = float(sub["sap_s"].mean())
        sap_m = float(sub["sap_m"].mean())
        sap_l = float(sub["sap_l"].mean())
        eff = float(sub["eff_e2e_mean"].mean())
        skip = float(sub["frame_skip_pct"].mean())
        L_s = map_L0["small"]  - sap_s
        L_m = map_L0["medium"] - sap_m
        L_l = map_L0["large"]  - sap_l
        L_by_bg_size[bg] = {"small": L_s, "medium": L_m, "large": L_l,
                             "eff": eff, "skip": skip,
                             "sap_s": sap_s, "sap_m": sap_m, "sap_l": sap_l}
        rows_sum.append({
            "detector": DETECTOR, "params_M": PARAMS_M, "bg_level": bg,
            "gpu_eff_e2e_ms": round(eff, 2),
            "gpu_frame_skip_pct": round(skip, 2),
            "L_small": round(L_s, 4),
            "L_medium": round(L_m, 4),
            "L_large": round(L_l, 4),
        })

    # Q(s) from TASK B summary
    Q_df = pd.read_csv(RES / "gen_decomp_summary.csv")
    Q = {row["size"]: float(row["Q"])
         for _, row in Q_df[Q_df.detector == DETECTOR].iterrows()}

    # Per-group Q_g and L_g(C) using GROUP_P weights
    group_curves = {}
    for grp, p in GROUP_P.items():
        Q_g = (p["small"] * Q["small"] +
               p["medium"] * Q["medium"] +
               p["large"] * Q["large"])
        L_g_by_bg = {}
        for bg in LADDER:
            if bg not in L_by_bg_size: continue
            d = L_by_bg_size[bg]
            L_g = (p["small"] * d["small"] +
                   p["medium"] * d["medium"] +
                   p["large"] * d["large"])
            L_g_by_bg[bg] = {"C": d["eff"], "L_g": L_g, "skip": d["skip"]}
        group_curves[grp] = {"Q_g": Q_g, "by_bg": L_g_by_bg}

    # ----- C* via piecewise-linear interpolation -----
    def cstar_interp(Q_g, points):
        """points: list of (C, L_g) sorted by C. Find first crossing where L>=Q.
        Returns (C*, ladder_pre, ladder_post, frac) or (None, ...) if never crosses."""
        pts = sorted(points, key=lambda p: p[0])
        for i in range(len(pts) - 1):
            C1, L1 = pts[i]
            C2, L2 = pts[i + 1]
            if (L1 - Q_g) * (L2 - Q_g) <= 0 and (L2 != L1):
                frac = (Q_g - L1) / (L2 - L1)
                Cstar = C1 + frac * (C2 - C1)
                return Cstar, i, i + 1, frac
        # Special case: already > Q at C=0 (shouldn't happen for L(0)~=0)
        if pts[0][1] >= Q_g:
            return pts[0][0], 0, 0, 0.0
        # Never crosses
        return None, None, None, None

    print("\n" + "=" * 78)
    print(" Per-group C* (YOLOv11s)")
    print("=" * 78)
    cstar_rows = []
    for grp, data in group_curves.items():
        Q_g = data["Q_g"]
        pts = [(d["C"], d["L_g"]) for d in data["by_bg"].values()]
        pts_named = [(bg, d["C"], d["L_g"]) for bg, d in data["by_bg"].items()]
        Cstar, _, _, _ = cstar_interp(Q_g, pts)
        cstar_rows.append({
            "detector": DETECTOR, "params_M": PARAMS_M,
            "group": grp, "Q_g": round(Q_g, 4),
            "Cstar_ms": round(Cstar, 2) if Cstar is not None else None,
        })
        print(f"  group={grp:<14s}  Q_g={Q_g:+.4f}")
        for bg, C, L in pts_named:
            print(f"     bg={bg:<9s}  C={C:6.2f} ms  L_g={L:+.4f}")
        if Cstar is None:
            print(f"  -> no crossover in ladder range")
        else:
            print(f"  -> C*={Cstar:.2f} ms")

    pd.DataFrame(rows_sum).to_csv(OUT_SUM, index=False)
    pd.DataFrame(cstar_rows).to_csv(RES / "cstar_per_group.csv", index=False)
    print(f"\nsaved {OUT_SUM}")
    print(f"saved {RES / 'cstar_per_group.csv'}")

    # ----- Figure: L_g vs C with Q_g horizontal lines and C* markers -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"small-rich": "#3680c4",
              "medium-mixed": "#3b9c4d",
              "large-rich": "#c43b3b"}

    # Panel A: L_g(C) curves vs scalar C
    ax = axes[0]
    for grp, data in group_curves.items():
        Q_g = data["Q_g"]
        items = sorted(data["by_bg"].items(), key=lambda x: x[1]["C"])
        Cs = [v["C"] for _, v in items]
        Ls = [v["L_g"] for _, v in items]
        ax.plot(Cs, Ls, "-o", color=colors[grp], linewidth=2, markersize=8,
                label=f"L_g(C)  {grp}")
        ax.axhline(Q_g, color=colors[grp], linestyle="--", linewidth=1.2,
                   alpha=0.7)
        # C* marker
        Cstar = next((r["Cstar_ms"] for r in cstar_rows if r["group"] == grp), None)
        if Cstar is not None:
            ax.plot(Cstar, Q_g, marker="*", color=colors[grp],
                    markersize=18, markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(f"C*={Cstar:.1f}ms",
                        xy=(Cstar, Q_g), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8, color=colors[grp])
        # Annotate Q_g
        ax.text(Cs[-1] + 0.4, Q_g, f"$Q_g$={Q_g:.3f}",
                color=colors[grp], fontsize=8, verticalalignment="center")
        # Annotate ladder points
        for (bg, _), C, L in zip(items, Cs, Ls):
            ax.annotate(bg, xy=(C, L), xytext=(0, -12),
                        textcoords="offset points",
                        fontsize=6.5, ha="center", color="#555")
    ax.set_xlabel("scalar contention proxy C  (GPU foreground effective E2E latency, ms)")
    ax.set_ylabel("L_g(C)  (offline mAP − streaming sAP)")
    ax.set_title(f"YOLOv11s ({PARAMS_M} M): L_g(C) crosses Q_g at C*")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: C* vs detector capacity (only one detector measured)
    ax = axes[1]
    for grp, color in colors.items():
        Cstar = next((r["Cstar_ms"] for r in cstar_rows if r["group"] == grp), None)
        if Cstar is not None:
            ax.scatter([PARAMS_M], [Cstar], color=color, s=120, marker="o",
                       edgecolor="black", linewidth=0.5, label=f"{grp}: C*={Cstar:.1f}ms",
                       zorder=4)
    # TBD markers for n and m
    for cap, lbl in [(2.6, "YOLOv11n"), (20.1, "YOLOv11m")]:
        ax.axvline(cap, color="gray", linestyle=":", alpha=0.4)
        ax.text(cap, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 50,
                f"{lbl}\n(TBD)", fontsize=7, color="gray",
                ha="center", va="top")
    ax.set_xlabel("detector capacity (parameters, M)")
    ax.set_ylabel("inversion threshold C*  (ms)")
    ax.set_xscale("log")
    ax.set_xticks([2.6, 9.4, 20.1])
    ax.set_xticklabels(["2.6\n(n)", "9.4\n(s)", "20.1\n(m)"])
    ax.set_title("C* vs detector capacity (only YOLOv11s measured; n/m TBD)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.suptitle("TASK C — Inversion threshold C* (YOLOv11s; capacity-scaling row TBD)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_OUT}")


if __name__ == "__main__":
    main()
