#!/usr/bin/env python3
"""
Generate bounded_recovery_analysis_accumulation_5cand.pdf — stacked bar
chart showing T_detect (orange) + T_post (dark blue) per workload scenario,
derived from the 5-candidate BoundGuard sweep (Q1.3-consistent).

Usage:
    python scripts/bounded_recovery_accumulation_plot_5cand.py
    python scripts/bounded_recovery_accumulation_plot_5cand.py \
        --from-json results/bounded_sweep_5cand/sweep.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR,
                       "bounded_recovery_analysis_accumulation_5cand.pdf")
DEFAULT_JSON = os.path.join(RESULTS_DIR, "bounded_sweep_5cand", "sweep.json")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default=DEFAULT_JSON,
                        help="Path to sweep JSON file.")
    parser.add_argument("--out", default=OUT_PDF)
    parser.add_argument("--bound", type=float, default=None,
                        help="Override: theoretical upper bound (seconds). "
                             "If not set, computed from "
                             "--T, --Tv, --N-cand, --delta, --T-stable.")
    parser.add_argument("--T", type=float, default=3.0,
                        help="Monitoring window length T (s). Default: 3.")
    parser.add_argument("--Tv", type=float, default=3.0,
                        help="Validation window length T_v (s). Default: 3.")
    parser.add_argument("--N-cand", type=int, default=5,
                        help="Number of candidate placements. Default: 5.")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Switching + warm-up overhead delta (s). Default: 1.")
    parser.add_argument("--T-stable", type=float, default=10.0,
                        help="Stabilization time after the final valid "
                             "placement (s). Default: 10.")
    args = parser.parse_args()

    if args.bound is not None:
        bound = args.bound
        bound_desc = f"T_recovery bound = {bound:.1f}s (manual)"
        bound_formula = rf"$T_{{recovery}} \leq {bound:.0f}$\,s"
    else:
        bound = args.T + args.N_cand * (args.Tv + args.delta) + args.T_stable
        bound_desc = (f"T_recovery bound = {bound:.1f}s "
                      f"(T={args.T} + {args.N_cand}*(Tv={args.Tv}+d={args.delta}) "
                      f"+ T_stable={args.T_stable})")
        bound_formula = (
            r"$T_{recovery} \leq T + N_{cand}\,(T_v + \delta) + T_{stable}"
            rf" = {args.T:.0f} + {args.N_cand}\!\cdot\!({args.Tv:.0f}+{args.delta:.0f})"
            rf" + {args.T_stable:.0f} = {bound:.0f}$\,s"
        )

    with open(args.from_json) as f:
        data = json.load(f)

    labels = []
    td_mean, td_sd, tp_mean, tp_sd = [], [], [], []
    for s in data["scenarios"]:
        labels.append(s["label"].replace("squeeze", "squez"))
        st = s["stats"]
        td_mean.append(st["T_detect"]["mean"])
        td_sd.append(st["T_detect"]["std"])
        tp_mean.append(st["T_post"]["mean"])
        tp_sd.append(st["T_post"]["std"])

    td_mean = np.array(td_mean); td_sd = np.array(td_sd)
    tp_mean = np.array(tp_mean); tp_sd = np.array(tp_sd)
    total = td_mean + tp_mean
    total_sd = np.sqrt(td_sd ** 2 + tp_sd ** 2)

    x = np.arange(len(labels))
    width = 0.55

    plt.rcParams["hatch.linewidth"] = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars_detect = ax.bar(
        x, td_mean, width, color="#fad7a8", edgecolor="#555555",
        linewidth=0.5, hatch="xxx",
        label=r"$T_{detect}$", zorder=2,
    )
    bars_post = ax.bar(
        x, tp_mean, width, bottom=td_mean,
        yerr=total_sd, capsize=4,
        color="#b9d0e8", edgecolor="#555555", linewidth=0.5,
        ecolor="#555555", label=r"$T_{post}$", zorder=2,
    )

    for bar, val in zip(bars_post, total):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=13, color="#333333")

    # Theoretical upper bound (horizontal dashed line) only — no legend
    # entry and no formula text inside the plot.
    ax.axhline(y=bound, color="#c0392b", linestyle="--", linewidth=1.6,
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Recovery latency (seconds)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Workload scenario", fontsize=15, fontweight="bold")
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, max(total.max(), bound) * 1.25)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[Plot] Saved: {args.out}")


if __name__ == "__main__":
    main()
