#!/usr/bin/env python3
"""
Generate bounded_recovery_analysis_accumulation.pdf — stacked bar chart
showing T_detect (orange) + T_post (dark blue) per workload scenario.

Usage:
    python scripts/bounded_recovery_accumulation_plot.py
    python scripts/bounded_recovery_accumulation_plot.py --from-json results/bounded_sweep/sweep.json
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
OUT_PDF = os.path.join(RESULTS_DIR, "bounded_recovery_analysis_accumulation.pdf")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json",
                        default=os.path.join(RESULTS_DIR, "bounded_sweep", "sweep.json"),
                        help="Path to sweep JSON file.")
    args = parser.parse_args()

    with open(args.from_json) as f:
        data = json.load(f)

    labels = []
    td_mean = []
    td_sd = []
    tp_mean = []
    tp_sd = []

    for s in data["scenarios"]:
        labels.append(s["label"].replace("squeeze", "squez"))
        st = s["stats"]
        td_mean.append(st["T_detect"]["mean"])
        td_sd.append(st["T_detect"]["std"])
        tp_mean.append(st["T_post"]["mean"])
        tp_sd.append(st["T_post"]["std"])

    td_mean = np.array(td_mean)
    td_sd = np.array(td_sd)
    tp_mean = np.array(tp_mean)
    tp_sd = np.array(tp_sd)
    total = td_mean + tp_mean
    total_sd = np.sqrt(td_sd ** 2 + tp_sd ** 2)

    x = np.arange(len(labels))
    width = 0.55

    plt.rcParams["hatch.linewidth"] = 0.4

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # T_detect (bottom, pastel peach, cross hatch)
    bars_detect = ax.bar(
        x, td_mean, width,
        color="#fad7a8",
        edgecolor="#555555",
        linewidth=0.5,
        hatch="xxx",
        label=r"$T_{detect}$",
        zorder=2,
    )

    # T_post (top, pastel blue, solid) stacked on T_detect
    bars_post = ax.bar(
        x, tp_mean, width,
        bottom=td_mean,
        yerr=total_sd,
        capsize=4,
        color="#b9d0e8",
        edgecolor="#555555",
        linewidth=0.5,
        ecolor="#555555",
        label=r"$T_{post}$",
        zorder=2,
    )

    # Total labels above each bar
    for i, (bar, val) in enumerate(zip(bars_post, total)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=13, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Recovery latency (seconds)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Workload scenario", fontsize=15, fontweight="bold")
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, total.max() * 1.20)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"[Plot] Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
