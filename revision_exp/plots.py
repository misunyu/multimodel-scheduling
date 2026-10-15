"""Figures for the retrained pipeline. Same size / axes / legend style as scripts/plot_model_count_vs_score.py and
scripts/plot_alpha_comparison.py, but every output path is an argument (nothing is written outside the given directory)."""
import sys

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402


def _style():
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"] + [f for f in rcParams["font.serif"] if f != "Times New Roman"]
    rcParams["hatch.linewidth"] = 0.3


def fig3(df, out_pdf, out_png):
    """df: rows N>= in {3..7}; columns Oracle, Ours, Best-of-5, Best-of-5_std, Latency-based, All-GPU."""
    _style()
    d = df[df["N>="].between(3, 7)].sort_values("N>=")
    x = d["N>="].values
    plt.figure(figsize=(3.2, 2.5))
    plt.plot(x, d["Oracle"], marker="D", label="Oracle", color="black", linewidth=0.5, markersize=2, linestyle="--")
    plt.plot(x, d["Ours"], marker="o", label="Ours", color="tab:red", linewidth=0.5, markersize=2)
    plt.plot(x, d["All-GPU"], marker="v", label="All-GPU", color="tab:orange", linewidth=0.5, markersize=2, linestyle=":")
    plt.plot(x, d["Best-of-5"], marker="^", label="Best-of-5", color="forestgreen", linewidth=0.5, markersize=2)
    plt.fill_between(x, d["Best-of-5"] - d["Best-of-5_std"], d["Best-of-5"] + d["Best-of-5_std"], color="forestgreen", alpha=0.15, linewidth=0)
    plt.plot(x, d["Latency-based"], marker="s", label="Latency-based", color="tab:blue", linewidth=0.5, markersize=2)
    plt.xlabel("Number of Application Instances (>= N)", fontsize=6)
    plt.ylabel("Average Score", fontsize=6)
    plt.title("Score vs. Number of Application Instances", fontsize=7)
    plt.xticks(x, fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True, linestyle="--", alpha=0.7, linewidth=0.1)
    ax = plt.gca()
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=5, frameon=False, columnspacing=0.8, handletextpad=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close()


def fig4(df, out_pdf, out_png):
    """df: one row per alpha with avg/top1/top5 for multi and single plus oracle_avg."""
    _style()
    d = df.sort_values("alpha")
    alphas = d["alpha"].values
    x = np.arange(len(alphas))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.75))
    panels = [("avg_score", "Score"), ("top1", "Top-1 Accuracy"), ("top5", "Top-5 Accuracy")]
    for i, (col, ylabel) in enumerate(panels):
        ax = axes[i]
        ax.bar(x - width / 2, d[f"{col}_multi"], width, label="Multi-Objective Predictor", color="skyblue", alpha=0.5, hatch="//",
               edgecolor="black", linewidth=0.3)
        ax.bar(x + width / 2, d[f"{col}_single"], width, label="Single-Objective Predictor", color="lightcoral", alpha=0.5, hatch="..",
               edgecolor="black", linewidth=0.3)
        if col == "avg_score":
            ax.plot(x, d["oracle_avg"], color="black", linestyle=":", linewidth=0.8, marker="D", markersize=2, label="Oracle")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(alphas, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel("Alpha", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth=0.15)
        for s in ax.spines.values():
            s.set_linewidth(0.15)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=120)
    plt.close()
