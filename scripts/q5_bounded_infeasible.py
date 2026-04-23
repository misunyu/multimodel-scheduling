#!/usr/bin/env python3
"""
Q5: Bounded behavior when no QoS-satisfying placement exists.

Produces results/q5_bounded_infeasible.pdf — a single-panel V(t) time
series comparing Adaptive hot-swap and BoundGuard on the heaviest
infeasible scenario (S6, +TinyLlama background), where none of the
top-5 candidates brings V(t) back below epsilon.

The figure highlights:
  * Adaptive commits to top-1 and, absent any feasible fallback,
    suffers an unbounded spike in V(t).
  * BoundGuard cycles the top-5 candidates in bounded T_v windows.
    Its V(t) oscillates but stays within a small envelope — failure
    impact is bounded even when recovery is impossible.

Usage:
    python scripts/q5_bounded_infeasible.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from qos_recovery_validation import compute_windowed_v, load_csv, WINDOW_T  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
Q13_DIR     = os.path.join(RESULTS_DIR, "q13_persistence")
OUT_PDF     = os.path.join(RESULTS_DIR, "q5_bounded_infeasible.pdf")

# Pastel palette — matches q13_failure_persistence.pdf.
COLOR_ADAPTIVE   = "#fad7a8"
COLOR_BOUNDGUARD = "#b9d0e8"
EDGE_ADAPTIVE    = "#c88a44"
EDGE_BOUNDGUARD  = "#5d87b5"

# Defaults: scenario 6 traces already produced by q13_failure_persistence.
DEFAULT_ADAPTIVE_CSV   = os.path.join(Q13_DIR, "s6_adaptive.csv")
DEFAULT_BOUNDGUARD_CSV = os.path.join(Q13_DIR, "s6_boundguard.csv")

# These phase durations are the ones used in q13_failure_persistence.py.
P_STABLE = 22
P_BURST  = 8


def _combo_transitions(rows):
    trans = []
    last = None
    for i, r in enumerate(rows):
        c = r["combination"]
        if c != last:
            trans.append((i, c))
            last = c
    return trans


def plot(adaptive_csv, boundguard_csv, out_pdf, epsilon=1.0):
    rows_a = load_csv(adaptive_csv)
    rows_b = load_csv(boundguard_csv)
    if not rows_a or not rows_b:
        raise RuntimeError("empty input CSV")

    v_a = compute_windowed_v(rows_a, T=WINDOW_T)
    v_b = compute_windowed_v(rows_b, T=WINDOW_T)
    t_a = np.arange(len(v_a))
    t_b = np.arange(len(v_b))

    trans_b = _combo_transitions(rows_b)
    burst_start = P_STABLE
    search_start = burst_start + P_BURST

    y_max = max(max(v_a), max(v_b)) * 1.15

    plt.rcParams["hatch.linewidth"] = 0.4
    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    ax.axvspan(search_start, max(t_a[-1], t_b[-1]),
               color="#f2f2f2", zorder=0,
               label="No feasible placement region")

    ax.axhline(epsilon, color="#555555", linestyle="--", linewidth=0.9,
               zorder=1, label=r"Threshold $\epsilon$")

    for i, c in trans_b:
        if not c.startswith("cand_"):
            continue
        ax.axvline(i, color="#8a8a8a", linestyle=":", linewidth=0.7, zorder=1)

    ax.plot(t_a, v_a, color=EDGE_ADAPTIVE, linewidth=1.8, zorder=3,
            label="Adaptive hot-swap", marker="o", markersize=3.5,
            markerfacecolor=COLOR_ADAPTIVE, markeredgecolor=EDGE_ADAPTIVE,
            markeredgewidth=0.5)
    ax.plot(t_b, v_b, color=EDGE_BOUNDGUARD, linewidth=1.8, zorder=3,
            label="BoundGuard", marker="s", markersize=3.5,
            markerfacecolor=COLOR_BOUNDGUARD, markeredgecolor=EDGE_BOUNDGUARD,
            markeredgewidth=0.5)

    # Annotate BoundGuard candidate labels above the top of the plot.
    cand_seq = [(i, c) for i, c in trans_b if c.startswith("cand_")]
    for i, c in cand_seq:
        label = c.replace("cand_", "c")
        ax.text(i + 0.1, y_max * 0.97, label, fontsize=8,
                color="#5d87b5", ha="left", va="top",
                rotation=0)

    # Envelope band around BoundGuard to emphasize bounded behavior.
    if cand_seq:
        s = cand_seq[0][0]
        e = min(cand_seq[-1][0] + 6, len(v_b))
        lo = min(v_b[s:e])
        hi = max(v_b[s:e])
        ax.axhspan(lo, hi, xmin=(s / ax.get_xlim()[1] if False else 0),
                   xmax=1, color="#b9d0e8", alpha=0.0)  # placeholder; we use hlines below
        ax.hlines([lo, hi], s, e, colors="#5d87b5", linestyles=":",
                  linewidth=0.7, zorder=2)
        ax.annotate(f"bounded envelope\n[{lo:.1f}, {hi:.1f}]",
                    xy=(e - 1, hi), xytext=(e + 1.5, hi + 0.4),
                    fontsize=9, color="#5d87b5",
                    arrowprops=dict(arrowstyle="->", color="#5d87b5",
                                    linewidth=0.7, shrinkA=2, shrinkB=2))

    # Annotate Adaptive peak.
    peak_i = int(np.argmax(v_a))
    peak_v = v_a[peak_i]
    ax.annotate(f"unbounded spike\n$V_{{peak}}={peak_v:.1f}$",
                xy=(peak_i, peak_v), xytext=(peak_i - 8, peak_v + 0.2),
                fontsize=9, color="#c88a44",
                arrowprops=dict(arrowstyle="->", color="#c88a44",
                                linewidth=0.7, shrinkA=2, shrinkB=2))

    ax.set_xlim(0, max(t_a[-1], t_b[-1]))
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"QoS violation score $V(t)$", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"[Plot] Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-csv",   default=DEFAULT_ADAPTIVE_CSV)
    parser.add_argument("--boundguard-csv", default=DEFAULT_BOUNDGUARD_CSV)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--out", default=OUT_PDF)
    args = parser.parse_args()
    plot(args.adaptive_csv, args.boundguard_csv, args.out, args.epsilon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
