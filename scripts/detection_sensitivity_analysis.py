#!/usr/bin/env python3
"""
Detection sensitivity analysis: produces detection_sensitivity.pdf.

Goal:
    Show how BoundGuard's two detection knobs -- the sliding-window
    length T and the violation-score threshold epsilon -- jointly
    affect (a) false positives in steady state, (b) detection latency,
    (c) recovery latency, and (d) the total accumulated QoS violation.

Approach:
    Re-use the per-tick `v_score` column already collected by the
    bounded-recovery sweep (results/bounded_sweep/run_s*_rep*.csv) and
    re-window it offline for every (T, eps) combination on a 3 x 3
    grid:

        T   in {1, 3, 5}    seconds (sliding-window length)
        eps in {0.5, 1.0, 2.0}      (threshold, centered on the
                                     operating ε = 1.0 used by the
                                     bounded-recovery sweep)

    For every cell we compute four metrics, averaged over all CSV
    runs from the sweep (6 scenarios x 3 reps = 18 runs), and plot
    them as four 3x3 heatmaps in a single figure.

Usage:
    python scripts/detection_sensitivity_analysis.py
"""
import argparse
import csv
import datetime
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Pastel heatmap palettes — white ramps toward the q13 pastel endpoints
# (pink, peach, blue, purple) so heatmap tones match the other paper figures.
_PASTEL_CMAPS = {
    "pastel_red":    LinearSegmentedColormap.from_list("pastel_red",
                        ["#ffffff", "#f4b5b5", "#c96b6b"]),
    "pastel_peach":  LinearSegmentedColormap.from_list("pastel_peach",
                        ["#ffffff", "#fad7a8", "#c88a44"]),
    "pastel_blue":   LinearSegmentedColormap.from_list("pastel_blue",
                        ["#ffffff", "#b9d0e8", "#5d87b5"]),
    "pastel_purple": LinearSegmentedColormap.from_list("pastel_purple",
                        ["#ffffff", "#d8c2ea", "#8a6aae"]),
}
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SWEEP_DIR = os.path.join(RESULTS_DIR, "bounded_sweep")
OUT_PDF = os.path.join(RESULTS_DIR, "detection_sensitivity.pdf")

T_VALUES   = [1, 3, 5]
EPS_VALUES = [0.5, 1.0, 2.0]


def load_run(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_per_tick = [float(r.get("v_score", 0) or 0) for r in rows]
    combos = [r.get("combination", "") for r in rows]
    return rows, v_per_tick, combos


def windowed(v_per_tick, T):
    out = []
    for i in range(len(v_per_tick)):
        win = v_per_tick[max(0, i - T + 1): i + 1]
        out.append(sum(win) / len(win) if win else 0.0)
    return out


def phase_indices(combos):
    """Return (p1_range, p2_range, p3_range) as Python ranges."""
    boundaries = []
    if not combos:
        return range(0), range(0), range(0)
    boundaries.append(0)
    for i in range(1, len(combos)):
        if combos[i] != combos[i - 1]:
            boundaries.append(i)
    boundaries.append(len(combos))
    if len(boundaries) < 4:
        # fewer than 3 phases -- pad with empty ranges
        while len(boundaries) < 4:
            boundaries.append(boundaries[-1])
    return (range(boundaries[0], boundaries[1]),
            range(boundaries[1], boundaries[2]),
            range(boundaries[2], boundaries[3]))


def compute_metrics_for_run(v_per_tick, combos, T, eps):
    """Return dict with the 4 sensitivity metrics for one run."""
    v_t = windowed(v_per_tick, T)
    p1, p2, p3 = phase_indices(combos)

    # 1. False triggers: how many phase-1 (healthy) ticks have V(t) > eps.
    false_triggers = sum(1 for i in p1 if v_t[i] > eps)

    # 2. Detection latency: ticks from start of phase 2 to first
    #    windowed V(t) > eps with at least T post-failure samples in
    #    the window.
    p2_start = p2.start if len(p2) > 0 else None
    detect_lat = None
    if p2_start is not None:
        earliest = p2_start + T - 1
        for i in range(earliest, p2.stop):
            if v_t[i] > eps:
                detect_lat = i - p2_start
                break

    # 3. Recovery latency: ticks from start of phase 2 to first phase-3
    #    tick where windowed V(t) <= eps.
    recover_lat = None
    if p2_start is not None and len(p3) > 0:
        for i in p3:
            if v_t[i] <= eps:
                recover_lat = i - p2_start
                break

    # 4. Cumulative QoS violation: sum_t max(0, V(t) - eps) across the
    #    entire run. Larger == more total degradation experienced
    #    before recovery (or never-recovered tail under high eps).
    cum_violation = sum(max(0.0, v - eps) for v in v_t)

    return {
        "false_triggers": float(false_triggers),
        "detect_lat": float(detect_lat) if detect_lat is not None else float("nan"),
        "recover_lat": float(recover_lat) if recover_lat is not None else float("nan"),
        "cum_violation": float(cum_violation),
        "rows": len(v_per_tick),
    }


def aggregate(grid_runs):
    """grid_runs[t][eps] = list of per-run metric dicts -> dict of mean arrays."""
    out = {}
    metric_names = ["false_triggers", "detect_lat", "recover_lat", "cum_violation"]
    for m in metric_names:
        arr = np.full((len(T_VALUES), len(EPS_VALUES)), np.nan)
        for ti in range(len(T_VALUES)):
            for ei in range(len(EPS_VALUES)):
                vals = [r[m] for r in grid_runs[ti][ei]
                        if not (isinstance(r[m], float) and np.isnan(r[m]))]
                if vals:
                    arr[ti, ei] = float(np.mean(vals))
        out[m] = arr
    return out


def render_heatmap(ax, data, title, fmt, cmap, lower_is_better=True,
                   highlight_min=True, vmin=None, vmax=None):
    arr = data
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    cmap_obj = _PASTEL_CMAPS.get(cmap, cmap) if isinstance(cmap, str) else cmap
    im = ax.imshow(arr, cmap=cmap_obj, aspect="auto",
                   vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(EPS_VALUES)))
    ax.set_xticklabels([f"$\\epsilon$={e}" for e in EPS_VALUES], fontsize=13)
    ax.set_yticks(range(len(T_VALUES)))
    ax.set_yticklabels([f"T={t}s" for t in T_VALUES], fontsize=13)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    # Annotate cells.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                txt = "n/a"
            else:
                txt = fmt.format(v)
            # Choose text color based on cell luminance.
            cell_norm = (v - vmin) / max(vmax - vmin, 1e-9)
            tc = "white" if cell_norm > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-glob",
                        default=os.path.join(SWEEP_DIR, "run_s*_rep*.csv"))
    parser.add_argument("--out", default=OUT_PDF)
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.sweep_glob))
    if not csv_paths:
        print(f"[Error] no sweep CSVs found at {args.sweep_glob}")
        return 1
    print(f"[Sensitivity] found {len(csv_paths)} CSV files")

    grid_runs = [[[] for _ in EPS_VALUES] for _ in T_VALUES]
    runs_meta = []
    for path in csv_paths:
        try:
            rows, v_per_tick, combos = load_run(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        for ti, T in enumerate(T_VALUES):
            for ei, eps in enumerate(EPS_VALUES):
                m = compute_metrics_for_run(v_per_tick, combos, T, eps)
                grid_runs[ti][ei].append(m)
        runs_meta.append({"csv": os.path.basename(path), "rows": len(rows)})

    agg = aggregate(grid_runs)

    # ---------- figure layout ------------------------------------------------
    # Row 1: Detection latency + Recovery latency (side-by-side)
    # Row 2: Cumulative violation (centered on the second row)
    fig = plt.figure(figsize=(9.0, 6.6))
    gs = GridSpec(2, 4, figure=fig,
                  hspace=0.30, wspace=0.60,
                  top=0.94, bottom=0.07, left=0.08, right=0.95)

    ax2 = fig.add_subplot(gs[0, 0:2])
    ax3 = fig.add_subplot(gs[0, 2:4])
    ax4 = fig.add_subplot(gs[1, 1:3])

    render_heatmap(ax2, agg["detect_lat"],
                   "Detection latency (ticks)",
                   fmt="{:.1f}",
                   cmap="pastel_blue")
    render_heatmap(ax3, agg["recover_lat"],
                   "Recovery latency (ticks)",
                   fmt="{:.1f}",
                   cmap="pastel_peach")
    render_heatmap(ax4, agg["cum_violation"],
                   r"Cumulative violation $\sum_t (V(t)-\epsilon)^+$",
                   fmt="{:.0f}",
                   cmap="pastel_purple")

    # fig.suptitle removed for paper figure

    fig.savefig(args.out)
    print(f"[Plot] Saved: {args.out}")

    # Also dump a JSON summary for traceability.
    summary = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "sources": runs_meta,
        "T_values": T_VALUES,
        "eps_values": EPS_VALUES,
        "metrics": {
            k: agg[k].tolist() for k in agg
        },
    }
    sum_path = os.path.splitext(args.out)[0] + ".json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Plot] Summary JSON: {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
