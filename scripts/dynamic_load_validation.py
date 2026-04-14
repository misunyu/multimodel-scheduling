#!/usr/bin/env python3
"""
Dynamic Load Adaptation Validation: produces dynamic_load_adaptation.pdf.

Goal of the figure:
    Compare the V(t) trajectory under a sudden input-rate increase
    between three execution modes driven by the same schedule YAML
    (tests/dynamic_load_views_schedule_npu.yaml):

        Static           (mode 3): placement never changes; V(t) keeps
                                   climbing as queues build up on the
                                   CPU-resident YOLO pair.
        Stop-and-restart (mode 0): phase_b -> phase_c triggers a worker
                                   stop-and-start; V(t) drops after the
                                   cold-start gap.
        Hot-swap         (mode 1): AdaptiveDeployManager brings the new
                                   placement up in parallel and swaps
                                   atomically, so service stays
                                   continuous through the reconfig and
                                   V(t) drops without a visible gap.

Scenario:
    Phase A — t = 0 .. ~22 s  placement P1, low input rate, V(t) ~ 0
    Phase B — t = 22 .. ~28 s placement P1, HIGH input rate
                              (the in-place "load change" event)
    Phase C — t ~ 28 s onward placement P2, HIGH input rate

This is a PLOT-ONLY script. Each V(t) curve must come from a data-
collection run that is fully isolated (separate Docker container, all
NPU drivers torn down between runs, same YAML schedule so input rates
and placements match across techniques). The canonical runner is
run_dynload_scenarios.sh — it sequences the three scenarios, kills
each one before starting the next, and then invokes this script to
produce the two PDFs from the three saved CSVs.

Usage:
    ./run_dynload_scenarios.sh            # run all three + plot
    ./run_dynload_scenarios.sh plot       # replot only (CSVs already present)
    python scripts/dynamic_load_validation.py --epsilon 10   # replot alone
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from qos_recovery_validation import (   # noqa: E402
    compute_windowed_v,
    find_phase_boundaries,
    load_csv,
    parse_timestamps,
    WINDOW_T,
    COLD_START_MIN_GAP,
)

DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule_npu.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR, "dynamic_load_adaptation_antara.pdf")
OUT_SAS_PDF = os.path.join(RESULTS_DIR, "dynamic_load_start_and_stop_antara.pdf")
ST_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_static.csv")
SAS_CSV = os.path.join(RESULTS_DIR, "dynamic_load_stop_and_restart.csv")
HS_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_hotswap.csv")

PHASE_A_DURATION = 18
PHASE_B_DURATION = 14
PHASE_C_DURATION = 20


def find_cold_start_gaps(times_sec, rows=None):
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt <= COLD_START_MIN_GAP:
            continue
        if rows is not None:
            prev_combo = rows[i - 1].get("combination", "")
            next_combo = rows[i].get("combination", "")
            if prev_combo == next_combo:
                continue
        gaps.append((times_sec[i - 1], times_sec[i]))
    return gaps


def load_curve(csv_path):
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times_sec = parse_timestamps(rows)
    boundaries = find_phase_boundaries(rows)
    cold_starts = find_cold_start_gaps(times_sec, rows=rows)
    return rows, v_t, times_sec, boundaries, cold_starts


def insert_nans_for_gaps(times_sec, v_t, cold_starts):
    if not cold_starts:
        return list(times_sec), list(v_t)
    out_t = []
    out_v = []
    gap_ends = {round(end, 3) for _, end in cold_starts}
    for i, (t, v) in enumerate(zip(times_sec, v_t)):
        if i > 0 and round(t, 3) in gap_ends:
            mid = (times_sec[i - 1] + t) / 2.0
            out_t.append(mid)
            out_v.append(float("nan"))
        out_t.append(t)
        out_v.append(v)
    return out_t, out_v


def _draw_gap_connectors(ax, times_sec, v_t, cold_starts, color):
    """Across each cold-start gap, draw a thin grey dotted line linking
    the last pre-gap and first post-gap sample so the eye can still
    follow the curve through the break."""
    if not cold_starts:
        return
    gap_ends = {round(end, 3) for _, end in cold_starts}
    for i in range(1, len(times_sec)):
        if round(times_sec[i], 3) not in gap_ends:
            continue
        ax.plot([times_sec[i - 1], times_sec[i]],
                [v_t[i - 1], v_t[i]],
                color=color, linestyle=":", linewidth=0.9,
                zorder=6, label="_nolegend_")


def _recover_time(v_t, times, boundaries, epsilon):
    if len(boundaries) < 3:
        return None
    p3_start = boundaries[2][0]
    for i in range(p3_start, len(v_t)):
        if v_t[i] <= epsilon:
            return times[i]
    return None


def make_combined_plot(st_data, sas_data, hs_data, epsilon, pdf_path,
                       x_max, load_change_sec):
    st_rows, st_v_t, st_times, st_bounds, st_cold = st_data
    sas_rows, sas_v_t, sas_times, sas_bounds, sas_cold = sas_data
    hs_rows, hs_v_t, hs_times, hs_bounds, hs_cold = hs_data

    st_t_plot, st_v_plot = insert_nans_for_gaps(st_times, st_v_t, st_cold)
    sas_t_plot, sas_v_plot = insert_nans_for_gaps(sas_times, sas_v_t, sas_cold)
    # Hot-swap keeps service continuous; do not break its line even if
    # the logger drops a couple of ticks during warmup.
    hs_t_plot, hs_v_plot = list(hs_times), list(hs_v_t)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    ax.plot(st_t_plot, st_v_plot, color="#a83232", linewidth=2.0,
            label="Static", zorder=10)
    ax.plot(sas_t_plot, sas_v_plot, color="#d98e00", linewidth=2.0,
            label="Stop-and-restart", zorder=11)
    ax.plot(hs_t_plot, hs_v_plot, color="#1f4e79", linewidth=2.0,
            label="BoundGuard", zorder=12)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=12, color="#333333")

    ax.axvline(x=load_change_sec, color="#7b3306", linestyle=":",
               linewidth=1.4, zorder=5)
    ax.text(load_change_sec + 0.3, 5,
            "Input rate\nincreases",
            color="#7b3306", fontsize=8.5, ha="left", va="bottom",
            zorder=13)

    _draw_gap_connectors(ax, sas_times, sas_v_t, sas_cold, "#888888")

    sas_recover = _recover_time(sas_v_t, sas_times, sas_bounds, epsilon)
    hs_recover = _recover_time(hs_v_t, hs_times, hs_bounds, epsilon)
    if sas_recover is not None:
        ax.axvline(x=sas_recover, color="#8a5a00", linestyle=":",
                   linewidth=1.1, zorder=5)
    if hs_recover is not None:
        ax.axvline(x=hs_recover, color="#155724", linestyle=":",
                   linewidth=1.1, zorder=5)

    candidate_max = max(max(st_v_t), max(sas_v_t), max(hs_v_t))
    y_max = max(epsilon * 2.4, candidate_max * 1.05)
    y_max = min(y_max, 220.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, x_max)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel(r"QoS Violation Score $V(t)$", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")
    return sas_recover, hs_recover


def make_start_and_stop_plot(st_data, sas_data, epsilon, pdf_path,
                             x_max, load_change_sec):
    """Focused figure: Static vs Stop-and-restart only. Highlights the
    visible cold-start gap that the hot-swap mode eliminates."""
    st_rows, st_v_t, st_times, st_bounds, st_cold = st_data
    sas_rows, sas_v_t, sas_times, sas_bounds, sas_cold = sas_data

    st_t_plot, st_v_plot = insert_nans_for_gaps(st_times, st_v_t, st_cold)
    sas_t_plot, sas_v_plot = insert_nans_for_gaps(sas_times, sas_v_t, sas_cold)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(st_t_plot, st_v_plot, color="#a83232", linewidth=2.0,
            label="Static", zorder=10)
    ax.plot(sas_t_plot, sas_v_plot, color="#d98e00", linewidth=2.0,
            label="Stop-and-restart", zorder=11)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=12, color="#333333")

    ax.axvline(x=load_change_sec, color="#7b3306", linestyle=":",
               linewidth=1.4, zorder=5)
    ax.text(load_change_sec + 0.3, 5,
            "Input rate\nincreases",
            color="#7b3306", fontsize=8.5, ha="left", va="bottom", zorder=12)

    _draw_gap_connectors(ax, sas_times, sas_v_t, sas_cold, "#888888")

    candidate_max = max(max(st_v_t), max(sas_v_t))
    y_max = max(epsilon * 2.4, candidate_max * 1.05)
    y_max = min(y_max, 220.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, x_max)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel(r"QoS Violation Score $V(t)$", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot dynamic-load V(t) curves from three CSVs "
                    "produced by independent run_dynload_scenarios.sh runs.")
    parser.add_argument("--epsilon", type=float, default=0.25)
    # --no-run retained as a no-op for backwards compatibility.
    parser.add_argument("--no-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    missing = [p for p in (ST_CSV, SAS_CSV, HS_CSV) if not os.path.exists(p)]
    if missing:
        print("Missing CSV(s); run ./run_dynload_scenarios.sh first:\n  "
              + "\n  ".join(missing))
        return 1

    st_data  = load_curve(ST_CSV)
    sas_data = load_curve(SAS_CSV)
    hs_data  = load_curve(HS_CSV)

    x_max = max(max(st_data[2]), max(sas_data[2]), max(hs_data[2])) + 1.0

    # Load-change wall-clock: use the start of phase_b in the stop-and-
    # restart CSV (which has clean phase boundaries recorded).
    sas_bounds = sas_data[3]
    if len(sas_bounds) >= 2:
        load_change_sec = sas_data[2][sas_bounds[1][0]]
    else:
        load_change_sec = float(PHASE_A_DURATION)

    sas_recover, hs_recover = make_combined_plot(
        st_data, sas_data, hs_data,
        epsilon=args.epsilon, pdf_path=OUT_PDF,
        x_max=x_max, load_change_sec=load_change_sec)

    make_start_and_stop_plot(
        st_data, sas_data,
        epsilon=args.epsilon, pdf_path=OUT_SAS_PDF,
        x_max=x_max, load_change_sec=load_change_sec)

    print()
    print("=" * 70)
    print("  Dynamic load scenario summary")
    print("=" * 70)
    print(f"  epsilon                       = {args.epsilon}")
    print(f"  Load-change wall-clock        = {load_change_sec:.2f}s")
    print(f"  Stop-and-restart cold-starts  = "
          + (", ".join(f"{e - s:.1f}s @ {s:.1f}-{e:.1f}s" for s, e in sas_data[4])
             if sas_data[4] else "(none)"))
    print(f"  Static V(t) max/end           = {max(st_data[1]):.2f} / {st_data[1][-1]:.2f}")
    print(f"  Stop-and-restart V(t) max/end = {max(sas_data[1]):.2f} / {sas_data[1][-1]:.2f}")
    print(f"  Hot-swap V(t) max/end         = {max(hs_data[1]):.2f} / {hs_data[1][-1]:.2f}")
    if sas_recover is not None:
        print(f"  Stop-and-restart t_stable     = {sas_recover:.2f}s")
    if hs_recover is not None:
        print(f"  Hot-swap t_stable             = {hs_recover:.2f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
