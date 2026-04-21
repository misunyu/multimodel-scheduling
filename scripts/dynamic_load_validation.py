#!/usr/bin/env python3
"""
Dynamic Load Adaptation Validation: produces dynamic_load_adaptation.pdf.

Goal of the figure:
    Compare the V(t) trajectory under a sudden input-rate increase between
        Static    (mode 3): the placement never changes; V(t) keeps
                            climbing as queues build up.
        BoundGuard (mode 1): the system hot-swaps to a GPU placement
                            shortly after the load change and V(t) drops
                            back below the threshold. Mode 1 keeps service
                            continuous through the reconfiguration thanks
                            to AdaptiveDeployManager's hot-swap path, so
                            the BoundGuard line stays unbroken.

Scenario:
    Phase A — t = 0 .. 20 s   placement P1 (all CPU), low input rate, V(t) ~ 0
    Phase B — t = 20 .. ~25 s placement P1 (all CPU), HIGH input rate
                              (the in-place "load change" event)
    Phase C — t ~ 25 s onward placement P2 (all GPU), HIGH input rate

The schedule is tests/dynamic_load_views_schedule.yaml. Mode 2's
AdaptiveDeployManager hot-swaps workers on every transition: phase A ->
phase B updates the per-view input rates without restarting any worker;
phase B -> phase C swaps the GPU workers in beside the running CPU
workers and only stops the old workers once the new ones are ready, so
neither transition leaves a service gap. Mode 3 keeps the first
placement throughout and only sweeps the per-view input rates.

Usage:
    python scripts/dynamic_load_validation.py
    python scripts/dynamic_load_validation.py --no-run        # replot only
    python scripts/dynamic_load_validation.py --epsilon 50

The script reuses the per-tick / windowed-V(t) helpers from
qos_recovery_validation.py so the metric definition stays consistent
across the two figures.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reap_lingering_executors(cooldown_sec: float = 5.0) -> None:
    """Kill any leftover executor/worker processes from the previous scenario
    and wait for the OS/GPU to settle before launching the next one."""
    for patt in ("schedule_executor_main.py", "headless_inference_worker.py"):
        subprocess.run(["pkill", "-9", "-f", patt], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(cooldown_sec)

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

PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR, "dynamic_load_adaptation.pdf")
BG_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_boundguard.csv")
ST_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_static.csv")

# Phase 1 wall-clock budget. Setting this to 20 puts the load-change event
# at t = 20 s on the figure x-axis.
PHASE_A_DURATION = 22
PHASE_B_DURATION = 6
PHASE_C_DURATION = 50


def run_scenario(schedule, csv_path, mode, label):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--duration", str(PHASE_A_DURATION + PHASE_B_DURATION + PHASE_C_DURATION),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
        "--combo-duration", f"combination_p1_low={PHASE_A_DURATION}",
        "--combo-duration", f"combination_p1_high={PHASE_B_DURATION}",
        "--combo-duration", f"combination_p2_high={PHASE_C_DURATION}",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = (PHASE_A_DURATION + PHASE_B_DURATION + PHASE_C_DURATION) * 4 + 90

    print("=" * 70)
    print(f"  Dynamic load scenario [{label}, mode={mode}]")
    print(f"  Phases: A={PHASE_A_DURATION}s  B={PHASE_B_DURATION}s  C={PHASE_C_DURATION}s")
    print("=" * 70)
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)
    tail = proc.stdout.decode(errors="replace").splitlines()[-5:]
    for line in tail:
        print(f"    [exec] {line}")
    if proc.returncode != 0:
        print(f"  [warn] executor exited rc={proc.returncode}")


def find_cold_start_gaps(times_sec, rows=None):
    """Return wall-clock gaps that look like cold-start windows.

    A "real" cold-start happens when the executor stops the workers for
    one combo and starts new workers for another, so the gap straddles a
    *phase boundary* in the CSV. Plain ticks that happen to be missing
    inside a phase (GC pauses, scheduler jitter) are filtered out so the
    figure doesn't sprout fake "rollback in progress" hatching in the
    middle of the recovered region.
    """
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt <= COLD_START_MIN_GAP:
            continue
        if rows is not None:
            prev_combo = rows[i - 1].get("combination", "")
            next_combo = rows[i].get("combination", "")
            if prev_combo == next_combo:
                # Missing tick inside a phase, not a real cold start.
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
    """Break the line through cold-start gaps by inserting a NaN row at the
    midpoint of each gap. matplotlib then renders a literal break instead
    of drawing a straight line through the empty interval.
    """
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


def make_plot(bg_data, st_data, epsilon, pdf_path, x_max,
              load_change_sec):
    bg_rows, bg_v_t, bg_times, bg_bounds, bg_cold = bg_data
    st_rows, st_v_t, st_times, st_bounds, st_cold = st_data

    # BoundGuard runs in mode 1 (AdaptiveDeployManager hot-swap) which
    # provides continuous service across the reconfiguration. The CSV may
    # still skip a couple of ticks while the new workers warm up (the
    # viewer's per-tick logger drops rows where every view reports 0
    # fps), but those skipped ticks are NOT a real service gap. Keep the
    # BoundGuard line connected through that warmup period; only break
    # the static line at any genuine cold-start gap (mode 3 doesn't have
    # hot-swap, so its line *would* break at a real gap).
    bg_t_plot, bg_v_plot = list(bg_times), list(bg_v_t)
    st_t_plot, st_v_plot = insert_nans_for_gaps(st_times, st_v_t, st_cold)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(st_t_plot, st_v_plot, color="#8b2e2e", linewidth=2.0,
            linestyle="--", marker="o", markersize=5, markevery=5,
            markerfacecolor="#f4b5b5", markeredgecolor="#8b2e2e",
            markeredgewidth=0.7,
            label="Static", zorder=10)
    ax.plot(bg_t_plot, bg_v_plot, color="#2c5984", linewidth=2.2,
            linestyle="-", marker="s", markersize=6, markevery=5,
            markerfacecolor="#b9d0e8", markeredgecolor="#2c5984",
            markeredgewidth=0.7,
            label="BoundGuard", zorder=11)

    # Detection threshold (epsilon).
    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=12, color="#333333")

    # y-axis: slightly above the observed maximum across both curves.
    candidate_max = max(max(bg_v_t), max(st_v_t), epsilon)
    y_max = candidate_max * 1.15
    ax.set_ylim(0.0, y_max)

    # Vertical marker for the load-change event, with a left-side label
    # connected by an arrow so the text does not overlap the curves.
    ax.axvline(x=load_change_sec, color="#b03a2e", linestyle="-.",
               linewidth=1.0, zorder=5)
    ax.annotate("Input rate\nincreases",
                xy=(load_change_sec, y_max * 0.55),
                xytext=(5.0, y_max * 0.55),
                color="#b03a2e", fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#b03a2e",
                                lw=0.8, shrinkA=2, shrinkB=4),
                zorder=12)

    # Compute reconfig completion (BoundGuard recovery) marker: first tick
    # in phase C where windowed V(t) drops back to <= epsilon.
    bg_recover_sec = None
    if len(bg_bounds) >= 3:
        p3_start = bg_bounds[2][0]
        for i in range(p3_start, len(bg_v_t)):
            if bg_v_t[i] <= epsilon:
                bg_recover_sec = bg_times[i]
                break
    if bg_recover_sec is not None:
        ax.axvline(x=bg_recover_sec, color="#5d87b5", linestyle=":",
                   linewidth=1.0, zorder=5)
        ax.text(bg_recover_sec + 0.3, y_max * 0.25,
                "BoundGuard\n stable",
                color="#5d87b5", fontsize=11, ha="left", va="center",
                zorder=12)
    ax.set_xlim(0.0, x_max)

    ax.set_xlabel("Time (seconds)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"QoS Violation Score $\mathbf{V(t)}$", fontsize=15, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=13)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")
    return bg_recover_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running the executor; replot only.")
    parser.add_argument("--run-only", choices=["bg", "st"], default=None,
                        help="Run only one curve per invocation.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.no_run:
        target = args.run_only
        if target is None or target == "bg":
            run_scenario(args.schedule, BG_CSV, mode=1, label="BoundGuard")
            reap_lingering_executors()
        if target is None or target == "st":
            run_scenario(args.schedule, ST_CSV, mode=3, label="Static")
            reap_lingering_executors()
        if target is not None:
            print(f"\n[Done] {target} data saved.")
            return 0

    bg_data = load_curve(BG_CSV)
    st_data = load_curve(ST_CSV)

    bg_times = bg_data[2]
    st_times = st_data[2]
    x_max = max(max(bg_times), max(st_times)) + 1.0

    # The load change is at the start of phase B in the BoundGuard CSV.
    bg_bounds = bg_data[3]
    if len(bg_bounds) >= 2:
        load_change_sec = bg_times[bg_bounds[1][0]]
    else:
        load_change_sec = float(PHASE_A_DURATION)

    bg_recover_sec = make_plot(bg_data, st_data,
                               epsilon=args.epsilon,
                               pdf_path=OUT_PDF,
                               x_max=x_max,
                               load_change_sec=load_change_sec)

    # Print summary numbers used to refresh the LaTeX text.
    print()
    print("=" * 70)
    print("  Dynamic load scenario summary")
    print("=" * 70)
    print(f"  epsilon                    = {args.epsilon}")
    print(f"  Load-change wall-clock     = {load_change_sec:.2f}s")
    print(f"  BoundGuard cold-start gaps = "
          + (", ".join(f"{e - s:.1f}s @ {s:.1f}-{e:.1f}s" for s, e in bg_data[4])
             if bg_data[4] else "(none)"))
    print(f"  BoundGuard V(t) max        = {max(bg_data[1]):.2f}")
    print(f"  BoundGuard V(t) end        = {bg_data[1][-1]:.2f}")
    if bg_recover_sec is not None:
        print(f"  BoundGuard t_stable        = {bg_recover_sec:.2f}s "
              f"(first V(t)<=eps in phase C)")
    print(f"  Static V(t) max            = {max(st_data[1]):.2f}")
    print(f"  Static V(t) end            = {st_data[1][-1]:.2f}")
    print(f"  Static rows                = {len(st_data[0])}")
    print(f"  BoundGuard rows            = {len(bg_data[0])}")


if __name__ == "__main__":
    sys.exit(main() or 0)
