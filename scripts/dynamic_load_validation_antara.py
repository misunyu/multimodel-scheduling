#!/usr/bin/env python3
"""
Dynamic Load Adaptation Validation (NPU/Antara): produces dynamic_load_adaptation_antara.pdf.

Four-curve figure (same style as ml_misprediction_fallback.pdf):
    Static            (mode 3): placement never changes; V(t) keeps climbing.
    Stop-and-restart  (mode 0): tears down workers and restarts.
    Adaptive (ML-only)(mode 1): hot-swaps to ML-predicted (suboptimal) placement;
                                stays there — V(t) remains elevated.
    BoundGuard        (mode 1): same ML pick, but after T=3s detects V(t)>ε
                                and falls back to optimal placement.

Scenario (tests/dynamic_load_views_schedule_npu.yaml):
    phase_a  all CPU, low input rate, V(t) ~ 0
    phase_b  all CPU, HIGH input rate (overload), V(t) >> ε
    phase_c  ML pick: resnet50 → NPU (wrong choice), yolov3 stays CPU
             V(t) stays > ε because yolov3 is the bottleneck
    phase_d  BoundGuard fallback: yolov3 → NPU (correct), resnet50 → CPU
             V(t) drops below ε

Usage:
    python scripts/dynamic_load_validation_antara.py --epsilon 2.0
    python scripts/dynamic_load_validation_antara.py --no-run --epsilon 2.0
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

PYTHON = sys.executable

DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule_npu.yaml")
MLONLY_SCHEDULE  = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule_npu_mlonly.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR, "dynamic_load_adaptation_antara.pdf")
BG_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_antara_boundguard.csv")
ML_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_antara_adaptive.csv")
SR_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_antara_stoprestart.csv")
ST_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_antara_static.csv")

PHASE_A_DURATION = 22   # warm-up (low infps, V(t)=0 throughout)
PHASE_B_MAX = 30        # max cap; phase_b ends on V(t)>ε (usually ≤ ~10s)
PHASE_C_MAX = 20        # max cap; phase_c ends on V(t)>ε after hold + T window
PHASE_D_DURATION = 40   # BoundGuard final observation (fixed, last combo)
NPU_INIT_SEC = 5        # NPU initialization / stabilization (excluded from T/T_v)
TV_WINDOW_SEC = 3       # validation window T_v (fill time before V(t) check)


def reap_lingering_executors(cooldown_sec: float = 5.0) -> None:
    for patt in ("schedule_executor_main.py", "headless_inference_worker.py"):
        subprocess.run(["pkill", "-9", "-f", patt], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(cooldown_sec)


def run_scenario(schedule, csv_path, mode, label, combo_durations,
                  trigger_epsilon=None, trigger_hold=None):
    """Run one scenario via schedule_executor_main.py.

    When `trigger_epsilon` is set, phase transitions fire on windowed V(t) > ε
    (after `trigger_hold[combo]` seconds of NPU-init stabilization that are
    excluded from T). `combo_durations` acts as the MAX cap per phase.
    """
    if os.path.exists(csv_path):
        os.remove(csv_path)
    total = sum(combo_durations.values())
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--duration", str(total),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
    ]
    for combo, dur in combo_durations.items():
        cmd += ["--combo-duration", f"{combo}={dur}"]
    if trigger_epsilon is not None:
        cmd += ["--phase-trigger-vscore", str(trigger_epsilon)]
    for combo, hold in (trigger_hold or {}).items():
        cmd += ["--phase-trigger-hold", f"{combo}={hold}"]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = total * 6 + 120

    print("=" * 70)
    print(f"  Dynamic load scenario [{label}, mode={mode}]")
    print(f"  Durations (max cap): {combo_durations}")
    if trigger_epsilon is not None:
        print(f"  V(t)-trigger: ε={trigger_epsilon}  hold={trigger_hold}")
    print("=" * 70)
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        tail = stdout.decode(errors="replace").splitlines()[-5:]
        for line in tail:
            print(f"    [exec] {line}")
        if proc.returncode != 0:
            print(f"  [warn] executor exited rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [info] Timeout after {timeout}s — killing executor")
        proc.kill()
        proc.wait()


def find_cold_start_gaps(times_sec, rows=None):
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt <= COLD_START_MIN_GAP:
            continue
        if rows is not None:
            if rows[i - 1].get("combination", "") == rows[i].get("combination", ""):
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
    out_t, out_v = [], []
    gap_ends = {round(end, 3) for _, end in cold_starts}
    for i, (t, v) in enumerate(zip(times_sec, v_t)):
        if i > 0 and round(t, 3) in gap_ends:
            mid = (times_sec[i - 1] + t) / 2.0
            out_t.append(mid)
            out_v.append(float("nan"))
        out_t.append(t)
        out_v.append(v)
    return out_t, out_v


def _trim_to_cutoff(times, values, cutoff):
    if cutoff is None:
        return list(times), list(values)
    out_t, out_v = [], []
    for t, v in zip(times, values):
        if t <= cutoff:
            out_t.append(t)
            out_v.append(v)
    return out_t, out_v


def make_plot(bg_data, ml_data, sr_data, st_data, epsilon, pdf_path, x_max,
              load_change_sec, data_cutoff=None):
    bg_rows, bg_v_t, bg_times, bg_bounds, bg_cold = bg_data
    ml_rows, ml_v_t, ml_times, ml_bounds, ml_cold = ml_data
    sr_rows, sr_v_t, sr_times, sr_bounds, sr_cold = sr_data
    st_rows, st_v_t, st_times, st_bounds, st_cold = st_data

    # Prepare plot data (optionally truncated to `data_cutoff` seconds so the
    # curves stop drawing at `data_cutoff` even though the axis extends to x_max)
    bg_t_plot, bg_v_plot = _trim_to_cutoff(bg_times, bg_v_t, data_cutoff)
    ml_t_plot, ml_v_plot = _trim_to_cutoff(ml_times, ml_v_t, data_cutoff)
    sr_t_plot_full, sr_v_plot_full = insert_nans_for_gaps(sr_times, sr_v_t, sr_cold)
    st_t_plot_full, st_v_plot_full = insert_nans_for_gaps(st_times, st_v_t, st_cold)
    sr_t_plot, sr_v_plot = _trim_to_cutoff(sr_t_plot_full, sr_v_plot_full, data_cutoff)
    st_t_plot, st_v_plot = _trim_to_cutoff(st_t_plot_full, st_v_plot_full, data_cutoff)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    sr_color = "#5a3e7c"
    sr_fill  = "#d8c2ea"

    # Static
    ax.plot(st_t_plot, st_v_plot, color="#8b2e2e", lw=2, ls="--",
            marker="o", markersize=5, markevery=5,
            markerfacecolor="#f4b5b5", markeredgecolor="#8b2e2e",
            markeredgewidth=0.7,
            label="Static", zorder=10)
    # Adaptive hot-swap
    ax.plot(ml_t_plot, ml_v_plot, color="#8e5a1c", lw=2, ls="-",
            marker="D", markersize=5, markevery=5,
            markerfacecolor="#fad7a8", markeredgecolor="#8e5a1c",
            markeredgewidth=0.7,
            label="Adaptive hot-swap", zorder=11)
    # Stop-and-restart
    ax.plot(sr_t_plot, sr_v_plot, color=sr_color, lw=2, ls="-.",
            marker="s", markersize=5, markevery=5,
            markerfacecolor=sr_fill, markeredgecolor=sr_color,
            markeredgewidth=0.6,
            label="Stop-and-restart", zorder=12)
    # Grey dotted connectors across SR cold-start gaps
    for gs, ge in sr_cold:
        idx_before = max(i for i, t in enumerate(sr_times) if t <= gs)
        idx_after = min(i for i, t in enumerate(sr_times) if t >= ge)
        ax.plot([gs, ge], [sr_v_t[idx_before], sr_v_t[idx_after]],
                color="#bbbbbb", lw=1.2, ls=":", zorder=9)
    # BoundGuard
    ax.plot(bg_t_plot, bg_v_plot, color="#2c5984", lw=2.2, ls="-",
            marker="^", markersize=6, markevery=5,
            markerfacecolor="#b9d0e8", markeredgecolor="#2c5984",
            markeredgewidth=0.7,
            label="BoundGuard", zorder=13)

    # Epsilon threshold
    ax.axhline(y=epsilon, color="gray", ls="--", lw=1.1, zorder=4)
    ax.text(1.2, epsilon, r"$\epsilon$",
            ha="left", va="bottom", fontsize=11, color="#555555",
            fontstyle="italic")
    from matplotlib.ticker import FixedLocator
    tick_tol = max(0.1, epsilon * 0.15)
    yticks = [t for t in ax.get_yticks() if abs(t - epsilon) > tick_tol]
    ax.yaxis.set_major_locator(FixedLocator(yticks))

    # y-axis
    candidate_max = max(max(bg_v_t), max(ml_v_t), max(sr_v_t), max(st_v_t), epsilon)
    y_max = candidate_max * 1.08
    ax.set_ylim(0.0, y_max)

    # Input rate increase marker
    ax.axvline(x=load_change_sec, color="#b03a2e", ls="-.", lw=1.0, zorder=5)
    ax.text(load_change_sec + 0.3, y_max * 0.19, "Input rate\nincreases",
            color="#b03a2e", fontsize=11, ha="left", va="bottom", zorder=14)

    # ML pick marker (phase_c start in BoundGuard CSV)
    if len(bg_bounds) >= 3:
        px = bg_times[bg_bounds[2][0]]
        ax.axvline(x=px, color="#c88a44", ls=":", lw=1.0, zorder=5)
        ax.text(px + 0.3, y_max * 0.93, "1st placement",
                color="#c88a44", fontsize=11, ha="left", va="center", zorder=14)

        # NPU init shaded region
        npu_init_end = px + NPU_INIT_SEC
        ax.axvspan(px, npu_init_end, color="#ffe0b2", alpha=0.3, zorder=1)
        ax.text((px + npu_init_end) / 2, y_max * 0.05, "NPU init",
                color="#c88a44", fontsize=9, ha="center", va="bottom",
                fontstyle="italic", zorder=14)

    # BoundGuard fallback marker (phase_d start in BoundGuard CSV)
    if len(bg_bounds) >= 4:
        fx = bg_times[bg_bounds[3][0]]
        ax.axvline(x=fx, color="#5d87b5", ls=":", lw=1.0, zorder=5)
        ax.text(fx + 0.3, y_max * 0.39,
                "2nd placement\n($V(t)>\\epsilon$ after T)",
                color="#5d87b5", fontsize=11, ha="left", va="center", zorder=14)

        # NPU init shaded region for phase_d
        npu_init_end_d = fx + NPU_INIT_SEC
        ax.axvspan(fx, npu_init_end_d, color="#bbdefb", alpha=0.3, zorder=1)
        ax.text((fx + npu_init_end_d) / 2, y_max * 0.05, "NPU init",
                color="#5d87b5", fontsize=9, ha="center", va="bottom",
                fontstyle="italic", zorder=14)

    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Time (seconds)", fontsize=17, fontweight="bold")
    ax.set_ylabel(r"QoS Violation Score $\mathbf{V(t)}$", fontsize=17, fontweight="bold")
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=13)
    ax.grid(True, ls=":", lw=0.5, color="#ccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--run-only", choices=["bg", "ml", "sr", "st"], default=None)
    parser.add_argument("--output", default=OUT_PDF,
                        help="Output PDF path (default: %(default)s)")
    parser.add_argument("--x-max", type=float, default=None,
                        help="Override the x-axis upper bound (seconds). "
                             "Defaults to the latest sample across all curves.")
    parser.add_argument("--data-cutoff", type=float, default=None,
                        help="Truncate each curve at t<=CUTOFF (seconds). "
                             "Independent of --x-max so the axis can extend "
                             "beyond the last drawn point.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # With V(t)-triggered transitions, combo_durations become MAX caps.
    # BoundGuard: a → b (V(t)>ε) → c (V(t)>ε after hold+T_v) → d (fixed obs)
    bg_durations = {
        "phase_a": PHASE_A_DURATION,
        "phase_b": PHASE_B_MAX,
        "phase_c": PHASE_C_MAX,
        "phase_d": PHASE_D_DURATION,
    }
    # Adaptive / Stop-restart: phase_c is the last combo → fixed observation.
    ml_durations = {
        "phase_a": PHASE_A_DURATION,
        "phase_b": PHASE_B_MAX,
        "phase_c": PHASE_D_DURATION,
    }
    # Static uses the 3-phase MLONLY schedule so the high-load infps in
    # phase_b is preserved through phase_c (phase_d in npu.yaml drops infps to
    # match NPU service rate for the BoundGuard recovery, which would
    # unrealistically drop Static's V(t) too). mode=3 ignores any placement
    # change, so what matters is that the infps stays high after the load bump.
    st_durations = {
        "phase_a": PHASE_A_DURATION,
        "phase_b": 6,
        "phase_c": PHASE_D_DURATION,
    }

    # Per-combo hold = NPU-init/stabilization time that is excluded from T / T_v.
    # phase_b has no placement change, but we still give T_v time (3s) for the
    # windowed V(t) to fill cleanly after the input-rate jump.
    bg_trigger_hold = {
        "phase_a": 0,
        "phase_b": 0,                 # no NPU init; V(t) window fills naturally
        "phase_c": NPU_INIT_SEC,      # NPU1 init
    }
    ml_trigger_hold = {
        "phase_a": 0,
        "phase_b": 0,
    }
    sr_trigger_hold = dict(ml_trigger_hold)

    if not args.no_run:
        target = args.run_only
        if target is None or target == "bg":
            run_scenario(args.schedule, BG_CSV, mode=1, label="BoundGuard",
                         combo_durations=bg_durations,
                         trigger_epsilon=args.epsilon,
                         trigger_hold=bg_trigger_hold)
            reap_lingering_executors()
        if target is None or target == "ml":
            run_scenario(MLONLY_SCHEDULE, ML_CSV, mode=1, label="Adaptive (ML-only)",
                         combo_durations=ml_durations,
                         trigger_epsilon=args.epsilon,
                         trigger_hold=ml_trigger_hold)
            reap_lingering_executors()
        if target is None or target == "sr":
            run_scenario(MLONLY_SCHEDULE, SR_CSV, mode=0, label="Stop-and-restart",
                         combo_durations=ml_durations,
                         trigger_epsilon=args.epsilon,
                         trigger_hold=sr_trigger_hold)
            reap_lingering_executors()
        if target is None or target == "st":
            # Static ignores the trigger — keep fixed durations to let V(t)
            # accumulate without adaptation. Use the MLONLY schedule so the
            # sustained-overload infps from phase_b carries into phase_c
            # (see st_durations comment above).
            run_scenario(MLONLY_SCHEDULE, ST_CSV, mode=3, label="Static",
                         combo_durations=st_durations)
            reap_lingering_executors()
        if target is not None:
            print(f"\n[Done] {target} data saved.")
            return 0

    bg_data = load_curve(BG_CSV)
    ml_data = load_curve(ML_CSV)
    sr_data = load_curve(SR_CSV)
    st_data = load_curve(ST_CSV)

    all_times = [bg_data[2], ml_data[2], sr_data[2], st_data[2]]
    x_max = args.x_max if args.x_max is not None else (max(max(t) for t in all_times) + 1.0)

    bg_bounds = bg_data[3]
    load_change_sec = bg_data[2][bg_bounds[1][0]] if len(bg_bounds) >= 2 else float(PHASE_A_DURATION)

    make_plot(bg_data, ml_data, sr_data, st_data,
              epsilon=args.epsilon, pdf_path=args.output,
              x_max=x_max, load_change_sec=load_change_sec,
              data_cutoff=args.data_cutoff)

    # Summary
    print()
    print("=" * 70)
    print("  Dynamic load scenario summary (Antara / NPU)")
    print("=" * 70)
    print(f"  epsilon = {args.epsilon}")
    for name, data in [("BoundGuard", bg_data), ("Adaptive", ml_data),
                        ("Stop-Restart", sr_data), ("Static", st_data)]:
        print(f"  {name:<14} V(t) max={max(data[1]):.2f}  end={data[1][-1]:.2f}  rows={len(data[0])}")


if __name__ == "__main__":
    sys.exit(main() or 0)
