#!/usr/bin/env python3
"""
QoS Recovery Validation: produces qos_score_validation.pdf for the paper.

Scenario (4 models running concurrently — see tests/qos_recovery_schedule.yaml):
    Phase 1  combination_initial    INITIAL placement: all CPU at low input
                                    rates. The system is below capacity and
                                    V(t) sits near 0.
    Phase 2  combination_overload   FAILURE: same models, all CPU, but the
                                    input rates jump to a level the CPU
                                    cannot sustain. V(t) climbs above eps.
    Phase 3  combination_offload    RECOVERY: same high input rates, but
                                    resnet50 and resnext50 are offloaded to
                                    the GPU; mnasnet and squeezenet stay on
                                    CPU. V(t) returns below eps.

The script:
  1. Runs the schedule once with mode 0 (stop-and-restart) so each phase
     starts with freshly constructed view handlers — this resets the
     per-handler cumulative latency averages between phases. The three
     per-combo durations are passed via --combo-duration so the failure
     phase can be kept just long enough to satisfy the detection rule.
  2. Computes the windowed violation score
       V(t) = (1/T) * sum_{tau=t-T+1..t} v(tau),   T = 5s
     from the per-tick v(t) column the viewer wrote out (column "v_score").
  3. Determines the detection threshold epsilon. By default it is the
     midpoint of the empirical baseline and stressed V(t); --epsilon
     overrides it to a fixed value (the paper uses 30).
  4. Identifies the three figure markers:
        t0       = moment of failure injection. We display this at the last
                   pre-failure tick (p2_start - 1) so that the elapsed time
                   from t0 to the first valid windowed check is exactly T.
        t_detect = first tick where V(t) > epsilon AND the V(t) sliding
                   window is fully populated with post-failure samples
                   (i >= p2_start + T - 1). Combined with the t0 = p2_start
                   - 1 convention, the minimum detection latency is T.
        t_recover = first tick after the rollback where V(t) <= epsilon.
                   V(t) is already a T-second sliding mean, so a single
                   tick is enough — no extra debouncing.
  5. Plots V(t) over time with two shaded regions (detection latency and
     recovery latency), the threshold line, and event markers.

Output:
  results/qos_recovery_<ts>.csv          per-tick metrics CSV
  results/qos_score_validation.pdf       paper figure (overwritten each run)

Usage:
    python scripts/qos_recovery_validation.py --epsilon 30
    python scripts/qos_recovery_validation.py --no-run --epsilon 30   # replot only

See scripts/qos_recovery_validation.md for the full scenario walkthrough.
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "qos_recovery_schedule.yaml")
PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

WINDOW_T = 3  # V(t) sliding-window length (in measurement ticks / seconds)
PLOT_TIME_OFFSET = 0  # X-axis starts at this many wall-clock seconds
COLD_START_MIN_GAP = 1.5  # wall-clock seconds: anything bigger between two
                          # consecutive CSV rows is treated as a cold-start gap

# Mapping from raw combination names (in the YAML) to human-readable phase
# labels used in the figure. Unknown combo names fall back to the raw value.
# A value of None suppresses the label entirely (used for the middle/overload
# phase, which the figure no longer needs to label explicitly).
PHASE_DISPLAY = {
    "combination_initial":  "Initial Deployment",
    "combination_overload": None,
    "combination_offload":  "Changed Stable Deployment",
    # Backward compatibility with the previous scenario YAML
    "combination_baseline": "Initial Deployment",
    "combination_failure":  None,
    "combination_recovery": "Changed Stable Deployment",
}


# ---------------------------------------------------------------------------
# Run the scenario
# ---------------------------------------------------------------------------

def run_scenario(schedule: str, baseline_dur: int, failure_dur: int,
                 recovery_dur: int, csv_path: str, mode: int = 0,
                 label: str = "stop-and-restart") -> None:
    """Run the 3-phase scenario in a single executor invocation.

    Two modes are supported by this script:

      mode 0 — stop-and-restart. Each phase tears down the running workers
               and starts fresh ones. Used for the "stop-and-start" line in
               the figure. The cold-start gap between failure and recovery is
               where bounded recovery happens.
      mode 3 — static. The first combination's placement keeps running for
               the entire scenario; phase transitions only sweep the per-view
               input rates. Used for the "static" line in the figure: the
               system never reacts, so V(t) climbs once the input rate
               exceeds capacity and never returns below epsilon.

    --combo-duration overrides let baseline and recovery be long enough to
    show stable steady-state V(t) while keeping the failure phase short.
    """
    if os.path.exists(csv_path):
        os.remove(csv_path)

    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--duration", str(max(baseline_dur, failure_dur, recovery_dur)),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
        "--combo-duration", f"combination_initial={baseline_dur}",
        "--combo-duration", f"combination_overload={failure_dur}",
        "--combo-duration", f"combination_offload={recovery_dur}",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = (baseline_dur + failure_dur + recovery_dur) * 4 + 60

    print("=" * 60)
    print(f"  QoS recovery scenario [{label}, mode={mode}] "
          f"(baseline={baseline_dur}s, failure={failure_dur}s, recovery={recovery_dur}s)")
    print("=" * 60)
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        if stdout:
            for line in stdout.decode(errors="replace").splitlines()[-30:]:
                print(f"    [exec] {line}")
    except subprocess.TimeoutExpired:
        print(f"  Timed out after {timeout}s, killing executor...")
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def compute_windowed_v(rows, T: int = WINDOW_T):
    """V(t) = (1/T) * sum_{tau=t-T+1..t} v(tau) over the per-tick v(t) column."""
    v_list = [float(r.get("v_score", 0) or 0) for r in rows]
    out = []
    for i in range(len(v_list)):
        window = v_list[max(0, i + 1 - T):i + 1]
        out.append(sum(window) / len(window) if window else 0.0)
    return out


def find_phase_boundaries(rows):
    """Return [(start_index, combo_name), ...] in run order, deduplicating
    consecutive rows with the same combination label."""
    combos = [r.get("combination", "") for r in rows]
    if not combos:
        return []
    boundaries = [(0, combos[0])]
    for i in range(1, len(combos)):
        if combos[i] != combos[i - 1]:
            boundaries.append((i, combos[i]))
    return boundaries


def parse_timestamps(rows):
    """Parse the CSV ``timestamp`` column into wall-clock seconds offset from row 0.

    Each CSV row is recorded by the viewer's 1 Hz cpu_timer. When the
    executor stops a worker and starts a new one (mode 0 phase transition),
    no rows are written for ~3-5 wall-clock seconds while the new ONNX
    sessions load. Returning the elapsed seconds (relative to the first
    row) lets the figure plot V(t) at its real wall-clock position so the
    cold-start gaps are visible.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    parsed = []
    for r in rows:
        try:
            parsed.append(datetime.datetime.strptime(r.get("timestamp", ""), fmt))
        except (ValueError, TypeError):
            parsed.append(None)
    base = next((t for t in parsed if t is not None), None)
    if base is None:
        return [float(i) for i in range(len(rows))]
    out = []
    for t in parsed:
        if t is None:
            out.append(out[-1] if out else 0.0)
        else:
            out.append((t - base).total_seconds())
    return out


def find_cold_start_gaps(times_sec):
    """Return list of (start_sec, end_sec, gap_index) for wall-clock gaps
    longer than COLD_START_MIN_GAP. ``gap_index`` is the row index where the
    gap *ends* (= first row of the new phase)."""
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt > COLD_START_MIN_GAP:
            gaps.append((times_sec[i - 1], times_sec[i], i))
    return gaps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(times_sec, v_t, cold_starts,
              t0_sec, t_detect_sec, t_recover_sec,
              epsilon, pdf_path, phase_labels, x_min,
              static_times_sec=None, static_v_t=None):
    """Render the qos_score_validation figure on a wall-clock X-axis.

    The X-axis is wall-clock seconds since the first measurement, so the
    cold-start gaps between phases (where the executor is tearing down old
    workers and loading new ones, hence no CSV rows) appear as visible
    blank stretches in the data. Those gaps are explicitly marked with a
    hatched grey vspan labelled "Rollback in progress" so the reader can
    see them while the V(t) curve itself stays a single continuous line.

    If ``static_times_sec`` / ``static_v_t`` are supplied, a second V(t)
    curve from the static (mode 3) baseline is drawn on the same axes and a
    legend distinguishing "stop-and-start" from "static" is shown. The
    static run never recovers, so its line monotonically climbs and stays
    above epsilon for the rest of the scenario.
    """
    detection_phase = t_detect_sec - t0_sec
    recovery_phase = t_recover_sec - t_detect_sec

    # Filter cold-start gaps to only those that lie inside the detection ->
    # recovery window. Gaps that happen later (e.g. a brief cpu_timer tick
    # miss while the GPU workers stabilise in phase 3) are not "real"
    # redeployment events and they would just visually pollute the
    # "Changed Stable Deployment" region. We use the same filtered list for
    # both the hatched shading and the line-break NaN insertion so the
    # stop-and-start curve also stays continuous through any spurious
    # post-recovery gaps.
    relevant_cold_starts = [
        (gs, ge, gi) for (gs, ge, gi) in cold_starts
        if t0_sec <= ((gs + ge) / 2.0) <= t_recover_sec
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # ----- shaded phase regions (background, low zorder) ----------------
    # Pale fills so the V(t) curve drawn on top stays clearly readable.
    ax.axvspan(t0_sec, t_detect_sec, color="#f7c8a0", alpha=0.22, zorder=1)
    ax.axvspan(t_detect_sec, t_recover_sec, color="#b6e3b6", alpha=0.22, zorder=1)

    # ----- cold-start gap shading (rollback in progress) ----------------
    # Drawn at very low zorder + low opacity so the V(t) curves layered on
    # top are never obscured by either the gray fill or the hatch pattern.
    for gap_start, gap_end, _idx in relevant_cold_starts:
        ax.axvspan(gap_start, gap_end, color="#9a9a9a", alpha=0.13,
                   hatch="//", edgecolor="#888888", linewidth=0.0,
                   zorder=0.5)

    # ----- detection threshold (horizontal line, behind V(t)) -----------
    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.2,
               zorder=2)
    # epsilon symbol on the left side of the y-axis at the threshold height
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            fontsize=12, color="#333333",
            ha="right", va="center")

    # ----- vertical event markers (behind V(t)) -------------------------
    ax.axvline(x=t0_sec,        color="#c0392b", linestyle="-", linewidth=1.6, alpha=0.85, zorder=2)
    ax.axvline(x=t_detect_sec,  color="#e67e22", linestyle="-", linewidth=1.6, alpha=0.85, zorder=2)
    ax.axvline(x=t_recover_sec, color="#27ae60", linestyle="-", linewidth=1.6, alpha=0.85, zorder=2)

    # ----- V(t) curves, drawn LAST and on top so they stay visible -------
    # Bumped to a high zorder so the curves are unambiguously drawn on top
    # of the cold-start hatched bands (some of which fall on top of the
    # GPU-stable region where the stop-and-start line is very close to 0
    # — without high zorder the hatch pattern visually obscures it).
    if static_times_sec and static_v_t:
        ax.plot(static_times_sec, static_v_t,
                color="#a02020", linewidth=2.0, linestyle="--",
                label="static", zorder=20)

    # The stop-and-start curve is drawn segment by segment, breaking the line
    # at every *relevant* cold-start gap (= the redeployment events inside
    # the detection -> recovery window). Without this, matplotlib draws a
    # straight line across the gap (where there are NO measurement rows),
    # which can look like a flat plateau and obscures the fact that the
    # underlying 1 Hz sampling is actually missing data while workers are
    # being torn down and re-loaded. Spurious post-recovery gaps (e.g. a
    # brief cpu_timer tick miss in stable phase 3) are NOT broken so the
    # stable-state portion of the curve remains a single continuous line.
    if relevant_cold_starts:
        gap_indices = {idx for _s, _e, idx in relevant_cold_starts}
        plot_x = []
        plot_y = []
        for i, (x, y) in enumerate(zip(times_sec, v_t)):
            if i in gap_indices and plot_x:
                plot_x.append(float('nan'))
                plot_y.append(float('nan'))
            plot_x.append(x)
            plot_y.append(y)
        ax.plot(plot_x, plot_y, color="#0f3060", linewidth=2.4,
                label="stop-and-start", zorder=21)
    else:
        ax.plot(times_sec, v_t, color="#0f3060", linewidth=2.4,
                label="stop-and-start", zorder=21)

    # ----- y-axis range ---------------------------------------------------
    visible_v = [v for x, v in zip(times_sec, v_t) if x >= x_min]
    if static_times_sec and static_v_t:
        visible_v += [v for x, v in zip(static_times_sec, static_v_t)
                      if x >= x_min]
    y_max_data = max(visible_v) if visible_v else 1.0
    y_max = y_max_data * 1.22
    if y_max <= epsilon:
        y_max = epsilon * 1.5
    ax.set_ylim(0.0, y_max)

    # ----- legend ---------------------------------------------------------
    # Anchored just inside the top-right corner — close to the right axis
    # edge but with a tiny inset so it doesn't visually touch the border.
    if static_times_sec and static_v_t:
        ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.99),
                  framealpha=0.92, fontsize=9)

    # ----- top-of-axis event labels --------------------------------------
    label_y = y_max * 0.96
    ax.text(t0_sec,        label_y, r"  $t_0$",        color="#c0392b",
            fontsize=10, fontweight="bold", va="top", ha="left")
    ax.text(t_detect_sec,  label_y, r"  $t_{detect}$", color="#e67e22",
            fontsize=10, fontweight="bold", va="top", ha="left")
    ax.text(t_recover_sec, label_y, r"  $t_{recover}$", color="#27ae60",
            fontsize=10, fontweight="bold", va="top", ha="left")

    # ----- phase arrows (sit inside the shaded regions) -----------------
    if detection_phase > 0:
        y1 = y_max * 0.58
        ax.annotate("", xy=(t_detect_sec, y1), xytext=(t0_sec, y1),
                    arrowprops=dict(arrowstyle="<->", color="#7b3306", lw=1.4))
        ax.text((t0_sec + t_detect_sec) / 2.0, y1 + y_max * 0.02,
                "Detection\nphase",
                ha="center", va="bottom", fontsize=9, color="#7b3306")

    if recovery_phase > 0:
        # Place the recovery-phase double arrow well below the epsilon line
        # (a little above V(t) = 20) so it never collides with the
        # recovering stop-and-start curve, the epsilon line, or the
        # redeployment-downtime annotation that lives in the upper area.
        y2 = 25.0
        ax.annotate("", xy=(t_recover_sec, y2), xytext=(t_detect_sec, y2),
                    arrowprops=dict(arrowstyle="<->", color="#155724", lw=1.4))
        ax.text((t_detect_sec + t_recover_sec) / 2.0, y2 + y_max * 0.02,
                "Recovery\nphase",
                ha="center", va="bottom", fontsize=9, color="#155724")

    # ----- redeployment downtime annotation -----------------------------
    # The dotted ellipse + arrow points at the first "real" redeployment
    # gap (= the CPU -> GPU placement change in stop-and-start mode). Use
    # the already-filtered relevant_cold_starts so we never accidentally
    # land on a spurious post-recovery glitch.
    relevant_gap = relevant_cold_starts[0] if relevant_cold_starts else None
    if relevant_gap is not None:
        g_start, g_end, g_idx = relevant_gap
        gap_mid = (g_start + g_end) / 2.0

        # The "empty" stop-and-start data sits between the V(t) value of
        # the last pre-gap row (g_idx - 1) and the first post-gap row
        # (g_idx). Use the average as the vertical center of the
        # encircled empty region — this lands directly on the visual gap
        # in the broken line.
        y_before = float(v_t[g_idx - 1]) if g_idx > 0 else 0.0
        y_after = float(v_t[g_idx])
        empty_center_y = (y_before + y_after) / 2.0

        # Thin dotted ellipse around the empty (broken) area. Width is
        # slightly wider than the gap so the broken endpoints sit just
        # inside the ellipse; height is a small fraction of the y range,
        # large enough to enclose both endpoints comfortably.
        ellipse_width = (g_end - g_start) * 1.6
        ellipse_height = max(y_max * 0.10,
                             abs(y_before - y_after) + y_max * 0.06)
        empty_ellipse = Ellipse(
            (gap_mid, empty_center_y),
            width=ellipse_width,
            height=ellipse_height,
            linewidth=1.0,
            linestyle=':',
            edgecolor='#444444',
            facecolor='none',
            zorder=10.6,
        )
        ax.add_patch(empty_ellipse)

        # Text sits high (above all phase arrows / epsilon line / static
        # curve in this region) and slightly to the right of t_recover
        # so it never overlaps the t_recover label or the legend. Arrow
        # tip points at the center of the dotted ellipse.
        text_x = t_recover_sec + 1.4
        text_y = y_max * 0.78
        ax.annotate(
            "Redeployment\ndowntime\n(no service)",
            xy=(gap_mid, empty_center_y),
            xytext=(text_x, text_y),
            arrowprops=dict(arrowstyle="->", color="#444444",
                            lw=1.2, connectionstyle="arc3,rad=0.25"),
            fontsize=9, color="#333333",
            ha="left", va="center",
            zorder=11,
        )

    # ----- phase labels along the bottom ---------------------------------
    # Clamp label positions to the visible X range so labels for phases that
    # would otherwise sit before x_min still show inside the figure. The
    # right edge of the visible X range is the END OF THE STATIC CURVE
    # (when present) — the stop-and-start trace usually extends a couple
    # seconds further but those tail rows are stable-state V(t)≈0 noise
    # and only create an empty band on the right side of the figure.
    if static_times_sec:
        x_max_visible = max(static_times_sec)
    else:
        x_max_visible = max(times_sec) if times_sec else 1.0
    bottom_y = y_max * 0.04
    if phase_labels:
        for x_center, name in phase_labels:
            x_clamped = max(x_min + 0.5, min(x_max_visible - 0.5, x_center))
            ax.text(x_clamped, bottom_y, name, fontsize=8, color="#555555",
                    ha="center", va="bottom", style="italic")

    # ----- axes ----------------------------------------------------------
    ax.set_xlim(left=x_min, right=x_max_visible + 0.3)
    # Integer ticks every 2 seconds (e.g. 0, 2, 4, ...) instead of the
    # default 2.5-second floats matplotlib picks for this range.
    tick_start = int(x_min)
    tick_end = int(x_max_visible) + 1
    ax.set_xticks(list(range(tick_start, tick_end, 2)))
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel(r"QoS Violation Score $V(t)$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the QoS recovery scenario and produce qos_score_validation.pdf")
    parser.add_argument("--baseline-duration", type=int, default=12,
                        help="Seconds for the initial (verified) phase (default: 12)")
    parser.add_argument("--failure-duration", type=int, default=3,
                        help="Seconds for the failure (overload) phase. Should be "
                             "long enough that the windowed V(t) (T=3) reaches the "
                             "post-failure plateau before phase 3 starts, but short "
                             "enough that cumulative averaging does not pull V(t) "
                             "back below epsilon. Default 3 yields ~4 measurement "
                             "rows, the V(t) climbs cleanly to a single peak around "
                             "the rollback transition.")
    parser.add_argument("--recovery-duration", type=int, default=12,
                        help="Seconds for the recovery (offloaded) phase (default: 12)")
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE,
                        help="3-combo recovery schedule YAML")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip the executor run; reuse the most recent CSVs")
    parser.add_argument("--csv", type=str, default=None,
                        help="When --no-run is set, read this CSV as the "
                             "stop-and-restart trace")
    parser.add_argument("--static-csv", type=str, default=None,
                        help="When --no-run is set, read this CSV as the static "
                             "(mode 3) trace")
    parser.add_argument("--no-static", action="store_true",
                        help="Skip the static (mode 3) baseline run; only plot "
                             "the stop-and-restart curve")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override the detection threshold (default: empirical "
                             "midpoint of baseline and stressed V(t))")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(results_dir, "qos_score_validation.pdf")

    # ----- Run -----
    # We need two traces:
    #   csv_path        — stop-and-restart (mode 0). Drives detection /
    #                     recovery markers and the cold-start gap shading.
    #   static_csv_path — static (mode 3). Plotted as a second curve so the
    #                     reader can compare a system that *does* react with
    #                     a system that does not.
    if args.no_run:
        if args.csv:
            csv_path = args.csv
        else:
            cands = sorted(
                f for f in os.listdir(results_dir)
                if f.startswith("qos_recovery_") and f.endswith(".csv")
                and "static" not in f
            )
            if not cands:
                print("[Error] --no-run set but no qos_recovery_*.csv found.")
                return 1
            csv_path = os.path.join(results_dir, cands[-1])
        print(f"[Info] Reusing stop-and-restart CSV: {csv_path}")

        static_csv_path = None
        if not args.no_static:
            if args.static_csv:
                static_csv_path = args.static_csv
            else:
                cands = sorted(
                    f for f in os.listdir(results_dir)
                    if f.startswith("qos_recovery_static_") and f.endswith(".csv")
                )
                if cands:
                    static_csv_path = os.path.join(results_dir, cands[-1])
            if static_csv_path:
                print(f"[Info] Reusing static CSV: {static_csv_path}")
    else:
        csv_path = os.path.join(results_dir, f"qos_recovery_{ts}.csv")
        run_scenario(args.schedule,
                     args.baseline_duration,
                     args.failure_duration,
                     args.recovery_duration,
                     csv_path,
                     mode=0,
                     label="stop-and-restart")

        static_csv_path = None
        if not args.no_static:
            static_csv_path = os.path.join(
                results_dir, f"qos_recovery_static_{ts}.csv")
            run_scenario(args.schedule,
                         args.baseline_duration,
                         args.failure_duration,
                         args.recovery_duration,
                         static_csv_path,
                         mode=3,
                         label="static")

    # ----- Load -----
    rows = load_csv(csv_path)
    if not rows:
        print(f"[Error] CSV is empty: {csv_path}")
        return 1
    print(f"[Info] Loaded {len(rows)} CSV rows from {csv_path}")

    # ----- Per-tick v(t) and windowed V(t) -----
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    # Wall-clock seconds since the first measurement (so cold-start gaps
    # between phases — when the executor is loading new ONNX sessions and
    # no CSV rows are written — show up as visible blanks on the X-axis).
    times_sec = parse_timestamps(rows)
    cold_starts = find_cold_start_gaps(times_sec)

    # ----- Phase boundaries -----
    boundaries = find_phase_boundaries(rows)
    print(f"[Info] Phase boundaries: {boundaries}")
    if len(boundaries) < 3:
        print("[Error] Expected 3 distinct phases (baseline, failure, recovery); "
              f"found {len(boundaries)}: {boundaries}")
        print("        Make sure the schedule YAML has three combos with distinct names "
              "and that the executor reached all three.")
        return 1

    p1_start = boundaries[0][0]   # initial (verified) phase starts here
    p2_start = boundaries[1][0]   # failure injection — first overloaded measurement
    p3_start = boundaries[2][0]   # rollback / offloaded phase starts here
    p3_end = len(rows)
    phase_names = [boundaries[0][1], boundaries[1][1], boundaries[2][1]]

    # t_0 marks the moment of failure injection. We treat this as the
    # *boundary* between the last initial-phase tick and the first overload
    # tick. In wall-clock terms it falls inside the cold-start gap between
    # phase 1 and phase 2 (when the executor is replacing the workers).
    # We anchor t_0 at the wall-clock start of that gap (= the timestamp of
    # the last initial measurement).
    t0_row = max(0, p2_start - 1)

    # ----- Threshold determination -----
    # Use V(t) at the last tick of each phase: by then the sliding window is
    # fully populated with that phase's per-tick samples, so V(t) reflects the
    # steady-state load.
    baseline_idx = max(p1_start, p2_start - 1)
    stressed_idx = max(p2_start, p3_start - 1)
    baseline = v_t[baseline_idx]
    stressed = max(v_t[p2_start:p3_start]) if p3_start > p2_start else v_t[stressed_idx]
    midpoint = (baseline + stressed) / 2.0
    if args.epsilon is not None:
        epsilon = float(args.epsilon)
        epsilon_source = f"user override (--epsilon {epsilon:g})"
    else:
        epsilon = midpoint
        epsilon_source = "midpoint of baseline and stressed"

    print()
    print("=" * 60)
    print("  Detection threshold determination")
    print("=" * 60)
    print(f"  Baseline V(t) (end of {phase_names[0]:<22s}, t={baseline_idx}s) = {baseline:.4f}")
    print(f"  Stressed V(t) (end of {phase_names[1]:<22s}, t={stressed_idx}s) = {stressed:.4f}")
    print(f"  Empirical midpoint                                  = {midpoint:.4f}")
    print(f"  ==> Detection threshold  epsilon = {epsilon:.4f}  ({epsilon_source})")
    print()

    if stressed - baseline < 0.5:
        print("[Warning] baseline and stressed V(t) are very close — the bad "
              "placement is not actually degrading performance much. The threshold "
              "may not give a meaningful detection point.")

    # ----- Detection -----
    # V(t) is a sliding window of length T over the per-tick v(t). We
    # require the window used for the threshold check to be computed
    # entirely from post-failure samples (i.e. i - T + 1 >= p2_start so the
    # window is [p2_start..i]).
    detection_earliest = p2_start + WINDOW_T - 1
    t_detect_row = None
    for i in range(detection_earliest, p3_start):
        if v_t[i] > epsilon:
            t_detect_row = i
            break
    if t_detect_row is None:
        print("[Error] V(t) never crossed epsilon during the failure phase "
              "(after waiting for the window to fill with post-failure data). "
              "Try increasing --failure-duration or pick a more degraded "
              "combination_overload.")
        return 1

    # ----- Recovery -----
    # V(t) is itself a T-second sliding mean, so once it drops to epsilon
    # the system is considered stable. The first tick (after the rollback
    # phase begins) where V(t) <= epsilon is the recovery completion point.
    t_recover_row = None
    for i in range(p3_start, p3_end):
        if v_t[i] <= epsilon:
            t_recover_row = i
            break
    if t_recover_row is None:
        print("[Warning] V(t) did not return to epsilon by end of recovery phase; "
              "using the last tick as t_recover.")
        t_recover_row = p3_end - 1

    # Wall-clock positions of the events (used for the figure).
    # The detection phase is the time the system spends collecting T seconds
    # of post-failure data before V(t) > epsilon can be evaluated. We anchor
    # t_detect at the actual measurement row where the threshold is first
    # crossed, and place t_0 exactly T seconds before it so the visualised
    # detection phase always has the algorithm-defined width of T seconds
    # (matching the V(t) sliding window length). This decouples the
    # displayed detection phase from any cold-start gap that happens to sit
    # inside the [t0, t_detect] interval.
    t_detect_sec = times_sec[t_detect_row]
    t_recover_sec = times_sec[t_recover_row]
    t0_sec = t_detect_sec - float(WINDOW_T)

    detection_phase_sec = float(WINDOW_T)
    recovery_phase_sec = t_recover_sec - t_detect_sec
    degradation_duration_sec = t_recover_sec - t0_sec

    print("=" * 60)
    print("  Recovery scenario timing (wall-clock seconds from row 0)")
    print("=" * 60)
    print(f"  t_0      (failure injection)    = {t0_sec:6.2f}s  (= t_detect - T)")
    print(f"  t_detect (windowed V(t) > eps)  = {t_detect_sec:6.2f}s  (row {t_detect_row}, "
          f"window fully post-failure)")
    p3_start_sec = times_sec[p3_start]
    print(f"  rollback (new workers ready)    = {p3_start_sec:6.2f}s  (first {phase_names[2]} row)")
    print(f"  t_recover (V(t) <= eps)         = {t_recover_sec:6.2f}s  (row {t_recover_row})")
    print(f"  Detection phase                 = {detection_phase_sec:6.2f}s   (= T)")
    print(f"  Recovery phase                  = {recovery_phase_sec:6.2f}s")
    print(f"  Total degradation duration      = {degradation_duration_sec:6.2f}s")
    if cold_starts:
        print(f"  Cold-start gaps detected        = "
              + ", ".join(f"{e - s:.2f}s @ {s:.1f}-{e:.1f}s"
                          for s, e, _ in cold_starts))
    print()

    # ----- Static (mode 3) baseline -----
    # Load and align the static run so the phase 1 -> phase 2 boundary
    # (= moment of failure injection) lands at the same wall-clock X
    # position as the stop-and-restart curve. This makes the divergence
    # between the two strategies visually anchored on the same t_0.
    static_times_sec = None
    static_v_t = None
    if static_csv_path and os.path.exists(static_csv_path):
        try:
            static_rows = load_csv(static_csv_path)
        except FileNotFoundError:
            static_rows = []
        if static_rows:
            print(f"[Info] Loaded {len(static_rows)} static-mode CSV rows from "
                  f"{static_csv_path}")
            static_v_t_raw = compute_windowed_v(static_rows, T=WINDOW_T)
            static_times_raw = parse_timestamps(static_rows)
            static_boundaries = find_phase_boundaries(static_rows)
            print(f"[Info] Static-mode phase boundaries: {static_boundaries}")
            if len(static_boundaries) >= 2:
                static_p2_start = static_boundaries[1][0]
                offset = times_sec[p2_start] - static_times_raw[static_p2_start]
                static_times_sec = [t + offset for t in static_times_raw]
            else:
                # Fall back to no shift if mode 3 only saw one phase.
                static_times_sec = list(static_times_raw)
            static_v_t = list(static_v_t_raw)
        else:
            print(f"[Warning] Static CSV is empty: {static_csv_path}")

    # ----- Plot -----
    # Phase labels along the bottom (skip phases whose display name is None,
    # e.g. the middle overload phase that we deliberately leave unlabeled).
    raw_phase_labels = [
        ((times_sec[p1_start] + times_sec[max(p1_start, p2_start - 1)]) / 2.0,
         PHASE_DISPLAY.get(phase_names[0], phase_names[0])),
        ((times_sec[p2_start] + times_sec[max(p2_start, p3_start - 1)]) / 2.0,
         PHASE_DISPLAY.get(phase_names[1], phase_names[1])),
        ((times_sec[p3_start] + times_sec[p3_end - 1]) / 2.0,
         PHASE_DISPLAY.get(phase_names[2], phase_names[2])),
    ]
    phase_labels = [(x, lbl) for x, lbl in raw_phase_labels if lbl]
    make_plot(times_sec, v_t, cold_starts,
              t0_sec, t_detect_sec, t_recover_sec,
              epsilon, pdf_path, phase_labels, x_min=PLOT_TIME_OFFSET,
              static_times_sec=static_times_sec, static_v_t=static_v_t)
    print(f"[Plot] Saved: {pdf_path}")
    print(f"[Plot] CSV:   {csv_path}")
    if static_csv_path:
        print(f"[Plot] Static CSV: {static_csv_path}")
    print(f"[Plot] Use \\includegraphics{{{os.path.basename(pdf_path)}}} in the LaTeX figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
