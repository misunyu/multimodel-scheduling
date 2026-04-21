#!/usr/bin/env python3
"""
ML-misprediction (r_t_v_y variant): produces ml_misprediction_fallback.pdf.

Models: resnet50 (view1), yolov4 (view2), tiny-llama (headless), vgg19 (headless)

BoundGuard flow (mode 2 concept, executed as mode 1 with 4-phase schedule):
  1. combination_stable    -- all CPU, low rates, V(t) ≈ 0
  2. combination_burst     -- all CPU, high rates, V(t) >> ε
  3. combination_xgb_pick  -- XGBoost pick (2 GPU + 2 CPU), T=3s check
  4. combination_all_gpu   -- heuristic fallback (all GPU) if V(t) still > ε

Static (mode 3): keeps the initial all-CPU placement; only sweeps infps.
Run separately so it has realistic per-run variance but similar trajectory.

Usage:
    python scripts/ml_misprediction_rtvy.py
    python scripts/ml_misprediction_rtvy.py --no-run --epsilon 50
"""
import argparse, os, subprocess, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reap_lingering_executors(cooldown_sec: float = 5.0) -> None:
    for patt in ("schedule_executor_main.py", "headless_inference_worker.py"):
        subprocess.run(["pkill", "-9", "-f", patt], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(cooldown_sec)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from qos_recovery_validation import (
    compute_windowed_v, find_phase_boundaries,
    load_csv, parse_timestamps, WINDOW_T, COLD_START_MIN_GAP,
)

PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON): PYTHON = sys.executable
EXECUTOR = os.path.join(PROJECT_DIR, "schedule_executor_main.py")
SCHEDULE = os.path.join(PROJECT_DIR, "tests", "ml_misprediction_rtvy_schedule.yaml")
RESULTS  = os.path.join(PROJECT_DIR, "results")
OUT_PDF  = os.path.join(RESULTS, "ml_misprediction_fallback.pdf")
BG_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_bg.csv")
ML_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_ml.csv")
SR_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_sr.csv")  # stop-and-restart
ST_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_st.csv")

P_STABLE  = 22
P_BURST   = 8
P_XGB     = 3    # T=3s: XGBoost pick monitoring window
P_ALL_GPU = 25   # heuristic fallback


def run(csv_path, mode, label, combo_durations):
    if os.path.exists(csv_path): os.remove(csv_path)
    total = sum(combo_durations.values())
    cmd = [PYTHON, EXECUTOR, "--schedule", SCHEDULE,
           "--duration", str(total), "--adaptive-mode", str(mode),
           "--metrics-csv", csv_path, "--auto_start_all"]
    for c, d in combo_durations.items():
        cmd += ["--combo-duration", f"{c}={d}"]
    env = os.environ.copy(); env["QT_QPA_PLATFORM"] = "offscreen"
    print(f"\n{'='*60}\n  [{label}, mode={mode}]\n{'='*60}")
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=total*4+120)
    for l in proc.stdout.decode(errors="replace").splitlines()[-3:]:
        print(f"    {l}")


def find_cold_start_gaps(times_sec, rows=None):
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt <= COLD_START_MIN_GAP:
            continue
        if rows is not None:
            if rows[i - 1].get("combination", "") == rows[i].get("combination", ""):
                continue
        gaps.append((times_sec[i - 1], times_sec[i], i))
    return gaps


def load_curve(csv_path):
    rows = load_csv(csv_path)
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times = parse_timestamps(rows)
    bounds = find_phase_boundaries(rows)
    cold_starts = find_cold_start_gaps(times, rows=rows)
    return rows, v_t, times, bounds, cold_starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--out", default=OUT_PDF)
    parser.add_argument("--run-only", choices=["bg", "ml", "sr", "st"],
                        default=None,
                        help="Run only one curve per invocation: bg=BoundGuard, "
                             "ml=ML-only, sr=Stop-and-restart, st=Static.")
    args = parser.parse_args()

    if not args.no_run:
        target = args.run_only
        if target is None or target == "bg":
            run(BG_CSV, mode=1, label="BoundGuard",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST,
                    "combination_xgb_pick": P_XGB,
                    "combination_all_gpu":  P_ALL_GPU,
                })
            reap_lingering_executors()
        if target is None or target == "ml":
            run(ML_CSV, mode=1, label="Adaptive (ML-only)",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST,
                    "combination_xgb_pick": P_XGB + P_ALL_GPU,
                })
            reap_lingering_executors()
        if target is None or target == "sr":
            run(SR_CSV, mode=0, label="Stop-and-restart",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST,
                    "combination_xgb_pick": P_XGB,
                    "combination_all_gpu":  P_ALL_GPU,
                })
            reap_lingering_executors()
        if target is None or target == "st":
            run(ST_CSV, mode=3, label="Static",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST,
                    "combination_xgb_pick": P_XGB,
                    "combination_all_gpu":  P_ALL_GPU,
                })
            reap_lingering_executors()
        # If running one at a time, exit after data collection
        if target is not None:
            print(f"\n[Done] {target} data saved. Run other curves separately, "
                  f"then use --no-run to plot.")
            return 0

    bg_rows, bg_v, bg_t, bg_bounds, bg_cold = load_curve(BG_CSV)
    ml_rows, ml_v, ml_t, ml_bounds, ml_cold = load_curve(ML_CSV)
    sr_rows, sr_v, sr_t, sr_bounds, sr_cold = load_curve(SR_CSV)
    st_rows, st_v, st_t, st_bounds, st_cold = load_curve(ST_CSV)

    eps = args.epsilon
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    # Clip all curves to the same wall-clock span
    clip_t = max(bg_t[-1], st_t[-1]) if bg_t and st_t else 999
    def clip(t_list, v_list):
        t_c = [t for t in t_list if t <= clip_t]
        return t_c, v_list[:len(t_c)]
    ml_t_clip, ml_v_clip = clip(ml_t, ml_v)
    sr_t_clip, sr_v_clip = clip(sr_t, sr_v)

    ax.plot(st_t, st_v, color="#a83232", lw=2, ls="--",
            label="Static", zorder=10)
    ax.plot(ml_t_clip, ml_v_clip, color="#e67e22", lw=2, ls="-",
            marker="o", markersize=5, markevery=5,
            label="Adaptive (ML-only)", zorder=11)

    # Stop-and-restart: break the line at cold-start gaps (service stops
    # during worker teardown/rebuild). Insert NaN to break, then draw
    # light grey dotted lines across each gap to show the connection.
    sr_color = "#8e44ad"
    sr_t_plot = list(sr_t_clip)
    sr_v_plot = list(sr_v_clip)
    # Collect gap endpoints before inserting NaN (for dotted connectors)
    sr_gap_segments = []
    for gap_start, gap_end, gap_idx in sr_cold:
        if gap_start > clip_t:
            continue
        # Values just before and after the gap
        v_before = sr_v[gap_idx - 1] if gap_idx > 0 else 0
        v_after  = sr_v[gap_idx] if gap_idx < len(sr_v) else 0
        sr_gap_segments.append((gap_start, v_before, gap_end, v_after))
    # Insert NaN at gap boundaries to break the solid line
    sr_t_nan, sr_v_nan = [], []
    gap_ends_set = {round(ge, 3) for _, ge, _ in sr_cold}
    for i, (t, v) in enumerate(zip(sr_t_plot, sr_v_plot)):
        if i > 0 and round(t, 3) in gap_ends_set:
            mid = (sr_t_plot[i - 1] + t) / 2.0 if i > 0 else t
            sr_t_nan.append(mid)
            sr_v_nan.append(float("nan"))
        sr_t_nan.append(t)
        sr_v_nan.append(v)
    ax.plot(sr_t_nan, sr_v_nan, color=sr_color, lw=2, ls="-.",
            marker="s", markersize=5, markevery=5,
            label="Stop-and-restart", zorder=12)
    # Draw light grey dotted connectors across each gap
    for gs, vb, ge, va in sr_gap_segments:
        ax.plot([gs, ge], [vb, va], color="#bbbbbb", lw=1.2, ls=":", zorder=9)

    ax.plot(bg_t, bg_v, color="#1f4e79", lw=2.8, ls="-",
            marker="^", markersize=6, markevery=5,
            label="BoundGuard", zorder=13)

    ax.axhline(y=eps, color="gray", ls="--", lw=1.1, zorder=4)
    # Place ε label inside the plot area (right of y-axis) to avoid
    # overlapping with the numeric y-tick at the same height.
    ax.text(1.2, eps, r"$\epsilon$", ha="left", va="bottom",
            fontsize=11, color="#555555", fontstyle="italic")
    # Remove the y-tick at eps value so it doesn't collide
    from matplotlib.ticker import FixedLocator
    tick_tol = max(0.1, eps * 0.15)
    yticks = [t for t in ax.get_yticks() if abs(t - eps) > tick_tol]
    ax.yaxis.set_major_locator(FixedLocator(yticks))

    cmax = max(max(bg_v, default=0), max(ml_v, default=0),
               max(sr_v, default=0), max(st_v, default=0))
    y_max = cmax * 1.08
    ax.set_ylim(0, y_max)

    # Burst marker (phase 2 start)
    if len(bg_bounds) >= 2:
        bx = bg_t[bg_bounds[1][0]]
        ax.axvline(x=bx, color="#7b3306", ls=":", lw=1.3, zorder=5)
        ax.text(bx+0.3, y_max*0.19, "Input rate\nincreases",
                color="#7b3306", fontsize=11, ha="left", va="bottom", zorder=12)

    # XGBoost pick marker (phase 3 start)
    if len(bg_bounds) >= 3:
        px = bg_t[bg_bounds[2][0]]
        ax.axvline(x=px, color="#2c3e50", ls=":", lw=1.2, zorder=5)
        ax.text(px+0.3, y_max*0.93, "XGBoost pick\n(2 GPU + 2 CPU)",
                color="#2c3e50", fontsize=11, ha="left", va="center", zorder=12)

    # All-GPU fallback marker (phase 4 start)
    if len(bg_bounds) >= 4:
        fx = bg_t[bg_bounds[3][0]]
        ax.axvline(x=fx, color="#155724", ls=":", lw=1.2, zorder=5)
        ax.text(fx+0.3, y_max*0.39, "All-GPU fallback\n($V(t)>\\epsilon$ after T)",
                color="#155724", fontsize=11, ha="left", va="center", zorder=12)

    # BoundGuard recovery marker
    bg_recover = None
    if len(bg_bounds) >= 4:
        for i in range(bg_bounds[3][0], len(bg_v)):
            if bg_v[i] <= eps:
                bg_recover = bg_t[i]; break
    # All three curves end at clip_t; set xlim just past that
    ax.set_xlim(0, clip_t + 1)
    ax.set_xlabel("Time (seconds)", fontsize=17, fontweight="bold")
    ax.set_ylabel(r"QoS Violation Score $\mathbf{V(t)}$", fontsize=17, fontweight="bold")
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=13)
    ax.grid(True, ls=":", lw=0.5, color="#ccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"\n[Plot] Saved: {args.out}")

    print(f"\n  epsilon = {eps}")
    print(f"  Static       V(t) max={max(st_v):.1f}  end={st_v[-1]:.1f}")
    print(f"  ML-only      V(t) max={max(ml_v):.1f}  end={ml_v[-1]:.1f}")
    print(f"  Stop-restart V(t) max={max(sr_v):.1f}  end={sr_v[-1]:.1f}")
    print(f"  BoundGuard   V(t) max={max(bg_v):.1f}  end={bg_v[-1]:.1f}")
    if bg_recover:
        print(f"  BG t_recover (V<=eps) = {bg_recover:.1f}s")
    else:
        print(f"  BG did not reach V<=eps by end of run")


if __name__ == "__main__":
    sys.exit(main() or 0)
