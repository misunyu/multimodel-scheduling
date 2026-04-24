#!/usr/bin/env python3
"""
ML-misprediction (r_t_v_y variant): produces ml_misprediction_alternative.pdf.

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
OUT_PDF  = os.path.join(RESULTS, "ml_misprediction_alternative.pdf")
BG_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_bg.csv")
ML_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_ml.csv")
SR_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_sr.csv")  # stop-and-restart
ST_CSV   = os.path.join(RESULTS, "ml_mis_rtvy_st.csv")

P_STABLE    = 22
# Burst is capped; executor advances as soon as V(t) > eps thanks to the
# "v-above" combo-trigger.
P_BURST_MAX = 30
# xgb_pick is capped; executor advances after T_v when V(t) > eps, otherwise
# commits. Cap is long enough to absorb workers' warm-up overhead.
P_XGB_MAX   = 30
P_ALL_GPU   = 25   # final commit duration


def run(csv_path, mode, label, combo_durations,
        combo_triggers=None, epsilon=1.0, tv=3.0, stop_after=None):
    if os.path.exists(csv_path): os.remove(csv_path)
    total = sum(combo_durations.values())
    cmd = [PYTHON, EXECUTOR, "--schedule", SCHEDULE,
           "--duration", str(total), "--adaptive-mode", str(mode),
           "--metrics-csv", csv_path, "--auto_start_all",
           "--qos-trigger-epsilon", str(epsilon),
           "--qos-trigger-tv",      str(tv)]
    for c, d in combo_durations.items():
        cmd += ["--combo-duration", f"{c}={d}"]
    for c, pol in (combo_triggers or {}).items():
        cmd += ["--combo-trigger", f"{c}={pol}"]
    if stop_after:
        cmd += ["--stop-after", stop_after]
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


def _trim_trailing_reset(rows):
    """Drop the teardown tail rows where the executor reverts the combo
    label back to the initial combo after all phases have finished. Those
    rows capture workers tearing down and produce a spurious V(t) spike
    that isn't part of any measured phase."""
    if not rows:
        return rows
    combos = [r.get("combination", "") for r in rows]
    n = len(combos)
    # Walk back from the end: if we see the initial combo reappear after a
    # different one, truncate at the boundary of that reset.
    last = combos[-1]
    i = n - 1
    # Find contiguous tail with the same label as the last row.
    while i > 0 and combos[i] == last:
        i -= 1
    # i now points at the last row that is NOT the trailing label. If the
    # trailing label differs from combos[i] and matches an *earlier* label
    # (i.e. a reset to a phase we already left), drop the trailing block.
    if combos[i] != last and last in combos[:i]:
        return rows[:i + 1]
    return rows


def load_curve(csv_path):
    rows = load_csv(csv_path)
    rows = _trim_trailing_reset(rows)
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times = parse_timestamps(rows)
    bounds = find_phase_boundaries(rows)
    cold_starts = find_cold_start_gaps(times, rows=rows)
    return rows, v_t, times, bounds, cold_starts


def _eps_tag(eps):
    if eps == int(eps):
        return str(int(eps))
    return str(eps).replace(".", "p")


def _warmup_pass(epsilon=1.0, tv=3.0):
    """Run one full stable-only pass so all measured runs start from the
    same warm state (ONNX sessions, CUDA context, OS page cache)."""
    warm_csv = os.path.join(RESULTS, "ml_mis_rtvy_warmup.csv")
    print()
    print("=" * 60)
    print("  [Warmup pass — results discarded]")
    print("=" * 60)
    run(warm_csv, mode=3, label="Warmup",
        combo_durations={
            "combination_stable": P_STABLE,
            "combination_burst":  P_BURST_MAX,
        },
        epsilon=epsilon, tv=tv,
        stop_after="combination_burst")
    reap_lingering_executors()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--tv", type=float, default=3.0,
                        help="Validation window T_v (default 3s).")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--out", default=None,
                        help="Output PDF (defaults to "
                             "results/ml_misprediction_alternative_epsilon_<eps>.pdf)")
    parser.add_argument("--run-only", choices=["bg", "ml", "sr", "st"],
                        default=None,
                        help="Run only one curve per invocation: bg=BoundGuard, "
                             "ml=ML-only, sr=Stop-and-restart, st=Static.")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip the warmup pass before the measured runs.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(
            RESULTS,
            f"ml_misprediction_alternative_epsilon_{_eps_tag(args.epsilon)}.pdf",
        )

    if not args.no_run:
        target = args.run_only
        # Warmup pass before the first measured method so all curves start
        # from the same warm state (ONNX sessions, CUDA context, OS page cache).
        if target is None and not args.skip_warmup:
            _warmup_pass(epsilon=args.epsilon, tv=args.tv)
        # BoundGuard: burst advances on V(t)>eps; xgb_pick validates for T_v
        # then advances to all_gpu if still > eps, else commits.
        if target is None or target == "bg":
            run(BG_CSV, mode=1, label="BoundGuard",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST_MAX,
                    "combination_xgb_pick": P_XGB_MAX,
                    "combination_all_gpu":  P_ALL_GPU,
                },
                combo_triggers={
                    "combination_burst":    "v-above",
                    "combination_xgb_pick": "validate",
                },
                epsilon=args.epsilon, tv=args.tv)
            reap_lingering_executors()
        # Adaptive hot-swap (ML-only): burst advances on V(t)>eps; xgb_pick
        # is the terminal commit (no validate — stays there).
        if target is None or target == "ml":
            run(ML_CSV, mode=1, label="Adaptive (ML-only)",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST_MAX,
                    "combination_xgb_pick": P_XGB_MAX + P_ALL_GPU,
                },
                combo_triggers={
                    "combination_burst":    "v-above",
                },
                stop_after="combination_xgb_pick",
                epsilon=args.epsilon, tv=args.tv)
            reap_lingering_executors()
        # Stop-and-restart (mode 0): burst advances on V(t)>eps; xgb_pick
        # is final (and also validates with T_v); this matches the original
        # figure's behaviour where Stop-and-restart commits to the ML pick.
        if target is None or target == "sr":
            run(SR_CSV, mode=0, label="Stop-and-restart",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST_MAX,
                    "combination_xgb_pick": P_XGB_MAX + P_ALL_GPU,
                },
                combo_triggers={
                    "combination_burst":    "v-above",
                },
                stop_after="combination_xgb_pick",
                epsilon=args.epsilon, tv=args.tv)
            reap_lingering_executors()
        # Static (mode 3): no switching, only the scheduled phase sweep.
        if target is None or target == "st":
            run(ST_CSV, mode=3, label="Static",
                combo_durations={
                    "combination_stable":   P_STABLE,
                    "combination_burst":    P_BURST_MAX + P_XGB_MAX + P_ALL_GPU,
                },
                stop_after="combination_burst")
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

    # Clip all curves to the same wall-clock span. Use the shortest of the
    # measured BoundGuard / Adaptive / Stop-and-restart traces so Static
    # doesn't extend past the other methods' last measurement.
    adaptive_ends = [tlist[-1] for tlist in (bg_t, ml_t, sr_t) if tlist]
    clip_t = min(adaptive_ends) if adaptive_ends else 60.0
    def clip(t_list, v_list):
        t_c = [t for t in t_list if t <= clip_t]
        return t_c, v_list[:len(t_c)]
    ml_t_clip, ml_v_clip = clip(ml_t, ml_v)
    sr_t_clip, sr_v_clip = clip(sr_t, sr_v)
    st_t_clip, st_v_clip = clip(st_t, st_v)
    bg_t_clip, bg_v_clip = clip(bg_t, bg_v)

    ax.plot(st_t_clip, st_v_clip, color="#8b2e2e", lw=2, ls="--",
            marker="o", markersize=5, markevery=5,
            markerfacecolor="#f4b5b5", markeredgecolor="#8b2e2e",
            markeredgewidth=0.7,
            label="Static", zorder=10)
    ax.plot(ml_t_clip, ml_v_clip, color="#8e5a1c", lw=2, ls="-",
            marker="D", markersize=5, markevery=5,
            markerfacecolor="#fad7a8", markeredgecolor="#8e5a1c",
            markeredgewidth=0.7,
            label="Adaptive hot-swap", zorder=11)

    # Stop-and-restart: break the line at cold-start gaps (service stops
    # during worker teardown/rebuild). Insert NaN to break, then draw
    # light grey dotted lines across each gap to show the connection.
    sr_color = "#5a3e7c"
    sr_fill  = "#d8c2ea"
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
            markerfacecolor=sr_fill, markeredgecolor=sr_color,
            markeredgewidth=0.6,
            label="Stop-and-restart", zorder=12)
    # Draw light grey dotted connectors across each gap
    for gs, vb, ge, va in sr_gap_segments:
        ax.plot([gs, ge], [vb, va], color="#bbbbbb", lw=1.2, ls=":", zorder=9)

    ax.plot(bg_t_clip, bg_v_clip, color="#2c5984", lw=2.2, ls="-",
            marker="^", markersize=6, markevery=5,
            markerfacecolor="#b9d0e8", markeredgecolor="#2c5984",
            markeredgewidth=0.7,
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
        ax.axvline(x=bx, color="#b03a2e", ls="-.", lw=1.0, zorder=5)
        ax.text(bx+0.3, y_max*0.19, "Input rate\nincreases",
                color="#b03a2e", fontsize=11, ha="left", va="bottom", zorder=12)

    # Locate the 1st / 2nd placement transitions by combo name, so the
    # markers are drawn correctly regardless of whether the burst phase was
    # recorded (it may be skipped when V(t) > eps triggers immediately).
    def _find_first(combo_name):
        for idx, name in bg_bounds:
            if name == combo_name:
                return bg_t[idx]
        return None
    first_transition_t = _find_first("combination_xgb_pick")
    second_transition_t = _find_first("combination_all_gpu")

    # First placement transition (stable/burst -> xgb_pick)
    if first_transition_t is not None:
        ax.axvline(x=first_transition_t, color="#c88a44", ls=":", lw=1.0, zorder=5)
        ax.text(first_transition_t+0.3, y_max*0.93, "1st placement\ntransition",
                color="#c88a44", fontsize=11, ha="left", va="center", zorder=12)

    # Second placement transition (xgb_pick -> all_gpu) — BoundGuard only
    if second_transition_t is not None:
        ax.axvline(x=second_transition_t, color="#5d87b5", ls=":", lw=1.0, zorder=5)
        ax.text(second_transition_t+0.3, y_max*0.39, "2nd placement\n($V(t)>\\epsilon$ after T)",
                color="#5d87b5", fontsize=11, ha="left", va="center", zorder=12)

    # BoundGuard recovery marker (first tick where V(t) drops to <= eps
    # after the 2nd placement transition).
    bg_recover = None
    if second_transition_t is not None:
        for i, t in enumerate(bg_t):
            if t >= second_transition_t and bg_v[i] <= eps:
                bg_recover = t; break
    # All three curves end at clip_t; set xlim just past that
    ax.set_xlim(0, clip_t + 1)
    # Tighter x-tick spacing (every 5 s) so the long flat V=0 stable region
    # doesn't dominate the plot visually.
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("Time (seconds)", fontsize=17, fontweight="bold")
    ax.set_ylabel(r"QoS Violation Score $\mathbf{V(t)}$", fontsize=17, fontweight="bold")
    ax.tick_params(axis='both', labelsize=15)
    _handles, _labels = ax.get_legend_handles_labels()
    _order = [0, 2, 1, 3]
    ax.legend([_handles[i] for i in _order], [_labels[i] for i in _order],
              loc="upper left", framealpha=0.92, fontsize=13)
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
