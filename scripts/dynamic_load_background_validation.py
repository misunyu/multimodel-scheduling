#!/usr/bin/env python3
"""
Dynamic Load Adaptation + Background Interference:
    produces dynamic_load_adaptation_background.pdf.

Same three-phase scenario as dynamic_load_validation.py, but with five
headless background ONNX workers (squeezenet, shufflenet, vgg19, gpt2,
tiny-llama) running for the full duration of every executor invocation.
This stresses both compute and memory subsystems while the foreground
models undergo the same input-rate burst and GPU-offload recovery.

Usage:
    python scripts/dynamic_load_background_validation.py
    python scripts/dynamic_load_background_validation.py --no-run
    python scripts/dynamic_load_background_validation.py --epsilon 50
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def reap_lingering_executors(cooldown_sec: float = 5.0) -> None:
    """Kill any leftover executor/worker processes from the previous scenario
    and wait for the OS/GPU to settle before launching the next one."""
    for patt in ("schedule_executor_main.py", "headless_inference_worker.py"):
        subprocess.run(["pkill", "-9", "-f", patt], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(cooldown_sec)

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

PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

WORKER = os.path.join(SCRIPT_DIR, "headless_inference_worker.py")
DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule.yaml")
STATIC_SCHEDULE  = os.path.join(PROJECT_DIR, "tests", "dynamic_load_static_schedule.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
BG_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_bg_boundguard.csv")
ST_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_bg_static.csv")

PHASE_A_DURATION = 22
PHASE_B_DURATION = 6
T_V              = 3
N_CANDIDATES     = 4
TAIL_DURATION    = 38
STATIC_TAIL_BURST = (PHASE_B_DURATION
                     + N_CANDIDATES * T_V
                     + TAIL_DURATION)

NOISE_WORKERS = [
    {"model": "models_onnx/squeezenet1.0-12.onnx",           "device": "cpu", "rate": 30},
    {"model": "models_onnx/shufflenet-v2-12.onnx",           "device": "cpu", "rate": 30},
    {"model": "models_onnx/vgg19.onnx",                      "device": "cpu", "rate": 5},
    {"model": "models_onnx/gpt2.onnx",                       "device": "cpu", "rate": 1},
    {"model": "models_onnx/tiny-llama-chat-onnx/model.onnx", "device": "cpu", "rate": 0},
]


def start_background(duration_s):
    procs = []
    for spec in NOISE_WORKERS:
        cmd = [
            PYTHON, WORKER,
            "--model", os.path.join(PROJECT_DIR, spec["model"]),
            "--device", spec["device"],
            "--rate", str(spec.get("rate", 0)),
            "--duration", str(int(duration_s)),
            "--quiet",
        ]
        p = subprocess.Popen(cmd, cwd=PROJECT_DIR,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             preexec_fn=os.setsid)
        procs.append(p)
    return procs


def stop_background(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 5.0
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def _cmd_for(mode, label, schedule, csv_path):
    common = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
    ]
    if label == "BoundGuard":
        durations = [
            ("combination_p1_low",  PHASE_A_DURATION),
            ("combination_p1_high", PHASE_B_DURATION),
            ("combination_cand_1",  T_V),
            ("combination_cand_2",  T_V),
            ("combination_cand_3",  T_V),
            ("combination_p2_high", T_V + TAIL_DURATION),
        ]
    else:
        durations = [
            ("combination_p1_low",  PHASE_A_DURATION),
            ("combination_p1_high", STATIC_TAIL_BURST),
        ]
    total = sum(d for _, d in durations)
    cmd = common + ["--duration", str(total)]
    for name, d in durations:
        cmd += ["--combo-duration", f"{name}={d}"]
    return cmd, total


def run_scenario(schedule, csv_path, mode, label):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd, total = _cmd_for(mode, label, schedule, csv_path)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = total * 4 + 90

    print()
    print("=" * 70)
    print(f"  Dynamic load + background [{label}, mode={mode}, total={total}s]")
    print(f"  Background: {[s['model'].split('/')[-1] for s in NOISE_WORKERS]}")
    print("=" * 70)

    bg_procs = start_background(total + 60)
    time.sleep(2.0)

    try:
        proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        tail = proc.stdout.decode(errors="replace").splitlines()[-5:]
        for line in tail:
            print(f"    [exec] {line}")
        if proc.returncode != 0:
            print(f"  [warn] executor exited rc={proc.returncode}")
    finally:
        stop_background(bg_procs)


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


def make_plot(bg_data, st_data, epsilon, pdf_path, x_max, load_change_sec):
    bg_rows, bg_v_t, bg_times, bg_bounds, bg_cold = bg_data
    st_rows, st_v_t, st_times, st_bounds, st_cold = st_data

    bg_t_plot, bg_v_plot = list(bg_times), list(bg_v_t)
    st_t_plot, st_v_plot = insert_nans_for_gaps(st_times, st_v_t, st_cold)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(st_t_plot, st_v_plot, color="#8b2e2e", linewidth=2.0,
            linestyle="--", marker="o", markersize=5, markevery=8,
            markerfacecolor="#f4b5b5", markeredgecolor="#8b2e2e",
            markeredgewidth=0.6,
            label="Static", zorder=10)
    ax.plot(bg_t_plot, bg_v_plot, color="#2c5984", linewidth=2.2,
            linestyle="-", marker="s", markersize=6, markevery=8,
            markerfacecolor="#b9d0e8", markeredgecolor="#2c5984",
            markeredgewidth=0.6,
            label="BoundGuard", zorder=11)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(0.015, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=12, color="#333333",
            zorder=5)

    candidate_max = max(max(bg_v_t), max(st_v_t), epsilon)
    y_max = candidate_max * 1.15
    ax.set_ylim(0.0, y_max)

    ax.axvline(x=load_change_sec, color="#b03a2e", linestyle="-.",
               linewidth=1.0, zorder=5)
    ax.annotate("Input rate\nincreases",
                xy=(load_change_sec, y_max * 0.55),
                xytext=(5.0, y_max * 0.55),
                color="#b03a2e", fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#b03a2e",
                                lw=0.8, shrinkA=2, shrinkB=4),
                zorder=12)

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
                color="#5d87b5", fontsize=11, ha="left", va="center", zorder=12)
    ax.set_xlim(0.0, x_max)

    ax.set_xlabel("Time (seconds)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"QoS Violation Score $\mathbf{V(t)}$", fontsize=15, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc="center right", framealpha=0.92, fontsize=13)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")
    return bg_recover_sec


def _eps_tag(eps):
    if eps == int(eps):
        return str(int(eps))
    return str(eps).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--out", default=None,
                        help="Output PDF (defaults to "
                             "results/dynamic_load_adaptation_background_epsilon_<eps>.pdf)")
    parser.add_argument("--run-only", choices=["bg", "st"], default=None,
                        help="Run only one curve per invocation.")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip the warmup pass before the measured runs.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(
            RESULTS_DIR,
            f"dynamic_load_adaptation_background_epsilon_{_eps_tag(args.epsilon)}.pdf",
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.no_run:
        target = args.run_only
        if target is None and not args.skip_warmup:
            warm_csv = os.path.join(RESULTS_DIR, "dynamic_load_bg_warmup.csv")
            print()
            print("=" * 70)
            print("  Warmup pass (results discarded)")
            print("=" * 70)
            run_scenario(STATIC_SCHEDULE, warm_csv, mode=3, label="Warmup")
            reap_lingering_executors()

        if target is None or target == "bg":
            run_scenario(args.schedule, BG_CSV, mode=1, label="BoundGuard")
            reap_lingering_executors()
        if target is None or target == "st":
            # Trimmed yaml (p1_low + p1_high only) so Static placement never
            # changes — no candidate combos for the executor to iterate into.
            run_scenario(STATIC_SCHEDULE, ST_CSV, mode=3, label="Static")
            reap_lingering_executors()
        if target is not None:
            print(f"\n[Done] {target} data saved.")
            return 0

    bg_data = load_curve(BG_CSV)
    st_data = load_curve(ST_CSV)

    bg_times = bg_data[2]
    st_times = st_data[2]
    x_max = min(100.0, max(max(bg_times), max(st_times)) + 1.0)

    bg_bounds = bg_data[3]
    if len(bg_bounds) >= 2:
        load_change_sec = bg_times[bg_bounds[1][0]]
    else:
        load_change_sec = float(PHASE_A_DURATION)

    bg_recover_sec = make_plot(bg_data, st_data,
                               epsilon=args.epsilon,
                               pdf_path=args.out,
                               x_max=x_max,
                               load_change_sec=load_change_sec)

    print()
    print("=" * 70)
    print("  Dynamic load + background scenario summary")
    print("=" * 70)
    print(f"  epsilon                    = {args.epsilon}")
    print(f"  Background models          = {[s['model'].split('/')[-1] for s in NOISE_WORKERS]}")
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
