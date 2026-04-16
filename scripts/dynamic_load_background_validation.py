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
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR, "dynamic_load_adaptation_background.pdf")
BG_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_bg_boundguard.csv")
ST_CSV  = os.path.join(RESULTS_DIR, "dynamic_load_bg_static.csv")

PHASE_A_DURATION = 22
PHASE_B_DURATION = 6
PHASE_C_DURATION = 22

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


def run_scenario(schedule, csv_path, mode, label):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    total = PHASE_A_DURATION + PHASE_B_DURATION + PHASE_C_DURATION
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--duration", str(total),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
        "--combo-duration", f"combination_p1_low={PHASE_A_DURATION}",
        "--combo-duration", f"combination_p1_high={PHASE_B_DURATION}",
        "--combo-duration", f"combination_p2_high={PHASE_C_DURATION}",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = total * 4 + 90

    print()
    print("=" * 70)
    print(f"  Dynamic load + background [{label}, mode={mode}]")
    print(f"  Phases: A={PHASE_A_DURATION}s  B={PHASE_B_DURATION}s  C={PHASE_C_DURATION}s")
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

    ax.plot(st_t_plot, st_v_plot, color="#a83232", linewidth=2.0,
            linestyle="--",
            label="Static", zorder=10)
    ax.plot(bg_t_plot, bg_v_plot, color="#1f4e79", linewidth=2.8,
            linestyle="-", marker="^", markersize=6, markevery=5,
            label="BoundGuard", zorder=11)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=12, color="#333333")

    ax.axvline(x=load_change_sec, color="#7b3306", linestyle=":",
               linewidth=1.4, zorder=5)
    ax.text(load_change_sec + 0.3, 5,
            "Input rate\nincreases",
            color="#7b3306", fontsize=11, ha="left", va="bottom", zorder=12)

    bg_recover_sec = None
    if len(bg_bounds) >= 3:
        p3_start = bg_bounds[2][0]
        for i in range(p3_start, len(bg_v_t)):
            if bg_v_t[i] <= epsilon:
                bg_recover_sec = bg_times[i]
                break
    if bg_recover_sec is not None:
        ax.axvline(x=bg_recover_sec, color="#155724", linestyle=":",
                   linewidth=1.2, zorder=5)
        ax.text(bg_recover_sec + 0.3, epsilon * 0.55,
                "BoundGuard\n stable",
                color="#155724", fontsize=11, ha="left", va="center", zorder=12)

    candidate_max = max(max(bg_v_t), max(st_v_t))
    y_max = 100.0
    ax.set_ylim(0.0, y_max)
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
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--out", default=OUT_PDF)
    parser.add_argument("--run-only", choices=["bg", "st"], default=None,
                        help="Run only one curve per invocation.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.no_run:
        target = args.run_only
        if target is None or target == "bg":
            run_scenario(args.schedule, BG_CSV, mode=1, label="BoundGuard")
        if target is None or target == "st":
            run_scenario(args.schedule, ST_CSV, mode=3, label="Static")
        if target is not None:
            print(f"\n[Done] {target} data saved.")
            return 0

    bg_data = load_curve(BG_CSV)
    st_data = load_curve(ST_CSV)

    bg_times = bg_data[2]
    st_times = st_data[2]
    x_max = max(max(bg_times), max(st_times)) + 1.0

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
