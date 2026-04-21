#!/usr/bin/env python3
"""
Runtime overhead analysis: produces runtime_overhead.pdf.

Goal:
    Quantify the steady-state cost of BoundGuard's safety layer
    (QoS monitoring, validation, rollback). The headline question
    reviewers ask is "good idea, but what does it cost when nothing
    is going wrong?" -- this script answers it by running the same
    stable workload under three execution modes and comparing four
    metrics:

        Static            (adaptive_mode 3) -- placement frozen
        Adaptive hot-swap (adaptive_mode 1) -- AdaptiveDeployManager
        BoundGuard        (adaptive_mode 2) -- ReactiveDeployManager
                                               (mode 1 + rollback)

Schedule:
    tests/steady_state_views_schedule.yaml -- single combination,
    moderate input rates, V(t) stays near zero. No placement
    transitions occur during the run, so any throughput/latency/CPU
    delta is attributable to monitoring infrastructure.

Metrics:
    1. Mean total throughput        (sum across the four views)
    2. Mean per-view inference time (avg of view{1..4}_infer_ms)
    3. Mean drop rate               (frames per second dropped at queue)
    4. Mean executor process CPU    (psutil sampling, summed over its tree)

The fourth metric is the closest direct proxy for "monitoring
overhead". The first three are user-visible quality metrics that
should remain (almost) identical between modes if the safety layer
is genuinely cheap.

Usage:
    python scripts/runtime_overhead_analysis.py
    python scripts/runtime_overhead_analysis.py --duration 30 --reps 2
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OVERHEAD_DIR = os.path.join(RESULTS_DIR, "runtime_overhead")
SCHEDULE = os.path.join(PROJECT_DIR, "tests", "steady_state_views_schedule.yaml")
EXECUTOR = os.path.join(PROJECT_DIR, "schedule_executor_main.py")
PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

OUT_PDF = os.path.join(RESULTS_DIR, "runtime_overhead.pdf")

MODES = [
    {"label": "Static",            "mode": 3},
    {"label": "Adaptive\nhot-swap", "mode": 1},
    {"label": "BoundGuard",        "mode": 2},
]


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def run_one(mode_int, csv_path, duration_s, sample_period=0.5):
    """Run schedule_executor_main once and sample the process tree CPU."""
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd = [
        PYTHON, EXECUTOR,
        "--schedule", SCHEDULE,
        "--duration", str(duration_s),
        "--adaptive-mode", str(mode_int),
        "--metrics-csv", csv_path,
        "--auto_start_all",
        "--combo-duration", f"combination_stable={duration_s}",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cpu_samples = []
    rss_samples = []
    started = time.time()
    try:
        ps_proc = psutil.Process(proc.pid)
        # prime cpu_percent: psutil's per-call delta needs an initial call
        try:
            ps_proc.cpu_percent(interval=None)
            for child in ps_proc.children(recursive=True):
                child.cpu_percent(interval=None)
        except psutil.Error:
            pass

        deadline = started + duration_s + 60
        # Skip the first ~3s (worker init) when collecting samples.
        warmup_until = started + 3.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(sample_period)
            try:
                if not ps_proc.is_running():
                    break
                cpu = ps_proc.cpu_percent(interval=None)
                rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        rss += child.memory_info().rss
                    except psutil.Error:
                        continue
                if time.time() >= warmup_until:
                    cpu_samples.append(cpu)
                    rss_samples.append(rss / (1024.0 ** 2))   # MB
            except psutil.NoSuchProcess:
                break
    finally:
        try:
            stdout, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
    rc = proc.returncode
    return {
        "rc": rc,
        "wallclock": time.time() - started,
        "cpu_samples_pct": cpu_samples,
        "rss_samples_mb": rss_samples,
        "stdout_tail": stdout.decode(errors="replace").splitlines()[-5:] if stdout else [],
        "cmd": cmd,
    }


def parse_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # Drop the first 3 ticks as warmup so we don't penalise modes that
    # take a tick longer to ramp up the workers.
    rows = rows[3:] if len(rows) > 6 else rows

    def _avg_float(col):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, 0) or 0))
            except (TypeError, ValueError):
                continue
        return float(np.mean(vals)) if vals else 0.0

    total_fps = _avg_float("total_fps")
    drop_fps = _avg_float("drop_rate_fps")

    view_lat_cols = ["view1_infer_ms", "view2_infer_ms",
                     "view3_infer_ms", "view4_infer_ms"]
    per_tick_avg = []
    for r in rows:
        per_view = []
        for c in view_lat_cols:
            try:
                v = float(r.get(c, 0) or 0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                per_view.append(v)
        if per_view:
            per_tick_avg.append(float(np.mean(per_view)))
    avg_latency_ms = float(np.mean(per_tick_avg)) if per_tick_avg else 0.0

    return {
        "rows": len(rows),
        "total_fps": total_fps,
        "drop_fps": drop_fps,
        "avg_latency_ms": avg_latency_ms,
    }


def stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "values": []}
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"mean": mean, "std": std, "n": n, "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30,
                        help="seconds to run each mode for (default 30)")
    parser.add_argument("--reps", type=int, default=2,
                        help="repetitions per mode (default 2)")
    parser.add_argument("--cooldown-sec", type=float, default=4.0)
    parser.add_argument("--out", default=OUT_PDF)
    parser.add_argument("--replot", action="store_true",
                        help="Skip running experiments; reload last sweep "
                             "from overhead_sweep.json and just regenerate the PDF.")
    args = parser.parse_args()

    os.makedirs(OVERHEAD_DIR, exist_ok=True)
    json_path = os.path.join(OVERHEAD_DIR, "overhead_sweep.json")

    if args.replot:
        if not os.path.exists(json_path):
            print(f"[Overhead] no JSON to replot from: {json_path}")
            return 1
        with open(json_path) as f:
            cached = json.load(f)
        per_mode = cached["modes"]
        args.duration = cached.get("duration_s", args.duration)
        args.reps = cached.get("reps_per_mode", args.reps)
        started = cached.get("started", _now_iso())
        finished = cached.get("finished", _now_iso())
        print(f"[Overhead] replotting from {json_path}")
    else:
        started = _now_iso()
        print(f"[Overhead] {started} starting steady-state overhead sweep")
        print(f"[Overhead] schedule={SCHEDULE} duration={args.duration}s reps={args.reps}")
        per_mode = _run_sweep(args)
        finished = _now_iso()
        out_obj = {
            "started": started,
            "finished": finished,
            "schedule": SCHEDULE,
            "duration_s": args.duration,
            "reps_per_mode": args.reps,
            "modes": per_mode,
        }
        with open(json_path, "w") as f:
            json.dump(out_obj, f, indent=2)
        print()
        print(f"[Overhead] JSON: {json_path}")

    _render(per_mode, args)
    return 0


def _run_sweep(args):
    per_mode = []
    for mi, m in enumerate(MODES):
        label_flat = m["label"].replace("\n", " ")
        print()
        print("=" * 70)
        print(f"  Mode {mi + 1}/{len(MODES)}: {label_flat} (adaptive_mode={m['mode']})")
        print("=" * 70)
        runs = []
        for rep in range(args.reps):
            print(f"  -- run {rep + 1}/{args.reps}")
            csv_path = os.path.join(OVERHEAD_DIR, f"mode{m['mode']}_rep{rep + 1}.csv")
            run_info = run_one(m["mode"], csv_path, args.duration)
            metrics = parse_metrics(csv_path)
            cpu_arr = run_info["cpu_samples_pct"]
            rss_arr = run_info["rss_samples_mb"]
            cpu_mean = float(np.mean(cpu_arr)) if cpu_arr else 0.0
            rss_mean = float(np.mean(rss_arr)) if rss_arr else 0.0
            run_entry = {
                "rep": rep + 1,
                "csv": csv_path,
                "cmd": run_info["cmd"],
                "rc": run_info["rc"],
                "wallclock": run_info["wallclock"],
                "cpu_mean_pct": cpu_mean,
                "cpu_samples_n": len(cpu_arr),
                "rss_mean_mb": rss_mean,
                "metrics": metrics,
                "stdout_tail": run_info["stdout_tail"],
            }
            runs.append(run_entry)
            print(f"     fps={metrics['total_fps']:.2f} "
                  f"lat={metrics['avg_latency_ms']:.1f}ms "
                  f"drop={metrics['drop_fps']:.2f} "
                  f"cpu%={cpu_mean:.1f} rss={rss_mean:.0f}MB")
            time.sleep(args.cooldown_sec)
        # Aggregate across reps
        per_mode.append({
            "label": m["label"],
            "mode": m["mode"],
            "runs": runs,
            "stats": {
                "total_fps":      stats([r["metrics"]["total_fps"] for r in runs]),
                "avg_latency_ms": stats([r["metrics"]["avg_latency_ms"] for r in runs]),
                "drop_fps":       stats([r["metrics"]["drop_fps"] for r in runs]),
                "cpu_mean_pct":   stats([r["cpu_mean_pct"] for r in runs]),
            },
        })

    return per_mode


def _render(per_mode, args):
    # ---------- figure ------------------------------------------------------
    labels = [m["label"] for m in per_mode]
    fps_mean   = np.array([m["stats"]["total_fps"]["mean"]      for m in per_mode])
    fps_sd     = np.array([m["stats"]["total_fps"]["std"]       for m in per_mode])
    lat_mean   = np.array([m["stats"]["avg_latency_ms"]["mean"] for m in per_mode])
    lat_sd     = np.array([m["stats"]["avg_latency_ms"]["std"]  for m in per_mode])
    drop_mean  = np.array([m["stats"]["drop_fps"]["mean"]       for m in per_mode])
    drop_sd    = np.array([m["stats"]["drop_fps"]["std"]        for m in per_mode])
    cpu_mean   = np.array([m["stats"]["cpu_mean_pct"]["mean"]   for m in per_mode])
    cpu_sd     = np.array([m["stats"]["cpu_mean_pct"]["std"]    for m in per_mode])

    base_color = "#d6d6d6"
    # Pastel palette matching q13_failure_persistence.pdf:
    # Static (pink), Adaptive hot-swap (peach), BoundGuard (blue)
    bar_colors   = ["#f4b5b5", "#fad7a8", "#b9d0e8"]
    bar_hatches  = ["///",     "xxx",     ""]
    plt.rcParams["hatch.linewidth"] = 0.4

    fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(wspace=0.40, left=0.11, right=0.97, top=0.93, bottom=0.18)

    def _bars(ax, mean, sd, ylabel, fmt="{:.1f}"):
        x = np.arange(len(labels))
        bars = ax.bar(x, mean, 0.55,
                      yerr=sd, capsize=4,
                      color=bar_colors, edgecolor="#555555", linewidth=0.5,
                      ecolor="#555555")
        for b, h in zip(bars, bar_hatches):
            b.set_hatch(h)
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, v + max(mean) * 0.02,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=13, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
        ax.tick_params(axis='y', labelsize=13)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(mean) * 1.25 if max(mean) > 0 else 1.0)

    _bars(ax2, lat_mean, lat_sd,  "Mean per-view latency (ms)", "{:.0f}")
    _bars(ax4, cpu_mean, cpu_sd,  "Mean process-tree CPU (%)",  "{:.0f}")

    # fig.suptitle removed for paper figure

    fig.savefig(args.out)
    print(f"[Plot] Saved: {args.out}")
    print()
    print("[Overhead] Summary:")
    print(f"  {'mode':<18} {'fps':>10} {'lat (ms)':>12} {'drop':>10} {'cpu %':>10}")
    for m in per_mode:
        st = m["stats"]
        print(f"  {m['label'].replace(chr(10),' '):<18} "
              f"{st['total_fps']['mean']:>10.2f} "
              f"{st['avg_latency_ms']['mean']:>12.2f} "
              f"{st['drop_fps']['mean']:>10.3f} "
              f"{st['cpu_mean_pct']['mean']:>10.1f}")


if __name__ == "__main__":
    sys.exit(main())
