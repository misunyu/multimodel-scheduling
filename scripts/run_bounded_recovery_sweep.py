#!/usr/bin/env python3
"""
Bounded-recovery sweep driver.

For each workload scenario:
  1. Launch a fixed set of *background* (non-displayed) ONNX inference
     workers via scripts/headless_inference_worker.py.
  2. Run the qos_recovery 3-phase scenario through schedule_executor_main
     with the four-view schedule (mnasnet/resnet50/resnext50/yolov4).
  3. Parse the resulting CSV, extract t_0, t_detect, t_recover, and
     compute T_detect, T_post, T_total.
  4. Stop the background workers, wait for cleanup, repeat N times.

The aggregated mean ± std for every scenario is written to a JSON file
that scripts/bounded_recovery_validation.py reads to render the figure.

This script saves enough metadata in the JSON output to fully reproduce
the experiment: scenario configs, background worker command lines,
schedule YAML path, environment knobs, and per-run raw measurements.

Usage:
    python scripts/run_bounded_recovery_sweep.py
    python scripts/run_bounded_recovery_sweep.py --reps 3 --epsilon 50
"""
import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from qos_recovery_validation import (   # noqa: E402
    compute_windowed_v,
    find_phase_boundaries,
    load_csv,
    parse_timestamps,
    WINDOW_T,
)

PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

EXECUTOR = os.path.join(PROJECT_DIR, "schedule_executor_main.py")
WORKER   = os.path.join(SCRIPT_DIR, "headless_inference_worker.py")
SCHEDULE = os.path.join(PROJECT_DIR, "tests", "bounded_recovery_views_schedule.yaml")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SWEEP_DIR   = os.path.join(RESULTS_DIR, "bounded_sweep")

# ---------------------------------------------------------------------------
# Scenarios
#
# Each scenario re-uses the SAME view schedule (4 view models). What
# differs is the background load: a list of headless ONNX inference
# processes that consume CPU/GPU resources but do NOT contribute to V(t)
# (the metric is computed only over view handlers).
#
# Background spec : list of dicts {model, device, rate}
#   model  : path under models_onnx/
#   device : "cpu" or "gpu"
#   rate   : target inferences per second (0 = unbounded)
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "label": "Baseline\n(no bg)",
        "background": [],
    },
    {
        "label": "+1 light\nCPU bg",
        "background": [
            {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
        ],
    },
    {
        "label": "+2 CPU bg\n(squeeze+shuf)",
        "background": [
            {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
            {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
        ],
    },
    {
        "label": "+heavy CPU\n(squeeze+shuf+vgg)",
        "background": [
            {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
            {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
            {"model": "models_onnx/vgg19.onnx",            "device": "cpu", "rate": 5},
        ],
    },
    {
        "label": "+GPT-2\n(+gpt2 CPU)",
        "background": [
            {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
            {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
            {"model": "models_onnx/vgg19.onnx",            "device": "cpu", "rate": 5},
            {"model": "models_onnx/gpt2.onnx",             "device": "cpu", "rate": 1},
        ],
    },
    {
        "label": "+TinyLlama\n(+llama CPU)",
        "background": [
            {"model": "models_onnx/squeezenet1.0-12.onnx",            "device": "cpu", "rate": 30},
            {"model": "models_onnx/shufflenet-v2-12.onnx",            "device": "cpu", "rate": 30},
            {"model": "models_onnx/vgg19.onnx",                       "device": "cpu", "rate": 5},
            {"model": "models_onnx/gpt2.onnx",                        "device": "cpu", "rate": 1},
            {"model": "models_onnx/tiny-llama-chat-onnx/model.onnx",  "device": "cpu", "rate": 0},
        ],
    },
]


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def start_background(spec_list, duration_s):
    """Spawn one Popen per background spec. Returns list of Popen handles."""
    procs = []
    for spec in spec_list:
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
        procs.append((p, cmd))
    return procs


def stop_background(procs):
    for p, _cmd in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 5.0
    for p, _cmd in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def run_executor(csv_path, baseline_dur, failure_dur, recovery_dur):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd = [
        PYTHON, EXECUTOR,
        "--schedule", SCHEDULE,
        "--duration", str(max(baseline_dur, failure_dur, recovery_dur)),
        "--adaptive-mode", "0",
        "--metrics-csv", csv_path,
        "--auto_start_all",
        "--combo-duration", f"combination_initial={baseline_dur}",
        "--combo-duration", f"combination_overload={failure_dur}",
        "--combo-duration", f"combination_offload={recovery_dur}",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = (baseline_dur + failure_dur + recovery_dur) * 4 + 90
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "wallclock": time.time() - started,
        "stdout_tail": proc.stdout.decode(errors="replace").splitlines()[-5:],
    }


def measure_from_csv(csv_path, epsilon):
    """Replicate the qos_recovery_validation marker logic on the CSV.

    Returns dict with raw row indices and seconds for t_0, t_detect,
    t_recover, plus T_detect/T_post/T_total. Raises RuntimeError on
    malformed/insufficient data so the driver can retry."""
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times_sec = parse_timestamps(rows)

    boundaries = find_phase_boundaries(rows)
    combos = [b[1] for b in boundaries]
    if len(boundaries) < 3:
        raise RuntimeError(f"expected 3 phases, got {combos}")
    p1_start = boundaries[0][0]
    p2_start = boundaries[1][0]
    p3_start = boundaries[2][0]
    p3_end = len(rows)

    detection_earliest = p2_start + WINDOW_T - 1
    t_detect_row = None
    for i in range(detection_earliest, p3_start):
        if v_t[i] > epsilon:
            t_detect_row = i
            break
    if t_detect_row is None:
        raise RuntimeError(
            f"V(t) never crossed epsilon={epsilon} in overload phase "
            f"(rows {p2_start}..{p3_start - 1})"
        )

    t_recover_row = None
    for i in range(p3_start, p3_end):
        if v_t[i] <= epsilon:
            t_recover_row = i
            break
    if t_recover_row is None:
        raise RuntimeError(
            f"V(t) never returned to epsilon={epsilon} during offload phase"
        )

    p2_start_sec  = times_sec[p2_start]
    t_detect_sec  = times_sec[t_detect_row]
    t_recover_sec = times_sec[t_recover_row]

    T_detect = t_detect_sec - p2_start_sec
    T_post   = t_recover_sec - t_detect_sec
    T_total  = t_recover_sec - p2_start_sec

    return {
        "rows": len(rows),
        "p1_start": p1_start,
        "p2_start": p2_start,
        "p3_start": p3_start,
        "t_detect_row":  t_detect_row,
        "t_recover_row": t_recover_row,
        "p2_start_sec":  p2_start_sec,
        "t_detect_sec":  t_detect_sec,
        "t_recover_sec": t_recover_sec,
        "T_detect": T_detect,
        "T_post":   T_post,
        "T_total":  T_total,
        "v_t_max":  max(v_t),
    }


def stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "values": []}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n   # population variance
    return {"mean": mean, "std": var ** 0.5, "n": n, "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3,
                        help="repetitions per scenario (default 3)")
    parser.add_argument("--epsilon", type=float, default=50.0,
                        help="V(t) detection threshold (default 50)")
    parser.add_argument("--baseline-duration", type=int, default=8)
    parser.add_argument("--failure-duration",  type=int, default=14)
    parser.add_argument("--recovery-duration", type=int, default=16)
    parser.add_argument("--bg-warmup-sec", type=float, default=2.0,
                        help="Wait this long after launching background "
                             "workers before starting the executor.")
    parser.add_argument("--cooldown-sec", type=float, default=4.0,
                        help="Wait between repetitions to let the OS / GPU "
                             "settle.")
    parser.add_argument("--out", default=os.path.join(SWEEP_DIR, "sweep.json"))
    args = parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    sweep_started = _now_iso()
    print(f"[Sweep] {sweep_started}: starting bounded recovery sweep")
    print(f"[Sweep] reps={args.reps} epsilon={args.epsilon} schedule={SCHEDULE}")

    per_scenario = []
    for s_idx, scenario in enumerate(SCENARIOS):
        s_label_flat = scenario["label"].replace("\n", " ")
        print()
        print("=" * 70)
        print(f"  Scenario {s_idx + 1}/{len(SCENARIOS)}: {s_label_flat}")
        print(f"  Background: {scenario['background'] or 'none'}")
        print("=" * 70)
        runs = []
        for rep in range(args.reps):
            print(f"  -- run {rep + 1}/{args.reps}")
            csv_name = f"run_s{s_idx + 1}_rep{rep + 1}.csv"
            csv_path = os.path.join(SWEEP_DIR, csv_name)

            bg_duration = (args.baseline_duration + args.failure_duration
                           + args.recovery_duration + 60)
            bg_procs = start_background(scenario["background"], bg_duration)
            time.sleep(args.bg_warmup_sec)

            run_entry = {
                "rep": rep + 1,
                "csv": csv_path,
                "background_cmds": [shlex.join(c) for _, c in bg_procs],
                "bg_warmup_sec": args.bg_warmup_sec,
                "started": _now_iso(),
            }
            try:
                exec_info = run_executor(
                    csv_path,
                    args.baseline_duration,
                    args.failure_duration,
                    args.recovery_duration,
                )
                run_entry["executor"] = {
                    "rc": exec_info["rc"],
                    "wallclock": exec_info["wallclock"],
                    "stdout_tail": exec_info["stdout_tail"],
                    "cmd": exec_info["cmd"],
                }
                metrics = measure_from_csv(csv_path, args.epsilon)
                run_entry.update(metrics)
                run_entry["status"] = "ok"
                print(f"     T_detect={metrics['T_detect']:.2f}s "
                      f"T_post={metrics['T_post']:.2f}s "
                      f"T_total={metrics['T_total']:.2f}s "
                      f"V(t)_max={metrics['v_t_max']:.1f}")
            except Exception as e:
                run_entry["status"] = "error"
                run_entry["error"] = str(e)
                print(f"     ERROR: {e}")
            finally:
                stop_background(bg_procs)
                run_entry["finished"] = _now_iso()

            runs.append(run_entry)
            time.sleep(args.cooldown_sec)

        ok_runs = [r for r in runs if r.get("status") == "ok"]
        td_vals = [r["T_detect"] for r in ok_runs]
        tp_vals = [r["T_post"]   for r in ok_runs]
        tt_vals = [r["T_total"]  for r in ok_runs]
        # Bound = window length T (worst-case detection) + per-run T_post
        bound_vals = [WINDOW_T + tp for tp in tp_vals]

        per_scenario.append({
            "index": s_idx + 1,
            "label": scenario["label"],
            "background": scenario["background"],
            "runs": runs,
            "stats": {
                "T_detect": stats(td_vals),
                "T_post":   stats(tp_vals),
                "T_total":  stats(tt_vals),
                "bound":    stats(bound_vals),
            },
        })

    sweep_finished = _now_iso()
    out_obj = {
        "started": sweep_started,
        "finished": sweep_finished,
        "epsilon": args.epsilon,
        "window_T": WINDOW_T,
        "schedule": SCHEDULE,
        "executor_script": EXECUTOR,
        "worker_script": WORKER,
        "python": PYTHON,
        "reps_per_scenario": args.reps,
        "baseline_duration": args.baseline_duration,
        "failure_duration": args.failure_duration,
        "recovery_duration": args.recovery_duration,
        "bg_warmup_sec": args.bg_warmup_sec,
        "cooldown_sec": args.cooldown_sec,
        "scenarios": per_scenario,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print()
    print("=" * 70)
    print(f"[Sweep] Done. Wrote {args.out}")
    print("[Sweep] Summary (mean of T_detect / T_post / T_total / bound):")
    for s in per_scenario:
        st = s["stats"]
        flat = s["label"].replace("\n", " ")
        print(f"  S{s['index']} {flat:<28s} "
              f"d={st['T_detect']['mean']:5.2f}±{st['T_detect']['std']:.2f}  "
              f"p={st['T_post']  ['mean']:5.2f}±{st['T_post']  ['std']:.2f}  "
              f"t={st['T_total'] ['mean']:5.2f}±{st['T_total'] ['std']:.2f}  "
              f"b={st['bound']   ['mean']:5.2f}±{st['bound']   ['std']:.2f}")


if __name__ == "__main__":
    main()
