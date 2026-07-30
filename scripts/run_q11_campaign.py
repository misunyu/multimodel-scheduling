#!/usr/bin/env python3
"""v34 Q1.1 steady-state overhead: what the safety layer costs when nothing is wrong.

Conditions: docs/supplementary_conditions_v34.md. vision-3 at a low uniform rate so
V(t) ~ 0 and no placement transition ever happens; only combination_stable is executed.

Arms (v33 mode mapping, NOT the old script's):
    Static      mode 3            -- no monitoring threads at all
    Adaptive    mode 1            -- V(t) monitored, transitions armed
    BoundGuard  mode 1 + validate -- the above plus post-swap validation armed

CAVEAT recorded rather than hidden: in steady state Adaptive and BoundGuard execute the
same code path, because everything BoundGuard adds (the dwell window, the ranked descent)
is per-transition and no transition occurs. The comparison that carries information is
therefore monitored (mode 1) vs unmonitored (mode 3); the BoundGuard arm exists to show
that arming validation costs nothing extra, which is a claim about zero, not a measurement
of a distinct system.

Metrics -- note metric 1 is NOT metrics.csv's total_fps. That column is the sum of
1000/infer_ms, i.e. CAPACITY, so the published figure's "throughput" and "per-view latency"
were the same quantity twice. Actual throughput is counted from perframe.csv.

usage: python3 scripts/run_q11_campaign.py [--platforms gpu npu] [--reps 3]
"""
import argparse, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNNER = os.path.join(PROJECT, "scripts", "run_scenario.py")
SCHED = {"gpu": "schedules/q11_steady_gpu_cpu-gpu.yaml",
         "npu": "schedules/q11_steady_npu_cpu-npu.yaml"}
ARMS = {"Static": (3, []), "Adaptive": (1, []),
        "BoundGuard": (1, ["--combo-trigger", "cand_1=validate",
                           "--combo-trigger", "cand_2=validate",
                           "--combo-trigger", "cand_3=validate",
                           "--combo-trigger", "cand_4=validate",
                           "--combo-trigger", "cand_5=validate"])}
DURATION = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="+", default=["gpu", "npu"])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    total = len(a.platforms) * len(ARMS) * a.reps
    done = 0
    for plat in a.platforms:
        for arm, (mode, extra) in ARMS.items():
            for rep in range(a.reps):
                tag = f"q11{plat}_{arm}"
                cmd = [PY, RUNNER, "--schedule", os.path.join(PROJECT, SCHED[plat]),
                       "--mode", str(mode), "--tag", tag, "--duration", str(DURATION),
                       "--perframe", "--proc-cpu", "--buffer", "2",
                       "--combo-duration", f"combination_stable={DURATION}",
                       "--stop-after", "combination_stable", "--rep", str(rep)] + extra
                done += 1
                print(f"[{done}/{total}] {tag} r{rep}", flush=True)
                rc = subprocess.call(cmd, cwd=PROJECT)
                if rc != 0:
                    print(f"FAILED rc={rc} on {tag} r{rep} -- stopping", flush=True)
                    return 1
    print("q11 campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
