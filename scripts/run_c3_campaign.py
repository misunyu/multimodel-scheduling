#!/usr/bin/env python3
"""v35 C3 fluid-model revalidation sweep: 2 platforms x 3 lambda/mu* x 3 reps.

Conditions are fixed in docs/c3_revalidation_conditions.md and are NOT adjusted here.

Phase structure per run:
    combination_stable   3 s   all-accelerator, settle
    combination_burst   18 s   ALL-CPU  -> accumulation, dQ/dt = lambda - mu_cpu
    cand_1              35 s   all-accelerator -> drain, dQ/dt = -(mu* - lambda)

No QoS triggers are passed, so the controller makes no decisions: `QoS-triggered advance`
must be 0 in every log, and that is the evidence the placement is schedule-dictated rather
than adaptive. The B->C transition is the experiment itself and goes through the hot-swap
path (mode 1) because it must PRESERVE the backlog -- mode 0 recreates the feeders and
resets the counters (docs/c3_stress_report.md section 3).

The drain is deliberately not run to completion: at 0.95 mu* emptying the queue would take
minutes, which IS the divergence. 35 s is a window to fit the slope in; T_stable is then
Q/|dQ/dt|.

Buffer 8000/view so nothing is dropped -- the fluid model assumes queue conservation, so a
drop invalidates the prediction rather than merely perturbing it.

usage: python3 scripts/run_c3_campaign.py [--platforms gpu npu] [--reps 3]
"""
import argparse, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNNER = os.path.join(PROJECT, "scripts", "run_scenario.py")
FRACS = ["070", "090", "095"]
SUFFIX = {"gpu": "cpu-gpu", "npu": "cpu-npu"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="+", default=["gpu", "npu"])
    ap.add_argument("--fracs", nargs="+", default=FRACS)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    total = len(a.platforms) * len(a.fracs) * a.reps
    done = 0
    for plat in a.platforms:
        for frac in a.fracs:
            sched = os.path.join(PROJECT, "schedules",
                                 f"c3_fluid_{plat}_{frac}_{SUFFIX[plat]}.yaml")
            for rep in range(a.reps):
                tag = f"c3{plat}_{frac}"
                cmd = [PY, RUNNER, "--schedule", sched, "--mode", "1", "--tag", tag,
                       "--duration", "60", "--buffer", "8000", "--perframe",
                       "--combo-duration", "combination_stable=3",
                       "--combo-duration", "combination_burst=18",
                       "--combo-duration", "cand_1=35",
                       "--stop-after", "cand_1", "--rep", str(rep)]
                done += 1
                print(f"[{done}/{total}] {tag} r{rep}", flush=True)
                rc = subprocess.call(cmd, cwd=PROJECT)
                if rc != 0:
                    print(f"FAILED rc={rc} on {tag} r{rep} -- stopping", flush=True)
                    return 1
    print("c3 campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
