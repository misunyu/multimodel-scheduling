#!/usr/bin/env python3
"""v34 Q1.5 buffer sweep (tab:b2): 2 platforms x 4 techniques x 3 beta_B.

Conditions are fixed in docs/supplementary_conditions_v34.md and are NOT adjusted here.

Time-matching (the published harness gave BoundGuard 18 s more than the others, which
docs/backlog_preserve_report.md flagged as a defect): every technique gets the same total
window -- 1 s stable + 43 s of failure/recovery. Static spends all 43 in the failure
placement; the others spend 8 there and 35 in the candidate. The arrival process, buffer
and deadline are identical across techniques, which is the whole condition (177-4): this
experiment judges request-level outcomes, not recovery, so recovery is recorded but is
not a gate.

Reps are asymmetric on purpose: beta_B=0.5 is the tab:b2 regime and gets 5 like every v33
scenario; the other two regimes feed the sweep figure only and get 3. Recorded rather than
smoothed over.

usage: python3 scripts/run_q15_campaign.py [--platforms gpu npu] [--betas 0.5 1.0 1.5]
"""
import argparse, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNNER = os.path.join(PROJECT, "scripts", "run_scenario.py")

# Q_max scaled with lambda so B/lambda (the absorbable seconds) is preserved -- see 177-3.
QMAX = {"gpu": 503, "npu": 379}
SCHED = {"gpu": "schedules/q15_buffer_gpu_cpu-gpu.yaml",
         "npu": "schedules/q15_buffer_npu_cpu-npu.yaml"}
SCHED_FAIL = {"gpu": "schedules/q15_buffer_gpu_cpu-gpu_failure_only.yaml",
              "npu": "schedules/q15_buffer_npu_cpu-npu_failure_only.yaml"}

# technique -> (mode, uses failure-only schedule, extra args)
TECH = {
    "Static":       (3, True,  ["--combo-duration", "combination_burst=44",
                                "--stop-after", "combination_burst"]),
    "Stop-restart": (0, False, ["--combo-duration", "combination_stable=1",
                                "--combo-duration", "combination_burst=8",
                                "--combo-duration", "cand_1=35",
                                "--stop-after", "cand_1"]),
    "Adaptive":     (1, False, ["--combo-duration", "combination_stable=1",
                                "--combo-duration", "combination_burst=8",
                                "--combo-duration", "cand_1=35",
                                "--stop-after", "cand_1"]),
    "BoundGuard":   (1, False, ["--combo-duration", "combination_stable=1",
                                "--combo-duration", "combination_burst=8",
                                "--combo-duration", "cand_1=35",
                                "--combo-trigger", "cand_1=validate",
                                "--stop-after", "cand_1"]),
}
REPS = {"0.5": 5, "1.0": 3, "1.5": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="+", default=["gpu", "npu"])
    ap.add_argument("--betas", nargs="+", default=["0.5", "1.0", "1.5"])
    ap.add_argument("--techniques", nargs="+", default=list(TECH))
    a = ap.parse_args()

    total = sum(REPS[b] for _ in a.platforms for _ in a.techniques for b in a.betas)
    done = 0
    for plat in a.platforms:
        for beta in a.betas:
            buf = int(round(float(beta) * QMAX[plat]))
            for tech in a.techniques:
                mode, fail_only, extra = TECH[tech]
                sched = (SCHED_FAIL if fail_only else SCHED)[plat]
                for rep in range(REPS[beta]):
                    tag = f"q15{plat}_{tech}_b{beta}"
                    cmd = [PY, RUNNER, "--schedule", os.path.join(PROJECT, sched),
                           "--mode", str(mode), "--tag", tag, "--duration", "44",
                           "--perframe", "--buffer", str(buf), "--rep", str(rep)] + extra
                    done += 1
                    print(f"[{done}/{total}] {tag} r{rep} buffer={buf}", flush=True)
                    # one scenario at a time: this blocks until the executor exits
                    rc = subprocess.call(cmd, cwd=PROJECT)
                    if rc != 0:
                        print(f"FAILED rc={rc} on {tag} r{rep} -- stopping this section",
                              flush=True)
                        return 1
    print("q15 campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
