#!/usr/bin/env python3
"""v38: re-measure hot-swap transition resource cost with 10 Hz sampling (ISSRE R4).

v18 answered R4, but its six re-run records are all 2026-07-27 and phantom-contaminated:
only 2 of 4 named views were actually running, so "+201 MiB host RSS, 308x headroom" is not
the peak of a four-model workload. The STRUCTURAL half of v18 survives -- every recovery
transition moves a model ACROSS devices and the generative path is kill-then-start, so old
and new never co-reside on one device -- because that came from reading the code, not from
the workload. Only the numbers need redoing.

Configuration is the confirmed table's Q3, the one misprediction scenario whose validity
survived the phantom audit: vision-3 foreground + LLaMA-1B background, CPU--GPU, lambda
25/80, buffer 12. The measurement must be taken DURING a contention-driven failure, which is
R4's actual scenario; a transition measured while idle has ample headroom and answers
nothing.

Two variants, because they differ in exactly the way R4 worries about:
  BoundGuard  q3 schedule, mode 1 + `validate` triggers -- descends five candidates, 6 swaps
  A (no dwell) q6 schedule, mode 1 + `v-above` triggers -- the most churn of any variant,
               7 swaps, which is where peaks would accumulate if they accumulate at all

usage:
  python3 scripts/run_hotswap_mem_campaign.py --reps 3
  python3 scripts/run_hotswap_mem_campaign.py --sampler-control   # observer-cost check
  python3 scripts/run_hotswap_mem_campaign.py --pilot             # one BoundGuard run
"""
import argparse, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNNER = os.path.join(PROJECT, "scripts", "run_scenario.py")

DURATIONS = ["combination_stable=15", "combination_burst=10", "cand_1=20", "cand_2=20",
             "cand_3=20", "cand_4=20", "cand_5=45"]
VARIANTS = {
    # tag: (schedule, trigger keyword, npu?)
    "memBG": ("schedules/q3_misprediction_cpu-gpu.yaml", "validate", False),
    "memA": ("schedules/q6_ablation_cpu-gpu.yaml", "v-above", False),
    # v39 task 206: the same two variants on CPU--NPU, where device memory is 16 GB rather
    # than 32 and the paper already names it the binding constraint. One schedule for both:
    # the Q5 file's own header says the corners are distinguished by executor flags alone.
    "npumemBG": ("schedules/q5_misprediction_cpu-npu.yaml", "validate", True),
    "npumemA": ("schedules/q5_misprediction_cpu-npu.yaml", "v-above", True),
}


def cmd_for(tag, rep, sampler=True):
    sched, trig, npu = VARIANTS[tag]
    c = [PY, RUNNER, "--schedule", os.path.join(PROJECT, sched), "--mode", "1",
         "--tag", tag, "--duration", "60", "--buffer", "12", "--background",
         "--rep", str(rep)]
    for d in DURATIONS:
        c += ["--combo-duration", d]
    for cand in ("cand_1", "cand_2", "cand_3", "cand_4", "cand_5"):
        c += ["--combo-trigger", f"{cand}={trig}"]
    if sampler:
        c += ["--mem"]
        if npu:
            c += ["--mem-npu"]
    return c


def run(c, label):
    print(f"[run] {label}", flush=True)
    rc = subprocess.call(c, cwd=PROJECT)
    print(f"[run] {label} rc={rc}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first rep index; v39 task 208 extends the v38 campaign to n>=8 "
                         "rather than re-running the reps that already exist")
    ap.add_argument("--variants", nargs="+", default=["memBG", "memA"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sampler-control", action="store_true")
    a = ap.parse_args()

    if a.pilot:
        return run(cmd_for("memBG", 99), "pilot memBG r99")

    if a.sampler_control:
        # Same scenario, sampler on and off, one run each. v18 saw throughput move by up to
        # -5.9% with the sampler attached; harmless for a memory footprint but recorded.
        rc = run(cmd_for("memBG", 90, sampler=True), "control WITH sampler")
        rc |= run([*cmd_for("memBG", 91, sampler=False)], "control WITHOUT sampler")
        return rc

    total, done, bad = len(a.variants) * a.reps, 0, 0
    for tag in a.variants:
        for rep in range(a.rep_start, a.rep_start + a.reps):
            done += 1
            if run(cmd_for(tag, rep), f"[{done}/{total}] {tag} r{rep}"):
                bad += 1
    print(f"campaign complete: {done} runs, {bad} nonzero rc", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
