#!/usr/bin/env python3
"""v31 main re-run driver: one scenario's methods x reps, into tracked run dirs.

Method -> executor configuration (docs/c2_reactive_baseline_report.md section 2):
  Static           mode 3, stays on the failure phase (never advances to candidates)
  Stop-restart     mode 0, applies cand_1 and stays
  Adaptive         mode 1, applies cand_1 and stays (hot-swap, no dwell/progress)
  BoundGuard       mode 1 + per-candidate `validate` triggers (dwell + progress)
  A (no-dwell)     mode 1 + `v-above` triggers            (progress, no dwell)
  B (re-invoke)    mode 1, no triggers                    (= Adaptive corner)
  C (hybrid)       mode 1 + `validate`, FSRR_NO_AUTOADVANCE=1 (dwell, no progress)

Settings come from the v29 confirmed table and are NOT adjusted based on results.
"""
import argparse, os, subprocess, sys, time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(PROJECT, ".venv", "bin", "python3")
RUNNER = os.path.join(PROJECT, "scripts", "run_scenario.py")
WINDOW = 100          # fixed observation window (c2_integrity_report_v2 section 7)
STABLE, BURST = 15, 10
CANDS = [f"cand_{i}" for i in range(1, 6)]


def method_args(method):
    """(mode, triggers, extra_env, combo_durations, stop_after)

    stop_after matters: without it the executor keeps walking the schedule, and every combo
    with no explicit --combo-duration runs for the DEFAULT duration -- a 100 s Static run
    actually took 600 s (15 + 85 + 100x5). Methods that must hold one phase are truncated
    there instead.
    """
    rest = WINDOW - STABLE
    if method == "Static":
        # mode 3 never reconfigures, so it must START in the failure placement -- running
        # stable first and expecting a later switch left it on the healthy placement and
        # V never rose (caught in stage 1).
        return 3, [], {}, [f"combination_burst={WINDOW}"], None
    dur = [f"combination_stable={STABLE}", f"combination_burst={BURST}",
           f"cand_1={WINDOW - STABLE - BURST}"]
    if method == "Stop-restart":
        return 0, [], {}, dur, "cand_1"
    if method in ("Adaptive", "B"):
        return 1, [], {}, dur, "cand_1"
    # traversing methods: triggers advance candidates. Cap each candidate so a trigger that
    # does not fire cannot stretch the tail; the last one holds the rest of the window.
    trig_dur = [f"combination_stable={STABLE}", f"combination_burst={BURST}"]
    trig_dur += [f"cand_{i}=20" for i in range(1, 5)] + ["cand_5=45"]
    if method == "BoundGuard":
        return 1, [f"{c}=validate" for c in CANDS], {}, trig_dur, None
    if method == "A":
        return 1, [f"{c}=v-above" for c in CANDS], {}, trig_dur, None
    if method == "C":
        return 1, [f"{c}=validate" for c in CANDS], {"FSRR_NO_AUTOADVANCE": "1"}, trig_dur, None
    raise SystemExit(f"unknown method {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--tag-prefix", required=True)
    ap.add_argument("--methods", required=True, help="comma separated")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--background", action="store_true")
    a = ap.parse_args()

    methods = a.methods.split(",")
    t0 = time.time()
    total = len(methods) * a.reps
    done = 0
    for method in methods:
        mode, trig, extra, dur, stop_after = method_args(method)
        for rep in range(a.reps):
            sched = a.schedule
            if method == "Static":
                fo = a.schedule.replace(".yaml", "_failure_only.yaml")
                if os.path.isfile(os.path.join(PROJECT, fo)):
                    sched = fo
            cmd = [PY, RUNNER, "--schedule", sched, "--mode", str(mode),
                   "--tag", f"{a.tag_prefix}_{method}", "--duration", str(WINDOW),
                   "--buffer", a.buffer, "--rep", str(rep)]
            for d in dur:
                cmd += ["--combo-duration", d]
            for t in trig:
                cmd += ["--combo-trigger", t]
            if stop_after:
                cmd += ["--stop-after", stop_after]

            if a.background:
                cmd += ["--background"]
            env = os.environ.copy()
            env.update(extra)
            r = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True)
            done += 1
            line = (r.stdout or "").strip().splitlines()
            print(f"[{done}/{total}] {method} rep{rep}: {line[-1] if line else r.returncode}",
                  flush=True)
    print(f"[campaign] {a.tag_prefix} done: {total} runs in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
