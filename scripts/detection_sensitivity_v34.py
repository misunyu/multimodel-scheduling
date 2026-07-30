#!/usr/bin/env python3
"""Q1.2 detection sensitivity, re-windowed from the COMMITTED v33 runs.

This is not a new measurement. The four metrics and the 3x3 (T, eps) grid are taken
unchanged from scripts/detection_sensitivity_analysis.py; the only thing that changes is
the input. The original read results/bounded_sweep/run_s*_rep*.csv, which is not in the
repository and cannot be reconstructed -- so the figure could not be regenerated at all.
The v33 re-run left 269 committed run directories whose metrics.csv carries the same
per-tick v_score column, which makes this figure reproducible for the first time.

Two deliberate differences from the original, both recorded:

  1. Phase split. The original took the FIRST THREE combination boundaries as
     (healthy, failure, recovery). That silently truncates any run that traverses more
     than one candidate: for a BoundGuard run landing on cand_5, "phase 3" was cand_1,
     so recovery latency was measured against a placement that never recovered. Here
     phase 3 is everything after the failure phase, which is what the metric means.

  2. Run selection. BoundGuard arms only -- the detection knobs (T, eps) govern when the
     controller acts, so arms that never act carry no information about them. The
     INVALID lambda=123/93 Q1.3 runs are excluded by name (they are preserved on disk but
     measured nothing; runs/README.md).

usage: python3 scripts/detection_sensitivity_v34.py [--json out.json]
"""
import argparse, csv, glob, json, os, statistics as st, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_VALUES = [1, 3, 5]
EPS_VALUES = [0.5, 1.0, 2.0]
# v33 scenarios, BoundGuard arm. q13gpu/q13npu WITHOUT the V2 suffix are the invalid
# lambda=123/93 runs and are matched out below.
SCENARIO_GLOBS = ["*_q13gpuV2_BoundGuard_r*", "*_q13npuV2_BoundGuard_r*",
                  "*_q4b_BoundGuard_r*", "*_q3_BoundGuard_r*", "*_q6_BoundGuard_r*"]

# Excluded from the grid, with the reason. Both are reported separately (--show-excluded)
# so the exclusion is visible rather than silent.
#
#   q5  its "healthy" phase is NOT healthy: the stable placement already violates
#       (stable maxV 13.56, and t0 precedes the burst by 9.6 s -- docs/main_rerun_report.md
#       section 1). Those ticks are TRUE detections, not false triggers. Worse, 13.56
#       exceeds every eps on the grid, so Q5 contributed an identical count to all nine
#       cells and flattened the false-trigger metric completely (1.37 in every cell).
#   q4  no candidate ever recovers and its V is two orders of magnitude larger
#       (cum_violation 16235 vs 130-385). Averaged in, it dominates the mean and the
#       spread exceeded the mean, so the grid said nothing about T or eps.
EXCLUDED = {"*_q5_BoundGuard_r*": "stable phase already violates; its ticks are true "
                                  "detections, not false triggers",
            "*_q4_BoundGuard_r*": "never recovers and V is 2 orders larger; dominates "
                                  "every average"}


def load_run(path):
    rows = list(csv.DictReader(open(path)))
    return ([float(r.get("v_score", 0) or 0) for r in rows],
            [r.get("combination", "") for r in rows])


def windowed(v, T):
    return [sum(v[max(0, i - T + 1):i + 1]) / len(v[max(0, i - T + 1):i + 1])
            for i in range(len(v))]


def phases(combos):
    """(healthy, failure, after) as ranges. 'after' is everything past the failure phase,
    not just the next combination -- see the docstring."""
    if not combos:
        return range(0), range(0), range(0)
    bounds = [0] + [i for i in range(1, len(combos)) if combos[i] != combos[i - 1]] + [len(combos)]
    if len(bounds) < 3:
        return range(bounds[0], bounds[-1]), range(0), range(0)
    return (range(bounds[0], bounds[1]),          # healthy: the stable phase
            range(bounds[1], bounds[2]),          # failure: the burst phase
            range(bounds[2], bounds[-1]))         # after: every candidate that follows


def metrics(v_tick, combos, T, eps):
    v = windowed(v_tick, T)
    p1, p2, p3 = phases(combos)
    false_triggers = sum(1 for i in p1 if v[i] > eps)
    detect = float("nan")
    if len(p2):
        for i in range(p2.start + T - 1, p2.stop):
            if v[i] > eps:
                detect = float(i - p2.start)
                break
    recover = float("nan")
    if len(p2) and len(p3):
        for i in p3:
            if v[i] <= eps:
                recover = float(i - p2.start)
                break
    return {"false_triggers": float(false_triggers), "detect_lat": detect,
            "recover_lat": recover,
            "cum_violation": sum(max(0.0, x - eps) for x in v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--show-excluded", action="store_true",
                    help="also print the excluded scenarios at the operating point")
    a = ap.parse_args()
    for g, why in EXCLUDED.items():
        n = len(glob.glob(os.path.join(PROJECT, "runs", g)))
        print(f"[excluded] {g}  ({n} runs) -- {why}")
        if a.show_excluded:
            for d in sorted(glob.glob(os.path.join(PROJECT, "runs", g)))[:1]:
                v, c = load_run(os.path.join(d, "metrics.csv"))
                r = metrics(v, c, 3, 1.0)
                print(f"           at T=3, eps=1.0: {r}")
    print()
    runs = []
    for g in SCENARIO_GLOBS:
        for d in sorted(glob.glob(os.path.join(PROJECT, "runs", g))):
            p = os.path.join(d, "metrics.csv")
            if os.path.isfile(p):
                runs.append(d)
    if not runs:
        print("no runs matched", file=sys.stderr)
        return 1
    loaded = [load_run(os.path.join(d, "metrics.csv")) for d in runs]
    grid = {}
    for T in T_VALUES:
        for eps in EPS_VALUES:
            per = [metrics(v, c, T, eps) for v, c in loaded]
            cell = {}
            for k in ("false_triggers", "detect_lat", "recover_lat", "cum_violation"):
                vals = [p[k] for p in per if p[k] == p[k]]
                cell[k] = (round(st.mean(vals), 3), round(st.pstdev(vals), 3),
                           len(vals)) if vals else None
            grid[f"T{T}_eps{eps}"] = cell
    out = {"n_runs": len(runs), "runs": [os.path.basename(d) for d in runs], "grid": grid}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    print(f"re-windowed {len(runs)} committed BoundGuard runs\n")
    for k in ("false_triggers", "detect_lat", "recover_lat", "cum_violation"):
        print(f"--- {k}")
        print("      " + "".join(f"{'eps='+str(e):>16s}" for e in EPS_VALUES))
        for T in T_VALUES:
            row = f"T={T}  "
            for eps in EPS_VALUES:
                c = grid[f"T{T}_eps{eps}"][k]
                row += f"{c[0]:10.2f}±{c[1]:<5.2f}" if c else f"{'--':>16s}"
            print(row)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
