#!/usr/bin/env python3
"""Recompute the v33 campaign aggregates from committed runs, per scenario and variant.

Why this exists: `docs/main_rerun_report.md` tabulates terminal V, cumulative violation and
hot-swap counts for every scenario, but the script that produced those tables was never
committed. Registering those numbers in the confirmed table requires `source` to point at a
run directory (runs/README.md), and a source you cannot recompute from is a source in name
only -- that is exactly how the audit chain broke three times before.

So this recomputes them. The confirmed table still carries the REPORT's values verbatim
(v37 does not recompute values); this script's job is to say whether those values are
re-derivable from what is committed, and to make any disagreement visible instead of silent.

Metrics, over a window of fixed length from the first sample:
  lastV   v_score of the final sample in the window
  maxV    max v_score in the window
  cumV    time-weighted integral of v_score over the window (1 s samples -> sum)
  hotswaps  per-view 'hot-swap complete' events in the executor log

usage:
  python3 scripts/aggregate_v33_campaign.py                 # every scenario, 94 s window
  python3 scripts/aggregate_v33_campaign.py --window 100    # the c2/Q6 convention
  python3 scripts/aggregate_v33_campaign.py --scenario q4 --json out.json
"""
import argparse, csv, datetime, glob, json, os, re, statistics as st, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = "%Y-%m-%d %H:%M:%S"

# scenario -> (tag glob prefix, variants in report order)
SCENARIOS = {
    "q13gpu": ("q13gpuV2", ["BoundGuard", "Adaptive", "Stop-restart", "Static"]),
    "q13npu": ("q13npuV2", ["BoundGuard", "Adaptive", "Stop-restart", "Static"]),
    "q3": ("q3", ["BoundGuard", "Adaptive", "Stop-restart", "Static"]),
    "q5": ("q5", ["BoundGuard", "Adaptive", "Stop-restart", "Static"]),
    "q4": ("q4", ["BoundGuard", "A", "B", "C", "Adaptive"]),
    "q4b": ("q4b", ["BoundGuard", "A", "B", "C", "Adaptive"]),
    "q6": ("q6", ["BoundGuard", "A", "B", "C", "Adaptive"]),
}


def one_run(d, window_s, eps=1.0):
    rows = []
    for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
        try:
            rows.append((datetime.datetime.strptime(r["timestamp"], TS), float(r["v_score"])))
        except Exception:
            continue
    if not rows:
        return None
    t0 = rows[0][0]
    win = [v for t, v in rows if (t - t0).total_seconds() < window_s]
    if not win:
        return None
    swaps = 0
    p = os.path.join(d, "executor.log")
    if os.path.isfile(p):
        swaps = sum(1 for ln in open(p, errors="replace") if "hot-swap complete" in ln)
    # recovered: some sample exceeds eps and the series ends at or below it and stays there
    v = [x[1] for x in rows]
    violated = any(x > eps for x in v)
    tail_ok = next((i for i in range(len(v)) if v[i] <= eps and all(y <= eps for y in v[i:])),
                   None)
    return {"run": os.path.basename(d), "n_samples": len(win),
            "lastV": round(win[-1], 4), "maxV": round(max(win), 4),
            "cumV": round(sum(win), 1), "hotswaps": swaps,
            "recovered": bool(violated and tail_ok is not None),
            "run_length_s": (rows[-1][0] - t0).total_seconds() + 1}


def cell(dirs, window_s):
    runs = [one_run(d, window_s) for d in dirs]
    runs = [r for r in runs if r]
    if not runs:
        return None

    def ms(f, nd=1):
        vals = [r[f] for r in runs]
        return round(st.mean(vals), nd), round(st.pstdev(vals), nd)
    return {"n": len(runs), "lastV": ms("lastV"), "maxV": ms("maxV"), "cumV": ms("cumV", 0),
            "hotswaps": ms("hotswaps", 1),
            "recoveries": sum(1 for r in runs if r["recovered"]),
            "run_length_s": ms("run_length_s", 0),
            "runs": [r["run"] for r in runs], "per_run": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=94.0)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    print(f"window {a.window:g} s\n")
    print(f"{'cell':26s} {'n':>2s} {'rec':>4s} {'lastV':>16s} {'maxV':>16s} "
          f"{'cumV':>14s} {'swaps':>6s} {'len':>5s}")
    print("-" * 100)
    for name, (tag, variants) in SCENARIOS.items():
        if a.scenario and a.scenario != name:
            continue
        for var in variants:
            dirs = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_{tag}_{var}_r*")))
            dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "metrics.csv"))]
            c = cell(dirs, a.window)
            if not c:
                continue
            out[f"{name}/{var}"] = c
            print(f"{name + '/' + var:26s} {c['n']:2d} {c['recoveries']:2d}/{c['n']:<1d} "
                  f"{c['lastV'][0]:9.1f}±{c['lastV'][1]:<5.1f} "
                  f"{c['maxV'][0]:9.1f}±{c['maxV'][1]:<5.1f} "
                  f"{c['cumV'][0]:8.0f}±{c['cumV'][1]:<4.0f} "
                  f"{c['hotswaps'][0]:6.1f} {c['run_length_s'][0]:5.0f}")
    if a.json:
        json.dump({"window_s": a.window, "cells": out}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
