#!/usr/bin/env python3
"""Request-level metrics for Q1.5 / tab:b2, recomputed from committed runs.

Everything here comes from <run>/perframe.csv (one row per completed frame, with the
deadline that was in force) and <run>/metrics.csv (per-second drops and backlog). Nothing
is read from a report.

Two deadlines are reported because the published 8.13 ms was NOT a chosen SLO -- it is the
`1000/infps` fallback in view_handlers._deadline_ms(), so it moved with lambda. v34 pins
slo_ms=15 in the schedule and recomputes miss at 8.13 ms offline for comparison. p99/p999/
mean are deadline-independent, so the published-vs-rerun comparison on the tail is exact.

miss = (late + drop) / arrived, arrived = served + drop. The denominator differs per
technique -- Stop-restart accepts no frames while restarting -- so served/late/drop are
always printed alongside (docs/backlog_preserve_report.md 4a-4).

usage: python3 scripts/analyze_q15.py [--platform gpu npu] [--json out.json]
"""
import argparse, csv, glob, json, os, statistics as st, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TECHS = ["Static", "Stop-restart", "Adaptive", "BoundGuard"]
BETAS = ["0.5", "1.0", "1.5"]
PUBLISHED_DEADLINE_MS = 8.13


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def one_run(d):
    e2e, late15 = [], 0
    with open(os.path.join(d, "perframe.csv")) as f:
        for r in csv.DictReader(f):
            v = float(r["e2e_ms"])
            e2e.append(v)
            late15 += int(r["late"])
    e2e.sort()
    rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    drop = int(rows[-1]["drop_count"])
    blmax = max(int(r["backlog_total"]) for r in rows)
    served = len(e2e)
    arrived = served + drop
    late813 = sum(1 for v in e2e if v > PUBLISHED_DEADLINE_MS)
    man = json.load(open(os.path.join(d, "run_manifest.json")))
    return {
        "run": os.path.basename(d), "served": served, "drop": drop, "arrived": arrived,
        "late_15ms": late15, "late_8.13ms": late813,
        "miss_15ms": (late15 + drop) / arrived if arrived else float("nan"),
        "miss_8.13ms": (late813 + drop) / arrived if arrived else float("nan"),
        "p99": pct(e2e, 0.99), "p999": pct(e2e, 0.999), "mean": st.mean(e2e) if e2e else float("nan"),
        "blmax": blmax, "buffer": man["env_knobs"].get("FSRR_FRAME_BUFFER"),
        "executor_rc": man.get("executor_rc"), "wallclock_s": man.get("wallclock_s"),
        # recorded, not judged (condition 177-4): recovery is not a gate for this experiment
        "recovered": max(float(r["v_score"]) for r in rows[-3:]) <= 1.0,
        "saturated_advances": sum(1 for ln in open(os.path.join(d, "executor.log"), errors="replace")
                                  if "QoS-triggered advance: saturated" in ln),
    }


def agg(runs, field):
    v = [r[field] for r in runs if r[field] == r[field]]
    if not v:
        return None
    return (round(st.mean(v), 4 if abs(st.mean(v)) < 10 else 0),
            round(st.pstdev(v), 4 if abs(st.mean(v)) < 10 else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", nargs="+", default=["gpu", "npu"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for plat in a.platform:
        for beta in BETAS:
            for tech in TECHS:
                ds = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_q15{plat}_{tech}_b{beta}_r*")))
                if not ds:
                    continue
                runs = [one_run(d) for d in ds]
                key = f"{plat}/{tech}/b{beta}"
                out[key] = {
                    "n": len(runs), "buffer": runs[0]["buffer"],
                    "served": agg(runs, "served"), "drop": agg(runs, "drop"),
                    "late_15ms": agg(runs, "late_15ms"), "late_8.13ms": agg(runs, "late_8.13ms"),
                    "miss_15ms": agg(runs, "miss_15ms"), "miss_8.13ms": agg(runs, "miss_8.13ms"),
                    "p99": agg(runs, "p99"), "p999": agg(runs, "p999"), "mean": agg(runs, "mean"),
                    "blmax": agg(runs, "blmax"),
                    "recovered": sum(1 for r in runs if r["recovered"]),
                    "saturated_advances": sum(r["saturated_advances"] for r in runs),
                    "teardown_aborts": sum(1 for r in runs if r["executor_rc"] != 0),
                    "source": os.path.basename(ds[0]),
                }
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    hdr = (f"{'cell':26s} {'n':>2s} {'buf':>5s} {'served':>7s} {'drop':>6s} "
           f"{'miss15':>7s} {'miss8':>7s} {'p99(ms)':>8s} {'p999':>8s} {'mean':>7s} "
           f"{'blmax':>6s} {'rec':>4s} {'sat':>4s} {'ab':>3s}")
    print(hdr); print("-" * len(hdr))
    for k, v in out.items():
        print(f"{k:26s} {v['n']:2d} {str(v['buffer']):>5s} {v['served'][0]:7.0f} {v['drop'][0]:6.0f} "
              f"{v['miss_15ms'][0]:7.3f} {v['miss_8.13ms'][0]:7.3f} {v['p99'][0]:8.0f} "
              f"{v['p999'][0]:8.0f} {v['mean'][0]:7.0f} {v['blmax'][0]:6.0f} "
              f"{v['recovered']:4d} {v['saturated_advances']:4d} {v['teardown_aborts']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
