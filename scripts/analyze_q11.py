#!/usr/bin/env python3
"""Q1.1 steady-state overhead, recomputed from committed runs.

Four metrics, following scripts/runtime_overhead_analysis.py -- with metric 1 corrected.

  1. throughput   frames/s actually completed, counted from perframe.csv.
                  NOT metrics.csv's total_fps: that column is the sum of 1000/infer_ms,
                  i.e. CAPACITY. Verified on a q15 run -- view1_fps 101.82 vs
                  1000/infer 101.83, while the real rate was 40.7 fps against a scheduled
                  41. Summing it made the published figure's "throughput" and its
                  "per-view latency" the same quantity twice.
  2. infer_ms     mean per-view inference time (metrics.csv, valid as published)
  3. drop_rate    frames/s dropped at the queue (metrics.csv)
  4. cpu          executor process-TREE CPU percent (proc_cpu.csv). The closest direct
                  proxy for monitoring cost. Per-process, so concurrent analysis on the
                  same box cannot leak into it.

The first 3 CPU samples of every run are discarded: model load and warm-up dominate them
and have nothing to do with steady-state monitoring.

usage: python3 scripts/analyze_q11.py [--json out.json]
"""
import argparse, csv, glob, json, os, statistics as st, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["Static", "Adaptive", "BoundGuard"]
WARMUP_SAMPLES = 3


def one_run(d):
    rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    infer = []
    for r in rows:
        vals = [float(r[k]) for k in r if k.endswith("_infer_ms") and float(r[k]) > 0]
        if vals:
            infer.append(st.mean(vals))
    pf = list(csv.DictReader(open(os.path.join(d, "perframe.csv"))))
    span = (float(pf[-1]["ts"]) - float(pf[0]["ts"])) if len(pf) > 1 else 0.0
    cpu, rss = [], []
    p = os.path.join(d, "proc_cpu.csv")
    if os.path.isfile(p):
        cr = list(csv.DictReader(open(p)))[WARMUP_SAMPLES:]
        cpu = [float(x["cpu_percent_tree"]) for x in cr if float(x["cpu_percent_tree"]) > 0]
        rss = [float(x["rss_mb_tree"]) for x in cr if float(x["rss_mb_tree"]) > 0]
    man = json.load(open(os.path.join(d, "run_manifest.json")))
    return {
        "run": os.path.basename(d),
        "throughput": (len(pf) / span) if span else float("nan"),
        "infer_ms": st.mean(infer) if infer else float("nan"),
        "drop_rate": (int(rows[-1]["drop_count"]) / span) if span else float("nan"),
        "cpu": st.mean(cpu) if cpu else float("nan"),
        "rss_mb": st.mean(rss) if rss else float("nan"),
        "maxV": max(float(r["v_score"]) for r in rows),
        "executor_rc": man.get("executor_rc"),
        "transitions": len(man["transitions"].get("applied_sequence") or []),
    }


def agg(runs, f):
    v = [r[f] for r in runs if r[f] == r[f]]
    return (round(st.mean(v), 3), round(st.pstdev(v), 3)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for plat in ("gpu", "npu"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_q11{plat}_{arm}_r*")))
            # run_manifest.json is written last, so its absence means the run is still in
            # flight. Skip it loudly rather than crashing -- analysis has to be safe to run
            # while a campaign is going.
            complete = [d for d in ds if os.path.isfile(os.path.join(d, "run_manifest.json"))]
            for d in ds:
                if d not in complete:
                    print(f"[skip] in flight: {os.path.basename(d)}", file=sys.stderr)
            if not complete:
                continue
            ds = complete
            runs = [one_run(d) for d in ds]
            out[f"{plat}/{arm}"] = {
                "n": len(runs),
                **{f: agg(runs, f) for f in ("throughput", "infer_ms", "drop_rate", "cpu", "rss_mb")},
                "maxV": round(max(r["maxV"] for r in runs), 4),
                "teardown_aborts": sum(1 for r in runs if r["executor_rc"] != 0),
                "transitions": max(r["transitions"] for r in runs),
                "source": os.path.basename(ds[0]),
            }
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    hdr = (f"{'cell':22s} {'n':>2s} {'thru(fps)':>10s} {'infer(ms)':>10s} "
           f"{'drop(fps)':>10s} {'cpu(%)':>10s} {'rss(MB)':>9s} {'maxV':>7s} {'trans':>6s} {'ab':>3s}")
    print(hdr); print("-" * len(hdr))
    for k, v in out.items():
        print(f"{k:22s} {v['n']:2d} {v['throughput'][0]:10.2f} {v['infer_ms'][0]:10.3f} "
              f"{v['drop_rate'][0]:10.3f} {v['cpu'][0]:10.2f} {v['rss_mb'][0]:9.0f} "
              f"{v['maxV']:7.4f} {v['transitions']:6d} {v['teardown_aborts']:3d}")
    # the comparison that carries information: monitored vs unmonitored
    for plat in ("gpu", "npu"):
        s, b = out.get(f"{plat}/Static"), out.get(f"{plat}/BoundGuard")
        if s and b:
            print(f"\n{plat}: monitoring cost = CPU {b['cpu'][0]-s['cpu'][0]:+.2f} pp "
                  f"({100*(b['cpu'][0]/s['cpu'][0]-1):+.1f}%), "
                  f"infer {b['infer_ms'][0]-s['infer_ms'][0]:+.3f} ms, "
                  f"throughput {b['throughput'][0]-s['throughput'][0]:+.2f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
