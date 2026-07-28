#!/usr/bin/env python3
"""Recompute a run's headline metrics from its COMMITTED artefacts alone.

This is the acceptance test for run persistence (v24 task 101 / v30 task 137): having
files is not the same as being able to trace a value back. If a metric cannot be
recomputed here, the run directory is missing something and the audit chain is still
broken -- three sets of artefacts have already been lost this way.

Definitions are the canonical ones (docs/metric_definitions.md), not new ones:
  t0      first sample with v_score > eps
  t_r     first sample with v_score <= eps that STAYS <= eps until the end (strict);
          absent => censored
  persist t_r - t0                  (censored if t_r absent)
  drain   t_r - t_commit            (commit = when the recovering candidate was applied)
  search  t_commit - t0             (detection onset -> candidate applied)
  hotswaps number of per-VIEW hot-swap events (adaptive_deploy logs one per view)

usage: recompute_from_run.py runs/<dir> [--eps 1.0]
"""
import argparse, csv, datetime, json, os, re, sys

TS = "%Y-%m-%d %H:%M:%S"


def load_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.csv")
    rows = list(csv.DictReader(open(p)))
    out = []
    for r in rows:
        try:
            t = datetime.datetime.strptime(r["timestamp"], TS)
        except Exception:
            continue
        out.append((t, float(r["v_score"]), r.get("combination", "")))
    return out


def parse_transitions(run_dir):
    """Candidate-application and hot-swap events from the executor log.

    The executor prints a line per applied combination and adaptive_deploy prints one
    'hot-swap complete' per VIEW, so hot-swaps are per-view by construction.
    """
    p = os.path.join(run_dir, "executor.log")
    if not os.path.isfile(p):
        return None
    applied, swaps = [], 0
    for line in open(p, errors="replace"):
        m = re.search(r"\[Executor\] Starting schedule: (\S+)", line)
        if m:
            applied.append(m.group(1))
        if "hot-swap complete" in line:
            swaps += 1
        m2 = re.search(r"QoS-triggered advance|exhausted -> revert to best", line)
        if m2:
            applied.append(f"<{m2.group(0)}>")
    return {"applied_sequence": applied, "hotswaps": swaps}


def compute(rows, eps):
    if not rows:
        return {}
    v = [x[1] for x in rows]
    t = [x[0] for x in rows]
    t0 = next((t[i] for i in range(len(v)) if v[i] > eps), None)
    tr = None
    for i in range(len(v)):
        if v[i] <= eps and all(x <= eps for x in v[i:]):
            tr = t[i]
            break
    res = {
        "n_samples": len(v), "maxV": round(max(v), 4), "lastV": round(v[-1], 4),
        "violated": t0 is not None,
        "t0": t0.strftime(TS) if t0 else None,
        "t_r": tr.strftime(TS) if tr else None,
        "recovered": bool(t0 and tr and tr >= t0),
    }
    if t0 and tr and tr >= t0:
        res["persist_s"] = (tr - t0).total_seconds()
    elif t0:
        res["persist_s"] = f"censored(> {(t[-1]-t0).total_seconds():.0f}s)"
    # combination changes give the applied-candidate timeline when present
    changes = []
    for i in range(1, len(rows)):
        if rows[i][2] != rows[i-1][2]:
            changes.append((rows[i][0], rows[i][2]))
    res["combination_changes"] = [(c[0].strftime(TS), c[1]) for c in changes]
    if t0 and changes:
        # Canonical definitions: search = detection onset -> the RECOVERING candidate is
        # applied; drain = that commit -> V<=eps. So the commit is the last placement
        # change at or before t_r (the one in effect when recovery happened), NOT the
        # first candidate tried. Using the first candidate understates search and
        # overstates drain.
        if tr:
            prior = [c for c in changes if c[0] <= tr]
            commit = prior[-1][0] if prior else None
            commit_combo = prior[-1][1] if prior else None
        else:
            commit = commit_combo = None
        if commit and commit >= t0:
            res["commit_combo"] = commit_combo
            res["search_s"] = (commit - t0).total_seconds()
            res["drain_s"] = (tr - commit).total_seconds()
            res["persist_check_search_plus_drain"] = res["search_s"] + res["drain_s"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--eps", type=float, default=None)
    a = ap.parse_args()
    man_p = os.path.join(a.run_dir, "run_manifest.json")
    man = json.load(open(man_p)) if os.path.isfile(man_p) else {}
    eps = a.eps if a.eps is not None else man.get("eps", 1.0)

    rows = load_metrics(a.run_dir)
    got = compute(rows, eps)
    tr = parse_transitions(a.run_dir)
    if tr is not None:
        got["hotswaps"] = tr["hotswaps"]
        got["applied_sequence"] = tr["applied_sequence"]

    print(f"run      : {a.run_dir}")
    print(f"eps      : {eps}   (from run_manifest.json)" if a.eps is None else f"eps      : {eps}")
    print("--- recomputed from committed artefacts ---")
    for k, v in got.items():
        print(f"  {k:20s}: {v}")
    rep = man.get("classification")
    if rep:
        print("--- reported by the harness (run_manifest.json) ---")
        for k, v in rep.items():
            print(f"  {k:20s}: {v}")
        print("--- agreement ---")
        for k in ("violated", "recovered", "maxV", "lastV"):
            if k in rep and k in got:
                ok = (rep[k] == got[k])
                print(f"  {k:20s}: {'MATCH' if ok else 'MISMATCH'}  ({rep[k]} vs {got[k]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
