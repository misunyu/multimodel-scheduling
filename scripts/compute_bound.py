#!/usr/bin/env python3
"""Instantiate the recovery envelope from a committed run and check it against the measurement.

The envelope is  T + k(T_v + delta)  (Q3/Q5 form; Q4 adds a trailing +delta for the
best-so-far return).  T and T_v are settings from the confirmed table; k and delta are NOT
settings -- they are properties of the run and must be read from the artifacts:

  k      = index of the RECOVERING candidate (cand_5 -> 5).  The envelope is over the
           candidates actually traversed, so a run that lands early has a smaller envelope.
  delta  = per-candidate overhead beyond the dwell: swap cost plus the sampling gap.
           Measured as max_i(interval_i - T_v) over the candidates that were REJECTED and
           advanced past, i.e. the worst one, because an envelope must cover the worst.
           Clamped at 0 -- a candidate that auto-advances faster than T_v (identical
           placement, no transition to validate) cannot make the envelope smaller.

           The recovering candidate is deliberately excluded: the run keeps dwelling on it
           until the schedule ends, so its "interval" is the remaining run length, not a
           search cost. Including it inflated delta to 40-100 s and made the envelope
           vacuous (bound 218 s vs search 27 s).

Reporting delta from a setting instead of from the run is what makes an envelope
unfalsifiable, so this script always prints the measured value next to the check.

usage: python3 scripts/compute_bound.py runs/<dir> [runs/<dir> ...] [--form q4]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_from_run import load_metrics, compute  # noqa: E402

T = 3.0
T_V = 3.0
EPS = 1.0
N_CAND = 5


def bound_for_run(run_dir, form="q35"):
    rows = load_metrics(run_dir)
    res = compute(rows, EPS)
    changes = res.get("combination_changes") or []
    # candidate applications, in order, with the timestamp each took effect
    cands = [(ts, c) for ts, c in changes if c.startswith("cand_")]
    if not cands:
        return {"run": os.path.basename(run_dir), "error": "no candidate applications"}
    import datetime
    TS = "%Y-%m-%d %H:%M:%S"
    times = [datetime.datetime.strptime(ts, TS) for ts, _ in cands]
    # only consecutive candidate-to-candidate gaps: each is one REJECTED candidate's dwell
    deltas = [max(0.0, (times[i + 1] - times[i]).total_seconds() - T_V)
              for i in range(len(times) - 1)]
    commit = res.get("commit_combo")
    k = int(commit.split("_")[1]) if commit and commit.startswith("cand_") else len(cands)
    rejected = deltas[:k - 1] if form != "q4" else deltas
    delta = max(rejected) if rejected else 0.0

    # lead = detection onset -> FIRST candidate applied. The published envelope substitutes
    # T for this, which is only right when the trigger is evaluated the instant the window
    # fills. In practice the executor evaluates at the dwell boundary of the failure phase,
    # so lead can exceed T -- measure it rather than assume it.
    #
    # lead decomposes into two parts, and the split matters for reading the result:
    #   pre-activation  = the controller is not yet running (the violation began in the
    #                     stable phase, before the failure phase was applied). Q5 only.
    #   post-activation = the failure phase's own dwell must elapse before the first
    #                     candidate can be applied -- 10 s here, not T.
    t0 = res.get("t0")
    burst = next((datetime.datetime.strptime(ts, TS) for ts, c in changes
                  if c == "combination_burst"), None)
    lead = pre = None
    if t0:
        t0d = datetime.datetime.strptime(t0, TS)
        lead = (times[0] - t0d).total_seconds()
        pre = max(0.0, (burst - t0d).total_seconds()) if burst else 0.0
    head = T if lead is None else max(lead, 0.0)
    if form == "q4":
        bound_T = T + N_CAND * (T_V + delta) + delta
        bound = head + N_CAND * (T_V + delta) + delta
    else:
        bound_T = T + k * (T_V + delta)
        bound = head + k * (T_V + delta)
    search = res.get("search_s")
    return {"run": os.path.basename(run_dir), "k": k, "delta_s": round(delta, 2),
            "lead_s": (None if lead is None else round(lead, 1)),
            "lead_pre_activation_s": (None if pre is None else round(pre, 1)),
            "commit": commit, "bound_s": round(bound, 1), "bound_T_s": round(bound_T, 1),
            "search_s": search, "persist_s": res.get("persist_s"),
            "holds": (None if search is None else bool(search <= bound)),
            "holds_published_form": (None if search is None else bool(search <= bound_T))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--form", choices=["q35", "q4"], default="q35")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = [bound_for_run(d.rstrip("/"), a.form) for d in a.run_dirs]
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        for r in out:
            if "error" in r:
                print(f"{r['run']:44s} {r['error']}")
                continue
            mark = {True: "OK", False: "VIOLATED", None: "-- (no recovery)"}[r["holds"]]
            print(f"{r['run']:44s} k={r['k']} delta={r['delta_s']:5.2f}s "
                  f"bound={r['bound_s']:6.1f}s search={r['search_s']} {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
