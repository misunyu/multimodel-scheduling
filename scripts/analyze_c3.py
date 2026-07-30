#!/usr/bin/env python3
"""C3 fluid-model check, recomputed from committed runs.

Conditions and pass/fail thresholds were fixed BEFORE the sweep ran and are recorded in
docs/c3_revalidation_conditions.md. This script only evaluates them.

What is measured, per run:

  accumulation slope   least-squares dQ/dt of the bottleneck view's queue over the all-CPU
                       phase.  Model: lambda - mu_cpu.
  drain slope          least-squares dQ/dt over the all-accelerator phase.
                       Model: -(mu* - lambda).
  mu_cpu, mu_drain     the SERVICE RATES those two phases actually ran at, counted as
                       completions per second from perframe.csv.

Counting completions from perframe.csv is not optional.  metrics.csv's `view*_fps` column is
1000/infer_ms -- a CAPACITY, not a throughput -- and the two differ by a factor of 1.8 here:
during the saturated mu* run view1's column reads 144 fps while perframe.csv shows 80
completions per second.  The same trap was already found in the Q1.1 overhead figure
(fig/FIGURE_SPECS.md, runtime_overhead 함정 2).  Deriving mu from the queue slope instead
would be circular -- the slope is the thing being tested.

Three candidate mu* values are carried through, because the old C3's central claim was that
mu* is the PIPELINE service rate and not the inference rate, and this run either reproduces
that or does not:

  pipeline_drain   121 (GPU) / 86 (NPU).  Measured on this system in runs/*_c3mudrain_* at
                   lambda=20/view, i.e. a near-idle drain.  THIS IS THE ANCHOR lambda WAS SET
                   FROM, so it cannot be revised after the fact.
  pipeline_sat      79 (GPU) / 65 (NPU).  Completions per second for view1 in
                   runs/*_c3mustar_* under 400 fps/view of arrivals.
  raw_infer        144 (GPU) / 231 (NPU).  1000/infer_ms for view1 at the end of that same
                   saturated phase, where the cumulative mean has converged.

Post-hoc, clearly separated from the pre-registered judgment, two further things are
reported: the drain regime split (while view2 still has backlog vs. after it empties) and
the lambda-sweep fit, which asks whether the model's FORM holds even where its constant does
not.

usage: python3 scripts/analyze_c3.py [--json out.json]
"""
import argparse, csv, datetime, glob, json, os, statistics as st, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOTTLENECK = "view1"          # yolo11s
MU_PIPE_DRAIN = {"gpu": 121.0, "npu": 86.0}
MU_PIPE_SAT = {"gpu": 79.0, "npu": 65.0}
MU_RAW_INFER = {"gpu": 144.0, "npu": 231.3}
LAMBDA = {"070": {"gpu": 85, "npu": 60}, "090": {"gpu": 109, "npu": 77},
          "095": {"gpu": 115, "npu": 82}}
FRACS = ["070", "090", "095"]

# Pre-registered tolerances (docs/c3_revalidation_conditions.md §2).
TOL_ACC_PCT = 20.0
TOL_DRAIN_PCT = 20.0
TOL_DRAIN_ABS = 3.0           # fps; satisfying EITHER counts as agreement
DIVERGENCE_BAND = (3.6, 8.4)  # 6x +-40%, predicted for 0.70 -> 0.95


def fit(xs, ys):
    """least-squares slope; None if fewer than 3 points"""
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def epoch(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def completions(perframe_rows, view, t_lo, t_hi):
    """observed service rate = completed frames per second in [t_lo, t_hi)"""
    if t_hi - t_lo < 2:
        return None
    n = sum(1 for r in perframe_rows if r["view"] == view and t_lo <= float(r["ts"]) < t_hi)
    return n / (t_hi - t_lo)


def one_run(d, plat, frac):
    rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    q = [int(r[f"{BOTTLENECK}_q"]) for r in rows]
    q2 = [int(r["view2_q"]) for r in rows]
    combo = [r["combination"] for r in rows]
    ts = [epoch(r["timestamp"]) for r in rows]
    drop = int(rows[-1]["drop_count"])
    pf = list(csv.DictReader(open(os.path.join(d, "perframe.csv"))))

    acc_i = [i for i, c in enumerate(combo) if c == "combination_burst"]
    dr_i = [i for i, c in enumerate(combo) if c.startswith("cand_")]
    # accumulation: skip the first 2 samples of the phase (the queue has to start building)
    acc_i = acc_i[2:]
    # drain: only while the queue is still non-empty, else the slope is floored at 0 and the
    # fit is dragged toward the model
    dr_use = []
    for i in dr_i:
        dr_use.append(i)
        if q[i] == 0:
            break
    acc_slope = fit(acc_i, [q[i] for i in acc_i])
    dr_slope = fit(dr_use, [q[i] for i in dr_use])

    # service rates over exactly those two windows, counted from perframe.csv
    mu_cpu = completions(pf, BOTTLENECK, ts[acc_i[0]], ts[acc_i[-1]] + 1) if acc_i else None
    mu_drain = completions(pf, BOTTLENECK, ts[dr_use[0]], ts[dr_use[-1]] + 1) if dr_use else None

    lam = LAMBDA[frac][plat]
    acc_pred = (lam - mu_cpu) if mu_cpu else None
    acc_err = ((acc_slope - acc_pred) / abs(acc_pred) * 100) if (acc_pred and acc_slope) else None

    q_at_switch = q[dr_i[0]] if dr_i else None
    preds = {}
    for name, mu in (("pipeline_drain", MU_PIPE_DRAIN[plat]),
                     ("pipeline_sat", MU_PIPE_SAT[plat]),
                     ("raw_infer", MU_RAW_INFER[plat])):
        pred = -(mu - lam)
        err_pp = ((dr_slope - pred) / abs(pred) * 100) if (dr_slope is not None and pred) else None
        err_abs = abs(dr_slope - pred) if dr_slope is not None else None
        preds[name] = {"mu": round(mu, 1), "pred_slope": round(pred, 2),
                       "err_pct": (round(err_pp, 1) if err_pp is not None else None),
                       "err_abs": (round(err_abs, 2) if err_abs is not None else None),
                       "agrees": bool(err_abs is not None and
                                      (abs(err_pp) <= TOL_DRAIN_PCT or err_abs <= TOL_DRAIN_ABS))}

    # POST-HOC: the drain has two regimes. While view2 still holds backlog the accelerator is
    # shared with it; once view2 empties, view1 has the pipeline to itself. A single-mu* fluid
    # model cannot express that, so both segments are reported separately -- never in place of
    # the pre-registered whole-phase fit.
    v2_empty = next((i for i in dr_i if q2[i] == 0), None)
    early = [i for i in dr_i if v2_empty is not None and i < v2_empty]
    late = [i for i in dr_i if v2_empty is not None and i >= v2_empty and q[i] > 0]
    dr_early, dr_late = fit(early, [q[i] for i in early]), fit(late, [q[i] for i in late])

    man = json.load(open(os.path.join(d, "run_manifest.json")))
    guard = sum(1 for ln in open(os.path.join(d, "executor.log"), errors="replace")
                if "QoS-triggered advance" in ln)
    t_stable = (q_at_switch / abs(dr_slope)) if (dr_slope and dr_slope < 0 and q_at_switch) else None
    r = lambda v, n=2: (round(v, n) if v is not None else None)
    return {"run": os.path.basename(d), "lambda": lam,
            "acc_slope": r(acc_slope), "mu_cpu_obs": r(mu_cpu),
            "acc_pred": r(acc_pred), "acc_err_pct": r(acc_err, 1),
            "acc_agrees": bool(acc_err is not None and abs(acc_err) <= TOL_ACC_PCT),
            "drain_slope": r(dr_slope), "mu_drain_obs": r(mu_drain),
            "mu_star_implied": r(lam - dr_slope) if dr_slope is not None else None,
            "drain_slope_early": r(dr_early), "drain_slope_late": r(dr_late),
            "q_at_switch": q_at_switch, "T_stable_s": r(t_stable, 1),
            "drained_to_zero": bool(dr_i and q[dr_use[-1]] == 0),
            "drop": drop, "guard_advances": guard, "background_lines": 0,
            "executor_rc": man.get("executor_rc"), "preds": preds}


def anchor_from_mudrain(plat):
    """Recompute the mu* anchor from the pilot it came from, so the number is traceable.
    Rule: least squares over the drain samples up to and including the first zero, dropping
    the first sample of the phase (it straddles the switch and can read higher than the
    burst's last sample). mu* = lambda + |slope|, lambda = 20/view."""
    ds = glob.glob(os.path.join(PROJECT, "runs", f"*_c3mudrain_{plat}_r*"))
    ds = [d for d in ds if os.path.isfile(os.path.join(d, "metrics.csv"))]
    out = []
    for d in sorted(ds):
        rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
        q = [int(r[f"{BOTTLENECK}_q"]) for r in rows]
        dr = [i for i, r in enumerate(rows) if r["combination"].startswith("cand_")]
        use = []
        for i in dr[1:]:
            use.append(i)
            if q[i] == 0:
                break
        s = fit(use, [q[i] for i in use])
        if s:
            out.append({"run": os.path.basename(d), "slope": round(s, 2),
                        "mu_star": round(20 - s, 1), "n_samples": len(use)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out, per_run = {}, []
    for plat in ("gpu", "npu"):
        for frac in FRACS:
            ds = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_c3{plat}_{frac}_r*")))
            ds = [d for d in ds if os.path.isfile(os.path.join(d, "run_manifest.json"))]
            if not ds:
                continue
            runs = [one_run(d, plat, frac) for d in ds]
            per_run += [dict(cell=f"{plat}/{frac}", **r) for r in runs]

            def ag(f, n=2):
                v = [r[f] for r in runs if r[f] is not None]
                return (round(st.mean(v), n), round(st.pstdev(v), n)) if v else None
            cell = {"n": len(runs), "lambda": runs[0]["lambda"],
                    "acc_slope": ag("acc_slope"), "mu_cpu_obs": ag("mu_cpu_obs"),
                    "acc_pred": ag("acc_pred"), "acc_err_pct": ag("acc_err_pct", 1),
                    "acc_agrees": sum(1 for r in runs if r["acc_agrees"]),
                    "drain_slope": ag("drain_slope"), "mu_drain_obs": ag("mu_drain_obs"),
                    "mu_star_implied": ag("mu_star_implied"),
                    "drain_slope_early": ag("drain_slope_early"),
                    "drain_slope_late": ag("drain_slope_late"),
                    "q_at_switch": ag("q_at_switch"), "T_stable_s": ag("T_stable_s", 1),
                    "drained_to_zero": sum(1 for r in runs if r["drained_to_zero"]),
                    "drop": sum(r["drop"] for r in runs),
                    "guard_advances": sum(r["guard_advances"] for r in runs),
                    "teardown_aborts": sum(1 for r in runs if r["executor_rc"] != 0),
                    "runs": [r["run"] for r in runs], "source": runs[0]["run"]}
            # T_stable from the cell's mean slope, not the mean of per-run T_stable: at
            # lambda -> mu* individual denominators approach zero and one run with a slope of
            # -0.54 drags a mean of ratios to 3700 s. Both are reported; the mean-slope form
            # is the one quoted.
            ms, mq = cell["drain_slope"][0], cell["q_at_switch"][0]
            cell["T_stable_from_mean_slope"] = round(mq / abs(ms), 1) if ms < 0 else None
            for name in ("pipeline_drain", "pipeline_sat", "raw_infer"):
                ps = [r["preds"][name] for r in runs]
                cell[name] = {
                    "mu": ps[0]["mu"], "pred_slope": ps[0]["pred_slope"],
                    "err_pct": round(st.mean(p["err_pct"] for p in ps), 1),
                    "err_abs": round(st.mean(p["err_abs"] for p in ps), 2),
                    "agrees_n": sum(1 for p in ps if p["agrees"])}
            out[f"{plat}/{frac}"] = cell

    # POST-HOC lambda-sweep fit. The model says drain = lambda - mu*, i.e. a line in lambda of
    # slope exactly +1 with intercept -mu*. Fitting it separates two questions the per-cell
    # error percentages conflate: is the FUNCTIONAL FORM right, and is the mu* we anchored on
    # right. A slope near 1 with a tight per-cell mu* says the form holds and only the
    # constant was off -- a different conclusion from "the model does not describe the drain".
    for plat in ("gpu", "npu"):
        keys = [k for k in out if k.startswith(plat)]
        if len(keys) < 2:
            continue
        for tag, field in (("", "drain_slope"), ("_late", "drain_slope_late")):
            pts = [(out[k]["lambda"], out[k][field][0]) for k in keys]
            mus = [l - d for l, d in pts]
            out[f"_fit_{plat}{tag}"] = {
                "dslope_dlambda": round(fit([p[0] for p in pts], [p[1] for p in pts]), 3),
                "dslope_dlambda_predicted": 1.0,
                "mu_star_eff_per_cell": [round(m, 1) for m in mus],
                "mu_star_eff_mean": round(st.mean(mus), 1),
                "mu_star_eff_sd": round(st.pstdev(mus), 2),
                "mu_star_anchor": MU_PIPE_DRAIN[plat],
                "anchor_error_pct": round((st.mean(mus) / MU_PIPE_DRAIN[plat] - 1) * 100, 1),
                "lambda_over_mu_eff": [round(p[0] / st.mean(mus), 3) for p in pts]}
        out[f"_anchor_{plat}"] = anchor_from_mudrain(plat)

    if a.json:
        json.dump({"cells": out, "runs": per_run}, open(a.json, "w"), indent=2)

    cells = {k: v for k, v in out.items() if not k.startswith(("_fit_", "_anchor_"))}
    print("ACCUMULATION  (model: dQ/dt = lambda - mu_cpu; mu_cpu counted from perframe.csv)")
    print(f"{'cell':10s} {'lam':>4s} {'mu_cpu':>7s} {'pred':>8s} {'meas':>8s} {'err':>7s} {'ok':>5s}")
    for k, v in cells.items():
        print(f"{k:10s} {v['lambda']:4d} {v['mu_cpu_obs'][0]:7.2f} {v['acc_pred'][0]:8.2f} "
              f"{v['acc_slope'][0]:8.2f} {v['acc_err_pct'][0]:+6.1f}% {v['acc_agrees']:d}/{v['n']:d}")

    print("\nDRAIN  (model: dQ/dt = -(mu* - lambda))")
    print(f"{'cell':10s} {'lam':>4s} {'meas':>8s} {'mu_obs':>7s} {'mu_impl':>8s} {'Q':>6s} "
          f"{'Tstab':>8s} {'->0':>4s} {'drop':>5s} {'guard':>5s}")
    for k, v in cells.items():
        t = v["T_stable_from_mean_slope"]
        print(f"{k:10s} {v['lambda']:4d} {v['drain_slope'][0]:8.2f} {v['mu_drain_obs'][0]:7.1f} "
              f"{v['mu_star_implied'][0]:8.1f} {v['q_at_switch'][0]:6.0f} "
              f"{(f'{t:8.1f}' if t else '   diverg')} {v['drained_to_zero']:4d} "
              f"{v['drop']:5d} {v['guard_advances']:5d}")

    print(f"\n{'cell':10s} " + " ".join(f"{n:>30s}" for n in
          ("pipeline_drain", "pipeline_sat", "raw_infer")))
    for k, v in cells.items():
        line = f"{k:10s} "
        for n in ("pipeline_drain", "pipeline_sat", "raw_infer"):
            p = v[n]
            line += (f"  mu={p['mu']:5.1f} pred={p['pred_slope']:7.2f} "
                     f"err={p['err_pct']:+7.1f}% {p['agrees_n']}/{v['n']}")
        print(line)

    print("\nPOST-HOC  regime split and lambda-sweep fit")
    for k, v in cells.items():
        print(f"{k:10s} drain full {v['drain_slope'][0]:7.2f} | "
              f"early (view2 backlogged) {v['drain_slope_early'][0]:7.2f} | "
              f"late (view2 empty) {v['drain_slope_late'][0]:7.2f}")
    for plat in ("gpu", "npu"):
        for tag, lab in (("", "full "), ("_late", "late ")):
            f = out.get(f"_fit_{plat}{tag}")
            if f:
                print(f"[{plat} {lab}] d(drain)/d(lambda)={f['dslope_dlambda']} (model 1.0)  "
                      f"mu*_eff {f['mu_star_eff_per_cell']} mean {f['mu_star_eff_mean']}"
                      f"+-{f['mu_star_eff_sd']} vs anchor {f['mu_star_anchor']} "
                      f"({f['anchor_error_pct']:+}%)  lambda/mu*_eff {f['lambda_over_mu_eff']}")
        print(f"[{plat}] anchor recomputed from mudrain pilot: {out[f'_anchor_{plat}']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
