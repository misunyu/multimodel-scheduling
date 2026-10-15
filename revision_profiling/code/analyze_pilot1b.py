"""Phase 1b pilot analysis (SPEC_PHASE1B_v1.1.md sections 3-5). Criteria come from the spec; nothing is fitted here.
  python3 -B code/analyze_pilot1b.py conservation | arrival | gpuload | select | repeat | dropmetric
"""
import sys

sys.dont_write_bytecode = True

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import analyze_pilot as ap
from analyze_pilot import PILOT, load_runs, offline, wkey, write_csv

BETAS = (0.05, 0.1, 0.25)
ALPHA = 0.3
F4 = "gpt2+mnasnet+resnet50+resnext50"


def all_formal_runs():
    out = []
    for d in ["smoke/runs", "arrival/runs", "repeat/runs", "rate_range/runs", "nice/nice_0", "nice/nice_10"] + [f"beta_{b}/single_gpu/runs" for b in BETAS]:
        v, inv = load_runs(PILOT / d)
        out += [(d, r) for r in v + inv]
    v, inv = load_runs(PILOT.parent / "offline_profile" / "runs") if (PILOT.parent / "offline_profile" / "runs").exists() else ([], [])
    for p in sorted((PILOT.parent / "offline_profile" / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        out.append(("offline_profile/runs", r))
    return out


def conservation():
    rows, cons, trace, diff, flags, lag = [], Counter(), Counter(), Counter(), [], []
    runs = all_formal_runs()
    for d, r in runs:
        for i in r["instances"]:
            w = i["window"]
            cons[w["conservation_flow"]] += 1
            trace[w["trace_check"]] += 1
            for side in ("start", "end"):
                diff[w[f"snapshot_diff_{side}"]] += 1
                lag.append(w[f"snapshot_read_lag_ms_{side}"])
                if abs(w[f"snapshot_diff_{side}"]) > 1:
                    flags.append({"dir": d, "run_key": r["run_key"], "instance": i["name"], "side": side, "diff": w[f"snapshot_diff_{side}"],
                                  "read_lag_ms": round(w[f"snapshot_read_lag_ms_{side}"], 3), "rate": i["rate"]})
        if not r["valid"]:
            rows.append({"dir": d, "run_key": r["run_key"], "reasons": ";".join(r["invalid_reasons"])})
    write_csv(PILOT / "invalid_runs.csv", rows)
    write_csv(PILOT / "snapshot_flags.csv", flags)
    summ = {"runs": len(runs), "instances": sum(cons.values()), "conservation_flow": dict(cons), "trace_check": dict(trace),
            "snapshot_diff(start+end)": dict(sorted(diff.items())), "snapshot_flags_abs_gt_1": len(flags),
            "read_lag_ms_median": round(float(np.median(lag)), 3), "read_lag_ms_max": round(max(lag), 3), "invalid_runs": len(rows)}
    (PILOT / "conservation_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    for x in rows:
        print("INVALID", x)


def _spread(vals, rel):
    m = float(np.mean(vals))
    r = max(vals) - min(vals)
    return (r / m if m > 0 else None) if rel else r


def variation_tables(by_wl, tag, out_dir):
    """by_wl: {workload key: {placement: [runs]}}. F, D_macro, S spread over repeats; SE of the mean; rank-uncertain pairs."""
    cand, pairs = [], []
    for key, pls in sorted(by_wl.items()):
        fmax = max(float(np.mean([r["system"]["throughput_fps"] for r in rs])) for rs in pls.values())
        stat = {}
        for pl, rs in sorted(pls.items()):
            f = [r["system"]["throughput_fps"] for r in rs]
            d = [r["system"]["drop_macro"] for r in rs]
            s = [fi / fmax - ALPHA * di for fi, di in zip(f, d)]
            n = len(rs)
            se = lambda v: float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else None  # noqa: E731
            stat[pl] = {"S_mean": float(np.mean(s)), "S_se": se(s)}
            cand.append({"family": key[0], "beta": key[1], "mult": key[2], "placement": pl, "repeats": n,
                         "F_mean": round(float(np.mean(f)), 2), "F_rel_range": round(_spread(f, True), 4), "F_se": round(se(f), 3),
                         "D_macro_mean": round(float(np.mean(d)), 4), "D_macro_abs_range": round(_spread(d, False), 4), "D_macro_se": round(se(d), 5),
                         "S_mean": round(float(np.mean(s)), 4), "S_abs_range": round(_spread(s, False), 4), "S_se": round(se(s), 5),
                         "F_within_5pct": _spread(f, True) <= 0.05})
        for a, b in itertools.combinations(sorted(stat), 2):
            diff = abs(stat[a]["S_mean"] - stat[b]["S_mean"])
            se_sum = stat[a]["S_se"] + stat[b]["S_se"]
            pairs.append({"family": key[0], "beta": key[1], "mult": key[2], "placement_a": a, "placement_b": b,
                          "S_mean_a": round(stat[a]["S_mean"], 4), "S_mean_b": round(stat[b]["S_mean"], 4), "abs_diff": round(diff, 4),
                          "se_sum": round(se_sum, 4), "rank_uncertain": diff <= se_sum})
    write_csv(out_dir / f"variation_{tag}_candidates.csv", cand)
    write_csv(out_dir / f"variation_{tag}_pairs.csv", pairs)
    q = lambda k: [x[k] for x in cand]  # noqa: E731
    summ = {"groups": len(cand)}
    for k in ("F_rel_range", "D_macro_abs_range", "S_abs_range"):
        v = np.array(q(k))
        summ[k] = {"median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4), "max": round(float(v.max()), 4)}
    summ["F_within_5pct"] = f"{sum(q('F_within_5pct'))}/{len(cand)}"
    d = np.array([p["abs_diff"] for p in pairs])
    summ["pairs"] = len(pairs)
    summ["rank_uncertain_pairs"] = int(sum(p["rank_uncertain"] for p in pairs))
    summ["S_diff_between_placements"] = {"median": round(float(np.median(d)), 4), "min": round(float(d.min()), 4), "max": round(float(d.max()), 4)}
    summ["S_se_median"] = round(float(np.median(q("S_se"))), 5)
    (out_dir / f"variation_{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    print(tag, json.dumps(summ))
    return summ


def arrival():
    runs, invalid = load_runs(PILOT / "arrival" / "runs")
    by = defaultdict(lambda: defaultdict(list))
    inst_rows = []
    for r in runs:
        by[wkey(r)][r["placement"]].append(r)
        for i in r["instances"]:
            w = i["window"]
            inst_rows.append({"family": r["meta"]["family"], "beta": r["meta"]["beta"], "mult": r["meta"]["multiplier"], "placement": r["placement"],
                              "rep": r["repeat"], "instance": i["name"], "device": i["execution"], "rate": round(i["rate"], 3),
                              "scheduled_ticks": w["scheduled_ticks"], "generated": w["generated"], "missed_ticks": w["missed_ticks"],
                              "realization": round(w["realization"], 4), "dropped_full": w["dropped_full"], "completed": w["completed"],
                              "q_i": round(w["q_i"], 4), "conservation_flow": w["conservation_flow"], "trace_check": w["trace_check"]})
    wl = []
    for key, pls in sorted(by.items()):
        rs_all = [r for rs in pls.values() for r in rs]
        reals = [i["window"]["realization"] for r in rs_all for i in r["instances"]]
        miss = [i["window"]["missed_ticks"] / i["window"]["scheduled_ticks"] for r in rs_all for i in r["instances"]]
        bad, checks, rules = [], 0, Counter()
        for rep in sorted({r["repeat"] for r in rs_all}):
            rr = [r for r in rs_all if r["repeat"] == rep]
            for name in [i["name"] for i in rr[0]["instances"]]:
                g = [next(i for i in r["instances"] if i["name"] == name)["window"] for r in rr]
                gen = [x["generated"] for x in g]
                checks += 1
                if min(x["scheduled_ticks"] for x in g) >= 200:
                    ok, rule = (max(gen) - min(gen)) / (sum(gen) / len(gen)) <= 0.02, "relative<=2%"
                else:
                    ok, rule = max(gen) - min(gen) <= 2, "absolute<=2"
                rules[rule] += 1
                if not ok:
                    bad.append(f"{name}(rep{rep},{rule},gen={gen})")
        complete = len(pls) == 3 and all(len(v) == 3 for v in pls.values())
        frange = {pl: _spread([r["system"]["throughput_fps"] for r in rs], True) for pl, rs in pls.items()}
        ok_real = min(reals) >= 0.95 and max(miss) <= 0.05
        wl.append({"family": key[0], "n": rs_all[0]["meta"]["n_instances"], "beta": key[1], "mult": key[2], "valid_runs": len(rs_all), "complete": complete,
                   "min_realization": round(min(reals), 4), "max_missed_ratio": round(max(miss), 4), "ok_realization_missed": ok_real,
                   "generated_checks": checks, "generated_rules": json.dumps(dict(rules)), "generated_violations": len(bad),
                   "generated_violation_list": "; ".join(bad)[:500], "ok_generated": not bad,
                   **{f"F_rel_range_{pl}": round(v, 4) for pl, v in sorted(frange.items())},
                   "F_rel_range_max": round(max(frange.values()), 4), "F_all_within_5pct(report only)": max(frange.values()) <= 0.05,
                   "pass_5_2": bool(complete and ok_real and not bad)})
    write_csv(PILOT / "arrival" / "arrival_5_2_instances.csv", inst_rows)
    write_csv(PILOT / "arrival" / "arrival_5_2_workloads.csv", wl)
    for w in wl:
        print(f"5.2 N={w['n']} b={w['beta']} x{w['mult']} runs={w['valid_runs']} minreal={w['min_realization']} maxmiss={w['max_missed_ratio']} "
              f"genviol={w['generated_violations']} Frange_max={w['F_rel_range_max']} PASS={w['pass_5_2']}")
    variation_tables(by, "5_2", PILOT / "arrival")
    res = {b: bool([w for w in wl if w["beta"] == b]) and all(w["pass_5_2"] for w in wl if w["beta"] == b) for b in BETAS}
    print("5.2 beta pass", res, "invalid", len(invalid))
    return res


def gpuload():
    prof = offline()
    runs, _ = load_runs(PILOT / "arrival" / "runs")
    A, B, C, D = [], [], [], []
    for r in runs:
        if set(r["placement"]) != {"G"}:
            continue
        ratios = [i["window"]["service_ms_p50"] / prof[i["model"]]["gpu"]["service_ms_p50"] for i in r["instances"] if i["window"]["service_ms_p50"]]
        nd = sum(1 for i in r["instances"] if i["window"]["dropped_full"] > 0)
        util = r["gpu_samples"]["util_mean"]
        c2u, c2l = util is not None and util >= 80.0, bool(ratios) and float(np.median(ratios)) >= 1.5
        A.append({"beta": r["meta"]["beta"], "mult": r["meta"]["multiplier"], "n": r["meta"]["n_instances"], "rep": r["repeat"],
                  "gpu_instances_with_drop": nd, "gpu_util_mean": util, "latency_ratio_median": round(float(np.median(ratios)), 3),
                  "latency_ratio_max": round(max(ratios), 3), "ok_i": nd > 0, "ok_ii_util": c2u, "ok_ii_latency_median": c2l,
                  "ii_latency_any": max(ratios) >= 1.5, "pass_A": nd > 0 and (c2u or c2l), "D_macro": round(r["system"]["drop_macro"], 4),
                  "process_cpu_sec": r["process_cpu_sec_in_window"]})
    sdir = {0.05: "beta_0.05/single_gpu/runs", 0.1: "smoke/runs", 0.25: "beta_0.25/single_gpu/runs"}
    for b, d in sdir.items():
        for r in load_runs(PILOT / d)[0]:
            i = r["instances"][0]
            if i["execution"] != "gpu":
                continue
            w = i["window"]
            B.append({"beta": b, "model": i["model"], "rate": round(i["rate"], 3), "scheduled_ticks": w["scheduled_ticks"], "generated": w["generated"],
                      "dropped_full": w["dropped_full"], "completed": w["completed"], "drop_ratio": round(w["q_i"], 5), "ok_iii": w["q_i"] < 0.01})
    rr, _ = load_runs(PILOT / "rate_range" / "runs")
    for b in BETAS:
        one = [r for r in rr if r["meta"]["beta"] == b and r["meta"]["multiplier"] == 1]
        four = [r for r in runs if r["meta"]["beta"] == b and r["meta"]["multiplier"] == 4 and r["meta"]["family"] == F4
                and set(r["placement"]) == {"G"} and r["repeat"] == 0]
        row = {"beta": b}
        if one:
            r = one[0]
            lam = sum(i["rate"] for i in r["instances"])
            row.update({"x1_D_macro": round(r["system"]["drop_macro"], 4), "x1_F": round(r["system"]["throughput_fps"], 2), "x1_Lambda": round(lam, 2),
                        "x1_F_over_Lambda": round(r["system"]["throughput_fps"] / lam, 4),
                        "ok_1x_unsaturated": r["system"]["drop_macro"] <= 0.3 and r["system"]["throughput_fps"] / lam >= 0.7})
        if four:
            r = four[0]
            nd = sum(1 for i in r["instances"] if i["window"]["dropped_full"] > 0)
            row.update({"x4_gpu_instances_with_drop": nd, "x4_D_macro": round(r["system"]["drop_macro"], 4), "ok_4x_saturated": nd > 0})
        row["pass_C"] = bool(row.get("ok_1x_unsaturated") and row.get("ok_4x_saturated"))
        C.append(row)
        bb = [x for x in B if x["beta"] == b]
        if bb:
            lo = min(bb, key=lambda x: x["generated"])
            D.append({"beta": b, "models": len(bb), "min_generated_1x": lo["generated"], "model": lo["model"], "rate": lo["rate"],
                      "q_i_resolution": round(1 / lo["generated"], 4) if lo["generated"] else None, "pass_D": len(bb) == 8 and lo["generated"] >= 30,
                      "models_below_30": ",".join(f"{x['model']}({x['generated']})" for x in bb if x["generated"] < 30)})
    write_csv(PILOT / "gpu_load_5_3_A.csv", A)
    write_csv(PILOT / "gpu_load_5_3_B.csv", B)
    write_csv(PILOT / "gpu_load_5_3_C.csv", C)
    write_csv(PILOT / "gpu_load_5_3_D.csv", D)
    res = {}
    for b in BETAS:
        a4 = [x for x in A if x["beta"] == b and x["mult"] == 4]
        bb = [x for x in B if x["beta"] == b]
        res[b] = {"A": bool(a4) and all(x["pass_A"] for x in a4), "A_detail": f"{sum(x['pass_A'] for x in a4)}/{len(a4)}",
                  "B": len(bb) == 8 and all(x["ok_iii"] for x in bb), "B_detail": f"{sum(x['ok_iii'] for x in bb)}/{len(bb)}",
                  "C": next((x["pass_C"] for x in C if x["beta"] == b), False), "D": next((x["pass_D"] for x in D if x["beta"] == b), False)}
    for x in A:
        print("5.3A", x)
    for x in C + D:
        print("5.3C/D", x)
    print("5.3", res)
    return res


def select():
    import csv
    smoke = list(csv.DictReader(open(PILOT / "smoke" / "smoke_5_1.csv")))
    ok51 = len(smoke) == 16 and all(r["pass_all"] == "True" for r in smoke)
    a, g = arrival(), gpuload()
    rows = []
    for b in BETAS:
        full = ok51 and a[b] and g[b]["A"] and g[b]["B"] and g[b]["C"] and g[b]["D"]
        cond = ok51 and a[b] and g[b]["A"] and g[b]["B"] and g[b]["C"] and not g[b]["D"]
        rows.append({"beta": b, "5.1": f"{sum(r['pass_all'] == 'True' for r in smoke)}/{len(smoke)}", "ok_5.1": ok51, "ok_5.2": a[b],
                     "ok_5.3A": g[b]["A"], "5.3A": g[b]["A_detail"], "ok_5.3B": g[b]["B"], "5.3B": g[b]["B_detail"], "ok_5.3C": g[b]["C"], "ok_5.3D": g[b]["D"],
                     "status": "pass" if full else "conditional(D only)" if cond else "fail"})
    write_csv(PILOT / "beta_selection.csv", rows)
    for r in rows:
        print("SELECT", r)
    elig = [r for r in rows if r["status"] != "fail"]
    print("SMALLEST ELIGIBLE:", elig[0] if elig else None)


def repeat():
    runs, invalid = load_runs(PILOT / "repeat" / "runs")
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by[wkey(r)][r["placement"]].append(r)
    variation_tables(by, "5_4", PILOT / "repeat")
    q = defaultdict(list)
    for r in runs:
        for i in r["instances"]:
            q[(r["placement"], i["name"])].append(i["window"]["q_i"])
    d = np.array([max(v) - min(v) for v in q.values()])
    print("5.4 runs", len(runs), "invalid", len(invalid), "q_i abs range median %.4f p90 %.4f max %.4f" % (np.median(d), np.percentile(d, 90), d.max()))
    (PILOT / "repeat" / "q_i_range.json").write_text(json.dumps({"groups": len(d), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": float(d.max())}))


if __name__ == "__main__":
    {"conservation": conservation, "arrival": arrival, "gpuload": gpuload, "select": select, "repeat": repeat, "dropmetric": ap.dropmetric}[sys.argv[1]]()
