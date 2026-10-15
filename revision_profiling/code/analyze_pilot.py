"""Phase 1 pilot analysis (spec v2.2 sections 5.2-5.5). Criteria are the spec's; nothing here is fitted to results.

  python3 -B code/analyze_pilot.py arrival      -> pilot/arrival/arrival_5_2*.csv
  python3 -B code/analyze_pilot.py gpuload      -> pilot/beta_*/gpu_load_5_3*.csv, pilot/beta_selection.csv
  python3 -B code/analyze_pilot.py repeat       -> pilot/repeat/repeat_5_4*.csv
  python3 -B code/analyze_pilot.py dropmetric   -> pilot/drop_metric_5_5_*.csv   (reporting only; never used to pick beta)

Rules fixed before the pilot runs were looked at:
  5.2 generated consistency: per workload, repeat and instance across the measured placements;
      scheduled_ticks >= 200 -> (max - min) / mean <= 2 %, otherwise max - min <= 2 requests.
  5.3 (ii) latency rule: median over the GPU instances of (service p50 / isolated GPU service p50) >= 1.5
      (the "any instance" variant is reported next to it).
"""
import sys

sys.dont_write_bytecode = True

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot"
BETAS = (0.25, 0.5, 1.0)
ALPHAS = [round(0.1 * k, 1) for k in range(1, 11)]


def load_runs(d):
    runs, invalid = [], []
    for p in sorted(Path(d).glob("*__rep*.json")):
        if ".rejected." in p.name:
            continue
        r = json.loads(p.read_text())
        r["_file"] = p.name
        (runs if r["valid"] else invalid).append(r)
    return runs, invalid


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def offline():
    return json.loads((ROOT / "offline_profile" / "offline_profile.json").read_text())["models"]


def wkey(r):
    m = r["meta"]
    return (m["family"], m["beta"], m["multiplier"])


def arrival():
    runs, invalid = load_runs(PILOT / "arrival" / "runs")
    inst_rows, wl_rows = [], []
    by = defaultdict(list)
    for r in runs:
        by[wkey(r)].append(r)
        for i in r["instances"]:
            w = i["window"]
            inst_rows.append({"family": r["meta"]["family"], "beta": r["meta"]["beta"], "mult": r["meta"]["multiplier"],
                              "placement": r["placement"], "rep": r["repeat"], "instance": i["name"], "device": i["execution"],
                              "rate": round(i["rate"], 3), "scheduled_ticks": w["scheduled_ticks"], "generated": w["generated"],
                              "missed_ticks": w["missed_ticks"], "realization": round(w["realization"], 4),
                              "missed_ratio": round(w["missed_ticks"] / w["scheduled_ticks"], 4),
                              "tick_delay_p95_ms": w["tick_delay_ms_p95"], "tick_delay_max_ms": w["tick_delay_ms_max"],
                              "dropped_full": w["dropped_full"], "completed": w["completed"], "failed": w["failed"],
                              "conservation_flow": w["conservation_flow"],
                              "ok_real": w["realization"] >= 0.95, "ok_missed": w["missed_ticks"] / w["scheduled_ticks"] <= 0.05})
    for (fam, beta, mult), rs in sorted(by.items()):
        placements = sorted({r["placement"] for r in rs})
        reals = [i["window"]["realization"] for r in rs for i in r["instances"]]
        miss = [i["window"]["missed_ticks"] / i["window"]["scheduled_ticks"] for r in rs for i in r["instances"]]
        fdiff = {}
        for pl in placements:
            f = [r["system"]["throughput_fps"] for r in sorted(rs, key=lambda x: x["repeat"]) if r["placement"] == pl]
            fdiff[pl] = abs(f[0] - f[1]) / (sum(f) / 2) if len(f) == 2 and sum(f) > 0 else None
        gen_bad, gen_checked = [], 0
        for rep in sorted({r["repeat"] for r in rs}):
            rr = [r for r in rs if r["repeat"] == rep]
            for name in [i["name"] for i in rr[0]["instances"]]:
                g = [next(i for i in r["instances"] if i["name"] == name)["window"] for r in rr]
                gen = [x["generated"] for x in g]
                sched = min(x["scheduled_ticks"] for x in g)
                gen_checked += 1
                if sched >= 200:
                    ok = (max(gen) - min(gen)) / (sum(gen) / len(gen)) <= 0.02 if sum(gen) else False
                    rule = "relative<=2%"
                else:
                    ok = max(gen) - min(gen) <= 2
                    rule = "absolute<=2"
                if not ok:
                    gen_bad.append(f"{name}(rep{rep},{rule},gen={gen})")
        complete = len(rs) == 2 * len(placements) and len(placements) == 3
        ok_real = min(reals) >= 0.95 and max(miss) <= 0.05
        ok_f = all(v is not None and v <= 0.05 for v in fdiff.values())
        wl_rows.append({"family": fam, "n": rs[0]["meta"]["n_instances"], "beta": beta, "mult": mult, "valid_runs": len(rs),
                        "complete": complete, "min_realization": round(min(reals), 4), "max_missed_ratio": round(max(miss), 4),
                        "ok_realization": ok_real,
                        **{f"F_rel_diff_{pl}": (round(v, 4) if v is not None else None) for pl, v in fdiff.items()},
                        "ok_F_repeat": ok_f, "generated_checks": gen_checked, "generated_violations": len(gen_bad),
                        "generated_violation_list": "; ".join(gen_bad)[:600], "ok_generated": not gen_bad,
                        "pass": bool(complete and ok_real and ok_f and not gen_bad)})
    write_csv(PILOT / "arrival" / "arrival_5_2_instances.csv", inst_rows)
    write_csv(PILOT / "arrival" / "arrival_5_2_workloads.csv", wl_rows)
    write_csv(PILOT / "arrival" / "invalid_runs.csv", [{"file": r["_file"], "reasons": ";".join(r["invalid_reasons"])} for r in invalid])
    beta_pass = {b: all(w["pass"] for w in wl_rows if w["beta"] == b) and any(w["beta"] == b for w in wl_rows) for b in BETAS}
    for w in wl_rows:
        print(f"5.2 {w['family'][:12]:12s} N={w['n']} b={w['beta']} x{w['mult']} runs={w['valid_runs']} minreal={w['min_realization']} "
              f"maxmiss={w['max_missed_ratio']} Fdiff={[w[k] for k in w if k.startswith('F_rel_diff_')]} genviol={w['generated_violations']} PASS={w['pass']}")
    print("5.2 beta pass:", beta_pass, "invalid runs:", len(invalid))
    return beta_pass, wl_rows


def gpuload():
    prof = offline()
    runs, _ = load_runs(PILOT / "arrival" / "runs")
    rows_a, rows_b = [], []
    for r in runs:
        if set(r["placement"]) != {"G"}:
            continue
        ratios = [i["window"]["service_ms_p50"] / prof[i["model"]]["gpu"]["service_ms_p50"] for i in r["instances"] if i["window"]["service_ms_p50"]]
        drop_inst = [i["name"] for i in r["instances"] if i["window"]["dropped_full"] > 0]
        util = r["gpu_samples"]["util_mean"]
        c1 = len(drop_inst) > 0
        c2u = util is not None and util >= 80.0
        c2l = len(ratios) > 0 and float(np.median(ratios)) >= 1.5
        rows_a.append({"beta": r["meta"]["beta"], "mult": r["meta"]["multiplier"], "family": r["meta"]["family"], "n": r["meta"]["n_instances"],
                       "rep": r["repeat"], "gpu_instances_with_drop": len(drop_inst), "gpu_util_mean": util,
                       "latency_ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
                       "latency_ratio_min": round(min(ratios), 3) if ratios else None, "latency_ratio_max": round(max(ratios), 3) if ratios else None,
                       "ok_i_drop": c1, "ok_ii_util>=80": c2u, "ok_ii_latency_median>=1.5": c2l, "ii_latency_any>=1.5": bool(ratios and max(ratios) >= 1.5),
                       "ok_ii": c2u or c2l, "pass_A": c1 and (c2u or c2l),
                       "process_cpu_sec_in_window": r["process_cpu_sec_in_window"],
                       "preprocess_ms_mean_by_instance": json.dumps({i["name"]: round(i["window"]["preprocess_ms_mean"], 3) for i in r["instances"] if i["window"]["preprocess_ms_mean"] is not None}),
                       "worker_cpu_sec_by_instance": json.dumps({i["name"]: i["cpu_sec_in_window"]["worker_thread"] for i in r["instances"]})})
    single_dirs = {0.25: PILOT / "beta_0.25" / "single_gpu" / "runs", 0.5: PILOT / "smoke" / "runs", 1.0: PILOT / "beta_1.0" / "single_gpu" / "runs"}
    for b, d in single_dirs.items():
        rs, _ = load_runs(d)
        for r in rs:
            i = r["instances"][0]
            if i["execution"] != "gpu":
                continue
            w = i["window"]
            q = w["dropped_full"] / w["generated"] if w["generated"] else None
            rows_b.append({"beta": b, "model": i["model"], "rate": round(i["rate"], 3), "generated": w["generated"], "dropped_full": w["dropped_full"],
                           "completed": w["completed"], "drop_ratio": round(q, 5) if q is not None else None, "realization": round(w["realization"], 4),
                           "gpu_util_mean": r["gpu_samples"]["util_mean"], "ok_iii": q is not None and q < 0.01})
    sel = []
    for b in BETAS:
        a4 = [x for x in rows_a if x["beta"] == b and x["mult"] == 4]
        bb = [x for x in rows_b if x["beta"] == b]
        write_csv(PILOT / f"beta_{b}" / "gpu_load_5_3_A_all_gpu.csv", [x for x in rows_a if x["beta"] == b])
        write_csv(PILOT / f"beta_{b}" / "gpu_load_5_3_B_single_gpu_1x.csv", bb)
        sel.append({"beta": b, "A_runs_4x": len(a4), "A_pass_4x": sum(x["pass_A"] for x in a4), "ok_A": bool(a4) and all(x["pass_A"] for x in a4),
                    "B_models": len(bb), "B_pass": sum(x["ok_iii"] for x in bb), "ok_B": len(bb) == 9 and all(x["ok_iii"] for x in bb),
                    "B_fail_models": ",".join(x["model"] for x in bb if not x["ok_iii"])})
    for x in rows_a:
        print(f"5.3A b={x['beta']} x{x['mult']} N={x['n']} rep{x['rep']} drop_inst={x['gpu_instances_with_drop']} util={x['gpu_util_mean']} "
              f"lat_med={x['latency_ratio_median']} pass={x['pass_A']}")
    for x in sel:
        print("5.3", x)
    return sel


def repeat():
    runs, invalid = load_runs(PILOT / "repeat" / "runs")
    by = defaultdict(list)
    for r in runs:
        by[r["combination"]].append(r)
    rows = []
    for comb, rs in sorted(by.items()):
        f = [r["system"]["throughput_fps"] for r in rs]
        q = defaultdict(list)
        for r in rs:
            for i in r["instances"]:
                q[i["name"]].append(i["window"]["q_i"])
        qd = {k: max(v) - min(v) for k, v in q.items()}
        rows.append({"combination": comb, "placement": rs[0]["placement"], "repeats": len(rs), "F_mean": round(float(np.mean(f)), 2),
                     "F_values": json.dumps([round(x, 2) for x in f]), "F_rel_range": round((max(f) - min(f)) / float(np.mean(f)), 4) if np.mean(f) > 0 else None,
                     "D_macro_values": json.dumps([round(r["system"]["drop_macro"], 4) for r in rs]),
                     "q_i_abs_range_max": round(max(qd.values()), 4), "q_i_abs_range_median": round(float(np.median(list(qd.values()))), 4),
                     "q_i_abs_range_by_instance": json.dumps({k: round(v, 4) for k, v in qd.items()})})
    write_csv(PILOT / "repeat" / "repeat_5_4.csv", rows)
    write_csv(PILOT / "repeat" / "invalid_runs.csv", [{"file": r["_file"], "reasons": ";".join(r["invalid_reasons"])} for r in invalid])
    fr = [x["F_rel_range"] for x in rows if x["F_rel_range"] is not None]
    summ = {"candidates": len(rows), "F_rel_range_median": round(float(np.median(fr)), 4) if fr else None, "F_rel_range_max": round(max(fr), 4) if fr else None,
            "q_i_abs_range_median_over_candidates": round(float(np.median([x["q_i_abs_range_median"] for x in rows])), 4) if rows else None,
            "q_i_abs_range_max": round(max(x["q_i_abs_range_max"] for x in rows), 4) if rows else None, "invalid_runs": len(invalid)}
    write_csv(PILOT / "repeat" / "repeat_5_4_summary.csv", [summ])
    for x in rows:
        print("5.4", x["combination"], x["placement"], x["F_values"], x["F_rel_range"], x["q_i_abs_range_max"])
    print("5.4 summary", summ)
    return summ


def dropmetric():
    """Three drop definitions on every pilot workload (reporting only). 'best' = best measured among pilot candidates."""
    runs = load_runs(PILOT / "arrival" / "runs")[0] + load_runs(PILOT / "repeat" / "runs")[0]
    by = defaultdict(lambda: defaultdict(list))
    cons_rows = []
    for r in runs:
        by[wkey(r)][r["placement"]].append(r)
        lam = sum(i["rate"] for i in r["instances"])
        s = r["system"]
        cons_rows.append({"family": r["meta"]["family"], "beta": r["meta"]["beta"], "mult": r["meta"]["multiplier"], "placement": r["placement"],
                          "rep": r["repeat"], "Lambda_requested": round(lam, 3), "generated_rate": round(s["generated_rate_fps"], 3),
                          "F": round(s["throughput_fps"], 3), "R": round(s["drop_rate_fps"], 3),
                          "rel_err_F_plus_R_vs_Lambda": round((s["throughput_fps"] + s["drop_rate_fps"] - lam) / lam, 5),
                          "rel_err_F_plus_R_vs_generated": round((s["throughput_fps"] + s["drop_rate_fps"] - s["generated_rate_fps"]) / s["generated_rate_fps"], 6),
                          "rel_err_generated_vs_Lambda": round((s["generated_rate_fps"] - lam) / lam, 5)})
    cand_rows, best_rows, alpha_rows, prot_rows, q_rows = [], [], [], [], []
    for key, pls in sorted(by.items()):
        fam, beta, mult = key
        c = {}
        for pl, rs in pls.items():
            q = defaultdict(list)
            for r in rs:
                for i in r["instances"]:
                    q[i["name"]].append(i["window"]["q_i"])
                    q_rows.append({"family": fam, "beta": beta, "mult": mult, "placement": pl, "rep": r["repeat"], "instance": i["name"],
                                   "device": i["execution"], "rate": round(i["rate"], 3), "generated": i["window"]["generated"],
                                   "dropped_full": i["window"]["dropped_full"], "completed": i["window"]["completed"], "q_i": round(i["window"]["q_i"], 5)})
            c[pl] = {"F": float(np.mean([r["system"]["throughput_fps"] for r in rs])), "R": float(np.mean([r["system"]["drop_rate_fps"] for r in rs])),
                     "Dmacro": float(np.mean([r["system"]["drop_macro"] for r in rs])), "Dmax": float(np.mean([r["system"]["drop_max"] for r in rs])),
                     "q": {k: float(np.mean(v)) for k, v in q.items()}, "reps": len(rs)}
        fmax = max(v["F"] for v in c.values())
        rmax = max(v["R"] for v in c.values())
        for pl, v in c.items():
            v["T"] = v["F"] / fmax if fmax > 0 else 0.0
            v["Da"] = v["R"] / rmax if rmax > 0 else 0.0
            cand_rows.append({"family": fam, "beta": beta, "mult": mult, "placement": pl, "reps": v["reps"], "F": round(v["F"], 3), "R": round(v["R"], 3),
                              "T=F/Fmax": round(v["T"], 4), "D_a=R/Rmax": round(v["Da"], 4), "D_macro": round(v["Dmacro"], 4), "D_max": round(v["Dmax"], 4)})
        tmax = max(c, key=lambda p: (c[p]["F"], p))
        res = {}
        for a in ALPHAS:
            b_a = max(c, key=lambda p: (c[p]["T"] - a * c[p]["Da"], p))
            b_b = max(c, key=lambda p: (c[p]["T"] - a * c[p]["Dmacro"], p))
            b_c = max(c, key=lambda p: (c[p]["T"] - a * c[p]["Dmax"], p))
            res[a] = (b_a, b_b, b_c)
            alpha_rows.append({"family": fam, "beta": beta, "mult": mult, "candidates": len(c), "alpha": a, "throughput_max": tmax,
                               "best_a_aggregate": b_a, "best_b_Dmacro": b_b, "best_c_Dmax": b_c,
                               "a_equals_throughput_max": b_a == tmax, "b_equals_throughput_max": b_b == tmax, "c_equals_throughput_max": b_c == tmax})
        b_a, b_b, b_c = res[0.3]
        best_rows.append({"family": fam, "n": len(next(iter(c.values()))["q"]), "beta": beta, "mult": mult, "candidates": len(c), "throughput_max": tmax,
                          "best_a_alpha0.3": b_a, "best_b_alpha0.3": b_b, "best_c_alpha0.3": b_c,
                          "a_equals_throughput_max": b_a == tmax, "b_equals_throughput_max": b_b == tmax, "c_equals_throughput_max": b_c == tmax,
                          "alphas_where_b_differs": ",".join(str(a) for a in ALPHAS if res[a][1] != tmax),
                          "alphas_where_c_differs": ",".join(str(a) for a in ALPHAS if res[a][2] != tmax),
                          "alphas_where_a_differs": ",".join(str(a) for a in ALPHAS if res[a][0] != tmax)})
        for label, idx in (("b_Dmacro", 1), ("c_Dmax", 2)):
            for a in ALPHAS:
                b = res[a][idx]
                if b == tmax:
                    continue
                for name in c[b]["q"]:
                    prot_rows.append({"family": fam, "beta": beta, "mult": mult, "definition": label, "alpha": a, "throughput_max": tmax, "best": b,
                                      "instance": name, "q_i_at_throughput_max": round(c[tmax]["q"][name], 4), "q_i_at_best": round(c[b]["q"][name], 4),
                                      "delta_q_i": round(c[b]["q"][name] - c[tmax]["q"][name], 4),
                                      "F_at_throughput_max": round(c[tmax]["F"], 2), "F_at_best": round(c[b]["F"], 2)})
                break   # smallest alpha at which the choice differs
    write_csv(PILOT / "drop_metric_5_5_candidates.csv", cand_rows)
    write_csv(PILOT / "drop_metric_5_5_best_measured.csv", best_rows)
    write_csv(PILOT / "drop_metric_5_5_alpha_sensitivity.csv", alpha_rows)
    write_csv(PILOT / "drop_metric_5_5_protected_instances.csv", prot_rows)
    write_csv(PILOT / "drop_metric_5_5_conservation_F_R_Lambda.csv", cons_rows)
    write_csv(PILOT / "drop_metric_5_5_q_i.csv", q_rows)
    n = len(best_rows)
    e1 = np.array([x["rel_err_F_plus_R_vs_Lambda"] for x in cons_rows])
    e2 = np.array([x["rel_err_F_plus_R_vs_generated"] for x in cons_rows])
    print(f"5.5 workloads={n} a==Tmax {sum(x['a_equals_throughput_max'] for x in best_rows)}/{n} b==Tmax {sum(x['b_equals_throughput_max'] for x in best_rows)}/{n} "
          f"c==Tmax {sum(x['c_equals_throughput_max'] for x in best_rows)}/{n} (alpha 0.3)")
    for x in best_rows:
        print("   ", x["family"][:12], x["beta"], x["mult"], "cand", x["candidates"], "Tmax", x["throughput_max"], "a", x["best_a_alpha0.3"], "b", x["best_b_alpha0.3"],
              "c", x["best_c_alpha0.3"], "| b differs at", x["alphas_where_b_differs"], "| c differs at", x["alphas_where_c_differs"], "| a differs at", x["alphas_where_a_differs"])
    print(f"5.5 (F+R-Lambda)/Lambda: median {np.median(e1):.4f} min {e1.min():.4f} max {e1.max():.4f}; (F+R-gen)/gen: median {np.median(e2):.5f} min {e2.min():.5f} max {e2.max():.5f}")


if __name__ == "__main__":
    {"arrival": arrival, "gpuload": gpuload, "repeat": repeat, "dropmetric": dropmetric}[sys.argv[1]]()
