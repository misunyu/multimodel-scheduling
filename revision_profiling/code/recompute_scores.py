"""Labels and dataset CSVs from run JSONs (Phase 2 v1.2 section 3.1).

  python3 -B code/recompute_scores.py --results results --out dataset
  python3 -B code/recompute_scores.py --results pilot/arrival/runs pilot/repeat_5_4/runs --beta 0.1 --out dataset_pilot_check \
          --train-mults 2 --test-mults 4 --allow-partial

Per candidate: mean over its valid repeats first (1 value if not repeated): F, R_agg (aggregate drop rate, req/s), D_macro, D_max.
Per workload (family x multiplier), from the MEANS:  F_max = max F,  R_max = max R_agg
  T = F / F_max;  D_agg = R_agg / R_max (R_max = 0 -> 0);  D_macro and D_max are not normalised.
Dataset CSV (3 columns, no comments): schedule_file, combination, json_content.  json_content["derived"]:
  throughput_norm = T, drop_rate_norm = D_macro, drop_agg_norm = D_agg, drop_max = D_max.
Schedule CSV (schedule_name, content): the layout the existing builder parses (top-level combination_xxx -> instance -> model,
execution, infps); the executor schedule files under schedules/ keep their own layout.
A workload is "complete" iff every one of its 2^N placements has at least one valid run (section 2.4).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.3


def load_runs(dirs, beta=None):
    """All non-rejected run JSONs under the given directories. Returns (valid, invalid)."""
    valid, invalid = [], []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            if ".rejected." in p.name:
                continue
            r = json.loads(p.read_text())
            if "run_key" not in r or r.get("mode") != "open_loop":
                continue
            if beta is not None and abs(r["meta"]["beta"] - beta) > 1e-12:
                continue
            r["_file"] = str(p)
            (valid if r["valid"] else invalid).append(r)
    return valid, invalid


def rate_tag(mult):
    return f"x{mult:g}"


def schedule_name(family, mult):
    return f"{family}_{rate_tag(mult)}.yaml"


def se(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def build_tables(valid):
    """Returns (cand DataFrame, inst DataFrame, workloads dict)."""
    wl = defaultdict(lambda: defaultdict(dict))
    for r in valid:
        m = r["meta"]
        key = (m["family"], float(m["multiplier"]))
        assert r["repeat"] not in wl[key][r["combination"]], f"duplicate run {r['run_key']}"
        wl[key][r["combination"]][r["repeat"]] = r
    cand_rows, inst_rows = [], []
    for (family, mult), combos in sorted(wl.items()):
        n = None
        tmp = []
        for comb, reps in sorted(combos.items()):
            ks = sorted(reps)
            r0 = reps[ks[0]]
            n = r0["meta"]["n_instances"]
            F = np.array([reps[k]["system"]["throughput_fps"] for k in ks])
            R = np.array([reps[k]["system"]["drop_rate_fps"] for k in ks])
            Dm = np.array([reps[k]["system"]["drop_macro"] for k in ks])
            Dx = np.array([reps[k]["system"]["drop_max"] for k in ks])
            tmp.append((comb, ks, reps, F, R, Dm, Dx))
        fmax = max(t[3].mean() for t in tmp)
        rmax = max(t[4].mean() for t in tmp)
        beta = tmp[0][2][tmp[0][1][0]]["meta"]["beta"]
        for comb, ks, reps, F, R, Dm, Dx in tmp:
            r0 = reps[ks[0]]
            T = F.mean() / fmax
            Dagg = R.mean() / rmax if rmax > 0 else 0.0
            S_rep = F / fmax - ALPHA * Dm
            row = {"family": family, "N": n, "beta": beta, "multiplier": mult, "rate": rate_tag(mult), "schedule_file": schedule_name(family, mult),
                   "combination": comb, "comb_num": int(comb.rsplit("_", 1)[1]), "placement": r0["placement"],
                   "n_gpu": r0["placement"].count("G"), "n_reps": len(ks), "reps": ";".join(str(k) for k in ks),
                   "F_mean": F.mean(), "F_se": se(F), "R_agg_mean": R.mean(), "R_agg_se": se(R),
                   "D_macro_mean": Dm.mean(), "D_macro_se": se(Dm), "D_max_mean": Dx.mean(), "D_max_se": se(Dx),
                   "F_max": fmax, "R_max": rmax, "throughput_norm": T, "drop_rate_norm": Dm.mean(), "drop_agg_norm": Dagg, "drop_max": Dx.mean(),
                   "S_macro_a0.3": T - ALPHA * Dm.mean(), "S_macro_a0.3_se": se(S_rep),
                   "S_agg_a0.3": T - ALPHA * Dagg, "S_max_a0.3": T - ALPHA * Dx.mean(),
                   "Lambda_sched": sum(i["rate"] for i in r0["instances"]),
                   "generated_mean": float(np.mean([reps[k]["system"]["generated"] for k in ks])),
                   "min_realization": min(i["window"]["realization"] for k in ks for i in reps[k]["instances"]),
                   "max_missed_ratio": max(i["window"]["missed_ticks"] / i["window"]["scheduled_ticks"] for k in ks for i in reps[k]["instances"]),
                   "execution_id_first": r0["execution_id"]}
            for k in (0, 1, 2):
                j = ks.index(k) if k in ks else None
                row[f"F_rep{k}"] = F[j] if j is not None else np.nan
                row[f"R_agg_rep{k}"] = R[j] if j is not None else np.nan
                row[f"D_macro_rep{k}"] = Dm[j] if j is not None else np.nan
                row[f"D_max_rep{k}"] = Dx[j] if j is not None else np.nan
                row[f"S_macro_a0.3_rep{k}"] = S_rep[j] if j is not None else np.nan
            cand_rows.append(row)
            for ii, inst in enumerate(r0["instances"]):
                w = [reps[k]["instances"][ii]["window"] for k in ks]
                assert all(reps[k]["instances"][ii]["name"] == inst["name"] for k in ks)
                irow = {"family": family, "N": n, "multiplier": mult, "rate": rate_tag(mult), "combination": comb, "placement": r0["placement"],
                        "instance_index": ii, "instance": inst["name"], "model": inst["model"], "device": inst["execution"],
                        "lambda_i": inst["rate"], "scheduled_ticks": w[0]["scheduled_ticks"], "n_reps": len(ks)}
                for col, key in (("completed", "completed"), ("dropped", "dropped_full"), ("generated", "generated"), ("q_i", "q_i"),
                                 ("latency_p50", "latency_ms_p50"), ("service_p50", "service_ms_p50"), ("realization", "realization"),
                                 ("missed_ticks", "missed_ticks")):
                    vals = [x[key] for x in w]
                    ok = [v for v in vals if v is not None]
                    irow[f"{col}_i"] = float(np.mean(ok)) if ok else np.nan
                    for k in (0, 1, 2):
                        irow[f"{col}_i_rep{k}"] = (vals[ks.index(k)] if k in ks and vals[ks.index(k)] is not None else np.nan)
                inst_rows.append(irow)
    return pd.DataFrame(cand_rows), pd.DataFrame(inst_rows), wl


def json_content(row, inst):
    models = {}
    for _, i in inst.sort_values("instance_index").iterrows():
        models[f"view{int(i['instance_index']) + 1}"] = {
            "model": i["model"], "execution": str(i["device"]).upper(),
            "completed": i["completed_i"], "dropped": i["dropped_i"], "generated": i["generated_i"], "q_i": i["q_i_i"],
            "latency_p50_ms": None if pd.isna(i["latency_p50_i"]) else i["latency_p50_i"]}
    return json.dumps({
        "timestamp": row["execution_id_first"], "window_sec": 10.0, "combination": row["combination"], "models": models,
        "total": {"total_throughput_fps": row["F_mean"]},
        "derived": {"throughput_norm": row["throughput_norm"], "drop_rate_norm": row["drop_rate_norm"],
                    "drop_agg_norm": row["drop_agg_norm"], "drop_max": row["drop_max"],
                    "drop_rate_fps": row["R_agg_mean"], "throughput_fps": row["F_mean"], "F_max": row["F_max"], "R_max": row["R_max"],
                    "n_reps": int(row["n_reps"])},
        "schedule_file": row["schedule_file"]})


def legacy_schedule_content(wl_runs):
    """Flattened layout for the existing parser: combination_xxx -> instance name -> {model, execution, infps, ...}."""
    doc = {}
    for comb, reps in sorted(wl_runs.items()):
        r = reps[sorted(reps)[0]]
        doc[comb] = {i["name"]: {"model": i["model"], "execution": i["execution"], "infps": i["rate"], "base_rate": i["base_rate"],
                                 "beta": i["beta"], "multiplier": i["multiplier"]} for i in r["instances"]}
    return yaml.safe_dump(doc, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--train-mults", default="1,2,4")
    ap.add_argument("--test-mults", default="3")
    ap.add_argument("--allow-partial", action="store_true", help="pilot format check only: workloads need not hold all 2^N placements")
    ap.add_argument("--exclude-workloads", default="", help="';'-separated <family>_x<rate> keys excluded as incomplete (from INTEGRITY)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    valid, invalid = load_runs([ROOT / d for d in a.results], a.beta)
    cand, inst, wl = build_tables(valid)
    assert cand["beta"].nunique() == 1, "more than one beta in the input"
    comp = cand.groupby(["family", "multiplier"]).agg(N=("N", "first"), measured=("combination", "nunique")).reset_index()
    comp["expected"] = 2 ** comp["N"]
    comp["complete"] = comp["measured"] == comp["expected"]
    comp["workload"] = [schedule_name(f, m)[:-5] for f, m in zip(comp["family"], comp["multiplier"])]
    excl = set(x for x in a.exclude_workloads.split(";") if x)
    comp["excluded"] = comp["workload"].isin(excl) | (~comp["complete"] & (not a.allow_partial))
    comp.to_csv(out / "workload_completeness.csv", index=False)
    keep = set(comp.loc[~comp["excluded"], "workload"])
    cand["workload"] = cand["schedule_file"].str[:-5]
    inst["workload"] = [schedule_name(f, m)[:-5] for f, m in zip(inst["family"], inst["multiplier"])]
    cand.to_csv(out / "labels_candidates_all.csv", index=False)
    inst.to_csv(out / "labels_instances_all.csv", index=False)
    cand_k, inst_k = cand[cand["workload"].isin(keep)], inst[inst["workload"].isin(keep)]
    inst_g = {k: v for k, v in inst_k.groupby(["workload", "combination"])}
    split = {"train": [float(x) for x in a.train_mults.split(",")], "test": [float(x) for x in a.test_mults.split(",")]}
    for name, mults in split.items():
        sub = cand_k[cand_k["multiplier"].isin(mults)].sort_values(["family", "multiplier", "comb_num"])
        rows = [{"schedule_file": r["schedule_file"], "combination": r["combination"],
                 "json_content": json_content(r, inst_g[(r["workload"], r["combination"])])} for _, r in sub.iterrows()]
        pd.DataFrame(rows, columns=["schedule_file", "combination", "json_content"]).to_csv(out / f"{name}_x3.csv", index=False)
        srows = [{"schedule_name": schedule_name(f, m), "content": legacy_schedule_content(wl[(f, m)])}
                 for (f, m) in sorted(wl) if m in mults and schedule_name(f, m)[:-5] in keep]
        pd.DataFrame(srows, columns=["schedule_name", "content"]).to_csv(out / f"{name}_schedules_x3.csv", index=False)
        sub.to_csv(out / f"labels_{name}.csv", index=False)
        print(f"[{name}] mults={mults} workloads={sub['workload'].nunique()} rows={len(sub)}")
    print(f"valid runs {len(valid)}, invalid {len(invalid)}, workloads {len(comp)}, incomplete {int((~comp['complete']).sum())}, excluded {int(comp['excluded'].sum())}")


if __name__ == "__main__":
    main()
