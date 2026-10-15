"""Phase N2-A A4: labels and dataset CSVs for the CPU-NPU board, with the rules of recompute_scores.py (Phase 2) unchanged.

  python3 -B code/recompute_scores_npu.py          # revision_profiling_npu/results/runs -> revision_profiling_npu/dataset

Reused as is (imported, not copied): recompute_scores.load_runs, recompute_scores.build_tables (per candidate: mean over its
valid repeats of F, R_agg, D_macro, D_max; per workload from the means: F_max, R_max; T = F/F_max, D_agg = R_agg/R_max,
S = T - 0.3 D_macro; repeat values *_rep0..2 and SE), recompute_scores.schedule_name, the same CSV columns and split rule.
Board-specific differences (everything else is identical):
  - workload completeness: expected candidates = the feasible placements of the family (gen_schedules_npu.enumerate_feasible,
    3/7/13/21 for N = 1..4) instead of 2^N;
  - n_gpu column = number of instances on an accelerator (npu0 or npu1); the column name is kept for compatibility;
  - instance identity in json_content / schedule CSVs = the schedule instance name (resnet50_big, ...), as in the January
    Antara data, not the registry key (the registry key is kept in labels_instances_all.csv, column registry_key);
  - json_content window_sec = the measured window (30 s) instead of the constant 10.0
    (recompute_scores.py json_content / legacy_schedule_content, lines 128-158, otherwise followed line by line).
"""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import pandas as pd
import yaml

import recompute_scores as RS
from gen_schedules_npu import FAMILIES, enumerate_feasible, read_family

ROOT = Path(__file__).resolve().parent.parent.parent / "revision_profiling_npu"
TRAIN, TEST = (1.0, 2.0, 4.0), (3.0,)


def json_content(row, inst, window):
    models = {}
    for _, i in inst.sort_values("instance_index").iterrows():
        models[f"view{int(i['instance_index']) + 1}"] = {
            "model": i["instance"], "execution": str(i["device"]).upper(),
            "completed": i["completed_i"], "dropped": i["dropped_i"], "generated": i["generated_i"], "q_i": i["q_i_i"],
            "latency_p50_ms": None if pd.isna(i["latency_p50_i"]) else i["latency_p50_i"]}
    return json.dumps({
        "timestamp": row["execution_id_first"], "window_sec": window, "combination": row["combination"], "models": models,
        "total": {"total_throughput_fps": row["F_mean"]},
        "derived": {"throughput_norm": row["throughput_norm"], "drop_rate_norm": row["drop_rate_norm"],
                    "drop_agg_norm": row["drop_agg_norm"], "drop_max": row["drop_max"],
                    "drop_rate_fps": row["R_agg_mean"], "throughput_fps": row["F_mean"], "F_max": row["F_max"], "R_max": row["R_max"],
                    "n_reps": int(row["n_reps"])},
        "schedule_file": row["schedule_file"]})


def legacy_schedule_content(wl_runs):
    doc = {}
    for comb, reps in sorted(wl_runs.items()):
        r = reps[sorted(reps)[0]]
        doc[comb] = {i["name"]: {"model": i["name"], "execution": i["execution"], "infps": i["rate"], "base_rate": i["base_rate"],
                                 "beta": i["beta"], "multiplier": i["multiplier"]} for i in r["instances"]}
    return yaml.safe_dump(doc, sort_keys=True)


def main():
    out = ROOT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    valid, invalid = RS.load_runs([ROOT / "results" / "runs"])
    cand, inst, wl = RS.build_tables(valid)
    assert cand["beta"].nunique() == 1 and abs(cand["beta"].iloc[0] - 0.25) < 1e-12
    windows = {round(r["timing"]["window_sec_actual"]) for r in valid}
    assert windows == {30}, windows
    cand["n_gpu"] = [sum(1 for x in p.split(",") if not x.endswith(":cpu")) for p in cand["placement"]]
    reg = {(r["meta"]["family"], float(r["meta"]["multiplier"]), r["combination"], i["name"]): i["model"]
           for r in valid for i in r["instances"]}
    inst.insert(inst.columns.get_loc("model") + 1, "registry_key",
                [reg[(f, m, c, n)] for f, m, c, n in zip(inst["family"], inst["multiplier"], inst["combination"], inst["instance"])])
    inst["model"] = inst["instance"]
    feas = {}
    for fam in FAMILIES:
        base, _ = read_family(fam)
        feas[fam] = len(enumerate_feasible(sorted(next(iter(base.values())))))
    comp = cand.groupby(["family", "multiplier"]).agg(N=("N", "first"), measured=("combination", "nunique")).reset_index()
    comp["expected"] = comp["family"].map(feas)
    comp["complete"] = comp["measured"] == comp["expected"]
    comp["workload"] = [RS.schedule_name(f, m)[:-5] for f, m in zip(comp["family"], comp["multiplier"])]
    comp["excluded"] = ~comp["complete"]
    assert len(comp) == 44, len(comp)
    comp.to_csv(out / "workload_completeness.csv", index=False)
    keep = set(comp.loc[~comp["excluded"], "workload"])
    cand["workload"] = cand["schedule_file"].str[:-5]
    inst["workload"] = [RS.schedule_name(f, m)[:-5] for f, m in zip(inst["family"], inst["multiplier"])]
    cand.to_csv(out / "labels_candidates_all.csv", index=False)
    inst.to_csv(out / "labels_instances_all.csv", index=False)
    cand_k, inst_k = cand[cand["workload"].isin(keep)], inst[inst["workload"].isin(keep)]
    inst_g = {k: v for k, v in inst_k.groupby(["workload", "combination"])}
    for name, mults in (("train", TRAIN), ("test", TEST)):
        sub = cand_k[cand_k["multiplier"].isin(mults)].sort_values(["family", "multiplier", "comb_num"])
        rows = [{"schedule_file": r["schedule_file"], "combination": r["combination"],
                 "json_content": json_content(r, inst_g[(r["workload"], r["combination"])], 30.0)} for _, r in sub.iterrows()]
        pd.DataFrame(rows, columns=["schedule_file", "combination", "json_content"]).to_csv(out / f"{name}_x3.csv", index=False)
        srows = [{"schedule_name": RS.schedule_name(f, m), "content": legacy_schedule_content(wl[(f, m)])}
                 for (f, m) in sorted(wl) if m in mults and RS.schedule_name(f, m)[:-5] in keep]
        pd.DataFrame(srows, columns=["schedule_name", "content"]).to_csv(out / f"{name}_schedules_x3.csv", index=False)
        sub.to_csv(out / f"labels_{name}.csv", index=False)
        print(f"[{name}] mults={mults} workloads={sub['workload'].nunique()} rows={len(sub)}")
    print(f"valid runs {len(valid)}, invalid {len(invalid)}, workloads {len(comp)}, incomplete {int((~comp['complete']).sum())}")


if __name__ == "__main__":
    main()
