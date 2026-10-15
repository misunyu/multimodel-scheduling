"""Phase 2 section 2.4: integrity checks after the campaign.  ->  results/INTEGRITY.md + results/integrity/*.csv

  python3 -B code/integrity.py

Generated-count difference (quality indicator, rule of Phase 1b section 5.2, unchanged):
  per workload, repeat and instance, over the candidates of that workload:
    scheduled_ticks >= 200 : relative difference (max - min) / mean, violation if > 2 %
    scheduled_ticks <  200 : absolute difference max - min, violation if > 2 requests
  gen_mismatch_max      = largest relative difference among the instances with >= 200 ticks
  gen_mismatch_abs_max  = largest absolute difference among the instances with < 200 ticks
  gen_violation         = any instance violates its rule;  clean subset = workloads without gen_violation
  (gen_mismatch_rel_all = largest relative difference over ALL instances, listed for reference only)
"""
import sys

sys.dont_write_bytecode = True

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import phase2_plan as P
import recompute_scores as RS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "integrity"
ALPHA = 0.3
TOL = 1e-9


def md(df, floatfmt=4):
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    J = pd.DataFrame(P.jobs())
    valid, invalid = RS.load_runs([RES])
    cand, inst, wl = RS.build_tables(valid)
    got = {(r["meta"]["family"], int(r["meta"]["multiplier"]), r["combination"], r["repeat"]) for r in valid}
    J["valid"] = [(f, m, c, r) in got for f, m, c, r in zip(J["family"], J["mult"], J["combination"], J["rep"])]
    unexpected = got - set(zip(J["family"], J["mult"], J["combination"], J["rep"]))
    ex = pd.read_csv(ROOT / "logs" / "exec_order.csv")
    att = ex.groupby(["family", "mult", "combination", "rep"]).agg(attempts=("attempt", "max")).reset_index()
    J = J.merge(att, on=["family", "mult", "combination", "rep"], how="left")
    fr = J.groupby(["family", "N", "mult"]).agg(expected=("valid", "size"), valid=("valid", "sum"), retried_runs=("attempts", lambda s: int((s > 1).sum())),
                                                 extra_attempts=("attempts", lambda s: int((s.fillna(1) - 1).sum()))).reset_index()
    fr["missing"] = fr["expected"] - fr["valid"]
    fr.to_csv(OUT / "runs_by_family_rate.csv", index=False)
    rej = sorted(str(p.relative_to(RES)) for p in RES.rglob("*.rejected.*.json"))
    bad_attempts = ex[ex["status"] != "ok"]
    bad_attempts.to_csv(OUT / "non_ok_attempts.csv", index=False)

    # ---- workload completeness (section 2.4, 100 % of the feasible candidates)
    comp = []
    for fi, key, models in P.families():
        for m in P.MULTS:
            sub = J[(J["family"] == key) & (J["mult"] == m)]
            have = sub[sub["valid"]].groupby("combination")["rep"].nunique()
            exp_reps = 3 if (m == P.REPEAT_MULT and len(models) >= P.REPEAT_MIN_N) else 1
            comp.append({"workload": f"{key}_x{m}", "family": key, "N": len(models), "mult": m, "candidates": 2 ** len(models),
                         "candidates_measured": int(len(have)), "expected_reps": exp_reps,
                         "candidates_with_all_reps": int((have == exp_reps).sum()), "complete": len(have) == 2 ** len(models),
                         "all_reps_complete": int((have == exp_reps).sum()) == 2 ** len(models),
                         "missing": ";".join(sorted(set(sub["combination"]) - set(have.index)))})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "workload_completeness.csv", index=False)
    excluded = comp[~comp["complete"]]

    # ---- conservation / trace / realization
    cons, trace, real, miss, snapflag = Counter(), Counter(), [], [], 0
    for r in valid:
        for i in r["instances"]:
            w = i["window"]
            cons[w["conservation_flow"]] += 1
            trace[w["trace_check"]] += 1
            real.append(w["realization"])
            miss.append(w["missed_ticks"] / w["scheduled_ticks"])
            snapflag += bool(w["snapshot_flag"])
    real, miss = np.array(real), np.array(miss)
    run_minreal = np.array([min(i["window"]["realization"] for i in r["instances"]) for r in valid])
    inv_rows = [{"file": str(Path(r["_file"]).relative_to(RES)), "reasons": ";".join(r["invalid_reasons"])} for r in invalid]

    # ---- generated-count difference per workload
    gm = []
    for (family, mult), combos in sorted(wl.items()):
        n = len(family.split("+"))
        rel_max, abs_max, rel_all, viol, detail = 0.0, 0, 0.0, 0, []
        reps = sorted({k for c in combos.values() for k in c})
        for k in reps:
            runs = [c[k] for c in combos.values() if k in c]
            if len(runs) < 2:
                continue
            for ii in range(n):
                g = np.array([r["instances"][ii]["window"]["generated"] for r in runs])
                ticks = runs[0]["instances"][ii]["window"]["scheduled_ticks"]
                rel = float((g.max() - g.min()) / g.mean()) if g.mean() > 0 else 0.0
                ab = int(g.max() - g.min())
                rel_all = max(rel_all, rel)
                if ticks >= 200:
                    rel_max = max(rel_max, rel)
                    v = rel > 0.02
                else:
                    abs_max = max(abs_max, ab)
                    v = ab > 2
                if v:
                    viol += 1
                    lo = runs[int(g.argmin())]
                    detail.append({"workload": f"{family}_x{mult:g}", "N": n, "mult": mult, "rep": k, "instance": lo["instances"][ii]["name"],
                                   "scheduled_ticks": ticks, "gen_min": int(g.min()), "gen_max": int(g.max()), "rel_diff": round(rel, 5), "abs_diff": ab,
                                   "placement_of_min": lo["placement"], "n_cpu_of_min": lo["placement"].count("C"),
                                   "device_of_instance_in_min": lo["instances"][ii]["execution"],
                                   "candidates_below_98pct": int((g < 0.98 * np.median(g)).sum()), "candidates": len(g)})
        gm.append({"workload": f"{family}_x{mult:g}", "family": family, "N": n, "mult": mult, "gen_mismatch_max": rel_max, "gen_mismatch_abs_max": abs_max,
                   "gen_mismatch_rel_all": rel_all, "violations": viol, "gen_violation": viol > 0, "_detail": detail})
    gmd = pd.DataFrame([d for g in gm for d in g["_detail"]])
    gm = pd.DataFrame(gm).drop(columns="_detail")
    gm.to_csv(OUT / "gen_mismatch.csv", index=False)
    gmd.to_csv(OUT / "gen_mismatch_violations.csv", index=False)

    # ---- repeated workloads (3x, N >= 3)
    rep = cand[(cand["multiplier"] == P.REPEAT_MULT) & (cand["N"] >= P.REPEAT_MIN_N) & (cand["n_reps"] == 3)].copy()
    rep["F_rel_range"] = (rep[["F_rep0", "F_rep1", "F_rep2"]].max(axis=1) - rep[["F_rep0", "F_rep1", "F_rep2"]].min(axis=1)) / rep["F_mean"]
    rep["D_macro_abs_range"] = rep[["D_macro_rep0", "D_macro_rep1", "D_macro_rep2"]].max(axis=1) - rep[["D_macro_rep0", "D_macro_rep1", "D_macro_rep2"]].min(axis=1)
    sc = [f"S_macro_a0.3_rep{k}" for k in range(3)]
    rep["S_abs_range"] = rep[sc].max(axis=1) - rep[sc].min(axis=1)
    rep[["family", "N", "combination", "placement", "F_mean", "F_se", "F_rel_range", "D_macro_mean", "D_macro_se", "D_macro_abs_range",
         "S_macro_a0.3", "S_macro_a0.3_se", "S_abs_range"]].to_csv(OUT / "repeat_variation_candidates.csv", index=False)
    orc = []
    for fam, sub in cand[(cand["multiplier"] == P.REPEAT_MULT) & (cand["N"] >= P.REPEAT_MIN_N)].groupby("family"):
        sub = sub.sort_values("comb_num")
        row = {"family": fam, "N": int(sub["N"].iloc[0]), "candidates": len(sub), "reps_min": int(sub["n_reps"].min())}
        allgpu = sub["comb_num"].max()
        for d, col in (("macro", "S_macro_a0.3"), ("agg", "S_agg_a0.3"), ("max", "S_max_a0.3")):
            s = sub[col].values
            oset = set(sub["comb_num"].values[s >= s.max() - TOL])
            row[f"oracle_{d}"] = "combination_%03d" % min(oset)
            row[f"oracle_{d}_placement"] = sub.loc[sub["comb_num"] == min(oset), "placement"].iloc[0]
            row[f"n_oracle_{d}"] = len(oset)
            row[f"all_gpu_is_oracle_{d}"] = allgpu in oset
            row[f"S_all_gpu_{d}"] = float(sub.loc[sub["comb_num"] == allgpu, col].iloc[0])
            row[f"S_oracle_{d}"] = float(s.max())
        # per repeat, each with its own normalisation (what a single-run campaign would have produced)
        per = []
        for k in range(3):
            if sub[f"F_rep{k}"].isna().any():
                per.append(None)
                continue
            s = (sub[f"F_rep{k}"] / sub[f"F_rep{k}"].max() - ALPHA * sub[f"D_macro_rep{k}"]).values
            per.append(int(sub["comb_num"].values[np.lexsort((sub["comb_num"].values, -s))[0]]))
        mean_or = int(row["oracle_macro"].rsplit("_", 1)[1])
        row["oracle_rep0"], row["oracle_rep1"], row["oracle_rep2"] = per
        row["reps_equal_to_each_other"] = len(set(per)) == 1
        row["reps_equal_to_mean_oracle"] = sum(p == mean_or for p in per)
        s_mean = dict(zip(sub["comb_num"], sub["S_macro_a0.3"]))
        row["max_regret_of_rep_oracles_on_mean_label"] = max(float(sub["S_macro_a0.3"].max() - s_mean[p]) for p in per if p is not None)
        se_or = float(sub.loc[sub["comb_num"] == mean_or, "S_macro_a0.3_se"].iloc[0])
        row["oracle_S_se"] = se_or
        srt = sub.sort_values("S_macro_a0.3", ascending=False)
        row["margin_oracle_to_2nd"] = float(srt["S_macro_a0.3"].iloc[0] - srt["S_macro_a0.3"].iloc[1])
        orc.append(row)
    orc = pd.DataFrame(orc)
    orc.to_csv(OUT / "oracle_x3.csv", index=False)

    # ---- time drift
    ex_ok = ex[ex["status"] == "ok"].copy()
    t_first = ex["start_epoch"].min()
    ex_ok["hour"] = (ex_ok["start_epoch"] - t_first) / 3600
    drift = []
    r3 = cand[(cand["multiplier"] == P.REPEAT_MULT) & (cand["N"] >= P.REPEAT_MIN_N) & (cand["n_reps"] == 3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ratio = (r3[f"F_rep{b}"] / r3[f"F_rep{a}"]).values
        drift.append({"compare": f"rep{b}/rep{a}", "candidates": len(ratio), "F_ratio_median": float(np.median(ratio)), "F_ratio_mean": float(ratio.mean()),
                      "F_ratio_p05": float(np.percentile(ratio, 5)), "F_ratio_p95": float(np.percentile(ratio, 95)),
                      "share_above_1": float((ratio > 1).mean())})
    drift = pd.DataFrame(drift)
    drift.to_csv(OUT / "drift_repeat_blocks.csv", index=False)
    single = []
    for r in valid:
        if r["meta"]["n_instances"] == 1 and r["meta"]["multiplier"] == 1:
            i = r["instances"][0]
            e = ex_ok[(ex_ok["family"] == r["meta"]["family"]) & (ex_ok["mult"] == 1) & (ex_ok["combination"] == r["combination"])]
            single.append({"hour": round(float(e["hour"].iloc[0]), 2), "model": i["model"], "device": i["execution"], "rate": round(i["rate"], 2),
                           "F": r["system"]["throughput_fps"], "F_over_rate": round(r["system"]["throughput_fps"] / i["rate"], 4),
                           "service_ms_p50": round(i["window"]["service_ms_p50"], 3), "q_i": i["window"]["q_i"]})
    single = pd.DataFrame(single).sort_values("hour")
    single.to_csv(OUT / "drift_single_1x.csv", index=False)
    gpu = pd.read_csv(ROOT / "logs" / "gpu_1min.csv")
    gpu["hour"] = ((gpu["epoch"] - t_first) / 3600).astype(int)
    gh = gpu.groupby("hour").agg(samples=("epoch", "size"), temp_mean=("gpu_temp_c", "mean"), temp_max=("gpu_temp_c", "max"),
                                 clock_mean=("gpu_clock_mhz", "mean"), power_mean=("gpu_power_w", "mean"), load1_mean=("load1", "mean")).reset_index()
    gh.to_csv(OUT / "gpu_by_hour.csv", index=False)
    blocks = ex.groupby("block").agg(attempts=("attempt", "size"), start=("start_ts", "min"), end=("end_ts", "max"),
                                     hours=("seconds", lambda s: s.sum() / 3600), sec_p50=("seconds", "median")).reset_index()
    blocks.to_csv(OUT / "blocks.csv", index=False)
    tn = ex_ok.groupby("N")["seconds"].agg(runs="size", p50="median", p95=lambda s: s.quantile(0.95)).reset_index()
    tn["formula"] = [P.est_sec(n) for n in tn["N"]]
    tn.to_csv(OUT / "run_seconds_by_N.csv", index=False)

    # ---- label statistics, all-GPU oracle counts on every rate (single runs for 1,2,4x)
    stat = []
    for (fam, mult), sub in cand.groupby(["family", "multiplier"]):
        allgpu = sub["comb_num"].max()
        row = {"family": fam, "N": int(sub["N"].iloc[0]), "mult": mult, "F_max": float(sub["F_max"].iloc[0]), "Lambda": float(sub["Lambda_sched"].iloc[0]),
               "D_macro_min": float(sub["D_macro_mean"].min()), "D_macro_median": float(sub["D_macro_mean"].median()), "D_macro_max": float(sub["D_macro_mean"].max()),
               "D_macro_all_gpu": float(sub.loc[sub["comb_num"] == allgpu, "D_macro_mean"].iloc[0]),
               "T_all_gpu": float(sub.loc[sub["comb_num"] == allgpu, "throughput_norm"].iloc[0])}
        for d, col in (("macro", "S_macro_a0.3"), ("agg", "S_agg_a0.3"), ("max", "S_max_a0.3")):
            s = sub[col].values
            row[f"all_gpu_is_oracle_{d}"] = bool(sub.loc[sub["comb_num"] == allgpu, col].iloc[0] >= s.max() - TOL)
        stat.append(row)
    stat = pd.DataFrame(stat)
    stat.to_csv(OUT / "label_stats_by_workload.csv", index=False)

    # ---- report
    n3 = comp[(comp["mult"] == 3) & (comp["N"] >= 3)]
    L = ["# Campaign 무결성 점검 (Phase 2 v1.2 §2.4)", "", "`code/integrity.py`가 생성. 상세 표는 `results/integrity/`.", "",
         "## 1. Run 수", "",
         f"- 계획 {len(J)} runs, valid {int(J['valid'].sum())}, 누락 {int((~J['valid']).sum())}. 계획에 없는 결과 파일 {len(unexpected)}개.",
         f"- 실행 시도 {len(ex)}회. 첫 시도에 valid가 아니어서 재시도한 run {int((att['attempts'] > 1).sum())}개, ok가 아닌 시도 {len(bad_attempts)}회, `*.rejected.*` 파일 {len(rej)}개.",
         f"- 최종 invalid로 남은 결과 파일 {len(invalid)}개.", "",
         md(blocks), "",
         "Family × rate별 표는 `integrity/runs_by_family_rate.csv`. 누락이 있는 행:", "",
         md(fr[fr["missing"] > 0]) if (fr["missing"] > 0).any() else "없음.", "",
         "## 2. Workload 완전성 (전체 feasible 후보 100%)", "",
         f"- 88 workloads(22 families × 4 rates) 중 완전한 것 {int(comp['complete'].sum())}개.",
         f"- 평가 대상(3×, N≥3) 12개 중 완전한 것 {int(n3['complete'].sum())}개, 3회 반복까지 모두 있는 것 {int(n3['all_reps_complete'].sum())}개.",
         f"- **Primary 집계에서 제외되는 workload: {len(excluded)}개.**" + ("" if not len(excluded) else " " + "; ".join(excluded["workload"])), ""]
    if len(excluded):
        L += [md(excluded[["workload", "N", "mult", "candidates", "candidates_measured", "missing"]]), ""]
    L += ["## 3. Conservation·trace·도착", "",
          f"- instance {sum(cons.values())}개: conservation_flow 분포 {dict(cons)}, trace_check {dict(trace)}, snapshot flag(|diff|>1) {snapflag}개.",
          f"- 실현율(generated / scheduled): 최소 {real.min():.4f}, 1% 분위 {np.percentile(real, 1):.4f}, 5% 분위 {np.percentile(real, 5):.4f}, 중앙값 {np.median(real):.4f}. "
          f"0.95 미만 instance {int((real < 0.95).sum())}개, 그런 instance가 있는 run {int((run_minreal < 0.95).sum())}개.",
          f"- missed_ticks 비율: 최대 {miss.max():.4f}, 99% 분위 {np.percentile(miss, 99):.4f}, 0보다 큰 instance {int((miss > 0).sum())}개, 5% 초과 {int((miss > 0.05).sum())}개.",
          "", "## 4. Generated 차이 (품질 지표)", "",
          f"- 위반 workload {int(gm['gen_violation'].sum())} / {len(gm)}. 그중 3× N≥3: {int(gm[(gm['mult'] == 3) & (gm['N'] >= 3)]['gen_violation'].sum())} / 12.",
          f"- `gen_mismatch_max` 분포: 중앙값 {gm['gen_mismatch_max'].median():.4f}, 90% 분위 {gm['gen_mismatch_max'].quantile(0.9):.4f}, 최대 {gm['gen_mismatch_max'].max():.4f}.", ""]
    if len(gmd):
        byn = gmd.groupby(["N", "mult"]).agg(violations=("instance", "size"), workloads=("workload", "nunique"), rel_max=("rel_diff", "max"),
                                             min_is_all_cpu=("n_cpu_of_min", lambda s: int((s == gmd.loc[s.index, "N"]).sum())),
                                             instance_on_cpu_in_min=("device_of_instance_in_min", lambda s: int((s == "cpu").sum()))).reset_index()
        L += [md(byn), "", "위반 instance가 가장 적게 생성된 배치의 CPU instance 수 분포: " + str(dict(Counter(zip(gmd["N"], gmd["n_cpu_of_min"])))), "",
              "위반 workload 목록:", "", md(gm[gm["gen_violation"]][["workload", "N", "mult", "gen_mismatch_max", "gen_mismatch_abs_max", "violations"]]), ""]
    L += ["## 5. 시간 drift", "", "반복 블록 사이의 F 비율(같은 후보, 3× N≥3):", "", md(drift), "",
          "1× 단일 instance run을 실행 시각 순으로:", "", md(single), "", "GPU 상태(1분 표본의 시간대별 평균):", "", md(gh), "",
          "Run 시간(N별):", "", md(tn), "",
          "## 6. 반복 3회의 편차 (3× N≥3, 후보 " + str(len(rep)) + "개)", "",
          "| 지표 | 중앙값 | 90% 분위 | 최대 |", "|---|---|---|---|"]
    for name, col in (("F 상대 범위", "F_rel_range"), ("D_macro 절대 범위", "D_macro_abs_range"), ("S 절대 범위", "S_abs_range"), ("S의 SE", "S_macro_a0.3_se")):
        L.append(f"| {name} | {rep[col].median():.4f} | {rep[col].quantile(0.9):.4f} | {rep[col].max():.4f} |")
    L += ["", f"반복별 oracle(각 반복의 자체 정규화)이 세 번 모두 같은 workload: {int(orc['reps_equal_to_each_other'].sum())} / {len(orc)}.", "",
          md(orc[["family", "N", "oracle_macro_placement", "n_oracle_macro", "oracle_rep0", "oracle_rep1", "oracle_rep2", "reps_equal_to_mean_oracle",
                  "max_regret_of_rep_oracles_on_mean_label", "margin_oracle_to_2nd", "oracle_S_se"]]), "",
          "## 7. 전부-GPU가 oracle인 workload (3×, N≥3, α=0.3, 3회 평균 label)", "",
          "| 정의 | 전부-GPU가 oracle | 분모 |", "|---|---|---|"]
    for d, nm in (("agg", "(a) aggregate"), ("macro", "(b) D_macro"), ("max", "(c) D_max")):
        L.append(f"| {nm} | {int(orc[f'all_gpu_is_oracle_{d}'].sum())} | {len(orc)} |")
    L += ["", md(orc[["family", "N", "oracle_agg_placement", "oracle_macro_placement", "oracle_max_placement", "S_oracle_macro", "S_all_gpu_macro"]]), ""]
    if inv_rows:
        L += ["## 8. Invalid로 남은 결과", "", md(pd.DataFrame(inv_rows)), ""]
    (RES / "INTEGRITY.md").write_text("\n".join(L))
    summ = {"planned": len(J), "valid": int(J["valid"].sum()), "missing": int((~J["valid"]).sum()), "attempts": len(ex), "rejected_files": len(rej),
            "excluded_workloads": list(excluded["workload"]), "eval_workloads_complete": int(n3["complete"].sum()),
            "conservation": {str(k): v for k, v in cons.items()}, "trace": dict(trace), "gen_violation_workloads": list(gm[gm["gen_violation"]]["workload"]),
            "all_gpu_oracle": {d: int(orc[f"all_gpu_is_oracle_{d}"].sum()) for d in ("agg", "macro", "max")}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
