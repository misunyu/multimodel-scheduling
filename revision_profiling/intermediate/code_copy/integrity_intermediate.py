"""Phase 4 integrity checks (copy of code/integrity.py, adapted; changes in ../CHANGES.md).  ->  intermediate/results/INTEGRITY.md + results/integrity/*.csv

  python3 -B revision_profiling/intermediate/code_copy/integrity_intermediate.py

Rules unchanged from Phase 2 (code/integrity.py):
  workload = (family, rate) is complete iff all 2^N candidates have a valid run; an incomplete workload is excluded (no correction / substitution)
  conservation_flow, trace_check, realization = generated / scheduled, missed_ticks / scheduled_ticks
  generated-count difference: scheduled_ticks >= 200 -> relative (max - min) / mean > 2 % is a violation; < 200 ticks -> absolute > 2 requests
Phase 4 additions (fixed before the campaign ended; no label is corrected with them):
  module paths of every valid run (loaded_modules) must lie in revision_profiling/code/
  time anomalies: GPU clock / temperature / load by hour; all-GPU candidate of every family listed in execution order with
    F, F / Lambda_sched and realization; flagged only if F / Lambda < 0.95 or realization < 0.95 (marked "확인", not called drift)
  side-by-side with Phase 2 3x (3-repeat mean) for the all-GPU candidates: F, generated, realization, D_macro at 2.5x, 3x, 3.5x;
    non-monotonic F (2.5x > 3x or 3.5x < 3x) is reported with the request-generation state and the GPU log of that run, not judged an error
"""
import sys

sys.dont_write_bytecode = True

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent
ROOT = IROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "code"))
import intermediate_plan as P  # noqa: E402
import recompute_scores as RS  # noqa: E402  (original)

RES = IROOT / "results"
OUT = RES / "integrity"
LOGS = IROOT / "logs"
ALPHA = 0.3
TOL = 1e-9


def md(df, floatfmt=4):
    cols = list(df.columns)
    L = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    J = pd.DataFrame(P.jobs())
    valid, invalid = RS.load_runs([RES])
    cand, inst, wl = RS.build_tables(valid)
    got = {(r["meta"]["family"], float(r["meta"]["multiplier"]), r["combination"], r["repeat"]) for r in valid}
    J["valid"] = [(f, float(m), c, r) in got for f, m, c, r in zip(J["family"], J["mult"], J["combination"], J["rep"])]
    unexpected = got - set(zip(J["family"], J["mult"].astype(float), J["combination"], J["rep"]))
    ex = pd.read_csv(LOGS / "exec_order.csv")
    att = ex.groupby(["family", "mult", "combination", "rep"]).agg(attempts=("attempt", "max")).reset_index()
    J = J.merge(att, on=["family", "mult", "combination", "rep"], how="left")
    fr = J.groupby(["family", "N", "mult"]).agg(expected=("valid", "size"), valid=("valid", "sum"), retried_runs=("attempts", lambda s: int((s > 1).sum())),
                                                 extra_attempts=("attempts", lambda s: int((s.fillna(1) - 1).sum()))).reset_index()
    fr["missing"] = fr["expected"] - fr["valid"]
    fr.to_csv(OUT / "runs_by_family_rate.csv", index=False)
    rej = sorted(str(p.relative_to(RES)) for p in RES.rglob("*.rejected.*.json"))
    bad_attempts = ex[ex["status"] != "ok"]
    bad_attempts.to_csv(OUT / "non_ok_attempts.csv", index=False)
    failed = pd.read_csv(LOGS / "failed_runs.csv") if (LOGS / "failed_runs.csv").exists() else pd.DataFrame()

    # ---- workload completeness (100 % of the feasible candidates)
    comp = []
    for fi, key, models in P.families():
        for m in P.MULTS:
            sub = J[(J["family"] == key) & (J["mult"] == m)]
            have = set(sub.loc[sub["valid"], "combination"])
            comp.append({"workload": f"{key}_{P.tag(m)}", "family": key, "N": len(models), "mult": m, "candidates": 2 ** len(models),
                         "candidates_measured": len(have), "complete": len(have) == 2 ** len(models),
                         "missing": ";".join(sorted(set(sub["combination"]) - have))})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "workload_completeness.csv", index=False)
    excluded = comp[~comp["complete"]]

    # ---- module paths
    bad_mod = []
    for r in valid:
        mods = r["loaded_modules"]
        for name, path in (mods.items() if isinstance(mods, dict) else []):
            if not str(path).startswith("revision_profiling/code/"):
                bad_mod.append({"run": r["run_key"], "module": name, "path": path})
    sched_ok = all(Path(r["meta"].get("schedule_path", r["run_key"].split("|")[0])).name.startswith(r["meta"]["family"]) for r in valid)

    # ---- conservation / trace / realization
    cons, cons_gen, trace, real, miss, snapflag = Counter(), Counter(), Counter(), [], [], 0
    for r in valid:
        for i in r["instances"]:
            w = i["window"]
            cons[w["conservation_flow"]] += 1
            cons_gen[w.get("conservation_gen", "n/a")] += 1
            trace[w["trace_check"]] += 1
            real.append(w["realization"])
            miss.append(w["missed_ticks"] / w["scheduled_ticks"])
            snapflag += bool(w["snapshot_flag"])
    real, miss = np.array(real), np.array(miss)
    run_minreal = np.array([min(i["window"]["realization"] for i in r["instances"]) for r in valid])
    inv_rows = [{"file": str(Path(r["_file"]).relative_to(RES)), "reasons": ";".join(r["invalid_reasons"])} for r in invalid]
    # scheduled ticks vs rate x window (tick rounding)
    tick_dev = []
    for r in valid:
        for i in r["instances"]:
            tick_dev.append(i["window"]["scheduled_ticks"] - i["rate"] * 10.0)
    tick_dev = np.array(tick_dev)

    # ---- generated-count difference per workload (Phase 2 rule)
    gm = []
    for (family, mult), combos in sorted(wl.items()):
        n = len(family.split("+"))
        rel_max, abs_max, rel_all, viol, detail = 0.0, 0, 0.0, 0, []
        runs = [c[0] for c in combos.values() if 0 in c]
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
                detail.append({"workload": f"{family}_x{mult:g}", "N": n, "mult": mult, "instance": lo["instances"][ii]["name"], "scheduled_ticks": ticks,
                               "gen_min": int(g.min()), "gen_max": int(g.max()), "rel_diff": round(rel, 5), "abs_diff": ab,
                               "placement_of_min": lo["placement"], "n_cpu_of_min": lo["placement"].count("C"),
                               "device_of_instance_in_min": lo["instances"][ii]["execution"]})
        gm.append({"workload": f"{family}_x{mult:g}", "family": family, "N": n, "mult": mult, "gen_mismatch_max": rel_max, "gen_mismatch_abs_max": abs_max,
                   "gen_mismatch_rel_all": rel_all, "violations": viol, "gen_violation": viol > 0, "_detail": detail})
    gmd = pd.DataFrame([d for g in gm for d in g["_detail"]])
    gm = pd.DataFrame(gm).drop(columns="_detail")
    gm.to_csv(OUT / "gen_mismatch.csv", index=False)
    gmd.to_csv(OUT / "gen_mismatch_violations.csv", index=False)

    # ---- time course: GPU state, run time, all-GPU candidates in execution order
    ex_ok = ex[ex["status"] == "ok"].copy()
    t_first = ex["start_epoch"].min()
    ex_ok["hour"] = (ex_ok["start_epoch"] - t_first) / 3600
    gpu = pd.read_csv(LOGS / "gpu_1min.csv")
    gpu["hour"] = ((gpu["epoch"] - t_first) / 3600).astype(int)
    gh = gpu.groupby("hour").agg(samples=("epoch", "size"), temp_mean=("gpu_temp_c", "mean"), temp_max=("gpu_temp_c", "max"),
                                 clock_mean=("gpu_clock_mhz", "mean"), clock_min=("gpu_clock_mhz", "min"), power_mean=("gpu_power_w", "mean"),
                                 load1_mean=("load1", "mean")).reset_index()
    gh.to_csv(OUT / "gpu_by_hour.csv", index=False)
    blocks = ex.groupby("mult").agg(attempts=("attempt", "size"), hours=("seconds", lambda s: s.sum() / 3600), sec_p50=("seconds", "median")).reset_index()
    sess = (LOGS / "campaign_sessions.log").read_text().strip().splitlines()
    wall_h = (ex["end_epoch"].max() - ex["start_epoch"].min()) / 3600
    tn = ex_ok.groupby("N")["seconds"].agg(runs="size", p50="median", p95=lambda s: s.quantile(0.95)).reset_index()
    tn["formula"] = [P.est_sec(n) for n in tn["N"]]
    tn.to_csv(OUT / "run_seconds_by_N.csv", index=False)
    ag = cand[cand["placement"].str.count("C") == 0][["family", "N", "multiplier", "combination", "F_mean", "Lambda_sched", "D_macro_mean", "generated_mean",
                                                       "min_realization", "max_missed_ratio"]].copy()
    ag = ag.merge(ex_ok[["family", "mult", "combination", "exec_order", "start_ts", "start_epoch", "hour"]].rename(columns={"mult": "multiplier"}),
                  on=["family", "multiplier", "combination"], how="left")
    ag["F_over_Lambda"] = ag["F_mean"] / ag["Lambda_sched"]

    def gpu_at(ep):
        s = gpu.iloc[(gpu["epoch"] - ep).abs().argsort()[:1]]
        return float(s["gpu_clock_mhz"].iloc[0]), float(s["gpu_temp_c"].iloc[0])
    ag["gpu_clock_near"], ag["gpu_temp_near"] = zip(*[gpu_at(e) for e in ag["start_epoch"]])
    ag["flag"] = np.where((ag["F_over_Lambda"] < 0.95) | (ag["min_realization"] < 0.95), "확인", "")
    ag = ag.sort_values("start_epoch")
    ag.to_csv(OUT / "all_gpu_by_time.csv", index=False)

    # ---- side by side with Phase 2 3x (all-GPU candidates)
    p2 = pd.read_csv(ROOT / "dataset" / "labels_candidates_all.csv")
    p2 = p2[(p2["multiplier"] == 3) & (p2["placement"].str.count("C") == 0) & (p2["N"] >= 3)]
    side = []
    for fam in sorted(ag["family"].unique()):
        row = {"family": fam, "N": int(ag.loc[ag["family"] == fam, "N"].iloc[0])}
        for lab_, src in (("2.5x (1 run)", ag[(ag["family"] == fam) & (ag["multiplier"] == 2.5)]), ("3x (Phase 2, 3-run mean)", p2[p2["family"] == fam]),
                          ("3.5x (1 run)", ag[(ag["family"] == fam) & (ag["multiplier"] == 3.5)])):
            if len(src):
                s = src.iloc[0]
                row[f"F {lab_}"] = float(s["F_mean"])
                row[f"generated/10s {lab_}"] = float(s["generated_mean"]) / 10.0
                row[f"min realization {lab_}"] = float(s["min_realization"])
                row[f"D_macro {lab_}"] = float(s["D_macro_mean"])
        f25, f3, f35 = row.get("F 2.5x (1 run)"), row.get("F 3x (Phase 2, 3-run mean)"), row.get("F 3.5x (1 run)")
        nm = []
        if f25 is not None and f3 is not None and f25 > f3:
            nm.append("F(2.5x) > F(3x)")
        if f35 is not None and f3 is not None and f35 < f3:
            nm.append("F(3.5x) < F(3x)")
        row["non_monotonic"] = "; ".join(nm)
        side.append(row)
    side = pd.DataFrame(side)
    side.to_csv(OUT / "all_gpu_vs_phase2_x3.csv", index=False)
    nonmono = side[side["non_monotonic"] != ""]
    nm_detail = ag[ag["family"].isin(nonmono["family"]) & ag["multiplier"].isin([2.5, 3.5])][
        ["family", "multiplier", "exec_order", "start_ts", "F_mean", "F_over_Lambda", "generated_mean", "min_realization", "max_missed_ratio", "gpu_clock_near", "gpu_temp_near"]]

    # ---- label statistics
    stat = []
    for (fam, mult), sub in cand.groupby(["family", "multiplier"]):
        allgpu = sub["comb_num"].max()
        s = sub["S_macro_a0.3"].values
        stat.append({"family": fam, "N": int(sub["N"].iloc[0]), "mult": mult, "F_max": float(sub["F_max"].iloc[0]), "Lambda": float(sub["Lambda_sched"].iloc[0]),
                     "D_macro_all_gpu": float(sub.loc[sub["comb_num"] == allgpu, "D_macro_mean"].iloc[0]),
                     "T_all_gpu": float(sub.loc[sub["comb_num"] == allgpu, "throughput_norm"].iloc[0]),
                     "all_gpu_is_oracle_macro": bool(sub.loc[sub["comb_num"] == allgpu, "S_macro_a0.3"].iloc[0] >= s.max() - TOL),
                     "margin_oracle_to_2nd": float(np.sort(s)[::-1][0] - np.sort(s)[::-1][1])})
    stat = pd.DataFrame(stat)
    stat.to_csv(OUT / "label_stats_by_workload.csv", index=False)

    # ---- report
    L = ["# Campaign 무결성 점검 — Phase 4 중간 rate (1.5×, 2.5×, 3.5×)", "",
         "`intermediate/code_copy/integrity_intermediate.py`가 생성(원본 `code/integrity.py`의 규칙). 상세 표는 `results/integrity/`.", "",
         "## 1. Run 수·시간", "",
         f"- 계획 {len(J)} runs, valid {int(J['valid'].sum())}, 누락 {int((~J['valid']).sum())}. 계획에 없는 결과 파일 {len(unexpected)}개.",
         f"- 실행 시도 {len(ex)}회. 재시도한 run {int((att['attempts'] > 1).sum())}개, ok가 아닌 시도 {len(bad_attempts)}회, `*.rejected.*` 파일 {len(rej)}개, "
         f"`logs/failed_runs.csv` {len(failed)}행. 최종 invalid로 남은 결과 파일 {len(invalid)}개.",
         f"- Run 시간 합 {ex['seconds'].sum() / 3600:.2f} h, 첫 시작–마지막 종료 {wall_h:.2f} h. 세션 기록: " + " / ".join(sess), "",
         md(blocks), "", "N별 run 시간:", "", md(tn), "",
         "## 2. Workload 완전성 (feasible 후보 100%)", "",
         f"- 36 workloads(12 families × 3 rates) 중 완전한 것 {int(comp['complete'].sum())}개.",
         f"- **평가에서 제외되는 workload: {len(excluded)}개.**" + ("" if not len(excluded) else " " + "; ".join(excluded["workload"])), ""]
    if len(excluded):
        L += [md(excluded[["workload", "N", "mult", "candidates", "candidates_measured", "missing"]]), ""]
    L += ["## 3. Runtime 동일성", "",
          f"- 모든 valid run의 `loaded_modules`가 `revision_profiling/code/` 안: {'예' if not bad_mod else '아니오 (' + str(len(bad_mod)) + '건)'}. "
          "Runner가 시작 시 runtime 파일 10개가 git HEAD와 같음을 확인했다(`logs/runtime_sha256.json`). Executor는 원본 그대로 호출했다.",
          f"- 결과의 mode가 모두 open_loop, β = 0.1: {all(r['mode'] == 'open_loop' and abs(r['meta']['beta'] - 0.1) < 1e-12 for r in valid)}.", "",
          "## 4. Conservation·trace·도착", "",
          f"- instance-run {sum(cons.values())}개: conservation_flow {dict(cons)}, conservation_gen {dict(cons_gen)}, trace_check {dict(trace)}, snapshot flag {snapflag}개.",
          f"- 실현율(generated / scheduled): 최소 {real.min():.4f}, 1% 분위 {np.percentile(real, 1):.4f}, 5% 분위 {np.percentile(real, 5):.4f}, 중앙값 {np.median(real):.4f}. "
          f"0.95 미만 instance {int((real < 0.95).sum())}개, 그런 instance가 있는 run {int((run_minreal < 0.95).sum())}개.",
          f"- missed_ticks 비율: 최대 {miss.max():.4f}, 99% 분위 {np.percentile(miss, 99):.4f}, 0보다 큰 instance {int((miss > 0).sum())}개, 5% 초과 {int((miss > 0.05).sum())}개.",
          f"- scheduled_ticks − rate×10 s: 최소 {tick_dev.min():.2f}, 최대 {tick_dev.max():.2f}(tick 격자 반올림; Phase 2와 같은 규칙).", "",
          "## 5. Generated 차이 (품질 지표, Phase 2 규칙: ≥200 tick 상대 2%, <200 tick 절대 2건)", "",
          f"- 위반 workload {int(gm['gen_violation'].sum())} / {len(gm)}.",
          f"- `gen_mismatch_max`(≥200 tick 상대 차) 중앙값 {gm['gen_mismatch_max'].median():.4f}, 최대 {gm['gen_mismatch_max'].max():.4f}; "
          f"`gen_mismatch_abs_max`(<200 tick 절대 차) 최대 {int(gm['gen_mismatch_abs_max'].max())}.", ""]
    if len(gmd):
        L += ["위반 workload:", "", md(gm[gm["gen_violation"]][["workload", "N", "mult", "gen_mismatch_max", "gen_mismatch_abs_max", "violations"]]), ""]
    L += ["## 6. 시간 경과 이상 징후 점검 (drift 검증 아님)", "",
          "GPU 상태(1분 표본, 시간대별):", "", md(gh), "",
          "각 family의 전부-GPU 후보를 실행 시각 순으로(rate가 다른 run의 F 차이는 부하 차이일 수 있어 drift 근거로 쓰지 않는다). "
          "`flag` = F/Λ < 0.95 또는 실현율 < 0.95인 run(확인 대상):", "",
          md(ag[["exec_order", "start_ts", "family", "N", "multiplier", "F_mean", "F_over_Lambda", "D_macro_mean", "min_realization", "gpu_clock_near", "gpu_temp_near", "flag"]]), "",
          f"표시된 run: {int((ag['flag'] != '').sum())}개.", "",
          "## 7. Phase 2 3×와 나란히 (전부-GPU 후보)", "",
          "중간 rate는 **1회 측정(rep0)**, 3×는 **Phase 2의 3회 반복 평균**이다. 비단조는 측정 오류로 판정하지 않는다.", "",
          md(side), ""]
    if len(nonmono):
        L += [f"비단조 family {len(nonmono)}개. 해당 run의 요청 생성 상태와 GPU 로그:", "", md(nm_detail), ""]
    else:
        L += ["비단조(F(2.5×) > F(3×) 또는 F(3.5×) < F(3×))인 family 없음.", ""]
    L += ["## 8. Label 통계", "", md(stat), ""]
    if inv_rows:
        L += ["## 9. Invalid로 남은 결과", "", md(pd.DataFrame(inv_rows)), ""]
    if bad_mod:
        L += ["## 10. 모듈 경로 위반", "", md(pd.DataFrame(bad_mod)), ""]
    (RES / "INTEGRITY.md").write_text("\n".join(L))
    summ = {"planned": len(J), "valid": int(J["valid"].sum()), "missing": int((~J["valid"]).sum()), "attempts": len(ex), "rejected_files": len(rej),
            "failed_runs": len(failed), "excluded_workloads": list(excluded["workload"]), "complete_workloads": int(comp["complete"].sum()),
            "conservation_flow": {str(k): v for k, v in cons.items()}, "trace": dict(trace), "module_path_violations": len(bad_mod),
            "realization_min": float(real.min()), "gen_violation_workloads": list(gm[gm["gen_violation"]]["workload"]),
            "run_hours": float(ex["seconds"].sum() / 3600), "wall_hours": float(wall_h), "flagged_all_gpu_runs": int((ag["flag"] != "").sum()),
            "non_monotonic_families": list(nonmono["family"]), "schedule_names_ok": bool(sched_ok)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
