"""Phase 4b integrity checks for rep1, rep2 (with rep0 of Phase 4).  ->  intermediate/logs/INTEGRITY_REPEATS.md + intermediate/logs/integrity_repeats/*.csv
New file; rules from Phase 4 integrity_intermediate.py / Phase 2 code/integrity.py (changes in ../CHANGES_REPEATS.md).

  python3 -B revision_profiling/intermediate/code_copy_repeats/integrity_repeats.py

Rules
  completeness       per repeat, every (workload, rep) needs all 2^N candidates valid. A workload with a missing (workload, rep) is EXCLUDED from the
                     3-repeat set (no correction, no substitution, no 1-/2-repeat means) -> logs/integrity_repeats/exclude_workloads.txt
  conservation_flow, trace_check, realization = generated / scheduled, missed_ticks / scheduled_ticks — per repeat
  generated-count difference (quality indicator, Phase 2 rule): per (workload, rep, instance) over the candidates;
                     >= 200 ticks: relative (max - min) / mean > 2 %; < 200 ticks: absolute > 2 requests
  repeat deviation   per candidate SE = std(ddof=1) / sqrt(3) of F, D_macro and S_r = F_r / F_max - 0.3 D_macro,r (F_max = common max of the 3-repeat
                     means; Phase 2 definition, PRECHECK_3REP 0-1); median / p90 / max per rate, next to Phase 2 3x
  per-repeat oracle  own normalisation per repeat (Phase 2 integrity.py), oracle SET with tol 1e-9; reported: workloads with a candidate optimal
                     in all three repeats (common set non-empty), and separately whether the representative (lowest) ids are equal
  time course        GPU clock / temperature / load by hour of the repeat campaign; the all-GPU candidate of every family at every rate listed by
                     repeat and start time; mark "급변" when |F_r - F_bar| / F_bar > 5 % (display rule only, fixed before the campaign ended;
                     it is not called drift and no label is corrected)
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
import labels_3rep as L3  # noqa: E402
import repeats_plan as PR  # noqa: E402

P4 = PR.P4
RS = L3.RS
LOGS = IROOT / "logs"
OUT = LOGS / "integrity_repeats"
ALPHA = 0.3
TOL = 1e-9
REPS = (0, 1, 2)
JUMP = 0.05


def md(df, floatfmt=4):
    cols = list(df.columns)
    L = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(L)


def rate_of(m):
    return P4.tag(float(m))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- run lists: rep0 (Phase 4 plan) + rep1, rep2
    J0 = pd.DataFrame(P4.jobs())
    J12 = pd.DataFrame(PR.jobs())
    J = pd.concat([J0, J12], ignore_index=True)
    valid, invalid, cand, inst, wl = L3.build(L3.REP_DIRS)
    got = {(r["meta"]["family"], float(r["meta"]["multiplier"]), r["combination"], r["repeat"]) for r in valid}
    J["valid"] = [(f, float(m), c, r) in got for f, m, c, r in zip(J["family"], J["mult"], J["combination"], J["rep"])]
    unexpected = got - set(zip(J["family"], J["mult"].astype(float), J["combination"], J["rep"]))
    ex = {0: pd.read_csv(LOGS / "exec_order.csv")}
    for r in (1, 2):
        p = LOGS / f"exec_order_rep{r}.csv"
        ex[r] = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=ex[0].columns)
    exa = pd.concat([ex[r].assign(rep=r) for r in REPS], ignore_index=True)
    failed = pd.read_csv(LOGS / "failed_runs.csv") if (LOGS / "failed_runs.csv").exists() else pd.DataFrame(columns=["rep"])
    rej = {r: sorted(str(p.relative_to(IROOT)) for p in L3.REP_DIRS[r].rglob("*.rejected.*.json")) for r in REPS}
    sess = (LOGS / "campaign_sessions_repeats.log").read_text().strip().splitlines() if (LOGS / "campaign_sessions_repeats.log").exists() else []
    runs = []
    for r in REPS:
        Jr, e = J[J["rep"] == r], ex[r]
        att = e.groupby(["family", "mult", "combination"])["attempt"].max() if len(e) else pd.Series(dtype=int)
        runs.append({"rep": r, "source": "Phase 4" if r == 0 else "Phase 4b", "planned": len(Jr), "valid": int(Jr["valid"].sum()),
                     "missing": int((~Jr["valid"]).sum()), "attempts": len(e), "retried_runs": int((att > 1).sum()),
                     "non_ok_attempts": int((e["status"] != "ok").sum()) if len(e) else 0, "rejected_files": len(rej[r]),
                     "failed_runs_csv": int((failed["rep"] == r).sum()) if len(failed) else 0,
                     "invalid_left": sum(1 for x in invalid if x["repeat"] == r),
                     "run_hours": float(e["seconds"].sum() / 3600) if len(e) else 0.0,
                     "wall_hours": float((e["end_epoch"].max() - e["start_epoch"].min()) / 3600) if len(e) else 0.0,
                     "first_start": e["start_ts"].min() if len(e) else "", "last_end": e["end_ts"].max() if len(e) else ""})
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "runs_by_rep.csv", index=False)
    exa[exa["status"] != "ok"].to_csv(OUT / "non_ok_attempts.csv", index=False)

    # ---- completeness per (workload, rep)
    comp = []
    for fi, key, models in P4.families():
        for m in P4.MULTS:
            row = {"workload": f"{key}_{P4.tag(m)}", "family": key, "N": len(models), "mult": m, "candidates": 2 ** len(models)}
            ok = True
            for r in REPS:
                sub = J[(J["family"] == key) & (J["mult"] == m) & (J["rep"] == r)]
                have = set(sub.loc[sub["valid"], "combination"])
                row[f"valid_rep{r}"] = len(have)
                row[f"missing_rep{r}"] = ";".join(sorted(set(sub["combination"]) - have))
                ok &= len(have) == 2 ** len(models)
            row["complete_all_reps"] = ok
            comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "workload_completeness_by_rep.csv", index=False)
    excluded = comp[~comp["complete_all_reps"]]
    (OUT / "exclude_workloads.txt").write_text(";".join(excluded["workload"]))

    # ---- module paths, conservation / trace / realization per repeat
    per_rep, bad_mod = [], []
    for r in REPS:
        vr = [x for x in valid if x["repeat"] == r]
        cons, trace, real, miss, snap, tick = Counter(), Counter(), [], [], 0, []
        for x in vr:
            mods = x["loaded_modules"]
            for name, path in (mods.items() if isinstance(mods, dict) else []):
                if not str(path).startswith("revision_profiling/code/"):
                    bad_mod.append({"rep": r, "run": x["run_key"], "module": name, "path": path})
            for i in x["instances"]:
                w = i["window"]
                cons[w["conservation_flow"]] += 1
                trace[w["trace_check"]] += 1
                real.append(w["realization"])
                miss.append(w["missed_ticks"] / w["scheduled_ticks"])
                snap += bool(w["snapshot_flag"])
                tick.append(w["scheduled_ticks"] - i["rate"] * 10.0)
        real, miss, tick = np.array(real), np.array(miss), np.array(tick)
        per_rep.append({"rep": r, "runs": len(vr), "instance_runs": int(sum(cons.values())), "conservation_flow": dict(cons), "trace_check": dict(trace),
                        "snapshot_flag": snap, "open_loop_beta_0.1": all(x["mode"] == "open_loop" and abs(x["meta"]["beta"] - 0.1) < 1e-12 for x in vr),
                        "realization_min": float(real.min()) if len(real) else np.nan, "realization_p1": float(np.percentile(real, 1)) if len(real) else np.nan,
                        "realization_median": float(np.median(real)) if len(real) else np.nan, "instances_realization_lt_0.95": int((real < 0.95).sum()),
                        "missed_ratio_max": float(miss.max()) if len(miss) else np.nan, "missed_ratio_p99": float(np.percentile(miss, 99)) if len(miss) else np.nan,
                        "instances_missed_gt_0": int((miss > 0).sum()), "instances_missed_gt_5pct": int((miss > 0.05).sum()),
                        "tick_minus_rate_x10_min": float(tick.min()) if len(tick) else np.nan, "tick_minus_rate_x10_max": float(tick.max()) if len(tick) else np.nan})
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(OUT / "arrival_by_rep.csv", index=False)

    # ---- generated-count difference per (workload, rep) — Phase 2 rule
    gm, gmd = [], []
    for (family, mult), combos in sorted(wl.items()):
        n = len(family.split("+"))
        for r in REPS:
            rr = [c[r] for c in combos.values() if r in c]
            if len(rr) < 2:
                continue
            rel_max, abs_max, viol = 0.0, 0, 0
            for ii in range(n):
                g = np.array([x["instances"][ii]["window"]["generated"] for x in rr])
                ticks = rr[0]["instances"][ii]["window"]["scheduled_ticks"]
                rel = float((g.max() - g.min()) / g.mean()) if g.mean() > 0 else 0.0
                ab = int(g.max() - g.min())
                if ticks >= 200:
                    rel_max = max(rel_max, rel)
                    v = rel > 0.02
                else:
                    abs_max = max(abs_max, ab)
                    v = ab > 2
                if v:
                    viol += 1
                    lo = rr[int(g.argmin())]
                    gmd.append({"workload": f"{family}_x{mult:g}", "rep": r, "N": n, "instance": lo["instances"][ii]["name"], "scheduled_ticks": ticks,
                                "gen_min": int(g.min()), "gen_max": int(g.max()), "rel_diff": round(rel, 5), "placement_of_min": lo["placement"]})
            gm.append({"workload": f"{family}_x{mult:g}", "rep": r, "N": n, "mult": mult, "gen_mismatch_max": rel_max, "gen_mismatch_abs_max": abs_max,
                       "violations": viol, "gen_violation": viol > 0})
    gm, gmd = pd.DataFrame(gm), pd.DataFrame(gmd)
    gm.to_csv(OUT / "gen_mismatch_by_rep.csv", index=False)
    gmd.to_csv(OUT / "gen_mismatch_violations_by_rep.csv", index=False)
    gsum = gm.groupby("rep").agg(workloads=("workload", "size"), violation_workloads=("gen_violation", "sum"), rel_median=("gen_mismatch_max", "median"),
                                 rel_max=("gen_mismatch_max", "max"), abs_max=("gen_mismatch_abs_max", "max")).reset_index()

    # ---- repeat deviation (3-repeat candidates of complete workloads)
    c3 = cand[(cand["n_reps"] == 3) & ~cand["workload"].isin(excluded["workload"])].copy()
    c3["rate"] = [rate_of(m) for m in c3["multiplier"]]
    dev = pd.concat([L3.se_stats(s).assign(rate=rt) for rt, s in c3.groupby("rate")], ignore_index=True)
    p2v = L3.build(L3.phase2_x3_dirs())[2]
    p2v = p2v[p2v["N"] >= 3]
    dev = pd.concat([dev, L3.se_stats(p2v).assign(rate="x3 (Phase 2)")], ignore_index=True)[["rate", "metric", "candidates", "median", "p90", "max"]]
    dev.to_csv(OUT / "repeat_variation_by_rate.csv", index=False)
    c3[["workload", "family", "N", "multiplier", "combination", "placement", "F_mean", "F_se", "D_macro_mean", "D_macro_se", "S_macro_a0.3", "S_macro_a0.3_se"]
       + [f"{c}_rep{k}" for c in ("F", "D_macro", "S_macro_a0.3") for k in REPS]].to_csv(OUT / "repeat_variation_candidates.csv", index=False)

    # ---- per-repeat oracles
    orc = L3.rep_oracles(c3)
    orc["rate"] = [rate_of(m) for m in orc["multiplier"]]
    orc.to_csv(OUT / "oracle_by_rep.csv", index=False)
    osum = orc.groupby("rate").agg(workloads=("workload", "size"), common_optimal_exists=("common_optimal_exists", "sum"),
                                   representative_ids_equal=("representative_ids_equal", "sum"),
                                   mean_label_oracle_in_common=("mean_label_oracle_in_common", "sum")).reset_index()

    # ---- time course
    gpu = pd.read_csv(LOGS / "gpu_1min_repeats.csv") if (LOGS / "gpu_1min_repeats.csv").exists() else pd.DataFrame()
    e12 = exa[exa["rep"].isin([1, 2])]
    t_first = e12["start_epoch"].min() if len(e12) else 0
    if len(gpu):
        gpu["hour"] = ((gpu["epoch"] - t_first) / 3600).astype(int)
        gh = gpu.groupby("hour").agg(samples=("epoch", "size"), temp_mean=("gpu_temp_c", "mean"), temp_max=("gpu_temp_c", "max"),
                                     clock_mean=("gpu_clock_mhz", "mean"), clock_min=("gpu_clock_mhz", "min"), power_mean=("gpu_power_w", "mean"),
                                     load1_mean=("load1", "mean")).reset_index()
    else:
        gh = pd.DataFrame()
    gh.to_csv(OUT / "gpu_by_hour_repeats.csv", index=False)
    tn = e12[e12["status"] == "ok"].groupby(["rep", "N"])["seconds"].agg(runs="size", p50="median", p95=lambda s: s.quantile(0.95)).reset_index()
    tn["formula"] = [P4.est_sec(n) for n in tn["N"]]
    tn.to_csv(OUT / "run_seconds_by_rep_N.csv", index=False)
    ag = c3[c3["placement"].str.count("C") == 0]
    ok_runs = exa[exa["status"] == "ok"].copy()
    ok_runs["mult"] = ok_runs["mult"].astype(float)
    rows = []
    for _, a in ag.iterrows():
        for r in REPS:
            e = ok_runs[(ok_runs["rep"] == r) & (ok_runs["family"] == a["family"]) & (ok_runs["mult"] == float(a["multiplier"])) & (ok_runs["combination"] == a["combination"])]
            F = float(a[f"F_rep{r}"])
            rows.append({"family": a["family"], "N": int(a["N"]), "rate": rate_of(a["multiplier"]), "rep": r,
                         "start_ts": e["start_ts"].iloc[-1] if len(e) else "", "start_epoch": float(e["start_epoch"].iloc[-1]) if len(e) else np.nan,
                         "F_r": F, "F_bar": float(a["F_mean"]), "dev_from_mean": F / float(a["F_mean"]) - 1, "D_macro_r": float(a[f"D_macro_rep{r}"])})
    agt = pd.DataFrame(rows).sort_values(["family", "rate", "rep"])
    agt["flag"] = np.where(agt["dev_from_mean"].abs() > JUMP, "급변", "")
    agt.to_csv(OUT / "all_gpu_by_rep.csv", index=False)

    # ---- report
    Lr = ["# Campaign 무결성 점검 — Phase 4b 반복 측정 (rep1, rep2; rep0 = Phase 4)", "",
          "`intermediate/code_copy_repeats/integrity_repeats.py`가 생성. 상세 표는 `logs/integrity_repeats/`. "
          "(Phase 4 rep0 폴더 `results/`는 보호 대상이라 이 보고서를 `logs/`에 둔다.)", "",
          "## 1. Run 수·시간", "", md(runs), "",
          f"- 계획에 없는 결과 파일 {len(unexpected)}개. 세션 기록: " + " / ".join(sess), "",
          "N별 run 시간(rep1, rep2):", "", md(tn), "",
          "## 2. 완전성 ((workload, rep)별 feasible 후보 100%)", "",
          f"- 36 workloads 중 세 반복 모두 완전한 것 {int(comp['complete_all_reps'].sum())}개.",
          f"- **3회 평균 집합에서 제외되는 workload: {len(excluded)}개.**" + ("" if not len(excluded) else " " + "; ".join(excluded["workload"])), ""]
    if len(excluded):
        Lr += [md(excluded[["workload", "N", "mult"] + [f"valid_rep{r}" for r in REPS] + [f"missing_rep{r}" for r in REPS]]), ""]
    Lr += ["## 3. Runtime·conservation·trace·도착 (rep별)", "",
           f"- 모든 valid run의 `loaded_modules`가 `revision_profiling/code/` 안: {'예' if not bad_mod else '아니오 (' + str(len(bad_mod)) + '건)'}. "
           "Runner가 각 블록 시작 시 runtime 파일 10개의 sha256이 Phase 4 `logs/runtime_sha256.json`과 같음을 확인했다(`logs/runtime_sha256_rep{1,2}.json`).", "",
           md(per_rep.astype({"conservation_flow": str, "trace_check": str})), "",
           "## 4. Generated 차이 (품질 지표, Phase 2 규칙: ≥200 tick 상대 2%, <200 tick 절대 2건) — rep별", "", md(gsum), ""]
    if len(gmd):
        Lr += ["위반 (workload, rep):", "", md(gm[gm["gen_violation"]][["workload", "rep", "N", "gen_mismatch_max", "gen_mismatch_abs_max", "violations"]]), ""]
    Lr += ["## 5. 반복 편차 (후보별 3회 SE = 표본표준편차(ddof=1)/√3)", "",
           "`S_r = F_r / F_max − 0.3·D_macro,r`, `F_max` = 3회 평균의 공통 최대값(Phase 2 정의, PRECHECK_3REP 0-1). Phase 2 3×(N≥3, 480 후보)와 나란히.", "",
           md(dev), "",
           "## 6. 반복별 oracle 일치", "",
           "각 반복의 자체 정규화(`F_r / max F_r − 0.3·D_macro,r`, Phase 2 integrity.py 규칙)와 동점 허용오차 1e-9로 oracle 후보 **집합**을 구했다. "
           "`common_optimal_exists` = 세 반복 모두에서 최적인 후보가 있는 workload 수, `representative_ids_equal` = 대표 ID(가장 작은 combination 번호)가 셋 다 같은 수.", "",
           md(osum), "", "Workload별(`oracle_by_rep.csv`):", "",
           md(orc[["rate", "family", "N", "oracle_rep0", "n_oracle_rep0", "oracle_rep1", "n_oracle_rep1", "oracle_rep2", "n_oracle_rep2",
                   "common_optimal_exists", "representative_ids_equal", "oracle_mean_label", "n_oracle_mean_label"]]), "",
           "## 7. 시간 경과 이상 징후 점검 (drift 검증 아님)", "", "GPU 상태(1분 표본, Phase 4b 시작부터 시간대별):", "", md(gh) if len(gh) else "없음", "",
           f"전부-GPU 후보의 반복별 F(rep0 = Phase 4, 2026-10-19). `flag` = |F_r − F̄| / F̄ > {JUMP:.0%}(표시 규칙만, label 보정 없음):", "",
           md(agt[["family", "N", "rate", "rep", "start_ts", "F_r", "F_bar", "dev_from_mean", "D_macro_r", "flag"]]), "",
           f"표시된 (후보, rep): {int((agt['flag'] != '').sum())}개.", ""]
    if invalid:
        Lr += ["## 8. Invalid로 남은 결과", "", md(pd.DataFrame([{"rep": x["repeat"], "file": x["_file"], "reasons": ";".join(x["invalid_reasons"])} for x in invalid])), ""]
    if bad_mod:
        Lr += ["## 9. 모듈 경로 위반", "", md(pd.DataFrame(bad_mod)), ""]
    (LOGS / "INTEGRITY_REPEATS.md").write_text("\n".join(Lr))
    summ = {"runs": runs.to_dict(orient="records"), "excluded_workloads": list(excluded["workload"]), "complete_all_reps": int(comp["complete_all_reps"].sum()),
            "module_path_violations": len(bad_mod), "gen_violations": gm[gm["gen_violation"]][["workload", "rep"]].to_dict(orient="records"),
            "oracle_summary": osum.to_dict(orient="records"), "flagged_all_gpu": int((agt["flag"] != "").sum()),
            "repeat_variation_S": dev[dev["metric"] == "S"].to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
