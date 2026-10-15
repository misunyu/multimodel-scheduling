"""Phase N2-A A3: campaign integrity for the CPU-NPU board -> revision_profiling_npu/results/INTEGRITY.md + results/integrity/*.csv

  python3 -B code/integrity_npu.py

Same items and rules as integrity.py (Phase 2 section 2.4); the generated-count rule is copied from integrity.py lines 1-12
(per workload, repeat and instance over the candidates: scheduled_ticks >= 200 -> (max - min)/mean <= 2 %, else max - min <= 2).
Board-specific: planned runs from run_plan.csv; completeness against the feasible placements (gen_schedules_npu); NPU sidecars
(Close rc, binary path per variant, driver module); residency re-checked from each run JSON; DMA32 free pages per run
(results/runs/dma32_log.csv). GPU state is replaced by DMA32 (no GPU on this board); NPU temperature is not exposed.
"""
import sys

sys.dont_write_bytecode = True

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import recompute_scores as RS
from gen_schedules_npu import FAMILIES, enumerate_feasible, read_family

ROOT = Path(__file__).resolve().parent.parent.parent / "revision_profiling_npu"
RES = ROOT / "results"
RUNS = RES / "runs"
OUT = RES / "integrity"
ALPHA, TOL, TIE = 0.3, 1e-9, 0.0002
EVAL = ("rbrsyb", "rbybys", "rsybys", "rbrsybys")


def md(df, floatfmt=4):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(ROOT / "run_plan.csv")
    plan["stem"] = [f"{Path(s).stem}__{c}__rep{r}" for s, c, r in zip(plan["schedule"], plan["combination"], plan["repeat"])]
    valid, invalid = RS.load_runs([RUNS])
    stems = {Path(r["_file"]).stem: r for r in valid}
    plan["valid"] = plan["stem"].isin(stems)
    unexpected = sorted(set(stems) - set(plan["stem"]))
    rej = sorted(p.name for p in RUNS.glob("*.rejected.*.json") if ".npu." not in p.name)
    retries = pd.read_csv(RES / "retries.csv") if (RES / "retries.csv").exists() else pd.DataFrame()
    dl = pd.read_csv(RUNS / "dma32_log.csv")
    cand, inst, wl = RS.build_tables(valid)

    # ---- per run: rc / sidecars / residency / binary mapping
    reg = json.loads((ROOT / "registry" / "files_sha256.json").read_text())
    run_rows, bin_map = [], Counter()
    for stem, r in stems.items():
        side = {json.loads(p.read_text())["instance"]: json.loads(p.read_text()) for p in RUNS.glob(stem + ".npu.*.json")}
        devs = [i["execution"] for i in r["instances"]]
        npu_inst = [i for i in r["instances"] if i["execution"] != "cpu"]
        res_ok = all(devs.count(d) <= 1 for d in ("npu0", "npu1")) and all(
            (i["model"].endswith("_npu")) == (i["execution"] != "cpu") for i in r["instances"])
        map_ok = True
        for i in npu_inst:
            s = side.get(i["name"])
            if s is None:
                map_ok = False
                continue
            ok = (s["registry_key"] == i["model"] and s["execution"] == i["execution"] and f"/{i['name']}/" in s["server"]["o_path"]
                  and s["server"]["o_path"].endswith(f"{i['name']}_neubla_p1.o") and s["server"]["npu_id"] == int(i["execution"][-1])
                  and s["server"]["module_file"].endswith("/lib/NeublaDriver.cpython-38-x86_64-linux-gnu.so"))
            map_ok &= ok
            bin_map[(i["name"], i["execution"], Path(s["server"]["o_path"]).name)] += 1
        run_rows.append({"run": stem, "valid": r["valid"], "n_npu": len(npu_inst), "sidecars": len(side),
                         "close_rc": ";".join(f"{k}:{v['close_rc']}" for k, v in sorted(side.items())),
                         "close_ok": len(side) == len(npu_inst) and all(v["close_rc"] == 0 for v in side.values()),
                         "residency_ok": res_ok, "binary_map_ok": map_ok,
                         "failed": sum(i["window"]["failed"] for i in r["instances"])})
    rr = pd.DataFrame(run_rows)
    rr.to_csv(OUT / "runs_checks.csv", index=False)
    bm = pd.DataFrame([{"instance": a, "device": b, "binary": c, "runs": n} for (a, b, c), n in sorted(bin_map.items())])
    bm.to_csv(OUT / "binary_mapping.csv", index=False)
    o_sha = {Path(k).name: v for k, v in reg["files"].items() if k.endswith(".o")}
    bm["registry_sha256"] = [o_sha.get(b, "MISSING")[:16] for b in bm["binary"]]
    bm.to_csv(OUT / "binary_mapping.csv", index=False)

    # ---- completeness
    comp = []
    for fam in FAMILIES:
        base, _ = read_family(fam)
        names = sorted(next(iter(base.values())))
        nf = len(enumerate_feasible(names))
        for m in (1, 2, 3, 4):
            sub = plan[plan["schedule"] == f"schedules/{fam}_x{m}.yaml"]
            have = sub[sub["valid"]].groupby("combination")["repeat"].nunique()
            exp_reps = 3 if (m == 3 and fam in EVAL) else 1
            comp.append({"workload": f"{fam}_x{m}", "family": fam, "N": len(names), "mult": m, "candidates": nf,
                         "candidates_measured": int(len(have)), "expected_reps": exp_reps,
                         "candidates_with_all_reps": int((have == exp_reps).sum()), "complete": len(have) == nf,
                         "all_reps_complete": int((have == exp_reps).sum()) == nf,
                         "missing": ";".join(sorted(set(sub["combination"]) - set(have.index)))})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "workload_completeness.csv", index=False)

    # ---- conservation / trace / realization
    cons_g, cons_f, trace, real, low = Counter(), Counter(), Counter(), [], []
    for r in valid:
        for i in r["instances"]:
            w = i["window"]
            cons_g[w["conservation_gen"]] += 1
            cons_f[w["conservation_flow"]] += 1
            trace[w["trace_check"]] += 1
            real.append(w["realization"])
            if w["realization"] < 0.95:
                low.append({"run": Path(r["_file"]).stem, "instance": i["name"], "device": i["execution"], "realization": w["realization"]})
    real = np.array(real)
    pd.DataFrame(low, columns=["run", "instance", "device", "realization"]).to_csv(OUT / "realization_below_095.csv", index=False)

    # ---- generated-count difference (integrity.py rule)
    gm, det = [], []
    for (family, mult), combos in sorted(wl.items()):
        reps = sorted({k for c in combos.values() for k in c})
        rel_max, abs_max, viol = 0.0, 0, 0
        for k in reps:
            runs = [c[k] for c in combos.values() if k in c]
            if len(runs) < 2:
                continue
            for ii in range(len(runs[0]["instances"])):
                g = np.array([x["instances"][ii]["window"]["generated"] for x in runs])
                ticks = runs[0]["instances"][ii]["window"]["scheduled_ticks"]
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
                    lo = runs[int(g.argmin())]
                    det.append({"workload": f"{family}_x{mult:g}", "split": "test" if mult == 3 else "train", "rep": k,
                                "instance": lo["instances"][ii]["name"], "scheduled_ticks": ticks, "gen_min": int(g.min()),
                                "gen_max": int(g.max()), "rel_diff": round(rel, 5), "abs_diff": ab, "placement_of_min": lo["placement"]})
        gm.append({"workload": f"{family}_x{mult:g}", "split": "test" if mult == 3 else "train", "gen_mismatch_max": rel_max,
                   "gen_mismatch_abs_max": abs_max, "violations": viol})
    gm = pd.DataFrame(gm)
    gm.to_csv(OUT / "gen_mismatch.csv", index=False)
    gmd = pd.DataFrame(det)
    gmd.to_csv(OUT / "gen_mismatch_violations.csv", index=False)

    # ---- repeats (3x, N >= 3 evaluation set)
    rep = cand[(cand["multiplier"] == 3) & (cand["family"].isin(EVAL))].copy()
    fr = ["F_rep0", "F_rep1", "F_rep2"]
    dr = ["D_macro_rep0", "D_macro_rep1", "D_macro_rep2"]
    rep["F_rel_range"] = (rep[fr].max(axis=1) - rep[fr].min(axis=1)) / rep["F_mean"]
    rep["D_macro_abs_range"] = rep[dr].max(axis=1) - rep[dr].min(axis=1)
    rep[["family", "combination", "placement", "n_reps", "F_mean", "F_rel_range", "D_macro_mean", "D_macro_abs_range",
         "S_macro_a0.3", "S_macro_a0.3_se"]].to_csv(OUT / "repeat_variation_candidates.csv", index=False)
    orc = []
    for fam, sub in rep.groupby("family"):
        sub = sub.sort_values("comb_num")
        s = sub["S_macro_a0.3"].values
        oset = list(sub["comb_num"].values[s >= s.max() - TOL])
        per = []
        for k in range(3):
            sk = (sub[f"F_rep{k}"] / sub[f"F_rep{k}"].max() - ALPHA * sub[f"D_macro_rep{k}"]).values
            per.append(int(sub["comb_num"].values[np.lexsort((sub["comb_num"].values, -sk))[0]]))
        srt = np.sort(s)[::-1]
        mo = min(oset)
        orc.append({"family": fam, "candidates": len(sub), "oracle": f"combination_{mo}",
                    "oracle_placement": sub.loc[sub["comb_num"] == mo, "placement"].iloc[0], "n_oracle": len(oset),
                    "S_oracle": float(s.max()), "oracle_rep0": per[0], "oracle_rep1": per[1], "oracle_rep2": per[2],
                    "reps_equal": len(set(per)) == 1, "reps_equal_to_mean_oracle": sum(p in oset for p in per),
                    "margin_1st_2nd": float(srt[0] - srt[1]), "tie_le_0.0002": bool(srt[0] - srt[1] <= TIE),
                    "oracle_S_se": float(sub.loc[sub["comb_num"] == mo, "S_macro_a0.3_se"].iloc[0])})
    orc = pd.DataFrame(orc)
    orc.to_csv(OUT / "oracle_x3_eval.csv", index=False)

    # ---- drift and run time
    drift = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ratio = (rep[f"F_rep{b}"] / rep[f"F_rep{a}"]).values
        drift.append({"compare": f"rep{b}/rep{a}", "candidates": len(ratio), "F_ratio_median": float(np.median(ratio)),
                      "F_ratio_p05": float(np.percentile(ratio, 5)), "F_ratio_p95": float(np.percentile(ratio, 95)),
                      "share_above_1": float((ratio > 1).mean())})
    drift = pd.DataFrame(drift)
    drift.to_csv(OUT / "drift_repeat_blocks.csv", index=False)
    ok = dl[dl["status"] == "ok"]
    blocks = dl.groupby("block").agg(attempts=("run", "size"), ok=("status", lambda s: int((s == "ok").sum())),
                                     hours=("sec", lambda s: s.sum() / 3600), sec_p50=("sec", "median"), sec_mean=("sec", "mean"),
                                     dma32_min=("dma32_free_after", "min"), dma32_first=("dma32_free_before", "first"),
                                     dma32_last=("dma32_free_after", "last")).reset_index()
    blocks.to_csv(OUT / "blocks.csv", index=False)
    nmap = {f"{fam}": len(sorted(next(iter(read_family(fam)[0].values())))) for fam in FAMILIES}
    ok = ok.assign(N=[nmap[r.split("_x")[0]] for r in ok["run"]])
    tn = ok.groupby("N")["sec"].agg(runs="size", p50="median", p95=lambda s: s.quantile(0.95)).reset_index()
    tn.to_csv(OUT / "run_seconds_by_N.csv", index=False)
    dq = dl.assign(chunk=(np.arange(len(dl)) // 50) * 50 + 1).groupby("chunk").agg(
        dma32_min=("dma32_free_after", "min"), dma32_mean=("dma32_free_after", "mean")).reset_index()
    dq.to_csv(OUT / "dma32_by_50_runs.csv", index=False)
    single = []
    for r in valid:
        if r["meta"]["n_instances"] == 1 and r["repeat"] == 0:
            i = r["instances"][0]
            single.append({"order": int(plan.loc[plan["stem"] == Path(r["_file"]).stem, "order"].iloc[0]), "instance": i["name"],
                           "device": i["execution"], "mult": r["meta"]["multiplier"], "rate": round(i["rate"], 3),
                           "F": r["system"]["throughput_fps"], "service_ms_p50": round(i["window"]["service_ms_p50"], 2),
                           "q_i": round(i["window"]["q_i"], 4)})
    single = pd.DataFrame(single).sort_values("order")
    single.to_csv(OUT / "single_instance_runs.csv", index=False)

    # ---- label statistics per workload (reference)
    stat = []
    for (fam, mult), sub in cand.groupby(["family", "multiplier"]):
        s = sub["S_macro_a0.3"].values
        best = sub.iloc[int(np.lexsort((sub["comb_num"].values, -s))[0])]
        allcpu = sub[[all(x.endswith(":cpu") for x in p.split(",")) for p in sub["placement"]]].iloc[0]
        stat.append({"workload": f"{fam}_x{mult:g}", "N": int(sub["N"].iloc[0]), "F_max": float(sub["F_max"].iloc[0]),
                     "oracle": best["combination"], "oracle_placement": best["placement"], "S_oracle": float(best["S_macro_a0.3"]),
                     "S_all_cpu": float(allcpu["S_macro_a0.3"]), "D_macro_all_cpu": float(allcpu["D_macro_mean"]),
                     "D_macro_min": float(sub["D_macro_mean"].min()), "D_macro_max": float(sub["D_macro_mean"].max())})
    stat = pd.DataFrame(stat)
    stat.to_csv(OUT / "label_stats_by_workload.csv", index=False)

    # ---- report
    nv = int(plan["valid"].sum())
    bad_runs = rr[~(rr["valid"] & rr["close_ok"] & rr["residency_ok"] & rr["binary_map_ok"] & (rr["failed"] == 0))]
    L = ["# Campaign 무결성 점검 (Phase N2-A A3)", "", "`code/integrity_npu.py`가 생성. 상세 표는 `results/integrity/`. 항목·규칙은 Phase 2 `integrity.py`와 같다.", "",
         "## 1. Run 수·유효성", "",
         f"- 계획 {len(plan)} runs, valid {nv}, 누락 {len(plan) - nv}. 계획에 없는 결과 파일 {len(unexpected)}개.",
         f"- 실행 시도 {len(dl)}회, ok가 아닌 시도 {int((dl['status'] != 'ok').sum())}회, 재시도 기록 {len(retries)}건, `*.rejected.*` 결과 파일 {len(rej)}개, 최종 invalid 결과 {len(invalid)}개.",
         f"- run 단위 검사(rc 0·valid·failed 0·Close rc 0·residency·변종–바이너리 매핑) 통과 {len(rr) - len(bad_runs)} / {len(rr)}.",
         "", md(blocks), ""]
    if len(bad_runs):
        L += ["검사 실패 run:", "", md(bad_runs), ""]
    if len(retries):
        L += ["재시도 기록:", "", md(retries), ""]
    L += ["## 2. Workload 완전성", "",
          f"- 44 workloads(11 families × 4 rates) 중 후보 100% valid: {int(comp['complete'].sum())}. 제외 workload: {int((~comp['complete']).sum())}.",
          f"- 3× N≥3 평가 4 workloads 중 3회 반복까지 모두 있는 것: {int(comp[(comp['mult'] == 3) & comp['family'].isin(EVAL)]['all_reps_complete'].sum())}.", ""]
    if (~comp["complete"]).any():
        L += [md(comp[~comp["complete"]]), ""]
    L += ["## 3. Conservation·trace·도착", "",
          f"- instance-run {sum(cons_f.values())}개: conservation_gen {dict(cons_g)}, conservation_flow {dict(cons_f)}, trace_check {dict(trace)}.",
          f"- 실현율(generated / scheduled): 최소 {real.min():.4f}, 1% 분위 {np.percentile(real, 1):.4f}, 중앙값 {np.median(real):.4f}. "
          f"0.95 미만 instance {len(low)}개" + ("." if not low else f" (`integrity/realization_below_095.csv`)."), "",
          "## 4. 후보 간 generated 차이", "",
          f"- 위반 workload: 학습 {int(((gm['violations'] > 0) & (gm['split'] == 'train')).sum())} / {int((gm['split'] == 'train').sum())}, "
          f"평가 {int(((gm['violations'] > 0) & (gm['split'] == 'test')).sum())} / {int((gm['split'] == 'test').sum())}.",
          f"- 상대 차(≥200 tick) 최대 {gm['gen_mismatch_max'].max():.4f}, 절대 차(<200 tick) 최대 {int(gm['gen_mismatch_abs_max'].max())}.", ""]
    if len(gmd):
        L += [md(gmd), ""]
    L += ["## 5. Residency·변종–바이너리 매핑", "",
          f"- residency 위반 run {int((~rr['residency_ok']).sum())}. 매핑 불일치 run {int((~rr['binary_map_ok']).sum())}. "
          "매핑 규칙: sidecar의 registry key = run의 key, NPU id = 배치, `.o` = `<instance>/npu_code/<instance>_neubla_p1.o`, driver = `/workspace/lib` 판.",
          "- `.o` 파일 sha256은 registry 목록(`registry/files_sha256.json`)과 campaign 전후 비교로 확인한다(REPORT_N2A).", "",
          md(bm), "",
          "## 6. 반복 편차 (3× N≥3, 60 후보 × 3회)", "",
          "| 지표 | 중앙값 | 90% 분위 | 최대 |", "|---|---|---|---|"]
    for name, col in (("F 상대 범위", "F_rel_range"), ("D_macro 절대 범위", "D_macro_abs_range"), ("S의 SE", "S_macro_a0.3_se")):
        L.append(f"| {name} | {rep[col].median():.4f} | {rep[col].quantile(0.9):.4f} | {rep[col].max():.4f} |")
    L += ["", f"- 반복별 oracle(각 반복 자체 정규화)이 세 번 모두 같은 workload: {int(orc['reps_equal'].sum())} / {len(orc)}.",
          f"- oracle 1·2위 실측 차 ≤ {TIE}: {int(orc['tie_le_0.0002'].sum())} / {len(orc)}.", "", md(orc), "",
          "## 7. 시간 drift·DMA32", "", "반복 블록 사이 F 비율(같은 후보):", "", md(drift), "",
          "DMA32 free pages (50 run 구간별):", "", md(dq), "",
          "단일 instance run(rep0, 실행 순서):", "", md(single), "",
          "Run 시간(N별, 초):", "", md(tn), "",
          "NPU 온도: driver·sysfs에 노출되지 않아 기록 없음 [확인 불가].", "",
          "## 8. Workload별 label 참고값", "", md(stat), ""]
    if invalid:
        L += ["## 9. Invalid 결과", "", md(pd.DataFrame([{"file": Path(r["_file"]).name, "reasons": ";".join(r["invalid_reasons"])} for r in invalid])), ""]
    (RES / "INTEGRITY.md").write_text("\n".join(L))
    summ = {"planned": len(plan), "valid": nv, "attempts": len(dl), "retries": len(retries), "rejected_files": len(rej),
            "invalid": len(invalid), "run_checks_failed": len(bad_runs), "complete_workloads": int(comp["complete"].sum()),
            "conservation_gen": {str(k): v for k, v in cons_g.items()}, "conservation_flow": {str(k): v for k, v in cons_f.items()},
            "trace": dict(trace), "realization_min": float(real.min()), "realization_below_095": len(low),
            "gen_violation_workloads": list(gm[gm["violations"] > 0]["workload"]),
            "residency_violations": int((~rr["residency_ok"]).sum()), "binary_map_mismatch": int((~rr["binary_map_ok"]).sum()),
            "o_files_in_registry": len(o_sha)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
