#!/usr/bin/env python3
"""Phase 4b: evaluation of the FROZEN predictors at 1.5x, 2.5x, 3.5x with 3-repeat labels (rep0 of Phase 4 + rep1, rep2). New file.

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp_adapter/run_intermediate_3rep.py --precheck   -> revision_exp/out_intermediate_3rep/PRECHECK_3REP.md
  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp_adapter/run_intermediate_3rep.py --final      -> revision_exp/out_intermediate_3rep/*

The Phase 4 adapter revision_exp_adapter/run_intermediate.py is imported unchanged (load_dataset, evaluate_set, evaluate_set_rate, summary_row, mae,
frozen_columns, load_models); nothing of it is edited and its log() is not called (it would append to out_intermediate/logs).
evaluate_rates() below is the computation of run_intermediate.final() for the intermediate rates, without writing; the precheck shows it
reproduces the Phase 4 CSVs from the Phase 4 dataset.

Rules = Phase 4 section 3 (unchanged): Ours (Phase 2 artifact, no clip), Per-instance (Phase 3 artifact, compose_select), Latency-based (= All-GPU),
Oracle (S = T - 0.3 D_macro, T = F / F_max), evaluator unchanged (tol 1e-9, ties -> lower combination number, Top-5 = first 5 of the predicted order).
Labels: dataset_3rep built by the original code/recompute_scores.py from rep0 + rep1 + rep2 (3-repeat means, Phase 2 v1.2 section 3.1).
Best-of-5: the trial samples of Phase 4 (SeedSequence([42, trial, w]), w = position of the workload in the sorted group order of the Phase 4 rate
set) are kept; only the pick of the best of the five uses the 3-repeat labels. w is not renumbered if a workload is excluded.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "revision_exp_adapter"))
import run_intermediate as RI  # noqa: E402  (Phase 4 adapter, unchanged)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B, D, E, P, R = RI.B, RI.D, RI.E, RI.P, RI.R
sys.path.insert(0, str(REPO / "revision_profiling" / "intermediate" / "code_copy_repeats"))
import labels_3rep as L3  # noqa: E402

ALPHA = RI.ALPHA
PROF = RI.PROF
IROOT = PROF / "intermediate"
DATA_P2 = RI.DATA_P2
DATA_REP0 = RI.DATA_INT
DATA_3REP = IROOT / "dataset_3rep"
OUT4 = RI.OUT                                   # Phase 4 outputs (read only)
OUT = REPO / "revision_exp" / "out_intermediate_3rep"
RATES = RI.RATES
NAME = RI.NAME
TRIALS, K = 1000, 5


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / "run_intermediate_3rep.log", "a") as f:
        f.write(line + "\n")


def sha(p):
    return RI.sha(p)


# ------------------------------------------------------------------------------------------------ Best-of-5 with fixed samples
def bo5_samples(cand):
    """{group: (w, comb_nums ascending, [trials, k] sampled comb_nums)} — the draws of baselines.best_of_k (they depend only on t, w, n)."""
    groups = sorted(cand["group"].unique())
    out = {}
    for w, g in enumerate(groups):
        cn = cand[cand["group"] == g].sort_values("comb_num")["comb_num"].values
        n = len(cn)
        smp = np.zeros((TRIALS, min(K, n)), dtype=int)
        for t in range(TRIALS):
            if n <= K:
                idx = np.arange(n)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([B.MASTER_SEED, t, w]))
                idx = np.sort(rng.choice(n, size=K, replace=False))
            smp[t] = cn[idx]
        out[g] = (w, cn, smp)
    return out


def bo5_from_samples(samples, cand, groups):
    """Best of the fixed samples on the labels of `cand` (ties -> lower combination number). Same return layout as B.best_of_k."""
    S_sel = np.zeros((TRIALS, len(groups)))
    top1 = np.zeros((TRIALS, len(groups)), dtype=bool)
    contains = np.zeros((TRIALS, len(groups)), dtype=bool)
    for gi, g in enumerate(groups):
        sub = cand[cand["group"] == g].sort_values("comb_num")
        cn = sub["comb_num"].values
        S = (sub["T"] - ALPHA * sub["D"]).values
        _, cn0, smp = samples[g]
        assert np.array_equal(cn, cn0), f"candidate set of {g} differs from Phase 4"
        pos = {c: i for i, c in enumerate(cn)}
        mx = S.max()
        for t in range(TRIALS):
            idx = np.array([pos[c] for c in smp[t]])
            j = idx[np.argmax(S[idx])]
            S_sel[t, gi] = S[j]
            top1[t, gi] = S[j] >= mx - 1e-9
            contains[t, gi] = bool((S[idx] >= mx - 1e-9).any())
    return groups, S_sel, top1, contains


# ------------------------------------------------------------------------------------------------ evaluation of one dataset (rates)
def evaluate_rates(root, samples=None, keep_groups=None):
    """run_intermediate.final() computation for the intermediate rates of the dataset at `root` (no 3x part, no writing).
    samples: fixed Best-of-5 draws per rate (None -> draw from this dataset, = Phase 4). keep_groups: per rate, the evaluated workloads."""
    cols = RI.frozen_columns()
    m_t, m_d, m_i = RI.load_models()
    dfi = RI.load_dataset(root)
    res, groups, fams, smp_all = {}, {}, {}, {}
    for rt in RATES:
        sub = dfi[dfi["rate"] == rt].reset_index(drop=True)
        if keep_groups is not None:
            sub = sub[sub["group"].isin(keep_groups[rt])].reset_index(drop=True)
        if not len(sub):
            continue
        r = RI.evaluate_set_rate(sub, cols, m_t, m_d, m_i, rt)
        n_of = sub.groupby("group")["N"].first()
        groups[rt] = sorted(n_of[n_of >= 3].index)
        fams[rt] = sorted(sub.groupby("group")["family"].first()[groups[rt]])
        cand = R.cand_of(sub)
        smp = bo5_samples(cand) if samples is None else samples[rt]
        smp_all[rt] = smp
        r["bo5_lib"] = r["bo5"]
        r["bo5"] = bo5_from_samples(smp, cand, sorted(cand["group"].unique()))
        r["order_ours"], _ = E.select(cand[["group", "comb_num"]], r["pred"]["S_hat"].values)
        r["X"] = D.build_features(sub, cols)
        r["X"].index = list(zip(sub["group"], sub["comb_num"]))
        res[rt] = r
    fam_of = dfi.groupby("group")["family"].first()
    return {"df": dfi, "res": res, "groups": groups, "fams": fams, "fam_of": fam_of, "samples": smp_all}


def tables(ev, label):
    """summary rows, MAE rows, per-workload rows of evaluate_rates() output (same columns as Phase 4)."""
    rows, maes, pw = [], [], []
    for rt, r in ev["res"].items():
        gs = ev["groups"][rt]
        for k in ("ours", "inst", "lat", "bo5", "orc"):
            rows.append({"rate": rt, "measurement": label, **RI.summary_row(r, gs, k)})
        maes.append({"rate": rt, **RI.mae(r, gs)})
        per = {k: v.set_index("group") for k, v in r["per"].items()}
        g_, S_sel, top1, contains = r["bo5"]
        bi = {g: i for i, g in enumerate(g_)}
        for g in gs:
            o, u, l_, ins = per["orc"].loc[g], per["ours"].loc[g], per["lat"].loc[g], per["inst"].loc[g]
            pw.append({"rate": rt, "family": ev["fam_of"][g], "group": g, "N": int(o["N"]), "candidates": int(o["n_candidates"]),
                       "oracle_selected": o["selected"], "oracle_S": o["oracle_S"], "n_oracle": int(o["n_oracle"]),
                       "ours_selected": u["selected"], "ours_S": u["S_selected"], "ours_top1": bool(u["top1"]), "ours_top5": bool(u["top5"]), "ours_regret": u["regret"],
                       "latency_selected": l_["selected"], "latency_S": l_["S_selected"], "latency_top1": bool(l_["top1"]),
                       "per_instance_selected": ins["selected"], "per_instance_S": ins["S_selected"], "per_instance_top1": bool(ins["top1"]),
                       "per_instance_top5": bool(ins["top5"]),
                       "bo5_mean_S": float(S_sel[:, bi[g]].mean()), "bo5_std_S": float(S_sel[:, bi[g]].std(ddof=0)), "bo5_top1_rate": float(top1[:, bi[g]].mean()),
                       "ours_minus_allgpu": u["S_selected"] - per["gpu"].loc[g, "S_selected"]})
    return pd.DataFrame(rows), pd.DataFrame(maes), pd.DataFrame(pw)


def paired(pw):
    pr = []
    for rt, sub in pw.groupby("rate", sort=False):
        s6 = sub[sub["N"] >= 6]
        d = s6["ours_minus_allgpu"].values
        pr.append({"rate": rt, "workloads_N>=6": len(s6), "mean_diff": float(d.mean()) if len(d) else np.nan,
                   "se_across_workloads": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                   "n_pos": int((d > 1e-9).sum()), "n_zero": int((np.abs(d) <= 1e-9).sum()), "n_neg": int((d < -1e-9).sum()),
                   "per_workload": "; ".join(f"{f}: {x:+.4f}" for f, x in zip(s6["family"], d))})
    return pd.DataFrame(pr)


def csv_equal(a, b, tol=0.0):
    """Two frames equal after the same CSV round trip (numeric |diff| <= tol, others exact). Returns (equal, max numeric diff, n mismatched cells)."""
    ra = pd.read_csv(io.StringIO(a.to_csv(index=False)))
    rb = pd.read_csv(io.StringIO(b.to_csv(index=False)))
    if list(ra.columns) != list(rb.columns) or ra.shape != rb.shape:
        return False, float("inf"), -1
    mx, bad = 0.0, 0
    for c in ra.columns:
        x, y = ra[c], rb[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.values.astype(float), y.values.astype(float)
            both = np.isnan(xv) & np.isnan(yv)
            d = np.where(both, 0.0, np.abs(xv - yv))
            d = np.where(np.isnan(d), np.inf, d)
            mx = max(mx, float(d.max()) if len(d) else 0.0)
            bad += int((d > tol).sum())
        else:
            bad += int((x.astype(str) != y.astype(str)).sum())
    return bad == 0, mx, bad


# ------------------------------------------------------------------------------------------------ precheck (section 0)
def precheck():
    OUT.mkdir(parents=True, exist_ok=True)
    chk, info = {}, {}
    # 0-1 label code on Phase 2 3x raw repeats
    p1 = L3.precheck_phase2()
    orc_p2 = p1.pop("oracles")
    orc_p2.to_csv(OUT / "precheck_phase2_rep_oracles.csv", index=False)
    for k, v in p1["checks"].items():
        chk[f"0-1 {k}"] = bool(v)
    info["0-1"] = {k: v for k, v in p1.items() if k != "checks"}
    # 0-2 (a) Phase 4 PRECHECK through the imported evaluate_set (Phase 2 3x data)
    cols = RI.frozen_columns()
    m_t, m_d, m_i = RI.load_models()
    df2 = RI.load_dataset(DATA_P2)
    r2 = RI.evaluate_set(df2, cols, m_t, m_d, m_i)
    n_of = df2.groupby("group")["N"].first()
    g3 = sorted(n_of[n_of >= 3].index)
    tab2 = pd.DataFrame([dict(RI.summary_row(r2, g3, k), key=k) for k in ("ours", "inst", "lat", "gpu", "bo5", "orc")])
    eq, mx, bad = csv_equal(tab2, pd.read_csv(OUT4 / "precheck_table.csv"))
    chk["0-2a Phase 4 PRECHECK table (precheck_table.csv) reproduced exactly"] = eq
    eqp, mxp, badp = csv_equal(r2["pred"], pd.read_csv(OUT4 / "precheck_predictions_x3.csv"))
    chk["0-2a Phase 4 PRECHECK candidate predictions (precheck_predictions_x3.csv) reproduced exactly"] = eqp
    info["0-2a"] = {"table_max_diff": mx, "table_bad_cells": bad, "pred_max_diff": mxp, "pred_bad_cells": badp}
    # 0-2 (b) Phase 4 rep0 evaluation from the Phase 4 dataset
    ev0 = evaluate_rates(DATA_REP0)
    tab0, mae0, pw0 = tables(ev0, "1 run (rep0)")
    t4 = pd.read_csv(OUT4 / "table_intermediate.csv")
    t4r = t4[t4["rate"].isin(RATES)].reset_index(drop=True)
    eqt, mxt, badt = csv_equal(tab0, t4r)
    eqw, mxw, badw = csv_equal(pw0, (lambda d: d[d["rate"].isin(RATES)].reset_index(drop=True))(pd.read_csv(OUT4 / "per_workload_intermediate.csv")))
    m4 = pd.read_csv(OUT4 / "mae_intermediate.csv")
    eqm, mxm, badm = csv_equal(mae0, m4[m4["rate"].isin(RATES)].reset_index(drop=True))
    eqpr, _, _ = csv_equal(pd.concat([r["pred"] for r in ev0["res"].values()], ignore_index=True), pd.read_csv(OUT4 / "predictions_intermediate.csv"))
    chk["0-2b Phase 4 rep0 table_intermediate.csv (15 rate rows) reproduced exactly"] = eqt
    chk["0-2b Phase 4 rep0 per_workload_intermediate.csv (36 rate rows) reproduced exactly"] = eqw
    chk["0-2b Phase 4 rep0 mae_intermediate.csv reproduced exactly"] = eqm
    chk["0-2b Phase 4 rep0 predictions_intermediate.csv reproduced exactly"] = eqpr
    # the fixed-sample Best-of-5 equals the library function on rep0
    same_bo5 = True
    for rt, r in ev0["res"].items():
        gl, Sl, tl, cl = r["bo5_lib"]
        gf, Sf, tf, cf = r["bo5"]
        same_bo5 &= list(gl) == list(gf) and np.array_equal(Sl, Sf) and np.array_equal(tl, tf) and np.array_equal(cl, cf)
    chk["0-2b Best-of-5 with the stored draws = baselines.best_of_k on rep0 (bit-identical, 3 rates)"] = bool(same_bo5)
    info["0-2b"] = {"table_max_diff": mxt, "table_bad": badt, "per_workload_max_diff": mxw, "per_workload_bad": badw, "mae_max_diff": mxm}
    # 0-3 identity: runtime vs Phase 4 runtime_sha256.json, offline profile, artifacts vs Phase 4 PRECHECK
    p4rt = json.loads((IROOT / "logs" / "runtime_sha256.json").read_text())["sha256"]
    rt_now = {f: sha(PROF / "code" / f) for f in p4rt}
    chk["0-3 runtime files sha256 = Phase 4 logs/runtime_sha256.json (10 files, direct comparison)"] = rt_now == p4rt and len(p4rt) == 10
    pre4 = json.loads((OUT4 / "precheck.json").read_text())
    art = {"T.json": sha(RI.ART_T), "D.json": sha(RI.ART_D), "per_instance/t_ratio.json": sha(RI.ART_I["t_ratio"]), "per_instance/q_i.json": sha(RI.ART_I["q_i"])}
    chk["0-3 predictor artifacts (T, D, t_ratio, q_i) sha256 = Phase 4 precheck.json"] = art == pre4["artifact_sha256"]
    prof = PROF / "offline_profile" / "offline_profile.json"
    prof_sha = sha(prof)
    head = hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{prof.relative_to(REPO)}"], capture_output=True, check=True).stdout).hexdigest()
    chk["0-3 offline profile sha256 = git HEAD blob (Phase 4 recorded no profile hash; Phase 4 RUN_PLAN checked 'unchanged vs git HEAD')"] = prof_sha == head
    info["0-3"] = {"runtime_sha256": rt_now, "artifact_sha256": art, "offline_profile_sha256": prof_sha, "offline_profile_git_HEAD_sha256": head}
    allp = all(chk.values())
    j = {"all_pass": allp, "checks": chk, "info": info, "ran_at": datetime.now().isoformat(sep=" ", timespec="seconds")}
    (OUT / "precheck_3rep.json").write_text(json.dumps(j, indent=1, default=str))
    se = pd.DataFrame(info["0-1"]["se_stats_N>=3"])
    Lm = ["# PRECHECK_3REP — Phase 4b §0 (측정 전)", "",
          f"실행 {j['ran_at']}. 코드 `revision_exp_adapter/run_intermediate_3rep.py --precheck`, label 코드 `revision_profiling/intermediate/code_copy_repeats/labels_3rep.py`.", "",
          f"**결과: {'PASS — rep1 측정 시작 가능' if allp else 'FAIL — 측정 시작 금지'}**", "",
          "## 검사", "", "| 검사 | 결과 |", "|---|---|"] + [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in chk.items()]
    Lm += ["", "## 0-1 3회 평균 label 코드", "",
           "- Label은 원본 `code/recompute_scores.py`의 `load_runs()`·`build_tables()`(수정 없음)가 만든다. Phase 2 3× label을 만든 함수와 같다.",
           "- 정의: 후보별 `F̄ = mean_r F_r`, `D̄_macro = mean_r D_macro,r`(`D_macro,r` = run의 `system.drop_macro` = instance별 `q_{i,r}`의 평균; "
           "아래 표에서 차 ≤ 1.2e-16 확인), workload별 `F_max = max_cand F̄`, `T = F̄/F_max`, `S = T − 0.3·D̄_macro`.",
           "- **반복별 `S_r`의 정규화 분모 (Phase 2 구현 확인)**: SE·반복 편차에 쓰는 `S_r`(`S_macro_a0.3_rep<r>`)는 `F_r / F_max − 0.3·D_macro,r`이고, "
           "`F_max`는 **3회 평균 `F̄`의 공통 최대값**이다(`recompute_scores.build_tables` 91행 `S_rep = F / fmax - ALPHA * Dm`, `fmax = max(F.mean())`). "
           "SE = 표본표준편차(ddof=1)/√3(`recompute_scores.se`). 반면 **반복별 oracle**은 Phase 2 `code/integrity.py` 151–158행대로 "
           "**각 반복 자체의 정규화** `F_r / max_cand F_r − 0.3·D_macro,r`로 구한다(단일 측정 campaign이 냈을 oracle). 이 Phase도 두 정의를 그대로 쓴다.",
           f"- Phase 2 3× 원시 반복 {info['0-1']['runs_loaded']} runs → 후보 {info['0-1']['candidates']}개, Phase 2 `dataset/labels_test.csv` {info['0-1']['rows_phase2_labels_test']}행과 1:1 대조. 최대 절대 차:", "",
           "| 열 | 최대 차 |", "|---|---|"] + [f"| {k} | {v:.3g} |" for k, v in info["0-1"]["max_abs_diff"].items()]
    Lm += ["", f"- `D_macro,r` vs `mean_i q_{{i,r}}`: {info['0-1']['D_macro_run_value_vs_mean_i_q_i']}", "",
           "- 반복 편차 재현(3× N≥3, 후보 480개; Phase 2 `results/INTEGRITY.md` §6: S의 SE 중앙값 0.0010, 90% 0.0027, 최대 0.0130):", "",
           "| 지표 | 후보 | 중앙값 | 90% 분위 | 최대 |", "|---|---|---|---|---|"]
    Lm += [f"| {r['metric']} | {r['candidates']} | {r['median']:.4f} | {r['p90']:.4f} | {r['max']:.4f} |" for _, r in se.iterrows()]
    Lm += ["", f"- 반복별 oracle(자체 정규화, 동점 1e-9): 대표 ID가 Phase 2 `results/integrity/oracle_x3.csv`와 36개(12 × 3) 모두 같다. "
           f"세 반복 대표 ID가 같은 workload {info['0-1']['workloads_rep_ids_all_equal']}/12(Phase 2 {info['0-1']['workloads_rep_ids_all_equal_phase2']}/12), "
           f"세 반복에 공통으로 최적인 후보가 있는 workload {info['0-1']['workloads_common_optimal']}/12(`precheck_phase2_rep_oracles.csv`).", "",
           "## 0-2 평가 경로", "",
           f"- (a) Phase 4 adapter의 `evaluate_set()`(import, 수정 없음)로 Phase 2 3× 데이터를 평가해 Phase 4 `precheck_table.csv`·`precheck_predictions_x3.csv`를 "
           f"같은 CSV 저장·읽기 후 비교: 표 최대 차 {info['0-2a']['table_max_diff']:.3g}, 예측 최대 차 {info['0-2a']['pred_max_diff']:.3g}(불일치 칸 {info['0-2a']['table_bad_cells']}, {info['0-2a']['pred_bad_cells']}). "
           "Phase 4 PRECHECK가 Phase 3 값과 같음을 이미 확인했으므로 Phase 3 값도 재현된다.",
           f"- (b) 이 adapter의 `evaluate_rates()`로 Phase 4 rep0 dataset을 평가해 Phase 4 `table_intermediate.csv`(rate 15행), `per_workload_intermediate.csv`(rate 36행), "
           f"`mae_intermediate.csv`, `predictions_intermediate.csv`와 비교: 최대 차 표 {info['0-2b']['table_max_diff']:.3g}, workload별 {info['0-2b']['per_workload_max_diff']:.3g}, MAE {info['0-2b']['mae_max_diff']:.3g}.",
           "- Best-of-5는 Phase 4 추출(`SeedSequence([42, t, w])`)을 후보 ID로 저장해 두고, 다섯 중 최적 선택에만 label을 쓰는 함수로 바꿨다. "
           "rep0에서 이 함수가 `baselines.best_of_k`와 bit 단위로 같다.", "",
           "## 0-3 동일성", "", "| 파일 | sha256 | 대조 |", "|---|---|---|"]
    Lm += [f"| `revision_profiling/code/{f}` | `{h}` | Phase 4 `logs/runtime_sha256.json` |" for f, h in rt_now.items()]
    Lm += [f"| `{k}` | `{v}` | Phase 4 `precheck.json` |" for k, v in art.items()]
    Lm += [f"| `revision_profiling/offline_profile/offline_profile.json` | `{prof_sha}` | git HEAD blob `{head[:12]}…`. Phase 4 PRECHECK에는 profile hash 기록이 없다. "
           "Phase 4 RUN_PLAN이 'git HEAD와 같음'을 확인했으므로 같은 기준으로 비교했다 |", ""]
    (OUT / "PRECHECK_3REP.md").write_text("\n".join(Lm))
    print("\n".join(Lm))
    return allp


# ------------------------------------------------------------------------------------------------ final (section 3)
def final():
    pre = json.loads((OUT / "precheck_3rep.json").read_text())
    assert pre["all_pass"], "PRECHECK_3REP has not passed"
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "git_status_before.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    art = {"T.json": sha(RI.ART_T), "D.json": sha(RI.ART_D), "per_instance/t_ratio.json": sha(RI.ART_I["t_ratio"]), "per_instance/q_i.json": sha(RI.ART_I["q_i"])}
    assert art == pre["info"]["0-3"]["artifact_sha256"]
    comp = pd.read_csv(DATA_3REP / "workload_completeness.csv")
    lab3 = pd.read_csv(DATA_3REP / "labels_test.csv")
    assert (lab3["n_reps"] == 3).all(), "a candidate without 3 repeats in dataset_3rep"
    # rep0 (Phase 4) through the same function -> stored draws, groups, selections
    ev0 = evaluate_rates(DATA_REP0)
    keep = {rt: [g for g in ev0["groups"][rt] if g[:-5] in set(lab3["workload"])] for rt in ev0["groups"]}
    excluded = {rt: sorted(set(ev0["groups"][rt]) - set(keep[rt])) for rt in ev0["groups"]}
    ev3 = evaluate_rates(DATA_3REP, samples=ev0["samples"])
    for rt in ev3["groups"]:
        assert ev3["groups"][rt] == keep[rt]
    # ---- selection invariance (features, predictions, predicted order, selected ids) for every candidate
    inv = []
    for rt, r3 in ev3["res"].items():
        r0 = ev0["res"][rt]
        X0 = r0["X"].loc[r3["X"].index]
        p0 = r0["pred"].set_index(["group", "comb_num"]).loc[list(r3["X"].index)]
        p3 = r3["pred"].set_index(["group", "comb_num"])
        i0 = r0["inst_cand"].set_index(["group", "comb_num"]).loc[list(r3["X"].index)]
        i3 = r3["inst_cand"].set_index(["group", "comb_num"]).loc[list(r3["X"].index)]
        per0 = {k: v.set_index("group") for k, v in r0["per"].items()}
        per3 = {k: v.set_index("group") for k, v in r3["per"].items()}
        gs = ev3["groups"][rt]
        row = {"rate": rt, "candidates": len(X0),
               "features_identical": bool(np.array_equal(X0.values, r3["X"].values)) and list(X0.columns) == list(r3["X"].columns),
               "ours_T_hat_D_hat_S_hat_identical": all(np.array_equal(p0[c].values, p3[c].values) for c in ("T_hat", "D_hat", "S_hat")),
               "ours_predicted_order_identical": all(list(r0["order_ours"][g]) == list(r3["order_ours"][g]) for g in gs),
               "per_instance_F_hat_D_hat_S_hat_identical": all(np.array_equal(i0[c].values, i3[c].values) for c in ("F_hat", "D_hat", "T_hat_select", "S_hat")),
               "ours_selected_identical": all(per0["ours"].loc[g, "selected"] == per3["ours"].loc[g, "selected"] for g in gs),
               "per_instance_selected_identical": all(per0["inst"].loc[g, "selected"] == per3["inst"].loc[g, "selected"] for g in gs),
               "latency_selected_identical": all(per0["lat"].loc[g, "selected"] == per3["lat"].loc[g, "selected"] for g in gs),
               "bo5_draws_identical": all(np.array_equal(ev0["samples"][rt][g][2], ev3["samples"][rt][g][2]) and ev0["samples"][rt][g][0] == ev3["samples"][rt][g][0]
                                          for g in gs)}
        inv.append(row)
    inv = pd.DataFrame(inv)
    inv.to_csv(OUT / "selection_invariance_check.csv", index=False)
    flags = inv.drop(columns=["rate", "candidates"]).values
    if not flags.all() or int(inv["candidates"].sum()) != int(lab3.shape[0]):
        log("selection invariance FAILED -> evaluation stopped\n" + inv.to_string(index=False))
        raise SystemExit("selection invariance check failed; see selection_invariance_check.csv")
    # ---- tables: 3-repeat, rep0 (Phase 4), 3x (Phase 3) on the same workloads
    tab3, mae3, pw3 = tables(ev3, "3-repeat mean")
    ev0k = evaluate_rates(DATA_REP0, samples=ev0["samples"], keep_groups=keep) if any(excluded.values()) else ev0
    tab0, mae0, pw0 = tables(ev0k, "1 run (rep0, Phase 4)")
    cols = RI.frozen_columns()
    m_t, m_d, m_i = RI.load_models()
    ref = RI.load_dataset(DATA_P2)
    res3x = RI.evaluate_set(ref, cols, m_t, m_d, m_i)
    n_ref = ref.groupby("group")["N"].first()
    g3_all = sorted(n_ref[n_ref >= 3].index)
    fam_of3 = ref.groupby("group")["family"].first()
    common = set.intersection(*[set(v) for v in ev3["fams"].values()])
    x3_sets = [("x3 (Phase 3, 12 workloads, 3-repeat mean)", g3_all)]
    if len(common) < 12:
        x3_sets.append(("x3 (Phase 3, families evaluated at every rate here)", sorted(g for g in g3_all if fam_of3[g] in common)))
    rows3x, mae3x = [], []
    for lab_, gs in x3_sets:
        for k in ("ours", "inst", "lat", "bo5", "orc"):
            rows3x.append({"rate": lab_, "measurement": "3-repeat mean", **RI.summary_row(res3x, gs, k)})
        mae3x.append({"rate": lab_, **RI.mae(res3x, gs)})
    tab = pd.concat([tab3, tab0, pd.DataFrame(rows3x)], ignore_index=True)
    tab.to_csv(OUT / "table_intermediate_3rep.csv", index=False)
    mae3["measurement"], mae0["measurement"] = "3-repeat mean", "1 run (rep0, Phase 4)"
    mt = pd.concat([mae3, mae0, pd.DataFrame(mae3x).assign(measurement="3-repeat mean")], ignore_index=True)
    mt.to_csv(OUT / "mae_intermediate_3rep.csv", index=False)
    # per-workload: 3-repeat with rep0 columns next to it
    pw = pw3.merge(pw0.drop(columns=["family", "N", "candidates"]), on=["rate", "group"], suffixes=("", "_rep0"), validate="one_to_one")
    lab3i = lab3.set_index(["workload", "combination"])
    for side in ("oracle", "ours", "per_instance", "latency"):
        pw[f"{side}_S_se"] = [float(lab3i.loc[(g[:-5], c), "S_macro_a0.3_se"]) for g, c in zip(pw["group"], pw[f"{side}_selected"].map(lambda s: "combination_%03d" % int(s.rsplit("_", 1)[1])))]
    pw["ours_top1_changed"] = pw["ours_top1"] != pw["ours_top1_rep0"]
    pw["ours_top5_changed"] = pw["ours_top5"] != pw["ours_top5_rep0"]
    pw["oracle_changed"] = pw["oracle_selected"] != pw["oracle_selected_rep0"]
    pw.to_csv(OUT / "per_workload_intermediate_3rep.csv", index=False)
    # paired N>=6
    p3 = paired(pw3).assign(measurement="3-repeat mean")
    p0 = paired(pw0).assign(measurement="1 run (rep0, Phase 4)")
    pw3x = (lambda d: d[d["rate"] == "x3"].reset_index(drop=True))(pd.read_csv(OUT4 / "per_workload_intermediate.csv"))
    p3x = paired(pw3x[pw3x["family"].isin(common)] if len(common) < 12 else pw3x).assign(measurement="3-repeat mean (Phase 3)")
    pr = pd.concat([p3, p0, p3x], ignore_index=True)
    pr.to_csv(OUT / "paired_n6_intermediate_3rep.csv", index=False)
    # selection ids vs rep0 and 3x
    sv = []
    for fam in sorted(fam_of3[g3_all].values):
        row = {"family": fam, "N": int(len(fam.split("+")))}
        for rt in ("x1.5", "x2.5", "x3", "x3.5"):
            if rt == "x3":
                s = pw3x[pw3x["family"] == fam]
                row["oracle_x3"] = s["oracle_selected"].iloc[0] if len(s) else "missing"
                row["ours_x3"] = s["ours_selected"].iloc[0] if len(s) else "missing"
                continue
            s = pw[(pw["family"] == fam) & (pw["rate"] == rt)]
            row[f"oracle_{rt}_3rep"] = s["oracle_selected"].iloc[0] if len(s) else "excluded"
            row[f"n_oracle_{rt}_3rep"] = int(s["n_oracle"].iloc[0]) if len(s) else None
            row[f"oracle_{rt}_rep0"] = s["oracle_selected_rep0"].iloc[0] if len(s) else "excluded"
            row[f"ours_{rt}"] = s["ours_selected"].iloc[0] if len(s) else "excluded"
            row[f"ours_top1_{rt}_3rep"] = bool(s["ours_top1"].iloc[0]) if len(s) else None
            row[f"ours_top1_{rt}_rep0"] = bool(s["ours_top1_rep0"].iloc[0]) if len(s) else None
        sv.append(row)
    sv = pd.DataFrame(sv)
    sv.to_csv(OUT / "selection_vs_3x_3rep.csv", index=False)
    pd.concat([r["pred"] for r in ev3["res"].values()], ignore_index=True).to_csv(OUT / "predictions_intermediate_3rep.csv", index=False)
    # manuscript numbers: x = max over rates of the rate-mean regret (Oracle S - Ours S), rounded UP to 3 decimals
    ours = tab3[tab3["method"] == NAME["ours"]].set_index("rate")
    regs = {rt: float(ours.loc[rt, "regret"]) for rt in ours.index}
    x = math.ceil(max(regs.values()) * 1000 - 1e-12) / 1000
    summ = {"rates": list(ev3["res"]), "workloads": {k: len(v) for k, v in ev3["groups"].items()}, "excluded_groups": excluded,
            "excluded_workloads_dataset": comp[comp["excluded"]]["workload"].tolist(), "artifact_sha256": art,
            "selection_invariance": inv.to_dict(orient="records"),
            "manuscript": {"top1": {rt: f"{int(ours.loc[rt, 'top1_hits'])}/{int(ours.loc[rt, 'workloads'])}" for rt in ours.index},
                           "top5": {rt: f"{int(ours.loc[rt, 'top5_hits'])}/{int(ours.loc[rt, 'workloads'])}" for rt in ours.index},
                           "rate_mean_regret": regs, "x_ceil_3dp": x},
            "best_of_5_rule": "Phase 4 draws kept (SeedSequence([42, trial, w]), w = position in the Phase 4 sorted group order of the rate); best of the 5 by 3-repeat labels"}
    (OUT / "summary_3rep.json").write_text(json.dumps(summ, indent=1, default=str))
    (OUT / "logs" / "git_status_after.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    log("\n" + tab.to_string(index=False))
    log("\n" + mt.to_string(index=False))
    log("\n" + pr.to_string(index=False))
    log("\n" + inv.to_string(index=False))
    log(json.dumps(summ["manuscript"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.precheck:
        sys.exit(0 if precheck() else 1)
    if a.final:
        final()
