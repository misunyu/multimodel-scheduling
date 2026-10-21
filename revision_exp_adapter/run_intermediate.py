#!/usr/bin/env python3
"""Phase 4: evaluation of the FROZEN Phase 2 / Phase 3 predictors at the intermediate rates 1.5x, 2.5x, 3.5x (no retraining).

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp_adapter/run_intermediate.py --precheck   -> revision_exp/out_intermediate/PRECHECK.md
  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp_adapter/run_intermediate.py --final      -> revision_exp/out_intermediate/*

Only existing modules are used, unchanged: revision_exp/{data,evaluate,baselines,predict,run_all}.py, revision_exp_adapter/run_reprofiled.py
(helpers only) and revision_exp/phase3/baselines/{instance_rows,compose_select}.py. Paths are pointed at a dataset folder at run time
(D.DS, D.SETS). The same function evaluate_set() is used for the precheck (Phase 2 3x data) and for the intermediate rates.

Rules (fixed before the campaign, Phase 4 instruction section 3):
  Ours          Phase 2 artifact out_reprofiled/artifacts/multi/{T,D}.json, S_hat = T_hat - 0.3 D_hat, no clip (P.multi_score(clip=False))
  Per-instance  Phase 3 artifact out_phase3/artifacts/per_instance/{t_ratio,q_i}.json, compose_select unchanged (clip t >= 0, q in [0,1];
                T_hat_select = F_hat / max_x F_hat; S_hat = T_hat_select - 0.3 D_hat, no clip)
  Latency-based isolated latency of the offline profile -> per-model device -> the matching candidate (= all-GPU on this profile)
  Oracle        measured S = T - 0.3 D_macro, T = F / F_max (workload), no clip
  Best-of-5     baselines.best_of_k (1,000 trials, SeedSequence([42, trial, workload index in sorted group order of the evaluated set]))
  Evaluator     evaluate.py unchanged (oracle tolerance 1e-9, ties -> lower combination number, Top-5 = first 5 of the predicted order)
Selection functions never receive measured columns: E.select() / CS.compose() / CS.select_candidates() assert it, and the direct features
are rebuilt from schedule-only windows (model, execution) and asserted identical to the loader features.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "revision_exp"
for p in (str(EXP), str(REPO / "revision_exp_adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baselines as B  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402
import run_reprofiled as RR  # noqa: E402
from phase3.baselines import compose_select as CS  # noqa: E402
from phase3.baselines import instance_rows as IR  # noqa: E402

ALPHA = 0.3
assert D.ALPHA == ALPHA
PROF = REPO / "revision_profiling"
P2 = EXP / "out_reprofiled"
P3 = EXP / "out_phase3"
ART_T, ART_D = P2 / "artifacts" / "multi" / "T.json", P2 / "artifacts" / "multi" / "D.json"
ART_I = {t: P3 / "artifacts" / "per_instance" / f"{t}.json" for t in ("t_ratio", "q_i")}
DATA_P2 = PROF / "dataset"
DATA_INT = PROF / "intermediate" / "dataset"
OUT = EXP / "out_intermediate"
RATES = ("x1.5", "x2.5", "x3.5")
NAME = {"ours": "Ours (XGBoost Full)", "inst": "Per-instance predictor", "lat": "Latency-based (= All-GPU)", "gpu": "All-GPU",
        "bo5": "Best-of-5", "orc": "Oracle"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / "run_intermediate.log", "a") as f:
        f.write(line + "\n")


def load_dataset(root):
    """Existing loader on <root>/test_x3.csv + test_schedules_x3.csv, labels from <root>/labels_test.csv (D = D_macro).
    family / rate are taken from the label file: data.family_key / rate_key only know the old '_x1-5' naming."""
    D.DS = root
    D.SETS = {"eval": ("test_x3.csv", "test_schedules_x3.csv", [])}
    df = RR.load("eval", "macro")
    lab = pd.read_csv(root / "labels_test.csv")[["schedule_file", "combination", "family", "rate", "multiplier"]]
    df = df.drop(columns=["family", "rate"]).merge(lab, on=["schedule_file", "combination"], how="left", validate="one_to_one")
    assert df["family"].notna().all()
    return df


def stripped_features(df, cols):
    """Direct features rebuilt from schedule-only windows (model + execution per instance, no measured value)."""
    s = D.suite()
    idx = s._index_schedules_from_csv(D.DS / "test_schedules_x3.csv")
    xs = []
    for sf, comb in zip(df["schedule_file"], df["combination"]):
        doc = s._find_schedule(idx, sf)
        inf = s._build_infps_lookup(doc, comb)
        rows = s._rows_from_combo_struct(dict(s._iter_combos_from_schedule(doc))[comb])
        w = {"schedule_file": sf, "combination": comb,
             "models": {f"view{i + 1}": {"model": r["model"], "execution": str(r["execution"]).upper()} for i, r in enumerate(sorted(rows, key=lambda r: r["model"]))}}
        x, _, _ = s.featurize_window(w, inf)
        xs.append(x)
    X = pd.DataFrame(xs).fillna(0.0)
    extra = sorted(set(X.columns) - set(cols))
    X = X.reindex(columns=cols, fill_value=0.0)
    X.index = df["row_uid"].values
    return X, extra


def raw_feature_columns(df):
    return sorted(set().union(*[set(x) for x in df["_x"]]))


def evaluate_set(df, cols, m_t, m_d, m_inst, groups_bo5=None):
    """All methods on one dataset frame. Returns dict of per-workload frames, predictions and checks."""
    out = {"checks": {}}
    extra_raw = sorted(set(raw_feature_columns(df)) - set(cols))
    out["checks"]["feature_columns_outside_frozen_list"] = extra_raw
    Xt = D.build_features(df, cols)
    Xs, extra_s = stripped_features(df, cols)
    out["checks"]["stripped_features_identical"] = bool(np.array_equal(Xt.values, Xs.values)) and not extra_s
    assert out["checks"]["stripped_features_identical"], "features depend on measured values"
    t_hat, d_hat = P.predict_multi(m_t, m_d, Xt)
    t_hat, d_hat = np.asarray(t_hat, float), np.asarray(d_hat, float)
    s_hat, _ = P.multi_score(t_hat, d_hat, ALPHA, clip=False)
    per = {"ours": R.ours_per_workload(df, s_hat, ALPHA, NAME["ours"])}
    # per-instance, from the schedule CSV and the offline profile only
    iso = IR.iso_capacity(PROF / "offline_profile" / "offline_profile.json")
    rows = IR.rows_from_schedules(D.DS / "test_schedules_x3.csv", iso)
    Xi = rows[IR.FEATURES].values
    pt, pq = m_inst["t_ratio"].predict(Xi), m_inst["q_i"].predict(Xi)
    comp = CS.compose(rows[sorted(CS.ALLOWED_INPUT & set(rows.columns))], pt, pq)
    s_i, order_i, ties_i = CS.select_candidates(comp, ALPHA)
    comp["S_hat"] = s_i
    assert set(zip(comp["group"], comp["comb_num"])) == set(zip(df["group"], df["comb_num"])), "per-instance candidates differ from measured"
    cand = R.cand_of(df)
    per["inst"] = E.score_selection(cand, order_i, ALPHA, NAME["inst"], top5=True, ties=ties_i)
    tg = B.latency_targets(RR.offline_latency())
    out["latency_targets"] = tg
    per["lat"] = E.score_selection(cand, E.single_choice(cand[["group", "comb_num"]], B.choose_by_targets(cand, tg)), ALPHA, NAME["lat"], top5=False)
    per["gpu"] = R.single_per_workload(df, B.choose_all_device(cand, "all_gpu"), ALPHA, NAME["gpu"])
    S_true = (df["T"] - ALPHA * df["D"]).values
    per["orc"] = R.ours_per_workload(df, S_true, ALPHA, NAME["orc"])
    bc = cand if groups_bo5 is None else cand[cand["group"].isin(groups_bo5)]
    out["bo5"] = B.best_of_k(bc, ALPHA)
    out["per"] = per
    out["pred"] = pd.DataFrame({"row_uid": df["row_uid"].values, "group": df["group"].values, "family": df["family"].values, "rate": df["rate"].values,
                                "combination": df["combination"].values, "comb_num": df["comb_num"].values, "N": df["N"].values,
                                "T": df["T"].values, "D": df["D"].values, "S": S_true, "T_hat": t_hat, "D_hat": d_hat, "S_hat": s_hat})
    cm = comp.merge(df[["group", "comb_num", "N", "T", "D", "F_mean", "F_max", "all_gpu"]], on=["group", "comb_num"], how="left", validate="one_to_one")
    out["inst_cand"] = cm
    out["inst_rows"] = rows.assign(pred_t_ratio=pt, pred_q_i=pq)
    return out


def summary_row(res, groups, method):
    if method == "bo5":
        g, S_sel, top1, contains = res["bo5"]
        m = np.array([x in groups for x in g])
        orc = res["per"]["orc"].set_index("group").loc[[x for x in g if x in groups], "oracle_S"].values
        means = S_sel[:, m].mean(axis=1)
        return {"method": NAME["bo5"], "workloads": int(m.sum()), "avg_score": float(means.mean()), "avg_score_std_over_trials": float(means.std(ddof=0)),
                "regret": float(orc.mean() - means.mean()), "top1_hits": None, "top1_rate": float(top1[:, m].mean()), "top5_hits": None,
                "sample_contains_oracle": float(contains[:, m].mean())}
    p = res["per"][method]
    a = E.aggregate(p[p["group"].isin(groups)], 1, NAME[method])
    return {"method": NAME[method], "workloads": a["workloads"], "avg_score": a["avg_score"], "avg_score_std_over_trials": None, "regret": a["regret"],
            "top1_hits": a["top1_hits"], "top1_rate": a["top1_hits"] / a["top1_n"] if a["top1_n"] else None,
            "top5_hits": a["top5_hits"], "sample_contains_oracle": None}


def mae(res, groups):
    pr = res["pred"][res["pred"]["group"].isin(groups) & (res["pred"]["N"] >= 3)]
    ci = res["inst_cand"][res["inst_cand"]["group"].isin(groups) & (res["inst_cand"]["N"] >= 3)]
    return {"candidates": len(pr), "ours_T": float((pr["T_hat"] - pr["T"]).abs().mean()), "ours_D": float((pr["D_hat"] - pr["D"]).abs().mean()),
            "inst_T": float((ci["T_hat_select"] - ci["T"]).abs().mean()), "inst_D": float((ci["D_hat"] - ci["D"]).abs().mean())}


def load_models():
    m_t, m_d = P.load_model(ART_T), P.load_model(ART_D)
    m_i = {t: P.load_model(p) for t, p in ART_I.items()}
    return m_t, m_d, m_i


def frozen_columns():
    cols = (P2 / "feature_columns.txt").read_text().split()
    assert cols == (P3 / "feature_columns_full.txt").read_text().split() and len(cols) == 204
    return cols


# ------------------------------------------------------------------------------------------------ precheck (section 3.0)
def precheck():
    OUT.mkdir(parents=True, exist_ok=True)
    cf2 = json.loads((P2 / "logs" / "config_frozen.json").read_text())
    cf3 = json.loads((P3 / "logs" / "config_frozen.json").read_text())
    hashes = {"T.json": sha(ART_T), "D.json": sha(ART_D), "per_instance/t_ratio.json": sha(ART_I["t_ratio"]), "per_instance/q_i.json": sha(ART_I["q_i"])}
    chk = {"T_sha_equals_phase2_config_frozen": hashes["T.json"] == cf2["artifact_sha256"]["artifacts/multi/T.json"] == cf3["xgb_full_artifact"]["T"],
           "D_sha_equals_phase2_config_frozen": hashes["D.json"] == cf2["artifact_sha256"]["artifacts/multi/D.json"] == cf3["xgb_full_artifact"]["D"]}
    cols = frozen_columns()
    m_t, m_d, m_i = load_models()
    df = load_dataset(DATA_P2)
    res = evaluate_set(df, cols, m_t, m_d, m_i)
    n_of = df.groupby("group")["N"].first()
    g3 = sorted(n_of[n_of >= 3].index)
    assert len(g3) == 12 and (df["N"] >= 3).sum() == 480
    tab = {k: summary_row(res, g3, k) for k in ("ours", "inst", "lat", "gpu", "bo5", "orc")}
    # targets: Phase 3 tables (and the instruction values)
    p3 = pd.read_csv(P3 / "table_all.csv").set_index("method")
    p3b = pd.read_csv(P3 / "table_baseline.csv").set_index("method")
    tgt = {"ours": (0.9930, 0.0044, 7, 11), "inst": (0.9868, 0.0106, 6, 11), "lat": (0.9727, 0.0247, 6, None), "orc": (0.9974, 0.0, 12, None)}
    for k, (avg, reg, t1, t5) in tgt.items():
        r = tab[k]
        chk[f"{k}_avg_{avg}"] = round(r["avg_score"], 4) == avg
        chk[f"{k}_regret_{reg}"] = round(r["regret"], 4) == reg
        chk[f"{k}_top1_{t1}/12"] = r["top1_hits"] == t1
        if t5 is not None:
            chk[f"{k}_top5_{t5}/12"] = r["top5_hits"] == t5
    b = tab["bo5"]
    bo5_p3 = p3b.loc["Best-of-5"]
    chk["bo5_avg_equals_phase3"] = round(b["avg_score"], 4) == float(bo5_p3["Avg score"]) == 0.9404
    chk["bo5_regret_equals_phase3"] = round(b["regret"], 4) == float(bo5_p3["regret"])
    chk["bo5_top1_rate_equals_phase3"] = f"{b['top1_rate']:.3f}" == str(bo5_p3["Top-1"]).split()[0]
    p2_bo5 = pd.read_csv(P2 / "table3a.csv")
    bo5_std_p2 = p2_bo5.loc[p2_bo5["strategy"].str.startswith("Best-of-5"), "avg_score_std_over_trials"]
    chk["bo5_std_equals_phase2_table3a"] = len(bo5_std_p2) == 1 and round(b["avg_score_std_over_trials"], 4) == round(float(bo5_std_p2.iloc[0]), 4)
    # candidate-level predictions vs Phase 3 output
    pr3 = pd.read_csv(P3 / "predictions_x3.csv")
    m = res["pred"].merge(pr3[["row_uid", "T_hat[XGBoost Full]", "D_hat[XGBoost Full]", "S_hat[XGBoost Full]"]], on="row_uid", validate="one_to_one")
    assert len(m) == len(res["pred"]) == 504
    dmax = {c: float((m[f"{c}_hat"] - m[f"{c}_hat[XGBoost Full]"]).abs().max()) for c in ("T", "D", "S")}   # in memory vs Phase 3 CSV
    # Phase 3 values went through DataFrame.to_csv / read_csv; compare after the same round trip (CSV float formatting differs by <= 1 ulp)
    rt_ = pd.read_csv(io.StringIO(res["pred"][["row_uid", "T_hat", "D_hat", "S_hat"]].to_csv(index=False))).merge(
        pr3[["row_uid", "T_hat[XGBoost Full]", "D_hat[XGBoost Full]", "S_hat[XGBoost Full]"]], on="row_uid", validate="one_to_one")
    dmax_rt = {c: float((rt_[f"{c}_hat"] - rt_[f"{c}_hat[XGBoost Full]"]).abs().max()) for c in ("T", "D", "S")}
    chk["ours_candidate_predictions_equal_phase3 (after the same CSV round trip: max abs diff 0)"] = all(v == 0.0 for v in dmax_rt.values())
    ic3 = pd.read_csv(P3 / "per_instance_candidates_x3.csv")
    mi = res["inst_cand"].merge(ic3[["group", "comb_num", "F_hat", "D_hat", "T_hat_select", "S_hat"]], on=["group", "comb_num"], suffixes=("", "_p3"), validate="one_to_one")
    assert len(mi) == len(res["inst_cand"]) == 504
    dmi = {c: float((mi[c] - mi[f"{c}_p3"]).abs().max()) for c in ("F_hat", "D_hat", "T_hat_select", "S_hat")}
    chk["per_instance_candidate_predictions_equal_phase3(max abs diff <= 1e-9)"] = all(v <= 1e-9 for v in dmi.values())
    # per-workload selections (incl. tie handling) vs Phase 3 per_workload_all.csv
    pw3 = pd.read_csv(P3 / "per_workload_all.csv")
    sel_cmp = []
    for k, nm3 in (("ours", "XGBoost Full"), ("inst", "Per-instance predictor ([14]-inspired)"), ("lat", "Latency-based (= All-GPU)"), ("orc", "Oracle")):
        a = res["per"][k].set_index("group")
        b3 = pw3[pw3["method"] == nm3].set_index("group")
        for g in g3:
            sel_cmp.append({"method": NAME[k], "group": g, "selected_this": a.loc[g, "selected"], "selected_phase3": b3.loc[g, "selected"],
                            "S_this": a.loc[g, "S_selected"], "S_phase3": b3.loc[g, "S_selected"], "n_tied_with_first": a.loc[g].get("n_tied_with_first"),
                            "same": a.loc[g, "selected"] == b3.loc[g, "selected"] and abs(a.loc[g, "S_selected"] - b3.loc[g, "S_selected"]) < 1e-12})
    sel_cmp = pd.DataFrame(sel_cmp)
    sel_cmp.to_csv(OUT / "precheck_selections.csv", index=False)
    chk["per_workload_selections_equal_phase3 (Ours, Per-instance, Latency, Oracle; 48 rows)"] = bool(sel_cmp["same"].all()) and len(sel_cmp) == 48
    chk["latency_targets_all_GPU"] = set(res["latency_targets"].values()) == {"GPU"}
    chk["latency_selection_equals_all_gpu"] = all(res["per"]["lat"].set_index("group").loc[g, "selected"] == res["per"]["gpu"].set_index("group").loc[g, "selected"] for g in g3)
    chk["features_selected_from_schedule_only (stripped windows identical)"] = res["checks"]["stripped_features_identical"]
    chk["no feature column outside the frozen 204"] = not res["checks"]["feature_columns_outside_frozen_list"]
    allp = all(chk.values())
    tab_df = pd.DataFrame([dict(v, key=k) for k, v in tab.items()])
    tab_df.to_csv(OUT / "precheck_table.csv", index=False)
    res["pred"].to_csv(OUT / "precheck_predictions_x3.csv", index=False)
    j = {"all_pass": allp, "checks": chk, "artifacts": {"T": str(ART_T.relative_to(REPO)), "D": str(ART_D.relative_to(REPO)),
                                                        **{f"per_instance_{t}": str(p.relative_to(REPO)) for t, p in ART_I.items()}},
         "artifact_sha256": hashes, "max_abs_diff_ours_vs_phase3_in_memory": dmax, "max_abs_diff_ours_vs_phase3_same_csv_round_trip": dmax_rt, "max_abs_diff_per_instance_vs_phase3": dmi,
         "table": {k: v for k, v in tab.items()}, "ran_at": datetime.now().isoformat(sep=" ", timespec="seconds")}
    (OUT / "precheck.json").write_text(json.dumps(j, indent=1, default=float))
    L = ["# PRECHECK — Phase 4 §3.0 (Phase 3 최종 규칙 재현, 측정 전)", "",
         f"실행 {j['ran_at']}. 코드 `revision_exp_adapter/run_intermediate.py --precheck`(이 Phase의 평가 함수 `evaluate_set()`을 Phase 2 3× 데이터에 적용). "
         "입력: `revision_profiling/dataset/{test_x3,test_schedules_x3,labels_test}.csv`, artifact는 아래.", "",
         f"**결과: {'PASS — campaign 시작 가능' if allp else 'FAIL — campaign 시작 금지'}**", "",
         "## 재현 표 (3×, N≥3, 12 workloads)", "", "| 방법 | Avg | Regret | Top-1 | Top-5 | 목표 |", "|---|---|---|---|---|---|"]
    tg_txt = {"ours": "0.9930 / 0.0044 / 7/12 / 11/12", "inst": "0.9868 / 0.0106 / 6/12 / 11/12", "lat": "0.9727 / 0.0247 / 6/12", "gpu": "(All-GPU, 참고)",
              "bo5": f"0.9404 ± {float(bo5_std_p2.iloc[0]):.4f} / {bo5_p3['Top-1']}", "orc": "0.9974 / 0 / 12/12"}
    for k, r in tab.items():
        if k == "bo5":
            L.append(f"| {r['method']} | {r['avg_score']:.4f} ± {r['avg_score_std_over_trials']:.4f} | {r['regret']:.4f} | {r['top1_rate']:.3f} (평균 적중률) | N/A | {tg_txt[k]} |")
        else:
            L.append(f"| {r['method']} | {r['avg_score']:.4f} | {r['regret']:.4f} | {r['top1_hits']}/12 | {'' if r['top5_hits'] is None else str(r['top5_hits']) + '/12'} | {tg_txt[k]} |")
    L += ["", "## 검사", "", "| 검사 | 결과 |", "|---|---|"] + [f"| {k} | {'PASS' if v else '**FAIL**'} |" for k, v in chk.items()]
    L += ["", "## Artifact (실제 파일명)", "", "| 파일 | sha256 | 대조 |", "|---|---|---|",
          f"| `{ART_T.relative_to(REPO)}` | `{hashes['T.json']}` | Phase 2 `config_frozen.json` artifact_sha256, Phase 3 `config_frozen.json` xgb_full_artifact와 같음 |",
          f"| `{ART_D.relative_to(REPO)}` | `{hashes['D.json']}` | 같음 |",
          f"| `{ART_I['t_ratio'].relative_to(REPO)}` | `{hashes['per_instance/t_ratio.json']}` | Phase 3 `config_frozen.json`·`REPORT_PHASE3.md`에 per-instance artifact hash 기록이 없다. 대신 후보 504개의 F̂·D̂·T̂_select·Ŝ가 Phase 3 `per_instance_candidates_x3.csv`와 같음을 확인했다 |",
          f"| `{ART_I['q_i'].relative_to(REPO)}` | `{hashes['per_instance/q_i.json']}` | 같음 |", "",
          f"- 후보별 예측 최대 차: Ours(메모리 값 vs Phase 3 CSV) {dmax}, 같은 CSV 저장·읽기를 거친 뒤 {dmax_rt}; Per-instance {dmi}.",
          "  첫 실행에서 Ours 검사를 '메모리 값과 CSV 값의 차 0'으로 두어 FAIL(최대 2.2e-16)이 났다. 두 값을 같은 `to_csv`/`read_csv`로 거치면 504개 모두 차 0이라, "
          "차이는 pandas CSV 실수 표기의 1 ulp 반올림이다. 검사를 같은 경로의 값끼리 비교하도록 고쳤다(평가 규칙·예측값은 바꾸지 않음).",
          "- Workload별 선택 ID와 실측 S(동점 처리 포함)를 Phase 3 `per_workload_all.csv`와 48행 대조했다(`precheck_selections.csv`).",
          "- Best-of-5 std 대조값은 Phase 2 `table3a.csv`(Phase 3 표에는 std가 없다). seed 규칙: (trial, 평가 집합의 정렬된 workload 순번). "
          "Phase 3은 3× 22 workloads 전체를 넣어 돌렸으므로 이 precheck도 22개를 넣었다. 중간 rate에서는 dataset에 N≥3 12 workloads만 있으므로 순번이 0–11이다.",
          "- Ours의 feature는 측정값을 뺀 schedule-only window(model, execution)로 다시 만든 것과 bit 단위로 같다. 204열 밖의 feature는 생기지 않았다.", ""]
    (OUT / "PRECHECK.md").write_text("\n".join(L))
    print("\n".join(L))
    return allp


# ------------------------------------------------------------------------------------------------ final (sections 3.1-3.3)
def final():
    pre = json.loads((OUT / "precheck.json").read_text())
    assert pre["all_pass"], "PRECHECK has not passed"
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "git_status_before.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    hashes = {"T": sha(ART_T), "D": sha(ART_D), "t_ratio": sha(ART_I["t_ratio"]), "q_i": sha(ART_I["q_i"])}
    assert hashes["T"] == pre["artifact_sha256"]["T.json"] and hashes["D"] == pre["artifact_sha256"]["D.json"]
    assert hashes["t_ratio"] == pre["artifact_sha256"]["per_instance/t_ratio.json"] and hashes["q_i"] == pre["artifact_sha256"]["per_instance/q_i.json"]
    cols = frozen_columns()
    m_t, m_d, m_i = load_models()
    # 3x reference through the same path (precheck has shown it equals Phase 3)
    ref = load_dataset(DATA_P2)
    n_ref = ref.groupby("group")["N"].first()
    res3 = evaluate_set(ref, cols, m_t, m_d, m_i)
    g3_all = sorted(n_ref[n_ref >= 3].index)
    fam_of3 = ref.groupby("group")["family"].first()
    # intermediate rates
    dfi = load_dataset(DATA_INT)
    comp = pd.read_csv(DATA_INT / "workload_completeness.csv")
    assert set(dfi["rate"]) <= set(RATES)
    res = {}
    groups = {}
    fams = {}
    infeat = {}
    for rt in RATES:
        sub = dfi[dfi["rate"] == rt].reset_index(drop=True)
        if not len(sub):
            continue
        # per-rate dataset view: same loader output, restricted rows; the per-instance rows come from the whole schedule CSV -> filter after
        r = evaluate_set_rate(sub, cols, m_t, m_d, m_i, rt)
        res[rt] = r
        n_of = sub.groupby("group")["N"].first()
        groups[rt] = sorted(n_of[n_of >= 3].index)
        fams[rt] = sorted(sub.groupby("group")["family"].first()[groups[rt]])
        infeat[rt] = r["checks"]
    log(f"evaluated rates {list(res)}; workloads per rate { {k: len(v) for k, v in groups.items()} }")

    # ---------------- summary table (rates x methods), 3x on the same family subset
    rows, maes = [], []
    for rt in list(res) + ["x3"]:
        if rt == "x3":
            fam_sets = {rt2: set(fams[rt2]) for rt2 in res}
            common = set.intersection(*fam_sets.values()) if fam_sets else set()
            variants = [("x3 (Phase 3, 12 workloads, 3-repeat mean)", g3_all)]
            if any(len(v) < 12 for v in fams.values()):
                variants.append(("x3 (Phase 3, families complete at every new rate)", sorted(g for g in g3_all if fam_of3[g] in common)))
            for lab_, gs in variants:
                for k in ("ours", "inst", "lat", "bo5", "orc"):
                    rows.append({"rate": lab_, "measurement": "3-repeat mean", **summary_row(res3, gs, k)})
                maes.append({"rate": lab_, **mae(res3, gs)})
            continue
        for k in ("ours", "inst", "lat", "bo5", "orc"):
            rows.append({"rate": rt, "measurement": "1 run (rep0)", **summary_row(res[rt], groups[rt], k)})
        maes.append({"rate": rt, **mae(res[rt], groups[rt])})
        if len(groups[rt]) < 12:
            gs = sorted(g for g in g3_all if fam_of3[g] in set(fams[rt]))
            for k in ("ours", "inst", "lat", "bo5", "orc"):
                rows.append({"rate": f"x3 on the {rt} subset", "measurement": "3-repeat mean", **summary_row(res3, gs, k)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_intermediate.csv", index=False)
    mt = pd.DataFrame(maes)
    mt.to_csv(OUT / "mae_intermediate.csv", index=False)

    # ---------------- per-workload table
    pw = []
    for rt in list(res) + ["x3"]:
        r, gs = (res3, g3_all) if rt == "x3" else (res[rt], groups[rt])
        per = {k: v.set_index("group") for k, v in r["per"].items()}
        g_, S_sel, top1, contains = r["bo5"]
        bi = {g: i for i, g in enumerate(g_)}
        for g in gs:
            o, u, l_, ins = per["orc"].loc[g], per["ours"].loc[g], per["lat"].loc[g], per["inst"].loc[g]
            fam = (fam_of3 if rt == "x3" else dfi.groupby("group")["family"].first())[g]
            pw.append({"rate": rt, "family": fam, "group": g, "N": int(o["N"]), "candidates": int(o["n_candidates"]),
                       "oracle_selected": o["selected"], "oracle_S": o["oracle_S"], "n_oracle": int(o["n_oracle"]),
                       "ours_selected": u["selected"], "ours_S": u["S_selected"], "ours_top1": bool(u["top1"]), "ours_top5": bool(u["top5"]), "ours_regret": u["regret"],
                       "latency_selected": l_["selected"], "latency_S": l_["S_selected"], "latency_top1": bool(l_["top1"]),
                       "per_instance_selected": ins["selected"], "per_instance_S": ins["S_selected"], "per_instance_top1": bool(ins["top1"]),
                       "per_instance_top5": bool(ins["top5"]),
                       "bo5_mean_S": float(S_sel[:, bi[g]].mean()), "bo5_std_S": float(S_sel[:, bi[g]].std(ddof=0)), "bo5_top1_rate": float(top1[:, bi[g]].mean()),
                       "ours_minus_allgpu": u["S_selected"] - per["gpu"].loc[g, "S_selected"]})
    pw = pd.DataFrame(pw)
    pw.to_csv(OUT / "per_workload_intermediate.csv", index=False)

    # ---------------- N>=6 paired Ours - All-GPU
    pr = []
    for rt, sub in pw.groupby("rate", sort=False):
        s6 = sub[sub["N"] >= 6]
        d = s6["ours_minus_allgpu"].values
        pr.append({"rate": rt, "workloads_N>=6": len(s6), "mean_diff": float(d.mean()) if len(d) else np.nan,
                   "se_across_workloads": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                   "n_pos": int((d > 1e-9).sum()), "n_zero": int((np.abs(d) <= 1e-9).sum()), "n_neg": int((d < -1e-9).sum()),
                   "per_workload": "; ".join(f"{f}: {x:+.4f}" for f, x in zip(s6["family"], d))})
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "paired_n6_intermediate.csv", index=False)

    # ---------------- selection vs 3x
    sv = []
    for fam in sorted(fam_of3[g3_all].values):
        row = {"family": fam, "N": int(pw.loc[pw["family"] == fam, "N"].iloc[0])}
        for rt in ("x1.5", "x2.5", "x3", "x3.5"):
            s = pw[(pw["family"] == fam) & (pw["rate"] == rt)]
            row[f"oracle_{rt}"] = s["oracle_selected"].iloc[0] if len(s) else "missing"
            row[f"ours_{rt}"] = s["ours_selected"].iloc[0] if len(s) else "missing"
        present = [rt for rt in ("x1.5", "x2.5", "x3", "x3.5") if row[f"oracle_{rt}"] != "missing"]
        row["oracle_changes_across_rates"] = len({row[f"oracle_{rt}"] for rt in present}) > 1
        row["ours_changes_across_rates"] = len({row[f"ours_{rt}"] for rt in present}) > 1
        sv.append(row)
    sv = pd.DataFrame(sv)
    sv.to_csv(OUT / "selection_vs_3x.csv", index=False)

    # ---------------- predictions and checks
    pd.concat([r["pred"] for r in res.values()], ignore_index=True).to_csv(OUT / "predictions_intermediate.csv", index=False)
    pd.concat([r["inst_cand"].assign(rate=rt) for rt, r in res.items()], ignore_index=True).to_csv(OUT / "per_instance_candidates_intermediate.csv", index=False)
    feat_rate = rate_feature_note(dfi, cols)
    summ = {"rates": list(res), "workloads": {k: len(v) for k, v in groups.items()}, "families": fams,
            "excluded_workloads": comp[comp["excluded"]]["workload"].tolist() if "excluded" in comp else [],
            "checks": infeat, "latency_targets": res3["latency_targets"], "artifact_sha256": hashes, "rate_in_features": feat_rate,
            "best_of_5_seed_rule": "SeedSequence([42, trial, w]); w = index of the workload in the sorted group order of the evaluated rate (12 workloads)"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    (OUT / "logs" / "git_status_after.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    log("\n" + tab.to_string(index=False))
    log("\n" + mt.to_string(index=False))
    log("\n" + pr.to_string(index=False))
    log("\n" + sv.to_string(index=False))


def evaluate_set_rate(sub, cols, m_t, m_d, m_i, rt):
    """evaluate_set() on one rate: the per-instance rows are built from the whole schedule CSV and restricted to this rate's groups
    inside a temporary dataset view (a filtered copy of the schedule CSV is not written; IR.rows_from_schedules is called on the full file)."""
    full_rows = IR.rows_from_schedules
    gset = set(sub["group"])

    def rows_rate(path, iso):
        r = full_rows(path, iso)
        return r[r["group"].isin(gset)].reset_index(drop=True)
    IR.rows_from_schedules = rows_rate
    try:
        r = evaluate_set(sub, cols, m_t, m_d, m_i)
    finally:
        IR.rows_from_schedules = full_rows
    return r


def rate_feature_note(dfi, cols):
    """Which features carry the request rate: the infps-valued columns (rate enters only through the schedule infps)."""
    rate_cols = [c for c in cols if "infps" in c or c.startswith("pairhash_")]
    X = D.build_features(dfi, cols)
    out = {"columns_carrying_infps": len(rate_cols), "examples": rate_cols[:6]}
    for rt in RATES:
        m = (dfi["rate"] == rt).values
        if m.any():
            out[rt] = {"views.sum.view.infps_range": [float(X.loc[m, "views.sum.view.infps"].min()), float(X.loc[m, "views.sum.view.infps"].max())]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.precheck:
        sys.exit(0 if precheck() else 1)
    if a.final:
        final()
