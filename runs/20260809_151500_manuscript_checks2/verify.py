"""작업 17 검증: 리포트의 줄 근거·수치 재확인 + facts.json. 읽기 전용."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))
import analysis_common as ac  # noqa: E402

problems, checks = [], {}


def lines(rel):
    return (ROOT / rel).read_text().splitlines()


AC = lines("xgboost_model/full_collection_540/scripts/analysis_common.py")
TGK = lines("xgboost_model/full_collection_540/scripts/train_groupkfold.py")
TUN = lines("xgboost_model/full_collection_540/scripts/train_unified.py")
SU = lines("xgboost_model/deploy_selector_xgb_suite.py")
J2W = lines("xgboost_model/full_collection_540/scripts/jsonl_to_windows.py")
CR = lines("xgboost_model/full_collection_540/scripts/collect_run.py")

CITES = [
    (TGK, 56, 'valid = y3_valid if tag == "y3"'),
    (TGK, 68, "y3 boosters are trained on valid rows only but PREDICT every held-out"),
    (TGK, 71, "pred[te] = ac.predict_booster(bst, Xv[te])"),
    (TUN, 56, 'valid = y3_valid if tag == "y3"'),
    (TUN, 64, "pred[te] = ac.predict_booster(bst, Xv[te])"),
    (AC, 186, "np.nan_to_num"),
    (AC, 188, "y1 - alpha * y2 + beta * y3 * hg"),
    (AC, 191, "def ranking_metrics"),
    (AC, 228, "def cv_select_params"),
    (AC, 229, "rounds: int = 300"),
    (AC, 239, "inner_fold = assign_group_folds"),
    (AC, 240, "n_folds=3, seed=seed"),
    (AC, 245, "for k in range(3)"),
    (AC, 252, "maes.append(float(np.mean(np.abs(pred - yv[te]))))"),
    (AC, 254, "if mae < best_mae"),
    (SU, 323, "y3_tokens += float(view.get"),
    (SU, 519, "_PARAMS = {"),
    (SU, 525, "_PARAM_GRID = ["),
    (SU, 534, "def _cv_select_params"),
    (SU, 537, "fold_id = np.arange(n) % folds"),
    (SU, 795, "predict_targets(prefix, X, with_y3=gen)"),
    (SU, 794, "combo_has_generative(combo_blob)"),
    (J2W, 35, 'reg.kind_of(model) in ("llm", "vlm") and execu.lower() in ("npu", "gpu")'),
    (J2W, 43, 'v["tokens_per_s"] = d["y3"]'),
    (CR, 122, '"models": {v["model"]: (v["execution"], v["throughput_fps"]'),
]
for src, ln, needle in CITES:
    got = src[ln - 1] if 0 < ln <= len(src) else "<범위 밖>"
    if needle not in got:
        problems.append(f"줄 근거 불일치: {ln} 에 {needle!r} 없음 -> {got.strip()!r}")

# ---- 그리드·기본값 실제 값 -----------------------------------------------------
from xgboost_model.deploy_selector_xgb_suite import _PARAMS, _PARAM_GRID  # noqa: E402

grid = [(g["max_depth"], g["eta"]) for g in _PARAM_GRID]
checks["grid_is_the_four_documented_points"] = grid == [(3, 0.1), (4, 0.1),
                                                        (5, 0.05), (6, 0.1)]
checks["objective_is_reg_squarederror"] = _PARAMS["objective"] == "reg:squarederror"
checks["fixed_params_as_documented"] = (
    _PARAMS["subsample"] == 0.8 and _PARAMS["colsample_bytree"] == 0.8
    and _PARAMS["min_child_weight"] == 3.0 and _PARAMS["seed"] == 42
    and _PARAMS["eval_metric"] == "rmse")
checks["inner_cv_uses_the_same_group_fold_function"] = (
    "assign_group_folds" in AC[238] or "assign_group_folds" in AC[239])

# ---- CPU-only 생성 후보 --------------------------------------------------------
S3_MODELS = "llama1b,resnet50,yolo11s"
detail = {}
for pl in ac.PLATFORMS:
    o = pd.read_csv(ac.ANALYSIS / f"groupkfold_{pl}_oof.csv", comment="#")
    s3 = o[o["models"] == S3_MODELS]
    masked = s3[~s3["y3_valid"].astype(bool)]
    detail[pl] = {
        "n_s3_rows": int(len(s3)),
        "n_cpu_only_generative_rows": int(len(masked)),
        "y3_pred_min": round(float(masked["y3_pred"].min()), 4),
        "y3_pred_max": round(float(masked["y3_pred"].max()), 4),
        "y3_meas_unique": sorted({float(x) for x in masked["y3_meas"]}),
        "has_gen_all_true": bool(masked["has_gen"].all()),
    }
    checks[f"{pl}_cpu_only_generative_is_12_rows"] = len(masked) == 12
    checks[f"{pl}_y3_pred_is_nonzero_for_them"] = bool(masked["y3_pred"].min() > 0)
    checks[f"{pl}_y3_meas_is_exactly_zero_for_them"] = (
        detail[pl]["y3_meas_unique"] == [0.0])
    checks[f"{pl}_has_gen_true_so_beta_term_applies"] = detail[pl]["has_gen_all_true"]
    # 랭킹 포함 여부: 그룹 크기가 8 (배제되지 않았다)
    sizes = s3.groupby("rate_factor").size().unique().tolist()
    checks[f"{pl}_s3_groups_keep_all_8_candidates"] = sizes == [8]
    # S3 그룹의 Top-1
    g = pd.read_csv(HERE.parent / "20260809_140605_checklist_r2" / "F_regret"
                    / f"per_group_direct_{pl}.csv")
    s3g = g[g["group"].str.contains(S3_MODELS)]
    checks[f"{pl}_s3_three_groups_present_in_ranking"] = len(s3g) == 3
    checks[f"{pl}_s3_all_top1_correct"] = bool((s3g["top1"] == 1).all())
    detail[pl]["s3_group_top1"] = s3g["top1"].tolist()
    detail[pl]["s3_group_oracle_ratio"] = [round(float(x), 4) for x in s3g["ratio"]]

# ---- (c) 창 총계 vs 뷰별 합 ----------------------------------------------------
e = json.loads((HERE.parent / "20260809_140605_checklist_r2" / "E_r3_definition"
                / "r3_facts.json").read_text())
diff = e["two_r3_conventions_in_repo"]["where_they_differ"]
checks["window_total_nonzero_while_viewsum_zero"] = (
    diff["n_windows"] == 18 and diff["sets"] == ["S3"])
checks["group_max_r3_unchanged_by_convention"] = (
    e["two_r3_conventions_in_repo"]["group_max_r3_for_S3"]["groups_whose_max_differs"] == [])

for k, v in checks.items():
    if not v:
        problems.append(f"검산 실패: {k}")

facts = {
    "q1_cpu_only_generative_candidates": {
        "who": "생성 세트인데 생성 뷰가 전부 CPU — S3 의 4배치 × 3 rate = 플랫폼당 12창",
        "a_prediction_path": {
            "y3_trained_on": "y3_valid 행만 (train_groupkfold.py:56 / train_unified.py:56)",
            "y3_predicted_for": "held-out fold 전 행 (train_groupkfold.py:71 / :64) — "
                                "마스크 없음",
            "beta_term": "has_gen 은 세트 단위 → S3 은 True → beta*yhat3 가 더해진다 "
                         "(analysis_common.py:188)",
            "special_handling": "없음 (0 대입도, beta 항 생략도 아니다)",
            "production_path": "deploy_selector_xgb_suite.py:794-795 도 세트 단위 "
                               "combo_has_generative 로 with_y3 를 정한다",
        },
        "b_included_in_ranking": {
            "included": True,
            "why": "ranking_metrics(analysis_common.py:191-212)에 배제 필터 없음",
            "evidence": {pl: {"group_size": 8,
                              "s3_groups": 3,
                              "top1": detail[pl]["s3_group_top1"],
                              "oracle_ratio": detail[pl]["s3_group_oracle_ratio"]}
                         for pl in ac.PLATFORMS},
        },
        "c_measured_score": {
            "uses_window_total_r3": False,
            "actual": "featurize_window 의 뷰별 tokens_per_s 합 "
                      "(deploy_selector_xgb_suite.py:323) = 이 후보들에서 0.0",
            "why": "수집 JSONL 이 뷰별 토큰 미보존(collect_run.py:122-123) + "
                   "jsonl_to_windows.py:35,43 가 총계를 '가속기 생성 뷰'에만 얹는데 "
                   "이 후보들에는 그 뷰가 없다",
            "window_total_range_tok_s": e["llm_on_cpu"]["window_total_r3_range"],
            "window_total_mean_tok_s": e["llm_on_cpu"]["window_total_r3_mean"],
            "numeric_impact": "없음 — S3 그룹의 max(r3) 가 두 관례에서 동일",
        },
        "per_platform_detail": detail,
    },
    "q2_hyperparameter_selection": {
        "grid": [{"max_depth": d, "eta": e_} for d, e_ in grid],
        "fixed_params": {k: _PARAMS[k] for k in sorted(_PARAMS)},
        "rounds": 300,
        "inner_cv_folds": 3,
        "inner_split": "assign_group_folds — 바깥 fold 와 같은 함수·같은 그룹 키 "
                       "(models, rate_factor), seed 42. 그룹 존중",
        "selection_metric": "MAE (3 inner fold 평균) 최소화",
        "objective_vs_selection": "학습 목적함수는 reg:squarederror, 선택 기준은 MAE — 다르다",
        "scope": "바깥 fold 마다 · 타깃(y1/y2/y3) 마다 독립 수행",
        "shipped_path_contrast": "deploy_selector_xgb_suite._cv_select_params(:534)는 "
                                 "같은 그리드지만 inner fold 가 arange(n)%folds — 그룹 미존중",
        "code": {"cv_select_params": "analysis_common.py:228-256",
                 "grid": "deploy_selector_xgb_suite.py:525-530",
                 "fixed": "deploy_selector_xgb_suite.py:519-523"},
    },
    "checks": checks,
    "problems": problems,
    "input_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                     for rel in ["xgboost_model/full_collection_540/scripts/analysis_common.py",
                                 "xgboost_model/full_collection_540/scripts/train_groupkfold.py",
                                 "xgboost_model/full_collection_540/scripts/train_unified.py",
                                 "xgboost_model/full_collection_540/scripts/jsonl_to_windows.py",
                                 "xgboost_model/deploy_selector_xgb_suite.py"]},
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))

print("grid:", grid)
print("inner CV: 3-fold, group-aware, metric = MAE, objective =", _PARAMS["objective"])
for pl in ac.PLATFORMS:
    d = detail[pl]
    print(f"{pl}: CPU-only 생성 {d['n_cpu_only_generative_rows']}창, "
          f"yhat3 {d['y3_pred_min']}~{d['y3_pred_max']}, y3_meas {d['y3_meas_unique']}, "
          f"S3 Top-1 {d['s3_group_top1']}")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
