"""E (#16): r3 의 집계식 확정 — 측정 시점 정의와 저장된 파일의 정의를 분리해서.

묻는 것:
  - 생성 모델이 2개인 세트(base1-5, S5-S8)에서 r3 는 합인가, 특정 모델인가,
    가속기 위 생성 모델만인가?
  - LLM 이 CPU 에 배치된 창에서 r3 는 어떻게 기록되는가?
  - y3 학습 마스킹(372창)과 정합하는가?

읽기 전용. 저장된 540창×2플랫폼 + 소스만 읽는다.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from xgboost_model.deploy_selector_xgb_suite import _is_vision, _norm_exec  # noqa: E402

GEN = {"llama1b", "qwen2_vl"}
SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
HAS_GEN = {s: any(m in GEN for m in v) for s, v in SETS.items()}

rows = []
for pl in ("gpu", "npu"):
    p = ROOT / f"xgboost_model/full_collection_540/cpu_{pl}/performance_{pl}_full540.json"
    for w in json.loads(p.read_text()):
        views = list(w["models"].values())
        sid = SID_OF[",".join(sorted(v["model"] for v in views))]
        per = {v["model"]: (float(v.get("tokens_per_s") or 0.0),
                            _norm_exec(v["execution"])) for v in views}
        accel_gen = [m for m, (t, d) in per.items() if m in GEN and d in ("gpu", "npu")]
        cpu_gen = [m for m, (t, d) in per.items() if m in GEN and d == "cpu"]
        rows.append({
            "platform": pl, "sid": sid, "rate": float(w["rate_factor"]),
            "combo": str(w["combination"]), "has_gen": HAS_GEN[sid],
            "r3_total": float(w["total"].get("total_tokens_per_s", 0.0)),
            "r3_viewsum": sum(t for t, _ in per.values()),
            "n_gen": len([m for m in per if m in GEN]),
            "accel_gen": accel_gen, "cpu_gen": cpu_gen,
            "nonzero_token_views": [m for m, (t, _) in per.items() if t > 0],
            "vision_with_tokens": [m for m, (t, _) in per.items()
                                   if _is_vision(m) and t > 0],
            "y3_valid": bool(accel_gen),
        })

checks = {}
EPS = 0.06   # 창 총계·뷰별 모두 소수 2자리 반올림 저장

# ---- 저장 파일 안에서 두 값이 갈리는 지점 ----
diff = [r for r in rows if abs(r["r3_total"] - r["r3_viewsum"]) > EPS]
checks["viewsum_equals_total_except_masked_generative"] = all(
    (r["has_gen"] and not r["y3_valid"]) for r in diff)
checks["all_diff_rows_have_viewsum_zero"] = all(r["r3_viewsum"] == 0.0 for r in diff)
checks["no_vision_view_carries_tokens"] = all(not r["vision_with_tokens"] for r in rows)
# 저장 파일에서 토큰이 0이 아닌 뷰는 항상 최대 1개 (재구성 산물)
checks["at_most_one_nonzero_token_view_per_window"] = all(
    len(r["nonzero_token_views"]) <= 1 for r in rows)
checks["nonzero_token_view_is_always_accel_generative"] = all(
    (not r["nonzero_token_views"])
    or (r["nonzero_token_views"][0] in r["accel_gen"]) for r in rows)
# 생성 2모델 창에서도 0이 아닌 뷰는 1개뿐 (=> 저장된 뷰별 값은 분해가 아니다)
gen2 = [r for r in rows if r["n_gen"] == 2]
checks["gen2_windows_never_show_two_nonzero_views"] = all(
    len(r["nonzero_token_views"]) <= 1 for r in gen2)

# ---- CPU 배치 생성 모델 ----
llm_cpu = [r for r in rows if "llama1b" in r["cpu_gen"]]
llm_cpu_only = [r for r in llm_cpu if not r["accel_gen"]]     # S3 의 12창
checks["llm_cpu_windows_390_per_2_platforms"] = len(llm_cpu) == 390
checks["llm_cpu_only_windows_are_12_per_platform"] = all(
    sum(1 for r in llm_cpu_only if r["platform"] == pl) == 12 for pl in ("gpu", "npu"))
checks["llm_cpu_only_windows_are_all_S3"] = {r["sid"] for r in llm_cpu_only} == {"S3"}
cpu_only_totals = [r["r3_total"] for r in llm_cpu_only]
checks["cpu_llm_still_produced_tokens_in_the_total"] = any(t > 0 for t in cpu_only_totals)

# ---- y3 마스킹 정합 ----
n_valid = {pl: sum(1 for r in rows if r["platform"] == pl and r["y3_valid"])
           for pl in ("gpu", "npu")}
stored = json.loads((ROOT / "xgboost_model/full_collection_540/analysis/"
                     "groupkfold_gpu_metrics.json").read_text())
checks["y3_valid_is_372_and_matches_stored"] = (
    n_valid["gpu"] == n_valid["npu"] == 372
    == stored["targets"]["y3"]["n_rows_evaluated"])
checks["masked_168_split_156_vision_only_plus_12_cpu_llm"] = all(
    sum(1 for r in rows if r["platform"] == pl and not r["y3_valid"]) == 168
    for pl in ("gpu", "npu")) and all(
    sum(1 for r in rows if r["platform"] == pl and not r["has_gen"]) == 156
    for pl in ("gpu", "npu"))

# ---- 두 r3 관례가 그룹 정규화에 미치는 영향 ----
# 파이프라인(P1/P4/P5/P6)은 featurize_window 의 뷰별 합을 쓰고,
# 기저 감사·Figure 2 는 창 총계를 쓴다. 갈리는 곳은 S3 뿐이므로 그 영향만 계산.
impact = {}
for key in ("r3_total", "r3_viewsum"):
    g = defaultdict(list)
    for r in rows:
        g[(r["platform"], r["sid"], r["rate"])].append(r[key])
    impact[key] = {f"{k[0]}/{k[1]}@{k[2]}": round(max(v), 4)
                   for k, v in g.items() if k[1] == "S3"}
s3_max_differs = [k for k in impact["r3_total"]
                  if abs(impact["r3_total"][k] - impact["r3_viewsum"][k]) > EPS]
checks["two_conventions_differ_only_in_S3_groups"] = all(r["sid"] == "S3" for r in diff)

facts = {
    "verdict_measurement_time": {
        "definition": "r3 = 창 안 모든 scheduled view 의 avg_tokens_per_s 단순 합",
        "device_filter": "없음 — CPU 에 배치된 생성 모델의 토큰도 포함된다",
        "model_filter": "없음 — vision 핸들러는 그 속성이 없어 0.0 으로 들어갈 뿐",
        "two_generative_models": "둘 다 더해진다 (특정 모델만이 아니다)",
        "unit": "tokens/s",
        "code": [
            "unified_viewer.py:1292-1294 — tokens_by_view = "
            "{v: getattr(<v>_handler, 'avg_tokens_per_s', 0.0) for v in self.view_names}",
            "unified_viewer.py:1296 — total_tokens_per_s = "
            "sum(tokens_by_view[v] for v in scheduled_views)   # execution 조건 없음",
            "unified_viewer.py:1337 — total.total_tokens_per_s 로 기록 (2자리 반올림)",
            "view_handlers.py:357,428,439 — avg_tokens_per_s 는 생성 핸들러만 갖는다",
        ],
    },
    "verdict_stored_file": {
        "warning": "performance_*_full540.json 의 **뷰별** tokens_per_s 는 측정값이 아니라 "
                   "재구성 산물이다. 수집 JSONL 이 창 총계만 보존했기 때문",
        "code": [
            "collect_run.py:122-123 — JSONL 은 뷰별로 (execution, throughput_fps, "
            "inference_count) 만 저장. 뷰별 토큰은 버려진다",
            "jsonl_to_windows.py:32-44 — 창 총계 y3 를 '가속기 위 생성 뷰' 하나에 통째로 "
            "얹고 나머지 뷰는 0. 해당 뷰가 없으면(gen_view is None) 아무 뷰에도 얹지 "
            "않는다 → 뷰별 합 0, 창 총계는 측정값 유지",
            "jsonl_to_windows.py:1-11 docstring — 이 재구성이 의도된 것임을 명시",
        ],
        "consequence": "저장 파일의 뷰별 토큰으로 '어느 생성 모델이 얼마를 냈는지'는 알 수 "
                       "없다. 창 총계만이 실측이다.",
    },
    "two_r3_conventions_in_repo": {
        "pipeline_P1_P4_P5_P6": "featurize_window 가 **뷰별 tokens_per_s 의 합**을 y3 로 쓴다 "
                                "(deploy_selector_xgb_suite.py:323)",
        "basis_audit_and_figure2": "build_basis_audit.load_rows 가 "
                                   "**total.total_tokens_per_s**(창 총계)를 r3 로 쓴다 "
                                   "(runs/20260730_190154_score_basis_audit/"
                                   "build_basis_audit.py:84)",
        "where_they_differ": {
            "n_windows": len(diff),
            "per_platform": {pl: sum(1 for r in diff if r["platform"] == pl)
                             for pl in ("gpu", "npu")},
            "sets": sorted({r["sid"] for r in diff}),
            "max_abs_diff": round(max(abs(r["r3_total"] - r["r3_viewsum"])
                                      for r in diff), 4) if diff else 0.0,
            "rows": [{k: r[k] for k in ("platform", "sid", "rate", "combo",
                                        "r3_total", "r3_viewsum")} for r in diff],
        },
        "group_max_r3_for_S3": {"total_convention": impact["r3_total"],
                                "viewsum_convention": impact["r3_viewsum"],
                                "groups_whose_max_differs": s3_max_differs},
        "note": "차이는 S3(LLM 만 있는 세트)에서 llama1b 가 CPU 에 놓인 창에만 나타난다. "
                "그 창들은 어차피 y3 학습에서 마스킹되지만, 그룹 정규화의 분모(max r3)와 "
                "측정 점수 S 에는 관례에 따라 다르게 들어간다.",
    },
    "generative_two_model_sets": {
        "sets": sorted({r["sid"] for r in gen2}),
        "n_windows": len(gen2),
        "n_windows_with_two_nonzero_stored_views": sum(
            1 for r in gen2 if len(r["nonzero_token_views"]) == 2),
        "explanation": "0 이지만 이는 '두 모델이 동시에 생성하지 않았다'는 뜻이 아니라 "
                       "재구성이 총계를 한 뷰에 몰아넣기 때문이다 (위 verdict_stored_file)",
    },
    "llm_on_cpu": {
        "n_windows_llm_on_cpu": len(llm_cpu),
        "n_windows_llm_on_cpu_and_no_accel_generative": len(llm_cpu_only),
        "stored_per_view_tokens": "전부 0.0 (재구성이 얹지 않음)",
        "window_total_r3_range": [round(float(np.min(cpu_only_totals)), 3),
                                  round(float(np.max(cpu_only_totals)), 3)],
        "window_total_r3_mean": round(float(np.mean(cpu_only_totals)), 3),
        "n_with_zero_total": int(sum(1 for t in cpu_only_totals if t == 0.0)),
    },
    "y3_masking": {
        "y3_valid_per_platform": n_valid,
        "definition": "가속기 위에 생성 뷰가 하나라도 있으면 valid "
                      "(featurize_window, deploy_selector_xgb_suite.py:326-330; "
                      "학습 마스킹은 train_targets 주석 :587-596)",
        "masked_per_platform": 168,
        "masked_breakdown": {"vision_only_sets": 156,
                             "generative_sets_with_all_generative_on_cpu": 12},
        "consistent": True,
    },
    "checks": checks,
}
(HERE / "r3_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False,
                                               default=str))

print("== 검산 ==")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
print(f"\n측정 시점 정의: 모든 뷰의 tokens/s 합 (디바이스·모델 필터 없음)")
print(f"저장 파일: 뷰별 토큰은 재구성 산물. 총계만 실측")
print(f"두 관례가 갈리는 창: {len(diff)}개 "
      f"(플랫폼당 {facts['two_r3_conventions_in_repo']['where_they_differ']['per_platform']}), "
      f"전부 {sorted({r['sid'] for r in diff})}")
print(f"S3 그룹 max r3 가 갈리는 그룹: {s3_max_differs}")
print(f"y3_valid {n_valid}, 마스킹 168 = vision-only 156 + CPU-생성 12")
if not all(checks.values()):
    sys.exit(1)
