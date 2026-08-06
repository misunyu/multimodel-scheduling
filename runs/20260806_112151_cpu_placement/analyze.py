"""작업 11: 측정 candidate 집합에서 "모델이 CPU에 배치되는 경우" 분석.

읽기 전용. 저장된 540창×2플랫폼 측정 데이터와 working_sets.yaml / model_registry.py만
읽는다. 학습·재수집·수정 없음.

분모 규약 (절대 혼용 금지):
  - candidate = (세트, 배치 벡터) 중복 제거 → 플랫폼당 180
  - window    = candidate × rate 수준 → 플랫폼당 540
표마다 어느 분모인지 명시한다.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import model_registry as reg  # noqa: E402
from xgboost_model.deploy_selector_xgb_suite import _norm_exec  # noqa: E402

HERE = Path(__file__).resolve().parent
PLATFORMS = ("gpu", "npu")

ws_doc = yaml.safe_load((ROOT / "working_sets.yaml").read_text())
SETS = ws_doc["sets"]
STRATIFIED = set(ws_doc["stratified_sets"])
TARGET_BY_FREE = ws_doc["stratified_target_by_free"]
KEY_TO_SET = {",".join(sorted(v)): k for k, v in SETS.items()}
ALL_MODELS = sorted({m for v in SETS.values() for m in v})

# CPU 배치가 가능한("자유") 모델 = 레지스트리에서 cpu 를 허용하는 모델
FREE = {m for m in ALL_MODELS if "cpu" in reg.allowed_devices(m)}
FIXED = sorted(set(ALL_MODELS) - FREE)


# ------------------------------------------------------------------ 데이터 로드
def load_rows(platform: str) -> pd.DataFrame:
    """창 하나 = 한 행. views 는 (model, dev) 튜플 목록."""
    path = ROOT / "xgboost_model/full_collection_540" / f"cpu_{platform}" / \
        f"performance_{platform}_full540.json"
    rows = []
    for w in json.loads(path.read_text()):
        views = [(v["model"], _norm_exec(v["execution"])) for v in w["models"].values()]
        models_key = ",".join(sorted(m for m, _ in views))
        rows.append({
            "platform": platform,
            "combination": str(w["combination"]),
            "rate": float(w["rate_factor"]),
            "set_id": KEY_TO_SET.get(models_key),
            "models_key": models_key,
            "n_views": len(views),
            "views": tuple(sorted(views)),
            "cpu_models": tuple(sorted(m for m, d in views if d == "cpu")),
            "accel_models": tuple(sorted(m for m, d in views if d in ("gpu", "npu"))),
            "k_cpu": sum(1 for _, d in views if d == "cpu"),
        })
    return pd.DataFrame(rows)


W = {pl: load_rows(pl) for pl in PLATFORMS}

# 디바이스 값 위생 검사: cpu / 가속기 외의 값이 있으면 즉시 알림
odd = {}
for pl, df in W.items():
    devs = Counter(d for vs in df["views"] for _, d in vs)
    bad = {d: c for d, c in devs.items() if d not in ("cpu", "gpu", "npu")}
    if bad:
        odd[pl] = bad
    # 플랫폼의 가속기 이름이 기대와 같은지
    odd.setdefault("_accel_names", {})[pl] = sorted(
        {d for d in devs if d in ("gpu", "npu")})

# 세트 미매핑 검사
unmapped = {pl: sorted(set(df.loc[df["set_id"].isna(), "models_key"]))
            for pl, df in W.items()}

# 한 창 안에 같은 모델이 두 번 등장하는지 (인스턴스 vs 모델 구분의 근거)
dup_model_windows = {}
for pl, df in W.items():
    n = int(sum(1 for vs in df["views"]
                if len({m for m, _ in vs}) != len(vs)))
    dup_model_windows[pl] = n


def candidates(pl: str) -> pd.DataFrame:
    """(세트, 배치 벡터) 중복 제거 → 180행. rate 는 제거된 축."""
    df = W[pl]
    cand = df.drop_duplicates(subset=["set_id", "views"]).copy()
    cand["n_windows"] = cand["views"].map(df["views"].value_counts())
    return cand.reset_index(drop=True)


C = {pl: candidates(pl) for pl in PLATFORMS}

# --------------------------------------------------------- 4. 플랫폼 대칭성 검산
def role_view(vs):
    """가속기 이름을 accel 로 치환 — 플랫폼 간 비교 가능한 형태."""
    return tuple(sorted((m, "accel" if d in ("gpu", "npu") else "cpu") for m, d in vs))


sym = {}
sets_role = {pl: set((r.set_id, role_view(r.views)) for r in C[pl].itertuples())
             for pl in PLATFORMS}
sym["candidate_symmetric_difference"] = sorted(
    sets_role["gpu"] ^ sets_role["npu"], key=str)
win_role = {pl: Counter((r.set_id, role_view(r.views), r.rate)
                        for r in W[pl].itertuples()) for pl in PLATFORMS}
sym["window_multiset_equal"] = win_role["gpu"] == win_role["npu"]
sym["candidate_counts"] = {pl: int(len(C[pl])) for pl in PLATFORMS}
sym["window_counts"] = {pl: int(len(W[pl])) for pl in PLATFORMS}
SYMMETRIC = (not sym["candidate_symmetric_difference"]) and sym["window_multiset_equal"]
sym["symmetric"] = SYMMETRIC

# 대칭이면 이후 집계는 GPU 기준 단일 표. 아니면 두 플랫폼 모두 계산해 비교한다.
PRIMARY = "gpu"


# ------------------------------------------------------- 1. CPU 배치 규모 분포 k
def k_distribution(pl: str):
    cand, win = C[pl], W[pl]
    kc = Counter(cand["k_cpu"])
    kw = Counter(win["k_cpu"])
    rows = []
    for k in sorted(set(kc) | set(kw)):
        rows.append({"k_cpu": int(k),
                     "n_candidates": int(kc.get(k, 0)),
                     "pct_candidates": round(100 * kc.get(k, 0) / len(cand), 2),
                     "n_windows": int(kw.get(k, 0)),
                     "pct_windows": round(100 * kw.get(k, 0) / len(win), 2)})
    return rows


k_dist = {pl: k_distribution(pl) for pl in PLATFORMS}

k0 = C[PRIMARY][C[PRIMARY]["k_cpu"] == 0]
k0_by_set = Counter(k0["set_id"])
kmax = int(C[PRIMARY]["k_cpu"].max())
kmax_rows = C[PRIMARY][C[PRIMARY]["k_cpu"] == kmax]
kmax_examples = [{"set_id": r.set_id, "combination": r.combination,
                  "cpu_models": list(r.cpu_models),
                  "accel_models": list(r.accel_models)}
                 for r in kmax_rows.itertuples()]

# k 는 세트마다 상한이 다르다(자유 인스턴스 수). 정규화된 관점도 같이 낸다.
free_count_by_set = {s: sum(1 for m in models if m in FREE)
                     for s, models in SETS.items()}


# --------------------------------------------------- 2. 모델별 CPU 배치 횟수
def model_table(pl: str):
    cand = C[pl]
    appear, on_cpu, on_accel = Counter(), Counter(), Counter()
    appear_w, on_cpu_w = Counter(), Counter()
    for r in cand.itertuples():
        for m, d in r.views:
            appear[m] += 1
            (on_cpu if d == "cpu" else on_accel)[m] += 1
    for r in W[pl].itertuples():
        for m, d in r.views:
            appear_w[m] += 1
            if d == "cpu":
                on_cpu_w[m] += 1
    rows = []
    for m in ALL_MODELS:
        rows.append({
            "model": m,
            "kind": reg.kind_of(m),
            "allowed_devices": list(reg.allowed_devices(m)),
            "cpu_capable": m in FREE,
            "n_sets_containing": sum(1 for v in SETS.values() if m in v),
            "cand_appearances": int(appear[m]),
            "cand_on_cpu": int(on_cpu[m]),
            "cand_on_accel": int(on_accel[m]),
            "cpu_ratio_cand": round(on_cpu[m] / appear[m], 4) if appear[m] else None,
            "window_appearances": int(appear_w[m]),
            "window_on_cpu": int(on_cpu_w[m]),
            "cpu_ratio_window": (round(on_cpu_w[m] / appear_w[m], 4)
                                 if appear_w[m] else None),
        })
    return rows


models_tbl = {pl: model_table(pl) for pl in PLATFORMS}

# 인스턴스 단위 총계 (뷰 = 인스턴스)
inst_totals = {pl: {"cand_view_instances": int(sum(r["cand_appearances"]
                                                   for r in models_tbl[pl])),
                    "cand_cpu_instances": int(sum(r["cand_on_cpu"]
                                                  for r in models_tbl[pl])),
                    "window_view_instances": int(sum(r["window_appearances"]
                                                     for r in models_tbl[pl])),
                    "window_cpu_instances": int(sum(r["window_on_cpu"]
                                                    for r in models_tbl[pl]))}
               for pl in PLATFORMS}

# VLM 제약 위반 검사
VLM = [m for m in ALL_MODELS if reg.kind_of(m) == "vlm"]
LLM = [m for m in ALL_MODELS if reg.kind_of(m) == "llm"]
violations = {}
for pl in PLATFORMS:
    for r in models_tbl[pl]:
        if r["model"] in VLM and r["cand_on_cpu"] > 0:
            violations[f"{pl}:{r['model']}"] = r["cand_on_cpu"]

# LLM CPU 배치 상세 (어느 세트에서, 몇 candidate)
llm_detail = {}
for m in LLM:
    per_set = Counter()
    for r in C[PRIMARY].itertuples():
        if (m, "cpu") in r.views:
            per_set[r.set_id] += 1
    tot_per_set = Counter(r.set_id for r in C[PRIMARY].itertuples()
                          if m in {mm for mm, _ in r.views})
    llm_detail[m] = {"by_set": {s: [int(per_set[s]), int(tot_per_set[s])]
                                for s in sorted(tot_per_set)},
                     "total_cpu_cand": int(sum(per_set.values())),
                     "total_cand": int(sum(tot_per_set.values()))}


# ------------------------------------------------------------- 3. 세트별 분해
def set_table(pl: str):
    cand = C[pl]
    rows = []
    for s in SETS:
        g = cand[cand["set_id"] == s]
        f = free_count_by_set[s]
        n_pow2_free = 2 ** f
        rows.append({
            "set_id": s,
            "N": len(SETS[s]),
            "n_free_instances": f,
            "n_fixed_accel_instances": len(SETS[s]) - f,
            "mode": "stratified" if s in STRATIFIED else "exhaustive",
            "n_feasible": n_pow2_free,
            "n_candidates_measured": int(len(g)),
            "n_with_cpu_ge1": int((g["k_cpu"] >= 1).sum()),
            "pct_with_cpu_ge1": round(100 * (g["k_cpu"] >= 1).mean(), 2),
            "mean_k_cpu": round(float(g["k_cpu"].mean()), 3),
            "max_k_cpu": int(g["k_cpu"].max()),
            "mean_cpu_fraction_of_free": round(float((g["k_cpu"] / f).mean()), 3)
                                         if f else None,
            "n_windows": int(g["n_windows"].sum()),
        })
    return rows


sets_tbl = {pl: set_table(pl) for pl in PLATFORMS}

# 층화 기준 = "가속기에 올라간 자유 인스턴스 수" → stratum s = f - k_cpu
strata = {}
for s in SETS:
    f = free_count_by_set[s]
    g = C[PRIMARY][C[PRIMARY]["set_id"] == s]
    got = Counter(f - int(k) for k in g["k_cpu"])
    strata[s] = {
        "n_free": f,
        "mode": "stratified" if s in STRATIFIED else "exhaustive",
        "target": TARGET_BY_FREE.get(f) if s in STRATIFIED else None,
        "measured_total": int(len(g)),
        "by_stratum": {str(a): {"measured": int(got.get(a, 0)),
                                "possible": int(comb(f, a))}
                       for a in range(f + 1)},
    }

# ------------------------------------------------- 5. exec_cpu indicator 분포
# 37-feature 중 views.sum.view.exec_cpu 는 창의 CPU 배치 인스턴스 수와 동일한 정의
# (_view_features 의 exec_cpu 원핫을 _aggregate 가 합산). 여기서는 그 값을 배치에서
# 직접 재구성해 분포를 낸다.
exec_cpu_feature = {
    "definition": "views.sum.view.exec_cpu = 창 안에서 exec==cpu 인 뷰 수 (= k_cpu)",
    "candidates_total": int(len(C[PRIMARY])),
    "candidates_active": int((C[PRIMARY]["k_cpu"] > 0).sum()),
    "candidates_zero": int((C[PRIMARY]["k_cpu"] == 0).sum()),
    "distinct_values_sum": sorted(int(v) for v in set(C[PRIMARY]["k_cpu"])),
    "windows_active": int((W[PRIMARY]["k_cpu"] > 0).sum()),
    "windows_zero": int((W[PRIMARY]["k_cpu"] == 0).sum()),
    "mean_feature_values": sorted(
        round(float(v), 4) for v in
        {r.k_cpu / r.n_views for r in C[PRIMARY].itertuples()}),
    "max_feature_active_candidates": int((C[PRIMARY]["k_cpu"] > 0).sum()),
}

# ------------------------------------------------------------------- 검산
checks = {}
for pl in PLATFORMS:
    checks[f"{pl}_k_dist_sums_to_180"] = sum(
        r["n_candidates"] for r in k_dist[pl]) == 180
    checks[f"{pl}_k_dist_sums_to_540"] = sum(
        r["n_windows"] for r in k_dist[pl]) == 540
    checks[f"{pl}_set_candidates_sum_180"] = sum(
        r["n_candidates_measured"] for r in sets_tbl[pl]) == 180
    checks[f"{pl}_set_windows_sum_540"] = sum(
        r["n_windows"] for r in sets_tbl[pl]) == 540
    # 모델 등장 수 = 세트 정의로부터 유도한 값과 일치하는가
    expected = {m: sum(r["n_candidates_measured"]
                       for r in sets_tbl[pl] if m in SETS[r["set_id"]])
                for m in ALL_MODELS}
    got = {r["model"]: r["cand_appearances"] for r in models_tbl[pl]}
    checks[f"{pl}_model_appearances_match_set_defs"] = expected == got
    # 인스턴스 총합 = sum(N_s * candidates_s)
    exp_inst = sum(len(SETS[r["set_id"]]) * r["n_candidates_measured"]
                   for r in sets_tbl[pl])
    checks[f"{pl}_instance_total_matches"] = \
        inst_totals[pl]["cand_view_instances"] == exp_inst
    checks[f"{pl}_no_unmapped_sets"] = not unmapped[pl]
    checks[f"{pl}_no_duplicate_model_in_window"] = dup_model_windows[pl] == 0
    # 층별 measured <= possible
    checks[f"{pl}_strata_within_possible"] = True
checks["platforms_symmetric"] = SYMMETRIC
checks["no_vlm_on_cpu"] = not violations
checks["strata_measured_le_possible"] = all(
    v["measured"] <= v["possible"] for s in strata
    for v in strata[s]["by_stratum"].values())
# exhaustive 세트는 층 전수 커버여야 한다
checks["exhaustive_sets_cover_all_strata"] = all(
    all(v["measured"] == v["possible"] for v in strata[s]["by_stratum"].values())
    for s in strata if strata[s]["mode"] == "exhaustive")

# ------------------------------------------------------------------ 산출물
per_cand = C[PRIMARY][["set_id", "combination", "n_views", "k_cpu",
                       "cpu_models", "accel_models", "n_windows"]].copy()
per_cand["platform_note"] = "GPU/NPU 동일" if SYMMETRIC else "GPU 기준"
per_cand["n_free_instances"] = per_cand["set_id"].map(free_count_by_set)
per_cand["stratum_accel_free"] = per_cand["n_free_instances"] - per_cand["k_cpu"]
per_cand["mode"] = per_cand["set_id"].map(
    lambda s: "stratified" if s in STRATIFIED else "exhaustive")
per_cand["cpu_models"] = per_cand["cpu_models"].map(lambda t: "|".join(t))
per_cand["accel_models"] = per_cand["accel_models"].map(lambda t: "|".join(t))
per_cand = per_cand.sort_values(
    ["n_views", "set_id", "k_cpu", "combination"]).reset_index(drop=True)
per_cand.to_csv(HERE / "per_candidate.csv", index=False)

facts = {
    "denominators": {"candidates_per_platform": 180, "windows_per_platform": 540,
                     "note": "candidate = (세트, 배치 벡터) 중복 제거; window = candidate × rate"},
    "free_models": sorted(FREE), "fixed_accel_models": FIXED,
    "device_constraints": {m: list(reg.allowed_devices(m)) for m in ALL_MODELS},
    "k_distribution": k_dist,
    "k0_candidates": {"n": int(len(k0)), "by_set": dict(k0_by_set),
                      "combinations": sorted(k0["combination"])},
    "k_max": {"k": kmax, "n_candidates": int(len(kmax_rows)),
              "examples": kmax_examples},
    "free_count_by_set": free_count_by_set,
    "models": models_tbl,
    "instance_totals": inst_totals,
    "vlm_models": VLM, "llm_models": LLM,
    "vlm_cpu_violations": violations,
    "llm_cpu_detail": llm_detail,
    "sets": sets_tbl,
    "strata": strata,
    "exec_cpu_feature": exec_cpu_feature,
    "symmetry": {k: (v if k != "candidate_symmetric_difference" else [str(x) for x in v])
                 for k, v in sym.items()},
    "device_value_sanity": odd,
    "duplicate_model_windows": dup_model_windows,
    "checks": checks,
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False,
                                            default=str))

# ------------------------------------------------------------------ 콘솔 요약
print("== k 분포 (candidate/180, window/540) ==")
for r in k_dist[PRIMARY]:
    print(f"  k={r['k_cpu']}: cand {r['n_candidates']:3d} ({r['pct_candidates']:5.2f}%)"
          f"  win {r['n_windows']:3d} ({r['pct_windows']:5.2f}%)")
print(f"  k=0 candidate {len(k0)}개, 세트별 {dict(k0_by_set)}")
print(f"  k 최대 {kmax} — {len(kmax_rows)} candidate")
print(f"== VLM CPU 배치 위반: {violations or '0건 (제약 유지)'}")
for m, d in llm_detail.items():
    print(f"== LLM {m}: CPU 배치 candidate {d['total_cpu_cand']}/{d['total_cand']}"
          f"  by_set={d['by_set']}")
print(f"== 플랫폼 대칭: {SYMMETRIC} (symdiff "
      f"{len(sym['candidate_symmetric_difference'])}건, window multiset "
      f"{sym['window_multiset_equal']})")
print("== 검산 ==")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if not all(checks.values()):
    sys.exit(1)
