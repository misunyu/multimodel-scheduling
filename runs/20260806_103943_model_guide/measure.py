"""작업 10-r3: MODEL_GUIDE.md 근거 실측.

읽기 전용. 저장소의 저장된 아티팩트/수집 데이터만 읽고, MODEL_GUIDE.md 2-A(파일 표)와
2-B(규모 수치)에 들어갈 값을 전부 실측해 measurements.json 으로 떨궈 놓는다.
학습·재수집 없음.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import analysis_common as ac  # noqa: E402
import train_decomposed as td  # noqa: E402
from xgboost_model.deploy_selector_xgb_suite import load_static_profiles  # noqa: E402

OUT = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(p: Path) -> dict:
    import datetime as dt
    st = p.stat()
    return {"size_bytes": st.st_size,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime).astimezone()
                       .isoformat(timespec="seconds")}


# ---------------------------------------------------------------- 2-A 파일 표
DATA_FILES = [
    # (상대경로, 역할, 생성 주체, 소비자)
    ("xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
     "CPU-GPU 원시 측정 창 540개 (학습·평가의 유일한 측정 원본)",
     "scripts/collect_run.py -> jsonl_to_windows.py", "P1(gpu), joint, P6(gpu)"),
    ("xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
     "CPU-NPU 원시 측정 창 540개",
     "scripts/collect_run.py -> jsonl_to_windows.py", "P1(npu), joint, P6(npu)"),
    ("xgboost_model/full_collection_540/cpu_gpu/collect_gpu_results.jsonl",
     "수집 원본 JSONL (창 변환 전)", "scripts/collect_run.py", "(변환 입력, 학습 직접 소비 없음)"),
    ("xgboost_model/full_collection_540/cpu_npu/collect_npu_results.jsonl",
     "수집 원본 JSONL (창 변환 전)", "scripts/collect_run.py", "(변환 입력, 학습 직접 소비 없음)"),
    ("xgboost_model/full_collection_540/cpu_gpu/collect_gpu_anchor.jsonl",
     "수집 앵커(단독 실행) JSONL", "scripts/collect_run.py", "(참고용)"),
    ("xgboost_model/full_collection_540/cpu_npu/collect_npu_anchor.jsonl",
     "수집 앵커(단독 실행) JSONL", "scripts/collect_run.py", "(참고용)"),
    ("xgboost_model/full_collection_540/cpu_gpu/oof_gpu.json",
     "출하 학습 경로(train_gpu.py)의 OOF 예측 백업", "scripts/train_gpu.py", "(레거시 참고)"),
    ("xgboost_model/full_collection_540/cpu_npu/oof_npu.json",
     "출하 학습 경로(train_npu.py)의 OOF 예측 백업", "scripts/train_npu.py", "(레거시 참고)"),
    ("working_sets.yaml", "15개 워킹셋 정의 + rate factor 규칙 + 층화 설정",
     "수작업", "gen_collection_schedules.py (간접: 스케줄 YAML 경유)"),
    ("model_registry.py", "모델 종류/허용 디바이스 (DEVICE_CONSTRAINTS)",
     "수작업", "gen_collection_schedules.py, deploy_selector_xgb_suite.py"),
    ("xgboost_model/schedules/collection/collect_cpu_gpu.yaml",
     "CPU-GPU 수집 스케줄 (combination -> 모델·디바이스·infps)",
     "gen_collection_schedules.py", "P1(gpu), joint, P6(gpu) — infps feature 원천"),
    ("xgboost_model/schedules/collection/collect_cpu_npu.yaml",
     "CPU-NPU 수집 스케줄", "gen_collection_schedules.py",
     "P1(npu), joint, P6(npu) — infps feature 원천"),
    ("xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json",
     "단독(isolated) 정적 프로파일: 모델×디바이스 추론시간/로드시간/토큰속도",
     "단독 프로파일링(외부)", "P1, joint, P6, P5 greedy — 전부 공유"),
    ("xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
     "P1(gpu) 결과 + group_to_fold (train/test 분할 정의)",
     "scripts/train_groupkfold.py", "P6(gpu) fold 상속"),
    ("xgboost_model/full_collection_540/analysis/groupkfold_npu_metrics.json",
     "P1(npu) 결과 + group_to_fold", "scripts/train_groupkfold.py", "P6(npu) fold 상속"),
    ("xgboost_model/full_collection_540/analysis/groupkfold_gpu_oof.csv",
     "P1(gpu) OOF 예측 (y3_pred 포함)", "scripts/train_groupkfold.py",
     "P6(gpu) ŷ3 공유, P3 대각선"),
    ("xgboost_model/full_collection_540/analysis/groupkfold_npu_oof.csv",
     "P1(npu) OOF 예측", "scripts/train_groupkfold.py", "P6(npu) ŷ3 공유, P3 대각선"),
    ("xgboost_model/artifacts/deploy_cpu_gpu_features.json",
     "정규 37-feature 순서 (FEATURE_ORDER의 실제 소스)", "출하 학습 경로",
     "P1, joint, P3 (열 순서 고정)"),
]

files_rows = []
for rel, role, producer, consumer in DATA_FILES:
    p = ROOT / rel
    row = {"path": rel, "role": role, "producer": producer, "consumer": consumer,
           "exists": p.exists()}
    if p.exists():
        row.update(stat(p))
        row["sha256"] = sha256(p)
    files_rows.append(row)

# ------------------------------------------------- 2-B(a,b) 후보 배치 / rate 수준
ws = yaml.safe_load((ROOT / "working_sets.yaml").read_text())
set_of_models = {",".join(sorted(v)): k for k, v in ws["sets"].items()}

per_platform = {}
for pl in ac.PLATFORMS:
    windows = ac.load_windows(pl)
    rows = []
    for w in windows:
        models = sorted(v["model"] for v in w["models"].values())
        devs = tuple(sorted((v["model"], ac._norm_exec(v["execution"]))
                            for v in w["models"].values()))
        rows.append({"combination": str(w["combination"]),
                     "models_key": ",".join(models),
                     "placement": devs,
                     "rate": w.get("rate_factor")})
    df = pd.DataFrame(rows)
    df["set_id"] = df["models_key"].map(set_of_models)
    per_platform[pl] = df

# 세트 식별: models_key -> set_id (working_sets.yaml 과 대조)
unmapped = {pl: sorted(set(df.loc[df["set_id"].isna(), "models_key"]))
            for pl, df in per_platform.items()}

candidates = {}
for pl, df in per_platform.items():
    by_set = {}
    for sid, g in df.groupby("set_id"):
        by_set[sid] = {
            "n_windows": int(len(g)),
            "n_unique_placements": int(g["placement"].nunique()),
            "rate_levels": sorted(float(r) for r in set(g["rate"])),
            "n_rate_levels": int(g["rate"].nunique()),
            "windows_per_rate": {str(float(r)): int(c)
                                 for r, c in sorted(Counter(g["rate"]).items())},
        }
    candidates[pl] = {
        "total_windows": int(len(df)),
        "total_unique_placements": int(df["placement"].nunique()),
        "n_sets": int(df["set_id"].nunique()),
        "n_groups_models_x_rate": int(df.groupby(["models_key", "rate"]).ngroups),
        "by_set": by_set,
    }

# 두 플랫폼의 후보 배치 집합이 (모델→디바이스 역할 수준에서) 같은지
def role_placement(devs):
    # gpu/npu 를 'accel' 로 치환해 플랫폼 간 비교 가능하게
    return tuple(sorted((m, "accel" if d in ("gpu", "npu") else d) for m, d in devs))

sets_by_pl = {pl: set(df["placement"].map(role_placement)) for pl, df in per_platform.items()}
identical_candidates = sets_by_pl["gpu"] == sets_by_pl["npu"]

# rate factor 규칙 대조
rate_rule = ws["rate_factors_by_size"]
rate_check = {}
for sid, models in ws["sets"].items():
    n = len(models)
    if n <= rate_rule["small"]["max_models"]:
        expect = [float(x) for x in rate_rule["small"]["factors"]]
        bucket = "small(<=2)"
    elif n <= rate_rule["medium"]["max_models"]:
        expect = [float(x) for x in rate_rule["medium"]["factors"]]
        bucket = "medium(3-5)"
    else:
        expect = [float(x) for x in rate_rule["large"]["factors"]]
        bucket = "large(>=6)"
    actual = {pl: candidates[pl]["by_set"].get(sid, {}).get("rate_levels")
              for pl in ac.PLATFORMS}
    rate_check[sid] = {"N": n, "bucket": bucket, "expected": expect,
                       "actual": actual,
                       "match": all(a == expect for a in actual.values())}

# ------------------------------------------------------- 2-B(c,d) 학습 행/fold
S = load_static_profiles(ac.STATIC_JSON)
model_rows = {}
folds = {}

for pl in ac.PLATFORMS:
    X, Yraw, M = ac.load_dataset(pl)
    groups = ac.group_keys(M)
    fold = ac.assign_group_folds(groups)
    y3_valid = M["y3_valid"].values.astype(bool)
    model_rows[f"P1_{pl}"] = {
        "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "n_groups": int(groups.nunique()),
        "n_y3_valid_rows": int(y3_valid.sum()),
        "n_has_gen_rows": int(M["has_gen"].sum()),
    }
    gf = pd.DataFrame({"group": groups.astype(str), "fold": fold})
    per_fold = []
    for f in range(ac.N_FOLDS):
        te = gf["fold"] == f
        per_fold.append({
            "fold": f,
            "test_groups": int(gf.loc[te, "group"].nunique()),
            "test_rows": int(te.sum()),
            "train_groups": int(gf.loc[~te, "group"].nunique()),
            "train_rows": int((~te).sum()),
            "train_rows_y3": int(((~te).values & y3_valid).sum()),
        })
    folds[f"P1_{pl}"] = per_fold
    # P1 metrics json 의 group_to_fold 와 동일한지
    p1 = json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
    stored = p1["group_to_fold"]
    recomputed = {str(g): int(f) for g, f in
                  sorted(set(zip(groups.values, fold.values)), key=str)}
    folds[f"P1_{pl}_matches_stored_group_to_fold"] = (stored == recomputed)
    folds[f"P1_{pl}_stored_fold_row_counts"] = p1["fold_row_counts"]

# joint
Xs, Ms = [], []
for pl in ac.PLATFORMS:
    X, Yraw, M = ac.load_dataset(pl)
    Xs.append(X); Ms.append(M)
Xj = pd.concat(Xs, ignore_index=True)
Mj = pd.concat(Ms, ignore_index=True)
gj = ac.group_keys(Mj, with_platform=True)
fj = ac.assign_group_folds(gj)
model_rows["joint"] = {"n_rows": int(len(Xj)), "n_features": int(Xj.shape[1]),
                       "n_groups": int(gj.nunique()),
                       "n_y3_valid_rows": int(Mj["y3_valid"].astype(bool).sum())}
gfj = pd.DataFrame({"group": gj.astype(str), "fold": fj})
folds["joint"] = [{"fold": f,
                   "test_groups": int(gfj.loc[gfj["fold"] == f, "group"].nunique()),
                   "test_rows": int((gfj["fold"] == f).sum()),
                   "train_groups": int(gfj.loc[gfj["fold"] != f, "group"].nunique()),
                   "train_rows": int((gfj["fold"] != f).sum())}
                  for f in range(ac.N_FOLDS)]
folds["joint_stored_fold_row_counts"] = json.loads(
    (ac.ANALYSIS / "unified_metrics.json").read_text())["fold_row_counts"]

# decomposed (P6): 뷰 행
for pl in ac.PLATFORMS:
    V = td.build_view_dataset(pl, S)
    _, _, M = ac.load_dataset(pl)
    w = pd.DataFrame({
        "combination": M["combination"].astype(str),
        "group": [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])]})
    p1 = json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
    w["fold"] = w["group"].map(p1["group_to_fold"])
    V = V.merge(w, on="combination", how="left")
    vis = V["is_vision"].values.astype(bool)
    model_rows[f"P6_{pl}"] = {
        "n_view_rows_total": int(len(V)),
        "n_vision_view_rows": int(vis.sum()),
        "n_gen_view_rows": int((~vis).sum()),
        "n_features": len(td.VIEW_FEATURES),
        "n_groups": int(V.loc[vis, "group"].nunique()),
        "n_trainable_rows": int(V.loc[vis, "target"].notna().sum()),
        "n_clipped_targets": int((V.loc[vis, "ratio_raw"] > 1.0).sum()),
    }
    sub = V[vis]
    folds[f"P6_{pl}"] = [{"fold": f,
                          "test_groups": int(sub.loc[sub["fold"] == f, "group"].nunique()),
                          "test_view_rows": int((sub["fold"] == f).sum()),
                          "train_groups": int(sub.loc[sub["fold"] != f, "group"].nunique()),
                          "train_view_rows": int((sub["fold"] != f).sum())}
                         for f in range(ac.N_FOLDS)]

# --------------------------------------------------------------- artifacts 목록
art = ROOT / "xgboost_model" / "artifacts"
artifacts = []
for p in sorted(art.iterdir()):
    if p.is_dir():
        artifacts.append({"name": p.name + "/", "kind": "dir",
                          "n_files": len(list(p.iterdir()))})
        continue
    artifacts.append({"name": p.name, "kind": "file", **stat(p)})

# infps=0 결함: 출하 아티팩트의 feature 결함 여부를 코드 경로로만 기록
guide_facts = {
    "alpha": ac.ALPHA, "beta": ac.BETA, "seed": ac.SEED, "n_folds": ac.N_FOLDS,
    "feature_order_len": len(ac.FEATURE_ORDER),
    "feature_order_source": "xgboost_model/artifacts/deploy_cpu_gpu_features.json",
    "gen_models": sorted(ac.GEN_MODELS),
    "identical_candidate_sets_across_platforms": bool(identical_candidates),
    "unmapped_models_keys": unmapped,
}

out = {"files": files_rows, "candidates": candidates, "rate_check": rate_check,
       "model_rows": model_rows, "folds": folds, "artifacts": artifacts,
       "facts": guide_facts}
(OUT / "measurements.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                                  default=str))

# ------------------------------------------------------------------- 콘솔 요약
print("== 2-A ==")
for r in files_rows:
    print(f"  {'OK ' if r['exists'] else 'MISS'} {r['path']}")
print("== 2-B a/b ==")
for pl in ac.PLATFORMS:
    c = candidates[pl]
    print(f"  {pl}: windows={c['total_windows']} unique_placements="
          f"{c['total_unique_placements']} sets={c['n_sets']} "
          f"groups={c['n_groups_models_x_rate']}")
print(f"  candidate sets identical across platforms: {identical_candidates}")
bad = [k for k, v in rate_check.items() if not v["match"]]
print(f"  rate rule mismatches: {bad or 'none'}")
print("== 2-B c ==")
for k, v in model_rows.items():
    print(" ", k, v)
print("== 2-B d ==")
for k, v in folds.items():
    print(" ", k, v)
