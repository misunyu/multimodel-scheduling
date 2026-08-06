"""MODEL_GUIDE.md 검증: 문서가 인용한 모든 경로의 실재 여부 + 줄 번호 근거 대조."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
GUIDE = ROOT / "xgboost_model" / "MODEL_GUIDE.md"
text = GUIDE.read_text()

problems = []

# ---- 1. 문서가 백틱으로 인용한 저장소 경로 전부 실재 확인 -------------------
PATH_RE = re.compile(r"`([A-Za-z0-9_./{},*\-]+\.(?:py|json|csv|md|yaml|jsonl))`")
BRACE_RE = re.compile(r"\{([^{}]*)\}")


def expand(p: str):
    """`a_{x,y}_b.json` 같은 중괄호 표기를 전개."""
    m = BRACE_RE.search(p)
    if not m:
        return [p]
    out = []
    for opt in m.group(1).split(","):
        out.extend(expand(p[:m.start()] + opt + p[m.end():]))
    return out


PREFIXES = ["", "xgboost_model/full_collection_540/",
            "xgboost_model/full_collection_540/scripts/",
            "xgboost_model/full_collection_540/analysis/", "xgboost_model/",
            "xgboost_model/artifacts/",
            "xgboost_model/performance_data/sample_profiling_data/"]
# 문서가 템플릿 자리표시자로 쓴 토막(실제 파일명이 아님)
PLACEHOLDER = re.compile(r"(^_fold\d|platform_metrics|_\{?tag\}?_)")
checked, missing = set(), []
for raw in sorted(set(PATH_RE.findall(text))):
    if "*" in raw or raw.startswith("..."):
        continue
    for cand in expand(raw):
        if cand in checked or PLACEHOLDER.search(cand):
            continue
        checked.add(cand)
        if any((ROOT / (pre + cand)).exists() for pre in PREFIXES):
            continue
        missing.append(cand)
if missing:
    problems.append(f"인용됐으나 찾지 못한 경로: {missing}")

# ---- 2. 줄 번호 근거 대조 ---------------------------------------------------
SUITE = (ROOT / "xgboost_model/deploy_selector_xgb_suite.py").read_text().splitlines()
AC = (ROOT / "xgboost_model/full_collection_540/scripts/analysis_common.py").read_text().splitlines()
TGK = (ROOT / "xgboost_model/full_collection_540/scripts/train_groupkfold.py").read_text().splitlines()
TUN = (ROOT / "xgboost_model/full_collection_540/scripts/train_unified.py").read_text().splitlines()
TDE = (ROOT / "xgboost_model/full_collection_540/scripts/train_decomposed.py").read_text().splitlines()
XPT = (ROOT / "xgboost_model/full_collection_540/scripts/cross_platform_transfer.py").read_text().splitlines()
REG = (ROOT / "model_registry.py").read_text().splitlines()
GCS = (ROOT / "xgboost_model/gen_collection_schedules.py").read_text().splitlines()

CITES = [
    (SUITE, 101, "def load_static_profiles"), (SUITE, 126, "def _device_static"),
    (SUITE, 224, "def _build_infps_lookup"), (SUITE, 250, "def _view_features"),
    (SUITE, 282, "def _aggregate"), (SUITE, 297, "def featurize_window"),
    (SUITE, 363, "def featurize_from_combo"), (SUITE, 382, "def _extract_schedule_hint"),
    (SUITE, 403, "def build_dataset_from_file"), (SUITE, 425, "def _normalize_targets"),
    (SUITE, 485, "rglob"), (SUITE, 519, "_PARAMS = {"), (SUITE, 525, "_PARAM_GRID = ["),
    (SUITE, 534, "def _cv_select_params"), (SUITE, 556, "def write_coverage"),
    (SUITE, 617, "save_model"), (SUITE, 677, "DEFAULT_ALPHA"), (SUITE, 678, "DEFAULT_BETA"),
    (SUITE, 681, "def score_combo"), (SUITE, 700, "def _align_features"),
    (SUITE, 711, "def predict_targets"), (SUITE, 759, "--rate_factors"),
    (AC, 44, "SEED = 42"), (AC, 48, "FEATURE_ORDER"), (AC, 88, "def load_windows"),
    (AC, 97, "def infps_map_for"), (AC, 106, "def load_dataset"),
    (AC, 133, "reindex(columns=FEATURE_ORDER)"), (AC, 139, "def normalize_targets"),
    (AC, 152, "def group_keys"), (AC, 160, "def assign_group_folds"),
    (AC, 168, "RandomState(seed)"), (AC, 183, "def scores_from"),
    (AC, 191, "def ranking_metrics"), (AC, 204, "1e-9"),
    (AC, 228, "def cv_select_params"), (AC, 259, "def train_booster"),
    (AC, 266, "def predict_booster"), (AC, 273, "def load_booster"),
    (TGK, 32, "def run_platform"), (TGK, 56, "y3_valid if tag"),
    (TGK, 75, "save_model"), (TGK, 82, "save_model"), (TGK, 110, "group_to_fold"),
    (TGK, 115, "def main"), (TGK, 141, "_features.json"),
    (TUN, 29, "def main"), (TUN, 38, "assert len(X) == 1080"),
    (TUN, 40, "group_keys"), (TUN, 15, "No boosters are exported"),
    (TDE, 50, "def build_view_dataset"), (TDE, 93, "def main"),
    (TDE, 100, "groupkfold_"), (TDE, 113, "assert"), (TDE, 114, "groupkfold_"),
    (TDE, 115, "y3_pred"), (TDE, 121, "merge"), (TDE, 124, "vis"),
    (TDE, 136, "train_booster"), (TDE, 185, "y3_pred"), (TDE, 214, "greedy_results.csv"),
    (XPT, 64, "def transfer_predict"),
    (REG, 45, "DEVICE_CONSTRAINTS"),
    (GCS, 128, "sort_keys=False"),
]
for lines, ln, needle in CITES:
    got = lines[ln - 1] if 0 < ln <= len(lines) else "<out of range>"
    if needle not in got:
        problems.append(f"줄 근거 불일치: {ln} 에 '{needle}' 없음 -> {got.strip()!r}")

# ---- 3. 수치 재검산 ----------------------------------------------------------
m = json.loads((Path(__file__).resolve().parent / "measurements.json").read_text())
checks = {
    "candidates_gpu_180": m["candidates"]["gpu"]["total_unique_placements"] == 180,
    "candidates_npu_180": m["candidates"]["npu"]["total_unique_placements"] == 180,
    "windows_540_each": all(m["candidates"][p]["total_windows"] == 540 for p in ("gpu", "npu")),
    "groups_45_each": all(m["candidates"][p]["n_groups_models_x_rate"] == 45
                          for p in ("gpu", "npu")),
    "joint_1080": m["model_rows"]["joint"]["n_rows"] == 1080,
    "joint_90_groups": m["model_rows"]["joint"]["n_groups"] == 90,
    "p6_vision_2220": all(m["model_rows"][f"P6_{p}"]["n_vision_view_rows"] == 2220
                          for p in ("gpu", "npu")),
    "p6_total_2952": all(m["model_rows"][f"P6_{p}"]["n_view_rows_total"] == 2952
                         for p in ("gpu", "npu")),
    "p6_gen_732": all(m["model_rows"][f"P6_{p}"]["n_gen_view_rows"] == 732
                      for p in ("gpu", "npu")),
    "y3_valid_372": all(m["model_rows"][f"P1_{p}"]["n_y3_valid_rows"] == 372
                        for p in ("gpu", "npu")),
    "features_37": m["facts"]["feature_order_len"] == 37,
    "p1_folds_match_stored": all(m["folds"][f"P1_{p}_matches_stored_group_to_fold"]
                                 for p in ("gpu", "npu")),
    "rate_rule_all_match": all(v["match"] for v in m["rate_check"].values()),
    "identical_candidates": m["facts"]["identical_candidate_sets_across_platforms"],
    "no_unmapped_sets": all(not v for v in m["facts"]["unmapped_models_keys"].values()),
}
# 뷰 수 = sum(모델 수 × 창 수)
ws = __import__("yaml").safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
view_total = sum(len(ws[s]) * m["candidates"]["gpu"]["by_set"][s]["n_windows"] for s in ws)
checks["view_total_2952_from_sets"] = view_total == 2952
# artifacts 개수
n_file = sum(1 for a in m["artifacts"] if a["kind"] == "file")
n_dir = sum(1 for a in m["artifacts"] if a["kind"] == "dir")
checks["artifacts_49_files_2_dirs"] = (n_file, n_dir) == (49, 2)
# Appendix A 커버리지 일치
import csv
cov = {r["set_id"]: r for r in csv.DictReader(
    (ROOT / "runs/20260730_172053_coverage/appendix_coverage.csv").open())}
checks["coverage_matches_appendix"] = all(
    int(cov[s]["n_measured_gpu"]) == m["candidates"]["gpu"]["by_set"][s]["n_unique_placements"]
    and int(cov[s]["n_measured_npu"]) == m["candidates"]["npu"]["by_set"][s]["n_unique_placements"]
    for s in cov)
for k, v in checks.items():
    if not v:
        problems.append(f"수치 검산 실패: {k}")

# ---- 4. save_model 호출이 3곳뿐인지 --------------------------------------------
sm = subprocess.run(["grep", "-rn", "--exclude-dir=.venv", "--exclude-dir=.git",
                     "save_model", "--include=*.py", "."], cwd=ROOT,
                    capture_output=True, text=True).stdout.strip().splitlines()
sm = [l for l in sm if not l.startswith("./runs/")]
checks_sm = sorted(l.split(":")[0] + ":" + l.split(":")[1] for l in sm)
expected_sm = sorted([
    "./xgboost_model/full_collection_540/scripts/train_groupkfold.py:75",
    "./xgboost_model/full_collection_540/scripts/train_groupkfold.py:82",
    "./xgboost_model/deploy_selector_xgb_suite.py:617"])
if checks_sm != expected_sm:
    problems.append(f"save_model 호출 위치 불일치: {checks_sm}")

# ---- 5. sample_profiling_data.json 을 쓰는 코드가 없는지 -----------------------
w = subprocess.run(["grep", "-rln", "--exclude-dir=.venv", "--exclude-dir=.git",
                    "sample_profiling_data", "--include=*.py", "."],
                   cwd=ROOT, capture_output=True, text=True).stdout.split()
writers = []
for f in w:
    if f.startswith("./runs/"):
        continue
    src = (ROOT / f).read_text()
    for ln in src.splitlines():
        if "sample_profiling_data" in ln and ("write_text" in ln or "json.dump" in ln
                                              or 'open(' in ln and '"w"' in ln):
            writers.append((f, ln.strip()))
if writers:
    problems.append(f"sample_profiling_data 를 쓰는 코드 발견(문서의 '미확인' 서술 수정 필요): {writers}")

# ---- 결과 -------------------------------------------------------------------
out = {"paths_checked": len(checked), "paths_missing": missing,
       "numeric_checks": checks, "problems": problems,
       "sample_profiling_readers": [f for f in w if not f.startswith("./runs/")]}
(Path(__file__).resolve().parent / "verification.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False))

print(f"인용 경로 {len(checked)}건 확인, 누락 {len(missing)}건")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
print(f"save_model 호출: {checks_sm}")
print(f"sample_profiling_data 참조 파일: {[f for f in w if not f.startswith('./runs/')]}")
if problems:
    print("\n=== 문제 ===")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n검증 통과")
