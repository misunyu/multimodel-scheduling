"""작업 10-r3 run_manifest.json 생성 — 참조 파일·해시·검증 결과."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


meas = json.loads((HERE / "measurements.json").read_text())
ver = json.loads((HERE / "verification.json").read_text())

# 문서가 근거로 인용한 소스 코드 파일
SOURCES = [
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/gen_collection_schedules.py",
    "xgboost_model/full_collection_540/scripts/analysis_common.py",
    "xgboost_model/full_collection_540/scripts/train_groupkfold.py",
    "xgboost_model/full_collection_540/scripts/train_unified.py",
    "xgboost_model/full_collection_540/scripts/train_decomposed.py",
    "xgboost_model/full_collection_540/scripts/cross_platform_transfer.py",
    "xgboost_model/full_collection_540/scripts/greedy_baseline.py",
    "xgboost_model/full_collection_540/scripts/train_gpu.py",
    "xgboost_model/full_collection_540/scripts/train_npu.py",
    "xgboost_model/full_collection_540/README.md",
    "xgboost_model/full_collection_540/analysis/SUMMARY.md",
    "xgboost_model/full_collection_540/analysis/groupkfold_ranking.md",
    "xgboost_model/full_collection_540/analysis/unified_metrics.json",
    "xgboost_model/full_collection_540/analysis/decomposed_metrics.json",
    "model_registry.py",
    "working_sets.yaml",
    "profile_models.py",
    "Y2_NORMALIZATION.md",
    "runs/20260730_172053_coverage/appendix_coverage.csv",
]

manifest = {
    "run": HERE.name,
    "task": "작업 10-r3 — XGBoost 모델 학습·저장·사용 문서화 (MODEL_GUIDE.md)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "outputs": {
        "xgboost_model/MODEL_GUIDE.md": sha256(ROOT / "xgboost_model/MODEL_GUIDE.md"),
        **{f"runs/{HERE.name}/{p.name}": sha256(p)
           for p in sorted(HERE.iterdir()) if p.name != "run_manifest.json"},
    },
    "scope": {
        "created": ["xgboost_model/MODEL_GUIDE.md"],
        "modified": [],
        "note": "코드·아티팩트·기존 문서 미수정. 학습·재수집 없음(읽기 전용 실측만).",
    },
    "input_data_files": {r["path"]: r.get("sha256") for r in meas["files"]},
    "input_source_files": {s: sha256(ROOT / s) for s in SOURCES
                           if (ROOT / s).exists()},
    "key_numbers": {
        "candidates_per_platform": meas["candidates"]["gpu"]["total_unique_placements"],
        "windows_per_platform": meas["candidates"]["gpu"]["total_windows"],
        "groups_per_platform": meas["candidates"]["gpu"]["n_groups_models_x_rate"],
        "rate_levels_per_set": 3,
        "P1_rows_per_platform": meas["model_rows"]["P1_gpu"]["n_rows"],
        "joint_rows": meas["model_rows"]["joint"]["n_rows"],
        "joint_groups": meas["model_rows"]["joint"]["n_groups"],
        "P6_vision_view_rows_per_platform": meas["model_rows"]["P6_gpu"]["n_vision_view_rows"],
        "P6_total_view_rows_per_platform": meas["model_rows"]["P6_gpu"]["n_view_rows_total"],
        "y3_valid_rows_per_platform": meas["model_rows"]["P1_gpu"]["n_y3_valid_rows"],
        "P1_fold_test_rows": [f["test_rows"] for f in meas["folds"]["P1_gpu"]],
        "joint_fold_test_rows": [f["test_rows"] for f in meas["folds"]["joint"]],
        "P6_fold_test_view_rows": [f["test_view_rows"] for f in meas["folds"]["P6_gpu"]],
    },
    "verification": {
        "paths_cited_and_checked": ver["paths_checked"],
        "paths_missing": ver["paths_missing"],
        "line_citation_mismatches": [p for p in ver["problems"] if "줄 근거" in p],
        "numeric_checks": ver["numeric_checks"],
        "all_passed": not ver["problems"],
    },
    "unconfirmed": [
        "'maxsize 하드코딩' 함정 — 학습·데이터 경로에서 해당 사례를 찾지 못함 (§6-2 마지막 항목)",
        "sample_profiling_data.pre_pathbug_backup.json 이 어떤 경로 버그의 백업인지 (§6-3)",
        "deploy_conc_cpu_gpu_* / deploy_xgb_* 를 만든 정확한 명령줄 (§6-3)",
    ],
}
(HERE / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(manifest["key_numbers"], indent=2, ensure_ascii=False))
print("verification all_passed:", manifest["verification"]["all_passed"])
