"""작업 13 run_manifest.json 생성."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


facts = json.loads((HERE / "facts.json").read_text())

INPUTS = [
    "xgboost_model/full_collection_540/scripts/greedy_baseline.py",
    "xgboost_model/full_collection_540/analysis/greedy_results.csv",
    "xgboost_model/full_collection_540/scripts/collect_run.py",
    "xgboost_model/gen_collection_schedules.py",
    "generate_schedules.py",
    "best_deploy_finder_executor.py",
    "runtime/llm_engine.py",
    "runtime/mobilint_vision.py",
    "model_processors.py",
    "model_registry.py",
    "unified_viewer.py",
    "requirements.txt",
    "runtime_env.sh",
    "working_sets.yaml",
    "PRE_COLLECTION_FINAL.md",
    "NPU_POSTPROCESS_UNIFICATION.md",
    "docs/llama1b_core_confound_report.md",
    "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json",
]

manifest = {
    "run": HERE.name,
    "task": "작업 13 — r13 원고 마무리용 확인 3건 (읽기 전용)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {"created": [f"runs/{HERE.name}/*"], "modified": [],
              "note": "코드·데이터·원고 미수정. 학습·수집·프로파일링 실행 없음. "
                      "라이브 호스트 조회는 버전·CPU 모델명 확인용 읽기만."},
    "inputs": {p: sha256(ROOT / p) for p in INPUTS if (ROOT / p).exists()},
    "outputs": {p.name: sha256(p) for p in sorted(HERE.iterdir())
                if p.name != "run_manifest.json"},
    "findings": {
        "q1_oracle_ratio": {
            "verdict": facts["q1_oracle_ratio"]["verdict"],
            "manuscript_sentence_matches": True,
            "formula": facts["q1_oracle_ratio"]["formula"],
            "denominators_by_cell": {k: v["n_collected"]
                                     for k, v in facts["q1_oracle_ratio"]["recomputed"].items()},
            "optional_refinements": [
                "'score' 가 measured score 임을 명시",
                "norm./raw 두 열이 점수 기저 차이임을 캡션에 명시",
                "정규화 기저의 비율은 하한이 0이 아니다 (최소 -0.156, 음수 7건)",
            ],
        },
        "q2_run_settings": {
            "table_rows": 26,
            "unknown_count": 7,
            "manuscript_conflict": "§142 'Intel Core i7-13700K' vs 저장소·라이브 호스트 "
                                   "'Intel Core Ultra 9 285K' — 저자 확인 필요",
        },
        "q3_candidate_policy": {
            "verdict": "동일 규칙 · 코드 이중화 · 샘플링 단계만 상이",
            "feasibility_rule_single_source": True,
            "two_enumerators_agree_on_full_set": True,
            "profiling_subsamples": ["S6", "S7", "S8", "S9", "S10"],
            "deployment_subsamples": False,
            "manuscript_sentence_insertable_as_is": False,
        },
    },
    "verification": {"checks": facts["checks"], "problems": facts["problems"],
                     "all_passed": not facts["problems"]},
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("all_passed:", manifest["verification"]["all_passed"])
print("q3 verdict:", manifest["findings"]["q3_candidate_policy"]["verdict"])
