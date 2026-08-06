"""작업 12 run_manifest.json 생성."""
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
    "profile_models.py",
    "runtime/llm_engine.py",
    "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/full_collection_540/scripts/greedy_baseline.py",
    "xgboost_model/full_collection_540/scripts/train_decomposed.py",
    "xgboost_model/MODEL_GUIDE.md",
    "runs/20260804_180759_feature_spec/feature_spec.md",
]

manifest = {
    "run": HERE.name,
    "task": "작업 12 — static_load 필드의 실체 확정 (읽기 전용)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {"created": [f"runs/{HERE.name}/*"], "modified": [],
              "note": "코드·데이터·원고 미수정. 프로파일러 실행도 하지 않았다(하드웨어 필요)."},
    "inputs": {p: sha256(ROOT / p) for p in INPUTS},
    "outputs": {p.name: sha256(p) for p in sorted(HERE.iterdir())
                if p.name != "run_manifest.json"},
    "verdict": facts["verdict"],
    "representative_pairs_ms": facts["representative_pairs"],
    "doc_consistency": facts["doc_consistency"],
    "unconfirmed": facts["unconfirmed"],
    "verification": {"checks": facts["checks"], "problems": facts["problems"],
                     "all_passed": not facts["problems"]},
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("verdict:", manifest["verdict"]["meaning"], "|", manifest["verdict"]["unit"])
print("all_passed:", manifest["verification"]["all_passed"])
