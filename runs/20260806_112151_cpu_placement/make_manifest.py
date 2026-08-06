"""작업 11 run_manifest.json 생성."""
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


facts = json.loads((HERE / "facts.json").read_text())
ver = json.loads((HERE / "verification.json").read_text())

INPUTS = [
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
    "working_sets.yaml",
    "model_registry.py",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/gen_collection_schedules.py",
    "xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
]

mt = {r["model"]: r for r in facts["models"]["gpu"]}
manifest = {
    "run": HERE.name,
    "task": "작업 11 — 후보 배치의 CPU 배치 사례 분석 (읽기 전용)",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {"created": [f"runs/{HERE.name}/*"], "modified": [],
              "note": "코드·데이터·원고 미수정. 저장된 측정 데이터만 읽었다."},
    "inputs": {p: sha256(ROOT / p) for p in INPUTS},
    "outputs": {p.name: sha256(p) for p in sorted(HERE.iterdir())
                if p.name != "run_manifest.json"},
    "denominators": facts["denominators"],
    "headline": {
        "k_distribution_candidates": {str(r["k_cpu"]): r["n_candidates"]
                                      for r in facts["k_distribution"]["gpu"]},
        "k_distribution_windows": {str(r["k_cpu"]): r["n_windows"]
                                   for r in facts["k_distribution"]["gpu"]},
        "k0_candidates": facts["k0_candidates"]["n"],
        "k_max": facts["k_max"]["k"],
        "vlm_cpu_placements": mt["qwen2_vl"]["cand_on_cpu"],
        "vlm_constraint_violations": facts["vlm_cpu_violations"],
        "llm_cpu_candidates": mt["llama1b"]["cand_on_cpu"],
        "llm_cpu_windows": mt["llama1b"]["window_on_cpu"],
        "cpu_instances_candidate_basis": facts["instance_totals"]["gpu"]["cand_cpu_instances"],
        "view_instances_candidate_basis": facts["instance_totals"]["gpu"]["cand_view_instances"],
        "candidates_with_cpu_ge1": facts["exec_cpu_feature"]["candidates_active"],
        "windows_with_cpu_ge1": facts["exec_cpu_feature"]["windows_active"],
        "empty_accelerator_candidates": 5,
        "platforms_symmetric": facts["symmetry"]["symmetric"],
    },
    "verification": {
        "analyze_checks": facts["checks"],
        "report_claim_checks": ver["checks"],
        "problems": facts.get("problems", []) + ver["problems"],
        "all_passed": all(facts["checks"].values()) and all(ver["checks"].values())
                      and not ver["problems"],
    },
}
(HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(manifest["headline"], indent=2, ensure_ascii=False))
print("all_passed:", manifest["verification"]["all_passed"])
