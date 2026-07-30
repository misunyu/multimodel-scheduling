"""Write run_manifest.json for this coverage run: inputs, hashes, commit, timestamp."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent

INPUTS = [
    "working_sets.yaml",
    "model_registry.py",
    "xgboost_model/gen_collection_schedules.py",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/schedules/collection/collect_cpu_gpu.yaml",
    "xgboost_model/schedules/collection/collect_cpu_gpu.yaml.meta.json",
    "xgboost_model/schedules/collection/collect_cpu_npu.yaml",
    "xgboost_model/schedules/collection/collect_cpu_npu.yaml.meta.json",
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
    "xgboost_model/full_collection_540/scripts/analysis_common.py",
    "xgboost_model/full_collection_540/analysis/platform_divergence.md",
]
# Consulted during exploration but deliberately NOT used as a numeric source.
NOT_USED = {
    "xgboost_model/performance_data/cpu_gpu/performance.json":
        "160-window pilot (2026-07-09/10), not the data the shipped predictors were trained on",
    "xgboost_model/performance_data/cpu_npu/performance.json":
        "160-window pilot, same reason",
    "xgboost_model/artifacts/deploy_cpu_gpu_coverage.json":
        "summary only (rows/model_sets/rate_factors); no per-set n_measured -- used for §D cross-check only",
    "xgboost_model/artifacts/deploy_cpu_npu_coverage.json": "same as above",
    "xgboost_model/full_collection_540/cpu_gpu/collect_gpu_results.jsonl":
        "duplicate of the windows file; read for cross-check only",
    "xgboost_model/full_collection_540/cpu_npu/collect_npu_results.jsonl": "same as above",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    facts = json.loads((OUT / "coverage_facts.json").read_text())
    manifest = {
        "run_id": OUT.name,
        "purpose": ("MLForSys appendix sampling-coverage table + verifications A-E; "
                    "read-only over already-collected artifacts (no re-collection)"),
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_commit_note": "HEAD at execution time (outputs committed on top of it)",
        "scripts": [
            {"file": "build_coverage.py", "sha256": sha256(OUT / "build_coverage.py"),
             "role": "computes appendix_coverage.csv + coverage_facts.json (A-E)"},
            {"file": "make_manifest.py", "sha256": sha256(OUT / "make_manifest.py"),
             "role": "this manifest"},
        ],
        "outputs": [
            {"file": "appendix_coverage.csv", "role": "artifact 1 -- 15 rows, one per working set"},
            {"file": "coverage_report.md", "role": "artifact 2 -- verifications A-E"},
            {"file": "coverage_facts.json", "role": "machine-readable form of every reported number"},
        ],
        "input_artifacts": [
            {"path": p, "sha256": sha256(ROOT / p), "bytes": (ROOT / p).stat().st_size}
            for p in INPUTS
        ],
        "consulted_not_used": NOT_USED,
        "parameters": {"alpha": 0.3, "betas": [1.0, 0.5], "tie_eps": 1e-9,
                       "score_fn": "xgboost_model/deploy_selector_xgb_suite.py:score_combo",
                       "tie_rule": facts["C_tie_rule"]},
        "headline_results": {
            "A_symdiff_all_zero": facts["A_all_zero"],
            "A_measured_equals_scheduled": facts["A_measured_equals_scheduled"],
            "B_qwen2_vl_cpu_rows": facts["B_qwen2_vl_cpu_rows"],
            "B_pow2_holds_for_all_sets": facts["B_pow2_holds_for_all"],
            "B_pow2_holds_iff_set_has_vlm": facts["B_pow2_holds_iff_has_vlm"],
            "B3_rate_levels_share_placement_set": facts["B3_rate_levels_share_placement_set"],
            "artifact1_granularity": facts["artifact1_granularity"],
            "C_exhaustive": {b: {"n_groups": facts["C"][b]["n_groups"],
                                 "n_disagree": facts["C"][b]["n_disagree"],
                                 "gen": facts["C"][b]["gen"],
                                 "vision": facts["C"][b]["vision"]}
                             for b in ("1.0", "0.5")},
            "C_all45_crosscheck": facts["C_all45_crosscheck"],
            "C_all45_matches_published_divergence_md": facts["C_all45_matches_published"],
            "D": {pl: {k: v for k, v in facts["D"][pl].items() if k.endswith("_ok")}
                  for pl in ("gpu", "npu")},
            "E": facts["E"],
        },
        "unknowns": [],
        "caveats": [
            "No manuscript .tex exists in this repository (find . -name '*.tex' -> 0 hits), so the "
            "tie-aware argmax rule was taken from analysis_common.ranking_metrics (1e-9 tie set) and "
            "validated by reproducing the published 29/45 and 30/45 divergence figures.",
            "The paths named in the task (xgboost_model/artifacts/cpu_gpu, .../cpu_npu) do not exist; "
            "artifacts/ is flat-named. Real per-placement data lives in full_collection_540/cpu_{gpu,npu}/.",
        ],
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["headline_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
