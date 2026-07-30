"""Write run_manifest.json for the Appendix D coefficient sweep run."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent

INPUTS = [
    "working_sets.yaml",
    "xgboost_model/deploy_selector_xgb_suite.py",
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
    "SCORE_WEIGHTS_REVIEW.md",
    "BETA1_CPU_PLACEMENT.md",
    "xgboost_model/full_collection_540/analysis/platform_divergence.md",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    f = json.loads((OUT / "sweep_facts.json").read_text())
    m = {
        "run_id": OUT.name,
        "purpose": ("MLForSys Appendix D coefficient sweep (alpha x beta) on measured "
                    "scores + verification of the body's sensitivity-asymmetry claim"),
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_commit_note": "HEAD at execution time (outputs committed on top of it)",
        "scripts": [{"file": n, "sha256": sha256(OUT / n)}
                    for n in ("build_sweep.py", "make_manifest.py")],
        "outputs": [
            {"file": "sweep_setcounts.csv",
             "role": "artifact 1 -- PRIMARY grid (pre-registered), 25 grid points"},
            {"file": "sweep_setcounts_declared_grid.csv",
             "role": "same columns on the task-declared grid, 35 points"},
            {"file": "sweep_setcounts_normalized.csv",
             "role": "secondary robustness variant on group-normalized measured targets"},
            {"file": "sweep_report.md", "role": "artifact 2 -- sections 0/A-F + UNKNOWNs"},
            {"file": "sweep_facts.json", "role": "artifact 3 -- machine-readable facts"},
        ],
        "grid_provenance": f["grid_provenance"],
        "grid_note": ("PRIMARY grid inherited from SCORE_WEIGHTS_REVIEW.md section 2 per the "
                      "pre-registration principle; only the alpha/beta VALUE LISTS were "
                      "inherited -- score basis (measured, not predicted), baseline "
                      "(beta=1.0, not 0.5) and 2-D grid shape follow this task's definitions. "
                      "No F3 artifact exists in the repository (see report section 0)."),
        "definitions": f["definitions"],
        "input_artifacts": [{"path": p, "sha256": sha256(ROOT / p),
                             "bytes": (ROOT / p).stat().st_size} for p in INPUTS],
        "headline_results": {
            "body_claim": ("GPU optimum unchanged across tested coefficients; NPU changes "
                           "in up to 7 of 15 sets (alpha in [0,0.6], beta in [0,1])"),
            "verdict": "NOT REPRODUCED on measured scores; asymmetry direction is reversed",
            "primary_grid": {
                "n_grid_points": f["primary"]["n_grid_points"],
                "gpu_max_changed_sets": f["primary"]["gpu_max_changed_sets"],
                "npu_max_changed_sets": f["primary"]["npu_max_changed_sets"],
                "gpu_violating_points": len(f["primary"]["gpu_violations"]),
            },
            "declared_grid": {
                "n_grid_points": f["declared"]["n_grid_points"],
                "gpu_max_changed_sets": f["declared"]["gpu_max_changed_sets"],
                "npu_max_changed_sets": f["declared"]["npu_max_changed_sets"],
            },
            "alpha_axis_has_no_effect_on_raw_scores": True,
            "baseline_selfcheck_ok": f["A_baseline_selfcheck"]["ok"],
            "C_beta_sensitive_group_counts": f["C_counts"],
            "E_divergence_matches_published": f["E_matches_platform_divergence_md"],
            "F_normalized_variant": {
                "gpu_max_changed_sets": f["F_normalized_variant"]["gpu_max_changed_sets"],
                "npu_max_changed_sets": f["F_normalized_variant"]["npu_max_changed_sets"],
            },
            "F_raw_scale": f["F_raw_scale"],
        },
        "unknowns": [
            "The manuscript .tex is not in this repository, so the body's exact wording, "
            "sweep range and the intended scale of 'measured-score' could not be checked "
            "against the source; the claim was compared against the text quoted in the task.",
            "No 'F3' identifier exists anywhere in the repository, so the F3 rerun plan's "
            "intended grid/baseline/scale is unverifiable; the grid was inherited from the "
            "only real prior sweep (SCORE_WEIGHTS_REVIEW.md section 2).",
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(m["headline_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
