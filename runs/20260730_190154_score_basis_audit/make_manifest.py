"""Write run_manifest.json for the score-basis audit run."""
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
    "xgboost_model/full_collection_540/scripts/analysis_common.py",
    "xgboost_model/full_collection_540/scripts/compare_platforms_v2.py",
    "xgboost_model/full_collection_540/scripts/train_groupkfold.py",
    "xgboost_model/full_collection_540/scripts/train_unified.py",
    "xgboost_model/full_collection_540/scripts/cross_platform_transfer.py",
    "xgboost_model/full_collection_540/scripts/train_decomposed.py",
    "xgboost_model/full_collection_540/scripts/greedy_baseline.py",
    "xgboost_model/full_collection_540/scripts/jsonl_to_windows.py",
    "xgboost_model/full_collection_540/analysis/platform_divergence.csv",
    "xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
    "xgboost_model/full_collection_540/analysis/groupkfold_npu_metrics.json",
    "xgboost_model/full_collection_540/analysis/greedy_summary.md",
    "xgboost_model/full_collection_540/analysis/SUMMARY.md",
]

PRIOR_RUNS = {
    "runs/20260730_172053_coverage": "sampling-coverage table; source of the raw-basis "
                                     "22/30 exhaustive figure and the exhaustive set list",
    "runs/20260730_184450_coeff_sweep": "alpha x beta sweep; its F_normalized_variant "
                                        "(GPU 9 / NPU 11) is cross-checked in part 2c",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    f = json.loads((OUT / "basis_facts.json").read_text())
    m = {
        "run_id": OUT.name,
        "purpose": ("Audit the score basis of every headline number against the paper's "
                    "declared Eq. (1) basis, and recompute on the declared basis"),
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_commit_note": "HEAD at execution time (outputs committed on top of it)",
        "scripts": [{"file": n, "sha256": sha256(OUT / n)}
                    for n in ("build_basis_audit.py", "make_manifest.py")],
        "outputs": [
            {"file": "basis_audit.md",
             "role": "part 1 basis-audit table + part 2 recomputation + changed-sentence list"},
            {"file": "basis_facts.json", "role": "machine-readable facts"},
            {"file": "sweep_declared_basis.csv", "role": "part 2c/2d sweep, 25 grid points"},
            {"file": "divergence_by_group.csv",
             "role": "per-group optima and agreement, both bases x both betas (180 rows)"},
        ],
        "declared_basis": {
            "group": "(set, rate)",
            "y1": "r1 / max(r1) within group",
            "y3": "r3 / max(r3) within group",
            "y2": "(r2 - min r2) / (max r2 - min r2); 0 when max == min",
            "score": "S = y1 - 0.3*y2 + beta*y3 (beta term only for generative sets)",
            "betas": [1.0, 0.5],
            "tie_rule": "1e-9 tie set; intersecting tie sets count as agreement",
            "y3_mask_convention": f["y3_mask_convention"]["rule"],
            "y3_mask_empirical_effect": f["y3_mask_convention"]["empirical_effect"],
            "r1_source": "full540 measured window totals (total.total_throughput_fps)",
        },
        "input_artifacts": [{"path": p, "sha256": sha256(ROOT / p),
                             "bytes": (ROOT / p).stat().st_size} for p in INPUTS],
        "prior_runs_referenced": PRIOR_RUNS,
        "headline_results": {
            "part1_basis_by_number": {
                "divergence_29_45_and_30_45": "raw",
                "gen_29_of_33_and_vision_0_of_12": "raw (derived from platform_divergence.csv)",
                "spearman_top1_top5": "normalized",
                "transfer_and_unified": "normalized",
                "greedy_and_decomposed": "normalized",
                "appendix_A_per_set_spearman": "normalized",
                "exhaustive_22_of_30": "raw",
            },
            "part1_raw_basis_reproduced": {
                "29_45_and_30_45": f["part1_reproduces_29_45_and_30_45"],
                "29_of_33_and_0_of_12": f["part1_reproduces_29of33_and_0of12"],
            },
            "part1_r1_convention_gap": f["r1_convention_gap"],
            "part2a_declared_basis": {
                b: {"n_disagree": f["part2a_declared_basis_divergence"][b]["n_disagree"],
                    "gen": f["part2a_declared_basis_divergence"][b]["gen"],
                    "vision": f["part2a_declared_basis_divergence"][b]["vision"]}
                for b in ("1.0", "0.5")},
            "part2a_beta_sensitive": f["part2a_beta_sensitive"],
            "part2b_exhaustive": {
                b: {"declared": f["part2b_exhaustive"][b]["declared_basis"]["n_disagree"],
                    "raw": f["part2b_exhaustive"][b]["raw_basis"]["n_disagree"]}
                for b in ("1.0", "0.5")},
            "part2c_sweep": {
                "gpu_max_changed_sets": f["part2c_sweep"]["gpu_max_changed_sets"],
                "npu_max_changed_sets": f["part2c_sweep"]["npu_max_changed_sets"],
                "matches_prior_F_variant": f["part2c_matches_prev_F"]},
            "part2d_beta_split": f["part2d_beta_split"],
            "part3": "NOT RUN -- its precondition (raw measured-best basis for c-f) is "
                     "false; c-f are already computed on the declared normalized basis",
        },
        "unknowns": [
            "Eq. (1)'s r1 definition (all-view sum vs vision-only sum) cannot be checked: "
            "no .tex in this repository. Divergence conclusions are identical under both "
            "conventions, so no reported number depends on it.",
            "Whether the manuscript describes divergence as following Eq. (1) or explicitly "
            "as raw measured totals is unverifiable for the same reason.",
            "The producing script for the 29/33 vs 0/12 gen/vision split was not found "
            "(no gen/vision split logic in any analysis script); the values were reproduced "
            "from platform_divergence.csv, but the original derivation path is unknown.",
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(m["headline_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
