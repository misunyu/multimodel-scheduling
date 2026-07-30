"""Run manifest for the read-only manuscript cross-check."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent

READ_ONLY_INPUTS = [
    "manuscript/mlforsys_main.tex",
    "xgboost_model/full_collection_540/analysis/figures/fig_divergence.pdf",
    "runs/20260730_190154_score_basis_audit/build_basis_audit.py",
    "runs/20260730_190154_score_basis_audit/basis_facts.json",
    "runs/20260730_190154_score_basis_audit/sweep_declared_basis.csv",
    "runs/20260730_172053_coverage/appendix_coverage.csv",
    "runs/20260730_172053_coverage/coverage_facts.json",
    "xgboost_model/full_collection_540/analysis/groupkfold_gpu_metrics.json",
    "xgboost_model/full_collection_540/analysis/groupkfold_npu_metrics.json",
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r1 = json.loads((OUT / "r1_convention_recheck.json").read_text())
    tex = ROOT / "manuscript/mlforsys_main.tex"
    fig = ROOT / "xgboost_model/full_collection_540/analysis/figures/fig_divergence.pdf"
    m = {
        "run_id": OUT.name,
        "purpose": ("Read-only cross-check of manuscript/mlforsys_main.tex against the "
                    "repository artifacts, to close the UNKNOWNs carried by the four "
                    "preceding runs"),
        "read_only": True,
        "manuscript_modified": False,
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "outputs": [
            {"file": "xcheck.md", "role": "sections 1-6: mirror currency, basis, r1, tie, "
                                          "UNKNOWN closure table, manuscript-side findings"},
            {"file": "r1_convention_recheck.json",
             "role": "the required re-verification: disagreement SETS under both r1 "
                     "conventions, both betas"},
        ],
        "step1_mirror_currency": {
            "marker_found": True,
            "marker_line": 5,
            "marker": "REVISION: 2026-07-30 r2 -- declared-basis numbers (11/45), Fig.1 "
                      "removed, Appendix D/E added, 4-page body confirmed, fig_divergence "
                      "v2 (sha256 3c13b1df...) embedded",
            "aborted": False,
            "embedded_figure_hash_check": {
                "claimed_prefix": "3c13b1df",
                "actual_sha256": sha256(fig),
                "matches": sha256(fig).startswith("3c13b1df"),
            },
            "manuscript_sha256": sha256(tex),
            "manuscript_lines": len(tex.read_text().splitlines()),
        },
        "step2_basis": {
            "verdict": "manuscript states the normalized (Eq. 1) basis explicitly",
            "quote_line_99": "All measured-best placements and divergence counts below use "
                             "this same normalized score; comparing raw measurements across "
                             "platforms would instead let unit-scale differences dominate.",
            "generators_agree": True,
            "generators": ["scripts/make_figures.py --fig f1 --basis declared_normalized",
                           "scripts/compare_platforms_v2.py --basis declared_normalized"],
            "numeric_crosscheck": {
                "appendix_D_cells_compared": 25, "appendix_D_mismatches": 0,
                "appendix_E_cells_compared": 75, "appendix_E_mismatches": 0,
                "headline_checks": 13, "headline_failures": 0,
                "raw_basis_figures_29_45_or_30_45_present_in_tex": False,
            },
        },
        "step3_r1_convention": {
            "manuscript_r1": "aggregate vision throughput (line 91) = vision-view sum",
            "generator_primary_r1": "total.total_throughput_fps = sum over ALL views",
            "rows_differing": {"gpu": 443, "npu": 461},
            "recheck_both_betas_identical_sets": r1["all_identical_both_betas"],
            "per_beta": r1["set_equality_by_beta"],
            "matches_audit_json": r1["matches_audit_json"],
            "aborted": False,
            "verdict": "no effect on any reported number; definitional mismatch flagged for "
                       "the chat session to resolve (manuscript half-sentence or generator "
                       "default), not changed here",
        },
        "step4_tie_rule": {
            "manuscript_has_tie_statement": True,
            "quotes": {
                "line_121": "The oracle selects the candidate with the highest measured "
                            "score S in each workload group and treats tied maxima as "
                            "correct.",
                "line_319": "whose group optimum leaves the baseline optimum set "
                            "(alpha=0.3, beta=1.0; tie-aware)",
            },
            "numeric_tolerance_in_manuscript": False,
            "code_convention": "1e-9 tie set (analysis_common.ranking_metrics, "
                               "build_basis_audit.TIE_EPS)",
            "semantics_match": True,
            "groups_with_ties_declared_basis_both_betas": 0,
            "note": "record as 'manuscript states tie-aware semantics; the 1e-9 tolerance "
                    "is a code convention only'. Whether to add a half-sentence to the "
                    "appendix is a chat-session decision.",
        },
        "step5_unknown_closure": {
            "closed": 5, "partially_closed": 1, "not_closed": 1,
            "not_closed_items": [
                "origin script for the 29/33 vs 0/12 gen/vision split (superseded by "
                "10/33 vs 1/12; no split logic in the repo, not stated in the manuscript)",
                "the 'F3' identifier itself (partially closed: Appendix D's table matches "
                "the pre-registered grid 25/25, but no 'F3' string exists in the manuscript)",
            ],
        },
        "manuscript_side_findings_reported_not_changed": [
            {"line": 101, "severity": "medium",
             "issue": "'device memory determines placement feasibility' -- no memory "
                      "constraint exists in the candidate generator; feasibility is the "
                      "VLM pin alone, as Appendix E's 2^(N-#VLM) already states"},
            {"lines": [270, 315], "severity": "low",
             "issue": "appendix subsections are ordered A, B, C, E, D"},
            {"lines": [8, 21], "severity": "low",
             "issue": "neurips_2025.sty in use; comment marks the 2026 swap as PENDING"},
            {"lines": [15, 61], "severity": "info",
             "issue": "\\RES/\\TODO macros defined; zero uses in body or appendix "
                      "(only a comment and a commented-out author line)"},
        ],
        "input_artifacts": [{"path": p, "sha256": sha256(ROOT / p),
                             "bytes": (ROOT / p).stat().st_size}
                            for p in READ_ONLY_INPUTS],
        "unknowns": [],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: m[k] for k in ("step1_mirror_currency", "step3_r1_convention",
                                        "step4_tie_rule", "step5_unknown_closure")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
