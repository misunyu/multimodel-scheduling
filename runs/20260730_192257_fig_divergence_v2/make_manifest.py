"""Write run_manifest.json for the Figure 2 (fig_divergence) declared-basis regeneration."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "xgboost_model/full_collection_540/analysis"

# Files this run created or rewrote (all outside runs/ -- this task explicitly
# directed in-place replacement, unlike the three preceding read-only runs).
MODIFIED = [
    "xgboost_model/full_collection_540/scripts/make_figures.py",
    "xgboost_model/full_collection_540/analysis/figures/fig_divergence.pdf",
    "xgboost_model/full_collection_540/analysis/figures/fig_divergence.png",
    "xgboost_model/full_collection_540/analysis/figures/README.md",
    "xgboost_model/full_collection_540/analysis/platform_divergence.md",
    "xgboost_model/full_collection_540/analysis/platform_divergence.csv",
]
RENAMED = {
    "xgboost_model/full_collection_540/analysis/platform_divergence.md":
        "xgboost_model/full_collection_540/analysis/platform_divergence_raw_v1.md",
    "xgboost_model/full_collection_540/analysis/platform_divergence.csv":
        "xgboost_model/full_collection_540/analysis/platform_divergence_raw_v1.csv",
}
INPUTS = [
    "runs/20260730_190154_score_basis_audit/divergence_by_group.csv",
    "runs/20260730_190154_score_basis_audit/build_basis_audit.py",
    "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json",
    "xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json",
]
# Stale references to the OLD (raw-basis) platform_divergence.{md,csv}, reported only.
STALE_REFERENCES = [
    {"path": "xgboost_model/full_collection_540/scripts/compare_platforms_v2.py",
     "lines": [9, 10, 86, 111],
     "issue": "WRITE COLLISION: this raw-basis generator still writes "
              "analysis/platform_divergence.{csv,md}; re-running it would silently "
              "overwrite the v2 declared-basis files",
     "severity": "high"},
    {"path": "xgboost_model/full_collection_540/analysis/SUMMARY.md",
     "lines": [58],
     "issue": "P2 output-path row points at platform_divergence.{csv,md}, whose content "
              "is now v2; SUMMARY's own P2 figure (29/45) and decision #6 ('P2 score is "
              "raw measured totals') now describe the renamed v1 file",
     "severity": "medium"},
    {"path": "xgboost_model/full_collection_540/analysis/platform_divergence_raw_v1.md",
     "lines": [26],
     "issue": "self-reference 'see platform_divergence.csv' now resolves to the v2 csv "
              "instead of its own platform_divergence_raw_v1.csv",
     "severity": "low"},
    {"path": "runs/20260730_172053_coverage/{coverage_report.md,build_coverage.py,"
             "make_manifest.py}",
     "lines": [],
     "issue": "historical run records citing platform_divergence.md for the raw 29/45 and "
              "30/45; correct as of their execution, now pointing at renamed content",
     "severity": "informational"},
    {"path": "runs/20260730_184450_coeff_sweep/{sweep_report.md,build_sweep.py,"
             "sweep_facts.json,make_manifest.py}",
     "lines": [],
     "issue": "same as above (E-1 reconciliation against the raw 29/45, 30/45)",
     "severity": "informational"},
    {"path": "runs/20260730_190154_score_basis_audit/{basis_audit.md,run_manifest.json,"
             "make_manifest.py,build_basis_audit.py}",
     "lines": [],
     "issue": "same as above; the audit's part-1 table intentionally documents the raw "
              "basis of the then-current platform_divergence.{md,csv}",
     "severity": "informational"},
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    figtxt = (OUT / "fig_divergence_text.txt")
    text = figtxt.read_text() if figtxt.exists() else ""
    m = {
        "run_id": OUT.name,
        "purpose": ("Regenerate manuscript Figure 2 (fig_divergence) on the declared "
                    "Eq. (1) basis and replace the raw-basis platform_divergence "
                    "analysis artifacts with basis-explicit v2 versions"),
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "git_commit_note": "HEAD at execution time (outputs committed on top of it)",
        "generator": {
            "script": "xgboost_model/full_collection_540/scripts/make_figures.py",
            "sha256": sha256(ROOT / "xgboost_model/full_collection_540/scripts/make_figures.py"),
            "invocation": "python make_figures.py --fig f1 --basis declared_normalized",
            "change": "added an explicit --basis parameter (default declared_normalized); "
                      "F1 now recomputes divergence through the audit run's "
                      "build_basis_audit.normalize()/score()/tie_sets() instead of reading "
                      "a basis-implicit csv, and asserts the result before drawing. "
                      "The raw path is retained as --basis raw over the renamed v1 csv. "
                      "Original version is preserved in git history (no file-copy backup).",
            "modified_in_place_not_new_script": True,
        },
        "basis": {
            "name": "declared_normalized",
            "group": "(set, rate)",
            "definition": "y1 = r1/max(r1); y3 = r3/max(r3); "
                          "y2 = (r2-min r2)/(max r2-min r2), 0 when max == min; "
                          "S = y1 - 0.3*y2 + beta*y3 (beta term only for generative sets)",
            "beta": 1.0,
            "beta_0_5_figure_generated": False,
            "tie_rule": "1e-9 tie set; intersecting tie sets count as agreement",
            "score_definitions_shared_from":
                "runs/20260730_190154_score_basis_audit/build_basis_audit.py",
        },
        "assert_results": {
            "n_disagree": 11, "n_groups": 45,
            "expected_groups": sorted([
                "S10@2.0", "S5@3.0", "S7@2.0", "S8@2.0", "base3@1.0", "base3@3.0",
                "base4@1.0", "base4@3.0", "base5@1.0", "base5@2.0", "base5@3.0"]),
            "matches_audit_csv": True,
            "matches_expected_11_group_list": True,
            "passed": True,
            "note": "both asserts live inside make_figures.load_divergence_declared(); "
                    "the figure is not written unless they pass",
        },
        "figure_text_verification": {
            "method": "pdftotext fig_divergence.pdf",
            "residual_29_occurrences": text.count("29"),
            "contains_11_of_45": "11/45" in text,
            "legend_line": next((l for l in text.splitlines()
                                 if "different placement" in l), None),
            "passed": text.count("29") == 0 and "11/45" in text,
        },
        "input_artifacts": [{"path": p, "sha256": sha256(ROOT / p),
                             "bytes": (ROOT / p).stat().st_size} for p in INPUTS],
        "outputs_modified_in_place": [
            {"path": p, "sha256": sha256(ROOT / p), "bytes": (ROOT / p).stat().st_size}
            for p in MODIFIED],
        "renamed_for_preservation": RENAMED,
        "stale_references_to_old_platform_divergence": STALE_REFERENCES,
        "stale_references_note": "reported only; no edits made to these files, per scope",
        "unknowns": [],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: m[k] for k in
                      ("assert_results", "figure_text_verification")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
