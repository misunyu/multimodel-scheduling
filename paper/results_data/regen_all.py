"""Regenerate ALL paper/results_data packages from existing logs, then cross-check
against main_vision.tex. Re-analysis only; paper is read-only source of truth.

Supersedes the pre-rev30 generator (kept as regen_all_stale_backup.py).
Order: existing-4 (rev30-aligned) -> table3 (rev30) -> new-5 -> appendix -> verify.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SCRIPTS = [
    "regen_table1_single_stream.py",
    "regen_table2_decomp.py",
    "regen_fig2_sweep.py",
    "regen_table3_main.py",
    "regen_table4_persize.py",
    "regen_table_policy_comparison.py",
    "regen_table_metric_sensitivity.py",
    "regen_table_detector_generality.py",
    "regen_qos.py",
    "regen_fig_qvsl.py",
    "regen_appendix.py",
]


def main():
    results = {}
    for s in SCRIPTS:
        print(f"\n{'='*70}\n### {s}\n{'='*70}")
        r = subprocess.run([PY, str(HERE / s)], cwd=HERE)
        results[s] = r.returncode
    print(f"\n{'='*70}\n### verify_against_paper.py\n{'='*70}")
    v = subprocess.run([PY, str(HERE / "verify_against_paper.py")], cwd=HERE)
    print("\n=== regen_all summary ===")
    for s, rc in results.items():
        print(f"  {'ok ' if rc == 0 else 'ERR'} {s}")
    print(f"  verify exit={v.returncode} (0=all pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
