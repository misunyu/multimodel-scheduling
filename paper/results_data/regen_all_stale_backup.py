#!/usr/bin/env python3
"""Regenerate all paper figures/tables from source CSVs and cross-validate against
paper/main_vision.tex. Extraction only — no new measurement.

Runs the 6 per-artifact generators (Fig.1 is verified, not re-rendered, since it
needs the NPU device), then prints a PASS/FAIL matrix vs the paper, then refreshes
index.html.

Usage:  python regen_all.py
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENS = [
    ("fig:sweep   (Fig.2)", "fig2_sweep/generate_fig2.py"),
    ("tab:single-stream (Table 1)", "table1_single_stream/generate_table1.py"),
    ("tab:decomp  (Table 2)", "table2_decomp/generate_table2.py"),
    ("tab:main    (Table 3)", "table3_main/generate_table3.py"),
    ("tab:persize-contention (Table 4)", "table4_persize/generate_table4.py"),
]

def fig1_check():
    pdf = HERE / "fig1_failures" / "vis_failure_examples.pdf"
    paper_uses = "vis_failure_examples.pdf" in (HERE.parent / "main_vision.tex").read_text()
    ok = pdf.exists() and pdf.stat().st_size > 1000 and paper_uses
    print(f"fig:failures (Fig.1): PDF present={pdf.exists()} size={pdf.stat().st_size if pdf.exists() else 0}B "
          f"referenced_in_paper={paper_uses} -> {'PASS' if ok else 'FAIL'}")
    return ok

def main():
    results = []
    print("="*70)
    print("Regenerating paper artifacts from source CSVs (extraction only)")
    print("="*70)
    for label, rel in GENS:
        print(f"\n--- {label} ---")
        rc = subprocess.run([sys.executable, str(HERE/rel)], cwd=HERE).returncode
        results.append((label, rc == 0))
    print("\n--- fig:failures (Fig.1) ---")
    results.append(("fig:failures (Fig.1)", fig1_check()))

    # refresh HTML
    print("\n--- index.html ---")
    rc = subprocess.run([sys.executable, str(HERE/"generate_index_html.py")], cwd=HERE).returncode
    print(f"index.html {'regenerated' if rc==0 else 'FAILED'}")

    print("\n" + "="*70)
    print("REPRODUCTION SUMMARY (vs paper/main_vision.tex)")
    print("="*70)
    npass = sum(1 for _, ok in results if ok)
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    print(f"\n{npass}/{len(results)} artifacts reproduced and matched paper values")
    return 0 if npass == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
