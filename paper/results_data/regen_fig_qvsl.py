"""fig:qvsl (C2 bar figure, now referenced at main_vision.tex L252 as c2_q_vs_l_bar.pdf).
The figure + underlying CSV are produced by analysis/a1c2e1_refresh.py. This package copies
them in and documents the regen path (no data invented here)."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ANALYSIS, RD, ROOT, backup_stale

PKG = RD / "fig_qvsl"
PAPER_FIG = ROOT / "paper" / "figures"


def main():
    PKG.mkdir(exist_ok=True)
    pdf = ANALYSIS / "c2_q_vs_l_bar.pdf"
    csv = ANALYSIS / "c2_per_size_losses.csv"
    if not pdf.exists() or not csv.exists():
        subprocess.run([sys.executable, str(ANALYSIS / "a1c2e1_refresh.py")], check=True, cwd=ANALYSIS)
    backup_stale(PKG / "c2_q_vs_l_bar.pdf")
    shutil.copy(pdf, PKG / "c2_q_vs_l_bar.pdf")
    shutil.copy(csv, PKG / "c2_per_size_losses.csv")
    # keep the paper-referenced copy fresh too
    PAPER_FIG.mkdir(exist_ok=True)
    shutil.copy(pdf, PAPER_FIG / "c2_q_vs_l_bar.pdf")
    (PKG / "README.txt").write_text(
        "fig:qvsl (main_vision.tex L252, \\includegraphics{c2_q_vs_l_bar.pdf}).\n"
        "Regenerate: python analysis/a1c2e1_refresh.py  (Q vs L per-size bar).\n"
        "Absolute dsAP labels; medium staleness -0.030; error bars = 3-rep std "
        "(Quantization std=0, bars omitted).\n")
    print("fig_qvsl: copied c2_q_vs_l_bar.pdf + c2_per_size_losses.csv (source: a1c2e1_refresh.py)")
    return True


if __name__ == "__main__":
    main()
