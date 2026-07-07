"""Shared helpers for the paper/reproduction package.

Adapted (logic unchanged) from paper/results_data/_common.py. The ONLY change
is path anchoring: this file lives at paper/reproduction/_common.py, so the
repo root is two levels up and the paper cross-check reads the real, read-only
paper/main_vision.tex. Each item's generate.py overrides RES/PKG to its own
local ./data (inputs) and ./ (outputs) so the item is self-contained.
"""
from __future__ import annotations
import shutil
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent          # paper/reproduction
ROOT = REPRO_ROOT.parent.parent                       # repo root
RES = ROOT / "accv_experiments" / "results"           # default; items override to ./data
RD = REPRO_ROOT                                        # default; items override to ./
ANALYSIS = ROOT / "analysis"
PAPER_TEX = ROOT / "paper" / "main_vision.tex"


def paper_text() -> str:
    return PAPER_TEX.read_text()


def backup_stale(path: Path):
    """Copy an existing output to <stem>_stale_backup<suffix> once (never overwrite a backup)."""
    path = Path(path)
    if not path.exists():
        return None
    bak = path.with_name(f"{path.stem}_stale_backup{path.suffix}")
    if not bak.exists():
        shutil.copy(path, bak)
    return bak


def tex_contains(value: str, text: str | None = None) -> bool:
    """Whether a literal token appears in the paper tex (used by verify)."""
    return value in (text if text is not None else paper_text())
