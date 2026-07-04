"""Shared helpers for paper/results_data regeneration (re-analysis only).

Rules: paper/main_vision.tex is source of truth and is READ-ONLY. No new
experiments. Every emitted number traces to accv_experiments/results/*.
"""
from __future__ import annotations
import shutil
from pathlib import Path

RD = Path(__file__).resolve().parent
ROOT = RD.parent.parent
RES = ROOT / "accv_experiments" / "results"
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
