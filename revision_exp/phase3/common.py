"""Shared setup for Phase 3 (reviewer experiments on the re-profiled CPU-GPU dataset).

Every module of revision_exp/ is imported unchanged; only D.DS, D.SETS and D.OUT are pointed at the Phase 2 dataset / Phase 3 output.
Import this module first. Scripts in this folder replace sys.path[0] (their own folder) by revision_exp/, so `import baselines`
keeps resolving to revision_exp/baselines.py and the per-instance code is imported as `phase3.baselines.*`.
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "revision_exp"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(REPO / "revision_exp_adapter") not in sys.path:
    sys.path.append(str(REPO / "revision_exp_adapter"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

DATA = REPO / "revision_profiling" / "dataset"
PROF = REPO / "revision_profiling"
P2 = EXP / "out_reprofiled"                     # Phase 2 results (read-only)
P2_ART = P2 / "artifacts" / "multi"
OUT = EXP / "out_phase3"
ALPHA = 0.3
SEED = 42
WINDOW_SEC = 10.0

D.DS = DATA
D.SETS = {"train": ("train_x3.csv", "train_schedules_x3.csv", []), "x3": ("test_x3.csv", "test_schedules_x3.csv", [])}
D.OUT = OUT
assert D.ALPHA == ALPHA

import run_reprofiled as RR  # noqa: E402  (Phase 2 adapter: helper functions only, main() is not called)

_T0 = time.time()
GUARD = {"final": False}       # set to True only inside the final evaluation


def allow_test_labels():
    GUARD["final"] = True


def assert_final(what):
    assert GUARD["final"], f"{what}: 3x labels may only be read in the final evaluation"


def log(msg, name="phase3.log"):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / name, "a") as f:
        f.write(line + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train():
    """Training candidates (1, 2, 4x) through the unchanged loader, D = D_macro."""
    df = RR.load("train", "macro")
    assert set(df["rate"]) == {"x1", "x2", "x4"}
    return df


def load_test():
    assert_final("load_test")
    df = RR.load("x3", "macro")
    assert set(df["rate"]) == {"x3"}
    return df


def phase2_folds(train):
    """Family -> fold as stored by Phase 2 (out_reprofiled/folds.csv), joined on the family key."""
    f = pd.read_csv(P2 / "folds.csv")
    m = dict(zip(f["family"], f["fold"]))
    assert set(m) == set(train["family"]), "fold file does not cover exactly the training families"
    return train["family"].map(m).astype(int).values


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"
