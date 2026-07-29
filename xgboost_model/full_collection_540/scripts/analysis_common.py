"""Shared utilities for the P1-P6 paper experiments (MLForSys workshop).

Everything here computes over the STORED 540-window collection only — no hardware,
no new profiling. All randomness is seed=42.

Dataset loading deliberately bypasses build_dataset(): the full540 windows carry no
"schedule file" hint, so build_dataset's schedule lookup silently fails and every
view.infps feature trains as 0 (this is how the shipped deploy_cpu_* artifacts were
trained, while the GUI predict path feeds real infps — a train/serve skew). Here the
per-combination infps map is taken directly from the platform's collection YAML, so
features match the predict-time path. Pass with_infps=False to reproduce the legacy
zero-infps behaviour for comparison.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
sys.path.insert(0, str(ROOT))

from xgboost_model.deploy_selector_xgb_suite import (  # noqa: E402
    _PARAM_GRID, _PARAMS, _build_infps_lookup, _norm_exec, _normalize_targets,
    featurize_window, load_static_profiles, score_combo,
)

FC = ROOT / "xgboost_model" / "full_collection_540"
ANALYSIS = FC / "analysis"
ARTIFACTS = ROOT / "xgboost_model" / "artifacts"
STATIC_JSON = ROOT / "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json"
SCHED_DIR = ROOT / "xgboost_model" / "schedules" / "collection"

PLATFORMS = ("gpu", "npu")
GEN_MODELS = {"llama1b", "qwen2_vl"}
ALPHA, BETA = 0.3, 1.0
SEED = 42
N_FOLDS = 3
# Canonical 37-feature order = the shipped artifacts' features.json (byte-identical
# across all six shipped artifacts). Keeps column order stable across platforms.
FEATURE_ORDER: List[str] = json.loads(
    (ARTIFACTS / "deploy_cpu_gpu_features.json").read_text())

TARGETS = [("y1", "y1_total_throughput_fps"),
           ("y2", "y2_deadline_miss_rate"),
           ("y3", "y3_total_tokens_per_s")]


# ---------- provenance ----------
def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def provenance(**extra) -> Dict:
    p = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_head() + " (pre-commit HEAD)",
        "alpha": ALPHA, "beta": BETA, "seed": SEED,
        "input_data": [str(FC / f"cpu_{pl}" / f"performance_{pl}_full540.json")
                       for pl in PLATFORMS],
    }
    p.update(extra)
    return p


def prov_md_header(**extra) -> str:
    p = provenance(**extra)
    lines = ["<!-- provenance"]
    for k, v in p.items():
        lines.append(f"  {k}: {v}")
    lines.append("-->")
    return "\n".join(lines) + "\n\n"


def prov_csv_comment(**extra) -> str:
    return "# provenance: " + json.dumps(provenance(**extra), ensure_ascii=False) + "\n"


# ---------- data loading ----------
def load_windows(platform: str) -> List[Dict]:
    path = FC / f"cpu_{platform}" / f"performance_{platform}_full540.json"
    return json.loads(path.read_text())


def load_schedule_doc(platform: str) -> Dict:
    return yaml.safe_load((SCHED_DIR / f"collect_cpu_{platform}.yaml").read_text())


def infps_map_for(platform: str) -> Dict[str, Dict[Tuple[str, str], float]]:
    """combination name -> {(model, device): infps} (rate factor already baked in)."""
    doc = load_schedule_doc(platform)
    out = {}
    for combo in doc:
        out[combo] = _build_infps_lookup(doc, combo)
    return out


def load_dataset(platform: str, with_infps: bool = True):
    """Return (X, Y_raw, M) for one platform's 540 windows.

    X: 37 features in the canonical artifact order.
    Y_raw: unnormalized targets (normalize with normalize_targets()).
    M: combination, workload, rate_factor, models, devices, y3_valid, has_gen,
       accel_set (sorted tuple of models placed on the accelerator), platform.
    """
    S = load_static_profiles(STATIC_JSON)
    imap = infps_map_for(platform) if with_infps else {}
    windows = load_windows(platform)
    X_rows, Y_rows, M_rows = [], [], []
    for w in windows:
        m = imap.get(str(w.get("combination"))) if with_infps else None
        X, (y1, y2, y3), meta = featurize_window(w, S, infps_map=m)
        if not np.isfinite(y2):
            y2 = 0.0
        X_rows.append(X)
        Y_rows.append({"y1_total_throughput_fps": y1,
                       "y2_deadline_miss_rate": y2,
                       "y3_total_tokens_per_s": y3})
        accel = tuple(sorted(v["model"] for v in w["models"].values()
                             if _norm_exec(v["execution"]) in ("gpu", "npu")))
        meta["accel_set"] = accel
        meta["has_gen"] = any(mm in GEN_MODELS for mm in meta["models"].split(","))
        meta["platform"] = platform
        M_rows.append(meta)
    X = pd.DataFrame(X_rows).reindex(columns=FEATURE_ORDER).fillna(0.0)
    Y = pd.DataFrame(Y_rows)
    M = pd.DataFrame(M_rows)
    return X, Y, M


def normalize_targets(Y_raw: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """Group-wise ((models, workload, rate)) normalization, as in the suite.

    Fold-safety note: the P1+ protocol assigns FOLDS BY GROUP, and every statistic
    here (Fmax, y2 min/max) is computed strictly within one group, so no statistic
    ever mixes rows from different folds. Within a held-out group this is a rank-
    preserving (monotone) transform of that group's raw values; since all evaluation
    is within-group ranking, using the group's own stats is not leakage.
    """
    return _normalize_targets(Y_raw, M)


# ---------- grouping / folds ----------
def group_keys(M: pd.DataFrame, with_platform: bool = False) -> pd.Series:
    """(models, rate_factor) per row — the paper's CV group unit."""
    if with_platform:
        return pd.Series(list(zip(M["platform"], M["models"], M["rate_factor"])),
                         index=M.index)
    return pd.Series(list(zip(M["models"], M["rate_factor"])), index=M.index)


def assign_group_folds(groups: pd.Series, n_folds: int = N_FOLDS,
                       seed: int = SEED) -> pd.Series:
    """GroupKFold with a seeded shuffle: every row of a group lands in ONE fold.

    Unique groups are shuffled with RandomState(seed) and dealt round-robin, which
    balances fold sizes to within one group. Deterministic for a given group list.
    """
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[j]: (i % n_folds) for i, j in enumerate(order)}
    return groups.map(fold_of)


# ---------- metrics ----------
def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def scores_from(y1, y2, y3, has_gen, alpha: float = ALPHA, beta: float = BETA):
    """Vectorized S = y1 - alpha*y2 (+ beta*y3 for generative sets)."""
    y1 = np.asarray(y1, float); y2 = np.asarray(y2, float)
    y3 = np.nan_to_num(np.asarray(y3, float), nan=0.0)
    hg = np.asarray(has_gen, bool)
    return y1 - alpha * y2 + beta * y3 * hg


def ranking_metrics(df: pd.DataFrame, group_col="group",
                    meas_col="s_meas", pred_col="s_pred") -> Dict:
    """Per-group Top-1 / Top-5 hit rates and mean group Spearman of scores.

    Top-1: predictor's argmax is one of the measured argmaxes (1e-9 tie set).
    Top-5: some measured argmax appears in the predictor's top-5 (the
    evaluate_model.py convention).
    """
    top1, top5, rhos = [], [], []
    for _, g in df.groupby(group_col, sort=False):
        meas = g[meas_col].values
        pred = g[pred_col].values
        best = meas.max()
        oracle = set(np.where(np.abs(meas - best) < 1e-9)[0])
        p_order = np.argsort(-pred, kind="stable")
        top1.append(1.0 if p_order[0] in oracle else 0.0)
        top5.append(1.0 if oracle & set(p_order[:5]) else 0.0)
        rhos.append(spearman(meas, pred))
    rhos = [r for r in rhos if np.isfinite(r)]
    return {"n_groups": int(df[group_col].nunique()),
            "top1": float(np.mean(top1)), "top5": float(np.mean(top5)),
            "group_spearman_mean": float(np.mean(rhos)) if rhos else float("nan")}


def per_group_spearman(df: pd.DataFrame, group_col, a_col, b_col) -> Dict:
    out = {}
    for k, g in df.groupby(group_col, sort=False):
        out[str(k)] = spearman(g[a_col].values, g[b_col].values)
    return out


# ---------- xgboost helpers ----------
def _lazy_xgb():
    import xgboost as xgb
    return xgb


def cv_select_params(Xv: np.ndarray, yv: np.ndarray, inner_groups: pd.Series,
                     rounds: int = 300, seed: int = SEED,
                     feature_names: List[str] = None) -> Tuple[Dict, float]:
    """Grid search (the suite's 4-combo grid) by GROUP-aware inner 3-fold CV, MAE.

    Unlike the suite's modulo-fold _cv_select_params, inner folds are also assigned
    by (models, rate) group so hyperparameter selection sees no within-group leakage.
    feature_names defaults to the 37-feature window schema; P6 passes its own.
    """
    xgb = _lazy_xgb()
    feat = feature_names or FEATURE_ORDER
    inner_fold = assign_group_folds(inner_groups.reset_index(drop=True),
                                    n_folds=3, seed=seed).values
    best, best_mae = None, float("inf")
    for extra in _PARAM_GRID:
        params = dict(_PARAMS); params.update(extra)
        maes = []
        for k in range(3):
            tr, te = inner_fold != k, inner_fold == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            d = xgb.DMatrix(Xv[tr], label=yv[tr], feature_names=feat)
            bst = xgb.train(params, d, num_boost_round=rounds)
            pred = bst.predict(xgb.DMatrix(Xv[te], feature_names=feat))
            maes.append(float(np.mean(np.abs(pred - yv[te]))))
        mae = float(np.mean(maes)) if maes else float("inf")
        if mae < best_mae:
            best_mae, best = mae, params
    return best or dict(_PARAMS), best_mae


def train_booster(Xv: np.ndarray, yv: np.ndarray, params: Dict, rounds: int = 300,
                  feature_names: List[str] = None):
    xgb = _lazy_xgb()
    d = xgb.DMatrix(Xv, label=yv, feature_names=feature_names or FEATURE_ORDER)
    return xgb.train(params, d, num_boost_round=rounds)


def predict_booster(bst, Xv: np.ndarray, clip=True,
                    feature_names: List[str] = None) -> np.ndarray:
    xgb = _lazy_xgb()
    p = bst.predict(xgb.DMatrix(Xv, feature_names=feature_names or FEATURE_ORDER))
    return np.clip(p, 0.0, 1.0) if clip else p


def load_booster(path: Path):
    xgb = _lazy_xgb()
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst
