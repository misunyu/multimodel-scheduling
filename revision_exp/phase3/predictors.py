"""R2-1 direct predictors (Ridge / Random Forest / MLP) and R2-2 XGBoost feature ablations.

Grids are fixed before any run (revision_audit/REPORT.md section D-1). Selection: mean over the 3 Phase 2 family folds of the
UNWEIGHTED validation MAE, ties -> first grid candidate (same rule as train_multi.cv_search). The D regressor of every method gets
the Phase 2 sample weight (train_multi.d_weights: 10 where D > 0.01, else 1).
"""
import sys

sys.dont_write_bytecode = True

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_multi as TM

SEED = 42
GRIDS = {
    "ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "rf": {"n_estimators": [300, 600], "max_depth": [None, 12], "min_samples_leaf": [1, 3]},
    "mlp": {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-4, 1e-2], "learning_rate_init": [1e-3]},
}
NAMES = {"ridge": "Ridge", "rf": "Random Forest", "mlp": "MLP (small)"}


def make(kind, params):
    if kind == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("est", Ridge(**params))])
    if kind == "rf":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    if kind == "mlp":
        return Pipeline([("scaler", StandardScaler()),
                         ("est", MLPRegressor(max_iter=2000, early_stopping=False, random_state=SEED, **params))])
    raise ValueError(kind)


def fit(kind, params, X, y, w=None):
    """Returns (model, number of ConvergenceWarnings). The weight goes to the estimator only (scaler unweighted)."""
    m = make(kind, params)
    kw = {} if w is None else ({"est__sample_weight": w} if isinstance(m, Pipeline) else {"sample_weight": w})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, y, **kw)
    if isinstance(m, Pipeline):
        sc = m.named_steps["scaler"]
        assert np.allclose(sc.mean_, X.mean(axis=0)), "scaler was not fitted on exactly the rows passed to fit()"
    return m, sum(issubclass(c.category, ConvergenceWarning) for c in caught)


def cv_search(kind, X, y, fold, w=None, label=""):
    Xv, yv = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    cands = TM.grid_candidates(GRIDS[kind])
    recs = []
    for ci, params in enumerate(cands):
        maes, conv, t0 = [], 0, time.perf_counter()
        for k in sorted(np.unique(fold)):
            tr, va = fold != k, fold == k
            m, cw = fit(kind, params, Xv[tr], yv[tr], None if w is None else w[tr])
            if isinstance(m, Pipeline):   # scaler statistics come from the training part of the fold only
                assert np.allclose(m.named_steps["scaler"].mean_, Xv[tr].mean(axis=0))
                assert m.named_steps["scaler"].n_samples_seen_ == tr.sum()
            conv += cw
            maes.append(float(np.abs(m.predict(Xv[va]) - yv[va]).mean()))
        recs.append({"method": NAMES[kind], "target": label, "candidate": ci, "params": str(params),
                     **{f"mae_fold{k}": v for k, v in enumerate(maes)}, "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                     "convergence_warnings": conv, "seconds": round(time.perf_counter() - t0, 2), "sample_weight": w is not None})
    df = pd.DataFrame(recs)
    best = int(df["mae_mean"].values.argmin())
    df["selected"] = df["candidate"] == best
    return cands[best], df


def train_direct(kind, X, train, fold, log=print):
    """T and D regressors. Returns ({'T': model, 'D': model}, records, info)."""
    yT, yD = train["T"].values, train["D"].values
    w = TM.d_weights(yD)
    models, recs, info = {}, [], {}
    for tgt, y, wt in (("T", yT, None), ("D", yD, w)):
        t0 = time.perf_counter()
        best, r = cv_search(kind, X, y, fold, wt, tgt)
        m, cw = fit(kind, best, np.asarray(X, dtype=float), y, wt)
        models[tgt] = m
        recs.append(r)
        sel = r[r["selected"]].iloc[0]
        info[tgt] = {"best_params": str(best), "cv_mae_mean": float(sel["mae_mean"]), "cv_mae_folds": [float(sel[f"mae_fold{k}"]) for k in range(3)],
                     "seconds": round(time.perf_counter() - t0, 1), "refit_convergence_warnings": int(cw), "sample_weight": wt is not None}
        log(f"[{NAMES[kind]}] {tgt}: best={best} cv_mae={sel['mae_mean']:.4f} ({info[tgt]['seconds']}s, conv.warn cv={int(r['convergence_warnings'].sum())} refit={cw})")
    return models, pd.concat(recs, ignore_index=True), info


def train_xgb_ablation(X, train, fold, art_dir, tag, log=print):
    """XGBoost with the Phase 2 grid / folds / weight on a column subset. Returns ({'T','D'}, records, info)."""
    art_dir.mkdir(parents=True, exist_ok=True)
    yT, yD = train["T"].values, train["D"].values
    w = TM.d_weights(yD)
    models, recs, info = {}, [], {}
    for tgt, y, wt in (("T", yT, None), ("D", yD, w)):
        t0 = time.perf_counter()
        best, r = TM.cv_search(X, y, fold, TM.GRID, weight=wt, label=f"{tag}:{tgt}")
        m = TM.fit_one(X.values, y, best, wt)
        m.save_model(str(art_dir / f"{tgt}.json"))
        models[tgt] = m
        sel = r[r["selected"]].iloc[0]
        r = r.assign(method=f"XGBoost {tag}", target=tgt)
        recs.append(r)
        info[tgt] = {"best_params": str(best), "cv_mae_mean": float(sel["mae_mean"]), "cv_mae_folds": [float(sel[f"mae_fold{k}"]) for k in range(3)],
                     "seconds": round(time.perf_counter() - t0, 1), "sample_weight": wt is not None}
        log(f"[XGBoost {tag}] {tgt}: best={best} cv_mae={sel['mae_mean']:.4f} ({info[tgt]['seconds']}s)")
    return models, pd.concat(recs, ignore_index=True), info
