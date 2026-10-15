"""Multi-Objective predictor: separate XGBoost regressors for T and D (weighted / unweighted)."""
import sys

sys.dont_write_bytecode = True

import itertools
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

# --- parameters confirmed in the existing training path -------------------------------------------------------------
# xgboost_model/deploy_selector_xgb_suite.py:754-758 (train_double) and :593-597 (train_score):
#     xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
# Nothing else is set there, so every other parameter is the library default. They are not guessed here: the resolved
# values are read back from the fitted booster (save_config) and written to meta.json.
BASE_PARAMS = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": -1}
# train_double grid (:743-748). ParameterGrid iterates keys in sorted order; ties are resolved to the first candidate.
GRID = {"learning_rate": [0.01, 0.05], "max_depth": [4, 6], "n_estimators": [500, 1000], "subsample": [0.8, 1.0]}
D_WEIGHT_THRESHOLD = 0.01   # :775  weights = 10.0 if y2 > 0.01 else 1.0
D_WEIGHT_VALUE = 10.0


def grid_candidates(grid):
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def d_weights(y):
    return np.where(np.asarray(y) > D_WEIGHT_THRESHOLD, D_WEIGHT_VALUE, 1.0)


def fit_one(X, y, params, weight=None, fixed=None):
    m = xgb.XGBRegressor(**BASE_PARAMS, **(fixed or {}), **params)
    m.fit(X, y, sample_weight=weight)
    return m


def cv_search(X, y, fold, grid, weight=None, fixed=None, subset_mask=None, label=""):
    """Exhaustive grid with the supplied folds. Selection: mean over folds of the UNWEIGHTED validation MAE.
    Returns (best_params, records). Each fold model is fitted on the other folds only."""
    Xv, yv = X.values, np.asarray(y, dtype=float)
    recs = []
    for ci, params in enumerate(grid_candidates(grid)):
        maes, sub_maes, secs = [], [], []
        for k in sorted(np.unique(fold)):
            tr, va = fold != k, fold == k
            t0 = time.perf_counter()
            m = fit_one(Xv[tr], yv[tr], params, None if weight is None else weight[tr], fixed)
            secs.append(time.perf_counter() - t0)
            err = np.abs(m.predict(Xv[va]) - yv[va])
            maes.append(float(err.mean()))
            if subset_mask is not None:
                s = subset_mask[va]
                sub_maes.append(float(err[s].mean()) if s.any() else float("nan"))
        rec = {"model": label, "candidate": ci, **params, **(fixed or {}),
               **{f"mae_fold{k}": v for k, v in enumerate(maes)}, "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
               "fit_seconds_total": float(np.sum(secs))}
        if subset_mask is not None:
            rec.update({f"mae_subset_fold{k}": v for k, v in enumerate(sub_maes)})
            rec["mae_subset_mean"] = float(np.nanmean(sub_maes))
        recs.append(rec)
    df = pd.DataFrame(recs)
    best = int(df["mae_mean"].values.argmin())  # argmin returns the first minimum
    df["selected"] = df["candidate"] == best
    return grid_candidates(grid)[best], df


def resolved_config(model):
    """Full parameter set actually used by the booster (library defaults included)."""
    cfg = json.loads(model.get_booster().save_config())
    learner = cfg["learner"]
    out = {"generic_param": learner.get("generic_param", {}), "objective": learner.get("objective", {}),
           "learner_train_param": learner.get("learner_train_param", {})}
    gb = learner.get("gradient_booster", {})
    out["booster_name"] = gb.get("name")
    out["tree_train_param"] = gb.get("tree_train_param", {})
    out["gbtree_model_param"] = gb.get("gbtree_model_param", {})
    out["updater"] = [u.get("name") for u in gb.get("updater", [])] if isinstance(gb.get("updater"), list) else gb.get("updater")
    out["sklearn_params_set"] = {k: v for k, v in model.get_params().items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    return out


def train_multi(X, train_df, fold, art_dir, log=print):
    y_t, y_d = train_df["T"].values, train_df["D"].values
    w = d_weights(y_d)
    sub = y_d > D_WEIGHT_THRESHOLD
    specs = [("T", y_t, None, None), ("D", y_d, w, sub), ("D_noweight", y_d, None, sub)]
    meta = {"mode": "multi", "features": list(X.columns), "grid": GRID, "base_params": BASE_PARAMS,
            "d_weight": {"threshold": D_WEIGHT_THRESHOLD, "value": D_WEIGHT_VALUE, "rows_weighted": int(sub.sum()), "rows": int(len(y_d))},
            "selection": "mean over 3 family-grouped folds of unweighted validation MAE; ties -> first grid candidate",
            "models": {}}
    all_recs = []
    for name, y, weight, mask in specs:
        t0 = time.perf_counter()
        best, recs = cv_search(X, y, fold, GRID, weight=weight, subset_mask=mask, label=name)
        cv_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = fit_one(X.values, y, best, weight)
        refit_s = time.perf_counter() - t1
        model.save_model(str(art_dir / f"{name}.json"))
        sel = recs[recs["selected"]].iloc[0]
        meta["models"][name] = {
            "target": "throughput_norm" if name == "T" else "drop_rate_norm", "sample_weight": weight is not None,
            "best_params": best, "cv_mae_mean": float(sel["mae_mean"]), "cv_mae_folds": [float(sel[f"mae_fold{k}"]) for k in range(3)],
            "cv_mae_subset_mean(D>0.01)": float(sel["mae_subset_mean"]) if mask is not None else None,
            "cv_seconds": cv_s, "refit_seconds": refit_s, "resolved_config": resolved_config(model),
            "n_trees": int(model.get_booster().num_boosted_rounds()),
        }
        all_recs.append(recs)
        log(f"[multi] {name}: best={best} cv_mae={sel['mae_mean']:.4f} cv={cv_s:.1f}s refit={refit_s:.1f}s")
    return meta, pd.concat(all_recs, ignore_index=True)
