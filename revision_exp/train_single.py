"""Single-Objective predictor: one XGBoost regressor per alpha on S = T - alpha * D."""
import sys

sys.dont_write_bytecode = True

import time

import pandas as pd

from train_multi import BASE_PARAMS, GRID, cv_search, fit_one, resolved_config

ALPHAS = [round(0.1 * i, 1) for i in range(1, 11)]
# Existing path, xgboost_model/deploy_selector_xgb_suite.py:571-612 (train_score):
#   target  y_score = Y["y1"] - alpha * Y["y2"]                      (:580)
#   grid    n_estimators, learning_rate, max_depth, subsample as in train_double, plus colsample_bytree [0.8]  (:583-589)
#   fit     grid_search.fit(X, y_score)  -> NO sample weight           (:607)
FIXED = {"colsample_bytree": 0.8}


def train_single(X, train_df, fold, art_dir, log=print):
    meta = {"mode": "single", "features": list(X.columns), "grid": GRID, "fixed": FIXED, "base_params": BASE_PARAMS,
            "sample_weight": False,
            "selection": "mean over 3 family-grouped folds of validation MAE; ties -> first grid candidate", "models": {}}
    recs_all = []
    for a in ALPHAS:
        y = (train_df["T"] - a * train_df["D"]).values
        t0 = time.perf_counter()
        best, recs = cv_search(X, y, fold, GRID, fixed=FIXED, label=f"S_alpha_{a}")
        cv_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = fit_one(X.values, y, best, None, FIXED)
        refit_s = time.perf_counter() - t1
        model.save_model(str(art_dir / f"alpha_{a}.json"))
        sel = recs[recs["selected"]].iloc[0]
        meta["models"][str(a)] = {"alpha": a, "best_params": {**best, **FIXED}, "cv_mae_mean": float(sel["mae_mean"]),
                                  "cv_mae_folds": [float(sel[f"mae_fold{k}"]) for k in range(3)],
                                  "cv_seconds": cv_s, "refit_seconds": refit_s, "resolved_config": resolved_config(model),
                                  "n_trees": int(model.get_booster().num_boosted_rounds())}
        recs.insert(0, "alpha", a)
        recs_all.append(recs)
        log(f"[single] alpha={a}: best={best} cv_mae={sel['mae_mean']:.4f} cv={cv_s:.1f}s refit={refit_s:.1f}s")
    return meta, pd.concat(recs_all, ignore_index=True)
