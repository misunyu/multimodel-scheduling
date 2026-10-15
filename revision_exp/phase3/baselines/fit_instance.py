"""Per-instance baseline: two shared XGBoost regressors (t_ratio, q_i). Same grid / selection rule as the direct predictor
(train_multi.GRID, mean unweighted validation MAE over 3 folds). Folds: the Phase 2 family -> fold mapping, joined on the family key
(no new GroupKFold on instance rows). q_i model: weight 10 where q_i > 0.01 (instance unit; the direct D model weights placement rows
with D_macro > 0.01 - same numbers, different unit of the loss weight)."""
import sys

sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd

import train_multi as TM

from phase3.baselines.instance_rows import FEATURES

Q_WEIGHT_THRESHOLD = TM.D_WEIGHT_THRESHOLD
Q_WEIGHT_VALUE = TM.D_WEIGHT_VALUE


def instance_folds(rows, family_fold):
    """family_fold: {family: fold} from the direct pipeline."""
    f = rows["family"].map(family_fold)
    assert f.notna().all(), "instance row with a family that has no direct fold"
    f = f.astype(int).values
    per_family = pd.Series(f).groupby(rows["family"].values).nunique()
    assert (per_family == 1).all()
    assert all(int(pd.Series(f)[rows["family"].values == fam].iloc[0]) == k for fam, k in family_fold.items() if (rows["family"] == fam).any())
    per_run = pd.Series(f).groupby([rows["group"].values, rows["comb_num"].values]).nunique()
    assert (per_run == 1).all(), "instances of one run are split across folds"
    return f


def fit_instance_models(rows, fold, art_dir, log=print):
    X = rows[FEATURES].reset_index(drop=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    w = np.where(rows["q_i"].values > Q_WEIGHT_THRESHOLD, Q_WEIGHT_VALUE, 1.0)
    models, recs, info = {}, [], {}
    for tgt, wt in (("t_ratio", None), ("q_i", w)):
        y = rows[tgt].values
        t0 = time.perf_counter()
        best, r = TM.cv_search(X, y, fold, TM.GRID, weight=wt, label=f"per-instance:{tgt}")
        m = TM.fit_one(X.values, y, best, wt)
        m.save_model(str(art_dir / f"{tgt}.json"))
        models[tgt] = m
        sel = r[r["selected"]].iloc[0]
        recs.append(r.assign(method="Per-instance predictor ([14]-inspired)", target=tgt))
        info[tgt] = {"best_params": str(best), "cv_mae_mean": float(sel["mae_mean"]), "cv_mae_folds": [float(sel[f"mae_fold{k}"]) for k in range(3)],
                     "seconds": round(time.perf_counter() - t0, 1), "sample_weight": wt is not None,
                     "rows_weighted": int((wt > 1).sum()) if wt is not None else 0, "rows": len(y)}
        log(f"[per-instance] {tgt}: best={best} cv_mae={sel['mae_mean']:.4f} ({info[tgt]['seconds']}s)")
    return models, pd.concat(recs, ignore_index=True), info
