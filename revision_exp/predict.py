"""Prediction with stored XGBoost artifacts (new or existing). No training here."""
import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from data import REPO, note_input

OLD_PREFIX = REPO / "xgboost_model" / "artifacts" / "gpu" / "xgb_model_x3_double"


def load_model(path):
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m


def predict_multi(model_t, model_d, X):
    """Batch prediction of T_hat and D_hat for all rows of X."""
    return np.asarray(model_t.predict(X.values), dtype=float), np.asarray(model_d.predict(X.values), dtype=float)


def multi_score(t_hat, d_hat, alpha, clip=True):
    """S_hat = T_hat - alpha * D_hat, clipped to [0, 1] as the existing predict path does (default --clip_pred_score)."""
    raw = t_hat - alpha * d_hat
    return (np.clip(raw, 0.0, 1.0) if clip else raw), raw


def single_score(model_s, X, clip=True):
    raw = np.asarray(model_s.predict(X.values), dtype=float)
    return (np.clip(raw, 0.0, 1.0) if clip else raw), raw


def load_old_artifact():
    """Existing artifact, read-only. Returns (model_T, model_D, feature list, meta)."""
    meta = json.loads(Path(note_input(str(OLD_PREFIX) + "_meta.json")).read_text())
    mt = load_model(note_input(str(OLD_PREFIX) + "_y1.json"))
    md = load_model(note_input(str(OLD_PREFIX) + "_y2.json"))
    return mt, md, meta["features"], meta
