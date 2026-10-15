"""Per-instance baseline: composition and selection from the baseline's OWN predictions.

  t_hat_i = clip(pred_t, 0) * lam_i            F_hat(x) = sum_i t_hat_i
  q_hat_i = clip(pred_q, 0, 1)                 D_hat_macro(x) = mean_i q_hat_i
  T_hat_select = F_hat / max_x F_hat (predicted maximum over all feasible candidates of the workload; 0 if that maximum is 0)
  S_hat = T_hat_select - alpha * D_hat_macro;  argmax, ties -> lower combination number (evaluate.select)
Measured F_max / D never enter compose() or select_candidates(): their inputs are restricted to schedule-derived columns.
"""
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import evaluate as E

from phase3.baselines.instance_rows import FEATURES

ALLOWED_INPUT = set(FEATURES) | {"group", "family", "rate", "combination", "comb_num", "model", "device"}


def compose(rows, pred_t, pred_q):
    assert set(rows.columns) <= ALLOWED_INPUT, f"compose() received measured columns: {sorted(set(rows.columns) - ALLOWED_INPUT)}"
    r = rows[["group", "comb_num", "lam"]].copy()
    r["t_hat"] = np.clip(np.asarray(pred_t, dtype=float), 0, None) * r["lam"].values
    r["q_hat"] = np.clip(np.asarray(pred_q, dtype=float), 0, 1)
    c = r.groupby(["group", "comb_num"], sort=True).agg(F_hat=("t_hat", "sum"), D_hat=("q_hat", "mean"), n_inst=("q_hat", "size")).reset_index()
    fmax = c.groupby("group")["F_hat"].transform("max")
    c["T_hat_select"] = np.where(fmax > 0, c["F_hat"] / fmax.where(fmax > 0, 1.0), 0.0)
    return c


def select_candidates(c, alpha):
    assert set(c.columns) == {"group", "comb_num", "F_hat", "D_hat", "n_inst", "T_hat_select"}, "select_candidates() must see predictions only"
    s_hat = (c["T_hat_select"] - alpha * c["D_hat"]).values
    order, ties = E.select(c[["group", "comb_num"]], s_hat)
    return s_hat, order, ties
