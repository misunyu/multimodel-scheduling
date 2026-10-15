"""Family-grouped folds for hyper-parameter selection (plain sklearn GroupKFold, no manual re-assignment)."""
import sys

sys.dont_write_bytecode = True

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_SPLITS = 3


def make_folds(train_df):
    """Return an int array (len = rows) with the validation fold of every row; groups = family key."""
    fold = np.full(len(train_df), -1, dtype=int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for k, (_tr, va) in enumerate(gkf.split(np.zeros(len(train_df)), groups=train_df["family"].values)):
        fold[va] = k
    assert (fold >= 0).all()
    per_family = pd.Series(fold).groupby(train_df["family"].values).nunique()
    assert (per_family == 1).all(), "a family is split across folds"
    return fold


def describe(train_df, fold):
    d = train_df.assign(fold=fold)
    fam = (d.groupby(["family", "fold"]).agg(rows=("row_uid", "size"), N=("N", "first"), schedules=("group", "nunique"))
           .reset_index().sort_values(["fold", "N", "family"]))
    per_fold = []
    for k, sub in d.groupby("fold"):
        per_fold.append({"scheme": "GroupKFold(3) by family (this work)", "fold": int(k), "rows": len(sub),
                         "families": sub["family"].nunique(), "schedule_files": sub["group"].nunique(),
                         "N_distribution(rows)": json.dumps({int(n): int(c) for n, c in sub.groupby("N").size().items()}),
                         "N_distribution(families)": json.dumps({int(n): int(c) for n, c in sub.groupby("N")["family"].nunique().items()})})
    return fam, pd.DataFrame(per_fold)
