#!/usr/bin/env python3
"""Accuracy validation for the 3-target deployment XGBoost model.

Two modes:

  cv       K-fold cross-validation over the training windows. Estimates how well
           the model predicts UNSEEN combinations (honest generalization).

  holdout  Evaluate an already-trained model on an independent test set
           (fresh contention runs the model never saw).

Metrics per target (y1 vision FPS, y2 drop rate, y3 tokens/sec): MAE, RMSE, R2,
and MAPE over non-zero actuals.

Usage:
  $PYTHON_BIN evaluate_model.py cv \
      --perf_dir xgboost_model/performance_data/train \
      --schedule_dir xgboost_model/schedules \
      --static_json .../sample_profiling_data.json [--folds 5]

  $PYTHON_BIN evaluate_model.py holdout \
      --model_in xgboost_model/artifacts/deploy_xgb \
      --test_perf xgboost_model/performance_data/test \
      --schedule_dir xgboost_model/schedules \
      --static_json .../sample_profiling_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import xgboost_model.deploy_selector_xgb_suite as S


TARGET_LABELS = {
    "y1_total_throughput_fps": "y1 norm throughput",
    "y2_deadline_miss_rate": "y2 deadline miss",
    "y3_total_tokens_per_s": "y3 norm tokens",
}


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
    nz = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100.0) if nz.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape, "mean_actual": float(actual.mean())}


def _print_report(title, per_target):
    print(f"\n=== {title} ===")
    print(f"{'target':<16}{'MAE':>10}{'RMSE':>10}{'R2':>8}{'MAPE%':>9}{'mean':>10}")
    for col, m in per_target.items():
        print(f"{TARGET_LABELS.get(col, col):<16}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}"
              f"{m['R2']:>8.3f}{m['MAPE%']:>9.1f}{m['mean_actual']:>10.3f}")


def cmd_cv(args):
    xgb = S._lazy_import_xgb()
    X, Y, M = S.build_dataset(Path(args.perf_dir), Path(args.static_json), Path(args.schedule_dir),
                              keep_rate_factors=args.rate_factors, normalize=True)
    n = len(X)
    print(f"[cv] {n} windows, {X.shape[1]} features, {args.folds}-fold")
    feat = list(X.columns)
    # deterministic fold assignment (no RNG needed)
    fold_id = np.arange(n) % args.folds
    cols = [c for _, c in S._TARGETS]
    preds = {c: np.zeros(n) for c in cols}
    for k in range(args.folds):
        tr = fold_id != k
        te = fold_id == k
        for _, col in S._TARGETS:
            dtr = xgb.DMatrix(X.values[tr], label=Y[col].values[tr], feature_names=feat)
            bst = xgb.train(S._PARAMS, dtr, num_boost_round=400)
            dte = xgb.DMatrix(X.values[te], feature_names=feat)
            preds[col][te] = bst.predict(dte)
    per_target = {col: _metrics(Y[col].values, preds[col]) for col in cols}
    _print_report(f"{args.folds}-fold cross-validation (unseen combinations)", per_target)
    return per_target


def cmd_holdout(args):
    Xte, Yte, Mte = S.build_dataset(Path(args.test_perf), Path(args.static_json), Path(args.schedule_dir),
                                    keep_rate_factors=args.rate_factors, normalize=True)
    print(f"[holdout] test windows: {len(Xte)} (rate factors {args.rate_factors})")
    y1, y2, y3 = S.predict_targets(Path(args.model_in), Xte)
    pred = {"y1_total_throughput_fps": y1, "y2_deadline_miss_rate": y2, "y3_total_tokens_per_s": y3}
    per_target = {col: _metrics(Yte[col].values, pred[col]) for col in pred}
    _print_report("Rate hold-out test set (production model)", per_target)
    return per_target


def cmd_select(args):
    """Paper-style placement selection: per-workload Top-1/Top-5 hit rate + oracle gap.

    A "workload" is a (active-model-set, input-rate) group. Within each group the
    measured (normalized) score defines the oracle ranking; the predictor ranks the
    same placements. Reported: Top-1/Top-5 hit rate and average score of the
    predicted pick vs the oracle (both S = T + beta*T_tok - alpha*miss).
    """
    Sprof = S.load_static_profiles(Path(args.static_json))
    # Normalized measured targets per test window (rate hold-out by default).
    Xte, Yte, Mte = S.build_dataset(Path(args.test_perf), Path(args.static_json),
                                    Path(args.schedule_dir),
                                    keep_rate_factors=args.rate_factors, normalize=True)
    sched = S._load_yaml_or_json(Path(args.schedule_yaml))
    blob_by_name = {name: blob for name, blob in S._iter_combos_from_schedule(sched)}

    # Build per-combo measured/predicted scores grouped by (workload, rate).
    groups = {}
    for i in range(len(Mte)):
        name = Mte.iloc[i]["combination"]
        g = (Mte.iloc[i].get("workload"), Mte.iloc[i].get("rate_factor"))
        meas = (float(Yte.iloc[i]["y1_total_throughput_fps"])
                + args.beta * float(Yte.iloc[i]["y3_total_tokens_per_s"])
                - args.alpha * float(Yte.iloc[i]["y2_deadline_miss_rate"]))
        blob = blob_by_name.get(name)
        if blob is None:
            continue
        y1, y2, y3 = S.predict_targets(Path(args.model_in), S.featurize_from_combo(Sprof, blob))
        pred = float(y1[0]) + args.beta * float(y3[0]) - args.alpha * float(y2[0])
        groups.setdefault(g, []).append((name, meas, pred))

    top1_hits, top5_hits, score_ratios, spearmans = [], [], [], []
    for g, rows in groups.items():
        if len(rows) < 2:
            continue
        names = [r[0] for r in rows]
        meas = np.array([r[1] for r in rows]); pred = np.array([r[2] for r in rows])
        best_meas = meas.max()
        oracle_set = {names[i] for i in range(len(names)) if abs(meas[i] - best_meas) < 1e-9}
        prank = [names[i] for i in np.argsort(-pred)]
        pred_best = prank[0]
        top1_hits.append(1.0 if pred_best in oracle_set else 0.0)
        top5_hits.append(1.0 if oracle_set & set(prank[:5]) else 0.0)
        ms = dict(zip(names, meas))
        score_ratios.append(ms[pred_best] / best_meas if best_meas > 1e-9 else 1.0)
        if len(names) > 2:
            rx = np.argsort(np.argsort(-pred)); ry = np.argsort(np.argsort(-meas))
            c = np.corrcoef(rx, ry)[0, 1]
            if np.isfinite(c):
                spearmans.append(float(c))

    print("\n=== Deployment selection accuracy (per-workload, rate hold-out) ===")
    print(f"workload groups evaluated: {len(top1_hits)}  (rate factors {args.rate_factors})")
    if top1_hits:
        print(f"Top-1 hit rate: {np.mean(top1_hits):.3f}")
        print(f"Top-5 hit rate: {np.mean(top5_hits):.3f}")
        print(f"avg predicted-pick score / oracle score: {np.mean(score_ratios):.3f}")
        print(f"avg Spearman rank corr (within group): {np.mean(spearmans):.3f}" if spearmans else "")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    _tr = [1.0, 2.0, 4.0]
    a = sub.add_parser("cv")
    a.add_argument("--perf_dir", required=True)
    a.add_argument("--schedule_dir", required=True)
    a.add_argument("--static_json", required=True)
    a.add_argument("--folds", type=int, default=3)
    a.add_argument("--rate_factors", nargs="*", type=float, default=_tr)

    b = sub.add_parser("holdout")
    b.add_argument("--model_in", required=True)
    b.add_argument("--test_perf", required=True)
    b.add_argument("--schedule_dir", required=True)
    b.add_argument("--static_json", required=True)
    b.add_argument("--rate_factors", nargs="*", type=float, default=[3.0])

    c = sub.add_parser("select")
    c.add_argument("--model_in", required=True)
    c.add_argument("--test_perf", required=True, help="dir with performance_*.json (rate hold-out windows)")
    c.add_argument("--schedule_dir", required=True)
    c.add_argument("--schedule_yaml", required=True)
    c.add_argument("--static_json", required=True)
    c.add_argument("--rate_factors", nargs="*", type=float, default=[3.0])
    c.add_argument("--alpha", type=float, default=0.3)
    c.add_argument("--beta", type=float, default=0.5)

    args = ap.parse_args()
    if args.cmd == "cv":
        cmd_cv(args)
    elif args.cmd == "holdout":
        cmd_holdout(args)
    else:
        cmd_select(args)


if __name__ == "__main__":
    main()
