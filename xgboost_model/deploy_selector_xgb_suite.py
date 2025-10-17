#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_selector_xgb_suite.py

Two-target XGBoost training/inference for multi-view performance logs.

Targets (window-level):
  - y1 = total.total_throughput_fps
  - y2 = derived.drop_rate_fps

Inputs (per window):
  - Per-view dynamic metrics exactly as specified:
      throughput_fps, avg_inference_time_ms, inference_count,
      avg_wait_to_preprocess_ms, dropped_frames_due_to_full_queue
  - Execution device one-hot: exec_cpu, exec_npu0, exec_npu1
  - Static features selected by the *used* device, looked up by model name from sample_profiling_data.json:
      static_infer_sel, static_load_sel
  - Two cross terms:
      throughput_fps * static_infer_sel
      avg_wait_to_preprocess_ms * static_load_sel
  - Then aggregated across views with sum/mean/max and views.count.views

NO leakage: window-level totals/derived fields are NOT used as features.

CLI
---
Train:
  python deploy_selector_xgb_suite.py train \
    --perf_dir ./xgboost_model/performance_data \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_out ./xgboost_model/artifacts/deploy_xgb \
    --dump_csv ./xgboost_model/artifacts/train_dataset_two_targets.csv

Predict:
  python deploy_selector_xgb_suite.py predict \
    --perf_file ./xgboost_model/performance_data/performance_20250913_134335.json \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_in ./xgboost_model/artifacts/deploy_xgb \
    --dump_csv ./xgboost_model/artifacts/features_for_this_file.csv
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- Utilities ----------

def _lazy_import_xgb():
    """Import xgboost only when needed to keep this script lightweight."""
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception as e:
        raise RuntimeError(
            "xgboost is required. Install it with: pip install xgboost\n"
            f"Original import error: {e}"
        )


@dataclass
class StaticProfile:
    """Static profiling values for a model."""
    cpu_infer: float
    npu0_load: float
    npu0_infer: float
    npu1_load: float
    npu1_infer: float


def load_static_profiles(static_json_path: Path) -> Dict[str, StaticProfile]:
    """Load 'total_data' from sample_profiling_data.json into a lookup table."""
    blob = json.loads(static_json_path.read_text(encoding="utf-8"))
    table: Dict[str, StaticProfile] = {}
    for row in blob.get("total_data", []):
        table[row["model"]] = StaticProfile(
            cpu_infer=float(row.get("cpu_infer", np.nan)),
            npu0_load=float(row.get("npu0_load", np.nan)),
            npu0_infer=float(row.get("npu0_infer", np.nan)),
            npu1_load=float(row.get("npu1_load", np.nan)),
            npu1_infer=float(row.get("npu1_infer", np.nan)),
        )
    return table


# Per-view dynamic keys (exactly as requested)
VIEW_DYNAMIC_KEYS = [
    "throughput_fps",
    "avg_inference_time_ms",
    "inference_count",
    "avg_wait_to_preprocess_ms",
    "dropped_frames_due_to_full_queue",
]


def _device_static_for(model: str, exec_dev: str, S: Dict[str, StaticProfile]) -> Tuple[float, float]:
    """Pick static (infer, load) based on the device actually used by the view."""
    prof = S.get(model)
    if prof is None:
        return (np.nan, np.nan)
    d = str(exec_dev).upper()
    if d == "CPU":
        return (prof.cpu_infer, 0.0)
    if d == "NPU0":
        return (prof.npu0_infer, prof.npu0_load)
    if d == "NPU1":
        return (prof.npu1_infer, prof.npu1_load)
    return (np.nan, np.nan)


def _nested(d: Dict[str, Any], dotted: str, field: str, default=np.nan) -> float:
    """Get d[k1][k2]...[field] where dotted='k1.k2'."""
    cur = d
    for part in dotted.split("."):
        cur = cur.get(part, {})
    val = cur.get(field, default) if isinstance(cur, dict) else default
    try:
        return float(val)
    except Exception:
        return default


# ---------- Feature engineering ----------

def featurize_window(window: Dict[str, Any], S: Dict[str, StaticProfile]) -> Tuple[Dict[str, float], Tuple[float, float], Dict[str, Any]]:
    """
    Convert a window into (X_features, (y1, y2), meta).

    Features come only from per-view dynamics + exec flags + selected static + cross terms,
    then aggregated across views.
    """
    models = window.get("models", {})
    per_view_rows: List[Dict[str, float]] = []

    for _, view in models.items():
        model_name = view.get("model")
        exec_dev = view.get("execution")
        if not model_name or not exec_dev:
            continue

        row: Dict[str, float] = {}
        # dynamic inputs
        for k in VIEW_DYNAMIC_KEYS:
            row[f"view.{k}"] = float(view.get(k, np.nan))

        # device flags
        dev = str(exec_dev).upper()
        row["view.exec_cpu"] = 1.0 if dev == "CPU" else 0.0
        row["view.exec_npu0"] = 1.0 if dev == "NPU0" else 0.0
        row["view.exec_npu1"] = 1.0 if dev == "NPU1" else 0.0

        # static inputs selected by the device
        s_infer, s_load = _device_static_for(model_name, exec_dev, S)
        row["view.static_infer_sel"] = s_infer
        row["view.static_load_sel"] = s_load

        # cross terms
        row["x.view_throughput__static_infer_sel"] = row["view.throughput_fps"] * row["view.static_infer_sel"]
        row["x.view_wait__static_load_sel"] = row["view.avg_wait_to_preprocess_ms"] * row["view.static_load_sel"]

        per_view_rows.append(row)

    # aggregate per-view to fixed size per-window
    X: Dict[str, float] = {}
    df = pd.DataFrame(per_view_rows)
    if not df.empty:
        for agg_name, s in {
            "sum": df.sum(numeric_only=True),
            "mean": df.mean(numeric_only=True),
            "max": df.max(numeric_only=True),
        }.items():
            for col, val in s.items():
                X[f"views.{agg_name}.{col}"] = float(val)
        X["views.count.views"] = float(len(df))
    else:
        X["views.count.views"] = 0.0

    # targets
    y1 = _nested(window, "total", "total_throughput_fps")  # target 1
    y2 = _nested(window, "derived", "drop_rate_fps")       # target 2

    meta = {
        "timestamp": window.get("timestamp"),
        "combination": window.get("combination"),
    }
    return X, (y1, y2), meta


def build_dataset_from_file(perf_json_path: Path, S: Dict[str, StaticProfile]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (X, Y, M) from a single performance JSON file."""
    blob = json.loads(perf_json_path.read_text(encoding="utf-8"))
    X_rows: List[Dict[str, float]] = []
    Y_rows: List[Dict[str, float]] = []
    M_rows: List[Dict[str, Any]] = []

    for w in blob.get("data", []):
        X, (y1, y2), meta = featurize_window(w, S)
        if math.isnan(y1) or math.isnan(y2):
            continue
        X_rows.append(X)
        Y_rows.append({"y1_total_throughput_fps": y1, "y2_drop_rate_fps": y2})
        M_rows.append(meta)

    X_df = pd.DataFrame(X_rows).fillna(0.0)
    Y_df = pd.DataFrame(Y_rows)
    M_df = pd.DataFrame(M_rows)
    return X_df, Y_df, M_df


def build_dataset(perf_dir: Path, static_json_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan perf_dir recursively and concatenate datasets."""
    S = load_static_profiles(static_json_path)
    X_all: List[pd.DataFrame] = []
    Y_all: List[pd.DataFrame] = []
    M_all: List[pd.DataFrame] = []

    json_paths = sorted([p for p in perf_dir.rglob("*.json") if p.is_file()])
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {perf_dir}")

    for path in json_paths:
        try:
            X, Y, M = build_dataset_from_file(path, S)
            if not X.empty:
                X_all.append(X)
                Y_all.append(Y)
                M["source_file"] = str(path)
                M_all.append(M)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)

    if not X_all:
        raise RuntimeError("No valid training rows were built. Check your input logs.")

    X_full = pd.concat(X_all, ignore_index=True).fillna(0.0)
    Y_full = pd.concat(Y_all, ignore_index=True)
    M_full = pd.concat(M_all, ignore_index=True)
    return X_full, Y_full, M_full


# ---------- Train / Predict ----------

def train_two_targets(X: pd.DataFrame, Y: pd.DataFrame, model_out_prefix: Path) -> None:
    """Train two XGBoost models (y1, y2) and save them as <prefix>_y1.json / <prefix>_y2.json."""
    xgb = _lazy_import_xgb()
    feat_names = list(X.columns)

    # y1 model
    dtrain_y1 = xgb.DMatrix(X.values, label=Y["y1_total_throughput_fps"].values, feature_names=feat_names)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3.0,
        "seed": 42,
    }
    bst1 = xgb.train(params, dtrain_y1, num_boost_round=400)
    p1 = str(model_out_prefix) + "_y1.json"
    Path(p1).parent.mkdir(parents=True, exist_ok=True)
    bst1.save_model(p1)

    # y2 model
    dtrain_y2 = xgb.DMatrix(X.values, label=Y["y2_drop_rate_fps"].values, feature_names=feat_names)
    bst2 = xgb.train(params, dtrain_y2, num_boost_round=400)
    p2 = str(model_out_prefix) + "_y2.json"
    bst2.save_model(p2)


def predict_two_targets(model_in_prefix: Path, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load two models and predict (y1, y2)."""
    xgb = _lazy_import_xgb()
    feat_names = list(X.columns)
    dmat = xgb.DMatrix(X.values, feature_names=feat_names)

    m1 = str(model_in_prefix) + "_y1.json"
    m2 = str(model_in_prefix) + "_y2.json"
    bst1 = xgb.Booster(model_file=m1)
    bst2 = xgb.Booster(model_file=m2)

    y1_pred = bst1.predict(dmat)
    y2_pred = bst2.predict(dmat)
    return y1_pred, y2_pred


# ---------- CLI ----------

def _normalize_model_prefix(path_str: str, default_name: str = "xgb2_model") -> Path:
    """
    Normalize --model_out / --model_in to be a 'prefix' path (not a directory).
    If the user gives a directory or a path ending with '/', append a default base name.
    """
    p = Path(path_str)
    # If ends with slash or is an existing directory, append default name
    if path_str.endswith("/") or (p.exists() and p.is_dir()):
        p = p / default_name
    return p


def main():
    ap = argparse.ArgumentParser(description="Two-target XGBoost trainer/inferencer for multi-view performance logs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="Train two-target model from a folder of performance JSON logs.")
    ap_tr.add_argument("--perf_dir", type=str, required=True, help="Directory containing performance JSON logs.")
    ap_tr.add_argument("--static_json", type=str, required=True, help="Path to sample_profiling_data.json.")
    ap_tr.add_argument("--model_out", type=str, required=True, help="Prefix path for saving models (without _y*.json).")
    ap_tr.add_argument("--dump_csv", type=str, default="", help="Optional: path to dump engineered dataset CSV.")

    ap_pr = sub.add_parser("predict", help="Predict two targets for a single performance JSON file.")
    ap_pr.add_argument("--perf_file", type=str, required=True, help="Performance JSON file to featurize.")
    ap_pr.add_argument("--static_json", type=str, required=True, help="Path to sample_profiling_data.json.")
    ap_pr.add_argument("--model_in", type=str, required=True, help="Model prefix path (expects _y1.json and _y2.json).")
    ap_pr.add_argument("--dump_csv", type=str, default="", help="Optional: path to dump engineered features CSV.")

    args = ap.parse_args()

    if args.cmd == "train":
        perf_dir = Path(args.perf_dir)
        static_json = Path(args.static_json)
        model_prefix = _normalize_model_prefix(args.model_out)

        X, Y, M = build_dataset(perf_dir, static_json)

        if args.dump_csv:
            df_dump = pd.concat([M.reset_index(drop=True), X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
            Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
            df_dump.to_csv(args.dump_csv, index=False)
            print(f"[INFO] wrote dataset -> {args.dump_csv}  rows={len(df_dump)}")

        train_two_targets(X, Y, model_prefix)
        print(f"[OK] saved -> {model_prefix}_y1.json, {model_prefix}_y2.json")

    elif args.cmd == "predict":
        perf_file = Path(args.perf_file)
        static_json = Path(args.static_json)
        model_prefix = _normalize_model_prefix(args.model_in)

        S = load_static_profiles(static_json)
        X, _, M = build_dataset_from_file(perf_file, S)
        if X.empty:
            raise RuntimeError("No feature rows built from perf_file.")

        if args.dump_csv:
            df_dump = pd.concat([M.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
            Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
            df_dump.to_csv(args.dump_csv, index=False)
            print(f"[INFO] wrote features -> {args.dump_csv}  rows={len(df_dump)}")

        y1_pred, y2_pred = predict_two_targets(model_prefix, X)
        for i, (ts, comb) in enumerate(zip(M.get("timestamp", []), M.get("combination", []))):
            print(
                f"{ts}\t{comb}\t"
                f"pred_total_throughput_fps={float(y1_pred[i]):.4f}\t"
                f"pred_drop_rate_fps={float(y2_pred[i]):.4f}"
            )

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
