#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_selector_xgb_suite.py

Two-target XGBoost training/inference for multi-view performance logs.

Targets (window-level):
  - y1 = total.total_throughput_fps
  - y2 = derived.drop_rate_fps

Inputs (per window):
  - Per-view dynamic metrics (from JSON logs, train only):
      throughput_fps, avg_inference_time_ms, inference_count,
      avg_wait_to_preprocess_ms, dropped_frames_due_to_full_queue
  - Execution device one-hot: exec_cpu, exec_npu0, exec_npu1
  - Static features selected by the used device from sample_profiling_data.json:
      static_infer_sel, static_load_sel
  - Per-view planned FPS from YAML schedule (train & predict):
      view.infps
  - Cross terms:
      throughput_fps * static_infer_sel
      avg_wait_to_preprocess_ms * static_load_sel
  - Aggregation across views: sum/mean/max + views.count.views

NO leakage: window-level totals/derived fields are NOT used as features.

CLI
---
Train (JSON + YAML folders):
  python deploy_selector_xgb_suite.py train \
    --perf_dir ./xgboost_model/performance_data \
    --schedule_dir ./xgboost_model/schedules \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_out ./xgboost_model/artifacts/deploy_xgb \
    [--dump_csv ./xgboost_model/artifacts/train_dataset_two_targets.csv]

Predict from YAML schedule (planned combinations):
  python deploy_selector_xgb_suite.py predict \
    --schedule_yaml ./xgboost_model/schedules/model_schedules.yaml \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_in ./xgboost_model/artifacts/deploy_xgb \
    [--alpha 0.2] [--topk 5]
"""


#python3 xgboost_model/deploy_selector_xgb_suite.py predict   --schedule_dir ./xgboost_model/schedules/test   --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json   --model_in ./xgboost_model/artifacts/deploy_xgb   --alpha 0.2 --topk 1   --repeats 10

import argparse
import json
import math
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ---------- Constants for combo featurization (predict 전용, 학습 입력엔 영향 없음) ----------
WINDOW_SEC = 30.0
ASSUME_WAIT_MS = 0.0


# ---------- Utilities ----------

def _lazy_import_xgb():
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
    cpu_infer: float
    gpu_infer: float
    npu0_load: float
    npu0_infer: float
    npu1_load: float
    npu1_infer: float


def load_static_profiles(static_json_path: Path) -> Dict[str, StaticProfile]:
    blob = json.loads(static_json_path.read_text(encoding="utf-8"))
    table: Dict[str, StaticProfile] = {}
    for row in blob.get("total_data", []):
        table[row["model"]] = StaticProfile(
            cpu_infer=float(row.get("cpu_infer", np.nan)),
            gpu_infer=float(row.get("gpu_infer", np.nan)),
            npu0_load=float(row.get("npu0_load", np.nan)),
            npu0_infer=float(row.get("npu0_infer", np.nan)),
            npu1_load=float(row.get("npu1_load", np.nan)),
            npu1_infer=float(row.get("npu1_infer", np.nan)),
        )
    return table


VIEW_DYNAMIC_KEYS = [
    "throughput_fps",
    "avg_inference_time_ms",
    "inference_count",
    "avg_wait_to_preprocess_ms",
    "dropped_frames_due_to_full_queue",
]


def _device_static_for(model: str, exec_dev: str, S: Dict[str, StaticProfile]) -> Tuple[float, float]:
    prof = S.get(model)
    if prof is None:
        return (np.nan, np.nan)
    d = str(exec_dev).upper()
    if d == "CPU":
        return (prof.cpu_infer, 0.0)
    if d == "GPU":
        # GPU has no separate load time in static profiles; treat load as 0 for prediction features
        try:
            return (prof.gpu_infer, 0.0)
        except Exception:
            # Backward compatibility: if gpu_infer missing, fall back to NPU1 infer
            return (getattr(prof, 'npu1_infer', np.nan), 0.0)
    if d == "NPU0":
        return (prof.npu0_infer, prof.npu0_load)
    if d == "NPU1":
        return (prof.npu1_infer, prof.npu1_load)
    return (np.nan, np.nan)


def _nested(d: Dict[str, Any], dotted: str, field: str, default=np.nan) -> float:
    cur = d
    for part in dotted.split("."):
        cur = cur.get(part, {})
    val = cur.get(field, default) if isinstance(cur, dict) else default
    try:
        return float(val)
    except Exception:
        return default


def _norm_exec(dev: str) -> str:
    d = str(dev).strip().lower()
    if d in ("cpu",):
        return "CPU"
    if d in ("gpu", "apple-gpu", "coreml-gpu"):
        return "GPU"
    if d in ("npu0", "npu-0", "npu_0", "npu 0"):
        return "NPU0"
    if d in ("npu1", "npu-1", "npu_1", "npu 1"):
        return "NPU1"
    return dev.upper()


# ---------- YAML 로더/인덱서 (train에서 사용) ----------

def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            data = yaml.safe_load(txt)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _iter_combos_from_schedule(schedule: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    if "combinations" in schedule:
        combos = schedule["combinations"]
        if isinstance(combos, list):
            for c in combos:
                if isinstance(c, dict):
                    name = c.get("combination") or c.get("name") or "combination_unknown"
                    out.append((str(name), c))
        elif isinstance(combos, dict):
            for name, c in combos.items():
                if isinstance(c, dict):
                    c = dict(c); c.setdefault("combination", name)
                    out.append((str(name), c))
        if out:
            return out
    if isinstance(schedule, dict):
        picks = [(k, v) for k, v in schedule.items()
                 if isinstance(v, dict) and str(k).lower().startswith("combination")]
        if picks:
            for name, blob in picks:
                out.append((str(name), blob))
            return out
    name = schedule.get("combination") or schedule.get("name") or "combination_unknown"
    out.append((str(name), schedule))
    return out


def _rows_from_combo_struct(combo_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "views" in combo_blob and isinstance(combo_blob["views"], list):
        return [dict(v) for v in combo_blob["views"]]
    if "models" in combo_blob and isinstance(combo_blob["models"], dict):
        return [dict(v) for _, v in sorted(combo_blob["models"].items())]
    if isinstance(combo_blob, dict):
        vals = list(combo_blob.values())
        if vals and all(isinstance(v, dict) for v in vals):
            rows = []
            for v in vals:
                if "model" in v and "execution" in v:
                    row = {"model": v["model"], "execution": v["execution"]}
                    for k in v:
                        if k not in ("model", "execution"):
                            row[k] = v[k]
                    rows.append(row)
            if rows:
                return rows
    raise ValueError("combo must have 'views' (list) or 'models' (dict), or an implicit dict of per-view entries.")


def _index_schedules(schedule_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not schedule_dir.exists():
        return index
    exts = {".yaml", ".yml", ".json"}
    for p in schedule_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                index[p.name.lower()] = _load_yaml_or_json(p)
            except Exception:
                pass
    return index


def _find_schedule(index: Dict[str, Dict[str, Any]], hint: str) -> Optional[Dict[str, Any]]:
    if not hint:
        return None
    base = Path(hint).name.lower()
    return index.get(base)


def _build_infps_lookup(schedule_doc: Dict[str, Any], combination_name: str) -> Dict[Tuple[str, str], float]:
    infps_map: Dict[Tuple[str, str], float] = {}
    try:
        combos = _iter_combos_from_schedule(schedule_doc)
        target_blob = None
        for name, blob in combos:
            if str(name) == str(combination_name):
                target_blob = blob
                break
        if target_blob is None and combos:
            if len(combos) == 1:
                target_blob = combos[0][1]
        if target_blob is None:
            return infps_map
        rows = _rows_from_combo_struct(target_blob)
        for r in rows:
            m = r.get("model")
            dev = _norm_exec(r.get("execution", ""))
            if not m or not dev:
                continue
            if "infps" in r and r["infps"] is not None:
                try:
                    infps_map[(m, dev)] = float(r["infps"])
                except Exception:
                    pass
    except Exception:
        pass
    return infps_map


# ---------- Feature engineering (train 전용) ----------

def featurize_window(window: Dict[str, Any],
                     S: Dict[str, StaticProfile],
                     infps_map: Optional[Dict[Tuple[str, str], float]] = None
                     ) -> Tuple[Dict[str, float], Tuple[float, float], Dict[str, Any]]:
    """
    Train용: JSON의 동적 지표 + YAML infps(view.infps) 병합.
    """
    models = window.get("models", {})
    per_view_rows: List[Dict[str, float]] = []

    for _, view in models.items():
        model_name = view.get("model")
        exec_dev_raw = view.get("execution")
        if not model_name or not exec_dev_raw:
            continue
        exec_dev = _norm_exec(exec_dev_raw)

        row: Dict[str, float] = {}
        for k in VIEW_DYNAMIC_KEYS:
            row[f"view.{k}"] = float(view.get(k, np.nan))

        row["view.exec_cpu"] = 1.0 if exec_dev == "CPU" else 0.0
        row["view.exec_npu0"] = 1.0 if exec_dev == "NPU0" else 0.0
        row["view.exec_npu1"] = 1.0 if exec_dev == "NPU1" else 0.0

        s_infer, s_load = _device_static_for(model_name, exec_dev, S)
        row["view.static_infer_sel"] = s_infer if np.isfinite(s_infer) else 0.0
        row["view.static_load_sel"] = s_load if np.isfinite(s_load) else 0.0

        infps_val = 0.0
        if infps_map is not None:
            try:
                infps_val = float(infps_map.get((model_name, exec_dev), 0.0))
            except Exception:
                infps_val = 0.0
        row["view.infps"] = infps_val

        row["x.view_throughput__static_infer_sel"] = row["view.throughput_fps"] * row["view.static_infer_sel"]
        row["x.view_wait__static_load_sel"] = row["view.avg_wait_to_preprocess_ms"] * row["view.static_load_sel"]

        per_view_rows.append(row)

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

    y1 = _nested(window, "total", "total_throughput_fps")
    y2 = _nested(window, "derived", "drop_rate_fps")

    meta = {
        "timestamp": window.get("timestamp"),
        "combination": window.get("combination"),
        "schedule_file": window.get("schedule file") or window.get("schedule_file") or window.get("schedule"),
    }
    return X, (y1, y2), meta


def _extract_schedule_hint(window: Dict[str, Any]) -> str:
    return str(window.get("schedule file") or window.get("schedule_file") or window.get("schedule") or "")


def build_dataset_from_file(perf_json_path: Path,
                            S: Dict[str, StaticProfile],
                            schedule_index: Optional[Dict[str, Dict[str, Any]]] = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blob = json.loads(perf_json_path.read_text(encoding="utf-8"))
    X_rows: List[Dict[str, float]] = []
    Y_rows: List[Dict[str, float]] = []
    M_rows: List[Dict[str, Any]] = []

    for w in blob.get("data", []):
        infps_map = None
        if schedule_index is not None:
            sched_hint = _extract_schedule_hint(w)
            sched_doc = _find_schedule(schedule_index, sched_hint)
            combo_name = str(w.get("combination") or "")
            if sched_doc is not None and combo_name:
                infps_map = _build_infps_lookup(sched_doc, combo_name)

        X, (y1, y2), meta = featurize_window(w, S, infps_map=infps_map)
        if math.isnan(y1) or math.isnan(y2):
            continue
        X_rows.append(X)
        Y_rows.append({"y1_total_throughput_fps": y1, "y2_drop_rate_fps": y2})
        M_rows.append(meta)

    X_df = pd.DataFrame(X_rows).fillna(0.0)
    Y_df = pd.DataFrame(Y_rows)
    M_df = pd.DataFrame(M_rows)
    return X_df, Y_df, M_df


def build_dataset(perf_dir: Path,
                  static_json_path: Path,
                  schedule_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    S = load_static_profiles(static_json_path)
    sched_index = _index_schedules(schedule_dir)

    X_all: List[pd.DataFrame] = []
    Y_all: List[pd.DataFrame] = []
    M_all: List[pd.DataFrame] = []

    json_paths = sorted([p for p in perf_dir.rglob("*.json") if p.is_file()])
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {perf_dir}")

    for path in json_paths:
        try:
            X, Y, M = build_dataset_from_file(path, S, schedule_index=sched_index)
            if not X.empty:
                X_all.append(X)
                Y_all.append(Y)
                M["source_file"] = str(path)
                M_all.append(M)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)

    if not X_all:
        raise RuntimeError("No valid training rows were built. Check your input logs & schedule_dir.")

    X_full = pd.concat(X_all, ignore_index=True).fillna(0.0)
    Y_full = pd.concat(Y_all, ignore_index=True)
    M_full = pd.concat(M_all, ignore_index=True)
    return X_full, Y_full, M_full


# ---------- Training / Predicting (two targets) ----------

def train_two_targets(X: pd.DataFrame, Y: pd.DataFrame, model_out_prefix: Path) -> None:
    xgb = _lazy_import_xgb()
    feat_names = list(X.columns)

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

    dtrain_y1 = xgb.DMatrix(X.values, label=Y["y1_total_throughput_fps"].values, feature_names=feat_names)
    bst1 = xgb.train(params, dtrain_y1, num_boost_round=400)
    p1 = str(model_out_prefix) + "_y1.json"
    Path(p1).parent.mkdir(parents=True, exist_ok=True)
    bst1.save_model(p1)

    dtrain_y2 = xgb.DMatrix(X.values, label=Y["y2_drop_rate_fps"].values, feature_names=feat_names)
    bst2 = xgb.train(params, dtrain_y2, num_boost_round=400)
    p2 = str(model_out_prefix) + "_y2.json"
    bst2.save_model(p2)


def predict_two_targets(model_in_prefix: Path, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible helper that loads models and predicts once.
    Note: Loading per call is slower; prefer using `load_two_models` +
    `predict_two_targets_loaded` when calling repeatedly.
    """
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


def load_two_models(model_in_prefix: Path):
    """Load y1 and y2 XGBoost boosters once and return them as a tuple."""
    xgb = _lazy_import_xgb()
    m1 = str(model_in_prefix) + "_y1.json"
    m2 = str(model_in_prefix) + "_y2.json"
    bst1 = xgb.Booster(model_file=m1)
    bst2 = xgb.Booster(model_file=m2)
    return bst1, bst2


def predict_two_targets_loaded(bst1, bst2, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using preloaded boosters (no reload)."""
    xgb = _lazy_import_xgb()
    feat_names = list(X.columns)
    dmat = xgb.DMatrix(X.values, feature_names=feat_names)
    y1_pred = bst1.predict(dmat)
    y2_pred = bst2.predict(dmat)
    return y1_pred, y2_pred


# ---------- predict (YAML만 사용) ----------

def featurize_from_combo(S: Dict[str, StaticProfile], combo_blob: Dict[str, Any]) -> pd.DataFrame:
    views = _rows_from_combo_struct(combo_blob)
    rows: List[Dict[str, float]] = []
    for v in views:
        m = v.get("model")
        dev = _norm_exec(v.get("execution", ""))
        if not m or not dev:
            continue

        s_infer, s_load = _device_static_for(m, dev, S)
        s_infer = float(s_infer) if np.isfinite(s_infer) else 0.0
        s_load = float(s_load) if np.isfinite(s_load) else 0.0

        if "infps" in v and v["infps"] is not None:
            try:
                fps = float(v["infps"])
            except Exception:
                fps = 0.0
            avg_inf_ms = 0.0 if fps <= 0 else (1000.0 / fps)
        else:
            fps = 0.0 if s_infer <= 0.0 else (1000.0 / s_infer)
            avg_inf_ms = s_infer

        inf_cnt = fps * WINDOW_SEC

        # Execution flags: map GPU onto legacy NPU1 flag for backward-compatible models
        exec_cpu = 1.0 if dev == "CPU" else 0.0
        exec_npu0 = 1.0 if dev == "NPU0" else 0.0
        exec_npu1 = 1.0 if dev in ("NPU1", "GPU") else 0.0
        r = {
            "view.throughput_fps": fps,
            "view.avg_inference_time_ms": avg_inf_ms,
            "view.inference_count": inf_cnt,
            "view.avg_wait_to_preprocess_ms": ASSUME_WAIT_MS,
            "view.dropped_frames_due_to_full_queue": 0.0,
            "view.exec_cpu": exec_cpu,
            "view.exec_npu0": exec_npu0,
            "view.exec_npu1": exec_npu1,
            "view.static_infer_sel": s_infer,
            "view.static_load_sel": s_load,
            "view.infps": fps,
        }
        r["x.view_throughput__static_infer_sel"] = r["view.throughput_fps"] * r["view.static_infer_sel"]
        r["x.view_wait__static_load_sel"] = r["view.avg_wait_to_preprocess_ms"] * r["view.static_load_sel"]
        rows.append(r)

    df = pd.DataFrame(rows)
    X: Dict[str, float] = {}
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

    return pd.DataFrame([X]).fillna(0.0)


# ---------- CLI ----------

def _normalize_model_prefix(path_str: str, default_name: str = "xgb2_model") -> Path:
    p = Path(path_str)
    if path_str.endswith("/") or (p.exists() and p.is_dir()):
        p = p / default_name
    return p


def main():
    ap = argparse.ArgumentParser(description="Two-target XGBoost trainer/inferencer for multi-view performance logs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # TRAIN (JSON + YAML)
    ap_tr = sub.add_parser("train", help="Train two-target model from JSON logs with YAML infps features.")
    ap_tr.add_argument("--perf_dir", type=str, required=True, help="Directory containing performance JSON logs.")
    ap_tr.add_argument("--schedule_dir", type=str, required=True, help="Directory containing YAML/JSON schedules.")
    ap_tr.add_argument("--static_json", type=str, required=True, help="Path to sample_profiling_data.json.")
    ap_tr.add_argument("--model_out", type=str, required=True, help="Prefix path for saving models (without _y*.json).")
    ap_tr.add_argument("--dump_csv", type=str, default="", help="Optional: path to dump engineered dataset CSV.")

    # PREDICT (YAML/JSON 스케줄, 디렉토리 또는 단일 파일)
    ap_pc = sub.add_parser("predict", help="Predict for planned combinations from schedules (single file or directory).")
    mx = ap_pc.add_mutually_exclusive_group(required=True)
    mx.add_argument("--schedule_yaml", type=str, help="Path to a YAML/JSON schedule with one or more combinations.")
    mx.add_argument("--schedule_dir", type=str, help="Directory containing YAML/JSON schedule files to run sequentially.")
    ap_pc.add_argument("--static_json", type=str, required=True, help="Path to sample_profiling_data.json.")
    ap_pc.add_argument("--model_in", type=str, required=True, help="Model prefix (expects _y1.json and _y2.json).")
    ap_pc.add_argument("--alpha", type=float, default=0.2, help="Score = FPS - alpha * DropRate (default: 0.2)")
    ap_pc.add_argument("--topk", type=int, default=0, help="If >0, print top-K combinations by score at the end.")
    ap_pc.add_argument("--repeats", type=int, default=1, help="Number of times to repeat prediction per schedule (default: 1).")

    args = ap.parse_args()

    if args.cmd == "train":
        perf_dir = Path(args.perf_dir)
        schedule_dir = Path(args.schedule_dir)
        static_json = Path(args.static_json)
        model_prefix = _normalize_model_prefix(args.model_out)

        X, Y, M = build_dataset(perf_dir, static_json, schedule_dir)

        if args.dump_csv:
            df_dump = pd.concat([M.reset_index(drop=True), X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
            Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
            df_dump.to_csv(args.dump_csv, index=False)
            print(f"[INFO] wrote dataset -> {args.dump_csv}  rows={len(df_dump)}")

        train_two_targets(X, Y, model_prefix)
        print(f"[OK] saved -> {model_prefix}_y1.json, {model_prefix}_y2.json")

    elif args.cmd == "predict":
        static_json = Path(args.static_json)
        model_prefix = _normalize_model_prefix(args.model_in)

        # Build list of schedule files
        sched_paths: List[Path] = []
        if getattr(args, "schedule_yaml", None):
            sched_paths = [Path(args.schedule_yaml)]
        elif getattr(args, "schedule_dir", None):
            d = Path(args.schedule_dir)
            if not d.exists() or not d.is_dir():
                print(f"[ERROR] schedule_dir not found or not a directory: {d}", file=sys.stderr)
                sys.exit(2)
            # collect .yaml/.yml/.json
            for ext in ("*.yaml", "*.yml", "*.json"):
                sched_paths.extend(sorted(d.glob(ext)))
            if not sched_paths:
                print(f"[ERROR] no schedule files (*.yaml|*.yml|*.json) under {d}", file=sys.stderr)
                sys.exit(2)
        else:
            print("[ERROR] either --schedule_yaml or --schedule_dir must be provided", file=sys.stderr)
            sys.exit(2)

        # Validate repeats
        repeats = int(getattr(args, "repeats", 1))
        if repeats < 1:
            print(f"[WARN] repeats < 1 ({repeats}); forcing to 1")
            repeats = 1

        # Preload static profiles and models once
        S = load_static_profiles(static_json)
        bst1, bst2 = load_two_models(model_prefix)
        xgb = _lazy_import_xgb()

        # Accumulators for per-schedule averages across repeats
        per_sched_infer_avgs: Dict[str, List[float]] = {}
        per_sched_y1_avgs: Dict[str, List[float]] = {}
        per_sched_y2_avgs: Dict[str, List[float]] = {}
        # Buffer to collect final summary lines for optional file output
        summary_lines: List[str] = []

        for sched_path in sched_paths:
            schedule = _load_yaml_or_json(Path(sched_path))
            combos = _iter_combos_from_schedule(schedule)
            base = Path(sched_path).name

            for r in range(repeats):
                # Per-run accumulators
                results: List[Tuple[str, float, float, float]] = []
                total_infer_s: float = 0.0
                total_y1_s: float = 0.0
                total_y2_s: float = 0.0

                # Optional run banner (no tabs to avoid interfering with parsers)
                if repeats > 1 or len(sched_paths) > 1:
                    print(f"-- Run {r+1}/{repeats} for {base} --")

                for name, combo_blob in combos:
                    X = featurize_from_combo(S, combo_blob)
                    feat_names = list(X.columns)
                    dmat = xgb.DMatrix(X.values, feature_names=feat_names)

                    t0 = time.perf_counter()
                    y1_pred = bst1.predict(dmat)
                    t1 = time.perf_counter()
                    y1_ms = (t1 - t0) * 1000.0
                    total_y1_s += (t1 - t0)

                    t2 = time.perf_counter()
                    y2_pred = bst2.predict(dmat)
                    t3 = time.perf_counter()
                    y2_ms = (t3 - t2) * 1000.0
                    total_y2_s += (t3 - t2)

                    infer_ms = y1_ms + y2_ms
                    total_infer_s += (t1 - t0) + (t3 - t2)

                    fps = float(y1_pred[0]); dropr = float(y2_pred[0])
                    score = fps - float(args.alpha) * dropr
                    results.append((name, fps, dropr, score))
                    print(f"{name}\t"
                          f"pred_total_throughput_fps={fps:.4f}\t"
                          f"pred_drop_rate_fps={dropr:.4f}\t"
                          f"pred_score(alpha={args.alpha:g})={score:.4f}\t"
                          f"infer_y1_ms={y1_ms:.2f}\t"
                          f"infer_y2_ms={y2_ms:.2f}\t"
                          f"infer_time_ms={infer_ms:.2f}")
                if results:
                    total_ms = total_infer_s * 1000.0
                    avg_ms = total_ms / max(len(results), 1)
                    total_y1_ms = total_y1_s * 1000.0
                    total_y2_ms = total_y2_s * 1000.0
                    avg_y1_ms = total_y1_ms / max(len(results), 1)
                    avg_y2_ms = total_y2_ms / max(len(results), 1)
                    print(f"TOTAL\tcombinations={len(results)}\ttotal_infer_time_ms={total_ms:.2f}\tavg_infer_time_ms={avg_ms:.2f}")
                    print(f"TOTAL_Y1\ttotal_infer_time_ms={total_y1_ms:.2f}\tavg_infer_time_ms={avg_y1_ms:.2f}")
                    print(f"TOTAL_Y2\ttotal_infer_time_ms={total_y2_ms:.2f}\tavg_infer_time_ms={avg_y2_ms:.2f}")

                    # Save per-run averages for final summary
                    per_sched_infer_avgs.setdefault(base, []).append(avg_ms)
                    per_sched_y1_avgs.setdefault(base, []).append(avg_y1_ms)
                    per_sched_y2_avgs.setdefault(base, []).append(avg_y2_ms)

                if results:
                    best_name, best_fps, best_drop, best_score = max(results, key=lambda x: x[3])
                    print(f"BEST\t{best_name}\t"
                          f"pred_total_throughput_fps={best_fps:.4f}\t"
                          f"pred_drop_rate_fps={best_drop:.4f}\t"
                          f"pred_score(alpha={args.alpha:g})={best_score:.4f}")
                    if int(args.topk) > 0:
                        topk = sorted(results, key=lambda x: x[3], reverse=True)[: int(args.topk)]
                        print("TOPK\t" + ", ".join([f"{n}:{s:.4f}" for n, _, __, s in topk]))

                    # Save minimal prediction summary JSON (backward compatibility)
                    try:
                        out_dir = Path("xgboost_model/performance_data/prediction_test_results")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        sched_stem = Path(sched_path).stem
                        out_path = out_dir / f"predict_performance_{stamp}_{sched_stem}.json"

                        # Build per-combination data list to include under the minimal summary as well
                        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        data_entries: List[Dict[str, Any]] = []
                        for (n, fps_v, drop_v, score_v) in results:
                            data_entries.append({
                                "timestamp": now_ts,
                                "window_sec": None,
                                "combination": n,
                                "total": {"total_throughput_fps": round(float(fps_v), 4)},
                                "derived": {"drop_rate_fps": round(float(drop_v), 4), "window_sec": 1.0},
                                "score": round(float(score_v), 4),
                            })

                        payload = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "schedule file": Path(sched_path).name,
                            "best deployment": best_name,
                            # We only predict total throughput and drop rate; use total as avg proxy
                            "total_throughput_fps": round(float(best_fps), 4),
                            "avg_throughput_fps": round(float(best_fps), 4),
                            "drop_rate_fps": round(float(best_drop), 4),
                            "score": round(float(best_score), 4),
                            "alpha": float(args.alpha),
                            "data": data_entries,
                        }
                        with out_path.open("w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                        print(f"[INFO] wrote prediction summary -> {out_path}")
                    except Exception as e:
                        print(f"[WARN] failed to write prediction summary JSON: {e}", file=sys.stderr)


        # Final summary across repeats per schedule (printed once before program exits)
        if per_sched_infer_avgs:
            line_all = "SUMMARY_ALL\t" + f"schedules={len(per_sched_infer_avgs)}"
            print(line_all)
            summary_lines.append(line_all)
            # Keep original order of sched_paths
            for sched_path in sched_paths:
                base = Path(sched_path).name
                if base not in per_sched_infer_avgs:
                    continue
                inf_list = per_sched_infer_avgs.get(base, [])
                y1_list = per_sched_y1_avgs.get(base, [])
                y2_list = per_sched_y2_avgs.get(base, [])
                # Use numpy for mean; guard against empty
                def _mean(lst: List[float]) -> float:
                    return float(np.mean(lst)) if lst else float("nan")
                line = (
                    f"SUMMARY\t{base}\t"
                    f"repeats={len(inf_list)}\t"
                    f"avg_infer_ms={_mean(inf_list):.2f}\t"
                    f"avg_y1_ms={_mean(y1_list):.2f}\t"
                    f"avg_y2_ms={_mean(y2_list):.2f}"
                )
                print(line)
                summary_lines.append(line)

            # Write summary to xgboost_model/performance_data/prediction_test_results/prediction_time_cpu_YYYYMMDD_HHMMSS.txt
            try:
                out_dir = Path("xgboost_model/performance_data/prediction_test_results")
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"prediction_time_cpu_{stamp}.txt"
                out_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
                print(f"[INFO] wrote final summary -> {out_path}")
            except Exception as e:
                print(f"[WARN] failed to write summary file: {e}", file=sys.stderr)

    else:
        ap.print_help()



def rows_from_schedule_yaml(schedule_yaml_path: str):
    """
    Compatibility shim for GUI predictor.
    Reads a YAML/JSON schedule file that contains one or more combinations,
    featurizes each combination using the same logic as CLI predict, and
    returns a list of rows: {"combination": name, "features": {...}}.
    """
    from pathlib import Path as _Path
    sched_path = _Path(schedule_yaml_path)
    schedule = _load_yaml_or_json(sched_path)

    # Load static profiling table located relative to this module
    module_dir = _Path(__file__).resolve().parent
    static_json = module_dir / "performance_data" / "sample_profiling_data" / "sample_profiling_data.json"
    S = load_static_profiles(static_json)

    rows = []
    for name, combo_blob in _iter_combos_from_schedule(schedule):
        Xdf = featurize_from_combo(S, combo_blob)
        # Convert single-row DataFrame to plain dict of features
        feats = {k: float(Xdf.iloc[0][k]) for k in Xdf.columns}
        rows.append({
            "combination": str(name),
            "features": feats,
        })
    return rows


if __name__ == "__main__":
    main()
