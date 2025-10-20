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

import argparse
import json
import math
import sys
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

        r = {
            "view.throughput_fps": fps,
            "view.avg_inference_time_ms": avg_inf_ms,
            "view.inference_count": inf_cnt,
            "view.avg_wait_to_preprocess_ms": ASSUME_WAIT_MS,
            "view.dropped_frames_due_to_full_queue": 0.0,
            "view.exec_cpu": 1.0 if dev == "CPU" else 0.0,
            "view.exec_npu0": 1.0 if dev == "NPU0" else 0.0,
            "view.exec_npu1": 1.0 if dev == "NPU1" else 0.0,
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

    # PREDICT (YAML만 사용)
    ap_pc = sub.add_parser("predict", help="Predict for planned combinations from a YAML/JSON schedule.")
    ap_pc.add_argument("--schedule_yaml", type=str, required=True, help="Path to YAML schedule with one or more combinations.")
    ap_pc.add_argument("--static_json", type=str, required=True, help="Path to sample_profiling_data.json.")
    ap_pc.add_argument("--model_in", type=str, required=True, help="Model prefix (expects _y1.json and _y2.json).")
    ap_pc.add_argument("--alpha", type=float, default=0.2, help="Score = FPS - alpha * DropRate (default: 0.2)")
    ap_pc.add_argument("--topk", type=int, default=0, help="If >0, print top-K combinations by score at the end.")

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
        sched_path = Path(args.schedule_yaml)
        static_json = Path(args.static_json)
        model_prefix = _normalize_model_prefix(args.model_in)

        S = load_static_profiles(static_json)
        schedule = _load_yaml_or_json(sched_path)
        combos = _iter_combos_from_schedule(schedule)

        results: List[Tuple[str, float, float, float]] = []
        for name, combo_blob in combos:
            X = featurize_from_combo(S, combo_blob)
            y1_pred, y2_pred = predict_two_targets(model_prefix, X)
            fps = float(y1_pred[0]); dropr = float(y2_pred[0])
            score = fps - float(args.alpha) * dropr
            results.append((name, fps, dropr, score))
            print(f"{name}\t"
                  f"pred_total_throughput_fps={fps:.4f}\t"
                  f"pred_drop_rate_fps={dropr:.4f}\t"
                  f"pred_score(alpha={args.alpha:g})={score:.4f}")

        if results:
            best_name, best_fps, best_drop, best_score = max(results, key=lambda x: x[3])
            print(f"BEST\t{best_name}\t"
                  f"pred_total_throughput_fps={best_fps:.4f}\t"
                  f"pred_drop_rate_fps={best_drop:.4f}\t"
                  f"pred_score(alpha={args.alpha:g})={best_score:.4f}")
            if int(args.topk) > 0:
                topk = sorted(results, key=lambda x: x[3], reverse=True)[: int(args.topk)]
                print("TOPK\t" + ", ".join([f"{n}:{s:.4f}" for n, _, __, s in topk]))

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
