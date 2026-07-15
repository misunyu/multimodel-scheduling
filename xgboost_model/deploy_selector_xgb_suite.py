#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_selector_xgb_suite.py

Three-target XGBoost training/inference for multi-view performance logs on the
Mobilint NPU + GPU + CPU stack.

Targets (window-level):
  - y1 = total vision throughput (sum of per-view FPS over vision models)
  - y2 = drop rate (derived.drop_rate_fps)
  - y3 = total generative throughput (sum of per-view tokens/sec over LLM/VLM)

Devices: cpu / gpu / npu.  Static features come from sample_profiling_data.json
(per-model, per-device: load ms, infer/prefill ms, tokens/sec).

CLI
---
Train:
  python deploy_selector_xgb_suite.py train \
    --perf_dir ./xgboost_model/performance_data \
    --schedule_dir ./xgboost_model/schedules \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_out ./xgboost_model/artifacts/deploy_xgb

Predict:
  python deploy_selector_xgb_suite.py predict \
    --schedule_yaml ./xgboost_model/schedules/model_schedules.yaml \
    --static_json ./xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_in ./xgboost_model/artifacts/deploy_xgb [--alpha 0.2] [--beta 0.1] [--topk 5]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Make the model registry importable whether run from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import model_registry as reg
except Exception:
    reg = None

WINDOW_SEC = 30.0
ASSUME_WAIT_MS = 0.0
DEVICES = ["cpu", "gpu", "npu"]


# ---------- xgboost lazy import ----------
def _lazy_import_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception as e:
        raise RuntimeError(f"xgboost is required (pip install xgboost). Original error: {e}")


# ---------- registry-independent helpers ----------
def _model_kind(model: str) -> str:
    if reg is not None:
        try:
            return reg.kind_of(model)
        except Exception:
            pass
    low = (model or "").lower()
    if "yolo" in low or "resnet" in low:
        return "vision"
    if "qwen" in low or "vl" in low:
        return "vlm"
    return "llm"


def _is_vision(model: str) -> bool:
    return _model_kind(model) == "vision"


def _norm_exec(dev: str) -> str:
    d = str(dev).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if d == "cpu":
        return "cpu"
    if d in ("gpu", "cuda"):
        return "gpu"
    if d.startswith("npu"):
        return "npu"
    return d


# ---------- static profiles ----------
def load_static_profiles(static_json_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {model: {device: {"load":.., "infer":.., "tokens":..}}}."""
    blob = json.loads(Path(static_json_path).read_text(encoding="utf-8"))
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in blob.get("total_data", []):
        model = row.get("model")
        if not model:
            continue
        per_dev: Dict[str, Dict[str, float]] = {}
        for dev in DEVICES:
            def _f(key):
                v = row.get(f"{dev}_{key}")
                try:
                    return float(v)
                except Exception:
                    return np.nan
            per_dev[dev] = {
                "load": _f("load"),
                "infer": _f("infer"),
                "tokens": _f("tokens_per_s"),
            }
        table[model] = per_dev
    return table


def _device_static(model: str, dev: str, S: Dict[str, Dict[str, Dict[str, float]]]) -> Tuple[float, float, float]:
    """Return (infer_ms, load_ms, tokens_per_s) for a model on a device."""
    prof = S.get(model)
    d = _norm_exec(dev)
    if prof is None or d not in prof:
        return (np.nan, np.nan, np.nan)
    e = prof[d]
    return (e.get("infer", np.nan), e.get("load", np.nan), e.get("tokens", np.nan))


# ---------- YAML schedule helpers ----------
def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    txt = Path(path).read_text(encoding="utf-8")
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
    raise ValueError("combo must have 'views' (list) or 'models' (dict) or per-view dict entries.")


def _index_schedules(schedule_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not Path(schedule_dir).exists():
        return index
    exts = {".yaml", ".yml", ".json"}
    for p in Path(schedule_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                index[p.name.lower()] = _load_yaml_or_json(p)
            except Exception:
                pass
    return index


def _find_schedule(index: Dict[str, Dict[str, Any]], hint: str) -> Optional[Dict[str, Any]]:
    if not hint:
        return None
    return index.get(Path(hint).name.lower())


def _build_infps_lookup(schedule_doc: Dict[str, Any], combination_name: str) -> Dict[Tuple[str, str], float]:
    infps_map: Dict[Tuple[str, str], float] = {}
    try:
        combos = _iter_combos_from_schedule(schedule_doc)
        target_blob = None
        for name, blob in combos:
            if str(name) == str(combination_name):
                target_blob = blob
                break
        if target_blob is None and len(combos) == 1:
            target_blob = combos[0][1]
        if target_blob is None:
            return infps_map
        for r in _rows_from_combo_struct(target_blob):
            m = r.get("model"); dev = _norm_exec(r.get("execution", ""))
            if m and dev and r.get("infps") is not None:
                try:
                    infps_map[(m, dev)] = float(r["infps"])
                except Exception:
                    pass
    except Exception:
        pass
    return infps_map


# ---------- feature engineering ----------
def _view_features(model: str, exec_dev: str, infps: float,
                   S: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    dev = _norm_exec(exec_dev)
    s_infer, s_load, s_tokens = _device_static(model, dev, S)
    kind = _model_kind(model)
    s_infer = float(s_infer) if np.isfinite(s_infer) else 0.0
    s_load = float(s_load) if np.isfinite(s_load) else 0.0
    s_tokens = float(s_tokens) if np.isfinite(s_tokens) else 0.0
    infps = float(infps or 0.0)
    # Device throughput ceiling (uncontended) and how hard the input rate pushes it.
    capacity = (1000.0 / s_infer) if s_infer > 0 else 0.0
    load_factor = (infps / capacity) if capacity > 0 else 0.0
    # NOTE: only static-profile + plan features are used (device, static latency,
    # tokens/sec, infps). Measured per-view dynamics are NOT features — they are
    # unavailable at predict time and using them leaks the target.
    row = {
        "view.infps": infps,
        "view.exec_cpu": 1.0 if dev == "cpu" else 0.0,
        "view.exec_gpu": 1.0 if dev == "gpu" else 0.0,
        "view.exec_npu": 1.0 if dev == "npu" else 0.0,
        "view.is_vision": 1.0 if kind == "vision" else 0.0,
        "view.is_llm": 1.0 if kind in ("llm", "vlm") else 0.0,
        "view.static_infer_sel": s_infer,
        "view.static_load_sel": s_load,
        "view.static_tokens_sel": s_tokens,
        "view.capacity_fps": capacity,
        "view.load_factor": load_factor,
    }
    row["x.infps__static_infer_sel"] = infps * s_infer
    return row


def _aggregate(per_view_rows: List[Dict[str, float]]) -> Dict[str, float]:
    X: Dict[str, float] = {}
    df = pd.DataFrame(per_view_rows)
    if not df.empty:
        for agg_name, s in {"sum": df.sum(numeric_only=True),
                            "mean": df.mean(numeric_only=True),
                            "max": df.max(numeric_only=True)}.items():
            for col, val in s.items():
                X[f"views.{agg_name}.{col}"] = float(val)
        X["views.count.views"] = float(len(df))
    else:
        X["views.count.views"] = 0.0
    return X


def featurize_window(window: Dict[str, Any],
                     S: Dict[str, Dict[str, Dict[str, float]]],
                     infps_map: Optional[Dict[Tuple[str, str], float]] = None
                     ) -> Tuple[Dict[str, float], Tuple[float, float, float], Dict[str, Any]]:
    models = window.get("models", {})
    per_view_rows: List[Dict[str, float]] = []
    model_names: List[str] = []
    devices: List[str] = []
    y1_vision_fps = 0.0
    y3_tokens = 0.0
    y3_valid = False   # True iff some LLM/VLM view is on an accelerator (npu/gpu)
    for _, view in models.items():
        model_name = view.get("model")
        exec_dev_raw = view.get("execution")
        if not model_name or not exec_dev_raw:
            continue
        dev = _norm_exec(exec_dev_raw)
        infps = 0.0
        if infps_map is not None:
            infps = float(infps_map.get((model_name, dev), 0.0))
        per_view_rows.append(_view_features(model_name, dev, infps, S))
        model_names.append(str(model_name))
        devices.append(dev)
        if _is_vision(model_name):
            y1_vision_fps += float(view.get("throughput_fps", 0.0) or 0.0)
        else:
            y3_tokens += float(view.get("tokens_per_s", 0.0) or 0.0)
            # A generative model on the CPU takes ~198s per generation, so in a 180s
            # window it completes 0-1 and its token throughput is pure noise (measured
            # in the pilot). Its y3 must not become a training label. y3 is a usable
            # label only when at least one LLM/VLM view actually runs on an accelerator.
            if dev in ("npu", "gpu"):
                y3_valid = True

    X = _aggregate(per_view_rows)

    def _nested(d, dotted, field, default=np.nan):
        cur = d
        for part in dotted.split("."):
            cur = cur.get(part, {}) if isinstance(cur, dict) else {}
        try:
            return float(cur.get(field, default)) if isinstance(cur, dict) else default
        except Exception:
            return default

    # y2 is the window-level deadline miss rate (already in [0,1]).
    y2 = _nested(window, "derived", "deadline_miss_rate")
    if not np.isfinite(y2):
        y2 = _nested(window, "total", "deadline_miss_rate")
    meta = {
        "timestamp": window.get("timestamp"),
        "combination": window.get("combination"),
        "schedule_file": window.get("schedule file") or window.get("schedule_file") or window.get("schedule"),
        "workload": window.get("workload"),
        "rate_factor": window.get("rate_factor"),
        # Model set / devices of this window, used to record training coverage.
        "models": ",".join(sorted(model_names)),
        "devices": ",".join(sorted(set(devices))),
        # Per-COMBINATION y3 validity, not per-set: a set with an LLM whose LLM sits on
        # CPU has no usable token label. Used to mask the y3 training rows.
        "y3_valid": bool(y3_valid),
    }
    # y1_vision_fps and y3_tokens are RAW here; normalized per-workload in build_dataset.
    return X, (y1_vision_fps, y2, y3_tokens), meta


def featurize_from_combo(S: Dict[str, Dict[str, Dict[str, float]]], combo_blob: Dict[str, Any]) -> pd.DataFrame:
    views = _rows_from_combo_struct(combo_blob)
    rows: List[Dict[str, float]] = []
    for v in views:
        m = v.get("model"); dev = _norm_exec(v.get("execution", ""))
        if not m or not dev:
            continue
        infps = 0.0
        if v.get("infps") is not None:
            try:
                infps = float(v["infps"])
            except Exception:
                infps = 0.0
        rows.append(_view_features(m, dev, infps, S))
    X = _aggregate(rows)
    return pd.DataFrame([X]).fillna(0.0)


# ---------- dataset build ----------
def _extract_schedule_hint(window: Dict[str, Any]) -> str:
    return str(window.get("schedule file") or window.get("schedule_file") or window.get("schedule") or "")


def _windows_from_blob(blob: Any) -> List[Dict[str, Any]]:
    if isinstance(blob, dict):
        if isinstance(blob.get("data"), list):
            return blob["data"]
        return [blob]
    if isinstance(blob, list):
        # list of windows, or list of {data:[...]} wrappers
        out = []
        for item in blob:
            if isinstance(item, dict) and isinstance(item.get("data"), list):
                out.extend(item["data"])
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []


def build_dataset_from_file(perf_json_path: Path,
                            S: Dict[str, Dict[str, Dict[str, float]]],
                            schedule_index: Optional[Dict[str, Dict[str, Any]]] = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blob = json.loads(Path(perf_json_path).read_text(encoding="utf-8"))
    X_rows, Y_rows, M_rows = [], [], []
    for w in _windows_from_blob(blob):
        infps_map = None
        if schedule_index is not None:
            sched_doc = _find_schedule(schedule_index, _extract_schedule_hint(w))
            combo_name = str(w.get("combination") or "")
            if sched_doc is not None and combo_name:
                infps_map = _build_infps_lookup(sched_doc, combo_name)
        X, (y1, y2, y3), meta = featurize_window(w, S, infps_map=infps_map)
        if math.isnan(y2):
            y2 = 0.0
        X_rows.append(X)
        Y_rows.append({"y1_total_throughput_fps": y1, "y2_deadline_miss_rate": y2, "y3_total_tokens_per_s": y3})
        M_rows.append(meta)
    return (pd.DataFrame(X_rows).fillna(0.0), pd.DataFrame(Y_rows), pd.DataFrame(M_rows))


def _normalize_targets(Y: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw throughput targets to [0,1] within each (models, workload, rate).

    T(x) = F(x) / Fmax, where Fmax is the max measured throughput across PLACEMENTS OF
    THE SAME WORKING SET at the same input-rate level. The group MUST include the model
    set (`models`): a 2-model set and an 8-model set can share the same workload
    (yolo11s is the detection model in nine of the fifteen sets) and the same rate, and
    normalising them together would divide the small set's throughput by the large
    set's Fmax -- a set-size confound. Normalising per working set makes T a within-set
    "fraction of the best placement", which is exactly what the score ranks. y2 (miss
    rate) is already in [0,1] and left unchanged.
    """
    Y = Y.copy()
    grp = list(zip(M.get("models", pd.Series([None] * len(M))),
                   M.get("workload", pd.Series([None] * len(M))),
                   M.get("rate_factor", pd.Series([None] * len(M)))))
    for col in ("y1_total_throughput_fps", "y3_total_tokens_per_s"):
        vals = Y[col].values.astype(float)
        out = np.zeros_like(vals)
        # group indices
        groups = {}
        for i, g in enumerate(grp):
            groups.setdefault(g, []).append(i)
        for g, idxs in groups.items():
            fmax = max((vals[i] for i in idxs), default=0.0)
            for i in idxs:
                out[i] = (vals[i] / fmax) if fmax > 1e-9 else 0.0
        Y[col] = out
    # y2 clip to [0,1] for safety
    Y["y2_deadline_miss_rate"] = np.clip(Y["y2_deadline_miss_rate"].values.astype(float), 0.0, 1.0)
    return Y


def build_dataset(perf_dir: Path, static_json_path: Path, schedule_dir: Path,
                  keep_rate_factors=None, normalize=True):
    S = load_static_profiles(static_json_path)
    sched_index = _index_schedules(schedule_dir)
    X_all, Y_all, M_all = [], [], []
    # Only performance_*.json are contention windows; ignore profiles / timing logs.
    json_paths = sorted([p for p in Path(perf_dir).rglob("performance*.json") if p.is_file()])
    if not json_paths:
        raise FileNotFoundError(f"No performance_*.json files found in: {perf_dir}")
    for path in json_paths:
        if "sample_profiling_data" in str(path):
            continue
        try:
            X, Y, M = build_dataset_from_file(path, S, schedule_index=sched_index)
            if not X.empty:
                X_all.append(X); Y_all.append(Y)
                M["source_file"] = str(path); M_all.append(M)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)
    if not X_all:
        raise RuntimeError("No valid training rows. Check performance logs & schedule_dir.")
    X_full = pd.concat(X_all, ignore_index=True).fillna(0.0)
    Y_full = pd.concat(Y_all, ignore_index=True)
    M_full = pd.concat(M_all, ignore_index=True).reset_index(drop=True)

    # Optional rate hold-out filter (e.g. train on 1x/2x/4x, hold out 3x).
    if keep_rate_factors is not None:
        keep = set(float(r) for r in keep_rate_factors)
        mask = M_full["rate_factor"].apply(lambda r: (r is not None) and (float(r) in keep)).values
        X_full = X_full[mask].reset_index(drop=True)
        Y_full = Y_full[mask].reset_index(drop=True)
        M_full = M_full[mask].reset_index(drop=True)

    # Normalize throughput targets to [0,1] within each (workload, rate) group.
    if normalize:
        Y_full = _normalize_targets(Y_full, M_full)
    return X_full, Y_full, M_full


# ---------- train / predict ----------
_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "max_depth": 5, "eta": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 3.0, "seed": 42,
}
# Small hyperparameter grid searched by 3-fold CV (MAE) per target (paper Sec 4.2).
_PARAM_GRID = [
    {"max_depth": 3, "eta": 0.1},
    {"max_depth": 4, "eta": 0.1},
    {"max_depth": 5, "eta": 0.05},
    {"max_depth": 6, "eta": 0.1},
]
_TARGETS = [("y1", "y1_total_throughput_fps"), ("y2", "y2_deadline_miss_rate"), ("y3", "y3_total_tokens_per_s")]


def _cv_select_params(xgb, Xv, yv, feat_names, folds=3, rounds=300):
    """Pick grid params minimizing mean absolute error via k-fold CV (paper Sec 4.2)."""
    n = len(yv)
    fold_id = np.arange(n) % folds
    best, best_mae = None, float("inf")
    for extra in _PARAM_GRID:
        params = dict(_PARAMS); params.update(extra)
        maes = []
        for k in range(folds):
            tr, te = fold_id != k, fold_id == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            d = xgb.DMatrix(Xv[tr], label=yv[tr], feature_names=feat_names)
            bst = xgb.train(params, d, num_boost_round=rounds)
            pred = bst.predict(xgb.DMatrix(Xv[te], feature_names=feat_names))
            maes.append(float(np.mean(np.abs(pred - yv[te]))))
        mae = float(np.mean(maes)) if maes else float("inf")
        if mae < best_mae:
            best_mae, best = mae, params
    return best or dict(_PARAMS), best_mae


def write_coverage(M: pd.DataFrame, model_out_prefix: Path) -> Dict[str, Any]:
    """Record which model sets / devices / rates the predictor actually saw.

    Consumers (e.g. the deploy-finder GUI) compare a user's selection against
    this to warn when a request falls outside the training distribution.
    """
    def _sets(col):
        if col not in M.columns:
            return []
        return sorted({s for s in M[col].dropna().astype(str) if s})

    model_sets = _sets("models")
    coverage = {
        "model_sets": model_sets,
        "models": sorted({m for s in model_sets for m in s.split(",") if m}),
        "view_counts": sorted({len([m for m in s.split(",") if m]) for s in model_sets}),
        "devices": sorted({d for s in _sets("devices") for d in s.split(",") if d}),
        "rate_factors": sorted({float(r) for r in M.get("rate_factor", pd.Series(dtype=float)).dropna()}),
        "rows": int(len(M)),
    }
    Path(str(model_out_prefix) + "_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return coverage


def train_targets(X: pd.DataFrame, Y: pd.DataFrame, model_out_prefix: Path,
                  M: pd.DataFrame = None) -> None:
    xgb = _lazy_import_xgb()
    feat_names = list(X.columns)
    Path(str(model_out_prefix)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(model_out_prefix) + "_features.json").write_text(json.dumps(feat_names), encoding="utf-8")

    # y3 is trained ONLY on windows whose token throughput is a real measurement: at
    # least one LLM/VLM view actually ran on an accelerator. This is a per-COMBINATION
    # decision, not per-set -- a set that contains an LLM but places it on the CPU has
    # y3~0 noise (a CPU generation takes ~198s, 0-1 finish in the 180s window), and a
    # vision-only set has no tokens at all. Both must be excluded, or the model learns
    # that certain feature combinations mean "no tokens" and then predicts ~0.89 tokens
    # for a set with no LLM in it. `M["y3_valid"]` carries this per-row (set in
    # featurize_window). Fall back to the coarse set-level feature only when M is absent.
    if M is not None and "y3_valid" in M.columns:
        gen_mask = M["y3_valid"].values.astype(bool)
    elif "views.sum.view.is_llm" in X.columns:
        gen_mask = X["views.sum.view.is_llm"].values > 0
    else:
        gen_mask = None

    for tag, col in _TARGETS:
        Xi, Yi = X, Y
        if tag == "y3" and gen_mask is not None:
            Xi, Yi = X[gen_mask], Y[gen_mask]
            dropped = int((~gen_mask).sum())
            print(f"[train] y3: {dropped} rows masked (no LLM/VLM on accelerator); "
                  f"{len(Xi)} of {len(X)} kept")
            if len(Xi) == 0:
                print("[train] y3: no valid generative rows; skipping the y3 model.")
                continue
        yv = Yi[col].values.astype(float)
        params, mae = _cv_select_params(xgb, Xi.values, yv, feat_names)
        print(f"[train] {tag} ({col}): params max_depth={params['max_depth']} eta={params['eta']} cv_mae={mae:.4f}")
        d = xgb.DMatrix(Xi.values, label=yv, feature_names=feat_names)
        bst = xgb.train(params, d, num_boost_round=300)
        bst.save_model(str(model_out_prefix) + f"_{tag}.json")


def out_of_fold_predictions(X: pd.DataFrame, Y: pd.DataFrame, M: pd.DataFrame,
                            folds: int = 3, seed: int = 42) -> pd.DataFrame:
    """Predict every row using models trained WITHOUT that row (k-fold), for honest
    validation. A row's y3 prediction is left NaN when its own y3 label is masked.

    Returns a DataFrame with pred_y1/pred_y2/pred_y3 aligned to X's index, so a caller
    can rank combinations within a set using predictions the model never saw at train.
    """
    xgb = _lazy_import_xgb()
    feat = list(X.columns)
    n = len(X)
    rng = np.random.RandomState(seed)
    fold_id = rng.randint(0, folds, size=n)
    y3_valid = (M["y3_valid"].values.astype(bool) if (M is not None and "y3_valid" in M.columns)
                else np.ones(n, bool))
    out = {t: np.full(n, np.nan) for _, t in [("y1", "pred_y1"), ("y2", "pred_y2"), ("y3", "pred_y3")]}
    name = {"y1_total_throughput_fps": "pred_y1", "y2_deadline_miss_rate": "pred_y2",
            "y3_total_tokens_per_s": "pred_y3"}
    for f in range(folds):
        tr = fold_id != f
        te = fold_id == f
        for tag, col in _TARGETS:
            tr_mask = tr & (y3_valid if tag == "y3" else np.ones(n, bool))
            te_mask = te & (y3_valid if tag == "y3" else np.ones(n, bool))
            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue
            d = xgb.DMatrix(X.values[tr_mask], label=Y[col].values[tr_mask].astype(float),
                            feature_names=feat)
            bst = xgb.train(_PARAMS, d, num_boost_round=300)
            pred = np.clip(bst.predict(xgb.DMatrix(X.values[te_mask], feature_names=feat)), 0.0, 1.0)
            out[name[col]][np.where(te_mask)[0]] = pred
    return pd.DataFrame(out, index=X.index)


def combo_has_generative(combo_blob) -> bool:
    """Does this working set contain any LLM/VLM view?

    Decided from the registry (`kind_of`), never from a hardcoded name list, so adding
    a model does not silently fall on the wrong side of the split.
    """
    for v in _rows_from_combo_struct(combo_blob):
        m = v.get("model")
        if not m:
            continue
        try:
            if _model_kind(m) in ("llm", "vlm"):
                return True
        except Exception:
            continue
    return False


def score_combo(y1: float, y2: float, y3: float, has_generative: bool,
                alpha: float, beta: float) -> float:
    """S = y1 - alpha*y2, plus beta*y3 only when the set actually generates tokens.

    A vision-only set has no token throughput to trade off: measured y3 is 0, but the
    predictor -- never trained on such a set -- happily emits ~0.89, which at beta=0.5
    handed every vision-only combination half a point of free score. The term is not
    zeroed after the fact; it is not in the objective at all.

    NOTE: this makes the score's scale depend on the set type, so scores are comparable
    only WITHIN one working set (which is all they are used for: ranking placements).
    Never compare a score across working sets.
    """
    s = float(y1) - float(alpha) * float(y2)
    if has_generative:
        s += float(beta) * float(y3)
    return s


def _align_features(X: pd.DataFrame, model_in_prefix: Path) -> pd.DataFrame:
    fpath = Path(str(model_in_prefix) + "_features.json")
    if fpath.exists():
        feat_names = json.loads(fpath.read_text(encoding="utf-8"))
        for c in feat_names:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feat_names]
    return X


def predict_targets(model_in_prefix: Path, X: pd.DataFrame,
                    with_y3: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict y1/y2 (and y3 only when asked).

    `with_y3=False` for a vision-only set: the y3 booster is not loaded and not run.
    Returning a number and discarding it later would leave the door open for someone to
    use it; the honest answer for a set that generates no tokens is that the question
    does not apply.
    """
    xgb = _lazy_import_xgb()
    X = _align_features(X.copy(), model_in_prefix)
    dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
    preds = {}
    for tag, _ in _TARGETS:
        if tag == "y3" and not with_y3:
            preds[tag] = np.zeros(X.shape[0], dtype=float)
            continue
        bst = xgb.Booster(model_file=str(model_in_prefix) + f"_{tag}.json")
        # All targets are normalized to [0,1]; clip predictions to the valid range.
        preds[tag] = np.clip(bst.predict(dmat), 0.0, 1.0)
    return preds["y1"], preds["y2"], preds["y3"]


# Backward-compatible alias (older callers expected 2 targets).
def predict_two_targets(model_in_prefix: Path, X: pd.DataFrame):
    y1, y2, _ = predict_targets(model_in_prefix, X)
    return y1, y2


# ---------- CLI ----------
def _normalize_model_prefix(path_str: str, default_name: str = "deploy_xgb") -> Path:
    p = Path(path_str)
    if path_str.endswith("/") or (p.exists() and p.is_dir()):
        p = p / default_name
    return p


def main():
    ap = argparse.ArgumentParser(description="Three-target XGBoost trainer/inferencer (cpu/gpu/npu).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--perf_dir", required=True)
    ap_tr.add_argument("--schedule_dir", required=True)
    ap_tr.add_argument("--static_json", required=True)
    ap_tr.add_argument("--model_out", required=True)
    ap_tr.add_argument("--dump_csv", default="")
    _train_rates = getattr(reg, "TRAIN_RATE_FACTORS", [1.0, 2.0, 4.0]) if reg else [1.0, 2.0, 4.0]
    ap_tr.add_argument("--rate_factors", nargs="*", type=float, default=_train_rates,
                       help="input-rate levels to train on (default 1x/2x/4x; 3x is held out)")

    ap_pc = sub.add_parser("predict")
    ap_pc.add_argument("--schedule_yaml", required=True)
    ap_pc.add_argument("--static_json", required=True)
    ap_pc.add_argument("--model_in", required=True)
    ap_pc.add_argument("--alpha", type=float, default=0.3, help="deadline-miss penalty (normalized)")
    ap_pc.add_argument("--beta", type=float, default=0.5, help="token-throughput reward (normalized)")
    ap_pc.add_argument("--topk", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "train":
        X, Y, M = build_dataset(Path(args.perf_dir), Path(args.static_json), Path(args.schedule_dir),
                                keep_rate_factors=args.rate_factors, normalize=True)
        prefix = _normalize_model_prefix(args.model_out)
        if args.dump_csv:
            dump = pd.concat([M.reset_index(drop=True), X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
            Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
            dump.to_csv(args.dump_csv, index=False)
            print(f"[INFO] wrote dataset -> {args.dump_csv} rows={len(dump)}")
        train_targets(X, Y, prefix)
        cov = write_coverage(M, prefix)
        print(f"[OK] saved -> {prefix}_y1.json, {prefix}_y2.json, {prefix}_y3.json (rows={len(X)})")
        print(f"[OK] coverage -> {prefix}_coverage.json "
              f"(models={cov['models']}, views={cov['view_counts']}, rates={cov['rate_factors']})")

    elif args.cmd == "predict":
        S = load_static_profiles(Path(args.static_json))
        schedule = _load_yaml_or_json(Path(args.schedule_yaml))
        prefix = _normalize_model_prefix(args.model_in)
        results = []
        for name, combo_blob in _iter_combos_from_schedule(schedule):
            X = featurize_from_combo(S, combo_blob)
            gen = combo_has_generative(combo_blob)
            y1, y2, y3 = predict_targets(prefix, X, with_y3=gen)
            T, miss, Ttok = float(y1[0]), float(y2[0]), float(y3[0])
            score = score_combo(T, miss, Ttok, gen, args.alpha, args.beta)
            results.append((name, T, miss, Ttok, score))
            print(f"{name}\tT={T:.3f}\tmiss={miss:.3f}\tT_tok={Ttok:.3f}\tscore={score:.3f}")
        if results:
            best = max(results, key=lambda x: x[4])
            print(f"BEST\t{best[0]}\tT={best[1]:.3f}\tmiss={best[2]:.3f}\tT_tok={best[3]:.3f}\tscore={best[4]:.3f}")
            if int(args.topk) > 0:
                topk = sorted(results, key=lambda x: x[4], reverse=True)[:int(args.topk)]
                print("TOPK\t" + ", ".join([f"{n}:{s:.3f}" for n, _, __, ___, s in topk]))
    else:
        ap.print_help()


def rows_from_schedule_yaml(schedule_yaml_path: str):
    """Compatibility shim for the GUI predictor."""
    sched_path = Path(schedule_yaml_path)
    schedule = _load_yaml_or_json(sched_path)
    module_dir = Path(__file__).resolve().parent
    static_json = module_dir / "performance_data" / "sample_profiling_data" / "sample_profiling_data.json"
    S = load_static_profiles(static_json)
    rows = []
    for name, combo_blob in _iter_combos_from_schedule(schedule):
        Xdf = featurize_from_combo(S, combo_blob)
        feats = {k: float(Xdf.iloc[0][k]) for k in Xdf.columns}
        rows.append({"combination": str(name), "features": feats})
    return rows


if __name__ == "__main__":
    main()
