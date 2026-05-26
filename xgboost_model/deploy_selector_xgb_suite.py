#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Train]
 1) Using directory (two_target):
    python ./xgboost_model/deploy_selector_xgb_suite.py train --perf_dir ./results_recompute --schedule_dir ./gen_schedules --model_out ./xgboost_model/artifacts/gpu/xgb_model

 2) Using CSV (two_target, Random split):
    python ./xgboost_model/deploy_selector_xgb_suite.py train --perf_csv train_random.csv --schedule_csv train_schedules_random.csv --model_out ./xgboost_model/artifacts/gpu/xgb_model_random

 3) Using CSV (two_target, Pattern x3 split):
    python ./xgboost_model/deploy_selector_xgb_suite.py train --perf_csv train_x3.csv --schedule_csv train_schedules_x3.csv --model_out ./xgboost_model/artifacts/gpu/xgb_model_x3

 4) Using Score Mode (1-target regression):
    python ./xgboost_model/deploy_selector_xgb_suite.py train --train_mode score --perf_csv train_x3.csv --schedule_csv train_schedules_x3.csv --model_out ./xgboost_model/artifacts/gpu/xgb_model_score --alpha 0.2

 5) Using Rank Mode (Learning-to-Rank):
    python ./xgboost_model/deploy_selector_xgb_suite.py train --train_mode rank --perf_csv train_x3.csv --schedule_csv train_schedules_x3.csv --model_out ./xgboost_model/artifacts/gpu/xgb_model_rank --alpha 0.2

[Predict / Validate]
  python ./xgboost_model/deploy_selector_xgb_suite.py predict --perf_csv ./xgboost_model/dataset/gpu/test_x3.csv --schedule_csv ./xgboost_model/dataset/gpu/test_schedules_x3.csv --model_in ./xgboost_model/artifacts/gpu/xgb_model_x3_double --alpha 0.2

 1) Predict for new schedules (Top-K output):
    python ./xgboost_model/deploy_selector_xgb_suite.py predict --schedule_dir ./gen_schedules --model_in ./xgboost_model/artifacts/gpu/xgb_model_random --topk 5 --alpha 0.2

 2) Validate with test CSV (Random split):
    python ./xgboost_model/deploy_selector_xgb_suite.py predict --perf_csv test_random.csv --schedule_csv test_schedules_random.csv --model_in ./xgboost_model/artifacts/gpu/xgb_model_random --alpha 0.2

 3) Validate with test CSV (Pattern x3 split):
    python ./xgboost_model/deploy_selector_xgb_suite.py predict --perf_csv test_x3.csv --schedule_csv test_schedules_x3.csv --model_in ./xgboost_model/artifacts/gpu/xgb_model_x3 --alpha 0.2

 4) Validate with Score/Rank model:
    python ./xgboost_model/deploy_selector_xgb_suite.py predict --perf_csv test_x3.csv --schedule_csv test_schedules_x3.csv --model_in ./xgboost_model/artifacts/gpu/xgb_model_score --alpha 0.2
"""

import hashlib
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

# ---------- Constants ----------
WINDOW_SEC = 30.0
ASSUME_WAIT_MS = 0.0
MODEL_HASH_BUCKETS = 128  # 모델 이름을 식별하기 위한 해싱 버킷 크기
PAIR_HASH_BUCKETS = 512


def _lazy_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception as e:
        raise RuntimeError(f"xgboost required. {e}")


def _norm_exec(dev: str) -> str:
    d = str(dev).strip().lower()
    if d in ("cpu",): return "CPU"
    if d in ("gpu", "apple-gpu", "coreml-gpu"): return "GPU"
    return dev.upper()


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            return yaml.safe_load(txt) or {}
        except Exception:
            pass
    try:
        return json.loads(txt) or {}
    except Exception:
        pass
    return {}


def _iter_combos_from_schedule(schedule: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    if "combinations" in schedule:
        combos = schedule["combinations"]
        if isinstance(combos, list):
            for c in combos:
                name = c.get("combination") or c.get("name") or "unknown"
                out.append((str(name), c))
        elif isinstance(combos, dict):
            for name, c in combos.items():
                c = dict(c);
                c.setdefault("combination", name)
                out.append((str(name), c))
        if out: return out

    if isinstance(schedule, dict):
        picks = [(k, v) for k, v in schedule.items()
                 if isinstance(v, dict) and str(k).lower().startswith("combination")]
        if picks:
            for name, blob in picks: out.append((str(name), blob))
            return out

    name = schedule.get("combination") or schedule.get("name") or "unknown"
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
                        if k not in ("model", "execution"): row[k] = v[k]
                    rows.append(row)
            if rows: return rows
    return []


def _index_schedules(schedule_dir: Path) -> Dict[str, Dict[str, Any]]:
    index = {}
    if not schedule_dir.exists(): return index
    for p in schedule_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}:
            try:
                index[p.name.lower()] = _load_yaml_or_json(p)
            except:
                pass
    return index


def _find_schedule(index, hint):
    if not hint: return None
    return index.get(Path(hint).name.lower())


def _build_infps_lookup(schedule_doc, combination_name):
    infps_map = {}
    try:
        combos = _iter_combos_from_schedule(schedule_doc)
        target_blob = None
        for name, blob in combos:
            if str(name) == str(combination_name):
                target_blob = blob;
                break
        if target_blob is None and len(combos) == 1: target_blob = combos[0][1]
        if target_blob:
            for r in _rows_from_combo_struct(target_blob):
                m = r.get("model")
                dev = _norm_exec(r.get("execution", ""))
                # [수정] infps 또는 intps를 가져옴 (intps는 infps와 동일하게 취급)
                val = r.get("infps")
                if val is None:
                    val = r.get("intps")
                
                if val is None:
                    print(f"Error: Neither 'infps' nor 'intps' found for model '{m}' in combination '{combination_name}'.")
                    sys.exit(1)

                if m and dev and val is not None:
                    infps_map[(m, dev)] = float(val)
    except:
        pass
    return infps_map


def _get_model_features(model_name: str) -> Dict[str, float]:
    """
    [FIX] 모델 이름을 피처로 변환 (Stable Hashing).
    Python의 내장 hash() 대신 md5 등을 사용하여 실행 시마다 고정된 값을 보장함.
    """
    feats = {}
    if not model_name:
        return feats

    # [수정] MD5를 사용하여 항상 같은 문자열에 대해 같은 정수값을 얻음
    enc = model_name.lower().encode("utf-8")
    fingerprint = int(hashlib.md5(enc).hexdigest(), 16)

    h_val = fingerprint % MODEL_HASH_BUCKETS

    feats[f"model_hash_{h_val}"] = 1.0
    return feats


def _pair_bucket(h1: int, h2: int, dev: str) -> int:
    a, b = sorted([h1, h2])
    key = f"{a}_{b}_{dev}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % PAIR_HASH_BUCKETS


# ---------- Feature Engineering (Fixed) ----------

def get_group_id(window: Dict[str, Any]) -> str:
    """
    Generate a stable group ID for ranking.
    Workloads from the same schedule file and with the same model composition
    should be ranked against each other.
    """
    s_name = window.get("schedule_file") or window.get("schedule file") or "unknown"
    models_blob = window.get("models", {})
    model_list = sorted([str(v.get("model", "")) for v in models_blob.values() if v.get("model")])
    model_composition = ",".join(model_list)
    
    # Include input rate to distinguish scenarios with same models but different rates
    # Input rate is often stored in 'input_fps' or can be derived from 'infps' of models
    rates = []
    for v in models_blob.values():
        r = v.get("infps") or v.get("intps") or 0.0
        rates.append(str(r))
    rate_str = ",".join(sorted(rates))
    
    return f"{s_name}_{model_composition}_{rate_str}"


def featurize_window(window: Dict[str, Any], infps_map=None) -> Tuple[
    Dict[str, float], Tuple[float, float], Dict[str, Any]]:
    models = window.get("models", {})
    per_view_rows = []
    cpu_items = []
    gpu_items = []
    npu0_items = []
    npu1_items = []

    for k, view in models.items():
        k_l = str(k).lower()
        if not (k_l.startswith("view") or k_l.startswith("headless")): continue

        model_name = view.get("model")
        exec_dev_raw = view.get("execution")
        if not model_name or not exec_dev_raw: continue

        exec_dev = _norm_exec(exec_dev_raw)

        row = {}

        # [FIX 1] 실제 측정값(throughput_fps 등)을 입력 피처에서 제거함!

        # [FIX 2] 모델 식별 정보 추가
        model_feats = _get_model_features(model_name)
        row.update(model_feats)

        # Device Flags
        exec_cpu = 1.0 if exec_dev == "CPU" else 0.0
        exec_gpu = 1.0 if exec_dev == "GPU" else 0.0
        exec_npu0 = 1.0 if exec_dev == "NPU0" else 0.0
        exec_npu1 = 1.0 if exec_dev == "NPU1" else 0.0

        # [NEW] 결합 피처 (model_hash * device)
        # Sparse generation: only create features for the actual hash bucket
        h_val = int(hashlib.md5(model_name.lower().encode("utf-8")).hexdigest(), 16) % MODEL_HASH_BUCKETS
        row[f"model_hash_{h_val}_on_cpu"] = 1.0 * exec_cpu
        row[f"model_hash_{h_val}_on_gpu"] = 1.0 * exec_gpu
        row[f"model_hash_{h_val}_on_npu0"] = 1.0 * exec_npu0
        row[f"model_hash_{h_val}_on_npu1"] = 1.0 * exec_npu1

        row["view.exec_cpu"] = exec_cpu
        row["view.exec_gpu"] = exec_gpu
        row["view.exec_npu0"] = exec_npu0
        row["view.exec_npu1"] = exec_npu1

        # Planned FPS (Demand)
        infps_val = 0.0
        if infps_map:
            infps_val = float(infps_map.get((model_name, exec_dev), 0.0))

        if exec_dev == "CPU": cpu_items.append((h_val, infps_val))
        if exec_dev == "GPU": gpu_items.append((h_val, infps_val))
        if exec_dev == "NPU0": npu0_items.append((h_val, infps_val))
        if exec_dev == "NPU1": npu1_items.append((h_val, infps_val))

        row["view.infps"] = infps_val
        row["view.infps_on_cpu"] = infps_val * exec_cpu
        row["view.infps_on_gpu"] = infps_val * exec_gpu
        row["view.infps_on_npu0"] = infps_val * exec_npu0
        row["view.infps_on_npu1"] = infps_val * exec_npu1

        per_view_rows.append(row)

    X = {}
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

        # Add pairwise features directly into X
        for i in range(len(cpu_items)):
            for j in range(i + 1, len(cpu_items)):
                h1, f1 = cpu_items[i]
                h2, f2 = cpu_items[j]
                b = _pair_bucket(h1, h2, "cpu")
                w = f1 + f2
                X[f"pairhash_{b}_on_cpu"] = X.get(f"pairhash_{b}_on_cpu", 0.0) + w

        for i in range(len(gpu_items)):
            for j in range(i + 1, len(gpu_items)):
                h1, f1 = gpu_items[i]
                h2, f2 = gpu_items[j]
                b = _pair_bucket(h1, h2, "gpu")
                w = f1 + f2
                X[f"pairhash_{b}_on_gpu"] = X.get(f"pairhash_{b}_on_gpu", 0.0) + w

        for i in range(len(npu0_items)):
            for j in range(i + 1, len(npu0_items)):
                h1, f1 = npu0_items[i]
                h2, f2 = npu0_items[j]
                b = _pair_bucket(h1, h2, "npu0")
                w = f1 + f2
                X[f"pairhash_{b}_on_npu0"] = X.get(f"pairhash_{b}_on_npu0", 0.0) + w

        for i in range(len(npu1_items)):
            for j in range(i + 1, len(npu1_items)):
                h1, f1 = npu1_items[i]
                h2, f2 = npu1_items[j]
                b = _pair_bucket(h1, h2, "npu1")
                w = f1 + f2
                X[f"pairhash_{b}_on_npu1"] = X.get(f"pairhash_{b}_on_npu1", 0.0) + w
    else:
        X["views.count.views"] = 0.0

    # Targets
    y1 = float(window.get("derived", {}).get("throughput_norm", np.nan))
    y2 = float(window.get("derived", {}).get("drop_rate_norm", np.nan))

    meta = {
        "timestamp": window.get("timestamp"),
        "combination": window.get("combination"),
    }
    return X, (y1, y2), meta


def featurize_from_combo(combo_blob: Dict[str, Any]) -> pd.DataFrame:
    views = _rows_from_combo_struct(combo_blob)
    rows = []
    cpu_items = []
    gpu_items = []
    npu0_items = []
    npu1_items = []

    for v in views:
        m = v.get("model")
        dev = _norm_exec(v.get("execution", ""))
        if not m or not dev: continue

        # [수정] infps 또는 intps를 가져옴 (intps는 infps와 동일하게 취급)
        fps_val = v.get("infps")
        if fps_val is None:
            fps_val = v.get("intps")

        if fps_val is None:
            print(f"Error: Neither 'infps' nor 'intps' found for model '{m}' in the schedule.")
            sys.exit(1)

        fps = float(fps_val) if fps_val is not None else 0.0

        r = {}
        # [FIX] 추론 시에도 모델 식별 정보 사용
        model_feats = _get_model_features(m)
        r.update(model_feats)

        exec_cpu = 1.0 if dev == "CPU" else 0.0
        exec_gpu = 1.0 if dev == "GPU" else 0.0
        exec_npu0 = 1.0 if dev == "NPU0" else 0.0
        exec_npu1 = 1.0 if dev == "NPU1" else 0.0

        # [NEW] 결합 피처 (model_hash * device)
        # Sparse generation: only create features for the actual hash bucket
        h_val = int(hashlib.md5(m.lower().encode("utf-8")).hexdigest(), 16) % MODEL_HASH_BUCKETS
        r[f"model_hash_{h_val}_on_cpu"] = 1.0 * exec_cpu
        r[f"model_hash_{h_val}_on_gpu"] = 1.0 * exec_gpu
        r[f"model_hash_{h_val}_on_npu0"] = 1.0 * exec_npu0
        r[f"model_hash_{h_val}_on_npu1"] = 1.0 * exec_npu1

        if dev == "CPU": cpu_items.append((h_val, fps))
        if dev == "GPU": gpu_items.append((h_val, fps))
        if dev == "NPU0": npu0_items.append((h_val, fps))
        if dev == "NPU1": npu1_items.append((h_val, fps))

        r["view.exec_cpu"] = exec_cpu
        r["view.exec_gpu"] = exec_gpu
        r["view.exec_npu0"] = exec_npu0
        r["view.exec_npu1"] = exec_npu1

        r["view.infps"] = fps
        r["view.infps_on_cpu"] = fps * exec_cpu
        r["view.infps_on_gpu"] = fps * exec_gpu
        r["view.infps_on_npu0"] = fps * exec_npu0
        r["view.infps_on_npu1"] = fps * exec_npu1

        rows.append(r)

    df = pd.DataFrame(rows)
    X = {}
    if not df.empty:
        for agg_name, s in {
            "sum": df.sum(numeric_only=True),
            "mean": df.mean(numeric_only=True),
            "max": df.max(numeric_only=True),
        }.items():
            for col, val in s.items():
                X[f"views.{agg_name}.{col}"] = float(val)
        X["views.count.views"] = float(len(df))

        # Add pairwise features directly into X
        for i in range(len(cpu_items)):
            for j in range(i + 1, len(cpu_items)):
                h1, f1 = cpu_items[i]
                h2, f2 = cpu_items[j]
                b = _pair_bucket(h1, h2, "cpu")
                w = f1 + f2
                X[f"pairhash_{b}_on_cpu"] = X.get(f"pairhash_{b}_on_cpu", 0.0) + w

        for i in range(len(gpu_items)):
            for j in range(i + 1, len(gpu_items)):
                h1, f1 = gpu_items[i]
                h2, f2 = gpu_items[j]
                b = _pair_bucket(h1, h2, "gpu")
                w = f1 + f2
                X[f"pairhash_{b}_on_gpu"] = X.get(f"pairhash_{b}_on_gpu", 0.0) + w

        for i in range(len(npu0_items)):
            for j in range(i + 1, len(npu0_items)):
                h1, f1 = npu0_items[i]
                h2, f2 = npu0_items[j]
                b = _pair_bucket(h1, h2, "npu0")
                w = f1 + f2
                X[f"pairhash_{b}_on_npu0"] = X.get(f"pairhash_{b}_on_npu0", 0.0) + w

        for i in range(len(npu1_items)):
            for j in range(i + 1, len(npu1_items)):
                h1, f1 = npu1_items[i]
                h2, f2 = npu1_items[j]
                b = _pair_bucket(h1, h2, "npu1")
                w = f1 + f2
                X[f"pairhash_{b}_on_npu1"] = X.get(f"pairhash_{b}_on_npu1", 0.0) + w
    else:
        X["views.count.views"] = 0.0

    return pd.DataFrame([X]).fillna(0.0)


# ---------- Builder & Trainer ----------

def _index_schedules_from_csv(csv_path: Path, names: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        name = str(row["schedule_name"]).lower()
        if names is not None and name not in names:
            continue
        content = str(row["content"])
        if yaml is not None:
            try:
                out[name] = yaml.safe_load(content) or {}
                continue
            except Exception:
                pass
        try:
            out[name] = json.loads(content) or {}
        except Exception:
            pass
    return out


def build_dataset_from_csv(csv_path: Path, schedule_dir: Optional[Path] = None, schedule_csv: Optional[Path] = None, is_constrained: bool = False):
    needed_names = set()
    df_csv = pd.read_csv(csv_path)
    for _, row in df_csv.iterrows():
        try:
            if "json_content" in df_csv.columns:
                w = json.loads(row["json_content"])
            else:
                w = json.loads(row[0])
            s_name = w.get("schedule_file") or w.get("schedule file")
            if s_name:
                needed_names.add(str(Path(s_name).name).lower())
        except Exception:
            pass

    if schedule_csv:
        sched_index = _index_schedules_from_csv(schedule_csv, names=needed_names)
    elif schedule_dir:
        sched_index = _index_schedules(schedule_dir)
    else:
        sched_index = {}

    X_all, Y_all, M_all = [], [], []

    for _, row in df_csv.iterrows():
        try:
            # Try to load 'json_content' column
            if "json_content" in df_csv.columns:
                w = json.loads(row["json_content"])
            else:
                # Fallback to the first column if no header matches
                w = json.loads(row[0])
            
            s_name = w.get("schedule_file") or w.get("schedule file")
            s_doc = _find_schedule(sched_index, s_name)
            c_name = w.get("combination")

            # if is_constrained:
            #     models_count = len(w.get("models", {}))
            #     if models_count < 3:
            #         continue

            infps_map = None
            if s_doc and c_name:
                infps_map = _build_infps_lookup(s_doc, c_name)

            X, (y1, y2), meta = featurize_window(w, infps_map)
            if math.isnan(y1) or math.isnan(y2): continue

            X_all.append(X)
            Y_all.append({"y1": y1, "y2": y2})
            meta["group_id"] = get_group_id(w)
            M_all.append(meta)
        except Exception as e:
            print(f"[WARN] CSV row error: {e}")

    if not X_all: raise RuntimeError("No valid data rows found in CSV.")
    return pd.DataFrame(X_all).fillna(0.0), pd.DataFrame(Y_all), pd.DataFrame(M_all)


def build_dataset(perf_dir: Path, schedule_dir: Path, is_constrained: bool = False):
    sched_index = _index_schedules(schedule_dir)
    X_all, Y_all, M_all = [], [], []

    for path in sorted(perf_dir.rglob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            for w in blob.get("data", []):
                s_name = w.get("schedule file") or w.get("schedule_file")
                s_doc = _find_schedule(sched_index, s_name)
                c_name = w.get("combination")

                # if is_constrained:
                #     models_count = len(w.get("models", {}))
                #     if models_count < 3:
                #         continue

                infps_map = None
                if s_doc and c_name:
                    infps_map = _build_infps_lookup(s_doc, c_name)

                X, (y1, y2), meta = featurize_window(w, infps_map)
                if math.isnan(y1) or math.isnan(y2): continue

                X_all.append(X)
                Y_all.append({"y1": y1, "y2": y2})
                meta["group_id"] = get_group_id(w)
                M_all.append(meta)
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")

    if not X_all: raise RuntimeError("No valid data rows found.")
    return pd.DataFrame(X_all).fillna(0.0), pd.DataFrame(Y_all), pd.DataFrame(M_all)


def train_score(X, Y, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    from sklearn.model_selection import GridSearchCV

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Calculate score target
    y_score = Y["y1"] - alpha * Y["y2"]

    # Hyperparameters for GridSearchCV
    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8],
    }

    print(f"Starting 3-fold Cross-Validation with GridSearchCV for Score Model (samples: {len(X)})...")

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )

    grid_search.fit(X, y_score)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV Score (MAE): {-grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    
    model.save_model(str(prefix) + "_score.json")
    # Meta info
    meta = {
        "mode": "score",
        "alpha": alpha,
        "features": cols,
        "best_params": grid_search.best_params_
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta, indent=2))


def train_rank(X, Y, M, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    from sklearn.model_selection import GroupKFold, GridSearchCV

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Calculate score target for ranking
    y_score = Y["y1"] - alpha * Y["y2"]
    
    df = X.copy()
    df["y_score"] = y_score
    df["group_id"] = M["group_id"]

    # 그룹별 relevance 계산 (0~31)
    def compute_group_relevance(group):
        if len(group) <= 1:
            group["target"] = 31
            return group
        ranks = group["y_score"].rank(method='min', ascending=True) - 1
        max_rank = ranks.max()
        if max_rank == 0:
            group["target"] = 31
        else:
            group["target"] = (ranks * 31.0 / max_rank).round().astype(int)
        return group

    print("Computing group relevance for ranking...")
    df = df.groupby("group_id", group_keys=False).apply(compute_group_relevance)
    y_relevance = df["target"]
    groups = df["group_id"]

    # Hyperparameters for GridSearchCV
    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6],
    }

    print(f"Starting 3-fold Cross-Validation with GridSearchCV for Rank Model (samples: {len(X)}, groups: {len(df['group_id'].unique())})...")

    base_model = xgb.XGBRanker(
        objective="rank:pairwise",
        random_state=42,
        n_jobs=-1
    )

    # GridSearchCV for XGBRanker needs group information in fit()
    # Scikit-learn's GridSearchCV with GroupKFold
    cv = GroupKFold(n_splits=3)

    # Custom scoring that handles groups if needed
    # But usually scikit-learn metrics don't take groups unless specified.
    # We'll use ndcg_score from sklearn as a custom scorer if needed, 
    # but GridSearchCV will pass y_true, y_pred.
    from sklearn.metrics import make_scorer, ndcg_score
    
    def my_ndcg_scorer(y_true, y_pred):
        # This is a bit tricky because ndcg_score expects [ [rel1, rel2, ...] ]
        # and we have a flat array across groups.
        # For simplicity, we'll return the mean of scores if we can't easily group here.
        # Or we can just use None and let XGBRanker's default scoring work if it's integrated.
        # Actually, let's try to provide fit_params to GridSearchCV.
        return 0.0 # Placeholder

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=None, 
        verbose=1
    )

    # We need to pass groups to fit(), and GridSearchCV passes it to cv.split().
    # However, XGBRanker.fit() also needs groups or qid.
    # We can pass them via fit_params.
    # But groups should be for the training subset of each fold.
    # This is why using GridSearchCV with XGBRanker is hard.
    
    # Alternative: Use XGBRegressor for ranking with a custom objective or just score regression.
    # But the user specifically asked for rank mode.
    
    # Let's fix it by using a custom loop or providing qid.
    df["qid"] = groups.factorize()[0]
    # qid must be sorted for XGBRanker
    df = df.sort_values("qid")
    X = X.loc[df.index]
    y_relevance = y_relevance.loc[df.index]
    qid = df["qid"]

    grid_search.fit(X, y_relevance, groups=qid, qid=qid)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV Score (NDCG): {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    
    model.save_model(str(prefix) + "_rank.json")
    # Meta info
    meta = {
        "mode": "rank",
        "alpha": alpha,
        "features": cols,
        "best_params": grid_search.best_params_
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta, indent=2))


def train_double(X, Y, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    from sklearn.model_selection import GridSearchCV

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Hyperparameters for GridSearchCV
    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0],
    }

    print(f"Starting 3-fold Cross-Validation with GridSearchCV for Double Model (samples: {len(X)})...")

    # 1. Train Throughput Model (y1)
    print("Optimizing Throughput (y1) model...")
    base_model_y1 = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    grid_y1 = GridSearchCV(
        estimator=base_model_y1,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid_y1.fit(X, Y["y1"])
    print(f"Best y1 params: {grid_y1.best_params_}, MAE: {-grid_y1.best_score_:.4f}")
    model_y1 = grid_y1.best_estimator_
    model_y1.save_model(str(prefix) + "_y1.json")

    # 2. Train Drop Rate Model (y2) with Weights
    print("Optimizing Drop Rate (y2) model...")
    # Since GridSearchCV doesn't easily support sample_weight per fold in some versions,
    # we can pass it to fit().
    weights = Y["y2"].apply(lambda x: 10.0 if x > 0.01 else 1.0).values
    
    base_model_y2 = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    grid_y2 = GridSearchCV(
        estimator=base_model_y2,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid_y2.fit(X, Y["y2"], sample_weight=weights)
    print(f"Best y2 params: {grid_y2.best_params_}, MAE: {-grid_y2.best_score_:.4f}")
    model_y2 = grid_y2.best_estimator_
    model_y2.save_model(str(prefix) + "_y2.json")

    # Meta info
    meta = {
        "mode": "double",
        "alpha": alpha,
        "features": cols,
        "best_params_y1": grid_y1.best_params_,
        "best_params_y2": grid_y2.best_params_
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta, indent=2))
    # For backward compatibility
    Path(str(prefix) + "_features.json").write_text(json.dumps(cols))


def train_two_targets(X, Y, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    from sklearn.model_selection import GridSearchCV

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Hyperparameters for GridSearchCV
    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6],
    }

    print(f"Starting 3-fold Cross-Validation with GridSearchCV for Two-Target Model (samples: {len(X)})...")

    # 1. Train Throughput Model (y1)
    print("Optimizing Throughput (y1) model...")
    base_model_y1 = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    grid_y1 = GridSearchCV(
        estimator=base_model_y1,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid_y1.fit(X, Y["y1"])
    print(f"Best y1 params: {grid_y1.best_params_}, MAE: {-grid_y1.best_score_:.4f}")
    model_y1 = grid_y1.best_estimator_
    model_y1.save_model(str(prefix) + "_y1.json")

    # 2. Train Drop Rate Model (y2) with Weights
    print("Optimizing Drop Rate (y2) model...")
    weights = Y["y2"].apply(lambda x: 10.0 if x > 0.01 else 1.0).values
    
    base_model_y2 = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    grid_y2 = GridSearchCV(
        estimator=base_model_y2,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid_y2.fit(X, Y["y2"], sample_weight=weights)
    print(f"Best y2 params: {grid_y2.best_params_}, MAE: {-grid_y2.best_score_:.4f}")
    model_y2 = grid_y2.best_estimator_
    model_y2.save_model(str(prefix) + "_y2.json")

    # Meta info
    meta = {
        "mode": "two_target",
        "alpha": alpha,
        "features": cols,
        "best_params_y1": grid_y1.best_params_,
        "best_params_y2": grid_y2.best_params_
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta, indent=2))
    # For backward compatibility
    Path(str(prefix) + "_features.json").write_text(json.dumps(cols))


def load_models(prefix):
    xgb = _lazy_import_xgb()
    
    meta_path = Path(str(prefix) + "_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        mode = meta.get("mode", "two_target")
        alpha = meta.get("alpha", 0.2)
        cols = meta.get("features")
    else:
        # Legacy mode
        mode = "two_target"
        alpha = 0.2
        cols = json.loads(Path(str(prefix) + "_features.json").read_text())

    if mode == "score":
        m_score = xgb.XGBRegressor()
        m_score.load_model(str(prefix) + "_score.json")
        return m_score, None, cols, mode, alpha
    elif mode == "rank":
        m_rank = xgb.XGBRanker()
        m_rank.load_model(str(prefix) + "_rank.json")
        return m_rank, None, cols, mode, alpha
    elif mode == "double":
        m1 = xgb.XGBRegressor()
        m1.load_model(str(prefix) + "_y1.json")
        m2 = xgb.XGBRegressor()
        m2.load_model(str(prefix) + "_y2.json")
        return m1, m2, cols, mode, alpha
    else:
        m1 = xgb.XGBRegressor()
        m1.load_model(str(prefix) + "_y1.json")
        m2 = xgb.XGBRegressor()
        m2.load_model(str(prefix) + "_y2.json")
        return m1, m2, cols, mode, alpha


def plot_multi_scatter(df: pd.DataFrame, output_path: Path, title: str, alpha: float):
    """
    Generate a PDF with three scatter plots: norm_throughput, norm_drop_rate, and score.
    """
    if df.empty:
        return

    # Set font to match LaTeX appearance (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle(title, fontsize=20)

    # 1. Throughput Plot
    if "actual_T_norm" in df.columns and "pred_T_norm" in df.columns:
        # Convert empty strings to NaN for plotting
        actual_t = pd.to_numeric(df["actual_T_norm"], errors='coerce')
        pred_t = pd.to_numeric(df["pred_T_norm"], errors='coerce')
        
        axes[0].scatter(actual_t, pred_t, alpha=0.5, color='blue')
        max_val = max(actual_t.max(), pred_t.max()) if not actual_t.isna().all() else 1.0
        min_val = min(actual_t.min(), pred_t.min()) if not actual_t.isna().all() else 0.0
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0].set_xlabel("Actual Norm Throughput", fontsize=18)
        axes[0].set_ylabel("Predicted Norm Throughput", fontsize=18)
        axes[0].set_title("Normalized Throughput", fontsize=20)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='both', which='major', labelsize=16)

    # 2. Drop Rate Plot
    if "actual_D_norm" in df.columns and "pred_D_norm" in df.columns:
        actual_d = pd.to_numeric(df["actual_D_norm"], errors='coerce')
        pred_d = pd.to_numeric(df["pred_D_norm"], errors='coerce')
        
        axes[1].scatter(actual_d, pred_d, alpha=0.5, color='green')
        max_val = max(actual_d.max(), pred_d.max()) if not actual_d.isna().all() else 1.0
        min_val = min(actual_d.min(), pred_d.min()) if not actual_d.isna().all() else 0.0
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1].set_xlabel("Actual Norm Drop Rate", fontsize=18)
        axes[1].set_ylabel("Predicted Norm Drop Rate", fontsize=18)
        axes[1].set_title("Normalized Drop Rate", fontsize=20)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='both', which='major', labelsize=16)

    # 3. Score Plot
    axes[2].scatter(df["actual_score"], df["pred_score"], alpha=0.5, color='purple')
    max_val = max(df["actual_score"].max(), df["pred_score"].max())
    min_val = min(df["actual_score"].min(), df["pred_score"].min())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[2].set_xlabel("Actual Score", fontsize=18)
    axes[2].set_ylabel("Predicted Score", fontsize=18)
    axes[2].set_title(f"Score (alpha={alpha})", fontsize=20)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Multi-scatter plot saved to: {output_path}")


def plot_score_scatter(df: pd.DataFrame, output_path: Path, title: str):
    """
    Generate a scatter plot of actual vs predicted scores.
    """
    if df.empty:
        return

    # Set font to match LaTeX appearance (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.figure(figsize=(8, 8))
    plt.scatter(df["actual_score"], df["pred_score"], alpha=0.5, color='blue')
    
    # 45-degree line
    max_val = max(df["actual_score"].max(), df["pred_score"].max())
    min_val = min(df["actual_score"].min(), df["pred_score"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel("Actual Score", fontsize=18)
    plt.ylabel("Predicted Score", fontsize=18)
    # plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Scatter plot saved to: {output_path}")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--perf_dir")
    tr.add_argument("--perf_csv")
    tr.add_argument("--schedule_dir")
    tr.add_argument("--schedule_csv")
    tr.add_argument("--model_out", required=True)
    tr.add_argument("--dump_csv", default="")
    tr.add_argument("--train_mode", choices=["two_target", "double", "score", "rank"], default="rank")
    tr.add_argument("--alpha", type=float, default=0.2)
    tr.add_argument("--clip_pred_score", action="store_true", default=True)
    tr.add_argument("--no_clip_pred_score", action="store_false", dest="clip_pred_score")

    pr = sub.add_parser("predict")
    pr.add_argument("--schedule_dir")
    pr.add_argument("--schedule_csv")
    pr.add_argument("--perf_csv")
    pr.add_argument("--model_in", required=True)
    pr.add_argument("--out_dir", default="xgboost_model/prediction_result")
    pr.add_argument("--alpha", type=float, default=None, help="If not set, uses alpha from model meta")
    pr.add_argument("--topk", type=int, default=5)
    pr.add_argument("--repeats", type=int, default=1)
    pr.add_argument("--clip_pred_score", action="store_true", default=True)
    pr.add_argument("--no_clip_pred_score", action="store_false", dest="clip_pred_score")

    args = ap.parse_args()

    if args.cmd == "train":
        # 모델 출력 경로에 모드명을 자동으로 포함 (기본값인 경우나 명시적으로 포함되지 않은 경우)
        model_out_path = Path(args.model_out)
        if args.train_mode not in model_out_path.name:
            # 확장자가 없는 형태일 것이므로 이름을 수정
            new_name = f"{model_out_path.name}_{args.train_mode}"
            model_out_path = model_out_path.with_name(new_name)
        
        is_constrained = "xgb_model_x3" in str(model_out_path) or "xgb_model_random" in str(model_out_path)
        if args.perf_csv:
            p_csv = Path(args.perf_csv)
            if not p_csv.exists():
                alt_p = Path("xgboost_model/dataset") / p_csv.name
                if alt_p.exists():
                    p_csv = alt_p
            
            s_dir = Path(args.schedule_dir) if args.schedule_dir else None
            
            s_csv = Path(args.schedule_csv) if args.schedule_csv else None
            if s_csv and not s_csv.exists():
                alt_s = Path("xgboost_model/dataset") / s_csv.name
                if alt_s.exists():
                    s_csv = alt_s
                    
            X, Y, M = build_dataset_from_csv(p_csv, s_dir, s_csv, is_constrained=is_constrained)
        elif args.perf_dir:
            X, Y, M = build_dataset(Path(args.perf_dir), Path(args.schedule_dir), is_constrained=is_constrained)
        else:
            print("Error: Either --perf_dir or --perf_csv must be provided for train command.")
            sys.exit(1)
        
        if args.train_mode == "score":
            train_score(X, Y, model_out_path, alpha=args.alpha)
        elif args.train_mode == "rank":
            train_rank(X, Y, M, model_out_path, alpha=args.alpha)
        elif args.train_mode == "double":
            train_double(X, Y, model_out_path, alpha=args.alpha)
        else:
            train_two_targets(X, Y, model_out_path, alpha=args.alpha)
        print(f"Training Done. Model saved with prefix: {model_out_path}")

    elif args.cmd == "predict":
        is_constrained = "xgb_model_x3" in str(args.model_in) or "xgb_model_random" in str(args.model_in)
        b1, b2, feats, mode, model_alpha = load_models(Path(args.model_in))
        
        # Override alpha if provided in CLI
        alpha = args.alpha if args.alpha is not None else model_alpha
        
        xgb = _lazy_import_xgb()

    # [추가] CSV 파일이 주어지면 해당 파일의 데이터에 대해 예측 수행
        if args.perf_csv:
            p_csv_path = Path(args.perf_csv)
            # 만약 지정된 경로에 파일이 없고 xgboost_model/dataset 아래에 있다면 해당 경로 사용
            if not p_csv_path.exists():
                alt_path = Path("xgboost_model/dataset") / p_csv_path.name
                if alt_path.exists():
                    p_csv_path = alt_path

            if args.schedule_csv:
                s_csv_path = Path(args.schedule_csv)
                if not s_csv_path.exists():
                    alt_s_path = Path("xgboost_model/dataset") / s_csv_path.name
                    if alt_s_path.exists():
                        s_csv_path = alt_s_path
                sched_index = _index_schedules_from_csv(s_csv_path)
            else:
                sched_index = _index_schedules(Path(args.schedule_dir))
            
            df_csv = pd.read_csv(p_csv_path)
            
            # [추가] 추론 결과를 저장할 리스트
            detailed_results = []
            
            # Group by schedule_file to find best in each context (for summary)
            scenario_data = {} # key: (schedule_file, timestamp), value: list of windows
            
            for index, row_csv in df_csv.iterrows():
                try:
                    if "json_content" in df_csv.columns:
                        w = json.loads(row_csv["json_content"])
                    else:
                        w = json.loads(row_csv[0])
                    
                    # if is_constrained:
                    #     models_count = len(w.get("models", {}))
                    #     if models_count < 3:
                    #         continue

                    s_name = w.get("schedule_file") or w.get("schedule file")
                    ts = w.get("timestamp")
                    c_name = w.get("combination")
                    
                    # Scenario grouping for summary
                    # Use get_group_id to group by context
                    key = get_group_id(w)
                    if key not in scenario_data:
                        scenario_data[key] = []
                    scenario_data[key].append(w)

                    # Prediction for this row
                    s_doc = _find_schedule(sched_index, s_name)
                    infps_map = _build_infps_lookup(s_doc, c_name) if s_doc else None
                    
                    # Ground Truth
                    y1_actual = float(w.get("derived", {}).get("throughput_norm", np.nan))
                    y2_actual = float(w.get("derived", {}).get("drop_rate_norm", np.nan))
                    actual_score_raw = y1_actual - alpha * y2_actual
                    actual_score_display = round(actual_score_raw, 2)

                    # Prediction
                    X_dict, _, _ = featurize_window(w, infps_map)
                    df_X = pd.DataFrame([X_dict])
                    df_X = df_X.reindex(columns=feats, fill_value=0.0)
                    
                    if mode == "score":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        # [Modified] Use raw prediction from model b1 without clipping
                        pred_score_raw = float(b1.predict(df_X)[0])
                    elif mode == "rank":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        # XGBRanker predict returns scores that represent relative ranking
                        pred_score_raw = float(b1.predict(df_X)[0])
                    else:
                        y1_pred = float(b1.predict(df_X)[0])
                        y2_pred = float(b2.predict(df_X)[0])
                        # [Modified] Calculate combined score
                        pred_score_raw = y1_pred - alpha * y2_pred
                    
                    # [Modified] Apply clipping only if enabled via --clip_pred_score
                    # For rank mode, clipping might not make sense as it's relative, 
                    # but we'll follow the same logic. Usually, Ranker outputs are not in [0,1].
                    if args.clip_pred_score and mode != "rank":
                        pred_score_raw = max(0.0, min(1.0, pred_score_raw))

                    pred_score_display = round(pred_score_raw, 2)

                    detailed_results.append({
                        "schedule_file": s_name,
                        "timestamp": ts,
                        "combination": c_name,
                        "actual_T_norm": round(y1_actual, 4),
                        "actual_D_norm": round(y2_actual, 4),
                        "actual_score": actual_score_display,
                        "pred_T_norm": round(y1_pred, 4) if not np.isnan(y1_pred) else "",
                        "pred_D_norm": round(y2_pred, 4) if not np.isnan(y2_pred) else "",
                        "pred_score": pred_score_display,
                        "diff_score": round(abs(actual_score_display - pred_score_display), 4)
                    })
                    
                except Exception as e:
                    print(f"[WARN] Predict CSV row load error at index {index}: {e}")

            # CSV 저장
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            top1_hits = 0
            top5_hits = 0
            score_gaps = []
            y1_errs = []
            y2_errs = []
            total_scenarios = 0

            # For summary table (similar to fix_prediction_summary.py)
            scenario_summary_data = []

            # summary calculation using grouped scenario_data
            for group_id, windows in scenario_data.items():
                total_scenarios += 1
                scenario_results = []
                
                # To determine model count for this scenario
                model_count = 0

                for w in windows:
                    s_name = w.get("schedule_file") or w.get("schedule file")
                    s_doc = _find_schedule(sched_index, s_name)
                    if not s_doc: continue
                    
                    if model_count == 0:
                        model_count = len(w.get("models", {}))

                    c_name = w.get("combination")
                    infps_map = _build_infps_lookup(s_doc, c_name)
                    
                    # Ground Truth
                    y1_actual = float(w.get("derived", {}).get("throughput_norm", np.nan))
                    y2_actual = float(w.get("derived", {}).get("drop_rate_norm", np.nan))
                    actual_score_raw = y1_actual - alpha * y2_actual

                    # Prediction
                    X_dict, _, _ = featurize_window(w, infps_map)
                    df_X = pd.DataFrame([X_dict])
                    df_X = df_X.reindex(columns=feats, fill_value=0.0)
                    
                    if mode == "score":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        pred_score_raw = float(b1.predict(df_X)[0])
                    elif mode == "rank":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        pred_score_raw = float(b1.predict(df_X)[0])
                    else:
                        y1_pred = float(b1.predict(df_X)[0])
                        y2_pred = float(b2.predict(df_X)[0])
                        pred_score_raw = y1_pred - alpha * y2_pred
                    
                    if args.clip_pred_score and mode != "rank":
                        pred_score_raw = max(0.0, min(1.0, pred_score_raw))
                    
                    if not np.isnan(y1_pred):
                        y1_errs.append(abs(y1_actual - y1_pred))
                    if not np.isnan(y2_pred):
                        y2_errs.append(abs(y2_actual - y2_pred))
                    
                    scenario_results.append({
                        "combination": c_name,
                        "actual_T_norm": y1_actual,
                        "actual_D_norm": y2_actual,
                        "actual_score": actual_score_raw,
                        "pred_score": pred_score_raw,
                        "schedule_file": s_name,
                        "timestamp": w.get("timestamp")
                    })
                
                if not scenario_results: continue
                
                # Sort by actual score to find actual best(s)
                actual_sorted = sorted(scenario_results, key=lambda x: x["actual_score"], reverse=True)
                max_actual_score = actual_sorted[0]["actual_score"]
                actual_best_names = [r["combination"] for r in actual_sorted if math.isclose(r["actual_score"], max_actual_score, rel_tol=1e-7)]
                
                # Sort by predicted score
                pred_sorted = sorted(scenario_results, key=lambda x: x["pred_score"], reverse=True)
                max_pred_score = pred_sorted[0]["pred_score"]
                pred_best_names = [r["combination"] for r in pred_sorted if math.isclose(r["pred_score"], max_pred_score, rel_tol=1e-7)]

                pred_best_row = pred_sorted[0]
                pred_best_name_first = pred_best_row["combination"]
                pred_best_actual_score = pred_best_row["actual_score"]
                actual_best_row = actual_sorted[0]
                actual_best_score = actual_best_row["actual_score"]
                
                # Apply coloring and marking to detailed_results
                for res in detailed_results:
                    # Match by schedule_file and timestamp and combination
                    # Note: res["combination"] might already be colored if multiple scenarios share same rows (unlikely but possible)
                    # We use group_id based grouping so it should be fine.
                    for s_res in scenario_results:
                        if res["schedule_file"] == s_res["schedule_file"] and res["timestamp"] == s_res["timestamp"] and res["combination"] == s_res["combination"]:
                            c_name = s_res["combination"]
                            is_actual_best = c_name in actual_best_names
                            is_pred_best = c_name in pred_best_names
                            
                            if is_actual_best and is_pred_best:
                                res["combination"] = f"<font color='purple'>{c_name}</font>"
                            elif is_actual_best:
                                res["combination"] = f"<font color='blue'>{c_name}</font>"
                            elif is_pred_best:
                                # Add (Actual: ...) hint if predicted best is not actual best
                                actual_hint = ", ".join([f"<font color='blue'>{n}</font>" for n in actual_best_names])
                                res["combination"] = f"<font color='red'>{c_name}</font> (Actual: {actual_hint})"
                            break
                
                # 1) Top-1 Accuracy: any(Predicted Top-1) in Actual Top-1
                # (Predicted Top-1 combinations: pred_best_names, Actual Top-1 combinations: actual_best_names)
                is_top1 = any(name in actual_best_names for name in pred_best_names)
                if is_top1:
                    top1_hits += 1

                # 2) Top-5 Accuracy: any(Actual Best) in Predicted Top-5 groups
                # (Predicted Top-5 groups: pred_score_raw >= threshold of 5th pred score group)
                unique_pred_scores = sorted(list(set([r["pred_score"] for r in scenario_results])), reverse=True)
                if len(unique_pred_scores) > 0:
                    top5_pred_threshold = unique_pred_scores[min(4, len(unique_pred_scores)-1)]
                    predicted_top5_names = [r["combination"] for r in scenario_results if r["pred_score"] >= (top5_pred_threshold - 1e-7)]
                else:
                    predicted_top5_names = []
                
                is_top5 = any(name in predicted_top5_names for name in actual_best_names)
                if is_top5:
                    top5_hits += 1
                
                # [Note] Keep top5_group_scores as predicted scores for the report as per previous instructions
                if len(unique_pred_scores) > 0:
                    # Use unique_pred_scores instead of raw pred_score for display in report
                    top5_group_scores = sorted(list(set([round(r["pred_score"], 2) for r in scenario_results if r["pred_score"] >= (top5_pred_threshold - 1e-7)])), reverse=True)
                else:
                    top5_group_scores = []

                # 3) Score gap
                gap = actual_best_score - pred_best_actual_score
                score_gaps.append(max(0, gap))

                # Best-of-5: best actual outcome among the model's predicted Top-5 group
                top5_candidates = [r for r in scenario_results if r["combination"] in predicted_top5_names]
                best5_row = max(top5_candidates, key=lambda x: x["actual_score"]) if top5_candidates else pred_best_row

                scenario_summary_data.append({
                    "m_count": model_count,
                    "is_top1": is_top1,
                    "is_top5": is_top5,
                    "top5_scores": top5_group_scores,
                    "oracle_T": actual_best_row["actual_T_norm"],
                    "oracle_D": actual_best_row["actual_D_norm"],
                    "oracle_S": actual_best_row["actual_score"],
                    "pred_T": pred_best_row["actual_T_norm"],
                    "pred_D": pred_best_row["actual_D_norm"],
                    "pred_S": pred_best_row["actual_score"],
                    "best5_T": best5_row["actual_T_norm"],
                    "best5_D": best5_row["actual_D_norm"],
                    "best5_S": best5_row["actual_score"]
                })

            # [Added] Also include alpha in the filename
            csv_out_path = out_dir / f"prediction_result_{p_csv_path.stem}_{mode}_alpha_{alpha}.csv"

            # Write Summary to CSV file
            with open(csv_out_path, "w", encoding="utf-8") as f:
                f.write(f"Alpha,{alpha}\n")
                
                top1_acc = top1_hits / total_scenarios if total_scenarios > 0 else 0
                top5_acc = top5_hits / total_scenarios if total_scenarios > 0 else 0
                f.write(f"Top-1 Accuracy,{top1_acc:.4f}\n")
                f.write(f"Top-5 Accuracy,{top5_acc:.4f}\n")
                
                # Top-k Accuracy (>= 3 models)
                ge3_scenarios = [d for d in scenario_summary_data if d["m_count"] >= 3]
                if ge3_scenarios:
                    top1_ge3 = sum(1 for d in ge3_scenarios if d["is_top1"]) / len(ge3_scenarios)
                    top5_ge3 = sum(1 for d in ge3_scenarios if d["is_top5"]) / len(ge3_scenarios)
                    f.write(f"Top-1 Accuracy (>= 3 models),{top1_ge3:.4f}\n")
                    
                    # Collect all Top-5 scores from GE3 scenarios
                    all_top5_scores_ge3 = []
                    for d in ge3_scenarios:
                        all_top5_scores_ge3.extend(d["top5_scores"])
                    # Unique and sorted scores for display
                    unique_top5_scores_ge3 = sorted(list(set(all_top5_scores_ge3)), reverse=True)
                    scores_str = " ".join([str(s) for s in unique_top5_scores_ge3])
                    
                    f.write(f"Top-5 Accuracy (>= 3 models),{top5_ge3:.4f},{scores_str}\n")
                
                # Actual Best Average
                avg_oracle_T = np.mean([d["oracle_T"] for d in scenario_summary_data])
                avg_oracle_D = np.mean([d["oracle_D"] for d in scenario_summary_data])
                avg_oracle_S = np.mean([d["oracle_S"] for d in scenario_summary_data])
                f.write(f"Actual Best Average,,{avg_oracle_T:.2f},{avg_oracle_D:.2f},{avg_oracle_S:.2f}\n")

                # Best-of-5 Average (best actual outcome within model's predicted Top-5)
                avg_best5_T = np.mean([d["best5_T"] for d in scenario_summary_data])
                avg_best5_D = np.mean([d["best5_D"] for d in scenario_summary_data])
                avg_best5_S = np.mean([d["best5_S"] for d in scenario_summary_data])
                f.write(f"Best-of-5 Average,,{avg_best5_T:.2f},{avg_best5_D:.2f},{avg_best5_S:.2f}\n")

                # Averages by model count
                max_m = max(d["m_count"] for d in scenario_summary_data) if scenario_summary_data else 0
                for n in range(max_m, 2, -1):
                    subset = [d for d in scenario_summary_data if d["m_count"] >= n]
                    if not subset:
                        continue
                    
                    o_T = np.mean([d["oracle_T"] for d in subset])
                    o_D = np.mean([d["oracle_D"] for d in subset])
                    o_S = np.mean([d["oracle_S"] for d in subset])
                    f.write(f"Actual Best Average (>= {n} models),,{o_T:.2f},{o_D:.2f},{o_S:.2f}\n")
                    
                    p_T = np.mean([d["pred_T"] for d in subset])
                    p_D = np.mean([d["pred_D"] for d in subset])
                    p_S = np.mean([d["pred_S"] for d in subset])
                    f.write(f"Average (>= {n} models),,{p_T:.2f},{p_D:.2f},{p_S:.2f}\n")

                    b5_T = np.mean([d["best5_T"] for d in subset])
                    b5_D = np.mean([d["best5_D"] for d in subset])
                    b5_S = np.mean([d["best5_S"] for d in subset])
                    f.write(f"Best-of-5 Average (>= {n} models),,{b5_T:.2f},{b5_D:.2f},{b5_S:.2f}\n")
                
                # Overall Average
                avg_pred_T = np.mean([d["pred_T"] for d in scenario_summary_data])
                avg_pred_D = np.mean([d["pred_D"] for d in scenario_summary_data])
                avg_pred_S = np.mean([d["pred_S"] for d in scenario_summary_data])
                f.write(f"Average,,{avg_pred_T:.2f},{avg_pred_D:.2f},{avg_pred_S:.2f}\n")
            
            # [Modified] Prepare summary rows for each scenario (best predicted)
            summary_rows = []
            for group_id, windows in scenario_data.items():
                scenario_results = []
                for w in windows:
                    s_name = w.get("schedule_file") or w.get("schedule file")
                    s_doc = _find_schedule(sched_index, s_name)
                    if not s_doc: continue
                    c_name = w.get("combination")
                    infps_map = _build_infps_lookup(s_doc, c_name)
                    
                    y1_actual = float(w.get("derived", {}).get("throughput_norm", np.nan))
                    y2_actual = float(w.get("derived", {}).get("drop_rate_norm", np.nan))
                    actual_score_raw = y1_actual - alpha * y2_actual

                    X_dict, _, _ = featurize_window(w, infps_map)
                    df_X = pd.DataFrame([X_dict]).reindex(columns=feats, fill_value=0.0)
                    
                    if mode == "score": pred_score_raw = float(b1.predict(df_X)[0])
                    elif mode == "rank": pred_score_raw = float(b1.predict(df_X)[0])
                    else: pred_score_raw = float(b1.predict(df_X)[0]) - alpha * float(b2.predict(df_X)[0])
                    
                    if args.clip_pred_score and mode != "rank":
                        pred_score_raw = max(0.0, min(1.0, pred_score_raw))
                    
                    scenario_results.append({
                        "combination": c_name,
                        "actual_T_norm": y1_actual,
                        "actual_D_norm": y2_actual,
                        "actual_score_raw": actual_score_raw,
                        "pred_score_raw": pred_score_raw,
                        "schedule_file": s_name
                    })
                
                if not scenario_results: continue
                
                # Best predicted (Multiple if tied)
                pred_sorted = sorted(scenario_results, key=lambda x: x["pred_score_raw"], reverse=True)
                max_pred_raw = pred_sorted[0]["pred_score_raw"]
                # Use math.isclose for tie判断 with raw float
                pred_best_all = [r for r in pred_sorted if math.isclose(r["pred_score_raw"], max_pred_raw, rel_tol=1e-7)]
                
                # Use the first one for numerical metrics
                pred_best = pred_best_all[0]
                
                # Actual bests
                actual_sorted = sorted(scenario_results, key=lambda x: x["actual_score_raw"], reverse=True)
                max_actual_raw = actual_sorted[0]["actual_score_raw"]
                actual_best_names = [r["combination"] for r in actual_sorted if math.isclose(r["actual_score_raw"], max_actual_raw, rel_tol=1e-7)]
                
                # Construct best_combination string with multiple names if tied
                comb_parts = []
                seen_combs = set()
                all_pred_best_are_actual_best = True
                for pb in pred_best_all:
                    c_name = pb["combination"]
                    if c_name in seen_combs: continue
                    seen_combs.add(c_name)

                    is_actual_best = c_name in actual_best_names
                    if is_actual_best:
                        comb_parts.append(f"<font color='purple'>{c_name}</font>")
                    else:
                        comb_parts.append(f"<font color='red'>{c_name}</font>")
                        all_pred_best_are_actual_best = False
                
                best_comb_str = ", ".join(comb_parts)
                
                if not all_pred_best_are_actual_best:
                    actual_hint = ", ".join([f"<font color='blue'>{n}</font>" for n in actual_best_names])
                    best_comb_str += f" (Actual: {actual_hint})"
                
                summary_rows.append({
                    "schedule_file": pred_best["schedule_file"],
                    "best_combination": best_comb_str,
                    "normalized_throughput": round(pred_best["actual_T_norm"], 2),
                    "drop_rate": round(pred_best["actual_D_norm"], 2),
                    "score": round(pred_best["actual_score_raw"], 2)
                })

                # [Added] Save detailed scores for all combinations in this scenario (JSON format)
                score_dir = out_dir / "score"
                score_dir.mkdir(parents=True, exist_ok=True)
                
                s_base = Path(pred_best["schedule_file"]).stem
                score_out_path = score_dir / f"scores_{s_base}.json"
                
                # Payload matching results_recompute format
                score_payload = {
                    "best deployment": best_comb_str,
                    "schedule file": pred_best["schedule_file"],
                    "data": []
                }
                
                for r in scenario_results:
                    score_payload["data"].append({
                        "combination": r["combination"],
                        "throughput_norm": round(r["actual_T_norm"], 4),
                        "drop_rate_norm": round(r["actual_D_norm"], 4),
                        "score": round(r["pred_score_raw"], 2)
                    })
                
                with open(score_out_path, "w", encoding="utf-8") as sf:
                    json.dump(score_payload, sf, indent=4)

            # Append summary results (one row per scenario)
            res_df = pd.DataFrame(summary_rows)
            res_df.to_csv(csv_out_path, mode="a", index=False)
            
            # [Added] PDF plots (using detailed data if needed, but here we just need to pass something)
            # Actually, the scatter plots need detailed data for all combinations.
            # So we should still keep a version of detailed results for plotting, but not for the CSV.
            plot_df = pd.DataFrame(detailed_results)
            if mode == "score":
                pdf_out_path = out_dir / f"prediction_result_{p_csv_path.stem}_{mode}_alpha_{alpha}.pdf"
                plot_score_scatter(plot_df, pdf_out_path, f"Score Prediction: Actual vs Predicted (alpha={alpha})")
            elif mode in ("two_target", "double"):
                pdf_out_path = out_dir / f"prediction_result_{p_csv_path.stem}_{mode}_alpha_{alpha}.pdf"
                plot_multi_scatter(plot_df, pdf_out_path, f"Multi-Model Prediction: {mode} (alpha={alpha})", alpha)

            top1_ratio = top1_hits / total_scenarios if total_scenarios > 0 else 0
            top5_ratio = top5_hits / total_scenarios if total_scenarios > 0 else 0
            
            print(f"Detailed prediction results saved to: {csv_out_path}")

            if total_scenarios > 0:
                print("--- CSV Prediction Summary ---")
                print(f"Total Scenarios: {total_scenarios}")
                if y1_errs:
                    print(f"Y1 (Throughput) MAE: {np.mean(y1_errs):.4f}")
                if y2_errs:
                    print(f"Y2 (Drop Rate) MAE: {np.mean(y2_errs):.4f}")
                print(f"1) Top-1 Hit Ratio: {top1_hits / total_scenarios:.4f} ({top1_hits}/{total_scenarios})")
                print(f"2) Top-5 Hit Ratio: {top5_hits / total_scenarios:.4f} ({top5_hits}/{total_scenarios})")
                print(f"3) Avg Score Gap: {np.mean(score_gaps):.4f}")
            
            return

        # 기존 로직: schedule_dir 내의 모든 json/yaml 처리
        import glob
        files = sorted(list(Path(args.schedule_dir).glob("*.yaml")) + list(Path(args.schedule_dir).glob("*.json")))

        for p in files:
            sched = _load_yaml_or_json(p)
            print(f"--- Processing {p.name} ---")
            results = []

            combos = _iter_combos_from_schedule(sched)
            if not combos: continue

            for name, blob in combos:
                df = featurize_from_combo(blob)
                # Feature Align: 학습 때 쓴 피처만 순서대로 추출
                for c in feats:
                    if c not in df.columns: df[c] = 0.0
                df = df[feats]

                if mode == "score":
                    y1 = np.nan
                    y2 = np.nan
                    # [Modified] Raw prediction
                    score = float(b1.predict(df)[0])
                elif mode == "rank":
                    y1 = np.nan
                    y2 = np.nan
                    score = float(b1.predict(df)[0])
                else:
                    y1 = float(b1.predict(df)[0])
                    y2 = float(b2.predict(df)[0])
                    # [Modified] Combined score
                    score = y1 - alpha * y2
                
                # [Modified] Clipping controlled by --clip_pred_score
                if args.clip_pred_score and mode != "rank":
                    score = max(0.0, min(1.0, score))

                # [수정] 모든 수치를 소수점 4자리로 반올림하여 일관성 유지
                if not np.isnan(y1): y1 = round(y1, 4)
                if not np.isnan(y2): y2 = round(y2, 4)
                score = round(score, 2)
                
                results.append((name, y1, y2, score))

            if results:
                # TOP-K 정렬 출력
                sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
                
                # [수정] top1과 score가 같은 것이 5개가 넘으면 다 보여줌
                top1_score = sorted_results[0][3]
                ties = [r for r in sorted_results if math.isclose(r[3], top1_score, rel_tol=1e-7)]
                
                if len(ties) > 5:
                    top_items = ties
                else:
                    topk = max(1, min(args.topk, len(sorted_results)))
                    top_items = sorted_results[:topk]

                actual_topk = len(top_items)
                print(f"TOP-{actual_topk}")
                for rank, r in enumerate(top_items, start=1):
                    name, y1, y2, score = r
                    y1_str = f"{y1:.4f}" if not np.isnan(y1) else "NaN"
                    y2_str = f"{y2:.4f}" if not np.isnan(y2) else "NaN"
                    print(f"{rank}\t{name}\tpred_score={score:.4f}\t(T_norm={y1_str}, D_norm={y2_str})")

                best = top_items[0]
                best_y1_str = f"{best[1]:.4f}" if not np.isnan(best[1]) else "NaN"
                best_y2_str = f"{best[2]:.4f}" if not np.isnan(best[2]) else "NaN"
                print(f"BEST\t{best[0]}\tpred_score={best[3]:.4f}\t(T_norm={best_y1_str}, D_norm={best_y2_str})")

                # 결과 파일 저장 (기존 형식 유지)
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                # [Modified] Include alpha in the filename
                out_path = out_dir / f"predict_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{p.stem}_alpha_{alpha}.json"

                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "schedule file": p.name,
                    "best deployment": best[0],
                    "score": best[3],
                    "data": []
                }
                # [수정] 결과를 score 큰 순으로 정렬하여 저장
                sorted_all_results = sorted(results, key=lambda x: x[3], reverse=True)
                for r in sorted_all_results:
                    payload["data"].append({
                        "combination": r[0],
                        "score": r[3],
                        "derived": {
                            "throughput_norm": r[1] if not np.isnan(r[1]) else None,
                            "drop_rate_norm": r[2] if not np.isnan(r[2]) else None
                        }
                    })
                with out_path.open("w") as f:
                    json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

"""
# ---------- Usage Examples ----------

1) two_target 학습:
    python xgboost_model/deploy_selector_xgb_suite.py train \
        --perf_csv xgboost_model/dataset/gpu/train_x3.csv \
        --schedule_csv xgboost_model/dataset/gpu/train_schedules_x3.csv \
        --model_out xgb_model_two_target \
        --alpha 0.2

2) score 학습 (1개 모델):
    python xgboost_model/deploy_selector_xgb_suite.py train \
        --train_mode score \
        --perf_csv xgboost_model/dataset/gpu/train_x3.csv \
        --schedule_csv xgboost_model/dataset/gpu/train_schedules_x3.csv \
        --model_out xgb_model_score \
        --alpha 0.2

3) score 모델로 테스트 평가:
    python xgboost_model/deploy_selector_xgb_suite.py predict \
        --model_in xgb_model_score \
        --perf_csv xgboost_model/dataset/gpu/test_x3.csv \
        --schedule_csv xgboost_model/dataset/gpu/test_schedules_x3.csv \
        --alpha 0.2 \
        --out_dir out_score

4) rank 학습 (Pairwise Ranking):
    python xgboost_model/deploy_selector_xgb_suite.py train \
        --train_mode rank \
        --perf_csv xgboost_model/dataset/gpu/train_x3.csv \
        --schedule_csv xgboost_model/dataset/gpu/train_schedules_x3.csv \
        --model_out xgb_model_rank \
        --alpha 0.2

5) rank 모델로 테스트 평가:
    python xgboost_model/deploy_selector_xgb_suite.py predict \
        --model_in xgb_model_rank \
        --perf_csv xgboost_model/dataset/gpu/test_x3.csv \
        --schedule_csv xgboost_model/dataset/gpu/test_schedules_x3.csv \
        --alpha 0.2 \
        --out_dir out_rank

# ---------- Sanity Check ----------
수정 후 아래 명령어로 정상 작동 여부를 확인할 수 있습니다:
(a) score 모델 학습: 
    python xgboost_model/deploy_selector_xgb_suite.py train --train_mode score --perf_csv xgboost_model/dataset/gpu/train_x3.csv --schedule_csv xgboost_model/dataset/gpu/train_schedules_x3.csv --model_out xgb_sanity --alpha 0.2
(b) predict 실행:
    python xgboost_model/deploy_selector_xgb_suite.py predict --model_in xgb_sanity --perf_csv xgboost_model/dataset/gpu/test_x3.csv --schedule_csv xgboost_model/dataset/gpu/test_schedules_x3.csv --alpha 0.2
(c) 결과 확인:
    - out_dir (기본 xgboost_model/prediction_result)에 prediction_result_{perf_csv_name}_{mode}_alpha_{alpha}.csv 생성 확인
    - 터미널에 "Total Scenarios", "Top-1 Hit Ratio", "Avg Score Gap" 출력 확인
"""