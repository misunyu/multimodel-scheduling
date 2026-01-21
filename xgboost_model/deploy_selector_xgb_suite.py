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

    for k, view in models.items():
        k_l = str(k).lower()
        if not (k_l.startswith("view") or k_l.startswith("headless")): continue

        model_name = view.get("model")
        exec_dev_raw = view.get("execution")
        if not model_name or not exec_dev_raw: continue

        exec_dev = _norm_exec(exec_dev_raw)
        if exec_dev in ("NPU0", "NPU1"): continue  # NPU Skip

        row = {}

        # [FIX 1] 실제 측정값(throughput_fps 등)을 입력 피처에서 제거함!

        # [FIX 2] 모델 식별 정보 추가
        model_feats = _get_model_features(model_name)
        row.update(model_feats)

        # Device Flags
        exec_cpu = 1.0 if exec_dev == "CPU" else 0.0
        exec_gpu = 1.0 if exec_dev == "GPU" else 0.0

        # [NEW] 결합 피처 (model_hash * device)
        # Sparse generation: only create features for the actual hash bucket
        h_val = int(hashlib.md5(model_name.lower().encode("utf-8")).hexdigest(), 16) % MODEL_HASH_BUCKETS
        row[f"model_hash_{h_val}_on_cpu"] = 1.0 * exec_cpu
        row[f"model_hash_{h_val}_on_gpu"] = 1.0 * exec_gpu

        row["view.exec_cpu"] = exec_cpu
        row["view.exec_gpu"] = exec_gpu

        # Planned FPS (Demand)
        infps_val = 0.0
        if infps_map:
            infps_val = float(infps_map.get((model_name, exec_dev), 0.0))
        
        if exec_dev == "CPU": cpu_items.append((h_val, infps_val))
        if exec_dev == "GPU": gpu_items.append((h_val, infps_val))

        row["view.infps"] = infps_val
        row["view.infps_on_cpu"] = infps_val * exec_cpu
        row["view.infps_on_gpu"] = infps_val * exec_gpu

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

    for v in views:
        m = v.get("model")
        dev = _norm_exec(v.get("execution", ""))
        if not m or not dev: continue
        if dev in ("NPU0", "NPU1"): continue

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

        # [NEW] 결합 피처 (model_hash * device)
        # Sparse generation: only create features for the actual hash bucket
        h_val = int(hashlib.md5(m.lower().encode("utf-8")).hexdigest(), 16) % MODEL_HASH_BUCKETS
        r[f"model_hash_{h_val}_on_cpu"] = 1.0 * exec_cpu
        r[f"model_hash_{h_val}_on_gpu"] = 1.0 * exec_gpu

        if dev == "CPU": cpu_items.append((h_val, fps))
        if dev == "GPU": gpu_items.append((h_val, fps))

        r["view.exec_cpu"] = exec_cpu
        r["view.exec_gpu"] = exec_gpu

        r["view.infps"] = fps
        r["view.infps_on_cpu"] = fps * exec_cpu
        r["view.infps_on_gpu"] = fps * exec_gpu

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
    else:
        X["views.count.views"] = 0.0

    return pd.DataFrame([X]).fillna(0.0)


# ---------- Builder & Trainer ----------

def _index_schedules_from_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        name = str(row["schedule_name"]).lower()
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
    if schedule_csv:
        sched_index = _index_schedules_from_csv(schedule_csv)
    elif schedule_dir:
        sched_index = _index_schedules(schedule_dir)
    else:
        sched_index = {}

    X_all, Y_all, M_all = [], [], []

    df_csv = pd.read_csv(csv_path)
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
    from sklearn.model_selection import train_test_split

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Calculate score target
    y_score = Y["y1"] - alpha * Y["y2"]

    # Hyperparameters for score mode
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "gamma": 0,
        "n_jobs": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50,
        "eval_metric": "mae"
    }

    # Split for Early Stopping (10~20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y_score, test_size=0.15, random_state=42)

    print(f"Training Score Model with {len(X_train)} samples, validating with {len(X_val)} samples.")

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    model.save_model(str(prefix) + "_score.json")
    # Meta info
    meta = {
        "mode": "score",
        "alpha": alpha,
        "features": cols
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta))


def train_rank(X, Y, M, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    
    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Calculate score target for ranking
    y_score = Y["y1"] - alpha * Y["y2"]
    
    df = X.copy()
    df["y_score"] = y_score
    df["group_id"] = M["group_id"]

    # [수정] 그룹별 relevance 계산 (0~31)
    # 각 그룹 내에서 y_score 순위에 따라 0~31로 매핑
    def compute_group_relevance(group):
        if len(group) <= 1:
            group["target"] = 31
            return group
        
        # rank() uses ascending by default, so higher score = higher rank
        ranks = group["y_score"].rank(method='min', ascending=True) - 1 # 0 to len(group)-1
        max_rank = ranks.max()
        if max_rank == 0:
            group["target"] = 31
        else:
            # Linear map to [0, 31]
            # Use floating point division then round or cast to int
            group["target"] = (ranks * 31.0 / max_rank).round().astype(int)
        return group

    df = df.groupby("group_id", group_keys=False).apply(compute_group_relevance)
    y_relevance = df["target"]
    
    # Group-based split to avoid leakage
    unique_groups = df["group_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    split_idx = int(len(unique_groups) * 0.85)
    train_groups = unique_groups[:split_idx]
    val_groups = unique_groups[split_idx:]
    
    df_train = df[df["group_id"].isin(train_groups)].sort_values("group_id")
    df_val = df[df["group_id"].isin(val_groups)].sort_values("group_id")
    
    X_train = df_train[cols]
    y_train = df_train["target"]
    g_train = df_train.groupby("group_id").size().values
    
    X_val = df_val[cols]
    y_val = df_val["target"]
    g_val = df_val.groupby("group_id").size().values
    
    print(f"Training Ranker with {len(X_train)} samples ({len(g_train)} groups), "
          f"validating with {len(X_val)} samples ({len(g_val)} groups).")

    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "objective": "rank:pairwise",
        "early_stopping_rounds": 100,
    }

    model = xgb.XGBRanker(**params)
    model.fit(
        X_train, y_train,
        group=g_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        verbose=100
    )
    
    model.save_model(str(prefix) + "_rank.json")
    # Meta info
    meta = {
        "mode": "rank",
        "alpha": alpha,
        "features": cols
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta))


def train_two_targets(X, Y, prefix, alpha=0.2):
    xgb = _lazy_import_xgb()
    from sklearn.model_selection import train_test_split

    # 컬럼 순서 고정
    cols = sorted(list(X.columns))
    X = X[cols]

    # Hyperparameters
    params = {
        "n_estimators": 10000,
        "max_depth": 8,
        "learning_rate": 0.005,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50
    }

    # Split for Early Stopping (10%)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples.")

    # 1. Train Throughput Model
    model_throughput = xgb.XGBRegressor(**params)
    model_throughput.fit(
        X_train, Y_train["y1"],
        eval_set=[(X_val, Y_val["y1"])],
        verbose=100
    )
    model_throughput.save_model(str(prefix) + "_y1.json")

    # 2. Train Drop Rate Model with Weights
    weights_train = Y_train["y2"].apply(lambda x: 10.0 if x > 0.01 else 1.0).values
    weights_val = Y_val["y2"].apply(lambda x: 10.0 if x > 0.01 else 1.0).values
    
    model_drop_rate = xgb.XGBRegressor(**params)
    model_drop_rate.fit(
        X_train, Y_train["y2"],
        sample_weight=weights_train,
        eval_set=[(X_val, Y_val["y2"])],
        sample_weight_eval_set=[weights_val],
        verbose=100
    )
    model_drop_rate.save_model(str(prefix) + "_y2.json")

    # Meta info
    meta = {
        "mode": "two_target",
        "alpha": alpha,
        "features": cols
    }
    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta))
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
    else:
        m1 = xgb.XGBRegressor()
        m1.load_model(str(prefix) + "_y1.json")
        m2 = xgb.XGBRegressor()
        m2.load_model(str(prefix) + "_y2.json")
        return m1, m2, cols, mode, alpha


def plot_score_scatter(df: pd.DataFrame, output_path: Path, title: str):
    """
    Generate a scatter plot of actual vs predicted scores.
    """
    if df.empty:
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(df["actual_score"], df["pred_score"], alpha=0.5, color='blue')
    
    # 45-degree line
    max_val = max(df["actual_score"].max(), df["pred_score"].max())
    min_val = min(df["actual_score"].min(), df["pred_score"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
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
    tr.add_argument("--train_mode", choices=["two_target", "score", "rank"], default="rank")
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
        is_constrained = "xgb_model_x3" in str(args.model_out) or "xgb_model_random" in str(args.model_out)
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
            train_score(X, Y, Path(args.model_out), alpha=args.alpha)
        elif args.train_mode == "rank":
            train_rank(X, Y, M, Path(args.model_out), alpha=args.alpha)
        else:
            train_two_targets(X, Y, Path(args.model_out), alpha=args.alpha)
        print("Training Done.")

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
                    actual_score = y1_actual - alpha * y2_actual

                    # Prediction
                    X_dict, _, _ = featurize_window(w, infps_map)
                    df_X = pd.DataFrame([X_dict])
                    df_X = df_X.reindex(columns=feats, fill_value=0.0)
                    
                    if mode == "score":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        # [Modified] Use raw prediction from model b1 without clipping
                        pred_score = float(b1.predict(df_X)[0])
                    elif mode == "rank":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        # XGBRanker predict returns scores that represent relative ranking
                        pred_score = float(b1.predict(df_X)[0])
                    else:
                        y1_pred = float(b1.predict(df_X)[0])
                        y2_pred = float(b2.predict(df_X)[0])
                        # [Modified] Calculate combined score
                        pred_score = y1_pred - alpha * y2_pred
                    
                    # [Modified] Apply clipping only if enabled via --clip_pred_score
                    # For rank mode, clipping might not make sense as it's relative, 
                    # but we'll follow the same logic. Usually, Ranker outputs are not in [0,1].
                    if args.clip_pred_score and mode != "rank":
                        pred_score = max(0.0, min(1.0, pred_score))

                    detailed_results.append({
                        "schedule_file": s_name,
                        "timestamp": ts,
                        "combination": c_name,
                        "actual_T_norm": round(y1_actual, 4),
                        "actual_D_norm": round(y2_actual, 4),
                        "actual_score": round(actual_score, 4),
                        "pred_T_norm": round(y1_pred, 4) if not np.isnan(y1_pred) else "",
                        "pred_D_norm": round(y2_pred, 4) if not np.isnan(y2_pred) else "",
                        "pred_score": round(pred_score, 4),
                        "diff_score": round(abs(actual_score - pred_score), 4)
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

            # summary calculation using grouped scenario_data
            for group_id, windows in scenario_data.items():
                # We need a schedule doc to get infps_map for each window
                # windows might have different schedule_files if they share same models
                # but usually scenario_data[group_id] will have same schedule_file per group_id
                # as per get_group_id implementation.
                
                total_scenarios += 1
                scenario_results = []
                
                for w in windows:
                    s_name = w.get("schedule_file") or w.get("schedule file")
                    s_doc = _find_schedule(sched_index, s_name)
                    if not s_doc: continue
                    
                    c_name = w.get("combination")
                    infps_map = _build_infps_lookup(s_doc, c_name)
                    
                    # Ground Truth
                    y1_actual = float(w.get("derived", {}).get("throughput_norm", np.nan))
                    y2_actual = float(w.get("derived", {}).get("drop_rate_norm", np.nan))
                    actual_score = y1_actual - alpha * y2_actual

                    # Prediction
                    X_dict, _, _ = featurize_window(w, infps_map)
                    df_X = pd.DataFrame([X_dict])
                    df_X = df_X.reindex(columns=feats, fill_value=0.0)
                    
                    if mode == "score":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        # [Modified] Use raw prediction from model b1
                        pred_score = float(b1.predict(df_X)[0])
                    elif mode == "rank":
                        y1_pred = np.nan
                        y2_pred = np.nan
                        pred_score = float(b1.predict(df_X)[0])
                    else:
                        y1_pred = float(b1.predict(df_X)[0])
                        y2_pred = float(b2.predict(df_X)[0])
                        # [Modified] Combined score
                        pred_score = y1_pred - alpha * y2_pred
                    
                    # [Modified] Clipping controlled by --clip_pred_score
                    if args.clip_pred_score and mode != "rank":
                        pred_score = max(0.0, min(1.0, pred_score))
                    
                    if not np.isnan(y1_pred):
                        y1_errs.append(abs(y1_actual - y1_pred))
                    if not np.isnan(y2_pred):
                        y2_errs.append(abs(y2_actual - y2_pred))
                    
                    scenario_results.append({
                        "combination": c_name,
                        "actual_score": actual_score,
                        "pred_score": pred_score
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

                pred_best_name_first = pred_sorted[0]["combination"]
                pred_best_actual_score = pred_sorted[0]["actual_score"]
                actual_best_score = actual_sorted[0]["actual_score"]
                
                # Apply coloring to detailed_results for this scenario
                for res in detailed_results:
                    if res["schedule_file"] == s_name and res["timestamp"] == ts:
                        c_name = res["combination"]
                        is_actual_best = c_name in actual_best_names
                        is_pred_best = c_name in pred_best_names
                        
                        if is_actual_best and is_pred_best:
                            res["combination"] = f"<font color='purple'>{c_name}</font>"
                        elif is_actual_best:
                            res["combination"] = f"<font color='blue'>{c_name}</font>"
                        elif is_pred_best:
                            res["combination"] = f"<font color='red'>{c_name}</font>"
                
                # 1) Check if predicted best is in actual bests (Top-1 Hit)
                if pred_best_name_first in actual_best_names:
                    top1_hits += 1

                # 2) Check if actual best is in top-5 predicted (Top-5 Hit)
                top5_pred_names = [r["combination"] for r in pred_sorted[:5]]
                if any(name in top5_pred_names for name in actual_best_names):
                    top5_hits += 1
                
                # 3) Score gap
                gap = actual_best_score - pred_best_actual_score
                score_gaps.append(max(0, gap))

            res_df = pd.DataFrame(detailed_results)
            csv_out_path = out_dir / f"prediction_result_{p_csv_path.stem}.csv"
            
            res_df.to_csv(csv_out_path, index=False)
            
            # [Added] Score scatter plot for score mode
            if mode == "score":
                pdf_out_path = out_dir / f"prediction_result_{p_csv_path.stem}.pdf"
                plot_score_scatter(res_df, pdf_out_path, f"Score Prediction: Actual vs Predicted (alpha={alpha})")

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
                score = round(score, 4)
                
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
                out_path = out_dir / f"predict_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{p.stem}.json"

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
    - out_dir (기본 xgboost_model/prediction_result)에 prediction_result_test_x3.csv 생성 확인
    - 터미널에 "Total Scenarios", "Top-1 Hit Ratio", "Avg Score Gap" 출력 확인
"""