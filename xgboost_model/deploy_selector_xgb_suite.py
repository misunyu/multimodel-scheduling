#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 python ./xgboost_model/deploy_selector_xgb_suite.py train --perf_dir ./xgboost_model/train/performance_results     --schedule_dir ./xgboost_model/train/schedules  --model_out ./xgboost_model/artifacts/xgb_model
 python ./xgboost_model/deploy_selector_xgb_suite.py predict     --schedule_dir ./xgboost_model/test/schedules     --model_in ./xgboost_model/artifacts/xgb_model
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

try:
    import yaml
except Exception:
    yaml = None

# ---------- Constants ----------
WINDOW_SEC = 30.0
ASSUME_WAIT_MS = 0.0
MODEL_HASH_BUCKETS = 20  # 모델 이름을 식별하기 위한 해싱 버킷 크기


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

    for i in range(MODEL_HASH_BUCKETS):
        feats[f"model_hash_{i}"] = 1.0 if i == h_val else 0.0
    return feats


# ---------- Feature Engineering (Fixed) ----------

def featurize_window(window: Dict[str, Any], infps_map=None) -> Tuple[
    Dict[str, float], Tuple[float, float], Dict[str, Any]]:
    models = window.get("models", {})
    per_view_rows = []

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
        row.update(_get_model_features(model_name))

        # Device Flags
        row["view.exec_cpu"] = 1.0 if exec_dev == "CPU" else 0.0
        row["view.exec_gpu"] = 1.0 if exec_dev == "GPU" else 0.0

        # Planned FPS (Demand)
        infps_val = 0.0
        if infps_map:
            infps_val = float(infps_map.get((model_name, exec_dev), 0.0))
        row["view.infps"] = infps_val

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
        r.update(_get_model_features(m))
        r["view.exec_cpu"] = 1.0 if dev == "CPU" else 0.0
        r["view.exec_gpu"] = 1.0 if dev == "GPU" else 0.0
        r["view.infps"] = fps

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
    else:
        X["views.count.views"] = 0.0

    return pd.DataFrame([X]).fillna(0.0)


# ---------- Builder & Trainer ----------

def build_dataset(perf_dir: Path, schedule_dir: Path):
    sched_index = _index_schedules(schedule_dir)
    X_all, Y_all, M_all = [], [], []

    for path in sorted(perf_dir.rglob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            for w in blob.get("data", []):
                s_name = w.get("schedule file") or w.get("schedule_file")
                s_doc = _find_schedule(sched_index, s_name)
                c_name = w.get("combination")

                infps_map = None
                if s_doc and c_name:
                    infps_map = _build_infps_lookup(s_doc, c_name)

                X, (y1, y2), meta = featurize_window(w, infps_map)
                if math.isnan(y1) or math.isnan(y2): continue

                X_all.append(X)
                Y_all.append({"y1": y1, "y2": y2})
                M_all.append(meta)
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")

    if not X_all: raise RuntimeError("No valid data rows found.")
    return pd.DataFrame(X_all).fillna(0.0), pd.DataFrame(Y_all), pd.DataFrame(M_all)


def train_two_targets(X, Y, prefix):
    xgb = _lazy_import_xgb()
    # 컬럼 순서 고정 (매우 중요)
    cols = sorted(list(X.columns))
    X = X[cols]

    params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1, "seed": 42}

    # Train Y1
    bst1 = xgb.train(params, xgb.DMatrix(X, label=Y["y1"]), num_boost_round=200)
    bst1.save_model(str(prefix) + "_y1.json")

    # Train Y2
    bst2 = xgb.train(params, xgb.DMatrix(X, label=Y["y2"]), num_boost_round=200)
    bst2.save_model(str(prefix) + "_y2.json")

    # Feature 이름 저장 (추론 시 정렬을 위해)
    Path(str(prefix) + "_features.json").write_text(json.dumps(cols))


def load_models(prefix):
    xgb = _lazy_import_xgb()
    b1 = xgb.Booster(model_file=str(prefix) + "_y1.json")
    b2 = xgb.Booster(model_file=str(prefix) + "_y2.json")
    cols = json.loads(Path(str(prefix) + "_features.json").read_text())
    return b1, b2, cols


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--perf_dir", required=True)
    tr.add_argument("--schedule_dir", required=True)
    tr.add_argument("--model_out", required=True)
    tr.add_argument("--dump_csv", default="")

    pr = sub.add_parser("predict")
    pr.add_argument("--schedule_dir", required=True)
    pr.add_argument("--model_in", required=True)
    pr.add_argument("--alpha", type=float, default=0.2)
    pr.add_argument("--topk", type=int, default=5)
    pr.add_argument("--repeats", type=int, default=1)

    args = ap.parse_args()

    if args.cmd == "train":
        X, Y, M = build_dataset(Path(args.perf_dir), Path(args.schedule_dir))
        train_two_targets(X, Y, Path(args.model_out))
        print("Training Done.")

    elif args.cmd == "predict":
        b1, b2, feats = load_models(Path(args.model_in))
        xgb = _lazy_import_xgb()

        # schedule_dir 내의 모든 json/yaml 처리
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

                dmat = xgb.DMatrix(df)
                y1 = float(b1.predict(dmat)[0])
                y2 = float(b2.predict(dmat)[0])
                score = y1 - args.alpha * y2
                
                # [수정] 모든 수치를 소수점 4자리로 반올림하여 일관성 유지
                y1 = round(y1, 4)
                y2 = round(y2, 4)
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
                    print(f"{rank}\t{name}\tpred_score={score:.4f}\t(T_norm={y1:.4f}, D_norm={y2:.4f})")

                best = top_items[0]
                print(f"BEST\t{best[0]}\tpred_score={best[3]:.4f}\t(T_norm={best[1]:.4f}, D_norm={best[2]:.4f})")

                # 결과 파일 저장 (기존 형식 유지)
                out_dir = Path("xgboost_model/test/performance_results/prediction")
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
                            "throughput_norm": r[1],
                            "drop_rate_norm": r[2]
                        }
                    })
                with out_path.open("w") as f:
                    json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()