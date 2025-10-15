#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deployment selection model (XGBoost w/ Ubuntu-friendly GPU/CPU control) — train & predict

- Ubuntu에서 GPU(CUDA)가 있으면 자동 사용(--device auto), 없으면 CPU로 동작
- CPU일 때 n_jobs/스레드 제어(OMP_NUM_THREADS) 지원
- train:  learn from ./performance/*.json logs
- predict: score candidate combinations from YAML (or JSON structure)
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# XGBoost import (fallback 준비)
# -------------------------
USING_XGB = False
try:
    import xgboost as xgb  # XGBoost 2.x 권장
    USING_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    USING_XGB = False


# -------------------------
# GPU 가용성 간단 탐지 (Ubuntu)
# -------------------------
def _detect_cuda_available() -> bool:
    """Lightweight CUDA availability check without hard deps."""
    # Honor CUDA_VISIBLE_DEVICES if explicitly disabled
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and cvd.strip() in {"", "-1"}:
        return False

    # Try common heuristics
    # 1) NVIDIA driver presence via proc
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    # 2) nvidia-smi in PATH
    from shutil import which
    if which("nvidia-smi"):
        return True
    return False


# -------------------------
# Feature engineering utils
# -------------------------
def _feat_from_models_dict(models: Dict[str, Dict]) -> Dict[str, float]:
    """Extract sparse categorical features from a 'models' dict in performance JSON.
    - One-hot for (view, model_name)
    - One-hot for (view, execution_device)
    - Counts per device (cpu/npu0/npu1) as numeric features
    """
    feat: Dict[str, float] = {}
    # View-level one-hot
    for view, spec in models.items():
        mname = spec["model"]
        execu = str(spec["execution"]).lower()
        feat[f"{view}__model={mname}"] = 1.0
        feat[f"{view}__exec={execu}"] = 1.0
    # Device counts
    dev_counts: Dict[str, int] = {}
    for spec in models.values():
        dev = str(spec["execution"]).lower()
        dev_counts[dev] = dev_counts.get(dev, 0) + 1
    for dev, cnt in dev_counts.items():
        feat[f"count_on__{dev}"] = float(cnt)
    return feat


def _feat_from_yaml_assigns(assigns: Dict[str, Dict]) -> Dict[str, float]:
    """Extract features from one combination entry of schedule YAML.
    assigns: { arbitrary_key: {display, execution, model, ...}, ... }
    """
    feat: Dict[str, float] = {}
    # View-level one-hot
    for _key, spec in assigns.items():
        view = spec.get("display")
        mname = spec.get("model")
        execu = str(spec.get("execution", "")).lower()
        if view:
            feat[f"{view}__model={mname}"] = 1.0
            feat[f"{view}__exec={execu}"] = 1.0
    # Device counts
    dev_counts: Dict[str, int] = {}
    for spec in assigns.values():
        dev = str(spec.get("execution", "")).lower()
        dev_counts[dev] = dev_counts.get(dev, 0) + 1
    for dev, cnt in dev_counts.items():
        feat[f"count_on__{dev}"] = float(cnt)
    return feat


# -------------------------
# Dataset loaders
# -------------------------
def rows_from_perf_json(fp: str) -> List[Dict]:
    """Load rows from a single performance JSON file."""
    doc = json.load(open(fp, "r"))
    rows: List[Dict] = []
    for item in doc.get("data", []):
        feat = _feat_from_models_dict(item["models"])
        y = float(item.get("score", 0.0))
        rows.append(
            {
                "source": os.path.basename(fp),
                "combination": item.get("combination"),
                "features": feat,
                "y": y,
            }
        )
    return rows


def rows_from_schedule_yaml(schedule_path: str) -> List[Dict]:
    """Load rows from a single schedule YAML file."""
    ydoc = yaml.safe_load(open(schedule_path, "r"))
    rows: List[Dict] = []
    for combo_name, assigns in ydoc.items():
        feat = _feat_from_yaml_assigns(assigns)
        rows.append({"combination": combo_name, "features": feat})
    return rows


# -------------------------
# Model builders (Ubuntu-aware)
# -------------------------
def _build_regressor(
    use_xgb: bool,
    device_opt: str = "auto",
    n_jobs: Optional[int] = None,
    seed: int = 42,
):
    """
    device_opt: "auto" | "cuda" | "cpu"
    - auto: CUDA 있으면 cuda, 아니면 cpu
    - cuda: 강제 GPU (실패하면 cpu로 폴백)
    - cpu : 강제 CPU
    """
    if not use_xgb:
        # Fallback: scikit-learn HGB
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        return HGB(random_state=seed, max_depth=8)

    # Decide device
    device = "cpu"
    # if device_opt == "cuda":
    #     device = "cuda"
    # elif device_opt == "auto":
    #     device = "cuda" if _detect_cuda_available() else "cpu"
    # else:
    #     device = "cpu"

    # CPU thread control (Ubuntu)
    # - n_jobs: XGB internal threads
    # - OMP_NUM_THREADS: OpenMP threads for histogram builder
    if device == "cpu":
        n_jobs = 8
        # if n_jobs is None or n_jobs <= 0:
        #     n_jobs = os.cpu_count() or 1
        os.environ.setdefault("OMP_NUM_THREADS", str(n_jobs))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_jobs))
        os.environ.setdefault("MKL_NUM_THREADS", str(n_jobs))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n_jobs))
    else:
        # On GPU, let CPU threads be minimal
        if n_jobs is None or n_jobs <= 0:
            n_jobs = 1

    # XGBoost 2.x uses device param; tree_method="hist" works on both CPU/GPU
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        device=device,
    )
    try:
        model = xgb.XGBRegressor(**params)
    except TypeError:
        # In case of older XGBoost (<2.0), fall back to legacy gpu params
        legacy_params = dict(params)
        legacy_params.pop("device", None)
        if device == "cuda":
            legacy_params["tree_method"] = "gpu_hist"
            legacy_params["predictor"] = "gpu_predictor"
        model = xgb.XGBRegressor(**legacy_params)
    return model


# -------------------------
# Train
# -------------------------
def cmd_train(
    perf_glob: str,
    out_dir: str,
    test_size: float = 0.2,
    device: str = "auto",
    n_jobs: int = 0,
    seed: int = 42,
) -> None:
    files = sorted(glob.glob(perf_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {perf_glob}")

    all_rows: List[Dict] = []
    for fp in files:
        all_rows.extend(rows_from_perf_json(fp))

    df = pd.DataFrame(all_rows)
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(df["features"])  # sparse one-hot
    y = df["y"].values

    model = _build_regressor(
        use_xgb=USING_XGB, device_opt=device, n_jobs=n_jobs, seed=seed
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "xgb_deploy_selector.model")
    joblib.dump(dv, out / "xgb_deploy_selector.feats")

    report = {
        "using_xgboost": USING_XGB,
        "device": getattr(model, "device", "cpu") if USING_XGB else "cpu",
        "n_jobs": getattr(model, "n_jobs", None) if USING_XGB else None,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "r2": float(r2),
        "mae": float(mae),
        "model_path": str(out / "xgb_deploy_selector.model"),
        "feats_path": str(out / "xgb_deploy_selector.feats"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


# -------------------------
# Predict
# -------------------------
def cmd_predict(
    model_path: str,
    feats_path: str,
    schedule_yaml: Optional[str],
    perf_json: Optional[str],
    schedule_glob: Optional[str] = None,
    out_csv: Optional[str] = None,
) -> None:
    model = joblib.load(model_path)
    dv = joblib.load(feats_path)

    rows: List[Dict] = []

    # Batch mode over multiple YAMLs (quote the glob in shell!)
    if schedule_glob:
        for yp in sorted(glob.glob(schedule_glob)):
            for r in rows_from_schedule_yaml(yp):
                r["source_yaml"] = os.path.basename(yp)
                rows.append(r)
    elif schedule_yaml:
        for r in rows_from_schedule_yaml(schedule_yaml):
            r["source_yaml"] = os.path.basename(schedule_yaml)
            rows.append(r)
    elif perf_json:
        doc = json.load(open(perf_json, "r"))
        for item in doc.get("data", []):
            rows.append(
                {
                    "source_yaml": os.path.basename(perf_json),
                    "combination": item.get("combination"),
                    "features": _feat_from_models_dict(item["models"]),
                }
            )
    else:
        raise ValueError("Provide --schedule_yaml or --perf_json or --schedule_glob.")

    X = dv.transform([r["features"] for r in rows])
    preds = model.predict(X)

    df = pd.DataFrame(
        {
            "source": [r.get("source_yaml", "-") for r in rows],
            "combination": [r["combination"] for r in rows],
            "pred_score": preds,
        }
    ).sort_values(["source", "pred_score"], ascending=[True, False]).reset_index(
        drop=True
    )

    out_path = Path(out_csv) if out_csv else Path("predictions.csv")
    out_path = out_path.resolve()
    df.to_csv(out_path, index=False)

    # Top-1 per source file (batch) or single top-1
    if schedule_glob or schedule_yaml:
        top_per_source = (
            df.groupby("source").head(1).reset_index(drop=True).to_dict(orient="records")
        )
        result = {"top_per_source": top_per_source, "predictions_csv": str(out_path)}
    else:
        result = {"top1": df.iloc[0]["combination"], "predictions_csv": str(out_path)}
    print(json.dumps(result, indent=2, ensure_ascii=False))


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Deployment selection trainer/predictor (Ubuntu-friendly)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--perf_glob", default="./performance/*.json", help="Training data glob")
    ap_tr.add_argument("--out_dir", default="./artifacts", help="Where to save model artifacts")
    ap_tr.add_argument("--test_size", type=float, default=0.2)
    ap_tr.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Training device for XGBoost (auto detects CUDA on Ubuntu)")
    ap_tr.add_argument("--n_jobs", type=int, default=0, help="CPU threads (0:auto). Ignored if device=cuda.")
    ap_tr.add_argument("--seed", type=int, default=42)

    # predict
    ap_pr = sub.add_parser("predict")
    ap_pr.add_argument("--model_path", default="./xgb_deploy_selector.model")
    ap_pr.add_argument("--feats_path", default="./xgb_deploy_selector.feats")
    grp = ap_pr.add_mutually_exclusive_group(required=False)
    grp.add_argument("--schedule_yaml", help="Predict for a single YAML file")
    grp.add_argument("--perf_json", help="Predict for a single performance JSON (structure only)")
    ap_pr.add_argument("--schedule_glob", help="Predict for many YAMLs (quote the glob! e.g., './tests/*.yaml')")
    ap_pr.add_argument("--out_csv", help="Where to save predictions CSV (default: ./predictions.csv)")

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train(args.perf_glob, args.out_dir, args.test_size, args.device, args.n_jobs, args.seed)
    elif args.cmd == "predict":
        cmd_predict(args.model_path, args.feats_path, args.schedule_yaml, args.perf_json, args.schedule_glob, args.out_csv)
    else:
        ap.error("Unknown command")


if __name__ == "__main__":
    main()
