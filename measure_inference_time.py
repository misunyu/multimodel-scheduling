import time
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import sys
import os
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

# Add the project root to sys.path to import from xgboost_model if needed
# But we can just copy necessary parts or import from the existing script.
sys.path.append(str(Path.cwd() / "xgboost_model"))
try:
    from deploy_selector_xgb_suite import featurize_from_combo, _norm_exec, _rows_from_combo_struct
except ImportError:
    print("Could not import from deploy_selector_xgb_suite. Make sure the path is correct.")
    sys.exit(1)

def _detect_xgb_device(model: xgb.XGBRegressor) -> Tuple[str, str]:
    """
    Detect device used by XGBoost model for prediction.
    Returns tuple (device_kind, reason) where device_kind in {"GPU", "CPU", "UNKNOWN"}.
    """
    try:
        # 0. Check parameters directly (set_params might store it here)
        params = model.get_params()
        dev = params.get("device")
        if dev:
            d = str(dev).lower()
            if d in ("cuda", "gpu") or d.startswith("cuda:"):
                return "GPU", f"params.device={dev}"
            if d == "cpu":
                return "CPU", f"params.device={dev}"

        # 1. Prefer booster-side config which records the runtime device in >=1.7/2.0
        cfg_txt = model.get_booster().save_config()
        cfg = json.loads(cfg_txt)
        dev = (
            cfg.get("learner", {})
               .get("generic_param", {})
               .get("device")
        )
        if isinstance(dev, str):
            d = dev.lower()
            if d in ("cuda", "gpu") or d.startswith("cuda:"):
                return "GPU", f"booster.config.device={dev}"
            if d == "cpu":
                return "CPU", f"booster.config.device={dev}"
        
        # 2. Legacy hints
        predictor = str(params.get("predictor", "")).lower()
        tree_method = str(params.get("tree_method", "")).lower()
        if predictor == "gpu_predictor" or tree_method == "gpu_hist":
            return "GPU", f"params: predictor={predictor}, tree_method={tree_method}"
        if predictor:
            return "CPU", f"params: predictor={predictor}"
        
        # Fallback
        return "UNKNOWN", "no explicit device found"
    except Exception as e:
        return "UNKNOWN", f"error: {e}"


def load_models_and_meta(prefix, force_gpu=False):
    meta_path = Path(str(prefix) + "_meta.json")
    if not meta_path.exists():
        print(f"Meta file {meta_path} not found.")
        sys.exit(1)
    
    meta = json.loads(meta_path.read_text())
    cols = meta.get("features")
    alpha = meta.get("alpha", 0.2)
    mode = meta.get("mode", "double")
    
    if mode == "double":
        m1 = xgb.XGBRegressor()
        m1.load_model(str(prefix) + "_y1.json")
        if force_gpu:
            m1.set_params(device="cuda")
        
        m2 = xgb.XGBRegressor()
        m2.load_model(str(prefix) + "_y2.json")
        if force_gpu:
            m2.set_params(device="cuda")

        dev1, why1 = _detect_xgb_device(m1)
        dev2, why2 = _detect_xgb_device(m2)
        print(f"Model y1 device: {dev1} ({why1})")
        print(f"Model y2 device: {dev2} ({why2})")
        
        return m1, m2, cols, alpha, mode
    elif mode == "score":
        m_score = xgb.XGBRegressor()
        m_score.load_model(str(prefix) + "_score.json")
        if force_gpu:
            m_score.set_params(device="cuda")
        
        dev, why = _detect_xgb_device(m_score)
        print(f"Model score device: {dev} ({why})")
        return m_score, None, cols, alpha, mode
    else:
        print(f"Unsupported mode: {mode}")
        sys.exit(1)

def measure_inference_time(model_prefix, schedule_path, num_runs=100):
    print(f"Loading schedule from {schedule_path}...")
    with open(schedule_path, 'r') as f:
        schedule_data = yaml.safe_load(f)
    
    # Get the first combination for testing
    combos = [k for k in schedule_data.keys() if k.startswith("combination_")]
    if not combos:
        print("No combinations found in schedule file.")
        return
    
    combo_name = combos[0]
    combo_blob = schedule_data[combo_name]

    # --- Double Model (CPU/GPU) ---
    print("\n" + "="*50)
    print("Testing Double Model Approach (y1, y2 -> Score)")
    print("="*50)
    
    print("\n" + "-"*20 + " Running on CPU " + "-"*20)
    _run_measurement(model_prefix, combo_blob, combo_name, num_runs, force_gpu=False)

    print("\n" + "-"*20 + " Running on GPU " + "-"*20)
    _run_measurement(model_prefix, combo_blob, combo_name, num_runs, force_gpu=True)

    # --- Score Model (CPU Only) ---
    score_model_prefix = "xgboost_model/artifacts/gpu/xgb_model_x3_score_alpha_0.2"
    if Path(score_model_prefix + "_meta.json").exists():
        print("\n" + "="*50)
        print("Testing Direct Score Model Approach (Single Model)")
        print("="*50)
        print("\n" + "-"*20 + " Running on CPU " + "-"*20)
        _run_measurement(score_model_prefix, combo_blob, combo_name, num_runs, force_gpu=False)
    else:
        print(f"\nScore model meta not found at {score_model_prefix}_meta.json")

def _run_measurement(model_prefix, combo_blob, combo_name, num_runs, force_gpu=False):
    print(f"Loading models from {model_prefix} (force_gpu={force_gpu})...")
    m1, m2, cols, alpha, mode = load_models_and_meta(model_prefix, force_gpu=force_gpu)
    
    if mode == "double":
        if force_gpu:
            print(f"Measuring sequential inference time (GPU) for '{combo_name}' over {num_runs} runs...")
        else:
            print(f"Measuring parallel inference time (CPU) for '{combo_name}' over {num_runs} runs...")
    else:
        print(f"Measuring single model inference time ({'GPU' if force_gpu else 'CPU'}) for '{combo_name}' over {num_runs} runs...")
    
    total_time = 0
    total_featurization = 0
    total_prediction = 0
    total_score_calc = 0

    # Thread pool for parallel inference (CPU and mode=double only)
    executor = ThreadPoolExecutor(max_workers=2) if (not force_gpu and mode == "double") else None

    def predict_y1(dm_or_df):
        if force_gpu:
            return m1.get_booster().predict(dm_or_df)[0]
        else:
            return m1.predict(dm_or_df)[0]

    def predict_y2(dm_or_df):
        if force_gpu:
            return m2.get_booster().predict(dm_or_df)[0]
        else:
            return m2.predict(dm_or_df)[0]

    def predict_score(dm_or_df):
        if force_gpu:
            return m1.get_booster().predict(dm_or_df)[0]
        else:
            return m1.predict(dm_or_df)[0]

    # Warm-up run
    X_df = featurize_from_combo(combo_blob)
    X_df = X_df.reindex(columns=cols, fill_value=0.0)
    
    if mode == "double":
        if force_gpu:
            dm = xgb.DMatrix(X_df)
            y1_pred = predict_y1(dm)
            y2_pred = predict_y2(dm)
        else:
            f1 = executor.submit(predict_y1, X_df)
            f2 = executor.submit(predict_y2, X_df)
            f1.result()
            f2.result()
    else: # mode == "score"
        if force_gpu:
            dm = xgb.DMatrix(X_df)
            predict_score(dm)
        else:
            predict_score(X_df)
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        # 1. Featurization
        feat_start = time.perf_counter()
        X_df = featurize_from_combo(combo_blob)
        X_df = X_df.reindex(columns=cols, fill_value=0.0)
        feat_end = time.perf_counter()
        
        # 2. Prediction
        pred_start = time.perf_counter()
        if mode == "double":
            if force_gpu:
                # Sequential on GPU
                dm = xgb.DMatrix(X_df)
                y1_pred = predict_y1(dm)
                y2_pred = predict_y2(dm)
            else:
                # Parallel on CPU
                f1 = executor.submit(predict_y1, X_df)
                f2 = executor.submit(predict_y2, X_df)
                y1_pred = f1.result()
                y2_pred = f2.result()
        else: # mode == "score"
            if force_gpu:
                dm = xgb.DMatrix(X_df)
                score_pred = predict_score(dm)
            else:
                score_pred = predict_score(X_df)
        pred_end = time.perf_counter()
        
        # 3. Score calculation
        score_start = time.perf_counter()
        if mode == "double":
            score = y1_pred * (1.0 - alpha) - y2_pred * alpha
        else:
            score = score_pred
        score_end = time.perf_counter()
        
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        total_featurization += (feat_end - feat_start)
        total_prediction += (pred_end - pred_start)
        total_score_calc += (score_end - score_start)
        
    if executor:
        executor.shutdown()

    avg_time_ms = (total_time / num_runs) * 1000
    avg_feat_ms = (total_featurization / num_runs) * 1000
    avg_pred_ms = (total_prediction / num_runs) * 1000
    avg_score_ms = (total_score_calc / num_runs) * 1000

    print(f"\nResults ({'GPU' if force_gpu else 'CPU'}, mode={mode}) for {num_runs} runs:")
    print(f"Average Total Time:    {avg_time_ms:.4f} ms")
    print(f"  - Featurization:     {avg_feat_ms:.4f} ms")
    print(f"  - Model Prediction:  {avg_pred_ms:.4f} ms")
    print(f"  - Score Calculation: {avg_score_ms:.4f} ms")
    print("-" * 30)
    if mode == "double":
        print(f"Predicted y1 (throughput): {y1_pred:.4f}")
        print(f"Predicted y2 (drop rate): {y2_pred:.4f}")
    print(f"Final Score: {score:.4f}")

if __name__ == "__main__":
    model_prefix = "xgboost_model/artifacts/gpu/xgb_model_x3_double"
    schedule_path = "gen_schedules/model_schedules_m_r_t_y_x3.yaml"
    
    if len(sys.argv) > 1:
        schedule_path = sys.argv[1]
        
    measure_inference_time(model_prefix, schedule_path)
