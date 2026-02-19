import json
import os
import math
import random
import re
import pandas as pd
import numpy as np
from pathlib import Path

def get_model_tags(filename):
    match = re.search(r'model_schedules_(.*?)\.json', filename)
    if match:
        tag_str = match.group(1)
        tag_str = re.sub(r'(_x\d+|_test)', '', tag_str)
        return tag_str.split('_')
    return []

def get_input_rate_suffix(filename):
    if '_x2' in filename: return '_x2'
    if '_x3' in filename: return '_x3'
    return ''

def find_single_model_file(results_dir, tag, suffix):
    candidates = []
    if not os.path.exists(results_dir):
        return None
    for f in os.listdir(results_dir):
        if f.startswith("performance_") and f.endswith(".json"):
            if suffix and suffix not in f: continue
            if not suffix and ('_x2' in f or '_x3' in f): continue
            
            # Simple tag matching
            f_tags = get_model_tags(f)
            if len(f_tags) == 1 and f_tags[0] == tag:
                candidates.append(f)
    
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(results_dir, candidates[-1])

def get_best_device_for_model(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        best_throughput = -1
        best_device = None
        
        for entry in data.get('data', []):
            models = entry.get('models', {})
            if len(models) != 1:
                continue
            
            view_key = list(models.keys())[0]
            model_info = models[view_key]
            throughput = model_info.get('throughput_fps', 0)
            device = model_info.get('execution', 'CPU').upper()
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_device = device
                
        return best_device
    except:
        return 'CPU'

def get_latency_based_combination(rows, results_dir, schedule_file):
    model_name_to_tag = {
        'gpt2': 'g',
        'mnasnet': 'm',
        'resnet50': 'r',
        'resnext50': 'r',
        'shufflenet-v2-12': 's',
        'shufflenet': 's',
        'tiny-llama-chat-onnx': 't',
        'vgg16': 'v',
        'yolov3': 'y',
        'vgg19': 'v',
        'squeezenet': 's'
    }

    suffix = ""
    if "_x2" in schedule_file: suffix = "_x2"
    elif "_x3" in schedule_file: suffix = "_x3"

    perf_file_pattern = schedule_file.replace(".yaml", ".json")
    perf_files = list(Path(results_dir).glob(f"performance_*_{perf_file_pattern}"))
    if not perf_files:
        return "combination_1"
    
    perf_file = sorted(perf_files)[-1]
    with open(perf_file, 'r') as f:
        perf_data = json.load(f)
    
    if not perf_data.get('data'):
        return "combination_1"
    
    first_entry = perf_data['data'][0]
    models_in_schedule = first_entry.get('models', {})
    
    model_best_devices = {}
    for view_key, info in models_in_schedule.items():
        m_name = info.get('model')
        tag = model_name_to_tag.get(m_name)
        if not tag:
            for k, v in model_name_to_tag.items():
                if k in m_name:
                    tag = v
                    break
        
        best_device = 'CPU'
        if tag:
            single_file = find_single_model_file(results_dir, tag, suffix)
            if single_file:
                best_device = get_best_device_for_model(single_file)
        
        model_best_devices[view_key] = best_device

    best_comb = None
    for entry in perf_data.get('data', []):
        match = True
        for view_key, best_dev in model_best_devices.items():
            current_dev = entry.get('models', {}).get(view_key, {}).get('execution', '').upper()
            if current_dev != best_dev:
                match = False
                break
        if match:
            best_comb = entry.get('combination')
            break
            
    return best_comb if best_comb else "combination_1"

def get_model_count_from_sched(sched_name, sched_index):
    # Normalize sched_name to match index keys
    key = str(Path(sched_name).name).lower()
    sched_content = sched_index.get(key)
    if not sched_content:
        print(f"DEBUG: Schedule {key} not found in index")
        return 0
    try:
        first_comb = next(iter(sched_content.values()))
        # If it's a dict with 'models' key (new format)
        if isinstance(first_comb, dict) and 'models' in first_comb:
            return len(first_comb['models'])
        # Old format might be different, let's just count keys if they look like models
        return len(first_comb)
    except Exception as e:
        print(f"DEBUG: Error counting models for {key}: {e}")
        return 0

def main():
    prediction_csv = Path("xgboost_model/prediction_result/gpu_x3_double/prediction_result_test_x3_double.csv")
    results_dir = "results"
    schedule_csv = Path("xgboost_model/dataset/gpu/test_schedules_x3.csv")
    output_latency_csv = Path("experimental_results/latency_based_best_results_x3.csv")
    output_random_csv = Path("experimental_results/random_search_base_results_x3.csv")

    sys.path.append(str(Path.cwd() / "xgboost_model"))
    from deploy_selector_xgb_suite import (
        _index_schedules_from_csv, 
        load_models, 
        featurize_window, 
        _find_schedule, 
        _build_infps_lookup
    )
    
    schedule_csv = Path("xgboost_model/dataset/gpu/test_schedules_x3.csv")

    # Read the detailed prediction data (not just the summary)
    # The CSV starts with summary lines, so we need to find where the data starts
    with open(prediction_csv, 'r') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('schedule_file,'):
            header_idx = i
            break
    
    if header_idx == -1:
        print("Detailed data not found in CSV.")
        return

    # We need the original detailed CSV with all combinations, but we only have the summary CSV.
    # We must RE-RUN the prediction to get all combinations.
    # NO, I'll use build_dataset_from_csv from deploy_selector_xgb_suite to get the data.
    # from deploy_selector_xgb_suite import build_dataset_from_csv, load_models, featurize_window
    
    p_csv = Path("xgboost_model/dataset/gpu/test_x3.csv")
    s_dir = None
    s_csv = Path("xgboost_model/dataset/gpu/test_schedules_x3.csv")
    # X, Y, M = build_dataset_from_csv(p_csv, s_dir, s_csv, is_constrained=True)
    needed_names = set()
    df_csv = pd.read_csv(p_csv)
    for _, row in df_csv.iterrows():
        try:
            w = json.loads(row["json_content"])
            s_name = w.get("schedule_file") or w.get("schedule file")
            if s_name:
                needed_names.add(str(Path(s_name).name).lower())
        except Exception:
            pass

    sched_index = _index_schedules_from_csv(s_csv, names=needed_names)

    X_all, Y_all, M_all = [], [], []
    for _, row in df_csv.iterrows():
        try:
            w = json.loads(row["json_content"])
            s_name = w.get("schedule_file") or w.get("schedule file")
            s_doc = _find_schedule(sched_index, s_name)
            c_name = w.get("combination")
            infps_map = _build_infps_lookup(s_doc, c_name) if s_doc and c_name else None
            X_feat, (y1, y2), meta = featurize_window(w, infps_map)
            X_all.append(X_feat)
            Y_all.append({"y1": y1, "y2": y2})
            meta["schedule_file"] = s_name
            M_all.append(meta)
        except Exception:
            pass
    
    X = pd.DataFrame(X_all).fillna(0.0)
    Y = pd.DataFrame(Y_all)
    M = pd.DataFrame(M_all)

    # We also need pred_score
    b1, b2, feats, mode, alpha = load_models(Path("xgboost_model/artifacts/gpu/xgb_model_x3_double"))
    X = X.reindex(columns=feats, fill_value=0.0)
    pred_scores = b1.predict(X)
    
    df = pd.DataFrame({
        'schedule_file': M['schedule_file'],
        'combination': M['combination'],
        'actual_T_norm': Y['y1'],
        'actual_D_norm': Y['y2'],
        'actual_score': Y['y1'] - alpha * Y['y2'],
        'pred_score': pred_scores
    })
    
    scenario_groups = df.groupby('schedule_file')
    
    latency_results = []
    random_results = []
    
    random.seed(42)
    
    for s_file, group in scenario_groups:
        # group.groupby might result in s_file being a tuple or string depending on pandas version/input
        if isinstance(s_file, tuple):
            s_file = s_file[0]
        
        rows = group.to_dict('records')
        m_count = get_model_count_from_sched(s_file, sched_index)
        
        # Latency-based
        latency_comb_name = get_latency_based_combination(rows, results_dir, s_file)
        latency_row = next((r for r in rows if r['combination'] == latency_comb_name), rows[0])
        latency_results.append({
            'schedule_file': s_file,
            'm_count': m_count,
            'actual_T_norm': latency_row['actual_T_norm'],
            'actual_D_norm': latency_row['actual_D_norm'],
            'actual_score': latency_row['actual_score']
        })
        
        # Random search (Best of 5 as a proxy for 'Best-of-5')
        # Actually Best-of-5 is usually random.seed(42) sample 5
        sample_5 = random.sample(rows, min(5, len(rows)))
        best_of_5 = max(sample_5, key=lambda x: x['actual_score'])
        random_results.append({
            'schedule_file': s_file,
            'm_count': m_count,
            'actual_T_norm': best_of_5['actual_T_norm'],
            'actual_D_norm': best_of_5['actual_D_norm'],
            'actual_score': best_of_5['actual_score']
        })

    def save_summary(results, out_path):
        with open(out_path, 'w') as f:
            if not results:
                print(f"Warning: No results to save for {out_path}")
                return
            max_m = max(r['m_count'] for r in results)
            # Find the actual minimum model count present in the results
            min_m = min(r['m_count'] for r in results)
            # Use 3 as the lower bound if possible, or min_m
            start_n = max(3, min_m)
            for n in range(max_m, start_n - 1, -1):
                subset = [r for r in results if r['m_count'] >= n]
                if not subset: continue
                avg_T = np.mean([r['actual_T_norm'] for r in subset])
                avg_D = np.mean([r['actual_D_norm'] for r in subset])
                avg_S = np.mean([r['actual_score'] for r in subset])
                f.write(f"Average (>= {n} models),,{avg_T:.2f},{avg_D:.2f},{avg_S:.2f}\n")
    
    save_summary(latency_results, output_latency_csv)
    save_summary(random_results, output_random_csv)
    print(f"Saved {output_latency_csv} and {output_random_csv}")

if __name__ == "__main__":
    import sys
    main()
