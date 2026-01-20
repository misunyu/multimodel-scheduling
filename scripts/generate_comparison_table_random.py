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
    # Mapping model name to tag (same as in get_latency_based_comb.py)
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

    # Use prediction_result filename to infer rate
    # But schedule_file is model_schedules_...yaml
    suffix = ""
    if "_x2" in schedule_file: suffix = "_x2"
    elif "_x3" in schedule_file: suffix = "_x3"

    # Find unique models in this schedule
    # We need a representative row to see what models are in this schedule.
    # The prediction CSV doesn't have per-model execution info easily, but we can 
    # look at the first row's JSON content if available. 
    # Wait, the prediction CSV doesn't have the models dict.
    # I should find the original performance file for this schedule.
    
    perf_file_pattern = schedule_file.replace(".yaml", ".json")
    perf_files = list(Path(results_dir).glob(f"performance_*_{perf_file_pattern}"))
    if not perf_files:
        return "combination_1"
    
    perf_file = sorted(perf_files)[-1]
    with open(perf_file, 'r') as f:
        perf_data = json.load(f)
    
    if not perf_data.get('data'):
        return "combination_1"
    
    # Get model info from the first entry of performance file
    first_entry = perf_data['data'][0]
    models_in_schedule = first_entry.get('models', {})
    
    model_best_devices = {} # (model_name, view_key) -> best_device
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

    # Now find the combination in perf_data that matches all best devices
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

def main():
    prediction_csv = Path("xgboost_model/prediction_result/gpu/prediction_result_test_random.csv")
    results_dir = "results"
    
    if not prediction_csv.exists():
        print(f"Error: {prediction_csv} not found.")
        return

    # Skip comment lines
    df = pd.read_csv(prediction_csv, comment='#')
    
    # Group by schedule_file to define a scenario
    scenario_groups = df.groupby(['schedule_file'])
    
    strategies = {
        "Latency-based (Greedy)": {"T": [], "D": [], "S": []},
        "Random Search (k=10)": {"T": [], "D": [], "S": []},
        "Predictive (Ours)": {"T": [], "D": [], "S": []},
        "Oracle (Upper Bound)": {"T": [], "D": [], "S": []}
    }

    random.seed(42) # For reproducibility

    for s_file_tuple, group in scenario_groups:
        s_file = s_file_tuple[0] if isinstance(s_file_tuple, tuple) else s_file_tuple
        # Convert group to list of dicts for easier handling
        rows = group.to_dict('records')
        
        # 1. Oracle (Actual best in this model set)
        oracle = max(rows, key=lambda x: x["actual_score"])
        
        # 2. Predictive (Ours - best according to predicted score in this model set)
        predictive = max(rows, key=lambda x: x["pred_score"])
        
        # 3. Latency-based (Greedy)
        # Identify optimal device for each model and find the corresponding combination
        latency_comb_name = get_latency_based_combination(rows, results_dir, s_file)
        
        latency_based = None
        for r in rows:
            if r["combination"] == latency_comb_name:
                latency_based = r
                break
        if not latency_based:
            # Fallback to combination_1 if not found in prediction CSV
            for r in rows:
                if r["combination"] == "combination_1":
                    latency_based = r
                    break
        if not latency_based:
            latency_based = rows[0]
            
        # 4. Random Search (k=10)
        random_sample = random.sample(rows, min(10, len(rows)))
        random_best = max(random_sample, key=lambda x: x["actual_score"])
        
        # Store results
        for strat, res in [
            ("Oracle (Upper Bound)", oracle),
            ("Predictive (Ours)", predictive),
            ("Latency-based (Greedy)", latency_based),
            ("Random Search (k=10)", random_best)
        ]:
            # Normalize Throughput and Score to Oracle
            strategies[strat]["T"].append(res["actual_T_norm"] / oracle["actual_T_norm"] if oracle["actual_T_norm"] > 0 else 0)
            strategies[strat]["S"].append(res["actual_score"] / oracle["actual_score"] if oracle["actual_score"] > 0 else 0)
            # Drop rate is already 0 to 1
            strategies[strat]["D"].append(res["actual_D_norm"])

    # Calculate final averages and normalize Drop Rate relative to Oracle
    oracle_avg_D = np.mean(strategies["Oracle (Upper Bound)"]["D"])

    print("\\begin{table}[t] ")
    print("\\centering ")
    print("\\caption{Comparison of placement strategies on the CPU--GPU platform under high-load conditions ($3\\times$). Results are normalized to the Oracle's performance.} ")
    print("\\label{tab:comparison_gpu} ")
    print("\\setlength{\\tabcolsep}{8pt} ")
    print("\\renewcommand{\\arraystretch}{1.2} \\begin{tabular}{cccc}")
    print("\\hline ")
    print("\\multirow{2}{*}{\\textbf{Strategy}} & \\textbf{Norm.} & \\textbf{Norm.} & \\textbf{Overall}\\tabularnewline")
    print(" & \\textbf{Throughput} & \\textbf{Drop Rate} & \\textbf{Score}\\tabularnewline")
    print("\\hline ")
    
    order = [
        "Latency-based (Greedy)",
        "Random Search (k=10)",
        "Predictive (Ours)",
        "Oracle (Upper Bound)"
    ]

    for strat in order:
        avg_T = np.mean(strategies[strat]["T"])
        avg_D = np.mean(strategies[strat]["D"]) - oracle_avg_D
        avg_S = np.mean(strategies[strat]["S"])
        
        print(f"\\textbf{{{strat}}} & {avg_T:.2f} & {max(0, avg_D):.2f} & {avg_S:.2f}\\tabularnewline")
        
    print("\\hline ")
    print("\\end{tabular}")
    print("\\end{table} ")

if __name__ == "__main__":
    main()

