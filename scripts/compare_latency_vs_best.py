#python3 scripts/compare_latency_vs_best.py results/performance_20251224_071639_model_schedules_g_t_test.json results/performance_20251224_071047_model_schedules_g_r_y_x2.json results/performance_20251224_081235_model_schedules_m_r_r_t_y_x3.json results/performance_20251223_180231_model_schedules_g_m_r_r_s_s_t_v_x3.json

### Latency-based Selection 결과
#1.  **results/performance_20251224_071639_model_schedules_g_t_test.json**
#    *   Combination: **combination_4**
#2.  **results/performance_20251224_071047_model_schedules_g_r_y_x2.json**
#    *   Combination: **combination_1**
#3.  **results/performance_20251224_081235_model_schedules_m_r_r_t_y_x3.json**
#    *   Combination: **combination_18**
#4.  **results/performance_20251223_180231_model_schedules_g_m_r_r_s_s_t_v_x3.json**
#    *   Combination: **combination_89**

### 확인 방법
#`scripts/get_latency_based_comb.py`

import json
import os
import argparse
import matplotlib.pyplot as plt
import re

def get_model_tags(filename):
    # performance_20251224_045508_model_schedules_g_m_r_r_s_s_t.json -> g, m, r, r, s, s, t
    match = re.search(r'model_schedules_(.*?)\.json', filename)
    if match:
        tag_str = match.group(1)
        # remove x2, x3, test suffixes
        tag_str = re.sub(r'(_x\d+|_test)', '', tag_str)
        return tag_str.split('_')
    return []

def get_input_rate_suffix(filename):
    if '_x2' in filename: return '_x2'
    if '_x3' in filename: return '_x3'
    return ''

def find_single_model_file(results_dir, tag, suffix):
    candidates = []
    for f in os.listdir(results_dir):
        if f.startswith("performance_") and f.endswith(".json"):
            # Ensure it matches the input rate suffix
            if suffix and suffix not in f: continue
            if not suffix and ('_x2' in f or '_x3' in f): continue
            
            f_tags = get_model_tags(f)
            if len(f_tags) == 1 and f_tags[0] == tag:
                candidates.append(f)
    
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(results_dir, candidates[-1])

def get_best_device_for_model(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    best_throughput = -1
    best_device = None
    
    for entry in data.get('data', []):
        # In single model file, there might be CPU and GPU combinations
        # We look at the total_throughput_fps or the specific model's throughput
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

def process_file(input_path, results_dir, model_name_to_tag):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
        
    best_deploy_id = input_data.get('best deployment')
    
    if not input_data.get('data'):
        print(f"No data in input file: {input_path}")
        return None

    input_filename = os.path.basename(input_path)
    input_tags = get_model_tags(input_filename)
    input_rate_suffix = get_input_rate_suffix(input_filename)
    
    tag_to_best_device = {}
    unique_tags = set(input_tags)
    for tag in unique_tags:
        single_file = find_single_model_file(results_dir, tag, input_rate_suffix)
        if single_file:
            best_device = get_best_device_for_model(single_file)
            tag_to_best_device[tag] = best_device
        else:
            tag_to_best_device[tag] = 'CPU' # Default
            
    def match_combination(entry):
        models = entry.get('models', {})
        for view_name, info in models.items():
            model_name = info.get('model')
            execution = info.get('execution', 'CPU').upper()
            tag = model_name_to_tag.get(model_name)
            if not tag:
                for k, v in model_name_to_tag.items():
                    if k in model_name:
                        tag = v
                        break
            
            if tag and tag in tag_to_best_device:
                if execution != tag_to_best_device[tag]:
                    return False
        return True

    latency_based_throughput = 0
    best_throughput = 0
    
    for entry in input_data.get('data', []):
        comb_name = entry.get('combination')
        total_tp = entry.get('total', {}).get('total_throughput_fps', 0)
        
        if comb_name == best_deploy_id:
            best_throughput = total_tp
            
        if match_combination(entry):
            latency_based_throughput = total_tp

    return best_throughput, latency_based_throughput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', help='Path(s) to the performance JSON file(s)')
    args = parser.parse_args()
    
    model_name_to_tag = {
        'gpt2': 'g',
        'mnasnet': 'm',
        'resnet50': 'r',
        'resnext50': 'r',
        'shufflenet-v2-12': 's',
        'shufflenet': 's',
        'tiny-llama-chat-onnx': 't',
        'vgg16': 'v',
        'yolov3': 'y'
    }
    
    results_list = []
    file_labels = []
    
    for input_path in args.input_files:
        results_dir = os.path.dirname(input_path) or '.'
        res = process_file(input_path, results_dir, model_name_to_tag)
        if res:
            best_tp, latency_based_tp = res
            results_list.append((best_tp, latency_based_tp))
            file_labels.append(os.path.basename(input_path))

    if not results_list:
        print("No valid data processed.")
        return

    # Plotting
    import numpy as np
    from matplotlib import rcParams

    # Set font to Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['hatch.linewidth'] = 0.3 # Thinner hatch lines
    
    n_files = len(results_list)
    x = np.arange(n_files)
    width = 0.35
    
    best_relative = [1.0] * n_files
    latency_based_relative = []
    processed_labels = []
    for i, (best_tp, latency_based_tp) in enumerate(results_list):
        if best_tp > 0:
            latency_based_relative.append(latency_based_tp / best_tp)
        else:
            latency_based_relative.append(0.0)
        
        # Format labels: (G, M, R)
        tags = [t.upper() for t in get_model_tags(file_labels[i])]
        processed_labels.append(f"({', '.join(tags)})")

    plt.figure(figsize=(max(4, n_files * 1.2), 2.67))
    
    edge_color1 = '#8888FF' # Lighter blue
    edge_color2 = '#FF8888' # Lighter red

    bar1 = plt.bar(x - width/2, best_relative, width, label='Oracle Placement', color='skyblue', alpha=0.5, hatch='//', edgecolor=edge_color1, linewidth=0.5)
    bar2 = plt.bar(x + width / 2, latency_based_relative, width, label='Latency-based Heuristic', color='lightcoral', alpha=0.5, hatch='..', edgecolor=edge_color2, linewidth=0.5)
    
    plt.ylabel('Normalized Throughput')
    plt.xticks(x, processed_labels, rotation=0, fontsize=6)
    leg = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8, frameon=False)
    
    all_vals = best_relative + latency_based_relative
    plt.ylim(0, max(all_vals) * 1.2 if all_vals else 1.2)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    autolabel(bar1)
    autolabel(bar2)

    plt.tight_layout()
    output_pdf = "multi_throughput_comparison.pdf"
    plt.savefig(output_pdf)
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    main()
