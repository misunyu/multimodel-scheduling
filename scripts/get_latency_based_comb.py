import json
import os
import re

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
    for f in os.listdir(results_dir):
        if f.startswith("performance_") and f.endswith(".json"):
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

def get_latency_based_combination(input_path):
    results_dir = os.path.dirname(input_path) or '.'
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
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
        
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
            tag_to_best_device[tag] = 'CPU'
            
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

    for entry in input_data.get('data', []):
        if match_combination(entry):
            return entry.get('combination')
    
    return None

files = [
    "results/performance_20251224_071639_model_schedules_g_t_test.json",
    "results/performance_20251224_071047_model_schedules_g_r_y_x2.json",
    "results/performance_20251224_081235_model_schedules_m_r_r_t_y_x3.json",
    "results/performance_20251223_180231_model_schedules_g_m_r_r_s_s_t_v_x3.json"
]

for f in files:
    comb = get_latency_based_combination(f)
    print(f"{f}: {comb}")
