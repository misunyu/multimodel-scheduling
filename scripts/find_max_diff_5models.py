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
    if not candidates: return None
    candidates.sort()
    return os.path.join(results_dir, candidates[-1])

def get_best_device_for_model(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    best_throughput = -1
    best_device = None
    for entry in data.get('data', []):
        models = entry.get('models', {})
        if len(models) != 1: continue
        view_key = list(models.keys())[0]
        model_info = models[view_key]
        throughput = model_info.get('throughput_fps', 0)
        device = model_info.get('execution', 'CPU').upper()
        if throughput > best_throughput:
            best_throughput = throughput
            best_device = device
    return best_device

def process_file(input_path, model_name_to_tag):
    results_dir = os.path.dirname(input_path) or '.'
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    best_deploy_id = input_data.get('best deployment')
    input_filename = os.path.basename(input_path)
    input_tags = get_model_tags(input_filename)
    input_rate_suffix = get_input_rate_suffix(input_filename)
    
    tag_to_best_device = {}
    for tag in set(input_tags):
        single_file = find_single_model_file(results_dir, tag, input_rate_suffix)
        if single_file:
            tag_to_best_device[tag] = get_best_device_for_model(single_file)
        else:
            tag_to_best_device[tag] = 'CPU'

    def match_combination(entry):
        models = entry.get('models', {})
        for info in models.values():
            model_name = info.get('model')
            execution = info.get('execution', 'CPU').upper()
            tag = model_name_to_tag.get(model_name)
            if not tag:
                for k, v in model_name_to_tag.items():
                    if k in model_name:
                        tag = v
                        break
            if tag and tag in tag_to_best_device:
                if execution != tag_to_best_device[tag]: return False
        return True

    best_tp = 0
    latency_tp = 0
    for entry in input_data.get('data', []):
        total_tp = entry.get('total', {}).get('total_throughput_fps', 0)
        if entry.get('combination') == best_deploy_id:
            best_tp = total_tp
        if match_combination(entry):
            latency_tp = total_tp
    return best_tp, latency_tp

def main():
    model_name_to_tag = {
        'gpt2': 'g', 'mnasnet': 'm', 'resnet50': 'r', 'resnext50': 'r',
        'shufflenet-v2-12': 's', 'shufflenet': 's', 'tiny-llama-chat-onnx': 't',
        'vgg16': 'v', 'yolov3': 'y'
    }
    
    files = [
        "results/performance_20251224_061843_model_schedules_g_m_r_r_t_test.json",
        "results/performance_20251224_062543_model_schedules_g_m_r_r_t_x2.json",
        "results/performance_20251224_063243_model_schedules_g_m_r_r_t_x3.json",
        "results/performance_20251224_063943_model_schedules_g_m_r_r_t.json",
        "results/performance_20251224_072143_model_schedules_m_r_r_s_s_test.json",
        "results/performance_20251224_072843_model_schedules_m_r_r_s_s_x2.json",
        "results/performance_20251224_073543_model_schedules_m_r_r_s_s_x3.json",
        "results/performance_20251224_074243_model_schedules_m_r_r_s_s.json",
        "results/performance_20251224_075835_model_schedules_m_r_r_t_y_test.json",
        "results/performance_20251224_080535_model_schedules_m_r_r_t_y_x2.json",
        "results/performance_20251224_081235_model_schedules_m_r_r_t_y_x3.json",
        "results/performance_20251224_081936_model_schedules_m_r_r_t_y.json",
        "results/performance_20251224_082636_model_schedules_m_r_r_v_y_test.json",
        "results/performance_20251224_083336_model_schedules_m_r_r_v_y_x2.json",
        "results/performance_20251224_084037_model_schedules_m_r_r_v_y_x3.json",
        "results/performance_20251224_084738_model_schedules_m_r_r_v_y.json",
        "results/performance_20251224_091230_model_schedules_r_r_s_s_t_test.json"
    ]
    
    results = []
    for f in files:
        if os.path.exists(f):
            btp, ltp = process_file(f, model_name_to_tag)
            diff = btp - ltp
            ratio = ltp / btp if btp > 0 else 0
            results.append((f, btp, ltp, diff, ratio))
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Filename':<60} | {'Best':<6} | {'Latency':<7} | {'Diff':<6} | {'Ratio':<5}")
    print("-" * 95)
    for res in results:
        print(f"{os.path.basename(res[0]):<60} | {res[1]:<6.2f} | {res[2]:<7.2f} | {res[3]:<6.2f} | {res[4]:<5.2f}")

if __name__ == "__main__":
    main()
