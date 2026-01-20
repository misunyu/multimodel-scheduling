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
    if not input_data.get('data'): return None
    
    input_filename = os.path.basename(input_path)
    input_tags = get_model_tags(input_filename)
    if len(input_tags) != 3: return None
    
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
                    if k in model_name: tag = v; break
            if tag and tag in tag_to_best_device:
                if execution != tag_to_best_device[tag]: return False
        return True

    latency_based_tp = 0
    best_tp = 0
    for entry in input_data.get('data', []):
        total_tp = entry.get('total', {}).get('total_throughput_fps', 0)
        if entry.get('combination') == best_deploy_id: best_tp = total_tp
        if match_combination(entry): latency_based_tp = total_tp
    
    return best_tp, latency_based_tp

def main():
    model_name_to_tag = {
        'gpt2': 'g', 'mnasnet': 'm', 'resnet50': 'r', 'resnext50': 'r',
        'shufflenet-v2-12': 's', 'shufflenet': 's', 'tiny-llama-chat-onnx': 't',
        'vgg16': 'v', 'yolov3': 'y'
    }
    results_dir = 'results'
    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith('performance_') and f.endswith('.json')]
    
    comparison_results = []
    for f in files:
        res = process_file(f, model_name_to_tag)
        if res:
            best_tp, latency_based_tp = res
            if best_tp > 0:
                diff = best_tp - latency_based_tp
                ratio = latency_based_tp / best_tp
                comparison_results.append({
                    'file': f,
                    'best': best_tp,
                    'latency': latency_based_tp,
                    'diff': diff,
                    'ratio': ratio
                })
    
    # Sort by diff descending
    comparison_results.sort(key=lambda x: x['diff'], reverse=True)
    
    print(f"{'File':<60} | {'Best':<8} | {'Latency':<8} | {'Diff':<8} | {'Ratio':<8}")
    print("-" * 100)
    for r in comparison_results[:10]:
        print(f"{os.path.basename(r['file']):<60} | {r['best']:<8.2f} | {r['latency']:<8.2f} | {r['diff']:<8.2f} | {r['ratio']:<8.2f}")

if __name__ == "__main__":
    main()
