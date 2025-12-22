import os
import json
import glob

def recompute_scores():
    input_dir = 'results'
    output_dir = 'results_recompute'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_files = glob.glob(os.path.join(input_dir, 'performance_*.json'))
    
    for file_path in json_files:
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            data_content = json.load(f)
            
        if 'data' not in data_content or not data_content['data']:
            print(f"Skipping {file_path}: No data found.")
            continue
            
        # 1) Find fmax and dmax
        fmax = 0.0
        dmax = 0.0
        
        for entry in data_content['data']:
            total_throughput = entry.get('total', {}).get('total_throughput_fps', 0.0)
            drop_rate = entry.get('derived', {}).get('drop_rate_fps', 0.0)
            
            if total_throughput > fmax:
                fmax = total_throughput
            if drop_rate > dmax:
                dmax = drop_rate
        
        # Avoid division by zero
        if fmax == 0:
            fmax = 1.0
        if dmax == 0:
            dmax = 1.0
            
        # 2) Calculate scores and find best deployment
        best_score = -float('inf')
        best_combination = None
        
        for entry in data_content['data']:
            total_throughput = entry.get('total', {}).get('total_throughput_fps', 0.0)
            drop_rate = entry.get('derived', {}).get('drop_rate_fps', 0.0)
            
            # score = (total_throughput_fps / fmax) - 0.2 * (drop_rate_fps / dmax)
            score = round((total_throughput / fmax) - 0.2 * (drop_rate / dmax), 2)
            entry['score'] = score
            
            if score > best_score:
                best_score = score
                best_combination = entry.get('combination')
                
        if best_combination:
            data_content['best deployment'] = best_combination
            
        # Save the result
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, 'w') as f:
            json.dump(data_content, f, indent=4)
        print(f"Saved recomputed results to {output_file_path}")

if __name__ == "__main__":
    recompute_scores()
