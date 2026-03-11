import os
import json
import glob
import argparse

def recompute_scores(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_files = glob.glob(os.path.join(input_dir, 'performance_*.json'))
    
    for file_path in json_files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data_content = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing {file_path}: {e}. Skipping.")
            continue
            
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
            
        # 2) Calculate scores and find best deployment for various alphas
        alphas = [round(i * 0.1, 1) for i in range(1, 11)]
        best_deployments = {alpha: {'score': -float('inf'), 'comb': None} for alpha in alphas}
        
        for entry in data_content['data']:
            total_throughput = entry.get('total', {}).get('total_throughput_fps', 0.0)
            drop_rate = entry.get('derived', {}).get('drop_rate_fps', 0.0)
            
            # Calculate normalized values
            throughput_norm = total_throughput / fmax
            drop_rate_norm = drop_rate / dmax
            
            # Add to derived field
            if 'derived' not in entry:
                entry['derived'] = {}
            entry['derived']['throughput_norm'] = throughput_norm
            entry['derived']['drop_rate_norm'] = drop_rate_norm
            
            # Current behavior: score = throughput_norm - 0.2 * drop_rate_norm
            score_02 = throughput_norm - 0.2 * drop_rate_norm
            entry['score'] = score_02
            
            # Calculate scores for all alphas and track best
            for alpha in alphas:
                s = throughput_norm - alpha * drop_rate_norm
                if s > best_deployments[alpha]['score']:
                    best_deployments[alpha]['score'] = s
                    best_deployments[alpha]['comb'] = entry.get('combination')
        
        # Add best deployments to data_content
        for alpha in alphas:
            if best_deployments[alpha]['comb']:
                key = f'best deployment-{alpha}'
                data_content[key] = best_deployments[alpha]['comb']
        
        # Keep 'best deployment' for backward compatibility (alpha=0.2)
        if best_deployments[0.2]['comb']:
            data_content['best deployment'] = best_deployments[0.2]['comb']
            
        # Save the result
        output_filename = "recompute_" + os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, output_filename)
        with open(output_file_path, 'w') as f:
            json.dump(data_content, f, indent=4)
        print(f"Saved recomputed results to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute scores for performance results.")
    parser.add_argument("--input_dir", default="results", help="Input directory containing JSON files")
    parser.add_argument("--output_dir", default="results_recompute", help="Output directory for recomputed JSON files")
    
    args = parser.parse_args()
    recompute_scores(args.input_dir, args.output_dir)
