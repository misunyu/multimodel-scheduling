import json
import os
import glob
import csv

def find_better_throughput(results_dir):
    performance_files = glob.glob(os.path.join(results_dir, "performance_*.json"))
    
    all_results = []
    
    for file_path in performance_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        best_deployment_id = data.get("best deployment")
        if not best_deployment_id:
            continue
            
        combinations_data = data.get("data", [])
        
        # Find throughput and drop rate of the best deployment
        best_throughput = None
        best_drop_rate = None
        for entry in combinations_data:
            if entry.get("combination") == best_deployment_id:
                best_throughput = entry.get("total", {}).get("total_throughput_fps")
                best_drop_rate = entry.get("derived", {}).get("drop_rate_fps")
                break
        
        if best_throughput is None:
            continue
            
        # Check other combinations
        for entry in combinations_data:
            comb_id = entry.get("combination")
            if comb_id == best_deployment_id:
                continue
                
            throughput = entry.get("total", {}).get("total_throughput_fps", 0)
            drop_rate = entry.get("derived", {}).get("drop_rate_fps")
            
            if throughput > best_throughput:
                all_results.append({
                    "directory": results_dir,
                    "filename": filename,
                    "best_deployment": best_deployment_id,
                    "best_throughput": best_throughput,
                    "best_drop_rate": best_drop_rate,
                    "better_combination": comb_id,
                    "better_throughput": throughput,
                    "better_drop_rate": drop_rate,
                    "diff": throughput - best_throughput
                })
                
    return all_results

if __name__ == "__main__":
    results = []
    print("Checking 'results' directory...")
    results.extend(find_better_throughput("results"))
    print("Checking 'results_recompute' directory...")
    results.extend(find_better_throughput("results_recompute"))
    
    if results:
        # Sort results by throughput difference in descending order
        results.sort(key=lambda x: x['diff'], reverse=True)
        
        output_file = "better_throughput_results.csv"
        keys = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nFound {len(results)} cases where a combination has higher throughput than the best deployment.")
        print(f"Results saved to {output_file}")
    else:
        print("\nNo cases found where another combination has higher throughput than the best deployment.")
