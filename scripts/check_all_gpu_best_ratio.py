import os
import json
import glob

def check_all_gpu_best_ratio():
    results_dir = "results_recompute"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    json_files.sort()

    total_valid_files = 0
    all_gpu_best_count = 0
    
    # x3 specific statistics
    total_x3_valid_files = 0
    all_gpu_x3_best_count = 0
    
    results_summary = []

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        is_x3_file = "_x3.json" in file_name or "_x3-" in file_name or file_name.endswith("_x3.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            results_summary.append(f"{file_name} | best=N/A | SKIPPED")
            continue

        best_deployment = data.get("best deployment")
        if not best_deployment:
            results_summary.append(f"{file_name} | best=N/A | SKIPPED")
            continue

        total_valid_files += 1
        if is_x3_file:
            total_x3_valid_files += 1
            
        found_combination = False
        is_all_gpu = False

        combinations = data.get("data", [])
        for entry in combinations:
            if entry.get("combination") == best_deployment:
                found_combination = True
                models = entry.get("models", {})
                
                if not models:
                    is_all_gpu = False
                    break

                all_gpu = True
                for model_info in models.values():
                    if model_info.get("execution") != "GPU":
                        all_gpu = False
                        break
                
                is_all_gpu = all_gpu
                break

        if not found_combination:
            results_summary.append(f"{file_name} | best={best_deployment} | SKIPPED")
            total_valid_files -= 1 # Not a valid data entry for statistics
            if is_x3_file:
                total_x3_valid_files -= 1
            continue

        if is_all_gpu:
            all_gpu_best_count += 1
            if is_x3_file:
                all_gpu_x3_best_count += 1
            status = "ALL_GPU"
        else:
            status = "NOT_ALL_GPU"
        
        results_summary.append(f"{file_name} | best={best_deployment} | {status}")

    # Output individual results
    for line in results_summary:
        print(line)

    # Output statistics
    percentage = (all_gpu_best_count / total_valid_files * 100) if total_valid_files > 0 else 0
    percentage_x3 = (all_gpu_x3_best_count / total_x3_valid_files * 100) if total_x3_valid_files > 0 else 0
    
    print(f"total_valid_files={total_valid_files}")
    print(f"all_gpu_best_count={all_gpu_best_count}")
    print(f"percentage={percentage:.1f}%")
    print("-" * 20)
    print(f"total_x3_valid_files={total_x3_valid_files}")
    print(f"all_gpu_x3_best_count={all_gpu_x3_best_count}")
    print(f"percentage_x3={percentage_x3:.1f}%")

if __name__ == "__main__":
    check_all_gpu_best_ratio()
