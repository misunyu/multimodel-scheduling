import argparse
import json
import csv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys

"""
[Usage Examples]
1) Run with default settings (Rank mode by default prefix):
   python scripts/xgb_best_selector.py

2) Run with specific model prefix:
   python scripts/xgb_best_selector.py --model_prefix xgboost_model/artifacts/gpu/xgb_model_score

3) Run without clipping:
   python scripts/xgb_best_selector.py --no_clip_pred_score

Note: Default model prefix is xgboost_model/artifacts/gpu/xgb_model_rank.
"""

# Import functions from deploy_selector_catboost_suite.py
sys.path.append(str(Path.cwd() / "xgboost_model"))
try:
    from deploy_selector_catboost_suite import (
        _index_schedules_from_csv
    )
except ImportError as e:
    print(f"Error importing from deploy_selector_catboost_suite: {e}")
    sys.exit(1)

def get_performance_index(results_dir):
    perf_index = {} # schedule_file_name -> { combination_name -> performance_data }
    results_path = Path(results_dir)
    
    for p_file in results_path.glob("*.json"):
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            sched_file = content.get("schedule file") or content.get("schedule_file")
            if not sched_file:
                continue
                
            sched_name = Path(sched_file).name
            
            if sched_name not in perf_index:
                perf_index[sched_name] = {}
                
            data_list = content.get("data", [])
            for item in data_list:
                comb_name = item.get("combination")
                if comb_name:
                    perf_index[sched_name][comb_name] = item
                    
        except Exception as e:
            print(f"Error indexing {p_file}: {e}")
            
    return perf_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_recompute_dir", type=str, default="results_recompute")
    parser.add_argument("--inference_csv", type=str, required=True, help="Path to the inference result CSV from deploy_selector_catboost_suite.py")
    parser.add_argument("--output_csv", type=str, default="catb_best_results.csv")
    args_cli = parser.parse_args()

    results_recompute_dir = args_cli.results_recompute_dir
    inference_csv = args_cli.inference_csv
    output_csv = args_cli.output_csv
    
    print(f"Loading inference results from {inference_csv}...")
    if not Path(inference_csv).exists():
        print(f"Error: {inference_csv} not found.")
        return
    
    df_pred = pd.read_csv(inference_csv)
    
    # Infer mode from filename or columns
    mode = "unknown"
    if "rank" in inference_csv.lower():
        mode = "rank"
    elif "score" in inference_csv.lower():
        mode = "score"
    elif "two_target" in inference_csv.lower():
        mode = "two_target"
    elif "double" in inference_csv.lower():
        mode = "double"
    
    print(f"Detected mode: {mode}")

    print("Indexing performance data from results_recompute...")
    perf_index = get_performance_index(results_recompute_dir)
    print(f"Indexed performance data for {len(perf_index)} schedules.")
    
    results = []
    
    top1_hits = 0
    top5_hits = 0
    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_valid_schedules = 0
    total_valid_schedules_ge3 = 0

    # Group by schedule_file
    for sched_name, group in df_pred.groupby("schedule_file"):
        print(f"Processing schedule: {sched_name}")
        
        # In this group, find predicted best and actual best
        # Group might have multiple timestamps, but usually we care about the set of combinations
        # If there are multiple timestamps, let's take the latest one or assume they are the same scenario
        
        # Sort by predicted score
        group_sorted_pred = group.sort_values(by="pred_score", ascending=False)
        max_pred_score = group_sorted_pred.iloc[0]["pred_score"]
        best_pred_combs = group_sorted_pred[np.isclose(group_sorted_pred["pred_score"].astype(float), float(max_pred_score), rtol=1e-7)]["combination"].tolist()
        
        # Sort by actual score
        group_sorted_actual = group.sort_values(by="actual_score", ascending=False)
        max_actual_score = group_sorted_actual.iloc[0]["actual_score"]
        best_actual_combs = group_sorted_actual[np.isclose(group_sorted_actual["actual_score"].astype(float), float(max_actual_score), rtol=1e-7)]["combination"].tolist()

        total_valid_schedules += 1
        
        # Top-1 Accuracy
        if any(c in best_actual_combs for c in best_pred_combs):
            top1_hits += 1
            
        # Top-5 Accuracy
        # Get unique predicted scores and find threshold for top 5 groups
        unique_scores = sorted(group["pred_score"].unique(), reverse=True)
        top5_threshold = unique_scores[min(4, len(unique_scores)-1)]
        top5_group_combs = group[group["pred_score"] >= top5_threshold - 1e-7]["combination"].tolist()
        
        if any(c in top5_group_combs for c in best_actual_combs):
            top5_hits += 1

        # Prepare display string
        display_parts = []
        for bpc in best_pred_combs:
            # Check if this combination is in best_actual_combs
            if bpc in best_actual_combs:
                display_parts.append(f"<font color='purple'>{bpc}</font>")
            else:
                display_parts.append(f"<font color='red'>{bpc}</font>")
        
        display_comb = ", ".join(display_parts)
        if not any(c in best_actual_combs for c in best_pred_combs) and best_actual_combs:
            actual_str = ", ".join([f"<font color='blue'>{c}</font>" for c in best_actual_combs])
            display_comb += f" (Actual: {actual_str})"

        # Get performance for the first predicted best
        best_pred_row = group_sorted_pred.iloc[0]
        
        # Get additional info from perf_index for model count and averages
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})
        if not sched_perf and pure_sched_name.endswith("_x3.yaml"):
            alt_name = pure_sched_name.replace("_x3.yaml", ".yaml")
            sched_perf = perf_index.get(alt_name, {})
            
        perf_item = sched_perf.get(best_pred_combs[0], {})
        models_count = len(perf_item.get('models', {}))

        if models_count >= 3:
            total_valid_schedules_ge3 += 1
            if any(c in best_actual_combs for c in best_pred_combs):
                top1_hits_ge3 += 1
            if any(c in top5_group_combs for c in best_actual_combs):
                top5_hits_ge3 += 1

        results.append({
            'schedule_file': sched_name,
            'best_combination': display_comb,
            'normalized_throughput': best_pred_row['actual_T_norm'] if not pd.isna(best_pred_row['actual_T_norm']) and best_pred_row['actual_T_norm'] != "" else "-",
            'drop_rate': best_pred_row['actual_D_norm'] if not pd.isna(best_pred_row['actual_D_norm']) and best_pred_row['actual_D_norm'] != "" else "-",
            'score': best_pred_row['actual_score'],
            'models_count': models_count
        })

    if results:
        output_df = pd.DataFrame(results)
        
        # Calculate accuracies
        top1_acc = round(top1_hits / total_valid_schedules, 4) if total_valid_schedules > 0 else 0
        top5_acc = round(top5_hits / total_valid_schedules, 4) if total_valid_schedules > 0 else 0
        
        top1_acc_ge3 = round(top1_hits_ge3 / total_valid_schedules_ge3, 4) if total_valid_schedules_ge3 > 0 else 0
        top5_acc_ge3 = round(top5_hits_ge3 / total_valid_schedules_ge3, 4) if total_valid_schedules_ge3 > 0 else 0

        # Calculate averages (numeric only)
        numeric_throughput = pd.to_numeric(output_df['normalized_throughput'], errors='coerce')
        numeric_drop_rate = pd.to_numeric(output_df['drop_rate'], errors='coerce')
        numeric_score = pd.to_numeric(output_df['score'], errors='coerce')
        
        avg_throughput = round(numeric_throughput.mean(), 2)
        avg_drop_rate = round(numeric_drop_rate.mean(), 2)
        avg_score = round(numeric_score.mean(), 2)
        
        # 모델 개수별 누적 평균값 계산 (>= 3, 4, 5, 6, 7, 8)
        avg_rows = []
        for n in range(8, 2, -1):
            mask = output_df['models_count'] >= n
            subset = output_df[mask]
            if not subset.empty:
                avg_t = round(pd.to_numeric(subset['normalized_throughput'], errors='coerce').mean(), 2)
                avg_d = round(pd.to_numeric(subset['drop_rate'], errors='coerce').mean(), 2)
                avg_s = round(pd.to_numeric(subset['score'], errors='coerce').mean(), 2)
                avg_rows.append([f'Average (>= {n} models)', '', avg_t, avg_d, avg_s])
            else:
                avg_rows.append([f'Average (>= {n} models)', '', 'nan', 'nan', 'nan'])

        # Create output filename with mode suffix
        if output_csv == "catb_best_results.csv":
            output_csv = f"catb_best_results_{mode}.csv"
            
        output_path = Path("experimental_results") / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Top-1 Accuracy', top1_acc])
            writer.writerow(['Top-5 Accuracy', top5_acc])
            writer.writerow(['Top-1 Accuracy (>= 3 models)', top1_acc_ge3])
            writer.writerow(['Top-5 Accuracy (>= 3 models)', top5_acc_ge3])
            for row in avg_rows:
                writer.writerow(row)
            writer.writerow(['Average', '', avg_throughput, avg_drop_rate, avg_score])
            writer.writerow(['schedule_file', 'best_combination', 'normalized_throughput', 'drop_rate', 'score'])
            
            for _, row in output_df.iterrows():
                writer.writerow([
                    row['schedule_file'],
                    row['best_combination'],
                    row['normalized_throughput'],
                    row['drop_rate'],
                    row['score']
                ])
                
        print(f"Saved {len(results)} results to {output_csv}")
        print(f"Average Throughput: {avg_throughput}, Drop Rate: {avg_drop_rate}, Score: {avg_score}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
