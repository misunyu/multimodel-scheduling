import json
import csv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys

# Import functions from deploy_selector_xgb_suite.py
sys.path.append(str(Path.cwd() / "xgboost_model"))
try:
    from deploy_selector_xgb_suite import (
        load_models, 
        featurize_from_combo, 
        _index_schedules_from_csv
    )
    import xgboost as xgb
except ImportError as e:
    print(f"Error importing from deploy_selector_xgb_suite: {e}")
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
    results_recompute_dir = "results_recompute"
    test_schedules_csv = "xgboost_model/dataset/gpu/test_schedules_x3.csv"
    model_prefix = "xgboost_model/artifacts/gpu/xgb_model_x3"
    output_csv = "xgb_best_results.csv"
    alpha = 0.2
    
    print("Loading XGBoost models...")
    try:
        model, feats = load_models(Path(model_prefix))
    except Exception as e:
        print(f"Error loading models from {model_prefix}: {e}")
        return

    print("Indexing performance data from results_recompute...")
    perf_index = get_performance_index(results_recompute_dir)
    print(f"Indexed performance data for {len(perf_index)} schedules.")
    
    print("Loading test schedules...")
    sched_index = _index_schedules_from_csv(Path(test_schedules_csv))
    print(f"Loaded {len(sched_index)} schedules from {test_schedules_csv}")

    results = []
    
    for sched_name, sched_doc in sched_index.items():
        print(f"Processing schedule: {sched_name}")
        
        best_pred_comb = None
        max_pred_score = -float('inf')
        
        # Iterate over all combinations in the schedule
        # The YAML structure has combinations as top-level keys
        for comb_name, combo_blob in sched_doc.items():
            if not isinstance(combo_blob, dict): continue
            
            try:
                # Featurize
                df_X = featurize_from_combo(combo_blob)
                
                # Ensure all features are present and in correct order
                for c in feats:
                    if c not in df_X.columns:
                        df_X[c] = 0.0
                df_X = df_X[feats]
                
                # Predict
                pred_score = float(model.predict(df_X)[0])
                
                if pred_score > max_pred_score:
                    max_pred_score = pred_score
                    best_pred_comb = comb_name
            except Exception as e:
                print(f"  [WARN] Error predicting for {comb_name}: {e}")
                continue
        
        if not best_pred_comb:
            print(f"  [LOG] Could not find any valid combination for {sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': '-',
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue

        # Get actual performance for the predicted best combination
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})
        perf_item = sched_perf.get(best_pred_comb)
        
        if not perf_item:
            print(f"  [LOG] Predicted best combination '{best_pred_comb}' not found in actual results for {pure_sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': best_pred_comb,
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        derived = perf_item.get('derived', {})
        models_count = len(perf_item.get('models', {}))
        results.append({
            'schedule_file': sched_name,
            'best_combination': best_pred_comb,
            'normalized_throughput': derived.get('throughput_norm'),
            'drop_rate': derived.get('drop_rate_norm'),
            'score': perf_item.get('score'),
            'models_count': models_count
        })

    if results:
        output_df = pd.DataFrame(results)
        
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

        output_path = Path("experimental_results") / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
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
