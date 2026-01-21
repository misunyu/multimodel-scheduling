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
        m1, m2, feats = load_models(Path(model_prefix))
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
    
    top1_hits = 0
    top5_hits = 0
    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_valid_schedules = 0
    total_valid_schedules_ge3 = 0

    for sched_name, sched_doc in sched_index.items():
        print(f"Processing schedule: {sched_name}")
        
        # Store all combinations with their predicted scores
        all_pred_results = []
        
        # Iterate over all combinations in the schedule
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
                y1_pred = float(m1.predict(df_X)[0])
                y2_pred = float(m2.predict(df_X)[0])
                pred_score = y1_pred - alpha * y2_pred
                
                all_pred_results.append({
                    'combination': comb_name,
                    'pred_score': pred_score
                })
            except Exception as e:
                print(f"  [WARN] Error predicting for {comb_name}: {e}")
                continue
        
        if not all_pred_results:
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

        # Sort by predicted score
        all_pred_results.sort(key=lambda x: x['pred_score'], reverse=True)
        max_pred_score = all_pred_results[0]['pred_score']
        
        # Get all predicted best combinations (those with max score)
        best_pred_combs = [r['combination'] for r in all_pred_results if math.isclose(r['pred_score'], max_pred_score, rel_tol=1e-7)]
        
        # Find ACTUAL best combination for this schedule
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})
        if not sched_perf:
            # Try removing _x3 suffix from pure_sched_name
            if pure_sched_name.endswith("_x3.yaml"):
                alt_name = pure_sched_name.replace("_x3.yaml", ".yaml")
                sched_perf = perf_index.get(alt_name, {})
        
        # Actual best combination(s) - handle ties in actual scores too if any
        best_actual_combs = []
        max_actual_score = -float('inf')
        for comb_name, perf_item in sched_perf.items():
            actual_score = perf_item.get('score', -float('inf'))
            if math.isclose(actual_score, max_actual_score, rel_tol=1e-7):
                best_actual_combs.append(comb_name)
            elif actual_score > max_actual_score:
                max_actual_score = actual_score
                best_actual_combs = [comb_name]

        total_valid_schedules += 1
        
        # Top-1 Accuracy: If one of predicted best is in actual best
        # Actually, user said: "one of the best_combinations is actual best_deployment"
        if any(c in best_actual_combs for c in best_pred_combs):
            top1_hits += 1
            
        # Top-5 Accuracy: Top-5 groups (where a group has same score)
        # Find combinations in Top-5 score groups
        pred_scores_sorted = sorted(list(set([r['pred_score'] for r in all_pred_results])), reverse=True)
        top5_threshold_score = pred_scores_sorted[min(4, len(pred_scores_sorted)-1)]
        top5_group_combs = [r['combination'] for r in all_pred_results if r['pred_score'] >= top5_threshold_score - 1e-7]
        
        if any(c in top5_group_combs for c in best_actual_combs):
            top5_hits += 1

        # Prepare display string for best combinations
        display_parts = []
        for bpc in best_pred_combs:
            if bpc in best_actual_combs:
                display_parts.append(f"<font color='purple'>{bpc}</font>")
            else:
                display_parts.append(f"<font color='red'>{bpc}</font>")
        
        display_comb = ", ".join(display_parts)
        
        # If no predicted best matches actual best, show actual best in blue
        if not any(c in best_actual_combs for c in best_pred_combs) and best_actual_combs:
            actual_str = ", ".join([f"<font color='blue'>{c}</font>" for c in best_actual_combs])
            display_comb += f" (Actual: {actual_str})"

        # Get performance for the first predicted best combination (for simplicity in other columns)
        perf_item = sched_perf.get(best_pred_combs[0])
        
        if not perf_item:
            print(f"  [LOG] Predicted best combination '{best_pred_combs[0]}' not found in actual results for {pure_sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': display_comb,
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        derived = perf_item.get('derived', {})
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
            'normalized_throughput': derived.get('throughput_norm'),
            'drop_rate': derived.get('drop_rate_norm'),
            'score': perf_item.get('score'),
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
