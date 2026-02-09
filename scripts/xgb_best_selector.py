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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_recompute_dir", type=str, default="results_recompute")
    parser.add_argument("--test_schedules_csv", type=str, default="xgboost_model/dataset/gpu/test_schedules_x3.csv")
    parser.add_argument("--model_prefix", type=str, default="xgboost_model/artifacts/gpu/xgb_model_rank")
    parser.add_argument("--output_csv", type=str, default="xgb_best_results.csv")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--no_clip_pred_score", action="store_true", help="Disable prediction score clipping")
    args_cli = parser.parse_args()

    results_recompute_dir = args_cli.results_recompute_dir
    test_schedules_csv = args_cli.test_schedules_csv
    model_prefix = args_cli.model_prefix
    output_csv = args_cli.output_csv
    alpha = args_cli.alpha
    clip_pred_score = not args_cli.no_clip_pred_score
    
    print(f"Loading XGBoost models (clip_pred_score={clip_pred_score})...")
    try:
        # load_models handles both two_target and score modes
        b1, b2, feats, mode, model_alpha = load_models(Path(model_prefix))
        print(f"Loaded mode={mode}, model_prefix={model_prefix}, alpha={alpha}")
        alpha = model_alpha # Use alpha from model meta
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
    actual_best_metrics = [] # List to store (throughput, drop_rate, score, models_count) of actual bests
    
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
        
        # Prepare features for all combinations in this schedule
        all_combo_feats = []
        comb_names = []
        
        for comb_name, combo_blob in sched_doc.items():
            if not isinstance(combo_blob, dict): continue
            try:
                df_X = featurize_from_combo(combo_blob)
                all_combo_feats.append(df_X.iloc[0].to_dict())
                comb_names.append(comb_name)
            except Exception as e:
                print(f"  [WARN] Error featurizing for {comb_name}: {e}")
                continue

        if not all_combo_feats:
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

        # Create a single DataFrame for all combinations and align features
        df_batch = pd.DataFrame(all_combo_feats)
        df_batch = df_batch.reindex(columns=feats, fill_value=0.0)
        
        # Batch Predict
        try:
            if mode == "score":
                preds = b1.predict(df_batch)
            elif mode == "rank":
                preds = b1.predict(df_batch)
            else:
                y1_preds = b1.predict(df_batch)
                y2_preds = b2.predict(df_batch)
                preds = y1_preds - alpha * y2_preds
            
            for i, comb_name in enumerate(comb_names):
                pred_score = float(preds[i])
                
                # Apply clipping if enabled
                if clip_pred_score and mode != "rank":
                    pred_score = max(0.0, min(1.0, pred_score))
                
                all_pred_results.append({
                    'combination': comb_name,
                    'pred_score': pred_score
                })
        except Exception as e:
            print(f"  [WARN] Error predicting for schedule {sched_name}: {e}")
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
        
        # [Added] Print Top-5 predicted combinations for this schedule
        print(f"  Top-5 predicted combinations for {sched_name}:")
        for i, res in enumerate(all_pred_results[:5]):
            comb = res['combination']
            p_score = res['pred_score']
            # Get actual score from perf_index
            pure_sched_name = Path(sched_name).name
            sched_perf = perf_index.get(pure_sched_name, {})
            if not sched_perf and pure_sched_name.endswith("_x3.yaml"):
                alt_name = pure_sched_name.replace("_x3.yaml", ".yaml")
                sched_perf = perf_index.get(alt_name, {})
            
            a_score = sched_perf.get(comb, {}).get('score', -1.0)
            print(f"    {i+1}. {comb}: Pred={p_score:.4f}, Actual={a_score:.4f}")

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

        # [Added] Collect metrics for actual best combinations
        if best_actual_combs:
            # Use the first one if there are ties
            first_actual_best = sched_perf.get(best_actual_combs[0])
            if first_actual_best:
                a_derived = first_actual_best.get('derived', {})
                a_t = a_derived.get('throughput_norm')
                a_d = a_derived.get('drop_rate_norm')
                a_s = first_actual_best.get('score')
                a_m_count = len(first_actual_best.get('models', {}))
                actual_best_metrics.append({
                    'throughput': a_t,
                    'drop_rate': a_d,
                    'score': a_s,
                    'models_count': a_m_count
                })

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
        
        # [Added] Calculate averages for actual best metrics
        actual_best_df = pd.DataFrame(actual_best_metrics)
        if not actual_best_df.empty:
            avg_actual_t = round(pd.to_numeric(actual_best_df['throughput'], errors='coerce').mean(), 2)
            avg_actual_d = round(pd.to_numeric(actual_best_df['drop_rate'], errors='coerce').mean(), 2)
            avg_actual_s = round(pd.to_numeric(actual_best_df['score'], errors='coerce').mean(), 2)
        else:
            avg_actual_t, avg_actual_d, avg_actual_s = 'nan', 'nan', 'nan'

        # 모델 개수별 누적 평균값 계산 (>= 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        avg_rows = []
        for n in range(12, 2, -1):
            mask = output_df['models_count'] >= n
            subset = output_df[mask]
            if not subset.empty:
                avg_t = round(pd.to_numeric(subset['normalized_throughput'], errors='coerce').mean(), 2)
                avg_d = round(pd.to_numeric(subset['drop_rate'], errors='coerce').mean(), 2)
                avg_s = round(pd.to_numeric(subset['score'], errors='coerce').mean(), 2)
                
                # [Added] Actual best average for >= n models
                subset_actual = actual_best_df[actual_best_df['models_count'] >= n]
                if not subset_actual.empty:
                    avg_a_t = round(pd.to_numeric(subset_actual['throughput'], errors='coerce').mean(), 2)
                    avg_a_d = round(pd.to_numeric(subset_actual['drop_rate'], errors='coerce').mean(), 2)
                    avg_a_s = round(pd.to_numeric(subset_actual['score'], errors='coerce').mean(), 2)
                    avg_rows.append([f'Actual Best Average (>= {n} models)', '', avg_a_t, avg_a_d, avg_a_s])
                
                avg_rows.append([f'Average (>= {n} models)', '', avg_t, avg_d, avg_s])
            else:
                avg_rows.append([f'Average (>= {n} models)', '', 'nan', 'nan', 'nan'])

        # Create output filename with mode suffix
        if output_csv == "xgb_best_results.csv":
            output_csv = f"xgb_best_results_{mode}.csv"
            
        output_path = Path("experimental_results") / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Alpha', alpha])
            writer.writerow(['Top-1 Accuracy', top1_acc])
            writer.writerow(['Top-5 Accuracy', top5_acc])
            writer.writerow(['Top-1 Accuracy (>= 3 models)', top1_acc_ge3])
            writer.writerow(['Top-5 Accuracy (>= 3 models)', top5_acc_ge3])
            writer.writerow(['Actual Best Average', '', avg_actual_t, avg_actual_d, avg_actual_s])
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
