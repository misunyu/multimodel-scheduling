import argparse
import json
import csv
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math

# Import functions from deploy_selector_xgb_suite.py
sys.path.append(str(Path.cwd() / "xgboost_model"))
try:
    from deploy_selector_xgb_suite_legacy import (
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
    
    for p_file in results_path.glob("recompute_performance_*_x3.json"):
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

def calculate_metrics_for_alpha(alpha, model_prefix, sched_index, perf_index, clip_pred_score=True):
    try:
        # Infer mode from model_prefix
        inferred_mode = None
        for m in ["rank", "score", "double", "two_target"]:
            if m in str(model_prefix):
                inferred_mode = m
                break
        b1, b2, feats, mode, _ = load_models(inferred_mode, alpha, prefix=Path(model_prefix))
    except Exception as e:
        print(f"Error loading models for alpha {alpha}: {e}")
        return None

    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_valid_schedules_ge3 = 0
    scores_ge3 = []

    for sched_name, sched_doc in sched_index.items():
        all_combo_feats = []
        comb_names = []
        
        # Scenario grouping for summary
        # Use get_group_id to group by context
        # We need to find at least one window for this schedule to get group_id
        # Actually, in calculate_metrics_for_alpha, we iterate over sched_index
        # which is {schedule_name: {combination: blob}}
        
        scenario_results = []
        models_count = 0
        
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})
        if not sched_perf and pure_sched_name.endswith("_x3.yaml"):
            alt_name = pure_sched_name.replace("_x3.yaml", ".yaml")
            sched_perf = perf_index.get(alt_name, {})

        for comb_name, combo_blob in sched_doc.items():
            if not isinstance(combo_blob, dict): continue
            
            # Ground truth from perf_index
            perf_item = sched_perf.get(comb_name)
            if not perf_item: continue

            if models_count == 0:
                models_count = len(perf_item.get('models', {}))
            
            # Skip if less than 3 models
            if models_count < 3:
                break

            try:
                df_X = featurize_from_combo(combo_blob)
                df_X = df_X.reindex(columns=feats, fill_value=0.0)
                
                if mode == "score":
                    pred_score = float(b1.predict(df_X)[0])
                elif mode == "rank":
                    pred_score = float(b1.predict(df_X)[0])
                else:
                    y1_p = b1.predict(df_X)[0]
                    y2_p = b2.predict(df_X)[0]
                    pred_score = y1_p - alpha * y2_p
                
                if clip_pred_score and mode != "rank":
                    pred_score = max(0.0, min(1.0, pred_score))
                
                # pred_score = round(pred_score, 2)
                
                actual_T = float(perf_item.get('derived', {}).get('throughput_norm', 0))
                actual_D = float(perf_item.get('derived', {}).get('drop_rate_norm', 0))
                actual_score = actual_T - alpha * actual_D

                scenario_results.append({
                    'combination': comb_name,
                    'actual_score': actual_score,
                    'pred_score': pred_score
                })
            except:
                continue

        if not scenario_results or models_count < 3:
            continue
        
        # Sort by actual score to find actual best(s)
        actual_sorted = sorted(scenario_results, key=lambda x: x['actual_score'], reverse=True)
        max_actual_score = actual_sorted[0]['actual_score']
        actual_top1_names = [r['combination'] for r in actual_sorted if math.isclose(r['actual_score'], max_actual_score, rel_tol=1e-7)]

        # Sort by predicted score
        pred_sorted = sorted(scenario_results, key=lambda x: x['pred_score'], reverse=True)
        max_pred_score = pred_sorted[0]['pred_score']
        pred_best_names = [r['combination'] for r in pred_sorted if math.isclose(r['pred_score'], max_pred_score, rel_tol=1e-7)]

        total_valid_schedules_ge3 += 1
        # Top-1 Accuracy: any(Predicted Top-1) in Actual Top-1
        is_top1 = any(name in actual_top1_names for name in pred_best_names)
        if is_top1:
            top1_hits_ge3 += 1
        
        # Top-5 Accuracy: any(Predicted Top-1) in Actual Top-5 groups
        unique_actual_scores = sorted(list(set([r['actual_score'] for r in scenario_results])), reverse=True)
        top5_actual_threshold = unique_actual_scores[min(4, len(unique_actual_scores)-1)]
        actual_top5_names = [r['combination'] for r in scenario_results if r['actual_score'] >= (top5_actual_threshold - 1e-7)]
        
        is_top5 = any(name in actual_top5_names for name in pred_best_names)
        if is_top5:
            top5_hits_ge3 += 1
        
        # Score of first predicted best
        pred_best_name_first = pred_best_names[0]
        pred_best_actual_score = next(r['actual_score'] for r in scenario_results if r['combination'] == pred_best_name_first)
        scores_ge3.append(pred_best_actual_score)

    if total_valid_schedules_ge3 == 0:
        return None

    return {
        'alpha': alpha,
        'avg_score_ge3': np.mean(scores_ge3),
        'top1_acc_ge3': top1_hits_ge3 / total_valid_schedules_ge3,
        'top5_acc_ge3': top5_hits_ge3 / total_valid_schedules_ge3
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_recompute_dir", type=str, default="results_recompute")
    parser.add_argument("--test_schedules_csv", type=str, default="xgboost_model/dataset/gpu/test_schedules_x3.csv")
    parser.add_argument("--model_prefix", type=str, default="xgboost_model/artifacts/gpu/xgb_model_x3_double")
    parser.add_argument("--output_csv", type=str, default="experimental_results/xgb_alpha_sweep_ge3.csv")
    args = parser.parse_args()

    print(f"Indexing performance data from {args.results_recompute_dir}...")
    perf_index = get_performance_index(args.results_recompute_dir)
    
    print(f"Loading test schedules from {args.test_schedules_csv}...")
    sched_index = _index_schedules_from_csv(Path(args.test_schedules_csv))

    sweep_results = []
    alphas = [round(x * 0.1, 1) for x in range(1, 11)]
    
    for alpha in alphas:
        print(f"Calculating metrics for alpha={alpha}...")
        res = calculate_metrics_for_alpha(alpha, args.model_prefix, sched_index, perf_index)
        if res:
            sweep_results.append(res)

    if sweep_results:
        df = pd.DataFrame(sweep_results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved alpha sweep results to {args.output_csv}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
