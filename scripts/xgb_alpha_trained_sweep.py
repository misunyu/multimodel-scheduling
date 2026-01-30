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

def calculate_metrics_for_alpha(alpha, model_prefix_template, sched_index, perf_index, clip_pred_score=True):
    model_prefix = model_prefix_template.format(alpha=alpha)
    print(f"Loading model: {model_prefix}")
    try:
        b1, b2, feats, mode, model_alpha = load_models(Path(model_prefix))
        # Note: model_alpha from meta might be different if it was trained with a specific alpha
    except Exception as e:
        print(f"Error loading models for alpha {alpha} from {model_prefix}: {e}")
        return None

    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_valid_schedules_ge3 = 0
    scores_ge3 = []

    for sched_name, sched_doc in sched_index.items():
        all_combo_feats = []
        comb_names = []
        
        for comb_name, combo_blob in sched_doc.items():
            if not isinstance(combo_blob, dict): continue
            try:
                df_X = featurize_from_combo(combo_blob)
                all_combo_feats.append(df_X.iloc[0].to_dict())
                comb_names.append(comb_name)
            except:
                continue

        if not all_combo_feats:
            continue

        df_batch = pd.DataFrame(all_combo_feats)
        df_batch = df_batch.reindex(columns=feats, fill_value=0.0)
        
        try:
            if mode == "score":
                preds = b1.predict(df_batch)
            elif mode == "rank":
                preds = b1.predict(df_batch)
            else:
                y1_preds = b1.predict(df_batch)
                y2_preds = b2.predict(df_batch)
                preds = y1_preds - alpha * y2_preds
            
            all_pred_results = []
            for i, comb_name in enumerate(comb_names):
                pred_score = float(preds[i])
                if clip_pred_score and mode != "rank":
                    pred_score = max(0.0, min(1.0, pred_score))
                all_pred_results.append({'combination': comb_name, 'pred_score': pred_score})
        except:
            continue
        
        all_pred_results.sort(key=lambda x: x['pred_score'], reverse=True)
        max_pred_score = all_pred_results[0]['pred_score']
        best_pred_combs = [r['combination'] for r in all_pred_results if math.isclose(r['pred_score'], max_pred_score, rel_tol=1e-7)]
        
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})
        if not sched_perf and pure_sched_name.endswith("_x3.yaml"):
            alt_name = pure_sched_name.replace("_x3.yaml", ".yaml")
            sched_perf = perf_index.get(alt_name, {})

        best_actual_combs = []
        max_actual_score = -float('inf')
        for comb_name, perf_item in sched_perf.items():
            actual_score = perf_item.get('score', -float('inf'))
            if math.isclose(actual_score, max_actual_score, rel_tol=1e-7):
                best_actual_combs.append(comb_name)
            elif actual_score > max_actual_score:
                max_actual_score = actual_score
                best_actual_combs = [comb_name]

        # Get models count for the first predicted best
        perf_item = sched_perf.get(best_pred_combs[0])
        if not perf_item: continue
        
        models_count = len(perf_item.get('models', {}))
        
        if models_count >= 3:
            total_valid_schedules_ge3 += 1
            # Top-1 Accuracy
            if any(c in best_actual_combs for c in best_pred_combs):
                top1_hits_ge3 += 1
            
            # Top-5 Accuracy
            pred_scores_sorted = sorted(list(set([r['pred_score'] for r in all_pred_results])), reverse=True)
            top5_threshold_score = pred_scores_sorted[min(4, len(pred_scores_sorted)-1)]
            top5_group_combs = [r['combination'] for r in all_pred_results if r['pred_score'] >= top5_threshold_score - 1e-7]
            if any(c in top5_group_combs for c in best_actual_combs):
                top5_hits_ge3 += 1
            
            scores_ge3.append(perf_item.get('score'))

    if total_valid_schedules_ge3 == 0:
        return None

    return {
        'alpha': alpha,
        'avg_score_ge3': round(np.mean(scores_ge3), 4),
        'top1_acc_ge3': round(top1_hits_ge3 / total_valid_schedules_ge3, 4),
        'top5_acc_ge3': round(top5_hits_ge3 / total_valid_schedules_ge3, 4)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_recompute_dir", type=str, default="results_recompute")
    parser.add_argument("--test_schedules_csv", type=str, default="xgboost_model/dataset/gpu/test_schedules_x3.csv")
    parser.add_argument("--model_prefix_template", type=str, default="xgboost_model/artifacts/gpu/xgb_model_x3_score_alpha_{alpha}")
    parser.add_argument("--output_csv", type=str, default="experimental_results/xgb_alpha_trained_sweep_ge3.csv")
    args = parser.parse_args()

    print(f"Indexing performance data from {args.results_recompute_dir}...")
    perf_index = get_performance_index(args.results_recompute_dir)
    
    print(f"Loading test schedules from {args.test_schedules_csv}...")
    sched_index = _index_schedules_from_csv(Path(args.test_schedules_csv))

    sweep_results = []
    alphas = [round(x * 0.1, 1) for x in range(1, 11)]
    
    for alpha in alphas:
        print(f"Calculating metrics for alpha={alpha}...")
        res = calculate_metrics_for_alpha(alpha, args.model_prefix_template, sched_index, perf_index)
        if res:
            sweep_results.append(res)

    if sweep_results:
        df = pd.DataFrame(sweep_results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved alpha trained sweep results to {args.output_csv}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
