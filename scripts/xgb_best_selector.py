import argparse
import json
import csv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys

#python3 scripts/xgb_best_selector.py \
#  --perf_csv xgboost_model/dataset/gpu/test_x3.csv \
#  --test_schedules_csv xgboost_model/dataset/gpu/test_schedules_x3.csv \
#  --model_prefix xgboost_model/artifacts/gpu/xgb_model_x3_double \
#  --output_csv xgb_best_results_x3_double.csv \
#  --alpha 0.2 \
#  --no_clip_pred_score

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
    from deploy_selector_xgb_suite_legacy import (
        load_models, 
        _index_schedules_from_csv,
        _find_schedule,
        _build_infps_lookup,
        featurize_window,
        get_group_id
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
    parser.add_argument("--perf_csv", type=str, default="xgboost_model/dataset/gpu/test_x3.csv")
    parser.add_argument("--model_prefix", type=str, default="xgboost_model/artifacts/gpu/xgb_model_rank")
    parser.add_argument("--output_csv", type=str, default="xgb_best_results.csv")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--no_clip_pred_score", action="store_true", help="Disable prediction score clipping")
    parser.add_argument("--debug_schedule", type=str, default=None, help="Substring of schedule name to show detailed debug info")
    args_cli = parser.parse_args()

    results_recompute_dir = args_cli.results_recompute_dir
    test_schedules_csv = args_cli.test_schedules_csv
    perf_csv_path = args_cli.perf_csv
    model_prefix = args_cli.model_prefix
    output_csv = args_cli.output_csv
    alpha = args_cli.alpha
    clip_pred_score = not args_cli.no_clip_pred_score
    debug_schedule = args_cli.debug_schedule
    
    print(f"Loading XGBoost models (clip_pred_score={clip_pred_score})...")
    try:
        # Infer mode from model_prefix
        inferred_mode = None
        for m in ["rank", "score", "double", "two_target"]:
            if m in str(model_prefix):
                inferred_mode = m
                break
        # load_models handles both two_target and score modes
        b1, b2, feats, mode, model_alpha = load_models(inferred_mode, alpha, prefix=Path(model_prefix))
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

    # [1] & [3] Load perf_csv and group by scenario (group_id)
    print(f"Loading and grouping performance data from {perf_csv_path}...")
    scenario_data = {} # group_id -> list of window objects
    try:
        with open(perf_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    w = json.loads(row['json_content'])
                    key = get_group_id(w)
                    if key not in scenario_data:
                        scenario_data[key] = []
                    scenario_data[key].append(w)
                except Exception as e:
                    print(f"Error parsing row in {perf_csv_path}: {e}")
    except Exception as e:
        print(f"Error reading {perf_csv_path}: {e}")
        return

    print(f"Grouped into {len(scenario_data)} scenarios.")

    results = []
    actual_best_metrics = [] # List to store (throughput, drop_rate, score, models_count) of actual bests
    
    top1_hits = 0
    top5_hits = 0
    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_scenarios = 0
    total_scenarios_ge3 = 0

    # [4] Process each scenario
    for group_id, windows in scenario_data.items():
        if not windows: continue
        
        s_name = windows[0].get("schedule_file") or windows[0].get("schedule file")
        is_debug = debug_schedule and s_name and debug_schedule in s_name

        if is_debug:
            print(f"\n[DEBUG] Scenario: {group_id} (Schedule: {s_name})")

        scenario_results = []
        
        # [NEW] Prepare features and keep successful entries (window, combination, X_dict)
        successful_entries = []
        fail_comb_names = []
        
        for w in windows:
            c_name = w.get("combination")
            try:
                # 1. Build infps_map
                s_doc = _find_schedule(sched_index, s_name)
                infps_map = _build_infps_lookup(s_doc, c_name) if s_doc else None
            
                # 2. Featurize
                X_dict, _, _ = featurize_window(w, infps_map)
            
                successful_entries.append({
                    "window": w,
                    "combination": c_name,
                    "X_dict": X_dict
                })
            except Exception as e:
                fail_comb_names.append(c_name)
                # Only print error if it's not a common case or during debug
                if is_debug:
                    print(f"  [WARN] Error featurizing for {c_name} in {s_name}: {e}")
                continue

        if is_debug:
            print(f"  [DEBUG] Total windows: {len(windows)}, Success: {len(successful_entries)}, Fail: {len(fail_comb_names)}")
            if fail_comb_names:
                print(f"  [DEBUG] Failed combinations: {fail_comb_names}")

        if not successful_entries:
            continue

        # Batch Predict for the scenario
        all_combo_feats = [entry["X_dict"] for entry in successful_entries]
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
        
            for i, entry in enumerate(successful_entries):
                pred_score_raw = float(preds[i])
                if clip_pred_score and mode != "rank":
                    pred_score_raw = max(0.0, min(1.0, pred_score_raw))
                
                # [5] Calculate actual_score_raw using the correct window from the entry
                w = entry["window"]
                c_name = entry["combination"]
                derived = w.get("derived", {})
                y1_actual = float(derived.get("throughput_norm", 0.0))
                y2_actual = float(derived.get("drop_rate_norm", 0.0))
                actual_score_raw = y1_actual - alpha * y2_actual
            
                scenario_results.append({
                    'combination': c_name,
                    'pred_score_raw': pred_score_raw,
                    'actual_score_raw': actual_score_raw,
                    'actual_T_norm': y1_actual,
                    'actual_D_norm': y2_actual,
                    'models_count': len(w.get('models', {}))
                })
                
                if is_debug:
                    print(f"    [DEBUG] #{i} Comb: {c_name}, Pred: {pred_score_raw:.4f}, Actual: {actual_score_raw:.4f}")
        except Exception as e:
            print(f"  [WARN] Error predicting for scenario {group_id}: {e}")
            continue
    
        if not scenario_results:
            continue

        # [6] Predicted best / Top-5 and Actual best
        scenario_results.sort(key=lambda x: x['pred_score_raw'], reverse=True)
        max_pred_score = scenario_results[0]['pred_score_raw']
        best_pred_combs = [r['combination'] for r in scenario_results if math.isclose(r['pred_score_raw'], max_pred_score, rel_tol=1e-7)]
        
        unique_pred_scores = sorted(list(set([r['pred_score_raw'] for r in scenario_results])), reverse=True)
        top5_threshold = unique_pred_scores[min(4, len(unique_pred_scores)-1)] if unique_pred_scores else -1.0
        predicted_top5_combs = [r['combination'] for r in scenario_results if r['pred_score_raw'] >= (top5_threshold - 1e-7)]

        actual_sorted = sorted(scenario_results, key=lambda x: x['actual_score_raw'], reverse=True)
        max_actual_score = actual_sorted[0]['actual_score_raw']
        best_actual_combs = [r['combination'] for r in scenario_results if math.isclose(r['actual_score_raw'], max_actual_score, rel_tol=1e-7)]

        if is_debug:
            print(f"  All Combinations in this Scenario:")
            for r in scenario_results:
                print(f"    - {r['combination']}: Pred={r['pred_score_raw']:.4f}, Actual={r['actual_score_raw']:.4f}")
            print(f"  Pred Top-1: {best_pred_combs} (score: {max_pred_score:.4f})")
            print(f"  Pred Top-5 Threshold: {top5_threshold:.4f}")
            print(f"  Pred Top-5 Combs: {predicted_top5_combs}")
            print(f"  Actual Best: {best_actual_combs} (score: {max_actual_score:.4f})")

        # Hits
        is_top1 = any(c in best_actual_combs for c in best_pred_combs)
        is_top5 = any(c in predicted_top5_combs for c in best_actual_combs)
        
        total_scenarios += 1
        if is_top1: top1_hits += 1
        if is_top5: top5_hits += 1

        # GE3
        model_count = scenario_results[0]['models_count']
        if model_count >= 3:
            total_scenarios_ge3 += 1
            if is_top1: top1_hits_ge3 += 1
            if is_top5: top5_hits_ge3 += 1

        # Metrics for Average
        actual_best_row = actual_sorted[0]
        actual_best_metrics.append({
            'throughput': actual_best_row['actual_T_norm'],
            'drop_rate': actual_best_row['actual_D_norm'],
            'score': actual_best_row['actual_score_raw'],
            'models_count': model_count
        })

        # [Added] Debug info for mismatch with deploy_selector_xgb_suite.py
        if not is_top5 and not is_debug:
             pass # Removed extra print to keep it clean, but logic remains same as deploy_selector_xgb_suite.py

        # [8] Results for CSV output
        display_parts = []
        seen_combs = set()
        for bpc in best_pred_combs:
            if bpc in seen_combs: continue
            seen_combs.add(bpc)
            if bpc in best_actual_combs:
                display_parts.append(f"<font color='purple'>{bpc}</font>")
            else:
                display_parts.append(f"<font color='red'>{bpc}</font>")
        
        display_comb = ", ".join(display_parts)
        if not any(c in best_actual_combs for c in best_pred_combs) and best_actual_combs:
            actual_str = ", ".join([f"<font color='blue'>{c}</font>" for c in best_actual_combs])
            display_comb += f" (Actual: {actual_str})"

        pred_best_row = scenario_results[0]
        results.append({
            'schedule_file': s_name,
            'best_combination': display_comb,
            'normalized_throughput': pred_best_row['actual_T_norm'],
            'drop_rate': pred_best_row['actual_D_norm'],
            'score': pred_best_row['actual_score_raw'],
            'models_count': model_count
        })

    if results:
        output_df = pd.DataFrame(results)
        
        # Calculate accuracies
        top1_acc = round(top1_hits / total_scenarios, 4) if total_scenarios > 0 else 0
        top5_acc = round(top5_hits / total_scenarios, 4) if total_scenarios > 0 else 0
        
        top1_acc_ge3 = round(top1_hits_ge3 / total_scenarios_ge3, 4) if total_scenarios_ge3 > 0 else 0
        top5_acc_ge3 = round(top5_hits_ge3 / total_scenarios_ge3, 4) if total_scenarios_ge3 > 0 else 0

        print(f"\n--- Prediction Summary ---")
        print(f"Total Scenarios: {total_scenarios}")
        print(f"Top-1 Accuracy: {top1_acc} ({top1_hits}/{total_scenarios})")
        print(f"Top-5 Accuracy: {top5_acc} ({top5_hits}/{total_scenarios})")
        if total_scenarios_ge3 > 0:
            print(f"Top-1 Accuracy (>= 3 models): {top1_acc_ge3} ({top1_hits_ge3}/{total_scenarios_ge3})")
            print(f"Top-5 Accuracy (>= 3 models): {top5_acc_ge3} ({top5_hits_ge3}/{total_scenarios_ge3})")
        print(f"---------------------------\n")

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
            writer.writerow(['Actual Best Average', '', round(avg_actual_t, 4) if isinstance(avg_actual_t, (int, float)) else avg_actual_t, round(avg_actual_d, 4) if isinstance(avg_actual_d, (int, float)) else avg_actual_d, round(avg_actual_s, 4) if isinstance(avg_actual_s, (int, float)) else avg_actual_s])
            for row in avg_rows:
                writer.writerow(row)
            writer.writerow(['Average', '', round(avg_throughput, 4), round(avg_drop_rate, 4), round(avg_score, 4)])
            writer.writerow(['schedule_file', 'best_combination', 'normalized_throughput', 'drop_rate', 'score'])
        
            for _, row in output_df.iterrows():
                writer.writerow([
                    row['schedule_file'],
                    row['best_combination'],
                    round(row['normalized_throughput'], 4) if isinstance(row['normalized_throughput'], (int, float)) else row['normalized_throughput'],
                    round(row['drop_rate'], 4) if isinstance(row['drop_rate'], (int, float)) else row['drop_rate'],
                    round(row['score'], 4) if isinstance(row['score'], (int, float)) else row['score']
                ])
                
        print(f"Saved {len(results)} results to {output_csv}")
        print(f"Average Throughput: {avg_throughput}, Drop Rate: {avg_drop_rate}, Score: {avg_score}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
