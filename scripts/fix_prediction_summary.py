import pandas as pd
import numpy as np
import sys
import os
import json

def fix_file(file_path, schedule_csv):
    # Read schedule info to get model counts
    sched_df = pd.read_csv(schedule_csv)
    sched_model_counts = {}
    for _, row in sched_df.iterrows():
        try:
            content = row['content']
            # YAML or JSON
            content = content.strip()
            if content.startswith('{'):
                data = json.loads(content)
            else:
                import yaml
                # Remove comments if they cause issues
                yaml_content = "\n".join([l for l in content.split("\n") if not l.strip().startswith("#")])
                data = yaml.safe_load(yaml_content)
            
            # Count models in 'models' or similar. 
            models = data.get('models', {})
            if not models and 'data' in data:
                # Might be a result file or something else?
                # Usually schedule files have 'models' at top level
                pass
            
            # If it's a schedule file, it should have model entries
            if not models:
                # If combinations are at top level
                first_combo = next(iter(data.values()))
                if isinstance(first_combo, dict):
                    # Count keys that are dicts (representing models in the combination)
                    models = [k for k, v in first_combo.items() if isinstance(v, dict)]
            
            sched_model_counts[row['schedule_name']] = len(models)
            if len(models) == 0:
                print(f"DEBUG: Could not find models in {row['schedule_name']}. Content preview: {str(data)[:100]}")
        except:
            sched_model_counts[row['schedule_name']] = 0

    # If file already has summary headers, skip them
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('schedule_file,'):
            data_start = i
            break
            
    # Write only data part to a temp file for pandas
    temp_csv = file_path + ".temp"
    data_lines = lines[data_start:]
    with open(temp_csv, 'w') as f:
        f.writelines(data_lines)
        
    df = pd.read_csv(temp_csv)
    os.remove(temp_csv)
    
    # [Correction] Filter out any extra headers that might have been accidentally added
    df = df[df['schedule_file'] != 'schedule_file']
    
    # Calculate metrics
    # [Modified] Group by schedule_file only, as timestamps might differ for combinations in the same scenario
    scenarios = df.groupby(['schedule_file'])
    
    top1_hits = 0
    top5_hits = 0
    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_scenarios = len(scenarios)
    total_scenarios_ge3 = 0
    
    for s_file_tuple, group in scenarios:
        s_file = s_file_tuple[0] if isinstance(s_file_tuple, (list, tuple)) else s_file_tuple
        # Actual best
        # Use a small tolerance for floating point comparison
        max_actual = group['actual_score'].astype(float).max()
        actual_best_names = group[group['actual_score'].astype(float) >= (max_actual - 1e-4)]['combination'].tolist()
        
        # Predicted best
        pred_sorted = group.sort_values(by='pred_score', ascending=False)
        pred_best_name = pred_sorted.iloc[0]['combination']
        
        is_top1 = False
        if pred_best_name in actual_best_names:
            is_top1 = True
            top1_hits += 1
            
        top5_pred_names = pred_sorted.iloc[:5]['combination'].tolist()
        is_top5 = any(name in top5_pred_names for name in actual_best_names)
        if is_top5:
            top5_hits += 1
            
        # Model count check
        s_name_base = os.path.basename(s_file).lower()
        m_count = 0
        for k, v in sched_model_counts.items():
            if os.path.basename(k).lower() == s_name_base:
                m_count = v
                break
        
        if m_count >= 3:
            total_scenarios_ge3 += 1
            if is_top1:
                top1_hits_ge3 += 1
            if is_top5:
                top5_hits_ge3 += 1
        
        # DEBUG print for the first few scenarios
        if total_scenarios <= 5 or total_scenarios % 10 == 0:
             print(f"Scenario: {s_name_base}, Models: {m_count}, Actual Max: {max_actual}, Pred Best: {pred_best_name}, Top1: {is_top1}")
            
    top1_acc = top1_hits / total_scenarios if total_scenarios > 0 else 0
    top5_acc = top5_hits / total_scenarios if total_scenarios > 0 else 0
    top1_acc_ge3 = top1_hits_ge3 / total_scenarios_ge3 if total_scenarios_ge3 > 0 else 0
    top5_acc_ge3 = top5_hits_ge3 / total_scenarios_ge3 if total_scenarios_ge3 > 0 else 0
    
    # Write summary + content
    with open(file_path, 'w') as f:
        f.write(f"Alpha,0.2\n")
        f.write(f"Top-1 Accuracy,{top1_acc:.4f}\n")
        f.write(f"Top-5 Accuracy,{top5_acc:.4f}\n")
        f.write(f"Top-1 Accuracy (>= 3 models),{top1_acc_ge3:.4f}\n")
        f.write(f"Top-5 Accuracy (>= 3 models),{top5_acc_ge3:.4f}\n")
        
        # Calculate and write Model-wise averages
        # Find max model count
        all_counts = list(sched_model_counts.values())
        max_m = max(all_counts) if all_counts else 0
        
        # We need to collect per-scenario bests and predictive bests for these averages
        scenario_data_list = []
        for s_file_tuple, group in scenarios:
            s_file = s_file_tuple[0] if isinstance(s_file_tuple, (list, tuple)) else s_file_tuple
            s_name_base = os.path.basename(s_file).lower()
            m_count = 0
            for k, v in sched_model_counts.items():
                if os.path.basename(k).lower() == s_name_base:
                    m_count = v
                    break
            
            # Oracle
            oracle_row = group.loc[group['actual_score'].astype(float).idxmax()]
            # Predictive
            pred_row = group.loc[group['pred_score'].astype(float).idxmax()]
            
            scenario_data_list.append({
                'm_count': m_count,
                'oracle_T': float(oracle_row['actual_T_norm']),
                'oracle_D': float(oracle_row['actual_D_norm']),
                'oracle_S': float(oracle_row['actual_score']),
                'pred_T': float(pred_row['actual_T_norm']),
                'pred_D': float(pred_row['actual_D_norm']),
                'pred_S': float(pred_row['actual_score'])
            })
        
        # Actual Best Average
        oracle_all_T = np.mean([d['oracle_T'] for d in scenario_data_list])
        oracle_all_D = np.mean([d['oracle_D'] for d in scenario_data_list])
        oracle_all_S = np.mean([d['oracle_S'] for d in scenario_data_list])
        f.write(f"Actual Best Average,,{oracle_all_T:.2f},{oracle_all_D:.2f},{oracle_all_S:.2f}\n")
        
        for n in range(max_m, 2, -1):
            subset = [d for d in scenario_data_list if d['m_count'] >= n]
            if not subset:
                f.write(f"Average (>= {n} models),,nan,nan,nan\n")
                continue
                
            o_avg_T = np.mean([d['oracle_T'] for d in subset])
            o_avg_D = np.mean([d['oracle_D'] for d in subset])
            o_avg_S = np.mean([d['oracle_S'] for d in subset])
            f.write(f"Actual Best Average (>= {n} models),,{o_avg_T:.2f},{o_avg_D:.2f},{o_avg_S:.2f}\n")
            
            p_avg_T = np.mean([d['pred_T'] for d in subset])
            p_avg_D = np.mean([d['pred_D'] for d in subset])
            p_avg_S = np.mean([d['pred_S'] for d in subset])
            f.write(f"Average (>= {n} models),,{p_avg_T:.2f},{p_avg_D:.2f},{p_avg_S:.2f}\n")
            
        # Overall Average
        p_all_T = np.mean([d['pred_T'] for d in scenario_data_list])
        p_all_D = np.mean([d['pred_D'] for d in scenario_data_list])
        p_all_S = np.mean([d['pred_S'] for d in scenario_data_list])
        f.write(f"Average,,{p_all_T:.2f},{p_all_D:.2f},{p_all_S:.2f}\n")
        
        f.write(content)

if __name__ == "__main__":
    fix_file(sys.argv[1], sys.argv[2])
