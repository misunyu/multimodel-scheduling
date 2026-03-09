#python3 scripts/compare_drop_rate.py results_recompute/recompute_performance_20260307_020658_model_schedules_m_resnet50_resnext50_shufflenet-v2-12_squeezenet1.0-12_v_y_x3.json --comp_a combination_97 --comp_b combination_113
#python3 scripts/compare_drop_rate.py results_recompute/recompute_performance_20260306_192733_model_schedules_g_m_resnet50_resnext50_s_t.json --comp_a combination_43 --comp_b combination_64
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def find_combinations_in_file(json_path, comp_a=None, comp_b=None):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        perf_data = data.get('data', [])
        if not perf_data:
            return None, None

        # Use explicitly provided combinations if available
        if comp_a and comp_b:
            entry_a = next((e for e in perf_data if e.get('combination') == comp_a), None)
            entry_b = next((e for e in perf_data if e.get('combination') == comp_b), None)
            if entry_a and entry_b:
                return entry_a, entry_b
            else:
                print(f"Warning: Specific combinations {comp_a} or {comp_b} not found. Falling back to default logic.")

        # Filter entries that have score, score >= 0, and model count >= 2
        valid_entries = []
        for entry in perf_data:
            if 'score' in entry and entry['score'] >= 0:
                models_count = len(entry.get('models', {}))
                if models_count >= 2:
                    valid_entries.append(entry)
                    
        if len(valid_entries) < 2:
            if len(valid_entries) == 1:
                return valid_entries[0], valid_entries[0]
            return None, None
            
        scores = [entry['score'] for entry in valid_entries]
        min_score = min(scores)
        max_score = max(scores)
        
        # Find combinations
        potential_best = [e for e in valid_entries if e['score'] == max_score]
        potential_worst = [e for e in valid_entries if e['score'] == min_score]
        
        return potential_worst[0], potential_best[0]
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None, None

def get_fractions(entry):
    derived = entry.get('derived', {})
    tp_norm = derived.get('throughput_norm')
    dr_norm = derived.get('drop_rate_norm')
    
    if tp_norm is not None and dr_norm is not None:
        return float(tp_norm), float(dr_norm)
        
    inf_count = sum(m.get('inference_count', 0) for m in entry.get('models', {}).values())
    drop_count = derived.get('drop_count', 0)
    total = inf_count + drop_count
    if total > 0:
        return inf_count / total, drop_count / total
    return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to the performance JSON file")
    parser.add_argument("--comp_a", help="Combination ID for Placement A", default=None)
    parser.add_argument("--comp_b", help="Combination ID for Placement B", default=None)
    args = parser.parse_args()

    worst_entry, best_entry = find_combinations_in_file(args.input_json, args.comp_a, args.comp_b)
    
    if not worst_entry or not best_entry:
        print(f"No valid data found in {args.input_json}.")
        return

    # Normalized throughput (processed) and drop rate (dropped)
    worst_tp, worst_drop = get_fractions(worst_entry)
    best_tp, best_drop = get_fractions(best_entry)

    # Plotting
    try:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams['hatch.linewidth'] = 0.3
    except:
        pass

    labels = ['Placement A', 'Placement B']
    tp_vals = [worst_tp, best_tp]
    drop_vals = [worst_drop, best_drop]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(4.0, 3.0))
    
    bar1 = plt.bar(x - width/2, tp_vals, width, label='Normalized Throughput',
                   color='skyblue', alpha=0.5, hatch='//', edgecolor='black', linewidth=0.5)
    bar2 = plt.bar(x + width/2, drop_vals, width, label='Normalized Drop Rate',
                   color='lightcoral', alpha=0.5, hatch='..', edgecolor='black', linewidth=0.5)

    plt.ylabel('Normalized Throughput and Drop Rate')
    plt.xticks(x, labels, fontsize=8)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8, frameon=False)
    
    plt.ylim(0, 1.2)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    autolabel(bar1)
    autolabel(bar2)

    plt.tight_layout()
    output_pdf = "drop_rate_comparison.pdf"
    plt.savefig(output_pdf)
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    main()
