# python3 scripts/search_and_plot_drop_rate.py results_recompute/recompute_performance_20260127_112106_model_schedules_m_r_t_y_x2.json
# combination_7 combination_6
# JSON file: results_recompute/recompute_performance_20260127_112106_model_schedules_m_r_t_y_x2.json
# Left combination: combination_7
# Right combination: combination_6
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def get_fractions(entry):
    derived = entry.get('derived', {})
    tp_norm = derived.get('throughput_norm', 0.0)
    dr_norm = derived.get('drop_rate_norm', 0.0)
    return tp_norm, dr_norm

def get_entry_by_combination(data, combination_name):
    perf_data = data.get('data', [])
    for entry in perf_data:
        if entry.get('combination') == combination_name:
            return entry
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the performance JSON file")
    parser.add_argument("left_comb", help="Combination name for the left side (e.g., combination_1)")
    parser.add_argument("right_comb", help="Combination name for the right side (e.g., combination_2)")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {args.input_file}: {e}")
        return

    left_entry = get_entry_by_combination(data, args.left_comb)
    right_entry = get_entry_by_combination(data, args.right_comb)

    if not left_entry:
        print(f"Combination '{args.left_comb}' not found in {args.input_file}.")
        return
    if not right_entry:
        print(f"Combination '{args.right_comb}' not found in {args.input_file}.")
        return

    l_tp, l_drop = get_fractions(left_entry)
    r_tp, r_drop = get_fractions(right_entry)

    print(f"Selected file: {args.input_file}")
    print(f"Left Combination: {args.left_comb} (TP: {l_tp:.2f}, Drop: {l_drop:.2f})")
    print(f"Right Combination: {args.right_comb} (TP: {r_tp:.2f}, Drop: {r_drop:.2f})")

    # Plotting
    try:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams['hatch.linewidth'] = 0.3
    except:
        pass

    labels = ['Placement A', 'Placement B']
    tp_vals = [l_tp, r_tp]
    drop_vals = [l_drop, r_drop]

    x = np.arange(len(labels))
    width = 0.25 

    plt.figure(figsize=(4.0, 3.0))
    
    # Using width exactly as spacing between centers to make them touch
    bar1 = plt.bar(x - width/2, tp_vals, width, label='Throughput', 
                   color='skyblue', alpha=0.5, hatch='//', edgecolor='black', linewidth=0.5)
    bar2 = plt.bar(x + width/2, drop_vals, width, label='Drop Rate', 
                   color='lightcoral', alpha=0.5, hatch='..', edgecolor='black', linewidth=0.5)

    plt.ylabel('Normalized Value')
    plt.xticks(x, labels, fontsize=8)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8, frameon=False)
    
    max_val = max(max(tp_vals), max(drop_vals))
    plt.ylim(0, max_val * 1.3)
    
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
