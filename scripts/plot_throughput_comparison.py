# Input file: results/performance_20260127_074757_model_schedules_g_m_t_x4.json
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib import rcParams

def find_combinations_in_file(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        perf_data = data.get('data', [])
        if not perf_data:
            return None, None, 0
            
        # Filter entries that have score, score >= 0, and model count >= 2
        valid_entries = []
        for entry in perf_data:
            if 'score' in entry and entry['score'] >= 0:
                models_count = len(entry.get('models', {}))
                if models_count >= 2:
                    valid_entries.append(entry)
                    
        if len(valid_entries) < 2:
            # If only one entry, best and worst are same, diff 0
            if len(valid_entries) == 1:
                return valid_entries[0], valid_entries[0], 0
            return None, None, 0
            
        scores = [entry['score'] for entry in valid_entries]
        min_score = min(scores)
        max_score = max(scores)
        diff = max_score - min_score
        
        # Find combinations
        potential_best = [e for e in valid_entries if e['score'] == max_score]
        potential_worst = [e for e in valid_entries if e['score'] == min_score]
        
        return potential_worst[0], potential_best[0], diff
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None, None, 0

def plot_throughput_fraction(input_file):
    worst_entry, best_entry, diff = find_combinations_in_file(input_file)
    
    if not worst_entry or not best_entry:
        print(f"No valid data found in {input_file}.")
        return

    print(f"Using performance file: {input_file}")
    print(f"Worst combination: {worst_entry['combination']} (Score: {worst_entry['score']})")
    print(f"Best combination: {best_entry['combination']} (Score: {best_entry['score']})")
    print(f"Score difference: {diff}")

    def get_fractions(entry):
        # Sum inference_count across all models
        inf_count = sum(m.get('inference_count', 0) for m in entry.get('models', {}).values())
        drop_count = entry.get('derived', {}).get('drop_count', 0)
        total = inf_count + drop_count
        if total > 0:
            return inf_count / total, drop_count / total
        return 0, 0

    frac_inf_worst, frac_drop_worst = get_fractions(worst_entry)
    frac_inf_best, frac_drop_best = get_fractions(best_entry)

    # Plotting
    try:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams['hatch.linewidth'] = 0.3
    except:
        pass

    labels = ['Worst Combination', 'Best Combination']
    inf_fractions = [frac_inf_worst, frac_inf_best]
    drop_fractions = [frac_drop_worst, frac_drop_best]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(2.25, 2.33))

    bar_inf = ax.bar(x, inf_fractions, width, label='Processed',
                     color='skyblue', alpha=0.5, edgecolor='black', hatch='//', linewidth=0.5)
    
    bar_drop = ax.bar(x, drop_fractions, width, bottom=inf_fractions, label='Dropped',
                      color='lightcoral', alpha=0.5, edgecolor='black', hatch='..', linewidth=0.5)

    for i in range(len(x)):
        # Processed fraction text
        if inf_fractions[i] > 0.05:
            ax.text(x[i], inf_fractions[i] / 2, 
                    f'{inf_fractions[i]:.2f}', 
                    ha='center', va='center', fontsize=7, fontfamily='serif')
        # Dropped fraction text
        if drop_fractions[i] > 0.05:
            ax.text(x[i], inf_fractions[i] + drop_fractions[i]/2, 
                    f'{drop_fractions[i]:.2f}', 
                    ha='center', va='center', fontsize=7, fontfamily='serif')

    ax.set_ylabel('Fraction of Requests', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.tick_params(axis='y', labelsize=5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=6, frameon=False)

    plt.tight_layout()
    output_pdf = "throughput_fraction_comparison.pdf"
    plt.savefig(output_pdf)
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to the performance JSON file")
    args = parser.parse_args()
    
    plot_throughput_fraction(args.input_json)
