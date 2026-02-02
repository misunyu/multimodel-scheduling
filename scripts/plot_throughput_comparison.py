import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def plot_throughput_fraction():
    csv_file = "better_throughput_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Read the first row (after header) from CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
    
    if not first_row:
        print("CSV is empty.")
        return

    results_dir = first_row['directory']
    filename = first_row['filename']
    best_deployment = first_row['best_deployment']
    better_combination = first_row['better_combination']

    json_path = os.path.join(results_dir, filename)
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    def get_counts(comb_id):
        for entry in data.get('data', []):
            if entry.get('combination') == comb_id:
                # Sum inference_count across all models
                inf_count = sum(m.get('inference_count', 0) for m in entry.get('models', {}).values())
                drop_count = entry.get('derived', {}).get('drop_count', 0)
                return inf_count, drop_count
        return 0, 0

    inf_best, drop_best = get_counts(best_deployment)
    inf_better, drop_better = get_counts(better_combination)

    # Fractions
    total_best = inf_best + drop_best
    total_better = inf_better + drop_better

    frac_inf_best = inf_best / total_best if total_best > 0 else 0
    frac_drop_best = drop_best / total_best if total_best > 0 else 0
    
    frac_inf_better = inf_better / total_better if total_better > 0 else 0
    frac_drop_better = drop_better / total_better if total_better > 0 else 0

    # Plotting
    # Set font to match LaTeX appearance (Times New Roman)
    try:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams['hatch.linewidth'] = 0.3  # Thinner hatch lines
    except:
        pass

    labels = ['Throughput-Max (CPU)', 'Best Score (Ours)']
    # better_combination is Throughput-Max (CPU), best_deployment is Best Score (Ours)
    inf_fractions = [frac_inf_better, frac_inf_best]
    drop_fractions = [frac_drop_better, frac_drop_best]

    x = np.arange(len(labels))
    width = 0.4  # bar width

    # Reduced height to 2/3: 3.5 * 2/3 approx 2.33
    # Reduced width to half: 4.5 / 2 = 2.25
    fig, ax = plt.subplots(figsize=(2.25, 2.33))

    # Inference (Processed) - Blue/Skyblue style
    edge_color1 = 'blue'
    # To make hatch color lighter than edge color, we can draw the bar twice
    # Once for the fill and hatch, once for the border.
    # But simpler is to use a lighter color for edgecolor and draw border separately if needed.
    # However, user said "연하고 얇게" (lighter and thinner).
    # Let's try setting edgecolor to a lighter version of blue/red and linewidth for hatch via rcParams.
    
    # We'll use a slightly lighter blue/red for the hatch/edge
    hatch_color1 = '#8888FF' # Lighter blue
    bar_inf = ax.bar(x, inf_fractions, width, label='Processed',
                     color='skyblue', alpha=0.5, edgecolor='black', hatch='//', linewidth=0.5)
    
    # Dropped - Red/Lightcoral style
    hatch_color2 = '#FF8888' # Lighter red
    bar_drop = ax.bar(x, drop_fractions, width, bottom=inf_fractions, label='Dropped',
                      color='lightcoral', alpha=0.5, edgecolor='black', hatch='..', linewidth=0.5)

    # Add text labels on bars (matches autolabel style)
    for i in range(len(x)):
        # Processed fraction text
        total_height = inf_fractions[i]
        ax.text(x[i], total_height / 2, 
                f'{inf_fractions[i]:.2f}', 
                ha='center', va='center', fontsize=7, fontfamily='serif')
        # Dropped fraction text
        ax.text(x[i], inf_fractions[i] + drop_fractions[i]/2, 
                f'{drop_fractions[i]:.2f}', 
                ha='center', va='center', fontsize=7, fontfamily='serif')

    ax.set_ylabel('Fraction of Requests', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    
    # Legend style from compare_latency_vs_best.py
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=6, frameon=False)

    plt.tight_layout()
    output_pdf = "throughput_fraction_comparison.pdf"
    plt.savefig(output_pdf)
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    plot_throughput_fraction()
