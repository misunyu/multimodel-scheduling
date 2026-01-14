import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_scatter(csv_path):
    try:
        df = pd.read_csv(csv_path, comment='#')
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # Check if required columns exist
    required_cols = ['actual_score', 'pred_score']
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {csv_path}: Missing required columns.")
        return

    # Set global font settings for Times New Roman and larger sizes
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    
    # Create figure with 3 subplots (Throughput, Drop Rate, Score)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    targets = [
        ('actual_T_norm', 'pred_T_norm', 'Throughput (T_norm)'),
        ('actual_D_norm', 'pred_D_norm', 'Drop Rate (D_norm)'),
        ('actual_score', 'pred_score', 'Score')
    ]
    
    for i, (actual_col, pred_col, title) in enumerate(targets):
        if actual_col in df.columns and pred_col in df.columns:
            ax = axes[i]
            actual = df[actual_col]
            pred = df[pred_col]
            
            ax.scatter(actual, pred, alpha=0.5, s=25)
            
            # Diagonal line
            min_val = min(actual.min(), pred.min())
            max_val = max(actual.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3)
            
            ax.set_xlabel('Ground Truth', fontsize=22)
            ax.set_ylabel('Predicted', fontsize=22)
            ax.set_title(title, fontsize=24, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Save as PDF
    output_pdf = csv_path.with_suffix('.pdf')
    plt.savefig(output_pdf)
    plt.close()
    print(f"Saved scatter plot to: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot scatter results for performance predictions.")
    parser.add_argument("--input_dir", default="xgboost_model/prediction_result", help="Directory containing prediction result CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist.")
        return

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        plot_scatter(csv_file)

if __name__ == "__main__":
    import argparse
    main()
