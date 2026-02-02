import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

def main():
    # 파일 경로 설정
    sweep_fixed_path = Path("experimental_results/xgb_alpha_sweep_ge3.csv")
    sweep_trained_path = Path("experimental_results/xgb_alpha_trained_sweep_ge3.csv")
    output_path = Path("experimental_results/alpha_comparison_plots.pdf")

    if not sweep_fixed_path.exists() or not sweep_trained_path.exists():
        print("Required CSV files not found.")
        return

    # 데이터 로드
    df_fixed = pd.read_csv(sweep_fixed_path)
    df_trained = pd.read_csv(sweep_trained_path)

    # 데이터 병합 (alpha 기준)
    df = pd.merge(df_fixed, df_trained, on='alpha', suffixes=('_fixed', '_trained'))
    df = df.sort_values('alpha')

    # 스타일 설정 (compare_latency_vs_best.py 스타일)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['hatch.linewidth'] = 0.3 # Thinner hatch lines

    alphas = df['alpha'].values
    x = np.arange(len(alphas))
    width = 0.35

    # 그래프 생성
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.75))
    
    metrics = [
        ('avg_score_ge3', 'Average Score (Models >= 3)'),
        ('top1_acc_ge3', 'Top-1 Accuracy (Models >= 3)'),
        ('top5_acc_ge3', 'Top-5 Accuracy (Models >= 3)')
    ]


    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        fixed_vals = df[f'{col}_fixed'].values
        trained_vals = df[f'{col}_trained'].values

        rects1 = ax.bar(x - width/2, fixed_vals, width, label='Multi-Objective Predictor', 
                        color='skyblue', alpha=0.5, hatch='//', edgecolor='black', linewidth=0.3)
        rects2 = ax.bar(x + width/2, trained_vals, width, label='Single-Objective Predictor', 
                        color='lightcoral', alpha=0.5, hatch='..', edgecolor='black', linewidth=0.3)

        if i == 0:
            ax.set_ylabel('Score', fontsize=10)
        else:
            ax.set_ylabel('Probability', fontsize=10)
        # ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(alphas, fontsize=9)
        ax.set_xlabel('Alpha', fontsize=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.15)
        
        # Thinner spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.15)
        
        # Y limit adjustment to fit labels
        max_val = max(fixed_vals.max(), trained_vals.max())
        ax.set_ylim(0, max_val * 1.4)

    # Global legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.82]) 
    plt.savefig(output_path)
    print(f"Comparison plots saved to {output_path}")

if __name__ == "__main__":
    main()
