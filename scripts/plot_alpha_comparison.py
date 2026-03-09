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
        ('avg_score_ge3', 'Average Score'),
        ('top1_acc_ge3', 'Top-1 Accuracy'),
        ('top5_acc_ge3', 'Top-5 Accuracy')
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
            ax.set_ylabel(title, fontsize=10)
        # ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(alphas, fontsize=9)
        ax.set_xlabel('Alpha', fontsize=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.15)
        
        # Thinner spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.15)
        
        # Y limit adjustment
        ax.set_ylim(0, 1.1)
        # Ensure 1.1 doesn't show in ticks
        ticks = [t for t in ax.get_yticks() if t <= 1.0]
        ax.set_yticks(ticks)

    # Global legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1.0]) 
    plt.savefig(output_path)
    print(f"Comparison plots saved to {output_path}")

    # CSV 파일로 저장
    csv_output_path = output_path.with_suffix('.csv')
    
    # xgboost 모델명 추가
    df_csv = df.copy()
    df_csv['xgboost_model_name_fixed'] = "xgb_model_x3_double"
    df_csv['xgboost_model_name_trained'] = df_csv['alpha'].apply(lambda a: f"xgb_model_x3_score_alpha{a}")
    
    # 열 순서 재배치 (각 그룹 옆에 모델명 배치)
    column_order = [
        'alpha',
        'xgboost_model_name_fixed', 'avg_score_ge3_fixed', 'top1_acc_ge3_fixed', 'top5_acc_ge3_fixed',
        'xgboost_model_name_trained', 'avg_score_ge3_trained', 'top1_acc_ge3_trained', 'top5_acc_ge3_trained'
    ]
    df_csv = df_csv[column_order]
    
    df_csv.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")

if __name__ == "__main__":
    main()
