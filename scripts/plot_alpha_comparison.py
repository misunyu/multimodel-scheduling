import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

def load_metrics_from_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return None
    
    # Custom parsing because the CSV has non-standard rows
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    metrics = {
        'top1_acc_ge3': None,
        'top5_acc_ge3': None,
        'avg_score_ge3': None
    }
    
    for line in lines:
        parts = line.split(',')
        if parts[0] == "Top-1 Accuracy (>= 3 models)":
            metrics['top1_acc_ge3'] = float(parts[1])
        elif parts[0] == "Top-5 Accuracy (>= 3 models)":
            metrics['top5_acc_ge3'] = float(parts[1])
        elif parts[0] == "Average (>= 3 models)":
            # Average (>= 3 models),,0.96,0.06,0.95 -> index 4 is the score
            if len(parts) >= 5 and parts[4]:
                metrics['avg_score_ge3'] = float(parts[4])
            
    return metrics

def main():
    # alpha 범위 설정 (0.1부터 1.0까지 0.1씩)
    alphas = [round(x * 0.1, 1) for x in range(1, 11)]
    
    prediction_dir = Path("xgboost_model/prediction_result")
    output_path = Path("experimental_results/alpha_comparison_plots.pdf")

    fixed_data = []
    trained_data = []

    for alpha in alphas:
        # Fixed model: xgb_model_x3_double
        fixed_csv = prediction_dir / f"prediction_result_test_x3_double_alpha_{alpha}.csv"
        f_metrics = load_metrics_from_csv(fixed_csv)
        if f_metrics:
            fixed_data.append({
                'alpha': alpha,
                'avg_score_ge3': f_metrics['avg_score_ge3'],
                'top1_acc_ge3': f_metrics['top1_acc_ge3'],
                'top5_acc_ge3': f_metrics['top5_acc_ge3']
            })
            
        # Trained model: xgb_model_x3_score_alphaXX
        trained_csv = prediction_dir / f"prediction_result_test_x3_score_alpha_{alpha}.csv"
        t_metrics = load_metrics_from_csv(trained_csv)
        if t_metrics:
            trained_data.append({
                'alpha': alpha,
                'avg_score_ge3': t_metrics['avg_score_ge3'],
                'top1_acc_ge3': t_metrics['top1_acc_ge3'],
                'top5_acc_ge3': t_metrics['top5_acc_ge3']
            })

    if not fixed_data or not trained_data:
        print("Required metrics data not found.")
        return

    # 데이터 로드
    df_fixed = pd.DataFrame(fixed_data)
    df_trained = pd.DataFrame(trained_data)

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
