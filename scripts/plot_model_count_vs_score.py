import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams
import csv

def extract_scores(file_path, actual=False):
    scores = {}
    if not Path(file_path).exists():
        print(f"Warning: {file_path} not found.")
        return scores
    
    prefix = "Actual Best Average (>= " if actual else "Average (>= "
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # "Average (>= N models)" 또는 "Actual Best Average (>= N models)" 행 찾기
            if row[0].startswith(prefix):
                try:
                    # n 추출
                    n_models = int(row[0].split(">= ")[1].split(" ")[0])
                    # score는 마지막 컬럼
                    score_val = float(row[-1])
                    scores[n_models] = score_val
                except (IndexError, ValueError):
                    continue
    return scores

def main():
    # 파일 경로 설정
    files = {
        'Ours': 'experimental_results/xgb_best_results_x3_double.csv',
        'Latency-based': 'experimental_results/latency_based_best_results_x3.csv',
        'Best-of-5': 'experimental_results/random_search_base_results_x3.csv'
    }
    
    output_path = Path("experimental_results/model_count_vs_score.pdf")

    # 데이터 추출
    data = {}
    for label, path in files.items():
        data[label] = extract_scores(path)
    
    # Actual Best Average 추출 (Ours 파일에서)
    actual_best_scores = extract_scores(files['Ours'], actual=True)

    # 그래프 스타일 설정
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    
    plt.figure(figsize=(3.2, 2.5))
    
    model_counts = sorted([3, 4, 5, 6, 7, 8])
    markers = ['o', 's', '^', 'D']
    # 진한 색상 설정
    colors = ['tab:red', 'tab:blue', 'forestgreen', 'black'] 

    # Oracle (Upper Bound) 플롯
    if actual_best_scores:
        y_actual = [actual_best_scores.get(n, np.nan) for n in model_counts]
        plt.plot(model_counts, y_actual, marker=markers[3], 
                 label='Oracle', color=colors[3], linewidth=0.5, markersize=2, linestyle='--')

    # 기존 방법들 플롯 (Ours, Latency-based, Best-of-5)
    labels_order = ['Ours', 'Best-of-5', 'Latency-based']
    plot_configs = {
        'Ours': (markers[0], colors[0]),
        'Latency-based': (markers[1], colors[1]),
        'Best-of-5': (markers[2], colors[2])
    }
    
    for label in labels_order:
        scores = data[label]
        marker, color = plot_configs[label]
        y_values = [scores.get(n, np.nan) for n in model_counts]
        plt.plot(model_counts, y_values, marker=marker, 
                 label=label, color=color, linewidth=0.5, markersize=2)

    plt.xlabel('Number of Models (>= N)', fontsize=6)
    plt.ylabel('Average Score', fontsize=6)
    plt.title('Score vs. Number of Models', fontsize=7)
    plt.xticks(model_counts, fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.1)
    
    # Legend 순서 조정: Oracle (Upper Bound), Ours, Best-of-5 (Oracle), Latency-based
    ax = plt.gca()
    # Thinner spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3] # 현재 [Oracle, Ours, Best-of-5, Latency-based] 순서임
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
               loc='upper center', bbox_to_anchor=(0.5, 1.25),
               ncol=4, fontsize=5.5, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path)
    print(f"Line graph saved to {output_path}")

    # CSV 파일로 저장
    csv_output_path = output_path.with_suffix('.csv')
    df_data = {'Model Count': model_counts}
    
    # Oracle 데이터 추가
    if actual_best_scores:
        df_data['Oracle'] = [actual_best_scores.get(n, np.nan) for n in model_counts]
    
    # 각 기법별 데이터 추가
    for label in labels_order:
        scores = data[label]
        df_data[label] = [scores.get(n, np.nan) for n in model_counts]
    
    df_results = pd.DataFrame(df_data)
    df_results.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")

if __name__ == "__main__":
    main()
