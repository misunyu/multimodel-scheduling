import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set standard academic plot style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def load_vscore_series(csv_path):
    df = pd.read_csv(csv_path)
    # v_score 컬럼이 없는 경우 예외 처리
    if 'v_score' not in df.columns:
        return None
    
    # 콤비네이션이 바뀌는 지점 찾기
    combos = df['combination'].tolist()
    switch_idx = 0
    for i in range(1, len(combos)):
        if combos[i] != combos[i-1]:
            switch_idx = i
            break
    
    # v_score 시리즈 추출
    v_scores = df['v_score'].tolist()
    
    # 전환 전후 데이터 필터링 (전환 전 약 20초, 전환 후 약 40초 정도)
    # t=20 지점이 전환점이 되도록 조정
    event_t = 20
    
    # 전환점(switch_idx)을 t=20으로 맞춤
    # 데이터가 부족하면 패딩 또는 절단
    start_idx = max(0, switch_idx - event_t)
    end_idx = min(len(v_scores), switch_idx + 40)
    
    series = v_scores[start_idx:end_idx]
    time = np.arange(len(series))
    
    # 만약 전환점이 t=20보다 앞에 있다면 (즉 start_idx가 0인 경우)
    # time 축을 밀어서 전환점이 20이 되게 함
    actual_switch_in_series = switch_idx - start_idx
    time = time + (event_t - actual_switch_in_series)
    
    return time, series

def generate_graph():
    # 데이터 경로 (실제 실험 결과)
    csv_mode1 = "results/adaptive_metrics_mode1_20260403_135343.csv" # Adaptive Hot-swap (Static Limitation 대용)
    csv_mode2 = "results/adaptive_metrics_mode2_20260403_135343.csv" # Reactive/Rollback (BoundGuard)
    
    t1, v1 = load_vscore_series(csv_mode1)
    t2, v2 = load_vscore_series(csv_mode2)
    
    plt.figure(figsize=(8, 5))
    
    # Static Limitation (Adaptive Hot-swap): 
    # 나쁜 배치가 적용된 후 계속 높은 스코어를 유지하거나 상승함
    plt.plot(t1, v1, label='Static/Hot-swap', color='tab:red', linestyle='--', linewidth=2.5)
    
    # BoundGuard:
    # 나쁜 배치 적용 직후 스코어가 치솟지만, 롤백 후 안정화됨
    plt.plot(t2, v2, label='BoundGuard (Rollback)', color='tab:blue', linestyle='-', linewidth=2.5)
    
    # Annotation for Event (t=20)
    t_event = 20
    plt.axvline(x=t_event, color='gray', linestyle=':', alpha=0.8)
    
    # 스코어 값에 따라 화살표 위치 조정
    max_v = max(max(v1), max(v2))
    plt.annotate('Input Rate Increase\n& Deploy Change', xy=(t_event, max_v*0.1), xytext=(t_event + 2, max_v*0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=11)
    
    # Rollback 지점 표시 (Mode 2에서 스코어가 급격히 떨어지는 지점)
    # 대략 t=25~30 사이
    plt.annotate('Rollback to Stable', xy=(28, v2[np.argmin(np.abs(t2-28))]), xytext=(35, v2[np.argmin(np.abs(t2-28))]+10),
                 arrowprops=dict(arrowstyle="->", color='tab:blue', connectionstyle="arc3,rad=.2"),
                 fontsize=11, color='tab:blue')

    plt.xlabel('Time (s)')
    plt.ylabel('Violation Score')
    plt.legend(loc='upper right')
    
    plt.ylim(0, max_v * 1.2)
    plt.xlim(0, 60)
    
    plt.savefig('dynamic_load_adaptation.pdf')
    print("Graph generated using real experimental data: dynamic_load_adaptation.pdf")

if __name__ == "__main__":
    generate_graph()
