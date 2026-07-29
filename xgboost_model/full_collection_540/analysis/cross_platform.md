<!-- provenance
  generated_at: 2026-07-29T18:36:14+09:00
  git_commit: 42bfd0c393920bb8765a9d7d0028fda5acdac72f (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P3 cross-platform transfer
  boosters: artifacts/gkf_* fold boosters (P1)
-->


# P3 — 교차 플랫폼 전이 (P1 fold 부스터, 재학습 없음)

대각선 = P1 OOF 그대로. 비대각선 = 학습 플랫폼 fold 부스터 3개 예측의 평균. 평가측 정규화는 평가 플랫폼 그룹 통계(순위 보존).

가속기 one-hot 주의: 전이 시 평가 플랫폼의 가속기 열은 학습에서 상수 0이던 열이다. `asis`는 무보정, `swap`은 exec_gpu↔exec_npu 스왑 — 두 변형의 차이 자체가 결과다. static/capacity feature는 항상 평가 플랫폼 값이다.

## 변형: asis

| 학습→평가 | y1 ρ | y2 ρ | y3 ρ | 점수 그룹-ρ | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| gpu→gpu | +0.985 | +0.924 | +0.981 | +0.986 | 0.933 | 1.000 |
| gpu→npu | +0.934 | +0.816 | +0.736 | +0.902 | 0.800 | 1.000 |
| npu→gpu | +0.956 | +0.580 | +0.750 | +0.927 | 0.933 | 1.000 |
| npu→npu | +0.955 | +0.894 | +0.960 | +0.972 | 0.867 | 1.000 |

## 변형: swap

| 학습→평가 | y1 ρ | y2 ρ | y3 ρ | 점수 그룹-ρ | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| gpu→gpu | +0.985 | +0.924 | +0.981 | +0.986 | 0.933 | 1.000 |
| gpu→npu | +0.936 | +0.835 | +0.736 | +0.903 | 0.800 | 1.000 |
| npu→gpu | +0.953 | +0.637 | +0.755 | +0.929 | 0.933 | 1.000 |
| npu→npu | +0.955 | +0.894 | +0.960 | +0.972 | 0.867 | 1.000 |

- 점수 그룹-ρ 평균: 대각선 +0.979 → 비대각선(asis) +0.915 (하락 0.064)
- 점수 그룹-ρ 평균: 대각선 +0.979 → 비대각선(swap) +0.916 (하락 0.063)
- fold 예측 표준편차(행 평균, gpu→npu 예): asis {'y1': 0.02232022024691105, 'y2': 0.033637672662734985, 'y3': 0.02681000903248787} / swap {'y1': 0.026612693443894386, 'y2': 0.03350943699479103, 'y3': 0.02701721340417862}

기계가독 전체 수치: `cross_platform_metrics.json`
