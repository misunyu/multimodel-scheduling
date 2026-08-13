<!-- provenance
  generated_at: 2026-08-13T14:32:50+09:00
  git_commit: 609fa501d7c6d47731c646e7f6929492a16967f2 (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P4 unified vs specialized
-->


# P4 — joint 예측기(1080행, 45 paired 그룹) vs 플랫폼별 예측기(P1)

동일 프로토콜(GroupKFold 3-fold seed=42, 그룹 키 = (models, rate) — 플랫폼 제외 → 45 paired 그룹, 한 워크로드의 GPU/NPU 행이 항상 같은 fold). feature 37차원 그대로 — exec one-hot이 플랫폼을 구분.

| 플랫폼 | 예측기 | 점수 그룹-ρ | Top-1 | Top-5 | y1 ρ | y2 ρ | y3 ρ |
|---|---|---|---|---|---|---|---|
| gpu | 플랫폼별(P1) | +0.986 | 0.933 | 1.000 | +0.985 | +0.924 | +0.981 |
| gpu | 통합(P4) | +0.975 | 0.933 | 1.000 | +0.989 | +0.922 | +0.976 |
| npu | 플랫폼별(P1) | +0.972 | 0.867 | 1.000 | +0.955 | +0.894 | +0.960 |
| npu | 통합(P4) | +0.984 | 0.867 | 1.000 | +0.973 | +0.890 | +0.962 |

## 세트별 점수 Spearman (통합 − 플랫폼별, 그룹 평균)

| 플랫폼 | 개선 그룹 | 동일(±0.02) | 악화 그룹 |
|---|---|---|---|
| gpu | 4 | 31 | 10 |
| npu | 9 | 34 | 2 |

## 판정 기준 (판정은 논문 쪽에서)

- 대등: 플랫폼별 대비 점수 그룹-ρ 차이 ≤ 0.01 그리고 Top-1 차이 ≤ 1그룹(0.022)
- 열세: 위 기준 초과 하락. 세트별 세부는 `unified_metrics.json`·`unified_oof.csv` 참조.
