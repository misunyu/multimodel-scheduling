<!-- provenance
  generated_at: 2026-07-29T18:30:03+09:00
  git_commit: a8728e67ddf0c96db6e5cd5d57684cedc57136fb (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P1 groupkfold ranking
  boosters: artifacts/gkf_cpu_{gpu,npu}_*.json
-->


# P1 — GroupKFold OOF 결과 (fold 누수 제거)

fold: `(models, rate_factor)` 그룹 단위 3-fold (seed=42), 정규화 통계는 그룹 내부로 한정.

## 점수 수준 (S = ŷ1 − 0.3·ŷ2 + 1.0·ŷ3[생성 세트만], 45그룹)

| 플랫폼 | Top-1 | Top-5 | 그룹 Spearman 평균 |
|---|---|---|---|
| CPU-GPU | 0.933 | 1.000 | +0.986 |
| CPU-NPU | 0.867 | 1.000 | +0.972 |

## target별 OOF (pooled / 그룹평균 Spearman, MAE)

| 플랫폼 | target | pooled ρ | 그룹 ρ 평균 | MAE | 평가 행수 |
|---|---|---|---|---|---|
| gpu | y1 | +0.985 | +0.989 | 0.0292 | 540 |
| gpu | y2 | +0.924 | +0.919 | 0.0810 | 540 |
| gpu | y3 | +0.981 | +0.913 | 0.0351 | 372 |
| npu | y1 | +0.955 | +0.962 | 0.0615 | 540 |
| npu | y2 | +0.894 | +0.917 | 0.0942 | 540 |
| npu | y3 | +0.960 | +0.920 | 0.0259 | 372 |

## 참고: infps=0 변형 (기존 아티팩트의 feature 결함 재현)

기존 학습 경로는 창에 schedule 힌트가 없어 view.infps 계열 feature가 전부 0으로 들어갔다. 동일 fold·동일 하이퍼파라미터로 feature만 바꾼 비교:

| 플랫폼 | 변형 | Top-1 | Top-5 | 그룹 Spearman | y1 ρ | y2 ρ | y3 ρ (pooled) |
|---|---|---|---|---|---|---|---|
| gpu | 실제 infps | 0.933 | 1.000 | +0.986 | +0.985 | +0.924 | +0.981 |
| gpu | infps=0 | 0.889 | 1.000 | +0.967 | +0.965 | +0.889 | +0.928 |
| npu | 실제 infps | 0.867 | 1.000 | +0.972 | +0.955 | +0.894 | +0.960 |
| npu | infps=0 | 0.800 | 1.000 | +0.954 | +0.946 | +0.867 | +0.928 |

채택 하이퍼파라미터·fold 구성은 `groupkfold_{platform}_metrics.json` 참조.
