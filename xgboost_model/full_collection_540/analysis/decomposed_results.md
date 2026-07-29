<!-- provenance
  generated_at: 2026-07-29T18:41:21+09:00
  git_commit: c37c5bd761fbbe8ea50202b2aca82eb9b17fca1b (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P6 decomposed modeling
-->


# P6 — 분해 모델링 (뷰 단위 저하율 예측 → 합성)

target r = min(1, throughput/min(infps, capacity)). vision 뷰만 학습 (뷰별 tokens 미보존 → y3 분해 불가, P1의 세트 수준 ŷ3 공유). fold·그룹은 P1과 동일.

## 뷰 수준 OOF

| 플랫폼 | vision 뷰 | clip된 target | OOF Spearman | OOF MAE |
|---|---|---|---|---|
| gpu | 2220 | 0 | +0.993 | 0.0170 |
| npu | 2220 | 0 | +0.992 | 0.0201 |

## y2 재구성 오차 (실측 저하율 합성 vs 실측 y2 — 모델 오차와 분리된 합성 근사 자체의 오차)

| 플랫폼 | raw MAE | pooled ρ | 그룹 ρ 평균 | (참고) y1 합성 상대오차 |
|---|---|---|---|---|
| gpu | 0.1489 | +0.834 | +0.877 | 0.0000 |
| npu | 0.1687 | +0.763 | +0.720 | 0.0000 |

(ŷ2_decomp = 뷰별 (1−r)의 비가중 평균은 창 수준 y2의 **근사**다: 실제 y2는 요청 수 가중이 다르고 생성 뷰도 포함한다. 생성 뷰는 합성에서 제외.)

## 점수 수준 3열 비교 (45그룹, S = ŷ1 − 0.3·ŷ2 + 1.0·ŷ3)

| 플랫폼 | 방법 | Top-1 | Top-5 | 그룹 Spearman |
|---|---|---|---|---|
| gpu | P1 직접 회귀 | 0.933 | 1.000 | +0.986 |
| gpu | P6 분해+합성 | 0.911 | 1.000 | +0.931 |
| gpu | P5 greedy(speedup) | 0.424(수집 33/45, 미수집 12) | — | — |
| gpu | P5 greedy(load_factor) | 0.341(수집 41/45, 미수집 4) | — | — |
| npu | P1 직접 회귀 | 0.867 | 1.000 | +0.972 |
| npu | P6 분해+합성 | 0.622 | 1.000 | +0.911 |
| npu | P5 greedy(speedup) | 0.333(수집 33/45, 미수집 12) | — | — |
| npu | P5 greedy(load_factor) | 0.250(수집 36/45, 미수집 9) | — | — |

(P5는 배치 1개만 내므로 Top-5·Spearman 미정의. 세부: `decomposed_metrics.json`, `decomposed_view_oof.csv`, `greedy_results.csv`.)
