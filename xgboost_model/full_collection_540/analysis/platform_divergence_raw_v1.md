<!-- provenance
  generated_at: 2026-07-29T18:32:09+09:00
  git_commit: 475a2089eb49a1906d663508055d3aeb5d5e2d43 (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P2 platform divergence
  betas: [1.0, 0.5]
  score_basis: raw measured window totals
-->


# P2 — 플랫폼별 최적 배치 불일치 (측정 점수 argmax, 45그룹)

점수: S = y1 − 0.3·y2 + β·y3(생성 세트만), **원시 측정 총계** 기준 (원본 compare_platforms.py와 동일 — 예측기 미사용).

| β | 동일 배치 | 다른 배치 | 불일치율 |
|---|---|---|---|
| 1.0 | 16 | 29 | 64.4% |
| 0.5 | 15 | 30 | 66.7% |

β에 민감한 그룹 (한쪽 β에서만 불일치): 1개
- S8 @rate1.5: β=1.0 일치 / β=0.5 불일치

그룹별 전체 내역은 `platform_divergence.csv` 참조.
