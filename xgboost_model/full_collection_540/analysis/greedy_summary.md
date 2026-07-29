<!-- provenance
  generated_at: 2026-07-29T18:33:47+09:00
  git_commit: 154f11480b290e3425256b0b0b20b74f093fdacc (pre-commit HEAD)
  alpha: 0.3
  beta: 1.0
  seed: 42
  input_data: ['/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json', '/home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json']
  experiment: P5 capacity greedy
  static_profile: /home/msyu/PycharmProjects/multimodel-scheduling-mobilint/xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json
-->


# P5 — Capacity greedy baseline (45그룹 × 2변형 × 2플랫폼)

배정: 정렬 순서대로 누적 load_factor ≤ 1.0이면 가속기, 아니면 CPU. qwen2_vl은 CPU 불가라 용량 초과여도 가속기 강제(누적에 반영).

| 플랫폼 | 변형 | Top-1 일치율(수집분) | 평균 oracle비(정규화) | 평균 oracle비(원시) | 미수집 그룹 |
|---|---|---|---|---|---|
| gpu | load_factor | 0.341 (14/41) | 0.575 | 0.669 | 4/45 |
| gpu | speedup | 0.424 (14/33) | 0.734 | 0.786 | 12/45 |
| npu | load_factor | 0.250 (9/36) | 0.590 | 0.669 | 9/45 |
| npu | speedup | 0.333 (11/33) | 0.793 | 0.760 | 12/45 |

## 미수집 그룹 목록 (greedy 산출 배치가 수집 조합에 없음)

- gpu/load_factor: S7@1.5, S7@1.0, S9@2.0, S6@2.0
- gpu/speedup: S8@2.0, S8@1.5, S8@1.0, S10@2.0, S10@1.5, S7@1.5, S7@2.0, S9@3.0, S9@2.0, S6@2.0, S6@1.5, S6@1.0
- npu/load_factor: S8@2.0, S8@1.5, S8@1.0, S10@1.0, S7@1.5, S7@2.0, S7@1.0, S9@2.0, S6@1.5
- npu/speedup: S8@2.0, S8@1.5, S8@1.0, S10@2.0, S10@1.5, S7@1.5, S7@2.0, S9@3.0, S9@2.0, S6@2.0, S6@1.5, S6@1.0

## 생성 모델의 capacity_fps에 쓰인 값 (판단은 논문 쪽)

capacity_fps = 1000/static_infer_sel. 생성 모델의 static_infer_sel은 prefill 지연(ms)이라 '초당 처리 가능 요청 수'로서의 의미가 제한적이다:

| 모델 | 플랫폼 | accel_infer(ms) | capacity_fps | infps | load_factor |
|---|---|---|---|---|---|
| qwen2_vl | gpu | 99.8 | 10.02 | 0.434 | 0.0433 |
| llama1b | gpu | 9.5 | 105.26 | 0.276 | 0.0026 |
| qwen2_vl | gpu | 99.8 | 10.02 | 0.326 | 0.0325 |
| llama1b | gpu | 9.5 | 105.26 | 0.207 | 0.0020 |
| qwen2_vl | gpu | 99.8 | 10.02 | 0.217 | 0.0217 |
| llama1b | gpu | 9.5 | 105.26 | 0.138 | 0.0013 |
| qwen2_vl | gpu | 99.8 | 10.02 | 0.651 | 0.0650 |
| llama1b | gpu | 9.5 | 105.26 | 0.414 | 0.0039 |

(NPU에서는 llama1b 81.2ms→12.3fps, qwen2_vl 613.8ms→1.63fps로 동일 정의를 적용. 전체 내역은 `greedy_results.csv`.)
