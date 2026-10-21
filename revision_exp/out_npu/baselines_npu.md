# Baselines for Table 3(b) (CPU–NPU)

이 규칙은 실행 전에 썼다(Phase N2 지시문 v1.3 §B3). 결과를 본 뒤 바꾸지 않는다.

- 코드: `revision_exp_adapter/run_npu.py`
- 평가: 3× N≥3 4 workloads(rbrsyb, rbybys, rsybys, rbrsybys; 60 후보), α = 0.3, D = D_macro, label은 3회 평균
- evaluator: `revision_exp/evaluate.py` 그대로
  - oracle 집합 = S ≥ max S − 1e-9
  - 선택 = 점수 내림차순, 동점은 combination 번호가 작은 쪽
  - Top-5 = 예측 순서 상위 5개 중 oracle이 있는지

## Oracle

- 실측 S = T − 0.3·D_macro가 최대인 후보다. 동점(1e-9 안)은 모두 정답이다.

## Predictive (Ours)

- XGBoost 2개(T, D), §0 grid, family GroupKFold(3).
- Ŝ = T̂ − 0.3·D̂, clip 없음. 동점은 combination 번호가 작은 쪽이다.

## Latency-based

workload에 지정된 instance identity(big/small)는 고정하고 **device만** 고른다.

- **순서:** instance는 family key 순서로 본다. resnet50_big → resnet50_small → yolov3_big → yolov3_small 중 해당 instance다.
- **배정:**
  - 각 instance를 isolated latency가 가장 짧은 device에 놓는다. latency는 그 instance(변종)의 값이다.
  - 그 device가 NPU이고 이미 점유됐으면, 그 instance에서 다음으로 latency가 짧은 device로 간다.
  - 예: npu0이 점유됐으면 npu1, 둘 다 점유됐으면 cpu.
  - CPU는 점유 제한이 없다.
- **isolated latency:**
  - 출처는 `revision_profiling_npu/dataset/offline_profile.json`의 `entries.<key>.service_ms_p50`이다(closed loop, 단일 instance).
  - CPU는 `<model>_cpu`이며 big/small이 같은 CPU 실행 파일을 쓴다(variant `common`).
  - NPU는 `<model>_<variant>_<npu0|npu1>`이다.
  - CPU–GPU 쪽 Latency-based도 같은 profile 필드(`service_ms_p50`)를 썼다.
- **동점:** cpu → npu0 → npu1 순으로 정한다.
- **확인:** 결과 배치가 그 workload의 측정 후보 안에 있는지 assert한다.
- **R1 참고 배치와 다르다.** R1은 N2-A §3-1의 배치로, N1 §4의 isolated 처리량 순 규칙이다. 이 규칙은 그것과 다르다.
- 3×에서 oracle과 모두 일치하더라도 오류로 보지 않으며 규칙을 바꾸지 않는다.

## Per-instance predictor ([14]-inspired)

`revision_exp/phase3/baselines/`의 코드(instance_rows, fit_instance, compose_select)를 그대로 쓴다. 바꾸는 것은 device set과 입력 경로뿐이다.

| 항목 | CPU–GPU (Phase 3) | CPU–NPU (이번) |
|---|---|---|
| device one-hot | dev_cpu, dev_gpu | dev_cpu, dev_npu0, dev_npu1 |
| isolated capacity | `offline_profile/offline_profile.json` `models[m][dev].throughput_fps` | NPU dataset `offline_profile.json` `entries[...].throughput_fps` (위 Latency-based와 같은 key 규칙, 해당 device·변종) |
| window | 10 s | 30 s (json `window_sec`) |
| 학습 CSV / schedule | `revision_profiling/dataset/*` | `revision_profiling_npu/dataset/*` |

- **그대로 쓰는 것:** instance feature 나머지, target(t_ratio = completed/window/λ, q_i), q_i weight 10(q_i > 0.01), grid, 선택 기준, 합성·선택 규칙.
  - 합성: F̂ = Σ clip(t̂,0)·λ, T̂ = F̂/예측 최대, D̂ = mean clip(q̂,0,1), Ŝ = T̂ − 0.3·D̂.
  - instance feature 나머지: model identity one-hot 128 bucket, λ, 같은/다른 device co-runner 수·λ 합, N.
- **fold:** 직접 예측기와 같은 family → fold 매핑이다.
- **N/A 조건:** 이 범위를 넘는 코드 수정이 필요하면 N/A로 보고한다.
  - 구현 방식: 다른 함수는 import해서 그대로 쓰고, device set이 하드코딩된 `rows_from_schedules`와 `iso_capacity`만 adapter 안에서 같은 로직으로 다시 쓴다.

## Best-of-5

- `revision_exp/baselines.py: best_of_k` 그대로 쓴다.
  - 후보 5개를 비복원으로 추출하고 실측 S 최대를 고른다. 1,000회 반복한다.
  - seed는 SeedSequence([42, trial, workload index])이다.
- **보고 값:** 평균 S, 추출 간 표준편차, oracle 포함률.
- **Top-1:** 추출 평균 적중률을 소수로 적는다. n/4 형식과 구분하기 위해서다.
- **후보가 5개 이하인 workload:** 후보 전체를 쓴다(N=1: 3개). 11 workloads 참고 표에만 해당한다. N=2(7개)는 해당하지 않는다.
- **sampler 검증:** primary 4 workloads의 무작위 Top-1 기대값은 (3×5/13 + 5/21)/4 ≈ 0.348이다. 측정값과 비교해 기록한다.

## All-CPU

- 모든 instance를 CPU에 두는 배치(`combination_1`)다. 참고 행으로만 둔다.

## All-accelerator

- NPU당 instance는 1개로 제한된다(residency). N≥3에서는 NPU 2개에 모든 instance를 올릴 수 없어 feasible하지 않다. 그래서 표에 넣지 않는다.
