# 예측기 y2 신호 검사 (Task A) — Q3 전제 검증

날짜: 2026-07-22 · 범위: cpu_gpu·cpu_npu full_collection_540 (n=540 each) · 검사 전용 (실험/코드변경 없음)

## 가설
"학습 데이터 수집 당시 miss가 거의 발생하지 않아 y2(deadline miss rate)가 degenerate했을 수 있다"
(rate-control이 C3에서 신규 → 구 feeder의 λ≈30fps가 가속기 μ 수백 fps보다 훨씬 낮아 miss≈0).

## A1. y2 원천 및 분포
- 파이프라인: `deploy_selector_xgb_suite.featurize_window` — `y2 = derived.deadline_miss_rate`
  (없으면 `total.deadline_miss_rate`, 그것도 nan이면 `y2=0.0` 기본값; L417-418).
- 학습 record `total`에 `deadline_miss_rate` **540/540 존재** (기본값 폴백 아님).

| 플랫폼 | n | y2 min | y2 median | y2 max | y2≈0 비율 |
|---|---|---|---|---|---|
| cpu_gpu | 540 | 0.152 | 0.858 | 1.000 | **0%** |
| cpu_npu | 540 | 0.287 | 0.935 | 1.000 | **0%** |

→ **y2는 degenerate 아님.** 넓은 분포(0.15~1.0), zero 비율 0%. 가설(신호 부재) **기각.**

## A2. 수집 λ
- `rate_factor` ∈ {1.0, 1.5, 2.0, 3.0, 4.0}.
- 달성 vision throughput_fps: min=1, median=39, **max=160** (파이프라인 μ*_GPU≈137 초과).
- → 고λ(경합) 구간도 **일부 수집됨**. 단 median 39fps로 저~중λ 편중. 학습 record는 llama1b·qwen2_vl와
  vision을 **동시 실행**(경합 포함)해 miss를 측정 → 예측기 y2는 LLM-vision GPU 경합을 원리적으로 반영.

## A4. α 민감도 (가장 결정적)
mixed set(llama1b, qwen2_vl, resnet50, yolo11s) 2^4 조합에 대해 α ∈ {0, 0.3, 1, 3, 10} 스윕:

| α | top-1 | top-1의 y2 |
|---|---|---|
| 0.0 → 10.0 (전부) | combination_15 (all-GPU) | 0.003 |

→ α를 33× 키워도 top-1 불변. **단, 이는 "y2 신호 부재"가 아니라 all-GPU가 y1(throughput)·y2(miss)
  양쪽에서 동시 지배**하기 때문 (all-GPU가 최고 throughput AND 최저 miss 0.003). y2가 랭킹을
  뒤집을 여지 자체가 없음 — 페널티를 아무리 키워도 최저-miss 후보가 top-1.

## Task A 판정: **y2 신호 있음 (present)**
- 분포 넓음(0.15~1.0), 예측기가 실제로 사용, 학습 시 경합 동시측정 반영.
- all-GPU가 y2=0.003으로 최저 → 예측기의 all-GPU 선택은 **y2 신호에 근거한 정당한 선택**.
- **주의(distribution shift 잔여 위험)**: 수집 median λ가 39fps로 저~중부하 편중. C3/Q3의 고λ 지속
  경합에서 all-GPU의 실제 miss가 예측 y2=0.003보다 클 수 있음(과소예측). 그러나 이는 "다른 배치가
  더 낫다"는 뜻이 아니라 **모든 배치가 함께 나빠지는 용량 문제** — Task B에서 확정.
