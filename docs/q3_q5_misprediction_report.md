# Q3/Q5 오예측 시나리오 — 탐색 결과 (게이트: 자연 오예측 미발견)

## 결론: **vision-only 셋에서 자연 발생 Q3/Q5 오예측(top-1 mis-rank)은 구조적으로 안 생긴다.** 억지 실패 없이 멈추고 보고.

## 탐색 데이터 (측정)

### 1. coverage 가드 — 정상 (차단 안 함)
- 학습 워킹셋 15개, 모델 9개. 미학습 조합(예: mobilenet_v2,yolo11n,yolo11x) 다수 존재.
- 미학습 조합으로 예측 시 **차단하지 않고 경고만**(설계대로) → 실험 자체는 가능.

### 2. Path A (미학습 조합 + 고부하) — 오예측 미발생
- 미학습 조합 × 입력률(15~200) × 양 플랫폼(cpu_gpu/cpu_npu)에서 **예측기 top-1이 항상 all-accelerator**
  (all-GPU / all-NPU), CPU 잔류 0, 예측 miss 0.00~0.11.
- 이유: vision-only + 빠른 가속기에선 all-accelerator가 **실제로 near-optimal**이라 예측기가 정확히 고름.
  y1(정규화 처리량) 최대·y2(miss) 최소가 모두 all-accelerator → mis-rank 구조가 없음.

### 3. Path B (NPU 제약) — 오예측 미발생
- runtime이 **NPU 단일-모델-상주 제약을 강제하지 않음**(각 뷰가 자체 .mxq를 상주 로드, dispose는 워커 종료 시만).
  → all-NPU가 all-GPU처럼 작동, reload 스래싱 없음. Q5가 노린 "제약 하 infeasible"이 현 구현엔 없음.

### 4. 런타임 실패 모드 확인 = Q4-like (Q3 아님)
- all-GPU 4모델(yolo11x/l/m/resnet50) 고부하(infps=90, rate-control) → **maxV=342 지속 위반**(GPU contention:
  yolo11x 54fps ≪ λ90).
- 그러나 **더 나은 대안 placement 없음**(무거운 모델을 CPU로 옮기면 더 느림). 즉 top-1이 실패해도 회복할
  대안이 없음 → **Q4(전 후보 실패·bounded envelope)**, Q3(대안으로 회복)가 아님.

## 판정 (§0.3, §4, §5 게이트)
- **Q3의 전제(top-1이 mis-rank, 더 나은 대안 존재)가 vision-only에선 성립 안 함.** top-1(all-accelerator)이
  진짜 최적이고, 실패는 mis-rank가 아니라 **용량 contention**(Q4 성격).
- 원래 논문 Q3의 자연 오예측은 **mixed 셋(LLM/VLM + vision)**에서 발생: LLM/VLM이 가속기를 점유해 예측기가
  vision을 CPU에 남기는 mis-rank. 이건 vision-only에 없다.
- **억지로 실패시키지 않음**(§0.1 조작 금지, §5 게이트).

## 선택지 (사람 판단 — PENDING)
1. **Mixed 셋(LLM/VLM)으로 진짜 Q3/Q5** — 원래 논문 시나리오. LLM/VLM 워커 필요 → vision(onnxruntime)과
   cuDNN 충돌로 **프로세스 분리 선행 필수**(후속 지시문). 이게 authentic한 경로.
2. **Path C — controlled misprediction injection** — top-1을 의도적으로 infeasible 후보로 대체(예: yolo11x를
   CPU로 강제)하고 BoundGuard가 all-GPU로 사이클. **injection임을 논문에 명시**해야 함(§0.2). 다만 대안(all-GPU)이
   자명해 설득력은 mixed보다 약함.
3. **NPU 단일-상주 제약을 runtime에 구현** — Q5의 자연 오예측을 만들려면 NPU reload/DRAM 제약을 실제로 강제해야.
   실행모델 변경이라 별도 설계·확인.

## 무결
- GPU 폴백 0(런타임 확인 run), NPU 무결. 예측기 스키마·배선·β 불변. 측정·탐색만, 코드 로직 무변경.
