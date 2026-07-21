# Phase A — 네 기법 §4b 실측 재검사 (vision-only)

날짜: 2026-07-21 · 범위: vision-only 셋 × 양 플랫폼(cpu_npu, cpu_gpu) · mixed(LLM/VLM)는 범위 밖(후속)

## 하네스
- `schedule_executor_main.py --schedule <yaml> --adaptive-mode {3,0,1,2} --combo-duration ... --metrics-csv ...`
  를 `QT_QPA_PLATFORM=offscreen`으로 헤드리스 실행.
- 시나리오: 2-페이즈 위반→회복. `combination_initial`(yolo11s+resnet50 둘 다 CPU, infps=60 → L_SLO=16.7ms,
  CPU 추론이 SLO 초과 → 위반) → `combination_offload`(가속기로 이동 → 회복). 각 페이즈 12s.
- 워킹셋: `resnet50, yolo11s`(coverage 내 vision-only).

## 8칸 매트릭스

| 기법 (mode) | 플랫폼 | 완료 | 크래시 | GPU 폴백 | 고유 동작 (V(t) 궤적) | 판정 |
|---|---|---|---|---|---|---|
| Static (3) | cpu_gpu | clean quit | 없음 | 0 | maxV 18.7 → **lastV 18.5**(회복 없음, 위반 지속) | PASS |
| Static (3) | cpu_npu | clean quit | 없음 | 0 | lastV 18.4 (위반 지속) | PASS |
| Stop-restart (0) | cpu_gpu | clean quit | 없음 | 0 | maxV **37.8**(재시작 콜드스타트 스파이크) → **lastV 0.0**(회복) | PASS |
| Stop-restart (0) | cpu_npu | clean quit | 없음 | 0 | lastV 0.0 (회복) | PASS |
| Adaptive (1) | cpu_gpu | clean quit | 없음 | 0 | 3 hot-swap, 스파이크 없이 **lastV 2.1**(회복) | PASS |
| Adaptive (1) | cpu_npu | clean quit | 없음 | 0 | lastV 2.5 (회복) | PASS |
| BoundGuard (2) | cpu_gpu | clean quit | 없음 | 0 | validation window + reactive rollback 동작. rows 34(긴 관측). **lastV 10.6** — 이 시나리오에서 CPU로 롤백 | **부분** |
| BoundGuard (2) | cpu_npu | clean quit | 없음 | 0 | lastV 10.9 — 동일 | **부분** |

## GPU/NPU 무결 (핵심)
- **전 8 run에서 cuDNN CPU 폴백 0** (`Falling back to CPUExecutionProvider` 없음). torch-free vision 유지 확인.
- 실가속 실측(offload 페이즈, stop-restart는 핸들러 리셋으로 순수 측정):
  - **cpu_gpu: yolo11s 5.9ms, resnet50 5.2ms** (GPU)
  - **cpu_npu: yolo11s 4.1ms, resnet50 1.9ms** (NPU .mxq)
- Adaptive의 offload infer_ms가 높게(59ms) 나오는 것은 **CPU 폴백이 아니라 핸들러 누적평균 아티팩트**
  (hot-swap은 카운터를 리셋하지 않아 초기 CPU 샘플이 평균에 잔존; stop-restart는 리셋하므로 순수 GPU값).
  폴백 0 + stop-restart 순수값이 실가속을 증명.

## 판정 요약
- **static / stop-restart / adaptive (6칸)**: **PASS**. 각 고유 동작 정확 관찰
  (static 위반 지속 / stop-restart 재시작 스파이크 후 회복 / adaptive 무스파이크 hot-swap 회복).
- **BoundGuard (2칸)**: 메커니즘(validation window + reactive rollback)은 **실제 동작**하고 크래시·폴백 없이
  완료했으나, **distinctive benefit(bounded persistence로 baselines보다 짧은 위반)은 이 최소 2-페이즈
  시나리오에서 미시연**. reactive 매니저가 hot-swap 직후 콜드스타트 구간의 stable V(t)를 측정해
  롤백을 발동, 위반 placement(CPU)로 되돌아가 lastV가 높게(10.6) 끝났다. 이는 코드 결함이 아니라
  **시나리오 설계 이슈**(12s 페이즈 + 콜드스타트, 후보 사이클/validate 트리거 부재)로 판단.

## 게이트 판단
- 6/8 명확 PASS + 8/8 clean 실행 + GPU/NPU 무결.
- **BoundGuard의 bounded-persistence 우위는 미시연** → §A3 관문 기준 "8칸 전원 PASS"에 **미달**.
- 필요 조치: BoundGuard용 **후보 사이클링 시나리오**(다중 후보 combination + `--combo-trigger ...=validate`
  + T_v/epsilon 설정, 충분한 페이즈 길이)로 재실행해 bounded persistence(단일 배치 ≤ T_v+δ, 위반 지속이
  static/adaptive보다 짧게 bound)를 관찰해야 관문 통과. → Phase B 착수 전 선행.

## 잔여
- Adaptive infer_ms 누적평균 아티팩트: 측정 명료화를 위해 hot-swap 시 핸들러 카운터 리셋 옵션 고려(측정
  이슈, 기법 로직 무관).
