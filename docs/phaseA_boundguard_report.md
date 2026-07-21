# Phase A 재도전 — BoundGuard bounded persistence (§4b 관문)

날짜: 2026-07-21 · 범위: vision-only(mobilenet_v2, resnet50, yolo11s) × cpu_gpu·cpu_npu

## 배경
Phase A에서 BoundGuard 2칸이 미시연(lastV≈10.6, 위반 placement로 회귀). 원인 후보 두 가지:
- **(A) 시나리오 부족**: 짧은 페이즈 + 잘못된 mode.
- **(B) 측정 윈도우 구조 문제**: hot-swap 직후 콜드스타트가 stable V(t) 측정에 섞여 정상 배치를 오기각.

## q13 하네스 구조 (읽기)
- **BoundGuard = mode 1(adaptive hot-swap) + 후보 시퀀스**(cand_1..cand_5), 각 후보를 T_v 유지 후
  validate 트리거로 검사. bounded persistence는 per-combo duration + validate 트리거로 강제.
- Phase A는 BoundGuard를 **mode 2(reactive)**로 돌린 것이 오류(D5 이중 의미 함정). reactive 롤백이
  콜드스타트 구간을 측정해 오발동한 것.

## 시나리오 (신규 모델 재현)
- 후보를 신규 예측기(DeployPredictor)로 랭킹 → cand_1..cand_4.
- 스케줄: `combination_stable`(all-CPU 저부하) → `combination_burst`(all-CPU 고부하=위반) →
  `cand_1..N`(랭킹된 placement) + `--combo-trigger cand_i=validate`, `--qos-trigger-tv 3 --epsilon 1.0`.
- mode: BoundGuard=1(+validate), Adaptive=1(단일 cand_1), Static=3.

## 원인 판정 (§3b) — **케이스 A 확정**
BoundGuard 로그(gpu):
```
QoS-trigger active: combo=cand_1 policy=validate eps=1.0 T_v=3.0s
QoS-validate combo=cand_1 elapsed=3.2s V(t)=0.015 eps=1.0 (post-T_v)
QoS-trigger: combo=cand_1 validated (V(t)=0.015 <= eps=1.0); committing.
```
- **validate가 T_v(3.2s) 이후, 즉 콜드스타트가 지난 뒤 V(t)=0.015를 정측정**해 커밋. Phase A의 롤백·오기각
  없음 → **측정 윈도우 코드 버그(케이스 B) 아님. 시나리오/모드 문제(케이스 A)였음** 확정.

## 결과 — failure persistence (V(t)>ε 지속)
| 기법 | cpu_gpu | cpu_npu | 비고 |
|---|---|---|---|
| **Static** | **65s** (maxV 9.2, 회복 없음) | **65s** | 위반 지속 baseline |
| Adaptive | 0s (즉시 회복) | 0s | 단일 hot-swap |
| **BoundGuard** | **0s** (maxV 0.8, lastV 0) | **0s** | validate 트리거로 bounded, 회복 |

- **GPU/NPU 무결**: 전 6 run cuDNN CPU 폴백 **0**. GPU 실가속 확인(cand infer yolo11s 11.5ms, resnet50
  5.7ms, mobilenet_v2 2.9ms — CPU ~28-52ms 대비 명백 가속).

## 관문 재판정
- **BoundGuard PASS (양 플랫폼)**: bounded persistence(각 후보 validate 윈도우 내, 무한 지속 없음) +
  위반 지속이 Static baseline(65s)보다 짧음(0s) + 회복(lastV 0, Phase A의 10.6 회귀 해소).
- **8칸(4기법 × 2플랫폼) 전원 PASS** → **§4b 관문 통과 → Phase B(C3) 착수 가능.**

## 한계·후속 (Phase B에서)
- 이 시나리오는 예측기 top-1이 정확해 **Adaptive도 0s** → BoundGuard의 *distinctive* 우위(오예측 하에서
  Adaptive가 나쁜 배치에 커밋될 때 BoundGuard가 후보 사이클로 회복)는 미시연. 이를 보이려면
  **오예측 시나리오**(top-1 후보 infeasible)가 필요 — `ml_misprediction` 하네스 스타일, Phase B 논문 그림.
- 스케줄이 커밋 후에도 combo-duration으로 다음 후보까지 순회(모든 후보가 양호해 위반 지표엔 무영향).
  "commit-and-stay" 정밀 재현은 Phase B에서 `--stop-after`/validate 커밋 시 잔여 페이즈 스킵으로 정리.
