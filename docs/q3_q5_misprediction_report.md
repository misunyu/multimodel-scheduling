> ℹ️ [2026-07-25] Q3 수치 정의: persist 정의·정본은 c2_integrity_report.md 참조 → [c2_integrity_report.md](c2_integrity_report.md) 작업 4.

# Q3/Q5 오예측 시나리오 재실험 — mixed 셋에서 자연 발생 Q3 실증

날짜: 2026-07-22 · Q3(CPU–GPU) 완료 · Q5(CPU–NPU) co-load feasible·게이트 · main.tex 불변

> 이전 판(vision-only 탐색)의 결론 "자연 오예측 미발견"은 **여전히 유효**하다: Q3는 vision-only가 아니라
> **mixed 셋(LLM background + vision)**에서만 발생한다. 그때 제시한 선택지 #1(mixed 셋)을 이 실험이 실행했다.

## 결론
- **Q3: 자연 발생 오예측 실증 성공.** 예측기의 진짜 top-1(gggg=all-GPU, LLM 포함)이 고부하에서 QoS 미달 →
  Static/Stop-restart/Adaptive는 지속 위반, **BoundGuard만 예측기 top-5를 순회해 회복**(cand_5=LLM→CPU).
  **injection 아님** — 후보는 예측기의 실제 top-5, 조작 없음(§0.1 준수).
- **Q5: co-load 실현 가능 확인**(vision .mxq + llama1b .mxq가 aries0 co-resident 성공). 전체 실험은 NPU
  전용 λ 튜닝 필요 → 게이트(§Q5).

## 오예측 유도 경로: **경로 A (자연 발생)**
- 워킹셋: llama1b(background) + yolo11s·resnet50·mobilenet_v2(foreground vision). cpu_gpu 예측기, α=0.3 β=0.5.
- coverage 가드: 미학습 조합을 **차단 안 함**(스키마 검사·경고만) → 실험 가능.
- **예측기 top-1 = gggg**(all-GPU, LLM도 GPU), pred_score=1.473, 예측 y2=0.000. 예측기의 **진짜** 선택.
- 실패 원인 = 목적함수 불일치: 점수가 y3(LLM 토큰)를 보상해 LLM을 GPU에 두길 선호하나, 고입력률(λ=80)에서
  LLM이 GPU를 점유해 vision이 deadline 미달. 예측기는 GPU 경합을 모델링 안 해 y2=0.000으로 과소예측.
- top-5(=BoundGuard 후보, `df.head(5)`): **gggg, gggc, ggcg, gcgg, cggg**. cand_1..4는 전부 LLM=GPU 유지
  (회복 불가), **cand_5=cggg = LLM→CPU**(회복). cand_5는 정확히 N_cand=5 경계(candidate_ranking_check.md).

## Q3 실험 (CPU–GPU) — 네 기법 동일 시나리오
- 시나리오: stable(feasible, λ=25) → burst(gggg, λ=80, 위반) → 후보 순회.
- 부하 노브(전부 기본=이전 동작 보존): `FSRR_RATE_REPLICATE=1`(λ=infps 강제), `FSRR_FRAME_BUFFER=12`
  (backlog→latency), slo_ms=15, ε=1.0, T_v=3s.
- BoundGuard = **mode 1 + validate**(D5 확인: 로그 `policy=validate` 4회, cand_1..4 각 V>ε로 advance).

| 기법 | mode | maxV | 최종 V | failure persist | 회복 |
|---|---|---|---|---|---|
| Static | 3 | 4.08 | **3.99** | 78s | ✗ |
| Stop-restart | 0 | 4.23 | **4.08** | 79s | ✗ |
| Adaptive | 1 | 4.43 | **4.09** | 78s | ✗ |
| **BoundGuard** | 1+validate | 5.17 | **0.54** | **35s** | **✓ (cand_5=LLM→CPU)** |

- **질적 패턴 재현**: 전 기법 위반 진입 → Static/Stop-restart/Adaptive는 top-1(gggg)에 머물러 **V≈4 지속** →
  **BoundGuard만** validate 실패를 감지해 cand_1→…→cand_5 순회, cand_5에서 **LLM을 GPU→CPU로 거친 전환**
  (bg 로그: `stop llama1b/gpu → start llama1b/cpu`)하여 **V가 ε 아래로 수렴**(0.54).
- BoundGuard validate 로그: cand_1 V=6.01, cand_2 V=3.89, cand_3 V=4.81, cand_4 V=4.75 (전부 >ε → advance),
  cand_5 회복.
- 그림: `docs/figures/q3_misprediction.pdf`.

### 한계 (정직 보고)
- BoundGuard는 **마지막 후보(cand_5)에서 제로 마진 회복** — cand_5가 정확히 N_cand=5 경계. vision 세트가
  커지면 cand_5가 예산 밖으로 밀려 회복 불가(candidate_ranking_check.md 민감도).
- 부하 생성에 FSRR_RATE_REPLICATE/FRAME_BUFFER 노브 사용(값 명시). 기본값이면 이전 결과 그대로 재현.

## Q4와의 구분
- 본 실험은 Q3: **일부 초기 후보 실패 후 대안 후보(cand_5)로 전환해 회복**. BoundGuard 최종 V=0.54<ε(회복).
- Q4는 **모든 후보가 검증 창 내 실패**해 회복 못 하나 bounded envelope 유지하는 worst case — 본 실험과 다름.
- (이전 판이 관측한 "vision-only 고부하 V=342, 대안 없음"이 바로 Q4-like. 본 mixed 실험은 대안(cand_5)이
  존재해 Q3.)

## Q5 (CPU–NPU) — co-load 게이트
- **feasibility 확인(측정)**: `build_vision_npu("yolo11s")`로 vision .mxq를 NPU 적재 후, 동시에
  `mobilint/Llama-3.2-1B-Instruct`(W8 .mxq)를 NPU 적재 → **둘 다 co-resident 성공**(aries0, qbruntime).
  DRAM 예산 수용, 단일-상주 제약이 이 조합을 막지 않음.
- 인프라 준비됨: `runtime/bg_entry.py`가 `--device npu` 지원(LLMEngine npu 경로), vision NPU 워커 존재,
  cpu_npu 예측기 아티팩트 존재.
- **게이트(사람 판단)**: Q5 전체 4기법 실험은 NPU 파이프라인 전용 λ 튜닝(NPU μ≈103fps)이 별도 필요.
  이전 "GPU-only" 결정 범위 밖 → 착수 승인 시 Q3와 동일 구조로 실행(vision NPU + llama1b NPU background,
  대안=llama1b→CPU, reload 스파이크 관찰).

## 무결 (§7)
- BoundGuard mode 1+validate 확인(D5, validate 4회). GPU 폴백 0(종료 후 895MiB idle). NPU co-load 정상.
- 네 기법 동일 시나리오·파라미터. 예측기 스키마·α·β·배선 불변. legacy/scripts/backup 무변경. main.tex 불변.
- 본 task는 코드 변경 없음(실험·그림·문서만; background 인프라는 이전 mixed_exec task).
