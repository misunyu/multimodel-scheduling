# 전면 재검증 캠페인 — commit-and-stay + P1~P9 프로토콜

날짜: 2026-07-23 · 모든 수정 적용(순서·post-swap V·추세·auto-advance·commit-and-stay) · main.tex 불변

## §1. commit-and-stay 구현
- `_commit_and_stay(combo)`: validate 통과 시 **잔여 후보 페이즈를 스킵**하고 committed 배치를 **잔여 시간
  동안 유지**(총 실행시간 보존 → 기법 간 time-matched). 로그: `committed 'cand_X'; skipping cand_Y..cand_tail
  (N phases); holding Ns`.
- **재트리거 의미(정정 반영)**: 커밋으로 recovery 에피소드 완결. 이후 재위반은 **새 에피소드**(새 감지창 T +
  현재 조건 기준 새 랭킹)이며 스킵 후보를 복원하지 않음. V(t) 모니터링은 CSV로 지속; 본 실험에선 committed
  배치가 feasible해 **재위반 미관측**(로그·CSV 확인). (harness는 페이즈 사전나열이라 새 에피소드 표현 불가 —
  harness 성격이지 알고리즘 한계 아님.)
- C1: `cand_tail = 마지막 후보 연장관찰`(옵션 b) → committed 배치가 흡수하는 것이 의도와 일치.
- C2: 커밋 후 전환 없음 → `bg.sync` 미호출 → background LLM 배치가 committed 배치로 유지(의도).
- 변경 국한: validate 경로(BoundGuard mode1+validate)만. `schedule_executor_main.py` 1파일.

## §2. 단계 1 — §4b 관문 (Q1.5 "no penalty" 최종 근거)
P1(순서 stable→burst→cand_1..) ✓ · P2(mode1+validate) ✓ · P3(fallback=0) ✓ · P9(스킵 로그) ✓

| 기법 | GPU persist/lastV | NPU persist/lastV |
|---|---|---|
| Static | 60s / 9.0 | 60s / 9.1 |
| Adaptive | 24s / 0.0 | 24s / 0.1 |
| **BoundGuard** | **24s / 0.07** | **24s / 0.08** |

- **cand_1(top-1 feasible) commit-and-stay**: `committed 'cand_1'; skipping cand_2..cand_tail (4 phases)`.
  V_postswap=0.11/0.38(정상) vs 오염 시 4.5(참고).
- **NPU 32s→24s 완전 수렴** — 이전 잔차(cand_3+cand_4 순회 스파이크)가 commit-and-stay로 제거됨.
- **판정: Q1.5 "no penalty in well-predicted case" 양 플랫폼 완전 복원. BoundGuard = Adaptive.**
  (개선분의 출처 = 순회 스파이크 제거, 명시.)

## §3. 단계 2 — B2 지표 (drop regime)
P1 ✓ · P3(fallback=0) ✓ · P6(B=754,λ=0.9μ 명시) ✓

| 기법 | GPU miss/p99 | NPU miss/p99 |
|---|---|---|
| Static | 1.000 / 24523ms | 1.000 / 24055ms |
| Stop-restart | 1.000 / 10730ms | 1.000 / 10664ms |
| Adaptive | 0.679 / 5547ms | 0.661 / 6134ms |
| **BoundGuard** | **0.665 / 5286ms** | **0.652 / 3775ms** |

- B2는 대형 버퍼(754)로 상속 backlog가 T_v 처리량 초과 → cand_1 post-swap V=6.17로 미커밋(commit-and-stay
  미발동). 단 cand_1=cand_2(동일 배치)라 최종 배치 동일 → 지표 변화 미미.
- **질적 서열 불변**: Adaptive≈BoundGuard(회복) >> Stop-restart≈Static(miss=1.0). 게이트 미해당.

## §4. 단계 3 — C3 fluid (Q1.4) 무영향 확인
- validate 트리거 미사용(단일콤보/burst-offload) → validate 경로 **미진입** → 구성상 무영향.
- **스팟체크**(slope_gpu_l0.90, mode0): burst backlog 누적 1947→3117, offload 배수 3509→1534.
  정성적 accumulate→drain 재현 ✓. **게이트 미해당.**

## §5. 단계 4 — Q3 / Q4
### Q3 (vision-3, λ=80) — **CLEAN, 교과서적 성립** ✓
P1 ✓ · P2 ✓ · P7(판정경로 로그) ✓
| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static | 3.77 | 3.77 | 106s | ✗ |
| Stop-restart | 3.72 | 3.72 | 106s | ✗ |
| **Adaptive** | 3.89 | **3.89** | 106s | **✗ (top-1 오예측 고착)** |
| **BoundGuard** | 3.67 | **0.87** | 66s | **✓** |

- **Adaptive는 예측기 top-1(ggggg=all-GPU 포함 LLM)에 커밋 → 오예측이라 실패 지속(lastV=3.89).**
- BoundGuard 판정경로: cand_1(=burst, auto-advance) → cand_2/3/4(LLM=GPU, V_postswap 4.1/7.5/14.8 → advance)
  → **cand_5(cggg=LLM→CPU)에서 회복**(lastV=0.87).
- **BoundGuard의 distinctive 우위 실증**: §4b(well-predicted)에선 Adaptive=BoundGuard, Q3(mispredicted)에선
  **BoundGuard가 Adaptive가 실패하는 곳에서 회복.** 억지 없이 자연 발생.

### Q4 (vision-4, λ=52) — **판정 불가 (경계, P8 위반)** ⚠️
| 기법 | maxV | lastV | persist |
|---|---|---|---|
| Static | 6.83 | 6.83 | 78s |
| Stop-restart | 0.42 | 0.41 | 0s |
| Adaptive | 0.86 | 0.58 | 0s |
| BoundGuard | 4.32 | 0.88 | 53s |

- **불일치**: Static(지속 ggggg)은 위반(6.83)하나 Adaptive/Stop-restart는 maxV<ε(위반 안 함).
- **원인**: λ=52는 vision-4의 **용량 경계(μ≈λ)** — ggggg 위반이 marginal이라 긴 지속(Static 78s)에서만 발현.
  Adaptive/Stop-restart의 burst→cand_1 전환(hot-swap/restart)이 backlog를 리셋해 marginal 위반을 가림.
- **P8(μ≈λ 경계 회피) 위반** → 결과가 알고리즘이 아니라 전환-리셋 아티팩트에 지배됨. **유효한 Q4 시연 아님.**
- vision-4는 앞선 분석(contention_alternative_check)대로 **깨끗한 Q4 레짐이 없음**(대안이 경계). 다른 워크로드
  필요 — 사람 판단.

## §6. 단계 5 — Q5 (CPU-NPU) — 미착수
- NPU background(llama1b .mxq) + 4기법. NPU DRAM 경험적. 앞 게이트(Q4) 대기 후 착수.

## 게이트 (§4)
- **§4.5 발동**: Q4(vision-4) 판정이 이전과 다르고 **P8 미충족(경계)** → 유효 Q4 아님.
- 그 외(§4b·B2·C3·Q3)는 기대와 일치 또는 개선(게이트 미해당).

## 논문 반영 대상 (예비)
| 항목 | 상태 | 근거 |
|---|---|---|
| Q1.5 "no penalty"(§4b) | **넣을 수 있음** | BoundGuard=Adaptive 양 플랫폼 |
| Q1.5 B2 표 | **넣을 수 있음**(수치 갱신) | Adaptive≈BoundGuard>>Static |
| Q1.4 C3 fluid | **유지** | 무영향 |
| Q3 misprediction(vision-3) | **넣을 수 있음** | Adaptive 실패, BoundGuard 회복 |
| Q4(vision-4) | **보류** | 경계, 유효 시연 아님 — 워크로드 재설계 필요 |
| Q5 | **미착수** | |

## 무결 (P1~P9)
- P1 순서 ✓(전 실행 로그 대조) · P2 mode ✓ · P3 fallback=0 ✓ · P7 판정경로/V_postswap 로그 ✓ · P9 commit-and-stay
  스킵 로그 ✓. P8은 vision-4에서 **위반(경계) → 무효 판정**. 코드변경 `schedule_executor_main.py` 1파일,
  legacy/scripts/backup·main.tex 불변.
