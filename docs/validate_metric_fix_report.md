# 검증 메트릭 오염 수정 (post-transition V) + §4b·B2 재실행

날짜: 2026-07-23 · validate 경로만 변경(T_v·ε·후보·α·β·예측기 불변) · main.tex 불변

## §1. 진단 — 오염원 = 누적 평균(윈도우 W = 전체 런)
- V(t)의 per-view latency `l_i = avg_infer_time + avg_wait_ms`이며, 둘 다 **누적 평균**(`total/count`,
  `view_handlers.py:141/199`) — 슬라이딩 윈도우가 아님.
- `reset_stats()`는 **run 시작 5s 워밍업 때만** 호출(`unified_viewer.py:1351`); **hot-swap 전환에선 리셋 안 됨**
  (adaptive는 `start_execution` 스킵).
- → 검증용 V의 오염 윈도우 W = **전체 런 이력(≫ T_v=3s)**. cand_1 검증 시 burst 수백 프레임의 high-wait가
  누적 평균에 포함 → feasible top-1이 V 오염으로 기각. (`_v_history_mode2` 슬라이딩(3 samples)은 전환 시
  리셋되나 그 입력 avg가 이미 오염.)
- **핵심 인식(지시문)**: 측정 윈도우보다 빨리 검증 불가. "전환 이후 샘플만 사용"이 필요조건.

## §2. 채택 방안 — 1순위(전환 이후 샘플만, T_v 유지), 비침습 스냅샷-델타
- 전환 시점(entered_at)에 각 뷰 핸들러의 누산기 `(total_infer, infer_count, total_wait, wait_count)`를
  **스냅샷**(`_handler_snapshot`). 검증 시점(T_v)에 **델타 평균** `Δtotal/Δcount`으로 **전환 이후 프레임만의
  V**를 계산(`_postswap_vscore`). 핸들러를 리셋하지 않음 → **보고용 CSV/그림 V(t)(누적)는 그대로**(연속성 유지),
  검증용 V만 분리.
- T_v=3s면 post-transition 샘플 충분(수십~수백 프레임) → **T_v 변경 불요**(2순위 불필요).
- 로그에 `V_postswap` / `V_cumulative` 병기(오염 gap 가시화).

## §3. 구현
- `schedule_executor_main.py`: `_handler_snapshot`, `_postswap_vscore` 추가; `check_validate`가 검증 판정에
  `V_postswap` 사용(commit 조건 `V_postswap≤ε` 또는 backlog 추세). 추세 판정·cand_1 auto-advance·duration cap 유지.
- **변경 국한 확인**: validate 경로(BoundGuard mode 1+validate)에만 영향. C3(트리거 없음)·Static·Adaptive·
  Stop-restart 무영향. 스냅샷 `backup/validate_metric_20260723_005756/`.

## §4. §4b 재실행 (Q1.5 주장의 운명) — 실행순서 P1 확인(stable→burst→cand_1..)
| 기법 | GPU persist/lastV | NPU persist/lastV |
|---|---|---|
| Static | 60s / 9.0 | 59s / 9.0 |
| Adaptive | 24s / 0.1 | 25s / 0.1 |
| **BoundGuard (수정)** | **24s / 0.7** | **32s / 0.9** |
| BoundGuard (오염, 감사) | 42s / 1.7 | 33s / 1.7 |

- **cand_1(top-1=all-GPU, feasible) 정상 commit**: `V_postswap=0.339` vs `V_cumulative=4.526`(오염 gap 확인).
  cand_3(yolo-CPU, infeasible)는 `V_postswap=7.988`로 정확히 기각.
- **판정: Q1.5 "no penalty in well-predicted case" 주장 복원.**
  - **GPU: BoundGuard 24s = Adaptive 24s (완전 대등).**
  - NPU: BoundGuard 32s vs Adaptive 25s (7s 잔차, lastV 0.9<ε 회복). 잔차 원인은 **검증 메트릭이 아니라**
    harness가 commit 후에도 duration cap으로 나머지 후보를 순회(각 전환에 소량 스파이크)하는 것 — validate
    수정과 별개의 harness 설계. (commit=조기-advance 방지일 뿐 "commit-and-stay"가 아님.)
- 오염(감사) 대비 BoundGuard가 미회복(1.7)→회복(0.7~0.9)으로 개선, Adaptive와 near-parity.

## §5. B2 영향 (Q1.5 지표)
기존 B2에서도 cand_1(feasible yolo-accelerator)이 오염 V=70.964로 기각됐음(순서 버그와 별개 축). 수정본 재실행
(fallback=0, 실행순서 burst→cand_1→cand_2 확인):

| | miss (수정 / 오염) | p99 (수정 / 오염) |
|---|---|---|
| gpu_BoundGuard | 0.685 / 0.725 | 5090 / 5203 ms |
| npu_BoundGuard | 0.649 / 0.662 | **3699 / 5596 ms** |
| gpu_Adaptive | 0.673 / 0.726 | 5651 / 5845 ms |

- **B2는 cand_1=cand_2(동일 배치)라 최종 배치 동일** → 지표 변화 **modest**(BoundGuard miss·p99 소폭 개선,
  특히 npu p99 5596→3699ms). 질적 서열 불변(모두 drop 레짐 고miss).
- **판정: B2 표/그림 수치는 소폭 갱신 가치 있으나 질적 결론 불변.** (버퍼 B=754가 커 상속 backlog가 T_v
  처리량을 초과 → post-swap V도 완전히는 못 씻김(cand_1 V_postswap=1.749). 그래도 같은 배치로 수렴해 지표엔
  경미.)

## §6. 무회귀
- **C3 fluid(Q1.4)**: validate 트리거 미사용(단일콤보/burst-offload) → validate 경로 미진입 → **무영향**(재실행 불요).
- **GPU 폴백 0**: §4b·B2 전 실행 `fallback=0` 확인.
- mode 1+validate(P2), 실행순서 P1 확인. legacy/scripts/backup 무변경, main.tex 불변.

## 종합 판정
- **검증 메트릭 오염 수정 성공.** post-swap V가 feasible top-1을 정상 commit(오염 gap 4.5→0.3 등 확인).
- **§4b: Q1.5 "no penalty" 주장 복원**(GPU 완전 대등, NPU 회복+소잔차).
- **B2: 지표 소폭 개선, 질적 불변** — 표/그림 수치 갱신은 선택(게이트: 사람 판단).
- **잔여 한계(정직 보고)**: (1) 매우 큰 버퍼(B2의 754)에선 상속 backlog가 T_v 처리량 초과 → post-swap V도
  완전히 못 씻김(단 동일-배치 수렴이라 지표 경미). (2) NPU §4b 7s 잔차는 harness의 "commit≠stay"(duration
  cap 순회) 때문 — 별도 설계 사항. 둘 다 validate 메트릭 수정 범위 밖.

## 논문 반영 메모(별도 대화)
설계 원칙: "검증 창은 QoS 메트릭의 평균 윈도우보다 짧을 수 없다 — 그렇지 않으면 전환 이전 데이터로 새 배치를
판정해 feasible 후보를 기각한다. 본 구현은 전환 이후 프레임만으로 검증 V를 계산해 이를 회피한다."
