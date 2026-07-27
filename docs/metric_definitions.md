# 지표 정의 — 단일 구현 (C2/Q6 데이터 무결성 작업 0)

날짜: 2026-07-25 · 파라미터 불변(T=3s, T_v=3s, ε=1.0, Δ=0.2s, N_cand=5) · main.tex 불변
단일 구현: `scratchpad/metrics_lib.py`(전 변형·전 시나리오 공유). 아래 정의가 정본.

## 정의

| 기호 | 정의 | 구현 |
|---|---|---|
| **t_inject** | 과부하(burst) 주입 시각 = 고정 윈도우 기준점 | 첫 `combination_burst` 샘플 타임스탬프 |
| **t0** | V(t)가 처음 ε을 초과한 시각 (과부하 주입 시각과 구분) | 첫 `v_score>ε` 샘플의 타임스탬프 |
| **t_r** | V(t)가 ε 미만으로 내려가 **이후 유지**된 최초 시각 | `v_score≤ε`가 되고 **런 종료까지 계속 ≤ε**인 첫 시각. 미충족이면 t_r=없음(censored) |
| **persist** | t_r − t0 | t_r 없으면 **`censored(>런종료−t0)`** 로 기록 |
| **T_valid** | 첫 후보 적용 시작 ~ 마지막 후보 판정(또는 best-so-far 복귀) 완료 | 첫 `cand_*` 진입 → commit/revert 로그 시각. **persist와 다른 양** |
| **lastV** | 런 종료 직전 T=3s 창의 V | 마지막 v_score (이미 T-창 평균) |
| **hotswaps** | **실제 배치가 바뀐** 전환 횟수 | 로그 `device changed ... hot-swapping` 카운트 |
| **no_op_swaps** | 현재 배치와 동일한 배치로의 전환 **시도** 횟수 | 후보 진입 중 전 뷰가 `same device, keeping worker`인 전환 카운트 (신규) |
| **skipped_by_guard** | auto-advance 가드가 스킵한 후보 수 | 로그 `placement identical ... auto-advancing` 카운트 (신규) |
| **censored** | 런 종료 시점에 여전히 V>ε인가 | `lastV > ε` (t_r 없음) |
| **Σδ** | 실측 전환 지연 합 | 전환별 δ = 전환 시작 → 정상 서비스 재개(작업 2, 로그 타임스탬프) |

## 기존 리포트 값 ↔ 정의 매핑

| 리포트 값 | 대응 정의 | 비고 |
|---|---|---|
| `q3_q5_misprediction_report` persist | persist (t_r−t0) | 단, **런 길이 미고정**이라 censored 변형은 런 길이를 측정(작업 3에서 해소) |
| `repetition_report` Q3 `T_valid=24.6±0.5` | **T_valid** (첫 cand→commit/revert) | persist와 **다른 양** — 세 값 불일치의 주원인 |
| `c2_reactive_baseline_report` persist=18.8 (BoundGuard) | persist | 회복하는 변형은 t_r 존재 → persist 유효 |
| `c2_reactive_baseline_report` persist=63.0 (reinvoke) | **censored** | 미회복 → persist가 **런 길이 측정**(작업 3 대상). 거동 아님 |
| `main.tex` Q3 `20.0±1.5` | persist (반복 5회, 회복 변형) | `repetition_report` B2의 persist(20±1.5)와 대응 — T_valid(24.6)와 혼동 금지 |

## 핵심 — 세 값 불일치의 원인 (작업 4 선결론)
- `main.tex 20.0` = **persist**(회복까지, 5회 반복) · `c2 18.8` = **persist**(회복까지, C2 5회) · `repetition 24.6`
  = **T_valid**(탐색 완료까지, 다른 양). → 20.0과 18.8은 같은 양의 다른 배치(반복 vs C2), 24.6은 **다른 지표**.
- **persist(회복)와 T_valid(탐색)는 근본적으로 다른 양**이므로 한 표에 섞으면 안 된다. 작업 4에서 정본 확정.

## 고정 관측 윈도우 (작업 5)

**문제**: strict t_r("ε 이하로 내려가 **런 종료까지** 유지")는 **런 길이가 회복 판정을 바꾼다.** 런이 길면
뒤늦게 재상승한 런이 미회복으로 뒤집힌다. 초기 매트릭스는 `T_run`을 per-combo duration 합으로 실현해
**최종 combo가 착지 지점마다 달라** 런 길이가 변형별 32s까지 벌어졌다(Adaptive는 cand_1을 길게 유지,
walk 변형은 cand_5의 짧은 tail, no-dwell은 cand_4 revert). → strict t_r 판정이 오염.

**해결(2층)**:
1. **실행 측** `FSRR_HARD_WINDOW_S`: 첫 `cand_*` 진입(=고정 prefix 20s 뒤, t_inject로부터 고정 오프셋)에서
   `_end_time`을 걸고 **독립 QTimer**로 commit/revert 여부와 무관하게 윈도우 끝에 종료. → raw 런 길이
   편차 32s → **~3s**(1Hz 샘플링 지터만 잔존). `schedule_executor_main._fire_window_end`.
2. **분석 측** `FSRR_EVAL_WINDOW`: 분석 시 `[t_inject, t_inject+W]`로 계열 절단 → 보고 지표가 **정확히 동일
   윈도우**를 사용(편차 0s). strict t_r·persist·censored 전부 이 공통 윈도우 위에서 계산.

**수용 기준**(사전 고정): 모든 런 dur 편차 ≤ 2s(eval window로 0s 달성), 회복 변형도 윈도우 끝까지 실행·계측,
그림 종료 마커 정렬. **이 기준 통과 전 회복 매트릭스는 확정본 아님.**
