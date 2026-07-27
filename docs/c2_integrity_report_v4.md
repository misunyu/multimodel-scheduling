# C2/Q6 무결성 v4 — §IV 전 구간 wall-clock 재계산 · 상한 여유 검증

> ⚠️ **[v5 정정]** 이 리포트의 권고 3건이 [c2_integrity_report_v5.md](c2_integrity_report_v5.md)에서 **철회**됐다:
> (1) **작업 13의 "레이블 반전"은 오판** — b2_analyze 코드상 persist(샘플수)·T_valid(burst→적용)는 서로 다른
> 양이며 SD도 다름(0.5 vs 1.4). 레이블 교체 권고 철회. (2) persist 25.4를 bound 대상으로 본 것도 범위 오류 —
> bound 대상은 **탐색(t0→적용, ≤23s)**, persist는 탐색+배수. (3) 작업 14의 "t0 기준 21.0 병기"도 철회(가설 A:
> 고정 스케줄+stable 위반). **정본 수치·최종 목록은 v5 참조.** 아래 v4 본문은 이력으로 보존.

날짜: 2026-07-26 · 재실험 없음(로그 재판독+통계) · main.tex 불변 · 파라미터 불변
선행: [c2_integrity_report_v3.md](c2_integrity_report_v3.md)(작업 8~10), [metric_definitions.md](metric_definitions.md)

> **판정 요약** (작업 11~14):
> - **작업 11 (게이트) = 정합 (1행). 상한 위반 없음.** Q4 첫위반→reversion = **24s = bound 24s(tight)**, c2
>   5회 22–23s(<24). **샘플 간격 누락 = 0s**(Q4는 background LLM 없음 → 거친 전환 없음 → sample-gap 미발생).
>   작업 11-1의 우려(누락→28s 초과)는 **Q4에서 미발현.** §II bound 서술 **생존** → 작업 12·13 진행.
> - **작업 12/13**: 같은 런(b2rep)에서 persist·T_valid 재계산 시 **part≤whole 성립**·분해 정확. v3의
>   "부분>전체" 모순은 **하네스 혼용 + 레이블 스왑** 아티팩트였다. Q3 정본: **persist(t0→회복)=25.4±1.4,
>   validation(cand1→회복)=19.4±1.4, 분해 25.4=6.0+19.4.** main.tex 720의 "persist 20.0/T_valid 24.6"은
>   **레이블 반전**(그들의 T_valid 24.6 = 실제 persistence 25.4, 그들의 persist 20.0 = 실제 validation 19.4).
> - **작업 9-1 정정**: 정본 persist는 **하네스에 따라 두 값** — 원본 b2rep **25.4±1.4**(§IV 다른 수치와 동일
>   출처, **정본 채택**), C2 고정윈도우 23.6±2.3(교차 재현, Threats). v3의 23.6 단일 정본은 **철회**.
> - **작업 14**: Q5·§4b는 **stable 단계가 이미 위반**(Q5 stable V=6.9>ε) → t0가 t_inject보다 6.8s 앞섬.
>   main.tex Q5 14.2는 **burst 기준**(정확), t0 기준은 21.0. 본문이 이 사실을 안 밝힘 → 문장 초안 제시.

## 작업 11 — Q4 상한 여유 재검증 (게이트) — **정합, 위반 없음**

논문 Q4 BoundGuard 런 = `q4_bsf_out`(best-so-far revert, lastV 55; = q4_experiment_report §3의 revert 판).
`q4_bf_out`은 cand_5 종료판(lastV 736). 둘 다 판독.

| run | 첫위반→reversion | t0→첫후보(하네스+감지) | **샘플간격 누락** | lastV |
|---|---|---|---|---|
| **q4_bsf_out (논문)** | **24s** | 9s | **0s** | 55 |
| q4_bf_out (cand5종료) | 21s (→cand_5) | 10s | 0s | 736 |
| c2_final2 rep1–5 | 22,22,23,22,23 | 7–8s | **0s** (전부) | ~55–150 |

- **샘플 간격 누락 = 0s (전 런).** Q4는 background LLM이 없어(BG=False) **LLM 거친 전환(~2.9s, 샘플 멈춤)이
  없다.** vision hot-swap은 실측 0.1–0.4s(콜드 1.1s)로 sub-second라 1Hz 샘플링을 멈추지 않는다. → 작업 9-1이
  Q3에서 본 과소보고(LLM 전환 간격 누락)가 **Q4엔 구조적으로 없다.**
- **δ 인스턴스화(11-4)**: Q4 전환은 vision(실측 0.1–1.1s). bound `T+N_cand(T_v+δ)+δ`를 δ=1s(vision, 보수적)로
  세우면 `3+5×4+1=24s`. 실측 δ가 더 작으므로 이 bound는 관대(=논문과 동일). 초과 없음.

### 판정 (매트릭스 11-3) — **1행 (정합)**
- **전 런 구간 ≤ bound(24s).** 논문 런 24s = bound 24s(tight, main.tex와 일치), c2 5회 22–23s(<24, 여유 +1~2s).
  **초과 런 0개.** 749행 "tight" 서술 **유효**, 수치 정본으로 교체만.
- main.tex가 우려한 "샘플 누락→상한 초과"는 **발생하지 않음**(Q4 sample-gap 0). §II bound 재작성 불요.

## 작업 12 — §IV 전 wall-clock 구간 재계산

원본 로그 재판독(metrics_lib 정본, wall-clock strict). "차이"는 현행(보고/논문)−정본.

| 위치 | 현행 | 정본 재계산 | 차이 | 누락분 | 상한/여유 |
|---|---|---|---|---|---|
| Q1.3 회복 3기법 | 9–12s | SR 11–13·Ad 11–12·BG 11–12 (진짜 회복) | ~0 | 0 | <33s ✓ |
| Q1.3 Static | 59.4s | **censored**(lastV 9.0, ≥58s) | 표기 | — | v3 §8-4 |
| Q3 persist | 20.0±1.5 | **25.4±1.4**(t0→회복, b2rep) | **+5.4** | 전환간격 | bound~26 → 여유 +0.6 |
| Q3 T_valid(validation) | 24.6±0.5 | **19.4±1.4**(cand1→회복) | **−5.2** | — | (레이블 반전, 작업13) |
| Q4 첫위반→reversion | 24s | **24s**(논문런), 22–23(c2) | 0 | **0** | bound 24 → tight ✓ |
| Q5 수렴(burst기준) | 15.2±0.4 | **14.2±0.4**(burst→회복) | −1.0 | — | — |
| Q5 t0→회복 | — | 21.0±0.9(stable위반 6.8s 포함) | — | — | 작업14 |
| Q5 baseline | 114s | **censored**(lastV 33) | 표기 | — | v3 §8-4 |
| 그림 caption(541) | 총회복<33s | 최대 실측 25.4(Q3)·24(Q4) < 33 ✓ | — | — | 성립 |

### 12-2. 정합성 검사 (동일 런, b2rep)
| 실험 | part≤whole (T_valid≤persist) | 분해 (persist≈T_detect+T_valid) |
|---|---|---|
| **Q3** | ✅ 19.4 ≤ 25.4 | ✅ 25.4 = 6.0 + 19.4 (정확) |
| **Q5** | ✅ 2.4 ≤ 21.0 | ✅ 21.0 = 18.6 + 2.4 (정확; T_detect 큰 이유=stable위반, 작업14) |
- **v3의 "부분>전체"(persist 23.6 < T_valid 24.6) 모순 해소**: 원인은 (i) persist를 C2 고정윈도우(23.6),
  T_valid를 repetition(24.6)에서 각각 가져와 **하네스 혼용**, (ii) **레이블 반전**. 같은 런으로 같은 정의를
  쓰면 part≤whole·분해 모두 성립.

## 작업 13 — 정본 persist 출처 통일 — **(a) 원본 하네스 채택**

- v3는 정본을 C2 고정윈도우 23.6±2.3으로 뒀으나, **§IV의 T_valid·δ·bound 대조 수치가 전부 b2rep(repetition)
  하네스 출신**이다. persist만 다른 하네스면 분해가 깨진다. → **(a) 채택: Q3 persist = 25.4±1.4(b2rep, t0→회복),
  T_valid = 19.4±1.4(cand1→회복), 같은 런.** part≤whole·분해 정합.
- **레이블 반전 정정**: main.tex 720의 "persist 20.0"은 실제 **validation(cand1→회복)=19.4**, "T_valid 24.6"은
  실제 **persistence(t0→회복)=25.4**. 두 레이블을 서로 바꿔야 의미와 일치. bound~26과 비교할 값은
  **persistence 25.4**(여유 +0.6, tight).
- **13-3 두 하네스 차이(25.4 vs 23.6, 1.8s) 규명**: b2rep는 가변 런길이·워밍업 포함 원본, C2는 고정 100s 윈도우
  + 별도 배치. 차이는 (i) 윈도우/하네스, (ii) 배치(v3 작업10 σ²_b=0.53). **1σ 내 재현**이며 조작 아님.
  C2 23.6은 **교차 하네스 재현값**으로 Threats에 병기. 정본은 25.4(§IV 내부 정합 우선).

## 작업 14 — Q5·§4b의 t0 선행 (stable 단계 위반)

- **Q5 stable-phase V = 6.9(max 8.0) > ε=1** → 과부하(burst) 주입 **전부터** 위반. t0가 t_inject보다 **6.8s 앞섬**.
  §4b도 동일(v3 9-2, stable V~6, t0 −7s). Q3는 stable 청정(V=0, t0 +2s).
- **결과**: main.tex Q5 14.2·15.2는 **burst 기준**(정확). metrics_lib t0 기준은 21.0(stable 6.8s 포함). baseline
  censdur 113>dur 106도 동일 원인.
- **t0 기준점·관측 창 표** (v3 §8-4 공통 항목 구체화):

| 실험 | stable V | t0 위치 | 관측창(정본) |
|---|---|---|---|
| Q3 | 0.00 | burst+2s (정상) | b2rep ~107s |
| Q5 | 6.9 (>ε) | **burst−6.8s** (stable) | b2rep ~107s |
| §4b | ~6 (>ε) | **burst−7s** (stable) | 고정 100s |
| Q4 | ~0 | burst+1s (정상) | 고정 100s |

- **Q5 본문 보완 문장 초안(사람 적용)**: *"In Q5, the all-CPU stable phase is itself infeasible for the
  background generative workload (V≈7 > ε before the burst); the reported convergence time is measured from
  the burst injection, and the pre-burst violation is a property of the stable placement, not of the
  overload."* §4b 시나리오 설명에도 동형 문장 필요.

## main.tex §IV 최종 수정 목록 (v3 §8-4 + 이번 재계산 통합, 사람 적용)

| 위치 | 수정 |
|---|---|
| Q3 720 persist | 20.0 → **25.4±1.4**(persistence, t0→회복). 레이블을 현재의 "persist"와 반대로 |
| Q3 720 T_valid | 24.6 → **19.4±1.4**(validation, cand1→회복). bound 비교 대상은 persistence 25.4(≤26) |
| Q4 749 | 24s vs 24s tight **유지**(정본 확인, sample-gap 0). c2 재현 22–23s 병기 가능 |
| Q5 774 수렴/T_valid | 15.2/14.2 **burst 기준 유지**(정확) + stable 위반 각주(작업14) |
| Q1.3 630 회복3기법 | 9–12s **유지**(진짜 회복) |
| Q1.3 630 Static · Q3·Q5 baseline | **"회복 없음"**(censored, lastV>ε, ≥창길이) — v3 §8-4 |
| Q1.3 "5–6×" | **삭제**(런길이÷회복시간) — v3 §8-4 |
| 공통 | 각 실험 관측 창·t0 기준 명시(작업14 표); Q5/§4b stable 위반 문장 추가 |
| §II bound | **불변**(작업11 정합, 위반 없음) |

## 무결성
- 재실험 없음(작업 11~14 전부 기존 로그 재판독+통계). 알고리즘·파라미터·main.tex·legacy/backup 불변.
- 코드 변경 없음(v4). 분석: metrics_lib 정본 정의, 동일 런 분해.
