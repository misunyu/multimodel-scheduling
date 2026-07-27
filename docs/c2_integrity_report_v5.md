# C2/Q6 무결성 v5 — 레이블 반전 코드검증 · Q3 런별 상한 · Q5 T_detect 규명 · 목록 확정

날짜: 2026-07-26 · 재실험 없음(로그·코드 판독) · main.tex 불변 · 파라미터 불변
선행: [c2_integrity_report_v4.md](c2_integrity_report_v4.md), [metric_definitions.md](metric_definitions.md)

> **판정 요약** (작업 15~18) — 이 라운드는 **v3/v4 권고 4건을 스스로 정정**한다:
> - **작업 15 (게이트) = 반전 아님. v4의 "레이블 교체" 권고 철회.** 720행 두 값은 코드상 **서로 다른 양**:
>   persist 20.0 = `V>ε 샘플 개수`(작업 9-1 과소집계), T_valid 24.6 = `burst→회복후보 적용`(결정적, SD 0.5).
>   내 25.4(t0→V≤ε)와는 SD가 달라(0.5 vs 1.4) 같은 양이 아니다.
> - **작업 16 = 상한 위반 없음.** persist(25.4)를 탐색-상한과 비교한 게 범위 오류였다. **탐색(t0→회복적용)
>   = max 23s < 상한 24–26s, 전 35런 이내.** 배수(적용→V≤ε, 4s)는 상한 밖 별개 항. b2rep의 외견상 초과는
>   vision δ 미로깅(δ 인스턴스화 불완전)일 뿐.
> - **작업 17 = 가설 A 확정(STOP 아님).** Q5 T_detect 18.6s는 감지 지연이 아니라 **고정 스케줄(stable+burst)
>   + stable 자체 위반**. 감지는 정상(cand_2 3.1s validate). → **t0 기준 수치 사용 금지**(v4 "21.0 병기" 철회).
> - **작업 18 = v3/v4의 "Q1.3 5–6× 삭제" 권고 철회.** 6배는 **누적위반 ∫V**(531 vs 85)이고 Static 창이 더
>   짧아 보수적. 삭제 대상은 persistence 59.4 하나뿐.

## 작업 15 — 레이블 반전 주장의 코드 검증 (게이트) — **반전 아님**

### 15-1. 출처 코드
main.tex 720의 `persist 20.0±1.5`·`T_valid 24.6±0.5` 산출 = **`scratchpad/b2_analyze.py`**(repetition_report 생성 경로,
`b2_rep.sh`가 호출). 계산식:
- **`persist = sum(1 for x in v if x>eps)`** → **V>ε인 CSV 샘플의 개수**(1Hz라 ≈초, 단 **전환 중 샘플 멈춤은
  미계수** = 작업 9-1 과소집계). 샘플 기반.
- **`Tvalid = rec_apply − burst`** → **burst 시작 → 회복후보 적용** 시각. 로그/CSV 타임스탬프 기반(wall-clock).

### 15-2. 런별 5쌍 대조 (b2rep_q3, 같은 런)
| rep | [orig]persist=샘플수 | [orig]T_valid=burst→적용 | [mine]persist=t0→V≤ε | [mine]T_valid=cand1→V≤ε |
|---|---|---|---|---|
| 1 | 17 | 25.0 | 23.0 | 17.0 |
| 2 | 20 | 24.0 | 25.0 | 19.0 |
| 3 | 21 | 25.0 | 26.0 | 20.0 |
| 4 | 21 | 24.0 | 26.0 | 20.0 |
| 5 | 21 | 25.0 | 27.0 | 21.0 |
| **평균±SD** | **20.0±1.55** | **24.6±0.49** | 25.4±1.36 | 19.4±1.36 |

### 15-3. 판정 — **3행 (정의 불일치), 반전 아님**
- [orig]T_valid(24.6±0.49)은 `burst→회복후보 적용`이지 내 persistence(25.4, t0→V≤ε)가 **아니다**.
- [orig]persist(20.0)은 `V>ε 샘플 개수`이지 내 T_valid(19.4, cand1→V≤ε)가 **아니다**(수치만 우연히 근접).
- 넷 다 다른 양. **v4의 "720행 레이블을 맞바꾸라" 권고는 오류이므로 철회.** 레이블을 바꾸면 없던 오류를 넣게 됨.

### 15-4. SD 불일치 설명 (의무)
- [orig]T_valid **24.6±0.49**(작은 SD): `burst→회복후보 적용`은 **스케줄 결정적** — BoundGuard가 매번 cand_5에서
  회복하고 burst-hold·T_v·후보수가 고정이라 적용 시각이 거의 상수.
- [mine]persist **25.4±1.36**(큰 SD): `t0→V≤ε`은 스왑 후 **큐 배수 동역학**에 좌우되어 변동.
- → **0.5 vs 1.4는 두 양이 다르다는 직접 증거.** 지시문의 "경쟁 설명"(24.6이 진짜 T_valid)이 옳다. 반전 판정 폐기.

### 15-5. 남는 실체 — persist 20.0의 성격
반전은 아니나 **persist 20.0은 샘플 개수라 wall-clock 위반 지속을 과소집계**(전환 중 멈춘 초 누락). 실제 위반
지속(회복까지 wall-clock)은 더 길다. 표기 시 "V>ε 샘플 수(하한)"임을 밝히거나 wall-clock 값을 별도 제시.

## 작업 16 — Q3 상한을 런 단위로 — **전 런 이내, 위반 없음**

**범위 오류 정정**: 상한 `T+N_cand(T_v+δ)+δ`은 **탐색(회복후보 적용까지)**을 경계한다. persist(t0→V≤ε)는
**탐색 + 배수(drain)**라 상한과 직접 비교 불가. 올바른 비교 = **탐색(t0→회복적용)**.

- 전 Q3 런 **회복 후보 = cand_5 (k=5 일정)** → N_cand=5 정확.
- **탐색(t0→적용): mean 19.9, max 23s, SD 1.9 (b2rep 5 + C2 30 = 35런).** **상한 24s(δ=1)·26s(paper) 초과 0건.**
- **배수(적용→V≤ε): mean 4.0, max 5s** — 상한 밖 별개. persist(25.4)=탐색(19.9)+배수(4)+계수차.
- **런별 실측-δ 상한**(`T+5·T_v+Σδ`): C2 30런(vision δ 완전 로깅, Σδ 3.5–5.7) **거의 전부 이내**. 초과 외견은
  **b2rep가 vision δ=ms를 로깅 안 해 Σδ가 LLM(2.9s)만 잡힌 under-count**(δ 인스턴스화 불완전, 작업 11-4).
  유효 상한(δ_max=LLM 2.9로 세우면 35.4s)엔 전 런 큰 여유.
  > ⚠️ [v6 정정] 여기 "b2rep **7런**"은 오기다. 실측-δ 상한 초과는 **7/35 = b2rep 5 + C2 2**(0.1·0.5s marginal).
  > **paper 상한(24–26s) 대비 초과는 0/35**(search max 23). → [c2_integrity_report_v6.md](c2_integrity_report_v6.md) 작업 21-2.

### 판정 (16-3) — **1행 (전 런 이내)**
- 720행은 **탐색(19.9±1.9, max 23)**을 상한과 비교해야 한다. "평균 25.4 vs 26" 표현은 **범위 오류이므로 삭제**.
- v3의 "관측 최대 28s > 상한 26" 우려는 **28s가 persist(탐색+배수)라 상한 대상이 아님** — 해소.

### 16-4. 상한 표기 (런마다 다름)
논문이 상한을 단일 `≈26s`로 쓴다면, 실제는 도달 후보 k·전환 종류에 따라 런마다 다르다. 제안:
*"search completes within T+k(T_v+δ) (k≤N_cand); across runs, observed search ≤ 23 s against the worst-case
bound of 24–26 s (δ instantiated per transition; the recovery transition in Q3 is the LLM CPU-load, δ≈2.9 s)."*

## 작업 17 — Q5 T_detect = 18.6s 규명 — **가설 A**

- **stable V 궤적(Q5 rep1)**: idx0 3.65 → idx3–6 ~7.4 (지속). **처음부터 계속 >ε** → 워밍업 아티팩트 아님
  (**가설 C 기각**). all-CPU stable이 Q5 background 생성 워크로드에 infeasible.
- **감지 정상 작동**(**가설 B 기각**): cand_2에서 `QoS-validate elapsed=3.1s V=0.077 → committing`(로그 line 98).
  T=3s/T_v=3s 감지가 설계대로 3.1s에 판정. 18.6s 감지 지연 아님.
- **가설 A 확정**: t0(stable 시작)→첫 적응(cand_2)의 ~18s는 **고정 실험 스케줄(stable 7s + burst 8s) +
  stable 위반을 그 온셋부터 계수**한 것. 적응 컨트롤러는 고정 stable/burst 구간엔 개입하지 않음. §4b 동형.

### 조치 (17-3 A)
- **t0 기준 수치(21.0)를 어디에도 쓰지 않음 — Threats 병기 포함. v4 작업14의 "21.0 병기" 철회.**
- burst 기준 **14.2/15.2만** 사용(정확).
- **실험 설정 명시 문장(사람 적용)**: *"In Q5/§4b the all-CPU stable phase is itself infeasible for the
  workload (V≈7>ε before the burst). Recovery times are measured from the burst injection; the adaptive
  controller acts only in the candidate phase of the fixed schedule, so pre-burst violation reflects the
  stable placement, not detection latency."*

## 작업 18 — 잘못된 권고 철회 (목록 재검토)

- **철회: "Q1.3 '5–6×' 삭제".** 630행 6배는 **누적위반 ∫V**(Static 531/530 vs 회복 3기법 76–91 = ~6.2×)이며
  persistence 비율이 아니다. 게다가 **Static 관측 창(58.4s) < 회복 3기법(63–68s)** — 짧은 창의 누적이라 비교는
  **보수적**(창을 맞추면 격차 확대). → 유지하되 *"despite Static's shorter observation window"* 추가 권고.
- **유일한 Q1.3 삭제 대상 = persistence `59.4s`**(censored, lastV 9.0). 이건 v3 §8-4대로 "회복 없음"으로.
- 나머지 목록 항목 재점검: censored 재표기(Q3/Q5/Q1.3 baseline persist)는 v3 작업8로 유효; Q4 tight는 작업11로
  유효; Q5 stable 각주는 작업17로 유효.

## main.tex §IV 수정 목록 — **확정본** (v3 §8-4 + v4 + v5 정정 통합)

| 위치 | 확정 조치 | 근거 |
|---|---|---|
| Q3 720 persist 20.0 | **레이블 유지**. "V>ε 샘플 수(하한)"임을 명시하거나 wall-clock 지속 별도 제시 | 작업15(반전 아님)·15-5 |
| Q3 720 T_valid 24.6 | **유지**(=burst→회복적용, 결정적). bound 비교 대상은 이 값(탐색), persist 아님 | 작업15·16 |
| Q3 720 "persist vs 상한" | 상한 비교는 **탐색(≤23s) vs 24–26s**로. "25.4 vs 26" 범위오류 표현 삭제 | 작업16 |
| Q3 720 상한 표기 | 단일 26s → **런별(k·δ 의존), 관측 탐색 ≤23s vs worst-case 24–26s** | 작업16-4 |
| Q4 749 | **24s = bound 24s (tight) 유지** (sample-gap 0, 정본 확인) | v4 작업11 |
| Q5 774 수렴/T_valid | **burst 기준 14.2/15.2 유지** + stable 위반 각주. **t0 기준 21.0 미사용** | 작업17 |
| Q5·§4b | stable 자체 infeasible(V≈7>ε) 문장 추가(초안 작업17) | 작업14·17 |
| Q1.3 630 "5–6×" | **유지**(누적위반 ∫V) + "Static 창 더 짧음에도" 추가. **삭제 권고 철회** | 작업18 |
| Q1.3 630 Static 59.4 · Q3/Q5 baseline persist | **"회복 없음"(censored, lastV>ε, ≥창길이)** | v3 작업8 |
| 공통 | 각 실험 관측 창·t0 기준 표(작업14) | v3·v4·v5 |
| §II bound | **불변**(작업11·16 위반 없음) | — |

## 무결성
- 재실험 없음(작업 15~18 전부 로그·코드 판독+통계). 알고리즘·파라미터·main.tex·legacy/backup 불변. 코드 변경 없음.
- **v3/v4 권고 4건 자기정정**: 레이블 교체·persist 25.4를 bound대상으로·t0 기준 병기·Q1.3 5–6× 삭제 — 모두 철회.
