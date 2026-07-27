# C2/Q6 데이터 무결성 — 지표 통일·대조·∑δ·정본화

날짜: 2026-07-25 · 스냅샷 `backup/schedule_executor_main.pre_integrity.py` · main.tex 불변
관련: [metric_definitions.md](metric_definitions.md), [c2_reactive_baseline_report.md](c2_reactive_baseline_report.md)

> **판정 요약** (작업 0~4):
> - **작업 0**: 지표를 단일 구현으로 통일. **persist(회복까지)와 T_valid(탐색까지)는 다른 양** — 세 곳
>   수치 불일치의 주원인.
> - **작업 1 게이트**: Q3에서 **B(re-invoke) ≡ Adaptive**(둘 다 top-1 고착·hotswaps 0·미회복=censored).
>   B의 "persist 63<baseline 105"는 **censored 런길이 아티팩트**. → **판정 매트릭스 2행**(런길이 차),
>   3행(의미 차 → main.tex 재작성) **미해당** → 작업 2~4 진행.
> - **추가 발견(무결성)**: C2 리포트의 **§4b 대조군 config 오류** — burst=all-accel(위반 없음)이라 "A
>   대조군 실패" 서술이 무효. burst=all-CPU로 수정해 재실행.
> - **작업 2**: ∑δ **실측**(공칭값 아님) — vision in-place ~50–180ms, background LLM 거친 전환 ~2.9s.
> - **작업 3**: 런 길이 T_run=120s 고정 → persist 절단 해소, censored 플래그 도입.
> - **작업 4**: main.tex 20.0=persist(반복), c2 18.8=persist(C2), repetition 24.6=**T_valid**(다른 양).

## 작업 0 — 지표 정의 통일
→ [metric_definitions.md](metric_definitions.md). 단일 구현 `scratchpad/metrics_lib.py`. t_r은 **런 종료까지
지속 ≤ε**(strict)으로 정의해 recovered와 persist를 정합. 미회복은 `censored(=런종료−t0)`로 기록.

## 작업 1 — B ↔ Adaptive 대조 (게이트) — **2행: 런길이 아티팩트**

Q3(고정 120s, 순차 5회):
| 변형 | 회복 n/5 | persist(회복,±SD) | cens_persist(런길이,±SD) | hotswaps | no_op |
|---|---|---|---|---|---|
| BoundGuard | 5/5 | 25.2 ± 1.0 | — | 7.0 | 1.0 |
| **Adaptive** | 0/5 | — | 120.6 ± 1.5 | **0.0** | 1.0 |
| **reinvoke (B)** | 1/5 | (82, 1회) | 89.0 ± 0.7 | **0.0** | 5.0 |

- **B와 Adaptive는 동작 동등**: 둘 다 top-1(=aaaa=위반 배치)에 고착, **실제 배치 변경 0회(hotswaps 0)**,
  거의 회복 못 함(0/5·1/5). 차이는 no_op 재적용 횟수(1 vs 5)와 **런길이**뿐.
- B의 이전 "persist 63"은 **미회복(censored) 변형의 런길이**였다 — 회복시간 아님. Adaptive도 censored
  (런길이 120.6). 두 censored 값 비교는 무의미.
- **판정 매트릭스 2행** — 런길이 차. B는 "무행동"이 아니라 Adaptive처럼 top-1 재적용(no-op)이므로 **3행
  (의미 차 → main.tex 재작성) 미해당.** → 작업 2~4 진행.

## 추가 발견 — §4b 대조군 config 오류 (무결성)
- C2 리포트(`c2_scenario`)의 §4b는 **burst=all-accelerator**로, all-GPU vision@λ=45가 feasible → **V=0,
  위반 없음.** "A가 대조군에서 실패(1/5)"는 위반이 없어 v-above가 안 걸리고 **duration 만료로 후보를
  맹목 순회하다 나쁜 후보에 착지**한 아티팩트다. **해당 §4b 수치 무효.**
- 원래 §4b(`bg_scenario`)는 **burst=all-CPU(위반)**였다. 수정: `c2_fixed_scenario`의 §4b burst=all-CPU →
  검증(V 6.7→9.4 위반, top-1=GPU offload가 cand_1 V 15.5→0 회복). 전 변형 재실행(작업 3와 통합).

## 작업 2 — ∑δ 실측 (C4 마무리)

**공칭값 미사용 — 로그 타임스탬프 실측**(`adaptive_deploy` 전환 시작→완료). 전환 종류별 분해:
- **vision in-place hot-swap δ**: ~43–180ms (워커 큐 스왑), cold-start 동반 시 ~1000ms.
- **background LLM 거친 전환 δ**: ~**2.9s** (llama1b CPU 워커 신규 적재). 논문의 vision≈1s vs LLM≈2.9s 구분을
  **실측 확인**.

Q3 변형별 Σδ(5회 평균, ms):
| 변형 | Σδ_vision | Σδ_bg(적재) | 해석 |
|---|---|---|---|
| BoundGuard | 1396 | 5690 | vision 4회 swap + LLM 초기적재+회복전환 |
| **A. no-dwell** | **3367** | 5713 | **vision swap 2.4배**(thrashing churn) |
| B. reinvoke | 0 | 2957 | swap 없음(top-1 고착), LLM 초기적재만 |
| C. hybrid | 0 | 2859 | 동일 |
| Adaptive | 0 | 2914 | 동일 |

- **A2의 세 번째 실패 모드("재구성 비용 미계상")를 수치화**: A는 vision 전환에 BoundGuard의 **2.4배** 시간을
  낭비(무익한 thrashing). B/C/Adaptive는 top-1 고착이라 vision swap 0(대신 회복 시도 자체를 안 함).
- Σδ_bg가 BoundGuard/A만 높음(~5.7s = 초기적재 2.9s + 회복 LLM전환 2.9s) → **이 둘만 LLM을 CPU로 전환
  (회복 시도)**, B/C/Adaptive는 초기적재(2.9s)만 = LLM 전환 없음(top-1 고착 확증).

## 작업 3 — 런 길이 고정(T_run=120s) + 정정 매트릭스 재실행

`c2_fixed_scenario`: T_run=120s 고정(과부하 후), §4b burst=all-CPU 수정, Adaptive 변형 추가. 전 변형이
정책 종료 후에도 120s까지 실행·계측. `censored` 플래그 기록. **5변형 × 3시나리오 × 5회.**

### 회복 성공률 (정정본)
| 변형 | §4b (예측정합) | Q3 (오예측) | Q4 (후보없음) |
|---|---|---|---|
| **BoundGuard** | 5/5 | 5/5 | 0/5 유계 |
| **Adaptive**(baseline) | 5/5 | **0/5** | 0/5 유계 |
| **A. no-dwell** | **1/5** (hs 13 폭주) | **1/5** | 0/5 유계(최악착지 lastV 134) |
| **B. re-invoke** | 5/5 | **0/5** | 0/5 유계 |
| **C. hybrid** | 5/5 | **1/5** | 0/5 유계 |

- **§4b config 수정 후에도 질적 결론 유지**: B/C는 예측 정합 시 회복(5/5)·오예측 시 실패(0–1/5),
  **A는 양쪽 실패**(dwell 보편 필수). 이전 broken-config는 "맞는 결론(A 실패)을 틀린 이유(위반 없음)"로 냈고,
  정정본은 **맞는 이유(feasible offload를 transient 중 이탈, hotswaps 13 폭주)**로 같은 결론.
- **persist(회복 변형, 정정본)**: BoundGuard §4b 18.6±0.5·Q3 **21.2±3.2**. Adaptive §4b 18.8±0.4.
  cens_persist(미회복)는 런길이라 비교 무의미(작업 1).
- 그림 `c2_reactive_comparison.pdf`는 정정본(Q3)으로 재생성.

## 작업 4 — Q3 수치 정본화
| 출처 | 값 | 정의 | 판정 |
|---|---|---|---|
| `main.tex` Q3 | 20.0±1.5s | **persist**(반복 5회, 회복까지) | `repetition_report` B2와 대응 |
| `repetition_report` | 24.6±0.5s | **T_valid**(탐색 완료까지) | **다른 양** — persist와 혼동 |
| `c2_reactive_baseline_report` | 18.8±0.4s | **persist**(C2 5회) | 회복 변형 유효 |
| 정정 매트릭스(고정 120s) | **21.2±3.2s** | persist | 정본 후보 |

- **20.0(persist)과 24.6(T_valid)은 근본적으로 다른 양** — 한 표에 섞으면 안 됨. persist(회복)를 Q3 headline으로,
  T_valid(탐색)는 bound 분해(별도 표)로 분리 권장.
- **persist 값들은 정의를 맞추면 정합**: main.tex 20.0 · c2 18.8 · 정정본 21.2 — 전부 persist(회복까지),
  서로 SD 범위 내(반복·C2·정정본의 배치 차이). **24.6만 다른 양(T_valid)**이라 튀어 보였던 것.
- **정본 후보: 정정 매트릭스 Q3 persist 21.2±3.2s**(고정 길이·단일 지표·5회). main.tex/c2 값은 이와 정합.

> ⚠️ **[2026-07-25 갱신]** 이 리포트의 "고정 길이"는 부분 수정 중간본이었다. 후속 무결성 감사에서 런 길이가
> 실제로는 변형별 32s 벌어져 있었음이 드러났고(strict t_r 오염), 완전 고정 윈도우로 재측정한 **정본은
> [c2_integrity_report_v2.md](c2_integrity_report_v2.md)**: Q3 persist **23.6±2.3s(관측 최대 28)**, 배치 효과 없음
> (F 0.64). 위 21.2±3.2는 중간본으로 STALE. §4b·hybrid Q3 수치도 v2가 정본.

## 판정 — Q6 표를 확정본으로 볼 수 있는가

**조건부 예 — 아래 3건을 반영하면 확정본.** 질적 결론(2×2, dwell/progress 필요성)은 정정 재실행으로
**모두 재확인**됐고, 무결성 결함은 수치·서술 차원이다:

1. **B(re-invoke) = Adaptive baseline으로 명기**: 결정적 예측기라 재호출=top-1 적용=Adaptive. B는 2×2의
   "둘 다 제거" 모서리이자 곧 baseline이다(격자 일관, 결함 아님). B의 persist(63)는 **censored 런길이
   아티팩트**였으므로 삭제하고 "미회복(=baseline)"으로 보고. → **`c2_reactive_baseline_report` 갱신 완료.**
2. **§4b 수치는 정정 매트릭스 값으로 교체**(broken-config 무효). A 1/5(hs 13), 나머지 5/5.
3. **persist(회복) vs T_valid(탐색) 분리**: Q6/Q3 표에 섞지 말 것. persist=21.2±3.2(정본), T_valid는 별도.

**main.tex 재작성(판정 매트릭스 3행)은 불필요** — B≡Adaptive는 의미 차가 아니라 격자의 예상된 모서리이고,
질적 주장은 전부 성립한다. 위 3건은 리포트/표의 **수치·라벨 정정**이며 서술 골격은 유지된다.

## 무결성
- 계측·정의만 변경. 알고리즘·파라미터 불변. 코드: `adaptive_deploy.py`(δ 타임스탬프), `view_handlers.py`(카운터,
  앞선 작업). 스냅샷 보관. main.tex 불변.
