# D2 — Trace-driven fluid simulation of bounded persistence

날짜: 2026-07-26 · 신규 시뮬레이션(하드웨어 재실험 없음) · main.tex 불변 · 파라미터(알고리즘) 불변
> **이 문서의 §1(파라미터 출처)과 §3(통과 기준)은 시뮬레이터를 실행하기 전에 확정해 기록한다.**
> 스윕 곡선은 §4 게이트를 통과한 뒤에만 §5에 싣는다.

## 목적
하드웨어 예산 밖 파라미터 공간에서 bounded persistence가 어디까지 성립하고 어디서 깨지는지 — 세 질문:
1. **dwell 용량반응**: T_v를 0→키우면 회복신뢰도·churn·persistence가 어떻게 변하나 (변형 A=T_v 0은 곡선 한 끝).
2. **예산×순위**: 회복 배치가 rank k*에 있을 때 예산 N_cand 요구량. (N_cand,k*) 평면의 회복 가능 영역·persistence 등고선.
3. **상한 긴밀도**: λ·k*에 따라 실측 탐색이 T+k(T_v+δ)에 얼마나 붙나.

## §1. 파라미터 — 전부 측정에서 고정 (자유 파라미터 0)

| 파라미터 | 값 | 출처 |
|---|---|---|
| T (감지창) | 3.0 s | 논문 설정 |
| T_v (검증창) | 3.0 s | 논문 설정 |
| ε | 1.0 | 논문 설정 |
| Δ (샘플링) | 0.2 s | 논문 설정 |
| δ_vision (in-place 전환) | 1.0 s | 실측(hot-swap; 논문 §IV Q3) |
| δ_llm (background 이전) | 2.9 s | 실측(b2rep_q3/q5 로그 `loaded (2837/2909 ms)`) |
| T_detect (t0→첫 후보 적용) | 측정 구조에서 (아래) | b2rep 로그 후보 진입시각 |
| λ, μ_i, μ* | 시나리오별 실측 서비스율 | q4_bsf_out·b2rep 로그, C3 파이프라인율 |
| k* (회복 후보 순위) | Q3=5, Q5=2, Q4=없음 | 실측 |

**측정된 구조적 타이밍**(t0 기준, b2rep rep3):
- Q3: 후보 진입 cand_1=6, cand_2=7, cand_3=10, cand_4=14, cand_5=23 s. LLM 이전 2.84/2.91 s.
- Q5: cand_1=14, cand_2=20 s(회복). t0가 burst−6(stable 위반).
- Q4: cand_2=9, cand_3=13, cand_4=17, cand_5=20 s, 이후 revert.

μ_i(후보 서비스율, Q4 실측): cand_2..5 = 237.3/202.9/215.8/186.2 fps(전부 미회복). Q3 회복 cand_5, Q5 회복 cand_2.
**이 표 밖의 수는 도입하지 않는다.** C3에서 확인된 파이프라인 서비스율(GPU μ*≈137, NPU≈103 fps)을 μ* 스케일로 쓴다.

## §2. 모형 (유체 수준, C3에서 검증된 백로그 동역학 위에)
- 큐: 축적 기울기 λ−μ_i, 배수 기울기 −(μ*−λ). 파이프라인 서비스율(raw inference rate 아님, C3).
- 후보: N개, 예측기 랭킹 순. 회복 배치(μ_i>λ)는 rank k*. 나머지 μ_i<λ, 값 상이(A의 "더 나쁜 배치 배회" 재현).
- 검출: 창 T의 V>ε. 검증: T_v 관찰 후 판정(T_v=0이면 즉시=변형 A). 전환: δ(종류별). 복귀: 예산 소진 시 관측 최선(μ 기준).
- **난수원(v13 공개)**: 도착/서비스는 **결정론적**이나 **판정은 확률적** — dwell T_v가 post-swap 정착시간
  τ_settle보다 짧으면 판정이 transient를 읽어 Bernoulli(f=min(1,T_v/τ)), 오판 시 동전. **이 판정노이즈가 회복률
  곡선의 유일한 난수원**(실패후보 μ 값은 feasibility boolean에만 쓰여 회복률에 무관, [d2_followup_report.md](d2_followup_report.md) 46).
- **모형이 담지 않는 것**: 간섭 미시구조, 스케줄러 상세, 확률적 도착/서비스 변동, **변형 A의 배회·고착
  메커니즘(k* 의존성)**. 결과는 **경향·임계점**이지 절대수치 예측 아님.

## §3. 통과 기준 (스윕 전 사전 고정 — 이 절은 시뮬레이터 실행 전에 확정)

재현 대상(값은 `fig/confirmed_values.json`에서 읽음):

| 시나리오 | 측정 search | 분류 | 회복 후보 |
|---|---|---|---|
| Q3 (k*=5) | 22.6 s | 회복 | cand_5 |
| Q5 (k*=2) | 14.2 s(burst 기준) | 회복 | cand_2 |
| Q4 (없음) | 24 s(첫위반→복귀) | 유계·미회복 | — |

**통과 기준 (사전 고정, 미조정)**:
1. **분류 일치**: 세 시나리오 회복/미회복 판정 전부 일치.
2. **탐색 시간**: sim search가 측정값의 **±20%** 이내(세 지점 모두).
3. **회복 후보**: 회복 후보 번호 일치(Q3 cand_5, Q5 cand_2).
4. **순서 보존**: Q3 search > Q5 search.

**판정 매트릭스**: 전부 충족→통과(작업 44); 일부 미달·원인이 모형 누락으로 특정→누락 추가(결과에 맞춘 파라미터
조정 금지)·재검증; 미달·원인 미상→**정지·보고**(스윕 안 함).

## §4. 재현 결과 (게이트) — **통과**

`fig/sim_d2.py`, 파라미터 §1(측정)에서만:

| 시나리오 | sim search | 측정 search | 비율 | 분류(sim/측정) | 회복후보 |
|---|---|---|---|---|---|
| Q3 (k*=5) | 20.9 s | 22.6 s | 0.93 | 회복/회복 ✓ | cand_5 ✓ |
| Q5 (k*=2) | 13.9 s | 14.2 s | 0.98 | 회복/회복 ✓ | cand_2 ✓ |
| Q4 (없음) | 22.0 s | 24 s | 0.92 | 미회복/미회복 ✓ | — ✓ |

- **기준 1 분류 일치** ✓ · **기준 2 ±20% 이내** ✓(최대 편차 8.3%) · **기준 3 회복후보 일치** ✓ · **기준 4 순서** Q3 20.9 > Q5 13.9 ✓.
- **정직 보고**: sim이 세 지점 모두 ~8% **과소추정**(체계적, 방향 일관). 잔여 ~2s는 harness settle/routing으로 추정하나
  모형에 넣지 않았다(±20% 이내라 게이트 목적상 불필요; 결과에 맞춘 조정 회피). t_detect는 측정된 t0→cand_1
  진입시각(Q3=6, Q5 burst→cand_1=8, Q4=5)에서 왔고 **search 출력에 맞춰 조정하지 않았다**.
- **판정: 통과 → 작업 44 스윕 진행.**

## §5. 스윕 (게이트 통과 후) — 각 설정 25 seed, 분포 보고

`fig/sim_d2_sweep.py`. 판정 노이즈는 **측정된 메커니즘**: dwell T_v가 post-swap 정착시간 τ_settle보다 짧으면
판정이 steady-state가 아니라 transient를 읽어 불신뢰(변형 A의 실패 원인). τ_settle=2s(측정: QoS-validate
`elapsed~3.1s`, transient ~1–2s) — §1에 추가, 아래 민감도 병기.

### 5-1. dwell 용량반응 (Q3-like, k*=5) — `d2_dwell_response.pdf`
> ⚠️ **[v13 정지]** 아래 회복률 수치의 **절대 높이·k* 의존성은 논문 미등재** — 하드웨어 2점 대조에서 순서
> 뒤집힘([d2_followup_report.md](d2_followup_report.md) 47). 회복률은 판정모형(f=T_v/τ)이 결정하며 μ 분포에
> robust(46-2)하나 하드웨어로 검증 안 됨. **유지되는 것: persist 추세(상한식 귀결)와 "어떤 임계 후 평탄" 형태만.**
> 그림은 persist를 primary로, 회복률을 미검증 shape로 표기.
| T_v (s) | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|
| 회복률 | 0.40 | 0.48 | 0.72 | 0.88 | **1.00** | 1.00 | 1.00 | 1.00 |
| churn(hot) | 4.6 | 4.5 | 4.3 | 4.1 | 4.0 | 4.0 | 4.0 | 4.0 |
| persist(회복,s) | 14 | 17 | 20 | 22 | 25 | 31 | 37 | 48 |

- **회복률은 T_v≈τ_settle(2s)에서 평탄(1.0)에 도달**, 그 위로는 불변. **T_v를 더 키우면 persist만 증가**(31→48).
  → 변형 A(T_v=0, 회복 0.40·churn 4.6)는 이 곡선의 한쪽 끝, BoundGuard(T_v=3, 1.0)는 평탄부. **"strawman 비교"
  반론 종결**: A는 dwell을 극단(0)으로 뺀 지점이고, 최적 T_v는 τ_settle 근방(논문 T_v=3은 약간 보수적).
- **τ 민감도**: 무릎 위치 = τ_settle(τ=1→무릎 1, τ=2→2, τ=3→3). **무릎 위치는 τ에 종속(구조적)**이나 **형태
  (상승→평탄, 이후 persist만 증가)는 τ 무관하게 robust.** 절대 무릎값이 아니라 이 형태가 결과다.
- 하드웨어 대조: T_v=0 sim 회복 0.40 vs 측정 A Q3 0.20 — **같은 방향**(둘 다 낮음). T_v=3 sim 1.0 vs 측정 5/5.

### 5-2. (N_cand, k*) 평면 — `d2_budget_rank.pdf`
- 회복 가능 영역 = **k* ≤ N_cand**(대각선 아래). 그 안에서 **persist가 k*에 선형 증가**(margin 0 기준 13→71s).
- **상충 수치화**: 예산을 키우면 깊은 k* 회복이 가능해지나 **최악 persist가 T+N_cand(T_v+δ)로 함께 증가**.
- §IV 연결: Q3(k*=5, N=5, **margin 0**)는 대각선 위(경계), 2-gen(k*=12 > N=5)은 미회복 영역. **Q4의 "rank-5 =
  예산 마지막 칸" 취약성이 평면 위 경계점으로 나타남** — 워킹셋에 vision 하나 더하면 k*=6>5로 미회복 영역 진입.

### 5-3. 부하와 상한 긴밀도 — `d2_load_tightness.pdf`
- **search는 부하 무관**(20.9s 일정) — 구조적(탐색은 후보 수·δ에만 의존). bound 대비 **tightness 0.91 일정**.
- **persist는 λ→μ*에서 발산**(λ/μ*=0.5→0.99: 27→1046s) — **C3의 T_stable 발산을 재현**. 상한이 느슨한 구간은
  없음(search는 항상 ~0.91×bound); 느슨해지는 것은 persist의 drain 항이며 이는 bound의 T_stable 부분.

## §6. 논문 반영안 (작업 45-2)

### 배치
**Q7 신설**(§IV Q6 뒤) 권장 — Q6가 "각 메커니즘을 극단에서만 제거"라는 limitation을 남겼고 D2가 그 스케일링을
채우므로, Q6 직후 확장 소절이 자연스럽다. (대안: Q6 limitation 3을 각주로 축소하고 D2를 Threats/appendix에.)

> ⚠️ **[v13 정정]** 아래 초안 중 **회복률 곡선(dwell capacity-response)의 절대 높이·k* 의존성은 논문에 싣지
> 않는다** — 하드웨어 2점 대조에서 순서가 뒤집혀 미검증([d2_followup_report.md](d2_followup_report.md) 47).
> 무릎 위치도 **모형 입력**(τ_settle)이지 발견이 아니다(50-1). 남는 주장: (i) 형태(신뢰도는 어떤 임계 이후
> 평탄, 그 뒤 persist만 증가; μ 분포에 robust), (ii) persist의 상한식 귀결, (iii) budget×rank 평면 위 하드웨어
> 점 위치. 절대 회복률 수치는 그림에서 정규화/제거.

### Q6 limitation 3 교체 문안 초안
> 현행: *"the ablation varies each mechanism at its extreme rather than sweeping it ... without characterizing
> how that contribution scales with T_v or N_cand."*
> **교체안**: *"We complement the extremal ablation with a trace-driven fluid simulation, parameterized entirely
> from the measured service rates and transition costs of Section~\ref{sec:evaluation} (no free parameter is fit
> to a sweep). Reproducing the three measured operating points to within 8\%, it shows that recovery reliability
> rises with the dwell window and plateaus once the window covers the post-swap settling time (~2s measured),
> beyond which a larger T_v only lengthens persistence---so variant A (T_v=0) is one end of a capacity-response
> curve rather than a strawman. In the (N_cand, k*) plane, recovery requires k* <= N_cand and the worst-case
> persistence grows as T+N_cand(T_v+delta) with the budget, making the zero-margin operating point of Q3
> (k*=5, N_cand=5) a boundary rather than an interior case. These are simulation trends and threshold locations,
> not absolute predictions."*

### §IV Q4 rank-5 margin-0 연결 문안
> Q4 서술("rank five is the last position the budget admits, so the margin there is zero ... with one more
> foreground model it would sit at rank six and never be evaluated")에 각주: *"The (N_cand, k*) sweep of the
> D2 simulation places this operating point on the boundary of the recoverable region; the same figure locates
> the two-generative-model case (k*=12) well inside the unrecoverable region."*

### 모형이 담지 않는 것 (반드시 명시)
> *"The simulation is a fluid-level model calibrated to measured rates; it does not model interference
> microstructure, scheduler internals, or stochastic service-time variation. Its results are trends and
> threshold locations (where reliability plateaus, where the recoverable region ends), not absolute predictions
> of recovery time. The knee of the dwell curve coincides with the measured settle time by construction; what
> the sweep establishes is the shape (plateau, then persistence-only growth), which is invariant to that
> constant."*

### 그림 구분 (작업 45-1)
- 신규 그림 3종 `d2_dwell_response`·`d2_budget_rank`·`d2_load_tightness`은 캡션·범례·각주에 **SIMULATION** 명시.
- `confirmed_values.json`·`check_figures.py`에 **`source_kind` 필드 추가**(기존 30항목 = `hardware`, 강제 검증).
  시뮬 값이 하드웨어 그림에 섞이면 checker가 `source_kind` 불일치로 잡는다.

## 무결성
- 하드웨어 재실험 없음. 알고리즘·파라미터·main.tex·legacy/backup 불변. 시뮬레이션 값은 하드웨어 값과 스키마·범례에서 구분(작업 45).
