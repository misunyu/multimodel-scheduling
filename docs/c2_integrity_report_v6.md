# C2/Q6 무결성 v6 — 샘플 기반 지표 편향 감사 · Q1.3 ∫V 검증 · 확정본 정정

날짜: 2026-07-26 · 재실험 없음(코드·로그 판독) · main.tex 불변 · 파라미터 불변
선행: [c2_integrity_report_v5.md](c2_integrity_report_v5.md), [metric_definitions.md](metric_definitions.md)

> **판정 요약** (작업 19~21):
> - **작업 19**: `b2_analyze.py`/`b3_aggregate.py`에서 **샘플 기반 편향 지표는 `persist` 하나뿐.** `cumV(∫V)`는
>   시간가중(dt), `T_valid`·`T_stable`은 wall-clock. persist 편향은 **전환 변형에만**(Q3: BoundGuard −5샘플,
>   Static/SR/Adaptive 0) → "20 vs 105" 대비의 분자만 축소.
> - **작업 20**: Q1.3 6배는 **∫V(시간가중, 무편향)**. 창 정합(58s) 시에도 ~5.7×, full 6.2×. Static 창이 더 짧아
>   **보수적**(6× 유지). 20-3 1행.
> - **작업 21**: (1) persist 20.0(샘플수)→wall-clock **25.4 = search 22.6 + drain 2.8**(정확 분해). 교체안·유지안
>   문장 초안 제시. (2) v5 "b2rep 7런"은 오기 → **7/35 = b2rep 5 + C2 2**(paper 상한 대비 초과는 0건). (3) Q5
>   문장을 하네스 제약→"실험 설계상 burst에서 활성"으로 정정.

## 작업 19 — 지표 샘플 기반 여부 감사

### 19-1. 지표 분류 (main.tex §IV 산출 스크립트)
| 지표 | 계산 방식 | main.tex 대응 | 편향(전환 중 누락) |
|---|---|---|---|
| `persist` (b2_analyze, b3_aggregate) | **`sum(1 for x in v if x>ε)` = 샘플 개수** | Q3 20.0·Q5·Q1.3 persistence | **예 (전환 변형만)** |
| `cumV`/∫V (b3_aggregate) | **`Σ v[i]·(t[i]−t[i-1])` = 시간가중** | Q1.3 630 ∫V(531 등) | 아니오 |
| `T_valid` (b2_analyze) | `rec_apply − burst` (타임스탬프) | Q3 720 24.6 | 아니오 |
| `T_stable` | `below_t − rec_apply` (타임스탬프) | — | 아니오 |
| `lastV`/`maxV` | 점값 | Q1.3/Q3/Q5 lastV | 아니오 |
| `delta`/loads | 로그 타임스탬프 | δ | 아니오 |
- **샘플 기반 편향 지표 = `persist` 하나.** (19-4 3행) 나머지는 무편향.

### 19-2/3. 편향 방향 정량화 (Q3, 변형별)
| 변형 | wall-clock | 누락 샘플 | gaps>1.5s | persist 편향 |
|---|---|---|---|---|
| Static | 106.8s (from burst; full-CSV 114) | **0.0** | 0.0 | 없음 |
| Stop-restart | 106.8s | **0.0** | 0.0 | 없음 |
| Adaptive | 107.0s | **0.0** | 0.0 | 없음(top-1 고착, 전환 0) |
| **BoundGuard** | 83.0s (from burst; full-CSV 90) | **5.0** | 1.0 | **−5s (20.0 vs wall-clock 25.4)** |

> ⚠️ [v7 정정] 위 wall-clock 열은 **full-CSV**(stable 포함) 기준이었다. **from-burst dur = 106.8/106.8/107.0/83.0**
> (v3 작업8 정합). 참조점 차이일 뿐 누락(0 vs 5) 결론은 불변. → [c2_integrity_report_v7.md](c2_integrity_report_v7.md) 작업24.
- **누락은 전환 변형(BoundGuard)에만.** 1개 gap = LLM CPU-load 전환(~2.9s). → **편향 방향 확정(19-4 1행)**:
  샘플 기반 persist는 전환하는 쪽만 깎아 "BoundGuard 20 vs baseline 105–106" 대비의 분자를 축소한다.
- baseline 105–106도 같은 persist 함수 출신이나 **누락 0**이라 wall-clock과 동일(단 censored=런길이, v3 §8-4).

## 작업 20 — Q1.3 ∫V 6배 검증 — **유지 (보수적)**

- **∫V는 시간가중 적분**(`v[i]·dt`), 샘플 개수 아님 → 전환 간격이 dt로 계수되어 **persist식 편향 없음**.
- **창 정합**(전 변형 58s 절단):

| plat | Static ∫V(full) | Static ∫V(58s) | 회복3기법 ∫V | 배수(full) |
|---|---|---|---|---|
| gpu | 531 | 482 | 82/89/85 | **6.2×** |
| npu | 530 | 480 | 76/87/91 | **6.3×** |

- 회복 변형은 회복 후 V=0이라 창 절단해도 ∫V 불변. Static은 창이 짧을수록 ∫V 작음 → **보고된 6×는 하한**:
  회복 변형의 더 긴 창(63–68s)에 맞추면 Static이 더 쌓여 배수↑. → **630행 6배 유지 + "Static 창 더 짧음에도" 추가.**
- 잔여: Stop-restart는 재시작 중 gap 5–7개. dt 시간가중이라 계수되나 gap을 post-gap V(낮음)로 가중 → 소량
  과소집계 가능(단방향, 배수를 낮추는 쪽). 최악 보정에도 배수 ≥ ~5(3기법 평균이므로 영향 제한). **판정 불변.**

## 작업 21 — 확정본 숫자·문구 정정

### 21-1. persist 20.0 처리 — 분해 + 문장 초안
- **분해(b2rep 동일 5런, 정확)**: **persist 25.4 = search(t0→회복적용) 22.6 + drain(적용→V≤ε) 2.8** (계수차 0.0).
  (v4의 25.4 = wall-clock, v5 작업16의 search ≤23s와 정합: b2rep search 22.6 ≤ 24.)
- **본문 "lasts 20.0s"는 지속시간 주장인데 값은 샘플 개수** → 부정확. 두 안:
  - **[교체안]** *"The violation persists for 25.4±1.4 s (wall-clock t0→recovery; = 22.6 s search + 2.8 s
    drain) before BoundGuard recovers, whereas no baseline recovers within the run."* (상한 비교는 search
    22.6 ≤ 24 s, 작업16과 정합)
  - **[유지안]** 20.0을 두되 *"20.0±1.5 violating samples (a lower bound; wall-clock persistence is 25.4 s as
    transitions pause sampling)"* 각주.
- **baseline 105–106**: 같은 persist 함수지만 누락 0 → wall-clock과 동일. 단 **censored(미회복)** → v3 §8-4대로
  **"회복 없음(≥창길이)"** 로. 교체/유지 어느 안이든 대비 축은 **"BoundGuard 회복(25.4s) vs baseline 미회복"**.

### 21-2. 런 수 정정
- v5 작업16의 *"b2rep 7런"*은 **오기**. 실측-δ 상한(T+15+Σδ_meas) 초과는 **7/35 = b2rep 5 + C2 2**
  (w6b1rep1 22>21.9, w6b2rep1 23>22.5, 0.1·0.5s). **b2rep 5는 vision δ 미로깅으로 Σδ under-count**, C2 2는
  0.1–0.5s marginal. **paper 상한(24–26s) 대비 초과는 0/35**(search max 23). 실제 위반 아님.

### 21-3. Q5 문장 정정 (하네스 제약을 시스템 성질로 쓰지 않음)
- v5 초안 *"the adaptive controller acts only in the candidate phase of the fixed schedule"*는 **하네스 제약**.
- **정정 초안**: *"In Q5/§4b the all-CPU stable phase is itself infeasible for the workload (V≈7>ε before the
  burst). By experimental design the controller is enabled at the burst injection, so recovery times are
  reported from that point; the pre-burst violation reflects the stable placement, not detection latency."*

## main.tex §IV 수정 목록 — **최종 확정본** (v3 §8-4 + v4 + v5 + v6)

| 위치 | 확정 조치 | 근거 |
|---|---|---|
| Q3 720 persist | 교체안(25.4 wall-clock, =22.6+2.8) **또는** 유지안(20.0=샘플수 하한+각주). 사람 택일 | 작업19·21-1 |
| Q3 720 T_valid 24.6 | 유지(=burst→적용, 결정적) | v5 작업15 |
| Q3 720 상한 비교 | **search(≤23s) vs 24–26s**. "persist vs 상한" 표현 삭제 | v5 작업16 |
| Q4 749 | 24s = bound 24s tight 유지(sample-gap 0) | v4 작업11 |
| Q5 774 | burst 기준 14.2/15.2 유지 + stable 위반 각주(21-3 초안). **t0 기준 미사용** | v5 작업17·21-3 |
| Q1.3 630 "6배" | **유지**(∫V 시간가중, 보수적) + "Static 창 더 짧음에도" | 작업20 |
| Q1.3 630 Static 59.4 · Q3/Q5 baseline persist | **"회복 없음"(censored, ≥창길이)** | v3 작업8 |
| 공통 | 각 실험 관측 창·t0 기준 표(v4 작업14) | v3–v6 |
| §II bound | **불변** | v4·v5 |

## 무결성
- 재실험 없음(작업 19~21 전부 코드·로그 판독+통계). 알고리즘·파라미터·main.tex·legacy/backup 불변. 코드 무변경.
- v6 자기정정: v5 "b2rep 7런" 오기 → 5+2. 그 외 v5 판정 4건은 유지.
