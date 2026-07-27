# Q3 재실행 — 논문 설정 정합(vision-4 + per-model SLO) + bound 분해

날짜: 2026-07-22 · **핵심: 논문 설정에선 Q3가 Q4-경계로 바뀐다(§0.2에 따라 그대로 보고)** · main.tex 불변

## 요약 (질적 변화 — 되돌리지 않고 보고)
- 이전 Q3 성공(BoundGuard 회복 V=0.54)은 **vision-3 + 균일 slo_ms=15** 설정 특유였다.
- **논문 설정(vision-4 foreground + per-model $L_{SLO}$=5×TableI)으로 재실행하니 결과가 질적으로 다르다**:
  top-1(ggggg)이 위반하고 **전 후보가 검증 창 내 실패** → BoundGuard가 ε 아래로 **완전 회복하지 못한다**
  (최종 V≈1.1~1.4, ε 경계). 그러나 baseline(V≈7~8.5) 대비 **bounded envelope**를 유지 → **Q3라기보다 Q4에 가깝다.**
- 원인 두 가지(§3): (i) **4 vision이 GPU를 포화**시켜 LLM 배치가 결정 요인이 아님 + LLM-CPU가 4-vision
  CPU측 파이프라인과 경합, (ii) **거친 전환 δ=3.7s > $T_v$=3s** — LLM-CPU 후보(cand_3)를 전환 완료 전에 검증해 기각.

## §2. 재실행 구성 (논문 정합) + 재확인한 후보 랭킹
- 워킹셋: llama1b(bg, view5) + **yolo11s, yolo11m, resnet50, mobilenet_v2**(foreground view1-4).
- $L_{SLO}$(모델별, 5×TableI): yolo11s=155, yolo11m=209, resnet50=56, mobilenet_v2=32 ms. (균일 15 폐기)
- **재확인 랭킹**(cpu_gpu, α=0.3, β=0.5, burst infps): top-5 = **ggggg, ggggc, cgggg, gcggg, gggcg**.
  → **LLM-CPU(cgggg) = 3위**(이전 vision-3의 5위 경계 아님; yolo11m 추가로 vision-이동 후보보다 상위). N_cand=5 이내.
- λ: **52**(top-1 ggggg가 V≈7.5로 명확히 위반; λ≤46은 feasible, λ=50~52에서 onset). ε=1.0, $T_v$=3s, N_cand=5.
- 부하 노브(명시): `FSRR_RATE_REPLICATE=1`, `FSRR_FRAME_BUFFER=30`. (기본값이면 이전 결과 재현.)

## §3a. 후보 순회 타임라인 + §3c. 예측 vs 실측 (y2 반전)
BoundGuard(mode 1+validate, D5 확인) validate 결과:

| 후보 | 배치 | 예측 $y_2$ | 실측 validate V(t) @ $T_v$ | 검증 |
|---|---|---|---|---|
| cand_1 | ggggg (top-1) | **0.030** (예측 최선) | **12.9** (실측 최악) | 실패→advance |
| cand_2 | ggggc | 0.291 | 11.6 | 실패→advance |
| cand_3 | **cgggg (LLM→CPU)** | 0.531 | **2.3** (전환 중, δ>Tv) | 실패→advance |
| cand_4 | gcggg (yolo11s→CPU) | 0.346 | 1.1 | 실패→advance |
| cand_5 | gggcg (resnet→CPU) | 0.336 | ~1.2 | 마지막→stay |

- **y2 반전 확정**: 예측기는 ggggg를 miss 최소(0.030)로 보나 실측 V 최대(12.9). cand_3~5는 예측 나쁨(0.34~0.53)이나
  실측 양호(1.1~2.3). → 예측기가 **GPU 경합을 모델링 안 해** all-GPU를 과대평가.
- **전 후보 validate>ε → 모두 advance.** BoundGuard는 cand_5(gggcg)에 안착, 최종 V≈1.1~1.4.

## §1/§3b. bound 분해 ($T_{valid}$ / $T_{stable}$) + δ
- **δ(거친 LLM 전환) = 3.7s**(bg 로그 `llama1b/cpu loaded 3663ms`) **> $T_v$=3s.**
  → cand_3(cgggg)는 LLM GPU→CPU 전환이 끝나기 전에 validate(3.1s)돼 V=2.3(과도기)로 기각. **거친 전환의 δ가
  검증 창을 초과하는 것이 회복 실패의 직접 원인 중 하나**(§1의 예상대로 δ가 hot-swap보다 큼).
- $T_{valid}$(탐색) = burst(8s) + cand_1..4 각 (~$T_v$+swap ≈3.5s) ≈ **22s**. bound $T+N_{cand}(T_v+\delta)$
  = T + 5×6.7 ≈ T+33.5s → **$T_{valid}$는 bound 이내**(35s persistence가 bound 위반이 아님).
- $T_{stable}$: 본 설정에선 V가 ε 아래로 수렴 안 함(경계 ~1.1) → **완전 배수 구간 부재**(Q4 성격). fluid 모델
  $Q/(\mu^*-\lambda)$ 검증은 후보가 feasible한 경우에만 적용 가능 → 여기선 해당 없음.

## §3d. Stop-restart 동작 확인
- 로그: `Mode 0: same placement cand_1 -> burst; applying infps in-place (no restart)`.
- **재시작 없음** — 예측기 top-1(ggggg)이 **이미 실행 중인 burst 배치와 동일**해 옮길 대상이 없다.
  → Stop-restart 곡선이 Static과 겹치는 이유(다운타임 없음, 재시작 자체가 안 일어남). 정직한 설명.

## §3e. 네 기법 비교 (그림: ~~`docs/figures/q3_paperset_q4like.pdf`~~ — **폐기**)

> ⚠️ **이 절의 수치는 무효다 (2026-07-24).** Δ-창 `V(t)` 수정
> ([vt_definition_fix_report.md](vt_definition_fix_report.md)) 이후 이 시나리오(vision-4 paper-set,
> λ=52, per-model `L_SLO`=5×Table I)를 재실행한 결과 **네 기법 모두 `persist=0s`, `maxV` 0.19–0.40**으로
> 위반에 진입조차 하지 않는다. 실측 지연(12–14ms / 5–7ms)이 `L_SLO`(155/209/56/32ms)보다 한 자릿수
> 낮고 backlog≈0, 드롭 0 — 이 워크로드는 λ=52에서 그냥 **feasible**하다.
>
> 아래 표의 `V≈7.5–11.8`은 `ℓ_i`가 lifetime 누적평균이던 시절 burst 전이가 감쇠하지 않아 생긴
> **지표 아티팩트**였다. 그림은 `q3_paperset_q4like_RETIRED.pdf`로 이동했다. Q3 시연은 실제로
> infeasible해지는 vision-3 + LLM 시나리오(`q3_misprediction.pdf`)만 사용한다. vision-4 paper-set으로
> Q3를 보이려면 λ를 52보다 높여 재설계해야 한다(별건). 상세: [hotswap_buffer_bias_fix.md](hotswap_buffer_bias_fix.md) §6c.

| 기법 | maxV | 최종 V | persist(V>ε) | 회복 |
|---|---|---|---|---|
| Static | 7.53 | 7.49 | 78s | ✗ |
| Stop-restart | 8.68 | 7.38 | 78s | ✗ (재시작 없음) |
| Adaptive | 10.93 | 8.44 | 78s | ✗ |
| **BoundGuard** | 11.84 | **1.39** | 53s | ✗(ε 경계) but **bounded envelope** |

- **BoundGuard maxV=11.84 > baseline** — 탐색 중 일시적으로 나쁜 후보(cand_1 cold+cand cycling)를 거쳐 정직하게 높음.
- **BoundGuard 최종 V≈1.1~1.4로 baseline(7~8.5)의 1/6** — 완전 회복은 아니나 유계·최저.

## 이전 실행(vision-3, 균일 SLO)과의 차이
| | vision-3 + slo15 | vision-4 + per-model SLO(논문) |
|---|---|---|
| LLM-CPU 순위 | 5위(경계) | **3위** |
| BoundGuard 최종 V | 0.54(**회복**) | 1.39(**미회복, 경계**) |
| 성격 | Q3 | **Q4-경계** |
| 원인 | LLM-CPU가 feasible(cand_5 default commit) | 4-vision GPU 포화 + LLM-CPU도 CPU경합 + δ>Tv |

## 판정 (§0.2 준수 — 되돌리지 않음)
- **논문 설정에서 이 워킹셋은 Q3(대안 회복)가 아니라 Q4(전 후보 실패·bounded envelope)에 가깝다.**
- BoundGuard의 우위는 **여전히 실증됨**(유계 최저 V, Adaptive의 단조 증가 대비). 다만 서술은 Q3가 아니라
  **Q4(bounded envelope)** 또는 "부분 회복"으로 해야 정직하다.
- **회복을 보이려면**(사람 판단): (a) 후보 중 feasible한 것이 있어야 함 → vision 부하를 줄이거나(3개) SLO를
  현실화(생성적 5× 대신 tighter), (b) $T_v > \delta$(거친 전환 δ=3.7s 초과)로 LLM-CPU 후보를 전환 완료 후 검증.
  — 어느 것도 후보/α/β 조작은 아니나, **논문 설정 자체의 재검토**가 필요.

## 무결
- BoundGuard mode 1+validate(D5) 확인. GPU 폴백 0. 네 기법 동일 시나리오·파라미터. 예측기·α·β·배선 불변.
  legacy/scripts/backup 무변경. main.tex 불변. 코드 변경 없음(실험·그림·문서만).
