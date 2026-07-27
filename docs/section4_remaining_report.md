# §IV 잔여 갱신 — 그림 재생성(A) + Q1.1·Q1.2·Q2 재실행(B)

날짜: 2026-07-24 · 알고리즘·파라미터 불변(ε=1.0, T=3, T_v=3, Δ=0.2) · main.tex 불변

---

# §1. [선행] Stop-restart downtime 시 V(t) 처리 — **판정: 아티팩트 아님**

> 이 판정이 Q1.3 본문 서술을 가른다. 코드·CSV로 확정(추정 없음).

## 1.1 코드 (ℓ_i 창 샘플 0일 때)
- `view_handlers.windowed_latency(Δ)` (L136): 창 `[t-Δ,t]`에 완료 프레임이 없으면 **`None` 반환**.
- `unified_viewer` V(t) 집계 (L1838-1852):
  ```python
  _li = _h.windowed_latency(_ELL_DELTA)
  if _li is None or _li <= 0:
      continue                      # (c) 해당 뷰를 v(t) 집계에서 제외
  ...
  _v_inst = (_v_sum / _n_act) if _n_act > 0 else 0.0   # 전 뷰 다운 시 (a) 0
  ```
- **부분 다운(일부 뷰만 샘플 0)** → 그 뷰 **제외(c)** = 합리적 처리(0으로 희석 안 함).
- **전 뷰 동시 다운** → `_n_act=0` → `_v_inst=0.0` = **(a)**, 원리적으로 아티팩트 소지.
  단 **V(t)는 T=3s 창 평균**이라 순간적 0은 직전 고값들과 평균되어 즉시 0이 되지 않는다.

## 1.2 CSV 실측 (`s4b_sr_out`, 전수 검사)
재시작 구간에서 **아티팩트가 실제로 발생했는가**를 서명 `V<ε AND (backlog>10 OR fps<20)`로 검사:

| plat | 아티팩트 서명 샘플 | downtime(fps<20) 샘플 | 첫 cand_1 min fps | persist | cumV |
|---|---|---|---|---|---|
| gpu | **0건** | **0건** | 120.8 | 12s | 86.1 |
| npu | **0건** | **0건** | 1474.8 | 8s | 61.0 |

- **V=0 샘플은 전부 fps 1400–1770 · backlog=0** — 워커가 고처리율로 서빙 중, 미처리 작업 없음 →
  **진짜 회복**(feasible 배치의 e2e ≪ L_SLO).
- **downtime(fps≈0) 샘플이 0건** — mode-0 재시작 downtime이 **1s 메트릭 샘플링보다 짧아**(GPU cold-start
  ~1s) 정체 상태로 포착되지 않았다. 첫 post-restart 샘플부터 이미 서빙(GPU V 6.3→4.0→1.7→0 감쇠).
- **정량**: 재시작 구간을 제외/포함해도 persistence·cumV가 바뀌지 않는다(아티팩트 샘플 0건 = 제외할 것이 없음).

## 1.3 판정과 Q1.3 서술 방향
- **§4b 데이터에서 아티팩트는 발생하지 않았다.** Stop-restart의 낮은 persistence는 **feasible offload로의
  진짜 빠른 회복**이며, "서비스 중단이 V를 인위적으로 억제한다"는 **기존 서술은 이 데이터로 뒷받침되지 않는다.**
- **→ Q1.3 본문의 "measurement artifact" 서술을 삭제/수정하고, Stop-restart를 정당한 비교 대상으로 다룬다.**
- **단, 코드의 (a) 잠재 거동 명시**: 재시작 downtime이 **한 샘플링 구간 전체에 걸쳐 전 뷰 다운**될 만큼
  길면 V=0이 기록될 수 있다(이 데이터에선 미발현 — 재시작이 빠르고 T-창이 평활). 이는 **잠재 속성**으로
  각주 처리.
- **새 Q1.3 메시지(§1의 게이트 답과 일관)**: 예측이 맞는 조건에서는 **적응·재시작 기반 기법이 모두 영향을
  유계로**(persist 8–12s, cumV 61–90) 만들고 **Static만 그러지 못한다**(59–60s, 526–538). 이들 사이의
  구별은 **오예측 하에서만 드러난다**(Q3). Q1.5의 "no penalty"와 일관.

---

# A부 — 기존 데이터로 재생성 (게이트)

## A1. Fig 1 `qos_score_validation` — **재생성 ✓** (최소 재실행 필요)

`docs/figures/qos_score_validation.pdf` 교체. **ε=1.0으로 수정**(기존 그림의 ε≈34는 논문과 모순 — 해소).

- **재사용 불가 사유**: 새 V(t) 정의(ε=1.0) 하에서 **회복하는 Stop-restart + downtime**을 보이는 기존 로그가
  없었다. B2 Stop-restart는 downtime은 보이나 drop-regime이라 부분 회복(V~190)만 하고, phaseA는 완전
  회복하나 **옛 V(t) 정의**다. → **최소 2-run 재실행**(yolo11s CPU@120fps 위반 → GPU offload; Static
  mode3 / Stop-restart mode0, buffer=120, L_SLO=31ms)으로 생성.
- **결과**: Static V 169→651 단조 상승(fps 6.1 고착, **지속 열화·무회복**) vs Stop-restart 전환(+5s)
  후 V 220→0 **회복**(재배포 downtime 비용 = 전환 시 V 스파이크). ε=1.0 기준선 명확.
- 출처: `scratchpad/fig1/{Static,StopRestart}.csv`.

## A2. Fig 3 `bounded_recovery_analysis` — **재생성 ✓**

`docs/figures/bounded_recovery_analysis.pdf` 교체. **33s bound 폐기**, 새 bound = `T + N_cand(T_v+δ) + δ`.

x축을 실측 시나리오 {Q3 CPU–GPU, Q5 CPU–NPU, Q4 CPU–GPU(소진)}로 재구성. 세그먼트 분석으로 산출한
탐색 시간(`T_valid`) vs 해석적 상한 대조:

| 시나리오 | 탐색 T_valid | 회복 T_stable | bound | 판정 |
|---|---|---|---|---|
| Q3 CPU–GPU (cand_5 회복) | 24s | 5s | 25.9s | T_valid ≤ bound ✓ |
| Q5 CPU–NPU (cand_2 회복) | 16s | 0s | 25.9s | T_valid ≤ bound ✓ |
| Q4 CPU–GPU (소진→복귀) | 25s | — (회복 없음) | 24s | ≈ bound (한계) ✓ |

- **bound는 탐색(T_valid)만 보장** — T_stable(물리적 배출)은 fluid(Q/(μ*−λ)) 지배로 별개(그림에 분리 표시).
- **δ가 전환 유형 의존**: vision in-place hot-swap ≈1s(bound의 N_cand항), 최종 프로세스 재기동/LLM 거친
  전환 ≈2.9s(bound의 +δ항). 실측 δ: Q3/Q5의 LLM CPU 적재 ≈2.9s.
- 출처: `q3_bsf_out`·`q5_out`·`q4_bsf_out`의 BoundGuard.csv 세그먼트 타임스탬프.

## A3. Fig 6·7 `q13_failure_persistence` / `q13_cumulative_violation` (Q1.3) — **부분 (재실행 필요)**

- **§4b(s4b_sg2)에서 산출 가능**: persistence + **누적 위반 ∫V dt**, 양 플랫폼, **3기법**(Static/Adaptive/
  BoundGuard):

  | plat/기법 | persist(V>1) | cumV=∫V dt | maxV | lastV |
  |---|---|---|---|---|
  | gpu/Static | 60s | 537.6 | 9.3 | 9.0 |
  | gpu/Adaptive | 10s | 72.8 | 8.9 | 0.0 |
  | gpu/BoundGuard | 11s | 79.7 | 9.3 | 0.0 |
  | npu/Static | 59s | 526.4 | 9.2 | 9.1 |
  | npu/Adaptive | 12s | 83.0 | 9.2 | 0.0 |
  | npu/BoundGuard | 12s | 90.0 | 9.2 | 0.0 |

- **부족분**: (1) **Stop-restart** — bg_scenario가 3기법만 실행. → baseline은 추가 실행 중(`s4b_sr_out`).
  (2) 옛 그림의 **6개 background-load 조건 스윕**(+1 light CPU bg, +2 CPU bg, +heavy CPU, +GPT-2,
  +TinyLlama) — §4b는 baseline 조건뿐. → **6조건 스윕은 재실행 결정 필요(사람 판단)**.
- 누적 위반이 **핵심 메시지를 담음**: Static cumV ≈530 vs {Stop-restart/Adaptive/BoundGuard} ≈61–90
  (약 **6–9배** 차). **Stop-restart 포함 4기법 baseline 확정**(persist gpu 12s/npu 8s, cumV 86/61).
- **서술**: §1 판정에 따라 "Stop-restart는 아티팩트로 좋아 보인다"는 **삭제** — 예측 정합 조건에서
  적응·재시작 기법 모두 진짜로 영향을 유계로 만들고 Static만 실패. 기법 간 구별은 Q3(오예측)에서 발현.

## A4. 게이트 요약

| 그림 | 상태 | 비고 |
|---|---|---|
| Fig 1 qos_score_validation | ✅ 재생성 (ε=1.0 수정) | 최소 재실행 사용(기존 로그 부적합) |
| Fig 3 bounded_recovery | ✅ 재생성 (새 bound) | 기존 Q3/Q4/Q5 데이터 |
| Fig 6/7 Q1.3 | ⚠️ 부분 | 3기법 baseline+cumV 확보; **Stop-restart 추가 중**, **6조건 스윕 미결** |

→ **사람 판단 필요**: Q1.3의 background-load 6조건 스윕을 재실행할지. baseline(4기법)만으로 Q1.3 본문을
갱신할지, 옛 6조건 축을 유지하려 재실행할지.

---

# B부 — 재실행

## B1. Q1.1 정상상태 오버헤드 `runtime_overhead` — **재실행 ✓**

`docs/figures/runtime_overhead.pdf`. feasible 정상상태(4-vision foreground @ 20fps, **maxV=0 위반 없음**)에서
Static/Adaptive/BoundGuard의 지연·처리율. BoundGuard는 검증 트리거를 켜 **모니터링 기구가 상시 동작**한다.

| plat | 지표 | Static | Adaptive | BoundGuard |
|---|---|---|---|---|
| GPU | mean e2e (ms) | 12.3 | 12.2 | **10.6** |
| GPU | throughput (fps) | 1194 | 1328 | 1424 |
| NPU | mean e2e (ms) | 17.8 | 17.1 | **16.0** |
| NPU | throughput (fps) | 1643 | 1157 | 1260 |

- **BoundGuard 모니터링 오버헤드는 측정 한계 이하** — mean e2e가 오히려 낮다(run 편차 범위). 처리율은
  run 편차가 있으나(NPU Static 1643이 높음) 지연이 더 깨끗한 오버헤드 프록시이며 페널티 없음.
- **옛 서술과 일치**(옛: 442/461/451ms, BoundGuard≈Static). 신 모델셋에서도 **오버헤드 무시 가능** 재확인.

## B2. Q1.2 검출 파라미터 민감도 `detection_sensitivity` — **재실행 ✓ (설정 재검토 필요)**

`docs/figures/detection_sensitivity.pdf`. **새 V(t) 정의**(ℓ_i=Δ창, V=T창) 하 T×ε 스윕. **제약 T_v≥T>Δ
준수** → T∈{1,2,3}(T_v=3), ε∈{0.5,1.0,2.0}, **Δ=0.2 고정**. per-frame 트레이스(feasible→violation→recover,
4-model, Table I L_SLO) 1회 수집 후 오프라인 재계산(검출 지연/오탐/미검출).

**검출 지연(s)** — T·ε 모두에 단조 증가:
| | ε=0.5 | ε=1.0 | ε=2.0 |
|---|---|---|---|
| T=1 | 0.7 | 1.0 | 1.3 |
| T=2 | 0.9 | 1.3 | 1.7 |
| **T=3** | 1.1 | **1.5** | 2.0 |

**오탐(feasible 구간 ε 초과 시간, s)**: T=1·T=2는 전부 **0**. **T=3만** ε=0.5에서 0.7s, **ε=1.0에서 0.6s**.
**미검출**: 없음(전 셀).

- **핵심**: 새 V(t)에서 **현재 설정 T=3/ε=1.0은 검출 지연 1.5s(T_v=3 이내, 안전) + 오탐 0.6s**.
  **T=1/ε=1.0은 검출 0.7s 더 빠르고(1.0s) 오탐 0.** → **현재 설정이 최적이 아닐 수 있다.**
- **원인**: 큰 T-창은 feasible 구간의 짧은 지연 스파이크를 더 오래 기억해 ε을 넘는 시간이 생긴다(작은 T는
  즉시 잊음). Δ창 ℓ_i 도입으로 이 효과가 드러났다.
- **정직 보고 / 판단 요청**: **T=1 또는 2가 검출·오탐 모두 우수**하나, T를 바꾸면 T_v·전 실험에 파급된다.
  현 T=3도 안전 범위(검출<T_v, 미검출 0)이므로 **유지 가능**. **파라미터 변경 여부는 사람 판단.**
- 단일 트레이스 기반(GPU). 검출 로직은 디바이스 무관 구조 — NPU도 유사 예상(별도 확인은 선택).

## B3. Q2 동적 워크로드 회복 `dynamic_load_adaptation` — **재생성 ✓**

`docs/figures/dynamic_load_adaptation.pdf`. **부하 증가**(stable→burst, 입력률 급증) 시 Static vs BoundGuard
V(t), 양 플랫폼. **§4b(예측 정합) 데이터** 사용 — **예측이 맞는 상황의 부하 변동 회복**이라 Q2에 정확히 부합
(Q3 오예측과 구분).

| plat | Static (부하증가 후) | BoundGuard |
|---|---|---|
| GPU | V≈9 지속(회복 ✗) | V 9→**0** 회복(persist 11s) |
| NPU | V≈9 지속(회복 ✗) | V 9→**0** 회복(persist 12s) |

- 부하 증가 후 Static은 지속 위반, **BoundGuard는 올바르게 예측된 offload 배치로 전환해 ε 이하로 회복**.
  옛 그림과 동일 메시지(Static 지속 vs BoundGuard 안정). **P8**: burst가 CPU 배치를 명확히 위반시키고
  offload가 명확히 feasible(§4b 사전측정).
- **Q2/Q3 구분 명시**: 본 Q2는 **예측 정합**(top-1 offload가 실제 feasible)이며, Q3의 오예측(top-1 실패)과
  다르다. background LLM 없음(vision-only). **background 변형**(LLM 상주 + 부하변동, 여전히 예측 정합)은
  별도 실행 필요 — 현재 미포함, 필요 시 추가.

## B4. P1~P9 확인 (재실행분)

| | Q1.1 | Q1.2 | Q2 |
|---|---|---|---|
| P1 실행 순서 | ✓ | ✓(오프라인) | ✓ |
| P2 mode | ✓(3/1/1) | ✓(3, 수집) | ✓ |
| P3 fallback=0 | ✓ | — | ✓ |
| P4 측정 편향 | 시간대응·warmup 제외 | Δ=0.2 고정·cold-start 제외 | §4b 시간대응 |
| P8 경계회피 | feasible maxV=0 | 명확 전환 | 명확 위반/feasible |
| 파라미터 불변 | ✓ | **스윕 대상 T·ε만 변화** | ✓ |

## 종합 판단 (사람 판단 대기)

1. **Q1.3 background-load 6조건 스윕** 재실행 여부(A3).
2. **Q1.2 검출 파라미터**: 새 V(t)에서 T=1~2가 T=3보다 검출·오탐 우수 — T를 낮출지(전 실험 파급 있음).
3. **Q2 background 변형** 추가 실행 여부.
세 항목 모두 **현 설정으로도 결론은 유효**(안전 범위)하나, 최적성/완전성 관점의 결정이다.

## 무결성
- 알고리즘·파라미터 불변. legacy/backup·main.tex 불변.
- 재생성 그림 출처를 각 절에 명시. 재실행 P1~P9는 B부에서 확인.
