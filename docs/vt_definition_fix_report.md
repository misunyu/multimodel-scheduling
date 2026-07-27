# V(t) 지표 정의 수정 — ℓ_i를 Δ 창 평균으로 (논문 정의 준수)

날짜: 2026-07-23 · 논문 정의: ℓ_i=Δ창평균, V=T창평균, T_v≥T>Δ · main.tex 불변

## §1. 진단 — 현행이 논문 정의 위반
- **ℓ_i**: `avg_infer_time + avg_wait_ms` = **lifetime 누적**(`total/count`, `view_handlers.py`). hot-swap 미리셋
  (5s 워밍업만). → Q3 T_stable 46s가 backlog 배수가 아니라 **누적평균 희석**이었음.
- **V(t) T-창**: CSV의 v_score는 **per-tick 값(T-창 미적용)**을 그대로 기록(`unified_viewer.py`). validate만
  3-sample 창(`_v_history_mode2`). → 논문 V=T창평균이 CSV에 미구현.
- **샘플링**: 메트릭 루프 1s cadence, validate poll 200ms. **foreground만** 집계(llama1b는 핸들러 뷰 아님) ✓.

## §2. Δ 결정 (오프라인 스윕, 기법 비교 배제)
### 2a. 선택 기준 (사전 고정)
- 오탐: feasible 정상상태에서 V(t)>ε **60s당 0회**.
- 응답 지연: 위반 감지 lag **≤ T/3 = 1s**.
- 규칙: 두 요건 만족하는 **가장 짧은 Δ**. 제약: Δ<3s.

### 2b. 오프라인 스윕 (per-frame 타임스탬프 1회 수집 → 사후 재계산)
수집: 3-vision, feasible(e2e 3.9ms) → infeasible(354ms) → recover(3.9ms), 21639 프레임.

| Δ | feasible 오탐/60s | feasible maxV | lag_up(위반감지) | lag_dn(회복) |
|---|---|---|---|---|
| **0.2** | **0** | **0.00** | **0.2s** | 3.0s |
| 0.5 | 0 | 0.00 | 0.3s | 3.1s |
| 1.0 | 0 | 0.62 | 0.6s | 3.6s |
| 2.0 | 0 | 0.85 | 0.6s | 4.4s |

- 전 Δ에서 오탐=0, lag_up≤0.6s(≤1s 충족). lag_dn은 T=3s 창 지배(Δ 무관).
### 2c. 결정: **Δ = 0.2s** (가장 짧고 feasible maxV도 최저). T=3s(=T_v) 유지.

## §3. 구현
- `view_handlers.py`: 핸들러에 **Δ-창 latency deque**(`record_latency`/`windowed_latency`) 추가. 프레임마다
  e2e(infer+wait) 기록, `windowed_latency(Δ)`=[t-Δ,t] 평균. reset_stats에서 clear.
- `reactive_deploy._collect_vscore`(validate)·`unified_viewer` CSV v_score: ℓ_i를 `windowed_latency(0.2)`로.
- CSV v_score에 **T=3s 창 평균** 적용(`_vt_window`).
- `check_validate`: post-swap 스냅샷-델타 우회 제거, Δ-창 기반 `_current_vscore` 사용(Δ-창이 이미 post-transition).
- 검증(live): feasible V=0.00, infeasible V 12→370(빠른 스파이크), recover 294→**0.00**(페이즈 내 회복). ✓

## §4. 재실행 (제어 동역학 변경)
> §0.4: **지표가 빨라져 전체 persistence가 균일 축소**됐고 상대 관계는 유지. "BoundGuard가 좋아졌다"가 아님.

### §4b (P1~P9 준수)
| 기법 | GPU persist/lastV | NPU persist/lastV | (이전 누적V) |
|---|---|---|---|
| Static | 60s / 9.0 | 61s / 9.1 | 60s |
| Adaptive | **12s / 0.0** | **12s / 0.0** | 24s |
| **BoundGuard** | **11s / 0.0** | **12s / 0.0** | 24s |
- **BoundGuard = Adaptive 유지**(양 플랫폼). persistence 24→12s 균일 축소(V(t)가 실제 회복 반영). cand_1
  V_postswap=0.000 → commit-and-stay. **Q1.5 "no penalty" 유지.**

### Q3 (vision-3, λ=80)
| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static/Stop-restart/**Adaptive** | 4.1-14.5 | **3.7-4.1** | 107s | **✗ (top-1 오예측 고착)** |
| **BoundGuard** | 14.5 | **0.00** | **23s** | **✓** |
- **교과서적 Q3 유지**: Adaptive 실패, BoundGuard 회복(persist 66→23s 축소). 판정경로(Δ-창 V): cand_1
  auto-advance, cand_2/3/4(LLM=GPU) V=4.2/7.3/13.3 → advance, **cand_5(LLM→CPU) 회복**.

### Q3 bound 분해 (T_valid / T_stable)
- **T_valid**(위반감지→cand_5 적용) ≈ **18s**: cand_1(auto ~1s)+cand_2(4s)+cand_3(3s)+cand_4(4s)+δ.
  **δ(LLM GPU→CPU 거친 전환) = 2.9s** 실측.
  - **bound 대조**: T + N_cand(T_v+δ) ≈ 3 + 4·(3+1_vision) + (3+3_LLM) ≈ 25s. **T_valid(18s) ≤ bound(25s) ✓.**
- **T_stable**(cand_5 적용→V<ε) ≈ **4s**: V 12.8→0.0을 ~4s에.
  - **fluid 대조 — 조건부**: 본 시나리오는 **buffer=12(drop-on-full)**라 cand_5 backlog Q(k)≈3~12로 **작음**.
    Q/(μ*−λ)=11/(137−80)=**0.2s** ≪ T_stable 4s → **T_stable은 fluid 배수가 아니라 T=3s 창에 floored**.
  - 즉 **작은 버퍼에선 T_stable ≈ T(모니터링 창)이 지배**. fluid $Q/(\mu^*-\lambda)$가 지배하려면 Q(k)가
    커야 함(대형 버퍼) — 별도 대형-버퍼 Q3 변형에서만 fluid-dominated T_stable 시연 가능.

## §5. 병행 감사 (유사 누적-지표 재사용)
- **avg_fps**(처리율, CSV view_fps)·**drop_rate**(`_td/_el`): lifetime/누적. **보고 전용(제어 신호 아님)** →
  V(t)보다 낮은 우선순위. per-phase 정확도 위해 윈도우화 여지 있음(수정은 별건, 목록화만).

## 무결
- P1(순서)·P2(mode)·P3(fallback=0)·P7(판정경로/V_postswap 로그)·P9(commit-and-stay) 확인.
- 변경: `view_handlers.py`(Δ-창), `unified_viewer.py`(CSV ℓ_i+T창), `reactive_deploy.py`(_collect_vscore),
  `schedule_executor_main.py`(validate). legacy/scripts/backup·main.tex 불변. 스냅샷 backup/vt_definition_*.

## 종합
- **V(t) 논문 정의 준수**(ℓ_i=Δ창). 회복시간이 이제 **실제 상태**(누적평균 아티팩트 제거).
- §4b(BoundGuard=Adaptive)·Q3(Adaptive 실패/BoundGuard 회복) **질적 결과 불변**, 타임스케일만 정정 축소.
- **T_valid는 bound로 설명**(핵심). **T_stable은 소형버퍼에선 T-창 지배**(fluid는 대형버퍼 조건) — 정직 보고.
