# 포화 가드 (saturation guard) — 설계·구현·재검증

날짜: 2026-07-24 · 대상: `schedule_executor_main.py::check_validate` · 파라미터 불변(N_cand=5, α=0.3, β=0.5, ε=1.0, T=3s, T_v=3s, Δ=0.2s, θ=0.5) · legacy/backup/main.tex 불변

## §1. 설계 (승인된 §1 게이트 반영)

### 1.1 문제 — 추세 검증의 사각지대
추세 기반 검증은 `backlog_slope < −θ`(배출 중)을 feasible 근거로 쓴다. 그러나 큐가 **cap에 클리핑**되면
이 근거가 무너진다.

- 큐가 가득 차 drop-on-full이 발생하면 backlog는 상한에 물려 **더 자랄 수 없다**. 관측 기울기는
  `λ−μ`가 아니라 `≈0 + 잡음`으로 절단되고, 워커 재시작 직후의 일시적 배출이 겹치면 **음수 기울기**가
  찍힌다. → 오검증(false commit).
- 동시에 `V(t)`도 **생존편향**된다. `ℓ_i`는 **완료된 프레임**의 Δ-창 평균인데, 드롭된 프레임은 영원히
  완료되지 않는다. 느린 프레임이 선택적으로 사라지므로 `ℓ_i`는 빠른 생존자만 평균하여 **낮게** 나온다.

즉 포화 상태에서는 **slope와 V 둘 다 신뢰할 수 없다**.

### 1.2 신호 — (a) 드롭 지속만 사용 (threshold-free)
승인된 설계대로 **(b) 큐 점유율(queue/cap ≥ 0.9) 신호는 채택하지 않았다.** 큐가 가득 찬 상태에서
정상 배출을 시작한 candidate("full queue에서 배출 중")를 오기각하기 때문이다.

채택 신호는 **(a) 드롭 지속** 하나뿐이다:

> `T_v`의 **마지막 1/3 구간**(=1.0s)에서 뷰 `i`의 누적 드롭이 **1개라도 증가**하면 뷰 `i`는 포화.

- **임계값 없음**: `Δdrop > 0` 자체가 "그 순간 큐가 cap에 닿아 있었다 ⇒ 도착 > 서비스"의 **직접 증거**다.
  튜닝 파라미터를 새로 도입하지 않는다.
- **뷰별 판정**: 한 뷰라도 포화면 그 배치는 그 뷰의 SLO를 지속적으로 깨고 있으므로 기각. (V가 4뷰
  평균이라 단일 뷰 포화를 희석하는 문제를 우회한다.)
- **마지막 1/3 창**: 전환 직후 전이 구간(inherited backlog 소각)의 드롭을 배제하고 **정착된 꼬리**만 본다.
  기존 slope 계산의 tail 창과 동일 구간이라 일관적이다.

### 1.3 판정 순서 — 포화 가드가 `V ≤ ε`보다 **먼저**
승인된 정정 사항. `V`가 생존편향으로 **낮게 나올 수 있으므로** `V ≤ ε`를 먼저 평가하면 포화 candidate가
"검증 통과"로 커밋된다. 실제 Q4에서 이 경로가 관측되었다(§4, cand_4: `V_postswap=1.198`인데 3뷰 모두
드롭 지속). 최종 순서:

```
(0) saturated_views 비어있지 않음  -> advance  ("saturated advance")
(1) V(t) <= eps                    -> commit-and-stay
(2) slope < -theta                 -> commit-and-stay
(3) 그 외                          -> advance
```

## §2. 구현

`schedule_executor_main.py`:

- `_current_drops()` (L646) — 두 피더(`video_feeder`, `resnet_feeder`)의 `drop_counts`를 뷰별로 합산한
  누적 드롭 `{view: n}`을 반환.
- validate 폴링 루프가 `drop_samples.append((elapsed, self._current_drops()))`로 `bl_samples`와 동일
  cadence(200ms)로 수집 (L824).
- 판정 시점(L851–858):
  ```python
  tail_start = self.qos_tv * (2.0 / 3.0)
  tail_drops = [(t, d) for (t, d) in drop_samples if t >= tail_start]
  drop_delta = {_vw: 0 for _vw in ("view1","view2","view3","view4")}
  if len(tail_drops) >= 2:
      d0 = tail_drops[0][1]; d1 = tail_drops[-1][1]
      for _vw in drop_delta:
          drop_delta[_vw] = int(d1.get(_vw,0)) - int(d0.get(_vw,0))
  saturated_views = [_vw for _vw, dd in drop_delta.items() if dd > 0]
  ```
- 판정 분기(L881–): `if saturated_views: self._qos_advance(...)` 를 `V<=eps` 분기 **앞**에 배치.
- **P7 로깅**: 모든 validate 라인에 `tail_drops={viewN:Δd,...}` 또는 `{none}` 을 기록하고, 기각 시
  `saturated: views [...] still dropping in the last 1.0s (queue clipped at cap -> arrival>service);
  slope/V unreliable, advancing` 을 출력.

**기본 동작 보존**: 드롭이 없으면 `saturated_views`가 비어 종전 판정 경로와 완전히 동일하다. 새 CLI
플래그·환경변수 없음.

## §3. 재검증 — 정상 커밋을 막지 않는가

### 3.1 §4b (well-predicted, top-1이 실제로 feasible)
| 기법 | GPU persist / lastV | NPU persist / lastV |
|---|---|---|
| Static | 60s / 9.02 | 59s / 9.09 |
| Adaptive | 10s / 0.00 | 12s / 0.00 |
| **BoundGuard** | **11s / 0.00** | **12s / 0.00** |

판정 경로: `cand_1 ... backlog_slope=0.00/s tail_drops={none}` → 포화 아님 → `V_postswap=0.000 ≤ ε`
→ **commit-and-stay**. **가드가 정상 커밋을 차단하지 않음**을 확인. BoundGuard ≈ Adaptive 유지 →
**Q1.5 "no penalty" 불변**.

### 3.2 Q3 (mispredicted, vision-3 + LLM, λ=80, buffer=12)
> 아래 수치는 [hotswap_buffer_bias_fix.md](hotswap_buffer_bias_fix.md)의 큐 깊이 편향을 수정한 뒤
> 재실행한 것이다(전 기법 큐 깊이 12로 동일).

| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static | 3.84 | 3.51 | 105s | ✗ |
| Stop-restart | 3.67 | 3.38 | 106s | ✗ |
| Adaptive | 3.88 | 3.42 | 106s | ✗ |
| **BoundGuard** | 61.33 | **0.00** | **20s** | **✓** |

판정 경로 (P7 로그 전문):
```
cand_1  placement identical to the current (violating) placement -> auto-advance
cand_2  V=3.691   slope=+1.14/s  tail_drops={view2:4}   -> saturated advance
cand_3  V=18.903  slope=-2.14/s  tail_drops={view3:62}  -> saturated advance   (*)
cand_4  V=66.090  slope= 0.00/s  tail_drops={view2:76}  -> saturated advance
cand_5  (cggg = LLM->CPU, terminal) -> V 61.33 -> 0.00, 회복
```
(*) **cand_3이 가드 없이는 오커밋되는 유일한 케이스다.** `slope=−2.14/s`로 "배출 중"으로 보이지만
view3이 마지막 1초에 62프레임을 버리는 중 — 큐가 cap에 물려 기울기가 `λ−μ`를 추정하지 못한다.
cand_2(+1.14)·cand_4(0.00)는 slope 검사만으로도 기각되므로, 이 시나리오에서 **가드의 순증 기여는
cand_3 한 건**이다. 그 한 건이 없으면 탐색이 거기서 멈춰 진짜 feasible한 cand_5에 도달하지 못한다.

**결론**: 가드는 (i) 드롭이 없는 진짜 feasible 후보(§4b cand_1)를 통과시키고, (ii) 드롭이 지속되는
포화 후보(Q3 cand_3/cand_4)를 기각한다. 두 방향 모두 의도대로 동작.

## §4. Q4 적용 결과 (요약; 상세는 [q4_experiment_report.md](q4_experiment_report.md))
결정적 과부하(heavy-4, λ=90/뷰, buffer=300)에서 **전 후보가 사전측정상 명백히 infeasible**(V 365~688).
BoundGuard 판정 경로 (큐 깊이 편향 수정 후):

| cand | 배치 | V_postswap | slope | tail_drops | 판정 |
|---|---|---|---|---|---|
| 1 | gggg | — | — | — | 현재(위반) 배치와 동일 → auto-advance |
| 2 | cggg | 279.121 | +89.69/s | view1:3, view2:17, view3:15 | **saturated advance** |
| 3 | ggcg | 203.105 | +120.54/s | view2:31 | **saturated advance** |
| 4 | gcgg | 126.151 | +135.25/s | view1:6 | **saturated advance** |
| 5 | gggc | (terminal, trigger 미부여) | | | — |

전 후보가 정당하게 기각됐다. 다만 **이 시나리오에서 가드의 순증 기여는 없다**: 버퍼가 정상 깊이라
큐가 실제로 쌓일 수 있어 slope가 +90~+135/s로 강하게 양수이고, slope 검사만으로도 세 후보 모두
기각된다.

### 4.1 §1.3(판정 순서) 정정의 근거는 유지된다
큐 깊이 편향 **수정 전** 같은 실험에서는 다음이 관측됐다:

```
cand_4  V_postswap=1.198  slope=+0.29/s  tail_drops={view1:32,view2:88,view3:25}
```

`V=1.198`은 ε=1.0에 근접한 낮은 값이지만(같은 배치의 사전측정 V는 445) 세 뷰 모두 초당 수십 프레임을
버리는 중이었다 — 생존편향으로 V가 낮아진 전형이다. 포화 가드를 `V≤ε` **뒤**에 놓았다면 여기서
오커밋했을 것이다. 즉 **순서 정정의 필요성은 실측으로 입증됐고**, 그 병리(얕은 큐 → 드롭 → 생존편향)는
버퍼가 작을 때 언제든 재현된다. Q3(buffer=12)의 cand_3이 수정 후에도 남아 있는 실증 사례다(§3.2).

### 4.2 가드의 적용 범위 — 정직한 한정
- **큐가 클리핑되는 레짐**(작은 버퍼, 또는 λ가 μ를 크게 초과해 즉시 cap 도달): slope와 V가 모두
  무력화된다 → **가드가 필수**. 실증: Q3 cand_3.
- **큐가 충분히 깊은 레짐**: slope가 `λ−μ`를 제대로 추정하므로 slope 검사가 이미 충분하고, 가드는
  같은 결론을 내리는 **중복 안전망**이다. 실증: Q4 cand_2/3/4.

가드는 후자에서 오탐(정상 커밋 차단)을 만들지 않는다 — §3.1 §4b에서 `tail_drops={none}` → 정상 커밋 확인.

## 무결성
- P1(실행 순서 `sort_keys=False`), P2(mode 1 + validate trigger = BoundGuard), P3(GPU fallback 0),
  P7(판정 경로 전량 로깅), P9(commit-and-stay) 확인.
- 변경 파일: `schedule_executor_main.py` 단일. legacy/scripts/backup·main.tex 불변.
- 관련 문서: [trend_validate_report.md](trend_validate_report.md),
  [vt_definition_fix_report.md](vt_definition_fix_report.md),
  [q4_experiment_report.md](q4_experiment_report.md).
