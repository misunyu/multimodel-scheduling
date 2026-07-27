# Q4 — 결정적 과부하 레짐에서의 유계 탐색

날짜: 2026-07-24 · 시나리오: heavy-4 vision, λ=90 fps/뷰, buffer=300 · 파라미터 불변(N_cand=5, α=0.3,
β=0.5, ε=1.0, T=3s, T_v=3s, Δ=0.2s, θ=0.5) · 그림: `docs/figures/q4_bounded_envelope.pdf`

> **판정 요약**: Q4의 다섯 성립 조건 중 **(i)(ii)(iii)(v)는 충족, (iv)는 불충족**이다. BoundGuard의
> V(t) envelope은 baseline보다 **낮지 않고 오히려 약 2배 높다**. 억지로 성립시키지 않고 그대로 보고한다.

## §1. 설계 의도
Q3(오예측 → BoundGuard만 회복)와 달리 Q4는 **회복이 불가능한 레짐**을 다룬다. 예측기 top-5 후보가
**모두** infeasible할 때, BoundGuard의 탐색이 `T + N_cand(T_v + δ)`로 **유계**임을 보이는 것이 목적이다.
회복(recovery)이 아니라 **유계(boundedness)** 가 주장이며, Q3와 혼동해서는 안 된다.

**P8(경계 회피)** 이 핵심 조건이다: 후보들이 "아슬아슬하게" infeasible하면 검증 판정이 잡음에
좌우되므로, **명백하게** infeasible해야 한다.

## §2. 사전측정 (P8 검증) — 각 후보를 독립 실행

`--adaptive-mode 3`(Static)로 다섯 후보를 각각 17s 단독 실행. λ=90/뷰, buffer=300 동일.

| 후보 | 배치 (y11x, y11l, y11m, rn50) | V_last | V_max | backlog_last | 판정 |
|---|---|---|---|---|---|
| cand1 | g g g g | 410.52 | 410.52 | 899 | infeasible |
| cand2 | **c** g g g | 364.59 | 456.28 | 897 | infeasible |
| cand3 | g g **c** g | 416.51 | 919.88 | 900 | infeasible |
| cand4 | g **c** g g | 444.96 | 932.33 | 897 | infeasible |
| cand5 | g g g **c** | **687.63** | 995.01 | 1194 | infeasible (**최악**) |

- 전 후보 `V` 365–688 (ε=1.0 대비 **2–3 자릿수**). 경계 사례 없음 → **P8 충족**.
- 모든 후보가 backlog cap(≈900/1200)에 도달 → 어느 배치로도 λ를 흡수할 수 없다.
- **cand_5(gggc)가 다섯 중 가장 나쁘다**. resnet50을 CPU로 보내면 view4까지 포화되어 4뷰 전부 cap에 닿는다.

모델: yolo11x(L_SLO 70ms) · yolo11l(52) · yolo11m(44) · resnet50(12).

## §3. 4기법 결과

실행 순서 `combination_stable(10s) → combination_burst(10s) → cand_*`. Static은 burst 유지,
Stop-restart/Adaptive는 cand_1 유지, BoundGuard만 cand_1..5 순회(각 8s, cand_5는 tail 40s).

| 기법 | maxV | lastV | persist(V>ε) | 회복 |
|---|---|---|---|---|
| Static | 383.96 | 381.71 | 89s | ✗ |
| Stop-restart | 375.08 | 367.88 | 89s | ✗ |
| Adaptive | 374.19 | 366.63 | 89s | ✗ |
| **BoundGuard** | **739.59** | **735.82** | 61s | ✗ |

- **(ii) 전 기법 위반 진입 충족** — 최저값도 374로 ε=1.0의 374배.
- `persist` 61s vs 89s는 **BoundGuard의 총 실행시간이 짧아서**(탐색으로 조기 advance) 생긴 것이며
  우위가 아니다. 같은 시점 비교는 `lastV`(735.8 vs 366.6~381.7)를 봐야 한다.

> **주의**: 이 수치는 [hotswap_buffer_bias_fix.md](hotswap_buffer_bias_fix.md)의 큐 깊이 편향을 수정한
> 뒤의 것이다. 수정 전에는 BoundGuard만 hot-swap 후 큐 깊이가 300→2로 떨어져 `maxV=275.4 / lastV=5.0`
> 으로 **유리하게** 보였으나, 그것은 전적으로 얕은 큐 아티팩트였다.

## §4. BoundGuard 후보별 판정 경로 (P7)

```
cand_1  placement identical to the current (violating) placement -> auto-advance
cand_2  V_postswap=279.121  slope= +89.69/s  tail_drops={view1:3,view2:17,view3:15}  -> saturated advance
cand_3  V_postswap=203.105  slope=+120.54/s  tail_drops={view2:31}                   -> saturated advance
cand_4  V_postswap=126.151  slope=+135.25/s  tail_drops={view1:6}                    -> saturated advance
cand_5  (terminal, C1 규약: validate trigger 미부여 → tail 연장관찰)
```

- **(i) 전 후보 검증 실패 충족**: cand_1은 현재 위반 배치와 동일하여 auto-advance(전이 없음), cand_2~4는
  포화로 정당하게 기각. **오커밋 0건.**
- slope가 +90~+135/s로 강한 양수 — 큐가 정상 깊이라 backlog가 실제로 발산 중임을 직접 보여준다.
  이 레짐에서는 포화 가드 없이 slope 검사만으로도 동일 결론이 난다(→
  [saturation_guard_report.md](saturation_guard_report.md) §4.2).

### 4.1 Bound 대조 — **(iii) 충족**

CSV 타임스탬프 기준(overload onset = burst 시작 = t≈0):

| 구간 | 시각 |
|---|---|
| 첫 `V>ε` 샘플 | +5.0s |
| burst 종료 → cand_1 진입 | +14.0s |
| cand_2 / cand_3 / cand_4 진입 | +15.0 / +19.0 / +23.0s |
| **cand_5(마지막 후보) 적용** | **+26.0s** |

- **탐색 구간**(cand_1 진입 → cand_5 적용) = **12.0s**.
- 후보당 실측 벽시계 ≈ 4s = `T_v`(실측 elapsed 3.0/3.2/3.1s) + `δ`(vision hot-swap ≈1s).
- **bound**: `N_cand(T_v + δ) = 5 × (3 + 1) = 20s` → **12.0s ≤ 20s ✓**.
- 감지항까지 포함해도 `T + N_cand(T_v+δ) = 3 + 20 = 23s`. (본 하네스에서 burst→cand_1 전이는 트리거가
  아니라 **고정 10s 스케줄**이며 이는 네 기법에 동일하게 적용된다. 따라서 "첫 위반 → cand_5" 21.0s
  중 10s는 하네스 상수이지 알고리즘의 반응 지연이 아니다.)
- 이 수치는 큐 깊이 수정 전후로 **완전히 동일**했다 — 탐색 시간은 버퍼와 무관하다.

## §5. 판정

| 조건 | 결과 |
|---|---|
| (i) 전 후보 검증 실패(포화로 정당하게 기각) | ✅ |
| (ii) 전 기법 위반 진입 | ✅ (374–740) |
| (iii) 탐색 persistence가 bound 내 | ✅ (12.0s ≤ 20s) |
| (iv) BoundGuard envelope이 baseline 대비 낮음 | ❌ **739.6 vs 374–384** |
| (v) P8 (전 후보 명백히 infeasible) | ✅ (사전측정 V 365–688) |

### 5.1 탐색 유계 (성립) vs tail (C1 규약의 귀결) — 구분

두 가지를 반드시 분리해서 읽어야 한다.

**(a) 탐색 유계 — 성립.** BoundGuard는 12.0s 안에 다섯 후보를 모두 소진했고, 각 후보를 `T_v` 관찰 후
포화 근거로 기각했다. 무한 순회도, 오커밋도 없었다. `T + N_cand(T_v+δ)` bound가 실측으로 지켜졌다.
이것이 Q4가 보이려던 주장이며 **성립한다**.

**(b) tail의 V=735.8 — bound 위반이 아니라 C1 규약의 귀결.** 하네스는 C1 결정
(`cand_tail = 마지막 후보 연장관찰`, [final_reverification_report.md](final_reverification_report.md))
에 따라 마지막 후보에 validate trigger를 부여하지 않고 잔여 시간을 흡수시킨다. 따라서 **모든 후보가
실패한 경우 실행은 정의상 cand_5 위에서 끝난다**. cand_5는 사전측정에서 다섯 중 최악(V=688)이었고,
실측 735.8이 이를 재현했다. 즉 tail 수치는 **탐색 알고리즘의 성능이 아니라 종료 규약의 산물**이다.

### 5.2 (iv) 불충족이 드러낸 것 — 실제 설계 공백

그럼에도 (iv) 불충족을 "규약 탓"으로 덮을 수는 없다. 드러난 사실은 명확하다:

> **bound는 탐색 *시간*을 보장하지만 종료 시점의 *배치 품질*은 보장하지 않는다.**
> 전 후보가 infeasible할 때 원배치(또는 관측된 최선)로 복귀하는 fallback 단계가 알고리즘에 없다.

이 시나리오에서 그 공백의 대가는 구체적이다. 아무것도 하지 않은 Static이 384인데, 탐색을 끝낸
BoundGuard는 736이다 — **탐색이 상황을 2배 악화시켰다**. 다섯 후보가 모두 실패한 뒤 원배치(gggg,
사전측정 410)로 복귀했다면 baseline 수준을 유지했을 것이다.

**따라서 다음 중 하나가 필요하다** (파라미터·알고리즘 변경이므로 결정은 별건):
1. 전 후보 실패 시 **원배치 복귀**(rollback) — mode 4가 이미 유사 경로를 가짐.
2. 전 후보 실패 시 **관측된 최선 후보로 복귀** — 탐색 중 `V_postswap`을 기록해 두면 추가 비용 없음.
3. C1 규약을 바꿔 마지막 후보도 검증 대상에 포함하고, 실패 시 1 또는 2를 적용.

현 구현·현 규약 하에서 Q4는 **"탐색은 유계이나 종료 배치는 무보장"** 으로 보고한다.

## §6. 무결성
- P1 실행 순서(`sort_keys=False`, stable→burst→cand_1..5) 로그 대조 ✓
- P2 mode: Static=3, Stop-restart=0, Adaptive=1, BoundGuard=1+validate trigger ✓
- P3 GPU fallback=0 ✓ · P4 큐 깊이 전 기법 동일(`viewN_q`≈300) ✓
- P5 서비스율: λ=90/뷰 `FSRR_RATE_REPLICATE=1`로 강제, 전 기법 동일 ✓
- P7 판정 경로·`V_postswap`·`tail_drops` 전량 로깅 ✓ · P8 §2 사전측정 ✓
- P9 commit-and-stay: 본 실험에서는 커밋이 발생하지 않음(전 후보 기각) — 해당 없음
- 관련: [saturation_guard_report.md](saturation_guard_report.md),
  [hotswap_buffer_bias_fix.md](hotswap_buffer_bias_fix.md),
  [q3_experiment_report.md](q3_experiment_report.md)
