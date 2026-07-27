# best-so-far 폴백 — 후보 예산 소진 시 관측 최선 배치로 복귀

날짜: 2026-07-24 · 알고리즘 **추가**(버그 수정 아님) · 스냅샷 `backup/best_so_far_20260724_150133/` · main.tex 불변
관련: [q4_experiment_report.md](q4_experiment_report.md), [saturation_guard_report.md](saturation_guard_report.md), `PENDING_DECISIONS.md [Q4-1]`

> **성격**: 논문 §III에 서술할 **알고리즘 추가**다. 탐색 로직(후보 열거·순서·N_cand·판정)은 불변이고,
> "예산 소진 시 관측 최선으로 복귀" 한 단계만 더해진다. 이 변경은 결과를 개선하는 방향이지만,
> **정당성은 "결과가 좋아서"가 아니라 "마지막-랭킹 후보에 머물 이유가 없어서"** 다(§1).

## §1. "best"의 기준 (게이트 확정)

### 1.1 왜 관측 V가 아닌가 — 상속 backlog 편향
순진한 기준(관측 V 최소)은 **초기 후보 쪽으로 편향**된다. cand_1은 burst backlog만 안고 평가되지만
cand_5는 **burst + 실패한 4개가 쌓은 backlog**를 안고 평가된다. 나중 후보일수록 V가 나쁘게 나오는데
이는 배치 품질이 아니라 상속 backlog 탓이다. → V는 기준으로 부적합.

### 1.2 채택: (b) 실측 서비스율
| 기준 | 장점 | 단점 | 채택 |
|---|---|---|---|
| (a) backlog 기울기 | λ−μ 추정, backlog 수준에 무오염 | **포화 시 큐가 cap에 clip → 무의미**. Q4가 정확히 포화 레짐 | ✗ |
| **(b) 서비스율**(완료 프레임/초) | μ_i 직접 추정, **포화에서도 유효** | — | **✓** |

서비스율 = 검증 창 동안 **완료된 프레임 수 / 시간**. 포화 상태에서도 "누가 더 많이 처리하는가"를
직접 재므로 Q4에서 유일하게 후보를 구별한다. 핸들러의 `infer_count`(누적 완료) 델타로 계산 — 신규 계측 없음.

### 1.3 세부 결정 (기존 로직과의 일관성)
- **측정 구간**: T_v **후반 1/3** (slope·drop 판정과 동일 — 전환 transient 배제).
- **기록 범위**: **검증된 후보만**. auto-advance된 후보(= 현재 위반 배치 그 자체)는 **기록하지 않는다** —
  그리로 복귀하면 알려진 최악으로 회귀하기 때문. 이 배제가 "복귀가 상황을 악화시키지 않음"을 구조적으로 보장.
- **동률·측정 실패**: 관측이 없으면(예: 완료 프레임 0) 현재 배치 유지(전환 없음).
- **터미널 후보도 검증**: 논문 알고리즘대로 **N_cand 전부** validate 트리거를 받는다. 마지막이 실패하면
  복귀. 기존의 "마지막은 무조건 머문다"가 C1 편법이었고, 이를 제거했다.

## §2. 구현

`schedule_executor_main.py` (스냅샷 후):
- `_total_infer_count()` — foreground 뷰 핸들러의 `infer_count` 합. 델타 = 집계 서비스율.
- `check_validate`: 매 poll에 `count_samples`(elapsed, count) 수집 → 판정 시 tail 1/3 구간 기울기로
  `service_rate` 계산 → **모든 검증 후보를 `self._cand_obs[combo]={rate,sig}`에 기록**(커밋 여부 무관).
  로그에 `service_rate=…fps` 추가(P7).
- 실패 분기(포화 / 미배출)에서 `is_last_cand`면 `_qos_advance` 대신 **`_revert_to_best(combo)`**.
- `_revert_to_best`: `_cand_obs`에서 서비스율 최대 후보 선택 → 현재 배치와 같으면 그대로 hold(δ 절약),
  다르면 **adaptive hot-swap(backlog 보존)으로 전환** 후 hold. 로그
  `exhausted -> revert to best (cand=…, service_rate=…)` + 후보별 관측값.
- `start()`에 `self._cand_obs={}` 초기화(에피소드 단위).
- 하네스(`q4_scenario.py`): cand_5에도 validate 트리거 부여(전 후보 검증).

**검증(스모크)**: 2-후보 전부 포화 시나리오에서 `revert to best (cand=cand_2, 92.1fps > cand_3 71.0fps)`
확인, 복귀 전환이 **backlog 보존 hot-swap**(`moved=299 dropped=0`)으로 수행됨 확인.

## §3. bound 갱신

복귀도 **전환이므로 δ가 한 번 더** 든다. 탐색 상한:
$$ T + N_{cand}(T_v + \delta) + \delta $$
(복귀가 현재 배치와 동일하면 마지막 δ는 0.) Q4 실측 persistence를 이 값과 대조(§4).

## §4. 재실행

### 4a. Q4 (핵심) — 그림 `docs/figures/q4_bounded_envelope.pdf`

> **주의(스케일 변화)**: 이번 Q4의 V(t) 절대값이 이전(≈370)보다 낮은 ≈75다. 이는 revert 효과가 아니라
> **직전 작업의 `slo_ms` 수정이 Q4에 전파**된 것이다. Q4 시나리오는 per-model L_SLO(yolo11x=70·11l=52·
> 11m=44·resnet50=12ms)를 설정하는데, 예전엔 명명뷰 경로가 이를 버리고 폴백(1000/90≈11ms)을 썼다. 이제
> **올바른 deadline**이 적용돼 V가 낮아졌다. 네 기법이 **동일한 새 deadline**을 쓰므로 상호 비교는 유효하다.

| 기법 | maxV | lastV | persist | (이전 lastV, 최악착지) |
|---|---|---|---|---|
| Static | 74.9 | 73.9 | 89s | 367.1 |
| Stop-restart | 77.8 | 75.8 | 88s | 373.7 |
| Adaptive | 77.8 | 75.9 | 89s | 378.9 |
| **BoundGuard** | 146.1 | **55.0** | 60s | **743.1** |

**후보별 판정 경로 (P7)** — 전 후보 validate·서비스율 기록:
```
cand_1  auto-advance (violating placement, 미기록)
cand_2  service_rate=237.3fps  saturated → advance
cand_3  service_rate=202.9fps  saturated → advance
cand_4  service_rate=215.8fps  saturated → advance
cand_5  service_rate=186.2fps  saturated, LAST → revert-to-best
exhausted → revert to best (cand_2, 237.3fps); obs {c2:237.3, c3:202.9, c4:215.8, c5:186.2}
reverted to 'cand_2' (backlog 보존 hot-swap), hold 37s
```
- **서비스율 랭킹이 해석 가능**: cand_2(cggg, **최중량 yolo11x를 CPU로**)가 237fps로 최고 — GPU에서 가장
  무거운 모델을 빼 나머지 3개가 빨라진다. cand_5(gggc, **경량 resnet50을 CPU로**)가 186fps로 최저 — 예측기가
  **마지막에 랭크한 cand_5가 실제로 최악**이었고, 복귀가 정확히 이를 회피했다.
- **결과**: BoundGuard가 cand_5(→135+로 향하던)에서 관측 최선 cand_2로 복귀, V 144→**55로 배출**.
  **세 baseline(74–76) 아래**로 내려와 이제 네 기법 중 최저다(이전엔 743 vs 367로 2배 열위).

**bound 대조 (§3 갱신형)**:
- 첫 위반(+6s) → 복귀 적용(+30s) = **24s**. 탐색 구간(cand_1 진입 +14s → 복귀 +30s) = 16s.
- bound = `T + N_cand(T_v+δ) + δ = 3 + 5·(3+1) + 1 = 24s`. **실측 24s ≤ bound 24s ✓**
  (하네스 고정 10s burst→cand_1 전이 포함, 이전 Q4와 동일 회계).

**Q4 성립 조건 판정**:
| 조건 | 결과 |
|---|---|
| (i) 전 후보 검증 실패(포화로 정당 기각) | ✅ cand_2~5 전부 saturated |
| (ii) 전 기법 위반 진입 | ✅ 74.9–146.1 |
| (iii) 탐색 유계 | ✅ 24s ≤ bound 24s |
| **(iv) BoundGuard envelope이 baseline 대비 열위 아님** | ✅ **55.0 < 74–76** (이전 ❌ 743 vs 367) |
| (v) P8 (전 후보 명백히 infeasible) | ✅ 사전측정 유지 |

**(iv)가 이제 충족**된다. 단, V=55 ≫ ε=1이므로 **회복이 아니라 유계**다(Q4의 본래 주장). 이전 리포트의
"탐색 유계이나 종료 배치 무보장"에서 **"종료 배치도 관측 최선으로 보장"**으로 갱신됨.

### 4b. §4b · Q3 무회귀 — **통과 (revert 미발동 = 소진 경로 미도달)**

복귀는 **후보를 전부 소진했을 때만** 발동한다. 두 시나리오 모두 그 전에 commit되므로 무영향이어야 하고,
실측이 그렇다(`grep "revert to best"` = 0회).

**§4b** (cand_1에서 commit):
| 기법 | GPU persist/lastV | NPU persist/lastV |
|---|---|---|
| Static | 60s / 9.2 | 59s / 9.1 |
| Adaptive | 12s / 0.0 | 12s / 0.0 |
| **BoundGuard** | **11s / 0.0** | **12s / 0.0** |

BoundGuard = Adaptive 유지 → **Q1.5 "no penalty" 불변**. revert 0회.

**Q3** (cand_5에서 commit):
| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static/Stop-restart/Adaptive | 3.04–3.13 | 2.65–2.77 | 105s | ✗ |
| **BoundGuard** | 50.6 | **0.00** | **22s** | **✓** |

BoundGuard 회복 유지(이전 21s→22s). revert 0회. 판정 경로: cand_2/3/4 검증·기록(서비스율 238/181/164fps)
후 cand_5(LLM→CPU) feasible → commit-and-stay. **모든 후보가 검증되므로 cand_2~4도 이제 서비스율이
기록**되지만, cand_5가 commit되어 복귀는 불필요.

### 4c. C3 무영향
C3는 mode 0(hot-swap·validate 미사용)이라 복귀 경로와 **구조적으로 무관**. 재실행 불요.

## §5. 개선분의 출처

BoundGuard가 Q4에서 나아진 것(이전 743 → 55, baseline 대비 열위→우위)의 출처는 **단 하나**:
**예측기가 마지막에 랭크한 최악 후보(cand_5=gggc)에 default-commit하던 것을, 관측된 최선(cand_2=cggg)으로
복귀**하도록 바꾼 것이다.

- 스케줄링 판정 로직·후보·파라미터는 불변이다. 탐색은 이전과 동일하게 cand_1~5를 순회한다.
- 바뀐 것은 **종료 지점**뿐이다: cand_5(186fps)에 머무는 대신 cand_2(237fps)로 되돌아간다.
- 서비스율 랭킹(237/216/203/186fps)은 예측기 랭킹과 **역순에 가깝다** — 예측기가 cand_5를 top-5의
  꼴찌로 놓았고 실제 최악이었으므로, "마지막 후보에 머문다"가 특히 나빴다. 복귀가 이를 교정한다.
- **이 개선은 "결과를 좋게 하려는 튜닝"이 아니다**: §1대로 "마지막-랭킹 후보에 머물 이유가 없다"는
  원칙에서 나온 것이고, 실제로 유리하게 작용했을 뿐이다. Q3·§4b에서는 commit이 먼저 일어나 이 단계가
  발동하지 않으므로, 결과를 바꾸지 않는다(무회귀).

## 무결성
- 탐색 로직·파라미터 불변(N_cand·α·β·ε·T·T_v·Δ·θ·추세 판정·포화 가드·auto-advance·commit-and-stay·
  backlog 보존). **추가된 것은 복귀 단계뿐.**
- 변경: `schedule_executor_main.py`. 하네스 `q4_scenario.py`(cand_5 트리거). legacy/backup·main.tex 불변.
- P1~P9 확인(§4에서).
