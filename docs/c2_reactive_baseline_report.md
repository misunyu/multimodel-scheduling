# C2 — reactive baseline (BoundGuard의 ablation 비교)

날짜: 2026-07-25 · 스냅샷 `backup/schedule_executor_main.pre_c2.py` · main.tex 불변
관련: §II/§III(A2) "기여는 mechanism이 아니라 property" 주장의 실험적 뒷받침

> ⚠️ **[2026-07-25 정정]** 이 리포트의 수치 일부는 무결성 감사에서 정정됐다 →
> [c2_integrity_report.md](c2_integrity_report.md). 요지: (1) **§4b config 오류**(burst=all-accel로
> 위반 없음)로 이전 §4b 수치 무효 → burst=all-CPU로 재실행(A 1/5·hs 13, 나머지 5/5). (2) **B(re-invoke)의
> persist 63은 censored 런길이 아티팩트** — B는 결정적 예측기 하에서 **Adaptive baseline과 동작 동등**(둘 다
> top-1 고착). (3) 런 길이 120s 고정. **질적 2×2 결론은 정정 재실행으로 전부 재확인**됨. 아래 본문의
> 원 수치는 참고용, 정본 수치는 무결성 리포트 참조.

> **설계**: strawman(별도 naive 시스템)을 피하고 **BoundGuard의 2×2 요인 ablation**으로 구성. 두 인자 —
> **dwell**(검증 창 T_v)과 **progress**(후보 순위 하강) — 를 각각 제거해 네 모서리를 만든다.

## §0. 2×2 요인 설계

| | **progress 유지** (순위 하강) | **progress 제거** (top-1 재적용) |
|---|---|---|
| **dwell 유지** (validate) | **BoundGuard** | **C. hybrid** |
| **dwell 제거** (v-above) | **A. no-dwell** | **B. re-invoke** |

- **A ↔ BoundGuard** = dwell의 효과 · **C ↔ BoundGuard** = progress의 효과 · **B** = 둘 다 없는 모서리.
- **각 ablation = 단일 변경점** (코드/설정 한 줄):
  - **dwell**: 트리거 정책 `validate`(T_v 관찰 후 판정) ↔ `v-above`(V>ε 즉시 전환, 관찰 없음). 검출 창
    T·ε·Δ는 불변 — **T_v(dwell)만** 제거(지시문 §1의 "공정" 정의).
  - **progress**: 스케줄이 `[cand_1..cand_5]`(예측기 top-5 순위) ↔ `[top-1 ×5]`(재호출 결과 반복).
- **나머지 전부 BoundGuard와 동일**: 검출 T·ε·Δ·전환 메커니즘·backlog 보존·**best-so-far 복귀·N_cand=5
  예산**(보완 1·2). B/C도 5회 예산 소진 후 best-so-far 복귀 → 무한 루프 아님, 차이는 "순위 하강 vs 재적용"
  단 하나.
  - *구현 주: hybrid(C)는 auto-advance 가드(현재 배치와 동일한 후보 스킵)를 끈다(`FSRR_NO_AUTOADVANCE`).
    이 가드 자체가 progress 인식 메커니즘이므로 progress 제거의 일부다. BoundGuard/A/B는 불변(가드 유지).*

## §1. 예측기 결정성 (보완 3) — **확인: 정적 순수함수**

`featurize_from_combo(S, combo_blob)`(`xgboost_model/deploy_selector_xgb_suite.py:363`)는 **정적 per-device
프로파일 S + 워크로드 구성(model·device·infps)만** 사용한다. `_view_features(m,dev,infps,S)`에 **런타임
상태(V(t)·backlog·현재 지연)가 없다.** 경험 확인: 동일 워크로드 2회 랭킹 **완전 동일**.

→ **예측기는 워크로드 구성의 결정적 순수함수.** 재호출은 조건 불변 시 **항상 같은 top-1**을 준다. 이것이
B/C 실패의 **정확한 근거**이자, B/C를 "top-1 ×5 반복"으로 충실히 구현할 수 있는 이유다(런타임 재호출 불필요).
**A2에 실험 없이도 넣을 수 있는 문장**: *"정책을 다시 호출하는 것만으로 안 되는 이유는 정책이 워크로드에
대해 결정적이어서 같은 입력에 같은 답을 주기 때문이다."*

## §2. 시나리오
| 시나리오 | 목적 | 조건(기존 그대로) |
|---|---|---|
| **Q3** (CPU–GPU 오예측) | 핵심 — 메커니즘 차이는 오예측에서만 발현 | llama1b bg + vision3, λ=80, buf=12 |
| **§4b** (예측 정합) | 대조군 — 전부 동일 회복해야 정상 | vision3, λ=45, buf=2 |
| **Q4** (가용 후보 없음) | 종료성 | heavy-4, λ=90, buf=300 |

각 5회 반복, warmup(P4), 이상치 미폐기.

## §3. 결과

### 3.1 Q3 (CPU–GPU 오예측) — 핵심, 5회

| 변형 (2×2) | 회복 n/5 | persist (±SD) | lastV (±SD) | hotswaps (±SD) |
|---|---|---|---|---|
| **BoundGuard** (dwell·progress) | **5/5** | 18.8 ± 0.4 | 0.0 ± 0.0 | 7.0 ± 0.0 |
| **A. no-dwell** (progress만) | **3/5** | 31.2 ± **27.2** | 9.9 ± **12.1** | 8.8 ± 0.4 |
| **B. re-invoke** (둘 다 없음) | **0/5** | 63.0 ± 0.6 | 1.2 ± 0.0 | 0 |
| **C. hybrid** (dwell만) | **0/5** | 79.2 ± 8.5 | 1.2 ± 0.0 | 0 |

- **progress 제거(B·C) → 0/5 결정적 실패.** 예측기 재호출이 매번 같은 top-1(=위반 배치)을 주므로 대안을
  탐색하지 못하고 top-1에 고착(lastV 1.2). B는 no-dwell로 top-1을 계속 재적용(hotswaps 0 = top-1을 떠나지
  않음), C는 top-1에 **5×T_v=15s를 dwell 낭비**한 뒤 top-1로 복귀. **둘 다 회복 불가.**
- **dwell 제거(A) → 3/5, 거대한 분산**(persist SD 27.2, lastV SD 12.1). dwell이 없어 각 후보의 서비스율이
  **전환 직후 transient**로만 측정되어 무의미 → best-so-far가 **코인플립**으로 좋은 배치를 고르거나(회복)
  쓰레기를 고른다(**실패 시 lastV 24.7 — 아무것도 안 한 것보다 나쁨**). hotswaps 8.8 > BoundGuard 7 (더 churn).
- **BoundGuard만 5/5 + tight SD** — property(신뢰성 있는 유계 회복) 성립.

### 3.2 §4b (예측 정합, 대조군) — 5회

| 변형 | 회복 n/5 | persist (±SD) | lastV (±SD) | hotswaps (±SD) |
|---|---|---|---|---|
| **BoundGuard** | **5/5** | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.0 ± 0.0 |
| **A. no-dwell** | **1/5** | 47.4 ± 21.7 | 3.3 ± 1.6 | **10.6 ± 0.8** |
| **B. re-invoke** | **5/5** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 |
| **C. hybrid** | **5/5** | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 |

- **대조군의 핵심 발견 — 두 메커니즘은 상보적 조건에서 실패한다.**
- **progress 제거(B·C)는 대조군에선 정상(5/5)** — 예측이 맞으면 top-1이 feasible이라 재적용으로 회복. B/C의
  실패(Q3 0/5)는 **오예측 조건 한정**이다.
- **dwell 제거(A)는 대조군에서도 실패(1/5)** — top-1(=cand_1)이 feasible인데도 no-dwell이 그 배치를 transient
  중에 **조기 이탈**하고, 폭주(hotswaps 10.6 vs BoundGuard 1)한다. **dwell 부재는 예측 정합 여부와 무관하게 치명적.**

### 3.3 Q4 (가용 후보 없음, 종료성) — 5회

| 변형 | 회복 n/5 | 종료 | lastV (±SD) | hotswaps |
|---|---|---|---|---|
| **BoundGuard** | 0/5 | ✓ 유계(best-so-far) | 65.3 ± 19.9 | 9.0 |
| **A. no-dwell** | 0/5 | ✓ 유계 | 66.6 ± 20.3 | 9.0 |
| **B. re-invoke** | 0/5 | ✓ 유계 | 74.4 ± 0.8 | 0 |
| **C. hybrid** | 0/5 | ✓ 유계 | 74.5 ± 1.1 | 0 |

- **네 변형 모두 종료(유계)** — feasible 후보가 없어 아무도 회복 못 하지만(정상), **best-so-far 예산(보완 1)을
  공정하게 부여했기에** 무한 루프 없이 전부 종료. **예산이 없었다면 B/C는 무한 재적용**이었을 것(그 실행은
  비종료이므로 돌리지 않고, 예산이 곧 종료 메커니즘임을 명시). wall-clock 상한 도달 사례 없음.
- 착지 품질: BoundGuard/A가 순회로 약간 나은 배치(~65) vs B/C는 top-1 고착(~74). A는 고분산(SD 20).

## 그림
`docs/figures/c2_reactive_comparison.pdf` (Q3, 4변형 V(t)): BoundGuard 회복(V→0), **A는 높은 V(~24)에
고착**(무의미 측정으로 best-so-far가 나쁜 배치 선택), **B·C는 낮은 V(~1.2)에 고착**(top-1 미탈출).

## §4. A2 실패 모드 ↔ 관측 수치 매핑

회복 성공률 요약(2×2 × 3시나리오):

| | §4b (예측 정합) | Q3 (오예측) | Q4 (후보 없음) |
|---|---|---|---|
| **BoundGuard** | 5/5 | 5/5 | 0/5 유계 |
| **A. no-dwell** (dwell 제거) | **1/5** | **3/5** | 0/5 유계 |
| **B. re-invoke** (둘 다 제거) | 5/5 | **0/5** | 0/5 유계 |
| **C. hybrid** (progress 제거) | 5/5 | **0/5** | 0/5 유계 |

| A2 주장(실패 모드) | ablation | 관측 수치 |
|---|---|---|
| **dwell-time 부재 → thrashing** | A ↔ BoundGuard | §4b·Q3 **양쪽에서 A 실패**(1/5·3/5), lastV·persist **거대한 분산**(SD 12·27), hotswaps 8.8–10.6 vs 1–7. validate(dwell) **0회** — 각 후보를 관찰하지 않아 best-so-far가 무의미 transient로 선택. 실패 시 lastV 24.7(무행동보다 나쁨) |
| **progress 구조 부재 → 진전 없음** | C(및 B) ↔ BoundGuard | Q3에서 **B·C 0/5**. 재호출이 **매번 같은 top-1**(=위반 배치) 반환 → 5개 후보가 **전부 동일 배치**(재방문 5회), 대안 미탐색. C는 validate **5회 = 5×T_v=15s를 같은 top-1에 dwell 낭비**. §4b(top-1이 맞음)에선 5/5 → **오예측 한정 실패** |
| **재구성 비용 미계상** | 전 ablation | A: hotswaps 10.6(폭주 churn). C: 동일 배치에 15s dwell 낭비. B: top-1 재적용 반복. 모두 **무익한 재구성에 시간/자원 소모** |

## §5. 판정 — **A2 뒷받침 + 정밀화**

- **두 메커니즘이 서로 다른(상보적) 실패를 막는다** — 단일 인자로 환원되지 않는다:
  - **dwell은 보편적 필수**: A는 예측 정합(§4b 1/5)·오예측(Q3 3/5) **양쪽 실패**. 관찰 없이는 feasible
    배치조차 transient 중 조기 이탈하고, 선택이 코인플립이 된다.
  - **progress는 오예측에서 필수**: B·C는 top-1이 맞으면(§4b 5/5) 정상, 틀리면(Q3 0/5) 결정적 실패.
    예측기가 결정적(§1)이라 재호출만으로는 같은 오답에서 못 벗어난다.
- **BoundGuard만 세 시나리오 전부에서 property 유지**(§4b 5/5·Q3 5/5·Q4 유계). → **"기여는 mechanism이
  아니라 property"**가 실험으로 성립: 어느 한 메커니즘을 빼도 property(신뢰성 있는 유계 회복)가 깨진다.
- **A2 서술 조정 제안**: 세 실패를 나열하는 대신 **2×2로 제시**하고, *"dwell은 관찰 없는 선택의 불안정을
  막고(예측 정합 여부 무관), progress는 오예측 시 대안 탐색을 가능케 한다. 정책이 결정적이므로 재호출만으로는
  대안에 도달할 수 없다"* 로 정밀화. **반증 아님 — 오히려 각 메커니즘의 역할이 조건별로 분리되어 더 강해진다.**

## 부가 — 예측기 결정성이 progress 필요성의 근거
§1에서 확인한 **예측기 결정성**이 B/C 실패의 직접 원인이다: 재호출이 같은 입력에 같은 top-1을 주므로,
progress(순위 하강) 없이는 오예측된 top-1을 절대 벗어나지 못한다. 이는 코드로 뒷받침되는 A2의 핵심 문장이다.

## 무결성
- BoundGuard 본체 불변(기본 `none`). 변경: `schedule_executor_main.py`(no-dwell 경로에 best-so-far 기록·복귀
  추가, auto-advance 가드를 FSRR_NO_AUTOADVANCE로 게이트). 파라미터 불변. 스냅샷 보관.
- P1~P9 확인(§3). legacy/backup·main.tex 불변.
