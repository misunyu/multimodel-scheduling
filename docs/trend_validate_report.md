# 추세 기반 검증 판정 (backlog 기울기) + Q3/Q4 재실행

날짜: 2026-07-22 · validate 판정만 변경(T_v·ε·후보·α·β 불변) · main.tex 불변

## §1. 판정 규칙 (구현됨)
`check_validate`를 순간 `V(t)≤ε` → **추세 기반**으로 교체:
- T_v 구간 poll(200ms)마다 총 backlog `Q(k)=Σ qsize()` 샘플 수집.
- **T_v 후반부**(elapsed ≥ T_v/2) 샘플로 최소제곱 기울기 추정. (전반부는 hot-swap 직후 transition transient
  — 재시작 워커가 상속 backlog를 일시 소진 후 infeasible이면 ceiling으로 refill — 이므로 제외.)
- 판정(결합):
  - `V(t) ≤ ε` (배수 완료) → **commit**
  - `slope < −θ` (배수 중, μ*>λ, feasible) → **commit** (V>ε여도 배수 중이면 회복 예정)
  - 그 외(slope ≥ −θ: 상승/평평, μ≈λ 발산 영역) → **advance** (보수적)
- θ = 0.5 frames/s (`--qos-slope-threshold`). 샘플 <3 → 전체 구간 fallback.
- 로그에 `backlog_slope`·판정 근거 기록.

**논문 반영 메모(§III)**: "상속 backlog 하에선 순간 위반이 아니라 backlog 추세로 feasibility를 판정한다.
그렇지 않으면 배수 중인 feasible 후보를 부당 기각한다." — 알고리즘 개선.

## §4. 무회귀 (vision-only §4b, mode 1+validate)
- gpu_BoundGuard: 후보 전부 `V≤ε`(0.0~0.45)라 **기존 V≤ε 경로로 commit**(추세 경로는 V>ε에서만 활성).
- **bounded persistence = 0s, maxV=0.80** — phaseA_boundguard 결과와 동일. **무회귀 확인.**
- (추세 변경은 feasible 후보 판정을 안 바꿈 — V≤ε면 그대로 commit.)

## §5. vision-3 Q3 재실행 (λ=80, 균일 slo15, buffer=12)
| 기법 | maxV | 최종 V | persist | 회복 |
|---|---|---|---|---|
| Static | 4.17 | 1.00 | 89s | ✗ |
| Stop-restart | 4.17 | 1.00 | 89s | ✗ |
| Adaptive | 4.18 | 0.99 | 88s | ✗ |
| **BoundGuard** | 4.70 | **0.01** | 50s | **✓** |

BoundGuard 후보별 추세 판정(2nd-half slope):
| 후보 | 배치 | validate V | slope(2nd-half) | 판정 |
|---|---|---|---|---|
| cand_1 | gggg (=top-1=burst) | 5.95 | −0.67 | commit*(경계, 아래 주) |
| cand_2 | gggc | 4.15 | −0.30 | advance |
| cand_3 | ggcg | 4.36 | +0.71 | advance |
| cand_4 | gcgg | 4.46 | −0.08 | advance |
| cand_5 | cggg (LLM-CPU) | (트리거 없음, 마지막) | — | 기본 commit → 배수 회복 |

- **BoundGuard 여전히 회복**(lastV=0.01). 단 회복은 **cand_5가 마지막이라 기본 commit + 긴 tail 배수** 때문 —
  진단대로 vision-3에는 cand_5 이전에 feasible 후보가 없어 **추세 판정이 결과를 바꾸지 않음.**
- *cand_1(=burst와 동일 배치 gggg)은 잔여 transient로 slope=−0.67(θ=0.5 미달)로 marginal false-commit.
  단 duration cap(8s)으로 advance하므로 결과 무영향. (2nd-half 도입 전엔 −5.95로 더 컸음.) cand_1은
  top-1 자체라 별도 처리(자동 advance) 여지 있음 — 사람 판단.

## §6. vision-4 재확인 (λ=52, per-model SLO, buffer=30) — Q4 유지
| 기법 | maxV | 최종 V | persist | 회복 |
|---|---|---|---|---|
| Static | 9.01 | 7.29 | 79s | ✗ |
| Stop-restart | 12.06 | 8.40 | 78s | ✗ |
| Adaptive | 13.79 | 7.91 | 78s | ✗ |
| **BoundGuard** | 11.76 | **1.94** | 54s | ✗ (bounded) |

BoundGuard 추세 판정:
| 후보 | 배치 | validate V | slope | 판정 |
|---|---|---|---|---|
| cand_1 | ggggg | 12.2 | +0.16 | advance ✓ |
| cand_2 | ggggc | 11.1 | +1.91 | advance ✓ |
| **cand_3** | **cgggg (LLM-CPU)** | 2.53 | **−0.33** | **advance**(−0.33 ≥ −0.5) |
| cand_4 | gcggg | 1.22 | +0.26 | advance ✓ |

- **cand_3(cgggg)이 −0.33/s로만 배수** — λ=52는 cgggg의 **용량 경계(μ≈λ)**라 배수가 매우 느림(T_stable→긴).
  보수적 θ=0.5로 advance. **fluid 모델 정합**(μ≈λ는 발산이므로 feasible로 안 봄).
- 별도 실측(hot-swap 진입 cgggg 28s 관찰): V 4.23→1.32(backlog 30→2)로 배수하나 ε 도달에 ~35s — 실용적 회복 아님.
- → **vision-4는 추세 판정 하에서도 Q4**(4-vision이 GPU 포화, 대안이 경계라 실용적 회복 없음). §0.3대로 그대로 보고.

## 종합 판정 (§0.3 준수)
- **추세 판정은 올바르게 구현·동작**: 상승(+1.91)·평평(+0.16) advance, feasible(V≤ε) commit, 경계(−0.33) 보수적 advance.
  fluid 모델(drain slope −(μ*−λ))과 정합, 상속 backlog에 오염 안 됨.
- **결과 자체는 이 두 설정에서 안 바뀜**: vision-3 Q3 회복은 원래 cand_5 마지막-commit 덕(추세 무관);
  vision-4 Q4는 cgggg가 진짜 경계(느린 배수)라 유지. 즉 **이전 Q3 성공이 "타이밍 버그 산물"은 아니었고**
  (cand_5 default-commit), **vision-4 Q4도 판정 버그가 아니라 실제 용량 한계**였음 — 둘 다 확정.
- **추세 판정이 결과를 바꾸는 조건**: feasible 후보가 **중간 순위**이고 상속 backlog가 큰 경우(순간 V로는
  부당 기각되나 추세로 commit). 현 두 워킹셋엔 그 조합이 없음(vision-3: feasible=마지막, vision-4: feasible=경계).
  이 조건을 실증하려면 별도 워킹셋 필요 — 사람 판단.

## 무결
- validate 판정만 변경(T_v·ε·후보·N_cand·α·β·예측기 불변). GPU 폴백 0. §4b 무회귀 0s 유지.
- legacy/scripts/backup 무변경. main.tex 불변. 스냅샷 `backup/trend_validate_20260722_162555/`.
- 그림: `docs/figures/q3_misprediction.pdf`(vision-3). ~~`q3_paperset_q4like.pdf`(vision-4)~~는
  **폐기**(2026-07-24) — Δ-창 `V(t)` 하에서 해당 시나리오는 위반에 진입하지 않는다.
  [q3_experiment_report.md](q3_experiment_report.md) §3e 참조.
