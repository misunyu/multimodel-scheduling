# 스케줄 순서 버그(`sort_keys=True`) 오염 범위 전수 판정

날짜: 2026-07-22 · 판정 전용(재실행은 별도) · main.tex 불변

## 요약
- **버그**: 일부 harness가 `yaml.safe_dump(sched)`(기본 `sort_keys=True`)로 콤보 키를 **알파벳 정렬** → executor가
  파일 순서대로 실행하므로 **실행 순서가 뒤바뀜**. 특히 `combination_stable`이 알파벳상 마지막이라 **저부하
  stable이 마지막에 실행 → "회복" artifact** 발생.
- **로드 경로는 정상**: `yaml.safe_load` + `list(keys())`는 파일 순서 보존, 재정렬 없음(`_placement_signature`의
  `sorted()`는 시그니처 내부 정렬일 뿐 콤보 순서와 무관). → 오염은 **덤프 측 정렬**에서만 발생.
- **[최우선] §4b BoundGuard "persistence 0s"는 artifact 확정** — 올바른 순서 재실행 시 결과가 질적으로 다름(§3).

## §1. 버그 영향 경로 (전수)
| 경로 | 정렬 | 판정 |
|---|---|---|
| `scripts/q13_failure_persistence.py:239` `write_runtime_yaml` | **sort_keys=False** | 순서 보존 → 유효 |
| `scripts/q13_...:175` (랭킹 temp) | sort_keys=True | 예측기 랭킹용, 순서 무관 → 무해 |
| `schedule_generator_logic.py:114,159`, `schedule_generator_app.py:1404,1471` | 정렬(기본) | 프로덕션: `combination_0..N` **독립 실행**(stable/burst 시퀀스 아님) → artifact 없음. 단 실행이 사전식 순서(combination_0,_1,_10,_2..)로 됨 — per-combo 측정엔 무해 |
| `reactive_deploy.py:344` fallback | 정렬(기본) | 단일 fallback 스케줄, 런타임 → 무해 |
| **scratchpad `q3_mixed_scenario.py`, `q3v4_scenario.py`** | sort_keys=True(수정 전) | **다중콤보 stable/burst/cand → stable 마지막 → 무효** |
| **scratchpad `bg_scenario.py`** | sort_keys=True(수정 전) | **동일 → 무효** (이 대화의 §4b no-regression 포함) |
| `c3/*.yaml` (C3) | (덤프 스크립트 불명) | 단일콤보 or `burst`<`offload`(알파벳=의도) → 유효 |
| `b2_run.sh` (B2) | **bash echo** | 정렬 없음, 순서 보존(burst→cand_1→cand_2) → 유효 |
| `mixed.yaml` (mixed_exec) | **bash heredoc** | 순서 보존(gpu→cpu) → 유효 |

- 수정 완료(sort_keys=False): q3_mixed, q3v4, bg_scenario.

## §2. 실험별 판정
| 실험 | 산출물 | harness | 콤보 | 순서영향 | 판정 |
|---|---|---|---|---|---|
| C3 fluid (Q1.4) | c3_stress_report, c3_fluid_validation.pdf | c3/*.yaml | 단일 or burst/offload | 알파벳=의도(b<o) | **유효** |
| B2 지표 (Q1.5) | c3_b2_metrics.pdf | b2_run.sh | burst,cand_1,cand_2 | bash echo 보존 | **유효** |
| benchmark Table I | benchmark_model_table.md | 격리 측정 | 스케줄 무관 | — | **유효** |
| 예측기 랭킹·y2·β 감사 | candidate_ranking_check 등 | 계산 전용 | — | — | **유효** |
| mixed 실행 검증 | mixed_exec_report.md | mixed.yaml | 2콤보 | heredoc 보존 | **유효** |
| **§4b Phase A 관문** | phaseA_boundguard_report | stable/burst/cand | 다중콤보 | **stable 마지막** | **무효(§3)** |
| **Q3/Q4 (vision-3/4)** | q3_experiment_report, trend_validate_report | q3_mixed/q3v4 | 다중콤보 | **stable 마지막** | **무효** |

## §3. [최우선] §4b BoundGuard "0s"는 artifact — 확정
수정된 harness(sort_keys=False)로 §4b 재실행, **올바른 순서 확인**(CSV: stable→burst→cand_1..cand_tail):

| 기법 | GPU persist / lastV | NPU persist / lastV |
|---|---|---|
| Static | 60s / 9.0 | 60s / 9.1 |
| **Adaptive** | **24s / 0.0 (회복)** | **25s / 0.1 (회복)** |
| **BoundGuard** | **42s / 1.7 (미회복)** | **33s / 1.7 (미회복)** |

- **이전 보고 "BoundGuard 0s, lastV 0"은 재현 안 됨.** 올바른 순서에서 BoundGuard는 **42s/1.7로 완전 회복
  못 하고, Adaptive(24s/0)보다 나쁨.**
- **원인**: 이전엔 `cand_1`(=top-1=all-GPU, feasible)이 **알파벳상 맨 먼저** 실행 → 선행 burst 없이 cold-start,
  상속 backlog 0 → V 낮음 → 즉시 "회복"처럼 보임. 올바른 순서(burst 후 cand_1)에선 cand_1이 burst의 상속
  backlog를 빠르게 배수하나 **V(t) 5s 윈도우가 뒤처져** validate 시점(T_v=3s)에 V=4.3(>ε), backlog는 이미
  ~0으로 배수돼 slope≈0 → **feasible top-1을 기각**. (원래 순간-V validate도 동일하게 기각했을 것.)
- **파급**: §4b는 "top-1이 정확한" 시나리오이므로 **Adaptive(top-1 커밋)가 최적**이고 BoundGuard 순회는 손해.
  이전 "BoundGuard bounded persistence 통과 → Phase B 착수 근거 → 논문 Q1.5 전제"가 **artifact에 의존**했다.
- **부수 발견(검증 메커니즘 자체 결함)**: hot-swap 후 상속 backlog가 큰 상태에서 **빠르게 배수하는 feasible
  후보**를 T_v=3s 창으로는 커밋 못 함(V 윈도우 lag + 배수 완료로 slope≈0). 이는 순서 버그와 별개의
  **validate 설계 문제**로, 추세 판정으로도 해결 안 됨.

## §4. 유효/무효 + 논문 영향 + 재실행 우선순위
- **유효(재실행 불요)**: C3(Q1.4), B2(Q1.5), Table I, 예측기 랭킹/감사, mixed 실행. — 순서 보존 근거 명확.
- **무효(재실행 필요)**:
  - §4b 관문 — **최우선**. 올바른 순서에서 Adaptive≥BoundGuard로 나오므로 **§4b 서술·관문 전면 재검토.**
  - Q3/Q4(vision-3/4) — 순서 수정 후 재실행 완료분 있음(§5).
- **논문 반영분 영향**:
  - Q1.4·Q1.5·Table I: **오염 없음**(유효 경로).
  - **§4b 관문(있다면): 오염** — BoundGuard 우위가 artifact. main.tex에 §4b 관문/Phase-B 착수 근거가
    반영돼 있다면 수정 필요(별도 대화).
- **검증 메커니즘 재검토 필요**: §3 부수 발견(빠른-배수 feasible 후보 기각)은 BoundGuard의 핵심 동작에
  영향 → 별도 설계 논의 필요.

## §5. TAIL=60 vision-3 (올바른 순서)
- Static/Stop-restart/Adaptive: lastV~3.5, persist 105-106s(고착).
- **BoundGuard: lastV=0.82(<ε, 회복), persist 63s.** — 이전 "lastV=0.01"은 stable-last artifact였으나, **올바른
  순서 + 충분한 tail(60s)에서도 BoundGuard는 (느리게) 회복**함. 단 회복은 cand_5(cggg=LLM-CPU)가 **마지막이라
  기본 commit + tail 배수** 덕이지, validate가 커밋해서가 아님(§3 결함과 정합).

## 판정 매트릭스 위치 / 게이트
- **게이트 §5.1 발동**: §4b BoundGuard 결과가 이전(0s)과 **질적으로 다름**(42s/1.7, Adaptive에 열세) → 관문
  통과 여부가 바뀜 → 이후 계획(Phase B, Q1.5 전제) 재검토 필요.
- **게이트 §5.4 관련**: Q3(vision-3)는 올바른 순서·긴 tail에서 회복하나 그 메커니즘이 validate가 아님.

## 무결
- 판정·측정만(코드 로직 변경은 이전 커밋의 트렌드 검증/cand1 fix + 이번 harness sort_keys 수정). GPU 폴백 0.
  legacy/backup/main.tex 불변.
