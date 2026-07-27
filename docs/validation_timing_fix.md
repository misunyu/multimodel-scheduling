# 검증 타이밍 진단 (§1) — 구현 전 재검토 (게이트)

날짜: 2026-07-22 · **진단이 지시문 전제(δ>T_v)와 다름 → 수정 전 재검토 필요** · 코드 변경 없음(진단만)

## 결론(먼저)
지시문 전제는 "거친 전환 δ=3.7s > T_v=3s라서 전환 완료 전에 검증한다"였다. **코드·실측 진단 결과 이 전제는
성립하지 않는다**: 검증 창은 **이미 LLM 전환(δ) 완료 후 시작**한다. 후보가 부당 기각되는 **실제 원인은
누적 backlog 배수 시간(~30s)이 T_v(3s)를 훨씬 초과**하는 것이며, 이는 별도의 수정을 요한다.

## §1a. 검증 창 시작 시점 — 이미 "전환 후"
- `_run_next`: `self._bg.sync(...)`(line 282) → (mode 1 경로) `update_combination` → `_maybe_start_qos_trigger`(line 530)에서 `entered_at=time.time()`(583).
- **`bg.sync`는 LLM ready 파일까지 블록**(`background_llm.py` 93-98: `while time.time()<deadline: if os.path.exists(ready_path): break`). 즉 LLM 거친 전환(δ=3.7s)이 **끝난 뒤** `entered_at`이 설정됨.
- → **LLM δ는 이미 흡수됨. T_v는 LLM 전환 후 시작.** 지시문의 "전환 시작부터 T_v" 전제는 LLM에 대해 틀림.

## §1b. 실제 원인 — 상속된 backlog 배수 >> T_v
- **hot-swap(mode 1)은 frame 큐를 리셋하지 않음**: 큐는 뷰어 최초 생성 시만 만들어지고(`unified_viewer.py` 574-597),
  hot-swap은 changed-device 워커만 교체·기존 큐 재사용. `_drain_and_close_all_queues`는 스케줄 **종료 시**만 호출(1530).
- 따라서 burst+cand_1+cand_2(전부 실패)에서 **큐가 buffer=30까지 참** → 그 backlog가 cand_3로 **상속**됨.
- cand_3(cgggg, LLM-CPU)이 feasible이어도 상속 backlog를 **~30s에 걸쳐 배수**해야 V<ε 도달.
- **실측(λ=52, hot-swap으로 cgggg 진입, 28s 관찰)**: V가 4.23→1.32로 단조 감소, backlog 30→2 배수.
  validate는 t+3s에 **V=2.3(배수 중)**을 측정 → V>ε → 기각. **feasible 후보를 과도기에서 부당 기각.**

## §1c. 단, 경계 불안정성 주의
- 동일 cgggg를 **깨끗한 시작**(빈 큐)으로 λ=52 측정 시 V~5-6, backlog~20 고정(배수 안 됨).
- 즉 **λ=52에서 cgggg는 용량 경계(μ≈λ)라 불안정** — 진입 경로(warm/cold, LLM 타이밍)에 따라 배수하기도, 정체하기도.
- → "cgggg가 명확히 feasible이라 timing만 고치면 Q3 회복"이라 단정 못 함. **경계 영역.**

## 판정 (§1 게이트: "이미 전환 후 시작이면 원인 재검토")
- **지시문 제안 수정("전환 완료 후 T_v 시작")은 이미 충족됨**(LLM δ 흡수). 그대로 구현해도 **효과 없음** — 전환은
  이미 끝났고, 문제는 전환 후의 **backlog 배수(T_stable)**가 T_v를 초과하는 것.
- 진짜 후보(원인에 맞는 수정):
  - **(A) 후보 hot-swap 시 frame 큐 드레인/리셋** → 각 후보를 상속 backlog 없이 **자기 상태에서** 평가. feasible→T_v 내 V낮음, infeasible→T_v 내 V상승. 단 **cold-start 스파이크 주입**(§4b/C3/B2 무회귀 위험 — mode 0은 의도적으로 이걸 피함).
  - **(B) validate를 절대 V가 아니라 V 추세(감소=회복 중)로** 판정. 코드 변경 최소, 의미 변경.
  - **(C) T_v를 배수시간 이상으로**(instruction 금지 — 회피책).
- **경계 불안정(§1c)** 때문에 어떤 수정도 vision-4/λ=52에서 깨끗한 Q3를 보장하지 못할 수 있음.

## 사람 판단 필요 (게이트)
1. 제안 수정(전환 후 시작)은 이미 충족 → 그래도 명시적 코드로 못박을지, 아니면 **원인(backlog)에 맞는 (A)/(B)로 갈지.**
2. (A) 큐 드레인은 §4b/C3/B2 무회귀 위험 — 감수할지.
3. λ=52 경계 불안정을 감안해 **더 낮은 λ**(cgggg가 안정적으로 feasible한 지점) 또는 **vision-3**로 Q3를 보이고, vision-4는 경계 사례로 별도 서술할지.
4. vision-3 Q3(이전 회복)는 cand_5가 마지막이라 default commit+긴 tail로 배수했기에 성립 — **backlog 배수 문제의 예외**였음(우연히 회복). 이 구조를 논문에 어떻게 반영할지.

## 무결
- 코드 변경 없음(진단·측정만). GPU 폴백 0(종료 후 1008MiB idle). legacy/backup/main.tex 불변.
