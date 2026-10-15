# Phase 1b (v1.1) — 조건 수정 후 pilot 재실행

선행: `revision_profiling/REPORT_PHASE1.md`, `CHANGES.md`, `LM_RESULT.md`, `NPU_COMPAT.md`, `SPEC_v2.2.md`.
v1 대비 변경: GPT-2를 CPU/GPU 양쪽 후보로 유지(GPU 고정 철회), conservation 판정을 timestamp 기준 정확 일치로, nice 결정 기준 확대, β 선택에 요청 수 충분성 조건 추가, pilot family 존재 사전 확인, 반복 편차 보고 방식, §5.2 3회.
Phase 1의 runtime·counter·feeder 구현은 그대로 쓴다. **Phase 2(전체 campaign)는 사용자 승인 전까지 실행하지 않는다.**

## 0. 사용자 결정 (확정)

| # | 항목 | 결정 |
|---|---|---|
| 1 | CPU session thread | `intra_op_num_threads = 4`, `session.intra_op.allow_spinning = "0"`. 모든 CPU session 동일. GPU session의 thread 설정도 같은 값으로 명시하고, CUDA EP session이 CPU thread pool을 쓰는지 ORT profiling 1회로 확인·기록 |
| 2 | LM | **GPT-2 포함, CPU/GPU 양쪽 후보 유지**(고정 없음). **TinyLLaMA 제외**(파일 불일치·완료 건수 부족·GPU 배치에서도 CPU 포화) |
| 3 | D10 | (b) D_macro 기본. (a)·(c) 계속 계산 |
| 4 | β 후보 | **{0.05, 0.1, 0.25}** |
| 5 | CPU–NPU | 보류. CPU–GPU 확정 후 NPU 보드에서 새 계측 정의로 재측정(기존 NPU 결과는 재사용 불가 — `NPU_COMPAT.md`) |
| 7 | 사양 밖 변경 | 분류 전처리(표준 ImageNet): **추인**. nice 10: **§2 비교 후 결정** |

## 1. 준비

1. `model_registry.yaml`에서 TinyLLaMA 제거(파일은 유지). GPT-2에 device 제약 없음.
2. Session 설정 변경(thread 4, spinning off) → `CHANGES.md`.
3. **Offline profile 재측정** (8 모델 × {CPU, GPU}). Phase 1 값과 나란히 표. 요청 rate `β × GPU capacity`와 §5.1 과부하 구분은 새 값으로.
4. **Family 재구성** (TinyLLaMA 제외만 반영, GPT-2는 자유 변수): family 수, N≥3 family 수, 후보 Σ2^N, run 수, Phase 2 시간. Phase 1의 "TinyLLaMA 제외" 시나리오(22 families, 2,016 runs)와 일치하는지 확인.
5. **Pilot family 사전 확인**: 재구성된 목록에서 (a) N=4 family, (b) N=8 family, (c) **LM이 전혀 없는 N=6 family**가 실제로 존재하는지 확인한다. 각 조건에서 이름순 첫 번째를 택한다. **없는 조건이 있으면 새 family를 만들거나 다른 N으로 대체하지 않고, 사용 가능한 family 목록을 보고한 뒤 사용자 확인으로 검증 대상을 확정한다.** 선정 결과를 실행 전에 `pilot/PILOT_FAMILIES.md`에 기록.
   - 이하 "전부-CPU" 배치는 GPT-2 포함 모든 instance가 CPU인 배치다.

## 2. nice 10 결정 (사전 고정 기준)

- 대상: N=8 family, β=0.1, 4×, 배치 3개(전부-GPU / 전부-CPU / 반반), **nice=0과 nice=10 각각 2회** (총 12 runs ≈ 5분).
- 비교 항목: (i) 실현율 ≥ 0.95, (ii) conservation(§3 규칙)과 failed=0, (iii) 배치별 F와 GPU/CPU instance latency p50, (iv) 같은 배치 2회의 F 상대 차이.
- 결정 규칙: nice=0에서 (i)(ii)를 모두 만족하고 (iv)가 nice=10보다 나쁘지 않으면(차이 ≤ 2%p) **nice 제거**. 아니면 nice=10 유지하고 "새 측정 플랫폼의 실행 조건"으로 `CHANGES.md`와 원고 구현 정보에 명시. (iii)은 판정용이 아니라 nice가 성능을 바꾸는지 기록용.
- 이후 pilot 전체는 결정된 한 설정으로만 실행.

## 3. Conservation·pending 판정 규칙 (v1의 "±1 허용" 삭제)

1. **Primary**: `[t0, t1)` 안의 이벤트 timestamp로 `generated, enqueued, dropped_full, completed, failed`를 집계하고, pending_start/end도 **같은 경계의 이벤트로 재구성**한다(각 요청의 enqueue ts와 completed/failed ts로 경계 시점의 상태 판정). 이 기준의 conservation은 **정수 오차 0**을 요구한다. 0이 아니면 invalid.
2. **Cross-check**: reader process의 경계 snapshot(queue 길이, in-flight)을 timestamp 기반 pending과 비교해 차이를 `snapshot_diff_start/end`로 기록한다. 이 차이는 **계측 오차 기록**이며 conservation 허용오차가 아니다. |diff| > 1이면 flag하고 원인(읽기 지연 ms)을 기록. 허용 여부는 판정에 쓰지 않는다.
3. **요청 ID 추적**: 이벤트 buffer에 요청 ID(instance, tick index)를 넣어, 모든 요청이 {dropped_full} ∪ {completed} ∪ {failed} ∪ {pending_end} 중 **정확히 하나**의 경로를 갖는지 instance별로 대조한다(중복·누락 0). 결과를 `trace_check` 열로. 이벤트 기반 pending이 정의상 보존식을 만족시키는 순환 논리를 막기 위한 독립 검증이다.
4. 이벤트 ts는 "완료 처리 직후 한 번"만 찍는다(이중 집계 방지). in-flight ∈ {0,1} assert 유지.
5. 낮은 β에서 LM 요청이 10건 수준이면 1건 차이가 10%다. 그래서 ±1을 무시하지 않는다.

## 4. Pilot 실행 (사양 v2.2 §5 순서, 아래 변경 적용)

- **§5.1 smoke**: β=0.1 고정, 1×. 8 모델 × 2 device. Conservation은 §3 규칙(정확 일치).
- **§5.2 arrival**: β ∈ {0.05, 0.1, 0.25} × {2×, 4×} × N=4·N=8 family × 3 배치 × **3회** (108 runs ≈ 40분). 기준: 실현율 ≥ 0.95, missed ≤ 5%, 배치 간 generated 일치(상대 2% / 절대 2건 규칙), 같은 배치 3회의 F 상대 차이(중앙값·최대) 기록. **F 차이 5% 기준은 보고하되, 통과/미통과 판정은 §6으로 넘긴다**(β 선택 조건에서 분리).
- **§5.3 GPU 부하**: (A) 4× 전부-GPU, (B) β별 1× 단일 GPU 전 모델, (C) **rate 범위**: N=4 전부-GPU에서 1×는 비포화(`D_macro ≤ 0.3` 그리고 `F/Λ ≥ 0.7`), 4×는 포화(GPU instance에 `dropped_full > 0`). β별 1×·4× 각 1 run.
- **§5.3 (D) 요청 수 충분성** (사전 고정): β별로 **가장 낮은 요청률 instance의 1× 계측 구간 `generated`** 를 보고한다. 자동 선택 조건은 모든 instance `generated ≥ 30`(q_i 해상도 ≤ 3.3%). 미달인 β는 자동 채택하지 않고 **사용자 결정 항목**으로 올린다(예: β=0.05에서 GPT-2 1× = 약 10건, β=0.1 = 약 20건은 미달 예상 — 그래도 실행·보고한다).
- **β 선택 규칙**: 5.1 통과 → 5.2의 실현율·missed·generated 일치 통과 → 5.3 (A)(B)(C)(D) 통과인 β 중 **가장 작은 값**. (D)만 미달인 β가 가장 작으면 "조건부"로 표시하고 사용자 결정. 만족 β가 없으면 중단·보고(후보를 또 바꾸지 않는다). 새 threshold를 결과를 보고 정하지 않는다.
- **§5.4 반복 편차**: 선택된(또는 조건부) β, 2×, LM-free N=6 family, 후보 8개(전부-GPU, 전부-CPU, 무작위 6 — seed 42), **3회**.
- **§5.5**: 세 정의 계산(보고용). "best measured among pilot candidates".

## 5. 반복 편차 보고 방식 (§5.4·§5.2 공통)

- 반복 간 변동을 **F, D_macro, S 각각** 분리해 보고(중앙값·90%·최대).
- 3회 평균의 **표준오차**와 **placement 간 S 차이 분포**를 나란히 놓고, 평균 차이가 표준오차 합보다 작은 placement 쌍은 **"순위 불확실"** 로 표시한다(3회 평균이 순위를 확정한다고 가정하지 않는다).
- 편차가 Phase 1(중앙값 9.2%)에서 얼마나 줄었는지, thread 4 변경이 원인 분리 근거가 되는지(nice 비교 §2 결과 포함)를 적는다. 여전히 크면 환경 불안정의 추가 원인(P-core/E-core 배치, 전처리 thread, feeder process, GPU clock)을 **조사 항목으로 제안**하되 이번 지시문 범위에서 새 실험을 추가하지 않는다.
- Phase 2 반복 범위 두 시나리오(3× N≥3만 3회 / 전체 3회)의 실행 시간을 반드시 보고. 결정은 사용자.

## 6. 보고 (`REPORT_PHASE1B.md`)

0. 변경 요약(thread·spinning·nice 결정과 근거·registry·family 재구성·pilot family 확인 결과), offline profile 신·구 표
1. §5.1–5.3 통과표(β별, (A)(B)(C)(D) 분리), β 선택 결과 또는 조건부/미선택 사유
2. Conservation: timestamp 기준 정확 일치 수, trace_check 결과, snapshot_diff 분포(별도)
3. 반복 편차(§5) — 순위 불확실 쌍 포함
4. §5.5 결과(참고)
5. Phase 2 예상: family·후보·run 수, 반복 범위 두 시나리오 시간
6. 원고 영향 갱신: 모델 8종(TinyLLaMA 제외 사유), thread 4, 요청 rate 규칙과 β, Δ, 전처리, nice 여부
7. 사용자 승인 요청: β(조건부면 요청 수 문제 포함), 반복 범위, Phase 2 실행

## 7. 규칙
- 원본 파일 불변, `revision_profiling/` 안에서만. Phase 1 결과는 `pilot_phase1/`로 보존, 새 결과는 `pilot/`.
- 기준에 맞추기 위해 counter·정의·기준·β 후보·thread 수를 바꾸지 않는다. 이 문서가 정한 결정 외의 변경은 보고 후 승인.
- 예상 시간: 준비 20분 + offline 10분 + nice 비교 5분 + pilot 약 70분. 실제 시간 기록.
