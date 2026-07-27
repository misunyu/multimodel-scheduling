# Hot-swap transition resource cost (ISSRE R4) — measurement + failure-path audit

날짜: 2026-07-27 · 알고리즘·파라미터·워크로드 불변 · `main.tex` 미수정(문안 초안만) · 새 그림 없음(표)

**R4 (ISSRE 메타리뷰)**: *"The hot-swap transition fundamentally assumes sufficient residual system
resources exist to execute both old and new model instances concurrently; initiating this during a
contention-driven performance failure risks severe transient resource exhaustion."*

**한 줄 결론**: 전환 구간의 **디바이스 메모리(VRAM/NPU DRAM) 증분은 0**이다 — 이 시스템의 모든 전환은
디바이스를 *가로질러*(GPU↔CPU) 모델을 옮기고 생성 모델은 kill-then-start라, 옛/새 인스턴스가 **같은
디바이스를 동시에 점유하지 않는다.** 겹침이 남기는 것은 **호스트 RSS의 일시적 +약 200 MiB**(4-모델 스왑,
여유 308×)뿐이다. 측정값은 안전하지만 **실패 경로가 정의돼 있지 않은 것이 실질 잔여 위험**이다(작업 69-3).

---

## 작업 68 — 게이트: 기존 로그에 자원 데이터가 있는가 → **없음**

| 확인 | 결과 |
|---|---|
| 메모리 사용량 시계열 (host RSS / device mem) | **없음.** `results/` 722개 `performance_*.json`은 조합별 **집계 요약**(throughput/latency/drop)만. 메모리 필드·시계열 0. `mem/rss/vram/dram` 키 0건. |
| 호스트 RSS vs 디바이스 메모리 구분 | 해당 없음(둘 다 없음) |
| 전환 시각과 정렬 | 전환 시각은 `adaptive_deploy` stdout(`hot-swap complete delta=…`)에만, 자원 축은 부재 |

→ 판정표의 **"없음"** → 작업 70(최소 재실행). Q3 + bounded-recovery **2 시나리오만** 계측 재실행,
전 매트릭스 재실행 없음.

---

## 작업 70 — 최소 재실행 (계측만 추가)

- **관찰자 사이드카**(`scratchpad/v18/mem_sampler.py`): 실행기 PID 트리의 host RSS(psutil) + GPU VRAM
  (`nvidia-smi --query-compute-apps`, 우리 PID 귀속분) + 총 VRAM을 **10 Hz**로 샘플링(요구 ≥5 Hz 충족;
  623 표본/62.5 s, 최대 간격 0.195 s). 워크로드·YAML·모드·duration은 기존 하네스와 **동일**, 측정만 추가.
- **출력은 `scratchpad/v18/`** — 게시된 `results/`·그림 불변.
- **샘플링 부하 대조**(요구사항): 동일 Q3를 샘플러 유·무로 각 1회. 정상 상태 throughput 차이
  **−0.5 % ~ −5.9 %**(ml_pick 최대). 측정 대상인 **메모리 풋프린트**는 throughput에 둔감하므로 무시 가능.
- 플랫폼: **RTX 5090 32607 MiB VRAM, 호스트 62.1 GiB RAM.** (NPU 축은 아래 §NPU에서 코드로 다룸 —
  Q3/BR는 vision GPU 케이스라 NPU 워커 없음.)

---

## 작업 69-1 — 전환 종류별 겹침(코드 확인)

| 전환 종류 | 겹침 실측/코드 |
|---|---|
| **vision in-place hot-swap** (`_hot_swap_view`, δ≈0.1–0.9 s) | **겹침 있음.** 새 워커 스레드를 `start()`(자체 ORT/NPU 세션 로드)한 뒤 **ready 후에야** 옛 워커를 정지(`adaptive_deploy.py:283→310`). 옛+새가 같은 프로세스 안에 동시 생존. 실측 delta 88–879 ms. |
| **background generative relocation** (δ≈2.9 s) | **겹침 없음(설계).** `BackgroundManager.sync`가 옮겨지는/불요 child를 **먼저 kill**하고(`background_llm.py:70-74`) 그 다음 새 child를 `Popen`으로 **시작**(76-78). 옛 인스턴스가 살아있지 않다 → 중복 피크가 아니라 **서비스 공백**이 비용. `adaptive_deploy._restart_headless`도 동일(정지-후-시작). |

> **지시문 69-1 가설의 반증**: 지시문은 generative relocation을 "중복이 가장 큼"으로 예상했으나, 구현은
> 반대로 **먼저 죽인다.** 따라서 시스템 전체에서 옛+새가 동시 생존하는 곳은 **vision hot-swap 하나뿐**이고,
> 그 대상은 작은 vision 모델이다.

**결정적 관측 — 같은-디바이스 중복은 어느 워크로드에서도 일어나지 않는다.** `tests/*.yaml` 전수 스캔:
한 뷰가 **같은 디바이스에 머물며 모델만 바꾸는** 전환은 **0건**. 모든 회복 전환은 GPU↔CPU 교차이동이다.
→ 옛 GPU 세션과 새 GPU 세션이 **동시에 VRAM을 점유하는 상황 자체가 없다.**

---

## 작업 69-2 — 전환 구간 자원 피크 (측정)

정상 상태 = 각 phase 중간 구간 평균. 피크 = (직전 phase 끝 −2 s ~ 다음 phase 시작 +5 s) 창의 최대.
"증분" = 피크 − 인접 정상 상태의 큰 값(= 겹침이 추가한 순수 초과분).

### GPU VRAM (우리 PID 귀속, MiB) — **증분 전부 0**

| 시나리오 | 전환 | 종류 | 정상 pre | 정상 post | 피크 | **증분** | 여유 배수 |
|---|---|---|---|---|---|---|---|
| Q3 (7 vision) | burst→ml_pick | CPU→GPU | 0 | 1946 | 1946 | **0** | ∞(N/A) |
| Q3 | ml_pick→fallback | GPU→CPU | 1946 | 538 | 1946 | **0** | ∞ |
| BR (4 vision) | initial→overload | GPU→CPU | 1948 | 538 | 1948 | **0** | ∞ |
| BR | overload→offload | CPU→GPU | 538 | 1950 | 1950 | **0** | ∞ |

VRAM은 새 구성의 정상 상태로 **단조 증감**할 뿐 오버슈트가 없다(교차이동이라 옛 디바이스 사본이 없음).
전 구간 VRAM 최대 = **1950 MiB = 32607의 6.0 %**.

### 호스트 RSS (프로세스 트리 합, MiB)

| 시나리오 | 전환 | 종류 | 정상 pre | 정상 post | 피크 | **증분** | 여유 배수 |
|---|---|---|---|---|---|---|---|
| Q3 | burst→ml_pick | CPU→GPU | 810 | 1382 | 1378 | ~0 | — |
| Q3 | ml_pick→fallback | GPU→CPU | 1382 | 1784 | 1783 | ~0 | — |
| BR | initial→overload | GPU→CPU | 1163 | 1745 | 1745 | ~0 | — |
| **BR** | **overload→offload** | **CPU→GPU** | 1745 | 1494 | **1946** | **+201** | **308×** |

**겹침의 순수 비용이 드러나는 유일 케이스**는 BR overload→offload(과부하 중 시작하는 회복 전환 = R4
시나리오 그 자체): 옛 CPU 워커가 아직 살아있는 채로 새 GPU 워커가 로드되며 **양쪽의 호스트측 메모리가
겹쳐 +201 MiB** 순간 초과. 목적지 정상 RSS(1494)가 출발 정상(1745)보다 낮아 오버슈트가 보인다.
호스트 RAM 62 GiB 기준 **여유 308×**. (Q3에서 안 보인 건 목적지 정상 RSS가 최고점이라 겹침이 그 아래에
묻혔기 때문 — 겹침이 없다는 뜻이 아니라 정상 상태보다 낮았다는 뜻.)

**런별·최대 동반**: 위 표는 각 전환의 최대(peak)와 인접 정상 평균을 함께 낸다. 전역 최대: host RSS
1946 MiB, GPU VRAM 1950 MiB(6.0 %), 총 VRAM(타 프로세스 포함) 2840 MiB.

### 가정 상황(같은-디바이스 중복)의 상한 — 측정 아님, 상한만
어느 워크로드도 하지 않지만, 만약 한 전환이 **전체 GPU 워킹셋을 같은 디바이스에서 중복**한다면
상한 2×1950 = 3900 MiB = **32 GB의 12 %, 16 GB 디바이스의 24 %**(여유 각 8×·4×). 실제로는 뷰 하나의
모델만 중복되므로 훨씬 작다. **워크로드를 바꿔 이를 강제 측정하지 않았다**(범위 밖: 워크로드 불변).

---

## 작업 69-3 — 할당 실패 시의 동작 (코드 판독)

**측정값이 안전해도 실패 경로가 없으면 답이 아니다.** 코드 확인 결과 **실패 경로가 정의돼 있지 않다.**

1. **사전 자원 점검 없음.** 저장소 전수 grep(`free_memory|headroom|can_fit|fits|sufficient|available_memory`)
   → 전환 전 잔여 VRAM/DRAM을 확인하는 코드 **0건**. R4의 *"fundamentally assumes sufficient residual
   resources"* 를 코드가 그대로 확증한다 — 가정할 뿐 **점검하지 않는다.**
2. **로드 실패 = 조용한 뷰 정지.** vision 워커 로드 중 예외(CUDA EP 부재 `model_processors.py:104`, ORT
   할당/OOM, NPU DRAM 초과)는 잡혀서 "FATAL" 출력 후 **그래도 `ready_event.set()`**(329-332). 예외 전파·재시도
   **없음**.
3. **롤백 없음(전방 전환).** watcher(`adaptive_deploy.py:292-336`)는 ready 후 죽은 새 워커로 큐를 무조건 교체하고
   **옛(정상) 워커를 정지**(310-313). → 뷰가 죽은 워커에 물려 **무프레임 정지**. "실패 시 옛 배치 유지"가 없다.
4. **후보 순회 반영은 간접·지연.** 로드 실패 자체는 후보 순회에 전달되지 않는다. 그 *결과*(정지→지속 V(t))만
   외곽 검증 루프(`restart_rollback_deploy`/validate)가 **T_v 뒤** 감지해 해당 후보를 기각/롤백한다. 즉 실패
   후보는 downstream에서 걸러지되 **T_v 지연 사각지대**가 있고, 전방 hot-swap 자체엔 가드가 없다.
5. **NPU DRAM 초과.** `build_vision_npu`(`runtime/mobilint_vision.py:35`)가 .mxq를 NPU에 적재; DRAM 고갈 시
   qbruntime/zoo가 raise → 위 2·3의 조용한 정지 경로로 귀결. 우리 코드엔 스왑·축출 없음(런타임의 로드 거부/에러를
   그대로 뷰 정지로 변환).
6. **generative 로드 실패.** `_start_one`(`background_llm.py:93-104`)은 child가 조기 종료하거나 ready 타임아웃이면
   **WARNING 로그 후 계속 진행**(미ready child도 등록). 옛 child는 이미 kill됨 → 폴백 대상 없음. 재시도·롤백 없음.

---

## 작업 71-1 — 판정

사전 고정 매트릭스: **여유 배수 > 2 (전 플랫폼·전 전환 종류)** 를 측정 전 조합 모두 충족
(GPU VRAM 증분 0 → ∞; host RSS 최악 308×). → **위험 미미(risk negligible).**

단, **정직한 조건**:
- 측정 환경이 32 GB VRAM·62 GiB RAM으로 넉넉하다. 그러나 핵심 근거는 하드웨어 여유가 아니라 **구조**다:
  교차이동 + generative kill-then-start → **전환 자체가 디바이스 메모리 피크를 만들지 않는다.** R4의 "transient
  exhaustion"은 이 구현에서 대체로 발현하지 않고, 남는 질문은 "**목적지 배치가 애초에 맞는가**"라는 **정상 상태**
  문제로 환원된다(전환 문제가 아님).
- 그래도 **실패 경로 미정의(69-3)**가 실질 잔여 위험이다. 목적지가 안 맞으면(작은 디바이스·큰 워킹셋) 로드가
  실패하고, 구현은 **조용히 뷰를 정지**시킨 뒤 옛 배치를 이미 헐어버린다. 이는 측정값과 무관한 **설계 공백**이다.

→ 판정표상 "정지·보고 후 사람 판단"이 필요한 "여유 배수 < 1" 조합은 **없음**. 하지만 §III 설계 서술은
실패 경로를 인정하는 방향으로 **보완 권고**(아래 문안).

---

## 작업 71-2 — 논문 문안 초안 (사람이 적용; `main.tex` 미수정)

**§III Placement Manager** (현재 *"This overlapping execution avoids any gap in processing capacity"* 를 보완):
> Overlapping execution assumes residual capacity for the incoming instance. In our design the overlap is
> confined to *vision* in-place swaps, where the outgoing and incoming workers are threads in one process;
> the background generative path relocates by terminating the old child before launching the new one, so the
> two never co-reside on a device. Because every candidate relocates a model *across* devices (GPU↔CPU),
> the old and new instances never occupy the same device memory simultaneously. We measured no device-memory
> (VRAM) increment at any transition; the only transient cost is host RSS, which peaks ≈200 MiB above steady
> during a 4-model recovery swap (≈13 % of that config's host footprint), well within host RAM.

**Threats to Validity**:
> Our measurements use a 32 GB GPU and a 62 GiB host, where transition transients leave ample headroom
> (≥300× on host RSS). Because transitions relocate models across devices rather than duplicating them on
> one device, the binding constraint is not a transient overlap peak but whether the *destination* placement
> fits — a steady-state property that tightens as the working set grows or on memory-constrained devices such
> as the 16 GB NPU DRAM (consistent with our design note that NPU memory is the constraint). Furthermore, the
> current implementation does not perform a pre-flight residual-resource check: a failed model load stalls the
> affected view and is recovered only indirectly, on the validation timescale T_v, by the outer BoundGuard loop.
> A same-device model swap (not exercised by our workloads) would duplicate one model's device footprint for
> the sub-second overlap window; bounding this explicitly is left to future work.

**§IV**: 표 하나(위 69-2)로 충분. **새 그림 없음.**

---

## 데이터 출처 · 산출물 · 무결성

- 출처: 게이트=없음 → **최소 재실행 2건**(Q3 BoundGuard 4-phase, bounded-recovery 3-phase), 10 Hz 사이드카.
  전 매트릭스 재실행·알고리즘·파라미터·워크로드 변경 **없음**. 샘플러는 별도 프로세스(관찰자).
- 산출물(모두 `scratchpad/v18/`): `mem_sampler.py`, `run_mem.py`/`run_q3_mem.py`, `analyze.py`,
  `q3_bg_*`·`br_views_*`(metrics/mem/exec.log), `q3_nosampler_*`(부하 대조).
- **게시 자산 불변**: `results/`·`docs/figures/`·`confirmed_values.json`·`main.tex` 미수정. 새 그림 없음
  (표로 충분; 그림을 만든다면 확정표 등재+사이드카 선행).
- **한계(정직)**: (1) NPU DRAM 전환 피크는 **직접 계측 안 함** — Q3/BR가 GPU 케이스라 NPU 워커 없음; NPU는
  코드로만(로드=적재, 실패=조용한 정지, 사전 점검 없음). (2) 같은-디바이스 중복은 어느 워크로드도 하지 않아
  **상한 계산만**(워크로드 불변 준수). (3) 이 Q3 실행은 자원 계측용 — 전환·과부하는 실측이나 논문 회복 곡선
  재현이 목적이 아니며 그렇게 주장하지 않음(실행기 내부 v_score 0–152 척도, 논문 windowed V(t) 0–1과 별개).
