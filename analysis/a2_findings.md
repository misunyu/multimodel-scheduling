# A2 Findings — co-tenant 대표성 (아카이브, 2026-07-04 종결)

> ## ⛔ 최종 disposition (먼저 읽을 것)
> **A2 신규 실험(Task 2 identity 불변성, CPU sweep의 원고 반영)은 이 투고에서
> 중단·미도입.** 이유: healthy-env에서도 paper-era anchor 미재현
> (ResNet k=1 GPU DM 24→59%, L2LM NPU DM 76→25%) → 신규 측정치와 논문 수치를
> 한 표에 혼재 불가.
>
> - **current-env 발견(imread 인과 메커니즘, CPU sweep, 스테이지 분해)은 그 자체로
>   유효**하나 paper-era 수치와 비교 불가 → 원고 미도입.
> - **"L2LM vs synth 대비(76% vs 27%, CPU util 비예측성)"는 broken-ORT-CUDA
>   아티팩트로 판명** (수리 후 L2LM 25% ≈ synth 32%). → §1.2/§2 철회.
> - **Task 2(identity 불변성)는 미착수 종결.**
> - 환경 이슈·재현 경로는 `env_snapshots/README.md`, ORT 수리는
>   `vendor_ort_cu12/README.md` 참조.

## 시간순 경위 (chronological)
1. **Task 1 (CPU sweep, synth)**: 사전등록 "Fig 3 거울상" 불성립. 두 경로 모두
   ~80% CPU까지 평탄, 100%에서 동반 붕괴. → "CPU util 비예측" 가설 제기(§1).
2. **Task 1b 시도**: L2LM NPU stall 스테이지 진단. inline 계측이 효과를 파괴
   (68→22%) → 인과 **imread ablation**으로 전환. **imread 제거 시 DM 0%로 붕괴**
   → 공유 host imread/preprocess가 stall locus로 확정(§3, current-env 유효).
3. **anchor 재현 시도 중 broken-ORT-CUDA 발견**: ResNet-ORT co-tenant가
   gpu_util 6%·CPU 100% (host-bound). 원인 = `onnxruntime-gpu 1.20.1`(CUDA-12)이
   CUDA-13 env에서 cuDNN 로드 실패 → CPU fallback.
4. **환경 수리**: 격리 CUDA-12 lib shim(`vendor_ort_cu12/`) → ORT-CUDA 복구,
   ResNet/L2LM co-tenant GPU-bound 복귀(gpu_util 74%).
5. **수리 후 anchor 재측정 → 드리프트 확인**: ResNet k=1 GPU DM **59%**(paper 24%),
   L2LM NPU DM **25%**(paper 76%), synth 32%(안정). **L2LM≈synth** → §1.2 대비가
   broken-ORT 아티팩트임이 드러남. GPU-path 드리프트는 ORT 무관(torch2.12/CUDA13
   추정) → paper-era와 비교 불가.
6. **결정(사용자)**: healthy-env 전면 재베이스라인 검토했으나, paper 수치 혼재
   불가로 **A2 신규 실험 중단**. 본 문서는 아카이브로 종결.

산출물(코드): `analysis/a2_util_sampler.py`, `a2_cpu_stress_worker.py`,
`a2_cotenants.py`, `a2_cpu_sweep.py`, `a2_aggregate_cpu.py`, `a2_stage_probe.py`,
`a2_stage_probe_1inst.py`, `a2_repro_check.py`, `a2_ablation.py`, `a2_affinity.py`,
`a2_l2_cell.py`, `a2_cotenant_proc.py`, `a2_task2_pilot.py`, `a2_anchor_check.py`.
데이터: 각 `analysis/a2_*.csv`. 환경: `vendor_ort_cu12/`, `env_snapshots/`.

> ⚠️ 아래 §1–§3b는 **작업 당시 기록**이며, §1.2 대비와 §2 재프레이밍은 위
> disposition대로 **철회**되었다(broken-ORT 아티팩트). current-env 인과 발견
> (§3 imread ablation, §3b Table2 스코프)은 유효하나 원고 미도입.

---

## 1. Task 1 — host-CPU 경합 sweep: "CPU utilization도 경로 열화의 예측자가 아니다"

### 1.1 측정 (All-GPU/All-NPU, PANEL4 N=4, ≥3 reps; `a2_cpu_sweep_agg.csv`)

합성 CPU-pinned stress 워커 수 c를 sweep. 좌표 = host CPU util(%) 및 각 경로 DM.

| CPU% (c) | All-GPU worst | GPU DM% | All-NPU worst | NPU DM% |
|---|---|---|---|---|
| 13–48% (c=0–8) | 0.1148 (평탄) | ~0.5 | 0.0835 (평탄) | ~0.3 |
| 81% (c=16) | 0.1139 | 2.6 | 0.0829 | 1.5 |
| **100% (c=24)** | **0.0766** | **62.6** | **0.0748** | **26.7** |
| anchor: LLM-CPU (~42%) | 0.1149 | 0.5 | 0.0836 | 0.2 |

### 1.2 핵심 대비 (재프레이밍의 근거) — ⛔ **철회 (broken-ORT 아티팩트)**
> 아래 "L2LM 86%CPU→68% vs synth 100%→27%" 대비는 **ORT-CUDA가 깨진 상태**에서
> 측정됨. 수리 후 L2LM=25%≈synth=32% (§5) → 대비 소멸. 원고 미도입.

CPU utilization은 NPU 경로 열화와 **비단조**:

| 상황 | host CPU% | NPU DM% | 측정 |
|---|---|---|---|
| 합성 CPU 포화 (c=24) | **100%** (99% user) | **26.7%** | a2_cpu_sweep (Task 1) |
| **L2LM co-tenant** | **86% user** (논문 93%) | **68% (63/80/61)** | a2_repro_check stock (Step 2) |
| anchor: LLM-CPU 추론 (합성 아님) | 42% | ~0 | Task 1 anchor |

→ **더 낮은 CPU(86%)의 L2LM이 더 높은 CPU(100%)의 합성보다 NPU DM을 2.5배 이상
크게 유발**한다(stock 하네스 확증, rev20 76%와 정합). host CPU utilization만으로는
NPU 경로 붕괴를 예측할 수 없다. 현실적 단일 CPU 추론(anchor, 42%)은 어느 경로도
못 깬다. 따라서 **"CPU-경합 자원축"은 결정 변수가 아니다** — 경합의 *양*(CPU%)이
아니라 *성질*(어느 경로의 host 의존 스테이지를, 어떤 시간 granularity로 때리는지)이 문제다.

**가설 iii 확인**: L2LM co-tenant는 이 환경에서 **CPU-bound**(gpu_util 4–6.6%,
power 18–32W로 GPU 거의 idle, cpu_user 86%). ResNet50+TinyLLaMA ORT-CUDA 세션이
GPU가 아니라 host CPU를 포화 → "GPU co-tenant"라는 명명이 이 환경엔 부정확.

### 1.3 판정
- 사전 등록한 "Fig 3의 거울상(NPU만 붕괴, GPU 평탄)"은 **불성립**(threads=4 canonical).
- 두 경로 모두 ~80% CPU까지 평탄, 100%에서 **동반 붕괴**(GPU DM 62.6% > NPU 26.7%).
- host 비용의 지배항은 양 경로가 공유하는 imread(1920×1200)+letterbox 전처리.
- **CPU utilization은 경로 열화의 예측자가 아니다** (§1.2 비단조성).

---

## 2. 2축 프레이밍 재작성 — "자원 축" → "경로 축" — ⛔ **철회**
> "경로 축" 재프레이밍은 §1.2 대비에 의존했고, 그 대비가 broken-ORT
> 아티팩트로 판명되어 **원고 미도입**. 아래 LaTeX 초안은 보존용(사용 금지).

### 폐기된 프레이밍
> ~~L1–L3는 GPU-경합축 × host-CPU-경합축의 2-자원축 공간 대표점.~~
(Task 1이 반증: CPU% ≠ NPU 경로 열화 예측자.)

### 새 프레이밍 (Sec 4.1 co-tenant 문단 초안, 영문 LaTeX)

```latex
% --- reframed: PATH-deadline axes, not resource axes ---
Our co-tenant configurations are not chosen as representative applications but as
representative points in a two-dimensional \emph{execution-path} stress space: the
GPU-path deadline-miss rate and the NPU-path deadline-miss rate. What determines
the deployment-time ranking is neither the identity of the co-tenant nor which
hardware resource is saturated, but which execution path misses its frame budget.
The GPU path is stressed by co-tenants that saturate GPU compute (L1--L3, swept
continuously via the ResNet50 co-tenant, Fig.~\ref{fig:sweep}); the NPU path is
stressed when host-side stages it shares with the streaming pipeline are stalled.
Crucially, host-CPU utilization alone does not predict NPU-path degradation:
the \Llm{} co-tenant drives the NPU deadline-miss rate to $68\%$ at $86\%$ host
CPU, whereas saturating all 24 cores with synthetic spin reaches $100\%$ CPU with
only a $27\%$ NPU deadline-miss rate, and a realistic CPU-only inference tenant at
$42\%$ CPU degrades neither path. A causal ablation localizes the NPU-path stall:
pre-decoding every frame into a preloaded input tensor---removing host image
decode and letterboxing from the runtime path---collapses the NPU deadline-miss
rate from $68\%$ (\Llm{}) and $27\%$ (synthetic) to under $1\%$ and restores the
uncontended worst-stream sAP, so the shared host \emph{image-decode/preprocess}
stage, not on-chip inference or post-processing, is the contended resource.
Pinning the co-tenant to disjoint cores does not relieve the stall, indicating
memory-bandwidth/cache rather than core-scheduling contention; this is why a
memory-bandwidth-heavy inference co-tenant stalls the NPU path more than pure
compute at higher CPU occupancy.
The two placements (All-GPU, All-NPU) therefore trace out the two path-deadline
axes, and L1--L3 sample distinct regions of this space regardless of which
specific applications instantiate them.
```

```latex
% vendor-demo realism anchor (Sec 4.1)
This configuration is not hypothetical: the NPU vendor's reference deployment runs
eight concurrent camera detectors alongside an on-demand vision-language model
~\cite{mobilint-aries}, matching our \Lvlm{} setup (eight-stream YOLO + VLM).
```

### identity 불변성 인용 (Task 2 지지 시 삽입할 자리 — 결과 확정 후)
```latex
% PENDING Task 2: if invariance holds
Moreover, matched GPU deadline-miss rates produced by structurally different GPU
co-tenants (ResNet50 vs. ViT/VGG) yield the same worst-stream sAP (within
$2\times$ pooled std), confirming that the GPU-path deadline-miss rate---not the
co-tenant's identity---is the sufficient coordinate.
```

---

## 3. Task 1b — L2LM NPU 경로 stall 귀속 (인과 ablation으로 확정)

목적: §1.2의 L2LM(NPU DM 68%) vs 합성(27%) 격차가 어느 host 스테이지에서 오는지 특정.

**계측 방법 전환 (중요)**: inline per-stage 타이밍(`a2_stage_probe.py`)은 L2LM
효과를 시스템 전역에서 파괴함 — 1개 스트림만 계측해도 전체 DM이 68%→22%로 붕괴
(Step 2 follow-up). 즉 효과가 sub-ms 스케줄링 granularity에서 작동하여 측정 행위가
효과를 없앰. 따라서 **inline 타이밍 대신 인과 ablation + 비침습 프로파일링**으로 귀속.

### (A) imread/preprocess ablation — 인과 확정 (`a2_ablation.csv`)
프레임을 사전 디코드+letterbox하여 RAM preload → 런타임 경로에서 read_pre 제거.

| config | ablated DM (read_pre 제거) | stock DM | ablated worst sAP |
|---|---|---|---|
| none | 0.0% | ~0% | 0.084 |
| synth_c24 | **0.3%** | ~27% | — |
| L2LM | **0.2%** | **~68%** | **0.0837 (=무경합 baseline)** |

→ **imread+preprocess 제거 시 L2LM·synth NPU DM이 0%로 붕괴, worst sAP 무경합값 복원.**
NPU 경로 stall은 인과적으로 **공유 host image-decode+preprocess(imread) 스테이지**
(on-chip·postproc·TinyLLaMA 고유 sync 경로 아님). synth도 붕괴 → 공유 전처리 병목 일반화.

### (B) CPU affinity 판별 (`a2_affinity.csv`)
별도 프로세스 co-tenant, 코어 분리(fg 0–11 / co-tenant 12–23) vs 비핀:

| variant | DM |
|---|---|
| shared (비핀) | 24.8% |
| pinned (분리) | 21.2% |

→ 코어 분리해도 DM 유지 → **core-scheduling 경합 아님 = memory-bandwidth/cache 경합.**
(부수: 별도 프로세스 25% ≪ in-process 68% → 공유 주소공간/allocator/CUDA-context가
memory-subsystem 경합을 증폭 — 역시 스케줄링 아님.)

### (C) py-spy 교차확인 (`a2_pyspy_l2.json`)
**비침습 확인**: py-spy 하에서 DM 64–67% 유지 (inline 계측 23%와 대조). 플레임그래프는
co-tenant ORT `run`(16%)+COCOeval 지배하나 foreground `raw_decode`(imread) 존재로 A 뒷받침.

### 최종 귀속 판정
**L2LM NPU 경로 stall = 공유 host imread/preprocess 스테이지의 memory-bandwidth/cache
경합.** core-scheduling도, TinyLLaMA-고유 sync도 아님. CPU utilization이 예측자가 아닌
이유 = 경합의 *성질*(ORT 추론의 memory-BW 압박 vs 순수 ALU spin)이 imread의 memory
자원을 더/덜 때리기 때문 (L2LM 86%CPU→68% vs synth 100%CPU→27%).

### 확정 확인 (Sec 4.3/4.4 원 측정 조건 + threads=4 값)
- 논문 line 245/413 "NPU single-stream latency is dominated by host-side
  post-processing"의 **원 측정은 threads=24(torch 기본값)**(rev16): postproc
  **27.3ms(70.2%)**, imread 6.9ms, on-chip 4.16ms, pre 0.54ms, total 38.9ms.
- **threads=4 확정 single-stream 분해 (신규 측정, `a2_step3_singlestream.csv`)**:
  **read_pre(imread+letterbox) 5.40 + on-chip 7.91 + postproc 1.30 = eff 14.63ms, DM 0%.**
  → postproc는 **최소 스테이지(1.30ms)**, on-chip·imread가 지배. 원고 "post-processing
  지배" 문구는 threads=4에서 **거짓**, 정정 필요.

### D — 원고 관련 확인 2건
- **D1 (co-tenant 실행 경로)**: ORT 세션 = CUDA EP(우선)+CPU EP(fallback);
  tinyllama "45 Memcpy nodes" + 실측 gpu_util 4–6.6% → **CUDA-EP 설정이나 실질
  host-bound**. 논문 line 149 "All co-tenants execute on CUDA"는 부정확
  (LLM은 상당부분 host 실행이며 그것이 NPU stall 메커니즘).
- **D2 (gpu_util 49.5% vs 4–6.6%)**: `cpuload_raw` L2_lm = npu_skip ~79%, gpu_util
  ~49.5%(과거) vs 현재 npu_skip 68%(재현)·gpu_util 4–6.6%(드리프트). **NPU DM은
  재현되나 co-tenant GPU 점유가 드리프트** — 현재 host-bound(paper의 host-CPU-stall
  메커니즘과 정합). tab:main L2LM 행 npu_skip은 유효(rev20 재현), gpu_util 각주만 갱신 대상.

---

## 3b. Table 2 latency 스테이지 범위 정합 확인 (책상)

- Table 2 `NPU 10.0ms / GPU 8.3ms`(rev19 `infer_ms`) = `fg_worker`의 `rt`
  = **imread(호스트 디코드)+letterbox+온디바이스 추론+호스트 postproc(dequant/
  decode/NMS)를 포함한 end-to-end per-frame wall time**, threads=4, **직렬**
  (디코드 별도 파이프라인 스레드 없음). NPU는 `cv2.imread`가 `npu_infer` 내부,
  GPU는 ultralytics `predict(path)` 내부 read — **양쪽 동일 정의**.
- **Step 3(14.63ms) vs Table 2(10.0ms)는 정의 차이 아님** (둘 다 imread 포함).
  4.6ms 차이 = on-chip 지연 드리프트(Step3 7.91 vs rev16 4.16ms), imread 아님.
- **원고 정합**: 10.0ms 내 threads=4 구성상 imread(~5–7)>on-chip(~4)>postproc(~1.3)
  → line 245/413 "dominated by host-side post-processing"는 Table 2 수치와도 모순.
- **캡션 제안**: "Latency is the mean end-to-end per-frame processing time at four
  post-processing threads, measured serially (no decode pipelining) and including
  host image decode and letterbox preprocessing, on-device inference, and host
  post-processing (dequantization, decoding, NMS). The NPU's higher latency
  reflects host-side post-processing overhead rather than on-chip compute."

## 3c. ⚠️ 환경 발견 — ONNX Runtime CUDA 깨짐 (Task 2 pilot, `a2_task2_pilot.csv`)

**ORT CUDAExecutionProvider가 세션 생성 시 로드 실패** (`libcudnn_cnn.so.9:
undefined symbol ... libcudnn_graph.so.9`; cuDNN 9 심볼 불일치, torch 설치
cuDNN과 충돌 추정) → **모든 ORT co-tenant(ResNet50, TinyLLaMA, VGG19)가 CPU
fallback 실행**. torch-CUDA는 정상.

이것이 A2 전반의 근본 원인:
- **ResNet50 co-tenant는 host-bound** (단독 gpu_util 6.7%, CPU 100%). 현재
  All-GPU N=4 GPU DM: k=1→**80.3%**(논문 rev30 24.3%), k=3→91%, k=8→94%,
  모두 CPU 91% — GPU가 아니라 **host CPU로 DM 유발**. **논문 24/50/67% 매칭 불가.**
- **ViT-B/16(torch-CUDA)만 진짜 GPU-bound** (단독 gpu_util 96%, CPU 8%;
  부하 시 gpu_util 81% CPU 17%). k=1에서 이미 GPU DM 77%.
- **Task 1b "L2LM host-bound"(§3 D1)의 근본 원인 = 깨진 ORT-CUDA**. 단 imread
  경합 메커니즘·NPU DM 재현(68% vs rev20 76%)은 유효. **교차검증**: L3_VLM은
  qwen2vl(torch-CUDA, gpu_util 82%)라 GPU-bound → host CPU 낮음 → NPU DM ~1%
  (안 깨짐), imread 가설과 정합.
- **D2 드리프트(gpu_util 49.5→5%)의 정체 = ORT-CUDA가 과거엔 작동, 현재 깨짐.**

→ Task 2(ResNet DM 매칭)는 현재 환경에서 진행 불가. 선택지: (a) ORT-CUDA 수리로
논문 환경 복원, (b) ViT-only identity 비교(cross-환경 한계), (c) 보류. **대기 중.**

## 4. 한계 (원고 반영 후보)
1. **메모리 대역폭 HW 카운터는 `perf_event_paranoid=4`(root 필요)로 측정 불가.**
   대신 CPU-affinity ablation(§3B)으로 core-scheduling vs memory-subsystem을
   판별 — mem-BW 여부는 **직접 카운터가 아니라 affinity 불변성으로부터의 추론**이다.
2. §3B의 affinity 판별은 별도 프로세스 co-tenant(DM 25%)에서 수행 — in-process
   본 효과(68%)보다 약한 프록시. 다만 코어 분리 불변성의 방향은 동일.
3. inline per-stage 타이밍은 L2LM 효과를 파괴(§3) → 스테이지 귀속은 인과 ablation
   (제거 실험)으로 확정했고 절대 스테이지-지연 프로파일은 미제공(측정 불가능).
4. L2LM co-tenant의 gpu_util은 측정 시점에 드리프트(49.5%→5%, §D2); NPU DM(핵심
   지표)은 재현. 원고의 co-tenant GPU 관련 각주는 갱신 대상.

---

## 5. ORT-CUDA 수리 + healthy-env anchor 재측정 (disposition의 근거)

### 5.1 수리 (`vendor_ort_cu12/`, `analysis/ort_env.sh`)
근본원인: env=CUDA13/torch2.12인데 onnxruntime-gpu 1.20.1=CUDA-12 빌드 →
시스템 cuDNN skew로 ORT-CUDA 로드 실패, CPU fallback. 수리: 격리 CUDA-12 lib
(`nvidia-cudnn-cu12==9.12.0.46`, `nvidia-cublas-cu12==12.6.4.1`,
`nvidia-cuda-runtime-cu12==12.6.77`)을 `LD_LIBRARY_PATH`로 ORT에만 연결
(site-packages·torch·NPU 무변경). 검증: ResNet co-tenant GPU-bound 복귀
(gpu_util 82%/CPU 10%), torch+ORT 공존 OK, NPU OK.

### 5.2 healthy-env anchor (`a2_anchor_check.csv`, `source ort_env.sh` 하)
| anchor | 측정 (healthy) | paper-era | 판정 |
|---|---|---|---|
| ResNet k=1 All-GPU | GPU DM **59.3%**, gpu_util 74%, CPU 11% | DM 24%, worst 0.098 | GPU-bound ✓ **DM 드리프트** |
| L2LM All-NPU | NPU DM **25.2%**, gpu_util 73.5%, CPU 92% | DM 76%, gpu_util 49.5% | co-tenant GPU-bound ✓ **DM≠76** |
| synth c=24 All-NPU | NPU DM **32.2%**, CPU 100% | ~27% | 안정(ORT 무관) ✓ |

**결론**: (1) 수리로 co-tenant는 GPU-bound 복귀. (2) 그러나 anchor DM이 paper와
불일치(ResNet 24→59, L2LM 76→25). (3) **healthy-env L2LM(25%)≈synth(32%)**
→ §1.2 "L2LM 특별함" 대비는 broken-ORT 아티팩트. (4) GPU-path 드리프트(동일
gpu_util~70%에서 DM 24→59)는 ORT 무관 → torch2.12/CUDA13 추정, **paper-era와
직접 비교 불가**. → A2 신규 실험 원고 미도입 결정. 재현 경로: `env_snapshots/README.md`.

## 6. Task 2 (identity 불변성) — 미착수 종결
calibration pilot에서 ORT-CUDA 깨짐 발견으로 진입 불가 판정. 수리 후에도 anchor
드리프트로 within-session 설계조차 paper 비교 불가 → **미착수 종결**. pilot
관찰(`a2_task2_pilot.csv`): ViT-B/16(torch)만 진짜 GPU-bound(gpu_util 96%),
ORT 계열(ResNet/VGG19)은 수리 전 host-bound. 재개 시 ViT torch 계열 우선.
