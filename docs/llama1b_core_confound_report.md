# llama1b 코어 구성 confound 조사 보고서

- 조사일: 2026-07-14
- 대상: `multimodel-scheduling-mobilint` (브랜치 `ubuntu_gpu_mobilint`)
- 교차참조: `MobilintTest/CORE_USAGE_REPORT.md` (직전 MLA100/qbruntime 실측 조사)
- 성격: **read-only 조사**. 코드 수정·재학습·커밋 없음. 이 문서가 유일한 산출물이다.

---

## 판정 (요약)

# ✅ CLEAN — confound 없음. 이 사유로는 재학습 불필요.

**llama1b는 프로파일 생성 / 학습 데이터 수집 / 실제 실행 세 경로 모두에서 동일하게 1코어(`single`, Cluster0/Core0)로 로드되었다.**

우려했던 "benchmark CLI 경유 = `global8` = 8코어" 경로는 **이 저장소에서 한 번도 사용된 적이 없다.** 저장소 전체(backup 제외)에서 `core_mode` 문자열과 benchmark CLI 참조는 **0건**이다. 8코어 llama1b는 애초에 이 데이터에 들어올 통로가 없었다.

단, CLEAN이라는 것은 "**일관되게 1코어였다**"는 뜻이지 "최적이었다"는 뜻이 아니다. 별도의 설계 이슈가 하나 남는다 — 아래 §5.

---

## 1. 데이터가 생성된 코드 경로 (항목 A)

핵심은 **세 경로가 모두 같은 로더로 수렴한다**는 것이다. llama1b/qwen2_vl의 NPU 로드 지점은 저장소 전체에 **단 한 곳**뿐이다: `runtime/llm_engine.py:73-80`.

### A-1. 정적 프로파일 → `profile_models.py`

생성 파일: `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json` (mtime 2026-07-09 09:57)
생성기 특정 근거: `profile_models.py:29` 의 usage 문자열이 이 경로를 그대로 지정한다.

| 모델 종류 | 로드 코드 | 코어 지정 |
|---|---|---|
| vision | `profile_models.py:59-61` → `build_vision_npu(model_name, infer_mode="global8")` | **명시적 `global8`** |
| llama1b / qwen2_vl | `profile_models.py:100-101` → `LLMEngine(model_name, device)` | **인자 없음 → 라이브러리 기본값** |

### A-2. 학습 데이터 수집 → `run_collection.py` (경유: `schedule_executor_main.py`)

`deploy_cpu_npu` 예측기의 학습 데이터 출처를 역추적한 결과:

```
run_platform_study.sh:26   run_collection.py --schedule schedules_cpu_npu.yaml --out .../cpu_npu/performance.json
  └─ run_collection.py:78-79   subprocess: schedule_executor_main.py --schedule ... (콤보당 1개 서브프로세스)
       └─ schedule_executor_main.py:19   from unified_viewer import UnifiedViewer
            └─ unified_viewer.py:499     Process(target=run_llm_process, ...)
                 └─ model_processors.py:375 run_llm_process
                      └─ model_processors.py:326-331 _run_generative → LLMEngine(model_name, device, ...)
                           └─ runtime/llm_engine.py:80  AutoModelForCausalLM.from_pretrained(...)   ← 코어 인자 없음
```

그리고 `run_platform_study.sh:31` 이 `--perf_dir xgboost_model/performance_data/cpu_npu` 로 학습 → `artifacts/deploy_cpu_npu_*`. 즉 **문제의 예측기가 먹는 데이터는 정확히 이 경로에서 나왔다.**

두 번째 수집기 `run_concurrency_study.py` 도 존재하지만(`run_concurrency_study.py:23,77` → 같은 `model_processors.run_llm_process` → 같은 `LLMEngine`), 이 스크립트가 만든 유일한 산출물 `concurrency_cpu_gpu/performance.json` 에는 **NPU 윈도우가 하나도 없다** (CPU/GPU 전용). NPU 라벨에는 기여하지 않는다.

### A-3. 실제 실행 → `schedule_executor_main.py`

`unified_viewer.py:481-499` 에서 워커를 스폰한다 — **A-2의 수집 경로와 문자 그대로 동일한 코드다.** 수집기가 실행기를 서브프로세스로 부르기 때문에, "학습 조건"과 "실행 조건"이 다를 수가 구조적으로 없다.

---

## 2. 모델별 실제 코어 구성 (항목 B)

`core_mode` 문자열이 아니라 **실제 점유 코어 수**로 판정했다. 실측값은 `MobilintTest/CORE_USAGE_REPORT.md` §2 (`Model.get_target_cores()` 직접 호출)에서 가져왔다.

| 모델 | 로드 방식 | `core_mode` | **실제 local 코어** | 근거 |
|---|---|---|---|---|
| **llama1b** | `LLMEngine` → `from_pretrained` (직접) | config.json에 **`core_mode` 키 없음** → 라이브러리 fallback **`single`** | **1** (C0/Core0) | `runtime/llm_engine.py:80` (코어 kwarg 없음); HF config.json: `core_mode` 부재, `target_cores: ["0:0"]`; 실측 `Target Cores: [CoreId(Cluster0, Core0)]` |
| **qwen2_vl** (text/LM) | 동일 | config.json `"core_mode": "single"`, `"target_cores": ["0:0"]` | **1** (C0/Core0) | HF config.json |
| **qwen2_vl** (vision enc.) | 동일 | `vision_config.core_mode: "multi"`, `target_clusters: [1]` | **4** (C1/Core0-3) | HF config.json |
| **qwen2_vl 합계** | — | — | **5 = 1 + 4** | ✅ **구조 유지됨** |
| **resnet50** | `build_vision_npu(infer_mode="global8")` | `global8` | **8** (+global 2) | `model_processors.py:259`, `profile_models.py:61` |
| **yolo11n/s/m/l/x** | 동일 | `global8` | **8** (+global 2) | `model_processors.py:171`, `profile_models.py:61`, `runtime/mobilint_vision.py:23,35` |

주의사항 반영:
- **vision의 `single`은 8코어**라는 함정은 여기선 무관하다 — vision은 세 경로 모두 `global8`을 **명시**하고 있고, `build_vision_npu`의 시그니처 기본값(`mobilint_vision.py:23`)도 `global8`이라 어느 쪽이든 8코어다.
- **`core_mode` 미지정 → 라이브러리 기본값** 규칙이 적용되는 것은 llama1b 하나뿐이며, 그 기본값이 `single`(1코어)임을 config.json 부재 + SDK fallback + 실측 3중으로 확인했다.

### 8코어 llama1b는 어디서 오는가 (그리고 왜 여기엔 없는가)

`global8` 기본값은 model-zoo의 **benchmark CLI** (`mblt_model_zoo/benchmark/transformers/benchmark_text_generation_models.py:1070`)에만 존재한다. 저장소 전수 검사 결과:

```
grep -rn "core_mode|benchmark|global8|infer_mode"  (backup/ 제외, *.py *.sh *.yaml *.json *.md)
  → core_mode      : 0건
  → benchmark CLI  : 0건
  → global8        : vision 로드 3곳뿐 (model_processors.py:171,259 / profile_models.py:61 / schedule_generator.py:175)
```

**benchmark CLI는 이 파이프라인의 어느 단계에서도 호출되지 않는다.** 8코어 llama1b가 데이터에 섞일 물리적 통로가 없다.

---

## 3. 불일치 여부 (항목 C)

### C-1. 경로 간 불일치 → **없음**

| 경로 | llama1b 코어 |
|---|---|
| 정적 프로파일 (`profile_models.py`) | 1 |
| 학습 데이터 수집 (`run_collection.py` → 실행기) | 1 |
| 실제 실행 (`schedule_executor_main.py`) | 1 |

세 경로 전부 `LLMEngine`을 거치고, `LLMEngine`은 **코어 파라미터를 아예 노출하지 않는다** (`llm_engine.py:42` 시그니처: `(model_name, device, max_new_tokens)`). 경로별로 다른 값을 줄 방법 자체가 없다.

### C-2. 데이터 내부(시점 간) 불일치 → **없음**

**git의 한계를 먼저 밝힌다.** Mobilint 계열 커밋은 전부 2026-07-10 17:16~17:24 에 몰아서 만들어졌고(`53083c5`…`59b4cd3`), 모든 데이터 파일은 그보다 **앞선** 07-09 09:57 ~ 07-10 12:41 에 생성됐다. 즉 데이터 수집은 커밋되지 않은 작업 트리에서 이뤄졌고, **git 히스토리만으로는 수집 도중의 코드 변경을 볼 수 없다.** (`git log -S "core_mode"` 가 잡아낸 4개 커밋은 전부 2026-03~07-04 의 ETRI 시절 것으로, 현재 Mobilint 파일을 **하나도 건드리지 않는다** — 확인 완료.)

그래서 git 대신 mtime + 구조 + 데이터 3중으로 판정했다:

1. **mtime**: `runtime/llm_engine.py` = **07-08 13:39** — 가장 이른 데이터셋(07-09 09:57)보다 **앞선다**. 이후 수정된 적 없고(`git log --follow` 결과 커밋 **1개**), 현재까지 mtime 그대로다. → 전체 수집 기간 동안 LLM 로더는 고정.
2. **구조**: `model_processors.py`가 수집 창 중간(07-09 14:48)에 수정되긴 했다. 그러나 그 파일의 LLM 분기는 `LLMEngine`에 위임할 뿐이고 **`LLMEngine`에는 넘길 코어 인자가 존재하지 않는다.** 뒤집을 스위치 자체가 없다. (vision 분기의 `global8` 리터럴은 `build_vision_npu`의 기본값과 동일해 어느 쪽이든 8코어.)
3. **체크포인트 고정**: llama1b NPU 스냅샷은 캐시에 **단 하나**뿐이다 (`refs/W8` → `f53e1b5a…`, 레지스트리의 `revision: "W8"` 과 일치). config.json이 갈릴 여지가 없다. (`model_registry.py` mtime이 정적 프로파일보다 11분 늦지만, 스냅샷이 하나뿐이라 이 경로도 닫힌다.)
4. **데이터**: 8시간 간격으로 **서로 다른 스크립트 실행**이 만든 두 NPU 데이터셋이 같은 값을 낸다 — §4 참조.

### C-3. 실행 시 불일치 → **없음**

학습 데이터 수집기가 실행기(`schedule_executor_main.py`)를 **그대로 서브프로세스로 호출**한다 (`run_collection.py:78-79`). 학습 조건 = 실행 조건이 코드 수준에서 보장된다.

---

## 4. 정량적 확인 (항목 D)

1↔8 코어가 섞였다면 llama1b의 NPU tokens/s 분포가 **이봉(bimodal)** 으로 갈라져야 한다. 갈라지지 않는다.

**`cpu_npu/performance.json` (07-10 00:15), llama1b @ NPU, n=70:**

```
min 10.32   median 12.09   max 13.83   CV 7.2%   spread(max/min) = 1.34x
정렬된 인접값 사이의 최대 간격 = 0.25 tok/s  (13.58 ↔ 13.83)
```

간격 0.25 tok/s — **분리된 고성능 군집이 존재하지 않는다.** 완전한 단봉 분포다.

**(workload, rate) 그룹별로 쪼개도 설명되지 않는 분산 없음** (20개 그룹, 그룹 내 CV 1.8%~8.2%):

| workload | rate 1x / 2x / 3x / 4x 의 median tok/s |
|---|---|
| yolo11n | 12.55 / 12.93 / 11.96 / 11.81 |
| yolo11s | 12.55 / 12.83 / 11.63 / 12.27 |
| yolo11m | 13.38 / 12.35 / 11.96 / 11.22 |
| yolo11l | 12.20 / 13.31 / 11.59 / 11.02 |
| yolo11x | 12.54 / 12.01 / 11.23 / 11.32 |

rate가 올라갈수록 완만히 내려가는 것 외에는 구조가 없다. 정상이다.

**시점이 다른 두 NPU 데이터셋의 일치 (C-2의 핵심 증거):**

| 데이터셋 | 수집 시각 | n | median | 범위 |
|---|---|---|---|---|
| `train/performance_ratesweep.json` | 07-09 16:22 | 19 | **12.76** | 12.18 – 13.33 |
| `cpu_npu/performance.json` | 07-10 00:15 | 70 | **12.09** | 10.32 – 13.83 |

8시간 간격, 별개의 수집 실행인데 **같은 구간**이다. 코어 구성이 중간에 바뀌었다면 나올 수 없는 결과다.

**정적 프로파일(21.94 tok/s) vs 경합 윈도우(~12.5 tok/s) 의 차이는 코어 수 차이가 아니다.** 정적 프로파일은 **단독 실행**(`profile_models.py`, iters=3, 경합 없음)이고 나머지는 다중 모델 경합 상태다. qwen2_vl도 정확히 같은 패턴을 보인다: 단독 16.01 → 경합 6.6~7.2 (2.3배 하락), llama1b 21.94 → 12.5 (1.75배 하락). 두 모델 모두 같은 `LLMEngine`·같은 1코어 설정으로 측정됐으므로, 이 격차는 **시분할 경합에 의한 저하**로 일관되게 설명된다.

> 과도한 통계적 주장은 하지 않는다. 여기서 말할 수 있는 것은 **"1↔8 코어 전환의 신호가 데이터에 없다"** 는 것뿐이며, 그것으로 충분하다.

---

## 5. CLEAN이지만 짚어야 할 별도 이슈 (confound 아님, 설계 문제)

조사 범위 밖이지만 발견된 김에 기록한다. **재학습 사유는 아니지만 논문 기술 사항이다.**

**llama1b는 NPU 8 local core 중 1개만 쓰고 있다.** 반면 vision 모델은 전부 8개를 요구한다 (`global8`). 즉 현재 "llama1b on npu" 숫자는 **1/8 코어로 돌린 llama1b**의 성능이다. 이것이 *일관되게* 그러했기 때문에 예측기의 내적 타당성(internal validity)은 깨지지 않는다 — 학습·프로파일·실행이 모두 같은 조건이다. 다만:

- 논문에 "llama1b on NPU"라고 쓸 때 그것이 **1코어 구성**임을 명시해야 한다. 독자는 8코어 NPU를 다 쓴 수치로 읽는다.
- llama1b의 `.mxq`는 `Single / Global4 / Global8` 을 지원한다 (`CORE_USAGE_REPORT` §1.3). 즉 8코어로 올릴 **여지가 있다**.
- 만약 llama1b를 `global8`로 바꾸고 싶다면, 그것은 **모든 NPU 데이터의 재수집 + 재학습을 요구한다** (지금 데이터는 1코어 조건에서 유효하므로). 이건 "confound 때문에 재학습"이 아니라 "구성을 바꾸기로 결정했으니 재수집"이다 — 성격이 전혀 다르다.

또한 §3.1(`CORE_USAGE_REPORT`)의 미해결 질문이 그대로 남는다: 현재 데이터는 vision이 전부 8코어를 요구하는 상황이라, **"코어를 겹치지 않게 나누면 실제로 병렬 실행되는가"** 는 이 데이터로 답할 수 없다. 논문에서 코어 분할의 이득을 주장하려면 별도 실험이 필요하다.

---

## 6. 잔여 불확실성 (솔직하게)

1. **git 히스토리가 수집 이후에 스쿼시됐다** (§C-2). 따라서 "수집 도중 코드가 안 바뀌었다"는 것은 git이 아니라 mtime·구조·데이터로 뒷받침된 결론이다. 세 증거가 모두 같은 방향을 가리키므로 판정을 바꿀 정도는 아니라고 본다. 다만 git 단독으로 증명된 것은 아니다.
2. `model_processors.py`의 07-09 14:48 수정 내용을 git으로 되짚을 수 없다. LLM 경로에는 코어 스위치가 없어 무관하고(§C-2.2), vision은 리터럴이든 기본값이든 `global8`이라 코어 **수**는 8로 동일하다. 다만 vision이 한때 `single`이었을 가능성을 git으로 배제하지는 못했다 — 배제하더라도 vision `single`은 **여전히 8 local core**이므로(zoo가 core_ids 8개를 전부 나열) 코어 수 confound는 되지 않는다. (실행 semantics는 달라지지만 이는 llama1b 조사 범위 밖이다.)

---

## 최종 판정

> # CLEAN
>
> llama1b는 **프로파일 생성·학습 데이터 수집·실제 실행 전 경로에서 동일하게 1코어(`single`, Cluster0/Core0)** 로 로드되었다. 8코어(`global8`) llama1b를 만들어내는 benchmark CLI는 이 저장소에서 **한 번도 호출되지 않았고**, `core_mode` 를 건드리는 코드도 **존재하지 않는다**. 시점이 다른 두 NPU 데이터셋이 동일한 처리량 구간을 재현하며, 라벨 분포는 완전한 단봉이다 (인접값 최대 간격 0.25 tok/s).
>
> **y1/y2/y3 라벨과 정적 프로파일에 코어 구성 confound는 없다. 이 사유로 인한 `deploy_cpu_npu` 재학습은 불필요하다.**
>
> 단, 현재 수치는 **1코어 llama1b**의 성능이라는 점을 논문에 명시할 것 (§5).
