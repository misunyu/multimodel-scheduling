<!-- 작업 13 — 읽기 전용 확인 3건. 코드·데이터·원고 미수정.
     검증 스크립트: checks.py (줄 근거 55건 + 검산 24건) / facts.json / run_manifest.json
     저장된 산출물과 소스만 읽었다. 학습·수집·프로파일링 실행 없음. -->

# r13 원고 마무리용 확인 3건

---

## 1. Oracle ratio의 평균 방식 — 원고 신규 정의문 **일치**

### 판정

> **(a) 그룹별 비율의 산술평균이다.** 분모는 그 그룹의 **best measured score**,
> 집계 대상은 **greedy 산출 배치가 수집 조합에 존재하는 그룹만**.

원고의 신규 문장 — *"the mean, over groups with measured greedy outputs, of the ratio
between the greedy-selected placement's score and the best measured score in its group"* —
과 **세 요소가 모두 일치**한다(평균의 대상 = 그룹별 비율, 분모 = 그룹 최고 측정 점수,
모집단 = 측정 산출이 있는 그룹). **수정 불필요.**

### 수식

Table 5의 각 셀(플랫폼 × 정렬변형 × {norm., raw})은

$$
\text{Oracle ratio} \;=\; \frac{1}{|G_{\text{measured}}|}\sum_{g \in G_{\text{measured}}}
\frac{S\big(\mathbf{x}^{\text{greedy}}_g\big)}{\displaystyle\max_{\mathbf{x}\in C_g} S(\mathbf{x})}
$$

- $C_g$ = 그룹 $g$의 **수집된** 후보 배치 집합
- $S(\cdot)$ = **측정** 점수 ($\alpha{=}0.3,\beta{=}1.0$). norm. 열은 그룹 정규화 점수,
  raw 열은 원시 측정 총계 점수 — 두 열은 **같은 식에 다른 점수 기저**를 넣은 것이다
  (`greedy_baseline.py:69` / `:71`)
- $G_{\text{measured}}$ = greedy 산출 배치가 $C_g$ 안에 있는 그룹
- **비율을 먼저 만들고 그다음에 평균**한다. 점수 합의 비율이 아니다.

### 코드 근거

| 요소 | 위치 | 코드 |
|---|---|---|
| 그룹 oracle | `greedy_baseline.py:93` | `oracle_n, oracle_r = g["s_norm"].max(), g["s_raw"].max()` |
| greedy 배치 조회 | `:92` | `match = g[g["accel_set"] == accel_set]` |
| **수집 그룹만** | `:94` / `:113-115` | `if len(match):` … `else: … ["unc"].append(...)` — 미수집은 `rn`/`rr`에 **들어가지 않는다** |
| 그룹별 비율 적재 | `:107-108` | `st["rn"].append(row["s_norm"] / oracle_n)` |
| 최종 평균 | `:134` | `np.nanmean(st['rn'])` / `np.nanmean(st['rr'])` |
| Top-1의 oracle | `:96` | `abs(row["s_norm"] - oracle_n) < 1e-9` — **비율의 분모와 동일한 값** |

### 재계산 대조 (원고 Table 5 vs 세 가지 후보 정의)

| 셀 | 원고 (norm./raw) | **(a) 그룹비율 평균** | (b) 점수 합의 비율 | (c) 45그룹 전체 평균 |
|---|---|---|---|---|
| GPU / load-factor | 0.575 / 0.669 | **0.575 / 0.669** ✅ | 0.632 | 0.524 |
| GPU / speedup | 0.734 / 0.786 | **0.734 / 0.786** ✅ | 0.757 | 0.538 |
| NPU / load-factor | 0.590 / 0.669 | **0.590 / 0.669** ✅ | 0.647 | 0.472 |
| NPU / speedup | 0.793 / 0.760 | **0.793 / 0.760** ✅ | 0.832 | 0.581 |

(a)만 소수 셋째 자리까지 정확히 재현하고 (b)·(c)는 네 셀 모두 어긋난다 —
원고 문장이 (a)임이 **배타적으로** 확정된다. Top-1 값과 분모(41/33/36/33, 미수집
4/12/9/12)도 함께 재현했다.

### 문장에 덧붙이면 더 정확해지는 것 (선택)

1. **"score"가 측정 점수임을 명시.** 현 문장은 greedy가 예측 점수를 쓴다고 오해될 수 있다.
   greedy는 점수를 전혀 예측하지 않고 배치만 내며, 그 배치의 **측정** 점수를 조회한다.
   → *"…the ratio between the **measured** score of the greedy-selected placement and the
   best measured score in its group."*
2. **norm./raw 두 열의 차이**가 점수 기저(그룹 정규화 vs 원시 총계)임을 캡션에 한 줄.
3. **비율은 1로 상한되지만 하한이 0이 아니다.** 정규화 기저에서 최소 $-0.156$, 45개 중
   7개 셀이 음수다($y_1{=}0$ 부근에서 $-\alpha y_2$가 남기 때문). "ratio"라는 이름이
   \[0,1\] 을 암시하므로, 값의 범위를 문제 삼는 리뷰어가 있다면 이 점을 알고 있어야 한다.
   raw 기저는 0.143–1.000으로 전부 양수다.

### 구현상 정확성 확인

- `match.iloc[0]`(`:95`)이 임의 선택이 아님: 한 그룹 안에서 `accel_set`(가속기에 올라간
  모델 집합)은 **중복이 없다**(1080창 전수 확인, 중복 그룹 0). 매칭은 항상 정확히 1행.
- `np.nanmean`이지만 **NaN은 한 건도 발생하지 않는다**(네 셀 모두 0건). 따라서 분모는
  "측정 산출이 있는 그룹 수"와 정확히 같고, 문장의 모집단 서술이 그대로 성립한다.

---

## 2. 실행 설정 명세 (#15용 부록 표)

수집은 두 플랫폼이 **같은 호스트·같은 스케줄·같은 실행기**를 쓰고 가속기만 교체한다.
값이 같은 항목은 한 칸에 병합했고, 소스에서 확정 불가한 것은 **UNKNOWN**으로 남겼다.

| 항목 | GPU 플랫폼 | NPU 플랫폼 |
|---|---|---|
| **호스트 CPU** | **Intel Core Ultra 9 285K** (24 스레드) — `PRE_COLLECTION_FINAL.md:115`(2026-07-15 기록) + `/proc/cpuinfo` 일치 | 〃 (동일 호스트) |
| 가속기 | NVIDIA RTX 5090 | Mobilint MLA100 (Aries) |
| 측정 창 | 20 s 워밍업 + **180 s** 측정 (`collect_run.py:25,67`, `unified_viewer.py:90`) | 〃 |
| 입력 영상 | `stockholm_1280x720.mp4` — **1280×720** (`unified_viewer.py:68`) | 〃 |
| **vision 런타임** | ONNX Runtime 1.20.1, `CUDAExecutionProvider`(+CPU 폴백 리스트) (`model_processors.py:82`) | `mblt_model_zoo.vision`, 사전 컴파일 `.mxq`, `infer_mode="global8"` (`model_processors.py:280`, `runtime/mobilint_vision.py:23,35`) |
| **vision precision** | ONNX **FP32** | **INT8** 양자화 `.mxq` (`NPU_POSTPROCESS_UNIFICATION.md:114`) |
| vision 입력 크기 | YOLO11 n/s/m/l/x **640×640** letterbox (`model_registry.py:68`), ResNet50·MobileNet-v2 **224×224** (`:85,:95`) | 〃 |
| vision 전처리 | YOLO: 중앙 letterbox, pad 값 114, BGR→RGB, /255 (`model_processors.py:139-143`) · ResNet/MobileNet: 중앙 크롭 → 224 리사이즈 → RGB, ImageNet mean/std 정규화 (`:350-357`) | NPU는 zoo 자체 preprocess(동일 letterbox 규약)에 RGB 프레임을 넣는다 — `model.preprocess(_to_rgb(frame))` (`model_processors.py:298`, 분류는 `:393`) |
| **vision batch size** | **1** — `arr[None, ...]` (`model_processors.py:143`, `:357`) | 1 (zoo 단일 프레임 호출) |
| vision NPU 코어 | — | `global8` = **local core 8개** (`docs/llama1b_core_confound_report.md:70-74`). ※ `global8`은 코어 모드이지 8-bit 정밀도 표기가 아님 |
| vision 워커 스레드 | torch 스레드 **1개로 고정** (`model_processors.py:159-175`) | 〃 |
| **LLM 체크포인트** | `unsloth/Llama-3.2-1B-Instruct` (`model_registry.py:102`) | `mobilint/Llama-3.2-1B-Instruct` **@revision `W8`** (가중치 8-bit, `:103-104`) |
| **VLM 체크포인트** | `Qwen/Qwen2-VL-2B-Instruct` (`:110`) | `mobilint/Qwen2-VL-2B-Instruct` @`main` (`:111-112`) |
| 생성 모델 precision | **torch.float16** (`runtime/llm_engine.py:72`) | NPU: `W8` 사전 양자화. CPU 배치 시에는 **float32** (같은 줄) |
| 생성 모델 NPU 코어 | — | llama1b: `core_mode="single"`, `target_cores=["0:0"]` = **1코어** (`llm_engine.py:44`). qwen2_vl: text single 0:0, vision multi/cluster1 = 4코어 (`llm_engine.py:40-42` 주석) |
| **max output tokens** | LLM **64**, VLM **32** (수집 실행기 기본값, `model_processors.py:529`, `:543`) | 〃 |
| **decoding** | `model.generate(..., do_sample=False)` — **greedy**. temperature·top_p·num_beams 미지정(기본값) (`llm_engine.py:135`) | 〃 |
| prompt (LLM) | `LLM_PROMPTS` **4개**를 생성마다 순환 (`prompts[idx % len(prompts)]`, `model_processors.py:499`; 목록은 `llm_engine.py:27-32`) | 〃 |
| prompt (VLM) | 고정 `DEFAULT_PROMPT` = *"Describe what is happening in one short sentence."* + 영상 프레임 1장 (`llm_engine.py:26`, `model_processors.py:496-497`) | 〃 |
| **prompt 길이(토큰)** | **UNKNOWN** — 채팅 템플릿 적용 후 길이가 어디에도 기록돼 있지 않다 (`_build_inputs`가 런타임에 계산) | UNKNOWN |
| 요청률 (infps) | 스케줄 YAML의 뷰별 `infps` = rate_factor × 모델 baseline (`collect_cpu_{gpu,npu}.yaml`) | 〃 |
| Python | 3.10.12 (`sample_profiling_data.json:profiling_runs[0].python`) | 〃 |
| torch / torchvision | **2.12.0+cu130** / 0.27.0 (`requirements.txt`, 07-14 실행 기록) | 〃 |
| onnxruntime-gpu | **1.20.1** (`requirements.txt`, 07-14 실행 기록) | 〃 |
| numpy / OpenCV | 2.2.6 / 4.13.0 (headless) (07-14 실행 기록) | 〃 |
| transformers | requirements 핀 **5.9.0** — 단, 현재 호스트는 5.6.2. **수집 시점 값은 UNKNOWN** | 〃 |
| Mobilint 런타임 | — | `mobilint-qb-runtime` **1.2.0** (`requirements.txt` 주석, 현재 호스트 `qbruntime v1.2.0`) |
| Mobilint 컴파일러 | — | `qbcompiler` — 로컬 wheel, **버전 UNKNOWN** (requirements가 `<ver>`로 비워 둠) |
| mblt-model-zoo / tracker | — | git 설치라 핀 없음. 현재 호스트 1.5.1 / 0.2.5, **수집 시점 값 UNKNOWN** |
| **CUDA / 드라이버** | torch 휠이 **cu130**(CUDA 13.0). ONNX Runtime의 CUDA 빌드 버전과 **NVIDIA 드라이버 버전은 수집 시점 기록 없음 → UNKNOWN** (현재 호스트 드라이버 580.159.03) | — |
| 실행 환경 변수 | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `QT_QPA_PLATFORM=offscreen`, `DISABLE_VISUALIZATION=1` (`collect_run.py:64-67`) | 〃 |
| 조합 실행 순서 | 세트 그룹 내 무작위, `random.seed(20260715)` 고정. 20 run마다 앵커(전부-가속기 S3) 재측정 (`collect_run.py:32-40, 99-108`) | 〃 |

### ⚠ 원고와 어긋나는 항목 1건

**§142의 "Intel Core i7-13700K"는 잘못된 것으로 보입니다.** 수집 직전인 2026-07-15에
작성된 `PRE_COLLECTION_FINAL.md:115`가 CPU 부하 A/B 실험의 대상 호스트를
**Intel Core Ultra 9 285K (최대 5100 MHz)** 로 기록하고 있고, 현재 호스트의
`/proc/cpuinfo`도 `Intel(R) Core(TM) Ultra 9 285K`(24 스레드)입니다. 저장소 어디에도
i7-13700K를 가리키는 기록이 없습니다. 다만 **수집 당시 호스트가 교체되지 않았다는 것을
직접 증명하는 기록은 없으므로**, 최종 확정은 원고 저자 확인이 필요합니다.

### UNKNOWN 목록 (추정 금지 원칙에 따라 비워 둔 것)

1. 생성 모델 prompt의 토큰 길이
2. NVIDIA 드라이버 버전 (수집 시점)
3. ONNX Runtime의 CUDA/cuDNN 빌드 버전
4. `qbcompiler` 버전
5. `mblt_model_zoo` / `mblt_tracker` 버전 (수집 시점)
6. `transformers` 버전 (수집 시점 — requirements 핀과 현재 호스트가 불일치)
7. NPU `.mxq`의 모델별 양자화 비트폭 상세 (문서는 vision을 int8, llama1b를 W8로만 기술)

---

## 3. Candidate-set 정책 일관성 — **판정: "동일 규칙 · 코드 이중화 · 샘플링 단계만 상이"**

"동일 정책 단일 소스"도 "경로별 상이"도 아닌 중간이라, 원고 문장을 **그대로 넣을 수는
없습니다.** 아래에 근거와 대체 문장을 제시합니다.

### 세 경로의 실제 구성

| 경로 | 후보를 만드는 코드 | 샘플링 |
|---|---|---|
| **profiling(수집)** | `xgboost_model/gen_collection_schedules.py` — `all_placements` `:45` → `combos_for` `:91` | S6–S10에 **층별 쿼터 적용** (`stratify:58`) |
| **evaluation** | 별도 열거 **없음** — 수집이 남긴 540창을 그대로 읽는다 (`analysis_common.load_windows`) | 수집의 결과를 상속 |
| **deployment** | `generate_schedules.py` — `enumerate_placements` `:58`, 호출은 `best_deploy_finder_executor.py:675,696` | **없음. 전 feasible 열거** |

### 확인된 사실

1. **feasibility 규칙은 단일 소스다.** 두 열거 함수 모두 `model_registry.allowed_devices`
   (= `DEVICE_CONSTRAINTS`)만 참조한다 (`gen_collection_schedules.py:42`,
   `generate_schedules.py:69`). VLM 고정 제약이 두 벌로 존재하지 않는다.
2. **두 열거 함수의 결과 집합은 완전히 동일하다.** 15세트 × 2플랫폼 = 30조합 전부에서
   `all_placements`(accel→실제 디바이스명으로 치환)와 `enumerate_placements`가
   **같은 배치 집합**을 낸다(집합 동일성 30/30). 즉 "무엇이 feasible인가"에 대한 답은
   두 경로가 일치한다.
3. **다른 것은 열거 다음 단계다.** 수집은 S6·S7·S8·S9·S10에서 전 feasible의 부분집합만
   내보낸다(32→16, 64→20, 128→24, 32→16, 128→24). 나머지 10세트는 전수.
   배포 경로에는 층화·샘플링 코드가 **없다**(해당 함수 본문 grep 0건) — 선택한 모델의
   feasible 배치를 전부 열거해 예측기로 순위를 매긴다.
4. **코드가 두 벌이다.** 같은 규칙을 두 파일이 각각 구현한다. 지금은 결과가 같지만
   한쪽만 고치면 조용히 갈라진다.
5. **잠재적 발산 지점 1곳**: `generate_schedules.py:70`의
   `choices.append(allowed or ["cpu"])` 폴백은, 모델의 허용 디바이스가 플랫폼 디바이스와
   교집합이 없을 때 **CPU로 되돌린다**. 확인 결과 `enumerate_placements(["qwen2_vl"], ["cpu"])`
   는 `{"qwen2_vl": "cpu"}`를 내지만 `all_placements(["qwen2_vl"])`는 accel로 고정한다.
   현재 GUI가 쓰는 두 프로파일(CPU-GPU / CPU-NPU)에서는 발동하지 않으므로 **실제 산출물에
   영향 없음** — 다만 "정책이 고정되어 있다"는 무조건적 주장은 이 폴백 때문에 성립하지 않는다.

### #4 문장 삽입 가부

원고 문장 *"candidate-set construction policy is fixed across profiling, evaluation, and
deployment"* 는 **그대로는 부정확**합니다. 배포 시에는 층화 샘플링이 적용되지 않아
S6–S10에서 profiling이 본 것보다 **더 많은** 후보를 순위 매기기 때문입니다(예: S8은
수집 24개 vs 배포 128개). 리뷰어가 "그럼 배포 때는 학습에서 못 본 배치도 채점하느냐"고
물으면 답은 "그렇다"입니다.

**제안 문장 (정확한 버전):**

> The feasibility rule that defines the candidate set is shared across profiling,
> evaluation, and deployment: a model may occupy any device its registry entry allows,
> and the VLM is pinned to the accelerator. Profiling additionally sub-samples the
> stratified sets to bound collection cost, whereas deployment enumerates the full
> feasible set; evaluation uses exactly the placements that were profiled.

이 문장이면 (i) 규칙이 고정이라는 원래 취지를 지키면서 (ii) 샘플링이 수집 단계에만
적용된다는 사실을 정확히 담고 (iii) 배포가 미학습 배치를 점수화할 수 있다는 한계도
자연스럽게 드러납니다. Limitations에 한 줄을 더 붙일 수도 있습니다: *"On the stratified
sets, deployment may therefore rank placements that were never measured."*

---

## 4. 검산 요약

`checks.py`가 코드·문서 줄 근거 **55건**과 수치·집합 검산 **24건**을 재확인했고 전부 통과했다.

| 검산 | 결과 |
|---|---|
| Table 5 네 셀 모두 "그룹비율 평균" 정의로 소수 셋째 자리까지 재현 | ✅ |
| 대안 정의 (합의 비율)는 네 셀 모두 불일치 → (a) 배타적 확정 | ✅ |
| Top-1 값·분모(41/33/36/33)·미수집(4/12/9/12) 재현 | ✅ |
| 비율 NaN 0건 → 분모 = "측정 산출이 있는 그룹" 정확히 성립 | ✅ |
| 그룹 내 `accel_set` 중복 0 → `match.iloc[0]` 유일 | ✅ |
| 두 열거 함수 결과 집합 동일 (15세트 × 2플랫폼 = 30/30) | ✅ |
| 수집이 부분집합을 내는 세트 = 층화 5세트와 정확히 일치 | ✅ |
| 배포 경로에 샘플링 코드 없음 | ✅ |
| `enumerate_placements` CPU 폴백 / `all_placements` accel 고정 동작 확인 | ✅ |

기계 판독 전체 수치는 `facts.json`.
