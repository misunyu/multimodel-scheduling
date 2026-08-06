<!-- 작업 12 — 읽기 전용 조사. 코드·데이터·원고 미수정.
     입력: profile_models.py, runtime/llm_engine.py,
           xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json,
           xgboost_model/deploy_selector_xgb_suite.py
     검증 스크립트: check.py / facts.json / run_manifest.json -->

# `static_load` 필드의 실체 확정

## 판정 (한 줄)

> **`load` = 모델을 그 디바이스에 적재해 추론 가능한 상태로 만들기까지의 벽시계 시간
> (모델/세션 초기화 시간), 단위 = 밀리초(ms).**
> 실행 중 부하 지표가 **아니다**. 제3의 것도 아니다.

`view.load_factor`의 "load"(= 제공 부하 비율)와는 **이름만 같고 완전히 별개의 값**이며,
후자는 `load` 필드를 전혀 참조하지 않는다(§3).

---

## 1. 생성 측 — `load` 값을 만드는 코드

`profile_models.py`가 유일한 생성자다
(`--out xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json`,
`profile_models.py:29`). 값이 JSON에 실리는 지점은 두 곳뿐이다:

| 대상 | 기록 지점 | 값의 출처 |
|---|---|---|
| vision 모델 | `row[f"{dev}_load"] = round(load_ms, 3)` — `profile_models.py:229` | `profile_vision()` 의 첫 반환값 `:86-129` |
| 생성(LLM/VLM) 모델 | `row[f"{dev}_load"] = round(load_ms, 3)` — `profile_models.py:234` | `profile_generative()` `:132-139` → `LLMEngine.profile()["load_ms"]` |

### 1-1. 무엇을, 어떤 구간으로 재는가

**vision · NPU** (`profile_models.py:90-92`):

```python
t0 = time.time()
model = build_vision_npu(model_name, infer_mode="global8")
load_ms = (time.time() - t0) * 1000.0
```

**vision · CPU/GPU** (`profile_models.py:114-116`):

```python
t0 = time.time()
sess = ort.InferenceSession(resolve_onnx_path(model_name), sess_options=so, providers=providers)
load_ms = (time.time() - t0) * 1000.0
```

**생성 모델** — `LLMEngine` 생성자 안에서 측정된 값을 그대로 전달한다
(`runtime/llm_engine.py:88-103`):

```python
t0 = time.time()
...  AutoProcessor/AutoTokenizer.from_pretrained(...)  +  AutoModel*.from_pretrained(...)
if host == "cuda":
    self.model = self.model.to("cuda")
self.model.eval()
self.load_ms = (time.time() - t0) * 1000.0
```

`profile()`이 `{"load_ms": self.load_ms, ...}`로 반환하고(`llm_engine.py:246`),
`profile_models.py:139`가 그 값을 그대로 꺼낸다.

**판정 근거 요약**

1. 타이머 구간이 전부 **"모델/세션 객체를 만드는 호출"** 하나만 감싼다 —
   NPU는 `build_vision_npu`, CPU/GPU는 `ort.InferenceSession` 생성자, 생성 모델은
   `from_pretrained` + `.to("cuda")` + `.eval()`. 추론 호출은 이 구간 **밖**이다.
2. 단위는 명시적으로 ms다: `(time.time() - t0) * 1000.0`, 그리고 `profile_models.py:12-14`
   스키마 주석의 `# ms`.
3. 조회 API로 얻은 부하 지표(utilization, 메모리 점유율 등)가 아니다 — `nvidia-smi`,
   `mobilint` 상태 조회, `psutil` 같은 호출이 이 경로에 전혀 없다.
4. 이름의 "load"는 **동사 load(적재하다)** 이지 명사 load(부하)가 아니다.

### 1-2. 측정 방식의 성질 (해석 시 주의)

`infer`와 달리 `load`는 **워밍업 없는 1회 측정**이다. `profile_vision`은 추론을
warmup 3회 + 본측정 15회 평균으로 재지만(`profile_models.py:98-102`, 실행 기록
`vision_warmup: 3, vision_iters: 15`), `load`는 반복도 평균도 없는 단발
`t0..t1` 하나다. 생성 모델도 `LLMEngine`을 한 번 만들 때의 값을 그대로 쓴다
(`profile()`의 warmup/iters는 prefill·tokens/s에만 적용, `llm_engine.py:237-250`).
따라서 **런타임 컨텍스트 최초 초기화 비용이 "그 시점에 처음 적재된 모델"에 흡수된다** —
§2에서 실제로 관측된다.

---

## 2. 값 확인 — 판정과 정합하는가

파일: `sample_profiling_data.json` (sha256 `7fec0eea36e9eb98…`, 9모델 × 3디바이스 = 27쌍,
결측 0건). 기록된 실행은 1건(`run_20260714_164004`, vision 7모델, host `5090pc`,
git `521e1d5`)이고 **생성 모델 2행(llama1b, qwen2_vl)에 대응하는 실행 기록은 파일에 없다**
— `profile_models.py:174-181`이 기존 파일과 병합하므로 이전 실행에서 남은 행으로 보인다.
(그 실행의 조건은 파일에서 확인 불가 → **미확인**.)

### 대표 3쌍

| # | (모델, 디바이스) | `load` (ms) | 판정과의 정합 |
|---|---|---|---|
| 1 | `mobilenet_v2` / **cpu** | **15.052** | 27쌍 중 두 번째로 작다. 13.99 MB ONNX를 CPU EP로 여는 비용 — 적재 시간으로 자연스럽다 |
| 2 | `yolo11x` / **cpu** | **295.199** | vision 최대. 228.06 MB로 ONNX 중 가장 큰 모델이며, CPU 적재가 GPU(109.2)·NPU(134.1)보다 비싸다 |
| 3 | `qwen2_vl` / **npu** | **1674.867** | **27쌍 전체 최대.** 가장 큰 모델(VLM) × NPU — 지시문이 "적재 시간이라면 자연스러워야 한다"고 한 바로 그 조합 |

전체 최소 3쌍: `mobilenet_v2/gpu` 12.891, `mobilenet_v2/cpu` 15.052, `yolo11n/cpu` 18.713.
전체 최대 3쌍: `llama1b/gpu` 1033.189, `llama1b/npu` 1449.563, `qwen2_vl/npu` 1674.867.
**생성 모델 6쌍(665~1675 ms)이 vision 21쌍(12.9~295.2 ms)을 전부 넘는다** — 모델 규모
순서와 일치한다.

### 정량 확인: `load`는 모델 **크기**를 따라가지 추론 비용을 따라가지 않는다

vision 7모델의 ONNX 파일 크기(`resolve_onnx_path` 기준: yolo11n 10.74 / mobilenet_v2 13.99 /
yolo11s 38.01 / yolo11m 80.67 / yolo11l 101.74 / resnet50 102.15 / yolo11x 228.06 MB)와의
Spearman 상관:

| 디바이스 | ρ(파일 크기, load) | resnet50 제외 시 | ρ(load, infer) |
|---|---|---|---|
| cpu | **+0.964** | +0.943 | +0.643 |
| gpu | **+0.929** | +0.943 | +0.464 |
| npu | **+0.929** | +0.943 | +0.464 |

크기와는 +0.93~0.96으로 거의 단조, 같은 행의 추론 시간과는 +0.46~0.64에 그친다.
**"적재해야 할 것의 양"에 비례하고 "실행 비용"에는 약하게만 연동된다** — 적재 시간이라는
판정과 정합한다.

### 관측된 이상: `resnet50`의 GPU/NPU load

`resnet50`은 102.15 MB로 `yolo11l`(101.74 MB)과 사실상 같은 크기인데 load는
GPU 281.2 vs 57.5 (4.9배), NPU 171.9 vs 117.9 (1.5배)다. CPU에서는 179.9 vs 117.0으로
크기 순서에 어긋나지 않는다. `resnet50`은 기록된 실행의 **모델 목록 첫 번째**이고
(`profiling_runs[0].models[0]`), 이상이 나타난 두 디바이스가 바로 **일회성 런타임 컨텍스트
초기화(CUDA 컨텍스트, NPU 디바이스 오픈)가 있는 디바이스**다. §1-2의 "1회 측정이라 최초
초기화 비용을 첫 모델이 떠안는다"는 성질과 맞아떨어진다.

다만 이는 **저장된 데이터로부터의 추론이지 분리 측정으로 확인한 것은 아니다** —
초기화 비용만 따로 잰 기록이 파일에 없다. 판정(= 적재 시간) 자체는 바뀌지 않고,
"resnet50의 GPU/NPU load 값은 순수 적재 시간보다 부풀려져 있을 가능성이 높다"는
단서로만 남긴다.

---

## 3. 소비 측 — `static_load_sel`과 `load_factor`는 별개다

### 조회 경로

```
sample_profiling_data.json
  └─ load_static_profiles()            suite:101-124   → {model: {dev: {"load","infer","tokens"}}}
       "load" ← row[f"{dev}_load"]     suite:118       (_f("load"), float 변환, 실패 시 NaN)
  └─ _device_static(model, dev, S)     suite:126-135   → (infer_ms, load_ms, tokens_per_s)
  └─ _view_features(...)               suite:250-279
       s_infer, s_load, s_tokens = _device_static(...)          :253
       "view.static_load_sel": s_load                           :273
```

즉 `view.static_load_sel`은 JSON의 `{dev}_load`를 **단위 변환도 스케일링도 없이 그대로**
싣는다(결측만 0.0으로 대체, `suite:255-257`). 단위는 ms 그대로다.

### `view.load_factor`는 `s_load`를 쓰지 않는다 — 코드로 확정

`_view_features:260-261`:

```python
capacity = (1000.0 / s_infer) if s_infer > 0 else 0.0
load_factor = (infps / capacity) if capacity > 0 else 0.0
```

`capacity`와 `load_factor`는 **`s_infer`(추론 시간)만** 참조한다. `s_load`가 등장하는
곳은 `_view_features` 전체에서 `"view.static_load_sel": s_load` 한 줄뿐이다.
두 값은 다음과 같이 완전히 다르다:

| | `view.static_load_sel` | `view.load_factor` |
|---|---|---|
| 의미 | 모델 **적재** 소요 시간 | 제공 입력률 / 무경합 처리 용량 = **제공 부하** 비율 |
| 단위 | ms | 무차원 비율 |
| 원천 | 정적 프로파일의 `{dev}_load` | 스케줄의 `infps` ÷ (1000 / `{dev}_infer`) |
| 계산에 `{dev}_load` 사용? | 예 (그 자체) | **아니오** |

**다른 "load" 이름들도 전부 `{dev}_load`와 무관하다**:

- `greedy_baseline.py:37-44` — `cpu_infer, _, _ = _device_static(...)`처럼 튜플의 **0번째
  (infer)만** 언팩하고 1번째(load)는 버린다. `load_factor = infps / capacity_fps`,
  `capacity_fps = 1000 / static_infer` (`greedy_baseline.py:4-5`).
- `train_decomposed.py:61-67` — 같은 방식으로 `s_infer, _, _`만 쓰고
  `"lf": infps / cap`를 만든다. 그 `lf`를 합산한 것이 `co.same_dev_load_sum` /
  `co.other_dev_load_sum` (`:75-77`)이므로 이 둘도 **적재 시간과 무관**하다.

정리하면 저장소에서 `{dev}_load`(적재 시간)를 실제로 소비하는 곳은
**`view.static_load_sel` 계열 3개 feature(sum/mean/max)뿐**이고, 이름에 "load"가 들어가는
나머지 feature(`load_factor` 3개, decomposed의 `co.*_load_sum` 2개)는 전부 추론 시간에서
파생된 부하 비율이다.

---

## 4. 문서 정합 — 불일치 목록

수정하지 않고 목록만 보고한다.

| 문서 | 위치 | 서술 | 판정과 |
|---|---|---|---|
| `runs/20260804_180759_feature_spec/feature_spec.md` | :44, :56, :68 | "배치된 디바이스의 **단독 모델 로드 시간** (load). 결측시 0", 단위 "ms (float)", 범주 "단독 프로파일" | **일치** |
| 〃 | :47, :59, :71 | `load_factor` = "view.infps / view.capacity_fps", 단위 "비율", 범주 "집계 부하" | **일치** (별개 값으로 분리 기술됨) |
| 〃 | :79, :84 | 범주 집계에서 `static_load_sel`은 "단독 프로파일", `load_factor`는 "집계 부하"로 분리 | **일치** |
| `xgboost_model/MODEL_GUIDE.md` | :390-393 | "`view.static_{infer,load,tokens}_sel` · `view.capacity_fps` · `view.load_factor`가 **선택된 디바이스의 단독 프로파일 값**을 담는다" | **부분 불일치** (아래) |
| 〃 | 2-A 표 11행 | 정적 프로파일 = "모델×디바이스 단독 추론시간/**로드시간**/토큰속도" | **일치** |

**불일치 1건 (MODEL_GUIDE.md:390-393).** 다섯 feature를 한 묶음으로 "단독 프로파일 값을
담는다"고 서술했는데, `view.load_factor`는 정적 프로파일 값이 아니라 **스케줄의 infps와
정적 추론시간으로 계산한 파생 비율**이다(`capacity_fps`도 `1000/s_infer`인 파생값). 해당
문단의 논지(전이 시 평가 플랫폼 기준이어야 한다)는 다섯 모두에 성립하므로 **결론은
영향받지 않으며**, "담는다"를 "정적 프로파일에서 유래한다" 정도로 좁히면 해소된다.
이 작업에서는 수정하지 않았다.

**불일치 아님이지만 기록해 둘 것**: 어느 문서도 `load`를 "실행 중 부하"로 서술하고 있지
않다. 즉 이번 조사로 뒤집히는 서술은 없다.

---

## 5. 미확인

- 생성 모델 2행(llama1b, qwen2_vl)의 `load` 값을 만든 **실행 기록이 파일에 없다**
  (`profiling_runs`에 vision 실행 1건만). 측정 시점·호스트·코드 리비전 **미확인**.
- `resnet50`의 GPU/NPU load 팽창이 최초 컨텍스트 초기화 때문이라는 것은 정황 추론이며,
  초기화 비용을 분리 측정한 기록이 없어 **확정 불가**.
- `build_vision_npu`가 NPU 적재 구간에서 무엇까지 수행하는지(예: 컴파일 캐시 여부)는
  이 조사 범위 밖 — **미확인**.
