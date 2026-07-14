# 수렴 실험 선행 점검 (Pre-Experiment Checks)

날짜: 2026-07-14 · 코드: `521e1d5` 기준 + 본 문서에서 기술한 수정
장비: RTX 5090 (driver 580.159.03) / Mobilint Aries NPU / onnxruntime 1.20.1 / torch 2.12.0+cu130

---

## 0. 요약 — **수렴 실험 진입 불가. 블로커 1건.**

세 항목을 점검하는 과정에서 **경로 해석 버그**를 발견했다. 이 버그 때문에
**CPU/GPU에서 실행된 모든 yolo 변종이 실제로는 yolo11s였다.** 정적 프로파일과
기존 수집 데이터의 CPU/GPU 항목이 오염되어 있다.

| 항목 | 판정 |
|---|---|
| 1. mobilenet_v2 프로파일 | ✅ 생성 완료 (조건 기록 포함) |
| 2. GPU 실행 경로 | ✅ **진짜 GPU** (폴백 아님) |
| 3. BGR→RGB의 detection **측정값** 영향 | ✅ **영향 없음** (통계적으로 유의하지 않음) |
| **0. (신규) yolo 경로 해석 버그** | 🔴 **블로커** — 기존 CPU/GPU 수집 데이터 재수집 필요 |

---

## 🔴 0. 신규 발견: 모든 yolo가 CPU/GPU에서 yolo11s로 실행됨

### 원인
`utils._canonical_name()` 이 **등록된 모델명까지 무조건** `model_registry._normalize()` 에
통과시켰고, `_normalize` 는 다음과 같다:

```python
if "yolo" in low:
    return "yolo11s"          # ← 이름에 "yolo"가 들어가면 전부 yolo11s
```

따라서 수정 전:

```
resolve_onnx_path("yolo11x") -> models/onnx/yolo11s.onnx      # ← 항상 yolo11s
resolve_mxq_path ("yolo11x") -> models/mobilint/yolo11s.mxq
```

`_normalize` 는 레거시 별칭(`yolov3`, `tiny` 등)용인데, 실제 모델명을 덮어써 버렸다.

### 영향 범위
- **CPU / GPU 경로**: `run_detection_process` 의 비-NPU 분기가 `resolve_onnx_path()` 를
  쓰므로, **yolo11n/m/l/x 를 스케줄해도 실제로는 yolo11s 가 추론**되었다.
- **NPU 경로**: 영향 **없음**. `build_vision_npu()` 는 레지스트리 spec의 `mxq` 를 직접 읽어
  올바른 파일을 쓴다. (NPU 프로파일만 정상적으로 스케일했던 이유)
- **정적 프로파일**: yolo11n/m/l/x 의 `cpu_infer` / `gpu_infer` / `cpu_load` / `gpu_load` 및
  그로부터 계산되는 `baseline_rate` 가 전부 틀렸다.
- **기존 수집 데이터(contention windows)**: CPU/GPU에 yolo가 배치된 모든 창은
  사실상 yolo11s 를 측정한 것이다.

### 증거 — 수정 전 프로파일은 모델 크기와 무관하게 평탄했다

| 모델 | params | cpu_infer (수정 전) | cpu_infer (수정 후) |
|---|---|---|---|
| yolo11n | 2.66 M | 21.28 ms | **11.73 ms** |
| yolo11s | 9.44 M | 21.27 ms | **22.26 ms** |
| yolo11m | 20.13 M | 22.76 ms | **55.23 ms** |
| yolo11l | 25.38 M | 27.49 ms | **71.59 ms** |
| yolo11x | 56.96 M | **20.24 ms** ← n보다 빠름(불가능) | **141.53 ms** |

수정 전에는 21배 큰 모델이 더 **빨랐다.** 수정 후 CPU·GPU·NPU 모두 단조 증가한다.

### 수정
`utils._canonical_name()` — 레지스트리에 이미 있는 이름은 그대로 반환하고,
모르는 이름만 `_normalize` 로 넘긴다. 레거시 별칭(`yolov3` → `yolo11s`)은 그대로 동작.

### 남은 조치 (실험 전 필수)
- [x] 정적 프로파일 재생성 (아래 1번)
- [ ] **CPU/GPU가 포함된 기존 수집 데이터 폐기 및 재수집** — `deploy_cpu_gpu` 예측기와
      `deploy_cpu_npu` 의 CPU 배치 항목은 잘못된 모델로 학습되었다.
      (본 작업 범위상 대규모 수집은 하지 않음. 결정 필요.)

---

## 1. mobilenet_v2 정적 프로파일

### 기존 모델들의 프로파일 조건 — 확인 결과

| 질문 | 답 |
|---|---|
| 언제 만들어졌나 | 파일 mtime **2026-07-09**, 커밋 `81cae87` (2026-07-10) |
| 어떤 스크립트/커밋 | `profile_models.py` (커밋 `85fbf0b`) — **그 이후 오늘까지 코드 변경 없음** (`git diff 85fbf0b HEAD -- profile_models.py` 결과 없음) |
| 워밍업 / 반복 | `warmup=3`, `iters=15` (vision) / `warmup=1`, `iters=3` (LLM·VLM) |
| 입력 소스 | `stockholm_1280x720.mp4` 의 **첫 프레임** 1장 |
| NPU 모드 | `infer_mode="global8"` |
| 타이밍 범위 | **forward pass만.** 전처리는 루프 밖에서 1회, **후처리는 아예 미포함** |
| **BGR 수정 전인가** | **그렇다.** 프로파일(07-09/07-10) < BGR 수정(07-14, `521e1d5`) |
| 조건 기록이 파일에 남아 있었나 | **아니오. 전혀 없었다** (타임스탬프·반복수·커밋 무엇도 없음) |

### BGR 수정이 기존 프로파일 수치에 미친 영향 — **없음 (구조적으로)**

`profile_vision()` 은 `pre = model.preprocess(frame)` 를 **타이밍 루프 밖에서 1회** 수행하고,
루프 안에서는 `model(pre)` 만 잰다 (`profile_models.py:63-68`). 즉 **색 순서는 전처리의
*내용*을 바꿀 뿐 측정 구간에 포함되지 않는다.** 고정 shape 그래프의 forward 지연은
입력 *내용*에 의존하지 않으므로, BGR/RGB 여부는 프로파일 수치를 바꾸지 못한다.
→ 기존 프로파일이 BGR 시절에 측정된 것은 사실이나, **그 자체로는 수치를 오염시키지 않았다.**
(오염시킨 것은 위 0번의 경로 버그다.)

### 수행

기존과 **동일한 조건**(warmup=3, iters=15, global8, 동일 프레임, forward-only)으로
vision 7종을 **모두 재프로파일**했다. 경로 버그 때문에 yolo 4종의 CPU/GPU 값이 틀렸으므로
mobilenet_v2만 추가할 수 없었다. LLM/VLM 행(`llama1b`, `qwen2_vl`)은 이 버그와 무관하여
(HF 체크포인트를 직접 로드) **그대로 보존**했다.

기존 파일은 `sample_profiling_data.pre_pathbug_backup.json` 으로 백업.

### 결과 (`run_20260714_164004`, git `521e1d5`)

| 모델 | cpu_infer | gpu_infer | npu_infer | baseline_rate |
|---|---|---|---|---|
| **mobilenet_v2** | **1.23 ms** | **0.78 ms** | **1.10 ms** | **814.3** |
| resnet50 | 6.36 | 1.71 | 1.52 | 157.3 |
| yolo11n | 11.73 | 2.46 | 2.92 | 85.2 |
| yolo11s | 22.26 | 3.93 | 3.82 | 44.9 |
| yolo11m | 55.23 | 5.58 | 6.01 | 18.1 |
| yolo11l | 71.59 | 7.87 | 7.74 | 14.0 |
| yolo11x | 141.53 | 11.20 | 14.14 | 7.1 |
| llama1b | (보존) | | | 0.138 |
| qwen2_vl | (보존) | | | 0.217 |

세 디바이스 모두 모델 크기 순으로 **단조 증가** — 물리적으로 타당하다.

### 조건 기록 (신규)

`profile_models.py` 가 이제 측정 조건을 JSON에 남긴다:
- 최상위 `profiling_runs[]`: `run_id`, `timestamp`, `git_commit`, `models`, `devices`,
  `vision_warmup/iters`, `generative_warmup/iters`, `max_new_tokens`, `npu_infer_mode`,
  `frame_source`, `host`, `python`, 라이브러리 버전, 타이밍 범위 주석
- 각 행: `profile_run_id`, `profiled_at`

`profile_run_id` 가 없는 행 = 이 기록 체계 이전에 만들어진 것 = **조건 불명**
(현재 `llama1b`, `qwen2_vl` 두 행이 여기 해당).

---

## 2. GPU 실행 경로 — **진짜 GPU (폴백 아님)**

`get_providers()` 만으로는 부족하므로 **4가지 독립 신호**를 모두 확인했다.

| 모델 | 세션 provider | 노드 배치 CUDA/CPU | gpu_ms | cpu_ms(강제) | 속도차 | GPU util | 프로세스 GPU 메모리 | 판정 |
|---|---|---|---|---|---|---|---|---|
| resnet50 | CUDA+CPU | **122 / 0** | 1.71 | 6.12 | 3.6× | 6% | 1296 MB | REAL GPU |
| mobilenet_v2 | CUDA+CPU | **100 / 0** | 0.76 | 1.11 | 1.5× | 8% | 1102 MB | REAL GPU |
| yolo11n | CUDA+CPU | **241 / 0** | 2.46 | 11.73 | 4.8× | 25% | 1172 MB | REAL GPU |
| yolo11s | CUDA+CPU | **241 / 0** | 3.93 | 22.26 | 5.7× | 10% | 1170 MB | REAL GPU |
| yolo11m | CUDA+CPU | **241 / 0** | 5.58 | 55.23 | 9.9× | 5% | 1298 MB | REAL GPU |
| yolo11l | CUDA+CPU | **241 / 0** | 7.87 | 71.59 | 9.1× | 11% | 1170 MB | REAL GPU |
| yolo11x | CUDA+CPU | **241 / 0** | 11.20 | 141.53 | 12.6× | 7% | 1168 MB | REAL GPU |

- **노드 배치**: ORT 프로파일링(`enable_profiling`)으로 노드별 실제 EP를 집계.
  **모든 모델에서 CPU EP에 떨어진 노드가 0개** — 그래프 전체가 CUDA에서 실행된다.
- **nvidia-smi**: 추론 중 해당 **프로세스의 GPU 메모리 점유 1.1–1.3 GB** 확인. 폴백이면 0이다.
- **속도 대조**: CPU 강제 대비 1.5×–12.6× 빠르다. 모델이 클수록 격차가 커지는 것도 정상.
- GPU utilization 수치(5–25%)가 낮게 보이는 것은 추론이 밀리초 단위라 50 ms 샘플링에
  잘 안 잡히기 때문이며, 메모리 점유와 노드 배치가 더 신뢰할 수 있는 신호다.

**판정: 폴백 없음. 7개 vision 모델 전부 진짜 GPU 실행.** 수정 사항 없음.

> 주의: 위 GPU/CPU 수치는 **경로 버그 수정 후**의 값이다. 수정 전에 측정하면 모든 yolo가
> yolo11s로 로드되어 gpu_ms가 3.4–4.2 ms로 평탄하게 나온다 — 이는 폴백이 아니라 0번 버그였다.

---

## 3. BGR→RGB 가 detection **측정값**에 미친 영향 — **영향 없음**

BGR 버그는 **NPU 경로에만** 있었다. CPU/GPU 경로는 `_yolo_letterbox()` 가 이미
`canvas[..., ::-1]` 로 RGB 변환을 하고 있었으므로 처음부터 정상이었다.
따라서 NPU에서 `stockholm_1280x720.mp4` **200 프레임**으로 두 조건을 비교했다.

### yolo11s (NPU, 200 프레임)

| 지표 | BGR (수정 전) | RGB (수정 후) | 차이 | Welch t |
|---|---|---|---|---|
| 박스 수/프레임 | 13.04 ± 2.17 | 14.37 ± 2.05 | **+1.33 (+10.2%)** | **−6.31** |
| 후처리(NMS) ms | 26.06 ± 14.08 | 25.85 ± 14.66 | −0.21 (−0.8%) | +0.15 |
| end-to-end ms | 31.04 ± 14.09 | 30.89 ± 14.59 | −0.15 (−0.5%) | +0.10 |

### yolo11x (NPU, 200 프레임)

| 지표 | BGR (수정 전) | RGB (수정 후) | 차이 | Welch t |
|---|---|---|---|---|
| 박스 수/프레임 | 15.72 ± 2.95 | 17.25 ± 2.68 | **+1.53 (+9.7%)** | **−5.42** |
| 후처리(NMS) ms | 31.75 ± 12.21 | 33.00 ± 12.40 | +1.24 (+3.9%) | −1.01 |
| end-to-end ms | 47.14 ± 12.34 | 48.28 ± 12.48 | +1.14 (+2.4%) | −0.92 |

### 판정: **측정값 영향 없음 → 기존 detection 데이터의 timing은 재사용 가능**

- **검출 결과는 유의하게 달라졌다** (박스 +10%, |t| = 5.4–6.3 ≫ 2). RGB가 더 많이 잡는다.
  즉 BGR 시절의 검출 **품질**은 실제로 나빴다.
- 그러나 **후처리·end-to-end 시간은 유의하지 않다** (|t| = 0.1–1.0 < 2).
  박스가 프레임당 1.3–1.5개 늘어난 정도로는 NMS 비용이 측정 가능한 수준으로 변하지 않는다.
  후처리 시간의 표준편차(12–15 ms)가 평균 차이(0.2–1.2 ms)보다 10배 이상 크다.
- 정적 프로파일은 애초에 후처리를 측정 구간에 포함하지 않으므로(1번 참조) 더더욱 무관하다.

**단, 이 결론은 timing 한정이다.** detection 정확도를 주장하는 결과가 있다면 그것은
BGR 시절 데이터로는 뒷받침되지 않는다.

---

## 4. 수렴 실험 진입 가능 여부

| # | 항목 | 상태 |
|---|---|---|
| 1 | mobilenet_v2 정적 프로파일 | ✅ 완료 (조건 기록됨, 기존과 동일 조건) |
| 2 | GPU 경로 진위 | ✅ 진짜 GPU, 폴백 없음 |
| 3 | BGR의 detection 측정값 영향 | ✅ 영향 없음 (수치 제시) |
| **0** | **yolo 경로 해석 버그** | 🔴 **코드는 수정됨. 데이터 재수집 미결.** |

### 🔴 남은 블로커

**기존에 수집된 CPU/GPU 성능 데이터는 폐기 대상이다.** yolo11n/m/l/x 를 CPU 또는 GPU에
배치한 모든 측정 창이 사실은 yolo11s 를 돌린 것이므로:

- `deploy_cpu_gpu` 예측기: 학습 데이터의 yolo 다양성이 **가짜**다. 모델 크기가 달라도
  CPU/GPU 특징이 동일했으므로, 예측기는 "yolo 크기는 CPU/GPU 성능에 영향 없음"을 학습했다.
- `deploy_cpu_npu` 예측기: NPU 배치는 정상이나, **CPU 배치된 yolo** 항목이 오염됐다.
- 정적 프로파일은 이미 재생성했으므로, **재수집만 하면 된다.**

재수집 범위(어느 working set을, 어느 rate factor까지)는 대규모 작업이라 지시를 기다린다.
현재 커버리지 파일 기준 기존 5개 working set + 신규 8개 = 13개.

---

## 부록: 이번 점검에서 변경된 코드

| 파일 | 변경 |
|---|---|
| `utils.py` | `_canonical_name()` — 등록된 모델명은 `_normalize` 로 덮어쓰지 않음 (**0번 버그 수정**) |
| `profile_models.py` | 측정 조건(provenance) 기록 추가; NPU 전처리를 RGB로 (타이밍 불변) |
| `model_registry.py` | `mobilenet_v2` 등록 (이전 커밋) |
| `model_processors.py` | NPU 경로 BGR→RGB (이전 커밋) |

측정용으로 되돌린 코드는 **커밋하지 않았다** (BGR 비교는 런타임 플래그로 수행).
