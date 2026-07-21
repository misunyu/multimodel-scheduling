# 네 기법 동작 검사 — 신규 모델·예측기 위에서

날짜: 2026-07-21 · 브랜치: `ubuntu_gpu_fsrr` · 대상: active 경로만 (legacy/scripts 제외)

## 한 줄 결론

**오프라인 XGBoost 예측기 스왑은 정상 동작하지만, 런타임 실행 백엔드(`model_processors.py`)가
신규 모델·MLA100 NPU로 이전되지 않았다.** 그 결과 네 기법 모두 *예측기 랭킹·기법 디스패치·QoS
로직*은 신규 스키마 위에서 멀쩡하나, *실제 모델 로드·추론* 단계에서 신규 셋(`yolo11*`,
`resnet50`, `mobilenet_v2`, `llama1b`, `qwen2_vl`)을 로드하지 못한다. 런타임은 여전히
`yolov3`/`resnet50_big`을 구 Neubla NPU(.o)로 부르고, LLM/VLM 실행 워커는 아예 없다.

이 이전은 최소 수정 범위가 아니라 **런타임 백엔드 포팅**이므로 게이트(사람 판단) 대상이다.

## §2 정합성 검사 (하드웨어 불필요) — 전부 PASS

| 항목 | 결과 |
|---|---|
| 2a 예측기 로드·스키마 | 6 booster(2 플랫폼 × y1/y2/y3) 로드, `num_features=37`, `feature_names`가 `*_features.json`과 완전 일치 |
| 2b coverage vs registry | coverage 9개 모델 = `model_registry` 9개 모델 완전 일치, kinds 정합(vision 7 / llm 1 / vlm 1) |
| 2c 정적 프로파일 | 9개 실행 모델 모두 cpu/gpu/npu per-device 프로파일 보유 |
| 예측기 축 드라이런 | vision-only·mixed × cpu_npu·cpu_gpu 4조합 모두 예외 없이 랭킹, y1/y2 ∈ [0,1], 스코어 내림차순, `has_generative` 분기 정상(β=0.5) |

## 축 × 기법 판정 매트릭스

축 정의: **예측기**(오프라인 랭킹, 기법이 소비) / **모델 로드·추론**(런타임) / **QoS V(t)** / **기법 고유 로직**.

| 기법 (mode) | 예측기 축 | 모델 로드·추론 축 | QoS 축 | 기법 고유 로직 | 종합 |
|---|---|---|---|---|---|
| **Static** (3) | 해당 없음 | **FAIL** (신규 모델 로드 불가) | 로직 OK, 입력 없음 | 고정 배치 디스패치 OK | **FAIL(런타임)** |
| **Stop-and-restart** (0) | PASS (오프라인) | **FAIL** | 로직 OK, 입력 없음 | 재시작 디스패치 OK | **FAIL(런타임)** |
| **Adaptive hot-swap** (1) | PASS (오프라인) | **FAIL** | 로직 OK, 입력 없음 | hot-swap 디스패치 OK, 단 라우팅 버그(아래) | **FAIL(런타임)** |
| **BoundGuard** (2) | PASS (오프라인) | **FAIL** | `_collect_vscore` 로직 OK | validate/T_v·rollback 로직 OK, 단 메모리 추정 버그(아래) | **FAIL(런타임)** |

> **중요한 구조적 사실**: 네 기법 모두 **런타임에는 XGBoost 예측기를 호출하지 않는다.** 예측기는
> 오프라인(GUI "Predict Best" 버튼 / scripts)에서만 돌아 후보 순서를 미리 정하고, 런타임은 그
> 미리 정해진 스케줄/후보열을 소비한다. 따라서 예측기 스왑은 런타임 기법 디스패치를 깨지 않는다 —
> 깨진 것은 그 아래 **모델 로드·추론**이다.

## 결함 목록 (기법 × 축)

### D1 — [치명] 런타임 워커가 신규 모델을 로드하지 못함 (모델 로드 축, 네 기법 공통)
- `model_processors.py`의 워커는 `yolov3_small`/`resnet50_big`을 하드코딩 로드하고, NPU는 구
  Neubla `.o`(`resolve_npu_object_o`)를 쓴다. Mobilint `.mxq`/qbruntime 경로가 **전무**.
- `utils.resolve_cpu_model_onnx("yolo11s")` → `models/yolo11s/model/yolo11s.onnx`(**존재하지 않음**)
  반환. 신규 자산은 `models/onnx/yolo11s.onnx` 레이아웃인데 resolver가 이를 모른다.
- LLM/VLM 실행 워커 부재 → mixed 셋의 `llama1b`/`qwen2_vl` 뷰는 실행 자체가 불가.
- 원인: 예측기 스왑 커밋(`f02331b`)이 런타임 실행부를 건드리지 않음. mobilint에는 이전된
  `model_processors.py`(qbruntime/.mxq/LLM) + `runtime/mobilint_vision.py`가 있으나 fsrr 미이식.

### D2 — [높음] Adaptive/hot-swap 워커 라우팅이 `"yolov4"` 문자열에 고정 (Adaptive·BoundGuard)
- `adaptive_deploy._create_worker_thread`는 `"yolov4" in model`이면 YOLO 워커, else ResNet 워커.
  신규 `yolo11s`는 `"yolov4"`에 매칭 안 되어 **ResNet 워커로 오라우팅**된다.
- 같은 `"yolov4"` 분기가 `unified_viewer.py`(비디오 피더 판정 등 6곳)에도 존재.

### D3 — [중간] BoundGuard 폴백 GPU 메모리 추정이 구 모델만 인지 (BoundGuard/reactive)
- `reactive_deploy._MODEL_GPU_MEM_MB`에 신규 모델 없음 → `resnet50`만 매칭, 나머지(`yolo11*`,
  `mobilenet_v2`, `llama1b`, `qwen2_vl`)는 200MB 기본값. LLM/VLM엔 심한 과소추정 → reactive
  메모리 예산 기반 폴백 배치 판정이 부정확(크래시는 아님).

### D4 — [낮음] 스케줄 생성 infps 폴백이 신규 vision 모델에 대부분 10 (네 기법 공통, 저위험)
- `schedule_generator_logic.py`의 폴백: `resnet50`→2, `yolov3`→30, else→10. 신규 vision 모델은
  대부분 else(10). 사용자가 입력레이트를 명시하면 무관(방금 이식한 baseline_rate 다이얼로그가
  완화). 예측 featurize에 미미하게 영향.

### D5 — [주의] mode 번호 이중 의미 (혼동 위험)
- `schedule_executor_main.py`: BoundGuard = **mode 2**(reactive). 그러나 일부 실험 스크립트는
  BoundGuard를 `best_deploy_finder_executor`의 **mode 1**(hot-swap)로 구동. 같은 이름이 진입점에
  따라 다른 mode를 가리킨다. active 버그는 아니나 실험 재현 시 혼선 소지.

## 하드웨어 검사 (§4b) — 미실행

D1로 인해 신규 모델을 로드할 수 없어, 실제 MLA100/GPU 실측(네 기법 × vision/mixed × cpu_npu/cpu_gpu)은
**의미가 없어 미실행**. 실측은 D1(런타임 백엔드 이전) 해소 후에 가능하다.

## 판정 요약 (§5 기준)

- **§2 정합성 + 예측기 축**: **PASS**. 예측기 스왑은 깨끗하며 네 기법이 그 출력을 정상 소비.
- **네 기법 런타임 종합**: **FAIL** — 모델 로드 축(D1)에서 신규 셋을 실행 불가. 가드레일이 정당하게
  막는 경우가 아니라 실제 미이전으로 인한 breakage.
- 기법 디스패치·QoS·고유 로직 자체는 모델-무관하게 **구조적으로 온전**(D2/D3는 신규 모델에서의 정확성 결함).

## 권장 (게이트)

D1은 런타임 실행 백엔드 포팅(= `model_processors.py` + `utils` resolver + `runtime/mobilint_vision.py`
+ LLM 워커를 mobilint에서 이식)으로, 최소 수정이 아니라 별도 대규모 작업이다. 사람 판단 필요:
- 이 검사에서 즉시 고칠 저위험 항목: 없음(런타임이 통째로 미이전이라 개별 패치는 무의미).
- 게이트 대상: D1(런타임 백엔드 이식) → 이후 D2·D3·D4가 그 이식에 흡수되어 해소되는지 재검사.

---

# D1 런타임 백엔드 포팅 결과 (2026-07-21, 추가)

방침: **비전 우선 + 환경 설치**. mobilint 워커를 fsrr 계약(Thread + ready_event + 기존 결과
튜플)에 맞춰 이식하고, 디스패치·QoS·view_handlers·예측기·β는 불변으로 유지.

## 이식 내용
- **환경 설치** (fsrr `.venv`): torch 2.12.0+cu130, torchvision, transformers 5.9, huggingface_hub,
  mblt-model-zoo 1.5.1, mobilint-qb-runtime 1.2.0, qbcompiler 1.1.2 (출처: `MobilintTest/`).
- **`runtime/mobilint_vision.py`** 이식 (MLA100 `.mxq` 비전 백엔드, `mblt_model_zoo`).
- **`model_processors.py` 재작성**: detection/classification 워커를 새 백엔드(cpu/gpu onnx + npu .mxq)로,
  fsrr 시그니처·결과 튜플(yolo 3-tuple, resnet 4-tuple)·ready_event 유지. `model_name` 파라미터 추가.
  registry 기반 라우팅 헬퍼(`classify_view`/`worker_target`) 추가.
- **라우팅 재배선** (D2): `unified_viewer.py`(팩토리 2 + 멤버십 5) + `adaptive_deploy.py`(팩토리 1 +
  멤버십 4)의 `"yolov4" in model` → `reg.kind_of`/task. **NPU 라우팅 재활성화**(구 "deprecated→GPU"
  제거). generative(llm/vlm) 뷰는 None 반환→스킵(hot-swap None 가드 추가).
- **D3**: `reactive_deploy._MODEL_GPU_MEM_MB`에 신규 9모델 값 추가(llama1b 2600, qwen2_vl 5200 등).
- **D5**: `schedule_executor_main.MODE_METHOD_MAP` 상수 + 두 진입점 상단 mode 매핑 주석.

## 검증 (실측 포함)
| 항목 | 결과 |
|---|---|
| 컴파일·import (전 런타임 모듈) | OK |
| resolver·라우팅 (9모델 × cpu/gpu/npu) | yolo11*→yolo, resnet50/mobilenet_v2→resnet, llama1b/qwen2_vl→스킵. onnx/mxq 경로 실존 |
| **추론 실측 7/7 PASS** | detection/classification × cpu·gpu·npu. NPU(.mxq, /dev/aries0) 정상(yolo11s 4.6ms, resnet50 1.9ms), CPU 정상. 결과 튜플 형태 정확 |
| D2 워커 라우팅 | clean (잔존 "yolov4"는 legacy 메모리 딕셔너리 데이터·주석뿐) |
| legacy/백업 무변경 | OK |

## 재판정 (모델 로드 축)
- **FAIL → PASS** (cpu/npu). 네 기법이 공유하는 모델 로드·추론 축이 신규 모델을 실제 로드·추론.
- 디스패치·QoS·view_handlers는 불변이므로 기법별 고유 로직도 그대로 유효.

## 잔여 항목 (사람 판단)
1. **[높음] GPU(CUDAExecutionProvider) cudnn 오류** — RTX 5090(sm120)+cuda13+onnxruntime-gpu
   1.23.2+cudnn 9.20에서 conv 노드가 `CUDNN_BACKEND_API_FAILED`로 실패, onnxruntime이 **CPU로
   자동 폴백**(크래시 아님, 조용한 성능 저하). cpu_gpu 전략의 vision GPU 실행이 실제로는 CPU에서 돎.
   환경/버전 호환 문제(코드 아님) → 호환 onnxruntime-gpu 빌드 필요. NPU 경로는 무관하게 정상.
2. **[낮음] `schedule_generator_app.py:1321`** `vision_with_view={mnasnet,resnet50,resnext50,yolov4}` —
   별도 스케줄 생성 GUI의 뷰 분류 집합(런타임 라우팅 아님). 신규 모델 미포함. 그 GUI를 쓸 때만 영향.
3. **[미실행] 네 기법 풀 GUI 실측(§4b)** — 워커 수준(모델 로드·추론)은 cpu/npu 실측 완료. 네 전략을
   실제 GUI로 30초씩 돌려 V(t) 궤적·전환·bounded persistence를 관찰하는 풀 실행은 미실행(대화형·NPU
   공유 자원). 워커·라우팅·튜플이 검증됐고 디스패치/QoS 불변이라 동작 예상.
4. **LLM/VLM(llama1b/qwen2_vl) 실행 워커 미이식**(비전 우선 방침) — mixed 셋의 LM 뷰는 스킵됨.
   다음 단계.
