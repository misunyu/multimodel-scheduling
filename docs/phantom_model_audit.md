# phantom model 오염 감사 (v20) — 판정: **광범위 오염**

작성 2026-07-27. 성격: **감사 전용.** 코드 수정·재실험 **없음**. 판정은 사전 고정 매트릭스(작업 81)에 따름.

---

## 판정 (작업 81, 사전 고정 매트릭스)

> **phantom이 다수·핵심 런에 존재 → 광범위 오염 → 정지·보고. 어휘 드리프트를 먼저 닫아야 한다(작업 82).
> 논문 수치는 그때까지 잠정.**

런타임 실행에 의존하는 **모든 QoS 시나리오**에서, 스케줄이 명명한 모델의 대다수가 실제로는 실행되지
않았다. 명명 4~7개 중 **실제로 돈 것은 1~2개**뿐이다. 이는 특정 런의 문제가 아니라 **코드 + 스케줄의
구조적 성질**이며, 저장소의 어느 시점에서 돌렸든 동일하게 발생한다.

---

## 근본 원인 요약 (어휘 드리프트 + 조용한 통과)

- 스케줄 YAML은 **XGBoost 오프라인 프로파일링 데이터셋의 어휘**(`mnasnet`, `resnext50`, `vgg19`,
  `shufflenet-v2-12`, `squeezenet1.0-12`, `yolov4`, `yolov3_*`, `resnet50_*` …)로 작성됐다.
- 런타임 레지스트리(`model_registry.MODELS`)가 **정적으로** 아는 vision 모델은
  `resnet50`, `yolo11{n,s,m,l,x}`, `mobilenet_v2` 뿐이다(git 이력상 폴더 자동등록·확장 없음).
- 변환 계층은 `_normalize()`의 **부분문자열 별칭**뿐이라 **거칠고 불완전**하다:
  - `"yolo"` 포함 → **전부 `yolo11s`** (yolov4, yolov3_big, yolov3_small … 구분 소멸)
  - `"resnet"` 포함 → **전부 `resnet50`** (resnet50_big, resnet50_small … 구분 소멸)
  - `resnext50`("resnet" 미포함), `mnasnet`, `vgg19`, `shufflenet-v2-12`, `squeezenet1.0-12`
    → 매핑 실패 → `reg.get()` **KeyError**.
- **조용한 통과 지점**: `worker_target()`→`classify_view()`가 KeyError를 `except`로 삼키고 `None`을 반환
  (`model_processors.py:487`, `:513-514`). 호출부는 워커를 **아예 시작하지 않고** 넘어간다
  (`unified_viewer.py:671-674` headless: print 후 `continue`; view: skip). 스케줄 파싱 시점에도
  `_resolve_model_path()`가 못 찾으면 조용히 `''` 반환(`:396-398`). **어디서도 "이 모델은 못 돌린다"고
  런을 멈추지 않는다.**
- 피더는 스케줄 `infps`대로 계속 프레임을 밀어넣으므로, 소비자 없는 뷰의 큐가 차서 **drop만 쌓이고
  `inference_count=0`** — "먹였는데 죽은" 상태가 된다.

> **정정(중요)**: 이 죽음은 v19가 지목한 FATAL fail-open 경로(`adaptive_deploy.py:329-332`)를 **타지
> 않는다.** 그 경로는 워커가 **시작된 뒤** 로드 예외를 던질 때다. phantom 모델은 그 이전, `worker_target`
> 단계에서 `None`이 되어 **워커가 시작조차 안 된다.** 즉 v19 작업 73(fail-safe)을 구현해도 phantom
> 문제는 안 풀린다. **어휘 드리프트가 원인, fail-open은 별개의(그리고 실측상 미발동인) 증상 은폐 후보.**

---

## 작업 79 — 게시 수치의 출처 런: **확정 불가 (증발)**

- 그림 값의 단일 소스로 문서들이 가리키는 `fig/confirmed_values.json`, 그림별 `.values.json` 사이드카,
  `fig/paper_figure_manifest.json`, `fig/check_figures.py`, `fig/build_pipeline_doc.py`는 **저장소에
  존재하지 않으며 git에 추적된 적도 없다**(`git ls-files 'fig/*'` 공집합). figure_pipeline.html은
  파이프라인이 `scratchpad/<run>/*.csv,*.log → fig/confirmed_values.json → PDF`임을 명시 — 즉 게시
  그림의 **원시 런과 값 대장이 세션 스크래치에 있었고, 지금은 사라졌다.**
- 따라서 게시 런을 경로·날짜로 직접 열거할 수 없다. **이것 자체가 작업 79의 보고 대상**(재현·감사
  체인이 저장소에 남지 않음).
- 저장소에 영구 보존된 런타임 산출물은 `results/*.json` **725개**뿐(조합별 집계 요약). 이 중 phantom
  모델을 참조하는 레코드는 **단 3건**(전부 2026-07-27, v18 계측분). 게시 그림의 원시 런은 여기 없다.
- 대신 **출처를 그림 생성 스크립트로 역추적**했다(아래 영향 목록).

---

## 작업 80 — 런×뷰 liveness

### 80-A. 영구 보존된 phantom 레코드 (results/ 전수 스캔, 3건)

| 파일 | 스케줄 | combo | 뷰:모델(dev) | inference_count | 판정 |
|---|---|---|---|---|---|
| perf_20260727_000038 | ml_misprediction_bg | fallback | view1:mnasnet(GPU) | 0 (drops 3879) | phantom |
| " | " | " | view2:resnext50(GPU) | 0 (drops 3879) | phantom |
| " | " | " | headless vgg19/shufflenet/squeezenet(CPU) | 0 (drops 0) | phantom(미기동) |
| " | " | " | view3:resnet50(CPU)=1888, view4:yolov4(CPU)=501 | >0 | live(2) |
| perf_20260727_000547 | ml_misprediction_bg | fallback | (동일 패턴) | mnasnet/resnext50=0 | phantom |
| perf_20260727_000736 | bounded_recovery | offload | view1:mnasnet(GPU)=0, view3:resnext50(GPU)=0 | 0 | phantom |
| " | " | " | view2:resnet50(GPU)=1019, view4:yolov4(GPU)=281 | >0 | live(2) |

- background generative(`llama1b`/`qwen2_vl`)의 `inference_count=0` 57건은 **구조적**(vision consumer
  부재)이라 제외. 다만 이들이 실제 child로 기동·ready 됐는지는 별도 확인 필요(로그 증발로 미확정).

### 80-B. 구조적 liveness (스케줄 파싱 → 실제 실행 모델), 시나리오별

| 시나리오(스케줄) | combo당 명명 | **실제 distinct 실행** | phantom(죽음) |
|---|---|---|---|
| Q3 `ml_misprediction_runtime_{bg,ml}` | 7 | **2** (resnet50, yolo11s) | mnasnet, resnext50, shufflenet-v2-12, squeezenet1.0-12, vgg19 |
| §4b `bounded_recovery_views` | 4 | **2** | mnasnet, resnext50 |
| `dynamic_load_views` | 4 | **2** | mnasnet, resnext50 |
| `qos_recovery` | 4 | **1** (resnet50) | mnasnet, squeezenet1.0-12, resnext50 |
| `steady_state_views` | 4 | **2** | mnasnet, resnext50 |
| Q5 `q5_run_infeasible`(foreground) | 4 | **2** (resnet50, yolo11s) | resnext50, vgg19 |

- 게다가 "실제 2개"조차 **별칭 붕괴** 결과다: `yolov4/yolov3_big/yolov3_small` → 모두 `yolo11s`,
  `resnet50/resnet50_big/resnet50_small` → 모두 `resnet50`. 서로 다른 모델로 적힌 이름이 **같은 2개
  ONNX로 수렴**한다.

### 80-C. 논문 서술과의 대조 (원칙 5)

§IV-A: *"four concurrent DNN applications"*, *"up to nine concurrent application instances"*.
실제 동시 실행된 **distinct 실 모델은 1~2개**. → **틀린 것은 수치가 아니라 워크로드 서술이다.**
살아 있던 뷰(resnet50, yolov4→yolo11s)의 drop·지연은 실측이지만, 그것은 **2모델 워크로드**의 값이다.

---

## 작업 81 — 영향 받은 그림/수치 목록

역추적한 provenance 기준. **런타임 실행 의존 = 오염**, **예측/시뮬레이션 = 무영향**.

### 오염 (런타임 QoS 실행 → phantom 뷰 포함)
| 그림 | 스크립트/스케줄 | 실 모델 |
|---|---|---|
| q3_misprediction(.pdf, _2gen) | ml_misprediction_validation → ml_misprediction_runtime_{ml,bg} | 2/7 |
| bounded_recovery_analysis | bounded_recovery_validation → bounded_recovery_views | 2/4 |
| q4_bounded_envelope | bounded_recovery sweep | 2/4 |
| dynamic_load_adaptation | dynamic_load_validation → dynamic_load_views | 2/4 |
| qos_score_validation | qos_recovery_validation → qos_recovery | **1/4** |
| q13_failure_persistence, q13_cumulative_violation | q13_failure_persistence (predictor+executor+headless) | ≤2 |
| detection_sensitivity | bounded_sweep 파생 CSV | 2/4 |
| q5_npu_generalization(.pdf, _2gen) | q5_run_infeasible foreground | 2/4 |
| runtime_overhead | runtime_overhead_analysis → steady_state_views | 2/4 |
| §IV Q1.x/Q2/Q6 중 위 런타임 런에 의존하는 항목 | (확정표 증발로 개별 매핑 불가) | — |

### 무영향으로 추정 (예측기/시뮬레이션, 런타임 미실행) — **사용자 확인 필요**
- d2_budget_rank, d2_dwell_response, d2_load_tightness (d2 시뮬레이션)
- b2_metrics, b2_buffer_sweep, c3_b2_metrics, c3_fluid_validation (fluid 모델/유체 검증 — 실행 의존
  여부 재확인 권장)
- comparison tables, alpha sweeps, plot_model_count_vs_score (XGBoost 예측 랭킹)
- c2_reactive_comparison (reactive baseline — 실행 의존 여부 재확인 권장)

> 확정표(`confirmed_values.json`)가 증발해 "그림↔§IV 수치↔런" 3자 매핑을 저장소만으로 못 맺는다.
> 위 분류는 **스크립트 provenance** 기준 최선의 역추적이며, 논문에 실제 실린 항목과의 대조는 사람이
> 해야 한다.

---

## 작업 82 — 어휘 드리프트 범위 (감사만, 수정 안 함)

1. **런타임이 아는 vision 모델**(정적): `resnet50`, `yolo11n/s/m/l/x`, `mobilenet_v2`.
   `_normalize` 별칭: `*yolo* → yolo11s`, `*resnet* → resnet50`, `*qwen*|*vl* → qwen2_vl`,
   `*llama*|*llm*|*tiny* → llama1b`.
2. **스케줄/시나리오에 등장하는 vision 이름 전체**: `mnasnet, resnet50, resnet50_big, resnet50_small,
   resnext50, shufflenet-v2-12, squeezenet1.0-12, vgg19, yolov3_big, yolov3_small, yolov4`
   (+ q5 배경 로드의 `models_onnx/*.onnx` 명시 경로).
3. **차집합 = 런타임 로드 불가**: `mnasnet, resnext50, shufflenet-v2-12, squeezenet1.0-12, vgg19`
   (5개). 별칭으로 "로드는 되나 다른 모델로 붕괴"하는 이름: `resnet50_big/_small`(→resnet50),
   `yolov3_big/_small`(→yolo11s), `yolov4`(→yolo11s).
4. **드리프트 발생점**: 스케줄이 XGBoost 데이터셋 어휘를 그대로 사용. 런타임에 그 어휘 → 실 모델의
   **번역 계층이 없고**, `_normalize`의 부분문자열 별칭이 **불완전한 대체물**이라 5개가 탈락하고 나머지가
   2개로 붕괴.
5. **검증 가능했던 지점**: 스케줄 파싱(`unified_viewer._resolve_model_path`, `worker_target`)이 로드
   불가/미해소를 **조용히 통과**시킨다. 여기서 "명명된 모델 중 K개가 미해소"를 fail-fast 했다면 게이트에서
   잡혔다. (수정은 원인 규명·사용자 판단 후 — 이번 라운드 범위 밖.)

---

## 작업 83 — v18 측정의 재검토

- v18 재실행 6 레코드(2026-07-27) = 위 3 파일(Q3×2, bounded_recovery×1). **전부 phantom 뷰 포함.**
- **흔들리는 것 (재측정 대상)**: v18이 보고한 **수치** —
  host RSS 최악 +201 MiB, 여유 308×, 전환 window 통계 등. 이는 4뷰 중 **2뷰만 살아 있던** 워크로드의
  값이다. "4모델 동시" 자원 피크가 아니다.
- **유지되는 것 (phantom 무관)**: v18의 **구조적 결론** — (a) 회복 전환이 전부 GPU↔CPU **교차이동**,
  (b) generative는 **kill-then-start**, (c) 그래서 옛/새가 같은 디바이스 메모리를 동시 점유하지 않음.
  이는 **코드 판독**에서 나왔고 실행 워크로드와 무관하므로 그대로다. VRAM 전환 증분 0의 **논거(교차이동)**
  는 유지되나, 그 **실측 확인**은 2모델 워크로드에서 이뤄졌다는 단서를 달아야 한다.

---

## 하지 않은 것 / 다음 결정 (사람 몫)

- **코드 수정 없음** — v19 작업 73 포함 전면 보류(원인=어휘 드리프트를 먼저 닫는 것이 순서).
- **재실행 없음** — 판정이 먼저(작업 81 규정).
- 결정 필요: (1) 어휘 드리프트를 닫는 방식(번역 계층 / 스케줄 어휘 교정 / 누락 모델 실제 도입 중 무엇),
  (2) 닫은 뒤 어떤 시나리오를 어떤 모델 구성으로 재실행할지, (3) 논문 워크로드 서술("4~9 concurrent")을
  실제 실행 구성에 맞출지. 모두 **사용자 판단**.
