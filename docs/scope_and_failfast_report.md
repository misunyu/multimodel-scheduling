# 범위 확정 · fail-fast · 검증장치 커밋 (v21)

작성 2026-07-27. 성격: 범위 확정 + 재발 방지(코드 수정 86·87) + 커밋(88). **재실행 없음.**

---

## 작업 84 — 예측기 세대 판정 (**게이트**) → 판정: **최대(maximal) 범위**

### 84-1. 확인 결과

1. **fsrr `xgboost_model/`**: `prediction_result/` 디렉터리는 **현재 없다**(git 이력상 삭제됨 — 옛 어휘
   score JSON들이 지워졌다). 남은 것은 `artifacts/{gpu,npu,cpu_gpu,cpu_npu}/`의 **신세대(3-target:
   rank/score/double)** XGBoost 모델과 `performance_data/sample_profiling_data/sample_profiling_data.json`.
2. **`../multimodel-scheduling-mobilint`(형제 저장소)**: **신세대 학습·스케줄 생성의 본가.**
   `model_registry.py`가 fsrr와 **동일한 신어휘**(`resnet50, yolo11n/s/m/l/x, mobilenet_v2, llama1b,
   qwen2_vl`). `generate_schedules.py`·`generate_concurrency_schedules.py`가 신어휘로 스케줄을 만든다
   (`BASE=["resnet50","llama1b","qwen2_vl"]`, `YOLO_ORDER=["yolo11n"…"yolo11x"]`). 신예측기
   (`deploy_predictor_logic.DeployPredictor` + fsrr `artifacts/`)는 fsrr **런타임에도 통합돼 있다.**
3. **시나리오 YAML은 손으로 작성**됐다. `tests/*.yaml`의 combo·후보 배치가 리터럴로 박혀 있다
   (`combination_cand_1..3`, `combination_ml_pick`, `combination_fallback`). 생성 스크립트 출력이 아니다.
4. **게시 후보 순서의 출처**: `scripts/ml_misprediction_validation.py`·`q13_failure_persistence.py`가
   **`deploy_predictor_logic_legacy`(pre-MLA100 2-target 예측기)**를 import 한다. 그 예측기 입력
   `test_schedules_x3.csv`는 `backup/results_pre_mla100_.../xgboost_model/dataset/gpu/`에 있다(옛 세대).

### 84-2. 근거 인용 (결정적)

`deploy_predictor_logic_legacy.py` 헤더:
> "Pre-MLA100 two-target predictor, kept for paper-figure reproduction only. **The working set these
> figures were produced on (resnext50, vgg19, yolov4, ...) is not in the MLA100 predictor's coverage**,
> so those scripts **cannot be repointed at the new bundle** -- they would fail the unprofiled-model check."

### 84-3. 판정 (사전 고정 매트릭스, 세 번째 칸)

**랭킹이 옛 예측기 출력이다.** 따라서 "rank 5"(Q3 cand_5), "rank 2"(Q5 cand_2)는 **옛 어휘 배치**를
가리킨다. §IV의 cand_5 마진 0 서술, budget×rank(d2), Q6 후보 순회가 **전부 재구성 대상**이다.

> **재실행 계획 이전에, 랭킹부터 신예측기로 다시 뽑아야 한다.** 옛 후보 순서를 그대로 두고 재실행하면
> "신어휘로 갈아끼운 옛 랭킹"이라는 또 다른 혼종이 된다.

**정정 확인**: v20의 "XGBoost 어휘 vs 런타임 어휘" 프레이밍은 틀렸다. 논문·런타임 레지스트리·신예측기가
모두 **신어휘로 일치**하고, 옛 어휘를 쓰는 것은 **시나리오 YAML 하나**다(한 세대 뒤처짐). `_normalize`
부분문자열 별칭이 7개 중 2개(yolo*, resnet*)를 "돌게" 만들어 **부분적 성공으로 실패를 은폐**했다.

---

## 작업 86 — fail-fast (구현 완료, **음성 테스트 통과**)

조용한 통과 지점 3곳을 닫았다(`unified_viewer.py`).

1. **스케줄 파싱 시점 검증**: 스케줄의 전 combo 모델명을 레지스트리로 해소. 하나라도 미해소면
   `ScheduleValidationError`로 **런 중단**(미해소 이름 전량 나열). 이 예외는 기존의 광범위 `except`가
   기본값으로 삼키지 못하도록 **재-raise** 처리(기본값 fallback = 옛 조용한 통과의 근원이었음).
2. **`worker_target` None = 치명**: view/headless 양쪽에서 None이면 raise. 단 **generative(llm/vlm)**는
   의도된 지연 실행이므로 `is_llm_like`일 때만 skip(방어적 이중화 — 파싱 검증이 이미 앞단에서 막음).
3. **기동 자기 점검 로그**: 명명 이름 → 실제 해소 모델을 **전량 출력**, 별칭 개입 시 `(ALIAS)` 표시.

### 86-4. 음성 테스트 출력 전문 (옛 어휘 스케줄 → 중단)

```
$ QT_QPA_PLATFORM=offscreen python3 schedule_executor_main.py \
    --schedule tests/bounded_recovery_views_schedule.yaml --duration 6 --adaptive-mode 1 --auto_start_all

[UnifiedViewer] model resolution self-check [bounded_recovery_views_schedule.yaml]:
    combination_initial: mnasnet -> UNRESOLVED
    combination_initial: resnet50 -> resnet50
    combination_initial: resnext50 -> UNRESOLVED
    combination_initial: yolov4 -> UNRESOLVED
    combination_overload: mnasnet -> UNRESOLVED
    combination_overload: resnet50 -> resnet50
    combination_overload: resnext50 -> UNRESOLVED
    combination_overload: yolov4 -> UNRESOLVED
    combination_offload: mnasnet -> UNRESOLVED
    combination_offload: resnet50 -> resnet50
    combination_offload: resnext50 -> UNRESOLVED
    combination_offload: yolov4 -> UNRESOLVED
Traceback (most recent call last):
    raise ScheduleValidationError(
unified_viewer.ScheduleValidationError: Schedule 'bounded_recovery_views_schedule.yaml' names models the
runtime cannot resolve: ['mnasnet', 'resnext50', 'yolov4']. They are not in model_registry.MODELS and no
explicit alias maps them. Refusing to run so they do not become silent phantom views
(see docs/phantom_model_audit.md).
exit code = 1
```

**확인 3항목**:
- ① 런이 **즉시·단일지점에서 중단**(exit 1). 기본값 fallback 없음, phantom 뷰 생성 없음.
- ② 미해소 이름 전량 나열(`mnasnet, resnext50, yolov4`). `resnet50`만 해소.
- ③ 자기 점검 로그가 이름→해소를 전량 출력.

### 86-(양성 대조) — 신어휘 스케줄은 정상 실행 (오탐 없음)

```
$ ... --schedule scratchpad/v21/newvocab_test.yaml (yolo11s/resnet50/mobilenet_v2/llama1b)
[UnifiedViewer] model resolution self-check [newvocab_test.yaml]:
    combination_a: yolo11s -> yolo11s
    combination_a: resnet50 -> resnet50
    combination_a: mobilenet_v2 -> mobilenet_v2
    combination_a: llama1b -> llama1b
[UnifiedViewer] Starting view1 with yolo11s on GPU (thread)
[UnifiedViewer] Starting view2 with resnet50 on GPU (thread)
```
→ 파싱 중단 없이 뷰 기동. `llama1b`는 generative로 지연(비전 워커 미기동, 정상).

---

## 작업 87 — `_normalize` 별칭 처리 (구현 완료)

`model_registry._normalize`의 **부분문자열 매칭 전면 제거**. 명시적 사전 `_ALIASES`(현재 비어 있음 —
논문 신어휘는 전부 `MODELS`의 정확 키라 별칭 불필요)로 대체. 표기 변형이 필요해지면 그때 명시 항목 추가.

- 부작용(의도됨): `yolov4/yolov3_big/yolov3_small → yolo11s`, `resnet50_big/_small → resnet50` 붕괴가
  사라짐. 이 이름들은 이제 **미해소로 즉시 실패**한다.
- 유닛 확인: 신어휘 9개 전부 해소(별칭 0), 옛 어휘 8개(`mnasnet, resnext50, vgg19, shufflenet-v2-12,
  squeezenet1.0-12, yolov4, yolov3_big, resnet50_small`) 전부 미해소.

**변경 파일**: `model_registry.py`(_normalize + `resolution`/`resolves`/`unresolved_models` 추가),
`unified_viewer.py`(`ScheduleValidationError`, 파싱 검증, 자기 점검 로그, 2개 None-site 치명화, except
재-raise). 백업: `backup/v21_pre_failfast/`.

---

## 작업 85 — 신어휘 스케줄 생성 규칙 (설계, **사람 확인 대기** — 아직 생성 안 함)

1. **정확 이름만.** 레지스트리가 별칭 없이 해소하는 이름만 사용(`_normalize` 의존 금지 — 이제 강제됨).
2. **워크로드 = §IV-A 규정.** foreground(모니터링 대상, 4개 고정): **YOLO11s, YOLO11m, ResNet50,
   MobileNet-v2.** background 풀(경쟁 유발): YOLO11n, YOLO11l, YOLO11x, LLaMA-1B, Qwen2-VL — 실험별로
   활성 집합 명시. 동시 인스턴스 up to 9.
3. **후보 배치는 신예측기 랭킹에서.** 순위를 YAML에 하드코딩하지 않는다(현재 하드코딩 = 보고 대상).
4. **옛→신 대응표**(재실행 결과를 옛 수치와 대조하기 위함) — 아래 초안.
5. **옛 스케줄 보존**: `tests/legacy/`로 이동 또는 접미사. 삭제 금지(감사 이력).

생성 도구는 형제 저장소 `generate_schedules.py`·`generate_concurrency_schedules.py`(신어휘)를 재사용
권장. **사람 확인 후** 실제 생성.

### 85-4. 옛→신 시나리오 대응표 (초안)

| 옛 스케줄(옛 어휘, 실행 1~2) | 실제 돌던 것 | 신 시나리오(안, §IV-A 준수) |
|---|---|---|
| ml_misprediction_runtime_{bg,ml} (7 명명) | resnet50, yolo11s | foreground 4(yolo11s/yolo11m/resnet50/mobilenet_v2) + bg(llama1b …) |
| bounded_recovery_views (4) | resnet50, yolo11s | foreground 4, 3-phase initial→overload→offload |
| dynamic_load_views (4) | resnet50, yolo11s | foreground 4, cand는 신예측기 랭킹 |
| qos_recovery (4) | **resnet50만** | foreground 4 |
| steady_state_views (4) | resnet50, yolo11s | foreground 4 (runtime_overhead 재측정용) |
| q5_run_infeasible (4 fg + bg) | resnet50, yolo11s | CPU–NPU, bg LLM on NPU |

---

## 작업 88 — 검증 장치 git 커밋

1. **미추적 사유**: `.gitignore`에 `fig/` 제외 규칙 **없음**. 단지 세션 스크래치에만 있었고 저장소 루트에
   생성된 적이 없어 추적되지 않았을 뿐. 예외 규칙 불필요.
2. `fig/`에 배치: `confirmed_values.json`, `check_figures.py`, `paper_figure_manifest.json`, `README.md`.
3. **잠정 표시 확인**: `confirmed_values.json._meta.status` =
   `"PROVISIONAL — superseded pending re-run (see docs/phantom_model_audit.md)"` (이미 포함).
4. `README.md` 포함(파이프라인·규약·source_kind·현재 잠정 경고).
5. 앞으로의 사이드카·플롯·스케줄도 커밋 대상.
6. `git ls-files fig/` 결과는 커밋 절에 기재.

> 사이드카(`*.values.json`)는 아직 없다 — 그림 재생성이 재실행 이후이므로. `check_figures.py`는 사이드카
> 부재 시 "사이드카를 찾지 못했다"로 종료(장치 자체는 정상).

---

## 작업 89 — 재실행 범위 산정 (**실행하지 않는다**)

**선행(필수)**: 신예측기로 후보 랭킹을 다시 뽑는다(84 최대판정). 이것 없이는 어떤 재실행도 옛 랭킹을
답습한다.

### 재실행 대상 (런타임 실행 의존 → phantom 영향)
우선순위(ISSRE 리뷰 대응 순):
1. **Q6 (2×2 ablation)** — 리뷰 핵심 지적("baselines weak", "no re-invoke baseline")의 답. 최우선.
2. **Q3·Q5 (오예측)** — 신어휘 foreground 4 + 신랭킹 cand.
3. **§4b/Q2 (bounded_recovery)** — 상한 대 실측.
4. **Q1.3·Q1.5**.
5. **Q4 (bounded envelope)**.
6. **C3 (fluid 검증)** — 런타임 측정분.
7. **v18 자원 재측정** — 4-모델 워크로드에서 host RSS/VRAM 재측정.

부수: dynamic_load, qos_recovery, q13, detection_sensitivity, runtime_overhead, c2_reactive — 전부
런타임 실행 의존이므로 재실행.

### 영향 없음 (재실행 불요)
- **d2_budget_rank/dwell_response/load_tightness** — analytic(상한식 평가) + 시뮬레이션. 단, budget×rank의
  **rank 축 자체**는 신랭킹으로 재확인 필요(값은 analytic이나 축의 의미가 옛 랭킹).
- **analytic bound 값**(q3_bound, q4_bound, q5_bound = T+k(T_v+δ)) — 측정 상수 대입식. δ만 유효하면 유지.
- XGBoost 예측 랭킹 그림의 **분석 평면** 부분.

> 확정표(`fig/confirmed_values.json`)의 `source` 필드가 각 값의 런 출처(b2rep_q3, b3rep, c2_final2 …)를
> 가리키나, 그 원시 런은 스크래치 증발로 저장소에 없다(작업 79). 재실행 시 신런으로 대체하고
> `_meta.status`의 PROVISIONAL을 제거한다.

---

## 하지 않은 것

- **재실행** — 범위 확정이 먼저(§0.1).
- **신 스케줄 실제 생성** — 규칙만, 사람 확인 대기(85).
- **정상 경로 알고리즘·파라미터 변경** — 수정은 86·87(오류 처리/해소 검증)에 한정.
- **누락 모델(mnasnet 등) 도입** — 논문 워크로드에 없음.
- **v19 작업 73 fail-safe** — 보류(phantom은 그 경로를 안 탐).
