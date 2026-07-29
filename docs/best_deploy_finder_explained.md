# `best_deploy_finder_executor.py` — 구현 설명

> 이 문서는 현재 저장소(`multimodel-scheduling-mobilint`, 브랜치 `ubuntu_gpu_mobilint`)에 구현된
> `best_deploy_finder_executor.py`의 동작을 다른 Claude 대화에 그대로 전달할 목적으로 작성한 자료입니다.
> 코드는 PyQt5 GUI이며, 사용자가 배치할 모델을 고르면 학습된 XGBoost 예측기로 **최적 디바이스 배치(placement)**를
> 찾아 실행까지 연결합니다.

---

## 1. 한 줄 요약

이 GUI는 **CPU + Mobilint NPU(MLA100)** 플랫폼을 대상으로,
사용자가 체크한 모델들을 CPU/NPU에 배치하는 모든 조합을 열거하고,
학습된 XGBoost 예측기(`deploy_cpu_npu`)로 각 조합의 점수를 매겨
**최적 조합 번호**를 화면에 표시하고, 그 조합으로 실행기(`schedule_executor_main.py`)를 띄웁니다.

과거의 **CPU + Neubla NPU** 흐름을 대체한 것으로, 플랫폼은 CPU+NPU로 고정되어 있고 두 디바이스 모두
여러 모델을 동시에 호스팅할 수 있는(shareable) 것으로 취급합니다.

---

## 2. 실행 방법 / 화면 구성

```bash
source runtime_env.sh
$PYTHON_BIN best_deploy_finder_executor.py [--models-root ./models]
```

UI는 `best_deploy_finder_executor.ui`(Qt Designer)로 로드되며, 주요 위젯:

| 위젯 | 역할 |
|------|------|
| `model_tree_view` | 배치 가능한 **모델 이름(확장자 없음)** 체크리스트 |
| `prediction_model_input` | 예측기 프리픽스 경로 (기본값 `xgboost_model/artifacts/deploy_cpu_npu`) |
| `deployment_model_input` | 모델 루트 경로 (기본 `./models`) |
| `device_config_input` | (현재는 사용 안 함, CPU+NPU 고정) |
| `input_rate_button` | 모델별 입력 rate(infps) 설정 다이얼로그 |
| `predict_best_button` | **Predict Best Deployment** — 핵심 버튼 |
| `load_execute_best_button` | 최적 조합으로 실행기 subprocess 실행 |
| `label_best_deploy_value` | 예측된 최적 조합 번호 표시 |
| `log_text_edit` | 로그 출력 |

---

## 3. 모델 목록은 어떻게 채워지나 (`_reload_model_list`)

트리에 표시되는 "배치 가능한 모델"은 두 종류를 합쳐서 만듭니다.

### 3.1 파일 기반 비전 모델 — `discover_file_backed_models(models_root)`
- `models/onnx/<name>.onnx` → CPU에서 실행 가능
- `models/mobilint/<name>.mxq` → NPU에서 실행 가능(컴파일된 형태)
- **두 파일이 모두 있는 모델만** deployable. `.onnx`만 있으면 `cpu_only`, `.mxq`만 있으면 `npu_only`로 로그에만 표시하고 목록에서 제외.
- 반환: `(deployable, cpu_only, npu_only)` — 모두 확장자 없는 이름 리스트.

### 3.2 생성 모델(LLM/VLM) — `discover_hf_backed_models(static_json_path)`
- `llama1b`, `qwen2_vl` 등은 로컬 `.onnx/.mxq`가 없고 Mobilint 런타임이 HuggingFace에서 로드.
- 그래서 **정적 프로파일(static profile) JSON에 항목이 있는 모델만** 노출 — 프로파일이 있어야 예측기가 필요로 하는 feature가 존재하기 때문.
- `model_registry`에서 `kind_of(m)`이 `llm`/`vlm`이고 프로파일된 모델을 반환.

최종 목록 = `set(file_backed) | set(hf_backed)` 를 정렬한 것.
`ModelChecklistModel`(QStandardItemModel)에 체크박스 아이템으로 채워집니다.

---

## 4. **Predict Best Deployment** 전체 흐름 (`on_predict_best_clicked`)

이 버튼이 핵심입니다. 순서대로:

### Step 0 — 입력 검증
- 체크된 모델이 없으면 에러.
- `prediction_model_input`이 실제 경로(프리픽스, 디렉터리, 또는 `_y1.json`)를 가리키는지 확인.

### Step 0.5 — **OOD(분포 밖) 경고** — `check_selection_coverage`
예측기 옆의 `<prefix>_coverage.json`을 읽어, 사용자의 선택이 **학습 분포 안에 있는지** 검사합니다.
`deploy_cpu_npu`는 항상
`{yolo11 변종 1개, resnet50, llama1b, qwen2_vl}` (4개, rate 1~4x) 조합으로만 학습되었습니다.

- 선택이 학습된 working set 중 하나와 **정확히 일치**하면 → 조용히 통과.
- 아니면 다음 경우를 각각 경고 문구로 수집:
  - **한 번도 학습 안 된 모델**이 포함됨 (`Never trained on: ...`)
  - **모델 수(view count)**가 학습 범위(예: 4개)와 다름
  - **모든 학습 working set에 항상 있던 모델**(llama1b, qwen2_vl 등)을 뺐음
- 경고가 있으면 로그에 남기고 **모달 다이얼로그(Ok/Cancel, 기본 Cancel)**를 띄움 → 사용자가 Ok 눌러야 진행.
  진행하더라도 그 순위는 "외삽(extrapolation)"임을 명시.

> 이 커버리지 파일은 학습 시 `deploy_selector_xgb_suite.py train`이 자동으로 기록합니다
> (모델 집합/모델 이름/view 수/디바이스/rate factor).

### Step 1 — 스케줄 YAML 생성 (`generate_all_combinations` → `build_schedule_from_selection`)
체크된 모델들에 대해 CPU/NPU 배치 조합을 **모두 열거**합니다.

- 플랫폼 디바이스 = `["cpu", "npu"]`, 두 디바이스 모두 shareable.
- `generate_schedules.enumerate_placements(models, ["cpu","npu"])`가
  각 모델을 **독립적으로** 허용 디바이스에 배정 → 기본 `2^N` 조합
  (단 per-model 제약 적용: `model_registry.DEVICE_CONSTRAINTS`에 `qwen2_vl: [gpu, npu]`이므로
  CPU+NPU 플랫폼에서 qwen2_vl은 **NPU에만** 배치 가능 → 실제 조합 수는 그만큼 줄어듦).
- 각 조합은 `combination_1`, `combination_2`, … 로 이름 붙고, 뷰별 엔트리:
  ```yaml
  combination_1:
    yolo11n_cpu:
      model: yolo11n
      execution: cpu
      display: view1
      infps: 46.984   # 입력 rate
  ```
- **infps 결정 규칙**: 사용자가 input-rate 다이얼로그에서 지정한 값이 있으면 그것,
  없으면 정적 프로파일의 `baseline_rate`(단독 최소 디바이스 처리량 = 1x), 그것도 없으면 1.0.
- 사이드카 `<out>.meta.json`에 조합별 `{rate_factor: 1.0, workload: <detection 모델>}` 기록.
- 결과 파일: `model_schedules.yaml` (+ `.meta.json`).

### Step 2 — XGBoost 예측 (`predict_best_combination`)
`deploy_selector_xgb_suite`의 함수를 그대로 사용:

1. `load_static_profiles(static_json)` — feature 소스(정적 프로파일) 로드.
2. `_iter_combos_from_schedule(schedule_doc)` — YAML의 각 조합을 순회.
3. 각 조합에 대해 `featurize_from_combo(S, combo_blob)` → **누수 없는 정적 feature 벡터**(뷰별 feature를 sum/mean/max로 집계, 37개).
4. `predict_targets(prefix, X)` → 3개 타깃 예측(각각 `[0,1]`로 클립):
   - **y1** = 정규화 처리량(normalized throughput)
   - **y2** = deadline miss rate
   - **y3** = 정규화 토큰 처리량(normalized tokens/s)
5. **런타임 점수**:
   ```
   S = y1 + β·y3 − α·y2      (기본 α=0.3, β=0.5)
   ```
   > ⚠️ α, β는 **학습에 포함되지 않습니다.** 순수 런타임 스코어링 가중치입니다.
6. 점수 내림차순 정렬 → `predictions.csv`로 저장, Top-1 조합을 반환.

### Step 3 — 결과 표시
- 최적 조합 번호를 `label_best_deploy_value`에 표시.
- Top-5 조합과 점수를 로그에 출력.

---

## 5. **Load & Execute Best** (`on_load_execute_best_clicked`)

1. `predictions.csv`가 있으면 첫 행(=최고 점수 조합)을 best로 사용.
2. 없으면 선택된 모델을 **전부 CPU에** 올린 단일 조합(`_build_cpu_only_schedule`)을 fallback으로 생성.
3. `schedule_executor_main.py`를 **별도 subprocess**로 실행
   (`--schedule <yaml> --schedule-name <combo> --duration 60`).
   별도 프로세스로 띄우는 이유는 중첩 QApplication을 피하기 위함.

---

## 6. 입력 rate 다이얼로그 (`on_input_rate_clicked`)

- `input_rate_dialog.ui`를 로드, 선택된 각 모델마다 `QDoubleSpinBox`(0.1~1000.0, 기본 10.0) 한 줄씩 생성.
- 확정 시 `self.input_fps_by_model[model] = value`에 저장 → 이후 스케줄 생성의 infps로 사용.

---

## 7. 예측기 경로 처리 (`_infer_model_prefix`)

`prediction_model_input`은 세 형태를 모두 허용:
- **bare 프리픽스**: `.../deploy_cpu_npu` (옆에 `_y1.json` 존재)
- **디렉터리**: 안에서 `_y1.json`+`_y2.json` 짝이 맞는 프리픽스 탐색
- **JSON 파일**: `..._y1.json` 또는 `..._y2.json` → 프리픽스와 짝 파일 존재 확인

기본값은 생성자에서 `xgboost_model/artifacts/deploy_cpu_npu`로 **명시적으로** 세팅
(여러 예측기가 있어 프리픽스가 모호해지지 않도록).

---

## 8. 예측기 아티팩트 구조

`xgboost_model/artifacts/deploy_cpu_npu_*`:

| 파일 | 내용 |
|------|------|
| `_y1.json` | 정규화 처리량 회귀기 (XGBoost) |
| `_y2.json` | deadline miss rate 회귀기 |
| `_y3.json` | 정규화 토큰 처리량 회귀기 |
| `_features.json` | feature 컬럼 순서 (예측 시 정렬용) |
| `_coverage.json` | **학습 커버리지** — model_sets / models / view_counts / devices / rate_factors (OOD 검사에 사용) |

저장소에는 4개의 예측기가 존재: `deploy_xgb`(3-device), `deploy_cpu_gpu`, `deploy_cpu_npu`, `deploy_conc_cpu_gpu`(N=4~8 동시 실행).
이 GUI는 그중 **`deploy_cpu_npu`**를 기본으로 사용.

---

## 9. 주요 함수 요약

| 함수 | 역할 |
|------|------|
| `discover_file_backed_models` | onnx ∩ mxq 로 CPU/NPU 둘 다 되는 비전 모델 탐색 |
| `discover_hf_backed_models` | 프로파일된 LLM/VLM 을 레지스트리에서 탐색 |
| `_reload_model_list` | 위 둘을 합쳐 트리뷰 채우기 |
| `check_selection_coverage` | 선택이 학습 분포 안인지 검사, 경고 리스트 반환 |
| `build_schedule_from_selection` | CPU/NPU 배치 조합 열거 → YAML + meta 작성 |
| `predict_best_combination` | 조합별 3-타깃 예측 → 점수 → predictions.csv, best 반환 |
| `on_predict_best_clicked` | Predict 버튼: 검증→OOD경고→스케줄생성→예측→표시 |
| `on_load_execute_best_clicked` | best 조합(또는 CPU-only fallback)으로 실행기 실행 |
| `_infer_model_prefix` | 예측기 경로(프리픽스/디렉터리/JSON) 정규화 |

---

## 10. 핵심 설계 포인트 (질문 시 참고)

- **플랫폼 고정**: CPU + Mobilint NPU, 둘 다 shareable → `2^N` 배치 공간 (per-model 제약으로 감소).
- **3개 독립 XGBoost 타깃**(y1/y2/y3), 각각 3-fold CV로 하이퍼파라미터 선택, 예측은 `[0,1]` 클립.
- **정규화**: y1/y3는 (workload, rate) 그룹 내에서 `T = F/Fmax`로 정규화되어 학습됨.
- **feature 누수 없음**: 측정된 런타임 처리량을 feature로 쓰지 않고, 정적 프로파일 + 배치/rate 계획만 사용(37개).
- **α, β는 학습 밖**: 런타임 점수식 `S = y1 + β·y3 − α·y2`에서만 사용 (기본 0.3 / 0.5).
- **OOD 경고**: 학습은 항상 4-뷰 `{yolo11 1종, resnet50, llama1b, qwen2_vl}` 조합. 비전-only나 5개 이상 선택 시 경고.
- **qwen2_vl 제약**: CPU 배치 불가(NPU에만) — `DEVICE_CONSTRAINTS`.
