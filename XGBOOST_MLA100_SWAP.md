# XGBoost 예측기 교체 (MLA100 재학습본)

NPU를 MLA100으로 바꾸고 모델 구성·working set을 변경해 재학습한 XGBoost 배치 예측기로
교체한 작업의 기록. 이전 예측기는 논문 그림 재현을 위해 legacy로 병존한다.

- 브랜치: `swap/xgboost-mla100-20260720_174523` (3 커밋, **미병합**)
- 신규 모델 출처: `/home/msyu/PycharmProjects/multimodel-scheduling-mobilint` (읽기 전용, 복사만 함)
- 백업: `backup/xgboost_pre_mla100_20260720_174523/` (65개 파일 + sha256 기록된 `MANIFEST.yaml`)

---

## 1. 왜 파일 교체가 아니라 번들 교체였나

모델 파일(.json)만 바꿨으면 깨졌다. 재학습 과정에서 가중치가 아니라 **파이프라인 전체**가 바뀌었다.

| | 구(pre-MLA100) | 신규(MLA100) |
|---|---|---|
| 피처 | 해시 기반 `pairhash_*` / `model_hash_*`, 60~169개 | 의미 기반 37개 (`view.infps`, `exec_cpu/gpu/npu`, `is_vision`, `is_llm`, `static_infer_sel`, `static_load_sel`, `static_tokens_sel`, `capacity_fps`, `load_factor` 의 sum/mean/max + `views.count.views`) |
| 타깃 | 2개 — y1=원시 FPS, y2=drop rate | 3개 — y1=정규화 처리량, y2=deadline miss rate, y3=정규화 토큰 처리량 |
| 스케일 | 원시값 | 셋 3개 모두 **워킹셋 내부** [0,1] 정규화 |
| 스코어 | `S = y1 − α·y2` | `S = y1 + β·y3 − α·y2` |
| featurize 시그니처 | `featurize_from_combo(combo_blob)` | `featurize_from_combo(S, combo_blob)` — 정적 프로파일 `S` 필요 |
| 추가 의존성 | 없음 | `model_registry.py`, `sample_profiling_data.json` |

정적 프로파일 형식도 호환되지 않는다. 구 FSRR은 `cpu_data` 리스트에 `.onnx` 접미사가 붙은
모델명을 쓰고, 신규는 `total_data`에 `{device}_{key}` 평면 키와 정제된 모델명을 쓴다.
그래서 MOBILINT의 `sample_profiling_data.json`을 함께 가져왔다.

### 스코어의 성질 (중요)

`y3` 항은 토큰을 생성하는 셋에서만 목적함수에 **들어간다** — 비전 전용 셋에서 계산 후
0으로 만드는 게 아니라 애초에 포함되지 않는다. 비전 전용 셋에 대해 예측기는 학습된 적 없는
0이 아닌 토큰 처리량을 내놓기 때문에, 사후 처리 방식이면 그런 조합마다 `β·y3` 만큼의
공짜 점수가 붙는다.

부작용으로 스코어의 스케일이 셋 유형에 따라 달라진다. **스코어는 하나의 워킹셋 안에서만
비교 가능하다** (배치 랭킹이 유일한 용도). 워킹셋 간 비교는 의미가 없다.

---

## 2. 현재 코드 구조 — 무엇을 쓸 것인가

### 신규 경로 (기본. 새 코드는 이쪽을 쓸 것)

| 파일 | 역할 |
|---|---|
| `deploy_predictor_logic.py` | `DeployPredictor` — 3-타깃 예측, 가드레일 포함 |
| `xgboost_model/deploy_selector_xgb_suite.py` | 피처 추출 / 전처리 / 정규화 / 로더 / 스코어링 |
| `model_registry.py` | 모델 종류 판정 (`kind_of` → vision/llm/vlm). 9개 모델 등록 |
| `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json` | 정적 per-device 프로파일 (featurizer 입력) |
| `xgboost_model/artifacts/cpu_npu/deploy_cpu_npu_{y1,y2,y3,features,coverage}.json` | MLA100 NPU 재학습본 |
| `xgboost_model/artifacts/cpu_gpu/deploy_cpu_gpu_{y1,y2,y3,features,coverage}.json` | 동일 540-row 수집 기반 GPU본 |

반환 DataFrame 컬럼: `source`, `combination`, `pred_norm_throughput`,
`pred_deadline_miss_rate`, `pred_norm_tokens`, `has_generative`, `pred_score`.

신규 예측기 coverage (`*_coverage.json`): 9개 모델
(`llama1b`, `mobilenet_v2`, `qwen2_vl`, `resnet50`, `yolo11{n,s,m,l,x}`),
15개 워킹셋, view 수 2~8, rate factor {1.0, 1.5, 2.0, 3.0, 4.0}, 540 rows.

### Legacy 경로 (논문 그림 재현 전용. 새 코드에서 쓰지 말 것)

| 파일 | 역할 |
|---|---|
| `deploy_predictor_logic_legacy.py` | 구 2-타깃 `DeployPredictor` |
| `xgboost_model/deploy_selector_xgb_suite_legacy.py` | 구 해시 기반 suite (git상 rename으로 이력 보존) |
| `xgboost_model/artifacts/{npu,gpu}/xgb_model_*` | 구 아티팩트 (그대로 유지) |

**legacy를 지우면 안 되는 이유**: 발표된 그림의 워킹셋(`resnext50`, `vgg19`, `yolov4`)이
신규 예측기 coverage에 없다. 이 스크립트들을 신규 번들로 돌리면 unprofiled-model 에러만 난다.
재현 가능한 그림을 에러와 맞바꾸는 셈이라 legacy에 고정했다.

legacy에 고정된 소비자: `measure_inference_time.py`,
`scripts/{q13_failure_persistence, ml_misprediction_validation, xgb_alpha_sweep,
xgb_alpha_trained_sweep, xgb_best_selector, recompute_comparison_baselines}.py`.

`grep`으로 `xgb_model_x3_double` 등이 잡히는 건 전부 이 재현 경로이며 의도된 상태다.

---

## 3. 배선에서 잡은 함정 두 가지

**(a) 아티팩트 디렉터리 분리** — GUI는 예측기에 **디렉터리 경로**를 넘긴다
(`best_deploy_finder_executor.py`가 `pred_model = base_dir`로 설정). 두 플랫폼 모델을
`artifacts/` 한 곳에 두면 prefix glob이 알파벳순으로 `deploy_cpu_gpu`를 잡아, NPU를
선택해도 GPU 모델로 조용히 예측한다. 그래서 `artifacts/cpu_npu`와 `artifacts/cpu_gpu`로
분리했다. **이 구조를 평탄화하지 말 것.**

**(b) GUI 라벨 의미 변경** — 구 GUI는 `pred_total_throughput_fps`를 `"N FPS"`로 표시했다.
신규 y1은 정규화 [0,1]이라 FPS가 아니다. 컬럼명(`pred_norm_throughput`)과 표시 라벨
(`"N (norm)"`)을 함께 고쳤다.

---

## 4. 가드레일 (조용한 오예측 차단)

신규 `DeployPredictor`는 예측 전에 두 가지를 검사하고, 실패하면 예외를 던진다. 둘 다
그냥 두면 "부분적으로 0인 피처 벡터에서 나온 자신만만한 숫자"가 되기 때문이다.

- **`_validate_feature_vector`** — 생성된 컬럼이 `*_features.json`의 학습 컬럼과 정확히
  일치하는지 확인. `_align_features`는 없는 컬럼을 조용히 0으로 채우고 남는 컬럼을 조용히
  버리므로, 신뢰하지 않고 검사한다.
- **미프로파일 모델 거부** — `_device_static`은 프로파일 없는 모델에 NaN을 반환하고
  `featurize_from_combo`가 NaN을 0.0으로 채운다. 정적 프로파일에 없는 모델이 조합에 있으면
  예측 자체를 거부한다.

두 가드 모두 실제 동작을 확인했다 (§5).

---

## 5. 검증 결과 (전부 통과)

| 항목 | 결과 |
|---|---|
| 6a 로드 & 스키마 | booster 6개(2 플랫폼 × y1/y2/y3) 전부 로드, `num_features=37`, `feature_names`가 `*_features.json`과 완전 일치 |
| 6b 엔드투엔드 | vision-only × mixed × cpu_npu × cpu_gpu = 4조합. `has_generative` 분기 정상, y1·y2 모두 [0,1], 스코어 내림차순 정렬 유효 |
| 가드레일 | 미프로파일 모델(`vgg19`) → `ValueError: No static profile for`. 구 아티팩트 오지정 → `ValueError: Feature mismatch (trained=169, produced=37)` |
| legacy 경로 | 구 워킹셋 16조합 정상 랭킹, 구 컬럼명 유지 |
| 6c 잔존 참조 | 활성 코드에 구 프리픽스 없음. 남은 참조는 전부 legacy 재현 경로 |
| 6d 회귀 | 변경된 13개 파일 전부 컴파일, 메인 앱 headless import 성공, 디렉터리 → prefix 해석이 두 플랫폼 모두 정확 |

---

## 6. 미결 항목 — β값 충돌 (사람 판단 필요)

**작업 지시문은 논문 공식으로 β=0.5를 명시했고, 이식해 온 MOBILINT 코드의 canonical
default는 β=1.0이다.**

- β는 학습된 파라미터가 아니라 런타임 정책 가중치다. **모델 자체에는 영향이 없다.**
- 비전 전용 셋은 y3 항이 아예 빠지므로 두 값에서 결과가 동일하다.
- **LM/VLM을 포함한 셋의 랭킹만 달라진다.**

현재 이 저장소는 지시문을 따라 `deploy_predictor_logic.DEFAULT_BETA = 0.5`로 설정돼 있고,
코드 주석에 MOBILINT(1.0)와의 차이를 명시해 뒀다. MOBILINT와 맞추려면 그 상수 하나만
바꾸면 된다. `xgboost_model/deploy_selector_xgb_suite.py`의 `DEFAULT_BETA`는 1.0 그대로
두었다(이식 원본 보존).

α는 양쪽 모두 0.3으로 일치한다.

---

## 7. 롤백

```bash
git switch -                                        # 원 브랜치로
git branch -D swap/xgboost-mla100-20260720_174523   # 필요 시 폐기
# 또는 백업 스냅샷에서 개별 복원
rsync -aR backup/xgboost_pre_mla100_20260720_174523/./<상대경로> .
```

백업 폴더와 `MANIFEST.yaml`은 읽기 전용으로 유지한다. 덮어쓰지 말 것.

---

## 8. 부수 사항

- 삭제돼 있던 `.venv`를 `requirements.txt` 기준으로 복구했다. 예측기 관련 pin
  (xgboost 3.2.0, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3)은
  MOBILINT와 이미 동일하다.
- `.gitignore`를 추가했다 (`__pycache__/`, `*.pyc`, `.venv/`, `.idea/`).
- 검증 스모크 런이 덮어쓴 `predictions.csv`는 HEAD로 되돌려 커밋에서 제외했다.
- 병합은 사람이 검토 후 결정한다. 자동 merge 하지 않았다.
