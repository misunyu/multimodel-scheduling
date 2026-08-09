<!-- 작업 17 — 읽기 전용 확인 2건. 코드·데이터·원고 미수정. -->

# 원고 마무리 확인 2건

---

## 1. CPU-only 생성 후보의 ŷ3 경로

대상: 생성 세트인데 생성 모델이 **전부 CPU**에 배치된 후보. S3(llama1b·resnet50·yolo11s)에서
llama1b가 CPU인 **4배치 × 3 rate = 12창** (플랫폼당). `qwen2_vl`은 항상 가속기이므로 이런
후보는 S3에서만 나온다.

### (a) 예측 경로 — **별도 처리 없다. ŷ3를 예측해 점수에 그대로 넣는다.**

| 단계 | 코드 | 동작 |
|---|---|---|
| y3 부스터 **학습** | `train_groupkfold.py:56` (joint은 `train_unified.py:56`) | `valid = y3_valid` — 이 후보들은 학습 행에서 **제외** |
| y3 **예측** | `train_groupkfold.py:71` (joint `:64`) | `pred[te] = predict_booster(bst, Xv[te])` — held-out fold의 **전 행**을 예측. `valid` 마스크가 걸려 있지 않다 |
| 점수화 | `analysis_common.py:188` | `y1 − α·y2 + β·y3·has_gen`. `has_gen`은 **세트 단위**(세트에 LLM/VLM이 있는가)이고 배치와 무관 → S3은 True → **β·ŷ3가 더해진다** |

`train_groupkfold.py:68-70` 주석이 이 설계를 명시한다: *"y3 boosters are trained on valid
rows only but PREDICT every held-out row: the production predictor scores all 12 placements
of a generative set, including LLM-on-CPU ones whose measured y3 is masked noise."*

`scores_from`의 `np.nan_to_num(y3, nan=0.0)` (`:186`)은 결측 방어일 뿐, 이 후보들의 ŷ3는
NaN이 아니라 실제 예측값이므로 0으로 치환되지 않는다.

**실측 확인 (S3의 12창, 플랫폼별):**

| 플랫폼 | ŷ3 (예측) | y3_meas (측정) | 예측 점수에 더해지는 β·ŷ3 |
|---|---|---|---|
| GPU | **0.123 ~ 0.263** | 0.0 | +0.12 ~ +0.26 |
| NPU | **0.904 ~ 0.981** | 0.0 | **+0.90 ~ +0.98** |

전부 0이 아니고, **NPU에서는 거의 최대치(≈1.0)의 토큰 보상이 붙는다.** NPU의 y3 부스터는
가속기-LLM 창만 보고 학습했는데 그 창들의 정규화 y3가 대부분 1.0 근처라, LLM이 CPU에 있는
입력에 대해서도 "토큰이 많이 나온다"고 외삽한다. GPU 부스터가 0.12~0.26에 머무는 것과
대조적이다 — 같은 결함이 플랫폼에 따라 4~8배 다른 크기로 나타난다.

운영(GUI) 경로도 동일하다: `deploy_selector_xgb_suite.py:795`가
`with_y3 = combo_has_generative(combo_blob)` — **세트 단위** 판정이라 CPU 배치 여부와 무관하게
y3 부스터를 로드해 예측한다.

### (b) 랭킹·Top-1 산정 — **포함된다.**

`ranking_metrics` (`analysis_common.py:191-212`)는 그룹의 **모든 행**을 대상으로 argmax를
잡는다. 배제 필터가 없다. S3의 8개 후보 전부(그중 4개가 CPU-only 생성)가 매 rate에서
경쟁하고, 45그룹 분모에도 S3의 3그룹이 그대로 들어간다.

실제 결과는 무해하다: **S3 6그룹(2플랫폼 × 3 rate) 전부 Top-1 정답**이고 oracle ratio 1.000이다.
가속기-LLM 후보는 y3_meas가 1.0이라 측정 점수에서 이미 +1.0을 받는데, CPU-only 후보는
ŷ3 과대평가로 최대 +0.98을 받아도 y1·y2 열세를 뒤집지 못한다. **NPU에서도 6/6 정답**이므로
이 결함이 보고된 지표를 바꾸지는 않는다.

### (c) 측정 점수 쪽 — **실측 r3를 쓰지 않는다. 0이 들어간다.** ⚠

파이프라인의 측정 y3는 `featurize_window`가 **뷰별 `tokens_per_s`를 합산**한 값이다
(`deploy_selector_xgb_suite.py:323`). 그런데 저장된 창 파일에서 이 후보들의 뷰별 토큰은
**전부 0**이다 — 수집 JSONL이 뷰별 토큰을 버렸고(`collect_run.py:122-123`),
`jsonl_to_windows.py:32-44`가 창 총계를 "가속기 위 생성 뷰"에만 얹는데 이 후보들에는
그런 뷰가 없어 **아무 뷰에도 얹지 못했기** 때문이다.

| | 값 |
|---|---|
| 창 총계 `total.total_tokens_per_s` (실측, CPU 토큰 포함) | **0.0 ~ 2.02 tok/s** (평균 0.535) |
| 파이프라인이 쓰는 y3 (뷰별 합) | **0.0** (12창 전부) |

즉 **CPU 위 LLM이 실제로 낸 토큰은 P1/joint/P5/P6의 측정 점수에 반영되지 않는다.**
(기저 감사·Figure 2는 창 총계를 쓰므로 이쪽은 반영된다 — 저장소에 r3 관례가 두 개다.)

**수치 영향은 없다**: 정규화 분모 `max(r3)`가 두 관례에서 S3 전 그룹 동일하고
(같은 그룹에 가속기-LLM 후보가 있어 그쪽이 최댓값), 45그룹 어디에서도 최적 배치가 바뀌지 않는다.

### 원고 초안 (1–2문장)

> A generative workload whose language model is placed on the CPU still receives a
> predicted token rate: the $y_3$ regressor is trained only on windows where a generative
> instance runs on the accelerator, but it scores every held-out candidate, and the
> $\beta$ term is applied per workload set rather than per placement, so such candidates
> compete in the ranking on equal footing. Their measured $y_3$ is zero, because per-view
> token rates were not retained during collection and the window total is attributed to
> the accelerator-side generative instance, which these candidates do not have.

한 문장만 넣는다면 앞 문장이 핵심이고, 뒤 문장은 "그럼 실측 r3는 어떻게 되느냐"는 리뷰어
질문을 닫는 용도다. 뒤 문장을 넣지 않으려면 최소한 §Performance targets 쪽에
"$y_3=0$ for candidates with no accelerator-side generative instance"를 명시해야 한다.

---

## 2. 하이퍼파라미터 선택 절차 (App F 보강용)

근거: `analysis_common.cv_select_params` (`:228-256`), 그리드·기본값은
`deploy_selector_xgb_suite.py:519-530`.

| 항목 | 값 |
|---|---|
| **탐색 그리드 (4점)** | `(max_depth, eta)` = **(3, 0.1), (4, 0.1), (5, 0.05), (6, 0.1)** (`suite:525-530`) |
| 고정 파라미터 | `objective=`**`reg:squarederror`**, `eval_metric=rmse`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3.0`, `seed=42` (`suite:519-523`). 그리드가 덮어쓰는 것은 `max_depth`·`eta` 뿐 |
| 부스팅 라운드 | **300** 고정 (`analysis_common.py:229`) |
| **inner CV fold 수** | **3** (`analysis_common.py:240`, `for k in range(3)` `:245`) |
| **분할 방식** | **그룹 존중**. `assign_group_folds(inner_groups, n_folds=3, seed=42)` — 바깥 fold와 같은 함수·같은 그룹 키 `(models, rate_factor)`. 한 그룹의 행이 inner fold를 가로지르지 않는다 (`:239-240`) |
| **선택 지표** | **MAE (mean absolute error) 최소화**. 3개 inner fold의 MAE 평균이 가장 작은 조합을 채택 (`:252-255`) |
| 적용 범위 | 바깥 fold마다 독립 수행. 타깃(y1/y2/y3)마다 독립 수행 |

**주의해서 쓸 점**: 학습 목적함수는 제곱오차(`reg:squarederror`)인데 **모델 선택 기준은
MAE**다. 서로 다른 지표이므로 원고에서 "selected by the training objective"라고 쓰면 부정확하다.

**대조 (원고에 쓸 필요는 없음)**: 출하 경로의 `deploy_selector_xgb_suite._cv_select_params`
(`:534`)는 같은 4점 그리드를 쓰지만 inner fold를 `arange(n) % folds`로 나눈다 — 그룹을
존중하지 않는다. P1~P6는 그룹 인지 버전을 쓴다.

### 원고 초안 (1–2문장)

> For each outer fold and each target we select $(\texttt{max\_depth}, \eta)$ from a
> four-point grid --- $(3, 0.1)$, $(4, 0.1)$, $(5, 0.05)$, $(6, 0.1)$ --- by an inner
> three-fold cross-validation that is split on the same workload groups as the outer
> folds, choosing the combination with the lowest mean absolute error; all other
> settings are fixed (squared-error objective, 300 boosting rounds, subsample and
> column-sample 0.8, minimum child weight 3, seed 42).

부록 표로 넣는다면:

| Setting | Value |
|---|---|
| Search grid | $(\texttt{max\_depth},\eta) \in \{(3,0.1),(4,0.1),(5,0.05),(6,0.1)\}$ |
| Selection | inner 3-fold CV, **group-aware** split, minimum **MAE** |
| Objective | `reg:squarederror`, 300 rounds |
| Fixed | subsample 0.8, colsample\_bytree 0.8, min\_child\_weight 3.0, seed 42 |

---

## 확인 방법

```bash
# 1(a) 예측 경로
sed -n '54,72p'   xgboost_model/full_collection_540/scripts/train_groupkfold.py
sed -n '183,212p' xgboost_model/full_collection_540/scripts/analysis_common.py
# 1(c) 측정 y3
sed -n '318,326p' xgboost_model/deploy_selector_xgb_suite.py
sed -n '32,44p'   xgboost_model/full_collection_540/scripts/jsonl_to_windows.py
# 2 하이퍼파라미터
sed -n '228,256p' xgboost_model/full_collection_540/scripts/analysis_common.py
sed -n '519,530p' xgboost_model/deploy_selector_xgb_suite.py
```

1의 수치(ŷ3 GPU 0.123~0.263 · NPU 0.904~0.981, y3_meas 0.0, S3 6그룹 Top-1 전부 정답)는
`analysis/groupkfold_{gpu,npu}_oof.csv` 와
`runs/20260809_140605_checklist_r2/F_regret/per_group_direct_{gpu,npu}.csv` 에서 읽었다.
1(c)의 창 총계 범위는 `runs/20260809_140605_checklist_r2/E_r3_definition/r3_facts.json`.
