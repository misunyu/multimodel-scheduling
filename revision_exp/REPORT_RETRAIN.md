# CPU–GPU 재학습 및 Table 3(a) · Fig. 3 · Fig. 4 재생성 (REPORT_RETRAIN)

작성일: 2026-10-13 · 범위: CPU–GPU, 저장된 profiling 데이터 그대로 · 선행 문서: `revision_audit/REPORT.md`, `REPORT2.md`, `REPORT3.md`

표기: **[확인]** 코드·데이터·실행 결과로 직접 확인. **[추정]** 추론. **[확인 불가]** 확인하지 못함.

## 결론

Family 단위 GroupKFold로 다시 골라도 T와 D의 hyperparameter는 기존과 같았고, Ours의 수치는 원고와 사실상 같다(Avg 0.951, Top-1 13/18, Top-5 17/18, MAE 0.060 / 0.090). 바뀐 것은 baseline이다. Latency-based는 0.63에서 0.676으로, Best-of-5는 0.78 ± 0.040에서 0.730 ± 0.039로 움직였고, 둘 다 예상 범위였다. 바뀌지 않은 것은 All-GPU와의 관계다. All-GPU는 0.943(17 workloads)이고 같은 17개에서 Ours는 0.950이다. 차이 0.007은 workload 2개에서 나오며, 같은 hyperparameter로 다시 학습한 것만으로 생기는 변동(기존 artifact 0.945 대 재학습 0.951)과 같은 크기다. Ours는 18개 중 15개에서 전부-GPU 배치를 골랐다. 즉 현 데이터에서 Ours가 All-GPU보다 낫다고 말할 근거는 없다. 예상과 다른 결과는 없었다.

---

## 0. 실행 기록

### 0.1 명령과 시간

작업 디렉터리는 저장소 루트, interpreter는 `.venv/bin/python3 -B`, `PYTHONDONTWRITEBYTECODE=1`, `MPLCONFIGDIR`는 세션 임시 폴더다.

| 실행 | 명령 | 시각 | 소요 |
|---|---|---|---|
| Probe | `revision_exp/run_all.py --probe-only` | 13:17:21 – 13:18:00 | 39 s |
| 전체 | `revision_exp/run_all.py` | 13:18:15 – 13:22:02 | 228 s |

전체 실행의 단계별 시간(`out/logs/stage_times.csv`):

| 단계 | 소요 (s) |
|---|---|
| data (5개 집합 load, feature 생성, 기존 builder와 대조) | 37.5 |
| timing probe | 1.6 |
| train (Multi 3 모델 + Single 10 모델) | 185.1 (Multi 42.4, Single 142.7) |
| validation, evaluate, fig data, plots, bookkeeping | 3.0 |

- **Fit당 시간**: 가장 무거운 grid 점(1,000 trees, depth 6)으로 T 모델 3 fold를 재서 평균 0.53 s였다. 637 fits(144 + 480 + 13)의 상한을 5.7분으로 추정했고 실제 학습은 3.1분이었다. 2시간 기준에 한참 못 미쳐 grid는 그대로 썼다.
- **환경**: Python 3.10.12, xgboost 3.0.5, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3. Intel Core Ultra 9 285K, 24 cores, `n_jobs=-1`. Master seed 42.
- 로그: `out/logs/run_all.log`, `run_all_stdout.txt`, `environment.json`, `timing_probe.json`, `training_time.json`.

### 0.2 부작용 확인

- `git status --short`는 실행 전후가 같다(`out/logs/git_status_before.txt`, `…_after.txt`). `revision_exp/` 외에 수정된 tracked file은 없다.
- 보호 대상 1,247개 파일(`experimental_results/`, `xgboost_model/artifacts/`, `prediction_result/`, `dataset/`, `scripts/`, `results*/`, `gen_schedules*/`, `sample_profiling_data/`, `deploy_selector_xgb_suite.py`)의 sha256이 실행 전후 **전부 같다**(`out/logs/protected_hash_diff.txt`).
- 프로젝트 디렉터리와 `revision_exp/`에 새 `__pycache__`는 없다.

### 0.3 입력

`out/input_manifest.csv`에 57개 파일의 경로, 크기, sha256이 있다: dataset CSV 10개, 기존 artifact 3개, N=1 결과 파일 39개(`results_recompute/` 12, `results_recompute_additional/` 27), offline profile 1개, 기존 그림 CSV 1개, audit 표 2개, `deploy_selector_xgb_suite.py`.

### 0.4 지시문과 다르게 한 것

- 기존 모듈에서 지시문에 적힌 함수 외에 `_index_schedules_from_csv`, `_find_schedule` 두 개를 더 import했다. Schedule CSV를 읽어 `_build_infps_lookup`에 넘기는 데 필요하고, 복사하지 않기 위해서다.
- Feature는 `featurize_window`를 행마다 호출해 만들었다(row_uid 정렬을 보장하려고). 5개 집합 모두에서 `build_dataset_from_csv`의 출력과 feature, label, group id가 완전히 같음을 확인했다(`out/data_summary.csv`).
- 이 보드에 Times New Roman이 없어 그림 글꼴은 기본 serif로 대체되었다.

---

## 1. Fold 구성

`GroupKFold(n_splits=3)`, group = family. 수동 재배치는 없다. 한 family가 두 fold에 걸치지 않음을 assert했다(`out/folds.csv`, `folds_summary.csv`).

| 방식 | Fold | 행 | Family | Schedule | N별 행 수 |
|---|---|---|---|---|---|
| **GroupKFold by family (이번)** | 0 | 1,192 | 11 | 33 | N1: 16, N2: 24, N3: 48, N4: 48, N5: 96, N6: 192, N8: 768 |
| | 1 | 1,191 | 6 | 18 | N1: 17, N3: 23, N7: 384, N8: 767 |
| | 2 | 1,191 | 12 | 36 | N1: 18, N3: 70, N4: 143, N6: 192, N7: 768 |
| KFold, shuffle 없음 (기존) | 0 | 1,192 | 3 | 7 | N6: 40, N7: 384, N8: 768 |
| | 1 | 1,191 | 25 | 75 | N1–N7 |
| | 2 | 1,191 | 3 | 7 | N7: 424, N8: 767 |

N 분포는 여전히 치우쳐 있다. 행의 75%(2,687 / 3,574)가 N≥7인 5개 family에 몰려 있어서, 각 fold의 validation 행은 대부분 N=7 또는 N=8 family 한두 개다. 지시대로 바꾸지 않았다.

**해석 범위**: Family-grouped CV는 hyperparameter 선택 과정의 hygiene을 개선한다. 최종 x3 평가는 train = x1/x2/x4, test = x3이므로 같은 family의 다른 rate가 학습에 포함된 **held-out-rate 평가**이며, unseen-family 평가가 아니다.

---

## 2. Hyperparameter

### 2.1 Parameter

기존 학습 경로가 설정하는 것은 `objective="reg:squarederror"`, `random_state=42`, `n_jobs=-1`과 grid 값뿐이다(`deploy_selector_xgb_suite.py:754-758`, `:593-597`). 나머지는 library 기본값이며, 추정하지 않고 학습된 booster의 `save_config()`에서 읽어 `artifacts/*/meta.json`의 `resolved_config`에 저장했다.

| Parameter | 값 | 출처 |
|---|---|---|
| objective | reg:squarederror | 코드 |
| random_state / seed | 42 | 코드 |
| n_jobs | −1 (24 threads) | 코드 |
| booster | gbtree | 기본값 (config) |
| tree_method (updater) | hist (`grow_quantile_histmaker`) | 기본값 (config) |
| colsample_bytree | 1 (Multi), 0.8 (Single, 코드의 grid 고정값) | 기본값 / 코드 |
| colsample_bylevel, colsample_bynode | 1 | 기본값 |
| min_child_weight | 1 | 기본값 |
| gamma (min_split_loss) | 0 | 기본값 |
| reg_alpha / reg_lambda | 0 / 1 | 기본값 |
| max_bin | 256 | 기본값 |
| max_delta_step, max_leaves | 0 | 기본값 |
| grow_policy, sampling_method | depthwise, uniform | 기본값 |

Grid(16 조합): `n_estimators {500, 1000}`, `learning_rate {0.01, 0.05}`, `max_depth {4, 6}`, `subsample {0.8, 1.0}`. 선택 기준은 3 fold의 unweighted validation MAE 평균이고, 동점이면 grid 순서상 첫 조합이다.

### 2.2 선택값 (`out/hparam_search_multi.csv`, `hparam_search_single.csv`)

| 모델 | n_estimators | learning_rate | max_depth | subsample | CV MAE (fold 0 / 1 / 2) | 2위와의 차이 | 기존 선택값 |
|---|---|---|---|---|---|---|---|
| T | 500 | 0.01 | 6 | 0.8 | 0.1005 (0.090 / 0.104 / 0.107) | 0.0008 | **같음** (500 / 0.01 / 6 / 0.8) |
| D (weight 10) | 1,000 | 0.01 | 6 | 0.8 | 0.1089 (0.123 / 0.100 / 0.103) | 0.0010 | **같음** (1000 / 0.01 / 6 / 0.8) |
| D_noweight | 500 | 0.01 | 4 | 0.8 | 0.1052 (0.118 / 0.094 / 0.103) | 0.0001 | 해당 없음 |

- 기존 KFold에서의 CV MAE는 T 0.1066, D 0.1132였다(`commands.docx`). Fold가 달라 직접 비교할 값은 아니다.
- D 모델의 D > 0.01 subset CV MAE는 weighted 0.1092, noweight 0.1089다.
- 16개 조합의 CV MAE 범위는 T 0.1005–0.1102, D 0.1089–0.1176으로 좁다.

Single-Objective(α별, `colsample_bytree=0.8` 고정, sample weight 없음 — 기존 `train_score` 경로와 같음):

| α | n_estimators | learning_rate | max_depth | subsample | CV MAE | 기존 artifact 선택값 |
|---|---|---|---|---|---|---|
| 0.1 | 500 | 0.01 | 6 | 0.8 | 0.1005 | 500 / 0.01 / 6 / 0.8 |
| 0.2 | 500 | 0.01 | 6 | 0.8 | 0.1042 | 1000 / 0.01 / 6 / 0.8 |
| 0.3 | 500 | 0.01 | 6 | 0.8 | 0.1101 | 1000 / 0.01 / 6 / 0.8 |
| 0.4 | 500 | 0.01 | 6 | 0.8 | 0.1170 | 1000 / 0.01 / 6 / 0.8 |
| 0.5 | 500 | 0.01 | 4 | 0.8 | 0.1244 | 1000 / 0.01 / 6 / 0.8 |
| 0.6 | 500 | 0.01 | 6 | 0.8 | 0.1317 | 1000 / 0.01 / 6 / 0.8 |
| 0.7 | 500 | 0.01 | 4 | 0.8 | 0.1387 | 1000 / 0.01 / 4 / 0.8 |
| 0.8 | 500 | 0.01 | 4 | 0.8 | 0.1459 | 1000 / 0.01 / 4 / 0.8 |
| 0.9 | 500 | 0.01 | 4 | 0.8 | 0.1530 | 1000 / 0.01 / 4 / 0.8 |
| 1.0 | 500 | 0.01 | 4 | 0.8 | 0.1616 | 1000 / 0.01 / 4 / 0.8 |

Single 쪽은 α ≥ 0.2에서 tree 수가 1,000에서 500으로 바뀌었다.

### 2.3 Feature

- 235개 column(`out/feature_columns.txt`). 기존 artifact는 169개다. 추가된 66개는 현재 `featurize_window`가 내는 `…_on_npu0`, `…_on_npu1` column이며 CPU–GPU 데이터에서는 전부 0이다. 기존 169개는 모두 포함된다.
- Leakage assert: 모든 column이 `views.*`, `pairhash_*` 패턴에 맞고 label·실측·평가 관련 이름이 없다.

---

## 3. 검증 A → B → C

3×, α = 0.3. Evaluator 단위 테스트(`s_hat = S`이면 18/18이고 Avg = Oracle, 동점은 combination 번호 순, `select()`는 실측 column을 받으면 실패)도 통과했다.

| 단계 | 모델 | 데이터 | 규칙 | N≥3: Avg | Top-1 | Top-5 | Regret | N≥1: Avg | Top-1 | Top-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** legacy replay | 기존 artifact | 중복 유지 (1,217행) | 기존 | 0.9451 | 13/18 | 17/18 | 0.0416 | 0.9608 | 21/29 | 28/29 |
| **B1** dedup만 | 기존 artifact | 중복 제거 (1,193행) | 기존 | 0.9453 | 13/18 | 17/18 | 0.0415 | 0.9608 | 21/29 | 28/29 |
| **B2** dedup + 새 규칙 | 기존 artifact | 중복 제거 | 새 규칙 | 0.9453 | 13/18 | 17/18 | 0.0415 | 0.9608 | 21/29 | 28/29 |
| **C** 재학습 | 새 artifact | 중복 제거 | 새 규칙 | 0.9505 | 13/18 | 17/18 | 0.0362 | 0.9623 | 20/29 | 28/29 |

Oracle은 모든 단계에서 0.9867(N≥3), 0.9918(N≥1)이다.

- **A**: REPORT2의 값(Oracle 0.987, Ours 0.945, 13/18, 17/18)을 재현한다. MAE(전체 1,217행)도 0.0621 / 0.0906으로 같다.
- **Dedup 효과**: Avg +0.0002. 중복 24행은 6개 schedule에 있고, 모두 2026-03-11의 재실행이 남고 03-06·03-07의 run이 빠졌다(`out/dedup_dropped.csv`). 재실행 run이 모든 후보를 포함해서 dedup 뒤에 두 run이 섞인 group은 없다. Timestamp로 순서를 정하지 못한 쌍은 없었다.
- **규칙 효과**: 없음. 재학습 모델의 3× 예측에서 N≥3 workload의 1위 동점이 0건이고, 기존 Top-5 규칙으로 계산한 값도 17/18로 같다.
- **재학습 효과**: Avg +0.0052, Regret −0.0053. Workload 하나(`m_…_v_y`, Fig. 1(b)의 workload)에서 선택이 combination_128(전부 GPU, S 0.510)에서 combination_127(S 0.605)로 바뀐 결과다. Hyperparameter는 같으므로 이는 학습 환경(기존 artifact는 다른 머신에서 학습)과 feature column 수의 차이에서 온 것으로 보인다[추정].
- 중복을 유지한 채 새 모델·새 규칙으로 계산하면 Ours 0.9504, 13/18, 17/18이다(`out/reference_duplicates_kept.csv`).

---

## 4. Table 3(a)

3× hold-out, N≥3, α = 0.3(`out/table3a.csv`, `.md`).

| 전략 | Workloads | Avg score | Top-1 | Top-5 | Regret | MAE(T) | MAE(D) | 원고 Table 3(a) |
|---|---|---|---|---|---|---|---|---|
| Latency-based | 18 | 0.676 | 3/18 (0.17) | N/A | 0.310 | | | 0.63 / hit 0.17 |
| **Ours (재학습)** | 18 | **0.951** | **13/18 (0.72)** | **17/18 (0.94)** | 0.036 | 0.060 | 0.090 | 0.95 / 0.72 / 0.94, MAE 0.06 / 0.09, gap 0.03 |
| Ours (기존 artifact + 새 evaluator) | 18 | 0.945 | 13/18 (0.72) | 17/18 (0.94) | 0.042 | 0.060 | 0.090 | |
| Best-of-5 (1,000회) | 18 | 0.730 ± 0.039 | 0.307 (평균 비율) | N/A | 0.257 | | | 0.78 ± 0.040 / hit 0.30 |
| **All-GPU** | 17 | **0.943** | 13/17 (0.76) | N/A | 0.043 | | | (없음) |
| Oracle | 18 | 0.987 | 18/18 | 18/18 | 0 | | | 0.99 |

참고 행: All-CPU −0.195 (0/18). Latency-based 저장 규칙 재현 0.618 (3/18). Ours에서 D weight를 뺀 경우 0.945 (5절).

**달라진 행과 이유**

- **Latency-based 0.63 → 0.676**: 원고 값은 α = 0.2였고, shufflenet·squeezenet·vgg19가 CPU로 선택된 결과였다. 이번에는 같은 rate의 단독 실행(N=1) 평균 latency가 작은 device를 고른다(동점은 GPU). 3×에서는 CNN 6종과 yolov4가 GPU, gpt2와 tiny-llama가 CPU다. 저장 규칙을 그대로 재현하면 0.618(α = 0.3)이 나온다.
- **Best-of-5 0.78 → 0.730**: 원고의 0.78을 만든 계산은 저장소에 없다. Seed를 `SeedSequence([42, trial, workload])`로 고정한 1,000회의 평균이며 표준편차 0.039는 원고의 0.040과 맞는다. 표본 5개에 oracle이 들어 있을 비율 0.307은 원고의 hit 0.30에 해당한다. Best-of-5는 표본 중 실측 최고를 고르므로 Top-1 비율과 이 값은 정의상 같다.
- **Ours**: 반올림하면 원고와 같다. Regret은 N≥3에서 0.036이다. 원고의 gap 0.03은 N≥1 기준이었고 이번 N≥1 값은 0.030이다.
- **Oracle**: 변화 없음.

**All-GPU 대비 Ours** (`out/per_workload_x3.csv`)

- All-GPU 배치가 측정되지 않은 workload는 `g_m_t` 하나라서 분모가 17이다.
- 같은 17개에서 Ours 0.950, All-GPU 0.943이다. 차이는 두 workload에서 나온다.

| Workload | N | Ours 선택 | Ours S | All-GPU S | Oracle S |
|---|---|---|---|---|---|
| `r_t_v_y` | 4 | combination_13 | 0.824 | 0.796 | 0.937 |
| `m_…_v_y` | 7 | combination_127 | 0.605 | 0.510 | 0.896 |

- 나머지 15개에서 Ours의 선택은 전부-GPU 배치와 같다. Oracle이 전부-GPU인 workload는 18개 중 13개다.
- 기존 artifact는 16개에서 전부-GPU를 골랐고 All-GPU와의 차이는 `r_t_v_y` 하나였다.

**Latency source** (`out/latency_profile_sources.csv`): 72개 (rate, model, device) 조합 모두 단독 실행 결과에서 얻었고 offline profile로 대체한 것은 없다. 3×의 shufflenet·squeezenet·vgg19 CPU는 run이 두 개라서 늦은 run(03-11)을 썼다.

**N≥1 전체** (`out/table_all_n.csv`, 29 workloads): Ours 0.962 (20/29, 28/29), 기존 artifact 0.961 (21/29, 28/29), All-GPU 0.960 (21/28), Latency-based 0.797 (12/29), Best-of-5 0.832 ± 0.024, Oracle 0.992. 후보 5개 이하 workload는 N≥3에 0개, N<3에 11개다.

---

## 5. D weight ablation

`out/ablation_d_weight.csv`, 3×, N≥3.

| 구성 | D 모델 선택값 | MAE(D) 전체 | MAE(D), D > 0.01 | Avg score | Top-1 | Top-5 | Regret |
|---|---|---|---|---|---|---|---|
| T + D (weight 10, primary) | 1000 / 0.01 / 6 / 0.8 | 0.0905 | 0.0940 | 0.9505 | 13/18 | 17/18 | 0.0362 |
| T + D_noweight | 500 / 0.01 / 4 / 0.8 | 0.0900 | 0.0937 | 0.9453 | 13/18 | 17/18 | 0.0415 |

- Weight의 영향은 작다. MAE(D)는 0.0005 차이이고 Top-1, Top-5는 같다. Avg score 차이 0.005는 3절과 같은 workload 하나(`m_…_v_y`)에서 나온다.
- 학습 행의 92%(3,283 / 3,574)가 D > 0.01이라 weight가 거의 모든 행에 걸린다. 사실상 D 손실 전체에 상수를 곱한 것에 가깝고, 그래서 효과가 작다[추정].
- 원고에는 "D 회귀기는 D > 0.01인 표본에 가중치 10을 준다"고 적고, 빼도 Top-k가 같다는 점을 함께 적을 수 있다.

---

## 6. Fig. 3, Fig. 4

### 6.1 Fig. 3 (`out/figs/fig3_model_count_vs_score.pdf`, `.png`, `out/fig3_data.csv`)

α = 0.3, "≥N" 누적 평균. 괄호 안은 기존 그림의 값이다.

| N≥ | Workloads | Oracle | Ours (재학습) | All-GPU | Best-of-5 (1,000회) | Latency-based |
|---|---|---|---|---|---|---|
| 3 | 18 | 0.987 (0.99) | 0.951 (0.95) | 0.943 (17개) | 0.730 ± 0.039 (0.77) | 0.676 (0.63) |
| 4 | 12 | 0.980 (0.98) | 0.929 (0.92) | 0.919 | 0.626 ± 0.054 (0.66) | 0.606 (0.52) |
| 5 | 8 | 0.978 (0.98) | 0.915 (0.90) | 0.904 | 0.537 ± 0.066 (0.56) | 0.580 (0.47) |
| 6 | 7 | 0.975 (0.97) | 0.903 (0.89) | 0.890 | 0.517 ± 0.067 (0.50) | 0.604 (0.47) |
| 7 | 5 | 0.979 (0.98) | 0.879 (0.86) | 0.860 | 0.458 ± 0.080 (0.42) | 0.573 (0.45) |
| 8 | 2 | 1.000 | 0.900 | 0.900 | 0.421 ± 0.121 | 0.561 |

- Oracle과 Ours는 기존 그림과 0.02 이내다.
- Latency-based는 전 구간에서 0.05–0.13 높아졌고 N에 따른 하락이 완만해졌다. N≥5부터는 Best-of-5보다 높다. 기존 그림에서는 N≥7에서만 그랬다.
- All-GPU 곡선은 Ours 바로 아래에 붙어 있다(차이 0.008–0.019).
- N≥8은 workload가 2개라 그림에는 넣지 않고 CSV에만 저장했다.

### 6.2 Fig. 4 (`out/figs/fig4_alpha_comparison.pdf`, `.png`, `out/fig4_data.csv`)

Multi는 재학습한 T, D 모델에 α만 바꾼 것이고, Single은 α별로 재학습한 모델이다. 18 workloads. 괄호 안은 기존 그림의 값이다.

| α | Oracle | Multi: Avg | Top-1 | Top-5 | Single: Avg | Top-1 | Top-5 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.996 | 0.956 (0.96) | 12/18 | 16/18 | 0.956 (0.96) | 12/18 | 16/18 |
| 0.2 | 0.991 | 0.951 (0.95) | 12/18 | 16/18 | 0.951 (0.95) | 12/18 | 16/18 |
| 0.3 | 0.987 | 0.951 (0.95) | 13/18 | 17/18 | 0.945 (0.95) | 13/18 | 17/18 |
| 0.4 | 0.984 | 0.948 (0.95) | 13/18 | 17/18 | 0.940 (0.94) | 13/18 | 17/18 |
| 0.5 | 0.981 | 0.945 (0.94) | 13/18 | 17/18 | 0.933 (0.94) | 13/18 | 15/18 (기존 17) |
| 0.6 | 0.978 | 0.942 (0.94) | 13/18 | 17/18 | 0.942 (0.94) | 13/18 | 16/18 (기존 17) |
| 0.7 | 0.976 | 0.939 (0.94) | 13/18 | 17/18 | 0.938 (0.94) | 13/18 | 15/18 |
| 0.8 | 0.973 | 0.937 (0.94) | 13/18 | 17/18 | 0.936 (0.94) | 13/18 | 15/18 |
| 0.9 | 0.971 | 0.934 (0.93) | 13/18 | 17/18 | 0.933 (0.93) | 13/18 | 15/18 |
| 1.0 | 0.968 | 0.931 (0.93) | 13/18 | 17/18 | 0.930 (0.93) | 13/18 | 15/18 |

- Top-1은 모든 α에서 두 예측기가 같다. 기존 그림과도 같다.
- Avg score는 α ≥ 0.3에서 Multi가 0.000–0.012 높다.
- Top-5는 Single이 α = 0.5에서 15/18, 0.6에서 16/18로 기존(17/18)보다 낮아졌다. α ≥ 0.7은 기존과 같다.
- 기존 Top-5 규칙(서로 다른 예측 값 상위 5개)으로 계산한 값은 20개 cell 모두 새 규칙과 같다(CSV의 `top5_legacy_rule_hits_*`).
- 두 예측기 모두 18개 중 15–17개에서 전부-GPU 배치를 고른다(CSV의 `selected_all_gpu_*`).

---

## 7. 중간 rate

재학습 artifact, α = 0.3, N≥3(`out/table_intermediate.csv`).

| Rate | 전략 | Workloads | Avg score | Top-1 | Top-5 | Regret | MAE(T) / MAE(D) |
|---|---|---|---|---|---|---|---|
| 1.5× | Ours (재학습) | 18 | 0.950 | 13/18 (0.72) | 17/18 (0.94) | 0.035 | 0.053 / 0.093 |
| | All-GPU | 15 | 0.950 | 13/15 | N/A | 0.032 | |
| | Latency-based | 16 | 0.813 | 5/16 | N/A | 0.171 | |
| | Best-of-5 | 18 | 0.734 ± 0.036 | 0.318 | N/A | 0.252 | |
| | Oracle | 18 | 0.985 | | | | |
| 2.5× | Ours (재학습) | 18 | 0.969 | 16/18 (0.89) | 18/18 (1.00) | 0.017 | 0.055 / 0.083 |
| | All-GPU | 16 | 0.947 | 13/16 | N/A | 0.038 | |
| | Latency-based | 18 | 0.688 | 3/18 | N/A | 0.298 | |
| | Best-of-5 | 18 | 0.728 ± 0.040 | 0.316 | N/A | 0.258 | |
| | Oracle | 18 | 0.986 | | | | |
| 3.5× | Ours (재학습) | 17 | 0.964 | 15/17 (0.88) | 16/17 (0.94) | 0.023 | 0.051 / 0.079 |
| | All-GPU | 16 | 0.947 | 14/16 | N/A | 0.039 | |
| | Latency-based | 17 | 0.847 | 4/17 | N/A | 0.140 | |
| | Best-of-5 | 17 | 0.746 ± 0.037 | 0.320 | N/A | 0.240 | |
| | Oracle | 17 | 0.986 | | | | |

- Ours의 Top-1, Top-5는 원고(0.72 / 0.88 / 0.88, 0.94 / 1.0 / 0.94)와 같다. 2.5×의 Top-1은 16/18 = 0.889다.
- All-GPU와 Latency-based는 해당 배치가 측정되지 않은 workload를 분모에서 뺐다. 목록은 CSV의 `not_evaluated` 열에 있다. 분모가 달라 Ours와 직접 비교할 때는 주의가 필요하다.
- **Latency-based는 rate에 따라 0.69–0.85로 흔들린다.** LM 2종의 "빠른 device"가 rate마다 바뀌기 때문이다(1.5×: gpt2 GPU, tiny-llama CPU / 3×와 2.5×: 둘 다 CPU / 3.5×: gpt2 CPU, tiny-llama GPU). LM instance는 두 device에서 같은 network를 약 3건/s로 실행했으므로(REPORT2) 이 선택은 측정 잡음이다.

---

## 8. 선행 보고서 정정

이번 작업 중 확인된 것으로, 선행 보고서의 두 문장을 바로잡는다. 해당 파일은 수정하지 않았다.

- REPORT2 3.3절의 "S_old에서 oracle이 전부-GPU인 workload 12/18"은 **13/18**이 맞다. 표에서 `m_r_t_y`의 GPU 목록이 잘려 3개로 보였으나 4개 모두 GPU다.
- REPORT2 3.2절과 지시문 0절 표 5번의 "3× 재실행 파일에 CPU 행만 있다"는 방향이 반대다. CPU 행 하나만 있는 것은 **원래 run(03-07)**이고, 재실행(03-11)은 CPU와 GPU가 모두 있다. 저장 규칙이 CPU를 고른 원인(행이 하나인 파일에서 score가 1.0으로 정규화되어 GPU와 동점)은 그대로다.

---

## 산출물

```
revision_exp/
  data.py  folds.py  train_multi.py  train_single.py  evaluate.py  baselines.py  predict.py  plots.py  run_all.py
  artifacts/multi/{T,D,D_noweight}.json, meta.json
  artifacts/single/alpha_{0.1..1.0}.json, meta.json
  out/
    input_manifest.csv  data_summary.csv  dedup_dropped.csv  feature_columns.txt  folds.csv  folds_summary.csv
    hparam_search_multi.csv  hparam_search_single.csv
    validation_A.csv  validation_B.csv  validation_C.csv  reference_duplicates_kept.csv
    latency_profile_sources.csv  ablation_d_weight.csv  mae_x3.csv
    predictions_x3.csv  predictions_x{1-5,2-5,3-5}.csv  per_workload_x3.csv
    table3a.csv  table3a.md  table_all_n.csv  table_all_n.md  table_intermediate.csv  table_intermediate.md
    fig3_data.csv  fig4_data.csv  figs/{fig3_model_count_vs_score,fig4_alpha_comparison}.{pdf,png}
    logs/
  REPORT_RETRAIN.md
```

재현: 저장소 루트에서 `.venv/bin/python3 -B revision_exp/run_all.py` (약 4분). `--reuse`를 주면 저장된 artifact로 평가와 그림만 다시 만든다.
