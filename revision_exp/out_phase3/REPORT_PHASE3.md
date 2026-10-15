# Phase 3 보고서 — 리뷰어 요청 실험 (재측정 CPU–GPU 데이터)

작성 2026-10-14. 지시문 v1.2. 표시: [확인] 측정·계산으로 확인, [추정] 근거는 있으나 직접 확인하지 않음.
모든 평가는 3× hold-out, N≥3 12 workloads(후보 480), α = 0.3, D = D_macro.

**2026-10-14 갱신**: 선택 점수를 모든 방법에서 clip 없는 Ŝ = T̂ − 0.3·D̂로 통일했다(원고 §3 수식). 이 때문에 MLP의 Top-1(5/12 → 6/12)과 XGBoost −P의 Avg·regret(0.9920 → 0.9918, 0.0054 → 0.0056)만 바뀌었다. 근거와 변경 전/후는 `CLIP_CHECK.md`.

## 요약

| 방법 | MAE(T) | MAE(D) | Avg score | Regret | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| XGBoost Full (Ours) | 0.0273 | 0.0300 | 0.9930 | 0.0044 | 7/12 | 11/12 |
| Ridge | 0.0300 | 0.0214 | 0.9941 | 0.0033 | 7/12 | 11/12 |
| Random Forest | 0.0307 | 0.0410 | 0.9948 | 0.0026 | 8/12 | 11/12 |
| MLP (small) | 0.0597 | 0.0474 | 0.9729 | 0.0245 | 6/12 | 9/12 |
| XGBoost −P | 0.0262 | 0.0447 | 0.9918 | 0.0056 | 6/12 | 10/12 |
| XGBoost −I | 0.0310 | 0.0347 | 0.9960 | 0.0014 | 8/12 | 12/12 |
| Per-instance predictor ([14]-inspired) | 0.0276 | 0.0236 | 0.9868 | 0.0106 | 6/12 | 11/12 |
| Latency-based (= All-GPU) | N/A | N/A | 0.9727 | 0.0247 | 6/12 | N/A |
| Best-of-5 | N/A | N/A | 0.9404 | 0.0570 | 0.402 (평균 적중률) | N/A |
| Oracle | N/A | N/A | 0.9974 | 0 | 12/12 | N/A |

- **R2-1**: XGBoost, Ridge, Random Forest는 Avg score 0.993–0.995, regret 0.003–0.004로 거의 같다. MLP만 뒤진다(0.9729, All-GPU와 같은 수준). [확인]
- **R2-2**: −P는 Full과 거의 같고(−0.0012), −I는 오히려 조금 높다(+0.0030). 차이는 1–3개 workload의 선택에서 나오고, oracle–All-GPU 차(0.025)의 1/8 이하다. Feature 그룹 하나를 빼서 성능이 떨어진다는 근거는 이 데이터에 없다. [확인]
- **R2-3**: per-instance predictor는 All-GPU보다 낫고(+0.014) XGBoost Full보다 뒤진다(−0.0062, SE 0.0028, 2 승 / 5 동률 / 5 패). [확인]

## 0. 설정 고정 확인

- 실행 전 점검 P1–P5 모두 통과(`precheck.json`).
  - **P1**: Phase 2 artifact로 추가 학습 없이 평가해 feature 204, 후보 480, workload 12, Avg 0.9930, regret 0.0044, Top-1 7/12, Top-5 11/12를 재현했다. Workload별 선택도 Phase 2와 같고, artifact hash가 Phase 2 `config_frozen.json`과 같다.
  - **P2**: `out_reprofiled/folds.csv`에 22 family 모두의 fold가 있다. `folds.make_folds`로 다시 만든 mapping과 같고, 그 fold로 Phase 2 선택 설정(T, D)을 다시 학습한 fold별 MAE가 저장값과 정확히 같다(차 0). `GroupKFold(shuffle=False)`라 seed는 fold에 영향이 없다.
  - **P3**: scikit-learn 1.7.2. Ridge·RF·MLP 모두 `sample_weight`를 받고(Pipeline에서는 `est__sample_weight`), weight가 결과를 바꾸며, 모든 weight가 1이면 weight 없는 결과와 같다. Feature matrix는 dense float64(0인 값 71%)라 `StandardScaler`를 기본값으로 썼다.
  - **P4**: −P 56열, −I 108열 제거(§2).
  - **P5**: q_i는 반복마다 d/g를 구한 뒤 평균하고, D_macro는 그 instance 평균이다(`counters.py`, `recompute_scores.py`). 학습 데이터 8,880 instance에서 label과 차가 1e-16 이하다. 3× 재구성은 §3.4.
- 학습 설정(grid, fold, weight, mask, 선택된 hyperparameter)은 3× label을 읽기 전에 `logs/config_frozen.json`에 기록했다. 3× label은 `final_eval()`에서만 읽는다(코드 assert). Per-instance 배치 선택은 그보다 먼저, schedule과 offline profile만으로 끝냈다.
- 결과를 본 뒤 grid·feature·규칙·지표를 바꾸지 않았다. 코드는 첫 실행 그대로다.
- 실행 시간: 학습 합계 약 87초(Ridge 0.1, RF 26, MLP 3, XGBoost −P 18, −I 15, per-instance 24초), 전체 93초.
- Artifact: XGBoost Full = `revision_exp/out_reprofiled/artifacts/multi/{T,D}.json`(재학습 없음). 새 artifact는 `out_phase3/artifacts/`.
- `revision_exp/` 기존 파일, Phase 2 산출물, tracked 파일 변경 없음. 새 `__pycache__` 없음.

## 1. R2-1 — 회귀기 비교 (`table_predictors.md`)

| 방법 | MAE(T) | MAE(D) | Avg | Regret | Top-1 | Top-5 | − All-GPU (참고) | − XGBoost Full |
|---|---|---|---|---|---|---|---|---|
| XGBoost Full | 0.0273 | 0.0300 | 0.9930 | 0.0044 | 7/12 | 11/12 | +0.0203 | – |
| Ridge | 0.0300 | 0.0214 | 0.9941 | 0.0033 | 7/12 | 11/12 | +0.0214 | +0.0011 (3/7/2) |
| Random Forest | 0.0307 | 0.0410 | 0.9948 | 0.0026 | 8/12 | 11/12 | +0.0221 | +0.0018 (2/10/0) |
| MLP (small) | 0.0597 | 0.0474 | 0.9729 | 0.0245 | 6/12 | 9/12 | +0.0002 | −0.0201 (2/5/5) |

괄호는 workload별 차의 부호(+ / 0 / −).

- 선택된 설정: Ridge α = 100(T, D 모두 grid의 끝값), RF `max_depth=None, min_samples_leaf=1`, n_estimators T 300 / D 600, MLP T `(64,)` / D `(64, 32)`, 둘 다 alpha 0.01. 모두 D에 weight 10.
- **내부 CV MAE와 3× MAE가 크게 다르다.** Family 단위 CV MAE는 Ridge 0.105 / 0.110, MLP 0.488 / 0.456인데, 3× MAE는 0.03 / 0.02, 0.06 / 0.05다. XGBoost(0.029 / 0.032)와 RF(0.032 / 0.040)는 CV와 3×가 비슷하다. CV는 학습에 없던 family로 일반화하는 과제이고, 3× 평가는 학습에 있던 family의 새 rate다. 표준화한 feature에서 학습 fold에 거의 없던 열(특정 모델 hash, pairhash)이 검증 fold에서 큰 값이 되어 선형·신경망 모델이 외삽한 것으로 보인다. [추정] 이 때문에 Ridge는 가장 강한 정규화(α = 100)가 골라졌다.
- MLP는 모든 fit이 수렴 경고 없이 max_iter 2,000 안에서 끝났다. 학습은 fit 1회에 0.2초 정도로 짧았다.
- 해석: XGBoost, Ridge, RF의 차이(Avg 최대 0.0018, regret 최대 0.0018)는 S의 반복 SE 중앙값(0.001)과 같은 크기이고, oracle–All-GPU 차(0.025)보다 한 자릿수 작다. 이 데이터에서 Ours의 이득은 특정 회귀기에서 나오는 것이 아니다. MLP는 GPU가 포화되는 두 N=7 workload에서 크게 틀려(regret 0.106, 0.045) All-GPU 수준이 됐다.

## 2. R2-2 — Feature ablation (`table_ablation.md`)

| 구성 | Feature 수 | 제거 열(학습에서 0이 아닌 열) | MAE(T) | MAE(D) | Avg | Regret | Top-1 | Top-5 | − XGBoost Full |
|---|---|---|---|---|---|---|---|---|---|
| Full | 204 | 0 | 0.0273 | 0.0300 | 0.9930 | 0.0044 | 7/12 | 11/12 | – |
| −P | 148 | 56 (56) | 0.0262 | 0.0447 | 0.9918 | 0.0056 | 6/12 | 10/12 | −0.0012 (0/11/1) |
| −I | 96 | 108 (54) | 0.0310 | 0.0347 | 0.9960 | 0.0014 | 8/12 | 12/12 | +0.0030 (3/8/1) |

- **Mask**: 지시문 초안 `^pairhash_`는 실제 열 이름에 그대로 맞는다(prefix 없음). 확정 패턴은 −P `^pairhash_\d+_on_`, −I `^views\.(sum|mean|max)\.model_hash_\d+_on_`과 `^views\.(sum|mean|max)\.view\.infps_on_`이다. 각 mask는 해당 그룹 전체만 지우고 다른 그룹은 건드리지 않는다(`ablation_columns.txt`).
- **기존 구성과 대조**(기존 169열 기준): pairwise 72 → 56(TinyLLaMA 관련 쌍이 빠짐). Identity×device 54 + demand×device 6 = 60 → 108열인데 그중 60열은 NPU 열이라 학습 데이터에서 모두 0이다. 실효 제거는 48 + 6 = 54열(모델 8개 × CPU/GPU × sum/mean/max, demand × CPU/GPU × 3)이다.
- **Collision** (`ablation_collisions.csv`): Full과 −P는 어떤 workload에서도 feature가 같은 후보가 없다. −I는 N=2의 두 family(gpt2+mnasnet, gpt2+yolov4)에서 "GPT-2 GPU + 다른 모델 CPU"와 그 반대 배치가 구분되지 않는다. 학습 6 workloads, 3× 2 workloads이고 묶음 안의 실측 S 차가 최대 0.66이다. N=2라 평가 12개에는 들어가지 않는다. N≥3에서는 남은 pairwise 열이 배치를 구분한다.
- 해석: 두 그룹 중 하나만 빼도 다른 그룹이 배치 정보를 거의 다 갖고 있어서(pairwise는 device별 쌍, identity×device는 모델별 device), 점수 변화는 −0.0012, +0.0030으로 잡음과 비슷한 크기다. 이 결과로 "각 그룹이 필요하다"고 쓸 수는 없고, "두 그룹은 서로 대체 가능하며 하나만 있어도 같은 수준"이 데이터가 뒷받침하는 서술이다. 다만 N=2에서는 −I가 배치를 구분하지 못한다는 구조적 사실이 있다. 둘 다 뺀 구성은 지시대로 하지 않았다.

## 3. R2-3 / R3-2 — Per-instance baseline (`table_baseline.md`)

Kim et al. [14]의 instance별 간섭 예측 아이디어만 참고한 변형이며 원 시스템의 재현이 아니다(SPEC_v2.2 §7 (b)).

| 방법 | MAE(T) | MAE(D) | Avg | Regret | Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| Latency-based (= All-GPU) | N/A | N/A | 0.9727 | 0.0247 | 6/12 | N/A |
| Per-instance predictor ([14]-inspired) | 0.0276 | 0.0236 | 0.9868 | 0.0106 | 6/12 | 11/12 |
| Ours (XGBoost Full) | 0.0273 | 0.0300 | 0.9930 | 0.0044 | 7/12 | 11/12 |
| Best-of-5 | N/A | N/A | 0.9404 | 0.0570 | 0.402 | N/A |
| Oracle | N/A | N/A | 0.9974 | 0 | 12/12 | N/A |

구성:

- 학습 행 8,880개(1·2·4×). Feature는 모델 identity one-hot(128 bucket), device, λ_i, isolated capacity(offline profile), 같은/다른 device co-runner 수와 λ 합, N이다.
- Target은 t_ratio = completed/10초/λ_i와 q_i다. XGBoost 2개를 쓰고, grid·선택 규칙은 direct와 같다. Fold는 Phase 2의 family→fold mapping을 그대로 join했다.
- **가중치 단위가 다르다.** Direct D 모델은 배치 행에서 D_macro > 0.01이면 weight 10이고, per-instance q_i 모델은 instance 행에서 q_i > 0.01이면 weight 10이다(3,291 / 8,880행). 같은 숫자지만 다른 단위의 손실 가중이라 "같은 손실 함수"가 아니다.
- **t_ratio 범위**: 학습 0.043–1.014이고 1을 넘는 행이 2,908개(33%)다. 측정 구간 시작 시 queue에 있던 요청이 구간 안에서 완료되면 completed가 λ·Δ를 조금 넘는다. 예측값은 `clip(·, 0)`(상한 없음)으로 두며, 이는 의도한 규칙이다. 3×에서는 최대 1.006, 1을 넘는 행 395개다.

진단:

| 단계 | 값 |
|---|---|
| (1) 실측 instance 값의 합산 재구성 | 504 후보 모두 정확(F 차 2e-13, D_macro 차 2e-16). 반복별 비율을 먼저 구하는 규칙으로 재구성된다. `mean(d)/mean(g)`로 바꾸면 411개 후보가 최대 1e-5 다르다 |
| (2) instance 예측 MAE (N≥3, 2,928행) | t_ratio 0.039, q_i 0.052 (q_i > 0.01 행 0.093). CPU instance t_ratio 0.066 / q_i 0.090, GPU instance 0.011 / 0.013 |
| (3) 합산 F̂ 오차, 정규화 영향 | F̂ 상대 오차 중앙값 −0.2%, 5–95% 분위 −5.0% ~ +8.0%, 절대 평균 3.6%. T̂_select MAE 0.0276, 실측 F_max로 정규화한 진단값 0.0281. **실측 정규화로 바꿔도 선택이 바뀌는 workload 0개** |
| (4) 최종 | regret 0.0106 (Ours 0.0044). Per-instance − Ours: −0.0062, SE 0.0028, 2 / 5 / 5 |

- Per-instance가 뒤진 5개 중 4개는 N≥6(gpt2…vgg19 N=7 −0.014, gpt2…squeezenet N=6 −0.023, mnasnet…vgg19 N=6 −0.021, mnasnet…yolov4 N=7 −0.018)이고, 나머지 1개(mnasnet+resnet50+resnext50)는 −0.00001이다. 이긴 2개(gpt2…shufflenet N=5, gpt2+mnasnet+resnet50)는 차이가 0.002 이하다. 4개 모두 Ours와 instance 하나의 device만 다른 배치를 골랐다(`paired_per_workload.csv`).
- 해석: 실측 instance 값은 system label을 정확히 재구성하므로, 분해 자체가 오차를 만드는 것은 아니다. Predicted-max 정규화도 선택을 바꾸지 않았다. 차이는 (2) CPU에 놓인 instance의 t_ratio·q_i 예측 오차에서 나온다. Co-runner를 개수와 λ 합으로만 요약하는 feature로는 CPU 쪽의 간섭(어느 모델과 함께 있는가)을 구분하기 어렵다. [추정] 그 결과 GPU가 포화되는 큰 workload에서 어떤 모델을 CPU로 보낼지를 direct 모델보다 덜 정확하게 고른다.

## 4. 통합 표와 각주 (`table_all.md`)

맨 위 요약 표가 통합 표다. 지표 적용 규칙:

- MAE는 후보 전체를 예측하는 방법에만 둔다. Per-instance는 T̂_select = F̂ / max F̂ 기준이다.
- Top-5는 후보 전체를 순위화하는 방법에만 둔다.
- Best-of-5의 Top-1은 1,000회 후보 추출의 평균 적중률이다.
- Oracle은 12/12다.

각주에 넣을 Top-1 관련 통계는 두 가지다.

- (i) oracle과 2위의 실측 S 차가 0.0002 이하인 workload: 6/12(그중 정확한 동점 2개, `top1_margin.csv`).
- (ii) 반복 3회의 oracle이 모두 같지는 않은 workload: 4/12(INTEGRITY §6).

권장 문장: "Top-1 accuracy is sensitive to small differences between the highest-ranked measured placements; we therefore report placement score and regret alongside Top-1 and Top-5." Top-1 실패가 전부 동률권 때문이라고 쓰지 않는다. 예를 들어 MLP와 per-instance의 실패에는 regret 0.01–0.1인 선택이 포함된다.

XGBoost의 D weight 없는 결과는 Phase 2 `ablation_d_weight.csv`에서 인용한다(Avg 0.9946, regret 0.0028, Top-5 12/12). 이 표에 행을 추가하지 않았다.

## 5. 핵심 질문

**(R2-1) 이득은 XGBoost에서 오는가, representation에서 오는가.** 같은 feature와 같은 fold에서 XGBoost, Ridge, RF의 차이는 0.002 이내이고, 셋 모두 All-GPU보다 +0.020–0.022 높다. Feature를 바꾼 −P·−I의 차이도 0.003 이내다. 회귀기 간 차이와 feature 그룹 간 차이가 모두 반복 잡음 수준이라 둘 중 어느 쪽이 이득을 만든다고 가를 수 없다. 데이터가 뒷받침하는 것은 "배치를 표현하는 placement-level feature 위에서 적당한 회귀기면 비슷한 선택에 도달한다"까지다. MLP만 예외로, 이 grid와 데이터 크기(1,512행)에서는 불안정하다.

**(R2-2) pairwise와 identity×device 각각의 기여.** 하나씩 빼면 −0.0012(−P), +0.0030(−I)으로 반복 잡음(0.001)의 1–3배이고, oracle–All-GPU 차(0.025)보다 훨씬 작다. 두 그룹이 같은 배치 정보를 중복해서 담고 있어 하나를 빼도 다른 하나가 대신한다. 구조적으로 −I는 N=2에서 일부 배치를 구분하지 못한다(평가 대상 밖).

**(R2-3) per-instance 분해는 어디서 부족한가.** 측정값 수준에서는 분해가 정확하고(재구성 오차 0), 예측 F̂의 정규화 방식도 선택에 영향이 없다. 부족한 곳은 instance 예측, 그중 CPU에 놓인 instance의 처리율·drop 예측이다(GPU 대비 오차 약 6–7배). 이 차이가 GPU가 포화되는 N≥6 workload에서 CPU로 보낼 모델의 선택 차이로 이어져 Ours보다 regret이 0.006 크다. 그래도 All-GPU보다는 0.014 높다.

## 6. 원고 반영용

- **새 비교표(6행)**: XGBoost Full / Ridge / RF / MLP / XGBoost −P / XGBoost −I. 값은 §1·§2 표.
- **Table 3(a) 추가 행**: Per-instance predictor ([14]-inspired) — Avg 0.9868, regret 0.0106, Top-1 6/12, Top-5 11/12, MAE 0.028 / 0.024.
- **§4.2 ablation 문장(초안)**: "Removing either the pairwise co-location features or the identity×device features changes the placement score by less than 0.004 (−0.001 and +0.003), because each group alone still encodes which model runs on which device; the two groups are largely redundant on this data."
- **§5.1.3 baseline 정의(초안)**: "Per-instance predictor (inspired by [14]). For each instance we predict its completed-request ratio t_i/λ_i and its drop fraction q_i from the model identity, its device, its request rate, its isolated capacity on that device, and a summary of its co-runners (count and total request rate on the same and on the other device, and N). Two XGBoost regressors are shared by all instances and tuned with the same grid and family-grouped folds as our predictor. A placement is scored by T̂ − 0.3·D̂_macro, where T̂ is the sum of the predicted instance throughputs normalised by its maximum over the candidate placements of the workload, and D̂_macro is the mean predicted drop fraction. This is an instance-level variant of the idea in [14], not a reproduction of that system."

## 7. 한계

- 평가 workload는 12개이고 N≥6은 4개다. 방법 간 차이 대부분이 이 4개에서 생긴다.
- Ours의 regret 0.0044는 "Ours가 oracle 쪽으로 더 나아갈 여지"의 평균이지 방법 간 차이의 상한이 아니다(MLP는 0.0245). 권장 문장: "Ours already achieves low regret (0.0044), leaving limited room for further improvement, although alternative predictors may still perform substantially worse."
- 새 MAE(0.027 / 0.030)와 기존 MAE(0.060 / 0.090), 새 Top-1 7/12와 기존 13/18은 모델 집합·실행 조건·drop 정의가 달라 같은 실험의 개선으로 해석하지 않는다.
- Ridge·MLP의 family 단위 CV MAE는 3× 성능을 잘 대변하지 못했다(§1). Hyperparameter 선택이 그 영향을 받았을 수 있으나 grid와 규칙은 바꾸지 않았다.
- Per-instance feature는 사전에 고정한 사양대로 co-runner 종류를 표현하지 않는다.

## 파일 (`revision_exp/out_phase3/`)

| 내용 | 파일 |
|---|---|
| 실행 전 점검 | `precheck.json`, `feature_columns_full.txt`, `ablation_columns.txt` |
| 표 | `table_predictors`, `table_ablation`, `table_baseline`, `table_all` (`.csv`, `.md`) |
| Hyperparameter | `hparam_search_predictors.csv`, `hparam_search_ablation.csv`, `hparam_search_instance.csv`, `logs/config_frozen.json` |
| Workload별 | `per_workload_all.csv`, `paired_per_workload.csv`, `paired_summary.csv`, `top1_margin.csv` |
| Collision | `ablation_collisions.csv`, `ablation_collisions_summary.csv` |
| Per-instance 진단 | `baseline_diagnostics.json`, `baseline_reconstruction_x3.csv`, `per_instance_rows_x3.csv`, `per_instance_candidates_x3.csv`, `baseline_fmax_x3.csv` |
| 예측값 | `predictions_x3.csv` |
| 코드 | `revision_exp/phase3/` (`common.py`, `precheck.py`, `predictors.py`, `run_phase3.py`, `baselines/{instance_rows,fit_instance,compose_select}.py`) |
