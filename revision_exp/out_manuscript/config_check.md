# §4.2 training configuration: draft vs frozen configuration

출처: `revision_exp/out_reprofiled/logs/config_frozen.json`(3× label을 읽기 전에 기록), `revision_exp/out_reprofiled/artifacts/multi/meta.json`(선택 기준, 실제 적용된 XGBoost 설정), `revision_exp/folds.py`, `revision_exp/train_multi.py`. 불일치가 있으면 config 값이 정답이다. 코드와 config는 바꾸지 않았다.

| 원고 초안 | config / 코드 | 판정 |
|---|---|---|
| grid n_estimators {500, 1000} | `grid.n_estimators` [500, 1000] | 일치 |
| learning_rate {0.01, 0.05} | `grid.learning_rate` [0.01, 0.05] | 일치 |
| max_depth {4, 6} | `grid.max_depth` [4, 6] | 일치 |
| subsample {0.8, 1.0} | `grid.subsample` [0.8, 1.0] | 일치 |
| 3-fold, model set(family) 단위 GroupKFold, shuffle 없음 | `fold_scheme` "GroupKFold(3) by family". `folds.py:18`이 `GroupKFold(n_splits=3)`(sklearn 1.7.2 기본 `shuffle=False`)이고 random_state는 없음 | 일치 |
| 선택 기준 validation MAE, T·D 각각 | meta `selection`: "mean over 3 family-grouped folds of **unweighted** validation MAE; ties -> first grid candidate". T와 D를 각각 선택함 | 일치. **누락**: D도 가중치 없는 MAE로 선택한다는 점, 동점 시 grid의 첫 후보 |
| D 회귀기 sample weight 10 (D_macro > 0.01) | `d_weight` {threshold 0.01, value 10.0}, `drop_definition` "macro". 학습 1,512행 중 1,205행이 weight 10. T에는 weight 없음 | 일치 |
| squared-error objective | `base_params.objective` "reg:squarederror". 적용된 objective도 같음 | 일치 |
| feature 204개 | `n_features` 204 | 일치 |
| 선택된 hyperparameter T | lr 0.05, max_depth 6, n_estimators 500, subsample 1.0 | (초안에 값 없음) → 이 값 |
| 선택된 hyperparameter D | lr 0.05, max_depth 4, n_estimators 1000, subsample 0.8 | (초안에 값 없음) → 이 값 |

초안에 없지만 원고에 쓸 수 있는 사실:

- **학습 데이터**: 1·2·4× 행만 쓴다(`train_rates` [x1, x2, x4]). 1,512 후보, 66 workloads, 22 families다.
- **나머지 XGBoost 설정은 library 기본값이다**(xgboost 3.0.5, meta의 `resolved_config`): colsample_bytree 1, min_child_weight 1, reg_lambda 1, reg_alpha 0, gamma 0, max_bin 256, hist tree method(`grow_quantile_histmaker`), depthwise. 그 밖에 `random_state` 42, `n_jobs` −1이다.
- **CV MAE**(선택된 설정): T 0.029, D 0.032.
- **α = 0.3은 학습에 쓰지 않는다.** T와 D를 따로 예측한 뒤 Ŝ = T̂ − 0.3·D̂로 합친다. 최종 선택 점수는 clip하지 않는다(`out_phase3/CLIP_CHECK.md`).
