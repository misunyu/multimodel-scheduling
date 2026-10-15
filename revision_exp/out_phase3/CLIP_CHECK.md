# Predicted score clipping 점검 (Phase 3 후속)

작성 2026-10-14. 재학습하지 않았고, 저장된 예측값(`predictions_x3.csv`, `per_instance_candidates_x3.csv`, Phase 2 `out_reprofiled/predictions_x3.csv`)만 사용했다.
코드: `revision_exp/phase3/clip_check.py`, `revision_exp/phase3/reaggregate.py`. 상세 표: `clip_check/`.

**결론: 차이가 있다.** (§6에 Fig. 4 Single, §7에 재현 경로 정리를 추가) 최종 Ŝ의 clip(0–1)이 12개 평가 workload 중 하나에서 두 방법(MLP, XGBoost −P)의 선택을 바꿨다. 지시대로 모든 방법의 Ŝ를 clip 없는 식으로 통일하고 영향 받은 표를 다시 집계했다. Grid·feature·fold·모델은 그대로다.

## 1. Ŝ 계산식 (코드 인용)

| 방법 | T̂ 개별 clip | D̂ 개별 clip | 최종 Ŝ clip (변경 전) | 위치 |
|---|---|---|---|---|
| XGBoost Full (Phase 2 artifact) | 없음 | 없음 | **[0, 1]** | `revision_exp/predict.py:28-31` `multi_score(..., clip=True)`, 호출 `phase3/run_phase3.py:175`, `revision_exp_adapter/run_reprofiled.py:291` |
| XGBoost −P / −I | 없음 | 없음 | **[0, 1]** | 같음 (`run_phase3.py:175`) |
| Ridge / Random Forest / MLP | 없음 | 없음 | **[0, 1]** | 같음 (`run_phase3.py:175`) |
| Per-instance ([14]-inspired) | t_ratio ≥ 0 (instance), 합산 뒤 T̂ = F̂ / max F̂ ∈ [0, 1] | q_i ∈ [0, 1] (instance), D̂ = 평균 | **없음** | `phase3/baselines/compose_select.py:26-27, 36` |
| (참고) Phase 2 단일 목적 (Fig. 4 Single) | – | – | [0, 1] | `revision_exp/predict.py:34-36` `single_score` |

```python
# revision_exp/predict.py:28-31
def multi_score(t_hat, d_hat, alpha, clip=True):
    """S_hat = T_hat - alpha * D_hat, clipped to [0, 1] as the existing predict path does (default --clip_pred_score)."""
    raw = t_hat - alpha * d_hat
    return (np.clip(raw, 0.0, 1.0) if clip else raw), raw

# revision_exp/phase3/baselines/compose_select.py:26-27, 36
r["t_hat"] = np.clip(np.asarray(pred_t, dtype=float), 0, None) * r["lam"].values
r["q_hat"] = np.clip(np.asarray(pred_q, dtype=float), 0, 1)
s_hat = (c["T_hat_select"] - alpha * c["D_hat"]).values
```

Clip의 출처는 기존 코드의 기본 옵션이다. `xgboost_model/deploy_selector_xgb_suite.py:1035`에 `--clip_pred_score`(기본 True)가 있고, `:1251-1252`에서 `pred_score_raw = max(0.0, min(1.0, pred_score_raw))`로 적용한다. Phase 2는 이 동작을 그대로 따랐다.

## 2. 실측 S (evaluator)에는 clip이 없다

- `revision_exp/evaluate.py:66` `S = (sub["T"] - alpha * sub["D"]).values`이고, oracle·regret·Top-k가 모두 이 값을 쓴다. clip은 없다.
- 실측 T = F/F_max ∈ (0, 1], D_macro ∈ [0, 1]이라 실측 S가 [0, 1] 밖으로 나갈 수 있는 것은 음수 쪽뿐이다(All-CPU 등).
- Best-of-5(`revision_exp/baselines.py:125`), paired 차이(`run_reprofiled.py:118`)도 clip 없이 같은 식이다.

## 3. Clip 범위를 벗어난 예측 (3× 후보 504개)

| 방법 | Ŝ_raw < 0 | Ŝ_raw > 1 | 그중 N≥3 (<0 / >1) | T̂ > 1 | D̂ < 0 |
|---|---|---|---|---|---|
| XGBoost Full | 3 | 8 | 2 / 4 | 8 | 13 |
| Ridge | 0 | 5 | 0 / 3 | 13 | 2 |
| Random Forest | 1 | 0 | 1 / 0 | 0 | 0 |
| MLP (small) | 4 | 22 | 3 / 19 | 38 | 18 |
| XGBoost −P | 3 | 15 | 2 / 9 | 16 | 12 |
| XGBoost −I | 3 | 2 | 2 / 0 | 2 | 3 |
| Per-instance | 3 | 0 | 2 / 0 | 0 | 0 |

- **음수 Ŝ 후보는 모두 그 workload의 꼴찌(8개 중 8위)다.** N≥3에서는 두 workload(gpt2+resnext50+vgg19, resnet50+vgg19+yolov4)의 all-CPU 후보만 해당한다. 하한 clip(0)은 순위 끝에서만 동점을 만들어 선택·Top-5에 영향이 없다.
- **영향은 상한 clip(1)에서 나온다.** T̂가 1을 넘는 후보 여러 개가 Ŝ = 1로 동점이 되고, 동점은 combination 번호가 작은 쪽으로 풀린다(`evaluate.select`). 그래서 예측상 1위가 아닌 후보가 선택될 수 있다.

## 4. 방법별 비교 (clip 있음 → 없음)

| 방법 | 선택이 바뀐 workload | Top-5 집합이 바뀐 workload | Top-5 순서만 바뀐 workload | Avg score | Regret | Top-1 | Top-5 |
|---|---|---|---|---|---|---|---|
| XGBoost Full | 0 | 0 | 0 | 0.9930 → 0.9930 | 0.0044 → 0.0044 | 7/12 → 7/12 | 11/12 → 11/12 |
| Ridge | 0 | 0 | 0 | 0.9941 → 0.9941 | 0.0033 → 0.0033 | 7/12 → 7/12 | 11/12 → 11/12 |
| Random Forest | 0 | 0 | 0 | 0.9948 → 0.9948 | 0.0026 → 0.0026 | 8/12 → 8/12 | 11/12 → 11/12 |
| **MLP (small)** | **1** | 0 | 2 | 0.97292 → 0.97293 | 0.02449 → 0.02447 | **5/12 → 6/12** | 9/12 → 9/12 |
| **XGBoost −P** | **1** | 0 | 1 | **0.9920 → 0.9918** | **0.0054 → 0.0056** | 6/12 → 6/12 | 10/12 → 10/12 |
| XGBoost −I | 0 | 0 | 0 | 0.9960 → 0.9960 | 0.0014 → 0.0014 | 8/12 → 8/12 | 12/12 → 12/12 |
| Per-instance | 0 | 0 | 0 | 0.9868 (원래 clip 없음) | 0.0106 | 6/12 | 11/12 |

달라진 (method, workload) 전부:

| method | workload | 선택 (clip 있음 → 없음) | 실측 S | Oracle? | 원인 |
|---|---|---|---|---|---|
| MLP (small) | gpt2+mnasnet+resnet50+resnext50+shufflenet (N=5) | combination_13 → combination_15 | 0.999815 → 0.999992 | 아니오 → **예** | Ŝ_raw 1.0996(15), 1.0363(13), 1.0035(30)가 모두 1로 잘려 3개 동점, 번호가 작은 13이 선택됨 |
| XGBoost −P | 같은 workload | combination_13 → combination_31 | 0.999815 → 0.997822 | 아니오 → 아니오 | Ŝ_raw 1.0116(31), 1.0113(15), 1.0053(29), 1.0050(13)이 모두 1로 잘려 4개 동점, 13이 선택됨 |

Top-5 순서만 바뀐 경우(집합은 같음): MLP 위 workload와 mnasnet…yolov4 (N=7, clip 시 1위 동점 5개), XGBoost −P 위 workload. Top-5 적중에는 영향이 없다.

이 workload(oracle = combination_15, 1·2위 차 0.00018)는 Top-1 동률권 통계 (i)에 들어가는 workload다. 바뀐 선택 둘 다 실측 S 차가 0.002 이하라 해석에는 영향이 없다.

## 5. 수정 내용

- 원고 §3 수식(clip 없음)에 맞춰 모든 direct 방법의 선택 점수를 **Ŝ = T̂ − 0.3·D̂(clip 없음)** 로 통일했다. Per-instance는 원래 clip이 없어 바뀌지 않는다.
- **재집계 절차**: 저장된 예측값으로 먼저 clip을 켠 상태에서 기존 표 7개(`table_predictors`, `table_ablation`, `table_baseline`, `table_all`, `paired_per_workload`, `paired_summary`, `per_workload_all`)를 정확히 재현했다(`clip_check/reproduce_check.json`, 전부 true). 그다음 clip만 끈 결과로 같은 파일들을 다시 썼다.
- XGBoost −P / −I의 예측값은 모델이 float32로 낸 값이 CSV에 짧은 float32 표기로 저장돼 있다. float32로 다시 읽어 원래 값을 정확히 복원했다(그대로 float64로 읽으면 3e-8 차이가 남).
- **변경 전 파일**은 `clip_check/before/`에 보관했다. 갱신된 파일은 위 7개 표와 각 `.md`, `predictions_x3.csv`(S_hat 열이 clip 없는 값), `summary.json`(`selection_score` 항목 추가)이다. 표 각주의 선택 점수 설명도 바꿨다.
- `REPORT_PHASE3.md`의 해당 숫자를 고쳤다. MLP Top-1 5/12 → 6/12, −P Avg 0.9920 → 0.9918, regret 0.0054 → 0.0056, −P − Full −0.0010 (1/10/1) → −0.0012 (0/11/1), −P − All-GPU +0.0193 → +0.0191.
- 결론 문장은 바뀌지 않는다. 회귀기·feature 그룹 간 차이는 여전히 반복 잡음 수준이고, MLP는 여전히 All-GPU 수준이다.

## 6. Phase 2 결과와 Fig. 4

Phase 2 산출물(`revision_exp/out_reprofiled/`)은 고치지 않았다.

| 항목 | clip 있음 vs 없음 |
|---|---|
| Table 3(a) Ours (XGBoost Full과 같은 예측) | 동일 |
| Ours, D without weight (`ablation_d_weight.csv`) | 동일 (0.9946, 7/12, 12/12) |
| Fig. 4 Multi, α = 0.1 … 1.0 | 10개 α 모두 선택·Top-5·Avg·regret 동일 |
| Fig. 4 Single | **다름** (아래) |
| 학습 데이터 성능(`train_performance.csv`) | 확인하지 않음(원고 수치 아님) |

### 6.1 Fig. 4 Single (`fig4_noclip/`, 코드 `revision_exp/phase3/fig4_noclip.py`)

재학습하지 않았다. 저장된 α별 artifact(`out_reprofiled/artifacts/single/alpha_*.json`)로 3× 후보를 예측만 다시 했다.

- **재현 확인**: clip을 켠 상태(`predict.single_score` 기본값)에서 α 10개 모두 Phase 2 `fig4_data.csv`의 Single 값과 같다. Avg score, regret, Top-1, Top-5, 전부-GPU 선택 수가 모두 일치한다(`fig4_noclip/reproduce_check.json`, `reproduced_all_alphas: true`). Phase 2는 workload별 Single 선택을 저장하지 않아, 선택은 이 집계값들로 대조했다.
- **clip 범위 밖 예측**: α에 따라 Ŝ_raw > 1인 후보가 6–15개(N≥3에서 2–9개), Ŝ_raw < 0인 후보가 0–12개다.

선택이 바뀐 (α, workload) 전부 (7건):

| α | workload | 선택 (clip 있음 → 없음) | 실측 S | Oracle? |
|---|---|---|---|---|
| 0.3 | gpt2+mnasnet+resnet50+resnext50+shufflenet (N=5) | 15 → 31 | 0.999992 → 0.997822 | 예 → 아니오 |
| 0.4 | mnasnet+resnet50+resnext50 (N=3) | 6 → 7 | 0.999981 → 1.000000 | 아니오 → 예 |
| 0.5 | gpt2+…+shufflenet (N=5) | 13 → 31 | 0.999810 → 0.997574 | 아니오 → 아니오 |
| 0.6 | mnasnet+resnet50+resnext50 (N=3) | 6 → 7 | 0.999972 → 1.000000 | 아니오 → 예 |
| 0.6 | mnasnet+resnext50+yolov4 (N=3) | 6 → 7 | 1.000000 → 1.000000 | 예 → 예 (oracle 동점 둘) |
| 0.8 | gpt2+…+shufflenet (N=5) | 13 → 31 | 0.999801 → 0.997201 | 아니오 → 아니오 |
| 1.0 | gpt2+…+shufflenet (N=5) | 29 → 31 | 0.999337 → 0.996953 | 아니오 → 아니오 |

7건 모두 clip 시 1위가 2–3개 동점(Ŝ = 1)이었다(`fig4_noclip/single_changed.csv`).

Single 집계 (clip 있음 = Phase 2 → 없음):

| α | Avg score | Regret | Top-1 | Top-5 |
|---|---|---|---|---|
| 0.1 | 0.9953 → 0.9953 | 0.0033 → 0.0033 | 6 → 6 | 12 → 12 |
| 0.2 | 0.9931 → 0.9931 | 0.0046 → 0.0046 | 6 → 6 | 10 → 10 |
| 0.3 | 0.9919 → **0.9917** | 0.0055 → **0.0057** | 8 → **7** | 9 → 9 |
| 0.4 | 0.9913 → 0.9913 | 0.0059 → 0.0059 | 6 → **7** | 10 → 10 |
| 0.5 | 0.9897 → **0.9895** | 0.0074 → **0.0076** | 6 → 6 | 11 → 11 |
| 0.6 | 0.9877 → 0.9877 | 0.0092 → 0.0092 | 5 → **6** | 12 → 12 |
| 0.7 | 0.9900 → 0.9900 | 0.0067 → 0.0067 | 7 → 7 | 11 → 11 |
| 0.8 | 0.9902 → **0.9900** | 0.0063 → **0.0065** | 6 → 6 | 11 → 11 |
| 0.9 | 0.9881 → 0.9881 | 0.0082 → 0.0082 | 6 → 6 | 11 → 11 |
| 1.0 | 0.9889 → **0.9887** | 0.0072 → **0.0074** | 7 → 7 | 11 → 11 |

- Multi는 clip과 무관하다(위 표). `fig4_noclip/fig4_data.csv`의 Multi 열은 Phase 2 값을 그대로 복사했고, Single 열은 clip 없는 새 값이다. clip 있는 Phase 2 값도 별도 열로 남겼다.
- **원고용 Fig. 4**는 `out_phase3/fig4_noclip/fig4_alpha_sensitivity.pdf`(.png)다. 그리기는 Phase 2 `plots.fig4`를 그대로 썼다.
- Phase 2 보고서 §4.4의 "Multi는 10개 α 중 9개에서 Single보다 regret이 작거나 같다"는 그대로다(α 0.1만 0.0034 대 0.0033). 차이의 최대값은 0.0042(α 0.6)다.

## 7. 재현 경로 정리 (`run_phase3.py` 기본값 변경)

- `run_phase3.py`의 최종 Ŝ clip을 끄도록 바꿨다(`CLIP_S_HAT = False`). 표 각주 문구도 `reaggregate.py`와 같게 했다. 그다음 같은 seed(42)로 전체를 다시 실행했다(94초).
- **표 7개가 `reaggregate.py`의 clip 없는 출력과 모두 일치한다**: `table_predictors`, `table_ablation`, `table_baseline`, `table_all`, `paired_per_workload`, `paired_summary`, `per_workload_all`.
- 학습 쪽도 같다.
  - 선택된 hyperparameter와 새 artifact(XGBoost −P/−I, per-instance)가 이전 실행과 byte 단위로 같다.
  - `baseline_diagnostics.json`, `ablation_collisions.csv`, `per_instance_candidates_x3.csv`도 byte 단위로 같다.
  - `hparam_search_*.csv`의 차이는 실행 시간 열과 RF fold MAE의 마지막 자리(최대 1e-16)뿐이다.
- 이제 최종 표는 `run_phase3.py` 한 번으로 재현된다. `reaggregate.py`는 clip 점검 기록으로 남긴다.
- 덮어쓰기 전 재집계 산출물은 `clip_check/reaggregate_final/`, clip 있는 최초 산출물은 `clip_check/before/`에 있다.
