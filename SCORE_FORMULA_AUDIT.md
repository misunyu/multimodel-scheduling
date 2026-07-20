# Predict Best — Top-5 score 계산식 검증 (조사 보고서)

> **성격**: 조사(read-only). 소스·학습 데이터 수정 없음. 아래는 전부 **실제 코드에서 추출**하고
> S1·S3 Predict Best를 실행해 **손계산으로 재현**한 결과다. (설계 문서를 옮긴 것이 아님.)

## 0. 결론 요약

| 항목 | 결과 |
|---|---|
| 실제 식 | `S = y1 − α·y2 (+ β·y3, **가산 조건부**)`, α=0.3, β=0.5 |
| y1/y2/y3 | 예측기(`deploy_cpu_{npu,gpu}_{y1,y2,y3}.json`) 출력, **[0,1] 클립**, **정규화된 타깃** |
| y1 정의 | 재수집 신정의 = **정규화 달성 처리량** (옛 정의 아님) |
| α/β | 런타임 스코어링 상수(하드코딩), 학습과 무관 |
| 화면 score == predictions.csv | **동일** (같은 df) |
| vision-only y3 항 | **완전 제외** (S1 실측 확인) |
| **y3 마스킹 일치성** | ⚠ **불일치** — 학습=조합단위, 스코어링=세트단위 (§4, 핵심) |

---

## 1. 실제 계산식 (코드 추출, 파일:줄)

**`xgboost_model/deploy_selector_xgb_suite.py:651` `score_combo()`**
```python
664   s = float(y1) - float(alpha) * float(y2)
665   if has_generative:
666       s += float(beta) * float(y3)
667   return s
```
→ **`S = y1 − α·y2`, 그리고 `has_generative`가 참일 때만 `+ β·y3`.**
(설계식 `y1 + β·y3 − α·y2`와 대수적으로 동일. 단 β·y3는 "사후 0 대입"이 아니라 **항 자체를
넣지 않는다**.)

**호출·정렬·표시 (`best_deploy_finder_executor.py`)**
- `predict_best_combination()` (763행)에서 combo마다:
  - `819  gen = combo_has_generative(combo_blob)`
  - `820  y1_pred, y2_pred, y3_pred = predict_targets(model_prefix, X, with_y3=gen)`
  - `823  score = score_combo(fps, miss, tok, gen, alpha, beta)`
- `834  df = ...sort_values(["pred_score"], ascending=[False])` → score 내림차순.
- `838  df.to_csv(predictions.csv)` → **저장 score = df의 pred_score**.
- 로그 Top-5: `1530–1532` `self.log(f"  {i+1}. {df.iloc[i]['combination']} -> {df.iloc[i]['pred_score']:.4f}")`
  → **화면 표시 score = predictions.csv의 pred_score (동일 df, 동일 값).**

---

## 2. y1 / y2 / y3 의 실체

**`deploy_selector_xgb_suite.py:681` `predict_targets()`**
```python
695   if tag == "y3" and not with_y3:
696       preds[tag] = np.zeros(X.shape[0], dtype=float)   # y3 booster 미로딩·미실행
697       continue
698   bst = xgb.Booster(model_file=str(model_in_prefix) + f"_{tag}.json")
700   preds[tag] = np.clip(bst.predict(dmat), 0.0, 1.0)     # 예측 직후 [0,1] 클립
```
- y1/y2/y3 = 각각 `deploy_cpu_{npu|gpu}_{y1,y2,y3}.json` 부스터의 **예측값**.
- **[0,1] 클립**은 예측 직후(700행) 적용. score식에는 클립된 값이 들어간다.
- `with_y3=False`이면 y3는 **부스터를 아예 안 돌리고 0** (698행 도달 안 함).
- **정규화 여부**: 부스터는 학습 시 `(models, workload, rate)` 그룹 내 [0,1]로 정규화된
  타깃(`_normalize_targets`)으로 학습됐으므로, 출력도 **정규화 스케일의 y1(정규화 달성 처리량)·
  y3(정규화 토큰)**. y2(데드라인 미스율)는 원래 [0,1]. → score는 **정규화 값**들로 계산되며,
  이 때문에 **세트 내 순위 비교에만 유효**(score_combo 도크스트링 660–662행).
- **y1은 재수집 신정의**(완료건수/측정창 = 달성 처리량)로 학습된 타깃
  `y1_total_throughput_fps`의 예측이다. 옛 정의(1000/avg_infer_time) 아님.

---

## 3. α, β 가중치

- 하드코딩 위치:
  - `best_deploy_finder_executor.py:763` `def predict_best_combination(..., alpha: float = 0.3, beta: float = 0.5)`
  - CLI 기본값 `deploy_selector_xgb_suite.py:736–737` `--alpha default=0.3`, `--beta default=0.5`
- **런타임 스코어링 상수**임이 코드에서 확인됨: 학습(`train_targets`)은 `reg:squarederror`로
  정규화 타깃을 회귀할 뿐 α/β를 쓰지 않는다. α/β는 **예측 후 랭킹 점수**에만 등장.
  (설정 파일/사용자 입력으로 바뀌지 않는 고정 기본값. GUI는 predict 시 이 기본값을 사용.)

---

## 4. y3 마스킹의 스코어링 반영 + 학습 마스킹과의 일치 여부 (핵심)

### 스코어링 쪽 판정: **세트 단위 (배치 무시)**
**`deploy_selector_xgb_suite.py:633` `combo_has_generative()`**
```python
639   for v in _rows_from_combo_struct(combo_blob):
640       m = v.get("model")
644       if _model_kind(m) in ("llm", "vlm"):
645           return True     # 모델이 '존재'하기만 하면 True — execution(cpu/npu) 안 봄
```
→ combo에 LLM/VLM이 **존재하면** `gen=True`. **그 LLM/VLM이 CPU에 있든 가속기에 있든 무관.**
이 `gen`이 `with_y3`(820행)와 `has_generative`(823행)에 그대로 쓰인다.
→ 생성 모델이 있는 세트의 **모든 조합**(CPU-LLM 포함)에 β·y3가 가산된다.

### 학습 쪽 판정: **조합 단위 (가속기 배치)**
**`deploy_selector_xgb_suite.py`**
```python
307   y3_valid = False   # True iff some LLM/VLM view is on an accelerator (npu/gpu)
328   if dev in ("npu", "gpu"):
329       y3_valid = True
...
574   if M is not None and "y3_valid" in M.columns:
575       gen_mask = M["y3_valid"].values.astype(bool)
583   if tag == "y3" and gen_mask is not None:
584       Xi, Yi = X[gen_mask], Y[gen_mask]   # y3는 '가속기-생성' 행으로만 학습
```
→ y3 부스터는 **LLM/VLM이 실제 가속기에 배치된 조합**(조합 단위)으로만 학습. CPU-LLM 조합은
y3 학습에서 제외(마스킹)됐다.

### ⚠ 불일치 (granularity mismatch)
| | 판정 단위 | CPU-LLM 조합의 y3 |
|---|---|---|
| **학습** (train_targets) | 조합 단위 (LLM on npu/gpu) | **마스킹 = 학습 제외** |
| **스코어링** (combo_has_generative) | 세트 단위 (LLM 존재) | **β·y3 가산 (외삽 예측)** |

- vision-only 세트(S1·S2·S9·S10): 생성 모델 없음 → 양쪽 다 y3 제외 → **일치**.
- 가속기-LLM 조합: 양쪽 다 y3 사용 → **일치**.
- **CPU-LLM 조합(생성 세트 내)**: 학습은 제외했는데 스코어링은 β·y3를 더한다 → **불일치**.
  y3 부스터가 학습에서 안 본 CPU-LLM 특성행에 대해 **외삽값**을 내고, 그 값이 점수에 가산된다.

---

## 5. 실측 재현 (S1 / S3 Top-5, 손계산 vs 화면)

CPU-NPU / deploy_cpu_npu, α=0.3, β=0.5. `hand = y1 − 0.3·y2 (+ 0.5·y3 if gen)`.

### S1 (resnet50, yolo11s) — vision-only
| combo | y1 | y2 | y3 | gen | 화면 score | 손계산 | 일치 |
|---|---:|---:|---:|:--:|---:|---:|:--:|
| combination_4 | 0.9990 | 0.6891 | 0.0000 | False | 0.79231 | 0.79231 | ✅ |
| combination_3 | 0.6312 | 0.9454 | 0.0000 | False | 0.34761 | 0.34761 | ✅ |
| combination_2 | 0.5019 | 1.0000 | 0.0000 | False | 0.20186 | 0.20186 | ✅ |
| combination_1 | 0.0901 | 1.0000 | 0.0000 | False | −0.20986 | −0.20986 | ✅ |

→ gen=False, y3=0, **β·y3 항이 실제로 빠짐**. `y1 − 0.3·y2`만으로 화면 값 정확 재현.

### S3 (llama1b, resnet50, yolo11s) — 생성 세트
| combo | y1 | y2 | y3 | gen | 화면 score | 손계산 | 일치 | llama1b 배치 |
|---|---:|---:|---:|:--:|---:|---:|:--:|---|
| **combination_4** | 0.9980 | 0.7945 | 0.8703 | True | **1.19480** | 1.19480 | ✅ | **cpu** |
| combination_8 | 0.7642 | 0.6165 | 0.9974 | True | 1.07796 | 1.07796 | ✅ | npu |
| combination_7 | 0.6757 | 0.7701 | 0.7888 | True | 0.83904 | 0.83904 | ✅ | npu |
| combination_3 | 0.6341 | 0.9207 | 0.7810 | True | 0.74837 | 0.74837 | ✅ | **cpu** |
| combination_6 | 0.5847 | 0.8214 | 0.7789 | True | 0.72771 | 0.72771 | ✅ | npu |

→ 손계산이 5개 모두 화면 값과 **정확히 일치**(식·α·β 확정).
→ **1위 combination_4는 llama1b가 CPU인데도** `gen=True`라 y3=0.8703이 예측되어
`β·y3 = 0.435`가 가산됐다. 이 보너스가 없었다면 combination_4의 점수는 0.760으로
combination_8(1.078)보다 낮아 **순위가 뒤바뀐다**.

**영향**: CPU-llama1b는 1회 생성 ~198초로 측정창에서 토큰이 거의 0 (바로 이 때문에 학습에서
y3 마스킹됨). 그런데 스코어링은 이 조합에 β·y3≈0.44의 토큰 보상을 준다. y3 부스터의 외삽값이
CPU-LLM(0.8703)을 가속기-LLM(0.9974)보다 낮게는 냈지만 **여전히 큰 보상**이라, 실제로 토큰을
못 내는 배치가 상위로 올라갈 수 있다.

---

## 6. 설계식 vs 실제 코드 차이

| 항목 | 설계(배경) | 실제 코드 | 판정 |
|---|---|---|---|
| 식 형태 | `y1 + β·y3 − α·y2` | `y1 − α·y2 (+β·y3 조건부)` | 동일 (대수적) |
| α, β | 0.3 / 0.5 | 0.3 / 0.5 (763, 736–737행) | 일치 |
| y1 정의 | 달성 처리량(신정의) | 정규화 `y1_total_throughput_fps` 예측 | 일치 |
| 정규화/클립 | (명시 안 됨) | (models,workload,rate) 정규화 타깃 + [0,1] 클립(700행) | 확인 |
| "LLM/VLM 없는 세트 y3 제외" | 세트 단위 서술 | `combo_has_generative` 세트 단위(존재) | **서술과는 일치** |
| **y3 마스킹 정합성(학습↔스코어링)** | (학습은 조합단위로 바뀜) | 스코어링은 **세트단위** 유지 | ⚠ **불일치** |

**핵심 결론**: 점수식·가중치·y1 신정의·클립·정규화는 모두 코드와 설계가 일치한다. 유일한
불일치는 **y3 마스킹 판정 단위**다 — 학습은 조합단위(`y3_valid`, LLM/VLM on accelerator)로
바뀌었는데, 스코어링의 `combo_has_generative`는 여전히 **세트단위(생성 모델 존재)**라, 생성 세트
안의 **CPU-LLM 조합에 외삽된 β·y3가 가산**된다. vision-only 세트와 가속기-LLM 조합은 문제없고,
**생성 세트에서 LLM을 CPU에 두는 후보의 순위가 부풀 수 있다** (S3 combination_4 실측으로 확인).

> 참고(수정 제안 아님, 조사 범위 밖): 스코어링을 학습과 정합시키려면 `combo_has_generative`
> 대신 featurize 시 계산되는 **조합단위 `y3_valid`**(LLM/VLM이 실제 npu/gpu에 배치)로 β·y3
> 가산 여부를 결정하면 된다. 본 보고서는 조사만 수행했고 코드는 변경하지 않았다.
