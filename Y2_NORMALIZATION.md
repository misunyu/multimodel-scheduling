# y2 정규화로 점수식 척도 통일 — 구현 보고서

> **배경**: `SCORE_FORMULA_AUDIT.md`에서 확인한 척도 불일치 — 점수식
> `S = y1 − α·y2 (+ β·y3)`에서 y1·y3는 `(models,workload,rate)` 그룹 정규화값인데
> **y2만 raw miss rate**였다. 세트마다 y2의 발언권이 달라져 α=0.3의 트레이드오프가 일관되지 않았다.
> 본 작업은 y2도 같은 그룹 키로 정규화해 **세 항을 모두 "세트 내 최적 대비 상대값"**으로 통일한다.

## 0. 결론 요약

| 항목 | 결과 |
|---|---|
| y2 정규화식 | `(models,workload,rate)` 그룹 내 **min-max**, best=0·worst=1 (작을수록 좋음 유지) |
| 적용 지점 | `_normalize_targets` (y1/y3 정규화와 **같은 함수·같은 그룹**) → 학습 타깃 정규화 |
| 스코어링 변경 | **불필요** — booster가 정규화 타깃을 직접 예측, `score_combo`는 그대로 |
| y2 재학습 | `deploy_cpu_{npu,gpu}_y2.json` 재학습 (540창), **y1/y3는 불변** |
| y2 out-of-fold Spearman | NPU **+0.806 → +0.869**, GPU **+0.876 → +0.894** (개선) |
| 세 항 척도 | y1·y2·y3 모두 세트 내 **[0,1]** |
| 순위 변화 | **7/15** 세트 top-5 변동 (base1·base2는 top-1까지) |
| y3 마스킹 | **불변** (y3_valid 372/540) |

---

## 1. 적용한 y2 정규화식과 방향

**`xgboost_model/deploy_selector_xgb_suite.py` `_normalize_targets()`** (y1/y3와 동일 함수·동일 그룹):
```python
# y1, y3 (클수록 좋음): T = F / Fmax
# y2 (작을수록 좋음): 같은 그룹 min-max
y2 = clip(y2, 0, 1)
for g, idxs in groups.items():          # 그룹 키 = (models, workload, rate) — y1/y3와 동일
    lo, hi = min(y2[g]), max(y2[g]); span = hi - lo
    y2_norm[i] = (y2[i]-lo)/span if span>1e-9 else 0.0   # best=0, worst=1
```
- **방향 유지(핵심 함정 회피)**: y1/y3의 `F/Fmax`를 y2에 그대로 쓰면 "작을수록 좋음"이 뒤집힌다.
  그래서 y2는 **min-max**로 그룹 최선(miss 최소)=0, 최악=1 → 점수식의 `−α·y2_norm`이 그대로
  성립(miss 적은 배치일수록 페널티 작음).
- **편차 0 그룹**(모든 조합 miss 동일) → `y2_norm=0` (분모 0 방지).
- **방향 실증** (합성 + 실측):
  - 합성: y2_raw [0.6, 0.8, 1.0] → y2_norm [0.0, 0.5, 1.0] (best 0.6 → 0).
  - 실측 S1: 최선 combination_4의 y2_norm=**0.0002** → 페널티 최소 → 1위(score 0.999).

## 2. 학습/스코어링 일관 적용

y1/y3와 **완전히 대칭**으로 처리했다:
- **학습**: `_normalize_targets`가 y1/y3/y2 세 타깃을 모두 그룹 정규화 → y2 booster는
  **정규화된 y2를 예측하도록 학습**된다.
- **스코어링**: `predict_targets`가 booster 출력(정규화 y2)을 그대로 반환, `score_combo`가
  `− α·y2` 계산. **스코어링 코드는 한 줄도 안 바꿨다** — 정규화가 타깃(=booster)에 내재되므로
  y1/y3와 같은 경로로 자동 반영된다.
- **불일치 방지**: y2는 (y3와 달리) **마스킹이 없다** — 전 540행 학습·스코어링에 동일 적용.
  y3_valid 마스킹은 y2와 무관하게 그대로 동작(372행, §6).

## 3. y2 예측기 재학습 결과

원시 데이터(본수집 540창, `xgboost_model/full_collection_540/cpu_{npu,gpu}/`,
파일럿 160창 아님) 재사용. 재수집 없음. 3-fold CV, out-of-fold, [0,1] 클립.
구 booster는 `xgboost_model/artifacts/pre_y2_norm_backup/`에 백업.

| 플랫폼 | 행수 | y2 oof Spearman (raw, before) | y2 oof Spearman (min-max, after) | cv MAE(y2) |
|---|---:|---:|---:|---:|
| CPU-NPU | 540 | +0.806 | **+0.869** | 0.091 |
| CPU-GPU | 540 | +0.876 | **+0.894** | 0.094 |

→ **정규화 후 y2 예측 난이도가 나빠지지 않았고 오히려 개선**됐다. min-max가 그룹 내 miss
스프레드를 [0,1] 전체로 펼쳐 순위 신호가 선명해졌기 때문. (y1/y3 예측기는 정의 불변이라
재학습하지 않음 — mtime·백업으로 확인.)

## 4. 세 항 척도 일치

동일 세트 내 예측값 범위 (NPU):

| 항 | 범위 | 평균 \|기여\| (α·/β· 반영) |
|---|---|---:|
| y1 | [0.055, 1.0] | 0.471 |
| **y2 (norm)** | **[0.0, 1.0]** | **α·y2 = 0.202** (α=0.3) |
| y3 | [0.0, 1.0] | β·y3 = 0.332 (β=0.5, 생성 조합 n=292) |

→ 세 항 모두 세트 내 **[0,1]**. 세 기여가 같은 자릿수(0.47 / 0.20 / 0.33)로, α·y2가 이제
**실제 스프레드**를 가져 트레이드오프에 참여한다. (raw 시절엔 y2가 0.5~1.0에 몰려 α·y2의
순위 영향력이 사실상 없었다 — 이 작업이 고친 문제.)

## 5. 손계산 재현 (변경 후, NEW booster)

`hand = y1 − 0.3·y2_norm (+ 0.5·y3 if gen)`. CPU-NPU.

**S1 (vision-only)** — y3 항 없음, y2_norm 방향 확인:
| combo | y1 | y2_norm | y3 | gen | 화면 score | 손계산 | 일치 |
|---|---:|---:|---:|:--:|---:|---:|:--:|
| combination_4 | 0.9990 | **0.0002** | 0.0000 | F | 0.99899 | 0.99899 | ✅ |
| combination_3 | 0.6312 | 0.7991 | 0.0000 | F | 0.39149 | 0.39149 | ✅ |
| combination_2 | 0.5019 | 1.0000 | 0.0000 | F | 0.20186 | 0.20186 | ✅ |
| combination_1 | 0.0901 | 1.0000 | 0.0000 | F | −0.20986 | −0.20986 | ✅ |

**S3 (generative)** — y2_norm이 낮은 배치가 정당하게 상승:
| combo | y1 | y2_norm | y3 | gen | 화면 score | 손계산 | 일치 | llama1b |
|---|---:|---:|---:|:--:|---:|---:|:--:|---|
| combination_4 | 0.9980 | 0.5401 | 0.8703 | T | 1.27112 | 1.27112 | ✅ | cpu |
| combination_8 | 0.7642 | **0.0196** | 0.9974 | T | 1.25701 | 1.25701 | ✅ | npu |
| combination_7 | 0.6757 | 0.3712 | 0.7888 | T | 0.95870 | 0.95870 | ✅ | npu |
| combination_6 | 0.5847 | 0.5044 | 0.7789 | T | 0.82281 | 0.82281 | ✅ | npu |
| combination_3 | 0.6341 | 0.7920 | 0.7810 | T | 0.78697 | 0.78697 | ✅ | cpu |

→ 손계산 5/5 정확 일치. combination_8(llama1b가 NPU, 그룹 내 **miss 최소** y2_norm=0.0196)이
페널티가 거의 0이 되어 1위에 근접(1.257 vs 1.271). raw 때는 y2=0.6165로 눌려 있었다.

## 6. 변경 전/후 순위 변화 + y3 마스킹 불변

- **순위(top-5) 바뀐 세트: 7/15** — S3, S4, base1, base2, S5, S7, S10.
  이 중 **base1·base2는 top-1(추천 배치)까지 변경** (combination_4 → combination_8).
- **y3 마스킹 불변**: y3_valid 372/540 (재학습 전후 동일), 스코어링의 `combo_has_generative`
  플래그도 그대로. 이번 변경은 y2 정규화만 건드렸고 y3 경로는 무접촉.

## 7. α, β 기여도 재검토 (보고만, 변경 없음)

- 척도 통일 후 평균 |기여|: **y1 0.471 / α·y2 0.202 / β·y3 0.332** (§4).
- α=0.3, β=0.5는 **유지**. 세 항이 이제 같은 자릿수라 현재 가중이 균형적으로 보인다
  (y1이 가장 크고, 생성 세트에서 β·y3가 그다음, α·y2가 조정항). 특별히 재튜닝이 필요해
  보이지는 않으나, 정밀 튜닝은 별도 작업으로 분리 권장 (본 작업 범위 밖).

---

## 부수효과 / 주의

- `predict_best_combination`의 df 컬럼 `pred_deadline_miss_rate`는 이제 **정규화된 y2**
  (raw miss rate 아님). GUI는 이 값을 사용자에게 "miss rate %"로 직접 표시하지 않고
  스코어링에만 쓰므로 기능 영향은 없으나, 이 컬럼을 절대 miss rate로 해석하면 안 된다.
  (원시 측정 miss rate는 `full_collection_540`의 window `total.deadline_miss_rate`에 그대로 보존.)

## 변경 파일

- 소스: `xgboost_model/deploy_selector_xgb_suite.py` (`_normalize_targets`에 y2 min-max 추가).
- 아티팩트: `deploy_cpu_{npu,gpu}_y2.json` 재학습(교체). y1/y3/features/coverage **불변**.
- 백업: `xgboost_model/artifacts/pre_y2_norm_backup/deploy_cpu_{npu,gpu}_y2.json`.
- 검증 스크립트: `<scratchpad>/retrain_y2_norm.py`, `verify_y2_norm.py`.
