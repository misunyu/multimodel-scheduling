# decomposed per-instance model 구현 검토 (P6)

- 작성: 2026-08-04
- 대상: `xgboost_model/full_collection_540/scripts/train_decomposed.py` + 실제 호출 공용 코드
- **읽기 전용.** 코드·데이터·아티팩트·원고 미수정. 산출물은 이 review.md뿐이다.
- 원고 대조 리비전: `manuscript/mlforsys_main.tex` **r4** (sha256 `d31d17a0…`)

---

## 1. predictor 구조 — **(a) 전 모델 공용 단일 회귀기**

`train_decomposed.py:129-140` (`main()`):

```python
pred = np.full(len(yv), np.nan)
for f in range(ac.N_FOLDS):                       # N_FOLDS = 3
    tr, te = fold != f, fold == f
    params, inner_mae = ac.cv_select_params(Xv[tr], yv[tr], grp[tr]…, feature_names=VIEW_FEATURES)
    bst = ac.train_booster(Xv[tr], yv[tr], params, feature_names=VIEW_FEATURES)
    pred[te] = ac.predict_booster(bst, Xv[te], feature_names=VIEW_FEATURES)
```

`Xv`/`yv`는 `:124-125`에서 **vision 뷰 전체를 한 덩어리로** 모은 행렬이다
(`V.loc[vis, VIEW_FEATURES]`). 모델별·디바이스별로 나누어 학습하는 분기가 없다.

- **회귀기 개수**: 플랫폼당 **3개**(fold당 1개), 두 플랫폼 합쳐 **6개**. P1과 달리
  "final" 부스터를 따로 학습·저장하지 않는다 (`ac.ARTIFACTS`로 내보내는 코드 없음).
- **모델 정체성의 반영 방식**: one-hot 모델 ID가 **없다**. `_view_features`
  (`deploy_selector_xgb_suite.py:250-279`)가 주는 것은 `view.is_vision`/`view.is_llm`
  두 종류 플래그와 정적 프로파일 수치(`static_infer_sel`, `capacity_fps` 등)뿐이므로,
  모델 구분은 **정적 프로파일 값의 차이로만 암묵적으로** 들어간다.
- **학습 행**: vision 뷰만. 플랫폼당 vision 2220행 / 생성 732행이며 생성 뷰는 학습·합성
  양쪽에서 제외된다 (`:122-127`, `:153`). 근거는 docstring `:3-4` — 뷰별 tokens가 보존되지
  않아 y3는 분해 불가.
- **그룹 분할**: 뷰는 자신이 속한 창의 `(models, rate_factor)` 그룹을 상속하고
  (`:121` merge), fold는 **P1이 확정한 매핑을 파일에서 읽어 그대로 쓴다**
  (`:100-101` `g2f = p1["group_to_fold"]`, `:112` `w["fold"] = w["group"].map(g2f)`,
  `:113` 전 그룹 커버 assert). 즉 P6가 fold를 새로 뽑지 않는다.

## 2. 학습기·검증 — XGBoost, direct와 동일 헬퍼·동일 3-fold groupwise

| 항목 | 값 / 경로 |
|---|---|
| 학습기 | XGBoost. `ac.train_booster` → `analysis_common.py:_lazy_xgb()` → `xgb.train` |
| 기본 파라미터 | `deploy_selector_xgb_suite.py:519-523` `_PARAMS`: `reg:squarederror`, rmse, `max_depth=5`, `eta=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3.0`, **`seed=42`** |
| 탐색 그리드 | 동 `:525-530` `_PARAM_GRID` 4조합 (`d3/η.1`, `d4/η.1`, `d5/η.05`, `d6/η.1`) |
| 부스팅 라운드 | `analysis_common.cv_select_params`/`train_booster`의 기본 `rounds=300` |
| 하이퍼파라미터 선택 | `ac.cv_select_params` (`analysis_common.py:~230`) — **그룹 인식 내부 3-fold**, MAE 기준 |
| 외부 검증 | 3-fold, **P1의 group→fold 매핑 그대로 상속** (§1) |

**direct(P1)와 공유하는 코드 경로**: `ac.cv_select_params`, `ac.train_booster`,
`ac.predict_booster`, `ac.spearman`, `ac.per_group_spearman`, `ac.scores_from`,
`ac.ranking_metrics` — 전부 `analysis_common.py`의 동일 함수다. P6 전용으로 추가된 것은
`feature_names` 인자뿐이며, 이는 P6가 18-feature 스키마를 쓰기 위해
`analysis_common`에 소폭 추가된 것이다(`SUMMARY.md` 구현결정 12에 기록).

**차이점**: P1은 `predict_booster`에서 `clip=True`로 [0,1] 클리핑하는데 P6도 기본값
`clip=True`를 그대로 쓴다(`:137`) — target이 `min(1.0, ratio)`로 [0,1]이므로 정합적이다.

## 3. 입력 feature — **co-runner 요약 포함** (Kim 구조 이식 주장의 판별점)

`VIEW_FEATURES`는 `train_decomposed.py:39-47`에 정의된 **18개**다.

**(A) instance 자신의 정보 12개** — `_view_features()`
(`deploy_selector_xgb_suite.py:250-279`)가 생성:
`view.infps`, `view.exec_cpu/gpu/npu`, `view.is_vision`, `view.is_llm`,
`view.static_infer_sel`, `view.static_load_sel`, `view.static_tokens_sel`,
`view.capacity_fps`, `view.load_factor`, `x.infps__static_infer_sel`

**(B) co-runner 요약 6개** — `train_decomposed.py:74-82`에서 추가:

```python
same  = [o for j,o in enumerate(views) if j != i and o["dev"] == me["dev"]]
other = [o for j,o in enumerate(views) if j != i and o["dev"] != me["dev"]]
gen   = [o for o in views if not o["is_vision"]]
feat.update({
    "co.same_dev_load_sum":  sum(o["lf"] for o in same),
    "co.same_dev_count":     float(len(same)),
    "co.other_dev_load_sum": sum(o["lf"] for o in other),
    "co.other_dev_count":    float(len(other)),
    "co.set_has_gen":  1.0 if gen else 0.0,
    "co.gen_on_accel": 1.0 if any(o["dev"] in ("gpu","npu") for o in gen) else 0.0,
})
```

→ **자기 정보만 쓰는 순수 per-instance 모델이 아니다.** 같은 디바이스/다른 디바이스에
동시 배치된 co-runner의 개수와 load_factor 합, 생성 모델의 존재·가속기 배치 여부까지
본다. 세트 구성과 동시 배치가 feature로 들어가므로, **"instance 단위로 쪼개되 contention
맥락은 요약해 준다"**는 구조다.

**모든 feature가 plan 수준**이다: 정적 프로파일 + 스케줄 infps + 배치 조합에서만 파생되고,
창 측정값은 하나도 들어가지 않는다(`_view_features` `:262-264` 주석이 명시: 측정 per-view
동역학은 predict 시점에 없고 target leakage라 제외). docstring `:6-8`도 동일 진술.

## 4. attainment 정의 — 코드 원문

`train_decomposed.py:60-63, 83-89` (`build_view_dataset()`):

```python
infps = float(vmap.get((m, dev), 0.0))            # 스케줄의 요청률 (rate 배율 반영)
s_infer, _, _ = _device_static(m, dev, S)         # 정적 프로파일 infer(ms)
cap = 1000.0 / s_infer if np.isfinite(s_infer) and s_infer > 0 else np.nan
eff = min(infps, cap) if np.isfinite(cap) else infps          # ← 분모
...
ratio_raw = me["thr"] / me["eff"] if me["eff"] > 1e-9 else np.nan
"target": min(1.0, ratio_raw) if np.isfinite(ratio_raw) else np.nan
```

- 분모 `eff = min(요청률, 고립 용량)` — 원고 서술과 일치.
- **isolated capacity의 출처**: `capacity_fps = 1000 / static_infer_ms`, 그 `static_infer_ms`는
  `load_static_profiles(ac.STATIC_JSON)` (`:94`) →
  `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json`의
  `{dev}_infer` 필드. greedy baseline이 쓰는 것과 **동일한 정적 프로파일**이다.
- `infps`의 출처는 `ac.infps_map_for(platform)` (`:52`) → 수집 스케줄 YAML
  (`schedules/collection/collect_cpu_{gpu,npu}.yaml`), rate 배율이 이미 곱해진 값.
- **상한 클리핑** `min(1.0, ·)`이 있고, 클리핑된 target 수는 `:123`에서 집계되어
  **양 플랫폼 0건**으로 기록된다(아티팩트 확인). 즉 클리핑은 실제로 발동하지 않았다.

## 5. 합성 함수 — ŷ1은 가중합/Fmax, **ŷ2는 비가중 평균 → 측정 min-max 재스케일**

`train_decomposed.py:151-167`:

```python
for name, col in (("pred","pred"), ("meas","target")):
    g = V[vis].groupby("combination")
    y1_sum = g.apply(lambda d: float((d[col] * d["eff"]).sum()))      # Σ r·eff
    miss   = g.apply(lambda d: float(np.clip((1.0 - d[col]).mean(), 0.0, 1.0)))
...
comp[f"y1_decomp_{name}"] = comp[f"y1_sum_{name}"] / comp["fmax"]
comp[f"y2_decomp_{name}"] = np.where(span > 1e-9,
        (comp[f"miss_{name}"] - comp["y2_lo"]) / span, 0.0)
```

**ŷ1** = `Σ_vision (r̂_v × eff_v) / Fmax_group`.
`Σ r·eff`는 뷰별 예측 처리량의 합(= 창 y1의 재구성), `fmax`는 `gstat`
(`:116-117`)의 그룹 **측정** `y1_raw` 최대값. 즉 P1의 정규화 y1과 같은 척도로 맞춘다.

**ŷ2** = `clip01( mean_vision(1 − r̂_v) )`를 그룹 **측정** y2의 `lo/hi`로 min-max한 값.
매핑 성격을 명확히 하면:

- **임계(threshold) 아님**, **학습된 함수 아님**. 두 단계 모두 고정 산식이다.
- 1단계: 뷰별 미달률 `(1−r̂)`의 **비가중 산술평균** → [0,1] 클리핑.
- 2단계: 그룹의 측정 y2 `lo/hi`에 대한 **아핀(affine) min-max 재스케일**.
  `span ≤ 1e-9`이면 0.

이 매핑이 **근사**임은 코드 자신이 명시한다 (`:17-18` docstring "an APPROXIMATE
reconstruction of window y2", 출력 md `:236-237`): 실제 창 y2는 **요청 수 가중**이 다르고
**생성 뷰도 포함**하는데, 합성은 vision 뷰의 비가중 평균만 쓴다. §7의 0.834/0.763이
바로 이 근사 자체의 손실이다.

**주목할 구조적 사실**: ŷ1은 그룹 측정 `fmax`로, ŷ2는 그룹 측정 `y2_lo/hi`로 재스케일된다.
즉 **P6의 합성 예측은 predict 시점에 그룹의 측정 통계를 필요로 한다.** 반면 P1의 `s_pred`
(`train_groupkfold.py:106`)는 부스터가 이미 정규화 공간에서 예측하므로 측정 통계가 필요
없다. 두 방법의 비교에서 **P6 쪽이 추가 정보를 받고도 지는** 구조이므로, "direct >
decomposed" 결론은 이 점에서 보수적(P6에 유리하게 기운) 비교다.

## 6. ŷ3 공유 — **P1의 OOF 예측을 파일에서 읽어 그대로 사용. 재학습 없음**

`train_decomposed.py:114-115`:

```python
p1_oof = pd.read_csv(ac.ANALYSIS / f"groupkfold_{platform}_oof.csv", comment="#")
w = w.merge(p1_oof[["combination", "y3_pred"]], on="combination", how="left")
```

- P1이 저장한 **OOF 예측 컬럼 `y3_pred`**를 combination 키로 병합해 그대로 쓴다
  (`:185`에서 `s_pred` 계산에 투입).
- direct predictor **아티팩트(`gkf_cpu_*_y3*.json`)를 로드하지 않고**, 재학습도 하지 않는다.
  P1이 남긴 산출 CSV에 의존한다. (`ac.ARTIFACTS`를 P6는 참조하지 않음.)
- docstring `:19` "y3 = P1's set-level OOF y3 prediction, shared verbatim"과 일치.
- 결과적으로 P1과 P6의 점수 차이는 **y1·y2 항에서만** 발생한다.

## 7. 부록 C 수치의 산출 지점과 재현 여부

| 원고 부록 C 값 (r4 `:277-279`) | 코드 산출 지점 | 아티팩트 값 (GPU / NPU) | 일치 |
|---|---|---|---|
| per-view attainment ρ **0.993 / 0.992** | `:145` `view_stats["oof_spearman_pooled"]` = `ac.spearman(yv, pred)` | 0.99309 / 0.99160 | ✓ |
| 동 MAE **0.017 / 0.020** | `:146` `oof_mae` | 0.017026 / 0.020131 | ✓ |
| y2 from measured attainments ρ **0.834 / 0.763** | `:172` `recon["y2_raw_spearman"]` = `spearman(miss_meas, y2_raw)` | 0.83435 / 0.76300 | ✓ |
| 동 MAE **0.149 / 0.169** | `:171` `recon["y2_raw_mae"]` | 0.148923 / 0.168677 | ✓ |
| composed 그룹 ρ **0.931 / 0.911** | `:186` `ranking["group_spearman_mean"]` | 0.93105 / 0.91082 | ✓ |
| composed Top-1 **0.911 / 0.622** | `:186` `ranking["top1"]` | 0.91111 / 0.62222 | ✓ |
| Top-5 **1.000** (본문 `:264`) | `:186` `ranking["top5"]` | 1.0 / 1.0 | ✓ |
| y1 상대오차 **0** (본문 `:264`) | `:176-178` `y1_raw_relerr_mean` | 1.34e-17 / 9.49e-18 | ✓ (부동소수 영) |

출처 아티팩트: `analysis/decomposed_metrics.json`. Table 1(`:166`)의 `.931 .911 / .911 .622`도
동일 값이다.

**재현(재실행) 여부**: **실행하지 않았다.** `train_decomposed.py`는 실행 시
`analysis/decomposed_{metrics.json,results.md,view_oof.csv}`를 덮어쓰므로, 본 작업의
"수정 금지" 범위에 저촉된다. 대신 위와 같이 **각 수치가 어느 줄에서 나오는지와 커밋된
아티팩트·원고 값의 일치**를 확인했다.

재현 가능성 자체는 높다고 판단한다: fold는 P1 JSON에서 읽는 고정값이고(`:100-112`),
XGBoost `seed=42`, 그리드 탐색도 결정적이며, 무작위 요소가 코드에 없다. 다만 이는 코드
검토에 근거한 판단이며 **실측 재현으로 확인한 것은 아니다.** (재현이 필요하면 출력
경로를 분리한 별도 실행이 필요하고, 이는 신규 스크립트 작성이라 본 작업 범위 밖이다.)

## 8. 원고 서술과 코드의 불일치

원고 r4 읽기 전용 대조. 수치 불일치는 **0건**이고(§7), 아래는 모두 **서술·정의 수준**이다.

| # | 원고 위치 | 서술 | 코드 실제 | 성격 |
|---|---|---|---|---|
| 1 | `:124` §3 | "predicts the degradation of **each task separately**" | **전 vision 모델 공용 단일 회귀기** 1개(fold당). 모델별 회귀기가 아니며 모델 one-hot도 없다 | **오독 유발 가능** — "task별로 예측"은 per-task *모델*로 읽히기 쉽다. "per-instance 단위의 target을 예측" 정도가 정확 |
| 2 | `:124` §3 | "The instance level estimates are **combined** to obtain ŷ1 and ŷ2" | 합성이 per-instance 추정만으로 끝나지 않는다. ŷ1은 그룹 **측정** Fmax로 나누고, ŷ2는 그룹 **측정** y2 lo/hi로 min-max한다 (`:164-167`) | **중요 누락** — P6는 predict 시점에 측정 그룹 통계를 요구하는데 P1은 아니다. 비교 조건의 비대칭(§5) |
| 3 | `:198` 관련연구 | "our decomposed baseline **ports their structure**" | 이식본은 co-runner 요약 6개 feature를 포함한다(§3). 순수 per-instance 정보만 쓰는 구조가 아니다 | **검증 불가 + 정보 누락** — Kim 원 논문이 co-runner feature를 쓰는지 이 저장소에서 확인 불가. 최소한 "co-runner 요약을 포함해 이식"임을 밝히는 편이 안전 |
| 4 | `:192` 본문 / `:277` 부록 C | attainment ρ "0.993/0.992" | `oof_spearman_pooled` = **pooled** ρ (`:145`) | **지표 관행 불일치** — 같은 문단의 Table 1 `.931/.911`은 **group-mean** ρ다. 한 문단에서 두 종류 Spearman이 구분 없이 쓰인다 |
| 5 | `:278` 부록 C | y2 재구성 ρ "0.834/0.763" | `y2_raw_spearman` = **pooled** (`:172`). 코드는 group-mean도 계산하나(`:173-175`, **0.877/0.720**) 원고에 없다 | 동상. 논문의 랭킹 지표가 그룹 단위인 점을 감안하면 group-mean 병기가 일관적 |
| 6 | `:124` §3 | attainment 정의에 상한 언급 없음 | target은 `min(1.0, ratio)`로 상한 클리핑(`:88`) | **경미** — 클리핑 실제 발동 0건이라 수치 무영향 |
| 7 | `:264` 부록 C | "composing y2 … already loses substantial fidelity"만 서술 | 손실의 구체적 원인이 코드에 명시돼 있다: 비가중 평균 + **생성 뷰 제외** + 요청 수 가중 미반영 (`:236-237`) | 설명 부족(오류 아님) — 부록 C에 한 구절 넣으면 "composition error"의 정체가 분명해진다 |

**y3 관련(`:124` "The generative target ŷ3 uses the same predictor as the direct model")은
일치**로 판정한다 — 코드는 direct의 OOF 산출값을 그대로 재사용하므로 값이 동일하다(§6).

---

## 판단에 사용한 파일 (전부 읽기만)

| 파일 | 용도 |
|---|---|
| `xgboost_model/full_collection_540/scripts/train_decomposed.py` | 주 대상 |
| `xgboost_model/full_collection_540/scripts/analysis_common.py` | `cv_select_params`, `train_booster`, `predict_booster`, `ranking_metrics`, `scores_from`, `N_FOLDS`, `STATIC_JSON` |
| `xgboost_model/deploy_selector_xgb_suite.py` | `_view_features`(`:250`), `_device_static`(`:126`), `load_static_profiles`(`:101`), `_PARAMS`/`_PARAM_GRID`(`:519-530`) |
| `xgboost_model/full_collection_540/analysis/decomposed_metrics.json` | §7 값 대조 |
| `xgboost_model/full_collection_540/analysis/groupkfold_{gpu,npu}_metrics.json`, `groupkfold_{gpu,npu}_oof.csv` | fold 매핑·`y3_pred` 공유 경로 확인 |
| `manuscript/mlforsys_main.tex` (r4) | §8 대조 |

추가로 필요한 파일은 없다. 다만 §8-3(Kim 원 논문의 feature 구조)은 저장소 밖 자료라
이 저장소만으로는 판정 불가이며, 이는 파일 부족이 아니라 범위 밖 사안이다.

## UNKNOWN

1. **Kim et al. (ICCD 2024) 원 모델이 co-runner 요약 feature를 쓰는지** — 원문 미보유로
   "ports their structure" 주장의 충실도를 판정할 수 없다. §8-3.
2. **§7 수치의 실행 재현** — 아티팩트 덮어쓰기 금지로 재실행하지 않았다. 코드상 결정적
   이라고 판단하나 실측 확인은 아니다.
