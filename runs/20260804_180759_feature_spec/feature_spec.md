# direct 회귀 모델 feature 명세 (37개)

- 작성: 2026-08-04
- **읽기 전용.** 코드·데이터·아티팩트·원고 미수정.
- 정본 소스: `xgboost_model/deploy_selector_xgb_suite.py`
  → `_view_features()` (`:250-279`) → `_aggregate()` (`:282-294`) → `featurize_window()` (`:296-361`)
- 교차 검증 아티팩트: `xgboost_model/artifacts/*_features.json` (6개 파일)
- 대조 원고 리비전: `manuscript/mlforsys_main.tex` r4

## 구조 요약

```
_view_features(model, dev, infps, S)        → 뷰당 12개 feature
_aggregate(per_view_rows)                   → 각 12개를 sum / mean / max 로 집계 = 36
                                            + views.count.views                  =  1
                                            ────────────────────────────────────────
                                                                          합계  =  37
```

`featurize_window()`는 창의 각 뷰에 `_view_features`를 적용해 12열 행을 쌓고(`:320`),
그 행렬을 `_aggregate`에 넘긴다(`:331`). feature 이름은
`views.{sum|mean|max}.{per-view 키}` 형식으로 `_aggregate:288`이 생성한다.

## 검증 결과 (요약)

| # | 항목 | 결과 |
|---|---|---|
| 1 | 총 개수 = 37 | **정확히 37** — 원고 수치 옳음 |
| 2 | CPU-GPU / CPU-NPU feature 목록 동일 | **완전 동일** (이름·순서·정의) |
| 3 | 아티팩트 목록 == 코드 산출 | **이름·순서까지 일치** |
| 4 | 전 feature가 plan 수준 | **예 — 측정값 참조 0건** |

## 37행 명세표

| # | feature_name | 정의 | 단위/타입 | 범주 |
|---|---|---|---|---|
| 1 | `views.sum.view.infps` | 창 내 전 뷰 합: 스케줄이 이 뷰에 지정한 초당 요청 수 (baseline_rate x rate_factor) | fps (float) | 요청률 |
| 2 | `views.sum.view.exec_cpu` | 창 내 전 뷰 합: 이 뷰가 CPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 3 | `views.sum.view.exec_gpu` | 창 내 전 뷰 합: 이 뷰가 GPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 4 | `views.sum.view.exec_npu` | 창 내 전 뷰 합: 이 뷰가 NPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 5 | `views.sum.view.is_vision` | 창 내 전 뷰 합: _model_kind(model)=='vision'이면 1 | 0/1 지시자 | 모델 유형 |
| 6 | `views.sum.view.is_llm` | 창 내 전 뷰 합: _model_kind(model)이 llm/vlm이면 1 | 0/1 지시자 | 모델 유형 |
| 7 | `views.sum.view.static_infer_sel` | 창 내 전 뷰 합: 배치된 디바이스의 단독 추론 지연 (_device_static의 infer). 결측시 0 | ms (float) | 단독 프로파일 |
| 8 | `views.sum.view.static_load_sel` | 창 내 전 뷰 합: 배치된 디바이스의 단독 모델 로드 시간 (load). 결측시 0 | ms (float) | 단독 프로파일 |
| 9 | `views.sum.view.static_tokens_sel` | 창 내 전 뷰 합: 배치된 디바이스의 단독 토큰 생성률 (tokens). 결측시 0 | tok/s (float) | 단독 프로파일 |
| 10 | `views.sum.view.capacity_fps` | 창 내 전 뷰 합: 1000 / static_infer_sel (static_infer_sel>0일 때, 아니면 0) | fps (float) | 단독 프로파일 |
| 11 | `views.sum.view.load_factor` | 창 내 전 뷰 합: view.infps / view.capacity_fps (capacity>0일 때, 아니면 0) | 비율 (float) | 집계 부하 |
| 12 | `views.sum.x.infps__static_infer_sel` | 창 내 전 뷰 합: view.infps x view.static_infer_sel 상호작용항 | fps·ms (float) | 기타 |
| 13 | `views.mean.view.infps` | 창 내 전 뷰 평균: 스케줄이 이 뷰에 지정한 초당 요청 수 (baseline_rate x rate_factor) | fps (float) | 요청률 |
| 14 | `views.mean.view.exec_cpu` | 창 내 전 뷰 평균: 이 뷰가 CPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 15 | `views.mean.view.exec_gpu` | 창 내 전 뷰 평균: 이 뷰가 GPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 16 | `views.mean.view.exec_npu` | 창 내 전 뷰 평균: 이 뷰가 NPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 17 | `views.mean.view.is_vision` | 창 내 전 뷰 평균: _model_kind(model)=='vision'이면 1 | 0/1 지시자 | 모델 유형 |
| 18 | `views.mean.view.is_llm` | 창 내 전 뷰 평균: _model_kind(model)이 llm/vlm이면 1 | 0/1 지시자 | 모델 유형 |
| 19 | `views.mean.view.static_infer_sel` | 창 내 전 뷰 평균: 배치된 디바이스의 단독 추론 지연 (_device_static의 infer). 결측시 0 | ms (float) | 단독 프로파일 |
| 20 | `views.mean.view.static_load_sel` | 창 내 전 뷰 평균: 배치된 디바이스의 단독 모델 로드 시간 (load). 결측시 0 | ms (float) | 단독 프로파일 |
| 21 | `views.mean.view.static_tokens_sel` | 창 내 전 뷰 평균: 배치된 디바이스의 단독 토큰 생성률 (tokens). 결측시 0 | tok/s (float) | 단독 프로파일 |
| 22 | `views.mean.view.capacity_fps` | 창 내 전 뷰 평균: 1000 / static_infer_sel (static_infer_sel>0일 때, 아니면 0) | fps (float) | 단독 프로파일 |
| 23 | `views.mean.view.load_factor` | 창 내 전 뷰 평균: view.infps / view.capacity_fps (capacity>0일 때, 아니면 0) | 비율 (float) | 집계 부하 |
| 24 | `views.mean.x.infps__static_infer_sel` | 창 내 전 뷰 평균: view.infps x view.static_infer_sel 상호작용항 | fps·ms (float) | 기타 |
| 25 | `views.max.view.infps` | 창 내 전 뷰 최대: 스케줄이 이 뷰에 지정한 초당 요청 수 (baseline_rate x rate_factor) | fps (float) | 요청률 |
| 26 | `views.max.view.exec_cpu` | 창 내 전 뷰 최대: 이 뷰가 CPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 27 | `views.max.view.exec_gpu` | 창 내 전 뷰 최대: 이 뷰가 GPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 28 | `views.max.view.exec_npu` | 창 내 전 뷰 최대: 이 뷰가 NPU에 배치되면 1, 아니면 0 | 0/1 지시자 | 디바이스 배치 |
| 29 | `views.max.view.is_vision` | 창 내 전 뷰 최대: _model_kind(model)=='vision'이면 1 | 0/1 지시자 | 모델 유형 |
| 30 | `views.max.view.is_llm` | 창 내 전 뷰 최대: _model_kind(model)이 llm/vlm이면 1 | 0/1 지시자 | 모델 유형 |
| 31 | `views.max.view.static_infer_sel` | 창 내 전 뷰 최대: 배치된 디바이스의 단독 추론 지연 (_device_static의 infer). 결측시 0 | ms (float) | 단독 프로파일 |
| 32 | `views.max.view.static_load_sel` | 창 내 전 뷰 최대: 배치된 디바이스의 단독 모델 로드 시간 (load). 결측시 0 | ms (float) | 단독 프로파일 |
| 33 | `views.max.view.static_tokens_sel` | 창 내 전 뷰 최대: 배치된 디바이스의 단독 토큰 생성률 (tokens). 결측시 0 | tok/s (float) | 단독 프로파일 |
| 34 | `views.max.view.capacity_fps` | 창 내 전 뷰 최대: 1000 / static_infer_sel (static_infer_sel>0일 때, 아니면 0) | fps (float) | 단독 프로파일 |
| 35 | `views.max.view.load_factor` | 창 내 전 뷰 최대: view.infps / view.capacity_fps (capacity>0일 때, 아니면 0) | 비율 (float) | 집계 부하 |
| 36 | `views.max.x.infps__static_infer_sel` | 창 내 전 뷰 최대: view.infps x view.static_infer_sel 상호작용항 | fps·ms (float) | 기타 |
| 37 | `views.count.views` | 창의 뷰(동시 인스턴스) 개수 = N. _aggregate가 len(df)로 산출 | 개수 (float) | 기타 |

### 범주별 분포

| 범주 | 개수 | 내역 |
|---|---|---|
| 단독 프로파일 | 12 | `static_infer_sel`, `static_load_sel`, `static_tokens_sel`, `capacity_fps` × {sum, mean, max} |
| 디바이스 배치 | 9 | `exec_cpu`, `exec_gpu`, `exec_npu` × {sum, mean, max} |
| 모델 유형 | 6 | `is_vision`, `is_llm` × {sum, mean, max} |
| **기타** | **4** | `x.infps__static_infer_sel` × {sum, mean, max} + `views.count.views` |
| 요청률 | 3 | `infps` × {sum, mean, max} |
| 집계 부하 | 3 | `load_factor` × {sum, mean, max} |

**"기타" 4개에 대한 보고** — 원고 §3(`:160`)의 5분류
("request rates, model types, device assignments, isolated device profiles, and
aggregate load")에 억지로 배정하지 않고 남겨 둔 항목이다.

- `views.{sum,mean,max}.x.infps__static_infer_sel` (3개): **요청률 × 단독 프로파일의
  상호작용항**이다(`_view_features:278`). 두 범주를 곱한 파생량이라 어느 한쪽으로
  분류할 수 없다. 5분류에 넣으려면 "요청률"이나 "단독 프로파일" 중 하나로 밀어야
  하는데, 둘 다 부정확하다.
- `views.count.views` (1개): **창의 동시 인스턴스 수(N)**다(`_aggregate:292`).
  "aggregate load"로 읽을 여지가 있으나 부하량이 아니라 개수이므로 보류했다.

즉 5분류로 서술하려면 33개는 그대로 들어가고 4개가 남는다. 원고가 5분류를 유지하려면
"and their pairwise interaction and the instance count" 정도의 보강이 필요하고,
분류를 늘리지 않으려면 §3 문장을 "features summarizing ... , aggregate load, and
derived interaction terms" 식으로 넓히는 선택지가 있다. **어느 쪽을 택할지는 원고 쪽
결정이며 본 작업 범위 밖이다.**

---

## 검증 상세

### 1. 총 개수 — **37 확정. 원고 수치가 옳다**

세 경로가 모두 37로 일치한다.

| 근거 | 값 |
|---|---|
| 코드 산출 (`_view_features` 12키 × 3집계 + count) | **37** |
| 학습 아티팩트 `deploy_cpu_gpu_features.json` 등 6개 | **37** |
| `analysis_common.py:FEATURE_ORDER` (아티팩트 로드) | **37** |

"37"이라는 수치가 코드에 상수로 하드코딩된 곳은 없다 — `_view_features`가 반환하는
dict 키 개수(12)와 `_aggregate`의 집계 3종에서 **파생**된다. 따라서 뷰 feature를 하나
추가하면 자동으로 40이 되는 구조이고, 현재 상태에서 37이 맞다.

원고 `:89` "37-dimensional feature representation" — **일치. 수정 불필요.**

### 2. 두 플랫폼 예측기의 feature 목록 — **완전 동일**

`xgboost_model/artifacts/` 아래 feature 목록 파일 6개
(`deploy_cpu_gpu`, `deploy_cpu_npu`, `deploy_conc_cpu_gpu`, `deploy_xgb`,
`gkf_cpu_gpu`, `gkf_cpu_npu`)가 **전부 바이트 동일**하다
(sha256 `79e34d26dbfe…` 공통). 이름·순서·개수 차이 0건.

플랫폼 차이는 feature **목록**이 아니라 feature **값**에서만 나타난다: `exec_gpu`/`exec_npu`
지시자 중 어느 쪽이 켜지는지, 그리고 `static_*`가 어느 디바이스 프로파일에서 오는지
(`_device_static(model, dev, S)`). 즉 원고의 "37-dimensional feature representation are
held fixed"(`:89`) 통제 주장은 **성립한다**.

### 3. 아티팩트 vs 코드 — **이름·순서까지 일치**

`_aggregate`를 직접 호출해 생성한 이름 리스트와 `deploy_cpu_gpu_features.json`을
비교한 결과:

- 집합 일치: **True**
- **순서까지 일치: True** (sum 12 → mean 12 → max 12 → count 1)

`analysis_common.py:FEATURE_ORDER`가 이 아티팩트 파일을 정본으로 읽어
(`:38-40` 주석 "byte-identical across all six shipped artifacts") 플랫폼 간 열 순서를
고정하는 구조와도 정합적이다.

### 4. plan 수준 여부 — **37개 전부 plan 수준. 측정값 참조 0건**

`_view_features`의 입력은 `(model, exec_dev, infps, S)` 넷뿐이다:

- `model`, `exec_dev` — 배치 계획에서 옴
- `infps` — 수집 스케줄 YAML의 지정 요청률(`_build_infps_lookup`), 측정된 달성 처리량이 아님
- `S` — 정적 단독 프로파일 JSON

`featurize_window`는 창에서 `throughput_fps`·`tokens_per_s`·`deadline_miss_rate`를 읽지만
그것들은 **전부 target(y1/y2/y3) 산출에만** 쓰이고 `X`에는 들어가지 않는다(`:320-330`
vs `:331`). 코드 주석도 명시한다 (`_view_features:262-264`):

> only static-profile + plan features are used … Measured per-view dynamics are NOT
> features — they are unavailable at predict time and using them leaks the target.

→ 원고의 measurement-free 주장 성립 조건을 만족한다.

**단, 별개의 알려진 결함 1건**: 출하된 `deploy_cpu_{gpu,npu}` 아티팩트는 `infps` 계열
feature가 **전부 0인 채로 학습**되어 있다(full540 창에 schedule 힌트가 없어
`build_dataset`의 조회가 실패). 예측 경로는 실제 infps를 넣으므로 train/serve 불일치다.
이는 `SUMMARY.md` 이상발견 1과 `analysis_common.py:14-20` docstring에 이미 기록돼 있고,
P1~P6 실험은 실제 infps를 복원해 사용했다. **feature 명세(무엇을 쓰는가)의 문제가 아니라
학습 입력값(어떤 값이 들어갔는가)의 문제**이므로 §4 판정을 바꾸지 않는다.

### 5. decomposed 18-feature와의 관계

| 구분 | 개수 | 항목 |
|---|---|---|
| **공통 기반량** (direct는 sum/mean/max로 집계, decomposed는 뷰별 원값 그대로) | 12 | `view.infps`, `view.exec_{cpu,gpu,npu}`, `view.is_vision`, `view.is_llm`, `view.static_{infer,load,tokens}_sel`, `view.capacity_fps`, `view.load_factor`, `x.infps__static_infer_sel` |
| **decomposed 전용** | 6 | `co.same_dev_load_sum`, `co.same_dev_count`, `co.other_dev_load_sum`, `co.other_dev_count`, `co.set_has_gen`, `co.gen_on_accel` |
| **direct 전용** | 1 | `views.count.views` |

- 두 모델은 **동일한 `_view_features` 12개를 공유**하되, direct는 창 단위로 3종 집계해
  36개로 펼치고, decomposed는 뷰 단위 원값 12개를 그대로 쓴다.
- decomposed의 `co.*` 6개는 direct에 대응 항목이 없다. 다만 direct의 sum/mean/max 집계가
  같은 정보를 **창 전체 수준에서** 이미 담고 있다는 점에서 역할이 겹친다 — decomposed는
  "뷰 i에서 본 나머지"를, direct는 "창 전체"를 요약한다.
- direct의 `views.count.views`(=N)는 decomposed에 없으나,
  `co.same_dev_count + co.other_dev_count + 1 = N`이므로 정보로는 복원 가능하다.

---

## 판단에 사용한 파일 (전부 읽기만)

| 파일 | 용도 |
|---|---|
| `xgboost_model/deploy_selector_xgb_suite.py` | `_view_features`(`:250`), `_aggregate`(`:282`), `featurize_window`(`:296`), `_device_static`(`:126`) |
| `xgboost_model/artifacts/*_features.json` (6개) | 아티팩트 feature 목록 교차 검증 |
| `xgboost_model/full_collection_540/scripts/analysis_common.py` | `FEATURE_ORDER` 정본 경로 |
| `xgboost_model/full_collection_540/scripts/train_decomposed.py` | `VIEW_FEATURES` 18개 (§5) |
| `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json` | 정적 프로파일 필드 확인 |
| `manuscript/mlforsys_main.tex` (r4) | `:89` 37차원 주장, `:160` 5분류 서술 |

## UNKNOWN

없음. 37개 전 항목의 이름·정의·단위·범주가 코드에서 확정되었고, 검증 1–4는 모두
코드·아티팩트 대조로 결론이 났다. §범주별 분포의 "기타 4개"는 미확정이 아니라
**원고 5분류에 대응시키지 않기로 한 판단**이며 사유를 명기했다.
