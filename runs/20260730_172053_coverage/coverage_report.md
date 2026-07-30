# 부록 샘플링 커버리지 — 수치 추출 및 검증 A~E

- 작성: 2026-07-30
- 스크립트: `runs/20260730_172053_coverage/build_coverage.py` (신규, 읽기 전용 분석)
- 산출물: `appendix_coverage.csv` (15행), `coverage_facts.json` (전 수치 기계 판독본)
- 입력: 이미 수집된 아티팩트만. 재수집·재실험 없음. 원고(.tex)·기존 아티팩트 미수정.

## 0. 탐색 결과 — 실제 소스 경로

지시된 `xgboost_model/artifacts/cpu_gpu`, `.../cpu_npu` **디렉토리는 존재하지 않는다.**
`artifacts/` 는 플랫폼별 하위 디렉토리 없이 `deploy_cpu_{gpu,npu}_{y1,y2,y3,coverage,features}.json`
평면 파일 구조다. 실제로 각 수치를 얻은 경로는 아래와 같다.

| 수치 | 소스 경로 | 필드 |
|---|---|---|
| 세트 정의, N, 층화 대상/타깃 | `working_sets.yaml` | `sets.*`, `stratified_sets`, `stratified_target_by_free` |
| VLM CPU 불가 제약 | `model_registry.py:45-47` | `DEVICE_CONSTRAINTS = {"qwen2_vl": ["gpu","npu"]}` |
| 후보 생성 로직 / 층화 축 | `xgboost_model/gen_collection_schedules.py` | `free_models()`, `all_placements()`, `stratify()`, `combos_for()` |
| 스케줄된 배치 (플랫폼별) | `xgboost_model/schedules/collection/collect_cpu_{gpu,npu}.yaml` + `.meta.json` | `<combo>.<model>_<dev>.execution`; meta의 `set`/`rate_factor`/`stratum`/`n_accel` |
| 측정된 창 (플랫폼별 540) | `xgboost_model/full_collection_540/cpu_{gpu,npu}/performance_{gpu,npu}_full540.json` | `[].models[].execution`, `[].rate_factor`, `[].total.*` |
| 점수 정의 α/β | `xgboost_model/deploy_selector_xgb_suite.py:681` `score_combo`; `full_collection_540/scripts/analysis_common.py` `ALPHA=0.3, BETA=1.0` | — |
| 대조용 기존 divergence 수치 | `xgboost_model/full_collection_540/analysis/platform_divergence.md` | β=1.0 → 29/45, β=0.5 → 30/45 |

확인했으나 사용하지 않은 경로 (사유 명기):

- `xgboost_model/performance_data/cpu_{gpu,npu}/performance.json` — **160창 파일럿**
  (2026-07-09~10). `full_collection_540/README.md`가 명시적으로 "현행 예측기는 이 데이터가
  아니라 540창으로 학습됐다"고 경고. 부록 커버리지에 부적합하여 제외.
- `xgboost_model/artifacts/deploy_cpu_{gpu,npu}_coverage.json` — `rows: 540`, `model_sets`
  15개, `rate_factors` 5개 값만 담긴 요약. **세트별 n_measured가 없어** 부록 표를 만들 수
  없다. 총계 대조(§D)에만 참조.
- `xgboost_model/full_collection_540/cpu_{gpu,npu}/collect_{gpu,npu}_results.jsonl` (540행)
  — 창 파일과 동일 내용의 수집 요약. 교차 확인용으로만 열람.
- 원고 `.tex`: **이 저장소에 존재하지 않는다** (`find . -name "*.tex"` → 0건). §C의 tie 규칙은
  원고 대조가 아니라 코드베이스 규약으로 확정했다 — 상세는 §C 참조.

---

## A. 플랫폼 간 후보집합 동일성

**결론: 전 15세트에서 대칭차 = 0. 논문의 divergence 주장 성립 조건을 만족한다.**

비교 방법: 각 측정 창의 배치를 플랫폼 중립 벡터로 정규화했다 —
`(model, 'cpu'|'accel')`를 모델명으로 정렬한 튜플. 가속기 토큰(`GPU`/`NPU`)을 `accel`로
접었기 때문에, 두 플랫폼의 배치 집합이 직접 비교 가능하다 (접지 않으면 모든 배치가
자동으로 다르게 나와 비교 자체가 무의미해진다).

| 검증 층위 | 결과 |
|---|---|
| 세트 단위 (15세트) 측정 배치 집합 대칭차 | **전부 0** |
| (세트, rate) 단위 (45그룹) 측정 배치 집합 대칭차 | **전부 0** |
| 스케줄 yaml 단위 (45그룹) 대칭차 | **전부 0** |
| 측정 집합 == 스케줄 집합 (플랫폼별) | GPU **일치**, NPU **일치** (불일치 그룹 0) |

즉 동일성은 사후적으로 우연히 성립한 것이 아니라, 생성 시점(`gen_collection_schedules.build()`가
두 플랫폼에 같은 `combos_for()` 결과를 사용)에 보장되고, 수집 단계에서도 누락 없이
그대로 실현되었다. 대칭차가 0이 아닌 세트는 없으므로 나열할 placement도 없다.

부수 확인: (세트, rate) 안에서 같은 배치가 두 번 측정된 중복은 양 플랫폼 모두 0건.

## B. feasible 수 정의 검증

### B-1. VLM CPU 배치 불가 제약이 실제로 후보 생성 시 적용됐는가 — **적용됨**

코드 경로가 확정된다:

1. `model_registry.py:45` `DEVICE_CONSTRAINTS = {"qwen2_vl": ["gpu", "npu"]}` —
   `allowed_devices()`가 이 dict를 우선 조회한다.
2. `gen_collection_schedules.py:free_models()` = `[m for m in models if "cpu" in
   reg.allowed_devices(m)]` → `qwen2_vl`은 자유 축에서 **배제**.
3. 같은 파일 `all_placements()`: 자유 모델만 `itertools.product(["cpu","accel"])`로 전개하고,
   비자유 모델(`fixed`)은 `p[m] = "accel"`로 **가속기에 고정**.

데이터 측 검증: 측정된 540창 × 2플랫폼에서 `qwen2_vl`이 CPU에 배치된 행은 **GPU 0건,
NPU 0건**. (`CPU_PLACEMENT_COVERAGE.md`의 "qwen2_vl CPU 배치 위반 0건"과 일치.)

### B-2. `n_feasible == 2^(N-1)` 이 성립하는가 — **전 세트 성립하지 않음. VLM 포함 세트에서만 성립한다.**

성립 여부가 정확히 `qwen2_vl` 포함 여부와 일치한다 (`B_pow2_holds_iff_has_vlm = true`).
일반식은 **`n_feasible = 2^(#free) = 2^(N - #VLM)`** 이다.

| 세트 | N | 2^N | 2^(N-1) | n_feasible | 2^(N-1) 성립 | qwen2_vl |
|---|---|---|---|---|---|---|
| S1 | 2 | 4 | 2 | 4 | ✗ | 없음 |
| S2 | 3 | 8 | 4 | 8 | ✗ | 없음 |
| S3 | 3 | 8 | 4 | 8 | ✗ | 없음 |
| S4 | 3 | 8 | 4 | 4 | ✓ | 포함 |
| base1–base5 | 4 | 16 | 8 | 8 | ✓ | 포함 |
| S5 | 5 | 32 | 16 | 16 | ✓ | 포함 |
| S6 | 6 | 64 | 32 | 32 | ✓ | 포함 |
| S7 | 7 | 128 | 64 | 64 | ✓ | 포함 |
| S9 | 5 | 32 | 16 | 32 | ✗ | 없음 |
| S8 | 8 | 256 | 128 | 128 | ✓ | 포함 |
| S10 | 7 | 128 | 64 | 128 | ✗ | 없음 |

성립하지 않는 5세트(S1, S2, S3, S9, S10)의 사유는 **추가 제약이 아니라 제약의 부재**다.
이들은 `qwen2_vl`을 포함하지 않으므로 모든 N개 모델이 자유 배치이고, 후보 수가 감소하지
않아 `2^N`이 된다. S3는 `llama1b`를 포함하지만 `llama1b`에는 디바이스 제약이 없다
(`DEVICE_CONSTRAINTS`에 `qwen2_vl` 단일 항목뿐).

**디바이스 메모리 등 추가 제약은 후보 생성 코드에 존재하지 않는다.** `all_placements()`는
`DEVICE_CONSTRAINTS` 외 어떤 필터도 적용하지 않으며, 메모리·용량 조건을 참조하는 코드
경로가 없다. 즉 `n_feasible`의 감소 요인은 VLM 제약 하나뿐이고, `n_measured < n_feasible`
인 세트의 축소는 제약이 아니라 **층화 샘플링** 때문이다(§B-2와 별개 축).

### B-3. 세트 내 3개 rate level이 동일한 placement 집합을 공유하는가 — **공유한다**

전 15세트 × 2플랫폼 = 30개 조합 모두 `identical = true`, 각 세트의 rate 수는 3개
(`rate_factors_by_size`로 크기별 3개씩: ≤2모델 `[2,3,4]×`, 3–5모델 `[1,2,3]×`,
≥6모델 `[1,1.5,2]×`). 코드 근거: `gen_collection_schedules.build()`가
`for rate in rate_for(n): for p, tag in zip(places, tags)` 로 **동일한 `places` 리스트를
모든 rate에 재사용**한다.

**→ 부록 표는 세트 단위 15행으로 쓸 수 있다.** 산출물 1(`appendix_coverage.csv`)을 15행으로
확정했고, 45행 `(set, rate)` 표는 필요하지 않아 생성하지 않았다 (스크립트는 B-3이 깨질
경우에만 `appendix_coverage_by_rate.csv`를 자동 생성하도록 되어 있다).

## C. exhaustive 그룹 한정 divergence 재계산

### tie 처리 규칙 (본 분석에서 사용)

`full_collection_540/scripts/analysis_common.py:ranking_metrics`의 규약을 그대로 쓴다:

> 그룹의 measured-best는 **그룹 최대 점수와의 차가 `1e-9` 미만인 모든 배치의 집합**
> (tie set)이다. 한 그룹은 두 플랫폼의 tie set이 **교집합을 가지면 "일치"**,
> 공집합이면 "불일치"로 센다.

원고 `.tex`가 이 저장소에 없어 문면 대조는 불가능하므로, 대신 **동일 규칙을 45그룹 전체에
적용해 기존 발표 수치를 재현하는지** 확인했다: β=1.0 → 29/45 불일치, β=0.5 → 30/45 불일치.
`analysis/platform_divergence.md`의 값과 **정확히 일치**한다. 따라서 본 분석의 tie 규칙은
논문 수치를 낳은 규칙과 같은 결과를 준다.

(참고: 기존 `compare_platforms_v2.py`는 `max(...)`의 first-max를 쓰는 tie-미인식 구현이다.
아래 exhaustive 부분집합에서는 **tie가 발생한 그룹이 0개**여서 두 규칙의 결과가 같고,
45그룹 전체에서도 재현 결과가 동일하다.)

점수: `S = y1 − 0.3·y2 (+ β·y3, 생성 모델 포함 세트만)`, **원시 측정 창 총계** 기준
(`score_combo`, 예측기 미사용).

### 결과 — exhaustive 그룹만 (mode == exhaustive)

exhaustive 세트 10개 = S1, S2, S3, S4, base1–base5, S5 → **30 그룹** (10세트 × 3 rate).

| 기준 | β = 1.0 | β = 0.5 |
|---|---|---|
| **불일치 그룹 / exhaustive 총수** | **22 / 30 (73.3%)** | **22 / 30 (73.3%)** |
| 생성 모델 포함 그룹 불일치 / 분모 | **22 / 24 (91.7%)** | **22 / 24 (91.7%)** |
| vision-only 그룹 불일치 / 분모 | **0 / 6 (0.0%)** | **0 / 6 (0.0%)** |
| tie 발생 그룹 수 | 0 | 0 |

β=1.0과 β=0.5가 완전히 동일하다. 두 β에서 불일치 그룹 집합 자체가 같다
(β 민감 그룹은 45그룹 중 S8@1.5 하나뿐이고, S8은 sampled이므로 이 부분집합에 없다).

세트별 불일치 그룹 수 (양 β 공통): S3 3/3, S4 1/3, S5 3/3, base1 3/3, base2 3/3,
base3 3/3, base4 3/3, base5 3/3. vision-only인 S1(0/3), S2(0/3)는 전 rate에서 일치.

분모 내역: 생성 모델 포함 exhaustive 세트 8개 = S3, S4, S5, base1–base5 → 24그룹.
vision-only exhaustive 세트 2개 = S1, S2 → 6그룹.

전형적 불일치 형태 (base1 @rate1.0): GPU 최적은 `llama1b→accel`, NPU 최적은
`llama1b→cpu` (나머지 3모델은 양쪽 모두 accel). 생성 모델의 가속기 적합도가 플랫폼별로
갈리는 것이 불일치의 주된 축이다.

sampled 그룹(15그룹)의 몫은 45그룹 전체 − exhaustive = β=1.0에서 7/15, β=0.5에서 8/15로,
exhaustive 그룹의 불일치율(73.3%)이 전체(64.4%)보다 **높다**. exhaustive로 한정하는 것이
divergence 주장을 약화시키지 않는다.

## D. 총계 정합성

**결론: 플랫폼당 180 placements / 540 windows / 45 groups 전부 일치. 불일치 없음.**

| 항목 | 기대 | GPU | NPU |
|---|---|---|---|
| 세트별 n_measured 합 (placements) | 180 | **180** ✓ | **180** ✓ |
| 창 수 (windows) | 540 | **540** ✓ | **540** ✓ |
| (세트, rate) 그룹 수 | 45 | **45** ✓ | **45** ✓ |
| 세트 매핑 실패 행 | 0 | **0** ✓ | **0** ✓ |

세트별 내역 (GPU/NPU 동일): S1 4, S2 8, S3 8, S4 4, base1–base5 각 8 (=40), S5 16,
S6 16, S7 20, S9 16, S8 24, S10 24 → 합 180. 창 수는 각 세트 n_measured × 3 rate이므로
180 × 3 = 540. 그룹은 15세트 × 3 rate = 45.

`artifacts/deploy_cpu_{gpu,npu}_coverage.json`의 `rows: 540`과도 일치한다. 단 이 파일의
`rate_factors`는 5개 값 `[1.0, 1.5, 2.0, 3.0, 4.0]`을 나열하는데, 이는 15세트에 걸쳐 등장하는
rate 값의 **합집합**이며 세트당 rate 수가 5라는 뜻이 아니다 (세트당 3개, §B-3).

## E. vision-only 통제군 상태

**S9와 S10은 둘 다 `sampled`이다 — exhaustive가 아니다.** S9: 32 후보 중 16 측정
(커버리지 50.0%), S10: 128 후보 중 24 측정 (18.75%). 둘 다 `working_sets.yaml`의
`stratified_sets: [S6, S7, S8, S9, S10]`에 포함되어 층화 샘플링 대상이다.

→ §C의 exhaustive 한정 재계산에서 vision-only 분모는 S1, S2의 6그룹뿐이고, 그 6그룹의
불일치는 0이다. vision-only 통제군을 "exhaustive에서 divergence 0"으로 서술할 때 근거가
되는 세트는 **S9/S10이 아니라 S1/S2(2·3모델 소형 세트)** 임을 부록에 정확히 반영해야 한다.

---

## UNKNOWN으로 남은 항목

없음. 산출물 1의 모든 셀과 A~E의 모든 수치가 위 표의 소스에서 확정되었다.

단, 소스에서 확정할 수 없어 **대체 근거로 확정한** 항목 1건을 명시한다:

- **§C tie 규칙의 원고 대조**: 원고 `.tex`가 이 저장소에 없어(0건) 문면 비교는 불가능했다.
  코드베이스 규약(`analysis_common.ranking_metrics`의 `1e-9` tie set)을 채택하고, 그 규칙이
  45그룹 전체에서 발표 수치(29/45, 30/45)를 정확히 재현함을 확인하는 방식으로 등가성을
  확보했다. 원고에 다른 tie 규칙이 문서화되어 있다면 재확인이 필요하다.
