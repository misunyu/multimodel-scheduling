<!--
  작성: 2026-08-06 (작업 10-r3). 근거 실측 스크립트·매니페스트:
  runs/20260806_103943_model_guide/{measure.py, measurements.json, run_manifest.json}
  이 문서의 모든 수치는 저장소의 저장된 데이터·아티팩트에서 실측한 값이다.
  재학습·재수집은 하지 않았다. 확인 불가한 항목은 "미확인"으로 명기했다.
  경로는 전부 저장소 루트 기준 상대경로.
-->

# XGBoost 예측기 3종 — 학습·저장·사용 가이드

논문의 세 예측 모델(**platform-specific** / **joint** / **decomposed**)이 어떤 코드로
학습되고, 어떤 데이터를 어디서 읽고, 산출물이 어디에 저장되며, 어떻게 재학습·사용하는지를
정리한다. 모든 진술에 소스 `파일:줄` 근거를 병기한다.

---

## 1. 개요

| | **platform-specific (P1)** | **joint (P4)** | **decomposed (P6)** |
|---|---|---|---|
| 학습 스크립트 | `xgboost_model/full_collection_540/scripts/train_groupkfold.py` | `.../scripts/train_unified.py` | `.../scripts/train_decomposed.py` |
| 진입 함수 | `main()` `:115` → `run_platform(platform, with_infps=True)` `:32` | `main()` `:29` | `main()` `:93` (+ `build_view_dataset()` `:50`) |
| 예측 단위 | 창(window) = 하나의 (배치, rate) | 창 | **뷰(view)** = 창 안의 모델 1개 |
| 학습 데이터 범위 | 한 플랫폼의 540창 | 두 플랫폼 1080창 합침 (`train_unified.py:35-38`) | 한 플랫폼의 vision 뷰 2220행 |
| 학습 표본 수 | 540행 × 37 feature (플랫폼당) | 1080행 × 37 feature | 2220행 × 18 feature (플랫폼당) |
| CV 그룹 키 | `(models, rate_factor)` → 45그룹 (`analysis_common.py:152-157`) | `(models, rate_factor)` → **45 paired 그룹**, 플랫폼 제외 (`train_unified.py:53`) | 창의 그룹을 뷰가 상속 → 45그룹 (`train_decomposed.py:121`) |
| 부스터 산출물 | `xgboost_model/artifacts/gkf_cpu_{gpu,npu}_{y1,y2,y3}.json` + `_fold{0,1,2}.json` (`train_groupkfold.py:75,82`) | **없음** (`train_unified.py:15-16`) | **없음** (`save_model` 호출 자체가 없음) |
| 지표 산출물 | `analysis/groupkfold_{gpu,npu}_metrics.json`, `groupkfold_{gpu,npu}_oof.csv`, `groupkfold_ranking.md` | `analysis/unified_metrics.json`, `unified_oof.csv`, `unified_vs_specialized.md` | `analysis/decomposed_metrics.json`, `decomposed_results.md`, `decomposed_view_oof.csv` |
| 논문 대응 수치 (점수 Top-1 / 그룹 ρ) | GPU 0.933 / +0.986, NPU 0.867 / +0.972 | GPU 0.933 / +0.975, **NPU 0.867 / +0.984** | GPU 0.911 / +0.931, NPU 0.622 / +0.911 |
| 대응 그림·표 | Fig. transfer의 `platform-specific` 막대 | Fig. transfer의 `joint (both platforms)` 막대 | 본문 decomposed 베이스라인 비교 |

`analysis/`는 `xgboost_model/full_collection_540/analysis/`의 줄임. 위 수치의 원본은
`groupkfold_ranking.md`, `unified_metrics.json:per_platform.*.ranking`,
`decomposed_metrics.json:{gpu,npu}.ranking`.

### 1-1. 세 모델 사이의 두 가지 의존

두 모델이 P1의 산출물을 **읽어서** 쓰기 때문에, P1을 다시 돌리면 P6도 다시 돌려야 한다.

1. **fold 상속 (P6 ← P1)**
   `train_decomposed.py:100-113` 이 `analysis/groupkfold_{platform}_metrics.json`의
   `group_to_fold` 딕셔너리를 읽어 창의 fold를 정하고, 뷰가 창의 fold를 그대로 받는다
   (`:121`). `assert w["fold"].notna().all()` (`:113`)로 모든 그룹이 매핑에 있는지 확인한다.
   → P6는 자체적으로 fold를 다시 나누지 않는다.
   `group_to_fold`는 `train_groupkfold.py:110-111`이 쓴다.
   **joint은 상속하지 않는다**: 1080행에 대해 `assign_group_folds`를 독립적으로 다시
   호출한다 (`train_unified.py:53-54`). 다만 그룹 키가 P1과 같은 `(models, rate_factor)`
   이고 배정이 정렬된 고유 키와 seed만의 함수라, **결과 파티션은 P1과 45/45 동일**하다.
   (2026-08-13 이전에는 키에 플랫폼이 들어가 90그룹이었고 27/45 쌍이 fold 를 가로질렀다.)

2. **ŷ3 공유 (P6 ← P1)**
   뷰별 토큰 수가 보존되지 않아 y3는 분해할 수 없다 (`train_decomposed.py:2-5`).
   그래서 P6는 P1의 **OOF** y3 예측을 그대로 가져온다:
   `p1_oof = pd.read_csv(analysis/groupkfold_{platform}_oof.csv)` 후
   `w.merge(p1_oof[["combination","y3_pred"]], on="combination")` (`:114-115`),
   점수 합성에서 `comp["y3_pred"]`를 그대로 쓴다 (`:185`).
   → P6의 점수는 y1·y2만 분해 모델이고 y3 항은 P1과 동일하다.

---

## 2. 학습·테스트 데이터: 위치와 규모

### 2-A. 데이터 위치

전부 실재를 확인했다(18/18 존재). sha256은 `runs/20260806_103943_model_guide/measurements.json`
에도 기계 판독 형태로 들어 있다.

| # | 경로 | 역할 | 생성 주체 | 소비 모델 | sha256 (앞 16자) / 크기 |
|---|---|---|---|---|---|
| 1 | `xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json` | **CPU-GPU 원시 측정 창 540개** — 학습·평가의 유일한 측정 원본 | `scripts/collect_run.py` → `jsonl_to_windows.py` | P1(gpu), joint, P6(gpu) | `813c915250bc9243` / 666,231 B |
| 2 | `xgboost_model/full_collection_540/cpu_npu/performance_npu_full540.json` | CPU-NPU 원시 측정 창 540개 | 동상 | P1(npu), joint, P6(npu) | `0fbf775ae5b59cf8` / 665,394 B |
| 3 | `xgboost_model/full_collection_540/cpu_gpu/collect_gpu_results.jsonl` | 수집 원본 JSONL (창 변환 전, 조합당 1행) | `scripts/collect_run.py` | 학습 직접 소비 없음 (1의 입력) | `8e320f8fbeba7998` / 179,207 B |
| 4 | `xgboost_model/full_collection_540/cpu_npu/collect_npu_results.jsonl` | 동상 | 동상 | 학습 직접 소비 없음 (2의 입력) | `1476805e42f09104` / 178,533 B |
| 5 | `xgboost_model/full_collection_540/cpu_gpu/collect_gpu_anchor.jsonl` | 앵커(전부-GPU S3) 반복 측정 — 드리프트 감시 | `scripts/collect_run.py` | 없음 (참고) | `d6f9d17094ebbd5d` / 2,154 B |
| 6 | `xgboost_model/full_collection_540/cpu_npu/collect_npu_anchor.jsonl` | 동상 | 동상 | 없음 (참고) | `d0d611d94f3372af` / 2,120 B |
| 7 | `working_sets.yaml` | 15개 워킹셋 정의 + rate factor 규칙 + 층화 설정 | 수작업 | 간접 (스케줄 생성 경유) | `42a15580c58efc00` / 2,685 B |
| 8 | `model_registry.py` | 모델 종류/허용 디바이스 (`DEVICE_CONSTRAINTS:45-47`) | 수작업 | `gen_collection_schedules.py`, `deploy_selector_xgb_suite.py` | `d66c50f684c80ef1` / 5,983 B |
| 9 | `xgboost_model/schedules/collection/collect_cpu_gpu.yaml` | **CPU-GPU 수집 스케줄** — combination → 모델·디바이스·infps. `view.infps` 계열 feature의 유일한 원천 | `xgboost_model/gen_collection_schedules.py` | P1(gpu), joint, P6(gpu) | `db65e8ee56f1708a` / 280,752 B |
| 10 | `xgboost_model/schedules/collection/collect_cpu_npu.yaml` | CPU-NPU 수집 스케줄 | 동상 | P1(npu), joint, P6(npu) | `bd6c9c2f72a4e769` / 280,752 B |
| 11 | `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json` | **isolated(단독) 프로파일** — 모델×디바이스 단독 추론시간/로드시간/토큰속도 | `profile_models.py --out <이 경로>` (`:29`) | P1·joint·P6·P5 greedy **전부 공유** | `7fec0eea36e9eb98` / 4,335 B |
| 12 | `analysis/groupkfold_gpu_metrics.json` | P1(gpu) 지표 **+ `group_to_fold` (train/test 분할 정의)** | `train_groupkfold.py` | P6(gpu) fold 상속 | `5f1b48849523ec17` / 16,365 B |
| 13 | `analysis/groupkfold_npu_metrics.json` | P1(npu) 지표 + `group_to_fold` | 동상 | P6(npu) fold 상속 | `0040264e8c84b99d` / 16,585 B |
| 14 | `analysis/groupkfold_gpu_oof.csv` | P1(gpu) OOF 예측 (`y3_pred` 포함) | 동상 | **P6(gpu) ŷ3 공유**, P3 대각선 | `2e10bbead96a9971` / 137,932 B |
| 15 | `analysis/groupkfold_npu_oof.csv` | P1(npu) OOF 예측 | 동상 | P6(npu) ŷ3 공유, P3 대각선 | `7c65760df3ac1afd` / 137,154 B |
| 16 | `xgboost_model/artifacts/deploy_cpu_gpu_features.json` | **정규 37-feature 순서**의 실제 소스 (`analysis_common.py:48-49`가 이 파일을 읽는다) | 출하 학습 경로 (`train_targets`) | P1, joint, P3 (열 순서 고정) | `79e34d26dbfe30a2` / 1,122 B |
| 17 | `xgboost_model/full_collection_540/cpu_gpu/oof_gpu.json` | 출하 학습 경로(`train_gpu.py`)의 OOF 백업 | `scripts/train_gpu.py` | 레거시 참고 (P1~P6 미사용) | `3ac1ac56e7f28dd6` / 85,339 B |
| 18 | `xgboost_model/full_collection_540/cpu_npu/oof_npu.json` | 동상 | `scripts/train_npu.py` | 레거시 참고 | `8f867e40e36d7875` / 85,106 B |

**중간 산출물**: feature 캐시나 병합 테이블 같은 별도 중간 파일은 **없다**.
`analysis_common.load_dataset()` (`analysis_common.py:106-136`)이 매 실행마다 창 JSON +
스케줄 YAML + 정적 프로파일에서 feature를 즉석 계산한다. 위 표의 12~15만이 "다른 실행의
산출물을 입력으로 쓰는" 중간 파일이고, 그 용도는 fold 상속과 ŷ3 공유다(§1-1).
`xgboost_model/artifacts/train_dataset.csv`(2026-07-09)는 파일럿 시절 `--dump_csv` 산출물로
현행 학습 경로가 읽지 않는다.

> **혼동 주의**: `xgboost_model/performance_data/cpu_{gpu,npu}/performance.json`은
> 160창짜리 **파일럿** 데이터다(`full_collection_540/README.md:19-24`). 현행 예측기는
> 이 데이터가 아니라 표의 1·2번으로 학습됐다.

#### "테스트 데이터"는 별도 파일이 아니다

세 모델 어디에도 별도의 테스트 세트 파일은 없다. 테스트는 **fold 매핑에 의한 논리적 분할**이다.

- 분할 함수: `analysis_common.assign_group_folds()` (`analysis_common.py:160-171`) —
  고유 그룹을 `RandomState(42).permutation`으로 섞어 3-fold 라운드로빈 배정. 시드 고정이라
  같은 그룹 목록에 대해 결정적이다.
- **어느 그룹이 어느 fold의 test가 되는지의 매핑 파일**:
  `analysis/groupkfold_{gpu,npu}_metrics.json`의 최상위 키 **`group_to_fold`**
  (45개 항목, `"('llama1b,qwen2_vl,resnet50,yolo11n', 1.0)": 2` 형식).
  `train_groupkfold.py:110-111`이 쓰고 `train_decomposed.py:100-112`가 읽는다.
- 이 파일이 결정하는 범위: **P1과 P6**. joint은 같은 함수를 독립 호출한다
  (`train_unified.py:53-54`) — 파일을 읽지는 않지만 키가 같아 **결과 파티션은 동일**하다.
  실행 시 45쌍이 fold 를 가로지르지 않는지 assert 로 확인한다 (`train_unified.py:58-62`).
- 검증: 저장된 `group_to_fold`를 `assign_group_folds`로 재계산해 두 플랫폼 모두 **완전 일치**
  함을 확인했다(`measurements.json:folds.P1_{gpu,npu}_matches_stored_group_to_fold = true`).
  joint의 fold 행 수도 저장값 `[320, 352, 408]`과 재계산이 일치한다.

### 2-B. 규모 (전부 실측)

#### a. candidate 배치 수 — 플랫폼당 **180**, 세트별 분해

| 세트 | N | candidate (gpu) | candidate (npu) | Appendix A `n_measured` | rate 수준 | 창 수 |
|---|---|---|---|---|---|---|
| S1 | 2 | 4 | 4 | 4 | 2.0 / 3.0 / 4.0 | 12 |
| S2 | 3 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| S3 | 3 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| S4 | 3 | 4 | 4 | 4 | 1.0 / 2.0 / 3.0 | 12 |
| base1 | 4 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| base2 | 4 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| base3 | 4 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| base4 | 4 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| base5 | 4 | 8 | 8 | 8 | 1.0 / 2.0 / 3.0 | 24 |
| S5 | 5 | 16 | 16 | 16 | 1.0 / 2.0 / 3.0 | 48 |
| S9 | 5 | 16 | 16 | 16 | 1.0 / 2.0 / 3.0 | 48 |
| S6 | 6 | 16 | 16 | 16 | 1.0 / 1.5 / 2.0 | 48 |
| S7 | 7 | 20 | 20 | 20 | 1.0 / 1.5 / 2.0 | 60 |
| S10 | 7 | 24 | 24 | 24 | 1.0 / 1.5 / 2.0 | 72 |
| S8 | 8 | 24 | 24 | 24 | 1.0 / 1.5 / 2.0 | 72 |
| **계** | | **180** | **180** | **180** | | **540** |

- 두 플랫폼의 candidate 집합은 가속기 이름(gpu/npu)을 `accel`로 치환하면 **집합으로 동일**
  하다 (`measurements.json:facts.identical_candidate_sets_across_platforms = true`).
- Appendix A 커버리지 표(`runs/20260730_172053_coverage/appendix_coverage.csv`)의
  `n_measured_gpu`/`n_measured_npu`와 **15/15 세트 전부 일치**한다.
- candidate 생성 규칙: 자유 배치 가능한 모델(=CPU 허용 모델)만 `{cpu, accel}` 전수 조합
  (`gen_collection_schedules.py:free_models`, `all_placements`), `qwen2_vl`은 CPU 불가라
  항상 accel 고정 (`model_registry.py:45-47`). S6~S10은 "가속기 위 자유 모델 수" 축으로 층화
  샘플링 (`working_sets.yaml:stratified_sets`, `stratified_target_by_free`).

#### b. input frequency 변이

- **세트당 수준 수: 정확히 3** — 15세트 × 2플랫폼 전부.
- 실제 배수 집합과 세트 크기별 규칙 (`working_sets.yaml:rate_factors_by_size`):

  | 버킷 | 조건 | 배수 | 적용 세트 |
  |---|---|---|---|
  | small | 모델 ≤ 2 | 2.0, 3.0, 4.0 | S1 |
  | medium | 모델 3–5 | 1.0, 2.0, 3.0 | S2, S3, S4, base1–base5, S5, S9 |
  | large | 모델 ≥ 6 | 1.0, 1.5, 2.0 | S6, S7, S8, S10 |

  15세트 전부 규칙과 실측 배수가 **일치**(`measurements.json:rate_check.*.match = true`).
- **(candidate × rate) = window**: 세트마다 각 rate에서 candidate 수만큼 창이 있음을 확인했다
  (예외 0건). 따라서 플랫폼당 180 × 3 = **540창**, 그룹 `(models, rate)` 수 = 15 × 3 = **45**.

#### c. 모델별 학습 행 수

| 모델 | 행 수 | feature 수 | 그룹 수 | 비고 |
|---|---|---|---|---|
| P1 (플랫폼당) | **540** | 37 | 45 | y3 학습 대상은 `y3_valid` 372행 (`train_groupkfold.py:56`), 생성 세트 창은 384행 |
| joint | **1080** | 37 | 90 | 두 플랫폼 concat, `assert len(X) == 1080` (`train_unified.py:38`); y3 valid 744행 |
| P6 (플랫폼당) | **2220** (vision 뷰) | 18 | 45 | 전체 뷰 2952 = vision 2220 + 생성 뷰 732. 생성 뷰는 학습 제외 (`train_decomposed.py:124-127`) |

- P6의 2220은 코드·문서에 상수로 적혀 있지 않아 실제로 카운트한 값이다. 저장된
  `decomposed_metrics.json:{gpu,npu}.view_level.n_vision_views = 2220`과 일치하며,
  두 플랫폼이 같다(배치 집합이 같으므로).
- 학습 가능 행(target 유한) = 2220/2220, target clip(>1.0) **0건** — 저장값과 일치.
- 뷰 수 검산: 세트별 (모델 수 × 창 수)의 합 = 2952. 예) S8은 8모델 × 72창 = 576.

#### d. fold별 train/test 크기

**P1 (두 플랫폼 동일 — 그룹 목록이 같아 분할도 같다)**

| fold | test 그룹 | test 창 | train 그룹 | train 창 | train 창(y3 valid) |
|---|---|---|---|---|---|
| 0 | 15 | 216 | 30 | 324 | 252 |
| 1 | 15 | 140 | 30 | 400 | 280 |
| 2 | 15 | 184 | 30 | 356 | 212 |

그룹 수는 15/15/15로 균등하지만 창 수는 216/140/184로 불균등하다 — 그룹당 창 수가
4~24로 다르기 때문이다. 저장값 `groupkfold_{gpu,npu}_metrics.json:fold_row_counts`와 일치.

**joint (45 paired 그룹)** — P1 과 같은 파티션, 창 수만 2배

| fold | test 그룹 | test 창 | train 그룹 | train 창 |
|---|---|---|---|---|
| 0 | 15 | 432 | 30 | 648 |
| 1 | 15 | 280 | 30 | 800 |
| 2 | 15 | 368 | 30 | 712 |

(폐기된 90그룹 분할의 fold 창 수는 320/352/408 이었다 —
`analysis/unified_metrics_90group_legacy.json` 참조.)

저장값 `unified_metrics.json:fold_row_counts = [320, 352, 408]`과 일치.

**P6 (뷰 행, 두 플랫폼 동일)**

| fold | test 그룹 | test 뷰 | train 그룹 | train 뷰 |
|---|---|---|---|---|
| 0 | 15 | 1080 | 30 | 1140 |
| 1 | 15 | 408 | 30 | 1812 |
| 2 | 15 | 732 | 30 | 1488 |

fold는 P1에서 상속하지만(§1-1) 뷰 수 비율은 창 수 비율과 다르다 — fold 0에 큰 세트
(모델 수가 많은 세트)가 몰려 있기 때문이다.

#### 코드 상수·논문 서술과의 대조

| 대조 대상 | 서술값 | 실측 | 결과 |
|---|---|---|---|
| 플랫폼당 배치 수 | 180 | 180 | 일치 |
| 플랫폼당 창 수 | 540 | 540 | 일치 |
| 그룹 수 | 45 | 45 | 일치 |
| 워킹셋 수 | 15 | 15 | 일치 |
| rate 수준 수 | 3 | 3 (15/15 세트) | 일치 |
| joint 행 수 | 1080 | 1080 | 일치 (`train_unified.py:38` assert) |
| feature 수 | 37 | 37 | 일치 (`runs/20260804_180759_feature_spec` 참조) |
| y3_valid 행 수 | 372 (`Y2_NORMALIZATION.md`) | 372 | 일치 |

**어긋나는 항목 없음.** P6 뷰 행 수(2220/2952/732)는 대조할 서술값이 없어 신규 기재다.

---

## 3. 학습 파이프라인

### 3-1. 단계별 함수

| 단계 | 함수 | 위치 |
|---|---|---|
| ① 창 로드 | `load_windows(platform)` | `analysis_common.py:88-90` |
| ② infps 맵 | `infps_map_for(platform)` → `_build_infps_lookup` | `analysis_common.py:97-103`, `deploy_selector_xgb_suite.py:224` |
| ③ 정적 프로파일 | `load_static_profiles(STATIC_JSON)` | `deploy_selector_xgb_suite.py:101` |
| ④ featurize (37차원) | `featurize_window(w, S, infps_map=m)` → `_view_features` `:250` + `_aggregate` `:282` | `deploy_selector_xgb_suite.py:297` |
| ⑤ 열 순서 고정 | `X.reindex(columns=FEATURE_ORDER)` | `analysis_common.py:133` (순서 원본: `artifacts/deploy_cpu_gpu_features.json`) |
| ⑥ 타깃 정규화 | `normalize_targets(Y_raw, M)` → `_normalize_targets` | `analysis_common.py:139-148`, `deploy_selector_xgb_suite.py:425` |
| ⑦ 그룹 키 | `group_keys(M)` — 세 모델 모두 `(models, rate_factor)` | `analysis_common.py:152-157` |
| ⑧ GroupKFold | `assign_group_folds(groups, n_folds=3, seed=42)` | `analysis_common.py:160-171` |
| ⑨ 하이퍼파라미터 선택 | `cv_select_params(Xtr, ytr, inner_groups)` | `analysis_common.py:228-256` |
| ⑩ 학습 | `train_booster(Xtr, ytr, params, rounds=300)` | `analysis_common.py:259-263` |
| ⑪ 예측 | `predict_booster(bst, Xte)` — `clip(0,1)` | `analysis_common.py:266-270` |
| ⑫ 점수화·랭킹 | `scores_from` `:183`, `ranking_metrics` `:191` | `analysis_common.py` |

**타깃 정규화**(⑥, `deploy_selector_xgb_suite.py:425-477`): 그룹 키
`(models, workload, rate_factor)` 내에서
`y1, y3 ← F/Fmax` (클수록 좋음), `y2 ← (y2−lo)/(hi−lo)` (작을수록 좋음 유지, 편차 0이면 0).
`analysis_common.py:139-148`의 docstring이 fold 안전성 근거를 적어 둔다 — fold가 그룹 단위라
어떤 통계도 fold 경계를 넘지 않고, 보류 그룹 안에서는 단조 변환이라 순위 보존.

**y3 마스킹**: y3 부스터는 `y3_valid` 행으로만 학습하되 보류 fold의 **전 행**을 예측한다
(`train_groupkfold.py:56, 68-71`) — 운영 예측기가 LLM-on-CPU 배치도 점수화하는 것과 맞춘 것.

### 3-2. 하이퍼파라미터 그리드와 시드

기본 파라미터 (`deploy_selector_xgb_suite.py:519-523`):

```python
_PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse",
           "max_depth": 5, "eta": 0.1, "subsample": 0.8,
           "colsample_bytree": 0.8, "min_child_weight": 3.0, "seed": 42}
```

탐색 그리드 4조합 (`:525-530`): `(max_depth, eta)` ∈ {(3, 0.1), (4, 0.1), (5, 0.05), (6, 0.1)}.
부스팅 라운드 300 고정(`analysis_common.py:229, 259`). 선택 기준은 **그룹 단위 내부 3-fold CV의
MAE**(`analysis_common.cv_select_params:228-256`) — 출하 경로의 모듈로 fold
(`deploy_selector_xgb_suite._cv_select_params:534`, `fold_id = arange(n) % folds`)와 달리
선택 단계의 그룹 내 누수도 제거한다.

시드는 전부 **42**: `analysis_common.SEED = 42` (`:44`), `_PARAMS["seed"] = 42`,
fold 셔플 `RandomState(42)` (`:168`). 난수는 이 셋뿐이다.

### 3-3. 재학습 CLI (⚠ 실행하지 말 것 — 아래 경로를 덮어쓴다)

```bash
# P1 — platform-specific (두 플랫폼 모두 한 번에)
python xgboost_model/full_collection_540/scripts/train_groupkfold.py

# joint (P1 결과를 읽어 비교표를 만들므로 P1 이후에)
python xgboost_model/full_collection_540/scripts/train_unified.py

# decomposed (P1의 group_to_fold·y3_pred에 의존 — 반드시 P1 이후에)
python xgboost_model/full_collection_540/scripts/train_decomposed.py
```

> **⚠ 덮어쓰는 경로**
>
> - `train_groupkfold.py`: `xgboost_model/artifacts/gkf_cpu_{gpu,npu}_{y1,y2,y3}.json`,
>   같은 이름의 `_fold{0,1,2}.json` 12개, `gkf_cpu_{gpu,npu}_{features,coverage}.json`,
>   `analysis/groupkfold_{gpu,npu}_metrics.json`, `analysis/groupkfold_{gpu,npu}_oof.csv`,
>   `analysis/groupkfold_ranking.md`
> - `train_unified.py`: `analysis/unified_metrics.json`, `analysis/unified_oof.csv`,
>   `analysis/unified_vs_specialized.md`
> - `train_decomposed.py`: `analysis/decomposed_metrics.json`,
>   `analysis/decomposed_results.md`, `analysis/decomposed_view_oof.csv`
>   (`decomposed_results.md`는 `analysis/greedy_results.csv`도 **읽는다** — `:214`)
>
> 셋 다 출하 `deploy_*` 아티팩트는 건드리지 않는다 (`train_groupkfold.py:18`).
> `train_decomposed.py`를 P1 재학습 없이 돌리면 옛 fold·옛 ŷ3와 섞인다.
> P1을 다시 돌리면 P3(`cross_platform_transfer.py`)도 fold 부스터를 다시 읽으므로 재실행 대상이다.

**출하 아티팩트(`deploy_cpu_*`) 재학습**은 별개 경로다 (`full_collection_540/README.md:38-52`):

```bash
python xgboost_model/deploy_selector_xgb_suite.py train \
  --perf_dir xgboost_model/full_collection_540/cpu_npu \
  --schedule_dir xgboost_model/schedules/collection \
  --static_json xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
  --model_out xgboost_model/artifacts/deploy_cpu_npu \
  --rate_factors 1.0 1.5 2.0 3.0 4.0
```

`--rate_factors`를 생략하면 기본값 `[1.0, 2.0, 4.0]` (`deploy_selector_xgb_suite.py:758-760`)로
필터되어 1.5×·3× 행이 통째로 빠진다. 반드시 전 rate를 명시할 것.

---

## 4. 산출물 디렉터리 명세

`xgboost_model/artifacts/` (2026-08-06 `ls` 기준, 총 49 파일 + 2 디렉터리):

| 파일명 패턴 | 개수 | 내용 | 생성 주체 | 최종 갱신 |
|---|---|---|---|---|
| `gkf_cpu_{gpu,npu}_{y1,y2,y3}.json` | 6 | **P1 최종 부스터** (전 데이터, 후속 재사용용) | `train_groupkfold.py:82` | 2026-07-29 18:29–18:30 |
| `gkf_cpu_{gpu,npu}_{y1,y2,y3}_fold{0,1,2}.json` | 18 | **P1 fold 부스터** — P3 교차 전이가 재학습 없이 소비 | `train_groupkfold.py:75` | 2026-07-29 18:29–18:30 |
| `gkf_cpu_{gpu,npu}_features.json` | 2 | 37-feature 순서 사이드카 | `train_groupkfold.py:141-142` | 2026-07-29 |
| `gkf_cpu_{gpu,npu}_coverage.json` | 2 | 학습 분포 커버리지(세트·디바이스·rate) | `write_coverage` (`suite:556`) | 2026-07-29 |
| `deploy_cpu_{gpu,npu}_{y1,y2,y3}.json` | 6 | **출하 운영 부스터** (GUI/deploy finder용) | `train_gpu.py` / `train_npu.py` → `train_targets` (`suite:617`) | y1·y3 2026-07-17~18, **y2만 2026-07-20** (y2 정규화 재학습) |
| `deploy_cpu_{gpu,npu}_{features,coverage}.json` | 4 | 동상 사이드카 | 동상 | 2026-07-17~18 |
| `deploy_conc_cpu_gpu_*` | 5 | 동시성(concurrency) 파일럿 산출물 | 파일럿 경로 | 2026-07-10 |
| `deploy_xgb_*` | 5 | 초기 파일럿 산출물 (160창) | 파일럿 경로 | 2026-07-09~10 |
| `train_dataset.csv` | 1 | 파일럿 `--dump_csv` 덤프 | `suite:main --dump_csv` | 2026-07-09 |
| `pre_recollect_backup/` | 10 | 본수집 이전 `deploy_cpu_*` 백업 | 수작업 | 2026-07-18 |
| `pre_y2_norm_backup/` | 2 | y2 정규화 이전 `deploy_cpu_*_y2.json` 백업 | 수작업 | 2026-07-20 |

**joint / decomposed 부스터는 저장되지 않는다.**
저장소 전체에서 `save_model` 호출은 3곳뿐이다 —
`train_groupkfold.py:75`(P1 fold), `train_groupkfold.py:82`(P1 final),
`deploy_selector_xgb_suite.py:617`(출하 `train_targets`).

- **joint(P4)**: 최종·fold 부스터 **둘 다 없음**. 의도적이며 근거가 명시돼 있다 —
  "No boosters are exported: nothing downstream consumes a unified booster"
  (`train_unified.py:15-16`). 재현은 `unified_oof.csv`로 한다.
- **decomposed(P6)**: 최종·fold 부스터 **둘 다 없음**. fold 부스터는
  `train_decomposed.py:136`에서 메모리상으로만 만들어 예측 후 버린다. 뷰 수준 예측은
  `analysis/decomposed_view_oof.csv`에 남는다.
  (즉 "final만 미저장, fold는 저장"이 아니라 **아무것도 저장하지 않는다**.)

지표·표·그림 산출물은 `xgboost_model/full_collection_540/analysis/` 아래에 있고
목록·의미는 같은 디렉터리의 `SUMMARY.md`, `figures/README.md`가 관리한다.

---

## 5. 사용법 (추론)

새 예제 스크립트는 만들지 않는다. 아래는 **저장된 P1 부스터로 한 워킹셋의 배치들을 랭킹**하는
최소 경로를 기존 함수 포인터로 적은 것이다.

1. **부스터 로드** — `analysis_common.load_booster(ARTIFACTS / f"gkf_cpu_{platform}_{tag}.json")`
   (`analysis_common.py:273-277`). tag ∈ {y1, y2, y3}.
   교차 전이·앙상블이 필요하면 fold 부스터 3개를 읽어 평균 내는
   `cross_platform_transfer.transfer_predict()` (`:64-75`)가 그 경로다.
2. **feature 구성** — 두 경로 중 하나:
   - 측정 창에서: `analysis_common.load_dataset(platform)` (`:106-136`) — 창 JSON + 스케줄 YAML의
     infps + 정적 프로파일로 37열을 만들고 `FEATURE_ORDER`로 열 순서를 맞춘다.
   - 스케줄만 있고 측정이 없을 때(운영 경로): `featurize_from_combo(S, combo_blob)`
     (`deploy_selector_xgb_suite.py:363`).
3. **예측** — `analysis_common.predict_booster(bst, X.values)` (`:266-270`, `clip(0,1)`).
   출하 아티팩트를 쓸 때는 `predict_targets(prefix, X, with_y3=gen)`
   (`deploy_selector_xgb_suite.py:711`)이 세 타깃을 한 번에 처리하고,
   내부의 `_align_features` (`:700`)가 `*_features.json`으로 열을 정렬·보충한다.
   `with_y3=False`(vision-only 세트)면 y3 부스터를 **로드조차 하지 않는다**.
4. **점수화** — `S = y1 − α·y2 + β·y3`, α=0.3, β=1.0.
   - 스칼라 1건: `score_combo(y1, y2, y3, has_generative, alpha, beta)`
     (`deploy_selector_xgb_suite.py:681-699`, 기본값 `DEFAULT_ALPHA/BETA` `:677-678`).
   - 벡터: `analysis_common.scores_from(y1, y2, y3, has_gen)` (`:183-188`).
   - **β·y3 항은 생성 세트에만 더한다.** vision-only 세트에서 y3는 목적함수에 들어가지
     않는다(사후 0 처리가 아니라 항 자체가 없다) — 근거는 `score_combo` docstring `:682-694`.
5. **랭킹** — 그룹(=세트×rate) 안에서 S 내림차순.
   Top-1/Top-5/Spearman은 `analysis_common.ranking_metrics(df)` (`:191-212`).
   동점은 `1e-9` 허용오차의 **동점 집합**으로 처리한다(`:204`) — 첫 argmax가 아니다.

> **점수는 세트 내에서만 비교 가능하다.** 생성 세트에만 β·y3가 붙어 점수 척도가 세트 유형에
> 따라 달라지고, 타깃이 그룹 내 정규화값이다. 세트를 가로질러 점수를 비교하지 말 것
> (`score_combo` docstring `:695-697`).

### zero-shot / joint 사용 시 주의

- **exec indicator(one-hot)와 정적 프로파일은 "평가 플랫폼" 기준이어야 한다.**
  37 feature 중 `views.{sum,mean,max}.view.exec_{cpu,gpu,npu}` 9개가 디바이스 배치를,
  `view.static_{infer,load,tokens}_sel` · `view.capacity_fps` · `view.load_factor`가
  **선택된 디바이스의 단독 프로파일 값**을 담는다(`_view_features:250-279`,
  `_device_static:126`). 평가 플랫폼의 값으로 만들지 않으면 예측이 무의미하다.
- **zero-shot 전이**의 구조적 문제: GPU에서 학습한 부스터에게 `exec_npu` 열은 학습 내내
  상수 0이던 열이다. `cross_platform_transfer.py`는 무보정(`asis`)과 열 스왑(`swap`) 두
  변형을 모두 평가했고 차이는 ±0.002로 무의미했다 — 전이 격차의 원인은 one-hot이 아니라
  `static_*`/`capacity` feature의 디바이스 값 차이다(`analysis/SUMMARY.md` 이상 발견 3).
- **joint 예측기**는 두 플랫폼을 한 모델로 다루므로 별도 스왑이 필요 없지만, 그 대신
  exec one-hot이 플랫폼 구분자 역할을 하므로(`train_unified.py:5-7`) 배치의 디바이스
  표기가 정확해야 한다. **joint 부스터는 저장돼 있지 않다**(§4) — 쓰려면 재학습해야 한다.

---

## 6. 알려진 주의사항

### 6-1. 출하 `deploy_*` 아티팩트의 `infps = 0` 결함

**증상.** 출하된 운영 부스터는 `view.infps` 계열 feature가 전부 0인 채로 학습됐다.
예측 경로(GUI `featurize_from_combo`)는 실제 infps를 넣으므로 **train/serve 스큐**다.

**원인.** `build_dataset` → `build_dataset_from_file`은 창의 `schedule file` 힌트로 스케줄
문서를 찾아 infps 맵을 만든다(`deploy_selector_xgb_suite.py:403-418`, 힌트 추출
`_extract_schedule_hint:382`). 그런데 `performance_*_full540.json`의 창에는 그 키가 없다 —
실측: **540창 중 0창**이 힌트를 가진다(키는 `combination, models, rate_factor, total,
window_sec, workload`뿐). 힌트가 없으면 `infps_map = None`으로 조용히 넘어가 관련 feature가 0이 된다.

**영향 아티팩트** (부스터의 분기 사용 횟수로 직접 확인):

| 아티팩트 | 전체 분기 수 | infps 계열 6개 feature의 분기 수 |
|---|---|---|
| `deploy_cpu_gpu_y1.json` | 5089 | **0** |
| `deploy_cpu_npu_y1.json` | 5597 | **0** |
| `gkf_cpu_gpu_y1.json` | 3690 | 611 |
| `gkf_cpu_npu_y1.json` | 5692 | 931 |

→ 영향 대상은 **`deploy_cpu_gpu_*`, `deploy_cpu_npu_*` (6 부스터 + 사이드카)**.
`deploy_xgb_*`·`deploy_conc_cpu_gpu_*`는 파일럿 창(`performance_data/…/performance.json`)으로
학습됐고 그 창들은 힌트를 **전부** 가진다(160/160, 192/192) — 결함 대상이 아니다.

**논문 산출물에는 영향 없음.** P1·P3·P4·P6은 `deploy_*`를 읽지 않는다. 전부
`analysis_common.load_dataset()`을 쓰고, 이 함수는 힌트 대신 수집 스케줄 YAML에서 combination별
infps를 직접 만든다(`analysis_common.py:97-103, 119-120`, 근거 docstring `:6-12`). 논문
수치를 만든 부스터는 `gkf_*`이고 위 표대로 infps feature를 실제로 쓴다. 참고로 P1은 레거시
재현용 `infps=0` 변형을 병기했다: Top-1 GPU 0.933→0.889, NPU 0.867→0.800
(`analysis/groupkfold_ranking.md`).

**권고.** 운영 아티팩트를 다시 학습할 때는 (a) `jsonl_to_windows.py`가 창에 `schedule file`
키를 쓰도록 고치거나, (b) `build_dataset`이 `analysis_common.infps_map_for()`처럼 스케줄
YAML에서 직접 맵을 만들게 하거나, (c) 아예 `train_groupkfold.py` 경로의 데이터셋 로더를 쓰는
셋 중 하나를 택한다. `SUMMARY.md`의 열린 질문 Q4가 같은 사안이다.

### 6-2. 재학습 시 재발할 수 있는 과거 함정

- **`--rate_factors` 기본값**: 생략하면 `[1.0, 2.0, 4.0]`으로 필터되어 1.5×·3× 창이 통째로
  빠진다 (`deploy_selector_xgb_suite.py:758-760`, `full_collection_540/README.md:48-51`).
- **`--perf_dir` 재귀 수집**: `build_dataset`이 `perf_dir` 아래 `performance*.json`을 **재귀로
  전부** 읽는다 (`suite:485`). 540창 본수집과 160창 파일럿을 같은 트리에 두면 섞인다 —
  `full_collection_540/`를 `performance_data/` 밖에 둔 이유다 (`README.md:16-24`).
- **y2 정규화 방향**: y2는 "작을수록 좋음"이라 y1/y3의 `F/Fmax`를 그대로 쓰면 의미가 뒤집힌다.
  반드시 그룹 내 min-max(best=0, worst=1) (`suite:466-477`, `Y2_NORMALIZATION.md` §1).
  이 수정 때문에 `deploy_cpu_*_y2.json`만 2026-07-20에 재학습됐다(§4 표).
- **정규화 그룹 키에 `models` 포함**: 키가 `(workload, rate)`뿐이면 2모델 세트와 8모델 세트가
  같은 workload·rate를 공유해 함께 정규화되어 세트 크기 교란이 생긴다 (`suite:432-437`).
- **P6를 P1 없이 돌리기**: `train_decomposed.py`는 `groupkfold_*_metrics.json`(fold)과
  `groupkfold_*_oof.csv`(ŷ3)를 읽는다. P1 재학습 후 P6를 갱신하지 않으면 옛 fold·옛 ŷ3와
  섞인다. `:113`의 assert는 그룹 **누락**만 잡고 fold **낡음**은 잡지 못한다.
- **`train_gpu.py` / `train_npu.py`의 경로**: 두 스크립트는 이제 없는 세션 scratchpad
  (`/tmp/claude-1001/.../6f0af394-…/scratchpad/collect/windows_{gpu,npu}`)를 가리킨다.
  실재하지 않음을 확인했다 — 그대로 실행하면 실패한다. 재학습은 §3-3의 CLI를 쓸 것.
- **스케줄 YAML의 `sort_keys=False`**: `gen_collection_schedules.py:128`이
  `yaml.dump(doc, sort_keys=False)`로 쓴다. combination 순서가 삽입 순이지 알파벳 순이 아니므로,
  스케줄을 다시 생성하면 파일 diff가 커질 수 있다. 다만 학습은 combination을 **이름으로** 조회하므로
  (`_build_infps_lookup:224`) 순서 변경이 결과에 영향을 주지는 않는다.
- **"maxsize 하드코딩"** — **미확인.** 저장소의 `maxsize` 출현은 전부 런타임 뷰어/스터디의
  큐 크기(`run_concurrency_study.py:61,67`, `unified_viewer.py:552,561-562`)이고 학습·데이터
  경로와 무관하다. 학습 파이프라인에서 재발할 수 있는 `maxsize` 함정은 확인하지 못했다.

### 6-3. 그 밖의 미확인 항목

- `sample_profiling_data.json`의 **생성 스크립트는 `profile_models.py`**다
  (`--out xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json`,
  `profile_models.py:29`; 스키마는 `:7-22`). 실행에 실제 하드웨어가 필요하므로 이 문서 작업에서
  실행하지 않았다. 다만 함께 있는 `sample_profiling_data.pre_pathbug_backup.json`이 어떤 경로
  버그의 백업인지는 **미확인**이다.
  (`schedule_generator/file_manager.py:334-341`도 `sample_profiling_data.json`을 쓰지만
  NPU1/NPU2 스키마의 **다른 레거시 파일**을 CWD에 쓰는 코드이고, 학습이 읽는 파일이 아니다.)
- `deploy_conc_cpu_gpu_*` / `deploy_xgb_*`를 만든 정확한 명령줄: 파일럿 시기 산출물로
  스크립트가 저장소에 남아 있지 않다 — **미확인**.
