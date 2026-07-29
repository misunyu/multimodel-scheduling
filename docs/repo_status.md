# 저장소 상태 확인 보고서 (읽기 전용 조사)

- 생성: 2026-07-29
- 브랜치: `ubuntu_gpu_mobilint`
- HEAD: `d1daa694f38bb5c0f6e1fc8965291927d51a8d43` (2026-07-20 17:09 +0900, "Add comprehensive sequential display test for all 15 working sets")
- **이전 기준점 `d1daa694`과의 관계: HEAD가 기준점 그 자체다.** `git log d1daa694..HEAD`는 비어 있다 — 이전 조사 이후 **커밋이 하나도 없다**.
- 따라서 이 보고서의 "신규 여부" 판정 기준: ① 커밋 기준으로는 신규 없음, ② 파일 단위로는 **mtime > 2026-07-20 17:09(기준점 커밋 시각) 또는 uncommitted 변경**을 신규로 간주.

한 줄 결론: **이전 조사 이후 실험 관련 신규 작업은 전무하다.** 기준점 이후 변한 것은 07-21의 GUI 단발 실행 산출물(`predictions.csv`, `results/*`), venv 재구축 스크립트(`setup_venv.sh`), OpenCV headless 자동수리(`runtime_env.sh` 수정)뿐이며 모두 실험과 무관하다. **P1~P6 전부가 신규 구현 대상이다.**

---

## §1. 저장소 식별과 이전 조사와의 차이

### 1.1 uncommitted 변경 (전량)

| 항목 | 성격 | 실험 관련성 |
|---|---|---|
| `M runtime_env.sh` (07-21 16:05) | `enforce_headless_opencv()` 추가 — pip가 non-headless opencv를 재설치해 PyQt xcb가 깨지는 것을 자동 수리 `[코드]` | 무관 (환경 정비) |
| `M predictions.csv` (07-21 15:58) | GUI(`best_deploy_finder_executor.py:837-843`)가 실행마다 덮어쓰는 8행짜리 throwaway. gitignore 대상 `[아티팩트]` | 무관 |
| `?? setup_venv.sh` (07-21 16:03) | .venv 재구축 스크립트 (Python 3.10, CUDA-13 torch, Mobilint SDK 로컬 설치) `[코드]` | 무관 |
| `?? docs/{best_deploy_finder_explained.md, llama1b_core_confound_report.md, xgboost_dataset_features.html, display_test/, ui_relayout/}` | 07-14~07-20 산출 문서·스크린샷 (기준점 이전 생성) `[문서]` | 기존 작업의 부속물 |
| `?? xgboost_model/artifacts/pre_recollect_backup/` (07-17~18) | 160창 파일럿으로 학습된 구 부스터 백업 `[아티팩트]` | 기존 |
| `?? xgboost_model/artifacts/pre_y2_norm_backup/` (07-20 15:29) | y2 정규화 변경 직전의 y2 부스터 백업 `[아티팩트]` | 기존 |

### 1.2 이전 조사 산출물의 부재

- **`docs/paper_facts.md`와 `docs/_paper_facts_analysis.py`는 이 저장소(워킹카피 포함)에 존재하지 않는다.** untracked 목록에도 없다. 이전 조사가 같은 HEAD(`d1daa694`)를 기준으로 수행됐다고 되어 있으므로, 산출물이 다른 워킹카피 또는 세션 scratchpad에만 남았을 가능성이 크다. → **[질문 Q1]** `paper_facts.md`는 어디에 있는가? 이 저장소로 가져와야 하는가?

### 1.3 이전 조사가 지적한 사항의 수정 여부 — **전부 미수정**

(코드가 기준점에서 한 줄도 안 변했으므로 당연한 결과이나, 각 지점을 재확인했다.)

| 지적 | 판정 | 근거 |
|---|---|---|
| I-5: OOF fold 행 단위 무작위 분할 | **미수정** | `xgboost_model/deploy_selector_xgb_suite.py:631-632` — `RandomState(seed).randint(0, folds, size=n)` `[코드]` |
| I-5: 정규화가 fold 분할 전에 그룹 전체를 봄 | **미수정** | `deploy_selector_xgb_suite.py:512-514` — `build_dataset`이 fold와 무관하게 전체 데이터에 `_normalize_targets`(425행, `(models, workload, rate)` 그룹 내 min-max) 적용 `[코드]` |
| I-4: 최종 하이퍼파라미터 stdout만 출력 | **미수정** | `deploy_selector_xgb_suite.py:613-614` — `_cv_select_params` 채택값을 `print`만 하고 저장 안 함. 부스터 JSON의 `learner.attributes`도 `{}`(빈 값) `[코드][아티팩트]` |
| I-8: β 기본값 불일치 | **미수정** | 1.0: `deploy_selector_xgb_suite.py:678`(`DEFAULT_BETA=1.0`), `evaluate_model.py:190`, `best_deploy_finder_executor.py:763` / **0.5: `xgboost_model/full_collection_540/scripts/compare_platforms.py:19`** `[코드]` |
| I-6: y3 마스킹 단위가 경로에 따라 다름 | **미수정** | `deploy_selector_xgb_suite.py:595-600` — M(`y3_valid`, 조합 단위)이 있으면 행 단위, 없으면 `views.sum.view.is_llm > 0`(세트 단위) fallback `[코드]` |

---

## §2. 확정 실험 P1~P6 구현·결과 존재 판정

### P1 — GroupKFold 재학습

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **없음.** `GroupKFold`/`group_kfold`/`groups=` 저장소 전체 0건. sklearn은 `requirements.txt:28`에 있으나 어떤 .py도 import하지 않음. 기존 CV는 세 곳 모두 비그룹: 모듈로 fold(`deploy_selector_xgb_suite.py:537`), 행 무작위 fold(`:631-632`), 모듈로 fold(`evaluate_model.py:76`) `[코드]` |
| (b) 계획 일치 | 해당 없음 |
| (c) 결과 존재 | 없음. 하이퍼파라미터 기록 파일도 없음 (I-4 미수정) |
| (d) 데이터 세대 | 해당 없음 |

### P2 — 플랫폼 비교 β=1.0 재실행

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **스크립트는 존재**: `xgboost_model/full_collection_540/scripts/compare_platforms.py` — (set, rate)별 측정 점수 argmax 배치를 NPU vs GPU 비교 `[코드]` |
| (b) 계획 일치 | **세 가지 불일치**: ① `BETA=0.5` 고정(`:19`) — β=1.0 아님. ② 결과를 stdout으로만 출력, 파일 저장 없음. ③ 데이터 경로가 **이전 세션의 /tmp scratchpad를 하드코딩**(`:14-15, :24` — `/tmp/claude-1001/.../6f0af394-.../scratchpad/collect/windows_{npu,gpu}/performance_*_all.json`). 현재 그 경로는 아직 살아 있고 파일 크기가 저장소 백업(`full_collection_540/cpu_*/performance_*_full540.json`)과 바이트 단위 일치하므로, 저장소 경로로 바꿔 실행해도 동일 결과다 `[코드][역추출]` |
| (c) 결과 존재 | **파일 없음.** β=0.5 실행 결과가 `FULL_COLLECTION_GPU_REPORT.md:108`("45개 그룹 중 30개에서 최적 배치 다름") 등 md 산문으로만 존재. `BETA1_CPU_PLACEMENT.md`(07-20)는 β=1 민감도 검토 문서이지 계획된 "그룹별 내역 파일"이 아님 `[문서]` |
| (d) 데이터 세대 | 540창 (스크립트가 읽는 `performance_*_all.json` = 540창 본수집과 동일 파일) `[아티팩트]` |

### P3 — 교차 플랫폼 전이 (4×4 Spearman/Top-k)

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **없음.** 부스터 로드는 단일 prefix 인자 경유가 유일(`deploy_selector_xgb_suite.py:728`); GPU 부스터를 NPU 데이터에 적용하는 코드 0건. `cross_platform`/`transfer` 0건. `oof_gpu.json`/`oof_npu.json`은 `train_{gpu,npu}.py`가 쓰기만 하고 **읽는 코드가 저장소에 없음** `[코드]` |
| (b) 계획 일치 | 해당 없음. 가장 가까운 것은 `evaluate_model.py:103-160`의 단일 플랫폼 Spearman/Top-1/Top-5 (`cmd_select`) — 4×4 행렬 아님 |
| (c) 결과 존재 | 없음. 보고서의 Spearman(+0.954/+0.970)은 md 산문뿐, 기계가독 지표 파일 없음 |
| (d) 데이터 세대 | 해당 없음 |

### P4 — 통합 예측기 (1080행)

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **없음.** `unified`/`combined`/`merged` 실험 코드 0건(`train_data_rate.py`의 `merged_train.json`은 존재하지 않는 구 경로를 읽는 죽은 스크립트). `build_dataset`(`:479`)은 단일 perf_dir만 읽고, `full_collection_540/README.md:14-19`는 오히려 데이터 혼입을 막는 배치를 명시 `[코드][문서]` |
| (b) 계획 일치 | 해당 없음. 참고: feature에 `exec_cpu/gpu/npu` one-hot(`:266-`)이 있어 통합 학습 자체는 데이터 병합만으로 가능한 구조 |
| (c) 결과 존재 | 없음. "1080" 언급 자체가 저장소에 없음 |
| (d) 데이터 세대 | 해당 없음 |

### P5 — Capacity greedy baseline

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **없음.** `capacity_fps`(`deploy_selector_xgb_suite.py:260`)와 `load_factor`(`:261`)는 **XGBoost 입력 feature로만** 존재 — 임계값 비교·누적·배정 로직 0건. 배치 선택은 전수 열거+ML 점수 정렬이 유일(`best_deploy_finder_executor.py:814-836`). `evaluate_model.py`는 oracle 대비 지표만 있고 heuristic baseline 없음. `schedule_generator/ui_components.py:258-296`의 makespan 브루트포스는 구세대 GUI 도구로 무관 `[코드]` |
| (b) 계획 일치 | 해당 없음 (정렬 2변형·누적 임계값·qwen2_vl CPU 불가 제약 모두 미구현) |
| (c) 결과 존재 | 없음 |
| (d) 데이터 세대 | 해당 없음 |

### P6 — Kim 방식 분해 모델링 (뷰 단위 저하율)

| 단계 | 판정 |
|---|---|
| (a) 구현 존재 | **없음.** `degradation`/`attainment`/`per_view`/`corunner` 학습 코드 0건. 현행 파이프라인은 의도적으로 창 단위: 뷰별 측정치는 feature로 쓰지 않는다는 설계 주석(`deploy_selector_xgb_suite.py:262-264`), 뷰별 throughput은 y1로 **합산**(`:321`) `[코드]` |
| (b) 계획 일치 | 해당 없음. `throughput_fps / min(infps, capacity_fps)` 비율은 어디서도 계산되지 않음 (y1 정규화 `T=F/Fmax`는 다른 양) |
| (c) 결과 존재 | 없음 |
| (d) 데이터 세대 | 해당 없음. 단, **재료는 540창 세대에 완비** — §5.2 참조. 제약 1건: `jsonl_to_windows.py:4-11`에 따르면 **뷰별 tokens_per_s는 미보존**(창 합계를 생성 뷰 1개에 부착) → y3의 뷰 분해는 불가, vision 뷰만 분해 가능 `[코드][아티팩트]` |

---

## §3. 데이터·아티팩트의 변화

### 3.1 신규 수집 데이터 — **없음**

- 2026-07-20 14:38(540창 백업 커밋 시점) 이후 새 수집 jsonl/performance JSON 없음. 기준점 이후 가장 최신 데이터 파일은 `results/performance_20260721_155821.json`(2,506B, GUI 단발 MPOpt 실행, gitignore 대상) — 수집 데이터 아님 `[아티팩트]`
- `full_collection_540/`: 플랫폼당 540행 jsonl + 540창 windows JSON + oof(3-fold, seed 42) + anchor 30회 — 이전 조사와 동일. 창 수·rate 분포 재확인: 양 플랫폼 동일하게 540창, rate {1.0:176, 1.5:84, 2.0:180, 3.0:96, 4.0:4}, 뷰 수 2~8 `[아티팩트]` (`docs/_repo_status_checks.py`로 계산)

### 3.2 artifacts 대조 — **부스터 교체 없음**

`xgboost_model/artifacts/` 파일 크기가 이전 조사 기록과 **전부 일치**:

| 파일 | 크기(B) | 이전 조사 기록 | mtime |
|---|---|---|---|
| deploy_cpu_gpu_y1.json | 664,934 | 664,934 ✓ | 07-18 18:54 |
| deploy_cpu_gpu_y2.json | 619,574 | 619,574 ✓ | 07-20 15:29 |
| deploy_cpu_gpu_y3.json | 626,844 | 626,844 ✓ | 07-18 18:54 |
| deploy_cpu_npu_y1.json | 717,601 | 717,601 ✓ | 07-17 06:28 |
| deploy_cpu_npu_y2.json | 318,031 | 318,031 ✓ | 07-20 15:29 |
| deploy_cpu_npu_y3.json | 305,083 | 305,083 ✓ | 07-17 06:28 |

- 추가/삭제된 패밀리 없음. 현존: `deploy_cpu_{gpu,npu}`(rows=540), `deploy_conc_cpu_gpu`(rows=192), `deploy_xgb`(rows=160), 백업 2종(`pre_recollect_backup` rows=160, `pre_y2_norm_backup` y2만) `[아티팩트]`
- y2만 07-20 재학습(y2 min-max 정규화 변경, `Y2_NORMALIZATION.md`)이라 **cpu_gpu/cpu_npu 삼중항이 단일 학습 실행 산물이 아님** — 이는 이전 조사 시점에도 그랬고 변화 없음
- features.json 6개 전부 바이트 동일 (37개 feature, md5 `72a438c6...`) `[아티팩트]`

### 3.3 provenance — **개선 없음**

부스터 JSON top-level은 `learner`/`version`뿐, `learner.attributes = {}`. coverage.json에도 커밋 해시·데이터 경로·타임스탬프 없음. provenance는 `full_collection_540/README.md`와 `FULL_COLLECTION_*_REPORT.md` 산문이 유일 `[아티팩트][문서]`.

### 3.4 신규 평가·비교 결과 파일 — **없음**

기준점 이후 생성된 `*.json/*.csv/*.md`는 `predictions.csv`(GUI throwaway)와 `results/*`(GUI 단발)뿐. 기존 보고서 군(`FULL_COLLECTION_*_REPORT.md`, `Y2_NORMALIZATION.md`, `SCORE_WEIGHTS_REVIEW.md`, `BETA1_CPU_PLACEMENT.md`, `LM_DEADLINE_AUDIT.md` 등 18종)은 모두 기준점 이전 산물로 이전 조사가 이미 본 것들이다.

---

## §4. 논문 관련 신규 파일

- **원고 없음**: `.tex`/`.bib`/`paper*.md` 0건. 그림·표 생성 스크립트도 현세대용은 0건 — `backup/generate_latex_table.py`, `backup/lambda_score_graph.py`는 2025 구세대(resnet50/yolov3 모델셋) 전용 `[코드]`
- 열린 질문에 대한 결정의 반영 여부:

| 질문 | 현재 상태 |
|---|---|
| 데드라인 서술 | **주기 기반 유지**: `deadline_ms = (1000/infps) × DEADLINE_FACTOR`, `DEADLINE_FACTOR=3.0` (`model_registry.py:27`, `view_handlers.py:89`) `[코드]` |
| β 통일 | **미통일**: 3파일 1.0 vs `compare_platforms.py` 0.5 (§1.3 I-8). `BETA1_CPU_PLACEMENT.md`·`SCORE_WEIGHTS_REVIEW.md`가 β=1 쪽 검토를 담고 있으나 코드 통일은 안 됨 `[코드][문서]` |
| isolated latency 표 | 변경 없음 (코드·데이터 모두 기준점 그대로이므로 결정 미반영) |
| 137/103 fps 서술 | **부분만 확인 가능**: 137은 GPU anchor(전부-GPU S3) 측정 y1 137.24–137.92와 일치(`FULL_COLLECTION_GPU_REPORT.md:58`) `[문서]`. **"103"은 저장소 어디에도 없음** — NPU anchor 30회 평균은 **115.1 fps**(`collect_npu_anchor.jsonl`에서 계산) `[역추출]`. → **[질문 Q2]** "103 fps"의 출처는? NPU 앵커 값이라면 115가 맞는 수치다 |

---

## §5. 데이터에서 계산한 예비 확인

사용 스크립트: **`docs/_repo_status_checks.py`** (읽기 전용 계산; suite의 `_build_infps_lookup`/`_device_static` 재사용).

### 5.1 P5 예비 (greedy 배치 손추적)

**수행 불가** — greedy 구현이 저장소에 존재하지 않으므로(§2 P5) "그 코드가 산출할 배치"를 추적할 대상이 없다. 신규 구현 후에 수행해야 한다.

### 5.2 P6 뷰 단위 target 재료 — **완비 (vision 한정)**

- 필요 필드 가용성: 뷰별 `throughput_fps`는 540창 JSON에 전 뷰 존재(결측 0/2,952 뷰, 플랫폼당). `infps`는 `xgboost_model/schedules/collection/collect_cpu_{gpu,npu}.yaml`에서 (model, device)별 전 뷰 복원(결측 0). `capacity_fps = 1000/s_infer`는 `sample_profiling_data.json`으로 전 뷰 계산(결측 0) `[아티팩트][역추출]`
- **rate_factor는 스케줄 infps에 이미 반영돼 있음** (예: resnet50 157.332 → rate 1.5에서 235.998 → 2.0에서 314.664) — 별도 곱셈 불필요 `[역추출]`
- 저하율 `r = throughput_fps / min(infps, capacity_fps)` 분포:

| 플랫폼 | 뷰 종류 | n | min | p25 | med | p75 | max |
|---|---|---|---|---|---|---|---|
| GPU | vision | 2,220 | 0.016 | 0.048 | 0.112 | 0.545 | 0.991 |
| GPU | llm/vlm | 732 | 0.000 | 0.106 | 0.714 | 0.798 | **1.014** |
| NPU | vision | 2,220 | 0.015 | 0.048 | 0.105 | 0.413 | 0.992 |
| NPU | llm/vlm | 732 | 0.000 | 0.145 | 0.483 | 0.599 | 0.783 |

- 주의 2건: ① GPU llm/vlm에서 r>1(최대 1.014) 소수 존재 — `min(infps, capacity)` 상한 대비 측정 지터로 보이며, target 정의 시 clip 여부 결정 필요. ② llm/vlm 뷰의 `throughput_fps`는 요청 처리율일 뿐 토큰 처리율이 아니고, **뷰별 tokens는 미보존**이므로 y3 분해는 불가(§2 P6-(d)). vision 저하율 중앙값이 0.1 근처로 매우 낮은 것은 CPU 배치 뷰의 심한 병목을 반영한다.

### 5.3 아티팩트 학습 행 수 (coverage.json 기준)

`deploy_cpu_gpu` = 540, `deploy_cpu_npu` = 540, `deploy_conc_cpu_gpu` = 192, `deploy_xgb` = 160, `pre_recollect_backup/*` = 160. 신규 아티팩트 없음 `[아티팩트]`.

---

## 신규 구현 필요 목록

**P1~P6 전부 신규 작업이 필요하다.**

| ID | 필요 작업 (한 줄) |
|---|---|
| P1 | `(models, rate_factor)` 그룹 fold의 GroupKFold CV + fold 내부 정규화로 재작성하고(현행 `:631-632` 무작위 fold, `:514` 사전 정규화 대체), 채택 하이퍼파라미터를 JSON으로 저장 |
| P2 | `compare_platforms.py`의 데이터 경로를 `full_collection_540/cpu_*/`로 교체, `BETA=1.0`으로 재실행, (set, rate)별 내역을 파일로 저장 |
| P3 | 저장된 4개 부스터×2 데이터셋 교차 적용 스크립트 신규 작성 — 4×4 Spearman/Top-k (oof 파일을 읽는 코드 자체가 현재 없음) |
| P4 | 양 플랫폼 1080행 병합 학습 스크립트 신규 작성 (exec one-hot feature는 기존에 있어 병합만으로 구조상 가능) |
| P5 | capacity greedy 2변형(speedup 순/load_factor 큰 순, 누적 load_factor ≤ 1.0, qwen2_vl CPU 불가) 신규 구현 + 산출 배치의 수집 180개 내 존재 여부 집계 |
| P6 | 뷰 단위 저하율 target 생성 변환 + co-runner 요약 feature + XGBoost 학습 신규 구현 — **단 vision 뷰 한정**(뷰별 tokens 미보존), y2 재구성 오차 보고 방식 별도 설계 필요 |

공통 선행 작업: `train_{gpu,npu}.py`·`compare_platforms.py`의 /tmp scratchpad 하드코딩 경로를 저장소 경로로 교체해야 어떤 재실행도 가능하다.

## 확인 중 발견한 이상

1. **뷰별 프레임 단위 원시 트레이스의 유일본이 /tmp에 있음**: `/tmp/claude-1001/.../6f0af394-.../scratchpad/collect/traces/` (1,140파일, 9.7GB, `{gpu,npu}_combination_N.jsonl`). 저장소 백업은 집계된 540창뿐이다. 현재 /tmp는 살아 있으나(창 파일은 저장소 사본과 크기 일치 확인) 재부팅/청소 시 소실된다. P6에서 창 내 시계열이 필요해지면 이것이 유일한 원천이다. → 백업 여부 결정 필요
2. **`docs/paper_facts.md` 부재** (§1.2, 질문 Q1): 이전 조사 산출물이 이 워킹카피에 없다
3. **"103 fps" 근거 부재** (§4, 질문 Q2): NPU anchor 실측 평균은 115.1 fps다
4. **P6 y3 분해 불가**: 뷰별 tokens_per_s가 수집 단계에서 보존되지 않아(`jsonl_to_windows.py:4-11`) Kim 방식 분해는 vision 뷰에만 적용 가능 — 계획의 "y2 재구성 오차 별도 보고"는 가능하나 y3 합성은 창 단위 값을 그대로 써야 한다
5. **뷰 단위 저하율이 1을 초과하는 뷰 존재** (GPU llm/vlm 최대 1.014): `min(infps, capacity)` 상한의 측정 지터. target 정의에서 clip(≤1.0) 여부를 결정해야 한다
6. **rate 분포 비균일**: 540창의 rate 분포가 {1.0:176, 1.5:84, 2.0:180, 3.0:96, 4.0:4}로 균일하지 않다(세트별 rate 셋이 다름). P1의 `(models, rate_factor)` 그룹 fold 설계 시 rate=4.0 그룹(4행)이 극소수인 점을 고려해야 한다
7. 이전 조사 결과와의 수치 모순: **없음** (부스터 크기 6종 전부 이전 기록과 일치)

### 남긴 질문

- **Q1**: `docs/paper_facts.md`/`_paper_facts_analysis.py`의 소재 — 다른 워킹카피에 있는가, 이 저장소로 옮길 것인가?
- **Q2**: 논문 서술 후보 "137/103 fps"에서 103의 출처 — NPU 앵커라면 실측은 115.1이다
- **Q3**: P2 "그룹별 내역 파일"의 저장 위치·형식 지정 (예: `docs/` md vs `xgboost_model/full_collection_540/` json)
- **Q4**: /tmp의 9.7GB 프레임 트레이스를 저장소 밖 어딘가에 백업할 것인가?
