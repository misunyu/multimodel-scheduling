# Phase N2-A 보고 — CPU–NPU campaign·무결성·dataset (NPU 보드)

작성: 2026-10-15 23:40 KST. Branch `revision-profiling-npu` (worktree `../mms_npu_port`).
표기: [확인] / [추정] / [확인 불가]. 사전 등록 값(§0)은 바꾸지 않았다.

## 0. 요약

| 항목 | 결과 |
|---|---|
| 실행 | 492 runs (rep0 372 + rep1 60 + rep2 60). 2026-10-15 17:53:48 – 23:21:09 KST (327.4 min). **중단 0, 재시도 0** |
| run 시간 | 평균 39.9 s/run (overhead 포함). 예상 40.9 s |
| 무결성 | **valid 492/492.** conservation_gen/flow 0, trace ok (instance-run 1,422개 전부). failed 0, Close rc 0, residency 위반 0, 변종–바이너리 불일치 0 |
| workload 완전성 | **44/44 workloads 후보 100% valid.** 제외 workload 0 |
| 도착 | 실현율 최소 0.9938. 후보 간 generated 차 위반 0 (학습 0/33, 평가 0/11) |
| 반복 | 3× N≥3 60 후보 × 3회. F 상대 범위 중앙값 0.0096 (최대 0.053). D_macro 절대 범위 중앙값 0.0041 (최대 0.016). S의 SE 중앙값 0.0019 (최대 0.0068). **반복별 oracle 4/4 일치** |
| dataset | `dataset/`: labels_train 279행, labels_test 93행, instances 1,020행 |
| 불변 | 현재 branch·`new_main` 변경 0. registry 21개 파일 sha256이 campaign 전후 같음 |

## 1. 실행 [확인]

- **계획:** `RUN_PLAN.md`, `run_plan.csv` (seed 42, 실행 전 기록).
  - rep0의 family 순서는 ybys, rbrsyb, rb, rbrsybys, ys, rsyb, yb, rbrs, rsybys, rs, rbybys이다. family 안의 (rate, 후보)는 섞었다.
  - rep1과 rep2는 3× N≥3 60 후보를 각각 새로 섞은 순서다.
- **실행:** `code/run_campaign_npu.py --plan run_plan.csv --window 30`, 컨테이너 1회 연속 실행. 세션 기록은 `logs/campaign_sessions.log`(START/END 1쌍씩)에 있다.
- **사전 확인 (A1, 17:50 KST):**
  - registry 21개 sha256이 pilot 후 스냅샷과 같다.
  - driver sha `92ad912c…`.
  - DMA32 free 약 178k pages.
  - npu0(ResNet50 small)와 npu1(YOLOv3 small) smoke 각 1 run(10 s): ok, Close rc 0 (`smoke/n2a_precheck/`).
- **실행 중 코드·registry·driver 변경 없음.**
  - campaign 전에 바꾼 것은 새 파일 `run_campaign_npu.py`의 `--plan` 모드(재시도·세션 기록)와 새 파일 `run_plan_npu.py`뿐이다.
  - `--plan` 모드는 1-run 시험(`smoke/n2a_precheck/plan_test/`)으로 확인했다.
- **블록별:**

| block | runs | 시간 (h) | run당 p50 (s) | DMA32 최소 |
|---|---|---|---|---|
| rep0 | 372 | 4.10 | 39.3 | 162,890 |
| rep1 | 60 | 0.68 | 40.1 | 162,826 |
| rep2 | 60 | 0.68 | 40.1 | 161,185 |

- **DMA32 free pages:** 178k에서 161k로 천천히 줄었다(약 35 pages/run). 하한 20,000과는 거리가 크다.
  - 원인 [추정]: 정상 load/close에서도 회수되지 않는 소량의 커널 할당이 쌓인다.
  - campaign을 다시 하면 같은 속도로 줄어들 것이다. 약 4,000 run 뒤에야 하한에 닿는다 [추정].

## 2. 무결성 (A3) [확인] — `results/INTEGRITY.md`, `results/integrity/*.csv`

### 2-1. 유효성
- 계획 492, valid 492, 누락 0. 계획에 없는 결과 0, rejected 0, retries 0, 최종 invalid 0.
- run 단위 검사(rc 0, valid, failed 0, NPU sidecar 수 = NPU instance 수, Close rc 0, residency, 매핑): 492/492 통과.

### 2-2. Conservation·trace·도착
- instance-run 1,422개 전부 conservation_gen = conservation_flow = 0, trace_check ok.
- 실현율(generated/scheduled): 최소 0.9938, 1% 분위 0.9988, 중앙값 1.0000. 0.95 미만 0개.

### 2-3. 후보 간 generated 차이 (Phase 2 규칙: ≥200 tick 상대 2%, <200 tick 절대 2건)
- 위반 workload: 학습 0/33, 평가 0/11.
- 상대 차 최대 0.0062. <200 tick instance의 절대 차 최대 0.

### 2-4. Residency·변종–바이너리 매핑
- NPU당 instance ≤ 1, NPU에 놓인 instance는 `_npu` key, CPU에 놓인 instance는 `_cpu` key: 위반 0.
- sidecar 기준 매핑 불일치 0: registry key = run의 key, NPU id = 배치, `.o` = `<instance>/npu_code/<instance>_neubla_p1.o`, driver = `/workspace/lib` 판.
- **변종 4개 `.o`:** resnet50_big `80d12c14…`, resnet50_small `dfd310e8…`, yolov3_big `c855b2b5…`, yolov3_small `9ea9a273…`.
  - registry sha256과 같다.
  - campaign 전후 sha 동일(`results/sha_after_campaign.json` vs `smoke/n2a_precheck/sha_precheck.json`: changed none).
  - 표는 `integrity/binary_mapping.csv`.

### 2-5. 반복 편차 (3× N≥3, 60 후보 × 3회)

| 지표 | 중앙값 | 90% 분위 | 최대 |
|---|---|---|---|
| F 상대 범위 | 0.0096 | 0.0291 | 0.0526 |
| D_macro 절대 범위 | 0.0041 | 0.0095 | 0.0160 |
| S의 SE (F_max = 3회 평균 F의 workload 최대) | 0.0019 | 0.0046 | 0.0068 |

- 반복별 oracle(각 반복 자체 정규화)이 세 번 모두 같은 workload: **4/4.** 세 번 모두 평균 label oracle과도 같다.
- oracle 1·2위 실측 차 ≤ 0.0002인 workload: **0/4.** 차는 0.0095–0.0187이다.

### 2-6. 시간 drift
- 같은 후보의 F 비율: rep1/rep0 중앙값 1.000 (5–95%: 0.983–1.028), rep2/rep0 중앙값 1.000 (0.983–1.034).
- NPU 온도는 driver와 sysfs에 노출되지 않는다 [확인 불가]. DMA32 추이는 §1에 있다.

### 2-7. Workload 완전성
44 workloads(11 × 4 rates) 모두 feasible 후보 100% valid다. 3× N≥3 4개는 3회 반복까지 모두 있다. **제외 0.**

## 3. Label 통계 [확인] — `results/integrity/label_stats_by_workload.csv`

### 3-1. 3× N≥3 평가 workload (3회 평균 label, α = 0.3, D_macro)

| family | 후보 | oracle (S) | oracle 배치 | R1 참고 (S) | all-CPU (S) |
|---|---|---|---|---|---|
| rbrsyb | 13 | combination_2 (0.9868) | rb:cpu, rs:cpu, yb:**npu0** | combination_2 (0.9868), oracle과 같음 | 0.5404 |
| rbybys | 13 | combination_10 (0.9922) | rb:cpu, yb:**npu0**, ys:**npu1** | combination_13 (0.9735) | 0.3110 |
| rsybys | 13 | combination_5 (0.9922) | rs:cpu, yb:**npu0**, ys:**npu1** | combination_7 (0.9737) | 0.3114 |
| rbrsybys | 21 | combination_5 (0.9881) | rb:cpu, rs:cpu, yb:**npu0**, ys:**npu1** | combination_7 (0.9786) | 0.3036 |

- **oracle의 device 구성:** 4개 모두 ResNet50은 CPU, YOLOv3는 NPU에 둔다. YOLOv3가 2개일 때는 **big이 npu0(Gen4×8), small이 npu1(Gen2×1)** 이다.
- **R1과의 차이:** R1은 N1 §4 규칙(isolated 최대 처리량 순)으로 정한 배치다. small이 isolated 처리량이 높아 npu0을 먼저 가져가고, big은 npu1로 간다.
  - 실측에서는 반대가 낫다. big이 npu1에서 잃는 처리량(10.23 → 9.50 req/s)이 small이 잃는 것보다 커서 3×에서 drop이 생기기 때문이다 [추정, isolated 수치 기준].
  - YOLO 2개 workload에서 R1의 regret은 0.010–0.019이다.
- 4× rbybys, rsybys, rbrsybys, ybys에서는 oracle이 R1 쪽(big:npu1, small:npu0)으로 바뀐다. 이 4개는 모두 학습 split이다.

### 3-2. 전체 workload
- **N = 1:**
  - ResNet(rb, rs)은 4개 rate 모두 CPU가 oracle이다. CPU 52.5 req/s, NPU 22–24 req/s.
  - YOLO(yb, ys)는 npu0이 oracle이다.
  - 1× workload에서는 drop이 없는 후보가 여러 개라 S = 1인 동점이 있다. 동점일 때 oracle은 index가 작은 것으로 정했다.
- **all-CPU의 3× S:** ResNet만 있는 workload는 1.0(rbrs_x3), YOLO가 있는 workload는 0.03–0.74다.

## 4. Dataset (A4) — `dataset/README.md`

| 파일 | 행 | sha256 (앞 16자) |
|---|---|---|
| `labels_train.csv` | 279 | 7f5acf8534fd58e4 |
| `labels_test.csv` | 93 | fed522320da95488 |
| `labels_candidates_all.csv` | 372 | 01d4dd13474f5d08 |
| `labels_instances_all.csv` | 1,020 | c2127f73f8090413 |
| `train_x3.csv` / `test_x3.csv` | 279 / 93 | ab9e9c4dd1cef64a / abcb6b78e28a3fa1 |
| `train_schedules_x3.csv` / `test_schedules_x3.csv` | 33 / 11 | c958716de928bd2d / e6c93d3441ee9752 |
| `workload_completeness.csv` | 44 | ffbcb68f1c324d54 |
| `offline_profile.json` | – | 8e156380f9caa006 |
| `schedules/*.yaml` | 44 files | – |

- **규칙:** `recompute_scores.py`의 `load_runs`와 `build_tables`를 import해 그대로 썼다. 열 이름은 CPU–GPU dataset과 같다(열 집합 일치 확인).
- **board 차이 4가지:** README에 있다.
  1. 완전성 기준은 2^N이 아니라 feasible 후보 수(3/7/13/21)다.
  2. `n_gpu`는 NPU에 놓인 instance 수다.
  3. instance identity는 instance 이름이다.
  4. `window_sec`는 30이다.
- **점검:** instance q_i 평균으로 다시 만든 D_macro와 label의 차는 최대 2.2e-16이다.
- **builder 호환 [확인, 코드 읽기만]:** `revision_exp/data.py`의 `family_key`/`rate_key`가 `<family>_x<k>.yaml`을 읽는다. feature 열 정규식에 `npu0|npu1`이 있다. 실제 builder 실행은 N2-B에서 한다.

## 5. [추정] / [확인 불가]

- [추정] DMA32가 run당 약 35 pages씩 감소한다(§1). 이번 규모(492 run)에서는 문제가 없다.
- [추정] 3× 평가 oracle이 R1과 다른 이유(YOLO big/small의 NPU 배정)는 isolated 처리량 차로 설명된다. 직접 분해 측정은 하지 않았다.
- [확인 불가] NPU 온도·clock: 노출되지 않음.
- [확인 불가] 1월 데이터와의 직접 비교: driver, window(10 s → 30 s), rate 규칙이 모두 달라 하지 않았다.

## 6. 불변 확인 [확인] (23:35 KST)

- **현재 branch (`ubuntu_gpu_fsrr_npu`, `multimodel-scheduling`):** tracked 수정은 작업 전과 같은 3개다.
  - pyc 2개
  - `ANTARA_NPU_DOCKER_RUN_GUIDE.md`: 1행 앞 `1`, N1 보고 때와 같음
  - 이번 작업으로 바뀐 것은 0이다.
- **`new_main` worktree:** `git status` 0행.
- **저장소 밖 파일:** 쓰지 않았다.
  - registry 21개 파일(모델 `.onnx`/`.o`, driver `.so`, host ONNX 2개 포함) sha256이 campaign 전후 같다.
  - 컨테이너는 종료 상태다.
- **port worktree:** 기존 파일 수정은 `revision_profiling/code/model_registry.yaml`(+109, N1과 같음)뿐이다.

## 7. Commit·push

§7 끝에 commit hash를 적는다. 포함 범위:
- 새 code 파일 15개: N1의 12개(`run_campaign_npu.py`는 `--plan` 모드 추가분 포함) + `run_plan_npu.py`, `integrity_npu.py`, `recompute_scores_npu.py`
- `model_registry.yaml`
- `revision_profiling_npu/` (결과 JSON, sidecar, dataset, 보고서)

제외: `*.events.npz`(659개, 40 MB), `imagenet-sample-images` symlink.
