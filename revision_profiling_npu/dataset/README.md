# dataset/ — CPU–NPU dataset (Phase N2-A campaign)

`code/recompute_scores_npu.py`로 생성. 입력은 `results/runs/`의 valid run 492개(invalid 0, 제외 workload 0).
규칙은 CPU–GPU dataset(`revision_profiling/dataset/README.md`, `recompute_scores.py`)과 같다. 함수(`load_runs`, `build_tables`)를 import해 그대로 쓴다.
CSV 파일 안에는 주석이 없고, 정의는 이 문서에만 둔다.

## 파일

| 파일 | 행 | 내용 |
|---|---|---|
| `train_x3.csv` | 279 | 1×·2×·4× 후보. 컬럼 `schedule_file, combination, json_content` (기존 loader 형식) |
| `test_x3.csv` | 93 | 3× 후보. 같은 형식 |
| `train_schedules_x3.csv`, `test_schedules_x3.csv` | 33, 11 | schedule 내용(`combination_x → instance → model, execution, infps, base_rate, beta, multiplier`) |
| `labels_train.csv`, `labels_test.csv` | 279, 93 | 후보별 label, 반복별 값, SE (열 이름은 CPU–GPU dataset과 동일) |
| `labels_candidates_all.csv` | 372 | 위 둘을 합친 전체 |
| `labels_instances_all.csv` | 1,020 | instance별 counter(반복 평균과 `*_rep0..2`). `registry_key` 열 추가 |
| `workload_completeness.csv` | 44 | workload별 기대 후보 수(feasible), 측정 후보 수 |
| `offline_profile.json` | – | isolated profile과 C^max (Phase N1 §4 사본) |
| `schedules/<family>_x<k>.yaml` | 44 | 실행에 쓴 executor schedule (β 0.25) |

## 이름 규칙

- **workload** = (family, rate).
  - `schedule_file = <family>_x<k>.yaml`이다.
  - family key는 `gen_schedules_antara/`의 이름이다. `rb`, `rs`, `yb`, `ys`, `rbrs`, `rsyb`, `ybys`, `rbrsyb`, `rbybys`, `rsybys`, `rbrsybys` (r = resnet50, y = yolov3, b/s = big/small).
  - fold 입력에는 `family` 열(또는 `schedule_file`에서 `_x<k>.yaml`을 뗀 값)을 쓴다.
- **instance identity**는 schedule의 instance 이름이다(`resnet50_big`, `resnet50_small`, `yolov3_big`, `yolov3_small`). json_content와 schedule CSV의 `model`도 이 이름이다(1월 Antara 데이터와 같은 규약). 실행에 쓴 registry key(`antara_<m>_cpu`, `antara_<m>_<v>_npu`)는 `labels_instances_all.csv`의 `registry_key`에 있다.
- **device**는 `cpu`, `npu0`, `npu1`이다(json_content에서는 대문자). NPU당 instance는 최대 1개이다.
- **combination 번호**는 `gen_schedules_antara/model_schedules_<family>.yaml`의 번호 그대로다(`combination_1`이 전부 CPU). CPU–GPU dataset처럼 bit mask가 아니다.
- `placement` 열은 `instance:device`를 `,`로 이은 것이다.
- **rate**: `lambda_i = 0.25 × C^max[model] × k`이다. C^max는 resnet50 52.50 req/s, yolov3 14.27 req/s다. 배치·변종과 무관하며, workload 안에서 모든 후보의 rate가 같다.

## Label 정의 (CPU–GPU dataset과 같음)

측정 구간은 warm-up 5초 뒤 **30초**다(CPU–GPU는 10초). instance i마다 generated_i, completed_i, dropped_i를 센다.

- **반복별 값:**
  - `F`: 완료 요청 합 / 30초
  - `R_agg`: drop 합 / 30초
  - `D_macro`: q_i = dropped_i / generated_i의 instance 평균
  - `D_max`: q_i의 최댓값
- **반복 평균을 먼저 낸다.** 3× N≥3 평가 workload(rbrsyb, rbybys, rsybys, rbrsybys)는 3회, 나머지는 1회다.
- **workload 안에서 정규화한다.**
  - `F_max`, `R_max`는 그 workload 모든 후보의 평균값 중 최댓값이다.
  - `throughput_norm` (T) = `F_mean / F_max`
  - `drop_rate_norm` (D, 기본) = `D_macro_mean` (정규화하지 않음)
  - `drop_agg_norm` = `R_agg_mean / R_max`
  - `drop_max` = `D_max_mean`
- **점수:** `S_macro_a0.3 = T − 0.3 × D_macro` (기본). `S_macro_a0.3_rep k`는 평균 `F_max`로 정규화한 반복별 점수다. `*_se`는 반복 표준편차 / √반복 수다.
- **board 차이로 뜻이 바뀐 열:** `n_gpu` = 가속기(npu0/npu1)에 놓인 instance 수. 열 이름은 호환을 위해 유지했다.
- **json_content `window_sec`** = 30.0이다.

## Split

- 학습: 1·2·4× (33 workloads, 279행)
- 평가: 3× (11 workloads, 93행). 그중 N≥3 4 workloads, 60행이 Table 3(b) 대상이다.

## 검증 (생성 시 assert/확인)

- workload 44개 모두 feasible 후보 100%가 valid다.
- β = 0.25 단일 값, window 30 s.
- instance q_i 평균으로 다시 만든 D_macro와 label의 차는 최대 2.2e-16이다.
- labels 열 집합은 CPU–GPU `labels_train.csv`와 같다.
- feature는 만들지 않았다. GPU 보드의 builder(`revision_exp/data.py`)로 만든다.
