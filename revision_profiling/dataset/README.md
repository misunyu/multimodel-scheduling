# dataset/ — 재측정 CPU-GPU dataset (Phase 2 campaign)

`code/recompute_scores.py --results results --out dataset`로 생성. 입력은 `results/`의 valid run 2,976개(invalid 0, 제외 workload 0).
CSV 파일 안에는 주석이 없다. 정의는 이 문서에만 둔다.

## 파일

| 파일 | 내용 |
|---|---|
| `train_x3.csv` | 1×·2×·4× 후보 1,512행. 컬럼 `schedule_file, combination, json_content` (기존 loader 형식) |
| `test_x3.csv` | 3× 후보 504행. 같은 형식 |
| `train_schedules_x3.csv`, `test_schedules_x3.csv` | schedule 내용. 기존 parser가 읽는 구조(`combination_xxx → instance → model, execution, infps`) |
| `labels_train.csv`, `labels_test.csv` | 후보별 label과 반복별 값·SE |
| `labels_candidates_all.csv` | 위 둘을 합친 전체 2,016 후보 |
| `labels_instances_all.csv` | instance별 counter (17,696 instance-run을 후보 단위로 평균) |
| `workload_completeness.csv` | workload별 후보 수와 반복 수 |
| `feature_columns.txt`, `compat_check.json` | `code/check_dataset_compat.py --final` 결과 |

## 이름 규칙

- workload = (family, rate). `schedule_file = <family>_x<m>.yaml`, family는 모델 이름을 정렬해 `+`로 이은 것.
- instance 순서 = 정렬된 모델 이름. `combination_<mask>`의 bit i가 1이면 instance i가 GPU. `combination_000`은 전부 CPU.
- rate: `lambda_i = 0.1 × (모델의 GPU isolated throughput) × m`. 배치와 무관하다.

## Label 정의

한 run의 측정 구간은 warm-up 5초 뒤 10초다. instance i에 대해 generated_i, completed_i, dropped_i를 센다.

- 후보의 반복별 값: `F`(완료 요청 합 / 10초), `R_agg`(drop 합 / 10초), `D_macro`(instance별 dropped_i / generated_i의 평균), `D_max`(그 최댓값).
- **반복 평균을 먼저 낸다.** 3×의 N≥3 workload는 3회, 나머지는 1회다. `F_mean`, `R_agg_mean`, `D_macro_mean`, `D_max_mean`.
- 그다음 workload 안에서 정규화한다. `F_max`, `R_max`는 그 workload의 모든 후보에 대한 `F_mean`, `R_agg_mean`의 최댓값.

| 컬럼 | 정의 | 정규화 |
|---|---|---|
| `throughput_norm` (T) | `F_mean / F_max` | workload 최댓값으로 나눔 |
| `drop_rate_norm` (D, 기본) | `D_macro_mean` | **정규화하지 않음** (이미 0–1의 비율) |
| `drop_agg_norm` | `R_agg_mean / R_max` | workload 최댓값으로 나눔 |
| `drop_max` | `D_max_mean` | 정규화하지 않음 |

**D_macro와 D_agg의 차이.** D_macro는 절대 비율이라 workload가 달라도 같은 뜻이다(0.1 = instance 평균 10% drop).
D_agg는 그 workload에서 가장 많이 drop한 후보를 1로 둔 상대값이라, drop이 거의 없는 workload에서도 최악 후보가 1이 된다.
또 D_agg는 요청 수가 많은 instance가 지배하고, D_macro는 모든 instance를 같은 무게로 센다(GPT-2처럼 rate가 낮은 instance의 drop이 드러난다).
학습과 평가의 기본 D는 D_macro다.

- 점수: `S_macro_a0.3 = T − 0.3 × D_macro` (기본), `S_agg_a0.3`, `S_max_a0.3`.
- 반복별 값은 `*_rep0..2`, 표준오차는 `*_se`(반복 표준편차 / √반복 수; 반복 1회면 비어 있음).
  `S_macro_a0.3_rep k`는 평균에서 얻은 `F_max`로 정규화한 반복별 점수다.
- `json_content.derived`에는 `throughput_norm`, `drop_rate_norm`(= D_macro), `drop_agg_norm`, `drop_max`가 들어 있다.
- 품질 지표(label 보정에 쓰지 않음): `min_realization`(generated / scheduled의 최솟값), `max_missed_ratio`.

## 호환성 검증 (`compat_check.json`, pass = true)

기존 `revision_exp/data.py`의 loader와 feature builder를 수정 없이 사용했다.

| 항목 | 결과 |
|---|---|
| loader 통과 | train 1,512행 / 66 workloads, test 504행 / 22 workloads, 중복 제거된 행 0 |
| builder가 읽은 infps = 실제 rate | 11,840개 값 대조, 불일치 0 |
| 배치·combination 번호 대응 | 불일치 0 |
| T = throughput_norm, D = D_macro | 일치 (최대 차 1e-16) |
| 후보 수 = 2^N | 모든 workload |
| feature 수 | 204 (기존 235) |
| 기존에만 있는 feature | 31개: TinyLLaMA(hash_17) 15개, 기존 데이터에만 있던 모델 쌍 hash 16개 |
| 새로 생긴 feature | 0 |
| label·counter 이름이 feature에 섞임 | 없음 |
| 학습 데이터에서 모두 0인 feature | 60개 (전부 NPU 관련; 이 데이터에는 NPU 배치가 없음) |

Split 점검은 `SPLIT_CHECK.md`.
