# PRECHECK — Phase 4 §3.0 (Phase 3 최종 규칙 재현, 측정 전)

실행 2026-10-19 15:59:32. 코드 `revision_exp_adapter/run_intermediate.py --precheck`(이 Phase의 평가 함수 `evaluate_set()`을 Phase 2 3× 데이터에 적용). 입력: `revision_profiling/dataset/{test_x3,test_schedules_x3,labels_test}.csv`, artifact는 아래.

**결과: PASS — campaign 시작 가능**

## 재현 표 (3×, N≥3, 12 workloads)

| 방법 | Avg | Regret | Top-1 | Top-5 | 목표 |
|---|---|---|---|---|---|
| Ours (XGBoost Full) | 0.9930 | 0.0044 | 7/12 | 11/12 | 0.9930 / 0.0044 / 7/12 / 11/12 |
| Per-instance predictor | 0.9868 | 0.0106 | 6/12 | 11/12 | 0.9868 / 0.0106 / 6/12 / 11/12 |
| Latency-based (= All-GPU) | 0.9727 | 0.0247 | 6/12 |  | 0.9727 / 0.0247 / 6/12 |
| All-GPU | 0.9727 | 0.0247 | 6/12 |  | (All-GPU, 참고) |
| Best-of-5 | 0.9404 ± 0.0210 | 0.0570 | 0.402 (평균 적중률) | N/A | 0.9404 ± 0.0210 / 0.402 (mean hit rate over 1,000 samplings) |
| Oracle | 0.9974 | 0.0000 | 12/12 | 12/12 | 0.9974 / 0 / 12/12 |

## 검사

| 검사 | 결과 |
|---|---|
| T_sha_equals_phase2_config_frozen | PASS |
| D_sha_equals_phase2_config_frozen | PASS |
| ours_avg_0.993 | PASS |
| ours_regret_0.0044 | PASS |
| ours_top1_7/12 | PASS |
| ours_top5_11/12 | PASS |
| inst_avg_0.9868 | PASS |
| inst_regret_0.0106 | PASS |
| inst_top1_6/12 | PASS |
| inst_top5_11/12 | PASS |
| lat_avg_0.9727 | PASS |
| lat_regret_0.0247 | PASS |
| lat_top1_6/12 | PASS |
| orc_avg_0.9974 | PASS |
| orc_regret_0.0 | PASS |
| orc_top1_12/12 | PASS |
| bo5_avg_equals_phase3 | PASS |
| bo5_regret_equals_phase3 | PASS |
| bo5_top1_rate_equals_phase3 | PASS |
| bo5_std_equals_phase2_table3a | PASS |
| ours_candidate_predictions_equal_phase3 (after the same CSV round trip: max abs diff 0) | PASS |
| per_instance_candidate_predictions_equal_phase3(max abs diff <= 1e-9) | PASS |
| per_workload_selections_equal_phase3 (Ours, Per-instance, Latency, Oracle; 48 rows) | PASS |
| latency_targets_all_GPU | PASS |
| latency_selection_equals_all_gpu | PASS |
| features_selected_from_schedule_only (stripped windows identical) | PASS |
| no feature column outside the frozen 204 | PASS |

## Artifact (실제 파일명)

| 파일 | sha256 | 대조 |
|---|---|---|
| `revision_exp/out_reprofiled/artifacts/multi/T.json` | `b6a7ce2d159f9e9874a93bc1ac1a4124a9b1aa642db2600468d2e4b45c3b20fa` | Phase 2 `config_frozen.json` artifact_sha256, Phase 3 `config_frozen.json` xgb_full_artifact와 같음 |
| `revision_exp/out_reprofiled/artifacts/multi/D.json` | `ab19e2ad2d0614b919799ff7d5dc0c29d3793c4b1859732ac56bfa3ec8bf9334` | 같음 |
| `revision_exp/out_phase3/artifacts/per_instance/t_ratio.json` | `2a19ee15dc58a8ee1d92013c552310baef18e838541758ef3d081778874388de` | Phase 3 `config_frozen.json`·`REPORT_PHASE3.md`에 per-instance artifact hash 기록이 없다. 대신 후보 504개의 F̂·D̂·T̂_select·Ŝ가 Phase 3 `per_instance_candidates_x3.csv`와 같음을 확인했다 |
| `revision_exp/out_phase3/artifacts/per_instance/q_i.json` | `b28007d87cfffd8530e914f4518d19f625fa7eac1258425c21ab24c31880bbcc` | 같음 |

- 후보별 예측 최대 차: Ours(메모리 값 vs Phase 3 CSV) {'T': 2.220446049250313e-16, 'D': 9.71445146547012e-17, 'S': 1.1102230246251565e-16}, 같은 CSV 저장·읽기를 거친 뒤 {'T': 0.0, 'D': 0.0, 'S': 0.0}; Per-instance {'F_hat': 2.2737367544323206e-13, 'D_hat': 1.1102230246251565e-16, 'T_hat_select': 1.1102230246251565e-16, 'S_hat': 1.1102230246251565e-16}.
  첫 실행에서 Ours 검사를 '메모리 값과 CSV 값의 차 0'으로 두어 FAIL(최대 2.2e-16)이 났다. 두 값을 같은 `to_csv`/`read_csv`로 거치면 504개 모두 차 0이라, 차이는 pandas CSV 실수 표기의 1 ulp 반올림이다. 검사를 같은 경로의 값끼리 비교하도록 고쳤다(평가 규칙·예측값은 바꾸지 않음).
- Workload별 선택 ID와 실측 S(동점 처리 포함)를 Phase 3 `per_workload_all.csv`와 48행 대조했다(`precheck_selections.csv`).
- Best-of-5 std 대조값은 Phase 2 `table3a.csv`(Phase 3 표에는 std가 없다). seed 규칙: (trial, 평가 집합의 정렬된 workload 순번). Phase 3은 3× 22 workloads 전체를 넣어 돌렸으므로 이 precheck도 22개를 넣었다. 중간 rate에서는 dataset에 N≥3 12 workloads만 있으므로 순번이 0–11이다.
- Ours의 feature는 측정값을 뺀 schedule-only window(model, execution)로 다시 만든 것과 bit 단위로 같다. 204열 밖의 feature는 생기지 않았다.
