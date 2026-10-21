# Campaign 무결성 점검 — Phase 4b 반복 측정 (rep1, rep2; rep0 = Phase 4)

`intermediate/code_copy_repeats/integrity_repeats.py`가 생성. 상세 표는 `logs/integrity_repeats/`. (Phase 4 rep0 폴더 `results/`는 보호 대상이라 이 보고서를 `logs/`에 둔다.)

## 1. Run 수·시간

| rep | source | planned | valid | missing | attempts | retried_runs | non_ok_attempts | rejected_files | failed_runs_csv | invalid_left | run_hours | wall_hours | first_start | last_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Phase 4 | 1440 | 1440 | 0 | 1440 | 0 | 0 | 0 | 0 | 0 | 7.435 | 7.435 | 2026-10-19 16:01:52 | 2026-10-19 23:27:57 |
| 1 | Phase 4b | 1440 | 1440 | 0 | 1440 | 0 | 0 | 0 | 0 | 0 | 7.45 | 7.45 | 2026-10-20 14:51:39 | 2026-10-20 22:18:38 |
| 2 | Phase 4b | 1440 | 1440 | 0 | 1440 | 0 | 0 | 0 | 0 | 0 | 7.441 | 7.441 | 2026-10-20 22:18:38 | 2026-10-21 05:45:06 |

- 계획에 없는 결과 파일 0개. 세션 기록: 2026-10-20 14:51:39 START reason='initial start (PRECHECK_3REP pass, RUN_PLAN_REPEATS pass)' planned=2880 todo=2880 est_h=14.85 pid=1512901 / 2026-10-20 14:51:39 BLOCK rep1 start (runtime sha256 = Phase 4, offline profile = git HEAD) todo_in_block=1440 / 2026-10-20 22:18:38 BLOCK rep2 start (runtime sha256 = Phase 4, offline profile = git HEAD) todo_in_block=1440 / 2026-10-21 05:45:06 END executed=2880 retries=0 failed=0 hours=14.89

N별 run 시간(rep1, rep2):

| rep | N | runs | p50 | p95 | formula |
|---|---|---|---|---|---|
| 1 | 3 | 144 | 17.71 | 18.87 | 17.11 |
| 1 | 4 | 48 | 17.83 | 18.57 | 17.58 |
| 1 | 5 | 96 | 17.98 | 18.74 | 18.05 |
| 1 | 6 | 384 | 18.06 | 18.84 | 18.52 |
| 1 | 7 | 768 | 19.05 | 20.05 | 18.99 |
| 2 | 3 | 144 | 17.67 | 18.84 | 17.11 |
| 2 | 4 | 48 | 17.78 | 18.68 | 17.58 |
| 2 | 5 | 96 | 17.99 | 18.77 | 18.05 |
| 2 | 6 | 384 | 18.06 | 18.81 | 18.52 |
| 2 | 7 | 768 | 18.99 | 20.05 | 18.99 |

## 2. 완전성 ((workload, rep)별 feasible 후보 100%)

- 36 workloads 중 세 반복 모두 완전한 것 36개.
- **3회 평균 집합에서 제외되는 workload: 0개.**

## 3. Runtime·conservation·trace·도착 (rep별)

- 모든 valid run의 `loaded_modules`가 `revision_profiling/code/` 안: 예. Runner가 각 블록 시작 시 runtime 파일 10개의 sha256이 Phase 4 `logs/runtime_sha256.json`과 같음을 확인했다(`logs/runtime_sha256_rep{1,2}.json`).

| rep | runs | instance_runs | conservation_flow | trace_check | snapshot_flag | open_loop_beta_0.1 | realization_min | realization_p1 | realization_median | instances_realization_lt_0.95 | missed_ratio_max | missed_ratio_p99 | instances_missed_gt_0 | instances_missed_gt_5pct | tick_minus_rate_x10_min | tick_minus_rate_x10_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1440 | 8784 | {0: 8784} | {'ok': 8784} | 0 | True | 0.9732 | 0.9889 | 1 | 0 | 0.02679 | 0.01106 | 1158 | 0 | -0.6 | 0.6 |
| 1 | 1440 | 8784 | {0: 8784} | {'ok': 8784} | 0 | True | 0.9719 | 0.9885 | 1 | 0 | 0.02807 | 0.01148 | 1162 | 0 | -0.6 | 0.6 |
| 2 | 1440 | 8784 | {0: 8784} | {'ok': 8784} | 1 | True | 0.9698 | 0.9893 | 1 | 0 | 0.03019 | 0.01067 | 1156 | 0 | -0.6 | 0.6 |

## 4. Generated 차이 (품질 지표, Phase 2 규칙: ≥200 tick 상대 2%, <200 tick 절대 2건) — rep별

| rep | workloads | violation_workloads | rel_median | rel_max | abs_max |
|---|---|---|---|---|---|
| 0 | 36 | 2 | 0.001566 | 0.02699 | 0 |
| 1 | 36 | 3 | 0.00148 | 0.02828 | 0 |
| 2 | 36 | 3 | 0.001573 | 0.03043 | 0 |

위반 (workload, rep):

| workload | rep | N | gen_mismatch_max | gen_mismatch_abs_max | violations |
|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12_x3.5 | 1 | 6 | 0.02336 | 0 | 1 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12_x3.5 | 2 | 6 | 0.02165 | 0 | 1 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19_x3.5 | 0 | 7 | 0.02229 | 0 | 1 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19_x3.5 | 1 | 7 | 0.02529 | 0 | 1 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19_x3.5 | 2 | 7 | 0.02057 | 0 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4_x3.5 | 0 | 7 | 0.02699 | 0 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4_x3.5 | 1 | 7 | 0.02828 | 0 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4_x3.5 | 2 | 7 | 0.03043 | 0 | 1 |

## 5. 반복 편차 (후보별 3회 SE = 표본표준편차(ddof=1)/√3)

`S_r = F_r / F_max − 0.3·D_macro,r`, `F_max` = 3회 평균의 공통 최대값(Phase 2 정의, PRECHECK_3REP 0-1). Phase 2 3×(N≥3, 480 후보)와 나란히.

| rate | metric | candidates | median | p90 | max |
|---|---|---|---|---|---|
| x1.5 | F (fps) | 480 | 0.1856 | 0.5699 | 2.237 |
| x1.5 | F relative (SE / F_bar) | 480 | 0.0003599 | 0.001461 | 0.0123 |
| x1.5 | D_macro | 480 | 0.0005665 | 0.00218 | 0.01252 |
| x1.5 | S | 480 | 0.0005202 | 0.001883 | 0.01467 |
| x2.5 | F (fps) | 480 | 0.6014 | 2.489 | 9.75 |
| x2.5 | F relative (SE / F_bar) | 480 | 0.0007418 | 0.003088 | 0.01389 |
| x2.5 | D_macro | 480 | 0.0005951 | 0.001673 | 0.006289 |
| x2.5 | S | 480 | 0.0008368 | 0.002844 | 0.01036 |
| x3.5 | F (fps) | 480 | 1.847 | 7.114 | 29.04 |
| x3.5 | F relative (SE / F_bar) | 480 | 0.001708 | 0.006255 | 0.02553 |
| x3.5 | D_macro | 480 | 0.0009744 | 0.002862 | 0.01379 |
| x3.5 | S | 480 | 0.001698 | 0.005799 | 0.02353 |
| x3 (Phase 2) | F (fps) | 480 | 0.9046 | 2.859 | 9.151 |
| x3 (Phase 2) | F relative (SE / F_bar) | 480 | 0.0009602 | 0.003009 | 0.01353 |
| x3 (Phase 2) | D_macro | 480 | 0.0006453 | 0.001617 | 0.008922 |
| x3 (Phase 2) | S | 480 | 0.0009704 | 0.002678 | 0.01303 |

## 6. 반복별 oracle 일치

각 반복의 자체 정규화(`F_r / max F_r − 0.3·D_macro,r`, Phase 2 integrity.py 규칙)와 동점 허용오차 1e-9로 oracle 후보 **집합**을 구했다. `common_optimal_exists` = 세 반복 모두에서 최적인 후보가 있는 workload 수, `representative_ids_equal` = 대표 ID(가장 작은 combination 번호)가 셋 다 같은 수.

| rate | workloads | common_optimal_exists | representative_ids_equal | mean_label_oracle_in_common |
|---|---|---|---|---|
| x1.5 | 12 | 12 | 9 | 12 |
| x2.5 | 12 | 11 | 11 | 11 |
| x3.5 | 12 | 12 | 12 | 12 |

Workload별(`oracle_by_rep.csv`):

| rate | family | N | oracle_rep0 | n_oracle_rep0 | oracle_rep1 | n_oracle_rep1 | oracle_rep2 | n_oracle_rep2 | common_optimal_exists | representative_ids_equal | oracle_mean_label | n_oracle_mean_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | combination_109 | 2 | combination_109 | 1 | combination_109 | 2 | True | True | combination_109 | 1 |
| x2.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | combination_079 | 1 | combination_079 | 1 | combination_079 | 1 | True | True | combination_079 | 1 |
| x3.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | combination_108 | 1 | combination_108 | 1 | combination_108 | 1 | True | True | combination_108 | 1 |
| x1.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | combination_045 | 4 | combination_015 | 4 | combination_015 | 5 | True | False | combination_045 | 3 |
| x2.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | combination_015 | 1 | combination_015 | 1 | combination_015 | 1 | True | True | combination_015 | 1 |
| x3.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | combination_046 | 1 | combination_046 | 1 | combination_046 | 1 | True | True | combination_046 | 1 |
| x1.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | combination_013 | 4 | combination_013 | 4 | combination_013 | 4 | True | True | combination_013 | 4 |
| x2.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | combination_013 | 1 | combination_013 | 2 | combination_013 | 2 | True | True | combination_013 | 1 |
| x3.5 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | combination_013 | 1 | combination_013 | 1 | combination_013 | 1 | True | True | combination_013 | 1 |
| x1.5 | gpt2+mnasnet+resnet50+resnext50 | 4 | combination_013 | 2 | combination_013 | 2 | combination_013 | 2 | True | True | combination_013 | 2 |
| x2.5 | gpt2+mnasnet+resnet50+resnext50 | 4 | combination_013 | 2 | combination_013 | 1 | combination_013 | 1 | True | True | combination_013 | 1 |
| x3.5 | gpt2+mnasnet+resnet50+resnext50 | 4 | combination_013 | 1 | combination_013 | 1 | combination_013 | 1 | True | True | combination_013 | 1 |
| x1.5 | gpt2+mnasnet+resnet50 | 3 | combination_005 | 2 | combination_005 | 2 | combination_005 | 1 | True | True | combination_005 | 1 |
| x2.5 | gpt2+mnasnet+resnet50 | 3 | combination_005 | 2 | combination_005 | 1 | combination_005 | 1 | True | True | combination_005 | 1 |
| x3.5 | gpt2+mnasnet+resnet50 | 3 | combination_005 | 2 | combination_005 | 1 | combination_005 | 2 | True | True | combination_005 | 1 |
| x1.5 | gpt2+resnext50+vgg19 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x2.5 | gpt2+resnext50+vgg19 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x3.5 | gpt2+resnext50+vgg19 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x1.5 | gpt2+resnext50+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x2.5 | gpt2+resnext50+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x3.5 | gpt2+resnext50+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x1.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | combination_118 | 4 | combination_127 | 1 | combination_111 | 5 | True | False | combination_127 | 1 |
| x2.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | combination_119 | 1 | combination_119 | 1 | combination_119 | 1 | True | True | combination_119 | 1 |
| x3.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | combination_118 | 1 | combination_118 | 1 | combination_118 | 1 | True | True | combination_118 | 1 |
| x1.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | combination_054 | 4 | combination_038 | 5 | combination_039 | 5 | True | False | combination_054 | 4 |
| x2.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | combination_054 | 1 | combination_055 | 1 | combination_063 | 1 | False | False | combination_054 | 1 |
| x3.5 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | combination_054 | 1 | combination_054 | 1 | combination_054 | 1 | True | True | combination_054 | 1 |
| x1.5 | mnasnet+resnet50+resnext50 | 3 | combination_006 | 2 | combination_006 | 2 | combination_006 | 2 | True | True | combination_006 | 2 |
| x2.5 | mnasnet+resnet50+resnext50 | 3 | combination_006 | 1 | combination_006 | 2 | combination_006 | 1 | True | True | combination_006 | 1 |
| x3.5 | mnasnet+resnet50+resnext50 | 3 | combination_006 | 1 | combination_006 | 1 | combination_006 | 1 | True | True | combination_006 | 1 |
| x1.5 | mnasnet+resnext50+yolov4 | 3 | combination_006 | 2 | combination_006 | 2 | combination_006 | 2 | True | True | combination_006 | 2 |
| x2.5 | mnasnet+resnext50+yolov4 | 3 | combination_006 | 1 | combination_006 | 1 | combination_006 | 1 | True | True | combination_006 | 1 |
| x3.5 | mnasnet+resnext50+yolov4 | 3 | combination_006 | 1 | combination_006 | 1 | combination_006 | 1 | True | True | combination_006 | 1 |
| x1.5 | resnet50+vgg19+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x2.5 | resnet50+vgg19+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |
| x3.5 | resnet50+vgg19+yolov4 | 3 | combination_007 | 1 | combination_007 | 1 | combination_007 | 1 | True | True | combination_007 | 1 |

## 7. 시간 경과 이상 징후 점검 (drift 검증 아님)

GPU 상태(1분 표본, Phase 4b 시작부터 시간대별):

| hour | samples | temp_mean | temp_max | clock_mean | clock_min | power_mean | load1_mean |
|---|---|---|---|---|---|---|---|
| 0 | 60 | 46.73 | 52 | 2506 | 195 | 118.4 | 10.45 |
| 1 | 60 | 48.12 | 55 | 2544 | 2520 | 134.5 | 10.62 |
| 2 | 60 | 45.62 | 51 | 2406 | 195 | 103 | 8.043 |
| 3 | 60 | 45.65 | 52 | 2462 | 195 | 98.72 | 8.88 |
| 4 | 60 | 46.57 | 51 | 2513 | 817 | 130.5 | 9.682 |
| 5 | 60 | 45.2 | 51 | 2483 | 660 | 122.2 | 10.83 |
| 6 | 60 | 44.47 | 48 | 2457 | 195 | 103.9 | 8.441 |
| 7 | 60 | 44.48 | 48 | 2503 | 195 | 115.5 | 8.131 |
| 8 | 60 | 44.25 | 48 | 2507 | 195 | 105.1 | 9.006 |
| 9 | 60 | 43.6 | 49 | 2509 | 472 | 112 | 8.366 |
| 10 | 60 | 45.68 | 51 | 2542 | 2520 | 141.7 | 10.73 |
| 11 | 59 | 45.12 | 49 | 2540 | 2512 | 138.3 | 10.96 |
| 12 | 60 | 44.63 | 50 | 2492 | 795 | 123.4 | 10.58 |
| 13 | 60 | 44.7 | 49 | 2359 | 307 | 125.5 | 10.68 |
| 14 | 54 | 43.17 | 47 | 2357 | 195 | 98.28 | 6.825 |

전부-GPU 후보의 반복별 F(rep0 = Phase 4, 2026-10-19). `flag` = |F_r − F̄| / F̄ > 5%(표시 규칙만, label 보정 없음):

| family | N | rate | rep | start_ts | F_r | F_bar | dev_from_mean | D_macro_r | flag |
|---|---|---|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 | 3 | x1.5 | 0 | 2026-10-19 23:17:32 | 187.7 | 187.6 | 0.0003553 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x1.5 | 1 | 2026-10-20 18:44:20 | 187.7 | 187.6 | 0.0003553 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x1.5 | 2 | 2026-10-21 05:34:44 | 187.5 | 187.6 | -0.0007106 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x2.5 | 0 | 2026-10-19 23:20:47 | 312.9 | 312.8 | 0.0002131 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x2.5 | 1 | 2026-10-20 18:40:48 | 312.8 | 312.8 | -0.0001066 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x2.5 | 2 | 2026-10-21 05:35:37 | 312.8 | 312.8 | -0.0001066 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x3.5 | 0 | 2026-10-19 23:17:15 | 438.1 | 437.9 | 0.0004567 | 0 |  |
| gpt2+mnasnet+resnet50 | 3 | x3.5 | 1 | 2026-10-20 18:46:07 | 437.6 | 437.9 | -0.0006851 | 0.0006743 |  |
| gpt2+mnasnet+resnet50 | 3 | x3.5 | 2 | 2026-10-21 05:32:57 | 438 | 437.9 | 0.0002284 | 0.0001204 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x1.5 | 0 | 2026-10-19 23:08:19 | 251.2 | 251.2 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x1.5 | 1 | 2026-10-20 18:49:58 | 251.2 | 251.2 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x1.5 | 2 | 2026-10-20 22:39:40 | 251.2 | 251.2 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x2.5 | 0 | 2026-10-19 22:59:57 | 418.8 | 418.7 | 0.0001592 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x2.5 | 1 | 2026-10-20 18:49:05 | 418.7 | 418.7 | -7.961e-05 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x2.5 | 2 | 2026-10-20 22:38:28 | 418.7 | 418.7 | -7.961e-05 | 0.0002275 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x3.5 | 0 | 2026-10-19 23:10:59 | 579.4 | 581 | -0.002754 | 0.008022 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x3.5 | 1 | 2026-10-20 18:54:27 | 582.1 | 581 | 0.001893 | 0.004479 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | x3.5 | 2 | 2026-10-20 22:35:47 | 581.5 | 581 | 0.0008606 | 0.00569 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x1.5 | 0 | 2026-10-19 19:14:08 | 378.4 | 378.4 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x1.5 | 1 | 2026-10-20 17:15:22 | 378.4 | 378.4 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x1.5 | 2 | 2026-10-21 04:55:35 | 378.4 | 378.4 | 2.22e-16 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x2.5 | 0 | 2026-10-19 19:28:02 | 630.6 | 630.5 | 0.0001057 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x2.5 | 1 | 2026-10-20 17:32:37 | 630.4 | 630.5 | -0.0002115 | 9.434e-05 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x2.5 | 2 | 2026-10-21 05:21:18 | 630.6 | 630.5 | 0.0001057 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x3.5 | 0 | 2026-10-19 19:13:33 | 871.2 | 858.9 | 0.01436 | 0.01055 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x3.5 | 1 | 2026-10-20 17:38:21 | 856 | 858.9 | -0.003338 | 0.02404 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | x3.5 | 2 | 2026-10-21 05:04:04 | 849.4 | 858.9 | -0.01102 | 0.0297 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x1.5 | 0 | 2026-10-19 22:40:33 | 580 | 580 | 0 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x1.5 | 1 | 2026-10-20 21:47:35 | 580 | 580 | 0 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x1.5 | 2 | 2026-10-21 00:40:25 | 580 | 580 | 0 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x2.5 | 0 | 2026-10-19 22:22:15 | 922.6 | 923.5 | -0.001011 | 0.02288 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x2.5 | 1 | 2026-10-20 22:12:17 | 926.5 | 923.5 | 0.003212 | 0.02102 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x2.5 | 2 | 2026-10-20 23:55:18 | 921.5 | 923.5 | -0.002202 | 0.0235 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x3.5 | 0 | 2026-10-19 22:04:57 | 1115 | 1106 | 0.008201 | 0.1123 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x3.5 | 1 | 2026-10-20 21:28:57 | 1094 | 1106 | -0.0107 | 0.1222 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | x3.5 | 2 | 2026-10-21 00:03:33 | 1108 | 1106 | 0.002503 | 0.1159 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x1.5 | 0 | 2026-10-19 17:25:55 | 664.5 | 663.8 | 0.001004 | 7.086e-05 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x1.5 | 1 | 2026-10-20 19:39:53 | 663.6 | 663.8 | -0.0003515 | 0.0007086 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x1.5 | 2 | 2026-10-21 04:07:18 | 663.4 | 663.8 | -0.0006528 | 0.0008503 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x2.5 | 0 | 2026-10-19 16:03:29 | 1001 | 1004 | -0.002756 | 0.06042 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x2.5 | 1 | 2026-10-20 20:24:30 | 1012 | 1004 | 0.0085 | 0.0544 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x2.5 | 2 | 2026-10-21 03:06:56 | 998.2 | 1004 | -0.005744 | 0.06362 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x3.5 | 0 | 2026-10-19 16:36:53 | 1119 | 1067 | 0.04849 | 0.1917 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x3.5 | 1 | 2026-10-20 20:35:50 | 1042 | 1067 | -0.02368 | 0.2318 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | x3.5 | 2 | 2026-10-21 04:42:40 | 1040 | 1067 | -0.02481 | 0.2341 |  |
| gpt2+resnext50+vgg19 | 3 | x1.5 | 0 | 2026-10-19 22:51:38 | 151.3 | 151.3 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x1.5 | 1 | 2026-10-20 21:12:53 | 151.3 | 151.3 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x1.5 | 2 | 2026-10-21 05:37:41 | 151.3 | 151.3 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x2.5 | 0 | 2026-10-19 22:48:13 | 252.1 | 252.1 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x2.5 | 1 | 2026-10-20 21:11:40 | 252.1 | 252.1 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x2.5 | 2 | 2026-10-21 05:42:01 | 252.1 | 252.1 | 0 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x3.5 | 0 | 2026-10-19 22:47:37 | 353 | 352.9 | 0.0002834 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x3.5 | 1 | 2026-10-20 21:06:22 | 353 | 352.9 | 0.0002834 | 0 |  |
| gpt2+resnext50+vgg19 | 3 | x3.5 | 2 | 2026-10-21 05:42:19 | 352.7 | 352.9 | -0.0005667 | 0.0001689 |  |
| gpt2+resnext50+yolov4 | 3 | x1.5 | 0 | 2026-10-19 19:09:19 | 86.7 | 86.7 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x1.5 | 1 | 2026-10-20 17:03:26 | 86.7 | 86.7 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x1.5 | 2 | 2026-10-20 22:24:59 | 86.7 | 86.7 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x2.5 | 0 | 2026-10-19 19:05:24 | 144.8 | 144.8 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x2.5 | 1 | 2026-10-20 17:03:44 | 144.8 | 144.8 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x2.5 | 2 | 2026-10-20 22:20:07 | 144.8 | 144.8 | 0 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x3.5 | 0 | 2026-10-19 19:05:42 | 202.5 | 202.4 | 0.0003293 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x3.5 | 1 | 2026-10-20 17:02:14 | 202.4 | 202.4 | -0.0001647 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | x3.5 | 2 | 2026-10-20 22:19:32 | 202.4 | 202.4 | -0.0001647 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x1.5 | 0 | 2026-10-19 23:22:11 | 248 | 248 | 0 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x1.5 | 1 | 2026-10-20 21:17:04 | 248 | 248 | 0 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x1.5 | 2 | 2026-10-21 05:30:20 | 248 | 248 | 0 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x2.5 | 0 | 2026-10-19 23:25:45 | 413.4 | 413.4 | -8.063e-05 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x2.5 | 1 | 2026-10-20 21:19:00 | 413.5 | 413.4 | 0.0001613 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x2.5 | 2 | 2026-10-21 05:28:24 | 413.4 | 413.4 | -8.063e-05 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x3.5 | 0 | 2026-10-19 23:25:28 | 578.9 | 578.9 | 0 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x3.5 | 1 | 2026-10-20 21:14:02 | 578.9 | 578.9 | 0 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | x3.5 | 2 | 2026-10-21 05:25:39 | 578.9 | 578.9 | 0 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x1.5 | 0 | 2026-10-19 18:39:19 | 661.4 | 661.4 | 0 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x1.5 | 1 | 2026-10-20 18:10:29 | 661.4 | 661.4 | 0 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x1.5 | 2 | 2026-10-20 22:57:38 | 661.4 | 661.4 | 0 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x2.5 | 0 | 2026-10-19 18:29:43 | 1102 | 1102 | 6.049e-05 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x2.5 | 1 | 2026-10-20 17:54:35 | 1102 | 1102 | -0.0002117 | 0.000248 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x2.5 | 2 | 2026-10-20 23:20:42 | 1102 | 1102 | 0.0001512 | 4.96e-05 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x3.5 | 0 | 2026-10-19 18:42:54 | 1338 | 1350 | -0.008544 | 0.08 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x3.5 | 1 | 2026-10-20 17:42:34 | 1356 | 1350 | 0.004716 | 0.07141 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | x3.5 | 2 | 2026-10-20 22:58:32 | 1355 | 1350 | 0.003827 | 0.0713 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x1.5 | 0 | 2026-10-19 20:06:56 | 681.4 | 681.5 | -0.0001467 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x1.5 | 1 | 2026-10-20 15:03:31 | 681.7 | 681.5 | 0.0002935 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x1.5 | 2 | 2026-10-21 01:05:39 | 681.4 | 681.5 | -0.0001467 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x2.5 | 0 | 2026-10-19 20:09:06 | 1134 | 1131 | 0.002151 | 0.0008078 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x2.5 | 1 | 2026-10-20 15:40:27 | 1130 | 1131 | -0.001473 | 0.002706 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x2.5 | 2 | 2026-10-21 01:33:13 | 1131 | 1131 | -0.0006776 | 0.002468 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x3.5 | 0 | 2026-10-19 21:16:38 | 1227 | 1222 | 0.004475 | 0.1651 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x3.5 | 1 | 2026-10-20 16:48:32 | 1231 | 1222 | 0.007912 | 0.1637 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | x3.5 | 2 | 2026-10-21 01:51:58 | 1207 | 1222 | -0.01239 | 0.178 |  |
| mnasnet+resnext50+yolov4 | 3 | x1.5 | 0 | 2026-10-19 22:54:52 | 202.1 | 202.1 | 0 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x1.5 | 1 | 2026-10-20 17:12:42 | 202.1 | 202.1 | 0 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x1.5 | 2 | 2026-10-20 23:40:23 | 202.1 | 202.1 | 0 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x2.5 | 0 | 2026-10-19 22:57:42 | 337.1 | 337.1 | 9.889e-05 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x2.5 | 1 | 2026-10-20 17:09:52 | 337.1 | 337.1 | 9.889e-05 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x2.5 | 2 | 2026-10-20 23:39:49 | 337 | 337.1 | -0.0001978 | 0.0001686 |  |
| mnasnet+resnext50+yolov4 | 3 | x3.5 | 0 | 2026-10-19 22:55:09 | 471.8 | 471.8 | 7.066e-05 | 0.0001204 |  |
| mnasnet+resnext50+yolov4 | 3 | x3.5 | 1 | 2026-10-20 17:09:18 | 471.9 | 471.8 | 0.0002826 | 0 |  |
| mnasnet+resnext50+yolov4 | 3 | x3.5 | 2 | 2026-10-20 23:42:05 | 471.6 | 471.8 | -0.0003533 | 0.0002408 |  |
| resnet50+vgg19+yolov4 | 3 | x1.5 | 0 | 2026-10-19 21:46:41 | 170.5 | 170.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x1.5 | 1 | 2026-10-20 14:53:44 | 170.5 | 170.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x1.5 | 2 | 2026-10-21 00:49:25 | 170.5 | 170.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x2.5 | 0 | 2026-10-19 21:40:07 | 284.5 | 284.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x2.5 | 1 | 2026-10-20 14:55:50 | 284.5 | 284.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x2.5 | 2 | 2026-10-21 00:47:20 | 284.5 | 284.5 | 0 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x3.5 | 0 | 2026-10-19 21:46:59 | 398 | 398 | -8.375e-05 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x3.5 | 1 | 2026-10-20 14:57:37 | 398.1 | 398 | 0.0001675 | 0 |  |
| resnet50+vgg19+yolov4 | 3 | x3.5 | 2 | 2026-10-21 00:44:21 | 398 | 398 | -8.375e-05 | 0 |  |

표시된 (후보, rep): 0개.
