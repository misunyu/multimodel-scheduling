# Campaign 무결성 점검 (Phase 2 v1.2 §2.4)

`code/integrity.py`가 생성. 상세 표는 `results/integrity/`.

## 1. Run 수

- 계획 2976 runs, valid 2976, 누락 0. 계획에 없는 결과 파일 0개.
- 실행 시도 2976회. 첫 시도에 valid가 아니어서 재시도한 run 0개, ok가 아닌 시도 0회, `*.rejected.*` 파일 0개.
- 최종 invalid로 남은 결과 파일 0개.

| block | attempts | start | end | hours | sec_p50 |
|---|---|---|---|---|---|
| rep0 | 2016 | 2026-10-13 19:30:56 | 2026-10-14 05:51:25 | 10.34 | 18.53 |
| rep1 | 480 | 2026-10-14 05:51:25 | 2026-10-14 08:20:04 | 2.477 | 18.62 |
| rep2 | 480 | 2026-10-14 08:20:04 | 2026-10-14 10:48:45 | 2.478 | 18.62 |

Family × rate별 표는 `integrity/runs_by_family_rate.csv`. 누락이 있는 행:

없음.

## 2. Workload 완전성 (전체 feasible 후보 100%)

- 88 workloads(22 families × 4 rates) 중 완전한 것 88개.
- 평가 대상(3×, N≥3) 12개 중 완전한 것 12개, 3회 반복까지 모두 있는 것 12개.
- **Primary 집계에서 제외되는 workload: 0개.**

## 3. Conservation·trace·도착

- instance 17696개: conservation_flow 분포 {0: 17696}, trace_check {'ok': 17696}, snapshot flag(|diff|>1) 2개.
- 실현율(generated / scheduled): 최소 0.9593, 1% 분위 0.9886, 5% 분위 0.9960, 중앙값 1.0000. 0.95 미만 instance 0개, 그런 instance가 있는 run 0개.
- missed_ticks 비율: 최대 0.0407, 99% 분위 0.0114, 0보다 큰 instance 2531개, 5% 초과 0개.

## 4. Generated 차이 (품질 지표)

- 위반 workload 2 / 88. 그중 3× N≥3: 0 / 12.
- `gen_mismatch_max` 분포: 중앙값 0.0000, 90% 분위 0.0049, 최대 0.0408.

| N | mult | violations | workloads | rel_max | min_is_all_cpu | instance_on_cpu_in_min |
|---|---|---|---|---|---|---|
| 7 | 4 | 2 | 2 | 0.04078 | 0 | 2 |

위반 instance가 가장 적게 생성된 배치의 CPU instance 수 분포: {(7, 6): 1, (7, 5): 1}

위반 workload 목록:

| workload | N | mult | gen_mismatch_max | gen_mismatch_abs_max | violations |
|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19_x4 | 7 | 4 | 0.02705 | 0 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4_x4 | 7 | 4 | 0.04078 | 0 | 1 |

## 5. 시간 drift

반복 블록 사이의 F 비율(같은 후보, 3× N≥3):

| compare | candidates | F_ratio_median | F_ratio_mean | F_ratio_p05 | F_ratio_p95 | share_above_1 |
|---|---|---|---|---|---|---|
| rep1/rep0 | 480 | 0.9999 | 0.9997 | 0.9916 | 1.007 | 0.4542 |
| rep2/rep0 | 480 | 0.9999 | 0.9997 | 0.9915 | 1.007 | 0.4417 |
| rep2/rep1 | 480 | 1 | 1 | 0.9925 | 1.007 | 0.4792 |

1× 단일 instance run을 실행 시각 순으로:

| hour | model | device | rate | F | F_over_rate | service_ms_p50 | q_i |
|---|---|---|---|---|---|---|---|
| 4.1 | yolov4 | gpu | 13.38 | 13.4 | 1.002 | 7.93 | 0 |
| 4.13 | yolov4 | cpu | 13.38 | 7.5 | 0.5605 | 132 | 0.4403 |
| 4.15 | vgg19 | gpu | 56.37 | 56.4 | 1 | 1.862 | 0 |
| 4.15 | vgg19 | cpu | 56.37 | 14.2 | 0.2519 | 69.59 | 0.7482 |
| 4.25 | resnext50 | gpu | 42.36 | 42.4 | 1.001 | 2.357 | 0 |
| 4.28 | resnext50 | cpu | 42.36 | 42.3 | 0.9986 | 21.56 | 0 |
| 4.29 | gpt2 | cpu | 2.12 | 2.1 | 0.9906 | 395.8 | 0 |
| 4.3 | gpt2 | gpu | 2.12 | 2.1 | 0.9906 | 46.44 | 0 |
| 5.63 | squeezenet1.0-12 | cpu | 134.4 | 134.4 | 1 | 2.976 | 0 |
| 5.64 | squeezenet1.0-12 | gpu | 134.4 | 134.4 | 1 | 0.83 | 0 |
| 8.74 | resnet50 | gpu | 43.96 | 44 | 1.001 | 2.248 | 0 |
| 8.75 | resnet50 | cpu | 43.96 | 43.9 | 0.9986 | 21.82 | 0 |
| 8.78 | shufflenet-v2-12 | gpu | 84.78 | 84.8 | 1 | 1.286 | 0 |
| 8.8 | shufflenet-v2-12 | cpu | 84.78 | 84.8 | 1 | 2.651 | 0 |
| 10.09 | mnasnet | cpu | 79.07 | 79 | 0.9991 | 3.409 | 0 |
| 10.11 | mnasnet | gpu | 79.07 | 79 | 0.9991 | 1.313 | 0 |

GPU 상태(1분 표본의 시간대별 평균):

| hour | samples | temp_mean | temp_max | clock_mean | power_mean | load1_mean |
|---|---|---|---|---|---|---|
| 0 | 60 | 46.9 | 53 | 2614 | 114.1 | 6.808 |
| 1 | 60 | 49.4 | 55 | 2825 | 154.8 | 10.72 |
| 2 | 60 | 49.18 | 56 | 2770 | 155.6 | 10.94 |
| 3 | 60 | 48.43 | 58 | 2647 | 145 | 10.12 |
| 4 | 60 | 46.38 | 52 | 2555 | 123.9 | 7.428 |
| 5 | 60 | 45.8 | 52 | 2667 | 125 | 7.445 |
| 6 | 60 | 48.98 | 57 | 2862 | 167.5 | 10.18 |
| 7 | 60 | 47.78 | 54 | 2780 | 157.4 | 10.54 |
| 8 | 60 | 47.9 | 55 | 2735 | 155.1 | 9.781 |
| 9 | 60 | 46.78 | 53 | 2822 | 156.1 | 9.596 |
| 10 | 60 | 48.05 | 57 | 2688 | 157.5 | 9.167 |
| 11 | 59 | 50.49 | 58 | 2792 | 167.4 | 10.93 |
| 12 | 60 | 48.93 | 54 | 2756 | 163.4 | 8.713 |
| 13 | 60 | 49.33 | 54 | 2797 | 161.4 | 8.911 |
| 14 | 60 | 52.88 | 57 | 2851 | 188.5 | 11.03 |
| 15 | 18 | 51.22 | 56 | 2783 | 161.6 | 9.778 |

Run 시간(N별):

| N | runs | p50 | p95 | formula |
|---|---|---|---|---|
| 1 | 64 | 16.15 | 16.91 | 16.17 |
| 2 | 32 | 17.52 | 18 | 16.64 |
| 3 | 288 | 17.64 | 18.82 | 17.11 |
| 4 | 96 | 17.73 | 18.5 | 17.58 |
| 5 | 192 | 18 | 18.67 | 18.05 |
| 6 | 768 | 18.02 | 18.77 | 18.52 |
| 7 | 1536 | 18.97 | 20.01 | 18.99 |

## 6. 반복 3회의 편차 (3× N≥3, 후보 480개)

| 지표 | 중앙값 | 90% 분위 | 최대 |
|---|---|---|---|
| F 상대 범위 | 0.0031 | 0.0101 | 0.0433 |
| D_macro 절대 범위 | 0.0021 | 0.0053 | 0.0286 |
| S 절대 범위 | 0.0032 | 0.0088 | 0.0417 |
| S의 SE | 0.0010 | 0.0027 | 0.0130 |

반복별 oracle(각 반복의 자체 정규화)이 세 번 모두 같은 workload: 8 / 12.

| family | N | oracle_macro_placement | n_oracle_macro | oracle_rep0 | oracle_rep1 | oracle_rep2 | reps_equal_to_mean_oracle | max_regret_of_rep_oracles_on_mean_label | margin_oracle_to_2nd | oracle_S_se |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 | 3 | GCG | 1 | 5 | 7 | 5 | 2 | 0.0001636 | 0.0001636 | 0.0001029 |
| gpt2+mnasnet+resnet50+resnext50 | 4 | GCGG | 2 | 13 | 13 | 13 | 3 | 0 | 0 | 6.634e-05 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | GGGGC | 1 | 13 | 15 | 29 | 1 | 0.0004371 | 0.0001762 | 5.247e-05 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | GCGGCG | 1 | 45 | 45 | 45 | 3 | 0 | 0.004904 | 0.001406 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | GGGGCCG | 1 | 79 | 79 | 79 | 3 | 0 | 0.00474 | 0.0007733 |
| gpt2+resnext50+vgg19 | 3 | GGG | 1 | 7 | 7 | 7 | 3 | 0 | 0.07572 | 0.0001102 |
| gpt2+resnext50+yolov4 | 3 | GGG | 1 | 7 | 7 | 7 | 3 | 0 | 0.0875 | 0 |
| mnasnet+resnet50+resnext50 | 3 | GGG | 1 | 7 | 6 | 6 | 1 | 1.405e-05 | 1.405e-05 | 0 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | CGGGGG | 1 | 55 | 62 | 62 | 2 | 0.0001695 | 0.0001281 | 0.0003581 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | CGGCGGG | 1 | 118 | 118 | 118 | 3 | 0 | 0.008824 | 4.218e-05 |
| mnasnet+resnext50+yolov4 | 3 | CGG | 2 | 6 | 6 | 6 | 3 | 0 | 0 | 0 |
| resnet50+vgg19+yolov4 | 3 | GGG | 1 | 7 | 7 | 7 | 3 | 0 | 0.1793 | 0 |

## 7. 전부-GPU가 oracle인 workload (3×, N≥3, α=0.3, 3회 평균 label)

| 정의 | 전부-GPU가 oracle | 분모 |
|---|---|---|
| (a) aggregate | 6 | 12 |
| (b) D_macro | 6 | 12 |
| (c) D_max | 6 | 12 |

| family | N | oracle_agg_placement | oracle_macro_placement | oracle_max_placement | S_oracle_macro | S_all_gpu_macro |
|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 | 3 | GCG | GCG | GCG | 1 | 0.9998 |
| gpt2+mnasnet+resnet50+resnext50 | 4 | GCGG | GCGG | GCGG | 1 | 1 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | GGGGC | GGGGC | GGGGC | 1 | 0.9978 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | CCGGGG | GCGGCG | GCGGCG | 0.9925 | 0.9175 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | CCGGCGG | GGGGCCG | GGGGCCG | 0.9769 | 0.8407 |
| gpt2+resnext50+vgg19 | 3 | GGG | GGG | GGG | 1 | 1 |
| gpt2+resnext50+yolov4 | 3 | GGG | GGG | GGG | 1 | 1 |
| mnasnet+resnet50+resnext50 | 3 | GGG | GGG | GGG | 1 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | CGGCGG | CGGGGG | CGGCGG | 0.9999 | 0.9965 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | CGGCGGG | CGGCGGG | CGGCGGG | 0.9996 | 0.9203 |
| mnasnet+resnext50+yolov4 | 3 | CGG | CGG | CGG | 1 | 1 |
| resnet50+vgg19+yolov4 | 3 | GGG | GGG | GGG | 1 | 1 |
