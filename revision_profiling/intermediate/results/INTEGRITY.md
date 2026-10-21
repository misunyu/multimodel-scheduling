# Campaign 무결성 점검 — Phase 4 중간 rate (1.5×, 2.5×, 3.5×)

`intermediate/code_copy/integrity_intermediate.py`가 생성(원본 `code/integrity.py`의 규칙). 상세 표는 `results/integrity/`.

## 1. Run 수·시간

- 계획 1440 runs, valid 1440, 누락 0. 계획에 없는 결과 파일 0개.
- 실행 시도 1440회. 재시도한 run 0개, ok가 아닌 시도 0회, `*.rejected.*` 파일 0개, `logs/failed_runs.csv` 0행. 최종 invalid로 남은 결과 파일 0개.
- Run 시간 합 7.43 h, 첫 시작–마지막 종료 7.43 h. 세션 기록: 2026-10-19 16:01:52 START reason='initial start (PRECHECK pass, RUN_PLAN pass)' planned=1440 todo=1440 est_h=7.43 pid=177098 / 2026-10-19 23:27:57 END executed=1440 retries=0 failed=0 hours=7.43

| mult | attempts | hours | sec_p50 |
|---|---|---|---|
| 1.5 | 480 | 2.475 | 18.61 |
| 2.5 | 480 | 2.478 | 18.62 |
| 3.5 | 480 | 2.481 | 18.67 |

N별 run 시간:

| N | runs | p50 | p95 | formula |
|---|---|---|---|---|
| 3 | 144 | 17.64 | 18.81 | 17.11 |
| 4 | 48 | 17.8 | 18.57 | 17.58 |
| 5 | 96 | 17.91 | 18.71 | 18.05 |
| 6 | 384 | 18.04 | 18.81 | 18.52 |
| 7 | 768 | 18.98 | 20.04 | 18.99 |

## 2. Workload 완전성 (feasible 후보 100%)

- 36 workloads(12 families × 3 rates) 중 완전한 것 36개.
- **평가에서 제외되는 workload: 0개.**

## 3. Runtime 동일성

- 모든 valid run의 `loaded_modules`가 `revision_profiling/code/` 안: 예. Runner가 시작 시 runtime 파일 10개가 git HEAD와 같음을 확인했다(`logs/runtime_sha256.json`). Executor는 원본 그대로 호출했다.
- 결과의 mode가 모두 open_loop, β = 0.1: True.

## 4. Conservation·trace·도착

- instance-run 8784개: conservation_flow {0: 8784}, conservation_gen {0: 8784}, trace_check {'ok': 8784}, snapshot flag 0개.
- 실현율(generated / scheduled): 최소 0.9732, 1% 분위 0.9889, 5% 분위 0.9974, 중앙값 1.0000. 0.95 미만 instance 0개, 그런 instance가 있는 run 0개.
- missed_ticks 비율: 최대 0.0268, 99% 분위 0.0111, 0보다 큰 instance 1158개, 5% 초과 0개.
- scheduled_ticks − rate×10 s: 최소 -0.60, 최대 0.60(tick 격자 반올림; Phase 2와 같은 규칙).

## 5. Generated 차이 (품질 지표, Phase 2 규칙: ≥200 tick 상대 2%, <200 tick 절대 2건)

- 위반 workload 2 / 36.
- `gen_mismatch_max`(≥200 tick 상대 차) 중앙값 0.0016, 최대 0.0270; `gen_mismatch_abs_max`(<200 tick 절대 차) 최대 0.

위반 workload:

| workload | N | mult | gen_mismatch_max | gen_mismatch_abs_max | violations |
|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19_x3.5 | 7 | 3.5 | 0.02229 | 0 | 1 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4_x3.5 | 7 | 3.5 | 0.02699 | 0 | 1 |

## 6. 시간 경과 이상 징후 점검 (drift 검증 아님)

GPU 상태(1분 표본, 시간대별):

| hour | samples | temp_mean | temp_max | clock_mean | clock_min | power_mean | load1_mean |
|---|---|---|---|---|---|---|---|
| 0 | 60 | 53.02 | 60 | 2789 | 532 | 156.9 | 10.24 |
| 1 | 60 | 53.98 | 61 | 2775 | 442 | 175.8 | 10.38 |
| 2 | 60 | 48.6 | 53 | 2431 | 195 | 117.1 | 9.552 |
| 3 | 60 | 50.62 | 58 | 2763 | 187 | 149.3 | 8.565 |
| 4 | 60 | 50.87 | 56 | 2757 | 2527 | 152.9 | 11.01 |
| 5 | 60 | 47.1 | 51 | 2436 | 195 | 125.2 | 10.05 |
| 6 | 60 | 45.63 | 52 | 2464 | 495 | 106 | 8.362 |
| 7 | 26 | 45.42 | 49 | 2231 | 195 | 92.08 | 6.425 |

각 family의 전부-GPU 후보를 실행 시각 순으로(rate가 다른 run의 F 차이는 부하 차이일 수 있어 drift 근거로 쓰지 않는다). `flag` = F/Λ < 0.95 또는 실현율 < 0.95인 run(확인 대상):

| exec_order | start_ts | family | N | multiplier | F_mean | F_over_Lambda | D_macro_mean | min_realization | gpu_clock_near | gpu_temp_near | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 2026-10-19 16:03:29 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 2.5 | 1001 | 0.9039 | 0.06042 | 1 | 2917 | 54 | 확인 |
| 108 | 2026-10-19 16:36:53 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 3.5 | 1119 | 0.7214 | 0.1917 | 0.9996 | 2580 | 51 | 확인 |
| 259 | 2026-10-19 17:25:55 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 1.5 | 664.5 | 0.9999 | 7.086e-05 | 1 | 2925 | 57 |  |
| 462 | 2026-10-19 18:29:43 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 2.5 | 1102 | 0.9999 | 0 | 1 | 2550 | 49 |  |
| 494 | 2026-10-19 18:39:19 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 1.5 | 661.4 | 1 | 0 | 1 | 2542 | 47 |  |
| 506 | 2026-10-19 18:42:54 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 3.5 | 1338 | 0.8673 | 0.08 | 1 | 2527 | 53 | 확인 |
| 581 | 2026-10-19 19:05:24 | gpt2+resnext50+yolov4 | 3 | 2.5 | 144.8 | 1.001 | 0 | 1 | 1215 | 46 |  |
| 582 | 2026-10-19 19:05:42 | gpt2+resnext50+yolov4 | 3 | 3.5 | 202.5 | 1 | 0 | 1 | 2617 | 49 |  |
| 594 | 2026-10-19 19:09:19 | gpt2+resnext50+yolov4 | 3 | 1.5 | 86.7 | 0.999 | 0 | 1 | 2932 | 50 |  |
| 608 | 2026-10-19 19:13:33 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 3.5 | 871.2 | 0.9866 | 0.01055 | 1 | 2940 | 51 |  |
| 610 | 2026-10-19 19:14:08 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 1.5 | 378.4 | 0.9999 | 0 | 1 | 2955 | 51 |  |
| 656 | 2026-10-19 19:28:02 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 2.5 | 630.6 | 0.9998 | 0 | 1 | 2932 | 54 |  |
| 782 | 2026-10-19 20:06:56 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 1.5 | 681.4 | 0.9999 | 0 | 1 | 2932 | 54 |  |
| 789 | 2026-10-19 20:09:06 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 2.5 | 1134 | 0.9984 | 0.0008078 | 1 | 2925 | 55 |  |
| 1005 | 2026-10-19 21:16:38 | mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 3.5 | 1227 | 0.7718 | 0.1651 | 1 | 2535 | 47 | 확인 |
| 1080 | 2026-10-19 21:40:07 | resnet50+vgg19+yolov4 | 3 | 2.5 | 284.5 | 1.001 | 0 | 1 | 2542 | 48 |  |
| 1102 | 2026-10-19 21:46:41 | resnet50+vgg19+yolov4 | 3 | 1.5 | 170.5 | 0.9996 | 0 | 1 | 2542 | 46 |  |
| 1103 | 2026-10-19 21:46:59 | resnet50+vgg19+yolov4 | 3 | 3.5 | 398 | 1 | 0 | 1 | 2550 | 47 |  |
| 1162 | 2026-10-19 22:04:57 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 3.5 | 1115 | 0.8236 | 0.1123 | 1 | 2550 | 46 | 확인 |
| 1219 | 2026-10-19 22:22:15 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 2.5 | 922.6 | 0.9544 | 0.02288 | 1 | 2527 | 47 |  |
| 1279 | 2026-10-19 22:40:33 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 1.5 | 580 | 1 | 0 | 1 | 2542 | 45 |  |
| 1302 | 2026-10-19 22:47:37 | gpt2+resnext50+vgg19 | 3 | 3.5 | 353 | 1 | 0 | 1 | 2542 | 45 |  |
| 1304 | 2026-10-19 22:48:13 | gpt2+resnext50+vgg19 | 3 | 2.5 | 252.1 | 0.9999 | 0 | 1 | 2535 | 48 |  |
| 1315 | 2026-10-19 22:51:38 | gpt2+resnext50+vgg19 | 3 | 1.5 | 151.3 | 1 | 0 | 1 | 2527 | 47 |  |
| 1326 | 2026-10-19 22:54:52 | mnasnet+resnext50+yolov4 | 3 | 1.5 | 202.1 | 0.9994 | 0 | 1 | 2542 | 47 |  |
| 1327 | 2026-10-19 22:55:09 | mnasnet+resnext50+yolov4 | 3 | 3.5 | 471.8 | 0.9999 | 0.0001204 | 1 | 1132 | 49 |  |
| 1336 | 2026-10-19 22:57:42 | mnasnet+resnext50+yolov4 | 3 | 2.5 | 337.1 | 1 | 0 | 1 | 2535 | 48 |  |
| 1344 | 2026-10-19 22:59:57 | gpt2+mnasnet+resnet50+resnext50 | 4 | 2.5 | 418.8 | 1 | 0 | 1 | 2550 | 47 |  |
| 1372 | 2026-10-19 23:08:19 | gpt2+mnasnet+resnet50+resnext50 | 4 | 1.5 | 251.2 | 0.9997 | 0 | 1 | 2542 | 45 |  |
| 1381 | 2026-10-19 23:10:59 | gpt2+mnasnet+resnet50+resnext50 | 4 | 3.5 | 579.4 | 0.9883 | 0.008022 | 0.9993 | 2535 | 44 |  |
| 1402 | 2026-10-19 23:17:15 | gpt2+mnasnet+resnet50 | 3 | 3.5 | 438.1 | 1 | 0 | 1 | 2527 | 46 |  |
| 1403 | 2026-10-19 23:17:32 | gpt2+mnasnet+resnet50 | 3 | 1.5 | 187.7 | 0.9999 | 0 | 1 | 2527 | 46 |  |
| 1414 | 2026-10-19 23:20:47 | gpt2+mnasnet+resnet50 | 3 | 2.5 | 312.9 | 1 | 0 | 1 | 195 | 43 |  |
| 1419 | 2026-10-19 23:22:11 | mnasnet+resnet50+resnext50 | 3 | 1.5 | 248 | 0.9997 | 0 | 1 | 2542 | 45 |  |
| 1431 | 2026-10-19 23:25:28 | mnasnet+resnet50+resnext50 | 3 | 3.5 | 578.9 | 1 | 0 | 1 | 2535 | 47 |  |
| 1432 | 2026-10-19 23:25:45 | mnasnet+resnet50+resnext50 | 3 | 2.5 | 413.4 | 0.9998 | 0 | 1 | 2535 | 47 |  |

표시된 run: 5개.

## 7. Phase 2 3×와 나란히 (전부-GPU 후보)

중간 rate는 **1회 측정(rep0)**, 3×는 **Phase 2의 3회 반복 평균**이다. 비단조는 측정 오류로 판정하지 않는다.

| family | N | F 2.5x (1 run) | generated/10s 2.5x (1 run) | min realization 2.5x (1 run) | D_macro 2.5x (1 run) | F 3x (Phase 2, 3-run mean) | generated/10s 3x (Phase 2, 3-run mean) | min realization 3x (Phase 2, 3-run mean) | D_macro 3x (Phase 2, 3-run mean) | F 3.5x (1 run) | generated/10s 3.5x (1 run) | min realization 3.5x (1 run) | D_macro 3.5x (1 run) | non_monotonic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 | 3 | 312.9 | 312.9 | 1 | 0 | 375.3 | 375.4 | 0.9987 | 0 | 438.1 | 438 | 1 | 0 |  |
| gpt2+mnasnet+resnet50+resnext50 | 4 | 418.8 | 418.8 | 1 | 0 | 502.5 | 502.6 | 1 | 0 | 579.4 | 586 | 0.9993 | 0.008022 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 630.6 | 630.8 | 1 | 0 | 755.5 | 757 | 1 | 0.001241 | 871.2 | 882.9 | 1 | 0.01055 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 922.6 | 966.8 | 1 | 0.02288 | 1072 | 1160 | 0.9995 | 0.04212 | 1115 | 1353 | 1 | 0.1123 |  |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 1001 | 1108 | 1 | 0.06042 | 1145 | 1329 | 1 | 0.08976 | 1119 | 1550 | 0.9996 | 0.1917 | F(3.5x) < F(3x) |
| gpt2+resnext50+vgg19 | 3 | 252.1 | 252.1 | 1 | 0 | 302.6 | 302.6 | 1 | 0 | 353 | 352.9 | 1 | 0 |  |
| gpt2+resnext50+yolov4 | 3 | 144.8 | 144.6 | 1 | 0 | 173.5 | 173.7 | 1 | 0 | 202.5 | 202.4 | 1 | 0 |  |
| mnasnet+resnet50+resnext50 | 3 | 413.4 | 413.5 | 1 | 0 | 496.1 | 496.2 | 1 | 0 | 578.9 | 578.8 | 1 | 0 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 1102 | 1102 | 1 | 0 | 1318 | 1323 | 0.9984 | 0.001997 | 1338 | 1543 | 1 | 0.08 |  |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 1134 | 1136 | 1 | 0.0008078 | 1265 | 1363 | 0.9998 | 0.03521 | 1227 | 1590 | 1 | 0.1651 | F(3.5x) < F(3x) |
| mnasnet+resnext50+yolov4 | 3 | 337.1 | 337 | 1 | 0 | 404.3 | 404.5 | 1 | 0 | 471.8 | 471.8 | 1 | 0.0001204 |  |
| resnet50+vgg19+yolov4 | 3 | 284.5 | 284.2 | 1 | 0 | 341.1 | 341.2 | 1 | 0 | 398 | 397.9 | 1 | 0 |  |

비단조 family 2개. 해당 run의 요청 생성 상태와 GPU 로그:

| family | multiplier | exec_order | start_ts | F_mean | F_over_Lambda | generated_mean | min_realization | max_missed_ratio | gpu_clock_near | gpu_temp_near |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 2.5 | 5 | 2026-10-19 16:03:29 | 1001 | 0.9039 | 1.108e+04 | 1 | 0 | 2917 | 54 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 3.5 | 108 | 2026-10-19 16:36:53 | 1119 | 0.7214 | 1.55e+04 | 0.9996 | 0.0004253 | 2580 | 51 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 2.5 | 789 | 2026-10-19 20:09:06 | 1134 | 0.9984 | 1.136e+04 | 1 | 0 | 2925 | 55 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 3.5 | 1005 | 2026-10-19 21:16:38 | 1227 | 0.7718 | 1.59e+04 | 1 | 0 | 2535 | 47 |

## 8. Label 통계

| family | N | mult | F_max | Lambda | D_macro_all_gpu | T_all_gpu | all_gpu_is_oracle_macro | margin_oracle_to_2nd |
|---|---|---|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 | 3 | 1.5 | 187.7 | 187.7 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50 | 3 | 2.5 | 312.9 | 312.9 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50 | 3 | 3.5 | 438.1 | 438 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50+resnext50 | 4 | 1.5 | 251.2 | 251.3 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50+resnext50 | 4 | 2.5 | 418.8 | 418.8 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50+resnext50 | 4 | 3.5 | 586.1 | 586.3 | 0.008022 | 0.9886 | False | 0.01384 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 1.5 | 378.4 | 378.4 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 2.5 | 630.8 | 630.7 | 0 | 0.9997 | False | 0.0001585 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 | 5 | 3.5 | 882.5 | 883 | 0.01055 | 0.9872 | False | 4.767e-05 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 1.5 | 580 | 580 | 0 | 1 | True | 0 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 2.5 | 966.6 | 966.7 | 0.02288 | 0.9545 | False | 0.002261 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 | 6 | 3.5 | 1334 | 1353 | 0.1123 | 0.8356 | False | 0.008815 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 1.5 | 664.6 | 664.6 | 7.086e-05 | 0.9998 | False | 0 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 2.5 | 1105 | 1108 | 0.06042 | 0.9064 | False | 0.0001703 |
| gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 7 | 3.5 | 1506 | 1551 | 0.1917 | 0.7429 | False | 0.002524 |
| gpt2+resnext50+vgg19 | 3 | 1.5 | 151.3 | 151.3 | 0 | 1 | True | 0.03786 |
| gpt2+resnext50+vgg19 | 3 | 2.5 | 252.1 | 252.1 | 0 | 1 | True | 0.06771 |
| gpt2+resnext50+vgg19 | 3 | 3.5 | 353 | 353 | 0 | 1 | True | 0.08337 |
| gpt2+resnext50+yolov4 | 3 | 1.5 | 86.7 | 86.79 | 0 | 1 | True | 0.0311 |
| gpt2+resnext50+yolov4 | 3 | 2.5 | 144.8 | 144.7 | 0 | 1 | True | 0.0799 |
| gpt2+resnext50+yolov4 | 3 | 3.5 | 202.5 | 202.5 | 0 | 1 | True | 0.09177 |
| mnasnet+resnet50+resnext50 | 3 | 1.5 | 248 | 248.1 | 0 | 1 | True | 0 |
| mnasnet+resnet50+resnext50 | 3 | 2.5 | 413.5 | 413.5 | 0 | 0.9998 | False | 0.0002418 |
| mnasnet+resnet50+resnext50 | 3 | 3.5 | 579 | 578.9 | 0 | 0.9998 | False | 0.0001727 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 1.5 | 661.4 | 661.4 | 0 | 1 | True | 0 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 2.5 | 1102 | 1102 | 0 | 0.9999 | False | 9.072e-05 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 | 6 | 3.5 | 1527 | 1543 | 0.08 | 0.8764 | False | 0.03418 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 1.5 | 681.4 | 681.5 | 0 | 1 | True | 0 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 2.5 | 1135 | 1136 | 0.0008078 | 0.9987 | False | 0.0001136 |
| mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 | 7 | 3.5 | 1498 | 1590 | 0.1651 | 0.8194 | False | 0.008437 |
| mnasnet+resnext50+yolov4 | 3 | 1.5 | 202.1 | 202.2 | 0 | 1 | True | 0 |
| mnasnet+resnext50+yolov4 | 3 | 2.5 | 337.2 | 337 | 0 | 0.9997 | False | 0.0002966 |
| mnasnet+resnext50+yolov4 | 3 | 3.5 | 472 | 471.8 | 0.0001204 | 0.9996 | False | 0.0004599 |
| resnet50+vgg19+yolov4 | 3 | 1.5 | 170.5 | 170.6 | 0 | 1 | True | 0.09462 |
| resnet50+vgg19+yolov4 | 3 | 2.5 | 284.5 | 284.3 | 0 | 1 | True | 0.1719 |
| resnet50+vgg19+yolov4 | 3 | 3.5 | 398 | 398 | 0 | 1 | True | 0.1848 |
