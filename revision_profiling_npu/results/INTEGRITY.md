# Campaign 무결성 점검 (Phase N2-A A3)

`code/integrity_npu.py`가 생성. 상세 표는 `results/integrity/`. 항목·규칙은 Phase 2 `integrity.py`와 같다.

## 1. Run 수·유효성

- 계획 492 runs, valid 492, 누락 0. 계획에 없는 결과 파일 0개.
- 실행 시도 492회, ok가 아닌 시도 0회, 재시도 기록 0건, `*.rejected.*` 결과 파일 0개, 최종 invalid 결과 0개.
- run 단위 검사(rc 0·valid·failed 0·Close rc 0·residency·변종–바이너리 매핑) 통과 492 / 492.

| block | attempts | ok | hours | sec_p50 | sec_mean | dma32_min | dma32_first | dma32_last |
|---|---|---|---|---|---|---|---|---|
| rep0 | 372 | 372 | 4.104 | 39.3 | 39.72 | 162890 | 178127 | 164633 |
| rep1 | 60 | 60 | 0.6759 | 40.05 | 40.55 | 162826 | 164633 | 166108 |
| rep2 | 60 | 60 | 0.6756 | 40.1 | 40.54 | 161185 | 166108 | 164131 |

## 2. Workload 완전성

- 44 workloads(11 families × 4 rates) 중 후보 100% valid: 44. 제외 workload: 0.
- 3× N≥3 평가 4 workloads 중 3회 반복까지 모두 있는 것: 4.

## 3. Conservation·trace·도착

- instance-run 1422개: conservation_gen {0: 1422}, conservation_flow {0: 1422}, trace_check {'ok': 1422}.
- 실현율(generated / scheduled): 최소 0.9938, 1% 분위 0.9988, 중앙값 1.0000. 0.95 미만 instance 0개.

## 4. 후보 간 generated 차이

- 위반 workload: 학습 0 / 33, 평가 0 / 11.
- 상대 차(≥200 tick) 최대 0.0062, 절대 차(<200 tick) 최대 0.

## 5. Residency·변종–바이너리 매핑

- residency 위반 run 0. 매핑 불일치 run 0. 매핑 규칙: sidecar의 registry key = run의 key, NPU id = 배치, `.o` = `<instance>/npu_code/<instance>_neubla_p1.o`, driver = `/workspace/lib` 판.
- `.o` 파일 sha256은 registry 목록(`registry/files_sha256.json`)과 campaign 전후 비교로 확인한다(REPORT_N2A).

| instance | device | binary | runs | registry_sha256 |
|---|---|---|---|---|
| resnet50_big | npu0 | resnet50_big_neubla_p1.o | 72 | 80d12c14dfc40555 |
| resnet50_big | npu1 | resnet50_big_neubla_p1.o | 72 | 80d12c14dfc40555 |
| resnet50_small | npu0 | resnet50_small_neubla_p1.o | 80 | dfd310e8333d2156 |
| resnet50_small | npu1 | resnet50_small_neubla_p1.o | 80 | dfd310e8333d2156 |
| yolov3_big | npu0 | yolov3_big_neubla_p1.o | 98 | c855b2b5c312f61e |
| yolov3_big | npu1 | yolov3_big_neubla_p1.o | 98 | c855b2b5c312f61e |
| yolov3_small | npu0 | yolov3_small_neubla_p1.o | 72 | 9ea9a273186d9c70 |
| yolov3_small | npu1 | yolov3_small_neubla_p1.o | 72 | 9ea9a273186d9c70 |

## 6. 반복 편차 (3× N≥3, 60 후보 × 3회)

| 지표 | 중앙값 | 90% 분위 | 최대 |
|---|---|---|---|
| F 상대 범위 | 0.0096 | 0.0291 | 0.0526 |
| D_macro 절대 범위 | 0.0041 | 0.0095 | 0.0160 |
| S의 SE | 0.0019 | 0.0046 | 0.0068 |

- 반복별 oracle(각 반복 자체 정규화)이 세 번 모두 같은 workload: 4 / 4.
- oracle 1·2위 실측 차 ≤ 0.0002: 0 / 4.

| family | candidates | oracle | oracle_placement | n_oracle | S_oracle | oracle_rep0 | oracle_rep1 | oracle_rep2 | reps_equal | reps_equal_to_mean_oracle | margin_1st_2nd | tie_le_0.0002 | oracle_S_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rbrsyb | 13 | combination_2 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0 | 1 | 0.9868 | 2 | 2 | 2 | True | 3 | 0.01332 | False | 0.0001166 |
| rbrsybys | 21 | combination_5 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.9881 | 5 | 5 | 5 | True | 3 | 0.009474 | False | 0.001685 |
| rbybys | 13 | combination_10 | resnet50_big:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.9922 | 10 | 10 | 10 | True | 3 | 0.01872 | False | 0.0001854 |
| rsybys | 13 | combination_5 | resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.9922 | 5 | 5 | 5 | True | 3 | 0.01854 | False | 7.85e-17 |

## 7. 시간 drift·DMA32

반복 블록 사이 F 비율(같은 후보):

| compare | candidates | F_ratio_median | F_ratio_p05 | F_ratio_p95 | share_above_1 |
|---|---|---|---|---|---|
| rep1/rep0 | 60 | 1 | 0.9829 | 1.028 | 0.4333 |
| rep2/rep0 | 60 | 1 | 0.9833 | 1.034 | 0.45 |
| rep2/rep1 | 60 | 1 | 0.9829 | 1.013 | 0.4167 |

DMA32 free pages (50 run 구간별):

| chunk | dma32_min | dma32_mean |
|---|---|---|
| 1 | 1.76e+05 | 1.781e+05 |
| 51 | 1.662e+05 | 1.757e+05 |
| 101 | 1.67e+05 | 1.702e+05 |
| 151 | 1.646e+05 | 1.693e+05 |
| 201 | 1.7e+05 | 1.703e+05 |
| 251 | 1.649e+05 | 1.683e+05 |
| 301 | 1.635e+05 | 1.677e+05 |
| 351 | 1.628e+05 | 1.659e+05 |
| 401 | 1.622e+05 | 1.659e+05 |
| 451 | 1.612e+05 | 1.651e+05 |

단일 instance run(rep0, 실행 순서):

| order | instance | device | mult | rate | F | service_ms_p50 | q_i |
|---|---|---|---|---|---|---|---|
| 81 | resnet50_big | npu0 | 1 | 13.12 | 13.1 | 44.63 | 0 |
| 82 | resnet50_big | cpu | 1 | 13.12 | 13.13 | 18.71 | 0 |
| 83 | resnet50_big | npu1 | 4 | 52.5 | 21.9 | 45 | 0.5835 |
| 84 | resnet50_big | npu0 | 2 | 26.25 | 22.13 | 44.53 | 0.1563 |
| 85 | resnet50_big | cpu | 4 | 52.5 | 52.5 | 18.63 | 0 |
| 86 | resnet50_big | npu1 | 2 | 26.25 | 21.87 | 45.05 | 0.1677 |
| 87 | resnet50_big | npu1 | 3 | 39.38 | 21.87 | 45.04 | 0.445 |
| 88 | resnet50_big | npu0 | 4 | 52.5 | 22.2 | 44.53 | 0.5771 |
| 89 | resnet50_big | npu1 | 1 | 13.12 | 13.1 | 45.05 | 0 |
| 90 | resnet50_big | cpu | 2 | 26.25 | 26.27 | 17.98 | 0 |
| 91 | resnet50_big | npu0 | 3 | 39.38 | 22.2 | 44.54 | 0.4365 |
| 92 | resnet50_big | cpu | 3 | 39.38 | 39.37 | 17.78 | 0 |
| 177 | yolov3_small | cpu | 4 | 14.27 | 3.367 | 300.6 | 0.7664 |
| 178 | yolov3_small | npu0 | 4 | 14.27 | 14.13 | 70.72 | 0.0093 |
| 179 | yolov3_small | cpu | 2 | 7.133 | 3.4 | 296.8 | 0.5234 |
| 180 | yolov3_small | npu0 | 2 | 7.133 | 7.133 | 70.74 | 0 |
| 181 | yolov3_small | npu1 | 3 | 10.7 | 10.7 | 78.53 | 0 |
| 182 | yolov3_small | npu0 | 3 | 10.7 | 10.7 | 70.7 | 0 |
| 183 | yolov3_small | cpu | 1 | 3.567 | 3.4 | 295.6 | 0.0374 |
| 184 | yolov3_small | npu0 | 1 | 3.567 | 3.567 | 70.3 | 0 |
| 185 | yolov3_small | cpu | 3 | 10.7 | 3.433 | 295.3 | 0.6791 |
| 186 | yolov3_small | npu1 | 2 | 7.133 | 7.133 | 78.76 | 0 |
| 187 | yolov3_small | npu1 | 4 | 14.27 | 12.7 | 78.57 | 0.1098 |
| 188 | yolov3_small | npu1 | 1 | 3.567 | 3.567 | 78.73 | 0 |
| 217 | yolov3_big | npu0 | 3 | 10.7 | 10.2 | 98.26 | 0.0467 |
| 218 | yolov3_big | cpu | 1 | 3.567 | 3.4 | 296.6 | 0.0374 |
| 219 | yolov3_big | cpu | 4 | 14.27 | 3.367 | 300.4 | 0.764 |
| 220 | yolov3_big | npu0 | 2 | 7.133 | 7.133 | 98.34 | 0 |
| 221 | yolov3_big | npu0 | 4 | 14.27 | 10.2 | 98.3 | 0.2827 |
| 222 | yolov3_big | cpu | 2 | 7.133 | 3.4 | 297.8 | 0.528 |
| 223 | yolov3_big | npu1 | 3 | 10.7 | 9.467 | 106.2 | 0.1184 |
| 224 | yolov3_big | npu1 | 1 | 3.567 | 3.567 | 106.3 | 0 |
| 225 | yolov3_big | npu0 | 1 | 3.567 | 3.567 | 97.99 | 0 |
| 226 | yolov3_big | npu1 | 4 | 14.27 | 9.433 | 106.1 | 0.3388 |
| 227 | yolov3_big | cpu | 3 | 10.7 | 3.4 | 293.2 | 0.6791 |
| 228 | yolov3_big | npu1 | 2 | 7.133 | 7.133 | 106.5 | 0 |
| 309 | resnet50_small | npu0 | 4 | 52.5 | 24.27 | 40.61 | 0.5378 |
| 310 | resnet50_small | npu0 | 3 | 39.38 | 24.2 | 40.63 | 0.3858 |
| 311 | resnet50_small | cpu | 3 | 39.38 | 39.37 | 17.69 | 0 |
| 312 | resnet50_small | cpu | 4 | 52.5 | 52.5 | 18.58 | 0 |
| 313 | resnet50_small | npu1 | 3 | 39.38 | 24.03 | 41.15 | 0.39 |
| 314 | resnet50_small | npu1 | 4 | 52.5 | 23.97 | 41.06 | 0.5435 |
| 315 | resnet50_small | npu1 | 2 | 26.25 | 23.93 | 41.16 | 0.0877 |
| 316 | resnet50_small | cpu | 2 | 26.25 | 26.27 | 17.84 | 0 |
| 317 | resnet50_small | npu1 | 1 | 13.12 | 13.1 | 41.19 | 0 |
| 318 | resnet50_small | cpu | 1 | 13.12 | 13.13 | 18.56 | 0 |
| 319 | resnet50_small | npu0 | 1 | 13.12 | 13.1 | 40.72 | 0 |
| 320 | resnet50_small | npu0 | 2 | 26.25 | 24.2 | 40.61 | 0.0775 |

Run 시간(N별, 초):

| N | runs | p50 | p95 |
|---|---|---|---|
| 1 | 48 | 36.85 | 40.7 |
| 2 | 84 | 38.1 | 42.1 |
| 3 | 234 | 39.65 | 42.8 |
| 4 | 126 | 40.3 | 43.27 |

NPU 온도: driver·sysfs에 노출되지 않아 기록 없음 [확인 불가].

## 8. Workload별 label 참고값

| workload | N | F_max | oracle | oracle_placement | S_oracle | S_all_cpu | D_macro_all_cpu | D_macro_min | D_macro_max |
|---|---|---|---|---|---|---|---|---|---|
| rb_x1 | 1 | 13.13 | combination_1 | resnet50_big:cpu | 1 | 1 | 0 | 0 | 0 |
| rb_x2 | 1 | 26.27 | combination_1 | resnet50_big:cpu | 1 | 1 | 0 | 0 | 0.1677 |
| rb_x3 | 1 | 39.37 | combination_1 | resnet50_big:cpu | 1 | 1 | 0 | 0 | 0.445 |
| rb_x4 | 1 | 52.5 | combination_1 | resnet50_big:cpu | 1 | 1 | 0 | 0 | 0.5835 |
| rbrs_x1 | 2 | 26.27 | combination_1 | resnet50_big:cpu,resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.01142 |
| rbrs_x2 | 2 | 52.53 | combination_1 | resnet50_big:cpu,resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.2961 |
| rbrs_x3 | 2 | 78.8 | combination_1 | resnet50_big:cpu,resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.5355 |
| rbrs_x4 | 2 | 86.33 | combination_1 | resnet50_big:cpu,resnet50_small:cpu | 0.9467 | 0.9467 | 0.1778 | 0.1778 | 0.6737 |
| rbrsyb_x1 | 3 | 29.83 | combination_2 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0 | 1 | 0.9444 | 0.081 | 0 | 0.3358 |
| rbrsyb_x2 | 3 | 59.67 | combination_2 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0 | 1 | 0.8497 | 0.2274 | 0 | 0.5581 |
| rbrsyb_x3 | 3 | 88.06 | combination_2 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0 | 0.9868 | 0.5404 | 0.4482 | 0.04396 | 0.6951 |
| rbrsyb_x4 | 3 | 91.93 | combination_2 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0 | 0.9224 | 0.4703 | 0.5862 | 0.2588 | 0.7737 |
| rbrsybys_x1 | 4 | 33.4 | combination_5 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.8284 | 0.236 | 0 | 0.45 |
| rbrsybys_x2 | 4 | 66.8 | combination_5 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.595 | 0.4583 | 0 | 0.6535 |
| rbrsybys_x3 | 4 | 98.2 | combination_5 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 0.9881 | 0.3036 | 0.64 | 0.03976 | 0.7692 |
| rbrsybys_x4 | 4 | 98.83 | combination_7 | resnet50_big:cpu,resnet50_small:cpu,yolov3_big:npu1,yolov3_small:npu0 | 0.9202 | 0.2817 | 0.7271 | 0.266 | 0.8253 |
| rbybys_x1 | 3 | 20.27 | combination_10 | resnet50_big:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.7988 | 0.243 | 0 | 0.5016 |
| rbybys_x2 | 3 | 40.53 | combination_10 | resnet50_big:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.5871 | 0.4828 | 0 | 0.6587 |
| rbybys_x3 | 3 | 59.92 | combination_10 | resnet50_big:cpu,yolov3_big:npu0,yolov3_small:npu1 | 0.9922 | 0.311 | 0.6513 | 0.02596 | 0.7728 |
| rbybys_x4 | 3 | 72.53 | combination_13 | resnet50_big:cpu,yolov3_big:npu1,yolov3_small:npu0 | 0.9521 | 0.202 | 0.7374 | 0.1598 | 0.8282 |
| rs_x1 | 1 | 13.13 | combination_1 | resnet50_small:cpu | 1 | 1 | 0 | 0 | 0 |
| rs_x2 | 1 | 26.27 | combination_1 | resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.08767 |
| rs_x3 | 1 | 39.37 | combination_1 | resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.39 |
| rs_x4 | 1 | 52.5 | combination_1 | resnet50_small:cpu | 1 | 1 | 0 | 0 | 0.5435 |
| rsyb_x1 | 2 | 16.7 | combination_4 | resnet50_small:cpu,yolov3_big:npu0 | 1 | 0.949 | 0.07009 | 0 | 0.3645 |
| rsyb_x2 | 2 | 33.4 | combination_4 | resnet50_small:cpu,yolov3_big:npu0 | 1 | 0.7818 | 0.3014 | 0 | 0.5015 |
| rsyb_x3 | 2 | 49.33 | combination_4 | resnet50_small:cpu,yolov3_big:npu0 | 0.9897 | 0.7417 | 0.3723 | 0.03427 | 0.6632 |
| rsyb_x4 | 2 | 62.3 | combination_4 | resnet50_small:cpu,yolov3_big:npu0 | 0.9539 | 0.5708 | 0.5032 | 0.1537 | 0.7469 |
| rsybys_x1 | 3 | 20.27 | combination_5 | resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.7997 | 0.2399 | 0 | 0.5016 |
| rsybys_x2 | 3 | 40.53 | combination_5 | resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.587 | 0.4831 | 0 | 0.6363 |
| rsybys_x3 | 3 | 59.9 | combination_5 | resnet50_small:cpu,yolov3_big:npu0,yolov3_small:npu1 | 0.9922 | 0.3114 | 0.6512 | 0.02596 | 0.757 |
| rsybys_x4 | 3 | 72.8 | combination_7 | resnet50_small:cpu,yolov3_big:npu1,yolov3_small:npu0 | 0.9528 | 0.1958 | 0.7378 | 0.1573 | 0.8189 |
| yb_x1 | 1 | 3.567 | combination_2 | yolov3_big:npu0 | 1 | 0.9421 | 0.03738 | 0 | 0.03738 |
| yb_x2 | 1 | 7.133 | combination_2 | yolov3_big:npu0 | 1 | 0.3182 | 0.528 | 0 | 0.528 |
| yb_x3 | 1 | 10.2 | combination_2 | yolov3_big:npu0 | 0.986 | 0.1296 | 0.6791 | 0.04673 | 0.6791 |
| yb_x4 | 1 | 10.2 | combination_2 | yolov3_big:npu0 | 0.9152 | 0.1009 | 0.764 | 0.2827 | 0.764 |
| ybys_x1 | 2 | 7.133 | combination_5 | yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.6626 | 0.2523 | 0 | 0.2523 |
| ybys_x2 | 2 | 14.27 | combination_5 | yolov3_big:npu0,yolov3_small:npu1 | 1 | 0.1799 | 0.6308 | 0 | 0.6308 |
| ybys_x3 | 2 | 20.8 | combination_5 | yolov3_big:npu0,yolov3_small:npu1 | 0.9921 | 0.03325 | 0.7492 | 0.02648 | 0.7492 |
| ybys_x4 | 2 | 23.33 | combination_7 | yolov3_big:npu1,yolov3_small:npu0 | 0.945 | -0.01891 | 0.8154 | 0.1834 | 0.8154 |
| ys_x1 | 1 | 3.567 | combination_2 | yolov3_small:npu0 | 1 | 0.9421 | 0.03738 | 0 | 0.03738 |
| ys_x2 | 1 | 7.133 | combination_2 | yolov3_small:npu0 | 1 | 0.3196 | 0.5234 | 0 | 0.5234 |
| ys_x3 | 1 | 10.7 | combination_2 | yolov3_small:npu0 | 1 | 0.1171 | 0.6791 | 0 | 0.6791 |
| ys_x4 | 1 | 14.13 | combination_2 | yolov3_small:npu0 | 0.9972 | 0.008301 | 0.7664 | 0.009346 | 0.7664 |
