# Pilot families and reference placements (Phase N1 §4-§5) — fixed before any pilot run

Source: `revision_profiling_npu/isolated/offline_profile.json` only.

Cmax (req/s): resnet50 52.50 (cpu/common), yolov3 14.27 (npu0/small)

R1 rule: instances in descending order of their best isolated option (cpu / npu0 / npu1 with the instance's own variant binary); each takes its best option still free (NPU <= 1 instance, CPU unlimited). R2: all CPU.

## rbybys (N = 3, 13 candidates)

| instance | cpu | npu0 | npu1 | (req/s, isolated) |
|---|---|---|---|---|
| resnet50_big | 52.50 | 22.07 | 21.83 | |
| yolov3_big | 3.37 | 10.23 | 9.50 | |
| yolov3_small | 3.37 | 14.27 | 12.80 | |

- order: resnet50_big, yolov3_small, yolov3_big
- **R1 = combination_13**: resnet50_big:cpu, yolov3_big:npu1, yolov3_small:npu0
- **R2 = combination_1**: all cpu

- beta 0.1: rate per instance at 1x = {'resnet50_big': 5.25, 'yolov3_big': 1.427, 'yolov3_small': 1.427} (x4: four times)
- beta 0.25: rate per instance at 1x = {'resnet50_big': 13.125, 'yolov3_big': 3.567, 'yolov3_small': 3.567} (x4: four times)

## rbrsybys (N = 4, 21 candidates)

| instance | cpu | npu0 | npu1 | (req/s, isolated) |
|---|---|---|---|---|
| resnet50_big | 52.50 | 22.07 | 21.83 | |
| resnet50_small | 52.50 | 24.43 | 24.07 | |
| yolov3_big | 3.37 | 10.23 | 9.50 | |
| yolov3_small | 3.37 | 14.27 | 12.80 | |

- order: resnet50_big, resnet50_small, yolov3_small, yolov3_big
- **R1 = combination_7**: resnet50_big:cpu, resnet50_small:cpu, yolov3_big:npu1, yolov3_small:npu0
- **R2 = combination_1**: all cpu

- beta 0.1: rate per instance at 1x = {'resnet50_big': 5.25, 'resnet50_small': 5.25, 'yolov3_big': 1.427, 'yolov3_small': 1.427} (x4: four times)
- beta 0.25: rate per instance at 1x = {'resnet50_big': 13.125, 'resnet50_small': 13.125, 'yolov3_big': 3.567, 'yolov3_small': 3.567} (x4: four times)

## Run plan

- main: (13 + 21) candidates x beta [0.1, 0.25] x rate [1, 4] = 136 runs
- repeat: rbrsybys R1 (combination_7) at 2x, 3 repeats x 2 beta = 6 runs
- total 142 runs; estimate 45 s/run -> 1.77 h (to be replaced by the measured run time)

## Criteria (fixed before the pilot; values are not changed after seeing results)

- (A) every run: timestamp conservation (gen, flow) = 0, trace_check ok, failed = 0
- (B) per workload (family, beta, k), per instance: generated spread over candidates <= 2 % of the max (instances with < 200 generated: <= 2 requests)
- (C1) 1x: R1 D_macro <= 0.02 and every instance completed/generated >= 0.95; (C2) 4x: R1 has an instance with q_i >= 0.05 (each pilot family)
- (D) lowest-rate instance generated >= 30 in the window; (E) realization >= 0.95 for every instance
- (F) residency violations 0, NPU load failures 0
- (G, reference) repeat: F relative range <= 1 %, D_macro absolute range <= 0.01
- beta: among betas passing A, B, F, the smallest passing C1, C2, D, E; none -> stop and report
