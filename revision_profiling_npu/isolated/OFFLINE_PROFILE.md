# Isolated profile (Phase N1 §4)

closed-loop single instance, warm-up 5 s + window 30 s; throughput = completed / window; service = dequeue -> output put (preprocess + host stages + NPU round trip + postprocess). Cmax[m] = max throughput over (device, variant) of model m.

| model | variant | device | req/s | service p50 ms | service p95 ms | driver call p50 ms | NPU / CPU | valid |
|---|---|---|---|---|---|---|---|---|
| resnet50 | big | npu0 | 22.07 | 44.69 | 48.66 | 42.17 | 0.42 | True |
| resnet50 | big | npu1 | 21.83 | 44.98 | 49.20 | 42.62 | 0.42 | True |
| resnet50 | common | cpu | 52.50 | 18.84 | 20.37 | - | 1.00 | True |
| resnet50 | small | npu0 | 24.43 | 40.39 | 43.18 | 38.23 | 0.47 | True |
| resnet50 | small | npu1 | 24.07 | 40.97 | 44.25 | 38.68 | 0.46 | True |
| yolov3 | big | npu0 | 10.23 | 97.68 | 98.83 | 88.66 | 3.04 | True |
| yolov3 | big | npu1 | 9.50 | 105.48 | 107.13 | 96.43 | 2.82 | True |
| yolov3 | common | cpu | 3.37 | 300.53 | 308.72 | - | 1.00 | True |
| yolov3 | small | npu0 | 14.27 | 70.04 | 71.38 | 61.00 | 4.24 | True |
| yolov3 | small | npu1 | 12.80 | 77.85 | 79.46 | 68.80 | 3.80 | True |

## Cmax (reference rate = beta x Cmax x k)

- resnet50: 52.50 req/s at cpu / common
- yolov3: 14.27 req/s at npu0 / small

## npu0 vs npu1 (PCIe Gen4x8 vs Gen2x1)

- resnet50 big: npu0 22.07, npu1 21.83 req/s (npu1/npu0 = 0.989)
- resnet50 small: npu0 24.43, npu1 24.07 req/s (npu1/npu0 = 0.985)
- yolov3 big: npu0 10.23, npu1 9.50 req/s (npu1/npu0 = 0.928)
- yolov3 small: npu0 14.27, npu1 12.80 req/s (npu1/npu0 = 0.897)
