# Smoke (Phase N1 §3): 1 instance, 2 req/s, warm-up 5 s + window 10 s

| run | registry key | device | generated | completed | dropped | failed | conservation (gen, flow) | trace_check | realization | service p50 ms (end-to-end) | latency p50 ms | preprocess ms | driver call p50 ms | Close rc | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50_big_npu0 | antara_resnet50_big_npu | npu0 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 44.71 | 44.85 | 0.96 | 42.47 | 0 | True |
| resnet50_big_npu1 | antara_resnet50_big_npu | npu1 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 45.19 | 45.23 | 0.92 | 42.89 | 0 | True |
| resnet50_cpu | antara_resnet50_cpu | cpu | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 18.78 | 18.81 | 0.81 | - | - | True |
| resnet50_small_npu0 | antara_resnet50_small_npu | npu0 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 40.69 | 40.76 | 0.93 | 38.53 | 0 | True |
| resnet50_small_npu1 | antara_resnet50_small_npu | npu1 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 41.07 | 41.17 | 0.98 | 38.86 | 0 | True |
| yolov3_big_npu0 | antara_yolov3_big_npu | npu0 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 98.02 | 98.07 | 2.18 | 88.56 | 0 | True |
| yolov3_big_npu1 | antara_yolov3_big_npu | npu1 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 105.86 | 105.90 | 2.28 | 96.38 | 0 | True |
| yolov3_cpu | antara_yolov3_cpu | cpu | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 283.60 | 283.66 | 2.18 | - | - | True |
| yolov3_small_npu0 | antara_yolov3_small_npu | npu0 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 70.53 | 70.61 | 2.24 | 61.14 | 0 | True |
| yolov3_small_npu1 | antara_yolov3_small_npu | npu1 | 20 | 20 | 0 | 0 | (0, 0) | ok | 1.00 | 79.99 | 80.05 | 2.28 | 70.25 | 0 | True |

## ResNet50 big vs small (end-to-end service p50 / driver call p50, ms)

- resnet50 npu0: big 44.71 / 42.47, small 40.69 / 38.53, diff (big - small) +4.02 / +3.93
- resnet50 npu1: big 45.19 / 42.89, small 41.07 / 38.86, diff (big - small) +4.12 / +4.03
- yolov3 npu0: big 98.02 / 88.56, small 70.53 / 61.14, diff (big - small) +27.50 / +27.41
- yolov3 npu1: big 105.86 / 96.38, small 79.99 / 70.25, diff (big - small) +25.87 / +26.14

All runs valid, conservation 0, trace_check ok, failed 0: **True** (10 runs)
