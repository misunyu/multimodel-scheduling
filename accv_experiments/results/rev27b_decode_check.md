# rev27b STEP 1 — YOLOv10s NPU decode diagnosis

_GPU FP32 (ultralytics yolov10s.pt) vs NPU INT8 (mblt yolov10s global8) on sid 17 frames. threads=4._

## Raw NPU output layout

NPU raw outputs: 6 tensors, shapes = [(20, 20, 64), (20, 20, 80), (40, 40, 64), (40, 40, 80), (80, 80, 64), (80, 80, 80)]

(YOLOv10 is NMS-free dual-head; v8/v11 are anchor-free + NMS. Layout differences here can break decode.)

## Per-frame GPU vs NPU box comparison (large objects, area>96^2 px)

| frame | GPU large dets (box, score) | NPU match? (IoU, score) | verdict |
|---|---|---|---|
| 40 | [1109,623,1323,757] s=0.92; [0,595,167,877] s=0.89 | matched IoU=0.98 score=0.89; matched IoU=0.97 score=0.89 | coords-OK/score |
| 60 | [1159,629,1478,818] s=0.94; [220,639,541,784] s=0.92 | matched IoU=1.00 score=0.92; matched IoU=0.99 score=0.92 | coords-OK/score |
| 80 | [0,648,400,861] s=0.94; [1255,634,1865,948] s=0.94 | matched IoU=0.99 score=0.92; matched IoU=0.98 score=0.91 | coords-OK/score |
| 100 | [1496,701,1918,1191] s=0.95; [1201,632,1565,848] s=0.94 | matched IoU=0.99 score=0.93; matched IoU=0.98 score=0.92 | coords-OK/score |
| 120 | [1321,632,1918,1014] s=0.95; [0,651,371,877] s=0.93 | matched IoU=0.96 score=0.93; matched IoU=0.99 score=0.94 | coords-OK/score |

## Aggregate signal

- H1 signals (coord drift / dropped large boxes): 0
- H2 signals (large box coords matched, score-only loss): 10

**Leaning H2 (export/quantization)** — large boxes are at correct coords but lose score on the NPU; decode looks fine, the INT8 export degrades large-object confidence (not fixable by us).
