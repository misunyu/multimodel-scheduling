# EXP-AUDIT-RES — YOLO detector input-resolution consistency audit (read-only)

Scope: confirm from code evidence the inference input resolution + preprocessing for the GPU
(Ultralytics FP32) and NPU (vendor INT8 .mxq) paths, across the four detectors. No measurement /
re-export / re-compile; NPU not loaded; paper .tex untouched.

## Resolution / preprocessing table
| model | path | input res | resize / preprocess | evidence (file:line) | 확정/미상 |
|---|---|---|---|---|---|
| YOLOv11s | GPU | 640×640 | Ultralytics letterbox (aspect-preserving, pad 114), RGB /255 | `step0_compare_devices.py:30` `IMG_SIZE=640`; `phase_rev6_sweep.py:127,130,132` `FGModelGPUGeneric.predict(imgsz=IMG_SIZE)` | 확정 |
| YOLOv11s | NPU | 640×640 | mblt LetterBox 640 (aspect-preserving) + Normalize "cv"; boxes rescaled to original via gain=min(640/h,640/w) | `mblt .../yolo11.py:67` `img_size [640,640]`; `_step_d_common.py:142,150-153` `m.preprocess`, `gain=min(IMG_SIZE/h0,IMG_SIZE/w0)` | 확정 |
| YOLOv8n | GPU | 640×640 | as YOLOv11s GPU | `phase_rev26_yolov8n.py:150` `FGModelGPUGeneric(...)` (no imgsz override) → IMG_SIZE=640 | 확정 |
| YOLOv8n | NPU | 640×640 | mblt LetterBox 640 + Normalize "cv" | `mblt .../yolov8.py:29` `img_size [640,640]`; npu_infer 640 gain | 확정 |
| YOLOv8s | GPU | 640×640 | as above | `phase_rev27_generality.py:130` `FGModelGPUGeneric(DET["ultralytics_pt"])` (no imgsz override) | 확정 |
| YOLOv8s | NPU | 640×640 | mblt LetterBox 640 + Normalize "cv" | `mblt .../yolov8.py:67` `img_size [640,640]`; npu_infer 640 gain | 확정 |
| YOLO12s | GPU | 640×640 | as above | `phase_rev27c_step0.py:29` `FGModelGPUGeneric("yolo12s.pt")` (no imgsz override) | 확정 |
| YOLO12s | NPU | 640×640 | mblt LetterBox 640 + Normalize "cv" | `mblt .../yolo12.py:67` `img_size [640,640]`; npu_infer 640 gain | 확정 |

Notes:
- GPU resolution is global: `step0_compare_devices.py:30 IMG_SIZE=640`, used by every Ultralytics
  adapter (`FGModelGPUGeneric`, `FGModelGPU`, `FGModelCPU`, `UltralyticsRunner`) via `imgsz=IMG_SIZE`.
  None of the per-model scripts (rev26/rev27/rev27c) override `imgsz`.
- NPU resolution is read from the mblt-model-zoo class config (`pre_cfg.LetterBox.img_size=[640,640]`)
  that loads each vendor .mxq, plus `npu_infer` which letterboxes to and unpads from 640. The raw
  .mxq input-tensor shape was NOT separately introspected (that would require loading the engine on
  the NPU, which is out of scope for this read-only audit); the 640 figure rests on the loader config
  + box-rescale code, which the engine must match to produce valid detections.

## Q1 — model-to-model consistency
**ALL FOUR detectors use 640×640 on both paths** (confirmed). No per-model resolution difference.

## Q2 — device-to-device consistency (per model)
**For every model, GPU and NPU use the same 640×640 input and the same letterbox resize**
(aspect-ratio-preserving with padding; both unpad/rescale detections back to original coords).
Confirmed for resolution and resize method.
- Preprocessing nuance (not a resolution issue): GPU = Ultralytics RGB, /255; NPU = mblt "cv"-style
  Normalize and OpenCV/BGR image reading (`mblt .../utils/results.py:61-73`). Exact channel-order
  handling inside each vendor .mxq is not code-introspectable here, but the NPU pipeline reproduces
  Table 1 closely (compiler-check vendor re-measure ≈ published Table 1), so the mblt preprocessing
  is consistent with what each .mxq expects. Resolution + letterbox geometry are identical across
  devices; the only un-pinned detail is the internal normalization/channel convention of the vendor
  INT8 graph, which does not change the 640 input size.

## Size-classification coordinate system (§1-4)
Size bins (small/medium/large) are computed in **original Argoverse-HD coordinates (1920×1200)**, not
inference resolution. Evidence: `_step_d_common.py:58-63` builds `coco_gt` from `val.json` (original
boxes); detections are rescaled to original coords before eval (`_step_d_common.py:150-156` NPU
unpad/÷gain; Ultralytics GPU boxes are already original-frame); `_step_d_common.py:272` runs standard
`COCOeval(...,"bbox")` whose default areaRng (small <32², medium 32²–96², large >96²) is in
original-pixel² (`stats[3/4/5]` → sap_small/medium/large). So size stratification is independent of
the 640 inference resolution. 확정.

## Verdict
- Q1: 일치 (네 모델 모두 640×640).
- Q2: 일치 (각 모델 GPU/NPU 모두 640×640 + aspect-preserving letterbox). 단, 벤더 INT8 그래프
  내부의 정규화/채널 컨벤션 세부는 코드로 완전 확정 불가(해상도·letterbox 기하는 동일).
- Size 좌표계: 원본 1920×1200 기준 (추론 해상도와 무관).

## Forbiddens compliance
No measurement/re-export/re-compile; NPU not loaded; no "probably 640" claimed without a code line;
paper .tex not opened or modified. Output is this fact report only.
