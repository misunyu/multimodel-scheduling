# EXP-FIG1-LEGEND — Stage A audit (read-only) + Case verdict

Target: Figure 1 `vis_failure_examples.pdf` (caption "Qualitative quantization failures").
Generators (identical logic): `accv_experiments/scripts/phase_vis_failure_examples.py` and
`paper/results_data/fig1_failures/generate_fig1.py`. The .tex includes `vis_failure_examples.pdf`
(main_vision.tex:191); canonical copy `paper/figures/vis_failure_examples.pdf`
(== results_data copy, sha 251762853c…).

## Stage A — box definitions (code evidence)
- **GREEN box = ALL GPU (FP32) detections** of the mapped AHD classes (definition (ii), NOT
  intersected with NPU). Evidence: `generate_fig1.py:179-181` (== phase_vis:178-181):
  `for bb,sc,lb in zip(frame_info["gpu_bb"], ...): cv2.rectangle(... (60,200,60) ...)` — iterates
  over every GPU detection. Header comment line 6: "GREEN = GPU detection (FP32, the reference)".
- **RED box = GPU∖NPU (subset)**: GPU detections of safety-critical small classes (person, bicycle,
  motorcycle, traffic_light, stop_sign) that have NO same-class NPU match. Evidence: missed-set
  computation `:128-138` (`iou_vals = iou_xyxy(gpu_bb[k], npu_bb[same_cls]); if (iou_vals>IOU_MATCH).any(): matched`),
  drawn `:182-185` in red (40,40,230). Comment line 7: "RED = object the GPU caught but the NPU did NOT detect".
- **Matching / thresholds (fixed in code)**: `IOU_MATCH = 0.4` (`:53`), same-class; detector
  `CONF = 0.25`, NMS `IOU = 0.45` (from `step0_compare_devices.py` import). Frames: panel/sid/frame =
  (1,2,5),(2,13,41),(3,17,163),(4,22,1) per `data/selected_frames.csv`.
- **Legend strings (current)**: `:223` green = "GPU (FP32) detection"; `:224-225` red = "object NPU
  (INT8) misses but GPU detects".

## Case verdict — **Case 2**
GREEN is *all* GPU detections, so every RED object (also a GPU detection) is drawn GREEN underneath
→ the two legend categories overlap. The legend ("GPU detection" vs "NPU misses but GPU detects") is
logically ambiguous: it does not convey that green should mean "both devices detect" as the contrast
to red. The figure's RED claim (NPU misses small safety-critical objects) is correct; only the
green/red definitional overlap is the problem. → directive routes Case 2 to Stage B-2 (redraw green
as GPU∩NPU).

## Modification status — NOT performed (blocked by §4)
Stage B-2 (and even B-1 legend-only) require re-running the generator to re-emit the PDF. Both
generators produce the figure by **live GPU + NPU inference** (`generate_fig1.py:105-122`
`load_gpu()`, `preload_npu_instances(1,"global8")`, `gpu_predict`, `npu_predict`). There is **no
saved per-frame detection cache**: `data/selected_frames.csv` holds only panel/sid/frame +
npu-missed class *names* (no box coordinates, no GPU/NPU detection arrays); no .npz/.npy/.pkl
detection dump exists anywhere. Redrawing green as GPU∩NPU needs the full per-frame GPU and NPU
detection boxes, which can only be obtained by NPU re-inference.

§4 forbids measurement / NPU re-inference and states: "저장된 결과가 없어 재추론이 필요하면
중단·보고(범위 밖)." Therefore the figure was **left unchanged**; no PDF regenerated, no backup
created (nothing modified), and per the directive the paper .tex was not touched.

## Note for separate channel (human decision; out of this directive's scope)
To actually fix Case 2, one of:
- (a) Authorize a one-off regeneration (re-runs GPU+NPU on the 4 saved frames in
  `data/selected_frames.csv`) that redraws green = GPU∩NPU using the existing fixed thresholds
  (IOU_MATCH=0.4, CONF=0.25) — this is B-2 but needs the now-forbidden NPU re-inference; or
- (b) Clarify in the .tex caption (separate channel) that green = all GPU detections including the
  red subset, since red (the paper's claim) is already correct.
Either path is outside this read-only/no-re-inference scope.

## Compliance
No measurement/re-inference/re-compile run; figure PDF and paper .tex unchanged; thresholds not
altered; Case not re-interpreted (Case 2 reported as found). Output = this audit file only.
