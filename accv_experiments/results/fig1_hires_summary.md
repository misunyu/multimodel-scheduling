# EXP-FIG1-HIRES — Figure 1 high-resolution regeneration

Goal: fix blurry panels (was 657×411 @ 100 ppi) + remove leftover duplicate legend text from the
earlier overlay + improve box-label readability — by **regenerating** at high resolution with one
permitted re-inference on the 4 fixed panel frames. Paper .tex untouched.

## Re-inference (1×, 4 panels only)
- Frames (fixed, no re-selection): (sid 2, frame 5), (sid 13, frame 41), (sid 17, frame 163),
  (sid 22, frame 1) — from `data/selected_frames.csv`.
- Models (== main): GPU `yolo11s.pt` sha `85a76fe8…`; NPU legacy mxq `b2441f9d…` (global8), located
  via `_find_yolo11s_mxq("global8")`.
- Thresholds fixed (unchanged): IOU_MATCH=0.4, CONF=0.25, NMS IoU=0.45, input 640 letterbox; boxes
  drawn in original 1920×1200 coords. Color rule unchanged (green = all GPU dets, red = NPU-missed
  safety-critical subset).
- Script: `accv_experiments/scripts/fig1_hires_regen.py` (helpers replicated verbatim from
  generate_fig1.py; that module not imported because its module-level rev7_pin.json path is broken).
  GPU cleared after run; the trailing AttributeError (cm.dispose_npu_for) fired AFTER savefig and is
  non-fatal — the NPU SDK logged "Model disposed" and `nvidia-smi` shows GPU clear.

## §2-2 box-consistency gate — PASS (NPU-miss lists identical to existing caption/titles)
| panel | recomputed NPU-miss classes | expected |
|---|---|---|
| sid 2, f5 | person, person, traffic_light, traffic_light, traffic_light | same ✓ |
| sid 13, f41 | traffic_light, traffic_light, traffic_light | same ✓ |
| sid 17, f163 | person, person, person | same ✓ |
| sid 22, f1 | person | same ✓ |
All four match → boxes consistent with the caption; no caption change needed.

## Verification
- **Resolution (pdfimages -list)**: before 657×411 @ **100 ppi** (4 panels) → after 1970×1232 @
  **300 ppi** (4 panels). ~3× linear, near native 1920×1200 source.
- **Aspect ratio (pdfinfo)**: page size **1021.02 × 681.812 pts** — *identical* to the original
  (ratio 1.4975) → `width=\linewidth` layout unaffected.
- **Duplicate legend removed (pdftotext)**: extracted text contains ONLY the new wording —
  "All GPU (FP32) detections" and "Missed by INT8 NPU (GPU detects)". The old strings
  ("GPU (FP32) detection", "object NPU (INT8) misses but GPU detects") are ABSENT (regeneration, not
  overlay → no leftover text layer).
- **Label readability**: box line widths and confidence labels scaled with image width
  (green ~3 px, red ~6 px, label font ~1.05× at native); labels e.g. "traffic_light 0.28 (NPU miss)"
  render crisply (visual check `hires_panel0.png`).
- **Legend**: single clean legend, swatches intact, no overlap (visual check `hires_legend.png`).

## Files
- Updated (same path/name): `paper/figures/vis_failure_examples.pdf` (300 ppi panels).
- Backups: pre-hires (overlay-modified) `accv_experiments/results/fig1_overlay/vis_failure_examples_prev_lowres.pdf`;
  original pre-overlay `…/vis_failure_examples_prev.pdf`.
- Renders: `accv_experiments/results/fig1_overlay/hires-1.png`, `hires_panel0.png`, `hires_legend.png`;
  before: `orig-1.png`, `orig_legend_band.png`.
- Script: `accv_experiments/scripts/fig1_hires_regen.py`; stdout `accv_experiments/results/fig1_hires_stdout.log`.
- NOTE: `paper/results_data/fig1_failures/vis_failure_examples.pdf` is a separate OLD copy (not the
  .tex-included path) — left unchanged; sync via separate channel if desired.

## Compliance
Re-inference limited to the 4 panel frames (allowed once for this figure); no other measurement/
re-train/re-compile. Thresholds/frames/box-meaning/labels-content unchanged. Filename/path/aspect
preserved. Paper .tex not modified.
