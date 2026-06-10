# Fig.1 (fig:failures) — qualitative quantization failures

`vis_failure_examples.pdf` is the figure used in the paper (`\includegraphics{vis_failure_examples.pdf}`),
copied here byte-for-byte from `paper/figures/vis_failure_examples.pdf`.

## How it was generated
- Generator: `generate_fig1.py` (copy of `accv_experiments/scripts/phase_vis_failure_examples.py`).
- It loads YOLOv11s on GPU (FP32) and NPU (INT8, mxq `b2441f9d`, global8), scans Argoverse-HD
  frames, and selects frames where the NPU drops a safety-critical small object the GPU keeps.
- Overlay: GPU detections = green; objects the NPU misses but GPU detects = red.
- Selected panels (`data/selected_frames.csv`):

| panel | sid | frame | NPU-missed classes |
|---|---|---|---|
| 1 | 2  | 5   | person, person, traffic_light ×3 |
| 2 | 13 | 41  | traffic_light ×3 |
| 3 | 17 | 163 | person ×3 |
| 4 | 22 | 1   | person |

## Reproduce
`generate_fig1.py` requires the NPU device + Argoverse-HD frames (GPU+NPU inference), so it is
**not** a pure-extraction step like the tables/Fig.2. Re-running on the same platform
(RTX 5090 + Mobilint MLA100, mxq `b2441f9d`, global8) regenerates the same PDF.
Minor font/margin differences are acceptable; the frames and boxes are identical.

```
python generate_fig1.py      # outputs paper/figures/vis_failure_examples.pdf
```
