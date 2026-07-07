# Reproduction: fig:iou_decay

Paper label: **`fig:iou_decay`** (main_vision.tex, `\includegraphics{b1_iou_decay_percent.pdf}`).

## Caption summary
Detector-independent temporal probe using Argoverse-HD ground truth. Using only
the GT annotations, each object's box is compared with the same object at later
timestamps (self-IoU), and an estimated sAP loss is derived from that decay.
Raw self-IoU decays fastest for **small** objects, but the estimated sAP loss is
largest for **large** objects and tracks the measured per-size staleness losses
from Table 2 (`tab:decomp`), drawn as dotted horizontal reference lines.

## Input data (`./data/`)
- `b1_temporal_iou_decay.csv` — per-size (small/medium/large) mean self-IoU and
  fraction of boxes with self-IoU < 0.5, as a function of temporal offset δ
  (columns: `size`, `delta_frames`, `delta_ms`, `mean_iou`, `frac_iou_lt_0.5`).
  Feeds panels (a) and (b).
- `b1_staleness_sap_proxy.csv` — per-size sAP-loss proxy vs temporal offset δ
  (columns include `size`, `delta_frames`, `delta_ms`, `sap_loss_proxy`).
  Feeds panel (c).

Both CSVs are copied verbatim from `analysis/`, where they were emitted by the
canonical generator `analysis/b1_temporal_iou_decay.py`.

## Run
```
cd paper/reproduction/fig_iou_decay
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Output
- `b1_iou_decay_percent.pdf` — the 3-panel figure used by `main_vision.tex`.

## Transform / scale conventions
- Panel (a) mean self-IoU: **absolute**, raw IoU in [0, 1] (no conversion).
- Panel (b) fraction self-IoU < 0.5: **absolute**, a fraction in [0, 1] (no conversion).
- Panel (c) sAP-loss proxy: **percent** — `sap_loss_proxy × 100`, y-axis fixed to
  0–22 so the δ=5 large-object marker (20.8%) is not clipped.
- Dotted reference lines in panel (c) = measured per-size staleness `−L_b × 100`
  from Table 2: large **9.8**, medium **3.0**, small **0.1** (%).
- Secondary top axis on all panels: δ in frames @30fps.

## Verification result
Ran with the venv interpreter from inside the folder. `b1_iou_decay_percent.pdf`
was written. Printed checks:
- Dotted measured L_b lines: large 9.8 / medium 3.0 / small 0.1 % — matches the
  fig:iou_decay caption and Table 2 staleness row (`−9.8` for large).
- Panel (c) proxy: small 0.44→1.01 %, medium 3.89→9.64 %, large 5.51→20.77 %
  (33→167 ms), so large-object estimated loss is largest and the δ=5 large
  marker peaks at 20.8 % within the 0–22 range.
- Panel (a) self-IoU stays absolute in [0.561, 0.920]; panel (b) fraction stays
  absolute in [0.009, 0.409]. No ×100 applied to (a)/(b).

**Verdict: MATCH.**

## Reproduction notes
Copied verbatim from `analysis/regen_b1_iou_decay_percent.py`. Only path
constants were changed (marked `# [reproduction] path adjusted`): inputs point to
`./data/`, output to the local folder. The `PAPER_PDF` definition and the
`shutil.copyfile(OUT_PDF, PAPER_PDF)` (plus its `print`) that copied into
`paper/figures/` are commented out. No computation or plotting logic changed.
