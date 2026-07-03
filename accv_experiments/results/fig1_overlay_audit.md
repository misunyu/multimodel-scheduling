# EXP-FIG1-OVERLAY — legend text overlay (no re-inference)

Target: `paper/figures/vis_failure_examples.pdf` (Figure 1). Replaced only the in-figure legend
**text** via vector overlay; panel images, boxes, color swatches, and layout untouched. No GPU/NPU
re-inference; `generate_fig1.py` not run. Paper .tex not modified.

## Stage A — legend location (Case O-1)
Rendered original at 150 dpi (`orig-1.png`, 2128×1421). The legend sits in the **bottom white
margin band below the 2×2 panels** (matplotlib `fig.legend(loc="lower center",
bbox_to_anchor=(0.5,-0.01))`, `tight_layout(rect=[0,0.04,1,1])`), with NO overlap onto panel
images → **Case O-1** (clean overlay possible). Detected coords (px → PDF pts, scale 0.4798 pt/px,
page 1021.02×681.812):
- green swatch: px x652-695 (PDF x312.8-333.5), y≈13.4-21.1 pt — **preserved**
- red swatch: px x975-1018 (PDF x467.8-488.4) — **preserved**
- green text (old "GPU (FP32) detection"): px x712-931 (PDF x341.6-446.7)
- red text (old "object NPU (INT8) misses but GPU detects"): px x1035-1471 (PDF x496.6-705.8)
Captures: `orig_legend_band.png` (before), `new_legend_band.png` (after), `coords.json`.

## Stage B — method B-i (reportlab overlay + pypdf merge)
Built a same-size overlay PDF (reportlab): white rectangles covering ONLY the old text spans (between
each swatch and the next element — swatches not covered), then new text; merged onto the original
page with `pypdf merge_page` (original loaded from the backup). Font: **DejaVuSans** (matplotlib's
default, registered from mpl-data) for visual match. New labels (per §0, fixed):
- green: "All GPU (FP32) detections"
- red: "Missed by INT8 NPU (GPU detects)"
Font size auto-fit with `stringWidth` to avoid intruding on the red swatch: common **9.25 pt**
(green new width 118.2 pt fits the 120.3 pt slot before the red swatch; red 159.4 pt within 514.6 pt).
Slightly smaller than the original 10 pt, per §2 (reduce to fit; no intrusion). Script:
`accv_experiments/results/fig1_overlay/make_overlay.py`.

## Verification
- **Panels bit-identical**: rendered new at 150 dpi; for the panel region (y < 1335 px, above the
  legend row) max pixel diff = **0**, nonzero-diff pixels = **0** vs original → boxes/images/layout
  unchanged. Only the legend row changed (6792 px).
- **Page geometry preserved**: `pdfinfo` before/after both `Pages: 1`, `Page size: 1021.02 ×
  681.812 pts` → `width=\linewidth` ratio unchanged.
- **Legend renders correctly**: swatches (green/red) intact, new text aligned, not clipped, not
  overlapping panels (visual check `new_legend_band.png`).

## Files
- Updated (same path/name): `paper/figures/vis_failure_examples.pdf` (sha 635eef1fc6f98957).
  NOTE: `paper/results_data/fig1_failures/vis_failure_examples.pdf` is a separate copy of the OLD
  figure (not the .tex-included path) and was left unchanged — flag for separate-channel sync if desired.
- Backup of original: `accv_experiments/results/fig1_overlay/vis_failure_examples_prev.pdf` (sha 251762853c00d7d8).
- Overlay + audit assets: `accv_experiments/results/fig1_overlay/` (make_overlay.py, overlay.pdf,
  coords.json, orig/new renders, legend bands).

## Compliance
No measurement/re-inference/re-compile; generate_fig1.py not run; panel pixels unchanged (diff=0);
swatches untouched; page dims preserved; filename/path unchanged; paper .tex not touched; new wording
fixed per §0.
