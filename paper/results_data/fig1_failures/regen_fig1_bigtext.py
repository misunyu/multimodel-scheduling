"""Re-label vis_failure_examples.pdf with LARGER top titles + bottom legend.

WHY THIS EXISTS (do not re-run generate_fig1.py for this):
  generate_fig1.py selects the failure frames by running live GPU(FP32) +
  NPU(INT8) inference, which (a) needs the NPU host and (b) is not guaranteed
  to reselect the identical frames. The committed PDF was also produced by a
  newer variant than generate_fig1.py (its legend text differs). So to change
  ONLY the font size we must NOT regenerate the content.

WHAT THIS DOES:
  Reuses the exact rasterized panels already baked into the committed figure
  (photo + green/red cv2 boxes, 1970x1232 @300ppi each) by extracting them
  with `pdfimages`, then redraws the SAME 2x2 layout with the SAME per-panel
  titles and the SAME bottom legend -- only the title/legend font sizes are
  enlarged. No inference, no NPU, no upscaling; panel pixels are untouched.

  Panel<->title mapping was verified against the baked red "(NPU miss)" class
  labels (pdfimages order != naive pdftotext reading order: the top-right and
  bottom-left titles are swapped).

Source (pristine Type3 original):  figs_backup_type3/vis_failure_examples.pdf
Output:                            paper/figures/vis_failure_examples.pdf
"""
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # Type 3 -> TrueType(42)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "figs_backup_type3" / "vis_failure_examples.pdf"
OUT = ROOT / "paper" / "figures" / "vis_failure_examples.pdf"

# --- font sizes (originals were title=9, legend=10) ---
TITLE_FS = 14
LEGEND_FS = 15
# Wrap long titles so the enlarged font stays within one panel width
# (the two top titles otherwise collide horizontally).
WRAP = 44

# Panels in matplotlib draw order (== pdfimages index == visual row-major:
# TL, TR, BL, BR). Titles verified against the baked red NPU-miss labels.
TITLES = [
    "sid 2, frame 5 — NPU misses: person, person, traffic_light, "
    "traffic_light, traffic_light",                                    # panel-000 TL
    "sid 13, frame 41 — NPU misses: traffic_light, traffic_light, "
    "traffic_light",                                                   # panel-001 TR
    "sid 17, frame 163 — NPU misses: person, person, person",     # panel-002 BL
    "sid 22, frame 1 — NPU misses: person",                       # panel-003 BR
]

# Legend text taken verbatim from the committed figure.
GREEN = (60 / 255, 200 / 255, 60 / 255)
RED = (230 / 255, 40 / 255, 40 / 255)
LEGEND = [
    (GREEN, "All GPU (FP32) detections"),
    (RED, "Missed by INT8 NPU (GPU detects)"),
]

# Layout constants copied verbatim from generate_fig1.render().
COLS, ROWS = 2, 2
FIGSIZE = (7.5 * COLS, 4.7 * ROWS)


def extract_panels(pdf, workdir):
    subprocess.run(["pdfimages", "-png", str(pdf), str(workdir / "panel")],
                   check=True)
    pngs = sorted(workdir.glob("panel-*.png"))
    if len(pngs) != ROWS * COLS:
        raise SystemExit(f"expected {ROWS*COLS} panels, got {len(pngs)}: {pngs}")
    return [Image.open(p).convert("RGB") for p in pngs]


def main():
    with tempfile.TemporaryDirectory() as td:
        panels = extract_panels(SRC, Path(td))

        fig, axes = plt.subplots(ROWS, COLS, figsize=FIGSIZE, squeeze=False)
        for k, img in enumerate(panels):
            ax = axes[k // COLS][k % COLS]
            ax.imshow(img, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(textwrap.fill(TITLES[k], WRAP), fontsize=TITLE_FS)

        handles = [mpatches.Patch(color=c, label=t) for c, t in LEGEND]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.01),
                   frameon=False)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        # dpi=300 matches the panels' native 300ppi (no up/down-scaling).
        fig.savefig(OUT, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    print(f"wrote {OUT}  (title_fs={TITLE_FS}, legend_fs={LEGEND_FS})")


if __name__ == "__main__":
    main()
