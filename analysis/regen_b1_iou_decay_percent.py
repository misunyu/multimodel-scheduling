"""Regenerate b1_iou_decay.pdf with panel (c) on a PERCENT scale.

Standalone regeneration for the paper's switch to reporting sAP in percent
(main_vision.tex L169). Does NOT modify the original generator
`analysis/b1_temporal_iou_decay.py`; it re-reads that generator's canonical,
already-emitted CSVs and re-draws the 3-panel figure.

Scope of the percent change (per task):
  panel (a) mean self-IoU    -> ABSOLUTE, no conversion (raw IoU in [0,1])
  panel (b) fraction<0.5     -> ABSOLUTE, no conversion (a fraction in [0,1])
  panel (c) sAP-loss proxy   -> PERCENT (proxy x100, y-axis 0-22);
            measured L_b dotted lines at 9.8 / 3.0 / 0.1 (= -L_b x100).

Sources:
  analysis/b1_temporal_iou_decay.csv    (panels a, b: mean_iou, frac_iou_lt_0.5)
  analysis/b1_staleness_sap_proxy.csv   (panel c: sap_loss_proxy)
Outputs:
  analysis/b1_iou_decay_percent.pdf
  paper/figures/b1_iou_decay.pdf        (copy used by main_vision.tex)
"""
import shutil
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # Type 3 -> TrueType(42)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DECAY_CSV = ROOT / "analysis/b1_temporal_iou_decay.csv"
PROXY_CSV = ROOT / "analysis/b1_staleness_sap_proxy.csv"
OUT_PDF = ROOT / "analysis/b1_iou_decay_percent.pdf"
PAPER_PDF = ROOT / "paper/figures/b1_iou_decay_percent.pdf"

FPS = 30.0
BINS = ["small", "medium", "large"]
COLORS = {"small": "#4477AA", "medium": "#EE6677", "large": "#228833"}
# Measured staleness loss L_b (Table 3), in the SAME percent units as panel (c).
# -L_b x100 -> dotted reference lines at large 9.8, medium 3.0, small 0.1.
L_B_MEASURED_PCT = {"small": 0.13, "medium": 2.95, "large": 9.79}


def series(df, size, xcol, ycol):
    sub = df[df["size"] == size].sort_values("delta_frames")
    return sub[xcol].tolist(), sub[ycol].tolist()


def main():
    decay = pd.read_csv(DECAY_CSV)
    proxy = pd.read_csv(PROXY_CSV)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.5))

    # (a) mean self-IoU -- ABSOLUTE, no conversion
    for b in BINS:
        xs, ys = series(decay, b, "delta_ms", "mean_iou")
        ax1.plot(xs, ys, "o-", color=COLORS[b], label=b, lw=1.8, ms=5)
    ax1.set_xlabel("temporal offset δ (ms)")
    ax1.set_ylabel("mean self-IoU")
    ax1.set_title("(a) Same-object IoU vs delay")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="GT size bin", frameon=False)

    # (b) fraction self-IoU < 0.5 -- ABSOLUTE, no conversion
    for b in BINS:
        xs, ys = series(decay, b, "delta_ms", "frac_iou_lt_0.5")
        ax2.plot(xs, ys, "s-", color=COLORS[b], label=b, lw=1.8, ms=5)
    ax2.set_xlabel("temporal offset δ (ms)")
    ax2.set_ylabel("fraction with self-IoU < 0.5")
    ax2.set_title("(b) Fraction below COCO match threshold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="GT size bin", frameon=False)

    # (c) sAP-loss proxy -- PERCENT (proxy x100), measured L_b dotted (x100)
    for b in BINS:
        xs, ys = series(proxy, b, "delta_ms", "sap_loss_proxy")
        ax3.plot(xs, [y * 100 for y in ys], "^-", color=COLORS[b],
                 label=b, lw=1.8, ms=5)
        ax3.axhline(L_B_MEASURED_PCT[b], color=COLORS[b], ls=":", lw=1.3, alpha=0.8)
    ax3.set_xlabel("temporal offset δ (ms)")
    ax3.set_ylabel(r"estimated staleness sAP loss (%)")
    ax3.set_title("(c) Estimated vs measured sAP loss")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 22)  # headroom so the δ=5 large proxy (20.8%) is not clipped
    ax3.legend(frameon=False, fontsize=8)

    # secondary top axis: frames @30fps (all panels)
    for ax in (ax1, ax2, ax3):
        secax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * FPS / 1000.0, lambda x: x * 1000.0 / FPS))
        secax.set_xlabel("δ (frames @30fps)")

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    shutil.copyfile(OUT_PDF, PAPER_PDF)
    print(f"wrote {OUT_PDF}")
    print(f"copied -> {PAPER_PDF}")

    # ---- verification ----
    print("\n== panel (c) percent-scale checks ==")
    print(f"  measured L_b dotted lines (%): "
          f"large {L_B_MEASURED_PCT['large']:.1f}  "
          f"medium {L_B_MEASURED_PCT['medium']:.1f}  "
          f"small {L_B_MEASURED_PCT['small']:.1f}   (tex/task: 9.8 / 3.0 / 0.1)")
    for b in BINS:
        _, ys = series(proxy, b, "delta_ms", "sap_loss_proxy")
        print(f"  proxy {b:>6s}: {ys[0]*100:5.2f} -> {ys[-1]*100:5.2f} %  (33->167 ms)")
    print("  y-axis range: 0-22 (percent; δ=5 large 20.8% marker not clipped)")
    a_lo = decay["mean_iou"].min(); a_hi = decay["mean_iou"].max()
    print(f"  panel (a) self-IoU stays absolute: [{a_lo:.3f}, {a_hi:.3f}] (no x100)")
    b_lo = decay["frac_iou_lt_0.5"].min(); b_hi = decay["frac_iou_lt_0.5"].max()
    print(f"  panel (b) fraction stays absolute: [{b_lo:.3f}, {b_hi:.3f}] (no x100)")


if __name__ == "__main__":
    main()
