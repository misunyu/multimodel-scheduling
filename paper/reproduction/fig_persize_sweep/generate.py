"""Regenerate persize_sweep.pdf at PERCENT scale (sAP reported as x100).

Standalone regeneration for the paper's switch to reporting sAP in percent
(main_vision.tex L169). Does NOT modify the original generator
`accv_experiments/scripts/phase_rev30_aggregate.py`; it re-reads that
generator's canonical, already-emitted per-size CSV and re-plots the loss
curves on a percent y-axis (0-12). No new measurement.

Source : accv_experiments/results/rev30_clean_resnet/rev30_persize.csv
Outputs: analysis/persize_sweep_percent.pdf
         paper/figures/persize_sweep.pdf   (copy used by main_vision.tex)

Style matches phase_rev30_aggregate.fig_persize (figsize 4.2x3.0,
tab:red/orange/blue, o/s/^ markers, no title -- caption is in the tex).
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
SRC_CSV = Path(__file__).resolve().parent / "data" / "rev30_persize.csv"  # [reproduction] path adjusted
OUT_PDF = Path(__file__).resolve().parent / "persize_sweep.pdf"  # [reproduction] path adjusted
# PAPER_PDF = ROOT / "paper/figures/persize_sweep.pdf"  # [reproduction] path adjusted (commented: paper/figures read-only)


def main():
    ps = pd.read_csv(SRC_CSV)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = ps["gpu_skip"]
    # sAP loss reported in percent: multiply the fractional loss by 100.
    ax.plot(x, ps["loss_large"] * 100, "-o", label="large", color="tab:red")
    ax.plot(x, ps["loss_medium"] * 100, "-s", label="medium", color="tab:orange")
    ax.plot(x, ps["loss_small"] * 100, "-^", label="small", color="tab:blue")
    ax.set_xlabel("GPU deadline misses (%)")
    ax.set_ylabel("sAP loss vs no contention (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 12.5)  # headroom so the 12.0 marker at 67.3% is not clipped
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    # shutil.copyfile(OUT_PDF, PAPER_PDF)  # [reproduction] path adjusted (commented: paper/figures read-only)
    print(f"wrote {OUT_PDF}")
    # print(f"copied -> {PAPER_PDF}")  # [reproduction] path adjusted (commented: paper/figures read-only)

    # ---- verification against body-cited values (main_vision.tex L293) ----
    def loss_pct(col, skip_near):
        r = ps.iloc[(ps["gpu_skip"] - skip_near).abs().idxmin()]
        return r["gpu_skip"], r[col] * 100

    print("\n== percent-scale checks vs tex L293 ==")
    s56, l56 = loss_pct("loss_large", 56)
    _, sm56 = loss_pct("loss_small", 56)
    print(f"  large@~56% skip: {l56:.1f}  (tex: 9.6)   skip={s56}")
    print(f"  small@~56% skip: {sm56:.1f}  (tex: 0.2)")
    s24, l24 = loss_pct("loss_large", 24)
    s67, l67 = loss_pct("loss_large", 67)
    print(f"  large sweep    : {l24:.1f} -> {l67:.1f}  (tex: 4.2 -> 12.0)"
          f"   skip {s24}->{s67}")
    print(f"  y-axis range   : 0-12.5 (percent; 12.0 marker not clipped)")


if __name__ == "__main__":
    main()
