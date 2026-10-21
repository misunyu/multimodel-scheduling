"""Fig. 1(b) v4: fig1b_v2 with the legend entry "Drop-aware (max $T-0.3D$)" shortened to "Drop-aware". Everything else (data, size,
fonts, colors/hatches, value labels, x labels, legend placement, no frame) is the fig1_v2.py fig1b_v2 code unchanged.
Data are read from fig1b_v2.csv as is (no recomputation). Writes fig1b_v4.{pdf,png,csv} only.

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/fig1b_v4.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "revision_exp"))

import shutil  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import plots as PL  # noqa: E402  (style only)

OUT = REPO / "revision_exp" / "out_manuscript"
W = 8.5 / 2.54
NAME = "fig1b_v4"
for ext in ["pdf", "png", "csv"]:
    assert not (OUT / f"{NAME}.{ext}").exists(), f"{NAME}.{ext} exists; refusing to overwrite"


def axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.1)
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.tick_params(labelsize=6, width=0.3, length=2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig1b_v4():
    PL._style()
    d = pd.read_csv(OUT / "fig1b_v2.csv")
    assert d["role"].tolist() == ["Throughput-maximizing", "Drop-aware"]
    rows = d.to_dict("records")
    shutil.copyfile(OUT / "fig1b_v2.csv", OUT / f"{NAME}.csv")

    colors = [("skyblue", "//"), ("lightcoral", "..")]
    labels = ["Throughput-maximizing", "Drop-aware"]  # v2: "Drop-aware (max $T-0.3D$)"
    width = 0.36

    def bars(ax, col, xc, fmt="{:.3f}"):
        for i, (r, (c, h)) in enumerate(zip(rows, colors)):
            x = xc + (i - 0.5) * width
            ax.bar(x, r[col], width, color=c, alpha=0.6, hatch=h, edgecolor="black", linewidth=0.3, label=labels[i])
            ax.text(x, r[col], fmt.format(r[col]), ha="center", va="bottom", fontsize=5.5)

    fig_h = W * 0.55
    ax_h_cm = fig_h * 2.54 * 0.75
    min_d_mm = min(r["D_macro"] for r in rows) / 1.1 * ax_h_cm * 10
    split = min_d_mm < 1.0
    if not split:
        fig, ax = plt.subplots(figsize=(W, fig_h))
        bars(ax, "T", 0)
        bars(ax, "D_macro", 1)
        ax.set_xticks([0, 1], ["Normalized throughput $T$", "Mean drop fraction $D$"])
        ax.set_ylim(0, 1.12)
        axes_style(ax)
        ax.legend(loc="upper right", fontsize=5.5, frameon=False, handletextpad=0.3)
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, fig_h))
        bars(a1, "T", 0)
        bars(a2, "D_macro", 0)
        a1.set_ylim(0, 1.15)
        a2.set_ylim(0, max(r["D_macro"] for r in rows) * 1.2)
        a1.set_xticks([0], ["Normalized throughput $T$"])
        a2.set_xticks([0], ["Mean drop fraction $D$"])
        for ax in (a1, a2):
            axes_style(ax)
        h, l = a1.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=5.5, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, NAME)
    return split


if __name__ == "__main__":
    print({"split_panels": fig1b_v4()})
