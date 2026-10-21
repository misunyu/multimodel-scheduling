"""Fig. 1(b) v6: fig1b_v6 with wspace widened to WSPACE_V5 x 1.8 (left/right figure margins from tight_layout kept, so the
overall width is unchanged and each panel gets narrower; bar width in data units unchanged). Otherwise identical to v5.

v5 docstring: Fig. 1(b) v5: fig1b_v2 with y-axis titles ("Normalized throughput $T$" left, "Mean drop fraction $D$" right) instead of the
x labels (x ticks removed). Everything else (data, figure size, font sizes, colors/hatches, value labels, legend text/placement,
no frame) is the fig1_v2.py fig1b_v2 code. Data are read from fig1b_v2.csv as is (no recomputation).
wspace = smallest value (0.01 steps) at which the right panel's y title / tick labels clear the left panel (axes + its texts) by
>= GAP_PT. Writes fig1b_v6.{pdf,png,csv} and prints the layout checks. 4.2 cm preview:
  pdftoppm -png -singlefile -scale-to-x 992 -scale-to-y -1 fig1b_v6.pdf fig1b_v6_4p2cm   (992 px = 4.2 cm at 600 dpi)

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/fig1b_v6.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "revision_exp"))

import itertools  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import plots as PL  # noqa: E402  (style only)

OUT = REPO / "revision_exp" / "out_manuscript"
W = 8.5 / 2.54
NAME = "fig1b_v6"
WSPACE_V5 = 0.39
WSPACE = round(WSPACE_V5 * 1.8, 2)
FS_LABEL = 6  # = x label size in v2 (x tick labels, tick_params labelsize=6)
LABELPAD = 2.0
GAP_PT = 1.0
for ext in ["pdf", "png", "csv"]:
    assert not (OUT / f"{NAME}.{ext}").exists(), f"{NAME}.{ext} exists; refusing to overwrite"


def axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.1)
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.tick_params(labelsize=6, width=0.3, length=2)


def ext(a, r):
    return a.get_window_extent(r)


def overlap(b1, b2):
    return b1.x0 < b2.x1 and b2.x0 < b1.x1 and b1.y0 < b2.y1 and b2.y0 < b1.y1


def fig1b_v6():
    PL._style()
    d = pd.read_csv(OUT / "fig1b_v2.csv")
    assert d["role"].tolist() == ["Throughput-maximizing", "Drop-aware"]
    rows = d.to_dict("records")
    shutil.copyfile(OUT / "fig1b_v2.csv", OUT / f"{NAME}.csv")

    colors = [("skyblue", "//"), ("lightcoral", "..")]
    labels = ["Throughput-maximizing", "Drop-aware (max $T-0.3D$)"]
    width = 0.36
    vals = {}

    def bars(ax, col, xc, fmt="{:.3f}"):
        for i, (r, (c, h)) in enumerate(zip(rows, colors)):
            x = xc + (i - 0.5) * width
            ax.bar(x, r[col], width, color=c, alpha=0.6, hatch=h, edgecolor="black", linewidth=0.3, label=labels[i])
            vals.setdefault(ax, []).append(ax.text(x, r[col], fmt.format(r[col]), ha="center", va="bottom", fontsize=5.5))

    fig_h = W * 0.55
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, fig_h))  # v2 used the split layout (min D bar < 1 mm)
    bars(a1, "T", 0)
    bars(a2, "D_macro", 0)
    a1.set_ylim(0, 1.15)
    a2.set_ylim(0, max(r["D_macro"] for r in rows) * 1.2)
    a1.set_xticks([])
    a2.set_xticks([])
    a1.set_ylabel("Normalized throughput $T$", fontsize=FS_LABEL, labelpad=LABELPAD)
    a2.set_ylabel("Mean drop fraction $D$", fontsize=FS_LABEL, labelpad=LABELPAD)
    for ax in (a1, a2):
        axes_style(ax)
    h, l = a1.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=5.5, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    sp = fig.subplotpars
    left, right, bottom, top = sp.left, sp.right, sp.bottom, sp.top

    def gap_at(ws):
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=ws)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lhs = [ext(a1, r)] + [ext(t, r) for t in vals[a1]]
        lo, hi = a2.get_ylim()
        rhs = [ext(a2.yaxis.label, r)] + [ext(t, r) for t, v in zip(a2.get_yticklabels(), a2.get_yticks()) if t.get_text() and lo <= v <= hi]
        return min(b.x0 for b in rhs) - max(b.x1 for b in lhs)  # px at fig dpi

    panel_w_cm = {}
    for tag, w_ in (("v5", WSPACE_V5), ("v6", WSPACE)):
        gap_at(w_)
        rr = fig.canvas.get_renderer()
        panel_w_cm[tag] = [round(ext(a, rr).width / fig.dpi * 2.54, 3) for a in (a1, a2)]
    ws = WSPACE
    gap_pt = gap_at(ws) * 72 / fig.dpi

    # overlap checks among all labelled elements
    r = fig.canvas.get_renderer()
    el = {"a1.ylabel": a1.yaxis.label, "a2.ylabel": a2.yaxis.label, "legend": leg}
    for k, ax in (("a1", a1), ("a2", a2)):
        lo, hi = ax.get_ylim()
        for i, t in enumerate(t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if t.get_text() and lo <= v <= hi):  # drawn ticks only
            el[f"{k}.tick{i}"] = t
        for i, t in enumerate(vals[ax]):
            el[f"{k}.value{i}"] = t
    bb = {k: ext(v, r) for k, v in el.items()}
    clashes = [(p, q) for p, q in itertools.combinations(bb, 2) if overlap(bb[p], bb[q])]
    lab_tick = {}
    for k, ax in (("a1", a1), ("a2", a2)):  # y title vs its own tick labels: horizontal clearance
        ticks = [bb[x] for x in bb if x.startswith(f"{k}.tick")]
        lab_tick[k] = (min(b.x0 for b in ticks) - bb[f"{k}.ylabel"].x1) * 72 / fig.dpi
    for k, ax in (("a1", a1), ("a2", a2)):  # value labels and legend inside/above axes
        axb = ext(ax, r)
        for t in vals[ax]:
            assert ext(t, r).y1 <= axb.y1 and not overlap(ext(t, r), bb["legend"])
        assert axb.y1 < bb["legend"].y0
    ylab_fit = {k: bb[f"{k}.ylabel"].height <= ext(ax, r).height for k, ax in (("a1", a1), ("a2", a2))}

    fig.savefig(OUT / f"{NAME}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{NAME}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    tb = fig.get_tightbbox(r).padded(0.02)
    plt.close(fig)
    return {"labelpad_pt": LABELPAD, "wspace": ws, "wspace_v5": WSPACE_V5, "panel_width_cm_at_fig_size": panel_w_cm, "gap_panel_to_right_ylabel_pt": gap_pt, "ylabel_to_ticks_pt": lab_tick,
            "pairwise_overlaps": clashes, "ylabel_fits_axes_height": ylab_fit, "tight_width_in": tb.width, "fig_width_in": W}


if __name__ == "__main__":
    print(json.dumps(fig1b_v6(), indent=1, default=float))
