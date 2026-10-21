"""Fig. 1(a)/(b) v3: half-column panels (1.65 x 1.55 in) for one row in an 8.5 cm column. Same font pt as v2.

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/out_manuscript/code/fig1_v3.py

Data: fig1a_v2.csv and fig1b_v2.csv read as is (no recomputation). Writes fig1a_v3.*, fig1b_v3.*, fig1_v3_preview.png,
fig1_v3_checks.json only; refuses to overwrite.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402

import figures as FG  # noqa: E402  (style helpers only)

OUT = FG.OUT
FIGSIZE = (1.65, 1.55)
FS_AXIS, FS_TICK, FS_VAL = 6, 6, 5.5  # v2 values
FS_LEG = float(__import__("os").environ.get("FIG1_V3_LEG_FS", 4.0))  # legend pt: v2 5.5, reduced on request; 4.0 = largest size clearing fig1a
NEW = [f"{n}.{e}" for n in ("fig1a_v3", "fig1b_v3") for e in ("pdf", "png", "csv")] + ["fig1_v3_preview.png", "fig1_v3_checks.json"]
for f in NEW:
    assert not (OUT / f).exists(), f"{f} exists; refusing to overwrite"


def bb(artist, r):
    return artist.get_window_extent(r)


def overlaps(a, b):
    return a.overlaps(b)


def save(fig, name):
    # "tight" crop, but the box is widened to the axis labels: matplotlib's tight box cut the right end of the x label
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [fig.get_tightbbox(r)] + [Bbox(lab.get_window_extent(r).get_points() / fig.dpi) for a in fig.axes for lab in (a.xaxis.label, a.yaxis.label)]
    box = Bbox.union(boxes).padded(0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=box)
    fig.savefig(OUT / f"{name}.png", bbox_inches=box, dpi=300)


def tick_label_overlaps(ax, r):
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        labs = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
        for i in range(len(labs) - 1):
            if overlaps(bb(labs[i], r), bb(labs[i + 1], r)):
                out.append((labs[i].get_text(), labs[i + 1].get_text()))
    return out


YLABEL_FIT = {}


def fit_ylabel(fig, ax, text):
    """One-line y label; font starts at FS_AXIS and shrinks in 0.25 pt steps until the label is no taller than the axes (requested)."""
    fs = FS_AXIS
    while True:
        ax.set_ylabel(text, fontsize=fs)
        fig.tight_layout()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lab_h, ax_h = ax.yaxis.label.get_window_extent(r).height, ax.get_window_extent(r).height
        if lab_h <= ax_h or fs <= 3:
            YLABEL_FIT.update({"text": text, "fontsize_pt": fs, "label_height_px": round(lab_h, 1), "axes_height_px": round(ax_h, 1)})
            return fs
        fs -= 0.25


# ---------------- fig1a_v3 ----------------
def fig1a_v3():
    d = pd.read_csv(OUT / "fig1a_v2.csv")
    d = d.sort_values(["N", "T_allgpu"]).reset_index(drop=True)
    xs = []
    for n, g in d.groupby("N", sort=True):
        k = len(g)
        xs += list(n + (np.zeros(1) if k == 1 else np.linspace(-0.15, 0.15, k)))
    d["x"] = xs
    FG.PL._style()
    checks = {}
    for loc in ("lower left", "upper right"):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for _, r in d.iterrows():
            ax.plot([r["x"], r["x"]], [r["T_allgpu"], 1.0], color="0.55", linewidth=0.5, zorder=1)
        s1 = ax.scatter(d["x"], d["T_allgpu"], marker="v", s=12, color="tab:orange", edgecolor="black", linewidth=0.3, zorder=3, label="All-GPU")
        s2 = ax.scatter(d["x"], np.ones(len(d)), marker="D", s=7, color="black", linewidth=0.3, zorder=3, label="Throughput-optimal")
        ax.set_xticks([3, 4, 5, 6, 7], ["3", "4", "5", "6", "7"])
        ax.set_xlim(2.5, 7.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 0.90/0.95/1.00 visible (<= 4 ticks); keeps 1.00 labelled with the raised top
        ax.set_ylim(top=1.02)  # headroom above the 1.0 markers (requested)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Concurrent instances $N$", fontsize=FS_AXIS)  # v2 wording is 1.54 in wide at 6 pt: wider than the panel
        fit_ylabel(fig, ax, "Normalized throughput")
        FG.axes_style(ax)
        ax.tick_params(labelsize=FS_TICK)
        leg = ax.legend(loc=loc, fontsize=FS_LEG, frameon=False, ncol=1, handlelength=1.0, labelspacing=0.2, handletextpad=0.3, borderaxespad=0.3)
        fig.tight_layout()  # final layout before measuring
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lb = bb(leg, r)
        pts = np.vstack([s1.get_offsets(), s2.get_offsets()])
        disp = ax.transData.transform(pts)
        pad = 3  # px: marker radius at 100 dpi
        hit = [tuple(np.round(p, 3)) for p, q in zip(pts, disp) if lb.x0 - pad <= q[0] <= lb.x1 + pad and lb.y0 - pad <= q[1] <= lb.y1 + pad]
        lines_hit = []  # connector segments (vertical, x fixed) crossing the legend box
        for _, rr in d.iterrows():
            (qx, qy0), (_, qy1) = ax.transData.transform([(rr["x"], rr["T_allgpu"]), (rr["x"], 1.0)])
            if lb.x0 <= qx <= lb.x1 and qy0 <= lb.y1 and qy1 >= lb.y0:
                lines_hit.append(round(rr["x"], 3))
        checks[loc] = {"legend_hits_points": hit, "legend_hits_connector_lines_at_x": lines_hit, "tick_label_overlaps": tick_label_overlaps(ax, r),
                       "xlabel_vs_ticks": any(overlaps(bb(ax.xaxis.label, r), bb(t, r)) for t in ax.get_xticklabels()),
                       "ylabel_vs_ticks": any(overlaps(bb(ax.yaxis.label, r), bb(t, r)) for t in ax.get_yticklabels()),
                       "yticks": [t.get_text() for t in ax.get_yticklabels()]}
        ok = not hit and not lines_hit and not checks[loc]["tick_label_overlaps"]
        if ok:
            fig.tight_layout()
            save(fig, "fig1a_v3")
            plt.close(fig)
            checks["legend_loc_used"] = loc
            break
        plt.close(fig)
    else:
        # both positions overlap: keep the one with fewer hits (markers count before connector lines), record it
        loc = min(("lower left", "upper right"), key=lambda k: (len(checks[k]["legend_hits_points"]), len(checks[k]["legend_hits_connector_lines_at_x"])))
        checks["legend_loc_used"] = loc + " (overlap remains)"
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for _, r in d.iterrows():
            ax.plot([r["x"], r["x"]], [r["T_allgpu"], 1.0], color="0.55", linewidth=0.5, zorder=1)
        ax.scatter(d["x"], d["T_allgpu"], marker="v", s=12, color="tab:orange", edgecolor="black", linewidth=0.3, zorder=3, label="All-GPU")
        ax.scatter(d["x"], np.ones(len(d)), marker="D", s=7, color="black", linewidth=0.3, zorder=3, label="Throughput-optimal")
        ax.set_xticks([3, 4, 5, 6, 7], ["3", "4", "5", "6", "7"])
        ax.set_xlim(2.5, 7.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(top=1.02)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Concurrent instances $N$", fontsize=FS_AXIS)
        fit_ylabel(fig, ax, "Normalized throughput")
        FG.axes_style(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(loc=loc, fontsize=FS_LEG, frameon=False, ncol=1, handlelength=1.0, labelspacing=0.2, handletextpad=0.3, borderaxespad=0.3)
        fig.tight_layout()
        save(fig, "fig1a_v3")
        plt.close(fig)
    d.to_csv(OUT / "fig1a_v3.csv", index=False)
    return checks


# ---------------- fig1b_v3 ----------------
def fig1b_v3():
    d = pd.read_csv(OUT / "fig1b_v2.csv")
    d.to_csv(OUT / "fig1b_v3.csv", index=False)
    rows = d.to_dict("records")
    FG.PL._style()
    labels = ["Throughput-\nmaximizing", "Drop-aware\n(max $T-0.3D$)"]  # two-line labels (requested) so the legend fits right of the T bars
    colors = [("skyblue", "//"), ("lightcoral", "..")]
    width = 0.36 * 0.7
    gx = [0.0, 1.0 * 0.7]  # group centres: spacing x0.7 of v2 (1.0)
    attempts = []
    for mode, off in (("offset", 0.015), ("offset", 0.03), ("offset", 0.05), ("stagger", 0.015)):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        texts = {}
        for gi, col in enumerate(["T", "D_macro"]):
            for i, (row, (c, h)) in enumerate(zip(rows, colors)):
                x = gx[gi] + (i - 0.5) * width
                ax.bar(x, row[col], width, color=c, alpha=0.6, hatch=h, edgecolor="black", linewidth=0.3, label=labels[i] if gi == 0 else None)
                y = row[col] + off
                if mode == "stagger" and col == "D_macro":
                    y = row[col] + off + (0.09 if i == 1 else 0.0)  # second D value on a second line above the first
                texts[(col, i)] = ax.text(x, y, f"{row[col]:.3f}", ha="center", va="bottom", fontsize=FS_VAL)
        ax.set_xticks(gx, ["Normalized\nthroughput $T$", "Mean drop\nfraction $D$"])
        ax.set_xlim(gx[0] - width - 0.02, gx[1] + width + 0.02)  # tight x range: bars as wide on paper as the panel allows
        ax.set_ylim(0, 1.18)  # headroom above the value labels (requested; was 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        FG.axes_style(ax)
        ax.tick_params(labelsize=FS_TICK)
        leg = ax.legend(loc="center right", fontsize=FS_LEG, frameon=False, ncol=1, handlelength=1.0, labelspacing=0.9,  # gap between the two-line entries (requested) handletextpad=0.3,
                        borderaxespad=0.3)
        fig.tight_layout()  # final layout before measuring
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tb = {k: bb(t, r) for k, t in texts.items()}
        ks = list(tb)
        val_pairs = [(f"{a[0]}{a[1]}", f"{b[0]}{b[1]}") for i, a in enumerate(ks) for b in ks[i + 1:] if overlaps(tb[a], tb[b])]
        lb = bb(leg, r)
        leg_vals = [f"{k[0]}{k[1]}" for k in ks if overlaps(tb[k], lb)]
        leg_bars = [i for i, p in enumerate(ax.patches) if overlaps(bb(p, r), lb)]
        ax_bb = ax.get_window_extent(r)
        out_of_axes = [f"{k[0]}{k[1]}" for k in ks if tb[k].y1 > ax_bb.y1 or tb[k].x0 < ax_bb.x0 or tb[k].x1 > ax_bb.x1]
        a = {"mode": mode, "offset": off, "value_overlaps": val_pairs, "legend_vs_values": leg_vals, "legend_vs_bars": len(leg_bars),
             "values_outside_axes": out_of_axes, "tick_label_overlaps": tick_label_overlaps(ax, r),
             "xticklabel_width_px": [round(bb(t, r).width, 1) for t in ax.get_xticklabels()], "axes_width_px": round(ax_bb.width, 1),
             "legend_width_px": round(lb.width, 1)}
        attempts.append(a)
        # value placement rules only govern value-value overlap; a legend that is wider than the free area cannot be fixed by them
        # and is recorded (legend_vs_values / legend_vs_bars) instead of driving further value moves
        # a value label crossing the top spine (ylim fixed at 1.1) is recorded, not a reason to stagger D
        if not val_pairs and not a["tick_label_overlaps"]:
            fig.tight_layout()
            save(fig, "fig1b_v3")
            plt.close(fig)
            break
        plt.close(fig)
    else:
        save(fig, "fig1b_v3")  # keep the last attempt, problems recorded
        plt.close(fig)
    min_d = d["D_macro"].min()
    return {"attempts": attempts, "used": attempts[-1], "min_D": min_d}


def preview():
    a = plt.imread(OUT / "fig1a_v3.png")
    b = plt.imread(OUT / "fig1b_v3.png")
    h = max(a.shape[0], b.shape[0])

    def padh(im):
        p = np.ones((h - im.shape[0], im.shape[1], im.shape[2]), dtype=im.dtype)
        return np.vstack([im, p])

    gap = np.ones((h, 20, a.shape[2]), dtype=a.dtype)
    img = np.hstack([padh(a), gap, padh(b)])
    w_in = 8.5 / 2.54
    fig = plt.figure(figsize=(w_in, w_in * img.shape[0] / img.shape[1]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(OUT / "fig1_v3_preview.png", dpi=img.shape[1] / w_in)
    plt.close(fig)
    return {"px": [int(img.shape[1]), int(img.shape[0])], "native_width_in_each": [a.shape[1] / 300, b.shape[1] / 300],
            "sum_width_cm": (a.shape[1] + b.shape[1]) / 300 * 2.54}


if __name__ == "__main__":
    ca = fig1a_v3()
    cb = fig1b_v3()
    pv = preview()
    ca["ylabel_fit"] = YLABEL_FIT
    res = {"fig1a_v3": ca, "fig1b_v3": cb, "preview": pv}
    (OUT / "fig1_v3_checks.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
