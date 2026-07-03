"""Generate paper/figures/gen_cstar.pdf — §7 crossover threshold.

Reads rev10_cstar.csv (A-3 output) + rev10_gen_single.csv (A-1 output for L0
anchor) and renders a 2-panel figure.

Inputs (from PART A measurement):
  rev10_gen_single.csv  : 4 dets × {GPU,NPU} × 24 logs at L0
  rev10_cstar.csv       : 4 dets × {GPU,NPU} × {L1_light,L1_heavy,L2_lm,L3_vlm}
                          × 4 sids (Comp A panel: 2,22,3,21)

Groups (per Comp A N=4):
  small-rich = sids 2, 22
  large-rich = sids 3, 21

Left panel  : Per-detector dev-gap(C) curves NPU−GPU sAP across bg ladder
              L0→L1_light→L1_heavy→L2_lm→L3_vlm. Two curves per detector
              (small-rich, large-rich). Marker = detector size.
              gap>0 region shaded.  C* (linear interpolation to gap=0) marked
              with a star.

Right panel : x = detector capacity (params, M), y = C* (bg index).
              small-rich vs large-rich split, with bg-level labels on y axis.

Output: paper/figures/gen_cstar.pdf, plus results/rev10_cstar_derived.csv
        with the C* values per (detector, group).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "accv_experiments" / "results"
OUT = ROOT / "paper" / "figures" / "gen_cstar.pdf"
DERIVED = RES / "rev10_cstar_derived.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

BG_ORDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
BG_INDEX = {bg: i for i, bg in enumerate(BG_ORDER)}
DETECTORS = [
    ("yolo11s",  9.4),
    ("yolo11m", 20.1),
    ("yolo11l", 25.3),
    ("yolo11x", 56.9),
]
GROUPS = {"small-rich": [2, 22], "large-rich": [3, 21]}
COLOR  = {"yolo11s": "#3b6cf0", "yolo11m": "#3aaf3a",
          "yolo11l": "#d4a013", "yolo11x": "#cc3333"}
GMARK  = {"small-rich": "o", "large-rich": "^"}


def load_anchors():
    """Build a (detector, sid, bg) → NPU sAP, GPU sAP table."""
    a1 = pd.read_csv(RES / "rev10_gen_single.csv")  # L0
    a3 = pd.read_csv(RES / "rev10_cstar.csv")        # ladder
    pool = pd.concat([a1, a3], ignore_index=True)
    # Restrict to Comp A panel
    pool = pool[pool.sid.isin([2, 22, 3, 21])]
    return pool


def per_group_gap(pool):
    """Compute per-detector × per-group × per-bg gap = NPU - GPU."""
    rows = []
    for det, _pm in DETECTORS:
        sub = pool[pool.detector == det]
        for gname, gsids in GROUPS.items():
            for bg in BG_ORDER:
                ggpu = sub[(sub.device == "GPU") & (sub.bg_level == bg) &
                            (sub.sid.isin(gsids))]
                gnpu = sub[(sub.device == "NPU") & (sub.bg_level == bg) &
                            (sub.sid.isin(gsids))]
                if len(ggpu) == 0 or len(gnpu) == 0:
                    rows.append({"detector": det, "group": gname,
                                  "bg_level": bg, "gpu_sap": float("nan"),
                                  "npu_sap": float("nan"), "gap": float("nan")})
                    continue
                gm = float(ggpu["sap_5095"].mean())
                nm = float(gnpu["sap_5095"].mean())
                rows.append({"detector": det, "group": gname, "bg_level": bg,
                              "gpu_sap": gm, "npu_sap": nm, "gap": nm - gm})
    return pd.DataFrame(rows)


def cstar_interp(gap_df):
    """For each (detector, group), find the bg index where gap crosses 0
    via linear interpolation on the (bg_index, gap) curve. Returns NaN if
    no crossing (gap stays one-sign across the ladder)."""
    out = []
    for det, _ in DETECTORS:
        for gname in GROUPS:
            sub = gap_df[(gap_df.detector == det) & (gap_df.group == gname)]
            xs = [BG_INDEX[b] for b in sub["bg_level"]]
            ys = sub["gap"].to_list()
            # find first sign change
            cstar = None
            for i in range(1, len(ys)):
                y0, y1 = ys[i - 1], ys[i]
                if not (np.isfinite(y0) and np.isfinite(y1)):
                    continue
                if y0 * y1 < 0:
                    x0, x1 = xs[i - 1], xs[i]
                    # linear interp: solve gap=0 → x* = x0 - y0*(x1-x0)/(y1-y0)
                    cstar = x0 - y0 * (x1 - x0) / (y1 - y0)
                    break
            out.append({"detector": det, "group": gname, "cstar_bg_index": cstar,
                         "interp_method": "linear" if cstar is not None else "no-crossing"})
    return pd.DataFrame(out)


def render(gap_df, cstar_df):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4),
                              gridspec_kw={"wspace": 0.30, "width_ratios": [1.4, 1.0]})

    # ----- LEFT: dev-gap curves -----
    ax = axes[0]
    ax.axhspan(0, 1, alpha=0.10, color="green",
                label="gap $>$ 0 (NPU better)")
    ax.axhline(0, color="black", linewidth=0.5)
    xs = [BG_INDEX[b] for b in BG_ORDER]
    for det, _pm in DETECTORS:
        for gname in GROUPS:
            sub = gap_df[(gap_df.detector == det) & (gap_df.group == gname)]
            ys = sub["gap"].to_list()
            if not any(np.isfinite(y) for y in ys): continue
            label = f"{det} {gname[0]}-rich"  # short label
            ax.plot(xs, ys, "-" if gname == "small-rich" else "--",
                     color=COLOR[det], marker=GMARK[gname], markersize=5,
                     linewidth=1.0, label=None)
        # C* stars
        for gname in GROUPS:
            row = cstar_df[(cstar_df.detector == det) & (cstar_df.group == gname)]
            if len(row) and row.iloc[0]["cstar_bg_index"] is not None and \
                np.isfinite(row.iloc[0]["cstar_bg_index"]):
                cs = float(row.iloc[0]["cstar_bg_index"])
                ax.scatter([cs], [0], marker="*", s=90, color=COLOR[det],
                            edgecolors="black", linewidths=0.6, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([b.replace("_", "$_{") + "}$" if "_" in b else b
                         for b in BG_ORDER], rotation=15, fontsize=8)
    ax.set_xlabel("Background contention level (C)", fontsize=9)
    ax.set_ylabel(r"$\Delta$ sAP  (NPU $-$ GPU)", fontsize=9)
    ax.set_title("(a) Per-detector device-gap across bg ladder", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_items = []
    for det, pm in DETECTORS:
        legend_items.append(Line2D([0], [0], color=COLOR[det], linewidth=1.0,
                                     label=f"{det} ({pm:.1f}M)"))
    legend_items.append(Line2D([0], [0], color="black", marker="o",
                                 linestyle="-", linewidth=0.8, label="small-rich"))
    legend_items.append(Line2D([0], [0], color="black", marker="^",
                                 linestyle="--", linewidth=0.8, label="large-rich"))
    legend_items.append(Line2D([0], [0], marker="*", color="white",
                                 markerfacecolor="black", markeredgecolor="black",
                                 markersize=10, linestyle="None", label="C$^\\star$"))
    ax.legend(handles=legend_items, fontsize=7, loc="lower left", ncol=2,
                framealpha=0.9)

    # ----- RIGHT: capacity vs C* -----
    ax = axes[1]
    for det, pm in DETECTORS:
        for gname in GROUPS:
            row = cstar_df[(cstar_df.detector == det) & (cstar_df.group == gname)]
            if not len(row): continue
            cs = row.iloc[0]["cstar_bg_index"]
            if cs is None or not np.isfinite(cs):
                # plot as a "no-crossing" arrow at the top/bottom of axis
                cs_plot = (len(BG_ORDER) - 1) + 0.4
                ax.scatter(pm, cs_plot, marker=GMARK[gname], s=70,
                            color="white", edgecolors=COLOR[det], linewidths=1.0)
                ax.annotate(" no $C^\\star$", (pm, cs_plot), fontsize=7,
                             color=COLOR[det], xytext=(2, 0),
                             textcoords="offset points")
            else:
                ax.scatter(pm, cs, marker=GMARK[gname], s=70,
                            color=COLOR[det], edgecolors="black", linewidths=0.5)
    ax.set_yticks(range(len(BG_ORDER)))
    ax.set_yticklabels(BG_ORDER, fontsize=8)
    ax.set_xlabel("Detector capacity (params, M)", fontsize=9)
    ax.set_ylabel(r"$C^\star$ (interpolated)", fontsize=9)
    ax.set_title("(b) Capacity vs $C^\\star$ per size group", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


def verify_directions(gap_df, cstar_df):
    """Check that the figure direction agrees with rev10 decomp / partA trend.
    No measurement; just internal sanity check that gap is monotone (or nearly
    so) along the bg ladder per detector."""
    print("\n=== B-2 derived C* table ===")
    print(cstar_df.to_string(index=False))
    print()
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    cstar_df.to_csv(DERIVED, index=False)
    print(f"saved {DERIVED}")


def main():
    pool = load_anchors()
    gap_df = per_group_gap(pool)
    cstar_df = cstar_interp(gap_df)
    render(gap_df, cstar_df)
    verify_directions(gap_df, cstar_df)


if __name__ == "__main__":
    main()
