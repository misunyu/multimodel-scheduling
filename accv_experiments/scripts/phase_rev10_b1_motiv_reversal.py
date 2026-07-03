"""Generate paper/figures/motiv_reversal.pdf — §5.2 reversal motivation.

Reads rev9_partA_npu.csv (NPU) + p1r6_ladder_yolo11s.csv (GPU at L1_light).
Both already exist; no measurement.

Output asserts (printed at the end):
  cell-match against paper/tables/partA.tex — every reported gap row.

Left panel  : 8 cameras sorted by NPU loss (dev-gap = NPU − GPU sAP).
              Colors: small-rich=blue, medium-mixed=gray, large-rich=goldenrod.
              Marker shapes also encode role (so it reads in B/W).
Right panel : x = offline mAP gap, y = streaming sAP gap, with y=x reference.
              Annotations show which group sits below y=x (NPU loses more on
              streaming than on offline).
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
OUT = ROOT / "paper" / "figures" / "motiv_reversal.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Per-sid role assignment used by partA generator (must stay aligned).
ROLE = {
    2:  "small-rich",
    13: "small-rich",
    22: "small-rich",
    8:  "medium-mixed",
    10: "medium-mixed",
    17: "medium-mixed",
    3:  "large-rich",
    21: "large-rich",
}
COLOR = {"small-rich": "#3b6cf0", "medium-mixed": "#888888", "large-rich": "#d4a013"}
MARKER = {"small-rich": "o", "medium-mixed": "s", "large-rich": "^"}


def load_anchors():
    npu = pd.read_csv(RES / "rev9_partA_npu.csv")
    npu = npu[(npu.device == "NPU") & (npu.bg_level == "L1_light")].set_index("sid")
    gpu = pd.read_csv(RES / "p1r6_ladder_yolo11s.csv")
    gpu = gpu[(gpu.device == "GPU") & (gpu.bg_level == "L1_light")].set_index("sid")
    sids = sorted(ROLE)
    rows = []
    for sid in sids:
        if sid not in npu.index or sid not in gpu.index:
            continue
        rows.append({
            "sid": sid,
            "role": ROLE[sid],
            "gpu_sap": float(gpu.loc[sid, "sap_5095"]),
            "npu_sap": float(npu.loc[sid, "sap_5095"]),
            "gpu_map": float(gpu.loc[sid, "map_5095"]),
            "npu_map": float(npu.loc[sid, "map_5095"]),
        })
    df = pd.DataFrame(rows)
    df["sap_gap"] = df["npu_sap"] - df["gpu_sap"]
    df["map_gap"] = df["npu_map"] - df["gpu_map"]
    return df


def render(df):
    df = df.sort_values("sap_gap", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), gridspec_kw={"wspace": 0.28})

    # ----- Left: dev-gap bars per camera, ordered by NPU loss -----
    ax = axes[0]
    x = np.arange(len(df))
    colors = [COLOR[r] for r in df["role"]]
    bars = ax.bar(x, df["sap_gap"], color=colors, edgecolor="black", linewidth=0.6)
    # marker overlay (shape encodes role) for B/W readability
    for xi, (_, row) in enumerate(df.iterrows()):
        ax.scatter(xi, row["sap_gap"], marker=MARKER[row["role"]],
                    s=42, color="white", edgecolors="black", linewidths=0.7, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"sid {s}" for s in df["sid"]], rotation=0, fontsize=8)
    ax.set_ylabel(r"$\Delta$ sAP  (NPU $-$ GPU)", fontsize=9)
    ax.set_title("(a) Per-camera quantization gap at L1$_{\\mathrm{light}}$, ordered by NPU loss",
                  fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["small-rich"],
                markeredgecolor="black", markersize=8, label="small-rich"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["medium-mixed"],
                markeredgecolor="black", markersize=8, label="medium-mixed"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR["large-rich"],
                markeredgecolor="black", markersize=8, label="large-rich"),
    ]
    ax.legend(handles=legend_items, fontsize=7.5, loc="lower left", framealpha=0.9)

    # ----- Right: offline mAP gap vs streaming sAP gap -----
    ax = axes[1]
    xs = df["map_gap"].to_numpy()
    ys = df["sap_gap"].to_numpy()
    for _, row in df.iterrows():
        ax.scatter(row["map_gap"], row["sap_gap"],
                    marker=MARKER[row["role"]], s=58,
                    color=COLOR[row["role"]], edgecolors="black", linewidths=0.5,
                    zorder=3)
        ax.annotate(f"  {row['sid']}", (row["map_gap"], row["sap_gap"]),
                     fontsize=7, color="black")
    lo = min(xs.min(), ys.min()) - 0.015
    hi = max(xs.max(), ys.max()) + 0.015
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.7, label="y = x", zorder=1)
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color="#fbb", alpha=0.18, zorder=0,
                     label="streaming gap > offline gap")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\Delta$ offline mAP  (NPU $-$ GPU)", fontsize=9)
    ax.set_ylabel(r"$\Delta$ streaming sAP  (NPU $-$ GPU)", fontsize=9)
    ax.set_title("(b) Offline mAP gap vs streaming sAP gap", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")
    return df


def verify_against_partA_tex(df):
    """Read paper/tables/partA.tex and confirm every row's gap matches."""
    txt = (ROOT / "paper" / "tables" / "partA.tex").read_text()
    body = [ln.strip() for ln in txt.splitlines()
              if ln.strip().endswith(r"\\") and ln.strip()[0].isdigit()]
    # The table is sorted ascending by gap (rank 1 = closest to zero).
    # Our `df` is sorted descending by gap (largest first), so reverse for comparison.
    df_asc = df.sort_values("sap_gap", ascending=False).reset_index(drop=True)
    # but the .tex displays smallest |gap| first (gap −0.008 then −0.010 ...).
    # So actually sort by gap descending (least negative first) for comparison.
    print("\n=== B-1 cell-match vs paper/tables/partA.tex ===")
    print(f"{'rank':<5} {'role':<13} {'tex GPU':<8} {'tex NPU':<8} {'tex gap':<9}"
            f"  {'csv GPU':<8} {'csv NPU':<8} {'csv gap':<9}  match?")
    parsed = []
    for ln in body:
        cols = [c.strip() for c in ln.rstrip(r"\\").split("&")]
        parsed.append({
            "rank": int(cols[0]),
            "role": cols[1],
            "gpu": float(cols[2]),
            "npu": float(cols[3]),
            "gap": float(cols[4].replace("$", "").replace("-", "−").replace("−", "-")),
        })
    parsed_sorted = sorted(parsed, key=lambda r: r["rank"])
    df_sorted = df.sort_values("sap_gap", ascending=False).reset_index(drop=True)
    all_match = True
    for tex_row, (_, csv_row) in zip(parsed_sorted, df_sorted.iterrows()):
        ok = (abs(tex_row["gpu"] - csv_row["gpu_sap"]) < 0.005 and
               abs(tex_row["npu"] - csv_row["npu_sap"]) < 0.005 and
               abs(tex_row["gap"] - csv_row["sap_gap"]) < 0.005)
        all_match = all_match and ok
        print(f"{tex_row['rank']:<5} {tex_row['role']:<13} {tex_row['gpu']:<8.3f} "
                f"{tex_row['npu']:<8.3f} {tex_row['gap']:<+9.3f}  "
                f"{csv_row['gpu_sap']:<8.4f} {csv_row['npu_sap']:<8.4f} "
                f"{csv_row['sap_gap']:<+9.4f}  {'✓' if ok else '✗'}")
    print(f"\nB-1 OVERALL CELL-MATCH: {'PASS' if all_match else 'FAIL'}")
    return all_match


def main():
    df = load_anchors()
    df = render(df)
    verify_against_partA_tex(df)


if __name__ == "__main__":
    main()
