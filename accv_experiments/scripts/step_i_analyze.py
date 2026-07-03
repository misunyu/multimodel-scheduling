"""Step I analysis — correct grid summary + figures.

Fixes the bug in step_i_natural_composition.build_figures() where Oracle was
extracted by name-prefix instead of by max worst_sap across the cell. This
script reuses the measurements written by step_i (and reuses step_h2 cells
for the 6 cells not re-measured) and produces correct figures.

Inputs:
  results/step_i_natural_composition.csv     (Step I new cells)
  results/step_h2_robustness.csv             (H2 Part A: 3 comp × N=8 × L1_light)
  results/step_h2_bg_ablation.csv            (H2 Part B: A × N=4,8 × 5 bg variants)

Outputs:
  results/step_i_grid_summary.csv            (correct 12-cell grid)
  results/figures/step_i_natural_comparison.pdf  (re-rendered)
  results/figures/step_i_schedule_natural.pdf    (re-rendered)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"

DF_I  = pd.read_csv(RES / "step_i_natural_composition.csv")
DF_HA = pd.read_csv(RES / "step_h2_robustness.csv")
DF_HB = pd.read_csv(RES / "step_h2_bg_ablation.csv")


def _canon(name):
    """Map a placement_name to a canonical strategy bucket."""
    if name.startswith("Naive_allGPU"):     return "Naive"
    if name.startswith("AllNPU"):            return "AllNPU"
    if name.startswith("SizeBlindRev"):      return "SizeBlindRev"
    if name.startswith("SizeAware"):         return "SizeAware"
    return None


def _cell_rows(comp, N, bg):
    """Return all measurement rows for a given (comp, N, bg) cell, merged
    from Step I and Step H2 (H2 only contributes for cells it covers)."""
    rows = []
    # Step I
    sub = DF_I[(DF_I.composition == comp) & (DF_I.n_streams == N) &
               (DF_I.bg_level == bg)]
    rows.extend(sub.to_dict("records"))
    # Step H2 Part B: only Composition A
    if comp == "A_baseline":
        sub = DF_HB[(DF_HB.n_streams == N) & (DF_HB.bg_level == bg)]
        rows.extend(sub.to_dict("records"))
    # Step H2 Part A: only N=8 × bg L1_light
    if N == 8 and bg == "L1_light":
        sub = DF_HA[DF_HA.composition == comp]
        rows.extend(sub.to_dict("records"))
    return rows


def cell_summary(comp, N, bg):
    """For a cell, find best worst_sap per canonical strategy + Oracle (=
    overall max). Returns dict {canon: (worst_sap, spec)}."""
    rows = _cell_rows(comp, N, bg)
    out = {}
    best_overall = (-1.0, None)
    for r in rows:
        w = float(r["worst_sap"])
        spec = r["placement_spec"]
        c = _canon(r["placement_name"])
        if c is not None:
            # Keep highest worst_sap per canonical
            if c not in out or out[c][0] < w:
                out[c] = (w, spec)
        if w > best_overall[0]:
            best_overall = (w, spec)
    if best_overall[1] is not None:
        out["Oracle"] = best_overall
    return out


COMPS = ["A_baseline", "B_medium_mixed", "C_diverse"]
NS = [4, 8]
BGS = ["L1_light", "L1_heavy"]
STRATS = ["Naive", "SizeAware", "AllNPU", "SizeBlindRev", "Oracle"]
COLORS = {"Naive": "#888888", "SizeAware": "#3680c4",
          "AllNPU": "#3b9c4d", "SizeBlindRev": "#c43b3b",
          "Oracle": "#c4a236"}
COMP_LABEL = {"A_baseline": "A: 4 small + 4 large (extreme)",
              "B_medium_mixed": "B: 4 medium + 2 small + 2 large",
              "C_diverse": "C: 3 small + 3 medium + 2 large"}


def build_grid():
    """Assemble the 12-cell grid table."""
    grid_rows = []
    for comp in COMPS:
        for N in NS:
            for bg in BGS:
                summary = cell_summary(comp, N, bg)
                row = {"composition": comp, "N": N, "bg": bg}
                for s in STRATS:
                    if s in summary:
                        row[f"{s}_sap"] = summary[s][0]
                        row[f"{s}_spec"] = summary[s][1]
                # offload fraction at the Oracle placement
                if "Oracle" in summary:
                    try:
                        spec = json.loads(summary["Oracle"][1])
                    except Exception:
                        spec = summary["Oracle"][1].split(",")
                    npu_count = sum(1 for d in spec if d == "NPU")
                    row["oracle_k_npu"] = npu_count
                    row["oracle_offload_frac"] = npu_count / N
                grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RES / "step_i_grid_summary.csv", index=False)
    return grid


def fig_grid(grid):
    """12-cell grid: rows = comp × bg; cols = N. Each cell shows 5 bars."""
    fig, axes = plt.subplots(len(COMPS) * len(BGS), len(NS),
                              figsize=(5.4 * len(NS), 2.4 * len(COMPS) * len(BGS)),
                              squeeze=False)
    for r_i, (comp, bg) in enumerate([(c, b) for c in COMPS for b in BGS]):
        for c_i, N in enumerate(NS):
            ax = axes[r_i][c_i]
            sub = grid[(grid.composition == comp) & (grid.N == N) &
                       (grid.bg == bg)]
            if not len(sub):
                ax.set_visible(False); continue
            r = sub.iloc[0]
            vals, labels, cols = [], [], []
            for s in STRATS:
                col = f"{s}_sap"
                if col in r and not pd.isna(r[col]):
                    vals.append(float(r[col]))
                    labels.append(s)
                    cols.append(COLORS[s])
            bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
            # outline the best strategy
            best_i = int(np.argmax(vals))
            bars[best_i].set_edgecolor("#000")
            bars[best_i].set_linewidth(2.5)
            ax.set_title(f"{comp.split('_')[0]} | N={N} | bg={bg}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=20)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)
            if c_i == 0:
                ax.set_ylabel("worst sAP", fontsize=8)
    plt.suptitle("Step I — Natural composition grid (worst-stream sAP per strategy)\n"
                 "A=extreme small/large; B=medium-mixed natural; C=size-diverse natural",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "step_i_natural_comparison.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_schedule(grid):
    """Optimal offload fraction vs bg, one panel per N."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    comp_color = {"A_baseline": "#c43b3b", "B_medium_mixed": "#3680c4",
                  "C_diverse": "#3b9c4d"}
    comp_marker = {"A_baseline": "o", "B_medium_mixed": "s", "C_diverse": "^"}
    for c_i, N in enumerate(NS):
        ax = axes[c_i]
        for comp in COMPS:
            xs, fracs, saps, ks = [], [], [], []
            for bg in BGS:
                r = grid[(grid.composition == comp) & (grid.N == N) &
                         (grid.bg == bg)]
                if len(r):
                    r = r.iloc[0]
                    xs.append(bg)
                    fracs.append(float(r["oracle_offload_frac"]))
                    saps.append(float(r["Oracle_sap"]))
                    ks.append(int(r["oracle_k_npu"]))
            ax.plot(xs, fracs, "-", marker=comp_marker[comp],
                    color=comp_color[comp], linewidth=2.4, markersize=11,
                    label=COMP_LABEL[comp])
            for j, (frac, sap, k) in enumerate(zip(fracs, saps, ks)):
                ax.annotate(f"k={k}/{N}\nsAP={sap:.3f}",
                            xy=(j, frac), xytext=(0, 11),
                            textcoords="offset points", fontsize=7,
                            ha="center", color="#333",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="#fffae6", ec="#bb9900", lw=0.4))
        ax.set_xlabel("background contention")
        if c_i == 0:
            ax.set_ylabel("optimal NPU offload fraction  k/N")
        ax.set_title(f"N = {N}")
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    plt.suptitle("Step I — Optimal offload schedule under natural compositions",
                 fontsize=12)
    plt.tight_layout()
    out = FIG / "step_i_schedule_natural.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_summary(grid):
    """Tabular text report."""
    print("\n" + "=" * 110)
    print(" 12-cell grid — worst-stream sAP per (composition, N, bg) × strategy")
    print("=" * 110)
    cols = [f"{s}_sap" for s in STRATS]
    pretty = grid[["composition", "N", "bg"] + cols].copy()
    for c in cols:
        pretty[c] = pretty[c].apply(lambda v: f"{v:.3f}" if not pd.isna(v) else "—")
    pretty.columns = ["composition", "N", "bg"] + STRATS
    print(pretty.to_string(index=False))

    print("\n" + "=" * 110)
    print(" Best strategy per cell")
    print("=" * 110)
    print(f"{'composition':<18s}  {'N':>2}  {'bg':<9s}  {'best strategy':<14s}  "
          f"{'worst sAP':>9s}  {'Oracle placement':<40s}  {'k/N':>5s}")
    for _, r in grid.iterrows():
        sap_vals = {s: r.get(f"{s}_sap") for s in STRATS if not pd.isna(r.get(f"{s}_sap"))}
        named_excl_oracle = {k: v for k, v in sap_vals.items() if k != "Oracle"}
        best_named = max(named_excl_oracle, key=named_excl_oracle.get) if named_excl_oracle else "-"
        spec = r.get("Oracle_spec", "—")
        try:
            spec_list = json.loads(spec) if isinstance(spec, str) else spec
            spec_str = "".join("N" if d == "NPU" else "G" for d in spec_list)
        except Exception:
            spec_str = str(spec)
        print(f"{r['composition']:<18s}  {int(r['N']):>2d}  {r['bg']:<9s}  "
              f"{best_named:<14s}  {float(r[f'{best_named}_sap']):>9.4f}  "
              f"{spec_str:<40s}  {int(r['oracle_k_npu'])}/{int(r['N'])}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    print_summary(grid)
    fig_grid(grid)
    fig_schedule(grid)
    print(f"\nsaved grid CSV: {RES / 'step_i_grid_summary.csv'}")


if __name__ == "__main__":
    main()
