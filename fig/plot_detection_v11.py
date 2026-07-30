#!/usr/bin/env python3
"""Regenerate detection_sensitivity.pdf -- Q1.2, the (T, eps) grid.

Spec: fig/FIGURE_SPECS.md. Values come ONLY from fig/confirmed_values.json, which was
filled from scripts/detection_sensitivity_v34.py re-windowing the committed v33 runs.

Four 3x3 heatmaps, one per metric, as published. Two choices worth stating:

  - The operating point (T=3, eps=1.0) is boxed in every panel. The figure exists to
    justify that choice, so it has to be findable.
  - Cells where a metric is undefined are hatched, not printed as zero. Recovery latency
    is undefined when no candidate recovers under that (T, eps); a zero there would read
    as "recovered instantly", which is the opposite of what happened.

usage: python3 fig/plot_detection_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

T_VALUES = [1, 3, 5]
EPS_VALUES = [0.5, 1.0, 2.0]
OPERATING = (3, 1.0)
# pastel ramps matching the other paper figures (q13 palette endpoints)
CMAPS = {
    "false_triggers": ["#ffffff", "#f4b5b5", "#c96b6b"],
    "detect_lat":     ["#ffffff", "#fad7a8", "#c88a44"],
    "recover_lat":    ["#ffffff", "#b9d0e8", "#5d87b5"],
    "cum_violation":  ["#ffffff", "#d8c2ea", "#8a6aae"],
}
TITLES = {"false_triggers": "false triggers in steady state (ticks)",
          "detect_lat": "detection latency (s)",
          "recover_lat": "recovery latency (s)",
          "cum_violation": r"cumulative violation $\sum\max(0,V-\epsilon)$"}


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def load():
    return {e["id"]: e for e in json.load(open(os.path.join(HERE, "confirmed_values.json")))["values"]}


def sidecar(entries, name, C):
    gen = {"path": SELF, "sha256": self_sha()}
    out = []
    for ident, drawn in entries:
        c = C[ident]
        out.append({"id": ident, "value": round(float(drawn), 4), "quantity": c["quantity"],
                    "reference": c["reference"], "unit": c["unit"],
                    "n": (int(c["n"]) if c.get("n") is not None else None),
                    "censored": bool(c["censored"]),
                    "recovered": (None if c["recovered"] is None else bool(c["recovered"])),
                    "window_s": (float(c["window_s"]) if c.get("window_s") is not None else None),
                    "sd": (float(c["sd"]) if c.get("sd") is not None else None),
                    "generator": gen})
    json.dump({"values": out}, open(os.path.join(HERE, name), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    C = load()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    ids = []
    for ax, (metric, ramp) in zip(axes.ravel(), CMAPS.items()):
        cmap = LinearSegmentedColormap.from_list(metric, ramp)
        grid, missing = [], []
        for ti, T in enumerate(T_VALUES):
            row = []
            for ei, eps in enumerate(EPS_VALUES):
                i = f"q12_{metric}_T{T}_eps{eps}"
                if i in C and C[i]["value"] == C[i]["value"]:
                    row.append(C[i]["value"]); ids.append((i, C[i]["value"]))
                else:
                    row.append(float("nan")); missing.append((ti, ei))
            grid.append(row)
        finite = [v for r in grid for v in r if v == v]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        if hi - lo < 1e-12:
            # a constant panel: matplotlib would map every cell to the MIDDLE of the ramp,
            # so an all-zero false-trigger panel rendered as saturated pink and read as if
            # the values were large. Anchor the scale at zero instead.
            lo, hi = 0.0, max(hi, 1.0)
            ax.annotate("constant across the grid", (0.5, 1.0), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=5, color="0.35")
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        for ti, T in enumerate(T_VALUES):
            for ei, eps in enumerate(EPS_VALUES):
                v = grid[ti][ei]
                if v != v:
                    ax.add_patch(plt.Rectangle((ei - .5, ti - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="0.6", lw=0.4))
                    ax.text(ei, ti, "undef.", ha="center", va="center", fontsize=5, color="0.4")
                else:
                    ax.text(ei, ti, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                            ha="center", va="center", fontsize=6)
                if (T, eps) == OPERATING:
                    ax.add_patch(plt.Rectangle((ei - .5, ti - .5), 1, 1, fill=False,
                                               edgecolor="#b0182b", lw=1.8))
        ax.set_xticks(range(len(EPS_VALUES)))
        ax.set_xticklabels([f"$\\epsilon$={e}" for e in EPS_VALUES], fontsize=6)
        ax.set_yticks(range(len(T_VALUES)))
        ax.set_yticklabels([f"$T$={t}s" for t in T_VALUES], fontsize=6)
        ax.set_title(TITLES[metric], fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5)
    fig.text(0.01, 0.005,
             "Re-windowed offline from the committed v33 BoundGuard runs; the published "
             "input (results/bounded_sweep/) is not in the repository. Red box = the "
             "operating point ($T$=3 s, $\\epsilon$=1.0). Hatched = undefined, not zero.",
             fontsize=4.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"detection_sensitivity.{ext}"), bbox_inches="tight")
    sidecar(ids, "detection_sensitivity.values.json", C)
    plt.close(fig)
    print("regenerated: detection_sensitivity (+ sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
