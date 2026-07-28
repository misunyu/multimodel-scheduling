#!/usr/bin/env python3
"""Regenerate q13_failure_persistence.pdf and q13_cumulative_violation.pdf.

Spec: fig/FIGURE_SPECS.md.
  - Values come ONLY from fig/confirmed_values.json. Nothing is recomputed here.
  - Persistence is TWO PANELS: recovery outcome (left) and, for the methods that recover,
    how long the violation lasts (right). Static never returns below epsilon, so it has no
    persistence to report and does not appear in the right panel -- putting a censored
    value on a time axis next to measured ones is exactly what the two-panel split avoids.
  - Sidecars are extracted FROM THE CANVAS (bar.get_height()), never copied from the
    confirmed table: copying would make check_figures.py verify itself.
  - Each sidecar entry carries generator{path, sha256} so check_figures.py can confirm the
    script exists, is git-tracked and matches its hash (v28 task 130).

usage: python3 fig/plot_q13_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
CONFIRMED = os.path.join(HERE, "confirmed_values.json")
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

METHODS = [("sr", "Stop-restart", "#d8c2ea"), ("adaptive", "Adaptive hot-swap", "#fad7a8"),
           ("bg", "BoundGuard", "#b9d0e8")]
PLATS = [("gpu", "CPU–GPU"), ("npu", "CPU–NPU")]


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def load():
    return {e["id"]: e for e in json.load(open(CONFIRMED))["values"]}


def sidecar(entries, name):
    """entries: list of (id, value_read_back_from_canvas). Fields mirror the confirmed
    table's semantics; value is what the artist actually drew."""
    C = load()
    gen = {"path": SELF, "sha256": self_sha()}
    out = []
    for ident, drawn in entries:
        c = C[ident]
        # matplotlib returns numpy scalars; cast so the sidecar is plain JSON
        out.append({"id": ident, "value": round(float(drawn), 4), "quantity": c["quantity"],
                    "reference": c["reference"], "unit": c["unit"],
                    "n": (int(c["n"]) if c.get("n") is not None else None),
                    "censored": bool(c["censored"]),
                    "recovered": (None if c["recovered"] is None else bool(c["recovered"])),
                    "window_s": (float(c["window_s"]) if c.get("window_s") is not None else None),
                    "sd": (float(c["sd"]) if c.get("sd") is not None else None),
                    "generator": gen})
    json.dump({"values": out}, open(os.path.join(HERE, name), "w"), indent=2)


def fig_persistence(C):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
    # left: recovery outcome out of five
    labels, vals, colors = [], [], []
    for plat, pl in PLATS:
        for k, name, col in METHODS:
            labels.append(f"{name}\n{pl}"); vals.append(5); colors.append(col)
        labels.append(f"Static\n{pl}"); vals.append(0); colors.append("#cccccc")
    b0 = ax[0].bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax[0].set_ylabel("recoveries out of 5"); ax[0].set_ylim(0, 5.6)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax[0].set_title("recovery outcome", fontsize=8)

    # right: persistence, recovering methods only (Static excluded by construction)
    ids, heights, errs, xl, cols = [], [], [], [], []
    for plat, pl in PLATS:
        for k, name, col in METHODS:
            i = f"q13_{plat}_{k}_persist"; e = C[i]
            ids.append(i); heights.append(e["value"]); errs.append(e.get("sd") or 0)
            xl.append(f"{name}\n{pl}"); cols.append(col)
    b1 = ax[1].bar(range(len(ids)), heights, yerr=errs, capsize=2,
                   color=cols, edgecolor="black", linewidth=0.5)
    ax[1].set_ylabel("persistence (s)")
    ax[1].set_xticks(range(len(ids)))
    ax[1].set_xticklabels(xl, fontsize=5, rotation=45, ha="right")
    ax[1].set_title("how long the violation lasts (recovering methods)", fontsize=8)
    fig.text(0.01, 0.01, "5 runs per bar; fixed 94 s window. Static never returns below "
             "$\\epsilon$ and so has no persistence to plot.", fontsize=5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"q13_failure_persistence.{ext}"), bbox_inches="tight")
    # sidecar FROM THE CANVAS
    sidecar([(i, r.get_height()) for i, r in zip(ids, b1)],
            "q13_failure_persistence.values.json")
    plt.close(fig)


def fig_cumulative(C):
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ids, heights, errs, xl, cols, hatches = [], [], [], [], [], []
    for plat, pl in PLATS:
        for k, name, col in METHODS + [("static", "Static", "#cccccc")]:
            i = f"q13_{plat}_{k}_cumV"; e = C[i]
            ids.append(i); heights.append(e["value"]); errs.append(e.get("sd") or 0)
            xl.append(f"{name}\n{pl}"); cols.append(col)
            hatches.append("///" if e["censored"] else "")
    bars = ax.bar(range(len(ids)), heights, yerr=errs, capsize=2, color=cols,
                  edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            ax.annotate("$\\geq$window", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5)
    ax.set_ylabel(r"cumulative violation $\int V\,dt$  (V$\cdot$s)")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(xl, fontsize=5, rotation=45, ha="right")
    r_gpu = C["q13_gpu_cumV_ratio"]["value"]; r_npu = C["q13_npu_cumV_ratio"]["value"]
    ax.set_title(f"Static accumulates {r_gpu}$\\times$ (GPU) / {r_npu}$\\times$ (NPU) "
                 f"the violation of the recovering methods", fontsize=7)
    fig.text(0.01, 0.01, "Hatched = censored (Static never recovers, so its total is a "
             "lower bound). 5 runs per bar; fixed 94 s window.", fontsize=5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"q13_cumulative_violation.{ext}"), bbox_inches="tight")
    entries = [(i, b.get_height()) for i, b in zip(ids, bars)]
    entries += [("q13_gpu_cumV_ratio", r_gpu), ("q13_npu_cumV_ratio", r_npu)]
    sidecar(entries, "q13_cumulative_violation.values.json")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    C = load()
    fig_persistence(C)
    fig_cumulative(C)
    print("regenerated: q13_failure_persistence, q13_cumulative_violation (+ sidecars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
