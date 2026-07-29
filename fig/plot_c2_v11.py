#!/usr/bin/env python3
"""Regenerate c2_reactive_comparison.pdf -- the 2x2 ablation (Q6 + the 4b control).

Spec: fig/FIGURE_SPECS.md. Values come ONLY from fig/confirmed_values.json.

Layout follows the factorial structure rather than a flat list, so neighbouring cells
isolate one mechanism: A<->BoundGuard is the effect of dwell, C<->BoundGuard the effect of
progress. A flat four-bar chart hides that, which is why the ablation is drawn as a grid.

Visual encoding:
  - recovery count is the primary quantity (left panel, per regime)
  - terminal V is the secondary (right panel), HATCHED where the run never recovered, since
    a censored terminal value must not read as a measured settling point
  - B is annotated as the adaptive hot-swap baseline: with neither dwell nor progress the
    predictor returns the same top-ranked placement, so the two are the same system

usage: python3 fig/plot_c2_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

# (key, label, colour) -- names/colours per FIGURE_SPECS variant_style
VARIANTS = [("bg", "BoundGuard\ndwell+prog.", "#b9d0e8"),
            ("a", "A\nprog. only", "#fad7a8"),
            ("c", "C\ndwell only", "#cfe3cf"),
            ("b", "B\nneither", "#e8c9c9"),
            ("adaptive", "Adaptive\nhot-swap", "#dddddd")]


def zero_labels(axis, bars):
    """A zero-height bar is indistinguishable from a missing one, and next to a tall
    neighbour it reads as if the neighbour owns the slot. Label the zeros explicitly."""
    for b in bars:
        if b.get_height() == 0:
            axis.annotate("0", (b.get_x() + b.get_width() / 2, 0), xytext=(0, 2),
                          textcoords="offset points", ha="center", fontsize=5, color="0.35")


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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axes.ravel()
    ids = []

    # left: recoveries out of five, misprediction regime (Q6) vs correct prediction (4b)
    x = range(len(VARIANTS))
    q6 = [C[f"q6_{k}_recoveries"]["value"] for k, _, _ in VARIANTS]
    q4b = [C[f"q4b_{k}_recoveries"]["value"] for k, _, _ in VARIANTS]
    w = 0.38
    b1 = ax[0].bar([i - w / 2 for i in x], q6, w, label="misprediction",
                   color=[c for _, _, c in VARIANTS], edgecolor="black", linewidth=0.5)
    # the control series is drawn hollow: same-colour fills made the two series read as one
    # wide bar, so a 0 under misprediction looked like a 5
    b2 = ax[0].bar([i + w / 2 for i in x], q4b, w, label="correct prediction (control)",
                   color="white", edgecolor=[c for _, _, c in VARIANTS], linewidth=1.2,
                   hatch="..")
    zero_labels(ax[0], b1)
    ax[0].set_ylabel("recoveries out of 5"); ax[0].set_ylim(0, 5.8)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([l for _, l, _ in VARIANTS], fontsize=5)
    ax[0].legend(fontsize=5, loc="upper right")
    ax[0].set_title("only the combination recovers under misprediction", fontsize=8)
    ids += [(f"q6_{k}_recoveries", r.get_height()) for (k, _, _), r in zip(VARIANTS, b1)]
    ids += [(f"q4b_{k}_recoveries", r.get_height()) for (k, _, _), r in zip(VARIANTS, b2)]

    # right: terminal V under misprediction; hatched where the run never recovered
    lv = [C[f"q6_{k}_lastV"] for k, _, _ in VARIANTS]
    bars = ax[1].bar(x, [e["value"] for e in lv], yerr=[e.get("sd") or 0 for e in lv],
                     capsize=2, color=[c for _, _, c in VARIANTS],
                     edgecolor="black", linewidth=0.5)
    for b, e in zip(bars, lv):
        if e["censored"]:
            b.set_hatch("///")
            ax[1].annotate(r"$\geq$window", (b.get_x() + b.get_width() / 2, b.get_height()),
                           textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5)
    ax[1].set_ylabel(r"terminal $V$"); ax[1].set_yscale("symlog", linthresh=1)
    ax[1].axhline(1.0, color="0.4", ls="--", lw=0.7)
    ax[1].annotate(r"$\epsilon=1$", (len(VARIANTS) - 0.4, 1.0), fontsize=5, color="0.4")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels([l for _, l, _ in VARIANTS], fontsize=5)
    ax[1].set_title("terminal violation (misprediction)", fontsize=8)
    for b, e in zip(bars, lv):
        if e["value"] == 0:
            ax[1].annotate("0 (recovered)", (b.get_x() + b.get_width() / 2, 0), xytext=(0, 2),
                           textcoords="offset points", ha="center", fontsize=5, color="0.35")
    ids += [(f"q6_{k}_lastV", b.get_height()) for (k, _, _), b in zip(VARIANTS, bars)]

    # bottom-left: hot-swaps performed. A swaps the most and still never recovers -- the count
    # separates "did nothing" (B/C/Adaptive: 0) from "acted and failed" (A), which the outcome
    # panels alone cannot distinguish.
    hs = [C[f"q6_{k}_hotswaps"] for k, _, _ in VARIANTS]
    b3 = ax[2].bar(x, [e["value"] for e in hs], color=[c for _, _, c in VARIANTS],
                   edgecolor="black", linewidth=0.5)
    ax[2].set_ylabel("hot-swaps performed")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels([l for _, l, _ in VARIANTS], fontsize=5)
    ax[2].set_title("swap effort (misprediction)", fontsize=8)
    for b, (k, _, _) in zip(b3, VARIANTS):
        if C[f"q6_{k}_recoveries"]["value"] == 0 and b.get_height() > 0:
            ax[2].annotate("no recovery", (b.get_x() + b.get_width() / 2, b.get_height()),
                           textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5)
    zero_labels(ax[2], b3)
    ids += [(f"q6_{k}_hotswaps", b.get_height()) for (k, _, _), b in zip(VARIANTS, b3)]

    # bottom-right: persistence under CORRECT prediction -- the control. Every variant lands in
    # the same band, so the separation above is attributable to misprediction, not to the
    # mechanisms costing anything when the predictor is right.
    ctrl = [(k, l, c) for k, l, c in VARIANTS if f"q4b_{k}_persist" in C]
    e4 = [C[f"q4b_{k}_persist"] for k, _, _ in ctrl]
    b4 = ax[3].bar(range(len(ctrl)), [e["value"] for e in e4],
                   yerr=[e.get("sd") or 0 for e in e4], capsize=2,
                   color=[c for _, _, c in ctrl], edgecolor="black", linewidth=0.5)
    ax[3].set_ylabel("persistence (s)")
    ax[3].set_xticks(range(len(ctrl)))
    ax[3].set_xticklabels([l for _, l, _ in ctrl], fontsize=5)
    ax[3].set_title("control: correct prediction (all variants alike)", fontsize=8)
    ids += [(f"q4b_{k}_persist", b.get_height()) for (k, _, _), b in zip(ctrl, b4)]

    fig.text(0.01, 0.005,
             "5 runs per bar, fixed 94 s window. Hatched = censored (never returned below "
             "$\\epsilon$). B is the adaptive hot-swap baseline: with neither mechanism the "
             "predictor re-returns the same top-ranked placement, so the two coincide. "
             "A is absent from the control panel: it recovered in only 2 of 5 runs there, so no "
             "persistence is reported over the full five.",
             fontsize=4.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"c2_reactive_comparison.{ext}"), bbox_inches="tight")
    sidecar(ids, "c2_reactive_comparison.values.json", C)
    plt.close(fig)
    print("regenerated: c2_reactive_comparison (+ sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
