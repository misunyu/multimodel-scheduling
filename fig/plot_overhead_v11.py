#!/usr/bin/env python3
"""Regenerate runtime_overhead.pdf -- Q1.1, what the safety layer costs at rest.

Spec: fig/FIGURE_SPECS.md. Values come ONLY from fig/confirmed_values.json.

The figure answers "good idea, but what does it cost when nothing is going wrong?", so it
is drawn as DELTAS against the unmonitored arm rather than as four absolute bars. Absolute
bars of near-identical numbers invite the reader to eyeball differences that are inside the
run-to-run spread; a delta panel with the spread drawn on it says how big the cost is and
how confident the answer is, which is the actual question.

Stated on the figure, not just in the report: in steady state Adaptive and BoundGuard
execute the same code path, because everything BoundGuard adds is per-transition and no
transition occurs. Their two bars are not two systems -- the BoundGuard bar is evidence
that arming validation costs nothing extra.

usage: python3 fig/plot_overhead_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

PLATS = [("gpu", "CPU–GPU"), ("npu", "CPU–NPU")]
MONITORED = [("adaptive", "Adaptive", "#e08214"), ("boundguard", "BoundGuard", "#1f5fa8")]
METRICS = [("cpu", "executor CPU", "percentage points"),
           ("infer_ms", "per-view inference", "ms"),
           ("throughput", "throughput", "fps")]


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
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ids = []
    for k, (metric, title, unit) in enumerate(METRICS):
        labels, vals, errs, cols = [], [], [], []
        # a delta is unreadable without its base: "-0.18 pp" means nothing until you know
        # the unmonitored arm sat at 16.68%. Printed on the panel and recorded in the
        # sidecar, so the baseline is verifiable too.
        base_txt = []
        for plat, plabel in PLATS:
            base_id = f"q11_{plat}_static_{metric}"
            if base_id not in C:
                continue
            base = C[base_id]["value"]
            base_txt.append(f"{base:.2f} ({plat[0].upper()})")   # G = CPU-GPU, N = CPU-NPU
            ids.append((base_id, base))
            for arm, alabel, col in MONITORED:
                i = f"q11_{plat}_{arm}_{metric}"
                if i not in C:
                    continue
                # delta against the unmonitored arm; the spread shown is the monitored
                # arm's own, which is the honest bound on "is this difference real"
                vals.append(C[i]["value"] - base)
                errs.append(C[i].get("sd") or 0)
                labels.append(f"{alabel}\n{plabel}")
                cols.append(col)
                ids.append((i, C[i]["value"]))
        bars = ax[k].bar(range(len(vals)), vals, yerr=errs, capsize=2, color=cols,
                         edgecolor="black", linewidth=0.5)
        ax[k].axhline(0, color="0.3", lw=0.8)
        ax[k].margins(y=0.22)   # headroom so the "within spread" labels clear the frame
        ax[k].set_xticks(range(len(labels)))
        ax[k].set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=5,
                              rotation=30, ha="right")
        ax[k].set_ylabel(f"$\\Delta$ {title} ({unit})", fontsize=7)
        # the baseline goes in the subtitle, outside the data area: inside the axes it
        # collided with the top of the error bars and was clipped by the frame
        ax[k].set_title(f"{title} vs unmonitored\nbaseline " + " · ".join(base_txt),
                        fontsize=7)
        for b, v, e in zip(bars, vals, errs):
            # a bar whose magnitude is inside its own spread is not a measured cost
            if abs(v) <= e:
                # place it past the whisker, on the side the bar points away from, so it
                # never sits on top of the bar or its error bar
                tip = v + e if v >= 0 else v - e
                ax[k].annotate("within spread", (b.get_x() + b.get_width() / 2, tip),
                               xytext=(0, 4 if v >= 0 else -10), textcoords="offset points",
                               ha="center", fontsize=4.4, color="0.35")
    fig.text(0.01, 0.005,
             "Deltas against Static (no monitoring threads), 3 runs per bar, 60 s steady "
             "state at 4 fps/view with $V(t)\\approx0$ and zero placement transitions. "
             "Adaptive and BoundGuard run the same code path here -- everything BoundGuard "
             "adds is per-transition -- so the BoundGuard bar shows that arming validation "
             "adds nothing, not that it is a second system. Baselines: G = CPU-GPU, "
             "N = CPU-NPU, in each panel's own unit.", fontsize=4.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"runtime_overhead.{ext}"), bbox_inches="tight")
    sidecar(ids, "runtime_overhead.values.json", C)
    plt.close(fig)
    print("regenerated: runtime_overhead (+ sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
