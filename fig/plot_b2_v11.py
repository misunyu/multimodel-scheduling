#!/usr/bin/env python3
"""Regenerate b2_buffer_sweep.pdf -- Q1.5, the buffer sweep behind tab:b2.

Spec: fig/FIGURE_SPECS.md ("p99 및 drop을 beta_B 축으로, 양 플랫폼"). Values come ONLY from
fig/confirmed_values.json; sidecars are read back off the canvas.

Two things this figure has to say at once, so it is two panels:

  left   p99 tail against beta_B. This is the discriminating metric -- and it is almost
         FLAT in beta_B, which is the point: enlarging the buffer does not fix a placement
         problem. The separation is between techniques, not between buffers.
  right  where the excess frames go. Static's drop count falls as the buffer grows while
         its miss rate barely moves: the frames stop being dropped and start being late.
         Plotting drop alone would read as "bigger buffer, better", so miss is overlaid on
         a second axis.

Static is drawn with open markers throughout: it never returns below epsilon, so every
Static value is censored and must not read as a settled measurement.

usage: python3 fig/plot_b2_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

BETAS = ["0.5", "1.0", "1.5"]
TECHS = [("Static", "#777777", ":", "s"), ("Stop-restart", "#7b3294", "-.", "^"),
         ("Adaptive", "#e08214", "--", "o"), ("BoundGuard", "#1f5fa8", "-", "D")]
PLATS = [("gpu", "CPU–GPU", "-"), ("npu", "CPU–NPU", "--")]


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
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ids = []
    x = range(len(BETAS))

    # left: p99 vs beta_B, per technique per platform
    for plat, plabel, pls in PLATS:
        for tech, col, _, mk in TECHS:
            key = tech.lower().replace("-", "")
            got = [(b, f"q15_{plat}_{key}_b{b}_p99") for b in BETAS
                   if f"q15_{plat}_{key}_b{b}_p99" in C]
            if not got:
                continue
            ys = [C[i]["value"] / 1000.0 for _, i in got]          # ms -> s
            es = [(C[i].get("sd") or 0) / 1000.0 for _, i in got]
            cens = C[got[0][1]]["censored"]
            ln = ax[0].errorbar([BETAS.index(b) for b, _ in got], ys, yerr=es, capsize=2,
                                color=col, linestyle=pls, marker=mk, markersize=4,
                                markerfacecolor=("none" if cens else col), lw=1.1,
                                label=f"{tech} {plabel}")
            ids += [(i, C[i]["value"]) for _, i in got]
            _ = ln
    ax[0].set_yscale("log")
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels([f"{b}×" for b in BETAS])
    ax[0].set_xlabel(r"buffer $\beta_B$  ($B=\beta_B\cdot Q_{max}$)")
    ax[0].set_ylabel("p99 end-to-end latency (s)")
    ax[0].set_title("the tail separates techniques, not buffers", fontsize=8)
    ax[0].legend(fontsize=4.6, ncol=2, loc="center right")

    # right: Static's excess frames -- dropped (bars) vs missed (line)
    w = 0.36
    for j, (plat, plabel, _) in enumerate(PLATS):
        dids = [f"q15_{plat}_static_b{b}_drop" for b in BETAS]
        if not all(i in C for i in dids):
            continue
        bars = ax[1].bar([i + (j - 0.5) * w for i in x], [C[i]["value"] for i in dids], w,
                         color=("#cccccc" if j == 0 else "#eeeeee"), edgecolor="black",
                         linewidth=0.5, hatch=("" if j == 0 else "//"),
                         label=f"dropped, {plabel}")
        ids += [(i, b.get_height()) for i, b in zip(dids, bars)]
    ax2 = ax[1].twinx()
    for j, (plat, plabel, pls) in enumerate(PLATS):
        mids = [f"q15_{plat}_static_b{b}_miss" for b in BETAS]
        if not all(i in C for i in mids):
            continue
        ax2.plot(list(x), [C[i]["value"] for i in mids], pls, color="#b0182b",
                 marker="s", markersize=4, markerfacecolor="none", lw=1.2,
                 label=f"miss rate, {plabel}")
        ids += [(i, C[i]["value"]) for i in mids]
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels([f"{b}×" for b in BETAS])
    ax[1].set_xlabel(r"buffer $\beta_B$")
    ax[1].set_ylabel("frames dropped (Static)")
    ax2.set_ylabel("miss rate (Static)", color="#b0182b")
    ax2.set_ylim(0, 1)
    ax[1].set_title("a bigger buffer converts drops into lateness", fontsize=8)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=4.6, loc="upper right")

    fig.text(0.01, 0.005,
             r"$\beta_B{=}0.5$ n=5, others n=3. Open markers = censored (Static never "
             r"returns below $\epsilon$). miss uses the pinned 15 ms deadline; the published "
             r"8.13 ms was the $1000/\lambda$ fallback and is reported in the table.",
             fontsize=4.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"b2_buffer_sweep.{ext}"), bbox_inches="tight")
    sidecar(ids, "b2_buffer_sweep.values.json", C)
    plt.close(fig)
    print("regenerated: b2_buffer_sweep (+ sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
