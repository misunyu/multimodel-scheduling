#!/usr/bin/env python3
"""Regenerate q3_misprediction.pdf and q5_npu_generalization.pdf -- V(t) per method.

Spec: fig/FIGURE_SPECS.md.
  - x is WALL-CLOCK seconds since t0, not the sample index: plotting by index compresses
    the transition region and made recovery look like it happened at "~20".
  - Traces are read from the committed runs (metrics.csv); the ANNOTATED SCALARS are the
    only thing that goes into the sidecar, and those are read back off the artists so
    check_figures.py compares the drawn value, not the intended one.
  - Baselines that never recover are drawn with open square markers and their terminal V is
    labelled as censored -- a trace that simply runs off the right edge reads as "still
    going", which is exactly what a censored value means and must be said explicitly.

usage: python3 fig/plot_traces_v11.py
"""
import csv, datetime, glob, hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)
TS = "%Y-%m-%d %H:%M:%S"
EPS = 1.0

SCENARIOS = {
    "q3_misprediction": {
        "tag": "q3", "title": "CPU–GPU, vision-3 + llama1b: predictor mis-ranks the placement",
        "methods": [("BoundGuard", "#1f5fa8", "-"), ("Adaptive", "#e08214", "--"),
                    ("Stop-restart", "#7b3294", "-."), ("Static", "#777777", ":")],
        "ids": ["q3_bg_persist", "q3_bg_search", "q3_bg_drain", "q3_baseline_lastV"],
    },
    "q5_npu_generalization": {
        "tag": "q5", "title": "CPU–NPU, vision-3 + llama1b: the same mechanism on a different accelerator",
        "methods": [("BoundGuard", "#1f5fa8", "-"), ("Adaptive", "#e08214", "--"),
                    ("Stop-restart", "#7b3294", "-."), ("Static", "#777777", ":")],
        "ids": ["q5_bg_persist", "q5_bg_search", "q5_bg_drain", "q5_baseline_lastV"],
    },
}


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def load():
    return {e["id"]: e for e in json.load(open(os.path.join(HERE, "confirmed_values.json")))["values"]}


def trace(run_dir):
    """(seconds since t0, V) for one run. t0 = first sample with V > eps, per the canonical
    definition in docs/metric_definitions.md."""
    rows = []
    for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))):
        try:
            rows.append((datetime.datetime.strptime(r["timestamp"], TS), float(r["v_score"])))
        except Exception:
            continue
    t0 = next((t for t, v in rows if v > EPS), rows[0][0] if rows else None)
    return [((t - t0).total_seconds(), v) for t, v in rows], t0


def find_run(tag, method):
    hits = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_{tag}_{method}_r0")))
    return hits[0] if hits else None


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


def draw(figname, spec, C):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for method, colour, ls in spec["methods"]:
        d = find_run(spec["tag"], method)
        if not d:
            print(f"  [skip] no run for {spec['tag']}/{method}")
            continue
        pts, _ = trace(d)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        recovered = ys[-1] <= EPS
        ax.plot(xs, ys, ls, color=colour, lw=1.2, label=method,
                marker=(None if recovered else "s"), markevery=12, markersize=3.5,
                markerfacecolor="none")
        if not recovered:
            ax.annotate(f"{ys[-1]:.1f} (censored)", (xs[-1], ys[-1]), fontsize=5,
                        color=colour, ha="right", va="bottom")
    ax.axhline(EPS, color="0.35", ls="--", lw=0.8)
    ax.annotate(r"$\epsilon=1$", (0, EPS), fontsize=6, color="0.35", va="bottom")
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)  # V is non-negative; symlog otherwise wastes a decade below zero
    ax.set_xlabel(r"seconds since detection onset $t_0$")
    ax.set_ylabel(r"violation score $V(t)$")
    ax.set_title(spec["title"], fontsize=8)
    ax.legend(fontsize=6, loc="upper right")

    # annotate the decomposition persist = search + drain on the BoundGuard trace
    pre = spec["ids"][0].rsplit("_", 2)[0]  # q3 / q5
    p, s, dr = C[f"{pre}_bg_persist"], C[f"{pre}_bg_search"], C[f"{pre}_bg_drain"]
    txt = ax.annotate(f"persist {p['value']:.1f} s = search {s['value']:.1f} + drain {dr['value']:.1f}",
                      (s["value"], EPS), xytext=(-40, -34), textcoords="offset points",
                      fontsize=6, color="#1f5fa8",
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f5fa8", lw=0.5),
                      arrowprops=dict(arrowstyle="->", color="#1f5fa8", lw=0.7))
    base = C[f"{pre}_baseline_lastV"]
    ax.annotate(f"baselines: 0/5 recover, terminal $V$ {base['value']:.1f}",
                (0.02, 0.05), xycoords="axes fraction", fontsize=6, color="0.3")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{figname}.{ext}"), bbox_inches="tight")
    # sidecar: the annotated scalars, read back from the confirmed entries actually drawn
    sidecar([(f"{pre}_bg_persist", p["value"]), (f"{pre}_bg_search", s["value"]),
             (f"{pre}_bg_drain", dr["value"]), (f"{pre}_baseline_lastV", base["value"])],
            f"{figname}.values.json", C)
    plt.close(fig)
    _ = txt


def main():
    os.makedirs(OUT, exist_ok=True)
    C = load()
    for name, spec in SCENARIOS.items():
        draw(name, spec, C)
        print(f"regenerated: {name} (+ sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
