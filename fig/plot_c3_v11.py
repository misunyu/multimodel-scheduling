#!/usr/bin/env python3
"""Regenerate c3_fluid_validation.pdf (v35 C3 revalidation).

Spec: fig/FIGURE_SPECS.md. Panel structure follows the caption already in the paper
(fig/paper_figure_manifest.json, fig:fluid_validation): (a) accumulation, (b) drain,
(c) the drain slope going to zero as lambda -> mu*.

  - Values come ONLY from fig/confirmed_values.json. Nothing is recomputed here.
  - Sidecars are read back FROM THE CANVAS (line.get_xydata(), annotation text), never
    copied from the confirmed table: copying would make check_figures.py verify itself.
  - Each sidecar entry carries generator{path, sha256}.

Design decisions:
  (a) and (b) are BOTH predicted-vs-measured with an identity line, deliberately the same
      form. The reader should be able to see that (a) sits on the line and (b) does not
      without re-reading the axes. Drawing (b) as "measured vs the one prediction we like"
      would hide that all three candidate mu* values miss.
  (b) uses symlog axes. The candidate predictions span -171 to +36 frames/s while the
      measurements sit between -38 and +3; on a linear axis every measured point would
      collapse onto one pixel column and the panel would show nothing.
  (c) plots the SIGNED drain slope, not its magnitude. The GPU 0.95 point is positive -- the
      queue grew on the accelerator -- and |slope| would draw that as a small drain instead
      of the divergence it is. The zero crossing of the fitted line IS mu*_eff, so the
      panel shows the confirmed functional form and the missed constant in one picture.
  T_stable is annotated only where a finite value exists. GPU 0.95 gets "no drain" and no
      number, because the queue never emptied: writing a number there would invent one.

usage: python3 fig/plot_c3_v11.py
"""
import hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
CONFIRMED = os.path.join(HERE, "confirmed_values.json")
OUT = os.path.join(PROJECT, "docs", "figures")
SELF = os.path.relpath(os.path.abspath(__file__), PROJECT)

PLATS = [("gpu", "CPU–GPU", "#2e6da4", "o"), ("npu", "CPU–NPU", "#b5651d", "s")]
FRACS = ["070", "090", "095"]
CANDS = [("pipe", "near-idle drain pilot", "#444444", "o"),
         ("sat", "saturated pipeline", "#7a7a7a", "^"),
         ("raw", "raw inference rate", "#aaaaaa", "v")]


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def load():
    return {e["id"]: e for e in json.load(open(CONFIRMED, encoding="utf-8"))["values"]}


def sidecar(entries, name):
    """entries: list of (id, value_read_back_from_canvas)."""
    C = load()
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
    read_back = []

    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))

    # ---- (a) accumulation: dQ/dt vs lambda - mu_cpu -------------------------------------
    lo, hi = 45, 120
    ax[0].plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=1)
    for plat, plabel, col, mk in PLATS:
        xs, ys, ids_x, ids_y = [], [], [], []
        for f in FRACS:
            ids_x.append(f"c3_{plat}_{f}_acc_pred"); ids_y.append(f"c3_{plat}_{f}_acc_slope")
            xs.append(C[ids_x[-1]]["value"]); ys.append(C[ids_y[-1]]["value"])
        ln = ax[0].plot(xs, ys, mk, color=col, ms=5, ls="none", label=plabel, zorder=3)[0]
        pts = ln.get_xydata()
        read_back += [(i, p[0]) for i, p in zip(ids_x, pts)]
        read_back += [(i, p[1]) for i, p in zip(ids_y, pts)]
    ax[0].set_xlabel(r"predicted  $\lambda-\mu_{cpu}$  (frames/s)")
    ax[0].set_ylabel("measured $dQ/dt$ (frames/s)")
    ax[0].set_title("(a) accumulation, all-CPU phase", fontsize=8)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].legend(fontsize=6, loc="upper left")

    # ---- (b) drain: dQ/dt vs -(mu* - lambda) for three candidate mu* --------------------
    ticks = [-100, -30, -10, -3, 0, 3, 10, 30]
    for a_ in (ax[1],):
        a_.set_xscale("symlog", linthresh=3, linscale=0.45)
        a_.set_yscale("symlog", linthresh=3, linscale=0.45)
        a_.set_xticks(ticks); a_.set_yticks(ticks)
        a_.set_xticklabels([str(t) for t in ticks], fontsize=6)
        a_.set_yticklabels([str(t) for t in ticks], fontsize=6)
        a_.minorticks_off()
    lo2, hi2 = -200, 45
    ax[1].plot([lo2, hi2], [lo2, hi2], color="0.6", lw=0.8, ls="--", zorder=1)
    ax[1].axhline(0, color="0.85", lw=0.6, zorder=0)
    ax[1].axvline(0, color="0.85", lw=0.6, zorder=0)
    for tag, clabel, ccol, cmk in CANDS:
        xs, ys, ids_x, ids_y = [], [], [], []
        for plat, plabel, col, mk in PLATS:
            for f in FRACS:
                ids_x.append(f"c3_{plat}_{f}_drain_pred_{tag}")
                ids_y.append(f"c3_{plat}_{f}_drain_slope")
                xs.append(C[ids_x[-1]]["value"]); ys.append(C[ids_y[-1]]["value"])
        ln = ax[1].plot(xs, ys, cmk, color=ccol, ms=4.5, ls="none",
                        label=r"$\mu^{\ast}$ = " + clabel, zorder=3)[0]
        pts = ln.get_xydata()
        read_back += [(i, p[0]) for i, p in zip(ids_x, pts)]
        read_back += [(i, p[1]) for i, p in zip(ids_y, pts)]
    ax[1].set_xlabel(r"predicted  $-(\mu^{\ast}-\lambda)$  (frames/s)")
    ax[1].set_ylabel("measured $dQ/dt$ (frames/s)")
    ax[1].set_title(r"(b) drain, all-accelerator phase", fontsize=8)
    ax[1].set_xlim(lo2, hi2); ax[1].set_ylim(lo2, hi2)
    ax[1].legend(fontsize=5.5, loc="lower right")

    # ---- (c) drain slope vs lambda: the zero crossing is mu*_eff ------------------------
    for plat, plabel, col, mk in PLATS:
        lam, ys, ids_y = [], [], []
        for f in FRACS:
            i = f"c3_{plat}_{f}_drain_slope"
            # lambda is the run's arrival rate; it is the setting the sweep varies and is
            # recovered here from two confirmed measurements rather than re-declared:
            # mu*_implied = lambda - drain slope  =>  lambda = implied + slope
            lam.append(C[f"c3_{plat}_{f}_mu_star_implied"]["value"] + C[i]["value"])
            ys.append(C[i]["value"]); ids_y.append(i)
        ln = ax[2].plot(lam, ys, mk + "-", color=col, ms=5, lw=1.0, label=plabel, zorder=3)[0]
        read_back += [(i, p[1]) for i, p in zip(ids_y, ln.get_xydata())]
        eff = C[f"c3_{plat}_mu_star_eff"]
        anc = C[f"c3_{plat}_mu_star_anchor"]
        e_ln = ax[2].axvline(eff["value"], color=col, lw=0.9, ls=":", zorder=2)
        a_ln = ax[2].axvline(anc["value"], color=col, lw=0.9, ls="-.", alpha=0.5, zorder=2)
        read_back.append((f"c3_{plat}_mu_star_eff", e_ln.get_xydata()[0][0]))
        read_back.append((f"c3_{plat}_mu_star_anchor", a_ln.get_xydata()[0][0]))
        for v, ls in ((eff["value"], ":"), (anc["value"], "-.")):
            ax[2].annotate(f"{v:g}", (v, -41), fontsize=5, color=col, rotation=90,
                           ha="right", va="bottom")
        # T_stable next to each point; the cell with no finite value says so in words
        for f, x, y in zip(FRACS, lam, ys):
            tid = f"c3_{plat}_{f}_T_stable"
            if tid in C:
                t = ax[2].annotate(f"{C[tid]['value']:.1f} s", (x, y), fontsize=5.5,
                                   textcoords="offset points", xytext=(4, 4), color=col,
                                   bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
                read_back.append((tid, float(t.get_text().split()[0])))
            else:
                ax[2].annotate("no drain", (x, y), fontsize=5.5, textcoords="offset points",
                               xytext=(-4, 6), ha="right", color=col,
                               bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax[2].axhline(0, color="0.4", lw=0.7)
    ax[2].set_ylim(-44, 9)
    ax[2].set_xlabel(r"arrival rate $\lambda$ per view (fps)")
    ax[2].set_ylabel("measured drain $dQ/dt$ (frames/s)")
    ax[2].set_title(r"(c) the drain slope reaches zero at $\mu^{\ast}_{eff}$", fontsize=8)
    ax[2].legend(handles=[Line2D([], [], color=c, marker=m, ls="-", ms=4, label=l)
                          for _, l, c, m in PLATS]
                 + [Line2D([], [], color="0.3", ls=":", label=r"$\mu^{\ast}_{eff}$ (fitted)"),
                    Line2D([], [], color="0.3", ls="-.", alpha=0.6,
                           label=r"$\mu^{\ast}$ anchor (pilot)")],
                 fontsize=5.5, loc="upper left")

    fig.text(0.005, 0.005,
             "3 runs per point; dropped frames 0 and saturation-guard advances 0 in all 18 "
             "runs, so the queue-conservation assumption holds throughout. Markers are "
             "measurements (source_kind hardware); the predictions on the x axes of (a) and "
             "(b) and the fitted $\\mu^{\\ast}_{eff}$ in (c) are analytic. Annotated "
             "$T_{stable}=Q/|dQ/dt|$ is an extrapolation except at CPU–NPU's lowest "
             "$\\lambda$, the one cell whose queue emptied inside the 35 s window.",
             fontsize=4.6, wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"c3_fluid_validation.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # de-duplicate: drain_slope is drawn in (b) three times and again in (c). Keep one entry
    # per id but assert every copy drew the same number -- if a panel drew something else,
    # that is exactly the bug this pipeline exists to catch.
    seen = {}
    for ident, val in read_back:
        if ident in seen and abs(seen[ident] - float(val)) > 1e-6:
            raise SystemExit(f"{ident}: panels drew different values "
                             f"({seen[ident]} vs {val})")
        seen[ident] = float(val)
    sidecar(sorted(seen.items()), "c3_fluid_validation.values.json")
    print(f"regenerated: c3_fluid_validation (+ sidecar, {len(seen)} values from canvas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
