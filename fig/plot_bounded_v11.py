#!/usr/bin/env python3
"""Regenerate bounded_recovery_analysis.pdf and q4_bounded_envelope.pdf.

Spec: fig/FIGURE_SPECS.md, plus the v33 correction below.

bounded_recovery_analysis decomposes recovery delay into lead + search + drain and puts the
envelope next to it. Four traps the spec calls out are handled explicitly:

  trap 1  Q4 is a DIFFERENT QUANTITY. No candidate recovers, so there is no search and no
          drain -- only the envelope. It is drawn in a separate panel, not as a fourth bar
          beside the recovering scenarios, because a shared axis invites reading a
          termination delay as a recovery delay.
  trap 2  Q4 is censored; the recovering scenarios are n=5 measured. Marked in the legend.
  trap 3  reference origins. v33 unifies every bar on t0 and shows the pre-activation part
          of Q5's lead separately, which is what made Q3 and Q5 look incommensurable before.
  trap 4  Q5's drain is 0.0 s -- a zero-height segment vanishes. It gets an explicit label.

The envelope is drawn in BOTH forms: as instantiated from the run (lead + k(T_v+delta)) and
as published (T substituted for lead). The published form fails wherever k is small, and a
figure that showed only the form that holds would be the more flattering and less honest
choice.

usage: python3 fig/plot_bounded_v11.py
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


def fig_bounded(C):
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [2, 1]})
    ids = []

    # left: recovering scenarios, stacked lead / search-after-lead / drain vs envelope
    # every recovering scenario, ordered by k: the k=1 cases are where the published form of
    # the envelope fails, so leaving them out would have hidden the discrepancy
    rows = [("Q1.3\nCPU–GPU", "q13_gpu"), ("Q1.3\nCPU–NPU", "q13_npu"), ("§4b\nCPU–GPU", "q4b"),
            ("Q5\nCPU–NPU", "q5"), ("Q3\nCPU–GPU", "q3")]
    x = range(len(rows))
    lead = [C[f"{k}_lead"]["value"] for _, k in rows]
    srch = [C[f"{k}_bg_search"]["value"] for _, k in rows]
    drain = [C[f"{k}_bg_drain"]["value"] for _, k in rows]
    # search is measured from t0 and therefore already contains lead; plot the remainder so
    # the stack reads as a timeline instead of double-counting the lead
    after = [max(0.0, s - l) for s, l in zip(srch, lead)]
    b_lead = ax[0].bar(x, lead, 0.5, color="#cfd8e3", edgecolor="black", linewidth=0.5,
                       label="lead (detection $\\to$ first candidate)")
    ax[0].bar(x, after, 0.5, bottom=lead, color="#7fa8d4", edgecolor="black", linewidth=0.5,
              label="search (remaining traversal)")
    b_dr = ax[0].bar(x, drain, 0.5, bottom=[l + a for l, a in zip(lead, after)],
                     color="#f2b56b", edgecolor="black", linewidth=0.5,
                     label="drain (commit $\\to V\\leq\\epsilon$)")
    for i, (_, k) in enumerate(rows):
        if drain[i] < 0.5:  # trap 4: a zero-height segment is invisible
            ax[0].annotate(f"drain {drain[i]:.1f} s", (i, srch[i] + drain[i]),
                           xytext=(14, 2), textcoords="offset points", fontsize=5.5,
                           color="#b8791f",
                           arrowprops=dict(arrowstyle="-", color="#b8791f", lw=0.6))
        bd = C[f"{k}_bound"]
        ax[0].plot([i - 0.34, i + 0.34], [bd["value"]] * 2, color="#b0182b", lw=1.6,
                   solid_capstyle="butt",
                   label=("envelope: lead + $k(T_v+\\delta)$" if i == 0 else None))
        pub = float(bd["note"].split("(T_v+delta)=")[1].split("s")[0])
        ax[0].plot([i - 0.34, i + 0.34], [pub] * 2, color="#b0182b", lw=1.0, ls=":",
                   label=("published form: $T$ substituted for lead" if i == 0 else None))
        ids.append((f"{k}_bound", bd["value"]))
    ids += [(f"{k}_lead", b.get_height()) for (_, k), b in zip(rows, b_lead)]
    ids += [(f"{k}_bg_search", s) for (_, k), s in zip(rows, srch)]
    ids += [(f"{k}_bg_drain", b.get_height()) for (_, k), b in zip(rows, b_dr)]
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels([l for l, _ in rows], fontsize=6)
    ax[0].set_ylabel("seconds since $t_0$")
    ax[0].set_title("recovery delay, decomposed (n=5 each); dotted = published form, "
                    "which the $k{=}1$ cases exceed", fontsize=7)
    ax[0].legend(fontsize=5, loc="upper left")

    # right: Q4 -- a different quantity, so a different panel (trap 1)
    bd4 = C["q4_bound"]; s4 = C["q4_search"]
    ax[1].bar([0], [bd4["value"]], 0.45, color="#eeeeee", edgecolor="#b0182b",
              linewidth=1.2, hatch="///", label="envelope (no candidate recovers)")
    ax[1].annotate("search: undefined\n(0/5 recover, censored)", (0, bd4["value"] * 0.45),
                   ha="center", fontsize=6, color="0.15",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
    ax[1].set_xticks([0]); ax[1].set_xticklabels(["Q4\nCPU–GPU\nheavy-4"], fontsize=6)
    ax[1].set_ylim(0, bd4["value"] * 1.35)
    ax[1].set_ylabel("seconds since $t_0$")
    ax[1].set_title("Q4: envelope only", fontsize=8)
    ax[1].legend(fontsize=5, loc="upper center")
    ids += [("q4_bound", bd4["value"]), ("q4_search", s4["value"])]

    fig.text(0.01, 0.005,
             "Q4 is not a recovery delay: no candidate is admissible, so the envelope bounds "
             "when the search terminates, not when $V$ returns below $\\epsilon$. Q5's lead "
             "includes 9.6 s before the failure phase is applied, during which the controller "
             "is not yet active.", fontsize=4.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"bounded_recovery_analysis.{ext}"), bbox_inches="tight")
    sidecar(ids, "bounded_recovery_analysis.values.json", C)
    plt.close(fig)


def fig_q4(C):
    """V(t) for Q4: every method stays above epsilon, so the figure is about the envelope
    terminating the search, not about recovery."""
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for method, colour, ls in (("BoundGuard", "#1f5fa8", "-"), ("A", "#e08214", "--"),
                               ("B", "#7b3294", "-."), ("C", "#4d9221", (0, (3, 1, 1, 1))),
                               ("Adaptive", "#777777", ":")):
        hits = sorted(glob.glob(os.path.join(PROJECT, "runs", f"*_q4_{method}_r0")))
        if not hits:
            continue
        rows = []
        for r in csv.DictReader(open(os.path.join(hits[0], "metrics.csv"))):
            try:
                rows.append((datetime.datetime.strptime(r["timestamp"], TS), float(r["v_score"])))
            except Exception:
                continue
        t0 = next((t for t, v in rows if v > EPS), rows[0][0])
        # linestyle as a keyword: the dash-pattern tuple used for C is not a valid fmt string
        ax.plot([(t - t0).total_seconds() for t, _ in rows], [v for _, v in rows],
                linestyle=ls, color=colour, lw=1.1, label=method, marker="s", markevery=15,
                markersize=3, markerfacecolor="none")
    bd = C["q4_bound"]; lv = C["q4_bg_lastV"]
    ax.axvline(bd["value"], color="#b0182b", lw=1.4)
    ax.annotate(f"envelope {bd['value']:.0f} s\n(search terminates)", (bd["value"], 2),
                xytext=(6, 0), textcoords="offset points", fontsize=6, color="#b0182b")
    ax.axhline(EPS, color="0.35", ls="--", lw=0.8)
    ax.annotate(r"$\epsilon=1$", (0, EPS), fontsize=6, color="0.35", va="bottom")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"seconds since detection onset $t_0$")
    ax.set_ylabel(r"violation score $V(t)$")
    ax.set_title("Q4 (heavy-4): no admissible placement exists; open squares = never recovers",
                 fontsize=8)
    ax.annotate(f"BoundGuard terminal $V$ {lv['value']:.0f} (censored, 0/5 recover)",
                (0.03, 0.06), xycoords="axes fraction", fontsize=6, color="0.3")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"q4_bounded_envelope.{ext}"), bbox_inches="tight")
    sidecar([("q4_bound", bd["value"]), ("q4_bg_lastV", lv["value"])],
            "q4_bounded_envelope.values.json", C)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    C = load()
    fig_bounded(C)
    fig_q4(C)
    print("regenerated: bounded_recovery_analysis, q4_bounded_envelope (+ sidecars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
