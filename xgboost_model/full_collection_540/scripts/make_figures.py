"""F1-F2: camera-ready figures for the MLForSys manuscript (NeurIPS single column).

All data comes from committed analysis/ outputs — no retraining, no boosters, no
hardware. Deterministic (no randomness). matplotlib only, vector PDF with
embedded (Type 42) fonts + a dpi=200 PNG proof for each figure.

  python make_figures.py --fig f1   -> analysis/figures/fig_divergence.{pdf,png}
  python make_figures.py --fig f2   -> analysis/figures/fig_transfer.{pdf,png}

F1 takes an explicit --basis (default "declared_normalized", the manuscript's Eq. (1)
per-group normalization). It recomputes the divergence through the basis-audit run's
build_basis_audit module -- the score definitions live there, not here -- and asserts
the resulting disagreement set against that run's divergence_by_group.csv and the
expected 11-group list before anything is drawn. "--basis raw" reproduces the
superseded v1 figure from platform_divergence_raw_v1.csv.

Palette (colorblind-safe, no red/green): blue #0173B2, orange #DE8F05, plus
grays. The "unified" bar uses #CCCCCC instead of the mid-gray #999999 because
#999999 is isoluminant with the orange (grayscale-print ambiguity); #CCCCCC
keeps a monotone lightness order blue < orange < gray.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac

FIGDIR = ac.ANALYSIS / "figures"
BLUE, ORANGE = "#0173B2", "#DE8F05"
GRAY_BAR, GRAY_CELL = "#CCCCCC", "#E8E8E8"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# Working sets ordered by model count N ascending, ties by name (documented in
# figures/README.md). Sizes come from working_sets.yaml (committed).
SET_ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4",
             "base5", "S5", "S9", "S6", "S10", "S7", "S8"]
SET_N = {"S1": 2, "S2": 3, "S3": 3, "S4": 3, "base1": 4, "base2": 4,
         "base3": 4, "base4": 4, "base5": 4, "S5": 5, "S9": 5, "S6": 6,
         "S10": 7, "S7": 7, "S8": 8}


# F1's score basis is an EXPLICIT parameter, not an implicit property of whichever
# csv happens to sit in analysis/. "declared_normalized" is the manuscript's Eq. (1)
# basis (per-group normalization); "raw" reproduces the superseded v1 figure built on
# raw measured window totals. See the basis audit run below and figures/README.md.
BASIS_CHOICES = ("declared_normalized", "raw")
AUDIT_RUN = ac.ROOT / "runs" / "20260730_190154_score_basis_audit"
# The 11 disagreeing groups the declared basis must yield (audit run, beta=1.0).
EXPECTED_DECLARED_DIFF = {
    "S10@2.0", "S5@3.0", "S7@2.0", "S8@2.0", "base3@1.0", "base3@3.0",
    "base4@1.0", "base4@3.0", "base5@1.0", "base5@2.0", "base5@3.0"}


def _load_audit_module():
    """Import the audit run's build_basis_audit so the score definitions are shared.

    F1 must not re-implement normalization/scoring/tie-breaking: it reuses that
    module's normalize() / score() / tie_sets() verbatim.
    """
    path = AUDIT_RUN / "build_basis_audit.py"
    spec = importlib.util.spec_from_file_location("build_basis_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _audit_csv_diff_groups(basis, beta=1.0):
    """Disagreeing groups recorded in the audit run's divergence_by_group.csv."""
    out = set()
    with open(AUDIT_RUN / "divergence_by_group.csv") as fh:
        for r in csv.DictReader(fh):
            if r["basis"] == basis and float(r["beta"]) == beta and int(r["agree"]) == 0:
                out.add(f"{r['set']}@{float(r['rate'])}")
    return out


def load_divergence_declared(beta=1.0):
    """(set, rate) -> same(0/1) on the declared (Eq. 1) basis, recomputed + asserted.

    Recomputes from the measured windows through the audit module, then checks the
    result against BOTH the audit run's per-group csv and the expected 11-group list.
    A mismatch is a hard failure: the figure is never written from unverified data.
    """
    bba = _load_audit_module()
    rows = {pl: bba.normalize(bba.load_rows(pl), "r1_totals") for pl in ("gpu", "npu")}
    tg = bba.tie_sets(rows["gpu"], bba.ALPHA, beta, "normalized")
    tn = bba.tie_sets(rows["npu"], bba.ALPHA, beta, "normalized")
    same = {k: int(bool(tg[k] & tn[k])) for k in tg}

    got = {f"{s}@{float(r)}" for (s, r), v in same.items() if not v}
    from_csv = _audit_csv_diff_groups("declared_normalized", beta)
    assert got == from_csv, (
        "declared-basis disagreement set differs from the audit csv:\n"
        f"  recomputed-only: {sorted(got - from_csv)}\n"
        f"  csv-only:        {sorted(from_csv - got)}")
    if beta == 1.0:
        assert got == EXPECTED_DECLARED_DIFF, (
            "declared-basis disagreement set differs from the expected 11 groups:\n"
            f"  unexpected: {sorted(got - EXPECTED_DECLARED_DIFF)}\n"
            f"  missing:    {sorted(EXPECTED_DECLARED_DIFF - got)}")
    print(f"F1 basis=declared_normalized beta={beta}: "
          f"{len(got)}/{len(same)} disagree, assert OK (audit csv + expected list)")
    return same


def load_divergence_raw():
    """(set, rate) -> same(0/1) for the beta=1.0 rows of the superseded v1 csv."""
    rows = {}
    with open(ac.ANALYSIS / "platform_divergence_raw_v1.csv") as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            if float(r["beta"]) == 1.0:
                rows[(r["set"], float(r["rate"]))] = int(r["same"])
    return rows


def fig_f1(basis="declared_normalized"):
    assert basis in BASIS_CHOICES, basis
    rows = (load_divergence_declared() if basis == "declared_normalized"
            else load_divergence_raw())
    rates_of = {s: sorted(r for (ss, r) in rows if ss == s) for s in SET_ORDER}
    assert all(len(v) == 3 for v in rates_of.values())
    # matrix[y][x]: y = set (N ascending, top->bottom), x = low/mid/high rate
    diff = [[1 - rows[(s, rates_of[s][x])] for x in range(3)] for s in SET_ORDER]
    n_diff = sum(sum(r) for r in diff)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for y, s in enumerate(SET_ORDER):
        for x in range(3):
            # inset cells: a visible white grid so the three rate columns read
            # as columns, not one continuous bar
            ax.add_patch(plt.Rectangle(
                (x + 0.06, len(SET_ORDER) - 1 - y + 0.10), 0.88, 0.80,
                facecolor=ORANGE if diff[y][x] else GRAY_CELL,
                edgecolor="none"))
    ax.set_xlim(0, 3); ax.set_ylim(0, len(SET_ORDER))
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_yticks([len(SET_ORDER) - 1 - i + 0.5 for i in range(len(SET_ORDER))])
    ax.set_yticklabels([f"{s} ({SET_N[s]})" for s in SET_ORDER])
    ax.set_xlabel("Input-rate level")
    ax.set_ylabel("Working set (model count)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=GRAY_CELL, edgecolor="white", label="same placement"),
        Patch(facecolor=ORANGE, edgecolor="white",
              label=f"different placement ({n_diff}/45)")],
        loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, frameon=False,
        handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_divergence")

    # caption material: disagreement by N
    by_n = {}
    for y, s in enumerate(SET_ORDER):
        d, n = sum(diff[y]), SET_N[s]
        a, b = by_n.get(n, (0, 0))
        by_n[n] = (a + d, b + 3)
    return {"basis": basis, "n_diff": n_diff, "by_n": by_n, "rates_of": rates_of}


def fig_f2():
    cp = json.loads((ac.ANALYSIS / "cross_platform_metrics.json").read_text())
    uni = json.loads((ac.ANALYSIS / "unified_metrics.json").read_text())
    spec = {pl: json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
            for pl in ("gpu", "npu")}
    other = {"gpu": "npu", "npu": "gpu"}
    vals = {}          # (metric, eval_pl) -> [specialized, transfer, unified]
    for pl in ("gpu", "npu"):
        s = spec[pl]["ranking"]
        t = cp[f"{other[pl]}->{pl}|asis"]["ranking"]
        u = uni["per_platform"][pl]["ranking"]
        vals[("rho", pl)] = [s["group_spearman_mean"], t["group_spearman_mean"],
                             u["group_spearman_mean"]]
        vals[("top1", pl)] = [s["top1"], t["top1"], u["top1"]]

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
    panels = [("rho", "Group Spearman (axis from 0.85)", (0.85, 1.0), 0.05),
              ("top1", "Top-1 rate (axis from 0.5)", (0.5, 1.0), 0.1)]
    colors = [BLUE, ORANGE, GRAY_BAR]
    labels = ["specialized (own platform)", "zero-shot transfer", "unified (both)"]
    width = 0.26
    for ax, (metric, ylab, ylim, step) in zip(axes, panels):
        rng = ylim[1] - ylim[0]
        for gi, pl in enumerate(("gpu", "npu")):
            vs = vals[(metric, pl)]
            prev_dodged = False
            for bi in range(3):
                v = vs[bi]
                x = gi + (bi - 1) * width
                ax.bar(x, v - ylim[0], width * 0.92, bottom=ylim[0],
                       color=colors[bi], edgecolor="white", linewidth=0.5)
                # stagger a label one line up when the NEIGHBORING label sits at
                # (nearly) the same height and was not itself staggered — keeps
                # the 0.933/0.933/0.933 triple readable (up, base, up pattern
                # never occurs adjacent)
                dodged = (bi > 0 and not prev_dodged
                          and abs(v - vs[bi - 1]) < rng * 0.03)
                ax.text(x, v + rng * 0.015 + (rng * 0.055 if dodged else 0.0),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
                prev_dodged = dodged
        ax.set_ylim(*ylim)
        ax.set_yticks([round(ylim[0] + i * step, 2)
                       for i in range(int(round((ylim[1] - ylim[0]) / step)) + 1)])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["eval on GPU", "eval on NPU"])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2)
        ax.margins(x=0.12)
    axes[0].legend(handles=[Patch(facecolor=c, label=l)
                            for c, l in zip(colors, labels)],
                   loc="lower center", bbox_to_anchor=(1.08, 1.0), ncol=3,
                   frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_transfer")
    return vals


def save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {FIGDIR / name}.pdf/.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", choices=["f1", "f2"], required=True)
    ap.add_argument("--basis", choices=list(BASIS_CHOICES),
                    default="declared_normalized",
                    help="F1 score basis (default: the manuscript's Eq. (1) basis)")
    args = ap.parse_args()
    info = fig_f1(args.basis) if args.fig == "f1" else fig_f2()
    print(json.dumps({str(k): v for k, v in (info or {}).items()
                      if k != "rates_of"}, default=str, ensure_ascii=False))
