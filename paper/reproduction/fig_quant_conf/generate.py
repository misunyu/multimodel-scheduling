"""fig:quant_conf — STAGE 2 of 2: figure rendering (NO dataset needed).

Reproduces paper Figure fig:quant_conf (per-size matched-detection confidence
distributions, FP32 vs INT8) from the bundled aggregated intermediate
data/quant_conf_matched_scores.csv. It does NOT read val.json or the raw npz
dumps — the Argoverse-HD dataset is only needed to (re)build that intermediate
via extract_data.py (STAGE 1).

The plotting block below is copied VERBATIM from the original single-file
generator; the only change is that the per-bin matched-score arrays
(score_all[b]["fp32"/"int8"]) are now LOADED from the intermediate CSV instead
of being recomputed from val.json + npz. No plotting/computation logic changed,
so the figure is identical.

  * INPUT : data/quant_conf_matched_scores.csv  (size, fp32_score, int8_score)
  * OUTPUT: b1_quant_conf_shift.pdf
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN_MATCHED = HERE / "data/quant_conf_matched_scores.csv"   # [reproduction] figure input: bundled aggregated intermediate
OUT_PDF = HERE / "b1_quant_conf_shift.pdf"                 # [reproduction] local output

BINS = ["small", "medium", "large"]


def load_matched_scores():
    """Reconstruct score_all[bin]['fp32'/'int8'] (Python floats, matched pairs)
    from the intermediate CSV — the exact arrays the original figure histograms."""
    if not IN_MATCHED.exists():
        raise SystemExit(
            f"[generate] intermediate not found:\n  {IN_MATCHED}\n"
            "Rebuild it from the Argoverse-HD dataset with:  python extract_data.py\n"
            "See README.md for the val.json download URL / placement.")
    score_all = {b: {"fp32": [], "int8": []} for b in BINS}
    with open(IN_MATCHED) as fh:
        header = fh.readline()  # size,fp32_score,int8_score
        for line in fh:
            line = line.strip()
            if not line:
                continue
            b, fp_s, np_s = line.split(",")
            score_all[b]["fp32"].append(float(fp_s))
            score_all[b]["int8"].append(float(np_s))
    return score_all


def main():
    score_all = load_matched_scores()

    # console parity: per-bin matched-score means/shift (derived from the
    # intermediate; identical to the original STAGE-1 report).
    print("=== Confidence shift (matched pairs, from intermediate) ===")
    for b in BINS:
        fa = np.asarray(score_all[b]["fp32"]); na = np.asarray(score_all[b]["int8"])
        if len(fa):
            print(f"  {b:7s}: FP32 mean={fa.mean():.3f}  INT8 mean={na.mean():.3f}  "
                  f"mean shift={na.mean()-fa.mean():+.3f}  (n={len(fa)})")

    # ---- figure: per-bin matched-score distributions FP32 vs INT8 ----
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42   # Type 3 -> TrueType(42)
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    colors = {"small": "#4477AA", "medium": "#EE6677", "large": "#228833"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    bins_h = np.linspace(0.25, 1.0, 26)
    for ax, b in zip(axes, BINS):
        fa = np.asarray(score_all[b]["fp32"]); na = np.asarray(score_all[b]["int8"])
        ax.hist(fa, bins=bins_h, alpha=0.5, color="#888888", label="FP32", density=True)
        ax.hist(na, bins=bins_h, alpha=0.6, color=colors[b], label="INT8", density=True)
        if len(fa):
            ax.axvline(fa.mean(), color="#555555", ls="--", lw=1.2)
            ax.axvline(na.mean(), color=colors[b], ls="-", lw=1.4)
            ax.set_title(f"{b}: shift {na.mean()-fa.mean():+.3f}")
        ax.set_xlabel("matched detection score")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
