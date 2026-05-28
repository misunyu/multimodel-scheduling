"""Step C — Scene × Device sAP analysis.

Joins Step A's per-(log,device) sAP measurements with Step B's scene labels,
then aggregates by (scene_label, device). Answers the three motivating-
experiment questions:

  Q1: does scene affect sAP?           (Light vs Dense t-test per device)
  Q2: do device gaps depend on scene?  (CPU-GPU, CPU-NPU, GPU-NPU per scene)
  Q3: is the best device scene-dependent?  (argmax sAP per scene)
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
FIG = SCRIPT_DIR.parent / "figures"
OUT_CSV = RES / "step_c_scene_device_sap.csv"

SCENE_ORDER = ["Light", "Medium", "Dense"]
DEVICE_ORDER = ["CPU", "GPU", "NPU"]


# ---------- statistical helpers (no scipy dependency) -----------------------

def welch_t_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-statistic + two-sided p-value via normal approx (n small)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    # Welch–Satterthwaite df
    df = (va / na + vb / nb) ** 2 / ((va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1)))
    # Two-sided p via Student's t survival; lacking scipy, use t -> z approx (df small but OK for direction sense).
    # Approximate via error function for the normal tail (decent for df>=6).
    from math import erf, sqrt
    z = abs(t)
    p_two = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return float(t), float(p_two)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


# ---------- core analysis ---------------------------------------------------

def main():
    a = pd.read_csv(RES / "step_a_baseline.csv")
    b = pd.read_csv(RES / "step_b_scene_classification.csv")
    df = a.merge(b[["log_id", "scene_label", "mean_objects"]], on="log_id", how="inner")
    assert len(df) == len(a), f"join missed rows: {len(df)} vs {len(a)}"

    # aggregate
    agg = (df.groupby(["scene_label", "device"])
             .agg(n=("sap_5095", "size"),
                  sap_mean=("sap_5095", "mean"),
                  sap_std=("sap_5095", "std"),
                  sap_min=("sap_5095", "min"),
                  sap_max=("sap_5095", "max"),
                  sap50_mean=("sap_50", "mean"),
                  sap_s_mean=("sap_small", "mean"),
                  sap_m_mean=("sap_medium", "mean"),
                  sap_l_mean=("sap_large", "mean"),
                  infer_mean_ms=("infer_mean_ms", "mean"))
             .reset_index())
    agg.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV}")

    # markdown table sAP mean ± std (rows=scene, cols=device)
    pivot_m = agg.pivot(index="scene_label", columns="device", values="sap_mean").reindex(SCENE_ORDER)[DEVICE_ORDER]
    pivot_s = agg.pivot(index="scene_label", columns="device", values="sap_std").reindex(SCENE_ORDER)[DEVICE_ORDER]
    pivot_lat = agg.pivot(index="scene_label", columns="device", values="infer_mean_ms").reindex(SCENE_ORDER)[DEVICE_ORDER]
    print("\n=== Scene × Device  sAP@[0.50:0.95] mean ± std ===")
    print(f"| scene  | {' | '.join([f'{d:>14s}' for d in DEVICE_ORDER])} |")
    print(f"|--------|{'|'.join(['-' * 16 for _ in DEVICE_ORDER])}|")
    for sc in SCENE_ORDER:
        cells = [f"{pivot_m.loc[sc, d]:.3f} ± {pivot_s.loc[sc, d]:.3f}" for d in DEVICE_ORDER]
        print(f"| {sc:<6s} | {' | '.join([f'{c:>14s}' for c in cells])} |")

    print("\n=== Scene × Device  avg inference latency (ms) ===")
    print(f"| scene  | {' | '.join([f'{d:>8s}' for d in DEVICE_ORDER])} |")
    print(f"|--------|{'|'.join(['-' * 10 for _ in DEVICE_ORDER])}|")
    for sc in SCENE_ORDER:
        cells = [f"{pivot_lat.loc[sc, d]:.1f}" for d in DEVICE_ORDER]
        print(f"| {sc:<6s} | {' | '.join([f'{c:>8s}' for c in cells])} |")

    # --- Q1: scene effect (Light vs Dense per device) ---
    print("\n=== Q1: Light vs Dense  per device  (Welch t + Cohen's d) ===")
    q1_summary = []
    for dev in DEVICE_ORDER:
        light = df[(df.scene_label == "Light") & (df.device == dev)]["sap_5095"].values
        dense = df[(df.scene_label == "Dense") & (df.device == dev)]["sap_5095"].values
        t, p = welch_t_p(light, dense)
        d = cohens_d(light, dense)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {dev}: Light mean={light.mean():.3f}  Dense mean={dense.mean():.3f}  "
              f"diff={light.mean()-dense.mean():+.3f}  t={t:+.2f}  p={p:.4f} {sig}  d={d:+.2f}")
        q1_summary.append((dev, light.mean(), dense.mean(), p, d, sig))

    # --- Q2: device gaps per scene ---
    print("\n=== Q2: device-pair gap (sAP) by scene ===")
    pairs = list(combinations(DEVICE_ORDER, 2))
    print(f"  {'scene':<6s}  " + "  ".join([f"{a}-{b}" for a, b in pairs]))
    gap_rows = []
    for sc in SCENE_ORDER:
        gaps = {}
        for a_dev, b_dev in pairs:
            ga = df[(df.scene_label == sc) & (df.device == a_dev)]["sap_5095"].values
            gb = df[(df.scene_label == sc) & (df.device == b_dev)]["sap_5095"].values
            gaps[(a_dev, b_dev)] = ga.mean() - gb.mean()
        print(f"  {sc:<6s}  " + "  ".join([f"{gaps[p]:+.3f}" for p in pairs]))
        gap_rows.append((sc, gaps))

    # gap variability across scene
    print("\n  pair gap range across scenes:")
    for p in pairs:
        vals = [g[1][p] for g in gap_rows]
        print(f"    {p[0]}-{p[1]}: min={min(vals):+.3f}  max={max(vals):+.3f}  range={max(vals)-min(vals):.3f}")

    # --- Q3: best device per scene ---
    print("\n=== Q3: best device per scene ===")
    bests = []
    for sc in SCENE_ORDER:
        row = pivot_m.loc[sc]
        best = row.idxmax()
        ranking = row.sort_values(ascending=False)
        bests.append((sc, best, row[best]))
        rank_str = " > ".join([f"{d}({row[d]:.3f})" for d in ranking.index])
        print(f"  {sc:<6s}: best={best}  ({row[best]:.3f})   ranking: {rank_str}")
    unique_bests = sorted({b for _, b, _ in bests})
    reversal = "YES" if len(unique_bests) > 1 else "NO"
    print(f"\n  reversal (best device differs across scenes)? {reversal}")
    print(f"  unique best devices: {unique_bests}")

    # --- visualizations ---
    FIG.mkdir(parents=True, exist_ok=True)

    # C1: heatmap
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow(pivot_m.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(DEVICE_ORDER))); ax.set_xticklabels(DEVICE_ORDER)
    ax.set_yticks(range(len(SCENE_ORDER))); ax.set_yticklabels(SCENE_ORDER)
    for i, sc in enumerate(SCENE_ORDER):
        for j, dv in enumerate(DEVICE_ORDER):
            ax.text(j, i, f"{pivot_m.loc[sc, dv]:.3f}",
                    ha="center", va="center", color="white", fontsize=11)
    ax.set_title("sAP@[0.50:0.95]  by scene × device")
    plt.colorbar(im, ax=ax, label="sAP")
    plt.tight_layout()
    fig.savefig(FIG / "scene_device_sap_heatmap.png", dpi=120)

    # C2: grouped bars with error bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SCENE_ORDER))
    w = 0.25
    colors = {"CPU": "#888888", "GPU": "#3680c4", "NPU": "#c4a236"}
    for i, dev in enumerate(DEVICE_ORDER):
        vals = [pivot_m.loc[sc, dev] for sc in SCENE_ORDER]
        errs = [pivot_s.loc[sc, dev] for sc in SCENE_ORDER]
        b = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
                   label=dev, color=colors[dev], edgecolor="black")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(SCENE_ORDER)
    ax.set_ylabel("sAP@[0.50:0.95]")
    ax.set_title("sAP by scene × device (mean ± std)")
    ax.legend(title="device")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "scene_device_sap_bars.png", dpi=120)

    # C3: device-pair gap lines
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a_dev, b_dev in pairs:
        ys = []
        for sc in SCENE_ORDER:
            ga = df[(df.scene_label == sc) & (df.device == a_dev)]["sap_5095"].values
            gb = df[(df.scene_label == sc) & (df.device == b_dev)]["sap_5095"].values
            ys.append(ga.mean() - gb.mean())
        ax.plot(SCENE_ORDER, ys, "-o", label=f"{a_dev} − {b_dev}")
        for i, y in enumerate(ys):
            ax.annotate(f"{y:+.3f}", (i, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("sAP gap (A − B)")
    ax.set_title("device-pair sAP gap vs scene density")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "device_gap_by_scene.png", dpi=120)

    print(f"\nsaved figures: heatmap / bars / gap to {FIG}")

    # final go/no-go decision
    print("\n=== Step C verdict ===")
    yes_q1 = any(s != "ns" for _, _, _, _, _, s in q1_summary)
    # Q2 yes if any pair has range > 0.02 (5% relative-ish)
    pair_ranges = []
    for p in pairs:
        vals = [g[1][p] for g in gap_rows]
        pair_ranges.append(max(vals) - min(vals))
    yes_q2 = max(pair_ranges) > 0.02
    yes_q3 = reversal == "YES"
    n_yes = sum([yes_q1, yes_q2, yes_q3])
    print(f"  Q1 (scene affects sAP)         : {'YES' if yes_q1 else 'NO'}")
    print(f"  Q2 (device gaps differ by scene): {'YES' if yes_q2 else 'NO'}  (max pair range={max(pair_ranges):.3f})")
    print(f"  Q3 (best device reverses)       : {'YES' if yes_q3 else 'NO'}")
    print(f"  → {n_yes}/3 signals")
    if n_yes == 3:
        print("  STRONG signal → proceed to Step D (multi-stream simulation)")
    elif n_yes >= 1:
        print("  MODERATE signal → tighten framing before Step D")
    else:
        print("  WEAK signal → reconsider hypothesis with user")


if __name__ == "__main__":
    main()
