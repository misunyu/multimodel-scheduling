"""Regenerate Step H2 Part C decomposition figure with full GPU baseline.

Uses both step_h2_decomposition.csv (extended with GPU N=1 L0/L1 supplement)
and Step F Part A. Also produces a refined Part B 'schedule shape' figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"

COMP_A = [2, 22, 13, 16, 3, 21, 14, 4]
BG_ORDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]


def gpu_n1_full(sid_list):
    """GPU N=1 bg L0 for each sid: prefer Step F Part A, fall back to Part C supp."""
    df_pa = pd.read_csv(RES / "step_f_partA_matrix.csv")
    df_c = pd.read_csv(RES / "step_h2_decomposition.csv")
    out = {}
    for sid in sid_list:
        pa = df_pa[(df_pa.device == "GPU") & (df_pa.sid == sid)]
        if len(pa):
            out[sid] = float(pa.iloc[0]["sap_5095"])
        else:
            sup = df_c[(df_c.placement_name == "N1_GPU_solo") &
                       (df_c.bg_level == "L0") & (df_c.s0_sid == sid)]
            if len(sup):
                out[sid] = float(sup.iloc[0]["s0_sap"])
    return out


def npu_n1(sid_list, bg):
    df_c = pd.read_csv(RES / "step_h2_decomposition.csv")
    out = {}
    for sid in sid_list:
        r = df_c[(df_c.placement_name == "N1_NPU_solo") &
                 (df_c.bg_level == bg) & (df_c.s0_sid == sid)]
        if len(r):
            out[sid] = float(r.iloc[0]["s0_sap"])
    return out


def gpu_n1_bg(sid_list, bg):
    df_c = pd.read_csv(RES / "step_h2_decomposition.csv")
    out = {}
    for sid in sid_list:
        r = df_c[(df_c.placement_name == "N1_GPU_solo") &
                 (df_c.bg_level == bg) & (df_c.s0_sid == sid)]
        if len(r):
            out[sid] = float(r.iloc[0]["s0_sap"])
    return out


def main():
    # --- Part C: decomposition data ---
    gpu_l0 = gpu_n1_full(COMP_A)
    gpu_l1 = gpu_n1_bg(COMP_A, "L1_light")
    npu_l0 = npu_n1(COMP_A, "L0")
    npu_l1 = npu_n1(COMP_A, "L1_light")

    df_c = pd.read_csv(RES / "step_h2_decomposition.csv")
    n8_l0_row = df_c[(df_c.n_streams == 8) & (df_c.bg_level == "L0") &
                     (df_c.placement_name == "AllNPU_N8_L0")].iloc[0]

    # Per-sid sAP under N=8 AllNPU bg L0
    npu_n8_l0 = {}
    for i in range(8):
        sid = int(n8_l0_row[f"s{i}_sid"])
        npu_n8_l0[sid] = float(n8_l0_row[f"s{i}_sap"])

    # Per-sid sAP under N=8 AllNPU bg L1_light (from Step G2)
    df_g2 = pd.read_csv(RES / "step_g2_single_mode.csv")
    g2_n8_allnpu = df_g2[(df_g2.n_streams == 8) &
                          (df_g2.placement_spec == '["NPU", "NPU", "NPU", "NPU", "NPU", "NPU", "NPU", "NPU"]')].iloc[0]
    npu_n8_l1 = {}
    for i in range(8):
        sid = int(g2_n8_allnpu[f"s{i}_sid"])
        npu_n8_l1[sid] = float(g2_n8_allnpu[f"s{i}_sap"])

    # Naive @ Step G2 (GPU N=8 bg L1) per-sid for comparison
    naive_n8_l1 = {}
    g2_n8_naive = df_g2[(df_g2.n_streams == 8) &
                         (df_g2.placement_name.str.contains("Naive_allGPU"))].iloc[0]
    for i in range(8):
        sid = int(g2_n8_naive[f"s{i}_sid"])
        naive_n8_l1[sid] = float(g2_n8_naive[f"s{i}_sap"])

    # Compute mean and worst over all 8 sids
    sids = COMP_A
    def stat(d):
        a = np.array([d[s] for s in sids])
        return a.mean(), a.min()

    g0_m, g0_w = stat(gpu_l0)
    g1_m, g1_w = stat(gpu_l1)
    n0_m, n0_w = stat(npu_l0)
    n0L1_m, n0L1_w = stat(npu_l1)
    n8L0_m, n8L0_w = stat(npu_n8_l0)
    n8L1_m, n8L1_w = stat(npu_n8_l1)
    naive_m, naive_w = stat(naive_n8_l1)

    print("=" * 88)
    print(" Composition A — N=1/N=8 baselines (worst- and mean-stream sAP)")
    print("=" * 88)
    print(f"{'Condition':<30s}  {'mean sAP':>10s}  {'worst sAP (sid)':>17s}")
    print("-" * 88)
    def line(name, m, w, d):
        sid_at_worst = min(d, key=d.get)
        print(f"{name:<30s}  {m:>10.4f}  {w:>10.4f} ({sid_at_worst:>2d})")
    line("GPU  N=1  bg L0   (ref)", g0_m, g0_w, gpu_l0)
    line("GPU  N=1  bg L1_light",   g1_m, g1_w, gpu_l1)
    line("NPU  N=1  bg L0   (+quant)", n0_m, n0_w, npu_l0)
    line("NPU  N=1  bg L1_light",     n0L1_m, n0L1_w, npu_l1)
    line("NPU  N=8 AllNPU bg L0",     n8L0_m, n8L0_w, npu_n8_l0)
    line("NPU  N=8 AllNPU bg L1_l (Step G2)", n8L1_m, n8L1_w, npu_n8_l1)
    line("GPU  N=8 Naive  bg L1_l (Step G2)", naive_m, naive_w, naive_n8_l1)

    # ---- Decomposition components (mean & worst) ----
    deltas_m = {
        "Quantization":      g0_m - n0_m,            # GPU → NPU at N=1 L0
        "Bg→NPU":            n0_m - n0L1_m,          # negligible (NPU isolated from GPU bg)
        "Concurrent N=8":    n0L1_m - n8L0_m,        # NPU N=1 → N=8 at L0  (we use L1_light reference here)
        "Interaction":       n8L0_m - n8L1_m,        # N=8 L0 → L1
    }
    deltas_w = {
        "Quantization":      g0_w - n0_w,
        "Bg→NPU":            n0_w - n0L1_w,
        "Concurrent N=8":    n0L1_w - n8L0_w,
        "Interaction":       n8L0_w - n8L1_w,
    }
    print("\n=== Decomposition (mean sAP) ===")
    for k, v in deltas_m.items():
        print(f"  Δ{k:<18s} = {v:+.4f}")
    print(f"  Total (GPU L0 → NPU L1_l N=8) = {g0_m - n8L1_m:+.4f}  (mean)")

    print("\n=== Decomposition (worst sAP) ===")
    for k, v in deltas_w.items():
        print(f"  Δ{k:<18s} = {v:+.4f}")
    print(f"  Total (GPU L0 → NPU L1_l N=8) = {g0_w - n8L1_w:+.4f}  (worst)")

    # ---- Counterfactual GPU loss ----
    # GPU at N=8 bg L1 = Naive measurement
    gpu_n8_l1_loss = g0_w - naive_w
    npu_n8_l1_loss = g0_w - n8L1_w
    saved = gpu_n8_l1_loss - npu_n8_l1_loss
    print(f"\n=== AllNPU vs Naive (Step G2 comparison, worst sAP) ===")
    print(f"  GPU N=1 L0 (best ref):    {g0_w:.3f}")
    print(f"  GPU N=8 Naive bg L1_l:    {naive_w:.3f}  (loses {gpu_n8_l1_loss:.3f} = contention)")
    print(f"  NPU N=8 AllNPU bg L1_l:   {n8L1_w:.3f}  (loses {npu_n8_l1_loss:.3f} = quant only)")
    print(f"  Worst-stream gain:        +{saved:.3f}  (= contention − quantization)")

    # ---- Figure: 2-panel ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: cumulative path
    ax = axes[0]
    labels = ["GPU N=1\nbg L0\n(ref)",
              "NPU N=1\nbg L0\n(+quant)",
              "NPU N=1\nbg L1_light\n(+bg→NPU)",
              "NPU N=8\nbg L0\n(+concurrent)",
              "NPU N=8\nbg L1_light\n(+interaction)",
              "GPU N=8 Naive\nbg L1_light\n(GPU saturation)"]
    means = [g0_m, n0_m, n0L1_m, n8L0_m, n8L1_m, naive_m]
    worsts = [g0_w, n0_w, n0L1_w, n8L0_w, n8L1_w, naive_w]
    x = np.arange(len(labels))
    w_bar = 0.38
    ax.bar(x - w_bar / 2, means, w_bar, color="#3680c4", edgecolor="black",
           linewidth=0.5, label="mean sAP")
    ax.bar(x + w_bar / 2, worsts, w_bar, color="#c43b3b", edgecolor="black",
           linewidth=0.5, label="worst sAP")
    for j, (m, w) in enumerate(zip(means, worsts)):
        ax.text(j - w_bar / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(j + w_bar / 2, w + 0.005, f"{w:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("sAP (over 8 streams in Composition A)")
    ax.set_title("Cumulative loss / saturation path")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel B: side-by-side bar of GPU vs NPU contention costs (worst stream)
    ax = axes[1]
    categories = ["Quantization\n(GPU N=1 L0 →\nNPU N=1 L0)",
                  "Bg → NPU\n(NPU N=1 L0 →\nL1_light)",
                  "Concurrent ×N=8\n(NPU N=1 →\nN=8 at L0)",
                  "Interaction\n(NPU N=8 L0\n→ L1_light)",
                  "GPU contention\n(GPU N=1 L0 →\nN=8 bg L1)"]
    values_w = [deltas_w["Quantization"], deltas_w["Bg→NPU"],
                deltas_w["Concurrent N=8"], deltas_w["Interaction"],
                gpu_n8_l1_loss]
    colors_b = ["#3680c4", "#7bbef0", "#c4a236", "#a6c8e8", "#c43b3b"]
    bars = ax.bar(categories, values_w, color=colors_b, edgecolor="black",
                  linewidth=0.5)
    for b, v in zip(bars, values_w):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.001 if v >= 0 else -0.005),
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("worst-stream sAP lost (Composition A, sid 21)")
    ax.set_title("Component cost — AllNPU avoids GPU contention at price of quantization")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)

    plt.suptitle("Part C — Loss decomposition: why AllNPU wins at high contention", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "step_h2_decomposition.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {FIG / 'step_h2_decomposition.pdf'}")

    # ---- Part B refined: optimal offload fraction vs bg ----
    df_b = pd.read_csv(RES / "step_h2_bg_ablation.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bg_idx = {b: i for i, b in enumerate(BG_ORDER)}
    for N in [4, 8]:
        opt_frac, opt_sap, opt_name = [], [], []
        for bg in BG_ORDER:
            sub = df_b[(df_b.n_streams == N) & (df_b.bg_level == bg)]
            if len(sub):
                best = sub.sort_values("worst_sap", ascending=False).iloc[0]
                spec = json.loads(best["placement_spec"])
                frac = spec.count("NPU") / N
                opt_frac.append(frac)
                opt_sap.append(float(best["worst_sap"]))
                opt_name.append(best["placement_name"])
            else:
                opt_frac.append(np.nan); opt_sap.append(np.nan); opt_name.append("")
        ax.plot(BG_ORDER, opt_frac, "-o", linewidth=2.4, markersize=10,
                label=f"N = {N}")
        for j, (frac, sap, nm) in enumerate(zip(opt_frac, opt_sap, opt_name)):
            if not np.isnan(frac):
                ax.annotate(f"{nm}\nsAP={sap:.3f}",
                            xy=(j, frac), xytext=(0, 12 if N == 4 else -28),
                            textcoords="offset points", fontsize=7,
                            ha="center", color="#444",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="#fffae6" if N == 4 else "#e6f3ff",
                                      ec="#888", lw=0.5))
    ax.set_xlabel("background contention level (severity →)")
    ax.set_ylabel("optimal NPU offload fraction  (k_NPU / N)")
    ax.set_ylim(-0.1, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0  (all-GPU)", "0.25", "0.5", "0.75", "1.0  (AllNPU)"])
    ax.set_title("Part B — Optimal offload schedule shifts toward AllNPU under heavier contention")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG / "step_h2_bg_schedule.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'step_h2_bg_schedule.pdf'}")


if __name__ == "__main__":
    main()
