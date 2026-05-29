"""Re-render the four motivation figures as PDFs for inclusion in main.tex.

Reads the existing CSVs (step_a_baseline / step_e_size_classification /
step_f_partA_matrix / step_f_partA_strategy / step_f_partB_placement) and
writes vector PDFs under figures/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "accv_experiments/results"

# ---------- Figure 1: object-size distribution across 24 logs ----------

def fig_size_dist():
    sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
    from step0_compare_devices import ANNOT
    with open(ANNOT) as f:
        val = json.load(f)
    SMALL_MAX, MEDIUM_MAX = 32 ** 2, 96 ** 2

    ann_by_img = {}
    for a in val["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    log_to_imgs = defaultdict(list)
    for img in val["images"]:
        log_to_imgs[img["sid"]].append(img)

    rows = []
    for sid in sorted(log_to_imgs):
        log_name = val["sequences"][sid]
        imgs = log_to_imgs[sid]
        areas = []
        for img in imgs:
            for a in ann_by_img.get(img["id"], []):
                ar = float(a["area"]) if a.get("area") else a["bbox"][2] * a["bbox"][3]
                areas.append(ar)
        if not areas:
            continue
        areas = np.asarray(areas)
        n = len(areas)
        n_s = int((areas < SMALL_MAX).sum())
        n_m = int(((areas >= SMALL_MAX) & (areas < MEDIUM_MAX)).sum())
        n_l = int((areas >= MEDIUM_MAX).sum())
        rows.append({"sid": sid, "log_id": log_name, "mean_area": float(areas.mean()),
                     "pct_small": n_s / n, "pct_medium": n_m / n, "pct_large": n_l / n})
    df = pd.DataFrame(rows)
    q1, q2 = np.quantile(df["pct_large"], [1/3, 2/3])
    df["size_label"] = df["pct_large"].apply(
        lambda v: "large-rich" if v >= q2 else ("small-rich" if v < q1 else "medium-mixed"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"small-rich": "#3680c4", "medium-mixed": "#888888", "large-rich": "#c4a236"}

    # left: histogram colored by class
    bins = np.linspace(0, df["pct_large"].max() + 0.02, 14)
    for lab in ["small-rich", "medium-mixed", "large-rich"]:
        sub = df[df["size_label"] == lab]
        axes[0].hist(sub["pct_large"], bins=bins, color=colors[lab], edgecolor="black",
                     label=f"{lab} ({len(sub)})", alpha=0.85)
    axes[0].set_xlabel("pct of objects with area $\\geq 96^2$ (large)")
    axes[0].set_ylabel("# logs")
    axes[0].set_title("Large-object share per log (tertile-based class)")
    axes[0].legend(title="size_label", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # middle: scatter mean area per log
    df_s = df.sort_values("mean_area").reset_index(drop=True)
    for lab in ["small-rich", "medium-mixed", "large-rich"]:
        sub = df_s[df_s["size_label"] == lab]
        axes[1].scatter(sub.index, sub["mean_area"], c=colors[lab], edgecolors="black",
                        s=80, label=lab)
    axes[1].axhline(SMALL_MAX, color="gray", linestyle="--", linewidth=0.8, label="$32^2$ (small/medium)")
    axes[1].axhline(MEDIUM_MAX, color="gray", linestyle=":", linewidth=0.8, label="$96^2$ (medium/large)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("log index (sorted by mean object area)")
    axes[1].set_ylabel("mean object area ($\\mathrm{px}^2$, log)")
    axes[1].set_title("Mean object area per log")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(alpha=0.3)

    # right: stacked composition sorted by pct_large desc
    df_b = df.sort_values("pct_large", ascending=False).reset_index(drop=True)
    x = np.arange(len(df_b))
    s = df_b["pct_small"].values; m = df_b["pct_medium"].values; l = df_b["pct_large"].values
    axes[2].bar(x, s, color="#3680c4", label="small")
    axes[2].bar(x, m, bottom=s, color="#888888", label="medium")
    axes[2].bar(x, l, bottom=s + m, color="#c4a236", label="large")
    for i, row in df_b.iterrows():
        axes[2].scatter(i, -0.04, marker="s", s=30, color=colors[row["size_label"]], clip_on=False)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{int(s)}" for s in df_b["sid"]], rotation=90, fontsize=7)
    axes[2].set_xlabel("log sid (sorted by pct_large desc)")
    axes[2].set_ylabel("size composition (fraction)")
    axes[2].set_title("Per-log size composition (bottom row = class color)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_ylim(-0.08, 1.05)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT / "motiv_size_dist.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------- Figure 2: per-device sAP + size + latency (from step_a) ----------

def fig_single_stream():
    df = pd.read_csv(RES / "step_a_baseline.csv")
    DEV = ["CPU", "GPU", "NPU"]
    colors = {"CPU": "#888888", "GPU": "#3680c4", "NPU": "#c4a236"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), gridspec_kw={"width_ratios": [1.0, 2.0, 1.0]})

    # Panel 1: overall sAP
    ax = axes[0]
    metrics = ["sAP$_{[.50:.95]}$", "sAP$_{0.50}$"]
    x = np.arange(len(metrics)); w = 0.25
    for i, dev in enumerate(DEV):
        sub = df[df.device == dev]
        vals = [sub.sap_5095.mean(), sub.sap_50.mean()]
        errs = [sub.sap_5095.std(), sub.sap_50.std()]
        bars = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
                      label=dev, color=colors[dev], edgecolor="black")
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("sAP (24-log mean $\\pm$ std)")
    ax.set_title("Overall sAP")
    ax.legend(title="device", loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    # Panel 2: AP by size
    ax = axes[1]
    sizes = ["small", "medium", "large"]
    x = np.arange(len(sizes)); w = 0.25
    for i, dev in enumerate(DEV):
        sub = df[df.device == dev]
        vals = [sub.sap_small.mean(), sub.sap_medium.mean(), sub.sap_large.mean()]
        errs = [sub.sap_small.std(), sub.sap_medium.std(), sub.sap_large.std()]
        bars = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
                      label=dev, color=colors[dev], edgecolor="black")
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("AP (24-log mean $\\pm$ std)")
    ax.set_title("AP by object size (streaming-paired)")
    ax.legend(title="device", loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: latency
    ax = axes[2]
    x = np.arange(len(DEV))
    means = [df[df.device == dev].infer_mean_ms.mean() for dev in DEV]
    p95s = [df[df.device == dev].infer_p95_ms.mean() for dev in DEV]
    stds = [df[df.device == dev].infer_mean_ms.std() for dev in DEV]
    b1 = ax.bar(x - 0.18, means, 0.36, yerr=stds, capsize=4, label="mean",
                color=[colors[d] for d in DEV], edgecolor="black")
    b2 = ax.bar(x + 0.18, p95s, 0.36, label="p95",
                color=[colors[d] for d in DEV], edgecolor="black", hatch="//", alpha=0.85)
    for rect, v in zip(b1, means):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    for rect, v in zip(b2, p95s):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1000 / 30, color="red", linestyle="--", linewidth=1, label="33ms budget")
    ax.set_xticks(x); ax.set_xticklabels(DEV)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Inference latency")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT / "motiv_single_stream.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------- Figure 3: reversal (NPU-GPU diff bar + sAP vs mAP scatter) ----------

def fig_reversal():
    df = pd.read_csv(RES / "step_f_partA_matrix.csv")
    piv_s = df.pivot(index=["sid", "log_id", "log_size_group"],
                     columns="device", values="sap_5095").reset_index()
    piv_s["delta_npu_gpu_sap"] = piv_s["NPU"] - piv_s["GPU"]
    piv_s = piv_s.sort_values("delta_npu_gpu_sap", ascending=False)

    piv_m = df.pivot(index=["sid", "log_id", "log_size_group"],
                     columns="device", values="map_5095").reset_index()
    piv_m["delta_npu_gpu_map"] = piv_m["NPU"] - piv_m["GPU"]

    merged = piv_s.merge(piv_m[["sid", "delta_npu_gpu_map"]], on="sid")

    colors = {"small-rich": "#3680c4", "medium-mixed": "#888888", "large-rich": "#c4a236"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))
    bars = ax1.bar(range(len(piv_s)), piv_s["delta_npu_gpu_sap"],
                   color=[colors[g] for g in piv_s["log_size_group"]], edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(range(len(piv_s)))
    ax1.set_xticklabels([f"sid={int(s)}\n{g}" for s, g in zip(piv_s["sid"], piv_s["log_size_group"])],
                        fontsize=8)
    ax1.set_ylabel("NPU sAP $-$ GPU sAP")
    ax1.set_title("Per-stream NPU $-$ GPU sAP difference (sorted)\n(positive = NPU-friendly, negative = GPU-preferred)")
    ax1.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()],
               title="size group", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.scatter(merged["delta_npu_gpu_map"], merged["delta_npu_gpu_sap"],
                c=[colors[g] for g in merged["log_size_group"]], s=120, edgecolors="black")
    for _, r in merged.iterrows():
        ax2.annotate(f"sid={int(r['sid'])}", (r["delta_npu_gpu_map"], r["delta_npu_gpu_sap"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax2.axhline(0, color="black", linewidth=0.4)
    ax2.axvline(0, color="black", linewidth=0.4)
    lim = max(abs(merged["delta_npu_gpu_map"].min()), merged["delta_npu_gpu_map"].max(),
              abs(merged["delta_npu_gpu_sap"].min()), merged["delta_npu_gpu_sap"].max()) + 0.005
    ax2.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.4)
    ax2.set_xlabel("NPU $-$ GPU mAP (offline)")
    ax2.set_ylabel("NPU $-$ GPU sAP (streaming)")
    ax2.set_title("sAP vs mAP difference\n(below $y=x$ $\\Rightarrow$ latency penalty dominates)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT / "motiv_reversal.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------- Figure 4: Part B worst-stream sAP comparison ----------

def fig_partB_worst():
    df = pd.read_csv(RES / "step_f_partB_placement.csv")
    BG = ["L1", "L2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.22
    x = np.arange(len(BG))
    for i, name in enumerate(["Naive_allGPU", "SizeAware", "SizeBlindRev"]):
        worst = []
        for bg in BG:
            sub = df[(df.bg_level == bg) & (df.placement_name.str.startswith(name))]
            worst.append(sub.iloc[0]["worst_sap"] if len(sub) else 0)
        axes[0].bar(x + (i - 1) * 0.25, worst, width, label=name)
    oracle_worst = [df[df.bg_level == bg]["worst_sap"].max() for bg in BG]
    axes[0].plot(x, oracle_worst, "k*--", markersize=15, label="Oracle (best worst sAP)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(BG)
    axes[0].set_ylabel("worst-stream sAP")
    axes[0].set_title("Worst-stream sAP by strategy $\\times$ bg level")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    colors = {"Naive_allGPU": "#888888", "SizeAware": "#3680c4", "SizeBlindRev": "#c43b3b"}
    for name in ["Naive_allGPU", "SizeAware", "SizeBlindRev"]:
        for bg in BG:
            sub = df[(df.bg_level == bg) & (df.placement_name.str.startswith(name))]
            if not len(sub): continue
            r = sub.iloc[0]
            marker = "o" if bg == "L1" else "s"
            axes[1].scatter(r["worst_map"], r["worst_sap"], color=colors[name], s=140,
                            edgecolors="black", marker=marker, label=f"{name}/{bg}")
    axes[1].plot([0, 0.2], [0, 0.2], "k--", linewidth=0.5)
    axes[1].set_xlabel("worst-stream mAP (offline)")
    axes[1].set_ylabel("worst-stream sAP (streaming)")
    axes[1].set_title("worst-stream sAP vs mAP\n(distance from $y=x$ = latency penalty)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)

    plt.tight_layout()
    out = OUT / "motiv_partB_worst.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_size_dist()
    fig_single_stream()
    fig_reversal()
    fig_partB_worst()
