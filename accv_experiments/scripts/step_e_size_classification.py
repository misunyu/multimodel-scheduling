"""Step E — Per-log object-size classification.

Classifies each Argoverse-HD val log into one of {small-dominant, medium-mixed,
large-dominant} based on the COCO area buckets of its GT bounding boxes.

Outputs:
  results/step_e_size_classification.csv
  results/figures/step_e_size_distribution.png

Cross-tabulates against step_b_scene_classification.csv (density labels) when
that file exists so we can see whether size and density are independent axes.
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
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
from step0_compare_devices import ANNOT  # noqa: E402

RES = SCRIPT_DIR.parent / "results"
FIG = RES / "figures"
OUT_CSV = RES / "step_e_size_classification.csv"
OUT_FIG = FIG / "step_e_size_distribution.png"
DENSITY_CSV = RES / "step_b_scene_classification.csv"

# COCO area buckets (px²)
SMALL_MAX = 32 ** 2   # 1024
MEDIUM_MAX = 96 ** 2  # 9216

# Initial fixed-threshold rules per task spec
LARGE_DOM_MIN_COUNT_PCT = 0.40
SMALL_DOM_MIN_COUNT_PCT = 0.40


def main():
    with open(ANNOT) as f:
        val = json.load(f)

    # Per-log object aggregation
    ann_by_img = {}
    for a in val["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    log_to_imgs = {}
    for img in val["images"]:
        log_to_imgs.setdefault(img["sid"], []).append(img)

    rows = []
    for sid in sorted(log_to_imgs):
        log_name = val["sequences"][sid]
        imgs = log_to_imgs[sid]
        n_frames = len(imgs)
        # collect every GT bbox in this log
        areas = []
        for img in imgs:
            for a in ann_by_img.get(img["id"], []):
                # Use COCO 'area' field if present, else compute from bbox w*h
                if "area" in a and a["area"] is not None:
                    ar = float(a["area"])
                else:
                    _, _, w, h = a["bbox"]
                    ar = float(w) * float(h)
                areas.append(ar)
        if not areas:
            continue
        areas = np.asarray(areas)
        n_obj = len(areas)

        small_mask = areas < SMALL_MAX
        med_mask = (areas >= SMALL_MAX) & (areas < MEDIUM_MAX)
        large_mask = areas >= MEDIUM_MAX
        n_s, n_m, n_l = int(small_mask.sum()), int(med_mask.sum()), int(large_mask.sum())
        a_s, a_m, a_l = float(areas[small_mask].sum()), float(areas[med_mask].sum()), float(areas[large_mask].sum())
        a_tot = float(areas.sum())

        rows.append({
            "log_id": log_name,
            "sid": sid,
            "n_frames": n_frames,
            "n_objects_total": n_obj,
            "mean_area": round(float(areas.mean()), 1),
            "median_area": round(float(np.median(areas)), 1),
            "pct_small_count":  round(n_s / n_obj, 4),
            "pct_medium_count": round(n_m / n_obj, 4),
            "pct_large_count":  round(n_l / n_obj, 4),
            "pct_small_area":   round(a_s / a_tot, 4),
            "pct_medium_area":  round(a_m / a_tot, 4),
            "pct_large_area":   round(a_l / a_tot, 4),
        })
    df = pd.DataFrame(rows)
    print(f"=== loaded {len(df)} logs ===")
    print(df[["log_id", "n_objects_total", "mean_area", "median_area",
              "pct_small_count", "pct_medium_count", "pct_large_count"]]
          .to_string(index=False))

    # --- initial labelling (fixed thresholds) ---
    def label_fixed(r):
        if r["pct_large_count"] >= LARGE_DOM_MIN_COUNT_PCT:
            return "large-dominant"
        if r["pct_small_count"] >= SMALL_DOM_MIN_COUNT_PCT:
            return "small-dominant"
        return "medium-mixed"

    df["size_label"] = df.apply(label_fixed, axis=1)
    bal = df["size_label"].value_counts().to_dict()
    print(f"\n=== fixed-threshold balance (large≥{LARGE_DOM_MIN_COUNT_PCT}, "
          f"small≥{SMALL_DOM_MIN_COUNT_PCT}) ===")
    print(f"  {bal}")

    threshold_kind = "fixed"
    # Fallback to tertile split when fixed thresholds yield empty class OR
    # any class holds ≥16/24 logs.
    classes_seen = {"large-dominant", "medium-mixed", "small-dominant"}
    empty_classes = classes_seen - set(bal.keys())
    max_class = max(bal.values()) if bal else 0
    if empty_classes or max_class >= 16:
        if empty_classes:
            print(f"  → empty class(es): {empty_classes}. switching to tertile split on pct_large_count")
        else:
            print("  → lopsided. switching to tertile split on pct_large_count")
        q1, q2 = np.quantile(df["pct_large_count"], [1/3, 2/3])
        def label_tert(r):
            if r["pct_large_count"] >= q2:
                return "large-dominant"
            if r["pct_large_count"] < q1:
                return "small-dominant"
            return "medium-mixed"
        df["size_label"] = df.apply(label_tert, axis=1)
        bal = df["size_label"].value_counts().to_dict()
        threshold_kind = f"tertile on pct_large_count (q1={q1:.3f}, q2={q2:.3f})"
        print(f"  → tertile cuts: q1={q1:.3f}, q2={q2:.3f}  balance={bal}")

    print(f"\n=== final classification ({threshold_kind}) ===")
    print(f"  {bal}")

    # Save CSV
    RES.mkdir(parents=True, exist_ok=True)
    df_out = df[["log_id", "n_frames", "n_objects_total",
                 "mean_area", "median_area",
                 "pct_small_count", "pct_medium_count", "pct_large_count",
                 "pct_small_area", "pct_medium_area", "pct_large_area",
                 "size_label"]]
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV} ({len(df_out)} rows)")

    # Stats summary
    print(f"\n=== per-log summary ===")
    print(df[["mean_area", "median_area", "pct_small_count", "pct_large_count",
              "n_objects_total"]].describe().round(2).to_string())

    # Group representatives
    print(f"\n=== group representatives (≤3 per class) ===")
    for label in ["small-dominant", "medium-mixed", "large-dominant"]:
        sub = df[df["size_label"] == label].sort_values("pct_large_count")
        if not len(sub):
            print(f"  {label}: (empty)")
            continue
        print(f"  {label} (n={len(sub)}):")
        for _, r in sub.head(3).iterrows():
            print(f"    sid={int(r['sid']):2d} {r['log_id'][:18]}…  "
                  f"pct_large={r['pct_large_count']:.3f}  "
                  f"mean_area={r['mean_area']:.0f}  n_obj={r['n_objects_total']}")

    # --- visualization (3 subplots) ---
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"small-dominant": "#3680c4", "medium-mixed": "#888888",
              "large-dominant": "#c4a236"}

    # Subplot 1: histogram of pct_large_count with class color
    ax = axes[0]
    bins = np.linspace(0, df["pct_large_count"].max() + 0.02, 14)
    for label in ["small-dominant", "medium-mixed", "large-dominant"]:
        sub = df[df["size_label"] == label]
        ax.hist(sub["pct_large_count"], bins=bins, color=colors[label],
                edgecolor="black", label=f"{label} ({len(sub)})", alpha=0.85)
    ax.set_xlabel("pct of objects with area ≥ 96² (large)")
    ax.set_ylabel("# logs")
    ax.set_title(f"Large-object share by log ({threshold_kind})")
    ax.legend(title="size_label", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Subplot 2: scatter mean_area (log-x) per log, color by class
    ax = axes[1]
    df_s = df.sort_values("mean_area").reset_index(drop=True)
    for label in ["small-dominant", "medium-mixed", "large-dominant"]:
        sub = df_s[df_s["size_label"] == label]
        ax.scatter(sub.index, sub["mean_area"], c=colors[label],
                   edgecolors="black", s=80, label=label)
    ax.axhline(SMALL_MAX, color="gray", linestyle="--", linewidth=0.8, label="32² (small/medium)")
    ax.axhline(MEDIUM_MAX, color="gray", linestyle=":", linewidth=0.8, label="96² (medium/large)")
    ax.set_yscale("log")
    ax.set_xlabel("log index (sorted by mean object area)")
    ax.set_ylabel("mean object area (px², log)")
    ax.set_title("Mean object area per log")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # Subplot 3: stacked bar per log, sorted by pct_large_count desc
    ax = axes[2]
    df_b = df.sort_values("pct_large_count", ascending=False).reset_index(drop=True)
    x = np.arange(len(df_b))
    s = df_b["pct_small_count"].values
    m = df_b["pct_medium_count"].values
    l = df_b["pct_large_count"].values
    ax.bar(x, s, color="#3680c4", label="small")
    ax.bar(x, m, bottom=s, color="#888888", label="medium")
    ax.bar(x, l, bottom=s + m, color="#c4a236", label="large")
    # Mark class regions visually
    for i, row in df_b.iterrows():
        ax.scatter(i, -0.04, marker="s", s=30, color=colors[row["size_label"]],
                   clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}" for s in df_b["sid"]], rotation=90, fontsize=7)
    ax.set_xlabel("log sid (sorted by pct_large desc)")
    ax.set_ylabel("size composition (fraction)")
    ax.set_title("Per-log size composition  (bottom row = class color)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.08, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Step E — Object-size classification (n={len(df)} logs)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=120)
    print(f"\nsaved {OUT_FIG}")

    # --- cross-table with density classification (step_b) ---
    if DENSITY_CSV.exists():
        dens = pd.read_csv(DENSITY_CSV)
        if "scene_label" in dens.columns and "log_id" in dens.columns:
            j = df.merge(dens[["log_id", "scene_label", "mean_objects"]], on="log_id", how="left")
            ct = pd.crosstab(j["size_label"], j["scene_label"]).reindex(
                index=["small-dominant", "medium-mixed", "large-dominant"],
                columns=["Light", "Medium", "Dense"], fill_value=0)
            print(f"\n=== size_label × density (scene_label) cross-table ===")
            print(ct.to_string())
            # Correlation of pct_large vs mean_objects
            corr = j["pct_large_count"].corr(j["mean_objects"])
            print(f"\n  Pearson corr(pct_large_count, mean_objects) = {corr:+.3f}")
            if abs(corr) >= 0.6:
                print("  → strong correlation: size and density carry mostly overlapping info")
            elif abs(corr) >= 0.3:
                print("  → moderate correlation: partially overlapping, size adds some new signal")
            else:
                print("  → weak correlation: size is an independent axis from density")
            # Pivot for which density logs are large/small dominant
            print(f"\n  joint label combos (size × density):")
            j["combo"] = j["size_label"] + " × " + j["scene_label"]
            for combo, n in j["combo"].value_counts().items():
                print(f"    {combo}: {n}")
        else:
            print(f"\n[skip cross-table] step_b CSV lacks expected columns")
    else:
        print(f"\n[skip cross-table] {DENSITY_CSV} not found")


if __name__ == "__main__":
    main()
