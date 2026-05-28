"""Step B — Scene classification by per-frame object density.

Reads Argoverse-HD val.json, computes per-log object-count statistics, and
assigns a scene label (Light / Medium / Dense). Threshold is initially a
hand-picked range; if it produces a lopsided split, falls back to quartile-
based 3-class binning so each class holds ~8 logs (24/3).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_b_scene_classification.csv"
OUT_FIG = SCRIPT_DIR.parent / "figures" / "step_b_scene_distribution.png"

# Initial thresholds (per task spec)
LIGHT_MAX = 10.0   # mean < 10 → Light
DENSE_MIN = 25.0   # mean ≥ 25 → Dense
# else Medium


def main():
    val = json.load(open(ANNOT))
    n_cat = len(val["categories"])
    cat_names = {c["id"]: c["name"] for c in val["categories"]}

    # Index annotations by image_id
    anns_per_img = defaultdict(list)
    for a in val["annotations"]:
        anns_per_img[a["image_id"]].append(a["category_id"])

    # Per-log stats
    per_log_imgs = defaultdict(list)
    for img in val["images"]:
        per_log_imgs[img["sid"]].append(img)
    rows = []
    for sid in sorted(per_log_imgs):
        log_name = val["sequences"][sid]
        imgs = per_log_imgs[sid]
        per_frame_counts = np.array([len(anns_per_img[img["id"]]) for img in imgs])
        all_cats = []
        for img in imgs:
            all_cats.extend(anns_per_img[img["id"]])
        cat_ctr = Counter(all_cats)
        # Shannon class diversity (entropy in nats, normalized to [0,1] by log(N))
        total = sum(cat_ctr.values())
        if total > 0:
            ps = np.array([n / total for n in cat_ctr.values() if n > 0])
            ent = float(-(ps * np.log(ps)).sum())
            diversity = ent / np.log(n_cat) if n_cat > 1 else 0.0
        else:
            diversity = 0.0
        rows.append({
            "log_id": log_name,
            "sid": sid,
            "frame_count": len(imgs),
            "mean_objects": round(float(per_frame_counts.mean()), 3),
            "median_objects": float(np.median(per_frame_counts)),
            "max_objects": int(per_frame_counts.max()),
            "std_objects": round(float(per_frame_counts.std()), 3),
            "class_diversity": round(diversity, 3),
            "scene_label": "",  # filled below
            "dominant_class": cat_names[cat_ctr.most_common(1)[0][0]] if cat_ctr else "",
        })
    df = pd.DataFrame(rows)

    # Initial fixed-threshold classification
    def label_fixed(m):
        if m < LIGHT_MAX:
            return "Light"
        if m >= DENSE_MIN:
            return "Dense"
        return "Medium"

    df["scene_label"] = df["mean_objects"].apply(label_fixed)
    bal_fixed = df["scene_label"].value_counts().to_dict()
    print(f"=== fixed threshold ({LIGHT_MAX=}, {DENSE_MIN=}) ===")
    print(f"  {bal_fixed}")

    # If lopsided (any class >= 16), switch to tertile (3-quantile) split
    n = len(df)
    threshold_kind = "fixed"
    if max(bal_fixed.values()) >= 16:
        print("  → too lopsided; switching to tertile split")
        q1, q2 = np.quantile(df["mean_objects"], [1/3, 2/3])
        print(f"  tertile cuts: 1/3={q1:.2f}, 2/3={q2:.2f}")
        def label_tert(m):
            if m < q1:
                return "Light"
            if m < q2:
                return "Medium"
            return "Dense"
        df["scene_label"] = df["mean_objects"].apply(label_tert)
        threshold_kind = f"tertile (q1={q1:.2f}, q2={q2:.2f})"

    bal = df["scene_label"].value_counts().to_dict()
    print(f"\n=== final classification ({threshold_kind}) ===")
    print(f"  {bal}")

    print(f"\n=== mean_objects summary ===")
    print(df["mean_objects"].describe().round(2).to_string())

    # Save CSV (drop sid + dominant_class from spec columns but keep for ref)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["log_id", "mean_objects", "median_objects", "max_objects",
            "std_objects", "class_diversity", "scene_label",
            "dominant_class", "frame_count", "sid"]
    df[cols].to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV} ({len(df)} rows)")

    # Per-group representative logs
    print(f"\n=== group representatives ===")
    for label in ["Light", "Medium", "Dense"]:
        sub = df[df["scene_label"] == label].sort_values("mean_objects")
        if not len(sub): continue
        print(f"  {label} ({len(sub)} logs, mean range {sub['mean_objects'].min():.1f}-{sub['mean_objects'].max():.1f}):")
        for _, r in sub.head(2).iterrows():
            print(f"     {r['log_id'][:36]}  mean={r['mean_objects']}  dominant={r['dominant_class']}")
        if len(sub) > 2:
            print(f"     … and {len(sub)-2} more")

    # Visualization
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Light": "#3b9c4d", "Medium": "#e9b341", "Dense": "#c43b3b"}
    bins = np.linspace(df["mean_objects"].min() - 0.5,
                       df["mean_objects"].max() + 0.5, 25)
    for label in ["Light", "Medium", "Dense"]:
        sub = df[df["scene_label"] == label]
        ax.hist(sub["mean_objects"], bins=bins, color=colors[label],
                label=f"{label} ({len(sub)})", edgecolor="black", alpha=0.85)
    ax.set_xlabel("mean object count per frame")
    ax.set_ylabel("# logs")
    ax.set_title(f"Step B — scene density distribution (n={n} val logs, {threshold_kind})")
    ax.legend(title="scene")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=120)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
