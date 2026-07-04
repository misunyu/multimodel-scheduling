"""B1 Task 1 — Temporal IoU decay (GT-only, detector-independent).

Mechanism probe for the *staleness* size-dependency reported in Table 3
(L_b: small -0.001, medium -0.030, large -0.098). Hypothesis: a stale
detection is the box from an EARLIER frame pasted onto a later frame; the
self-IoU of the same tracked object across a delay delta measures how fast
that staleness erodes localization. Near (=large) objects have larger
ego-motion pixel displacement than far (=small) objects, so large-object
self-IoU should decay fastest -> largest staleness loss.

Pure annotation analysis: no detector, no GPU. Uses per-(sid, track) box
identity from Argoverse-HD val.json. Size bin is taken from the frame-t box
using the SAME COCO areaRng thresholds pycocotools COCOeval uses on native
1920x1200 pixels (small<1024, 1024<=medium<9216, large>=9216), so bins are
consistent with the paper's sAP_s/m/l.

Outputs:
  analysis/b1_temporal_iou_decay.csv   (bin x delta x stats)
  analysis/b1_iou_decay.pdf            (mean self-IoU + IoU<0.5 fraction panels)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
OUT_CSV = ROOT / "analysis/b1_temporal_iou_decay.csv"
OUT_PDF = ROOT / "analysis/b1_iou_decay.pdf"

FPS = 30.0
DELTAS = [1, 2, 3, 4, 5]                       # frames -> 33.3 .. 166.7 ms
# COCO areaRng on native pixels (matches pycocotools COCOeval defaults).
AREA_SMALL = 1024.0       # 32**2
AREA_MEDIUM = 9216.0      # 96**2
BINS = ["small", "medium", "large"]
# COCO AP@[.50:.05:.95] IoU thresholds — sAP averages AP over these 10.
COCO_THRS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# Baseline GPU sAP per bin (A_b) from analysis/c2_per_size_losses.csv, used
# only to turn per-bin IoU-retention into an sAP-loss proxy for ordering.
A_B = {"small": 0.0159, "medium": 0.1839, "large": 0.4768}
# Measured staleness loss L_b (Table 3) for the ordering comparison.
L_B_MEASURED = {"small": -0.0013, "medium": -0.0295, "large": -0.0979}


def size_bin(area):
    if area < AREA_SMALL:
        return "small"
    if area < AREA_MEDIUM:
        return "medium"
    return "large"


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def main():
    val = json.load(open(ANNOT))
    img = {im["id"]: im for im in val["images"]}

    # (sid, track) -> {fid: bbox}  and frame-t area for binning
    tracks = defaultdict(dict)
    for a in val["annotations"]:
        if a.get("ignore") or a.get("iscrowd"):
            continue
        m = img[a["image_id"]]
        tracks[(m["sid"], a["track"])][m["fid"]] = (a["bbox"], a["area"])

    # accumulate per (bin, delta)
    ious = {(b, d): [] for b in BINS for d in DELTAS}
    disp = {(b, d): [] for b in BINS for d in DELTAS}

    for (sid, tid), frames in tracks.items():
        fids = frames
        for f, (box_t, area_t) in fids.items():
            b = size_bin(area_t)
            ct = center(box_t)
            for d in DELTAS:
                nxt = fids.get(f + d)
                if nxt is None:
                    continue
                box_td = nxt[0]
                ious[(b, d)].append(iou_xywh(box_t, box_td))
                cd = center(box_td)
                disp[(b, d)].append(float(np.hypot(cd[0] - ct[0], cd[1] - ct[1])))

    # write CSV
    rows = []
    header = ["size", "delta_frames", "delta_ms", "n_pairs",
              "mean_iou", "median_iou", "frac_iou_lt_0.5", "mean_center_disp_px"]
    for b in BINS:
        for d in DELTAS:
            arr = np.asarray(ious[(b, d)])
            dsp = np.asarray(disp[(b, d)])
            n = len(arr)
            rows.append([
                b, d, round(1000.0 * d / FPS, 1), n,
                round(float(arr.mean()), 4) if n else "",
                round(float(np.median(arr)), 4) if n else "",
                round(float((arr < 0.5).mean()), 4) if n else "",
                round(float(dsp.mean()), 2) if n else "",
            ])
    with open(OUT_CSV, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {OUT_CSV}")

    # ---- console summary + verdict ----
    print("\nMean self-IoU (bin x delta):")
    print(f"{'delta_ms':>9s} " + " ".join(f"{b:>8s}" for b in BINS))
    for d in DELTAS:
        line = f"{1000.0*d/FPS:9.1f} "
        for b in BINS:
            arr = np.asarray(ious[(b, d)])
            line += f" {arr.mean():7.3f}" if len(arr) else "     nan"
        print(line)

    print("\nFraction self-IoU < 0.5 (drops below COCO match threshold):")
    print(f"{'delta_ms':>9s} " + " ".join(f"{b:>8s}" for b in BINS))
    for d in DELTAS:
        line = f"{1000.0*d/FPS:9.1f} "
        for b in BINS:
            arr = np.asarray(ious[(b, d)])
            line += f" {(arr<0.5).mean():7.3f}" if len(arr) else "     nan"
        print(line)

    # verdict: does large decay fastest? use IoU drop from delta=1..5
    def mean_iou(b, d):
        arr = np.asarray(ious[(b, d)])
        return arr.mean() if len(arr) else np.nan
    drop = {b: mean_iou(b, 1) - mean_iou(b, 5) for b in BINS}
    frac5 = {b: (np.asarray(ious[(b, 5)]) < 0.5).mean() for b in BINS}
    print("\nIoU drop (delta1 -> delta5):", {b: round(drop[b], 3) for b in BINS})
    print("frac IoU<0.5 at delta5     :", {b: round(float(frac5[b]), 3) for b in BINS})
    order_ok = drop["large"] > drop["medium"] > drop["small"]
    print(f"\nVERDICT: large decays fastest? {order_ok} "
          f"(order large>med>small on IoU drop)")
    print(f"  measured L_b order was large(-0.098) > medium(-0.030) > small(-0.001)")

    # ---- reconciliation: sAP-style IoU-threshold retention ------------------
    # sAP averages AP over 10 IoU thresholds. A stale detection reuses box_t as
    # the "detection" for GT box_{t+d}; its match quality == self-IoU. The
    # fraction of objects still matched at threshold tau is P(self-IoU >= tau).
    # Averaging that over the 10 COCO thresholds mirrors how sAP aggregates, and
    # weights the HIGH-IoU regime where large objects live and thresholds are
    # dense. staleness sAP-loss proxy(bin, d) = A_b * (1 - retention).
    print("\n" + "=" * 68)
    print("RECONCILIATION via COCO IoU-threshold retention (mirrors sAP@[.5:.95])")
    print("retention = mean_tau P(self-IoU >= tau), tau in .50..0.95")

    def retention(b, d):
        arr = np.asarray(ious[(b, d)])
        if not len(arr):
            return np.nan
        return float(np.mean([(arr >= t).mean() for t in COCO_THRS]))

    recon_rows = [["size", "delta_frames", "delta_ms", "retention",
                   "one_minus_ret", "A_b", "sap_loss_proxy"]]
    print(f"\n{'delta_ms':>9s}  " + "  ".join(f"{b:>18s}" for b in BINS)
          + "   [retention | sAP-loss proxy]")
    for d in DELTAS:
        cells = []
        for b in BINS:
            r = retention(b, d)
            proxy = A_B[b] * (1.0 - r)
            recon_rows.append([b, d, round(1000.0 * d / FPS, 1),
                               round(r, 4), round(1.0 - r, 4),
                               A_B[b], round(proxy, 4)])
            cells.append(f"{r:5.3f} | -{proxy:5.3f}")
        print(f"{1000.0*d/FPS:9.1f}  " + "  ".join(f"{c:>18s}" for c in cells))

    recon_csv = ROOT / "analysis/b1_staleness_sap_proxy.csv"
    with open(recon_csv, "w") as f:
        for r in recon_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {recon_csv}")

    # ordering of the proxy vs measured L_b, per delta
    print("\nOrdering check — does sAP-loss proxy reproduce measured L_b order")
    print(f"  measured L_b: large {L_B_MEASURED['large']:+.4f} > "
          f"medium {L_B_MEASURED['medium']:+.4f} > small {L_B_MEASURED['small']:+.4f}")
    for d in DELTAS:
        pr = {b: A_B[b] * (1.0 - retention(b, d)) for b in BINS}
        ok = pr["large"] > pr["medium"] > pr["small"]
        print(f"  delta={d} ({1000*d/FPS:.0f}ms): proxy large -{pr['large']:.4f} "
              f"med -{pr['medium']:.4f} small -{pr['small']:.4f}  "
              f"order large>med>small? {ok}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"small": "#4477AA", "medium": "#EE6677", "large": "#228833"}
    xs_ms = [1000.0 * d / FPS for d in DELTAS]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.5))

    for b in BINS:
        y = [mean_iou(b, d) for d in DELTAS]
        ax1.plot(xs_ms, y, "o-", color=colors[b], label=b, lw=1.8, ms=5)
    ax1.set_xlabel("temporal offset δ (ms)")
    ax1.set_ylabel("mean self-IoU")
    ax1.set_title("(a) Same-object IoU vs delay")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="GT size bin", frameon=False)

    for b in BINS:
        y = [(np.asarray(ious[(b, d)]) < 0.5).mean() for d in DELTAS]
        ax2.plot(xs_ms, y, "s-", color=colors[b], label=b, lw=1.8, ms=5)
    ax2.set_xlabel("temporal offset δ (ms)")
    ax2.set_ylabel("fraction with self-IoU < 0.5")
    ax2.set_title("(b) Fraction below COCO match threshold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="GT size bin", frameon=False)

    # panel (c): sAP-loss proxy = A_b * (1 - threshold retention) vs measured L_b
    for b in BINS:
        y = [A_B[b] * (1.0 - retention(b, d)) * 100 for d in DELTAS]
        ax3.plot(xs_ms, y, "^-", color=colors[b], label=f"{b} (proxy)", lw=1.8, ms=5)
        ax3.axhline(-L_B_MEASURED[b] * 100, color=colors[b], ls=":", lw=1.3, alpha=0.8)
    ax3.set_xlabel("temporal offset δ (ms)")
    ax3.set_ylabel(r"staleness sAP-loss proxy (%)")
    ax3.set_title("(c) Proxy vs measured $L_b$ (dotted)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 20)
    ax3.legend(frameon=False, fontsize=8)

    # secondary axis: frames
    for ax in (ax1, ax2, ax3):
        secax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * FPS / 1000.0, lambda x: x * 1000.0 / FPS))
        secax.set_xlabel("δ (frames @30fps)")

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
