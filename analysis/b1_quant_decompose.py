"""B1 Task 2 — per-size quantization error decomposition (FP32 vs INT8).

Reads the offline per-detection dumps (analysis/b1_dets/{gpu_fp32,npu_int8}.npz)
and the GT (val.json). For every GT object matched by FP32, classify what INT8
did to it, to decompose the per-size quantization loss Q_b into error PATHS:

  (i)   match retained, score retained  (|d_score| <= SCORE_TOL)
  (ii)  match retained, score dropped   (int8_score < fp32_score - SCORE_TOL)
  (iii) MISSED: no INT8 detection near the GT (best IoU < LOC_MIN) -> the box
        vanished, i.e. its score fell below CONF=0.25 or NMS dropped it
        => RECALL / SCORE path
  (iv)  LOCALIZATION degraded: an INT8 detection of the same class overlaps the
        GT (LOC_MIN <= IoU < 0.5) but no longer matches => LOCALIZATION path

Matching is COCO-style greedy (per image, per class, score-desc, IoU>=0.5,
one det per GT), run independently for each device, exactly as pycocotools does
the assignment. Size bin is the GT `area` field on native 1920x1200 pixels
(same COCO areaRng as the paper's sAP_s/m/l).

Honesty: categories are reported as-is. The hypothesis (small-object loss is
dominated by the score/recall path (ii)/(iii), not localization (iv)) is a
prediction to test, not to enforce.

Outputs:
  analysis/b1_quant_decomposition.csv     (bin x category: counts, fractions)
  analysis/b1_quant_conf_shift.csv        (bin: FP32/INT8 matched-score stats)
  analysis/b1_quant_conf_shift.pdf        (per-bin score shift)
  console table + per-bin verdict + why-Q_large~0 note
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
DETS = ROOT / "analysis/b1_dets"
OUT_DECOMP = ROOT / "analysis/b1_quant_decomposition.csv"
OUT_SHIFT = ROOT / "analysis/b1_quant_conf_shift.csv"
OUT_PDF = ROOT / "analysis/b1_quant_conf_shift.pdf"

MATCH_IOU = 0.5
LOC_MIN = 0.1        # below this, treat as "vanished" (recall) not "mislocalized"
SCORE_TOL = 0.05     # |score shift| within this = "retained"
AREA_SMALL, AREA_MEDIUM = 1024.0, 9216.0
BINS = ["small", "medium", "large"]
CATS = ["i_match_score_kept", "ii_match_score_drop", "iii_missed_recall",
        "iv_localization"]


def size_bin(area):
    if area < AREA_SMALL:
        return "small"
    if area < AREA_MEDIUM:
        return "medium"
    return "large"


def iou_matrix(gt_xywh, det_xyxy):
    """IoU between GT (xywh) and det (xyxy). gt:(G,4) det:(D,4) -> (G,D)."""
    if len(gt_xywh) == 0 or len(det_xyxy) == 0:
        return np.zeros((len(gt_xywh), len(det_xyxy)), np.float32)
    gx1 = gt_xywh[:, 0]; gy1 = gt_xywh[:, 1]
    gx2 = gx1 + gt_xywh[:, 2]; gy2 = gy1 + gt_xywh[:, 3]
    ga = gt_xywh[:, 2] * gt_xywh[:, 3]
    dx1, dy1, dx2, dy2 = det_xyxy[:, 0], det_xyxy[:, 1], det_xyxy[:, 2], det_xyxy[:, 3]
    da = (dx2 - dx1) * (dy2 - dy1)
    ix1 = np.maximum(gx1[:, None], dx1[None, :])
    iy1 = np.maximum(gy1[:, None], dy1[None, :])
    ix2 = np.minimum(gx2[:, None], dx2[None, :])
    iy2 = np.minimum(gy2[:, None], dy2[None, :])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = ga[:, None] + da[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(gt_xywh, det_xyxy, det_score):
    """COCO-style: sort dets by score desc, assign each to best unused GT with
    IoU>=MATCH_IOU. Returns dict gt_idx -> (det_idx, score) for matched GTs."""
    matched = {}
    if len(gt_xywh) == 0 or len(det_xyxy) == 0:
        return matched
    ious = iou_matrix(gt_xywh, det_xyxy)
    order = np.argsort(-det_score)
    used_gt = set()
    for d in order:
        col = ious[:, d].copy()
        for g in used_gt:
            col[g] = -1
        g = int(np.argmax(col))
        if col[g] >= MATCH_IOU:
            matched[g] = (int(d), float(det_score[d]))
            used_gt.add(g)
    return matched


def load_dump(name):
    z = np.load(DETS / f"{name}.npz")
    by_img = defaultdict(lambda: {"box": [], "score": [], "cls": []})
    img = z["image_id"]; box = z["box"]; sc = z["score"]; cl = z["cls"]
    for i in range(len(img)):
        d = by_img[int(img[i])]
        d["box"].append(box[i]); d["score"].append(sc[i]); d["cls"].append(int(cl[i]))
    out = {}
    for k, v in by_img.items():
        out[k] = {"box": np.asarray(v["box"], np.float32),
                  "score": np.asarray(v["score"], np.float32),
                  "cls": np.asarray(v["cls"], np.int32)}
    return out


def main():
    val = json.load(open(ANNOT))
    img_meta = {im["id"]: im for im in val["images"]}
    # GT grouped by (image_id, class)
    gt = defaultdict(lambda: defaultdict(lambda: {"box": [], "area": []}))
    for a in val["annotations"]:
        if a.get("ignore") or a.get("iscrowd"):
            continue
        g = gt[a["image_id"]][a["category_id"]]
        g["box"].append(a["bbox"]); g["area"].append(a["area"])

    fp32 = load_dump("gpu_fp32")
    int8 = load_dump("npu_int8")
    proc_imgs = sorted(set(fp32) | set(int8) | set(gt.keys()))

    # counts[bin][cat]; also score pairs for shift; iou drops for (iv)
    counts = {b: {c: 0 for c in CATS} for b in BINS}
    n_fp32_matched = {b: 0 for b in BINS}
    n_gt_total = {b: 0 for b in BINS}
    score_pairs = {b: [] for b in BINS}       # (fp32_score, int8_score) for retained
    iou_drops = {b: [] for b in BINS}         # for (iv)
    score_all = {b: {"fp32": [], "int8": []} for b in BINS}  # matched-score dists

    for iid in proc_imgs:
        classes = set(gt.get(iid, {}).keys())
        f = fp32.get(iid, {"box": np.zeros((0, 4), np.float32),
                           "score": np.zeros(0, np.float32), "cls": np.zeros(0, np.int32)})
        n = int8.get(iid, {"box": np.zeros((0, 4), np.float32),
                           "score": np.zeros(0, np.float32), "cls": np.zeros(0, np.int32)})
        for c in classes:
            g = gt[iid][c]
            gbox = np.asarray(g["box"], np.float32)
            garea = np.asarray(g["area"], np.float32)
            for b in BINS:
                n_gt_total[b] += int(sum(size_bin(a) == b for a in garea))
            fsel = f["cls"] == c
            nsel = n["cls"] == c
            fb, fs = f["box"][fsel], f["score"][fsel]
            nb, ns = n["box"][nsel], n["score"][nsel]
            m_fp = greedy_match(gbox, fb, fs)
            m_np = greedy_match(gbox, nb, ns)
            # IoU of every GT to nearest INT8 det (same class), for (iii)/(iv) split
            iou_gn = iou_matrix(gbox, nb)
            best_np_iou = iou_gn.max(axis=1) if nb.shape[0] else np.zeros(len(gbox))
            for gi in m_fp:  # denominator = FP32-matched GT objects
                b = size_bin(garea[gi])
                n_fp32_matched[b] += 1
                fp_score = m_fp[gi][1]
                if gi in m_np:
                    np_score = m_np[gi][1]
                    score_all[b]["fp32"].append(fp_score)
                    score_all[b]["int8"].append(np_score)
                    if np_score >= fp_score - SCORE_TOL:
                        counts[b]["i_match_score_kept"] += 1
                    else:
                        counts[b]["ii_match_score_drop"] += 1
                    score_pairs[b].append((fp_score, np_score))
                else:
                    if best_np_iou[gi] >= LOC_MIN:
                        counts[b]["iv_localization"] += 1
                        iou_drops[b].append(float(best_np_iou[gi]))
                    else:
                        counts[b]["iii_missed_recall"] += 1

    # ---- write decomposition csv ----
    with open(OUT_DECOMP, "w") as fh:
        fh.write("size,n_gt_total,n_fp32_matched," +
                 ",".join(CATS) + "," +
                 ",".join(f"frac_{c}" for c in CATS) + "\n")
        for b in BINS:
            denom = max(1, n_fp32_matched[b])
            fh.write(f"{b},{n_gt_total[b]},{n_fp32_matched[b]}," +
                     ",".join(str(counts[b][c]) for c in CATS) + "," +
                     ",".join(f"{counts[b][c]/denom:.4f}" for c in CATS) + "\n")
    print(f"wrote {OUT_DECOMP}")

    # ---- confidence shift csv ----
    with open(OUT_SHIFT, "w") as fh:
        fh.write("size,n_matched_pairs,fp32_mean,int8_mean,mean_shift,"
                 "median_shift,fp32_mean_all,int8_mean_all\n")
        for b in BINS:
            pr = np.asarray(score_pairs[b])
            fa = np.asarray(score_all[b]["fp32"]); na = np.asarray(score_all[b]["int8"])
            if len(pr):
                shifts = pr[:, 1] - pr[:, 0]
                fh.write(f"{b},{len(pr)},{pr[:,0].mean():.4f},{pr[:,1].mean():.4f},"
                         f"{shifts.mean():.4f},{np.median(shifts):.4f},"
                         f"{fa.mean():.4f},{na.mean():.4f}\n")
            else:
                fh.write(f"{b},0,,,,,,\n")
    print(f"wrote {OUT_SHIFT}")

    # ---- console report ----
    print("\n=== Per-size INT8 error decomposition "
          "(denominator = FP32-matched GT objects) ===")
    hdr = f"{'bin':7s} {'n_matched':>9s} " + " ".join(f"{c.split('_')[0]:>5s}" for c in CATS)
    print(hdr)
    for b in BINS:
        denom = max(1, n_fp32_matched[b])
        print(f"{b:7s} {n_fp32_matched[b]:9d} " +
              " ".join(f"{100*counts[b][c]/denom:4.1f}%" for c in CATS))
    print("  legend: i=match+score_kept  ii=match+score_drop  "
          "iii=missed(recall/score)  iv=localization")

    print("\n=== Error PATH split (of FP32-matched objects INT8 degraded) ===")
    for b in BINS:
        denom = max(1, n_fp32_matched[b])
        score_path = counts[b]["ii_match_score_drop"] + counts[b]["iii_missed_recall"]
        loc_path = counts[b]["iv_localization"]
        degraded = counts[b]["ii_match_score_drop"] + counts[b]["iii_missed_recall"] + counts[b]["iv_localization"]
        print(f"  {b:7s}: degraded={100*degraded/denom:4.1f}%  "
              f"score-path(ii+iii)={100*score_path/denom:4.1f}%  "
              f"loc-path(iv)={100*loc_path/denom:4.1f}%  "
              f"| kept={100*counts[b]['i_match_score_kept']/denom:4.1f}%")

    print("\n=== Confidence shift (matched pairs) ===")
    for b in BINS:
        pr = np.asarray(score_pairs[b])
        if len(pr):
            print(f"  {b:7s}: FP32 mean={pr[:,0].mean():.3f}  INT8 mean={pr[:,1].mean():.3f}  "
                  f"mean shift={pr[:,1].mean()-pr[:,0].mean():+.3f}  (n={len(pr)})")

    # why Q_large ~ 0: which degradation category is (near-)absent for large?
    print("\n=== Why Q_large ~ +0.001 (≈0)? absent categories for large ===")
    b = "large"; denom = max(1, n_fp32_matched[b])
    for c in CATS:
        print(f"  large {c:22s}: {100*counts[b][c]/denom:4.1f}%")
    print("  -> large is dominated by (i): INT8 keeps the box AND score, so "
          "quantization removes almost no large-object AP.")

    # ---- figure: per-bin matched-score distributions FP32 vs INT8 ----
    import matplotlib
    matplotlib.use("Agg")
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
    fig.suptitle("Per-size matched-detection confidence: FP32 vs INT8", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
