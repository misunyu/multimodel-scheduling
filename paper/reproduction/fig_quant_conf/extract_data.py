"""fig:quant_conf — STAGE 1 of 2: data extraction (needs Argoverse-HD val.json).

This is the val.json/npz-dependent half of the original single-file generator,
split out so that the figure itself (STAGE 2, generate.py) can be reproduced
WITHOUT the Argoverse-HD dataset. The data-processing/calculation logic below is
copied VERBATIM from the original generate.py; only the I/O has been split:

  * INPUTS  (raw, dataset-dependent):
      - data/b1_dets/{gpu_fp32,npu_int8}.npz   (offline per-detection dumps; bundled)
      - Argoverse-HD val.json                  (GT annotations; NOT bundled, see README)
  * OUTPUT (minimal aggregated intermediate for the figure; bundled, license-safe):
      - data/quant_conf_matched_scores.csv     (size, fp32_score, int8_score per
        FP32&INT8-matched GT object — model-output confidence scores only; contains
        NO GT bbox/image/annotation content)
  * OUTPUT (diagnostic aggregates, unchanged from the original):
      - b1_quant_decomposition.csv
      - b1_quant_conf_shift.csv
      + the console decomposition/verdict report

STAGE 2 (generate.py) reads ONLY data/quant_conf_matched_scores.csv and draws the
identical figure. Re-run this stage only when regenerating the intermediate from
the raw dumps + val.json.

--- original module docstring (behaviour unchanged) ---
B1 Task 2 — per-size quantization error decomposition (FP32 vs INT8).
For every GT object matched by FP32, classify what INT8 did to it, decomposing
the per-size quantization loss Q_b into error PATHS (i)-(iv). Matching is
COCO-style greedy (per image, per class, score-desc, IoU>=0.5, one det per GT),
run independently for each device. Size bin is the GT `area` field on native
1920x1200 pixels (same COCO areaRng as the paper's sAP_s/m/l).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # [reproduction] path adjusted: repo root is 4 levels up
HERE = Path(__file__).resolve().parent                        # [reproduction] path adjusted: this reproduction item dir
# [reproduction] path adjusted: external Argoverse-HD GT annotation kept at original repo path (not copied, ~39MB bulky external dataset). See README for how to obtain it.
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
DETS = HERE / "data/b1_dets"                                  # [reproduction] path adjusted: detection dumps copied into ./data/b1_dets
OUT_DECOMP = HERE / "b1_quant_decomposition.csv"              # [reproduction] path adjusted: local diagnostic output
OUT_SHIFT = HERE / "b1_quant_conf_shift.csv"                  # [reproduction] path adjusted: local diagnostic output
OUT_MATCHED = HERE / "data/quant_conf_matched_scores.csv"     # [reproduction] NEW: minimal aggregated intermediate consumed by generate.py

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
    if not ANNOT.exists():
        raise SystemExit(
            f"[extract_data] Argoverse-HD annotation not found:\n  {ANNOT}\n"
            "This STAGE-1 script needs val.json. To only reproduce the FIGURE, run\n"
            "generate.py instead (it uses the bundled data/quant_conf_matched_scores.csv).\n"
            "See README.md for the Argoverse-HD download URL and where to place val.json.")

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

    # ---- NEW: minimal aggregated intermediate for the figure ----
    # Per-bin matched-detection confidence scores (score_all == the FP32&INT8
    # matched pairs the figure histograms). Stored at %.10g so the doubles the
    # figure consumes round-trip exactly (float32-origin scores need <=9 sig
    # digits); no GT bbox/image/annotation content is written. This is the ONLY
    # file generate.py needs.
    OUT_MATCHED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MATCHED, "w") as fh:
        fh.write("size,fp32_score,int8_score\n")
        for b in BINS:
            fps = score_all[b]["fp32"]; nps = score_all[b]["int8"]
            for fp_s, np_s in zip(fps, nps):
                fh.write(f"{b},{fp_s:.10g},{np_s:.10g}\n")
    print(f"wrote {OUT_MATCHED}")

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
    print("\n[extract_data] done. Now run: python generate.py")


if __name__ == "__main__":
    main()
