"""Step F Part A — Per-stream per-device sAP/mAP/latency matrix.

For each (log, device) pair: run single foreground stream + L1 background,
measure both sAP (streaming-paired) and mAP (offline-paired with input frame
ignoring time) and latency.

Outputs:
  results/step_f_partA_matrix.csv          (24 rows)
  results/step_f_partA_strategy.json       (Part B strategy definitions)
  results/figures/step_f_partA_difference.png
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelCPU, FGModelGPU, FPS, WARMUP_FRAMES,
                            fg_worker, get_npu_model, load_split_for_sid,
                            load_val, per_stream_sap,
                            preload_background_models, preload_npu_instances,
                            start_background, stop_background)

# 8 candidate logs (from step_e classification)
LOGS_A = [
    {"sid": 2,  "group": "small-rich"},
    {"sid": 22, "group": "small-rich"},
    {"sid": 13, "group": "small-rich"},
    {"sid": 17, "group": "medium-mixed"},
    {"sid": 8,  "group": "medium-mixed"},
    {"sid": 10, "group": "medium-mixed"},
    {"sid": 3,  "group": "large-rich"},
    {"sid": 21, "group": "large-rich"},
]
DEVICES = ["CPU", "GPU", "NPU"]
BG = "L1"
PARTB_SIDS = [2, 22, 3, 21]  # 2 small-rich + 2 large-rich for Part B

OUT_CSV = SCRIPT_DIR.parent / "results" / "step_f_partA_matrix.csv"
OUT_STRATEGY_JSON = SCRIPT_DIR.parent / "results" / "step_f_partA_strategy.json"
FIG_DIFF = SCRIPT_DIR.parent / "results" / "figures" / "step_f_partA_difference.png"
STEP_E_CSV = SCRIPT_DIR.parent / "results" / "step_e_size_classification.csv"


def per_stream_map_offline(split, result, warmup=WARMUP_FRAMES):
    """Offline mAP: pair each detection with its INPUT frame (no streaming
    time penalty). Eval over warmup-skipped imgIds, so frames the detector
    skipped contribute zero recall (still penalized) — this isolates the
    latency penalty when compared with sAP."""
    imgs = split["imgs"]
    coco_gt = split["coco_gt"]
    out = {"map_5095": 0.0, "map_50": 0.0, "map_s": 0.0, "map_m": 0.0, "map_l": 0.0}
    ccf = []
    for tidx in range(len(result["results"])):
        fidx = result["input_fidx"][tidx]
        if fidx < warmup or fidx >= len(imgs):
            continue
        img = imgs[fidx]
        bb, sc, lb = result["results"][tidx]
        for k in range(len(bb)):
            x1, y1, x2, y2 = bb[k]
            ccf.append({"image_id": int(img["id"]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc[k]), "category_id": int(lb[k])})
    if not ccf:
        return out
    img_ids_eval = sorted({img["id"] for img in imgs[warmup:]})
    coco_dt = coco_gt.loadRes(ccf)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = img_ids_eval
    e.evaluate(); e.accumulate(); e.summarize()
    out.update({"map_5095": float(e.stats[0]), "map_50": float(e.stats[1]),
                "map_s": float(e.stats[3]), "map_m": float(e.stats[4]),
                "map_l": float(e.stats[5])})
    return out


def measure_cell(device, split, model):
    result = defaultdict(list)
    stop = threading.Event()
    bg_stops, bg_threads = start_background(BG)
    t0 = time.time()
    fg_worker(0, device, split, model, result, stop)
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)

    sap = per_stream_sap(split, result)
    mp = per_stream_map_offline(split, result)
    return {
        "sap_5095": round(sap["sap_5095"], 4),
        "sap_50":   round(sap["sap_50"], 4),
        "sap_s":    round(sap["sap_small"], 4),
        "sap_m":    round(sap["sap_medium"], 4),
        "sap_l":    round(sap["sap_large"], 4),
        "map_5095": round(mp["map_5095"], 4),
        "map_50":   round(mp["map_50"], 4),
        "map_s":    round(mp["map_s"], 4),
        "map_m":    round(mp["map_m"], 4),
        "map_l":    round(mp["map_l"], 4),
        "latency_mean":     round(sap["infer_mean_ms"], 2),
        "latency_p95":      round(sap["infer_p95_ms"], 2),
        "latency_e2e_mean": round(sap["eff_mean_ms"], 2),
        "frame_skip_pct":   round(sap["frame_skip_pct"], 1),
        "n_processed":      sap["n_processed"],
        "wall_sec":         round(wall, 2),
    }


def main():
    # Resolve sid → log_id directly from val.json (step_e CSV lacks sid column).
    val = load_val()
    sid_to = {sid: {"log_id": name} for sid, name in enumerate(val["sequences"])}
    if STEP_E_CSV.exists():
        sm = pd.read_csv(STEP_E_CSV)
        name_to_label = dict(zip(sm["log_id"], sm["size_label"]))
        for sid, meta in sid_to.items():
            meta["size_label"] = name_to_label.get(meta["log_id"], "?")

    print("[partA] preloading bg + models (ORT first, then torch, then NPU)...")
    t0 = time.time()
    preload_background_models(max_level=BG)
    gpu_model = FGModelGPU()
    cpu_model = FGModelCPU()
    preload_npu_instances(1, infer_mode="multi")
    npu_model = get_npu_model(0)
    print(f"[partA] preload done in {time.time()-t0:.1f}s")

    rows = []
    n_total = len(LOGS_A) * len(DEVICES)
    cell = 0
    t_all = time.time()
    for log_spec in LOGS_A:
        sid = log_spec["sid"]
        group = log_spec["group"]
        log_id = sid_to[sid]["log_id"]
        split = load_split_for_sid(val, sid)
        for dev in DEVICES:
            cell += 1
            print(f"\n[A {cell:2d}/{n_total}] sid={sid:2d} ({group:<13s}) dev={dev}")
            model = {"CPU": cpu_model, "GPU": gpu_model, "NPU": npu_model}[dev]
            t0 = time.time()
            m = measure_cell(dev, split, model)
            print(f"   sAP={m['sap_5095']:.3f}  mAP={m['map_5095']:.3f}  "
                  f"infer={m['latency_mean']:5.1f}ms  skip={m['frame_skip_pct']:5.1f}%  "
                  f"({time.time()-t0:.1f}s)")
            rows.append({"log_id": log_id, "sid": sid, "log_size_group": group,
                         "device": dev, **m})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[partA] csv: {OUT_CSV}  ({len(df)} rows, wall={time.time()-t_all:.1f}s)")

    # Analysis: NPU−GPU sAP difference per stream
    piv_s = df.pivot(index=["sid", "log_id", "log_size_group"],
                     columns="device", values="sap_5095").reset_index()
    piv_s["delta_npu_gpu_sap"] = piv_s["NPU"] - piv_s["GPU"]
    piv_s = piv_s.sort_values("delta_npu_gpu_sap", ascending=False)
    print("\n=== NPU − GPU sAP difference per stream (higher = NPU-friendlier) ===")
    print(piv_s[["sid", "log_size_group", "CPU", "GPU", "NPU", "delta_npu_gpu_sap"]].round(4).to_string(index=False))

    piv_m = df.pivot(index=["sid", "log_id", "log_size_group"],
                     columns="device", values="map_5095").reset_index()
    piv_m["delta_npu_gpu_map"] = piv_m["NPU"] - piv_m["GPU"]
    print("\n=== NPU − GPU mAP difference per stream ===")
    print(piv_m[["sid", "log_size_group", "GPU", "NPU", "delta_npu_gpu_map"]].round(4).sort_values("delta_npu_gpu_map", ascending=False).to_string(index=False))

    # Size-group summary
    print("\n=== group-mean NPU − GPU sAP diff ===")
    print(piv_s.groupby("log_size_group")["delta_npu_gpu_sap"].agg(["mean","min","max","count"]).round(4).to_string())

    # Part B strategy: rank streams in PARTB_SIDS by NPU-friendliness, top-2 → NPU
    sub = piv_s[piv_s["sid"].isin(PARTB_SIDS)].copy().sort_values("delta_npu_gpu_sap", ascending=False)
    npu_friendly_2 = sub.iloc[:2]["sid"].tolist()
    gpu_friendly_2 = sub.iloc[2:]["sid"].tolist()
    print(f"\n=== Part B 4-stream strategy derivation ===")
    print(f"  PARTB sids (fixed order for stream slot 0..3): {PARTB_SIDS}")
    print(f"  → among these, NPU-friendly top-2: sids {npu_friendly_2}")
    print(f"  → GPU-keeping bot-2:               sids {gpu_friendly_2}")

    def make_placement(npu_sids):
        return ["NPU" if sid in set(npu_sids) else "GPU" for sid in PARTB_SIDS]

    strategies = {
        "Naive_allGPU":   ["GPU"] * 4,
        "SizeAware":      make_placement(npu_friendly_2),
        "SizeBlindRev":   make_placement(gpu_friendly_2),
    }
    print(f"\n  placements (slot order matches PARTB_SIDS):")
    for name, plc in strategies.items():
        print(f"    {name:14s}: {plc}")

    # Save strategy JSON for Part B
    OUT_STRATEGY_JSON.write_text(json.dumps({
        "partb_sids": PARTB_SIDS,
        "partb_log_ids": [sid_to[s]["log_id"] for s in PARTB_SIDS],
        "partb_size_groups": [next(L["group"] for L in LOGS_A if L["sid"] == s) for s in PARTB_SIDS],
        "strategies": strategies,
        "npu_friendly_ranking": [
            {"sid": int(r["sid"]), "log_size_group": r["log_size_group"],
             "GPU_sap": float(r["GPU"]), "NPU_sap": float(r["NPU"]), "CPU_sap": float(r["CPU"]),
             "delta_npu_gpu": float(r["delta_npu_gpu_sap"])}
            for _, r in piv_s.iterrows()
        ],
    }, indent=2))
    print(f"\nsaved {OUT_STRATEGY_JSON}")

    # Figure
    FIG_DIFF.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))
    colors = {"small-rich": "#3680c4", "medium-mixed": "#888888", "large-rich": "#c4a236"}

    bars = ax1.bar(range(len(piv_s)), piv_s["delta_npu_gpu_sap"],
                   color=[colors[g] for g in piv_s["log_size_group"]], edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(range(len(piv_s)))
    ax1.set_xticklabels([f"sid={int(s)}\n{g}" for s, g in zip(piv_s["sid"], piv_s["log_size_group"])],
                        fontsize=8)
    ax1.set_ylabel("NPU sAP − GPU sAP")
    ax1.set_title("Per-stream NPU − GPU sAP difference (sorted)\n(positive = NPU-friendly, negative = GPU-preferred)")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()],
               title="size group", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Side by side: sAP vs mAP diff scatter
    merged = piv_s.merge(piv_m[["sid", "delta_npu_gpu_map"]], on="sid")
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
    ax2.set_xlabel("NPU − GPU  mAP (offline)")
    ax2.set_ylabel("NPU − GPU  sAP (streaming)")
    ax2.set_title("sAP vs mAP difference\n(below y=x line ⇒ latency penalty larger for sAP)")
    ax2.grid(alpha=0.3)

    plt.suptitle("Step F Part A — per-stream per-device sAP/mAP matrix  (bg L1)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIFF, dpi=120)
    print(f"saved {FIG_DIFF}")


if __name__ == "__main__":
    main()
