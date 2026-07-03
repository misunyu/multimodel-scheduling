"""rev7 §5–6 re-measurement sweep on current SDK + pinned legacy v11s binary.

The pinned mxq is the same one rev6 used for v11s (legacy b2441f9d,
mode-flexible). All measurements below run YOLOv11s exclusively, on the
current `mblt_model_zoo`, so §5–6 anchors land on the same SDK / binary as
§7-s and are reproducible from a fresh clone.

Tables produced (one CSV each):
  results/rev7_single_stream.csv   tab:single-stream  (CPU/GPU/NPU L0 over 24 logs)
  results/rev7_per_class.csv       tab:per-class      (24 logs × {GPU,NPU} × 8 classes)
  results/rev7_partA.csv           tab:partA          (8 anchor sids × {GPU,NPU} × L1_light)
  results/rev7_main_cmp.csv        tab:main-comparison (N=4 size-diverse × bg{L1l,L2lm} × strategies + oracle)
  results/rev7_schedule.csv        tab:schedule-shift  (N∈{4,8} × bg{L0,L1l,L1h,L2lm,L3vlm} × {Naive,SizeAware,AllNPU})
  results/rev7_capacity.csv        tab:capacity        (N∈{2,3,4,5,6,8} × bg L1_light × {Naive,SizeAware,SizeBlindRev,AllNPU})
  results/rev7_natural.csv         tab:natural         (Comp A/B/C × N∈{4,8} × bg{L1l,L1h} × strategies + oracle samples)
  results/rev7_decomp.csv          tab:decomposition   (loss attribution path for the worst stream at N=8)
  results/rev7_reversal_gate.csv   STEP 2 gate

Reuses rev6 (`p1r6_*_yolo11s.csv`) wherever the cell key matches exactly:
  (sid, device, bg_level, infer_mode, placement). Otherwise measures fresh.

Manifest: `results/manifest_rev7.json` — idempotent resume.

Run order: tables 1, 2, 3 first (cheap, sanity), then 6, 5, 7 (multi-stream),
then 4 + 8 last (use earlier rows as references).
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
import traceback
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _step_d_common as cm
from _step_d_common import (FGModelCPU, FGModelGPU, fg_worker, load_split_for_sid,
                            load_val, per_stream_sap,
                            preload_background_models, stop_background)
from step_f_partA_matrix import per_stream_map_offline
from step_h2_robustness import start_bg_custom

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_rev7.json"

PIN = json.loads((RES / "rev7_pin.json").read_text())
MXQ_PATH = PIN["mxq_path"]
DET_NAME = "yolo11s"
DET_PARAMS_M = 9.4

# Per the rev6 SDK probe, the legacy mxq is mode-flexible: global8 for
# single-stream phases (matches Table 1 protocol), single for multi-stream
# (global8 cannot host >1 NPU instance).
MODE_SS = "global8"
MODE_MS = "single"

# anchor sid set for tab:partA (same 8 sids step_f Part A used)
PARTA_SIDS = [2, 3, 8, 10, 13, 17, 21, 22]
COMP_A_N4 = [2, 22, 3, 21]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]
COMP_B_N8 = [17, 8, 10, 11, 2, 13, 3, 14]
COMP_B_N4 = [17, 8, 10, 11]
COMP_C_N8 = [15, 19, 23, 0, 1, 20, 5, 7]
COMP_C_N4 = [15, 0, 5, 7]

LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}


# ============================ manifest ============================

def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except Exception: pass
    return {}

def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))

def mark_done(manifest, table, key):
    manifest.setdefault(table, {})[key] = {"status": "done", "ts": int(time.time())}
    save_manifest(manifest)

def is_done(manifest, table, key):
    return manifest.get(table, {}).get(key, {}).get("status") == "done"


# ============================ NPU loaders ============================

class FGModelGPUGeneric:
    def __init__(self, pt_name="yolo11s.pt"):
        import os
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO
        from step0_compare_devices import IMG_SIZE, CONF, IOU
        self.IMG_SIZE = IMG_SIZE; self.CONF = CONF; self.IOU = IOU
        self.m = YOLO(pt_name)
        self.m.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                       imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False, device="cuda")
    def predict(self, img_path):
        return self.m.predict(img_path, imgsz=self.IMG_SIZE, conf=self.CONF, iou=self.IOU,
                              verbose=False, device="cuda")[0]


_NPU = {}

def load_npu(mode, n=1):
    key = (MXQ_PATH, mode)
    insts = _NPU.setdefault(key, [])
    while len(insts) < n:
        from mblt_model_zoo import vision as mv
        cls = mv.YOLO11s
        m = cls(local_path=MXQ_PATH, infer_mode=mode, product="aries")
        from step0_compare_devices import CONF, IOU
        dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
        x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
        insts.append(m)
    return insts[:n]

def dispose_npu(mode=None):
    keys = list(_NPU.keys()) if mode is None else [k for k in _NPU if k[1] == mode]
    for k in keys:
        for m in _NPU[k]:
            try: m.dispose()
            except Exception: pass
        del _NPU[k]

def set_active_npu(insts):
    cm._NPU_INSTANCES.clear()
    cm._NPU_INSTANCES.extend(insts)


# ============================ measurement primitives ============================

def measure_one(sid, split, device, model, bg):
    res = defaultdict(list)
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg)
    t0 = time.time()
    fg_worker(0, device, split, model, res, stop)
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    sap = per_stream_sap(split, res)
    mp = per_stream_map_offline(split, res)
    eff = float(np.mean(res["eff_ms"])) if res["eff_ms"] else 0.0
    return {
        "sap_5095": sap["sap_5095"], "sap_50": sap["sap_50"],
        "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
        "map_5095": mp["map_5095"], "map_s": mp["map_s"],
        "map_m": mp["map_m"], "map_l": mp["map_l"],
        "latency_mean": sap["infer_mean_ms"], "eff_e2e_mean": eff,
        "frame_skip_pct": sap["frame_skip_pct"], "wall_sec": wall,
    }, res


def measure_multistream(sids, splits, placement, gpu_models, npu_models, bg):
    n = len(placement)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg)
    gi = ni = 0; threads = []
    for i, dev in enumerate(placement):
        if dev == "GPU":
            model = gpu_models[gi]; gi += 1
        elif dev == "NPU":
            model = npu_models[ni]; ni += 1
        else:
            raise ValueError(dev)
        t = threading.Thread(target=fg_worker,
                             args=(i, dev, splits[i], model, results[i], stop),
                             daemon=True)
        threads.append(t)
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    per_stream = []
    for i in range(n):
        sap = per_stream_sap(splits[i], results[i])
        mp = per_stream_map_offline(splits[i], results[i])
        per_stream.append({
            "stream_id": i, "sid": sids[i], "device": placement[i],
            "sap_5095": sap["sap_5095"], "sap_s": sap["sap_small"],
            "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
            "map_5095": mp["map_5095"], "map_s": mp["map_s"],
            "map_m": mp["map_m"], "map_l": mp["map_l"],
            "latency_mean": sap["infer_mean_ms"],
            "frame_skip_pct": sap["frame_skip_pct"],
        })
    saps = [s["sap_5095"] for s in per_stream]
    maps = [s["map_5095"] for s in per_stream]
    return {
        "n_streams": n, "placement_spec": json.dumps(placement),
        "mean_sap": float(np.mean(saps)), "worst_sap": float(np.min(saps)),
        "mean_map": float(np.mean(maps)), "worst_map": float(np.min(maps)),
        "per_stream": per_stream, "wall_sec": wall,
    }


# ============================ Table 1 — single-stream ============================

def table1_single_stream(val, manifest, gpu, npu_ss):
    """tab:single-stream: 24 logs × {CPU, GPU, NPU} × L0."""
    out = RES / "rev7_single_stream.csv"
    cols = ["sid", "log_id", "device",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l",
            "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    cpu_model = None  # lazy
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for device in ["CPU", "GPU", "NPU"]:
            key = f"{device}_sid{sid}"
            if is_done(manifest, "table1", key):
                continue
            try:
                if device == "GPU":
                    model = gpu
                elif device == "NPU":
                    set_active_npu(npu_ss)
                    model = npu_ss[0]
                else:
                    if cpu_model is None:
                        cpu_model = FGModelCPU()
                    model = cpu_model
                m, _ = measure_one(sid, split, device, model, "L0")
            except Exception as e:
                print(f"  table1 FAIL {key}: {type(e).__name__}: {e}")
                continue
            row = {"sid": sid, "log_id": val["sequences"][sid], "device": device,
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            mark_done(manifest, "table1", key)
            print(f"  table1 {device} sid={sid:>2d}: sap={m['sap_5095']:.3f} ({m['wall_sec']:.1f}s)")


# ============================ Table 2 — per-class ============================

def table2_per_class(val, manifest, gpu, npu_ss):
    """tab:per-class: collect detections per (sid, device) then COCOeval per class.
    We write the per-class aggregate; per-sid raw detections are kept in memory."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    out = RES / "rev7_per_class.csv"
    cols = ["device", "cat_id", "cat_name", "ap_5095", "ap_50", "n_gt", "n_dets"]
    if is_done(manifest, "table2", "done"):
        return
    # Build per-device detection bag across 24 logs
    cats = sorted(val["categories"], key=lambda c: c["id"])
    cat_names = {c["id"]: c["name"] for c in cats}

    # Pre-load splits
    splits = {sid: load_split_for_sid(val, sid) for sid in range(len(val["sequences"]))}

    def collect(device, model):
        all_ccf = []
        for sid in range(len(val["sequences"])):
            split = splits[sid]
            res = defaultdict(list); stop = threading.Event()
            bg_stops, bg_threads = start_bg_custom("L0")
            fg_worker(0, device, split, model, res, stop)
            stop_background(bg_stops, bg_threads)
            # post-warmup detections paired to input frame
            from _step_d_common import WARMUP_FRAMES
            from step0_compare_devices import FPS
            imgs = split["imgs"]
            ts = res["timestamps"]
            inp = res["input_fidx"]
            r = res["results"]
            tidx_p1 = 0
            for ii, img in enumerate(imgs):
                t_gt = ii / FPS
                while tidx_p1 < len(ts) and ts[tidx_p1] <= t_gt:
                    tidx_p1 += 1
                if ii < WARMUP_FRAMES: continue
                if tidx_p1 == 0: continue
                ti = tidx_p1 - 1
                bb, sc, lb = r[ti]
                for k in range(len(bb)):
                    x1, y1, x2, y2 = bb[k]
                    all_ccf.append({
                        "image_id": int(img["id"]),
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(sc[k]), "category_id": int(lb[k]),
                    })
            print(f"  per-class collect {device} sid={sid:>2d}: dets={len(all_ccf)} so far")
        return all_ccf

    # Global GT (post-warmup) over 24 logs
    from _step_d_common import WARMUP_FRAMES
    imgs_keep = []
    for sid in range(len(val["sequences"])):
        sub = sorted([i for i in val["images"] if i["sid"] == sid], key=lambda x: x["fid"])
        imgs_keep.extend(sub[WARMUP_FRAMES:])
    img_ids = {i["id"] for i in imgs_keep}
    anns = [a for a in val["annotations"] if a["image_id"] in img_ids]
    coco_gt = COCO()
    coco_gt.dataset = {"info": {}, "licenses": [], "categories": val["categories"],
                       "images": imgs_keep, "annotations": anns}
    coco_gt.createIndex()
    img_ids_eval = sorted(img_ids)

    rows = []
    for device, model in [("GPU", gpu), ("NPU", npu_ss[0])]:
        if device == "NPU":
            set_active_npu(npu_ss)
        all_ccf = collect(device, model)
        print(f"  per-class device={device}: total dets={len(all_ccf)}")
        for cat in cats:
            cid = cat["id"]
            n_gt = sum(1 for a in coco_gt.dataset["annotations"] if a["category_id"] == cid)
            sub_dets = [d for d in all_ccf if d["category_id"] == cid]
            if not sub_dets:
                rows.append({"device": device, "cat_id": cid, "cat_name": cat["name"],
                              "ap_5095": 0.0, "ap_50": 0.0, "n_gt": n_gt, "n_dets": 0})
                continue
            coco_dt = coco_gt.loadRes(sub_dets)
            e = COCOeval(coco_gt, coco_dt, "bbox")
            e.params.imgIds = img_ids_eval; e.params.catIds = [cid]
            e.evaluate(); e.accumulate(); e.summarize()
            rows.append({"device": device, "cat_id": cid, "cat_name": cat["name"],
                          "ap_5095": round(float(e.stats[0]), 4),
                          "ap_50":   round(float(e.stats[1]), 4),
                          "n_gt": n_gt, "n_dets": len(sub_dets)})

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    mark_done(manifest, "table2", "done")
    print(f"saved {out} ({len(rows)} rows)")


# ============================ Table 3 — partA ============================

def table3_partA(val, manifest, gpu, npu_ss):
    """tab:partA: 8 anchor sids × {GPU,NPU} × bg L1_light."""
    out = RES / "rev7_partA.csv"
    cols = ["sid", "log_id", "device", "bg_level",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l",
            "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    for sid in PARTA_SIDS:
        split = load_split_for_sid(val, sid)
        for device in ["GPU", "NPU"]:
            key = f"{device}_sid{sid}"
            if is_done(manifest, "table3", key):
                continue
            try:
                if device == "NPU":
                    set_active_npu(npu_ss)
                    model = npu_ss[0]
                else:
                    model = gpu
                m, _ = measure_one(sid, split, device, model, "L1_light")
            except Exception as e:
                print(f"  partA FAIL {key}: {type(e).__name__}: {e}")
                continue
            row = {"sid": sid, "log_id": val["sequences"][sid], "device": device,
                   "bg_level": "L1_light",
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            mark_done(manifest, "table3", key)
            print(f"  partA {device} L1 sid={sid:>2d}: sap={m['sap_5095']:.3f}")


# ============================ multi-stream tables (4–7) ============================

def _ms_csv_init(path, n_max=8):
    cols = ["table", "tag", "n_streams", "bg_level", "placement_name", "placement_spec",
            "mean_sap", "worst_sap", "mean_map", "worst_map", "wall_sec"]
    for i in range(n_max):
        for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "map", "skip"]:
            cols.append(f"s{i}_{k}")
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    return cols


def _ms_write(path, cols, table_tag, tag, agg, bg, name):
    spec = json.loads(agg["placement_spec"])
    row = {"table": table_tag, "tag": tag, "n_streams": agg["n_streams"],
           "bg_level": bg, "placement_name": name, "placement_spec": agg["placement_spec"],
           "mean_sap": round(agg["mean_sap"], 4),
           "worst_sap": round(agg["worst_sap"], 4),
           "mean_map": round(agg["mean_map"], 4),
           "worst_map": round(agg["worst_map"], 4),
           "wall_sec": round(agg["wall_sec"], 1)}
    for s in agg["per_stream"]:
        i = s["stream_id"]
        row[f"s{i}_dev"] = s["device"]; row[f"s{i}_sid"] = s["sid"]
        row[f"s{i}_sap"] = round(s["sap_5095"], 4)
        row[f"s{i}_sap_s"] = round(s["sap_s"], 4)
        row[f"s{i}_sap_m"] = round(s["sap_m"], 4)
        row[f"s{i}_sap_l"] = round(s["sap_l"], 4)
        row[f"s{i}_map"] = round(s["map_5095"], 4)
        row[f"s{i}_skip"] = round(s["frame_skip_pct"], 1)
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writerow(row)


def _make_placement(sids, npu_set):
    return ["NPU" if s in set(npu_set) else "GPU" for s in sids]


def table4_main_cmp(val, manifest, gpu_models, npu_ms_4):
    """tab:main-comparison: N=4 sids 2,22,3,21 × bg{L1_light,L2_lm} × strategies + oracle."""
    out = RES / "rev7_main_cmp.csv"
    cols = _ms_csv_init(out, n_max=8)
    sids = COMP_A_N4
    splits = [load_split_for_sid(val, s) for s in sids]
    placements = {
        "Naive_allGPU":  ["GPU","GPU","GPU","GPU"],
        "SizeAware":     ["NPU","NPU","GPU","GPU"],  # small (2,22) → NPU
        "SizeBlindRev":  ["GPU","GPU","NPU","NPU"],  # large (3,21) → NPU
        "AllNPU":        ["NPU","NPU","NPU","NPU"],
    }
    # exhaustive 16-placement oracle pool
    for k in range(5):
        for npu_pos in combinations(range(4), k):
            spec = ["NPU" if i in npu_pos else "GPU" for i in range(4)]
            name = "Oracle_k{}_{}".format(k, "_".join(map(str, npu_pos)) if npu_pos else "none")
            if not any(s == spec for s in placements.values()):
                placements[name] = spec
    for bg in ["L1_light", "L2_lm"]:
        for name, placement in placements.items():
            key = f"{bg}_{name}"
            if is_done(manifest, "table4", key):
                continue
            n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
            try:
                agg = measure_multistream(sids, splits, placement,
                                          gpu_models[:n_gpu] if n_gpu else [],
                                          npu_ms_4[:n_npu] if n_npu else [],
                                          bg)
            except Exception as e:
                print(f"  main_cmp FAIL {key}: {type(e).__name__}: {e}")
                continue
            _ms_write(out, cols, "main-comparison", bg, agg, bg, name)
            mark_done(manifest, "table4", key)
            print(f"  main_cmp {bg} {name[:24]:<24s}: worst={agg['worst_sap']:.4f} mean={agg['mean_sap']:.4f}")


def table5_schedule_shift(val, manifest, gpu_models, npu_ms_8):
    """tab:schedule-shift: N∈{4,8} × bg{L0,L1_light,L1_heavy,L2_lm,L3_vlm} × {Naive,SizeAware,AllNPU}."""
    out = RES / "rev7_schedule.csv"
    cols = _ms_csv_init(out, n_max=8)
    for N in [4, 8]:
        sids = COMP_A_N4 if N == 4 else COMP_A_N8
        splits = [load_split_for_sid(val, s) for s in sids]
        # SizeAware spec by N: small → NPU first
        if N == 4:
            sa = ["NPU","NPU","GPU","GPU"]
            named = {"Naive_allGPU": ["GPU"]*4, "SizeAware": sa, "AllNPU": ["NPU"]*4}
        else:
            sa = ["NPU","NPU","NPU","NPU","GPU","GPU","GPU","GPU"]
            named = {"Naive_allGPU": ["GPU"]*8, "SizeAware_NPU4": sa,
                     "SizeAware_NPU6": ["NPU","NPU","NPU","NPU","NPU","NPU","GPU","GPU"],
                     "AllNPU": ["NPU"]*8}
        for bg in LADDER:
            for name, placement in named.items():
                key = f"N{N}_{bg}_{name}"
                if is_done(manifest, "table5", key):
                    continue
                n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
                if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models):
                    print(f"  schedule SKIP {key}: not enough engines"); continue
                try:
                    agg = measure_multistream(sids, splits, placement,
                                              gpu_models[:n_gpu],
                                              npu_ms_8[:n_npu], bg)
                except Exception as e:
                    print(f"  schedule FAIL {key}: {type(e).__name__}: {e}")
                    continue
                _ms_write(out, cols, "schedule-shift", f"N={N}", agg, bg, name)
                mark_done(manifest, "table5", key)
                print(f"  schedule N={N} {bg:<9s} {name[:20]:<20s}: worst={agg['worst_sap']:.4f}")


def table6_capacity(val, manifest, gpu_models, npu_ms_8):
    """tab:capacity: N∈{2,3,4,5,6,8} × bg L1_light × {Naive,SizeAware,SizeBlindRev,AllNPU}."""
    out = RES / "rev7_capacity.csv"
    cols = _ms_csv_init(out, n_max=8)
    COMP = {
        2: ([2],            [3]),
        3: ([2, 22],        [3]),
        4: ([2, 22],        [3, 21]),
        5: ([2, 22, 13],    [3, 21]),
        6: ([2, 22, 13],    [3, 21, 14]),
        8: ([2, 22, 13, 16],[3, 21, 14, 4]),
    }
    bg = "L1_light"
    for N in [2, 3, 4, 5, 6, 8]:
        small, large = COMP[N]
        sids = small + large
        splits = [load_split_for_sid(val, s) for s in sids]
        named = {
            "Naive_allGPU":  ["GPU"] * N,
            "SizeAware":     ["NPU" if s in small else "GPU" for s in sids],
            "SizeBlindRev":  ["NPU" if s in large else "GPU" for s in sids],
            "AllNPU":        ["NPU"] * N,
        }
        for name, placement in named.items():
            key = f"N{N}_{name}"
            if is_done(manifest, "table6", key):
                continue
            n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
            if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models):
                print(f"  capacity SKIP {key}: not enough engines"); continue
            try:
                agg = measure_multistream(sids, splits, placement,
                                          gpu_models[:n_gpu] if n_gpu else [],
                                          npu_ms_8[:n_npu] if n_npu else [], bg)
            except Exception as e:
                print(f"  capacity FAIL {key}: {type(e).__name__}: {e}")
                continue
            _ms_write(out, cols, "capacity", f"N={N}", agg, bg, name)
            mark_done(manifest, "table6", key)
            print(f"  capacity N={N} {name[:18]:<18s}: worst={agg['worst_sap']:.4f}")


def table7_natural(val, manifest, gpu_models, npu_ms_8):
    """tab:natural: 3 compositions × N∈{4,8} × bg{L1_light,L1_heavy} × strategies + oracle samples."""
    out = RES / "rev7_natural.csv"
    cols = _ms_csv_init(out, n_max=8)
    COMPS = {
        "A_baseline": {4: COMP_A_N4, 8: COMP_A_N8},
        "B_medium_mixed": {4: COMP_B_N4, 8: COMP_B_N8},
        "C_diverse": {4: COMP_C_N4, 8: COMP_C_N8},
    }
    # per-group sid -> "small"/"medium"/"large"
    SIZE_LABEL = {}
    for grp, sids in SIZE_GROUPS.items():
        for s in sids:
            SIZE_LABEL[s] = {"small-rich":"small","medium-mixed":"medium","large-rich":"large"}[grp]

    def named_placements(comp, N, sids):
        if N == 4:
            small_priority = [s for s in sids if SIZE_LABEL.get(s) == "small"]
            sa = ["NPU" if s in small_priority[:2] else "GPU" for s in sids]
            large_priority = [s for s in sids if SIZE_LABEL.get(s) == "large"]
            sbr = ["NPU" if s in large_priority[:2] else "GPU" for s in sids]
            return {"Naive_allGPU": ["GPU"]*4, "SizeAware": sa,
                    "SizeBlindRev": sbr, "AllNPU": ["NPU"]*4}
        else:
            small_priority = [s for s in sids if SIZE_LABEL.get(s) == "small"]
            large_priority = [s for s in sids if SIZE_LABEL.get(s) == "large"]
            sa = ["NPU" if s in small_priority[:4] else "GPU" for s in sids]
            sbr = ["NPU" if s in large_priority[:4] else "GPU" for s in sids]
            return {"Naive_allGPU": ["GPU"]*8, "SizeAware_NPU4": sa,
                    "SizeAware_NPU6": ["NPU"]*6 + ["GPU"]*2,
                    "SizeBlindRev_NPU4": sbr, "AllNPU": ["NPU"]*8}

    for comp, ns_map in COMPS.items():
        for N in [4, 8]:
            sids = ns_map[N]
            splits = [load_split_for_sid(val, s) for s in sids]
            for bg in ["L1_light", "L1_heavy"]:
                placements = named_placements(comp, N, sids)
                # add oracle samples (k=0..min(N,4) for small grids)
                import random
                rng = random.Random(42 + N + hash(comp) & 0xffff)
                for k in range(min(N, 4) + 1):
                    candidates = list(combinations(range(N), k))
                    sample = rng.sample(candidates, min(3, len(candidates))) if k > 0 else candidates
                    for pos in sample:
                        spec = ["NPU" if i in pos else "GPU" for i in range(N)]
                        nm = f"Oracle_k{k}_p{'_'.join(map(str,pos)) if pos else 'none'}"
                        if nm not in placements:
                            placements[nm] = spec
                for name, placement in placements.items():
                    key = f"{comp}_N{N}_{bg}_{name}"
                    if is_done(manifest, "table7", key):
                        continue
                    n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
                    if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models):
                        print(f"  natural SKIP {key}"); continue
                    try:
                        agg = measure_multistream(sids, splits, placement,
                                                  gpu_models[:n_gpu] if n_gpu else [],
                                                  npu_ms_8[:n_npu] if n_npu else [], bg)
                    except Exception as e:
                        print(f"  natural FAIL {key}: {type(e).__name__}: {e}")
                        continue
                    _ms_write(out, cols, "natural", f"{comp}_N{N}", agg, bg, name)
                    mark_done(manifest, "table7", key)
                    print(f"  natural {comp} N={N} {bg:<9s} {name[:20]:<20s}: worst={agg['worst_sap']:.4f}")


def table8_decomposition(val, manifest, gpu, npu_ss, npu_ms_8):
    """tab:decomposition: loss attribution for the worst camera at N=8.
    We need: GPU L0 N=1, GPU L1_light N=1, NPU L0 N=1, NPU L1_light N=1,
             AllNPU N=8 bg L0, AllNPU N=8 bg L1_light, Naive_allGPU N=8 bg L1_light
    over Composition A. The single-stream rows can reuse Table 1 (sid-by-sid) and
    Table 3 (partA NPU L1_light), or measure fresh."""
    out = RES / "rev7_decomp.csv"
    cols = ["row_kind", "bg_level", "infer_mode", "placement_name",
            "n_streams", "mean_sap", "worst_sap", "worst_sid",
            "mean_map", "worst_map", "wall_sec"]
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    sids = COMP_A_N8
    splits = [load_split_for_sid(val, s) for s in sids]

    # 1) N=1 GPU L0 for each sid (use Table 1 GPU rows)
    df_t1 = pd.read_csv(RES / "rev7_single_stream.csv") if (RES / "rev7_single_stream.csv").exists() else None
    if df_t1 is not None:
        for sid in sids:
            key = f"N1_GPU_L0_sid{sid}"
            if is_done(manifest, "table8", key): continue
            row = df_t1[(df_t1.sid == sid) & (df_t1.device == "GPU")]
            if not len(row): continue
            r = row.iloc[0]
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow({
                    "row_kind": "N1_GPU_L0", "bg_level": "L0", "infer_mode": "",
                    "placement_name": "GPU_single", "n_streams": 1,
                    "mean_sap": round(float(r["sap_5095"]), 4),
                    "worst_sap": round(float(r["sap_5095"]), 4),
                    "worst_sid": sid,
                    "mean_map": round(float(r["map_5095"]), 4),
                    "worst_map": round(float(r["map_5095"]), 4),
                    "wall_sec": float(r["wall_sec"]),
                })
            mark_done(manifest, "table8", key)
    # 2) N=1 NPU L0 (reuse Table 1 NPU rows)
    if df_t1 is not None:
        for sid in sids:
            key = f"N1_NPU_L0_sid{sid}"
            if is_done(manifest, "table8", key): continue
            row = df_t1[(df_t1.sid == sid) & (df_t1.device == "NPU")]
            if not len(row): continue
            r = row.iloc[0]
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow({
                    "row_kind": "N1_NPU_L0", "bg_level": "L0", "infer_mode": MODE_SS,
                    "placement_name": "NPU_single", "n_streams": 1,
                    "mean_sap": round(float(r["sap_5095"]), 4),
                    "worst_sap": round(float(r["sap_5095"]), 4),
                    "worst_sid": sid,
                    "mean_map": round(float(r["map_5095"]), 4),
                    "worst_map": round(float(r["map_5095"]), 4),
                    "wall_sec": float(r["wall_sec"]),
                })
            mark_done(manifest, "table8", key)
    # 3) N=1 GPU L1_light per sid
    for sid in sids:
        key = f"N1_GPU_L1_sid{sid}"
        if is_done(manifest, "table8", key): continue
        split = load_split_for_sid(val, sid)
        m, _ = measure_one(sid, split, "GPU", gpu, "L1_light")
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writerow({
                "row_kind": "N1_GPU_L1", "bg_level": "L1_light", "infer_mode": "",
                "placement_name": "GPU_single", "n_streams": 1,
                "mean_sap": round(m["sap_5095"], 4), "worst_sap": round(m["sap_5095"], 4),
                "worst_sid": sid, "mean_map": round(m["map_5095"], 4),
                "worst_map": round(m["map_5095"], 4), "wall_sec": round(m["wall_sec"], 1),
            })
        mark_done(manifest, "table8", key)
    # 4) N=1 NPU L1_light per sid
    set_active_npu(npu_ss)
    for sid in sids:
        key = f"N1_NPU_L1_sid{sid}"
        if is_done(manifest, "table8", key): continue
        split = load_split_for_sid(val, sid)
        m, _ = measure_one(sid, split, "NPU", npu_ss[0], "L1_light")
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writerow({
                "row_kind": "N1_NPU_L1", "bg_level": "L1_light", "infer_mode": MODE_SS,
                "placement_name": "NPU_single", "n_streams": 1,
                "mean_sap": round(m["sap_5095"], 4), "worst_sap": round(m["sap_5095"], 4),
                "worst_sid": sid, "mean_map": round(m["map_5095"], 4),
                "worst_map": round(m["map_5095"], 4), "wall_sec": round(m["wall_sec"], 1),
            })
        mark_done(manifest, "table8", key)
    # 5) N=8 AllNPU L0 (Composition A) — needs 8 single-mode NPU instances
    # 6) N=8 AllNPU L1_light, 7) N=8 Naive L1_light — measured implicitly by table5
    # Pull table5 rows after table5 completes.


# ============================ orchestrator ============================

def reversal_gate_from(main_cmp_csv):
    if not Path(main_cmp_csv).exists():
        return None
    df = pd.read_csv(main_cmp_csv)
    sub = df[(df.n_streams == 4) & (df.bg_level == "L1_light")]
    sa = sub[sub.placement_name == "SizeAware"]
    sbr = sub[sub.placement_name == "SizeBlindRev"]
    if not len(sa) or not len(sbr): return None
    wg = float(sa.iloc[0]["worst_sap"]) - float(sbr.iloc[0]["worst_sap"])
    mg = float(sa.iloc[0]["mean_sap"]) - float(sbr.iloc[0]["mean_sap"])
    return {"worst_gain": wg, "mean_gain": mg, "inverts": wg > 0,
            "sa_worst": float(sa.iloc[0]["worst_sap"]),
            "sbr_worst": float(sbr.iloc[0]["worst_sap"])}


def main():
    manifest = load_manifest()
    print("[rev7] preloading bg max=L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[rev7] bg preload {time.time()-t0:.1f}s")
    val = load_val()
    print(f"[rev7] {len(val['sequences'])} logs, pinned mxq sha256[:12]={PIN['mxq_sha256'][:12]}")

    # Single-stream phase resources
    print("[rev7] loading GPU + 1 NPU global8…")
    gpu = FGModelGPUGeneric()
    npu_ss = load_npu(MODE_SS, n=1)

    t_all = time.time()
    # Table 1 (single-stream)
    print("\n--- TABLE 1 (single-stream) ---")
    table1_single_stream(val, manifest, gpu, npu_ss)
    # Table 2 (per-class)
    print("\n--- TABLE 2 (per-class) ---")
    table2_per_class(val, manifest, gpu, npu_ss)
    # Table 3 (partA)
    print("\n--- TABLE 3 (partA) ---")
    table3_partA(val, manifest, gpu, npu_ss)

    # Dispose global8 engines; need single-mode for multistream
    print("\n[rev7] swapping NPU mode: global8 -> single (multistream phase)…")
    dispose_npu(MODE_SS)
    npu_ms = load_npu(MODE_MS, n=8)
    print(f"[rev7] {len(npu_ms)} single-mode NPU engines loaded")

    # Need 8 GPU instances for high-N tables
    print("[rev7] loading 8 GPU instances total…")
    gpu_models = [gpu] + [FGModelGPUGeneric() for _ in range(7)]
    print(f"[rev7] {len(gpu_models)} GPU instances ready")

    # Tables 4-7 (multistream)
    print("\n--- TABLE 4 (main-comparison) ---")
    table4_main_cmp(val, manifest, gpu_models, npu_ms[:4])

    # STEP 2 REVERSAL GATE — write a CSV either way
    gate = reversal_gate_from(RES / "rev7_main_cmp.csv")
    with open(RES / "rev7_reversal_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sa_worst","sbr_worst","worst_gain","mean_gain","inverts","pass"])
        w.writeheader()
        if gate:
            w.writerow({**gate, "pass": bool(gate["inverts"])})
            print(f"\n[rev7] STEP 2 REVERSAL GATE: {'PASS' if gate['inverts'] else 'FAIL'} "
                  f"(worst gain {gate['worst_gain']:+.4f})")
        else:
            w.writerow({"pass": "TBD"})
            print(f"\n[rev7] STEP 2 REVERSAL GATE: TBD (main_cmp incomplete)")

    print("\n--- TABLE 6 (capacity) ---")
    table6_capacity(val, manifest, gpu_models, npu_ms)
    print("\n--- TABLE 5 (schedule-shift) ---")
    table5_schedule_shift(val, manifest, gpu_models, npu_ms)
    print("\n--- TABLE 7 (natural composition) ---")
    table7_natural(val, manifest, gpu_models, npu_ms)
    # Table 8 (decomposition): single-stream rows reuse Table 1; N=8 rows reuse Table 5
    print("\n--- TABLE 8 (decomposition) ---")
    # Swap NPU back to global8 for any N=1 NPU rows we still need
    dispose_npu(MODE_MS)
    npu_ss = load_npu(MODE_SS, n=1)
    table8_decomposition(val, manifest, gpu, npu_ss, None)

    print(f"\n[rev7] sweep complete. wall_total={time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
