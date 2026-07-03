"""rev9 §5–6 normal-state remeasurement.

Per the rev9 directive:
  STEP 0  normal-state gate (Table-1 large cell, range [-0.110, -0.082]; 1 fast gate; abort on outlier)
  STEP 1  reuse inventory from rev6 (p1r6_*) where cell keys match
  STEP 2  new measurements only for cells not covered by reuse
  STEP 3  REVERSAL GATE on the new main-comparison
  STEP 4  rebuild paper/tables/*.tex + rev9_delta.md + rev9_PROPOSED_EDITS.md (NO main.tex prose edits)
  STEP 5  marker cleanup (auto comments only)
  STEP 6  assertion vs CSV + pdflatex skip if missing

Binary: legacy b2441f9d (`models/mobilint_backup/yolo11s.mxq`).
Single-stream NPU mode: global8; multi-stream NPU mode: single.

Manifest: results/manifest_rev9.json — idempotent resume.
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
from _step_d_common import (FGModelCPU, fg_worker, load_split_for_sid, load_val,
                            per_stream_sap, preload_background_models,
                            stop_background)
from step_f_partA_matrix import per_stream_map_offline
from step_h2_robustness import start_bg_custom

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_rev9.json"
PIN = json.loads((RES / "rev7_pin.json").read_text())
MXQ = PIN["mxq_path"]
DETECTOR = "yolo11s"
DETECTOR_PARAMS = 9.4

MODE_SS = "global8"  # single-stream NPU mode
MODE_MS = "single"   # multistream NPU mode

# §5–6 anchor sids + compositions
PARTA_SIDS = [2, 3, 8, 10, 13, 17, 21, 22]
COMP_A_N4 = [2, 22, 3, 21]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]
COMP_B_N4 = [17, 8, 10, 11]
COMP_B_N8 = [17, 8, 10, 11, 2, 13, 3, 14]
COMP_C_N4 = [15, 0, 5, 7]
COMP_C_N8 = [15, 19, 23, 0, 1, 20, 5, 7]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}

# Gate
GATE_LO = -0.110
GATE_HI = -0.082


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


# ============================ adapters ============================

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
    key = (MXQ, mode)
    insts = _NPU.setdefault(key, [])
    while len(insts) < n:
        from mblt_model_zoo import vision as mv
        m = mv.YOLO11s(local_path=MXQ, infer_mode=mode, product="aries")
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


# ============================ STEP 0 gate ============================

def step0_gate(val, gpu, npu_ss):
    """Fast gate: 6-log NPU L0 sweep → measure mean large gap."""
    print("[rev9 STEP0] normal-state gate: 6-log NPU L0 sweep…")
    gate_sids = [2, 5, 10, 13, 17, 21]  # spread across size groups
    sap_l_npu = []; sap_l_gpu = []
    for sid in gate_sids:
        split = load_split_for_sid(val, sid)
        # GPU
        m_g, _ = measure_one(sid, split, "GPU", gpu, "L0")
        sap_l_gpu.append(m_g["sap_l"])
        # NPU
        set_active_npu(npu_ss)
        m_n, _ = measure_one(sid, split, "NPU", npu_ss[0], "L0")
        sap_l_npu.append(m_n["sap_l"])
        print(f"  sid {sid:>2d}  GPU sap_l={m_g['sap_l']:.4f}  NPU sap_l={m_n['sap_l']:.4f}  gap={m_n['sap_l']-m_g['sap_l']:+.4f}")
    gap = float(np.mean(sap_l_npu) - np.mean(sap_l_gpu))
    with open(RES / "rev9_state_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts_iso", "n_sids", "gpu_l_mean", "npu_l_mean", "large_gap", "gate_lo", "gate_hi", "pass"])
        w.writeheader()
        w.writerow({"ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_sids": len(gate_sids),
                    "gpu_l_mean": round(float(np.mean(sap_l_gpu)), 4),
                    "npu_l_mean": round(float(np.mean(sap_l_npu)), 4),
                    "large_gap": round(gap, 4),
                    "gate_lo": GATE_LO, "gate_hi": GATE_HI,
                    "pass": GATE_LO <= gap <= GATE_HI})
    ok = GATE_LO <= gap <= GATE_HI
    print(f"[rev9 STEP0] gate gap={gap:+.4f} (target [{GATE_LO}, {GATE_HI}]) → {'PASS' if ok else 'FAIL'}")
    return ok, gap


# ============================ I/O writers ============================

BASE_COLS = ["sid", "log_id", "device", "infer_mode", "bg_level",
             "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
             "map_5095", "map_s", "map_m", "map_l",
             "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
MS_COLS = ["table", "tag", "n_streams", "bg_level", "infer_mode",
           "placement_name", "placement_spec",
           "mean_sap", "worst_sap", "mean_map", "worst_map", "wall_sec"]
for i in range(8):
    for k in ["dev","sid","sap","sap_s","sap_m","sap_l","map","skip"]:
        MS_COLS.append(f"s{i}_{k}")

def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})

def ms_write(path, table_tag, tag, agg, bg, mode_tag, name):
    row = {"table": table_tag, "tag": tag, "n_streams": agg["n_streams"],
           "bg_level": bg, "infer_mode": mode_tag,
           "placement_name": name, "placement_spec": agg["placement_spec"],
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
    append_csv(path, MS_COLS, row)


# ============================ STEP 2 — new measurements ============================

def step2_table1_cpu(val, manifest):
    """Table 1: CPU L0 column (GPU+NPU reused from rev6)."""
    out = RES / "rev9_table1_cpu.csv"
    cpu_model = None
    for sid in range(len(val["sequences"])):
        if is_done(manifest, "table1_cpu", f"sid{sid}"): continue
        split = load_split_for_sid(val, sid)
        if cpu_model is None:
            cpu_model = FGModelCPU()
        try:
            m, _ = measure_one(sid, split, "CPU", cpu_model, "L0")
        except Exception as e:
            print(f"  T1 CPU FAIL sid={sid}: {e}"); continue
        row = {"sid": sid, "log_id": val["sequences"][sid], "device": "CPU",
               "infer_mode": "", "bg_level": "L0",
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
        append_csv(out, BASE_COLS, row)
        mark_done(manifest, "table1_cpu", f"sid{sid}")
        print(f"  T1 CPU sid={sid:>2d}: sap={m['sap_5095']:.3f}  ({m['wall_sec']:.1f}s)")


def step2_partA_npu(val, manifest, npu_ss):
    """tab:partA: 8 sids × NPU L1_light (GPU L1 reused from rev6 p1r6_ladder)."""
    out = RES / "rev9_partA_npu.csv"
    set_active_npu(npu_ss)
    for sid in PARTA_SIDS:
        if is_done(manifest, "partA_npu", f"sid{sid}"): continue
        split = load_split_for_sid(val, sid)
        try:
            m, _ = measure_one(sid, split, "NPU", npu_ss[0], "L1_light")
        except Exception as e:
            print(f"  partA NPU FAIL sid={sid}: {e}"); continue
        row = {"sid": sid, "log_id": val["sequences"][sid], "device": "NPU",
               "infer_mode": MODE_SS, "bg_level": "L1_light",
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
        append_csv(out, BASE_COLS, row)
        mark_done(manifest, "partA_npu", f"sid{sid}")
        print(f"  partA NPU L1 sid={sid:>2d}: sap={m['sap_5095']:.3f}  sap_l={m['sap_l']:.3f}")


def step2_per_class(val, manifest, gpu, npu_ss):
    """tab:per-class: 24 logs × {GPU, NPU} × L0; per-class COCOeval."""
    out = RES / "rev9_per_class.csv"
    if is_done(manifest, "per_class", "done"):
        return
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from _step_d_common import WARMUP_FRAMES
    from step0_compare_devices import FPS
    cats = sorted(val["categories"], key=lambda c: c["id"])
    splits = {sid: load_split_for_sid(val, sid) for sid in range(len(val["sequences"]))}
    # global GT (post-warmup)
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

    def collect(device, model):
        all_ccf = []
        for sid in range(len(val["sequences"])):
            split = splits[sid]
            res = defaultdict(list); stop = threading.Event()
            bg_stops, bg_threads = start_bg_custom("L0")
            fg_worker(0, device, split, model, res, stop)
            stop_background(bg_stops, bg_threads)
            imgs = split["imgs"]; ts = res["timestamps"]; inp = res["input_fidx"]; r = res["results"]
            tidx_p1 = 0
            for ii, img in enumerate(imgs):
                t_gt = ii / FPS
                while tidx_p1 < len(ts) and ts[tidx_p1] <= t_gt: tidx_p1 += 1
                if ii < WARMUP_FRAMES: continue
                if tidx_p1 == 0: continue
                ti = tidx_p1 - 1
                bb, sc, lb = r[ti]
                for k in range(len(bb)):
                    x1, y1, x2, y2 = bb[k]
                    all_ccf.append({"image_id": int(img["id"]),
                                     "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                     "score": float(sc[k]), "category_id": int(lb[k])})
            print(f"  per-class collect {device} sid={sid:>2d} dets={len(all_ccf)}")
        return all_ccf

    rows = []
    for device, model in [("GPU", gpu), ("NPU", npu_ss[0])]:
        if device == "NPU":
            set_active_npu(npu_ss)
        all_ccf = collect(device, model)
        for cat in cats:
            cid = cat["id"]
            n_gt = sum(1 for a in coco_gt.dataset["annotations"] if a["category_id"] == cid)
            sub_dets = [d for d in all_ccf if d["category_id"] == cid]
            if not sub_dets:
                rows.append({"device": device, "cat_id": cid, "cat_name": cat["name"],
                              "ap_5095": 0.0, "ap_50": 0.0, "n_gt": n_gt, "n_dets": 0}); continue
            coco_dt = coco_gt.loadRes(sub_dets)
            e = COCOeval(coco_gt, coco_dt, "bbox")
            e.params.imgIds = img_ids_eval; e.params.catIds = [cid]
            e.evaluate(); e.accumulate(); e.summarize()
            rows.append({"device": device, "cat_id": cid, "cat_name": cat["name"],
                          "ap_5095": round(float(e.stats[0]), 4),
                          "ap_50": round(float(e.stats[1]), 4),
                          "n_gt": n_gt, "n_dets": len(sub_dets)})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device","cat_id","cat_name","ap_5095","ap_50","n_gt","n_dets"])
        w.writeheader()
        for r in rows: w.writerow(r)
    mark_done(manifest, "per_class", "done")
    print(f"saved {out}")


def step2_main_cmp(val, manifest, gpu_models, npu_ms_4):
    """tab:main-comparison: N=4 sids 2,22,3,21 × bg{L1_light, L2_lm} × strategies + Oracle.
    Reuses rev6's SA/SBR at L1_light from p1r6_multistream_yolo11s.csv."""
    out = RES / "rev9_main_cmp.csv"
    sids = COMP_A_N4
    splits = [load_split_for_sid(val, s) for s in sids]
    placements = {
        "Naive_allGPU":  ["GPU","GPU","GPU","GPU"],
        "SizeAware":     ["NPU","NPU","GPU","GPU"],
        "SizeBlindRev":  ["GPU","GPU","NPU","NPU"],
        "AllNPU":        ["NPU","NPU","NPU","NPU"],
    }
    # 16-placement Oracle
    for k in range(5):
        for npu_pos in combinations(range(4), k):
            spec = ["NPU" if i in npu_pos else "GPU" for i in range(4)]
            name = "Oracle_k{}_{}".format(k, "_".join(map(str, npu_pos)) if npu_pos else "none")
            if not any(s == spec for s in placements.values()):
                placements[name] = spec
    for bg in ["L1_light", "L2_lm"]:
        for name, placement in placements.items():
            key = f"{bg}_{name}"
            if is_done(manifest, "main_cmp", key): continue
            n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
            if n_npu > len(npu_ms_4) or n_gpu > len(gpu_models):
                print(f"  main_cmp SKIP {key}: not enough engines"); continue
            try:
                agg = measure_multistream(sids, splits, placement,
                                           gpu_models[:n_gpu] if n_gpu else [],
                                           npu_ms_4[:n_npu] if n_npu else [], bg)
            except Exception as e:
                print(f"  main_cmp FAIL {key}: {e}"); continue
            ms_write(out, "main-comparison", bg, agg, bg, MODE_MS, name)
            mark_done(manifest, "main_cmp", key)
            print(f"  main_cmp {bg} {name[:24]:<24s}: worst={agg['worst_sap']:.4f} mean={agg['mean_sap']:.4f}")


def step2_schedule(val, manifest, gpu_models, npu_ms_8):
    """tab:schedule-shift: N∈{4,8} × bg{L0,L1l,L1h,L2lm,L3vlm} × {Naive,SizeAware,AllNPU} (+SA_NPU6 for N=8)."""
    out = RES / "rev9_schedule.csv"
    for N in [4, 8]:
        sids = COMP_A_N4 if N == 4 else COMP_A_N8
        splits = [load_split_for_sid(val, s) for s in sids]
        if N == 4:
            named = {"Naive_allGPU": ["GPU"]*4,
                     "SizeAware": ["NPU","NPU","GPU","GPU"],
                     "AllNPU": ["NPU"]*4}
        else:
            named = {"Naive_allGPU": ["GPU"]*8,
                     "SizeAware_NPU4": ["NPU","NPU","NPU","NPU","GPU","GPU","GPU","GPU"],
                     "SizeAware_NPU6": ["NPU","NPU","NPU","NPU","NPU","NPU","GPU","GPU"],
                     "AllNPU": ["NPU"]*8}
        for bg in LADDER:
            for name, placement in named.items():
                key = f"N{N}_{bg}_{name}"
                if is_done(manifest, "schedule", key): continue
                n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
                if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models):
                    print(f"  schedule SKIP {key}"); continue
                try:
                    agg = measure_multistream(sids, splits, placement,
                                               gpu_models[:n_gpu], npu_ms_8[:n_npu], bg)
                except Exception as e:
                    print(f"  schedule FAIL {key}: {e}"); continue
                ms_write(out, "schedule-shift", f"N={N}", agg, bg, MODE_MS, name)
                mark_done(manifest, "schedule", key)
                print(f"  schedule N={N} {bg:<9s} {name[:20]:<20s}: worst={agg['worst_sap']:.4f}")


def step2_capacity(val, manifest, gpu_models, npu_ms_8):
    """tab:capacity: N∈{2,3,4,5,6,8} × bg L1_light × {Naive,SizeAware,SizeBlindRev,AllNPU}."""
    out = RES / "rev9_capacity.csv"
    COMP = {2: ([2],[3]), 3: ([2,22],[3]), 4: ([2,22],[3,21]),
            5: ([2,22,13],[3,21]), 6: ([2,22,13],[3,21,14]),
            8: ([2,22,13,16],[3,21,14,4])}
    bg = "L1_light"
    for N in [2,3,4,5,6,8]:
        small, large = COMP[N]
        sids = small + large
        splits = [load_split_for_sid(val, s) for s in sids]
        named = {"Naive_allGPU": ["GPU"]*N,
                 "SizeAware": ["NPU" if s in small else "GPU" for s in sids],
                 "SizeBlindRev": ["NPU" if s in large else "GPU" for s in sids],
                 "AllNPU": ["NPU"]*N}
        for name, placement in named.items():
            key = f"N{N}_{name}"
            if is_done(manifest, "capacity", key): continue
            n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
            if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models): continue
            try:
                agg = measure_multistream(sids, splits, placement,
                                           gpu_models[:n_gpu] if n_gpu else [],
                                           npu_ms_8[:n_npu] if n_npu else [], bg)
            except Exception as e:
                print(f"  capacity FAIL {key}: {e}"); continue
            ms_write(out, "capacity", f"N={N}", agg, bg, MODE_MS, name)
            mark_done(manifest, "capacity", key)
            print(f"  capacity N={N} {name[:18]:<18s}: worst={agg['worst_sap']:.4f}")


def step2_natural(val, manifest, gpu_models, npu_ms_8):
    """tab:natural: Comp B + C × N∈{4,8} × bg{L1_light,L1_heavy} × strategies + oracle samples."""
    out = RES / "rev9_natural.csv"
    COMPS = {"B_medium_mixed": {4: COMP_B_N4, 8: COMP_B_N8},
             "C_diverse":      {4: COMP_C_N4, 8: COMP_C_N8}}
    SIZE_LABEL = {}
    for grp, sids in SIZE_GROUPS.items():
        for s in sids:
            SIZE_LABEL[s] = {"small-rich":"small","medium-mixed":"medium","large-rich":"large"}[grp]

    def named_placements(N, sids):
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
            return {"Naive_allGPU": ["GPU"]*8,
                    "SizeAware_NPU4": sa,
                    "SizeAware_NPU6": ["NPU"]*6 + ["GPU"]*2,
                    "SizeBlindRev_NPU4": sbr,
                    "AllNPU": ["NPU"]*8}

    import random
    for comp, ns_map in COMPS.items():
        for N in [4, 8]:
            sids = ns_map[N]
            splits = [load_split_for_sid(val, s) for s in sids]
            for bg in ["L1_light", "L1_heavy"]:
                placements = named_placements(N, sids)
                rng = random.Random(42 + N + hash(comp) & 0xffff)
                for k in range(min(N, 4) + 1):
                    cands = list(combinations(range(N), k))
                    sample = rng.sample(cands, min(2, len(cands))) if k > 0 else cands
                    for pos in sample:
                        spec = ["NPU" if i in pos else "GPU" for i in range(N)]
                        nm = f"Oracle_k{k}_p{'_'.join(map(str,pos)) if pos else 'none'}"
                        if nm not in placements: placements[nm] = spec
                for name, placement in placements.items():
                    key = f"{comp}_N{N}_{bg}_{name}"
                    if is_done(manifest, "natural", key): continue
                    n_npu = placement.count("NPU"); n_gpu = placement.count("GPU")
                    if n_npu > len(npu_ms_8) or n_gpu > len(gpu_models): continue
                    try:
                        agg = measure_multistream(sids, splits, placement,
                                                   gpu_models[:n_gpu] if n_gpu else [],
                                                   npu_ms_8[:n_npu] if n_npu else [], bg)
                    except Exception as e:
                        print(f"  natural FAIL {key}: {e}"); continue
                    ms_write(out, "natural", f"{comp}_N{N}", agg, bg, MODE_MS, name)
                    mark_done(manifest, "natural", key)
                    print(f"  natural {comp} N={N} {bg:<9s} {name[:20]:<20s}: worst={agg['worst_sap']:.4f}")


# ============================ main ============================

def reversal_gate(main_cmp_csv):
    if not Path(main_cmp_csv).exists(): return None
    df = pd.read_csv(main_cmp_csv)
    sub = df[(df.n_streams == 4) & (df.bg_level == "L1_light")]
    sa = sub[sub.placement_name == "SizeAware"]
    sb = sub[sub.placement_name == "SizeBlindRev"]
    if not len(sa) or not len(sb): return None
    wg = float(sa.iloc[0]["worst_sap"]) - float(sb.iloc[0]["worst_sap"])
    mg = float(sa.iloc[0]["mean_sap"]) - float(sb.iloc[0]["mean_sap"])
    return {"worst_gain": wg, "mean_gain": mg, "inverts": wg > 0,
            "sa_worst": float(sa.iloc[0]["worst_sap"]),
            "sbr_worst": float(sb.iloc[0]["worst_sap"])}


def main():
    manifest = load_manifest()
    print("[rev9] preloading bg max=L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[rev9] bg preload {time.time()-t0:.1f}s")

    val = load_val()
    print(f"[rev9] {len(val['sequences'])} logs; mxq sha[:12]={PIN['mxq_sha256'][:12]}")

    # Load 1 GPU + 1 NPU global8 for gate + single-stream tables
    print("[rev9] loading GPU + 1 NPU global8…")
    gpu = FGModelGPUGeneric()
    npu_ss = load_npu(MODE_SS, n=1)

    # STEP 0 — gate
    ok, gap = step0_gate(val, gpu, npu_ss)
    if not ok:
        print(f"[rev9] STEP0 gate FAIL (gap {gap:+.4f}). Aborting per spec.")
        return

    t_all = time.time()
    # STEP 2 — single-stream tables
    print("\n--- T1 CPU L0 (Table 1 missing column) ---")
    step2_table1_cpu(val, manifest)
    print("\n--- partA NPU L1_light ---")
    step2_partA_npu(val, manifest, npu_ss)
    print("\n--- per-class (24 logs × {GPU,NPU} × L0) ---")
    step2_per_class(val, manifest, gpu, npu_ss)

    # Swap NPU mode: single (multistream)
    print("\n[rev9] swapping NPU mode global8 → single…")
    dispose_npu(MODE_SS)
    npu_ms = load_npu(MODE_MS, n=8)
    print("[rev9] loading 8 GPU instances…")
    gpu_models = [gpu] + [FGModelGPUGeneric() for _ in range(7)]

    # STEP 2 — multistream tables
    print("\n--- main-comparison ---")
    step2_main_cmp(val, manifest, gpu_models, npu_ms[:4])

    gate = reversal_gate(RES / "rev9_main_cmp.csv")
    with open(RES / "rev9_reversal_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sa_worst","sbr_worst","worst_gain","mean_gain","inverts","pass"])
        w.writeheader()
        if gate:
            w.writerow({**gate, "pass": bool(gate["inverts"])})
            print(f"\n[rev9 STEP3] REVERSAL GATE: {'PASS' if gate['inverts'] else 'FAIL'} (worst gain {gate['worst_gain']:+.4f})")
            if not gate["inverts"]:
                print("[rev9] STEP3 GATE FAIL — stopping per spec.")
                return

    print("\n--- capacity ---")
    step2_capacity(val, manifest, gpu_models, npu_ms)
    print("\n--- schedule-shift ---")
    step2_schedule(val, manifest, gpu_models, npu_ms)
    print("\n--- natural (Comp B + C) ---")
    step2_natural(val, manifest, gpu_models, npu_ms)

    print(f"\n[rev9] sweep complete. wall_total={time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
