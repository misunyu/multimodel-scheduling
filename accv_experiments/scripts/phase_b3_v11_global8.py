"""STEP 2 + STEP 3 of B3 (rev 4) — v11 family global8 sweep.

For each detector in {yolo11s, yolo11m, yolo11l, yolo11x} (yolo11n is FAIL,
recorded as TBD by phase_b3_probe.py):

  BASELINE phase  (24 logs, no bg):
      24 × GPU L0     (FP32 single-stream)
      24 × NPU L0     (INT8 global8 single-stream)

  LADDER phase    (24 logs, GPU side only):
      24 × GPU × {L1_light, L1_heavy, L2_lm, L3_vlm}

  MULTISTREAM phase  (Composition A bg L1_light):
      N=4 [2,22,3,21]     × {SizeAware [NNGG], SizeBlindRev [GGNN]}
      N=8 [2,22,13,16,3,21,14,4] × {SizeAware [4N+4G], SizeBlindRev [4G+4N]}

  NPU mode policy:
      BASELINE NPU       → global8  (matches Table 1 / step_a configuration)
      MULTISTREAM NPU    → single   (global8 cannot host >1 instance; we re-load
                                     the same .mxq under infer_mode='single' just
                                     for this phase). Documented in the per-row CSV.

STEP 2 gate: v11s baseline must reproduce step_a (paired NPU-GPU diff per size
within ±0.005 of Table 1). If it fails, the detector is recorded but the main
table is NOT updated; the inconsistency is logged.

Outputs (per detector):
  results/p1g8_baseline_<det>.csv        baseline (GPU + NPU global8)
  results/p1g8_ladder_<det>.csv          GPU ladder
  results/p1g8_multistream_<det>.csv     multistream (single mode for NPU)

Manifest checkpoint:
  results/manifest_b3.json
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _step_d_common as cm
from _step_d_common import (FGModelGPU, fg_worker, load_split_for_sid,
                            load_val, per_stream_sap,
                            preload_background_models, stop_background)
from step_f_partA_matrix import per_stream_map_offline
from step_h2_robustness import start_bg_custom

RES = SCRIPT_DIR.parent / "results"
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_b3.json"

# ---- detector registry (yolo11n absent: 404 from HF; see v11_mxq_probe.csv) ----
DETECTORS = [
    {"name": "yolo11s", "params_M": 9.4,  "cls": "YOLO11s",
     "ultralytics_pt": "yolo11s.pt"},
    {"name": "yolo11m", "params_M": 20.1, "cls": "YOLO11m",
     "ultralytics_pt": "yolo11m.pt"},
    {"name": "yolo11l", "params_M": 25.3, "cls": "YOLO11l",
     "ultralytics_pt": "yolo11l.pt"},
    {"name": "yolo11x", "params_M": 56.9, "cls": "YOLO11x",
     "ultralytics_pt": "yolo11x.pt"},
]
LADDER_FG = ["L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
COMP_A_N4 = [2, 22, 3, 21]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]

# ---- v11s repro tolerance ----
REPRO_TOL = 0.005
TABLE1_TARGET = {"sap_small": -0.007, "sap_medium": -0.036, "sap_large": +0.001}


# ============================ manifest helpers ============================

def load_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            pass
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))


def mark_done(manifest, det, phase, key):
    manifest.setdefault(det, {}).setdefault(phase, {})[key] = {
        "status": "done", "ts": int(time.time())
    }
    save_manifest(manifest)


def is_done(manifest, det, phase, key):
    return manifest.get(det, {}).get(phase, {}).get(key, {}).get("status") == "done"


# ============================ GPU/NPU adapters ============================

class FGModelGPUGeneric:
    def __init__(self, pt_name):
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


_NPU_BY_DETECTOR = {}


def load_npu_engines(det_cfg, n, infer_mode):
    from mblt_model_zoo import vision as mv
    cls = getattr(mv, det_cfg["cls"])
    instances = _NPU_BY_DETECTOR.setdefault((det_cfg["name"], infer_mode), [])
    while len(instances) < n:
        # mblt_model_zoo handles cache resolution internally.
        m = cls(infer_mode=infer_mode, product="aries")
        from step0_compare_devices import CONF, IOU
        dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
        x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
        instances.append(m)
    return instances[:n]


def dispose_npu_engines(det_name=None, infer_mode=None):
    keys = list(_NPU_BY_DETECTOR.keys())
    for k in keys:
        if det_name is not None and k[0] != det_name: continue
        if infer_mode is not None and k[1] != infer_mode: continue
        for m in _NPU_BY_DETECTOR[k]:
            try: m.dispose()
            except Exception: pass
        del _NPU_BY_DETECTOR[k]


def set_active_npu_engines(insts):
    cm._NPU_INSTANCES.clear()
    cm._NPU_INSTANCES.extend(insts)


# ============================ measurement ============================

def measure_single_stream(sid, split, device, model, bg):
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
    }


def measure_multistream(sids, splits, placement, gpu_models, npu_models, bg):
    n = len(placement)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg)
    gpu_i = npu_i = 0
    threads = []
    for i, dev in enumerate(placement):
        if dev == "GPU":
            model = gpu_models[gpu_i]; gpu_i += 1
        elif dev == "NPU":
            model = npu_models[npu_i]; npu_i += 1
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
            "map_5095": mp["map_5095"], "latency_mean": sap["infer_mean_ms"],
            "frame_skip_pct": sap["frame_skip_pct"],
        })
    saps = [s["sap_5095"] for s in per_stream]
    return {
        "n_streams": n, "mean_sap": float(np.mean(saps)),
        "worst_sap": float(np.min(saps)),
        "per_stream": per_stream, "wall_sec": wall,
    }


# ============================ I/O ============================

BASE_COLS = ["detector", "params_M", "phase", "sid", "device", "infer_mode",
             "bg_level",
             "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
             "map_5095", "map_s", "map_m", "map_l",
             "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
MS_COLS = ["detector", "params_M", "phase", "n_streams", "bg_level", "infer_mode",
           "placement_name", "placement_spec",
           "mean_sap", "worst_sap", "wall_sec"]


def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


# ============================ STEP 2 — v11s repro gate ============================

def check_v11s_gate(out_b):
    """Compare v11s baseline (global8) to step_a/Table 1."""
    df = pd.read_csv(out_b)
    # mean per-device per-size
    g = df[(df.device == "GPU") & (df.bg_level == "L0")]
    n = df[(df.device == "NPU") & (df.bg_level == "L0")]
    if len(g) < 24 or len(n) < 24:
        return {"status": "incomplete", "n_gpu": len(g), "n_npu": len(n)}
    out = {"status": "checked", "deltas": {}, "within_tol": {}}
    for size_col, key in [("sap_s", "sap_small"),
                           ("sap_m", "sap_medium"),
                           ("sap_l", "sap_large")]:
        mg = float(g[size_col].mean())
        mn = float(n[size_col].mean())
        diff = mn - mg
        target = TABLE1_TARGET[key]
        within = abs(diff - target) <= REPRO_TOL
        out["deltas"][key] = {"gpu_mean": round(mg, 4),
                               "npu_mean": round(mn, 4),
                               "diff": round(diff, 4),
                               "table1_target": target,
                               "within_tol": within}
        out["within_tol"][key] = within
    out["pass"] = all(out["within_tol"].values())
    return out


# ============================ detector sweep ============================

def sweep_detector(det_cfg, val, manifest, gate_only=False):
    name = det_cfg["name"]
    params_M = det_cfg["params_M"]
    print(f"\n{'='*78}\n  DETECTOR: {name} ({params_M} M params)\n{'='*78}")

    t0 = time.time()
    print(f"[{name}] loading GPU {det_cfg['ultralytics_pt']}…")
    gpu = FGModelGPUGeneric(det_cfg["ultralytics_pt"])
    print(f"[{name}] loading 1 NPU engine (global8)…")
    npu_g8 = load_npu_engines(det_cfg, 1, "global8")
    set_active_npu_engines(npu_g8)
    print(f"[{name}] preload {time.time()-t0:.1f}s")

    # ---------- BASELINE (global8) ----------
    out_b = RES / f"p1g8_baseline_{name}.csv"
    print(f"[{name}] BASELINE phase (global8) → {out_b}")
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for device, bg in [("GPU", "L0"), ("NPU", "L0")]:
            key = f"{device}_{bg}_sid{sid}"
            if is_done(manifest, name, "baseline_g8", key):
                continue
            model = gpu if device == "GPU" else npu_g8[0]
            try:
                m = measure_single_stream(sid, split, device, model, bg)
            except Exception as e:
                print(f"  FAIL {key}: {type(e).__name__}: {e}")
                continue
            row = {"detector": name, "params_M": params_M, "phase": "baseline",
                   "sid": sid, "device": device,
                   "infer_mode": "global8" if device == "NPU" else "",
                   "bg_level": bg,
                   **{k: round(v, 4) if isinstance(v, float) else v
                      for k, v in m.items()}}
            append_csv(out_b, BASE_COLS, row)
            mark_done(manifest, name, "baseline_g8", key)
            print(f"  {device} L0 sid={sid:>2d}: sap={m['sap_5095']:.3f} "
                  f"sap_s/m/l={m['sap_s']:.3f}/{m['sap_m']:.3f}/{m['sap_l']:.3f} "
                  f"({m['wall_sec']:.1f}s)")
    # STEP 2 gate (for yolo11s only)
    if name == "yolo11s":
        gate = check_v11s_gate(out_b)
        with open(RES / "v11s_repro.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["size", "gpu_mean", "npu_mean",
                                              "diff", "table1_target", "within_tol"])
            w.writeheader()
            for size, d in gate["deltas"].items():
                w.writerow({"size": size, **{k: d[k] for k in
                            ["gpu_mean", "npu_mean", "diff",
                             "table1_target", "within_tol"]}})
        print(f"\n[{name}] STEP 2 gate: {'PASS' if gate.get('pass') else 'FAIL'}")
        for size, d in gate["deltas"].items():
            print(f"  {size:<12s} gpu={d['gpu_mean']:.4f} npu={d['npu_mean']:.4f} "
                  f"diff={d['diff']:+.4f} (Table 1 target {d['table1_target']:+.4f}, within {d['within_tol']})")
        if gate_only:
            return gate

    # ---------- LADDER (global8 GPU only) ----------
    out_l = RES / f"p1g8_ladder_{name}.csv"
    print(f"\n[{name}] LADDER phase → {out_l}")
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for bg in LADDER_FG:
            key = f"GPU_{bg}_sid{sid}"
            if is_done(manifest, name, "ladder_g8", key):
                continue
            try:
                m = measure_single_stream(sid, split, "GPU", gpu, bg)
            except Exception as e:
                print(f"  FAIL {key}: {type(e).__name__}: {e}")
                continue
            row = {"detector": name, "params_M": params_M, "phase": "ladder",
                   "sid": sid, "device": "GPU", "infer_mode": "",
                   "bg_level": bg,
                   **{k: round(v, 4) if isinstance(v, float) else v
                      for k, v in m.items()}}
            append_csv(out_l, BASE_COLS, row)
            mark_done(manifest, name, "ladder_g8", key)
            print(f"  GPU {bg:<9s} sid={sid:>2d}: sap={m['sap_5095']:.3f} "
                  f"eff={m['eff_e2e_mean']:.1f}ms skip={m['frame_skip_pct']:.0f}% "
                  f"({m['wall_sec']:.1f}s)")

    # ---------- MULTISTREAM (need NPU in single mode) ----------
    print(f"\n[{name}] disposing global8 NPU engines; loading 4 NPU engines (single)…")
    dispose_npu_engines(name, "global8")
    try:
        npu_s = load_npu_engines(det_cfg, 4, "single")
    except Exception as e:
        print(f"  FAIL: single-mode load failed: {e}")
        # Skip multistream phase, but log it
        manifest.setdefault(name, {})["multistream_skip_reason"] = str(e)[:200]
        save_manifest(manifest)
        return
    set_active_npu_engines(npu_s)
    gpu_models = [gpu] + [FGModelGPUGeneric(det_cfg["ultralytics_pt"]) for _ in range(3)]

    out_ms = RES / f"p1g8_multistream_{name}.csv"
    ms_plan = [
        ("N=4", "L1_light", "SizeAware",    COMP_A_N4, ["NPU", "NPU", "GPU", "GPU"]),
        ("N=4", "L1_light", "SizeBlindRev", COMP_A_N4, ["GPU", "GPU", "NPU", "NPU"]),
        ("N=8", "L1_light", "SizeAware_NPU4",    COMP_A_N8,
            ["NPU", "NPU", "NPU", "NPU", "GPU", "GPU", "GPU", "GPU"]),
        ("N=8", "L1_light", "SizeBlindRev_NPU4", COMP_A_N8,
            ["GPU", "GPU", "GPU", "GPU", "NPU", "NPU", "NPU", "NPU"]),
    ]
    for tag, bg, pname, sids, placement in ms_plan:
        key = f"{tag}_{bg}_{pname}"
        if is_done(manifest, name, "multistream_single", key):
            continue
        splits = [load_split_for_sid(val, s) for s in sids]
        n = len(placement)
        if n == 8:
            cur_gpu = gpu_models[:4]; cur_npu = npu_s[:4]
        else:
            cur_gpu = gpu_models[:2]; cur_npu = npu_s[:2]
        try:
            agg = measure_multistream(sids, splits, placement, cur_gpu, cur_npu, bg)
        except Exception as e:
            print(f"  FAIL {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        row = {"detector": name, "params_M": params_M, "phase": "multistream",
               "n_streams": n, "bg_level": bg, "infer_mode": "single",
               "placement_name": pname, "placement_spec": json.dumps(placement),
               "mean_sap": round(agg["mean_sap"], 4),
               "worst_sap": round(agg["worst_sap"], 4),
               "wall_sec": round(agg["wall_sec"], 1)}
        cols = MS_COLS + [f"s{i}_{k}" for i in range(8)
                          for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "skip"]]
        for s in agg["per_stream"]:
            i = s["stream_id"]
            row[f"s{i}_dev"] = s["device"]; row[f"s{i}_sid"] = s["sid"]
            row[f"s{i}_sap"] = round(s["sap_5095"], 4)
            row[f"s{i}_sap_s"] = round(s["sap_s"], 4)
            row[f"s{i}_sap_m"] = round(s["sap_m"], 4)
            row[f"s{i}_sap_l"] = round(s["sap_l"], 4)
            row[f"s{i}_skip"] = round(s["frame_skip_pct"], 1)
        append_csv(out_ms, cols, row)
        mark_done(manifest, name, "multistream_single", key)
        print(f"  {tag} {bg} {pname}: worst={agg['worst_sap']:.4f}  "
              f"mean={agg['mean_sap']:.4f}  ({agg['wall_sec']:.1f}s)")

    print(f"[{name}] disposing single NPU engines…")
    dispose_npu_engines(name, "single")


# ============================ main ============================

def main():
    manifest = load_manifest()

    print("[B3] preloading bg models max=L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[B3] bg preload {time.time()-t0:.1f}s")

    val = load_val()
    print(f"[B3] {len(val['sequences'])} logs")

    t_all = time.time()
    for det_cfg in DETECTORS:
        try:
            sweep_detector(det_cfg, val, manifest)
        except Exception as e:
            print(f"[B3] DETECTOR {det_cfg['name']} TOP-LEVEL FAIL: {e}")
            traceback.print_exc()
            manifest.setdefault(det_cfg["name"], {})["fatal"] = str(e)[:200]
            save_manifest(manifest)
    print(f"\n[B3] sweep complete. wall_total={time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
