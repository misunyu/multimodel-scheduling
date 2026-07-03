"""Phase P1 — YOLOv8 family generalization sweep (B' + C' + D').

Per claude_code_experiments.md (rev 3) §4. Measures generalization across
detector capacity using mxq files already in the Mobilint HF cache, so no
compilation is required. YOLOv11s is added as a cross-family marker but its
data is REUSED from prior steps (step_a + gen_decomp + cstar + gen_gain).

For each detector d in {v8n, v8s, v8m, v8l, v8x}:
  Phase BASELINE  (24 logs):
      24 × {GPU L0, NPU L0}  →  size-stratified sAP + mAP per sid
  Phase LADDER    (24 logs):
      24 × {GPU L1_light, GPU L1_heavy, GPU L2_lm, GPU L3_vlm}
      →  per-sid streaming sAP per size, eff E2E latency, frame-skip
  Phase MULTISTREAM   (Composition A):
      N=4 [2,22,3,21] × {SizeAware [NNGG], SizeBlindRev [GGNN]} × bg L1_light
      N=8 [2,22,13,16,3,21,14,4] × {SizeAware [4N+4G], SizeBlindRev [4G+4N]} × bg L1_light

NPU mode = single (consistent with multi-stream feasibility). v11s reuse uses
step_a (global8) for the single-camera baseline AND existing single-mode data
for multistream — clearly noted in the merged CSVs.

Manifest-checkpointed: a (detector, phase, cell) marked done in results/manifest.json
is skipped on resume.

Outputs:
  results/manifest.json                       checkpoint
  results/p1_baseline_<det>.csv               per-detector baseline
  results/p1_ladder_<det>.csv                 per-detector ladder
  results/p1_multistream_<det>.csv            per-detector multistream
  results/p1_family.csv                       combined long-format
  results/gen_decomp_v2.csv                   per-detector Q(s), L(s,L1)
  results/cstar_v2_family.csv                 per-detector C*_g (sAP-gap def)
  results/gen_gain_v2.csv                     per-detector worst gain at N=4 L1_light
  paper/tables/gen_decomp.tex                 (regenerated, 5 v8 + v11s)
  paper/tables/gen_gain.tex                   (regenerated, 5 v8 + v11s)
  paper/figures/gen_cstar.pdf                 multi-capacity C* curves
"""

from __future__ import annotations

import csv
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
from _step_d_common import (fg_worker, load_split_for_sid, load_val,
                            per_stream_sap, preload_background_models,
                            stop_background)
from step_f_partA_matrix import per_stream_map_offline
from step_h2_robustness import start_bg_custom

RES = SCRIPT_DIR.parent / "results"
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest.json"

# ---- detector registry (capacity-ordered) ----
DETECTORS = [
    {"name": "yolov8n", "params_M": 3.2,  "npu_cls": "YOLOv8n"},
    {"name": "yolov8s", "params_M": 11.2, "npu_cls": "YOLOv8s"},
    {"name": "yolov8m", "params_M": 25.9, "npu_cls": "YOLOv8m"},
    {"name": "yolov8l", "params_M": 43.7, "npu_cls": "YOLOv8l"},
    {"name": "yolov8x", "params_M": 68.2, "npu_cls": "YOLOv8x"},
]
MARKER = {"name": "yolo11s", "params_M": 9.4, "reuse": True}

LADDER_FG = ["L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
COMP_A_N4 = [2, 22, 3, 21]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]

# ---- group definitions ----
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}


# ====================== manifest helpers ============================

def load_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            pass
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))


def mark_done(manifest, det, phase, cell_key):
    manifest.setdefault(det, {}).setdefault(phase, {})[cell_key] = {
        "status": "done", "ts": int(time.time())
    }
    save_manifest(manifest)


def is_done(manifest, det, phase, cell_key):
    return manifest.get(det, {}).get(phase, {}).get(cell_key, {}).get("status") == "done"


# ====================== adapter — load by detector ============================

class FGModelGPUGeneric:
    """Per-stream Ultralytics YOLO on CUDA — accepts any .pt that ultralytics knows."""
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


_NPU_BY_DETECTOR = {}  # det name -> list of N model instances


def load_npu_engines(det_cfg, n):
    """Preload n NPU instances of the given detector in single mode."""
    from mblt_model_zoo import vision as mv
    cls = getattr(mv, det_cfg["npu_cls"])
    # Resolve local mxq path under the HF cache so we don't redownload
    import glob
    cands = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--mobilint--{det_cfg['npu_cls']}/snapshots/*/aries/{det_cfg['name']}.mxq"))
    local_path = cands[0] if cands else None

    instances = _NPU_BY_DETECTOR.setdefault(det_cfg["name"], [])
    while len(instances) < n:
        m = cls(local_path=local_path, infer_mode="single", product="aries")
        from step0_compare_devices import IMG_SIZE, CONF, IOU
        dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
        x = m.preprocess(dummy); o = m(x); m.postprocess(o, conf_thres=CONF, iou_thres=IOU)
        instances.append(m)
    return instances[:n]


def dispose_npu_engines(det_name):
    insts = _NPU_BY_DETECTOR.get(det_name, [])
    for m in insts:
        try:
            m.dispose()
        except Exception:
            pass
    _NPU_BY_DETECTOR[det_name] = []


# Monkey-patch _step_d_common's NPU instance list so fg_worker / npu_infer
# can use these per-detector engines transparently.
def set_active_npu_engines(insts):
    cm._NPU_INSTANCES.clear()
    cm._NPU_INSTANCES.extend(insts)


# ====================== per-cell measurement ============================

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
        "latency_mean": sap["infer_mean_ms"],
        "eff_e2e_mean": eff,
        "frame_skip_pct": sap["frame_skip_pct"],
        "wall_sec": wall,
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
            raise ValueError(f"unknown device {dev}")
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
        "n_streams": n,
        "mean_sap": float(np.mean(saps)),
        "worst_sap": float(np.min(saps)),
        "per_stream": per_stream,
        "wall_sec": wall,
    }


# ====================== detector sweep ============================

BASE_COLS = ["detector", "params_M", "phase", "sid", "device", "bg_level",
             "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
             "map_5095", "map_s", "map_m", "map_l",
             "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]
MS_COLS = ["detector", "params_M", "phase", "n_streams", "bg_level",
           "placement_name", "placement_spec",
           "mean_sap", "worst_sap", "wall_sec"]
# per-stream columns appended later


def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def sweep_detector(det_cfg, val, manifest):
    name = det_cfg["name"]
    params_M = det_cfg["params_M"]
    print(f"\n{'='*78}\n  DETECTOR: {name}  ({params_M} M params)\n{'='*78}")

    # Preload GPU (single instance — single-stream phases) and NPU (4 instances for N=4
    # multistream; NPU baseline reuses instance 0).
    t0 = time.time()
    pt_name = f"{name}.pt"
    print(f"[{name}] loading GPU {pt_name}…")
    gpu = FGModelGPUGeneric(pt_name)
    print(f"[{name}] loading 4 NPU engines (single mode)…")
    npu_instances = load_npu_engines(det_cfg, 4)
    set_active_npu_engines(npu_instances)
    print(f"[{name}] preload {time.time()-t0:.1f}s")

    # ---------- BASELINE ----------
    out_b = RES / f"p1_baseline_{name}.csv"
    print(f"[{name}] BASELINE phase → {out_b}")
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for device, bg in [("GPU", "L0"), ("NPU", "L0")]:
            key = f"{device}_{bg}_sid{sid}"
            if is_done(manifest, name, "baseline", key):
                continue
            t0 = time.time()
            model = gpu if device == "GPU" else npu_instances[0]
            try:
                m = measure_single_stream(sid, split, device, model, bg)
            except Exception as e:
                print(f"  FAIL {key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            row = {"detector": name, "params_M": params_M, "phase": "baseline",
                   "sid": sid, "device": device, "bg_level": bg,
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            append_csv(out_b, BASE_COLS, row)
            mark_done(manifest, name, "baseline", key)
            print(f"  {device} L0 sid={sid:>2d}: sap={m['sap_5095']:.3f} "
                  f"sap_s={m['sap_s']:.3f}/m={m['sap_m']:.3f}/l={m['sap_l']:.3f} "
                  f"({m['wall_sec']:.1f}s)")

    # ---------- LADDER ----------
    out_l = RES / f"p1_ladder_{name}.csv"
    print(f"[{name}] LADDER phase → {out_l}")
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for bg in LADDER_FG:
            key = f"GPU_{bg}_sid{sid}"
            if is_done(manifest, name, "ladder", key):
                continue
            t0 = time.time()
            try:
                m = measure_single_stream(sid, split, "GPU", gpu, bg)
            except Exception as e:
                print(f"  FAIL {key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            row = {"detector": name, "params_M": params_M, "phase": "ladder",
                   "sid": sid, "device": "GPU", "bg_level": bg,
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            append_csv(out_l, BASE_COLS, row)
            mark_done(manifest, name, "ladder", key)
            print(f"  GPU {bg:<9s} sid={sid:>2d}: sap={m['sap_5095']:.3f} "
                  f"eff={m['eff_e2e_mean']:.1f}ms skip={m['frame_skip_pct']:.0f}% "
                  f"({m['wall_sec']:.1f}s)")

    # ---------- MULTISTREAM ----------
    out_ms = RES / f"p1_multistream_{name}.csv"
    # We need at least 4 GPU models for N=4 SizeBlindRev (4 NPU + 0 GPU? actually SBR places
    # 2 small on GPU and 2 large on NPU at N=4; at N=8 SBR places 4 small on GPU + 4 large
    # on NPU). So 4 GPU instances + 4 NPU instances.
    print(f"[{name}] MULTISTREAM phase: preloading 3 more GPU models…")
    gpu_models = [gpu] + [FGModelGPUGeneric(pt_name) for _ in range(3)]
    print(f"[{name}] MULTISTREAM phase → {out_ms}")
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
        if is_done(manifest, name, "multistream", key):
            continue
        splits = [load_split_for_sid(val, s) for s in sids]
        n = len(placement)
        # If N=8 we need 4 GPU + 4 NPU instances → already have 4 + 4
        if n == 8:
            cur_gpu = gpu_models[:4]
            cur_npu = npu_instances[:4]
        else:  # n=4: 2 gpu + 2 npu
            cur_gpu = gpu_models[:2]
            cur_npu = npu_instances[:2]
        try:
            agg = measure_multistream(sids, splits, placement, cur_gpu, cur_npu, bg)
        except Exception as e:
            print(f"  FAIL {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        row = {"detector": name, "params_M": params_M, "phase": "multistream",
               "n_streams": n, "bg_level": bg,
               "placement_name": pname, "placement_spec": json.dumps(placement),
               "mean_sap": round(agg["mean_sap"], 4),
               "worst_sap": round(agg["worst_sap"], 4),
               "wall_sec": round(agg["wall_sec"], 1)}
        # Append per-stream columns (s0..s7)
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
        mark_done(manifest, name, "multistream", key)
        print(f"  {tag} {bg} {pname}: worst={agg['worst_sap']:.4f}  mean={agg['mean_sap']:.4f}  "
              f"({agg['wall_sec']:.1f}s)")

    # Cleanup NPU engines before moving to next detector
    print(f"[{name}] disposing NPU engines…")
    dispose_npu_engines(name)


# ====================== top-level orchestration ============================

def main():
    manifest = load_manifest()

    print("[P1] preloading bg models max=L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[P1] bg preload {time.time()-t0:.1f}s")

    val = load_val()
    n_logs = len(val["sequences"])
    print(f"[P1] {n_logs} logs")

    t_all = time.time()
    for det_cfg in DETECTORS:
        try:
            sweep_detector(det_cfg, val, manifest)
        except Exception as e:
            print(f"[P1] DETECTOR {det_cfg['name']} FAILED at top level: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Continue with the next detector — record fact in manifest
            manifest.setdefault(det_cfg["name"], {})["fatal"] = str(e)[:200]
            save_manifest(manifest)
    print(f"\n[P1] sweep complete. wall_total={time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
