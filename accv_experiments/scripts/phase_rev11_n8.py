"""rev11 — N=8 capacity·survival 3-rep confirmation in normal state.

GATE-Q (state-normal re-verification, yolo11s only):
  single-stream GPU FP32 vs NPU INT8 at L0, 24 logs.
  target rel large gap = -0.21 ± 0.03  (same as rev10 GATE-Q).
  FAIL → stop, do NOT report N=8 values.

A: N=8 L1_light × {Naive_allGPU, SizeAware (NPU 4), SizeBlindRev (NPU 4),
   AllNPU} × 3 reps. Records per-stream sAP + skip rate.

Outputs:
  results/rev11_n8.csv      raw rows
  results/rev11_n8_log.txt  per-line log
  results/manifest_rev11.json
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, measure_multistream,
                                DETECTORS)
from phase_rev10_measure import _native, short_sha

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_rev11.json"
LOG = RES / "rev11_n8_log.txt"
OUT_CSV = RES / "rev11_n8.csv"

# ---- fixed plan (per user spec) ----
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
COMP_A_N8 = [2, 22, 13, 16, 3, 21, 14, 4]
BG = "L1_light"
STRATS = [
    ("Naive_allGPU",  ["GPU"] * 8),
    ("SizeAware",     ["NPU", "NPU", "NPU", "NPU", "GPU", "GPU", "GPU", "GPU"]),
    ("SizeBlindRev",  ["GPU", "GPU", "GPU", "GPU", "NPU", "NPU", "NPU", "NPU"]),
    ("AllNPU",        ["NPU"] * 8),
]
N_REPS = 3
GATE_Q_TARGET = -0.21
GATE_Q_TOL    = 0.03

# Survival = per-stream skip-rate < SURVIVE_THR
SURVIVE_THR = 50.0


# ============================ manifest ============================

def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except Exception: pass
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(_native(m), indent=2))


def mark_done(manifest, key):
    manifest.setdefault("done", {})[key] = {"ts": int(time.time())}
    save_manifest(manifest)


def is_done(manifest, key):
    return key in manifest.get("done", {})


def append_log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts}  {msg}\n"
    with open(LOG, "a") as f: f.write(line)
    print(line, end="", flush=True)


def probe_env():
    info = {}
    try:
        info["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip()
    except Exception as e: info["driver"] = f"unknown ({e})"
    try:
        gu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
              "--format=csv,noheader,nounits"], text=True).strip()
        u, t = gu.split(",")
        info["gpu_util_pct"] = int(u.strip()); info["gpu_temp_c"] = int(t.strip())
    except Exception as e:
        info["gpu_util_pct"] = -1; info["gpu_temp_c"] = -1
    return info


# ============================ CSV writer ============================

def _row_cols():
    base = ["phase", "rep_idx", "n_streams", "bg_level", "infer_mode",
             "placement_name", "placement_spec",
             "mean_sap", "worst_sap", "wall_sec",
             "gpu_util_at_start", "gpu_temp_at_start"]
    per = [f"s{i}_{k}" for i in range(8)
            for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "map", "skip"]]
    return base + per


def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


# ============================ GATE-Q ============================

def run_gate_q(val):
    """yolo11s single-stream GPU + NPU at L0 over 24 logs, then compute large rel gap."""
    append_log("---- GATE-Q  single-stream L0 (yolo11s 24 logs × 2 dev) ----")
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)
    set_active_npu_engines(npu)
    gpu_sap_l = []; npu_sap_l = []
    for sid in range(len(val["sequences"])):
        split = load_split_for_sid(val, sid)
        for device, model, store in [("GPU", gpu, gpu_sap_l),
                                       ("NPU", npu[0], npu_sap_l)]:
            try:
                m = measure_single_stream(sid, split, device, model, "L0")
                store.append(m["sap_l"])
            except Exception as e:
                append_log(f"  FAIL gate sid={sid} {device}: {type(e).__name__}: {e}")
    dispose_npu_for(DET["name"])
    del gpu
    if len(gpu_sap_l) < 24 or len(npu_sap_l) < 24:
        return {"pass": False, "reason": f"incomplete (gpu={len(gpu_sap_l)} npu={len(npu_sap_l)})"}
    gl = float(np.mean(gpu_sap_l)); nl = float(np.mean(npu_sap_l))
    rel = (nl - gl) / gl if gl else float("nan")
    ok = abs(rel - GATE_Q_TARGET) <= GATE_Q_TOL
    return {"pass": bool(ok), "rel": rel, "gpu_l": gl, "npu_l": nl,
             "target": GATE_Q_TARGET, "tol": GATE_Q_TOL,
             "n_gpu": len(gpu_sap_l), "n_npu": len(npu_sap_l)}


# ============================ N=8 measurement ============================

def run_n8(val, manifest):
    append_log("---- A  N=8 L1_light × 4 strat × 3 reps ----")
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(8)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"],
                                    DET["multistream_mode"], 8)
    set_active_npu_engines(npu_models)
    append_log(f"[yolo11s] N=8 engines ready ({DET['multistream_mode']})")
    cols = _row_cols()
    splits = [load_split_for_sid(val, s) for s in COMP_A_N8]
    for rep in range(N_REPS):
        for pname, placement in STRATS:
            key = f"n8/{pname}/rep{rep}"
            if is_done(manifest, key): continue
            n_gpu = sum(1 for d in placement if d == "GPU")
            n_npu = sum(1 for d in placement if d == "NPU")
            env_pre = probe_env()
            try:
                agg = measure_multistream(COMP_A_N8, splits, placement,
                                            gpu_models[:n_gpu],
                                            npu_models[:n_npu], BG)
            except Exception as e:
                append_log(f"  FAIL {key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            row = {"phase": "rev11_n8", "rep_idx": rep, "n_streams": 8,
                    "bg_level": BG,
                    "infer_mode": DET["multistream_mode"],
                    "placement_name": pname,
                    "placement_spec": json.dumps(placement),
                    "mean_sap": round(agg["mean_sap"], 4),
                    "worst_sap": round(agg["worst_sap"], 4),
                    "wall_sec": round(agg["wall_sec"], 1),
                    "gpu_util_at_start": env_pre.get("gpu_util_pct", -1),
                    "gpu_temp_at_start": env_pre.get("gpu_temp_c", -1)}
            for s_info in agg["per_stream"]:
                i = s_info["stream_id"]
                row[f"s{i}_dev"] = s_info["device"]
                row[f"s{i}_sid"] = s_info["sid"]
                row[f"s{i}_sap"] = round(s_info["sap_5095"], 4)
                row[f"s{i}_sap_s"] = round(s_info["sap_s"], 4)
                row[f"s{i}_sap_m"] = round(s_info["sap_m"], 4)
                row[f"s{i}_sap_l"] = round(s_info["sap_l"], 4)
                row[f"s{i}_map"] = round(s_info.get("map_5095", 0), 4)
                row[f"s{i}_skip"] = round(s_info["frame_skip_pct"], 1)
            append_csv(OUT_CSV, cols, row)
            mark_done(manifest, key)
            survived = sum(1 for s in agg["per_stream"]
                            if s["frame_skip_pct"] < SURVIVE_THR)
            append_log(f"  {pname} rep{rep}: worst={agg['worst_sap']:.4f} "
                        f"mean={agg['mean_sap']:.4f} survived={survived}/8 "
                        f"({agg['wall_sec']:.1f}s)")
    dispose_npu_for(DET["name"])
    for m in gpu_models: del m


# ============================ main ============================

def main():
    manifest = load_manifest()
    env = probe_env()
    append_log(f"==== rev11 start ==== driver={env.get('driver')} "
                f"gpu_util={env.get('gpu_util_pct')}% temp={env.get('gpu_temp_c')}C")
    manifest["env_start"] = env

    append_log("[rev11] preloading bg models max=L1 (only L1_light needed)…")
    t0 = time.time()
    preload_background_models(max_level="L1")
    append_log(f"[rev11] bg preload {time.time()-t0:.1f}s")
    val = load_val()
    append_log(f"[rev11] {len(val['sequences'])} logs")

    # ---- GATE-Q ----
    gate_done = manifest.get("gate_q", {}).get("pass", None)
    if gate_done is None:
        g = run_gate_q(val)
        manifest["gate_q"] = g
        save_manifest(manifest)
        append_log(f"[GATE-Q]  pass={g.get('pass')}  rel={g.get('rel',0):+.3f}  "
                    f"target={GATE_Q_TARGET:+.3f}±{GATE_Q_TOL:.3f}  "
                    f"gpu_l={g.get('gpu_l',0):.4f} npu_l={g.get('npu_l',0):.4f}")
        if not g.get("pass"):
            append_log("[GATE-Q] FAIL — stopping. State not normal. "
                        "N=8 measurements NOT recorded.")
            return 1
    else:
        append_log(f"[GATE-Q] previously verified pass=True; skip re-measure.")

    # ---- N=8 ----
    run_n8(val, manifest)

    append_log("==== rev11 done ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
