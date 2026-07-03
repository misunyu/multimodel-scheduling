"""rev10 — re-measure §7 family (s/m/l/x) in normal state.

A-1  single-stream GPU FP32 vs NPU INT8 at L0, 24 logs, per detector.
     (GATE-Q on yolo11s: large rel gap = -21% ± 3%.)
A-2  multi-stream N=4 Comp A × {Naive, SizeAware, SizeBlindRev, AllNPU} × {L1_light, L2_lm}
     × 3 reps per detector.
     (GATE-S on yolo11s: worst gain SA-SBR at L1_light reproduces rev9 +0.002 ± 0.005.)
A-3  single-stream NPU+GPU on N=4 Comp A panel across ladder {L1_light, L1_heavy,
     L2_lm, L3_vlm} per detector (per-group gap; L0 reused from A-1).

Outputs (manifest-checkpointed):
  results/rev10_gen_single.csv
  results/rev10_gen_gain.csv
  results/rev10_cstar.csv
  results/manifest_rev10.json
  results/rev10_log.txt

DOES NOT modify step0~step_h2. Reuses utilities from phase_rev6_sweep.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
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

# Reuse rev6 utilities
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, measure_multistream,
                                DETECTORS)

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_rev10.json"
LOG = RES / "rev10_log.txt"

# ---------- plan ----------
COMP_A_N4 = [2, 22, 3, 21]
SIZE_GROUP = {2: "small-rich", 22: "small-rich",
              3: "large-rich", 21: "large-rich"}
LADDER_FG = ["L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
A2_BG     = ["L1_light", "L2_lm"]
A2_STRAT = [
    ("Naive_allGPU", ["GPU", "GPU", "GPU", "GPU"]),
    ("SizeAware",     ["NPU", "NPU", "GPU", "GPU"]),
    ("SizeBlindRev",  ["GPU", "GPU", "NPU", "NPU"]),
    ("AllNPU",        ["NPU", "NPU", "NPU", "NPU"]),
]
N_REPS = 3

# Gates (only applied for yolo11s)
GATE_Q_LARGE_TARGET = -0.21      # rel gap target from T-D0
GATE_Q_LARGE_TOL    = 0.03
GATE_S_TARGET       = +0.002     # rev9 SA-SBR worst gain at L1_light
GATE_S_TOL          = 0.005


# ============================ manifest ============================

def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except Exception: pass
    return {}


def _native(obj):
    """Convert numpy/bool types to JSON-serializable native Python types."""
    if isinstance(obj, dict): return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_native(x) for x in obj]
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj


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
    print(line, end="")


# ============================ env probe ============================

def probe_env():
    """Driver + GPU idle at start. Recorded once per run start."""
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
        info["gpu_util_pct"] = int(u.strip())
        info["gpu_temp_c"]   = int(t.strip())
    except Exception as e: info["gpu_util_pct"] = -1; info["gpu_temp_c"] = -1
    return info


# ============================ CSV writers ============================

A1_COLS = ["detector", "params_M", "phase", "rep_idx", "sid", "device",
            "infer_mode", "mxq_path", "mxq_sha8", "bg_level",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_s", "map_m", "map_l",
            "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec"]

A2_COLS = (["detector", "params_M", "phase", "rep_idx",
             "n_streams", "bg_level", "infer_mode", "mxq_path",
             "placement_name", "placement_spec",
             "mean_sap", "worst_sap", "wall_sec"]
            + [f"s{i}_{k}" for i in range(4)
                for k in ["dev", "sid", "sap", "sap_s", "sap_m", "sap_l", "skip"]])

A3_COLS = A1_COLS[:]


def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def short_sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()[:8]


# ============================ PART A-1 ============================

def run_a1(val, manifest):
    out = RES / "rev10_gen_single.csv"
    append_log("---- A-1 single-stream L0 (4 detectors × 2 devices × 24 logs) ----")
    for det in DETECTORS:
        name = det["name"]; pm = det["params_M"]
        gpu = FGModelGPUGeneric(det["ultralytics_pt"])
        try:
            sha8 = short_sha(det["baseline_mxq"])
        except Exception as e:
            sha8 = f"err_{e}"
        npu = load_npu_engines(det, det["baseline_mxq"], det["baseline_mode"], 1)
        set_active_npu_engines(npu)
        append_log(f"[{name}] sha8={sha8}  mode={det['baseline_mode']}")
        for sid in range(len(val["sequences"])):
            split = load_split_for_sid(val, sid)
            for device, model in [("GPU", gpu), ("NPU", npu[0])]:
                key = f"a1/{name}/{device}/{sid}"
                if is_done(manifest, key): continue
                try:
                    m = measure_single_stream(sid, split, device, model, "L0")
                except Exception as e:
                    append_log(f"  FAIL {key}: {type(e).__name__}: {e}")
                    continue
                row = {"detector": name, "params_M": pm, "phase": "a1",
                        "rep_idx": 0, "sid": sid, "device": device,
                        "infer_mode": det["baseline_mode"] if device == "NPU" else "",
                        "mxq_path": det["baseline_mxq"] if device == "NPU" else "",
                        "mxq_sha8": sha8 if device == "NPU" else "",
                        "bg_level": "L0",
                        **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in m.items()}}
                append_csv(out, A1_COLS, row)
                mark_done(manifest, key)
        dispose_npu_for(name)
        del gpu
    return out


def gate_q(out_a1):
    """yolo11s large gap = NPU - GPU at L0; rel = (NPU - GPU)/GPU."""
    df = pd.read_csv(out_a1)
    s = df[df.detector == "yolo11s"]
    g = s[s.device == "GPU"]; n = s[s.device == "NPU"]
    if len(g) < 24 or len(n) < 24:
        return {"pass": False, "reason": f"incomplete (gpu={len(g)} npu={len(n)})"}
    gpu_l = g["sap_l"].mean(); npu_l = n["sap_l"].mean()
    rel = (npu_l - gpu_l) / gpu_l if gpu_l else float("nan")
    ok = abs(rel - GATE_Q_LARGE_TARGET) <= GATE_Q_LARGE_TOL
    return {"pass": ok, "rel": float(rel), "target": GATE_Q_LARGE_TARGET,
             "tol": GATE_Q_LARGE_TOL,
             "gpu_l": float(gpu_l), "npu_l": float(npu_l)}


# ============================ PART A-2 ============================

def run_a2(val, manifest):
    out = RES / "rev10_gen_gain.csv"
    append_log("---- A-2 multi-stream N=4 Comp A × 4 strat × 2 bg × 3 reps ----")
    for det in DETECTORS:
        name = det["name"]; pm = det["params_M"]
        gpu_models = [FGModelGPUGeneric(det["ultralytics_pt"]) for _ in range(4)]
        npu_models = load_npu_engines(det, det["multistream_mxq"],
                                        det["multistream_mode"], 4)
        set_active_npu_engines(npu_models)
        append_log(f"[{name}] A-2 npu engines loaded ({det['multistream_mode']})")
        for bg in A2_BG:
            for rep in range(N_REPS):
                for pname, placement in A2_STRAT:
                    key = f"a2/{name}/{bg}/{pname}/rep{rep}"
                    if is_done(manifest, key): continue
                    splits = [load_split_for_sid(val, s) for s in COMP_A_N4]
                    n_gpu = sum(1 for d in placement if d == "GPU")
                    n_npu = sum(1 for d in placement if d == "NPU")
                    try:
                        agg = measure_multistream(COMP_A_N4, splits, placement,
                                                    gpu_models[:n_gpu],
                                                    npu_models[:n_npu], bg)
                    except Exception as e:
                        append_log(f"  FAIL {key}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        continue
                    row = {"detector": name, "params_M": pm, "phase": "a2",
                            "rep_idx": rep, "n_streams": 4, "bg_level": bg,
                            "infer_mode": det["multistream_mode"],
                            "mxq_path": det["multistream_mxq"],
                            "placement_name": pname,
                            "placement_spec": json.dumps(placement),
                            "mean_sap": round(agg["mean_sap"], 4),
                            "worst_sap": round(agg["worst_sap"], 4),
                            "wall_sec": round(agg["wall_sec"], 1)}
                    for s_info in agg["per_stream"]:
                        i = s_info["stream_id"]
                        row[f"s{i}_dev"] = s_info["device"]
                        row[f"s{i}_sid"] = s_info["sid"]
                        row[f"s{i}_sap"] = round(s_info["sap_5095"], 4)
                        row[f"s{i}_sap_s"] = round(s_info["sap_s"], 4)
                        row[f"s{i}_sap_m"] = round(s_info["sap_m"], 4)
                        row[f"s{i}_sap_l"] = round(s_info["sap_l"], 4)
                        row[f"s{i}_skip"] = round(s_info["frame_skip_pct"], 1)
                    append_csv(out, A2_COLS, row)
                    mark_done(manifest, key)
                    append_log(f"  {name} {bg} {pname} rep{rep}: "
                                f"worst={agg['worst_sap']:.4f} "
                                f"mean={agg['mean_sap']:.4f} "
                                f"({agg['wall_sec']:.1f}s)")
        dispose_npu_for(name)
        for m in gpu_models: del m
    return out


def gate_s(out_a2):
    """yolo11s SA - SBR worst gain at L1_light."""
    df = pd.read_csv(out_a2)
    s = df[(df.detector == "yolo11s") & (df.bg_level == "L1_light")]
    sa = s[s.placement_name == "SizeAware"]["worst_sap"]
    sbr = s[s.placement_name == "SizeBlindRev"]["worst_sap"]
    if len(sa) < 1 or len(sbr) < 1:
        return {"pass": False, "reason": "incomplete"}
    sa_mean = float(sa.mean()); sbr_mean = float(sbr.mean())
    gain = sa_mean - sbr_mean
    ok = abs(gain - GATE_S_TARGET) <= GATE_S_TOL
    return {"pass": ok, "gain": gain, "sa_mean": sa_mean, "sbr_mean": sbr_mean,
             "target": GATE_S_TARGET, "tol": GATE_S_TOL,
             "n_sa": int(len(sa)), "n_sbr": int(len(sbr))}


# ============================ PART A-3 ============================

def run_a3(val, manifest):
    out = RES / "rev10_cstar.csv"
    append_log("---- A-3 single-stream NPU+GPU ladder (4 det × 4 bg × 2 dev × 4 sids) ----")
    for det in DETECTORS:
        name = det["name"]; pm = det["params_M"]
        try:
            sha8 = short_sha(det["baseline_mxq"])
        except Exception: sha8 = "err"
        gpu = FGModelGPUGeneric(det["ultralytics_pt"])
        npu = load_npu_engines(det, det["baseline_mxq"], det["baseline_mode"], 1)
        set_active_npu_engines(npu)
        append_log(f"[{name}] A-3 mxq sha8={sha8} mode={det['baseline_mode']}")
        for sid in COMP_A_N4:
            split = load_split_for_sid(val, sid)
            for bg in LADDER_FG:
                for device, model in [("GPU", gpu), ("NPU", npu[0])]:
                    key = f"a3/{name}/{device}/{bg}/sid{sid}"
                    if is_done(manifest, key): continue
                    try:
                        m = measure_single_stream(sid, split, device, model, bg)
                    except Exception as e:
                        append_log(f"  FAIL {key}: {type(e).__name__}: {e}")
                        continue
                    row = {"detector": name, "params_M": pm, "phase": "a3",
                            "rep_idx": 0, "sid": sid, "device": device,
                            "infer_mode": det["baseline_mode"] if device == "NPU" else "",
                            "mxq_path": det["baseline_mxq"] if device == "NPU" else "",
                            "mxq_sha8": sha8 if device == "NPU" else "",
                            "bg_level": bg,
                            **{k: round(v, 4) if isinstance(v, float) else v
                                for k, v in m.items()}}
                    append_csv(out, A3_COLS, row)
                    mark_done(manifest, key)
        dispose_npu_for(name)
        del gpu
    return out


# ============================ main ============================

def main():
    manifest = load_manifest()
    env = probe_env()
    append_log(f"==== rev10 start ==== driver={env.get('driver')} "
                f"gpu_util={env.get('gpu_util_pct')}% temp={env.get('gpu_temp_c')}C")
    manifest["env_start"] = env; save_manifest(manifest)

    append_log("[rev10] preloading bg models max=L3…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    append_log(f"[rev10] bg preload {time.time()-t0:.1f}s")

    val = load_val()
    append_log(f"[rev10] {len(val['sequences'])} logs")

    # ---- A-1 ----
    out_a1 = run_a1(val, manifest)
    g = gate_q(out_a1)
    manifest["gate_q"] = g; save_manifest(manifest)
    append_log(f"[GATE-Q]  pass={g.get('pass')}  rel={g.get('rel', 0):+.3f}  "
                f"target={GATE_Q_LARGE_TARGET:+.3f}±{GATE_Q_LARGE_TOL:.3f}")
    if not g.get("pass"):
        append_log("[GATE-Q] FAIL — stopping. State not normal. No further measurement.")
        return 1

    # ---- A-2 ----
    out_a2 = run_a2(val, manifest)
    g = gate_s(out_a2)
    manifest["gate_s"] = g; save_manifest(manifest)
    append_log(f"[GATE-S]  pass={g.get('pass')}  gain={g.get('gain', 0):+.4f}  "
                f"target={GATE_S_TARGET:+.4f}±{GATE_S_TOL:.4f}")
    if not g.get("pass"):
        append_log("[GATE-S] FAIL — both values reported; not stopping (per spec, only state-not-normal is hard stop).")

    # ---- A-3 ----
    out_a3 = run_a3(val, manifest)

    append_log("==== rev10 done ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
