"""Smoke test for phase_rev10_measure — yolo11s single-stream L0 on sid 0
+ N=4 L1_light SizeAware single rep. Confirms imports, model loads, CSV
write paths. Does NOT touch manifest_rev10.json.

Expected: completes in < 60s.
"""

from __future__ import annotations

import sys, time, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, measure_multistream,
                                DETECTORS)
from phase_rev10_measure import (probe_env, append_log, A1_COLS, A2_COLS,
                                   append_csv, short_sha, COMP_A_N4)

RES = Path("accv_experiments/results")
OUT_A1 = RES / "_smoke_rev10_a1.csv"
OUT_A2 = RES / "_smoke_rev10_a2.csv"
for p in (OUT_A1, OUT_A2):
    if p.exists(): p.unlink()

t_total = time.time()
env = probe_env()
print(f"env: {env}")

print("preloading bg max=L1 (smoke, not L3)…")
preload_background_models(max_level="L1")

val = load_val()
print(f"val: {len(val['sequences'])} logs")

det = DETECTORS[0]  # yolo11s
name = det["name"]; pm = det["params_M"]
print(f"\n=== det: {name} ===")

t = time.time()
print("loading GPU model…")
gpu = FGModelGPUGeneric(det["ultralytics_pt"])
print(f"  GPU load {time.time()-t:.1f}s")

t = time.time()
print("loading 1 NPU engine baseline mode…")
npu = load_npu_engines(det, det["baseline_mxq"], det["baseline_mode"], 1)
set_active_npu_engines(npu)
print(f"  NPU load {time.time()-t:.1f}s")

# --- A-1 smoke: 1 sid × 2 device × L0 ---
sid = 0
split = load_split_for_sid(val, sid)
sha8 = short_sha(det["baseline_mxq"])

for device, model in [("GPU", gpu), ("NPU", npu[0])]:
    t = time.time()
    m = measure_single_stream(sid, split, device, model, "L0")
    elapsed = time.time() - t
    row = {"detector": name, "params_M": pm, "phase": "a1",
            "rep_idx": 0, "sid": sid, "device": device,
            "infer_mode": det["baseline_mode"] if device == "NPU" else "",
            "mxq_path": det["baseline_mxq"] if device == "NPU" else "",
            "mxq_sha8": sha8 if device == "NPU" else "",
            "bg_level": "L0",
            **{k: round(v, 4) if isinstance(v, float) else v
                for k, v in m.items()}}
    append_csv(OUT_A1, A1_COLS, row)
    print(f"  A-1 {device} sid={sid}: sap={m['sap_5095']:.4f} sap_l={m['sap_l']:.4f} ({elapsed:.1f}s)")

dispose_npu_for(name)

# --- A-2 smoke: N=4 L1_light SizeAware 1 rep ---
print("\nloading 4 GPU + 4 NPU for N=4 multistream…")
t = time.time()
gpu_models = [gpu] + [FGModelGPUGeneric(det["ultralytics_pt"]) for _ in range(3)]
npu_models = load_npu_engines(det, det["multistream_mxq"], det["multistream_mode"], 4)
set_active_npu_engines(npu_models)
print(f"  multistream load {time.time()-t:.1f}s")

splits = [load_split_for_sid(val, s) for s in COMP_A_N4]
placement = ["NPU", "NPU", "GPU", "GPU"]
t = time.time()
agg = measure_multistream(COMP_A_N4, splits, placement,
                            gpu_models[:2], npu_models[:2], "L1_light")
print(f"  A-2 N=4 SizeAware L1_light: worst={agg['worst_sap']:.4f} "
        f"mean={agg['mean_sap']:.4f} ({time.time()-t:.1f}s)")
row = {"detector": name, "params_M": pm, "phase": "a2",
        "rep_idx": 0, "n_streams": 4, "bg_level": "L1_light",
        "infer_mode": det["multistream_mode"],
        "mxq_path": det["multistream_mxq"],
        "placement_name": "SizeAware",
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
append_csv(OUT_A2, A2_COLS, row)

dispose_npu_for(name)

print(f"\n=== smoke done in {time.time()-t_total:.1f}s ===")
print(f"a1 csv: {OUT_A1.stat().st_size} bytes")
print(f"a2 csv: {OUT_A2.stat().st_size} bytes")
