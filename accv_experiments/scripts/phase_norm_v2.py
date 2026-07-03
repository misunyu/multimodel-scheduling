"""EXP-NORM v2 — worst-stream cause separation: content difficulty vs contention (W5).

Re-measures the 4 cells of Table 5 (tab:metric-sensitivity) at per-stream + per-size
granularity (existing per-stream files lack per-size), 3 reps each, NO protection.
  U-GPU : All-GPU, uncontended (L0)
  U-NPU : All-NPU, uncontended (L0)
  C-GPU : All-GPU, resnet_k=8 (SWEEP8, GPU skip ~80%)  -- Table 5 high-contention point
  C-NPU : All-NPU, resnet_k=8
Reuses rev22/rev6 measurement path exactly; core scripts unmodified.
Output: results/exp_norm_results.csv
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import FPS

RES = Path("accv_experiments/results")
OUT = RES / "exp_norm_results.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
THREADS = 4; N_REPS = 3
PANEL4 = [2, 22, 3, 21]
K_HI = 8
# rev22 reference worst-stream means for the ±0.006 sanity (§2)
REF_WORST = {"U-GPU": 0.1149, "C-GPU": 0.0663, "U-NPU": 0.0836, "C-NPU": 0.0836}

COLS = ["cell_id", "placement", "resnet_k", "repeat", "stream_id", "log_name",
        "sap", "sap_small", "sap_medium", "sap_large", "fg_skip_pct",
        "data_source", "run_timestamp", "notes"]

def app(row):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})

def reg_sweep(k):
    name = f"SWEEP{k}"; h2.BG_VARIANTS[name] = [_bg_resnet50_loop] * k; return name

def main():
    torch.set_num_threads(THREADS)
    print(f"=== EXP-NORM v2 threads={THREADS} panel={PANEL4} k_hi={K_HI} ===")
    preload_background_models(max_level="L1")  # resnet only (SWEEP8 cotenant)
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print(f"[panel] logs={[sp['log_name'] for sp in splits]}")
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    print("[npu] loading 4 engines (single mode)…")
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)

    bg_hi = reg_sweep(K_HI)
    cells = [
        ("U-GPU", ["GPU"] * 4, "L0", 0),
        ("U-NPU", ["NPU"] * 4, "L0", 0),
        ("C-GPU", ["GPU"] * 4, bg_hi, K_HI),
        ("C-NPU", ["NPU"] * 4, bg_hi, K_HI),
    ]
    sanity = {}
    for cell_id, placement, bg, rk in cells:
        ng = sum(1 for d in placement if d == "GPU"); nn = sum(1 for d in placement if d == "NPU")
        worsts = []
        for rep in range(N_REPS):
            torch.set_num_threads(THREADS)
            agg = measure_multistream(PANEL4, splits, placement, gpu_models[:ng], npu_models[:nn], bg)
            ts = int(time.time())
            for s in agg["per_stream"]:
                app({"cell_id": cell_id, "placement": "All-GPU" if ng else "All-NPU",
                     "resnet_k": rk, "repeat": rep, "stream_id": s["stream_id"],
                     "log_name": splits[s["stream_id"]]["log_name"],
                     "sap": round(s["sap_5095"], 4), "sap_small": round(s["sap_s"], 4),
                     "sap_medium": round(s["sap_m"], 4), "sap_large": round(s["sap_l"], 4),
                     "fg_skip_pct": round(s["frame_skip_pct"], 1),
                     "data_source": "measured", "run_timestamp": ts, "notes": ""})
            worsts.append(agg["worst_sap"])
            print(f"  {cell_id} rep{rep}: worst={agg['worst_sap']:.4f} mean={agg['mean_sap']:.4f} "
                  f"perstream={[round(s['sap_5095'],4) for s in agg['per_stream']]}")
        mw = float(np.mean(worsts)); ref = REF_WORST[cell_id]; diff = abs(mw - ref); ok = diff <= 0.006
        sanity[cell_id] = {"worst_mean": round(mw, 4), "ref": ref, "abs_diff": round(diff, 4), "pass": ok}
        print(f"  >> {cell_id} SANITY: worst mean={mw:.4f} vs rev22 {ref:.4f} |diff|={diff:.4f} PASS={ok}")
    dispose_npu_for("yolo11s")
    print("\n=== SANITY SUMMARY ===")
    for c, v in sanity.items(): print(f"  {c}: {v}")
    if not all(v["pass"] for v in sanity.values()):
        print("*** A cell failed sanity (>0.006 from rev22) — REPORT; environment may have drifted. ***")
    print("=== EXP-NORM done ===")

if __name__ == "__main__":
    main()
