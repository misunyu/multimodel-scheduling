"""EXP-FT-COMPILER-CHECK Step 4 — equivalence: local-compiled mxq vs vendor b2441f9d.
Table 1 protocol: isolated single-stream NPU, 24 logs, threads=4 (skip~0), 3 reps, per-size sAP.
NPU only (GPU FP32 reference reused from published Table 1). global8 mode (= Table 1 baseline).
Output: results/compiler_check_results.csv
"""
import sys, csv, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (load_npu_engines, set_active_npu_engines, dispose_npu_for,
                              measure_single_stream, DETECTORS)

RES = Path("accv_experiments/results")
OUT = RES / "compiler_check_results.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
THREADS = 4; N_REPS = 3; MODE = "global8"
VENDOR = "models/mobilint_backup/yolo11s.mxq"   # b2441f9d (Table 1 source)
LOCAL  = "accv_experiments/results/qbc_local_yolo11s_global8.mxq"
COLS = ["mxq_label","mxq_path","rep","sid","sap","sap_small","sap_medium","sap_large",
        "npu_infer_ms","skip_pct","run_timestamp"]

def app(row):
    new = not OUT.exists()
    with open(OUT,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(row)

def run_one(label, mxq_path, val, sids):
    print(f"\n=== {label}: {mxq_path} (mode={MODE}, threads={THREADS}) ===", flush=True)
    eng = load_npu_engines(DET, mxq_path, MODE, 1)[0]
    set_active_npu_engines([eng])
    for rep in range(N_REPS):
        for sid in sids:
            torch.set_num_threads(THREADS)
            sp = load_split_for_sid(val, sid)
            m = measure_single_stream(sid, sp, "NPU", eng, "L0")
            app({"mxq_label":label,"mxq_path":mxq_path,"rep":rep,"sid":sid,
                 "sap":round(m["sap_5095"],4),"sap_small":round(m["sap_s"],4),
                 "sap_medium":round(m["sap_m"],4),"sap_large":round(m["sap_l"],4),
                 "npu_infer_ms":round(m["latency_mean"],2),"skip_pct":round(m["frame_skip_pct"],1),
                 "run_timestamp":int(time.time())})
        print(f"  {label} rep{rep} done", flush=True)
    dispose_npu_for("yolo11s")

def main():
    torch.set_num_threads(THREADS)
    preload_background_models(max_level="L0")
    val = load_val()
    sids = list(range(len(val["sequences"])))  # 24 logs
    print(f"[info] {len(sids)} logs, 3 reps, NPU-only, mode={MODE}")
    run_one("vendor_b2441f9d", VENDOR, val, sids)
    run_one("local_qbc", LOCAL, val, sids)
    print("=== equivalence eval done ===")

if __name__ == "__main__":
    main()
