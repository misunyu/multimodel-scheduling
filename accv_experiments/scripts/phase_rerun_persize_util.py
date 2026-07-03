"""Re-run of the N=4 ResNet50 GPU-contention sweep, adding (a) per-size sAP at
every k and (b) concurrent GPU-utilization sampling — WITHOUT editing any core
measurement script.

Reuses measure_multistream (which already computes per-stream sap_s/m/l and
frame_skip_pct; the original rev22 just did not log the size fields). Contention
lever is the SAME as rev22/rev25: k co-located ResNet50 GPU loops registered via
h2.BG_VARIANTS (dict mutation, no core edit). Both devices torch.set_num_threads(4).

GPU util is sampled by a SEPARATE pynvml process (taskset-pinned to an unused
core), running continuously during the sampler-ON phase. Each measurement window
records (t_start, t_end) so util can be sliced per cell afterward.

Non-interference: k in {2,8} are additionally run with the sampler OFF; the
ON-vs-OFF gpu_skip / worst_sap deltas are compared against the 3-rep noise.

Outputs (results/sweep_rerun/):
  rerun_raw.csv         per (phase,placement,k,rep): gpu_skip, per-stream/size sAP, worst, t_start/t_end
  util_samples.csv      raw pynvml samples (epoch,util_gpu,util_mem,mem_used,power)
  manifest_rerun.json
"""
from __future__ import annotations
import csv, json, sys, time, subprocess, signal, os
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import FPS

RES = Path("accv_experiments/results/sweep_rerun"); RES.mkdir(parents=True, exist_ok=True)
MAN = RES/"manifest_rerun.json"
DET = [d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS = 4; N_REPS = int(os.environ.get("RR_REPS", "3")); PERIOD = 1000.0/FPS
PANEL4 = [2,22,3,21]
RESNET_COUNTS = [int(x) for x in os.environ.get("RR_KS", "0,1,2,3,4,6,8").split(",")]
NI_KS = [int(x) for x in os.environ.get("RR_NI_KS", "2,8").split(",") if x != ""]
SAMPLER_CORE = "23"; SAMPLE_INTERVAL = 0.2

def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except Exception: pass
    return {"done":{}}
def save(m): MAN.write_text(json.dumps(m,indent=2))

def reg_level(k):
    name = f"RR{k}"; h2.BG_VARIANTS[name] = [_bg_resnet50_loop]*k; return name

RAW_COLS = ["phase","placement","k","rep","sampler","t_start","t_end","gpu_skip","npu_skip",
            "worst_sap","mean_sap",
            "sap_s","sap_m","sap_l",
            "s0_skip","s1_skip","s2_skip","s3_skip"]
def app(path, cols, row):
    new = not path.exists()
    with open(path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow(row)

def cell_summary(agg, dev):
    streams = [s for s in agg["per_stream"] if s["device"]==dev]
    skips_g = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"]
    skips_n = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
    return {
        "gpu_skip": round(float(np.mean(skips_g)),2) if skips_g else 0.0,
        "npu_skip": round(float(np.mean(skips_n)),2) if skips_n else 0.0,
        "worst_sap": round(agg["worst_sap"],4), "mean_sap": round(agg["mean_sap"],4),
        "sap_s": round(float(np.mean([s["sap_s"] for s in streams])),4),
        "sap_m": round(float(np.mean([s["sap_m"] for s in streams])),4),
        "sap_l": round(float(np.mean([s["sap_l"] for s in streams])),4),
        "per_stream_skip": [round(s["frame_skip_pct"],1) for s in streams],
    }

def run_cell(phase, placement_name, placement, k, rep, sids, splits, gpu_models, npu_models, sampler_on, m):
    key = f"{phase}/{placement_name}/k{k}/rep{rep}"
    if m["done"].get(key):
        print(f"  skip (done) {key}"); return
    lvl = reg_level(k)
    ng = sum(1 for d in placement if d=="GPU"); nn = sum(1 for d in placement if d=="NPU")
    torch.set_num_threads(THREADS)
    t0 = time.time()
    agg = measure_multistream(sids, splits, placement, gpu_models[:ng], npu_models[:nn], lvl)
    t1 = time.time()
    dev = "GPU" if placement_name.startswith("All-GPU") else "NPU"
    cs = cell_summary(agg, dev)
    row = {"phase":phase,"placement":placement_name,"k":k,"rep":rep,
           "sampler":"ON" if sampler_on else "OFF",
           "t_start":round(t0,3),"t_end":round(t1,3),
           "gpu_skip":cs["gpu_skip"],"npu_skip":cs["npu_skip"],
           "worst_sap":cs["worst_sap"],"mean_sap":cs["mean_sap"],
           "sap_s":cs["sap_s"],"sap_m":cs["sap_m"],"sap_l":cs["sap_l"]}
    for i in range(4):
        row[f"s{i}_skip"] = cs["per_stream_skip"][i] if i < len(cs["per_stream_skip"]) else ""
    app(RES/"rerun_raw.csv", RAW_COLS, row)
    m["done"][key] = {"ts":int(time.time())}; save(m)
    print(f"  {key} sampler={'ON' if sampler_on else 'OFF'}: gpu_skip={cs['gpu_skip']} "
          f"worst={cs['worst_sap']} sap_s/m/l={cs['sap_s']}/{cs['sap_m']}/{cs['sap_l']} ({t1-t0:.1f}s)")

def start_sampler():
    out = RES/"util_samples.csv"
    p = subprocess.Popen(["taskset","-c",SAMPLER_CORE,str(SCRIPT_DIR.parent.parent/".venv/bin/python"),
                          str(SCRIPT_DIR/"gpu_util_sampler.py"), str(out), str(SAMPLE_INTERVAL)])
    time.sleep(2.0)  # let it spin up
    return p

def stop_sampler(p):
    if p is None: return
    p.send_signal(signal.SIGTERM)
    try: p.wait(timeout=5)
    except Exception: p.kill()

def main():
    torch.set_num_threads(THREADS)
    m = man()
    print(f"=== rerun per-size + util, threads={THREADS}, period={PERIOD:.1f}ms ===")
    preload_background_models(max_level="L1")  # resnet50 only, same as rev22
    val = load_val()
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    ALLGPU = ["GPU"]*4; ALLNPU = ["NPU"]*4

    # ---------- PHASE A: sampler ON ----------
    print("\n--- PHASE A: sampler ON ---")
    samp = start_sampler()
    try:
        for k in RESNET_COUNTS:
            for rep in range(N_REPS):
                run_cell("A","All-GPU",ALLGPU,k,rep,PANEL4,splits,gpu_models,npu_models,True,m)
        # All-NPU flat reference (lowest & highest k present, to show insensitivity to GPU contention)
        npu_ref_ks = sorted(set([RESNET_COUNTS[0], RESNET_COUNTS[-1]]))
        for k in npu_ref_ks:
            for rep in range(N_REPS):
                run_cell("A","All-NPU",ALLNPU,k,rep,PANEL4,splits,gpu_models,npu_models,True,m)
    finally:
        stop_sampler(samp)
    print("--- sampler stopped ---")

    # ---------- PHASE B: sampler OFF (non-interference check) ----------
    if NI_KS:
        print(f"\n--- PHASE B: sampler OFF (non-interference k={NI_KS}) ---")
        for k in NI_KS:
            for rep in range(N_REPS):
                run_cell("B","All-GPU-OFF",ALLGPU,k,rep,PANEL4,splits,gpu_models,npu_models,False,m)

    dispose_npu_for("yolo11s")
    print("=== rerun done ===")

if __name__ == "__main__":
    main()
