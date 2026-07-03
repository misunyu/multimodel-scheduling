"""rev25 — size-stratified sAP under real GPU contention (reviewer #6).

Direct evidence (not thread-lever proxy) that GPU staleness loss is large-biased
in the PARTIAL-contention regime, and size-neutral collapse at full saturation.

N=4, both devices threads=4. Contention via ResNet50 GPU-pressure (skip ~0/~50)
+ L3_vlm (Qwen2-VL) for ~100%. Capture per-size sAP (small/medium/large) for
All-GPU and All-NPU, averaged across the placement's streams. 3 reps.

Output:
  results/rev25_persize_under_contention.csv
  results/rev25_sanity.csv
  results/manifest_rev25.json
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)); sys.path.insert(0,str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                                dispose_npu_for, measure_single_stream, measure_multistream, DETECTORS)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import FPS

RES=Path("accv_experiments/results"); MAN=RES/"manifest_rev25.json"
DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; PERIOD=1000.0/FPS
PANEL4=[2,22,3,21]
# contention points: (label, bg_level_or_resnet_k)
POINTS=[("skip0","resnet0"),("skip~42","resnet1"),("skip~58","resnet2"),("skip~100","L3_vlm")]

def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except: pass
    return {"done":{}}
def save(m): MAN.write_text(json.dumps(m,indent=2))

def reg(spec):
    if spec.startswith("resnet"):
        k=int(spec[6:]); name=f"R{k}"; h2.BG_VARIANTS[name]=[_bg_resnet50_loop]*k; return name
    return spec  # L3_vlm already defined

def persize_for(agg, dev):
    streams=[s for s in agg["per_stream"] if s["device"]==dev]
    if not streams: return None
    return (float(np.mean([s["sap_s"] for s in streams])),
            float(np.mean([s["sap_m"] for s in streams])),
            float(np.mean([s["sap_l"] for s in streams])),
            float(np.mean([s["frame_skip_pct"] for s in streams])))

COLS=["point","strategy","rep","gpu_skip","npu_skip","sAP_small","sAP_medium","sAP_large"]
def app(row):
    new=not (RES/"rev25_persize_under_contention.csv").exists()
    with open(RES/"rev25_persize_under_contention.csv","a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(row)

def sanity(val,gpu,npu,tag):
    torch.set_num_threads(THREADS); set_active_npu_engines([npu])
    gl=[];nl=[];lat=[];sk=[]
    for sid in [2,3,21,22,13]:
        sp=load_split_for_sid(val,sid)
        mg=measure_single_stream(sid,sp,"GPU",gpu,"L0")
        set_active_npu_engines([npu]); mn=measure_single_stream(sid,sp,"NPU",npu,"L0")
        gl.append(mg["sap_l"]);nl.append(mn["sap_l"]);lat.append(mn["latency_mean"]);sk.append(mn["frame_skip_pct"])
    gap=float(np.mean(nl)-np.mean(gl))
    new=not (RES/"rev25_sanity.csv").exists()
    with open(RES/"rev25_sanity.csv","a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["tag","large_gap","npu_infer","npu_skip"]);
        if new:w.writeheader()
        w.writerow({"tag":tag,"large_gap":round(gap,4),"npu_infer":round(float(np.mean(lat)),2),"npu_skip":round(float(np.mean(sk)),2)})
    print(f"  SANITY {tag}: gap={gap:+.4f} infer={np.mean(lat):.1f}ms skip={np.mean(sk):.1f}%")

def main():
    torch.set_num_threads(THREADS)
    m=man()
    print(f"=== rev25 per-size under contention, threads={THREADS} ===")
    preload_background_models(max_level="L3")
    val=load_val()
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models=load_npu_engines(DET,DET["multistream_mxq"],DET["multistream_mode"],4)
    set_active_npu_engines(npu_models)
    sanity(val,gpu_models[0],npu_models[0],"pre")
    splits=[load_split_for_sid(val,s) for s in PANEL4]
    for label,spec in POINTS:
        lvl=reg(spec)
        for rep in range(N_REPS):
            key=f"{label}/rep{rep}"
            if m["done"].get(key): continue
            torch.set_num_threads(THREADS)
            for st,pl in [("All-GPU",["GPU"]*4),("All-NPU",["NPU"]*4)]:
                ng=sum(1 for d in pl if d=="GPU"); nn=sum(1 for d in pl if d=="NPU")
                agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],lvl)
                dev = "GPU" if st=="All-GPU" else "NPU"
                ps=persize_for(agg,dev)
                gs=float(np.mean([s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"])) if any(s["device"]=="GPU" for s in agg["per_stream"]) else 0.0
                ns=float(np.mean([s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"])) if any(s["device"]=="NPU" for s in agg["per_stream"]) else 0.0
                app({"point":label,"strategy":st,"rep":rep,"gpu_skip":round(gs,1),"npu_skip":round(ns,1),
                     "sAP_small":round(ps[0],4),"sAP_medium":round(ps[1],4),"sAP_large":round(ps[2],4)})
            m["done"][key]={"ts":int(time.time())}; save(m)
            print(f"  {label} rep{rep}: done ({lvl})")
    sanity(val,gpu_models[0],npu_models[0],"post")
    dispose_npu_for("yolo11s")
    print("=== rev25 done ===")

if __name__=="__main__":
    main()
