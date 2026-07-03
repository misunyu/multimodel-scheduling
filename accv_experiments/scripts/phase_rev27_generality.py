"""rev27 — detector-generality extension. Same protocol as rev26 (YOLOv8n), detector only changed.

STEP 0 guard (run separately): YOLOv8s vendor INT8 export available (cached, global8) -> measured.
YOLO26s has NO COCO vendor INT8 export (HF mobilint/YOLO26s 404; only custom fall-detection/weapon
mxq locally, non-COCO + head-shape mismatch) -> EXCLUDED, reported honestly.

Usage: python phase_rev27_generality.py <model_cls> <ultralytics_pt> <mxq_glob>
  e.g. python phase_rev27_generality.py YOLOv8s yolov8s.pt "*/models--mobilint--YOLOv8s/snapshots/*/aries/yolov8s.mxq"

Outputs (results/):
  rev27_<name>_single_stream.csv, rev27_<name>_contention.csv, rev27_<name>_sweep.csv
core eval scripts unmodified; prior 11s/8n artifacts / paper / results unchanged. threads=4 both devices.
"""
from __future__ import annotations
import csv, glob, json, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)); sys.path.insert(0,str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                                dispose_npu_for, measure_single_stream, measure_multistream)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import FPS

RES=Path("accv_experiments/results")
THREADS=4; N_REPS=3; PERIOD=1000.0/FPS
PANEL4=[2,22,3,21]; RESNET_K=[0,1,2,4,8]

CLS=sys.argv[1] if len(sys.argv)>1 else "YOLOv8s"
PT =sys.argv[2] if len(sys.argv)>2 else "yolov8s.pt"
GLOB=sys.argv[3] if len(sys.argv)>3 else str(Path.home()/".cache/huggingface/hub/models--mobilint--YOLOv8s/snapshots/*/aries/yolov8s.mxq")
NAME=CLS.lower()
MXQ=glob.glob(GLOB)[0]
DET={"name":NAME,"cls":CLS,"ultralytics_pt":PT,"baseline_mxq":MXQ,"baseline_mode":"global8",
     "multistream_mxq":MXQ,"multistream_mode":"global8"}

def reg(k): nm=f"R{k}"; h2.BG_VARIANTS[nm]=[_bg_resnet50_loop]*k; return nm
def dev_skips(agg):
    g=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"]
    n=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
    return (round(float(np.mean(g)),1) if g else 0.0, round(float(np.mean(n)),1) if n else 0.0)
def named(sids,k): return ["GPU"]*len(sids) if k=="All-GPU" else ["NPU"]*len(sids)
def app(path,cols,row):
    new=not path.exists()
    with open(path,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols)
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
    print(f"  SANITY {tag}: large_gap={np.mean(nl)-np.mean(gl):+.4f} npu_infer={np.mean(lat):.1f}ms npu_skip={np.mean(sk):.1f}%",flush=True)

def step1(val,gpu,npu):
    out=RES/f"rev27_{NAME}_single_stream.csv"
    if out.exists(): out.unlink()
    n_sids=len(val["sequences"]); perrep={d:{k:[] for k in ["s","m","l"]} for d in ["GPU","NPU"]}
    infskip={"infer":[],"skip":[]}
    for rep in range(N_REPS):
        torch.set_num_threads(THREADS); rep_={d:{k:[] for k in ["s","m","l"]} for d in ["GPU","NPU"]}
        for sid in range(n_sids):
            sp=load_split_for_sid(val,sid)
            for dev,model in [("GPU",gpu),("NPU",npu)]:
                if dev=="NPU": set_active_npu_engines([npu])
                m=measure_single_stream(sid,sp,dev,model,"L0")
                rep_[dev]["s"].append(m["sap_s"]);rep_[dev]["m"].append(m["sap_m"]);rep_[dev]["l"].append(m["sap_l"])
                if dev=="NPU": infskip["infer"].append(m["latency_mean"]);infskip["skip"].append(m["frame_skip_pct"])
        for dev in ["GPU","NPU"]:
            for k in ["s","m","l"]: perrep[dev][k].append(float(np.mean(rep_[dev][k])))
        print(f"  STEP1 rep{rep}: GPU l={np.mean(rep_['GPU']['l']):.4f} NPU l={np.mean(rep_['NPU']['l']):.4f}",flush=True)
    cols=["size","gpu_mean","npu_mean","gap","rel_gap_pct","gpu_std","npu_std","reps","npu_skip","npu_infer_ms"]
    for sz,k in [("small","s"),("medium","m"),("large","l")]:
        gm=float(np.mean(perrep["GPU"][k])); nm=float(np.mean(perrep["NPU"][k]))
        app(out,cols,{"size":sz,"gpu_mean":round(gm,4),"npu_mean":round(nm,4),"gap":round(nm-gm,4),
                      "rel_gap_pct":round((nm-gm)/gm*100,1) if gm else 0,
                      "gpu_std":round(float(np.std(perrep["GPU"][k])),4),"npu_std":round(float(np.std(perrep["NPU"][k])),4),
                      "reps":N_REPS,"npu_skip":round(float(np.mean(infskip["skip"])),1),
                      "npu_infer_ms":round(float(np.mean(infskip["infer"])),1)})
    print(f"  saved {out}",flush=True)

def step2(val,gpu_models,npu_models):
    out=RES/f"rev27_{NAME}_contention.csv"
    if out.exists(): out.unlink()
    cols=["bg","N","strategy","worst_sap","worst_std","mean_sap","gpu_skip","npu_skip","reps"]
    splits=[load_split_for_sid(val,s) for s in PANEL4]
    for bg in ["L3_vlm","L1_light","L2_lm"]:
        for st in ["All-GPU","All-NPU"]:
            worst=[];mean=[];gss=[];nss=[]
            for rep in range(N_REPS):
                torch.set_num_threads(THREADS); pl=named(PANEL4,st); ng=pl.count("GPU"); nn=pl.count("NPU")
                agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],bg)
                gs,ns=dev_skips(agg); worst.append(agg["worst_sap"]);mean.append(agg["mean_sap"]);gss.append(gs);nss.append(ns)
            app(out,cols,{"bg":bg,"N":4,"strategy":st,"worst_sap":round(float(np.mean(worst)),4),
                          "worst_std":round(float(np.std(worst)),4),"mean_sap":round(float(np.mean(mean)),4),
                          "gpu_skip":round(float(np.mean(gss)),1),"npu_skip":round(float(np.mean(nss)),1),"reps":N_REPS})
            print(f"  STEP2 {bg} {st}: worst={np.mean(worst):.4f} mean={np.mean(mean):.4f} GPUsk={np.mean(gss):.0f}% NPUsk={np.mean(nss):.0f}%",flush=True)
    print(f"  saved {out}",flush=True)

def step3(val,gpu_models,npu_models):
    out=RES/f"rev27_{NAME}_sweep.csv"
    if out.exists(): out.unlink()
    cols=["resnet_k","gpu_skip","strategy","worst_sap","worst_std","npu_skip","reps"]
    splits=[load_split_for_sid(val,s) for s in PANEL4]
    for k in RESNET_K:
        lvl=reg(k)
        for st in ["All-GPU","All-NPU"]:
            worst=[];gss=[];nss=[]
            for rep in range(N_REPS):
                torch.set_num_threads(THREADS); pl=named(PANEL4,st); ng=pl.count("GPU"); nn=pl.count("NPU")
                agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],lvl)
                gs,ns=dev_skips(agg); worst.append(agg["worst_sap"]);gss.append(gs);nss.append(ns)
            app(out,cols,{"resnet_k":k,"gpu_skip":round(float(np.mean(gss)),1),"strategy":st,
                          "worst_sap":round(float(np.mean(worst)),4),"worst_std":round(float(np.std(worst)),4),
                          "npu_skip":round(float(np.mean(nss)),1),"reps":N_REPS})
            print(f"  STEP3 k={k} {st}: GPUsk={np.mean(gss):.0f}% worst={np.mean(worst):.4f} NPUsk={np.mean(nss):.0f}%",flush=True)
    print(f"  saved {out}",flush=True)

def main():
    torch.set_num_threads(THREADS)
    print(f"=== rev27 {CLS} generality, threads={THREADS}, mxq={MXQ} ===",flush=True)
    preload_background_models(max_level="L3"); val=load_val()
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models=load_npu_engines(DET,DET["baseline_mxq"],DET["baseline_mode"],4)
    set_active_npu_engines(npu_models)
    sanity(val,gpu_models[0],npu_models[0],"pre")
    print("\n--- STEP 1 ---",flush=True); step1(val,gpu_models[0],npu_models[0])
    print("\n--- STEP 2 ---",flush=True); step2(val,gpu_models,npu_models)
    print("\n--- STEP 3 ---",flush=True); step3(val,gpu_models,npu_models)
    sanity(val,gpu_models[0],npu_models[0],"post")
    dispose_npu_for(NAME); print("=== rev27 done ===",flush=True)

if __name__=="__main__":
    main()
