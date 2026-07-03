"""rev26 — detector-generality check on YOLOv8n (vendor INT8 export).

YOLO11n has no published INT8 export (HF 404); per user decision we use the
vendor-published YOLOv8n mxq (cached) as the smaller-detector generality check.
Same protocol as YOLOv11s (rev19/rev22/rev25): both devices threads=4, same
co-tenants, same N, same ResNet50 GPU-pressure sweep lever. detector ONLY changed.

This is a REPRODUCIBILITY check, not a performance measurement: we test whether
the DIRECTION reproduces (isolated=GPU on small / large tie; VLM contention=NPU),
not absolute values. Report honestly either way.

STEP 1  single-stream per-size (Table 1 equiv)        -> rev26_yolov8n_single_stream.csv
STEP 2  VLM contention worst/mean (Table 3 L3 equiv)  -> rev26_yolov8n_contention.csv
STEP 3  ResNet sweep crossover (Fig.2 equiv)          -> rev26_yolov8n_sweep.csv

core eval scripts unmodified; YOLOv11s artifacts / paper / results unchanged.
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

RES=Path("accv_experiments/results"); MAN=RES/"manifest_rev26.json"
THREADS=4; N_REPS=3; PERIOD=1000.0/FPS

MXQ=glob.glob(str(Path.home()/".cache/huggingface/hub/models--mobilint--YOLOv8n/snapshots/*/aries/yolov8n.mxq"))[0]
DET={"name":"yolov8n","cls":"YOLOv8n","ultralytics_pt":"yolov8n.pt",
     "baseline_mxq":MXQ,"baseline_mode":"global8",
     "multistream_mxq":MXQ,"multistream_mode":"global8"}
PANEL4=[2,22,3,21]; SIZE_GROUP={2:"small",22:"small",3:"large",21:"large"}
RESNET_K=[0,1,2,4,8]

def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except: pass
    return {"done":{}}
def save(m): MAN.write_text(json.dumps(m,indent=2))
def reg(k): name=f"R{k}"; h2.BG_VARIANTS[name]=[_bg_resnet50_loop]*k; return name
def dev_skips(agg):
    g=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"]
    n=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
    return (round(float(np.mean(g)),1) if g else 0.0, round(float(np.mean(n)),1) if n else 0.0)
def named(sids,k):
    return ["GPU"]*len(sids) if k=="All-GPU" else ["NPU"]*len(sids)
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

# ---------- STEP 1 ----------
def step1(val,gpu,npu):
    out=RES/"rev26_yolov8n_single_stream.csv"
    if out.exists(): out.unlink()
    n_sids=len(val["sequences"])
    acc={d:{k:[] for k in ["s","m","l","infer","skip"]} for d in ["GPU","NPU"]}
    perrep={d:{k:[] for k in ["s","m","l"]} for d in ["GPU","NPU"]}
    for rep in range(N_REPS):
        torch.set_num_threads(THREADS)
        repacc={d:{k:[] for k in ["s","m","l","infer","skip"]} for d in ["GPU","NPU"]}
        for sid in range(n_sids):
            sp=load_split_for_sid(val,sid)
            for dev,model in [("GPU",gpu),("NPU",npu)]:
                if dev=="NPU": set_active_npu_engines([npu])
                m=measure_single_stream(sid,sp,dev,model,"L0")
                repacc[dev]["s"].append(m["sap_s"]); repacc[dev]["m"].append(m["sap_m"]); repacc[dev]["l"].append(m["sap_l"])
                repacc[dev]["infer"].append(m["latency_mean"]); repacc[dev]["skip"].append(m["frame_skip_pct"])
        for dev in ["GPU","NPU"]:
            for k in ["s","m","l"]: perrep[dev][k].append(float(np.mean(repacc[dev][k])))
            for k in ["s","m","l","infer","skip"]: acc[dev][k]+=repacc[dev][k]
        print(f"  STEP1 rep{rep}: GPU l={np.mean(repacc['GPU']['l']):.4f} NPU l={np.mean(repacc['NPU']['l']):.4f}",flush=True)
    cols=["size","gpu_mean","npu_mean","gap","rel_gap_pct","gpu_std","npu_std","reps","npu_skip","npu_infer_ms"]
    for sz,k in [("small","s"),("medium","m"),("large","l")]:
        gm=float(np.mean(perrep["GPU"][k])); nm=float(np.mean(perrep["NPU"][k]))
        app(out,cols,{"size":sz,"gpu_mean":round(gm,4),"npu_mean":round(nm,4),
                      "gap":round(nm-gm,4),"rel_gap_pct":round((nm-gm)/gm*100,1) if gm else 0,
                      "gpu_std":round(float(np.std(perrep["GPU"][k])),4),"npu_std":round(float(np.std(perrep["NPU"][k])),4),
                      "reps":N_REPS,"npu_skip":round(float(np.mean(acc["NPU"]["skip"])),1),
                      "npu_infer_ms":round(float(np.mean(acc["NPU"]["infer"])),1)})
    print(f"  saved {out}",flush=True)

# ---------- STEP 2 ----------
def step2(val,gpu_models,npu_models):
    out=RES/"rev26_yolov8n_contention.csv"
    if out.exists(): out.unlink()
    cols=["bg","N","strategy","worst_sap","worst_std","mean_sap","gpu_skip","npu_skip","reps"]
    splits=[load_split_for_sid(val,s) for s in PANEL4]
    for bg in ["L3_vlm","L1_light","L2_lm"]:
        for st in ["All-GPU","All-NPU"]:
            worst=[];mean=[];gss=[];nss=[]
            for rep in range(N_REPS):
                torch.set_num_threads(THREADS)
                pl=named(PANEL4,st); ng=pl.count("GPU"); nn=pl.count("NPU")
                agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],bg)
                gs,ns=dev_skips(agg); worst.append(agg["worst_sap"]);mean.append(agg["mean_sap"]);gss.append(gs);nss.append(ns)
            app(out,cols,{"bg":bg,"N":4,"strategy":st,"worst_sap":round(float(np.mean(worst)),4),
                          "worst_std":round(float(np.std(worst)),4),"mean_sap":round(float(np.mean(mean)),4),
                          "gpu_skip":round(float(np.mean(gss)),1),"npu_skip":round(float(np.mean(nss)),1),"reps":N_REPS})
            print(f"  STEP2 {bg} {st}: worst={np.mean(worst):.4f} mean={np.mean(mean):.4f} GPUsk={np.mean(gss):.0f}% NPUsk={np.mean(nss):.0f}%",flush=True)
    print(f"  saved {out}",flush=True)

# ---------- STEP 3 ----------
def step3(val,gpu_models,npu_models):
    out=RES/"rev26_yolov8n_sweep.csv"
    if out.exists(): out.unlink()
    cols=["resnet_k","gpu_skip","strategy","worst_sap","worst_std","npu_skip","reps"]
    splits=[load_split_for_sid(val,s) for s in PANEL4]
    for k in RESNET_K:
        lvl=reg(k)
        for st in ["All-GPU","All-NPU"]:
            worst=[];gss=[];nss=[]
            for rep in range(N_REPS):
                torch.set_num_threads(THREADS)
                pl=named(PANEL4,st); ng=pl.count("GPU"); nn=pl.count("NPU")
                agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],lvl)
                gs,ns=dev_skips(agg); worst.append(agg["worst_sap"]);gss.append(gs);nss.append(ns)
            app(out,cols,{"resnet_k":k,"gpu_skip":round(float(np.mean(gss)),1),"strategy":st,
                          "worst_sap":round(float(np.mean(worst)),4),"worst_std":round(float(np.std(worst)),4),
                          "npu_skip":round(float(np.mean(nss)),1),"reps":N_REPS})
            print(f"  STEP3 k={k} {st}: GPUsk={np.mean(gss):.0f}% worst={np.mean(worst):.4f} NPUsk={np.mean(nss):.0f}%",flush=True)
    print(f"  saved {out}",flush=True)

def main():
    torch.set_num_threads(THREADS)
    print(f"=== rev26 YOLOv8n generality check, threads={THREADS} ===",flush=True)
    print(f"    mxq={MXQ}",flush=True)
    preload_background_models(max_level="L3")
    val=load_val()
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models=load_npu_engines(DET,DET["baseline_mxq"],DET["baseline_mode"],4)
    set_active_npu_engines(npu_models)
    sanity(val,gpu_models[0],npu_models[0],"pre")
    print("\n--- STEP 1 single-stream per-size ---",flush=True); step1(val,gpu_models[0],npu_models[0])
    print("\n--- STEP 2 contention worst/mean ---",flush=True); step2(val,gpu_models,npu_models)
    print("\n--- STEP 3 ResNet sweep ---",flush=True); step3(val,gpu_models,npu_models)
    sanity(val,gpu_models[0],npu_models[0],"post")
    dispose_npu_for("yolov8n")
    print("=== rev26 done ===",flush=True)

if __name__=="__main__":
    main()
