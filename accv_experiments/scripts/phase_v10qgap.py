"""EXP-AUDIT-V10QGAP: YOLOv10s FP32 vs INT8 per-size AP (single-stream, tab:single-stream path).
Reuses measure_single_stream (rev6/rev19), threads=4, isolated N=1, L0, 24 logs x 3 reps.
Detector swapped to YOLOv10s; INT8 = vendor mblt yolov10s.mxq (global8). No fine-tune, no co-tenant."""
import sys, os, csv, time
import numpy as np, torch, cv2
SD=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SD); sys.path.insert(0,os.path.join(SD,"..","minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models, npu_infer, DATA
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_single_stream)
RES="accv_experiments/results"
THREADS=4; N_REPS=3
MXQ=os.path.expanduser("~/.mblt_model_zoo/vision/aries/global8/yolov10s.mxq")
DET={"name":"yolov10s","cls":"YOLOv10s","ultralytics_pt":"yolov10s.pt"}
V11_FP32={"small":0.016,"medium":0.184,"large":0.477}  # tab:single-stream reference (for context only)

def agg(rows,dev,key):
    # per-rep mean over sids, then mean/std over reps
    per=[]
    for rep in range(N_REPS):
        v=[r[key] for r in rows if r["device"]==dev and r["rep"]==rep]
        if v: per.append(float(np.mean(v)))
    return (round(float(np.mean(per)),4), round(float(np.std(per)),4)) if per else (float('nan'),0)

def main():
    torch.set_num_threads(THREADS)
    print(f"=== EXP-AUDIT-V10QGAP threads={THREADS} ===",flush=True)
    print(f"FP32: yolov10s.pt | INT8 mxq: {MXQ} (global8, nmsfree=True)",flush=True)
    print(f"mxq exists: {os.path.exists(MXQ)}",flush=True)
    preload_background_models(max_level="L0")
    val=load_val(); nsid=len(val["sequences"])
    gpu=FGModelGPUGeneric("yolov10s.pt")
    npu=load_npu_engines(DET,MXQ,"global8",1)[0]; set_active_npu_engines([npu])
    # STEP 0b/0d smoke: input shape + non-degenerate detections on 30 frames
    sp0=load_split_for_sid(val,0); fs=(sp0["imgs"][0]["height"],sp0["imgs"][0]["width"])
    x=npu.preprocess(cv2.imread(str(DATA/sp0["seq_dir"]/sp0["imgs"][0]["name"])))
    print(f"[0b] NPU preprocess shape: {np.asarray(x).shape if not hasattr(x,'shape') else x.shape}",flush=True)
    ndet=sum(len(npu_infer(DATA/sp0["seq_dir"]/im["name"],fs,npu)[0]) for im in sp0["imgs"][:30])
    print(f"[0d] NPU smoke detections over 30 frames: {ndet}",flush=True)
    if ndet==0:
        print("VERDICT C — degenerate INT8 output (all-empty). STOP.",flush=True); return
    # STEP 1/2: 24 logs x 3 reps, GPU(FP32) + NPU(INT8)
    rows=[]
    for rep in range(N_REPS):
        torch.set_num_threads(THREADS)
        for sid in range(nsid):
            sp=load_split_for_sid(val,sid)
            mg=measure_single_stream(sid,sp,"GPU",gpu,"L0")
            set_active_npu_engines([npu]); mn=measure_single_stream(sid,sp,"NPU",npu,"L0")
            for dev,m in [("GPU",mg),("NPU",mn)]:
                rows.append({"device":dev,"rep":rep,"sid":sid,
                             "sap_s":m["sap_s"],"sap_m":m["sap_m"],"sap_l":m["sap_l"],
                             "skip":m["frame_skip_pct"],"lat":m["latency_mean"]})
        print(f"  rep{rep} done",flush=True)
    dispose_npu_for("yolov10s")
    with open(f"{RES}/v10qgap_raw.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    # aggregate
    out={}
    for dev in ["GPU","NPU"]:
        out[dev]={k:agg(rows,dev,k) for k in ["sap_s","sap_m","sap_l","skip","lat"]}
    print("\n=== STEP1 FP32 (GPU) / STEP2 INT8 (NPU): per-size sAP mean±std (3 reps), skip, infer ms ===",flush=True)
    for dev in ["GPU","NPU"]:
        o=out[dev]
        print(f"  {dev}: small={o['sap_s'][0]}±{o['sap_s'][1]} medium={o['sap_m'][0]}±{o['sap_m'][1]} "
              f"large={o['sap_l'][0]}±{o['sap_l'][1]} | skip={o['skip'][0]}% infer={o['lat'][0]}ms",flush=True)
    print(f"  [ref] YOLOv11s FP32: small={V11_FP32['small']} medium={V11_FP32['medium']} large={V11_FP32['large']}",flush=True)
    # STEP3 gap
    print("\n=== STEP3 per-size quantization gap (FP32-INT8) ===",flush=True)
    g={}
    for k,nm in [("sap_s","small"),("sap_m","medium"),("sap_l","large")]:
        fp=out["GPU"][k][0]; iq=out["NPU"][k][0]; gap=fp-iq; rel=(gap/fp*100) if fp>1e-6 else float('nan')
        g[nm]=(fp,iq,gap,rel)
        print(f"  {nm:>6}: FP32={fp:.4f} INT8={iq:.4f} gap={gap:+.4f} rel={rel:+.1f}%",flush=True)
    # gates + verdict
    npu_skip=out["NPU"]["skip"][0]; gpu_skip=out["GPU"]["skip"][0]
    fp_large=g["large"][0]; int_large=g["large"][1]; gap_large=g["large"][2]; gap_small=g["small"][2]
    print(f"\n[0e] frame skip: GPU={gpu_skip}% NPU={npu_skip}% (need ~0 both)",flush=True)
    print(f"FP32 large {fp_large:.4f} vs YOLOv11s FP32 large {V11_FP32['large']} | gap_large={gap_large:+.4f} gap_small={gap_small:+.4f}",flush=True)
    if npu_skip>5 or gpu_skip>5:
        verdict="INCONCLUSIVE (contaminated: skip not ~0 at threads=4)"
    else:
        fp_healthy = fp_large >= 0.8*V11_FP32["large"]   # 'not substantially below' ref
        large_dominated = gap_large > gap_small
        if fp_healthy and large_dominated:
            verdict="A (export-induced large-object collapse; FP32 controlled; no fine-tune needed)"
        elif not fp_healthy:
            verdict="B (FP32 large itself weak; accuracy confound plausible)"
        else:
            verdict="A? (FP32 healthy but gap not large-dominated — report numbers)"
    print(f"\nVERDICT: {verdict}",flush=True)
    import json; json.dump({"out":out,"gap":g,"verdict":verdict},open(f"{RES}/v10qgap_means.json","w"),indent=2,default=str)
    print("saved v10qgap_raw.csv + v10qgap_means.json",flush=True)
if __name__=="__main__": main()
