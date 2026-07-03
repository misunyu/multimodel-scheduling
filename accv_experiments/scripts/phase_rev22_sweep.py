"""rev22 — VLM/GPU saturation sweep (gradual reversal threshold) + per-stream + bit-identical.

Contention lever: number of ResNet50 GPU-loops (0..k) registered at runtime into
BG_VARIANTS (dict mutation, NO core file edit). ResNet50 is GPU-bound (ORT-CUDA),
minimal host-CPU → isolates the GPU-contention axis, keeping the NPU host
postprocess fast (avoids the L2_lm CPU-confound). x-axis = measured GPU skip%.

Both devices torch.set_num_threads(4). Per-cell GPU skip% AND NPU skip% recorded.
Per-stream sAP captured (PART B). Single-stream sanity pre/post. Bit-identical
detection hash (PART C).

Outputs:
  results/rev22_vlm_sweep.csv      (lever, gpu_skip, npu_skip, strategy, worst, mean, +N)
  results/rev22_perstream_sap.csv  (placement, stream_id, sid, device, sap, skip)
  results/rev22_bitident.csv       (frame hash diff threads 4 vs 24)
  results/rev22_sanity.csv
  results/manifest_rev22.json
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)); sys.path.insert(0,str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models, npu_infer
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                                dispose_npu_for, measure_single_stream, measure_multistream, DETECTORS)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import DATA, CONF, IOU, FPS

RES=Path("accv_experiments/results"); MAN=RES/"manifest_rev22.json"
DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; PERIOD=1000.0/FPS
SIZE_GROUP={2:"small",22:"small",3:"large",21:"large"}
PANEL4=[2,22,3,21]; PANEL2=[2,21]
RESNET_COUNTS=[0,1,2,3,4,6,8]   # GPU-pressure lever

def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except: pass
    return {"done":{}}
def save(m): MAN.write_text(json.dumps(m,indent=2))

def reg_level(k):
    name=f"SWEEP{k}"
    h2.BG_VARIANTS[name]=[_bg_resnet50_loop]*k
    return name

def dev_skips(agg):
    g=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"]
    n=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
    return (round(float(np.mean(g)),1) if g else 0.0, round(float(np.mean(n)),1) if n else 0.0)

SWEEP_COLS=["rep","N","resnet_k","bg_level","strategy","placement","worst_sap","mean_sap","gpu_skip","npu_skip"]
PS_COLS=["rep","N","resnet_k","strategy","stream_id","sid","device","sap","skip"]
def app(path,cols,row):
    new=not path.exists()
    with open(path,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        if new: w.writeheader()
        w.writerow(row)

def named(sids,kind):
    if kind=="All-GPU": return ["GPU"]*len(sids)
    if kind=="All-NPU": return ["NPU"]*len(sids)
    if kind=="mixed2":  return ["NPU" if SIZE_GROUP[s]=="small" else "GPU" for s in sids]  # 2 NPU at N=4

def sanity(val,gpu,npu,tag):
    torch.set_num_threads(THREADS); set_active_npu_engines([npu])
    gl=[];nl=[];lat=[];sk=[]
    for sid in [2,3,21,22,13]:
        sp=load_split_for_sid(val,sid)
        mg=measure_single_stream(sid,sp,"GPU",gpu,"L0")
        set_active_npu_engines([npu]); mn=measure_single_stream(sid,sp,"NPU",npu,"L0")
        gl.append(mg["sap_l"]);nl.append(mn["sap_l"]);lat.append(mn["latency_mean"]);sk.append(mn["frame_skip_pct"])
    gap=float(np.mean(nl)-np.mean(gl))
    row={"tag":tag,"large_gap":round(gap,4),"npu_infer":round(float(np.mean(lat)),2),"npu_skip":round(float(np.mean(sk)),2),
         "ok":bool(np.mean(lat)<18 and np.mean(sk)<5)}
    app(RES/"rev22_sanity.csv",["tag","large_gap","npu_infer","npu_skip","ok"],row)
    print(f"  SANITY {tag}: gap={gap:+.4f} infer={np.mean(lat):.1f} skip={np.mean(sk):.1f}% ok={row['ok']}")

def part_c_bitident(val,npu):
    """threads=4 vs 24 detection identity, persisted."""
    sp=load_split_for_sid(val,2)
    paths=[DATA/sp["seq_dir"]/im["name"] for im in sp["imgs"]]
    fs=(sp["imgs"][0]["height"],sp["imgs"][0]["width"])
    frames=paths[40:60]
    def run(nt):
        torch.set_num_threads(nt); out=[]
        for p in frames: out.append(npu_infer(p,fs,npu))
        return out
    r4=run(4); r24=run(24)
    for i,((b4,s4,c4),(b24,s24,c24)) in enumerate(zip(r4,r24)):
        ident = (len(b4)==len(b24)) and (len(b4)==0 or (np.array_equal(b4,b24) and np.array_equal(s4,s24)))
        mbd=float(np.abs(b4-b24).max()) if (len(b4)==len(b24) and len(b4)>0) else 0.0
        msd=float(np.abs(s4-s24).max()) if (len(b4)==len(b24) and len(b4)>0) else 0.0
        app(RES/"rev22_bitident.csv",["frame_id","n_t4","n_t24","max_box_diff","max_score_diff","identical"],
            {"frame_id":40+i,"n_t4":len(b4),"n_t24":len(b24),"max_box_diff":mbd,"max_score_diff":msd,"identical":ident})
    torch.set_num_threads(THREADS)
    print("  PART C bit-identical: 20 frames hashed -> rev22_bitident.csv")

def main():
    torch.set_num_threads(THREADS)
    m=man()
    print(f"=== rev22 sweep threads={THREADS} period={PERIOD:.1f}ms ===")
    preload_background_models(max_level="L1")  # resnet50 only
    val=load_val()
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models=load_npu_engines(DET,DET["multistream_mxq"],DET["multistream_mode"],4)
    set_active_npu_engines(npu_models)
    sanity(val,gpu_models[0],npu_models[0],"pre")
    part_c_bitident(val,npu_models[0])

    sweeps=[(4,PANEL4,["All-GPU","All-NPU","mixed2"])]  # core
    # optional N=2,8 single line (All-GPU, All-NPU only)
    sweeps+=[(2,PANEL2,["All-GPU","All-NPU"]),(8,[2,22,13,16,3,21,14,4],["All-GPU","All-NPU"])]
    for N,sids,strats in sweeps:
        splits=[load_split_for_sid(val,s) for s in sids]
        if N==8: # need 8 models
            while len(gpu_models)<8: gpu_models.append(FGModelGPUGeneric(DET["ultralytics_pt"]))
            if len(npu_models)<8:
                dispose_npu_for("yolo11s")
                npu_models=load_npu_engines(DET,DET["multistream_mxq"],DET["multistream_mode"],8)
                set_active_npu_engines(npu_models)
        for k in RESNET_COUNTS:
            lvl=reg_level(k)
            for rep in range(N_REPS):
                key=f"N{N}/k{k}/rep{rep}"
                if m["done"].get(key): continue
                torch.set_num_threads(THREADS)
                for st in strats:
                    pl=named(sids,st)
                    ng=sum(1 for d in pl if d=="GPU"); nn=sum(1 for d in pl if d=="NPU")
                    agg=measure_multistream(sids,splits,pl,gpu_models[:ng],npu_models[:nn],lvl)
                    gs,ns=dev_skips(agg)
                    app(RES/"rev22_vlm_sweep.csv",SWEEP_COLS,{"rep":rep,"N":N,"resnet_k":k,"bg_level":lvl,
                        "strategy":st,"placement":"".join("N" if d=="NPU" else "G" for d in pl),
                        "worst_sap":round(agg["worst_sap"],4),"mean_sap":round(agg["mean_sap"],4),
                        "gpu_skip":gs,"npu_skip":ns})
                    # PART B per-stream (only N=4 mixed/core, rep0 to limit rows)
                    if N==4 and rep==0:
                        for s in agg["per_stream"]:
                            app(RES/"rev22_perstream_sap.csv",PS_COLS,{"rep":rep,"N":N,"resnet_k":k,"strategy":st,
                                "stream_id":s["stream_id"],"sid":s["sid"],"device":s["device"],
                                "sap":round(s["sap_5095"],4),"skip":round(s["frame_skip_pct"],1)})
                m["done"][key]={"ts":int(time.time())}; save(m)
                ag=[r for r in [agg] ]  # noop
                print(f"  N={N} k={k} rep{rep}: done ({lvl})")
            # quick line after reps
            agc=lambda st: None
    sanity(val,gpu_models[0],npu_models[0],"post")
    dispose_npu_for("yolo11s")
    print("=== rev22 done ===")

if __name__=="__main__":
    main()
