"""Step 7 (bit-identical) + Step 8 (4-cell Table-1-protocol measurement).
Reuses rev6/rev19 harness (FGModelGPUGeneric / load_npu_engines / measure_single_stream),
threads=4, isolated single-stream, 24 logs x 3 reps. NPU mxq = train-calib global8.
Output: exp_ft_local_results.csv
"""
import sys, csv, time, hashlib, numpy as np, torch
from pathlib import Path
SD=Path(__file__).resolve().parent
sys.path.insert(0,str(SD)); sys.path.insert(0,str(SD.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models, npu_infer
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_single_stream, DETECTORS)
from step0_compare_devices import DATA
RES=Path("accv_experiments/results"); OUT=RES/"exp_ft_local_results.csv"
DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; MODE="global8"
COCO_PT="yolo11s.pt"; FT_PT="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.pt"
COCO_MXQ="accv_experiments/results/qbc_coco_traincalib_global8.mxq"
FT_MXQ="accv_experiments/results/qbc_ft_traincalib_global8.mxq"
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
COLS=["cell","repeat","log_or_stream_id","sap","sap_small","sap_medium","sap_large",
      "offline_map","infer_ms_mean","skip_pct","model_sha","run_timestamp","notes"]
def app(r):
    new=not OUT.exists()
    with open(OUT,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(r)

def bitident(mxq,label):
    eng=load_npu_engines(DET,mxq,MODE,1)[0]
    val=load_val(); sp=load_split_for_sid(val,2)
    paths=[DATA/sp["seq_dir"]/im["name"] for im in sp["imgs"]][40:60]
    fs=(sp["imgs"][0]["height"],sp["imgs"][0]["width"])
    def run(): return [npu_infer(p,fs,eng) for p in paths]
    r1=run(); r2=run(); mb=ms=0.0; ident=True
    for (b1,s1,c1),(b2,s2,c2) in zip(r1,r2):
        if len(b1)!=len(b2): ident=False; continue
        if len(b1): mb=max(mb,float(np.abs(b1-b2).max())); ms=max(ms,float(np.abs(s1-s2).max()))
    print(f"[bitident {label}] identical={ident} max_box={mb:.2e} max_score={ms:.2e}",flush=True)
    dispose_npu_for("yolo11s"); return ident

def measure_gpu(cell,pt):
    print(f"=== {cell} (GPU {pt}) ===",flush=True)
    m=FGModelGPUGeneric(pt); s=sha(pt); val=load_val()
    for rep in range(N_REPS):
        for sid in range(len(val["sequences"])):
            torch.set_num_threads(THREADS); sp=load_split_for_sid(val,sid)
            r=measure_single_stream(sid,sp,"GPU",m,"L0")
            app({"cell":cell,"repeat":rep,"log_or_stream_id":sid,"sap":round(r["sap_5095"],4),
                 "sap_small":round(r["sap_s"],4),"sap_medium":round(r["sap_m"],4),"sap_large":round(r["sap_l"],4),
                 "offline_map":round(r["map_5095"],4),"infer_ms_mean":round(r["latency_mean"],2),
                 "skip_pct":round(r["frame_skip_pct"],1),"model_sha":s,"run_timestamp":int(time.time()),
                 "notes":"invalid_skip" if r["frame_skip_pct"]>5 else ""})
        print(f"  {cell} rep{rep} done",flush=True)

def measure_npu(cell,mxq):
    print(f"=== {cell} (NPU {mxq}) ===",flush=True)
    eng=load_npu_engines(DET,mxq,MODE,1)[0]; s=sha(mxq); val=load_val()
    for rep in range(N_REPS):
        for sid in range(len(val["sequences"])):
            torch.set_num_threads(THREADS); sp=load_split_for_sid(val,sid)
            r=measure_single_stream(sid,sp,"NPU",eng,"L0")
            app({"cell":cell,"repeat":rep,"log_or_stream_id":sid,"sap":round(r["sap_5095"],4),
                 "sap_small":round(r["sap_s"],4),"sap_medium":round(r["sap_m"],4),"sap_large":round(r["sap_l"],4),
                 "offline_map":round(r["map_5095"],4),"infer_ms_mean":round(r["latency_mean"],2),
                 "skip_pct":round(r["frame_skip_pct"],1),"model_sha":s,"run_timestamp":int(time.time()),
                 "notes":"invalid_skip" if r["frame_skip_pct"]>5 else ""})
        print(f"  {cell} rep{rep} done",flush=True)
    dispose_npu_for("yolo11s")

def main():
    torch.set_num_threads(THREADS); preload_background_models(max_level="L0")
    print("=== STEP 7 bit-identical ===",flush=True)
    bitident(COCO_MXQ,"COCO-local"); bitident(FT_MXQ,"FT-local")
    print("=== STEP 8 4-cell measurement ===",flush=True)
    measure_gpu("fp32_coco",COCO_PT)
    measure_gpu("fp32_ft",FT_PT)
    measure_npu("int8_coco_local",COCO_MXQ)
    measure_npu("int8_ft_local",FT_MXQ)
    print("=== ft_measure done ===",flush=True)
if __name__=="__main__": main()
