"""Step 10 Stretch (direction-only, internal): N=4 + L3_VLM, All-GPU(FT) vs All-NPU(FT-local).
Confirms reversal DIRECTION on the fine-tuned model. Not comparable to main Table; internal only."""
import sys, csv, time, numpy as np, torch
from pathlib import Path
SD=Path(__file__).resolve().parent
sys.path.insert(0,str(SD)); sys.path.insert(0,str(SD.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)
RES=Path("accv_experiments/results"); OUT=RES/"exp_ft_local_results.csv"
DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; PANEL4=[2,22,3,21]
FT_PT="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.pt"
FT_MXQ_SINGLE="accv_experiments/results/qbc_ft_traincalib_single.mxq"
COLS=["cell","repeat","log_or_stream_id","sap","sap_small","sap_medium","sap_large",
      "offline_map","infer_ms_mean","skip_pct","model_sha","run_timestamp","notes"]
def app(r):
    new=not OUT.exists()
    with open(OUT,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(r)
def main():
    torch.set_num_threads(THREADS)
    print("preload L3 (resnet+qwen)…",flush=True); preload_background_models(max_level="L3")
    val=load_val(); splits=[load_split_for_sid(val,s) for s in PANEL4]
    gpu_models=[FGModelGPUGeneric(FT_PT) for _ in range(4)]
    npu_models=load_npu_engines(DET,FT_MXQ_SINGLE,"single",4); set_active_npu_engines(npu_models)
    plans=[("stretch_ft_allgpu_vlm",["GPU"]*4,4,0),("stretch_ft_allnpu_vlm",["NPU"]*4,0,4)]
    for cell,pl,ng,nn in plans:
        print(f"=== {cell} ===",flush=True)
        for rep in range(N_REPS):
            torch.set_num_threads(THREADS)
            agg=measure_multistream(PANEL4,splits,pl,gpu_models[:ng],npu_models[:nn],"L3_vlm")
            ts=int(time.time())
            for s in agg["per_stream"]:
                app({"cell":cell,"repeat":rep,"log_or_stream_id":s["sid"],"sap":round(s["sap_5095"],4),
                     "sap_small":round(s["sap_s"],4),"sap_medium":round(s["sap_m"],4),"sap_large":round(s["sap_l"],4),
                     "offline_map":"","infer_ms_mean":round(s["latency_mean"],2),"skip_pct":round(s["frame_skip_pct"],1),
                     "model_sha":"ft_single" if nn else "ft_pt","run_timestamp":ts,"notes":"stretch_internal_only"})
            print(f"  {cell} rep{rep}: worst={agg['worst_sap']:.4f} mean={agg['mean_sap']:.4f}",flush=True)
    dispose_npu_for("yolo11s"); print("=== stretch done ===",flush=True)
if __name__=="__main__": main()
