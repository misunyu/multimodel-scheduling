"""EXP-CPULOAD: host CPU util under baseline / +LM / +VLM (All-NPU N=4, threads=4).
Reuses rev20/rev6 path (measure_multistream, same co-tenants) + psutil CPU sampler + pynvml GPU.
Gate: NPU skip must reproduce baseline~0 / LM~76% / VLM~1% else config mismatch -> report.
Outputs: results/cpuload_raw.csv, results/cpuload_results.md"""
import sys, csv, time, threading, statistics as st
from pathlib import Path
import numpy as np, torch, psutil
SD=Path(__file__).resolve().parent
sys.path.insert(0,str(SD)); sys.path.insert(0,str(SD.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)
try:
    import pynvml; pynvml.nvmlInit(); _H=pynvml.nvmlDeviceGetHandleByIndex(0); HAVE_NV=True
except Exception: HAVE_NV=False
RES=Path("accv_experiments/results"); DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; PANEL4=[2,22,3,21]
CONDS=[("baseline","L0"),("+LM","L2_lm"),("+VLM","L3_vlm")]
PROC=psutil.Process()

class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=threading.Event(); self.sys=[]; self.proc=[]; self.maxcore=[]; self.gpu=[]
    def run(self):
        psutil.cpu_percent(interval=None, percpu=True); PROC.cpu_percent(interval=None)
        time.sleep(0.1)
        while not self.stop.is_set():
            pc=psutil.cpu_percent(interval=0.1, percpu=True)   # blocks 0.1s
            self.sys.append(sum(pc)/len(pc)); self.maxcore.append(max(pc))
            self.proc.append(PROC.cpu_percent(interval=None))
            if HAVE_NV:
                try: self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(_H).gpu)
                except Exception: pass
def stats(a):
    if not a: return (0,0,0)
    return (round(float(np.mean(a)),1), round(float(np.percentile(a,50)),1), round(float(np.percentile(a,99)),1))

def main():
    torch.set_num_threads(THREADS)
    print(f"=== EXP-CPULOAD threads={THREADS} cores={psutil.cpu_count()} nvml={HAVE_NV} ===",flush=True)
    preload_background_models(max_level="L3")
    val=load_val(); splits=[load_split_for_sid(val,s) for s in PANEL4]
    npu=load_npu_engines(DET,DET["multistream_mxq"],DET["multistream_mode"],4); set_active_npu_engines(npu)
    rows=[]; agg={}
    for cond,bg in CONDS:
        agg[cond]=[]
        for rep in range(N_REPS):
            torch.set_num_threads(THREADS)
            smp=Sampler(); smp.start()
            t0=time.time()
            res=measure_multistream(PANEL4,splits,["NPU"]*4,[],npu[:4],bg)
            wall=time.time()-t0
            smp.stop.set(); smp.join(timeout=5)
            skip=float(np.mean([s["frame_skip_pct"] for s in res["per_stream"]]))
            lat=float(np.mean([s["latency_mean"] for s in res["per_stream"]]))
            sysm,sysp50,sysp99=stats(smp.sys); pm,pp50,pp99=stats(smp.proc); _,_,mcore=stats(smp.maxcore); gm,_,_=stats(smp.gpu)
            row={"cond":cond,"bg":bg,"rep":rep,"npu_skip_pct":round(skip,1),"npu_lat_ms":round(lat,2),
                 "worst_sap":round(res["worst_sap"],4),"mean_sap":round(res["mean_sap"],4),
                 "cpu_sys_mean":sysm,"cpu_sys_p99":sysp99,"cpu_proc_mean":pm,"cpu_proc_p99":pp99,
                 "cpu_maxcore_p99":mcore,"gpu_util_mean":gm,"n_samples":len(smp.sys),"wall_s":round(wall,1)}
            rows.append(row); agg[cond].append(row)
            print(f"  {cond} rep{rep}: skip={skip:.1f}% npu_lat={lat:.1f}ms cpu_sys={sysm:.1f}% cpu_proc={pm:.0f}% gpu={gm:.0f}% (n={len(smp.sys)})",flush=True)
    dispose_npu_for("yolo11s")
    # write raw csv
    cols=list(rows[0].keys())
    with open(RES/"cpuload_raw.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    # condition means
    def cm(cond,k): return round(float(np.mean([r[k] for r in agg[cond]])),1)
    print("\n=== CONDITION MEANS ===",flush=True)
    for cond,_ in CONDS:
        print(f"  {cond}: skip={cm(cond,'npu_skip_pct')}% npu_lat={cm(cond,'npu_lat_ms')}ms "
              f"cpu_sys={cm(cond,'cpu_sys_mean')}% cpu_proc={cm(cond,'cpu_proc_mean')}% gpu={cm(cond,'gpu_util_mean')}%",flush=True)
    # gate
    sk={c:cm(c,'npu_skip_pct') for c,_ in CONDS}
    repro=(sk['baseline']<10 and sk['+LM']>=60 and sk['+VLM']<10)
    print(f"\nGATE skip reproduce (base<10/LM>=60/VLM<10): {repro}  ({sk})",flush=True)
    import json; json.dump({"means":{c:{k:cm(c,k) for k in ['npu_skip_pct','npu_lat_ms','cpu_sys_mean','cpu_proc_mean','gpu_util_mean']} for c,_ in CONDS},"repro":repro},
                           open(RES/"cpuload_means.json","w"),indent=2)
    print("saved cpuload_raw.csv + cpuload_means.json",flush=True)
if __name__=="__main__": main()
