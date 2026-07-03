"""rev20 (Option B) — heavy-background reversal search at threads=4 (fair).

Both devices forced to torch.set_num_threads(4). Per-cell GPU skip% AND NPU skip%
recorded separately (to detect the mixed-state artifact that faked the old N=4
reversal). Single-stream sanity before measuring.

Matrix: bg {L1_light, L2_lm, L3_vlm} x N {2,4,8} x 5 strategies x 3 reps.
Priority order (hypothesis-critical first): L2_lm/L3_vlm N=4, then N=8, N=2,
then L1_light control. Oracle: full enumeration for N=2,4; N=8 named-only.

Outputs (manifest-checkpointed):
  results/rev20_5strat_heavybg.csv
  results/rev20_sanity.csv
  results/manifest_rev20.json
"""
from __future__ import annotations
import csv, json, itertools, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                                dispose_npu_for, measure_single_stream, measure_multistream, DETECTORS)
from step0_compare_devices import FPS

RES=Path("accv_experiments/results"); MAN=RES/"manifest_rev20.json"
OUT=RES/"rev20_5strat_heavybg.csv"; SAN=RES/"rev20_sanity.csv"
DET=[d for d in DETECTORS if d["name"]=="yolo11s"][0]
THREADS=4; N_REPS=3; PERIOD=1000.0/FPS
SIZE_GROUP={2:"small",22:"small",13:"small",16:"small",3:"large",21:"large",14:"large",4:"large"}
PANELS={2:[2,21], 4:[2,22,3,21], 8:[2,22,13,16,3,21,14,4]}
# priority: hypothesis-critical heavy-bg N=4 first
CELLS=[("L2_lm",4),("L3_vlm",4),("L2_lm",8),("L3_vlm",8),("L2_lm",2),("L3_vlm",2),
       ("L1_light",4),("L1_light",8),("L1_light",2)]

def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except: pass
    return {"done":{}}
def save(m): MAN.write_text(json.dumps(m,indent=2))

def named(sids,kind):
    if kind=="All-GPU": return ["GPU"]*len(sids)
    if kind=="All-NPU": return ["NPU"]*len(sids)
    if kind=="Isolated": return ["NPU" if SIZE_GROUP[s]=="large" else "GPU" for s in sids]
    if kind=="Cont-aware": return ["NPU" if SIZE_GROUP[s]=="small" else "GPU" for s in sids]

COLS=["rep","bg","N","strategy","placement","worst_sap","mean_sap","gpu_skip","npu_skip"]
def append(row):
    new=not OUT.exists()
    with open(OUT,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        if new: w.writeheader()
        w.writerow(row)

def dev_skips(agg):
    g=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"]
    n=[s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="NPU"]
    return (round(float(np.mean(g)),1) if g else 0.0, round(float(np.mean(n)),1) if n else 0.0)

def sanity(val, gpu, npu, tag):
    torch.set_num_threads(THREADS)
    set_active_npu_engines([npu])
    gl=[]; nl=[]; lat=[]; skip=[]
    for sid in [2,3,21,22,13]:
        sp=load_split_for_sid(val,sid)
        mg=measure_single_stream(sid,sp,"GPU",gpu,"L0")
        set_active_npu_engines([npu]); mn=measure_single_stream(sid,sp,"NPU",npu,"L0")
        gl.append(mg["sap_l"]); nl.append(mn["sap_l"]); lat.append(mn["latency_mean"]); skip.append(mn["frame_skip_pct"])
    gap=float(np.mean(nl)-np.mean(gl))
    row={"tag":tag,"large_gap":round(gap,4),"npu_infer":round(float(np.mean(lat)),2),
         "npu_skip":round(float(np.mean(skip)),2),
         "ok":bool(abs(gap-0.0)<=0.01 and np.mean(lat)<18)}
    new=not SAN.exists()
    with open(SAN,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)
    print(f"  SANITY {tag}: large_gap={gap:+.4f} infer={np.mean(lat):.1f}ms skip={np.mean(skip):.1f}% ok={row['ok']}")
    return row["ok"]

def main():
    torch.set_num_threads(THREADS)
    m=man()
    print(f"=== rev20 threads={THREADS} period={PERIOD:.1f}ms ===")
    print("preloading bg max=L3 ...")
    preload_background_models(max_level="L3")
    val=load_val()
    gpu_models=[FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(8)]
    npu_models=load_npu_engines(DET,DET["multistream_mxq"],DET["multistream_mode"],8)
    set_active_npu_engines(npu_models)

    sanity(val, gpu_models[0], npu_models[0], "pre")

    for bg,N in CELLS:
        sids=PANELS[N]; splits=[load_split_for_sid(val,s) for s in sids]
        # placements: full enumeration for oracle if N<=4 else named-only
        if N<=4:
            placements=list(itertools.product(["GPU","NPU"],repeat=N))
        else:
            placements=[tuple(named(sids,k)) for k in ["All-GPU","Isolated","Cont-aware","All-NPU"]]
        for rep in range(N_REPS):
            key=f"{bg}/N{N}/rep{rep}"
            if m["done"].get(key): continue
            torch.set_num_threads(THREADS)
            best_w=-1; best_pl=None; best_m=None; named_res={}
            for pl in placements:
                ng=sum(1 for d in pl if d=="GPU"); nn=sum(1 for d in pl if d=="NPU")
                agg=measure_multistream(sids,splits,list(pl),gpu_models[:ng],npu_models[:nn],bg)
                gs,ns=dev_skips(agg); w=agg["worst_sap"]; mn=agg["mean_sap"]
                if w>best_w: best_w=w; best_pl=pl; best_m=mn; best_skips=(gs,ns)
                for kind in ["All-GPU","Isolated","Cont-aware","All-NPU"]:
                    if list(pl)==named(sids,kind): named_res[kind]=(w,mn,gs,ns,pl)
            for kind,(w,mn,gs,ns,pl) in named_res.items():
                append({"rep":rep,"bg":bg,"N":N,"strategy":kind,
                        "placement":"".join("N" if d=="NPU" else "G" for d in pl),
                        "worst_sap":round(w,4),"mean_sap":round(mn,4),"gpu_skip":gs,"npu_skip":ns})
            if N<=4:
                append({"rep":rep,"bg":bg,"N":N,"strategy":"Oracle",
                        "placement":"".join("N" if d=="NPU" else "G" for d in best_pl),
                        "worst_sap":round(best_w,4),"mean_sap":round(best_m,4),
                        "gpu_skip":best_skips[0],"npu_skip":best_skips[1]})
            m["done"][key]={"ts":int(time.time())}; save(m)
            line=" ".join(f"{k}={named_res[k][0]:.4f}(G{named_res[k][2]:.0f}/N{named_res[k][3]:.0f})" for k in named_res)
            print(f"  {bg} N={N} rep{rep}: {line}"+(f" Oracle={best_w:.4f}" if N<=4 else ""))

    sanity(val, gpu_models[0], npu_models[0], "post")
    dispose_npu_for("yolo11s")
    print("=== rev20 done ===")

if __name__=="__main__":
    main()
