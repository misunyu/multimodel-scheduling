"""rev27c STEP 0 — YOLO12s export load + small-biased sanity gate."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)); sys.path.insert(0,str(SCRIPT_DIR.parent/"minimal_pipeline"))
torch.set_num_threads(4)
OUT=Path("accv_experiments/results/rev27c_step0_sanity.md")

def main():
    from phase_rev6_sweep import (FGModelGPUGeneric, set_active_npu_engines,
                                    dispose_npu_for, measure_single_stream)
    from _step_d_common import load_val, load_split_for_sid
    from mblt_model_zoo import vision as mv
    b=["# rev27c STEP 0 — YOLO12s export load + small-biased gate\n\n"]
    try:
        npu_obj=mv.YOLO12s(infer_mode="global8",product="aries")
        x=npu_obj.preprocess(np.zeros((1200,1920,3),dtype=np.uint8)); raw=npu_obj(x)
        shapes=[tuple(getattr(t,'shape',())) for t in raw] if isinstance(raw,(list,tuple)) else [tuple(getattr(raw,'shape',()))]
        b.append(f"## (0a) Load\nYOLO12s global8 LOADS+RUNS. NPU raw output shapes: {shapes}\n\n")
        npu_obj.dispose()
    except Exception as e:
        b.append(f"## (0a) Load\n**FAIL**: {type(e).__name__}: {str(e)[:200]}\n\nGATE FAIL (no usable export). Stop.\n")
        OUT.write_text("".join(b)); print("STEP0 LOAD FAIL:", e); return

    val=load_val()
    gpu=FGModelGPUGeneric("yolo12s.pt")
    npu=mv.YOLO12s(infer_mode="global8",product="aries")
    set_active_npu_engines([npu])
    sids=[2,3,8,17,21,22]
    acc={d:{k:[] for k in ["s","m","l"]} for d in ["GPU","NPU"]}
    infskip={"infer":[],"skip":[]}
    for rep in range(3):
        torch.set_num_threads(4)
        for sid in sids:
            sp=load_split_for_sid(val,sid)
            for dev,model in [("GPU",gpu),("NPU",npu)]:
                if dev=="NPU": set_active_npu_engines([npu])
                m=measure_single_stream(sid,sp,dev,model,"L0")
                acc[dev]["s"].append(m["sap_s"]);acc[dev]["m"].append(m["sap_m"]);acc[dev]["l"].append(m["sap_l"])
                if dev=="NPU": infskip["infer"].append(m["latency_mean"]);infskip["skip"].append(m["frame_skip_pct"])
    b.append("## (0b) small-biased sanity (6 sids x 3 reps)\n\n")
    b.append(f"NPU infer {np.mean(infskip['infer']):.1f}ms, skip {np.mean(infskip['skip']):.1f}% (threads=4)\n\n")
    b.append("| size | GPU | NPU | gap | rel% |\n|---|---|---|---|---|\n")
    rels={}
    for sz,k in [("small","s"),("medium","m"),("large","l")]:
        gm=np.mean(acc["GPU"][k]); nm=np.mean(acc["NPU"][k]); rel=(nm-gm)/gm*100 if gm else 0
        rels[sz]=rel
        b.append(f"| {sz} | {gm:.4f} | {nm:.4f} | {nm-gm:+.4f} | {rel:+.1f}% |\n")
    small_biased = abs(rels["small"]) >= abs(rels["large"]) - 5
    large_damaged = abs(rels["large"]) > abs(rels["small"]) + 15
    b.append("\n## Gate verdict\n\n")
    if large_damaged:
        b.append(f"**FAIL — large-biased export damage** (large {rels['large']:+.1f}% >> small {rels['small']:+.1f}%, v10s-style). Do NOT run STEP 1-3.\n")
        verdict="FAIL_large_damaged"
    elif small_biased:
        b.append(f"**PASS — small-biased** (small {rels['small']:+.1f}% worst, large {rels['large']:+.1f}% mildest). Proceed to STEP 1-3.\n")
        verdict="PASS"
    else:
        b.append(f"**INCONCLUSIVE** (small {rels['small']:+.1f}%, large {rels['large']:+.1f}%). Report as-is.\n")
        verdict="INCONCLUSIVE"
    OUT.write_text("".join(b))
    print(f"saved {OUT}")
    print(f"GATE={verdict}  small={rels['small']:.1f}% med={rels['medium']:.1f}% large={rels['large']:.1f}%")
    try: dispose_npu_for("yolo12s")
    except: pass

if __name__=="__main__":
    main()
