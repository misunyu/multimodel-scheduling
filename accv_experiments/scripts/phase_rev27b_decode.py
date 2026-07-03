"""rev27b STEP 1 — YOLOv10s NPU decode diagnosis (H1 decode-bug vs H2 export-damage).

On a few frames with large objects: compare GPU (FP32 ultralytics) vs NPU (INT8 mblt global8)
raw output layout + decoded boxes. If large boxes are coord-mangled/dropped in decode -> H1.
If coords match but scores are just low -> H2 (quantization/export).

No core eval script modified. threads=4. yolov10s.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, cv2, torch

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR)); sys.path.insert(0,str(SCRIPT_DIR.parent/"minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, npu_infer
from step0_compare_devices import DATA, CONF, IOU, IMG_SIZE, N_AHD
from mblt_model_zoo import vision as mv

torch.set_num_threads(4)
OUT=Path("accv_experiments/results/rev27b_decode_check.md")

def iou_xyxy(a,b):
    if len(b)==0: return np.zeros(0)
    x1=np.maximum(a[0],b[:,0]);y1=np.maximum(a[1],b[:,1]);x2=np.minimum(a[2],b[:,2]);y2=np.minimum(a[3],b[:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa+ab-inter+1e-9)

def main():
    val=load_val(); split=load_split_for_sid(val,17)  # sid 17 medium-mixed, has large vehicles
    imgs=split["imgs"]; seq=split["seq_dir"]; cm=split["coco_mapping"]
    fs=(imgs[0]["height"],imgs[0]["width"])
    # GPU
    import os; os.environ.setdefault("YOLO_VERBOSE","False")
    from ultralytics import YOLO
    gpu=YOLO("yolov10s.pt")
    # NPU
    mxq=None
    # yolov10s auto-downloads; get its cached path after instantiation
    npu=mv.YOLOv10s(infer_mode="global8",product="aries")
    b=[]
    b.append("# rev27b STEP 1 — YOLOv10s NPU decode diagnosis\n\n")
    b.append("_GPU FP32 (ultralytics yolov10s.pt) vs NPU INT8 (mblt yolov10s global8) on sid 17 frames. threads=4._\n\n")

    # --- raw NPU output layout vs what postprocess expects ---
    b.append("## Raw NPU output layout\n\n")
    p0=DATA/seq/imgs[40]["name"]; img0=cv2.imread(str(p0))
    x=npu.preprocess(img0); raw=npu(x)
    if isinstance(raw,(list,tuple)):
        b.append(f"NPU raw outputs: {len(raw)} tensors, shapes = {[tuple(getattr(t,'shape',()))[:] for t in raw]}\n\n")
    else:
        b.append(f"NPU raw output: single tensor shape {tuple(getattr(raw,'shape',()))}\n\n")
    b.append("(YOLOv10 is NMS-free dual-head; v8/v11 are anchor-free + NMS. Layout differences here can break decode.)\n\n")

    # --- per-frame GPU vs NPU box comparison, focus on LARGE objects ---
    b.append("## Per-frame GPU vs NPU box comparison (large objects, area>96^2 px)\n\n")
    b.append("| frame | GPU large dets (box, score) | NPU match? (IoU, score) | verdict |\n|---|---|---|---|\n")
    h0,w0=fs; LARGE_AREA=96*96*(h0*w0)/(IMG_SIZE*IMG_SIZE)  # large in original-scale approx
    h1_signals=0; h2_signals=0
    for fi in [40,60,80,100,120]:
        p=DATA/seq/imgs[fi]["name"]
        r=gpu.predict(str(p),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")[0]
        gb=r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0,4))
        gs=r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        # NPU
        nb,ns,ncls=npu_infer(p,fs,npu)
        # large GPU boxes
        areas=(gb[:,2]-gb[:,0])*(gb[:,3]-gb[:,1]) if len(gb) else np.zeros(0)
        large_idx=np.where(areas>96*96)[0]
        cells=[]
        for li in large_idx[:2]:
            box=gb[li]; sc=gs[li]
            if len(nb):
                ious=iou_xyxy(box,nb); j=int(np.argmax(ious)); miou=float(ious[j])
            else: miou=0.0; j=-1
            if miou>0.5:
                nsc=float(ns[j]); cells.append(f"matched IoU={miou:.2f} score={nsc:.2f}")
                h2_signals+=1  # coords fine, score (maybe low) -> quantization/export
            elif miou>0.1:
                cells.append(f"PARTIAL IoU={miou:.2f} (coord drift)"); h1_signals+=1
            else:
                cells.append(f"NO MATCH (IoU={miou:.2f}, dropped)"); h1_signals+=1
        gpu_desc="; ".join(f"[{int(gb[i][0])},{int(gb[i][1])},{int(gb[i][2])},{int(gb[i][3])}] s={gs[i]:.2f}" for i in large_idx[:2])
        verdict="coords-OK/score" if (cells and all("matched" in c for c in cells)) else ("decode-issue" if any(("NO MATCH" in c or "PARTIAL" in c) for c in cells) else "no-large")
        b.append(f"| {fi} | {gpu_desc or '(none)'} | {'; '.join(cells) or '(no NPU large match)'} | {verdict} |\n")
    b.append("\n")
    # also total det counts
    b.append("## Aggregate signal\n\n")
    b.append(f"- H1 signals (coord drift / dropped large boxes): {h1_signals}\n")
    b.append(f"- H2 signals (large box coords matched, score-only loss): {h2_signals}\n\n")
    if h1_signals>h2_signals:
        b.append("**Leaning H1 (decode/layout mismatch)** — large GPU boxes are not matched by NPU at correct coords; "
                 "consistent with a YOLOv10 head-decode mismatch in the NPU output path.\n")
    elif h2_signals>h1_signals:
        b.append("**Leaning H2 (export/quantization)** — large boxes are at correct coords but lose score on the NPU; "
                 "decode looks fine, the INT8 export degrades large-object confidence (not fixable by us).\n")
    else:
        b.append("**Inconclusive from box matching** — see raw layout + STEP 2 vendor mAP.\n")
    OUT.write_text("".join(b))
    print(f"saved {OUT}")
    print(f"H1_signals={h1_signals} H2_signals={h2_signals}")
    try: npu.dispose()
    except: pass

if __name__=="__main__":
    main()
