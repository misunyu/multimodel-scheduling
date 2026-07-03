"""Step 5 FP32 validation gate: offline per-size mAP, COCO-pretrained vs FT.
Runs each FP32 .pt over ALL val images (Ultralytics PyTorch cuda, same CONF/IOU
as the project harness), maps COCO->AHD, COCOeval per-size. Offline (no streaming).
Gate: if FT improves NO size by >=+0.01 -> Outcome C.
"""
import json, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0,"accv_experiments/minimal_pipeline")
from step0_compare_devices import ANNOT, DATA, IMG_SIZE, CONF, IOU, N_AHD
import os; os.environ.setdefault("YOLO_VERBOSE","False")
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

val=json.load(open(ANNOT))
cm=np.asarray(val["coco_mapping"]); seqd=val["seq_dirs"]
coco_gt=COCO(); coco_gt.dataset={"info":{},"licenses":[],"categories":val["categories"],
    "images":val["images"],"annotations":val["annotations"]}; coco_gt.createIndex()

def evaluate(pt):
    m=YOLO(pt); dets=[]
    imgs=val["images"]
    for k,im in enumerate(imgs):
        p=DATA/seqd[im["sid"]]/im["name"]
        r=m.predict(str(p),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")[0]
        if len(r.boxes)==0: continue
        cls=r.boxes.cls.cpu().numpy().astype(int); ahd=cm[cls]; sel=ahd<N_AHD
        xyxy=r.boxes.xyxy.cpu().numpy()[sel]; conf=r.boxes.conf.cpu().numpy()[sel]; lab=ahd[sel]
        for j in range(len(xyxy)):
            x1,y1,x2,y2=xyxy[j]
            dets.append({"image_id":int(im["id"]),"category_id":int(lab[j]),
                "bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],"score":float(conf[j])})
        if (k+1)%3000==0: print(f"  {pt}: {k+1}/{len(imgs)}",flush=True)
    if not dets: return {"all":0,"small":0,"medium":0,"large":0,"ndet":0}
    dt=coco_gt.loadRes(dets); e=COCOeval(coco_gt,dt,"bbox"); e.evaluate(); e.accumulate(); e.summarize()
    return {"all":float(e.stats[0]),"small":float(e.stats[3]),"medium":float(e.stats[4]),
            "large":float(e.stats[5]),"ndet":len(dets)}

print(f"=== Step 5 offline mAP gate (CONF={CONF} IOU={IOU} imgsz={IMG_SIZE}) ===",flush=True)
t0=time.time(); coco=evaluate("yolo11s.pt"); print("COCO done",time.time()-t0,flush=True)
t0=time.time(); ft=evaluate("accv_experiments/results/ft_runs/ft_yolo11s/weights/best.pt"); print("FT done",time.time()-t0,flush=True)
print("\n=== RESULT (offline mAP per-size) ===")
print(f"{'size':>8} {'COCO':>8} {'FT':>8} {'Δ(FT-COCO)':>11}")
deltas={}
for s in ["all","small","medium","large"]:
    d=ft[s]-coco[s]; deltas[s]=d
    print(f"{s:>8} {coco[s]:>8.4f} {ft[s]:>8.4f} {d:>+11.4f}")
print(f"ndet COCO={coco['ndet']} FT={ft['ndet']}")
improved=any(deltas[s]>=0.01 for s in ["small","medium","large"])
print(f"\nGATE: any per-size Δ >= +0.01 ? -> {improved}")
print("VERDICT:", "PROCEED (FT raised accuracy)" if improved else "OUTCOME C (fine-tuning did not raise per-size accuracy >= +0.01) -> STOP/REPORT")
json.dump({"coco":coco,"ft":ft,"deltas":deltas,"improved":improved},
          open("accv_experiments/results/ft_gate_offline_map.json","w"),indent=2)
