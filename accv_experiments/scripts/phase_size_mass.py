"""EXP-SIZE-MASS: recoverable accuracy (true positives) by object size at clean operating point.
GT counts + GPU FP32 (yolo11s.pt) clean single-stream detections (regenerated; tab:single-stream
point: threads=4, conf=0.25, skip~0). TP counts via COCO-style greedy match (IoU 0.5 + 0.5:0.95).
Honest additive TP counts (NOT share-weighted AP)."""
import json, sys, csv, os
from collections import defaultdict, Counter
import numpy as np, torch
sys.path.insert(0,"accv_experiments/minimal_pipeline")
from step0_compare_devices import ANNOT, DATA, IMG_SIZE, CONF, IOU, N_AHD
os.environ.setdefault("YOLO_VERBOSE","False")
from ultralytics import YOLO
RES="accv_experiments/results"; THREADS=4
THRS=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
def xywh2xyxy(b): x,y,w,h=b; return (x,y,x+w,y+h)
def sizebin(area): return "small" if area<1024 else ("medium" if area<9216 else "large")
def iou(a,b):  # a,b xyxy
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)

def main():
    torch.set_num_threads(THREADS)
    d=json.load(open(ANNOT)); cm=np.asarray(d["coco_mapping"]); seqd=d["seq_dirs"]
    byimg={i["id"]:i for i in d["images"]}
    gt=defaultdict(list)  # image_id -> [(xyxy, ahd_cat, area)]
    gtc=Counter()
    for a in d["annotations"]:
        if a.get("ignore"): continue
        area=a["bbox"][2]*a["bbox"][3]; sb=sizebin(area)
        gt[a["image_id"]].append((xywh2xyxy(a["bbox"]), a["category_id"], area)); gtc[sb]+=1
    tot=sum(gtc.values())
    print(f"[STEP1] GT total={tot} small={gtc['small']}({100*gtc['small']/tot:.1f}%) "
          f"medium={gtc['medium']}({100*gtc['medium']/tot:.1f}%) large={gtc['large']}({100*gtc['large']/tot:.1f}%)",flush=True)
    print(f"[0b] prediction source = REGENERATED clean single-stream GPU FP32 (yolo11s.pt, threads=4, conf={CONF}, imgsz={IMG_SIZE})",flush=True)
    m=YOLO("yolo11s.pt")
    m.predict(np.zeros((IMG_SIZE,IMG_SIZE,3),np.uint8),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")
    tp05=Counter(); tp_range=defaultdict(float); imgs=d["images"]
    for k,im in enumerate(imgs):
        p=DATA/seqd[im["sid"]]/im["name"]
        r=m.predict(str(p),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")[0]
        gl=gt.get(im["id"],[])
        if len(r.boxes)==0 or not gl:
            if (k+1)%4000==0: print(f"  {k+1}/{len(imgs)}",flush=True)
            continue
        cls=r.boxes.cls.cpu().numpy().astype(int); ahd=cm[cls]; sel=ahd<N_AHD
        xyxy=r.boxes.xyxy.cpu().numpy()[sel]; sc=r.boxes.conf.cpu().numpy()[sel]; lab=ahd[sel]
        order=np.argsort(-sc)
        # per-IoU-threshold greedy match (recompute matching at each thr, COCO-style)
        for thr in THRS:
            used=set()
            for di in order:
                dl=lab[di]; best=-1; bi=-1
                for gi,(gb,gc,ga) in enumerate(gl):
                    if gi in used or gc!=dl: continue
                    v=iou(xyxy[di],gb)
                    if v>=thr and v>best: best=v; bi=gi
                if bi>=0:
                    used.add(bi); sb=sizebin(gl[bi][2])
                    if thr==0.5: tp05[sb]+=1
                    tp_range[sb]+=1
        if (k+1)%4000==0: print(f"  {k+1}/{len(imgs)}",flush=True)
    nthr=len(THRS)
    print(f"\n[STEP2] TP@IoU0.5 by size + recall (clean point):",flush=True)
    ttp=sum(tp05.values())
    rows=[]
    for sb in ["small","medium","large"]:
        share=100*tp05[sb]/ttp if ttp else 0; rec=tp05[sb]/gtc[sb] if gtc[sb] else 0
        tpr_avg=tp_range[sb]/nthr
        rows.append({"size":sb,"gt":gtc[sb],"gt_share_pct":round(100*gtc[sb]/tot,1),
                     "tp_iou05":tp05[sb],"tp_share_pct":round(share,1),"recall_iou05":round(rec,4),
                     "tp_0.5_0.95_avg":round(tpr_avg,1)})
        print(f"  {sb:>7}: GT={gtc[sb]} ({100*gtc[sb]/tot:.1f}%)  TP={tp05[sb]} share={share:.1f}%  recall={rec:.3f}",flush=True)
    # 0.5:0.95 share ordering cross-check
    tr_tot=sum(tp_range.values())
    print(f"  [0.5:0.95 avg TP share] " + " ".join(f"{sb}={100*tp_range[sb]/tr_tot:.1f}%" for sb in ['small','medium','large']),flush=True)
    with open(f"{RES}/size_mass_raw.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    # STEP3 + heuristic
    frac_below={"small":0.4297,"medium":0.3421,"large":0.2229}   # EXP-GEOM-STALENESS d=2 (measured prior session, cited)
    stale_loss={"small":0.0,"medium":0.030,"large":0.098}        # Table2 |dAP| (paper, cited)
    print(f"\n[STEP3] three-way (TP share / geom frac-below d=2 / measured |staleness dAP|):",flush=True)
    sh={sb:tp05[sb]/ttp for sb in ['small','medium','large']}
    for sb in ['small','medium','large']:
        print(f"  {sb:>7}: TPshare={100*sh[sb]:.1f}%  fracBelow={frac_below[sb]}  staleLoss={stale_loss[sb]}",flush=True)
    prod={sb: sh[sb]*frac_below[sb] for sb in sh}
    print(f"  [heuristic, NOT a derivation] TPshare x fracBelow: " + " ".join(f"{sb}={prod[sb]:.4f}" for sb in ['small','medium','large']),flush=True)
    ord_prod = prod['large']>prod['medium']>prod['small']
    print(f"  heuristic product ordering large>medium>small (matches staleness)? {ord_prod}",flush=True)
    # verdict
    small_share=100*sh['small']; large_share=100*sh['large']
    if large_share+100*sh['medium'] >= 70 and small_share < large_share:
        verdict="SUPPORTED (TP concentrated on large/medium; small low recall/share)"
    elif small_share >= large_share:
        verdict="UNSUPPORTED (TP not large/medium-dominated)"
    else:
        verdict="PARTIAL"
    print(f"\nVERDICT: {verdict}",flush=True)
    print(f"  driving: small TPshare={small_share:.1f}% large TPshare={large_share:.1f}% | small recall={tp05['small']/gtc['small']:.3f} large recall={tp05['large']/gtc['large']:.3f}",flush=True)
    json.dump({"rows":rows,"tp_share":{k:round(100*sh[k],1) for k in sh},"verdict":verdict},
              open(f"{RES}/size_mass_means.json","w"),indent=2)
    print("saved size_mass_raw.csv + size_mass_means.json",flush=True)
if __name__=="__main__": main()
