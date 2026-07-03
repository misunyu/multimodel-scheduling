"""EXP-GEOM-BIGDELTA: extend GT-geometry frozen-IoU probe to large delays + translation/scale
decomposition + coverage. Tests whether large overtakes medium fragility at realistic high-skip Δ.
GT only. Reuses track-ID association + frac-below (IoU<0.5:0.95 avg) of EXP-GEOM-STALENESS."""
import json, sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0,"accv_experiments/minimal_pipeline")
from step0_compare_devices import ANNOT
RES="accv_experiments/results"
DELTAS=[1,2,3,5,8,12]; THRS=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
def xywh2xyxy(b): x,y,w,h=b; return (x,y,x+w,y+h)
def sizebin(area): return "small" if area<1024 else ("medium" if area<9216 else "large")
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)
def cen(b): x,y,w,h=b; return (x+w/2,y+h/2)
def fb(ious):  # frac-below averaged over 0.5:0.95
    ious=np.asarray(ious); return float(np.mean([np.mean(ious<t) for t in THRS])) if len(ious) else float('nan')

d=json.load(open(ANNOT)); byimg={i["id"]:i for i in d["images"]}
tracks=defaultdict(dict)
for a in d["annotations"]:
    if a.get("ignore"): continue
    im=byimg[a["image_id"]]; tracks[(im["sid"],a["track"])][im["fid"]]=a["bbox"]
print(f"[0a] tracks={len(tracks)} association=track-ID; frac-below=mean_{{0.5:0.95}} IoU<thr (same as EXP-GEOM-STALENESS)")

# frozen IoU + translation-only + scale-only, per cohort
def collect(cohort):
    # cohort 'per' = all f with f+D present (per D); 'fixed' = f with ALL D present
    iou_by=defaultdict(lambda: defaultdict(list)); cov_num=defaultdict(lambda: defaultdict(int)); cov_den=defaultdict(lambda: defaultdict(int))
    tr_by=defaultdict(lambda: defaultdict(list)); sc_by=defaultdict(lambda: defaultdict(list))
    for (sid,trk),fbx in tracks.items():
        fids=set(fbx)
        for f in fids:
            b0=fbx[f]; c0=cen(b0); a0=b0[2]*b0[3]; s0=sizebin(a0)
            if cohort=="fixed" and not all((f+D) in fids for D in DELTAS): 
                # coverage denom still counts? fixed cohort coverage trivially 100% by construction
                continue
            for D in DELTAS:
                # coverage (per cohort): for 'per', denom = objects present at f (size at f)
                if cohort=="per":
                    cov_den[s0][D]+=1
                present=(f+D) in fids
                if cohort=="per" and present: cov_num[s0][D]+=1
                if not present: 
                    continue
                bd=fbx[f+D]; cd=cen(bd); ad=bd[2]*bd[3]; sb=sizebin(ad)  # delivery-frame size
                iou_by[sb][D].append(iou(xywh2xyxy(b0),xywh2xyxy(bd)))
                # translation-only: center moved to cd, size = t size
                tb=(cd[0]-b0[2]/2,cd[1]-b0[3]/2,cd[0]+b0[2]/2,cd[1]+b0[3]/2)
                tr_by[sb][D].append(iou(xywh2xyxy(b0),tb))
                # scale-only: center fixed c0, size = t+D size
                sbx=(c0[0]-bd[2]/2,c0[1]-bd[3]/2,c0[0]+bd[2]/2,c0[1]+bd[3]/2)
                sc_by[sb][D].append(iou(xywh2xyxy(b0),sbx))
    return iou_by,cov_num,cov_den,tr_by,sc_by

iouP,covN,covD,trP,scP = collect("per")
iouF,_,_,trF,scF = collect("fixed")
nfix=sum(1 for (s,t),fb_ in tracks.items() for f in fb_ if all((f+D) in fb_ for D in DELTAS))
print(f"[0c] fixed-cohort anchors (f with all Δ present)={nfix}")

# 0a regression: per-cohort Δ=2
print("\n[0a regression Δ=2 frac-below] " + " ".join(f"{s}={fb(iouP[s][2]):.3f}" for s in ['small','medium','large']) + "  (expect ~0.43/0.34/0.22)")

# 0b coverage table (per cohort)
print("\n[0b] per-size track coverage (per-Δ cohort): num/den = fraction trackable to t+Δ")
print(f"{'size':>7} " + " ".join(f"Δ{D:>2}" for D in DELTAS))
covok={}
for s in ['small','medium','large']:
    cells=[]
    for D in DELTAS:
        c = covN[s][D]/covD[s][D] if covD[s][D] else 0; cells.append(c); covok[(s,D)]=c
    print(f"{s:>7} " + " ".join(f"{100*c:>4.0f}%{'!' if c<0.70 else ' '}" for c in cells) + "   (! = <70% survivorship-flag)")

# STEP1 frac-below tables both cohorts
def tbl(iou_by,name):
    print(f"\n[STEP1] frac-below — cohort {name}")
    print(f"{'size':>7} " + " ".join(f"Δ{D:>2}" for D in DELTAS))
    res={}
    for s in ['small','medium','large']:
        row=[fb(iou_by[s][D]) for D in DELTAS]; res[s]=row
        print(f"{s:>7} " + " ".join(f"{v:.3f}" for v in row))
    return res
rP=tbl(iouP,"(i) per-Δ"); rF=tbl(iouF,"(ii) fixed")

# STEP2 crossover + gap
print("\n[STEP2] large−medium frac-below gap vs Δ (negative = large less fragile):")
for name,res in [("per",rP),("fixed",rF)]:
    gaps=[res['large'][i]-res['medium'][i] for i in range(len(DELTAS))]
    cx=next((DELTAS[i] for i in range(len(DELTAS)) if res['large'][i]>=res['medium'][i]),None)
    print(f"  {name}: " + " ".join(f"Δ{DELTAS[i]}={gaps[i]:+.3f}" for i in range(len(DELTAS))) + f"  -> crossover Δ={cx}")

# STEP3 translation vs scale at large Δ (use Δ=8 and 12)
print("\n[STEP3] translation-only vs scale-only frac-below (per-Δ cohort):")
for D in [8,12]:
    print(f"  Δ={D}: " + " | ".join(f"{s}: trans={fb(trP[s][D]):.3f} scale={fb(scP[s][D]):.3f} full={fb(iouP[s][D]):.3f}" for s in ['small','medium','large']))

# STEP4 skip->delta mapping
print("\n[STEP4] approx skip->Δ: s=0.5->Δ2, 0.8->Δ5, 0.9->Δ10 (1/(1-s)). Paper points: ~48% crossover->Δ~2, "
      "~59% skip->Δ~2.4, ~80%(k8)->Δ~5, VLM 100%->saturated(Δ>>12).")

# verdict: largest adequately-covered Δ where compare valid (all sizes >=70%)
covered=[D for D in DELTAS if all(covok[(s,D)]>=0.70 for s in ['small','medium','large'])]
maxD=max(covered) if covered else None
cxP=next((DELTAS[i] for i in range(len(DELTAS)) if rP['large'][i]>=rP['medium'][i] and all(covok[(s,DELTAS[i])]>=0.70 for s in ['small','medium','large'])),None)
realistic = cxP is not None and cxP<=5
if cxP is not None and realistic: verdict="CLOSED (large overtakes medium within realistic Δ, adequate coverage)"
elif maxD is None: verdict="INCONCLUSIVE (coverage too low)"
elif cxP is None: verdict="NOT CLOSED (large does not overtake medium within covered Δ)"
else: verdict=f"NOT CLOSED (crossover only at Δ={cxP}, beyond realistic high-contention range)"
lm = f"large={rP['large'][DELTAS.index(maxD)]:.3f} medium={rP['medium'][DELTAS.index(maxD)]:.3f}" if maxD else "n/a"
print(f"\nVERDICT: {verdict}")
print(f"  largest adequately-covered Δ={maxD}; at it {lm}; crossover Δ(covered)={cxP}")
# save
out=[]
for name,res,iou_by in [("per",rP,iouP),("fixed",rF,iouF)]:
    for s in ['small','medium','large']:
        for i,D in enumerate(DELTAS):
            out.append({"cohort":name,"size":s,"delta":D,"frac_below":round(res[s][i],4),
                        "coverage":round(covok.get((s,D),float('nan')),3) if name=="per" else "",
                        "n":len(iou_by[s][D])})
with open(f"{RES}/geom_bigdelta_raw.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); [w.writerow(r) for r in out]
json.dump({"verdict":verdict,"crossover_per_covered":cxP,"max_covered_delta":maxD,
           "per":rP,"fixed":rF},open(f"{RES}/geom_bigdelta_means.json","w"),indent=2,default=str)
print("saved geom_bigdelta_raw.csv + geom_bigdelta_means.json")
