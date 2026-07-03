"""EXP-GEOM-STALENESS: GT-only geometric staleness probe. No detector/GPU/NPU.
Frozen-detection overlap IoU(box_t, box_{t+d}) of the SAME track, stratified by delivery-frame
size (COCO bins). Argoverse-HD val, 24 logs, track-ID association."""
import json, sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0,"accv_experiments/minimal_pipeline")
from step0_compare_devices import ANNOT
RES="accv_experiments/results"
DELTAS=[1,2,3]
THRS=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
def xywh2xyxy(b): x,y,w,h=b; return (x,y,x+w,y+h)
def iou(a,b):
    ax1,ay1,ax2,ay2=xywh2xyxy(a); bx1,by1,bx2,by2=xywh2xyxy(b)
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    aa=(ax2-ax1)*(ay2-ay1); ab=(bx2-bx1)*(by2-by1)
    return inter/(aa+ab-inter+1e-9)
def center(b): x,y,w,h=b; return (x+w/2,y+h/2)
def sizebin(area):
    return "small" if area<1024 else ("medium" if area<9216 else "large")

d=json.load(open(ANNOT))
byimg={i["id"]:(i["sid"],i["fid"]) for i in d["images"]}
# per (sid,track): fid -> bbox  (exclude ignore)
tracks=defaultdict(dict); nign=0
for a in d["annotations"]:
    if a.get("ignore"): nign+=1; continue
    sid,fid=byimg[a["image_id"]]
    tracks[(sid,a["track"])][fid]=a["bbox"]
print(f"[0a] logs=24 frames={len(d['images'])} anns={len(d['annotations'])} (ignored {nign}) tracks={len(tracks)}")
print("[0b] association = TRACK IDs (val.json 'track' field). [0c] size bin by delivery-frame area (t+d), COCO px^2. fps=30 (33ms/frame)")

# 0d smoke: largest & smallest track in sid 0 (by median area), per-frame disp + IoU(t,t+1)
def med_area(fb): return float(np.median([b[2]*b[3] for b in fb.values()]))
s0={k:v for k,v in tracks.items() if k[0]==0 and len(v)>=4}
big=max(s0,key=lambda k:med_area(s0[k])); small=min(s0,key=lambda k:med_area(s0[k]))
def smoke(k):
    fb=s0[k]; fids=sorted(fb)[:5]; out=[]
    for i in range(len(fids)-1):
        f0,f1=fids[i],fids[i+1]
        if f1==f0+1:
            c0,c1=center(fb[f0]),center(fb[f1]); disp=((c0[0]-c1[0])**2+(c0[1]-c1[1])**2)**.5
            out.append(f"f{f0}->{f1} disp={disp:.1f}px IoU={iou(fb[f0],fb[f1]):.3f}")
    return out
print(f"[0d smoke] NEAR/LARGE track{big[1]} medA={med_area(s0[big]):.0f}: {smoke(big)}")
print(f"[0d smoke] FAR/SMALL track{small[1]} medA={med_area(s0[small]):.0f}: {smoke(small)}")

# STEP1/2: accumulate per (size,delta)
acc=defaultdict(lambda: {"disp":[], "ndisp":[], "scale":[], "iou":[]})
cover={dl:[0,0] for dl in DELTAS}  # [pairs_with_t+d, pairs_with_t]
for (sid,tr),fb in tracks.items():
    fids=set(fb)
    for f in fids:
        for dl in DELTAS:
            cover[dl][1]+=1
            if f+dl in fids:
                cover[dl][0]+=1
                b0,bd=fb[f],fb[f+dl]
                a0=b0[2]*b0[3]; ad=bd[2]*bd[3]
                c0,cd=center(b0),center(bd); disp=((c0[0]-cd[0])**2+(c0[1]-cd[1])**2)**.5
                sb=sizebin(ad)  # delivery-frame size
                A=acc[(sb,dl)]
                A["disp"].append(disp); A["ndisp"].append(disp/max(1e-6,a0**.5))
                A["scale"].append((ad/max(1e-6,a0))**.5); A["iou"].append(iou(b0,bd))
print("\n[0b] per-delta track coverage (objects present at both t and t+d / present at t):")
for dl in DELTAS: print(f"   d={dl}: {cover[dl][0]}/{cover[dl][1]} = {100*cover[dl][0]/cover[dl][1]:.1f}%")

# reduce + table
rows=[]
def frac_below_range(ious):
    ious=np.array(ious); return float(np.mean([np.mean(ious<t) for t in THRS]))
print("\n=== STEP2: per-size geometric staleness (mean) ===")
print(f"{'d':>2} {'size':>7} {'n':>7} {'disp_px':>8} {'ndisp':>7} {'scale':>6} {'frozenIoU':>10} {'IoU<0.5':>8} {'below0.5:0.95':>13}")
for dl in DELTAS:
    for sb in ["small","medium","large"]:
        A=acc[(sb,dl)]; n=len(A["iou"])
        if n==0: continue
        r={"delta":dl,"size":sb,"n":n,"disp_px":round(np.mean(A["disp"]),2),
           "ndisp":round(np.mean(A["ndisp"]),3),"scale":round(np.mean(A["scale"]),4),
           "frozen_iou":round(np.mean(A["iou"]),4),"frac_iou_lt0.5":round(float(np.mean(np.array(A["iou"])<0.5)),4),
           "frac_below_0.5_0.95":round(frac_below_range(A["iou"]),4)}
        rows.append(r)
        print(f"{dl:>2} {sb:>7} {n:>7} {r['disp_px']:>8} {r['ndisp']:>7} {r['scale']:>6} {r['frozen_iou']:>10} {r['frac_iou_lt0.5']:>8} {r['frac_below_0.5_0.95']:>13}")
with open(f"{RES}/geom_staleness_raw.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

# STEP3: ordering at each delta (frac_below_0.5_0.95)
print("\n=== STEP3: ordering of frac_below_0.5:0.95 (geometric recall-loss proxy) ===")
def get(dl,sb,k): 
    for r in rows:
        if r["delta"]==dl and r["size"]==sb: return r[k]
    return None
verdicts=[]
for dl in DELTAS:
    s,m,l=(get(dl,x,"frac_below_0.5_0.95") for x in ["small","medium","large"])
    order="large>medium>small" if (l>m>s) else ("monotonic? no")
    mono = (l>m>s)
    print(f"  d={dl}: small={s} medium={m} large={l}  -> {'large>medium>small (monotonic)' if mono else 'NON-monotonic'}")
    verdicts.append(mono)
# measured staleness ordering (paper, for comparison only — not re-measured here)
print("\n  [paper measured staleness ordering] Table2 thread-lever dAP: small~0 / medium -0.030 / large -0.098;")
print("  Table3 (~59% skip) dsAP: small +0.002 / medium +0.040 / large +0.095  => large>medium>small")
# verdict at delta=2 primary
s2,m2,l2=(get(2,x,"frac_below_0.5_0.95") for x in ["small","medium","large"])
if all(verdicts): verdict="SUPPORTED (large>medium>small monotonic at all d)"
elif l2>s2: verdict="PARTIAL (large>small at d=2 but not strictly monotonic at all d)"
else: verdict="UNSUPPORTED (large not > small)"
print(f"\nVERDICT: {verdict}")
print(f"  d=2 frac_below_0.5:0.95: small={s2} medium={m2} large={l2}")
import json as J; J.dump({"rows":rows,"coverage":{str(k):cover[k] for k in cover},"verdict":verdict,
    "d2":{"small":s2,"medium":m2,"large":l2}}, open(f"{RES}/geom_staleness_means.json","w"),indent=2)
print("saved geom_staleness_raw.csv + geom_staleness_means.json")
