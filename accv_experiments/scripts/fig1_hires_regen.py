"""EXP-FIG1-HIRES: regenerate Figure 1 hi-res (1 re-inference, 4 fixed panels).
Helpers replicated verbatim from generate_fig1.py (same thresholds/logic) to avoid
importing its broken module-level PIN path. Only resolution/legend/label-font change."""
import sys, hashlib
from pathlib import Path
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
ROOT=Path("/home/msyu/PycharmProjects/multimodel-scheduling-video")
sys.path.insert(0,str(ROOT/"accv_experiments/scripts"))
sys.path.insert(0,str(ROOT/"accv_experiments/minimal_pipeline"))
import _step_d_common as cm
from _step_d_common import load_val, load_split_for_sid, npu_infer, preload_npu_instances
from step0_compare_devices import DATA, CONF, IOU, IMG_SIZE, N_AHD

SAFETY_SMALL={0,1,3,6,7}
CLASS_NAMES={0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"bus",5:"truck",6:"traffic_light",7:"stop_sign"}
IOU_MATCH=0.4
PANELS=[(2,5),(13,41),(17,163),(22,1)]
EXPECT={2:["person","person","traffic_light","traffic_light","traffic_light"],
        13:["traffic_light","traffic_light","traffic_light"],
        17:["person","person","person"], 22:["person"]}
OUT=ROOT/"paper/figures/vis_failure_examples.pdf"
NEW_GREEN="All GPU (FP32) detections"; NEW_RED="Missed by INT8 NPU (GPU detects)"

def load_gpu():
    import os; os.environ.setdefault("YOLO_VERBOSE","False")
    from ultralytics import YOLO
    m=YOLO("yolo11s.pt")
    m.predict(np.zeros((IMG_SIZE,IMG_SIZE,3),np.uint8),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")
    return m
def gpu_predict(m,p,cmap):
    r=m.predict(str(p),imgsz=IMG_SIZE,conf=CONF,iou=IOU,verbose=False,device="cuda")[0]
    if not len(r.boxes): return np.zeros((0,4),np.float32),np.zeros(0,np.float32),np.zeros(0,np.int32)
    ci=r.boxes.cls.cpu().numpy().astype(int); ahd=cmap[ci]; sel=ahd<N_AHD
    return r.boxes.xyxy.cpu().numpy()[sel].astype(np.float32),r.boxes.conf.cpu().numpy()[sel].astype(np.float32),ahd[sel].astype(np.int32)
def npu_predict(npu,p,fs,cmap):
    xyxy,sc,cc=npu_infer(p,fs,npu)
    if len(xyxy)==0: return np.zeros((0,4),np.float32),np.zeros(0,np.float32),np.zeros(0,np.int32)
    ahd=cmap[cc]; sel=ahd<N_AHD; return xyxy[sel],sc[sel],ahd[sel].astype(np.int32)
def iou_xyxy(a,b):
    if len(b)==0: return np.zeros(0,np.float32)
    x1=np.maximum(a[0],b[:,0]);y1=np.maximum(a[1],b[:,1]);x2=np.minimum(a[2],b[:,2]);y2=np.minimum(a[3],b[:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa+ab-inter+1e-9)
def annotate_hires(img,info):
    img=img.copy(); H,W=img.shape[:2]; s=W/1920.0
    gt=max(2,round(3*s)); rt=max(3,round(6*s)); fs=1.05*s; ft=max(2,round(3*s))
    for bb in info["gpu_bb"]:
        x1,y1,x2,y2=bb.astype(int); cv2.rectangle(img,(x1,y1),(x2,y2),(60,200,60),gt)
    for m in info["missed"]:
        x1,y1,x2,y2=m["bbox"].astype(int); cv2.rectangle(img,(x1,y1),(x2,y2),(40,40,230),rt)
        lab=f"{CLASS_NAMES[m['class']]} {m['score']:.2f} (NPU miss)"
        (lw,lh),_=cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,fs,ft)
        cv2.rectangle(img,(x1,max(y1-lh-8,0)),(x1+lw+6,y1),(40,40,230),-1)
        cv2.putText(img,lab,(x1+3,max(y1-5,lh)),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),ft)
    return img

def main():
    val=load_val(); gpu=load_gpu()
    preload_npu_instances(1,infer_mode="global8"); npu=cm._NPU_INSTANCES[0]
    mxq=cm._find_yolo11s_mxq("global8")
    print("mxq:",mxq)
    print("yolo11s.pt sha:",hashlib.sha256(open("yolo11s.pt","rb").read()).hexdigest()[:16])
    print("mxq sha:",hashlib.sha256(open(mxq,"rb").read()).hexdigest()[:16] if mxq else "NA")
    infos=[]
    for sid,fidx in PANELS:
        sp=load_split_for_sid(val,sid); imgs=sp["imgs"]; cmap=sp["coco_mapping"]; seq=sp["seq_dir"]
        fshape=(imgs[0]["height"],imgs[0]["width"]); path=DATA/seq/imgs[fidx]["name"]
        gbb,gsc,glb=gpu_predict(gpu,path,cmap); nbb,nsc,nlb=npu_predict(npu,path,fshape,cmap)
        missed=[]
        for k in range(len(gbb)):
            if glb[k] not in SAFETY_SMALL: continue
            same=nlb==glb[k]
            if same.any() and (iou_xyxy(gbb[k],nbb[same])>IOU_MATCH).any(): continue
            missed.append({"bbox":gbb[k],"score":float(gsc[k]),"class":int(glb[k])})
        infos.append({"sid":sid,"frame_idx":fidx,"img_path":str(path),
                      "gpu_bb":gbb,"gpu_sc":gsc,"gpu_lb":glb,"missed":missed})
        print(f"  sid={sid} frame={fidx}: missed={[CLASS_NAMES[m['class']] for m in missed]}")
    ok=True
    for inf in infos:
        got=sorted(CLASS_NAMES[m["class"]] for m in inf["missed"]); exp=sorted(EXPECT[inf["sid"]])
        if got!=exp: ok=False; print(f"  MISMATCH sid={inf['sid']}: {got} != {exp}")
    if not ok: print("*** §2-2 mismatch — STOP, not saving ***"); sys.exit(2)
    print("§2-2 consistency PASS")
    cols=rows=2
    fig,axes=plt.subplots(rows,cols,figsize=(7.5*cols,4.7*rows),squeeze=False)
    for k,c in enumerate(infos):
        ax=axes[k//cols][k%cols]; ann=cv2.cvtColor(annotate_hires(cv2.imread(c["img_path"]),c),cv2.COLOR_BGR2RGB)
        ax.imshow(ann); ax.set_xticks([]); ax.set_yticks([])
        ms=", ".join(CLASS_NAMES[m["class"]] for m in c["missed"][:5])
        ax.set_title(f"sid {c['sid']}, frame {c['frame_idx']} — NPU misses: {ms}",fontsize=9)
    legend=[mpatches.Patch(color=(60/255,200/255,60/255),label=NEW_GREEN),
            mpatches.Patch(color=(230/255,40/255,40/255),label=NEW_RED)]
    fig.legend(handles=legend,loc="lower center",ncol=2,fontsize=10,bbox_to_anchor=(0.5,-0.01),frameon=False)
    plt.tight_layout(rect=[0,0.04,1,1]); fig.savefig(OUT,format="pdf",dpi=300,bbox_inches="tight"); plt.close(fig)
    print("saved dpi=300",OUT); cm.dispose_npu_for("yolo11s")
if __name__=="__main__": main()
