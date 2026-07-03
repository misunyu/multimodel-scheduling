"""Build Ultralytics YOLO dataset from Argoverse-HD (COCO-80 class space).
Labels use COCO class ids (AHD->COCO) so the fine-tuned model keeps the 80-class
head and the rev19 eval harness (coco_mapping) works unchanged. Images symlinked.
"""
import json, os
from pathlib import Path
BASE=Path("/home/msyu/PycharmProjects/multimodel-scheduling-video")
ANN=BASE/"accv_experiments/data/argoverse_hd/Argoverse-HD/annotations"
TRK=BASE/"accv_experiments/data/argoverse_hd/Argoverse-1.1/argoverse-tracking"
OUT=BASE/"accv_experiments/data/ahd_yolo"

def ahd2coco(coco_mapping):
    m={}
    for coco_id,ahd in enumerate(coco_mapping):
        if ahd<8 and ahd not in m: m[ahd]=coco_id
    return [m[i] for i in range(8)]

def build(split):
    j=json.load(open(ANN/f"{split}.json"))
    cm=j["coco_mapping"]; a2c=ahd2coco(cm)
    imgs={im["id"]:im for im in j["images"]}
    by_img={}
    for an in j["annotations"]:
        if an.get("ignore"): continue
        by_img.setdefault(an["image_id"],[]).append(an)
    imgdir=OUT/"images"/split; lbldir=OUT/"labels"/split
    imgdir.mkdir(parents=True,exist_ok=True); lbldir.mkdir(parents=True,exist_ok=True)
    seqd=j["seq_dirs"]; n_img=0; n_box=0
    for iid,im in imgs.items():
        seq_dir=seqd[im["sid"]]  # e.g. train/<seq>/ring_front_center
        src=TRK/seq_dir/im["name"]
        if not src.exists(): continue
        flat=seq_dir.replace("/","_")+"__"+im["name"]
        link=imgdir/flat
        if not link.exists():
            try: link.symlink_to(src)
            except FileExistsError: pass
        W,H=im["width"],im["height"]
        lines=[]
        for an in by_img.get(iid,[]):
            x,y,w,h=an["bbox"]; cid=a2c[an["category_id"]]
            cx=(x+w/2)/W; cy=(y+h/2)/H; nw=w/W; nh=h/H
            if nw<=0 or nh<=0: continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box+=1
        (lbldir/(Path(flat).stem+".txt")).write_text("\n".join(lines))
        n_img+=1
    print(f"{split}: images={n_img} boxes={n_box} ahd2coco={a2c}")
    return n_img

OUT.mkdir(parents=True,exist_ok=True)
nt=build("train"); nv=build("val")
# COCO-80 names (standard) so nc=80 head is preserved
coco_names=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
yaml=f"""# Argoverse-HD as COCO-80 class space (AHD labels mapped to COCO ids)
path: {OUT}
train: images/train
val: images/val
nc: 80
names: {coco_names}
"""
(OUT/"ahd_coco80.yaml").write_text(yaml)
print("wrote", OUT/"ahd_coco80.yaml", "| train_imgs", nt, "val_imgs", nv)
