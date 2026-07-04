"""B1 Task 2 — per-detection dump, FP32 (GPU) vs INT8 (NPU), isolated offline.

Purpose: enable the per-size quantization error decomposition (Task 2). We need
the FULL per-frame detection list (box + score + class) from BOTH devices, with
NO streaming / staleness confound, so the only FP32->INT8 difference is INT8
quantization. This is the "pure quantization" state (threads=4, skip=0) that
Table 2 uses; here we simplify to a plain OFFLINE pass over every frame.

Faithfulness: uses the SAME model loaders and inference adapters as the paper's
eval path (FGModelGPUGeneric.predict for GPU FP32; npu_infer for NPU INT8),
SAME IMG_SIZE=640 letterbox, CONF=0.25, IOU=0.45 NMS, and SAME coco->AHD class
filter (ahd < N_AHD). Boxes are in native 1920x1200 coords (un-letterboxed),
matching GT so downstream size-binning uses identical COCO areaRng.

Determinism: NPU is re-run twice on a probe subset; box/score must be identical
(rev18 measured std=0). torch threads set to 4 (the non-thrashing config).

Output (per device): analysis/b1_dets/<device>.npz with concatenated arrays
  image_id (M,), fid (M,), sid (M,), box (M,4 xyxy native), score (M,), cls (M,)
plus meta json listing frames processed per sid.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

from _step_d_common import (load_val, load_split_for_sid, npu_infer, N_AHD,
                            DATA)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                              set_active_npu_engines, dispose_npu_for, DETECTORS)

OUT_DIR = ROOT / "analysis/b1_dets"
OUT_DIR.mkdir(exist_ok=True)
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
torch.set_num_threads(4)  # non-thrashing postprocess config


def gpu_infer(model, img_path, coco_mapping):
    r = model.predict(str(img_path))
    if not len(r.boxes):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    coco_ids = r.boxes.cls.cpu().numpy().astype(int)
    ahd = coco_mapping[coco_ids]
    sel = ahd < N_AHD
    return (r.boxes.xyxy.cpu().numpy()[sel].astype(np.float32),
            r.boxes.conf.cpu().numpy()[sel].astype(np.float32),
            ahd[sel].astype(np.int32))


def npu_infer_ahd(model, img_path, frame_shape, coco_mapping):
    xyxy, scores, coco_cls = npu_infer(img_path, frame_shape, model)
    if not len(xyxy):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    ahd = coco_mapping[coco_cls]
    sel = ahd < N_AHD
    return xyxy[sel], scores[sel], ahd[sel].astype(np.int32)


def dump_device(label, infer_fn, val, sids):
    print(f"\n===== dumping {label} =====", flush=True)
    rec = {k: [] for k in ("image_id", "fid", "sid", "score", "cls")}
    boxes = []
    frames_per_sid = {}
    t0 = time.time()
    for si, sid in enumerate(sids):
        split = load_split_for_sid(val, sid)
        imgs = split["imgs"]
        cm = split["coco_mapping"]
        seq_dir = split["seq_dir"]
        frame_shape = (imgs[0]["height"], imgs[0]["width"])
        for im in imgs:
            img_path = DATA / seq_dir / im["name"]
            bb, sc, lb = infer_fn(img_path, frame_shape, cm)
            n = len(bb)
            if n:
                boxes.append(bb)
                rec["image_id"].extend([im["id"]] * n)
                rec["fid"].extend([im["fid"]] * n)
                rec["sid"].extend([sid] * n)
                rec["score"].extend(sc.tolist())
                rec["cls"].extend(lb.tolist())
        frames_per_sid[sid] = len(imgs)
        el = time.time() - t0
        print(f"  [{si+1}/{len(sids)}] sid={sid} frames={len(imgs)} "
              f"cum_dets={len(rec['image_id'])} elapsed={el:.0f}s", flush=True)
    box_arr = np.concatenate(boxes, 0).astype(np.float32) if boxes else np.zeros((0, 4), np.float32)
    out = OUT_DIR / f"{label}.npz"
    np.savez_compressed(
        out,
        image_id=np.asarray(rec["image_id"], np.int64),
        fid=np.asarray(rec["fid"], np.int64),
        sid=np.asarray(rec["sid"], np.int64),
        box=box_arr,
        score=np.asarray(rec["score"], np.float32),
        cls=np.asarray(rec["cls"], np.int32),
    )
    (OUT_DIR / f"{label}_meta.json").write_text(json.dumps(
        {"device": label, "n_dets": len(rec["image_id"]),
         "frames_per_sid": frames_per_sid,
         "img_size": 640, "conf": 0.25, "iou": 0.45,
         "wall_sec": round(time.time() - t0, 1)}, indent=2))
    print(f"  wrote {out}  ({len(rec['image_id'])} dets, {time.time()-t0:.0f}s)", flush=True)


def determinism_check(npu, val, sid=5, n=50):
    print(f"\n===== NPU determinism check (sid={sid}, first {n} frames, 2 reps) =====",
          flush=True)
    split = load_split_for_sid(val, sid)
    imgs = split["imgs"][:n]
    cm = split["coco_mapping"]
    seq_dir = split["seq_dir"]
    frame_shape = (imgs[0]["height"], imgs[0]["width"])

    def run():
        allb, alls = [], []
        for im in imgs:
            bb, sc, _ = npu_infer_ahd(npu, DATA / seq_dir / im["name"], frame_shape, cm)
            allb.append(bb); alls.append(sc)
        return allb, alls

    b1, s1 = run()
    b2, s2 = run()
    max_box = 0.0; max_sc = 0.0; nd1 = 0
    for a, b in zip(b1, b2):
        nd1 += len(a)
        if len(a) == len(b) and len(a):
            max_box = max(max_box, float(np.abs(a - b).max()))
    for a, b in zip(s1, s2):
        if len(a) == len(b) and len(a):
            max_sc = max(max_sc, float(np.abs(a - b).max()))
    same_counts = all(len(a) == len(b) for a, b in zip(b1, b2))
    print(f"  frames={len(imgs)} dets={nd1} same_counts={same_counts} "
          f"max_box_diff={max_box:.2e} max_score_diff={max_sc:.2e}", flush=True)
    return same_counts and max_box == 0.0 and max_sc == 0.0


def main():
    val = load_val()
    sids = list(range(len(val["sequences"])))
    total_frames = sum(1 for im in val["images"])
    print(f"{len(sids)} logs, {total_frames} frames total, 2 devices", flush=True)

    # GPU FP32
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    dump_device("gpu_fp32", lambda p, fs, cm: gpu_infer(gpu, p, cm), val, sids)
    del gpu
    torch.cuda.empty_cache()

    # NPU INT8
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)[0]
    set_active_npu_engines([npu])
    ok = determinism_check(npu, val)
    print(f"  DETERMINISM: {'PASS' if ok else 'FAIL'}", flush=True)
    dump_device("npu_int8", lambda p, fs, cm: npu_infer_ahd(npu, p, fs, cm), val, sids)
    dispose_npu_for("yolo11s")
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
