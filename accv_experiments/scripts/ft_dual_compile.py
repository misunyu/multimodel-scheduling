"""Step 6 dual compile: COCO-local and FT-local INT8, SAME train-calib (200).
Same recipe as compiler-check (yolo_640 preset, global8, opset13 onnx)."""
import time, hashlib
from pathlib import Path
from qbcompiler import mxq_compile
CALIB="accv_experiments/results/qbc_calib200_train"
JOBS=[
  ("COCO-local","yolo11s.onnx","accv_experiments/results/qbc_coco_traincalib_global8.mxq"),
  ("FT-local","accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx","accv_experiments/results/qbc_ft_traincalib_global8.mxq"),
]
for name,onnx,out in JOBS:
    print(f"=== compile {name}: {onnx} -> {out} ===",flush=True)
    t0=time.time()
    mxq_compile(model=onnx, calib_data_path=CALIB, save_path=out,
                config_preset="yolo_640", inference_scheme="global8",
                device="gpu", backend="onnx")
    p=Path(out)
    if p.exists():
        print(f"OK {name} {time.time()-t0:.1f}s size={p.stat().st_size} sha={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}",flush=True)
    else:
        print(f"FAIL {name}: output missing",flush=True)
