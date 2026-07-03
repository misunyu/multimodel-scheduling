"""EXP-FT-COMPILER-CHECK — reproduction compile of COCO yolo11s via local qbcompiler.
Documented assumptions (vendor recipe for b2441f9d is undocumented):
  - input ONNX: Ultralytics export, 640x640, opset 13
  - calibration: 200 Argoverse-HD val frames, uniform sample (vendor calib unknown)
  - preset: yolo_640 (uint8 input + letterbox640 pad114; extends 'detection')
  - inference_scheme: global8 (the mode the vendor mxq is used in for Table 1 baseline)
"""
import sys, time, hashlib
from pathlib import Path
from qbcompiler import mxq_compile

ONNX = "yolo11s.onnx"
CALIB = "accv_experiments/results/qbc_calib200"
OUT = "accv_experiments/results/qbc_local_yolo11s_global8.mxq"

print(f"=== qbc reproduction compile -> {OUT} ===", flush=True)
t0 = time.time()
try:
    mxq_compile(
        model=ONNX,
        calib_data_path=CALIB,
        save_path=OUT,
        config_preset="yolo_640",
        inference_scheme="global8",
        device="gpu",
        backend="onnx",
    )
    dt = time.time() - t0
    p = Path(OUT)
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"COMPILE OK in {dt:.1f}s  size={p.stat().st_size}  sha256={h}", flush=True)
    else:
        print(f"COMPILE returned but {OUT} missing", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"COMPILE FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
