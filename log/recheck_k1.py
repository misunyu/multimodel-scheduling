"""k=1 anchor re-measurement — NO shim, documented init order.

Purpose: test whether rev30's ResNet k=1 anchor (All-GPU GPU frame-skip 24.3%,
worst sAP 0.0979) reproduces today in .venv, when
  (a) analysis/ort_env.sh is NOT sourced, and
  (b) preload_background_models() is called BEFORE any torch CUDA model,
      as _step_d_common.py:286-290 requires and as phase_rev30_clean.main() does.

Reuses phase_rev30_clean's own functions verbatim (imported, not copied) so the
measurement path is identical to rev30. Nothing in the repo is written: results
go to the scratchpad. Does NOT call phase_rev30_clean.start_sampler(), which
would truncate the existing rev30 util_samples.csv.

Block structure mirrors rev30: for each rep, the 5 placements of the k=1 block
are measured back-to-back in the same fixed order (GGGG, GGNG, GGNN, GNNN, NNNN).
Order is deliberately NOT randomized here — this is a reproduction control, and
matching rev30's order is what makes the comparison valid.
"""
from __future__ import annotations
import csv, json, os, signal, subprocess, sys, time, hashlib, platform
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-video")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

RAW = OUT / "recheck_k1_raw.csv"
UTIL = OUT / "recheck_k1_util.csv"
META = OUT / "recheck_k1_meta.json"
MANIFEST = OUT / "recheck_k1_manifest.csv"      # every attempted cell, incl. failures

K = 1
REPS = 3
SAMPLER_CORE = "23"
SAMPLE_INTERVAL = 0.2

# ---- metadata (recorded before anything is measured) ------------------------
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}>"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

import torch  # noqa: E402  (after os.chdir; torch CUDA is not initialised by import)
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

meta = {
    "purpose": "rev30 k=1 anchor reproduction, no shim, documented init order",
    "t_wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_commit": sh("git rev-parse HEAD"),
    "git_dirty": bool(sh("git status --porcelain -- accv_experiments analysis")),
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "onnxruntime": ort.__version__,
    "ort_available_providers": ort.get_available_providers(),
    "numpy": np.__version__,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "shim_sourced": "vendor_ort_cu12" in os.environ.get("LD_LIBRARY_PATH", ""),
    "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader"),
    "seed": "none — harness draws no random numbers; order fixed to match rev30",
    "order": "per rep: GGGG, GGNG, GGNN, GNNN, NNNN (rev30-identical, not randomized)",
}
try:
    import ultralytics
    meta["ultralytics"] = ultralytics.__version__
except Exception as e:
    meta["ultralytics"] = f"<{type(e).__name__}>"

# ---- documented init order: ORT co-tenant sessions FIRST --------------------
import phase_rev30_clean as R30                      # noqa: E402
from _step_d_common import _BG_PRELOADED             # noqa: E402
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,   # noqa: E402
                              set_active_npu_engines, dispose_npu_for)

torch.set_num_threads(R30.THREADS)
print(f"[meta] torch={meta['torch']} cuda={meta['torch_cuda']} ort={meta['onnxruntime']} "
      f"shim={meta['shim_sourced']}", flush=True)
print(f"[meta] LD_LIBRARY_PATH={meta['LD_LIBRARY_PATH']!r}", flush=True)

t0 = time.time()
R30.preload_background_models(max_level="L1")        # <-- BEFORE any torch CUDA model
sess = _BG_PRELOADED["resnet"][0]
meta["resnet_cotenant_providers"] = sess.get_providers()
meta["resnet_cotenant_on_gpu"] = "CUDAExecutionProvider" in sess.get_providers()
print(f"[gate] ResNet50 co-tenant providers = {sess.get_providers()} "
      f"({time.time()-t0:.1f}s)", flush=True)

val = R30.load_val()
n_models = 8                                          # rev30 full run loaded 8 of each
gpu_models = [FGModelGPUGeneric(R30.DET["ultralytics_pt"]) for _ in range(n_models)]
npu_models = load_npu_engines(R30.DET, R30.DET["multistream_mxq"], R30.NPU_MODE, n_models)
set_active_npu_engines(npu_models)
meta["mxq_path"] = R30.DET["multistream_mxq"]
meta["mxq_sha256"] = sha256(ROOT / R30.DET["multistream_mxq"])
meta["npu_infer_mode"] = R30.NPU_MODE
meta["n_engines_loaded"] = n_models
meta["panel4_sids"] = R30.PANEL4
print(f"[load] {n_models} GPU + {n_models} NPU engines ready ({time.time()-t0:.1f}s) "
      f"mxq={meta['mxq_sha256'][:16]} mode={R30.NPU_MODE}", flush=True)
META.write_text(json.dumps(meta, indent=2))

sp4 = [R30.load_split_for_sid(val, s) for s in R30.PANEL4]

# ---- sampler (own output path; rev30's util_samples.csv is left untouched) ---
samp = subprocess.Popen(["taskset", "-c", SAMPLER_CORE, sys.executable,
                         str(ROOT / "accv_experiments/scripts/gpu_util_sampler.py"),
                         str(UTIL), str(SAMPLE_INTERVAL)])
time.sleep(2.0)

RAW_COLS = ["run_id", "k", "rep", "placement", "gpu_skip_pct", "npu_skip_pct",
            "worst_sap", "mean_sap", "median_sap", "p10_sap",
            "sap_small", "sap_medium", "sap_large", "deadline_margin_ms_mean",
            "per_stream_json", "t_start", "t_end", "period_ms", "fps",
            "detector", "dataset", "timestamp"]
MAN_COLS = ["run_id", "k", "rep", "placement", "status", "detail", "t_start", "t_end"]

def app(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)

run_id = f"recheck_{int(time.time())}"
bg = R30.reg_level(K)
try:
    for rep in range(REPS):
        for r in range(5):                                  # rev30-identical order
            placement = R30.placement_for_ratio(r)
            ps = R30.pstr(placement)
            ng = sum(1 for d in placement if d == "GPU")
            nn = sum(1 for d in placement if d == "NPU")
            torch.set_num_threads(R30.THREADS)
            ta = time.time()
            try:
                agg = R30.measure_multistream_full(R30.PANEL4, sp4, placement,
                                                   gpu_models[:ng], npu_models[:nn], bg)
                tb = time.time()
            except Exception as e:                          # record, never silently drop
                tb = time.time()
                app(MANIFEST, MAN_COLS, {"run_id": run_id, "k": K, "rep": rep, "placement": ps,
                                         "status": "FAILED", "detail": f"{type(e).__name__}: {e}"[:300],
                                         "t_start": round(ta, 3), "t_end": round(tb, 3)})
                print(f"  !! {ps} rep{rep} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            s = R30.summarize(agg, placement)
            row = {"run_id": run_id, "k": K, "rep": rep, "placement": ps,
                   "t_start": round(ta, 3), "t_end": round(tb, 3),
                   "per_stream_json": json.dumps(agg["per_stream"]),
                   "period_ms": round(R30.PERIOD, 3), "fps": R30.FPS,
                   "detector": "yolo11s", "dataset": R30.DATASET, "timestamp": int(tb)}
            row.update(s)
            app(RAW, RAW_COLS, row)
            app(MANIFEST, MAN_COLS, {"run_id": run_id, "k": K, "rep": rep, "placement": ps,
                                     "status": "OK", "detail": "", "t_start": round(ta, 3),
                                     "t_end": round(tb, 3)})
            print(f"  k{K}/rep{rep}/{ps}: gpu_skip={s['gpu_skip_pct']} npu_skip={s['npu_skip_pct']} "
                  f"worst={s['worst_sap']} ({tb-ta:.1f}s)", flush=True)
finally:
    samp.send_signal(signal.SIGTERM)
    try:
        samp.wait(timeout=5)
    except Exception:
        samp.kill()
    try:
        dispose_npu_for("yolo11s")
    except Exception as e:
        print(f"  (dispose warning: {e})", flush=True)

meta["t_wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
META.write_text(json.dumps(meta, indent=2))
print("=== recheck done ===", flush=True)
