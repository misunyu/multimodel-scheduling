"""k=1 anchor re-measurement WITH rev30's execution prefix (2026-08-28, post-reboot).

Difference from log/recheck_k1.py: that run entered k=1 cold. rev30's k=1 cells
were all preceded by the k=0 block (3 reps x 5 placements), so they were measured
from a warm state. This run reproduces that prefix in a single process:

    for k in [0, 1]:
        for rep in 0..2:
            for r in 0..4:   # GGGG, GGNG, GGNN, GNNN, NNNN  (rev30 order)

Everything else is identical to rev30 and to the previous re-measurement:
  - analysis/ort_env.sh NOT sourced (LD_LIBRARY_PATH empty)
  - preload_background_models() called BEFORE any torch CUDA model
  - phase_rev30_clean's measure_multistream_full / summarize imported, not copied
  - 8 GPU + 8 NPU engines, mxq b2441f9d, infer_mode "single", threads 4

Pre-registered decision rule (fixed BEFORE this run, do not revise after seeing
results) — compare the three k=1 All-GPU (GGGG) reps against rev30 k=1 (n=10):
  PASS if all 3 reps fall inside rev30's observed range
       GPU deadline miss in [21.75, 28.56] %  AND  worst sAP in [0.0901, 0.1042]
       AND the paired d = worst(GGGG) - worst(NNNN) 95% t-CI overlaps rev30's
       [+0.01233, +0.01762].
  PASS  -> existing rev30 reps and new reps may be pooled; measure only the shortfall.
  FAIL  -> do not pool; measure the representative points at n=10 from scratch.
No cell is excluded on the basis of its measured values. Every attempted cell is
recorded in the manifest with its status.

Writes only into log/. Does NOT call phase_rev30_clean.start_sampler(), which
would truncate the existing rev30 util_samples.csv.
"""
from __future__ import annotations
import csv, json, os, signal, subprocess, sys, time, hashlib, platform
from pathlib import Path

OUT = Path(__file__).resolve().parent           # log/
ROOT = OUT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

RAW = OUT / "recheck_prefix_raw.csv"
UTIL = OUT / "recheck_prefix_util.csv"
META = OUT / "recheck_prefix_meta.json"
MANIFEST = OUT / "recheck_prefix_manifest.csv"

K_SEQ = [(0, 3), (1, 3)]                        # (k, reps) — k-major, as in rev30
SAMPLER_CORE = "23"
SAMPLE_INTERVAL = 0.2


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


import torch  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

meta = {
    "purpose": "rev30 k=1 anchor reproduction WITH k=0 warm-up prefix, no shim, documented init order",
    "t_wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "boot_time": sh("uptime -s"),
    "uptime": sh("uptime -p"),
    "git_commit": sh("git rev-parse HEAD"),
    "python": platform.python_version(),
    "interpreter": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "onnxruntime": ort.__version__,
    "numpy": np.__version__,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "shim_sourced": "vendor_ort_cu12" in os.environ.get("LD_LIBRARY_PATH", ""),
    "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
    "seed": "none — harness draws no random numbers; order fixed to match rev30",
    "order": "k-major: k=0 x3 reps then k=1 x3 reps; within each rep GGGG, GGNG, GGNN, GNNN, NNNN",
    "decision_rule": ("PASS if all three k=1 GGGG reps have gpu_skip in [21.75,28.56] and "
                      "worst_sap in [0.0901,0.1042], and paired-d 95% t-CI overlaps "
                      "rev30's [+0.01233,+0.01762]. Fixed before the run."),
}
try:
    import ultralytics
    meta["ultralytics"] = ultralytics.__version__
except Exception as e:
    meta["ultralytics"] = f"<{type(e).__name__}>"

import phase_rev30_clean as R30                      # noqa: E402
from _step_d_common import _BG_PRELOADED             # noqa: E402
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,   # noqa: E402
                              set_active_npu_engines, dispose_npu_for)

torch.set_num_threads(R30.THREADS)
print(f"[meta] boot={meta['boot_time']} torch={meta['torch']} cuda={meta['torch_cuda']} "
      f"ort={meta['onnxruntime']} shim={meta['shim_sourced']}", flush=True)

t0 = time.time()
R30.preload_background_models(max_level="L1")        # BEFORE any torch CUDA model
sess = _BG_PRELOADED["resnet"][0]
meta["resnet_cotenant_providers"] = sess.get_providers()
meta["resnet_cotenant_on_gpu"] = "CUDAExecutionProvider" in sess.get_providers()
print(f"[gate] ResNet50 co-tenant providers = {sess.get_providers()} ({time.time()-t0:.1f}s)", flush=True)

val = R30.load_val()
n_models = 8
gpu_models = [FGModelGPUGeneric(R30.DET["ultralytics_pt"]) for _ in range(n_models)]
npu_models = load_npu_engines(R30.DET, R30.DET["multistream_mxq"], R30.NPU_MODE, n_models)
set_active_npu_engines(npu_models)
meta.update({"mxq_path": R30.DET["multistream_mxq"],
             "mxq_sha256": sha256(ROOT / R30.DET["multistream_mxq"]),
             "npu_infer_mode": R30.NPU_MODE,
             "n_engines_loaded": n_models,
             "panel4_sids": R30.PANEL4})
print(f"[load] {n_models} GPU + {n_models} NPU engines ready ({time.time()-t0:.1f}s) "
      f"mxq={meta['mxq_sha256'][:16]} mode={R30.NPU_MODE}", flush=True)
META.write_text(json.dumps(meta, indent=2))

sp4 = [R30.load_split_for_sid(val, s) for s in R30.PANEL4]

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


run_id = f"prefix_{int(time.time())}"
try:
    for k, reps in K_SEQ:
        bg = R30.reg_level(k)
        for rep in range(reps):
            for r in range(5):
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
                except Exception as e:
                    tb = time.time()
                    app(MANIFEST, MAN_COLS, {"run_id": run_id, "k": k, "rep": rep, "placement": ps,
                                             "status": "FAILED",
                                             "detail": f"{type(e).__name__}: {e}"[:300],
                                             "t_start": round(ta, 3), "t_end": round(tb, 3)})
                    print(f"  !! k{k}/rep{rep}/{ps} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                s = R30.summarize(agg, placement)
                row = {"run_id": run_id, "k": k, "rep": rep, "placement": ps,
                       "t_start": round(ta, 3), "t_end": round(tb, 3),
                       "per_stream_json": json.dumps(agg["per_stream"]),
                       "period_ms": round(R30.PERIOD, 3), "fps": R30.FPS,
                       "detector": "yolo11s", "dataset": R30.DATASET, "timestamp": int(tb)}
                row.update(s)
                app(RAW, RAW_COLS, row)
                app(MANIFEST, MAN_COLS, {"run_id": run_id, "k": k, "rep": rep, "placement": ps,
                                         "status": "OK", "detail": "",
                                         "t_start": round(ta, 3), "t_end": round(tb, 3)})
                print(f"  k{k}/rep{rep}/{ps}: gpu_skip={s['gpu_skip_pct']} "
                      f"npu_skip={s['npu_skip_pct']} worst={s['worst_sap']} ({tb-ta:.1f}s)", flush=True)
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
print("=== prefix recheck done ===", flush=True)
