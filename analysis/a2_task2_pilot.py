"""A2 Task 2 calibration PILOT — alternative GPU co-tenants for identity invariance.

Goal: find, for each alternative co-tenant, the instance count that matches the
ResNet50 reference GPU deadline-miss rates (~24/50/67%) within +-3pp, AND verify
the co-tenant is genuinely GPU-BOUND (D1-style: gpu_util high when run alone),
since ORT-CUDA co-tenants were shown to become host-bound. host CPU util must not
exceed the ResNet reference by >10pp (else it stresses the NPU/host path, not GPU).

Priority: ViT-B/16 (torch-CUDA) first; VGG19 (ORT-CUDA) only if it passes the
GPU-bound check.

Everything measured in ONE session (no cross-time drift): ResNet reference is
re-measured now with util logging, then alternatives calibrated to match.
All-GPU N=4, PANEL4, threads=4, 1 rep per trial (pilot).

Output: analysis/a2_task2_pilot.csv, console calibration table + gate estimate.
"""
from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "accv_experiments/scripts"
sys.path.insert(0, str(SD))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

import psutil
import pynvml
import torch
from _step_d_common import load_val, load_split_for_sid, _bg_resnet50_loop, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, DETECTORS)
from phase_rev30_clean import measure_multistream_full, summarize, PANEL4, THREADS
import step_h2_robustness as h2
import a2_cotenants as a2c

DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
OUT = ROOT / "analysis/a2_task2_pilot.csv"
TARGETS = [24.3, 51.5, 67.3]     # ResNet k=1,3,8 reference GPU DM
REF_K = {24.3: 1, 51.5: 3, 67.3: 8}


class UtilSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self._ev = threading.Event()
        self.cpu, self.gpu = [], []
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self._ev.is_set():
            c = psutil.cpu_percent(self.interval)
            self.cpu.append(c)
            self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu)

    def stop(self):
        self._ev.set()
        self.join(timeout=2)
        return (round(float(np.mean(self.cpu)), 1) if self.cpu else -1,
                round(float(np.mean(self.gpu)), 1) if self.gpu else -1)


def gpu_bound_check(name, start_fn, secs=4):
    """Run co-tenant ALONE, measure gpu_util (D1-style GPU-bound verdict)."""
    stops, threads = start_fn()
    samp = UtilSampler(); samp.start()
    time.sleep(secs)
    cpu, gpu = samp.stop()
    from _step_d_common import stop_background
    stop_background(stops, threads)
    verdict = "GPU-BOUND" if gpu >= 40 else ("host-bound" if gpu < 15 else "mixed")
    print(f"  [gpu-bound check] {name}: gpu_util={gpu}% cpu={cpu}% -> {verdict}", flush=True)
    return gpu, cpu, verdict


def measure_gpu_cell(gpu_models, splits, bg_level):
    torch.set_num_threads(THREADS)
    samp = UtilSampler(); samp.start()
    agg = measure_multistream_full(PANEL4, splits, ["GPU"] * 4, gpu_models[:4], [], bg_level)
    cpu, gpu = samp.stop()
    s = summarize(agg, ["GPU"] * 4)
    return s["gpu_skip_pct"], s["worst_sap"], cpu, gpu


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 GPU models + resnet co-tenant ...", flush=True)
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    preload_background_models("L1")           # resnet
    a2c.preload_vit(batch=1)                   # ViT-B/16 torch-CUDA
    rows = []

    def reg(loop_fn, k, tag):
        name = f"{tag}{k}"
        h2.BG_VARIANTS[name] = [loop_fn] * k
        return name

    # ---- GPU-bound checks (D1-style) ----
    print("\n=== GPU-bound verification (co-tenant alone) ===", flush=True)
    gpu_bound_check("resnet50x4", lambda: h2.start_bg_custom(reg(_bg_resnet50_loop, 4, "R")))
    vit_gpu, vit_cpu, vit_verdict = gpu_bound_check(
        "vit_b16x2", lambda: h2.start_bg_custom(reg(a2c._bg_vit_loop, 2, "V")))

    # ---- ResNet reference (re-measured now, k=1,3,8) ----
    print("\n=== ResNet reference (current env, All-GPU N=4) ===", flush=True)
    ref = {}
    for k in [1, 3, 8]:
        dm, worst, cpu, gpu = measure_gpu_cell(gpu_models, splits, reg(_bg_resnet50_loop, k, "R"))
        ref[k] = {"dm": dm, "cpu": cpu, "gpu": gpu, "worst": worst}
        rows.append({"cotenant": "resnet50", "param": f"k={k}", "gpu_dm": dm,
                     "worst_sap": worst, "cpu": cpu, "gpu_util": gpu})
        print(f"  resnet k={k}: GPU_DM={dm:.1f}% worst={worst:.4f} cpu={cpu}% gpu_util={gpu}%", flush=True)

    # ---- ViT calibration search (only if GPU-bound) ----
    if vit_verdict == "GPU-BOUND":
        print("\n=== ViT-B/16 calibration (search k for GPU_DM 24/50/67) ===", flush=True)
        tried = {}
        for k in [1, 2, 4, 8, 12, 16]:
            dm, worst, cpu, gpu = measure_gpu_cell(gpu_models, splits, reg(a2c._bg_vit_loop, k, "V"))
            tried[k] = dm
            rows.append({"cotenant": "vit_b16", "param": f"k={k}", "gpu_dm": dm,
                         "worst_sap": worst, "cpu": cpu, "gpu_util": gpu})
            print(f"  ViT k={k}: GPU_DM={dm:.1f}% worst={worst:.4f} cpu={cpu}% gpu_util={gpu}%", flush=True)
            if dm >= 70:
                break
        # nearest k to each target
        print("\n  ViT k -> GPU_DM:", {k: round(v, 1) for k, v in tried.items()})
    else:
        print(f"\n  ViT NOT GPU-bound ({vit_verdict}) -> flag; calibration skipped", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cotenant", "param", "gpu_dm", "worst_sap", "cpu", "gpu_util"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT}", flush=True)
    print("\nResNet reference host CPU (for +10pp flag rule):",
          {k: ref[k]["cpu"] for k in ref})
    del gpu_models


if __name__ == "__main__":
    main()
