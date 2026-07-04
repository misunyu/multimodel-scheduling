"""A2 Step 2 — anchor reproduction after ORT-CUDA repair (run with ort_env.sh).

(a) ResNet50 k=1, All-GPU N=4, 3 reps  -> paper baseline (GPU DM ~24%, worst ~0.098)?
(b) L2LM, All-NPU N=4, 3 reps           -> NPU DM ~76% AND co-tenant gpu_util ~49.5%?
(c) synth c=24, All-NPU 1 rep           -> DM ~27% (ORT-independent control)
Logs GPU util + CPU util per cell. Output: analysis/a2_anchor_check.csv + verdict.
"""
from __future__ import annotations

import csv
import subprocess
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

DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
OUT = ROOT / "analysis/a2_anchor_check.csv"
STRESS = ROOT / "analysis/a2_cpu_stress_worker.py"
PY = ROOT / ".venv/bin/python"


class UtilSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self._ev = threading.Event()
        self.cpu, self.gpu = [], []
        pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self._ev.is_set():
            self.cpu.append(psutil.cpu_percent(self.interval))
            self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu)

    def stop(self):
        self._ev.set(); self.join(timeout=2)
        return (round(float(np.mean(self.cpu)), 1) if self.cpu else -1,
                round(float(np.mean(self.gpu)), 1) if self.gpu else -1)


def reg(fn, k, tag):
    name = f"{tag}{k}"; h2.BG_VARIANTS[name] = [fn] * k; return name


def cell(placement, gpu_models, npu_models, splits, bg):
    torch.set_num_threads(THREADS)
    samp = UtilSampler(); samp.start()
    ng = sum(d == "GPU" for d in placement); nn = sum(d == "NPU" for d in placement)
    agg = measure_multistream_full(PANEL4, splits, placement, gpu_models[:ng], npu_models[:nn], bg)
    cpu, gpu = samp.stop()
    s = summarize(agg, placement)
    dm = s["gpu_skip_pct"] if ng else s["npu_skip_pct"]
    return dm, s["worst_sap"], cpu, gpu


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    preload_background_models("L2")
    rows = []

    print("\n=== (a) ResNet k=1, All-GPU N=4, 3 reps ===", flush=True)
    for rep in range(3):
        dm, worst, cpu, gpu = cell(["GPU"] * 4, gpu_models, npu_models, splits, reg(_bg_resnet50_loop, 1, "R"))
        rows.append({"anchor": "a_resnet_k1_allGPU", "rep": rep, "dm": dm, "worst": worst, "cpu": cpu, "gpu_util": gpu})
        print(f"  rep{rep}: GPU_DM={dm:.1f}% worst={worst:.4f} cpu={cpu}% gpu_util={gpu}%", flush=True)

    print("\n=== (b) L2LM, All-NPU N=4, 3 reps ===", flush=True)
    for rep in range(3):
        dm, worst, cpu, gpu = cell(["NPU"] * 4, gpu_models, npu_models, splits, "L2_lm")
        rows.append({"anchor": "b_L2LM_allNPU", "rep": rep, "dm": dm, "worst": worst, "cpu": cpu, "gpu_util": gpu})
        print(f"  rep{rep}: NPU_DM={dm:.1f}% worst={worst:.4f} cpu={cpu}% gpu_util={gpu}%", flush=True)

    print("\n=== (c) synth c=24, All-NPU 1 rep (ORT-independent control) ===", flush=True)
    procs = [subprocess.Popen([str(PY), str(STRESS), str(c)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for c in range(24)]
    time.sleep(1.0)
    try:
        dm, worst, cpu, gpu = cell(["NPU"] * 4, gpu_models, npu_models, splits, "L0")
    finally:
        for p in procs: p.terminate()
        for p in procs:
            try: p.wait(timeout=3)
            except Exception: p.kill()
    rows.append({"anchor": "c_synth_c24_allNPU", "rep": 0, "dm": dm, "worst": worst, "cpu": cpu, "gpu_util": gpu})
    print(f"  rep0: NPU_DM={dm:.1f}% worst={worst:.4f} cpu={cpu}%", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {OUT}", flush=True)

    def mean(anchor, key):
        v = [r[key] for r in rows if r["anchor"] == anchor]
        return float(np.mean(v))
    print("\n=== VERDICT ===")
    a_dm, a_w = mean("a_resnet_k1_allGPU", "dm"), mean("a_resnet_k1_allGPU", "worst")
    b_dm, b_gpu = mean("b_L2LM_allNPU", "dm"), mean("b_L2LM_allNPU", "gpu_util")
    print(f"  (a) ResNet k=1 All-GPU: DM={a_dm:.1f}% (paper ~24%), worst={a_w:.4f} (paper ~0.098)")
    print(f"      {'REPRODUCES paper' if abs(a_dm-24)<=6 else 'DRIFT — use within-session design'}")
    print(f"  (b) L2LM All-NPU: DM={b_dm:.1f}% (paper ~76%), co-tenant gpu_util={b_gpu:.1f}% (past ~49.5%)")
    print(f"  (c) synth c=24 All-NPU: DM={mean('c_synth_c24_allNPU','dm'):.1f}% (Task1 ~27%)")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
