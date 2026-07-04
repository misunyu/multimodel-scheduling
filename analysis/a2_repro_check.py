"""A2 Step 2 + Step 3 — L2LM reproduction discriminator + N=1 breakdown.

Step 2 (discriminator): run the STOCK, non-instrumented harness
(phase_rev6_sweep.measure_multistream) exactly as rev20 did — All-NPU N=4,
PANEL4, threads=4, bg L2_lm, 3 reps — and read npu_skip. This isolates whether
the probe's 23% (vs rev20's ~76%) is a probe artifact or environment drift.
Also logs host CPU user/sys AND GPU util/power during each cell (tests
hypothesis iii: is the L2LM co-tenant actually GPU-bound or CPU-saturating?).
A no-co-tenant (L0) control cell anchors the baseline.

Step 3: N=1, threads=4, no-co-tenant single-stream STAGE decomposition
(read_pre / on-chip / post) — the confirmed value for the manuscript's
"single-stream latency" wording.

Runs GPU (L2 co-tenant) + NPU. Pre-approved batch.
Output: analysis/a2_repro_check.csv, a2_step3_singlestream.csv, console verdict.
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
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)
from a2_stage_probe import npu_infer_timed, timed_npu_worker  # instrumented N=1 path

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
OUT = ROOT / "analysis/a2_repro_check.csv"
OUT3 = ROOT / "analysis/a2_step3_singlestream.csv"
REPS = 3


class UtilSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self._ev = threading.Event()
        self.cpu, self.gpu, self.pw = [], [], []
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        psutil.cpu_times_percent(0.0)
        while not self._ev.is_set():
            ct = psutil.cpu_times_percent(self.interval)
            u = pynvml.nvmlDeviceGetUtilizationRates(self.h)
            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            except Exception:
                pw = -1
            self.cpu.append((ct.user, ct.system))
            self.gpu.append(u.gpu)
            self.pw.append(pw)

    def stop(self):
        self._ev.set()
        self.join(timeout=2)
        c = np.array(self.cpu) if self.cpu else np.zeros((1, 2))
        return {"cpu_user": round(float(c[:, 0].mean()), 1),
                "cpu_sys": round(float(c[:, 1].mean()), 1),
                "gpu_util": round(float(np.mean(self.gpu)), 1) if self.gpu else -1,
                "power_w": round(float(np.mean(self.pw)), 1) if self.pw else -1}


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 GPU + 4 NPU engines + L2 co-tenants ...", flush=True)
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    preload_background_models(max_level="L2")

    rows = []
    print("\n=== STEP 2: stock measure_multistream, All-NPU N=4, threads=4 ===", flush=True)
    for bg in ["L0", "L2_lm", "L2_lm", "L2_lm"]:
        torch.set_num_threads(4)
        samp = UtilSampler()
        samp.start()
        agg = measure_multistream(PANEL4, splits, ["NPU"] * 4, [], npu_models[:4], bg)
        util = samp.stop()
        # extract per-stream npu skip
        ps = agg["per_stream"]
        nskip = float(np.mean([s["frame_skip_pct"] for s in ps]))
        worst = float(min(s["sap_5095"] for s in ps))
        row = {"step": 2, "bg": bg, "N": 4, "placement": "AllNPU",
               "npu_skip": round(nskip, 2), "worst_sap": round(worst, 4),
               **util}
        rows.append(row)
        print(f"  bg={bg:6s} npu_skip={nskip:5.1f}% worst={worst:.4f} "
              f"cpu_user={util['cpu_user']}% cpu_sys={util['cpu_sys']}% "
              f"gpu_util={util['gpu_util']}% pw={util['power_w']}W", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}", flush=True)

    # ---- verdict ----
    l2 = [r["npu_skip"] for r in rows if r["bg"] == "L2_lm"]
    l2m = float(np.mean(l2))
    print(f"\n=== STEP 2 VERDICT ===")
    print(f"  stock harness L2LM All-NPU N=4 npu_skip = {l2} -> mean {l2m:.1f}%")
    print(f"  rev20 (same config) measured 78.8/73.8/74.9 -> ~76%")
    if l2m >= 65:
        print(f"  -> REPRODUCES ~76%. The probe's 23% is an INSTRUMENTATION artifact "
              f"(diff timed_npu_worker vs fg_worker).")
    elif l2m <= 55:
        print(f"  -> DOES NOT reproduce (drift). Escalate: tab:main L2LM row (83/76) "
              f"reproducibility issue. STOP for instruction.")
    else:
        print(f"  -> partial ({l2m:.0f}%). Report.")

    # ---- STEP 3: N=1 threads=4 single-stream stage breakdown ----
    print("\n=== STEP 3: N=1 threads=4 no-co-tenant single-stream breakdown ===", flush=True)
    torch.set_num_threads(4)
    s0 = load_split_for_sid(val, PANEL4[0])
    rows3 = []
    stop = threading.Event()
    timed_npu_worker(0, s0, npu_models[0], stop, rows3)
    arr = np.array([(r[2], r[3], r[4], r[5], r[6]) for r in rows3], float)  # read,infer,post,other,eff
    labels = ["read_pre", "infer", "post", "other", "eff"]
    means = arr.mean(0)
    dm = 100.0 * np.mean(arr[:, 4] > (1000.0 / 30))
    with open(OUT3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "mean_ms", "p50_ms", "p95_ms"])
        for i, lb in enumerate(labels):
            w.writerow([lb, round(means[i], 3), round(np.percentile(arr[:, i], 50), 3),
                        round(np.percentile(arr[:, i], 95), 3)])
    print(f"  N=1 threads=4: " + "  ".join(f"{lb}={means[i]:.2f}ms" for i, lb in enumerate(labels))
          + f"  DM={dm:.1f}%")
    print(f"  wrote {OUT3}")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
