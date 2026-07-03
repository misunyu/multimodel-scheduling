"""rev18 STAGE 1 — optimized-postprocess lever gate.

Discovery (rev18 probe): the ~27ms NPU host postprocess is torch CPU thread
thrashing on tiny YOLO11 head tensors. torch.set_num_threads(default=24) →
25.8ms; threads=4 → 1.33ms (19x faster), with IDENTICAL computation (same
math, only thread count) → accuracy preserved by construction.

This is the optimized-postprocess lever. It is applied as a process-level
setting (torch.set_num_threads) BEFORE calling the unmodified
measure_single_stream — no core eval script is edited.

Gate (single-stream L0, yolo11s, 24 logs, >=3 reps):
  G1: NPU L0 effective latency < 33.3ms AND skip < 5%
  G2: large NPU-GPU gap = -0.096 +- 0.005  (accuracy preserved)
  G3: >=3 reps reproducible (std small)

Baseline (default threads) measured too, to show the before/after.

Output: results/rev18_postproc_levers.csv + console gate verdict.
"""

from __future__ import annotations

import csv, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))

from _step_d_common import load_val, load_split_for_sid
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, DETECTORS)
from step0_compare_devices import FPS

RES = Path("accv_experiments/results")
OUT = RES / "rev18_postproc_levers.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD_MS = 1000.0 / FPS
N_REPS = 3
GATE_LARGE_TARGET = -0.096
GATE_LARGE_TOL = 0.005

THREAD_SETTINGS = [24, 4]   # default(thrash) vs optimized


def measure_all_logs(val, npu, gpu, n_sids):
    """Single-stream L0 for all sids, both devices. Returns per-size means + skip + lat."""
    g = {"sap_s": [], "sap_m": [], "sap_l": [], "lat": [], "skip": []}
    n = {"sap_s": [], "sap_m": [], "sap_l": [], "lat": [], "skip": []}
    for sid in range(n_sids):
        split = load_split_for_sid(val, sid)
        mg = measure_single_stream(sid, split, "GPU", gpu, "L0")
        set_active_npu_engines([npu])
        mn = measure_single_stream(sid, split, "NPU", npu, "L0")
        for d, m in [(g, mg), (n, mn)]:
            d["sap_s"].append(m["sap_s"]); d["sap_m"].append(m["sap_m"])
            d["sap_l"].append(m["sap_l"]); d["lat"].append(m["latency_mean"])
            d["skip"].append(m["frame_skip_pct"])
    return g, n


def main():
    val = load_val()
    n_sids = len(val["sequences"])
    print(f"period={PERIOD_MS:.2f}ms, {n_sids} logs, {N_REPS} reps per thread setting")

    rows = []
    summary = {}
    for nt in THREAD_SETTINGS:
        print(f"\n=== torch threads = {nt} ===")
        torch.set_num_threads(nt)
        gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
        npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)[0]
        set_active_npu_engines([npu])
        # warmup
        s0 = load_split_for_sid(val, 0)
        measure_single_stream(0, s0, "NPU", npu, "L0")

        rep_npu_lat = []; rep_npu_skip = []; rep_gap = []
        for rep in range(N_REPS):
            torch.set_num_threads(nt)  # ensure
            g, n = measure_all_logs(val, npu, gpu, n_sids)
            npu_lat = float(np.mean(n["lat"])); npu_skip = float(np.mean(n["skip"]))
            gpu_l = float(np.mean(g["sap_l"])); npu_l = float(np.mean(n["sap_l"]))
            gap = npu_l - gpu_l
            rep_npu_lat.append(npu_lat); rep_npu_skip.append(npu_skip); rep_gap.append(gap)
            print(f"  rep{rep}: NPU lat={npu_lat:.2f}ms skip={npu_skip:.1f}%  "
                    f"large gap={gap:+.4f}  (GPU_l={gpu_l:.4f} NPU_l={npu_l:.4f})")
            rows.append({"threads": nt, "rep": rep,
                          "npu_lat_ms": round(npu_lat, 3),
                          "npu_skip_pct": round(npu_skip, 2),
                          "gpu_sap_l": round(gpu_l, 4), "npu_sap_l": round(npu_l, 4),
                          "large_gap": round(gap, 4),
                          "npu_sap_s": round(float(np.mean(n["sap_s"])), 4),
                          "npu_sap_m": round(float(np.mean(n["sap_m"])), 4),
                          "gpu_sap_s": round(float(np.mean(g["sap_s"])), 4),
                          "gpu_sap_m": round(float(np.mean(g["sap_m"])), 4)})
        summary[nt] = {
            "npu_lat_mean": float(np.mean(rep_npu_lat)), "npu_lat_std": float(np.std(rep_npu_lat)),
            "npu_skip_mean": float(np.mean(rep_npu_skip)), "npu_skip_std": float(np.std(rep_npu_skip)),
            "gap_mean": float(np.mean(rep_gap)), "gap_std": float(np.std(rep_gap)),
        }
        dispose_npu_for("yolo11s"); del gpu

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nsaved {OUT}")

    # Gate on optimized setting (threads=4)
    opt = summary[4]
    g1 = opt["npu_lat_mean"] < PERIOD_MS and opt["npu_skip_mean"] < 5.0
    g2 = abs(opt["gap_mean"] - GATE_LARGE_TARGET) <= GATE_LARGE_TOL
    g3 = opt["npu_lat_std"] < 2.0 and opt["gap_std"] < 0.005  # reproducible
    print("\n=== STAGE 1 GATE (threads=4) ===")
    print(f"  baseline threads=24: NPU lat={summary[24]['npu_lat_mean']:.2f}ms "
            f"skip={summary[24]['npu_skip_mean']:.1f}% gap={summary[24]['gap_mean']:+.4f}")
    print(f"  optimized threads=4: NPU lat={opt['npu_lat_mean']:.2f}±{opt['npu_lat_std']:.2f}ms "
            f"skip={opt['npu_skip_mean']:.1f}±{opt['npu_skip_std']:.1f}% "
            f"gap={opt['gap_mean']:+.4f}±{opt['gap_std']:.4f}")
    print(f"  G1 (lat<{PERIOD_MS:.1f}ms & skip<5%): {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 (gap -0.096±0.005, accuracy preserved): {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 (>=3 reps reproducible): {'PASS' if g3 else 'FAIL'}")
    print(f"  OVERALL: {'PASS -> STAGE 2' if (g1 and g2 and g3) else 'FAIL -> stop'}")
    return 0 if (g1 and g2 and g3) else 1


if __name__ == "__main__":
    sys.exit(main())
