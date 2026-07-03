"""rev13 smoke — verify whether current NPU pipeline can achieve rev9-partA-like
low frame_skip_pct (< 5%) for yolo11s at L0 and L1_light.

Goal: determine if R1 gate is achievable BEFORE committing to a full sweep.

Test: yolo11s, sid=2 (small-rich, well-known anchor), NPU + GPU at L0 and L1_light.
Expected wall time: ~60-90 seconds.

Output: results/rev13_smoke.csv + verdict (PASS/FAIL skip<5% gate).
"""

from __future__ import annotations

import csv, json, subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, DETECTORS)

RES = Path("accv_experiments/results")
OUT = RES / "rev13_smoke.csv"

DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
TEST_SID = 2
GATE_SKIP_THR = 5.0  # %

def probe_env():
    info = {}
    try: info["driver"] = subprocess.check_output(
        ["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
        text=True).strip()
    except Exception: info["driver"] = "?"
    try:
        gu = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu",
              "--format=csv,noheader,nounits"], text=True).strip()
        u,t = gu.split(","); info["gpu_util"]=int(u.strip()); info["gpu_temp"]=int(t.strip())
    except Exception:
        info["gpu_util"]=-1; info["gpu_temp"]=-1
    try: info["git_commit"] = subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip()[:12]
    except Exception: info["git_commit"]="?"
    return info


def main():
    env = probe_env()
    print(f"env: driver={env['driver']} util={env['gpu_util']}% temp={env['gpu_temp']}C  git={env['git_commit']}")

    t = time.time()
    preload_background_models(max_level="L1")
    print(f"[smoke] bg preload {time.time()-t:.1f}s")

    val = load_val()
    print(f"[smoke] loaded {len(val['sequences'])} logs, testing sid={TEST_SID}")

    t = time.time()
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    print(f"[smoke] GPU load {time.time()-t:.1f}s")
    t = time.time()
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)
    set_active_npu_engines(npu)
    print(f"[smoke] NPU load {time.time()-t:.1f}s  (mode={DET['baseline_mode']})")

    split = load_split_for_sid(val, TEST_SID)
    rows = []
    for bg in ["L0", "L1_light"]:
        for device, model in [("GPU", gpu), ("NPU", npu[0])]:
            t = time.time()
            m = measure_single_stream(TEST_SID, split, device, model, bg)
            wall = time.time() - t
            print(f"  {device} {bg}: sap={m['sap_5095']:.4f}  "
                    f"latency={m['latency_mean']:.1f}ms  "
                    f"skip={m['frame_skip_pct']:.1f}%  ({wall:.1f}s)")
            rows.append({"device": device, "bg": bg,
                          "sap_5095": round(m["sap_5095"], 4),
                          "latency_mean_ms": round(m["latency_mean"], 1),
                          "frame_skip_pct": round(m["frame_skip_pct"], 1),
                          "wall_sec": round(wall, 1)})

    dispose_npu_for(DET["name"])

    # write CSV
    cols = ["device","bg","sap_5095","latency_mean_ms","frame_skip_pct","wall_sec"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"saved {OUT}")

    # Gate verdict
    npu_skips = [r["frame_skip_pct"] for r in rows if r["device"] == "NPU"]
    max_npu_skip = max(npu_skips)
    gate_pass = max_npu_skip < GATE_SKIP_THR
    print()
    print(f"=== R1 gate check ===")
    print(f"max NPU skip across L0/L1_light: {max_npu_skip:.1f}%  (threshold < {GATE_SKIP_THR:.0f}%)")
    print(f"verdict: {'PASS — proceed to full R1' if gate_pass else 'FAIL — current state cannot achieve rev9-partA-like low skip'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
