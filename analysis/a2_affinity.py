"""A2 Step-final B — CPU-affinity discrimination (core-sharing vs memory-bandwidth).

Stock (non-ablated) foreground: All-NPU N=4, PANEL4, threads=4, no in-process bg.
The L2LM co-tenant runs as a SEPARATE process (a2_cotenant_proc.py). Two variants:
  shared : co-tenant + foreground both unpinned (all 24 cores)  -> baseline
  pinned : co-tenant -> cores {12..23}, foreground -> cores {0..11}  (disjoint)
If pinned DM << shared DM  -> core-sharing SCHEDULING contention is the cause.
If pinned DM ~= shared DM  -> memory-bandwidth / cache contention (perf-free proxy).
3 reps each. Output: analysis/a2_affinity.csv.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid
from phase_rev6_sweep import (load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PY = ROOT / ".venv/bin/python"
COT = ROOT / "analysis/a2_cotenant_proc.py"
OUT = ROOT / "analysis/a2_affinity.csv"
FG_CORES = list(range(0, 12))
CT_CORES = list(range(12, 24))
REPS = 3


def start_cotenant(cores):
    arg = ",".join(str(c) for c in cores) if cores else ""
    p = subprocess.Popen([str(PY), str(COT), arg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # wait for ready line
    for _ in range(120):
        line = p.stdout.readline().decode(errors="ignore")
        if "L2 ready" in line:
            break
    time.sleep(1.0)
    return p


def stop_cotenant(p):
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()


def run_variant(name, fg_cores, ct_cores, splits, npu):
    if fg_cores:
        os.sched_setaffinity(0, set(fg_cores))
    else:
        os.sched_setaffinity(0, set(range(24)))
    ct = start_cotenant(ct_cores)
    rows = []
    try:
        for rep in range(REPS):
            torch.set_num_threads(4)
            res = measure_multistream(PANEL4, splits, ["NPU"] * 4, [], npu[:4], "L0")
            skip = float(np.mean([s["frame_skip_pct"] for s in res["per_stream"]]))
            worst = float(min(s["sap_5095"] for s in res["per_stream"]))
            rows.append({"variant": name, "rep": rep, "npu_skip": round(skip, 2),
                         "worst_sap": round(worst, 4)})
            print(f"  [{name}/rep{rep}] npu_skip={skip:.1f}% worst={worst:.4f}", flush=True)
    finally:
        stop_cotenant(ct)
    os.sched_setaffinity(0, set(range(24)))
    return rows


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 NPU engines ...", flush=True)
    npu = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu)

    rows = []
    print("\n=== variant: shared (co-tenant + fg unpinned) ===", flush=True)
    rows += run_variant("shared", None, None, splits, npu)
    print("\n=== variant: pinned (co-tenant cores 12-23, fg cores 0-11) ===", flush=True)
    rows += run_variant("pinned", FG_CORES, CT_CORES, splits, npu)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT}")

    def m(v):
        return float(np.mean([r["npu_skip"] for r in rows if r["variant"] == v]))
    sh, pn = m("shared"), m("pinned")
    print(f"\n=== AFFINITY VERDICT ===")
    print(f"  shared DM = {sh:.1f}%   pinned DM = {pn:.1f}%")
    if sh - pn >= 20:
        print("  -> pinning apart LARGELY removes the stall => CORE-SHARING scheduling contention.")
    elif sh - pn <= 8:
        print("  -> pinning apart does NOT help => MEMORY-BANDWIDTH / cache contention (cores disjoint).")
    else:
        print(f"  -> partial ({sh-pn:.0f}pp drop) => mixed core-sharing + memory contention.")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
