"""A2 Step-final C helper — one STOCK L2LM All-NPU cell, for py-spy profiling.

Runs exactly the stock reproduction cell (measure_multistream, All-NPU N=4,
PANEL4, threads=4, bg L2_lm) once and prints NPU DM, so an external py-spy
sampler can profile it WITHOUT any inline instrumentation. If DM stays ~68%
under py-spy, the flamegraph time-in-function cross-checks the ablation A
(expect host imread/preprocess dominant). Optional argv: 'none' for control.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_multistream, DETECTORS)

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
BG = sys.argv[1] if len(sys.argv) > 1 else "L2_lm"


def main():
    torch.set_num_threads(4)
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    npu = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu)
    preload_background_models("L2")
    # a couple of reps so py-spy has samples
    for rep in range(3):
        torch.set_num_threads(4)
        res = measure_multistream(PANEL4, splits, ["NPU"] * 4, [], npu[:4], BG)
        skip = float(np.mean([s["frame_skip_pct"] for s in res["per_stream"]]))
        worst = float(min(s["sap_5095"] for s in res["per_stream"]))
        print(f"[pyspy-cell {BG} rep{rep}] npu_skip={skip:.1f}% worst={worst:.4f}", flush=True)
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
