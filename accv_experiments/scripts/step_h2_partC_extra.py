"""Step H2 Part C supplement — GPU N=1 baselines for the 3 Composition A sids
not measured in Step F Part A (16, 14, 4), plus GPU N=1 bg L1_light for all
8 sids to motivate the AllNPU offload (showing GPU bg degradation).

Appends to: results/step_h2_decomposition.csv
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, fg_worker, load_split_for_sid,
                            load_val, per_stream_sap,
                            preload_background_models, stop_background)
from step_f_partA_matrix import per_stream_map_offline
from step_h2_robustness import start_bg_custom, base_cols_for, append_row

OUT = SCRIPT_DIR.parent / "results" / "step_h2_decomposition.csv"
COMP_A = [2, 22, 13, 16, 3, 21, 14, 4]
MISSING_FROM_PARTA = [16, 14, 4]


def measure_gpu_n1(sid, splits, bg_level, gpu_model):
    results = [defaultdict(list)]
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg_level)
    t = threading.Thread(target=fg_worker,
                         args=(0, "GPU", splits[0], gpu_model, results[0], stop),
                         daemon=True, name=f"hC_gpu_sid{sid}_{bg_level}")
    t0 = time.time(); t.start(); t.join(); wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    sap = per_stream_sap(splits[0], results[0])
    mp = per_stream_map_offline(splits[0], results[0])
    return sap, mp, wall


def main():
    val = load_val()
    print("[partC-extra] preloading bg max=L1")
    preload_background_models(max_level="L1")
    gpu_model = FGModelGPU()
    COLS = base_cols_for(8)

    rows_to_add = []
    # 1) GPU N=1 bg L0 for 3 missing sids
    for sid in MISSING_FROM_PARTA:
        splits = [load_split_for_sid(val, sid)]
        sap, mp, wall = measure_gpu_n1(sid, splits, "L0", gpu_model)
        print(f"  GPU N=1 bg L0 sid={sid}: sap={sap['sap_5095']:.3f}  ({wall:.1f}s)")
        row = {
            "part": "C_supp", "composition": "A_baseline", "n_streams": 1,
            "bg_level": "L0", "placement_name": "N1_GPU_solo",
            "placement_spec": json.dumps(["GPU"]),
            "mean_sap": round(sap["sap_5095"], 4),
            "worst_sap": round(sap["sap_5095"], 4),
            "sap_var": 0.0,
            "mean_map": round(mp["map_5095"], 4),
            "worst_map": round(mp["map_5095"], 4),
            "mean_latency": round(sap["infer_mean_ms"], 2),
            "frame_skip_total": round(sap["frame_skip_pct"], 1),
            "n_active_streams": 1 if sap["frame_skip_pct"] < 50 else 0,
            "wall_sec": round(wall, 1),
            "s0_dev": "GPU", "s0_sid": sid,
            "s0_sap": round(sap["sap_5095"], 4),
            "s0_sap_s": round(sap["sap_small"], 4),
            "s0_sap_m": round(sap["sap_medium"], 4),
            "s0_sap_l": round(sap["sap_large"], 4),
            "s0_map": round(mp["map_5095"], 4),
            "s0_lat": round(sap["infer_mean_ms"], 2),
            "s0_skip": round(sap["frame_skip_pct"], 1),
        }
        rows_to_add.append(row)

    # 2) GPU N=1 bg L1_light for all 8 (motivates AllNPU offload)
    for sid in COMP_A:
        splits = [load_split_for_sid(val, sid)]
        sap, mp, wall = measure_gpu_n1(sid, splits, "L1_light", gpu_model)
        print(f"  GPU N=1 bg L1_light sid={sid}: sap={sap['sap_5095']:.3f}  ({wall:.1f}s)")
        row = {
            "part": "C_supp", "composition": "A_baseline", "n_streams": 1,
            "bg_level": "L1_light", "placement_name": "N1_GPU_solo",
            "placement_spec": json.dumps(["GPU"]),
            "mean_sap": round(sap["sap_5095"], 4),
            "worst_sap": round(sap["sap_5095"], 4),
            "sap_var": 0.0,
            "mean_map": round(mp["map_5095"], 4),
            "worst_map": round(mp["map_5095"], 4),
            "mean_latency": round(sap["infer_mean_ms"], 2),
            "frame_skip_total": round(sap["frame_skip_pct"], 1),
            "n_active_streams": 1 if sap["frame_skip_pct"] < 50 else 0,
            "wall_sec": round(wall, 1),
            "s0_dev": "GPU", "s0_sid": sid,
            "s0_sap": round(sap["sap_5095"], 4),
            "s0_sap_s": round(sap["sap_small"], 4),
            "s0_sap_m": round(sap["sap_medium"], 4),
            "s0_sap_l": round(sap["sap_large"], 4),
            "s0_map": round(mp["map_5095"], 4),
            "s0_lat": round(sap["infer_mean_ms"], 2),
            "s0_skip": round(sap["frame_skip_pct"], 1),
        }
        rows_to_add.append(row)

    # Append all
    for row in rows_to_add:
        append_row(OUT, row, COLS)
    print(f"\n[partC-extra] added {len(rows_to_add)} rows to {OUT}")


if __name__ == "__main__":
    main()
