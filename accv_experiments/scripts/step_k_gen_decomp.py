"""Step K — TASK B: Per-size Q(s) and L(s, L1) for YOLOv11s.

Implements §3 of claude_code_experiments.md exactly:
  Q(s)   = AP_GPU_FP32_offline(s) - AP_NPU_INT8_offline(s)
  L(s,C) = AP_GPU_offline(s)      - sAP_GPU(s, C)        with C = L1_light

For each of 24 Argoverse-HD val logs, we run a single foreground stream on
each of three (device, bg) cells:
   1. GPU @ L0           -> offline mAP per size (basis for Q(s) and L baseline)
   2. NPU @ L0           -> offline mAP per size (basis for Q(s))
   3. GPU @ L1_light     -> streaming sAP per size (basis for L(s, L1))
(NPU @ L1_light is recorded but not used in the table since L is GPU-only.)

We then aggregate mean ± std over 24 logs per stratum (small/medium/large)
and emit:

  results/gen_decomp.csv              per-sid rows (full data)
  results/gen_decomp_summary.csv      24-log aggregated Q(s), L(s,L1)
  paper/tables/gen_decomp.tex         table scaffold matching tab:gen-decomp
                                       (s row filled, n/m TBD per Q1(c))

Wall-clock estimate: 24 * 3 * ~15s = ~18 min.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import (FGModelGPU, fg_worker, get_npu_model,
                            load_split_for_sid, load_val, per_stream_sap,
                            preload_background_models, preload_npu_instances,
                            start_background, stop_background)
from step_f_partA_matrix import per_stream_map_offline

RES = SCRIPT_DIR.parent / "results"
OUT_CSV = RES / "gen_decomp.csv"
OUT_SUM = RES / "gen_decomp_summary.csv"
OUT_TEX = SCRIPT_DIR.parent.parent / "paper" / "tables" / "gen_decomp.tex"

DETECTOR = "YOLOv11s"
PARAMS_M = "9.4"   # millions; reported as-is in the table


def measure_one(sid, split, device, model, bg_level):
    """Single-stream sAP + offline mAP for one (sid, device, bg) cell."""
    res = defaultdict(list)
    stop = threading.Event()
    bg_stops, bg_threads = start_background(bg_level)
    t0 = time.time()
    fg_worker(0, device, split, model, res, stop)
    wall = time.time() - t0
    stop_background(bg_stops, bg_threads)
    sap = per_stream_sap(split, res)
    mp = per_stream_map_offline(split, res)
    return {
        "sap_5095": sap["sap_5095"], "sap_50": sap["sap_50"],
        "sap_s": sap["sap_small"], "sap_m": sap["sap_medium"], "sap_l": sap["sap_large"],
        "map_5095": mp["map_5095"], "map_50": mp["map_50"],
        "map_s": mp["map_s"], "map_m": mp["map_m"], "map_l": mp["map_l"],
        "latency_mean": sap["infer_mean_ms"], "frame_skip_pct": sap["frame_skip_pct"],
        "wall_sec": wall,
    }


def main():
    val = load_val()
    n_logs = len(val["sequences"])
    sids = list(range(n_logs))
    print(f"[stepK] {n_logs} logs, detector={DETECTOR}")

    print("[stepK] preloading bg up to L1 (only ResNet50 needed for L1_light)…")
    t0 = time.time()
    preload_background_models(max_level="L1")
    print(f"[stepK] bg preload {time.time()-t0:.1f}s")

    print("[stepK] preloading GPU + NPU YOLO11s (single-mode)…")
    t0 = time.time()
    gpu = FGModelGPU()
    preload_npu_instances(1, infer_mode="single")
    npu = get_npu_model(0)
    print(f"[stepK] model preload {time.time()-t0:.1f}s")

    # Plan: 3 (device, bg) combinations × 24 logs
    plan = [
        ("GPU", "L0",       gpu),
        ("NPU", "L0",       npu),
        ("GPU", "L1_light", gpu),  # L1_light defined in step_h2 but we use start_background("L1")
    ]
    # NOTE: _step_d_common.start_background uses BG_REGISTRY which maps "L1"
    # → ResNet50 once. That matches our L1_light. We pass "L1".
    plan = [("GPU", "L0", gpu), ("NPU", "L0", npu), ("GPU", "L1", gpu)]
    # The CSV records the canonical L1_light tag for clarity later.
    BG_LABEL = {"L0": "L0", "L1": "L1_light"}

    if OUT_CSV.exists(): OUT_CSV.unlink()
    cols = ["detector", "sid", "log_id", "device", "bg_level",
            "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
            "map_5095", "map_50", "map_s", "map_m", "map_l",
            "latency_mean", "frame_skip_pct", "wall_sec"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    t_all = time.time()
    for sid in sids:
        split = load_split_for_sid(val, sid)
        for device, bg, model in plan:
            print(f"[K sid={sid:>2d}  dev={device}  bg={BG_LABEL[bg]:<9s}]", end=" ", flush=True)
            m = measure_one(sid, split, device, model, bg)
            row = {
                "detector": DETECTOR, "sid": sid, "log_id": val["sequences"][sid],
                "device": device, "bg_level": BG_LABEL[bg],
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            }
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            print(f"sAP={m['sap_5095']:.3f}  mAP={m['map_5095']:.3f}  "
                  f"map_s={m['map_s']:.3f}/m={m['map_m']:.3f}/l={m['map_l']:.3f}  "
                  f"({m['wall_sec']:.1f}s)")
    print(f"\n[stepK] all measurements done. wall_total={time.time()-t_all:.1f}s")

    # ----- Aggregate Q(s), L(s, L1) -----
    df = pd.read_csv(OUT_CSV)
    gpu_L0 = df[(df.device == "GPU") & (df.bg_level == "L0")]
    npu_L0 = df[(df.device == "NPU") & (df.bg_level == "L0")]
    gpu_L1 = df[(df.device == "GPU") & (df.bg_level == "L1_light")]
    print(f"\n=== aggregation: GPU L0={len(gpu_L0)}, NPU L0={len(npu_L0)}, GPU L1={len(gpu_L1)} sids ===")

    # Mean offline mAP per size for GPU L0 and NPU L0
    def mean(df_, col):
        v = df_[col].astype(float)
        return float(v.mean()), float(v.std(ddof=0))

    Q_rows = []
    L_rows = []
    for size_col, size_name in [("map_s", "small"), ("map_m", "medium"), ("map_l", "large")]:
        g_l0, g_l0_sd = mean(gpu_L0, size_col)
        n_l0, n_l0_sd = mean(npu_L0, size_col)
        Q = g_l0 - n_l0
        Q_rows.append({"detector": DETECTOR, "params_M": PARAMS_M,
                       "size": size_name,
                       "gpu_L0_mAP": round(g_l0, 4),
                       "npu_L0_mAP": round(n_l0, 4),
                       "Q": round(Q, 4)})
    for size_col, size_sap_col, size_name in [
        ("map_s", "sap_s", "small"),
        ("map_m", "sap_m", "medium"),
        ("map_l", "sap_l", "large"),
    ]:
        g_l0, _ = mean(gpu_L0, size_col)
        g_l1_sap, _ = mean(gpu_L1, size_sap_col)
        L = g_l0 - g_l1_sap
        L_rows.append({"detector": DETECTOR, "params_M": PARAMS_M,
                       "size": size_name,
                       "gpu_L0_mAP": round(g_l0, 4),
                       "gpu_L1_sAP": round(g_l1_sap, 4),
                       "L": round(L, 4)})

    summary = pd.DataFrame([{**q, **{f"L_{l['size']}": l['L'] for l in L_rows if l['size']==q['size']}}
                             for q in Q_rows])
    # cleaner: separate frame
    Qdf = pd.DataFrame(Q_rows)
    Ldf = pd.DataFrame(L_rows)
    merged = Qdf.merge(Ldf[["size", "gpu_L1_sAP", "L"]], on="size")
    merged.to_csv(OUT_SUM, index=False)
    print("\n=== Q(s) and L(s, L1) for YOLOv11s ===")
    print(merged.to_string(index=False))

    # ----- LaTeX scaffold (tab:gen-decomp) -----
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    Qs = {r["size"]: r["Q"] for r in Q_rows}
    Ls = {r["size"]: r["L"] for r in L_rows}
    with open(OUT_TEX, "w") as f:
        f.write("% Auto-generated by accv_experiments/scripts/step_k_gen_decomp.py\n")
        f.write("% Per-size Q(s) (offline FP32-INT8 mAP gap) and L(s, L1) (offline\n")
        f.write("% mAP - streaming sAP under L1_light), aggregated over 24 logs.\n")
        f.write("% YOLOv11n/m rows are TBD: Mobilint NPU INT8 mxq not available; see\n")
        f.write("% RESULTS_STATUS.md. Sign convention follows the paper: Q,L are losses\n")
        f.write("% reported as negative values.\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        f.write("& & \\multicolumn{3}{c|}{$Q(s)$ (quantization)} & "
                "\\multicolumn{3}{c}{$L(s,\\text{L1})$ (staleness)} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        f.write("YOLOv11n & 2.6\\,M  & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n")
        f.write(f"YOLOv11s & 9.4\\,M  & "
                f"${-Qs['small']:+.3f}$ & ${-Qs['medium']:+.3f}$ & ${-Qs['large']:+.3f}$ & "
                f"${-Ls['small']:+.3f}$ & ${-Ls['medium']:+.3f}$ & ${-Ls['large']:+.3f}$ \\\\\n")
        f.write("YOLOv11m & 20.1\\,M & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nsaved {OUT_TEX}")
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_SUM}")


if __name__ == "__main__":
    main()
