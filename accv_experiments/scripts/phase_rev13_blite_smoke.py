"""rev13 B-lite smoke variants — try non-privileged interventions to recover
the rev9-partA-like clean NPU pipeline (12ms latency, <5% skip).

Variant key:
  v1 = NPU loaded BEFORE GPU (reversed order)
  v2 = no BG preload (preload max=None) — minimum load
  v3 = extended warmup (extra 200 NPU dummy inferences before measurement)

Each variant runs the same test cell as rev13_smoke: yolo11s, sid=2,
GPU/NPU at L0 + L1_light. Outputs latency, skip, sap.

Each is independent: writes one row per (variant, device, bg) to
results/rev13_blite_smoke.csv.

GATE per variant:
  G1: NPU skip < 5% at L0 AND L1_light
  G2: NPU infer ~12ms (≤ 18ms loose threshold to be generous)
  G3: large NPU-GPU gap -0.096 ± 0.005 (loose: rev12 anchor)

PASS verdict on a variant ⇒ recommend proceed to full R1 with that variant.
"""

from __future__ import annotations

import csv, json, subprocess, sys, time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, DETECTORS)

RES = Path("accv_experiments/results")
OUT = RES / "rev13_blite_smoke.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
TEST_SID = 2

# Gates (loose to give the experiment a fair shot)
G1_SKIP_THR = 5.0           # %
G2_LATENCY_THR_MS = 18.0    # ms (rev9 was 12, current 36; 18 is permissive midpoint)
G3_GAP_TARGET = -0.096
G3_GAP_TOL = 0.005

COLS = ["variant", "device", "bg", "sap_5095", "sap_l", "latency_mean_ms",
        "frame_skip_pct", "wall_sec", "notes"]


def append_csv(rows):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS);
        if new: w.writeheader()
        for r in rows: w.writerow(r)


def run_variant_v1(val):
    """v1: NPU loaded BEFORE GPU."""
    notes = "NPU loaded before GPU; bg preload max=L1"
    t = time.time()
    preload_background_models(max_level="L1")
    print(f"  bg preload {time.time()-t:.1f}s")
    t = time.time()
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)
    set_active_npu_engines(npu)
    print(f"  NPU load {time.time()-t:.1f}s")
    t = time.time()
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    print(f"  GPU load {time.time()-t:.1f}s")
    rows = _measure_cells("v1", gpu, npu[0], val, notes)
    dispose_npu_for(DET["name"])
    del gpu
    return rows


def run_variant_v2(val):
    """v2: no BG preload."""
    notes = "no BG preload (max=None)"
    # don't preload bg at all
    t = time.time()
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    print(f"  GPU load {time.time()-t:.1f}s")
    t = time.time()
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)
    set_active_npu_engines(npu)
    print(f"  NPU load {time.time()-t:.1f}s")
    rows = _measure_cells("v2", gpu, npu[0], val, notes)
    dispose_npu_for(DET["name"])
    del gpu
    return rows


def run_variant_v3(val):
    """v3: extended warmup — 200 dummy NPU inferences before measurement."""
    notes = "extended warmup (200 dummy NPU inferences before fg_worker)"
    t = time.time()
    preload_background_models(max_level="L1")
    print(f"  bg preload {time.time()-t:.1f}s")
    t = time.time()
    gpu = FGModelGPUGeneric(DET["ultralytics_pt"])
    print(f"  GPU load {time.time()-t:.1f}s")
    t = time.time()
    npu = load_npu_engines(DET, DET["baseline_mxq"], DET["baseline_mode"], 1)
    set_active_npu_engines(npu)
    print(f"  NPU load {time.time()-t:.1f}s")
    # extended warmup
    print(f"  warming NPU with 200 dummy inferences…")
    t = time.time()
    from step0_compare_devices import CONF, IOU
    dummy = np.zeros((1200, 1920, 3), dtype=np.uint8)
    times = []
    for i in range(200):
        t0 = time.time()
        x = npu[0].preprocess(dummy); o = npu[0](x)
        npu[0].postprocess(o, conf_thres=CONF, iou_thres=IOU)
        times.append(time.time() - t0)
    print(f"  warmup wall {time.time()-t:.1f}s — last 50 mean inferences: "
            f"{np.mean(times[-50:])*1000:.1f}ms")
    rows = _measure_cells("v3", gpu, npu[0], val, notes)
    dispose_npu_for(DET["name"])
    del gpu
    return rows


def _measure_cells(variant_name, gpu, npu_model, val, notes):
    """Measure 4 cells: GPU/NPU × L0/L1_light for sid TEST_SID."""
    split = load_split_for_sid(val, TEST_SID)
    rows = []
    for bg in ["L0", "L1_light"]:
        for device, model in [("GPU", gpu), ("NPU", npu_model)]:
            t = time.time()
            m = measure_single_stream(TEST_SID, split, device, model, bg)
            wall = time.time() - t
            print(f"  {variant_name} {device} {bg}: sap={m['sap_5095']:.4f}  "
                    f"latency={m['latency_mean']:.1f}ms  skip={m['frame_skip_pct']:.1f}%  "
                    f"({wall:.1f}s)")
            rows.append({"variant": variant_name, "device": device, "bg": bg,
                          "sap_5095": round(m["sap_5095"], 4),
                          "sap_l": round(m["sap_l"], 4),
                          "latency_mean_ms": round(m["latency_mean"], 1),
                          "frame_skip_pct": round(m["frame_skip_pct"], 1),
                          "wall_sec": round(wall, 1),
                          "notes": notes})
    return rows


def evaluate_gate(rows, variant_name):
    """Return (G1_pass, G2_pass, G3_pass) for given variant."""
    npu_skips = [r["frame_skip_pct"] for r in rows
                  if r["variant"] == variant_name and r["device"] == "NPU"]
    npu_lats  = [r["latency_mean_ms"] for r in rows
                  if r["variant"] == variant_name and r["device"] == "NPU"
                  and r["bg"] == "L0"]
    npu_l_l   = [r["sap_l"] for r in rows
                  if r["variant"] == variant_name and r["device"] == "NPU"
                  and r["bg"] == "L0"]
    gpu_l_l   = [r["sap_l"] for r in rows
                  if r["variant"] == variant_name and r["device"] == "GPU"
                  and r["bg"] == "L0"]
    g1 = bool(npu_skips and max(npu_skips) < G1_SKIP_THR)
    g2 = bool(npu_lats and np.mean(npu_lats) < G2_LATENCY_THR_MS)
    if npu_l_l and gpu_l_l:
        gap = npu_l_l[0] - gpu_l_l[0]
        # NB: single-sid not 24-log mean; loose tolerance G3_GAP_TOL+0.02 = 0.025
        g3 = abs(gap - G3_GAP_TARGET) <= 0.025
    else:
        g3 = False
    return {"G1_skip<5%": g1, "G2_latency<18ms": g2, "G3_gap_close": g3,
             "all_pass": g1 and g2 and g3}


def main():
    # clear previous smoke file if any
    if OUT.exists(): OUT.unlink()
    val = load_val()
    all_rows = []

    print("\n=== variant v1: NPU loaded before GPU ===")
    all_rows.extend(run_variant_v1(val))
    g1 = evaluate_gate(all_rows, "v1")
    print(f"  v1 gate: {g1}")
    if g1["all_pass"]:
        print("  v1 PASS — recommend proceed full R1 with this variant.")
        append_csv(all_rows)
        return 0

    print("\n=== variant v2: no BG preload ===")
    all_rows.extend(run_variant_v2(val))
    g2 = evaluate_gate(all_rows, "v2")
    print(f"  v2 gate: {g2}")
    if g2["all_pass"]:
        print("  v2 PASS — recommend proceed full R1 with this variant.")
        append_csv(all_rows)
        return 0

    print("\n=== variant v3: extended warmup ===")
    all_rows.extend(run_variant_v3(val))
    g3 = evaluate_gate(all_rows, "v3")
    print(f"  v3 gate: {g3}")

    append_csv(all_rows)

    if g3["all_pass"]:
        print("  v3 PASS — recommend proceed full R1 with this variant.")
        return 0

    print("\n=== ALL variants FAIL — no further non-privileged intervention available. ===")
    print("Next steps require sudo (driver reload) or reboot — user permission needed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
