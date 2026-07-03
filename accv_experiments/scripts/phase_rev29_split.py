"""rev29 — Condition-dependent optimal placement (All-GPU / split / All-NPU / Oracle)
on a STABLE contention sweep (ResNet50 GPU-pressure; avoids 33ms skip-unstable cells).

Goal (diagnostic, NOT a scheduler proposal): show that the *optimal* placement is
condition-dependent (low contention -> All-GPU, moderate -> split, saturation -> All-NPU),
while isolated single-stream evaluation prescribes All-GPU everywhere. Oracle = achievable
upper bound over the measured placements (diagnostic), used to test / correct the paper's
line-389 claim that mixed placement never beats the best homogeneous at robust operating points.

Honesty: if split does NOT win at a STABLE operating point, report that as-is (line-389 stands).
Never lean split-superiority on the skip-unstable (33ms-boundary) cell.

Protocol (identical to existing sweeps for comparability):
  Argoverse-HD forward (24 val logs), YOLOv11s, streaming sAP, BOTH devices threads=4,
  NPU = MLA100 global8 (matches rev27 N=4 loading), contention lever = ResNet50 GPU-pressure
  (stable; ORT-CUDA, GPU-bound, minimal host CPU). One VLM-saturation point for contrast.

Split design (efficient: split RATIO, not full 2^N). N=4 fixed. r = #streams on NPU in
{0,1,2,3,4}. Streams moved to NPU in a PRE-DECLARED, mechanism-motivated order: large-object
streams FIRST (staleness is large-biased -> large streams are worst-hit on GPU and cheapest to
quantize on NPU). Order = [3, 21, 22, 2]  (large 3,21 ; then small 22,2).
  r=0 -> GGGG (All-GPU), r=4 -> NNNN (All-NPU), r=1..3 -> mixed.
Oracle(level) = max worst-stream sAP over the 5 measured ratios (upper bound over these 5).

Outputs:
  results/rev29_split_sweep.csv
  results/rev29_oracle_by_contention.csv
  results/rev29_sanity.csv
  results/manifest_rev29.json
core scripts unmodified; existing results / paper untouched.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))
from _step_d_common import load_val, load_split_for_sid, preload_background_models
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines, set_active_npu_engines,
                              dispose_npu_for, measure_single_stream, measure_multistream, DETECTORS)
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop
from step0_compare_devices import FPS

RES = Path("accv_experiments/results"); MAN = RES / "manifest_rev29.json"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
THREADS = 4; N_REPS = 3; PERIOD = 1000.0 / FPS

PANEL4 = [2, 22, 3, 21]
SIZE_GROUP = {2: "small", 22: "small", 3: "large", 21: "large"}
# pre-declared move order: large-object streams first (mechanism: staleness is large-biased)
MOVE_ORDER = [3, 21, 22, 2]
RESNET_COUNTS = [0, 1, 2, 4, 8]          # stable GPU-pressure levels (skip ~0/20/35/50/65%)
# (contention_level label, bg-spec); VLM saturation point appended for contrast
LEVELS = [(f"RES{k}", ("resnet", k)) for k in RESNET_COUNTS] + [("VLM", ("vlm", None))]


def man():
    if MAN.exists():
        try: return json.loads(MAN.read_text())
        except Exception: pass
    return {"done": {}}
def save(m): MAN.write_text(json.dumps(m, indent=2))

def reg_level(spec):
    kind, k = spec
    if kind == "vlm":
        return "L3_vlm"
    name = f"SWEEP{k}"
    h2.BG_VARIANTS[name] = [_bg_resnet50_loop] * k
    return name

def dev_skips(agg):
    g = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"] == "GPU"]
    n = [s["frame_skip_pct"] for s in agg["per_stream"] if s["device"] == "NPU"]
    return (round(float(np.mean(g)), 1) if g else 0.0,
            round(float(np.mean(n)), 1) if n else 0.0)

def placement_for_ratio(r):
    npu_set = set(MOVE_ORDER[:r])
    return ["NPU" if sid in npu_set else "GPU" for sid in PANEL4]

def strat_name(r):
    return {0: "All-GPU", 4: "All-NPU"}.get(r, f"split{r}")

SWEEP_COLS = ["contention_level", "resnet_k", "split_ratio_npu", "strategy",
              "worst_sap", "worst_std", "mean_sap", "gpu_skip", "npu_skip", "reps"]
ORACLE_COLS = ["contention_level", "gpu_skip_allgpu", "allgpu_worst", "allnpu_worst",
               "oracle_worst", "oracle_ratio"]
SAN_COLS = ["tag", "npu_infer_ms", "npu_skip_pct", "gpu_skip_pct", "large_gap", "ok"]

def app(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow(row)

def sanity(val, gpu, npu, tag):
    """Confirm threads=4 operating point: NPU infer ms / skip% at normal values."""
    torch.set_num_threads(THREADS); set_active_npu_engines([npu])
    gl = []; nl = []; lat = []; nsk = []
    for sid in [2, 3, 21, 22, 13]:
        sp = load_split_for_sid(val, sid)
        mg = measure_single_stream(sid, sp, "GPU", gpu, "L0")
        set_active_npu_engines([npu]); mn = measure_single_stream(sid, sp, "NPU", npu, "L0")
        gl.append(mg["sap_l"]); nl.append(mn["sap_l"])
        lat.append(mn["latency_mean"]); nsk.append(mn["frame_skip_pct"])
    infer = float(np.mean(lat)); skip = float(np.mean(nsk))
    ok = bool(infer < 18 and skip < 5)
    row = {"tag": tag, "npu_infer_ms": round(infer, 2), "npu_skip_pct": round(skip, 2),
           "gpu_skip_pct": 0.0, "large_gap": round(float(np.mean(nl) - np.mean(gl)), 4), "ok": ok}
    app(RES / "rev29_sanity.csv", SAN_COLS, row)
    print(f"  SANITY {tag}: npu_infer={infer:.1f}ms npu_skip={skip:.1f}% "
          f"large_gap={row['large_gap']:+.4f} ok={ok}", flush=True)
    return ok

def main():
    torch.set_num_threads(THREADS)
    m = man()
    print(f"=== rev29 split sweep  threads={THREADS} period={PERIOD:.1f}ms  N=4 ratios=0..4 ===", flush=True)
    print(f"    move order (large-first): {MOVE_ORDER}  levels: {[l for l,_ in LEVELS]}", flush=True)
    preload_background_models(max_level="L3")   # resnet (lever) + qwen2vl (VLM point)
    val = load_val()
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], "global8", 4)  # directive: global8
    set_active_npu_engines(npu_models)

    if not sanity(val, gpu_models[0], npu_models[0], "pre"):
        print("  !! SANITY FAILED (NPU infer/skip off threads=4 baseline) — aborting.", flush=True)
        return

    splits = [load_split_for_sid(val, s) for s in PANEL4]

    for lvl_label, spec in LEVELS:
        bg = reg_level(spec)
        # accumulate per ratio across reps
        acc = {r: {"worst": [], "mean": [], "gs": [], "ns": []} for r in range(5)}
        for rep in range(N_REPS):
            key = f"{lvl_label}/rep{rep}"
            if m["done"].get(key):
                print(f"  [skip cached] {key}", flush=True); continue
            torch.set_num_threads(THREADS)
            for r in range(5):
                pl = placement_for_ratio(r)
                ng = sum(1 for d in pl if d == "GPU"); nn = sum(1 for d in pl if d == "NPU")
                agg = measure_multistream(PANEL4, splits, pl, gpu_models[:ng], npu_models[:nn], bg)
                gs, ns = dev_skips(agg)
                acc[r]["worst"].append(agg["worst_sap"]); acc[r]["mean"].append(agg["mean_sap"])
                acc[r]["gs"].append(gs); acc[r]["ns"].append(ns)
                print(f"  {lvl_label} ({bg}) rep{rep} r={r} {''.join('N' if d=='NPU' else 'G' for d in pl)}: "
                      f"worst={agg['worst_sap']:.4f} mean={agg['mean_sap']:.4f} "
                      f"gpu_skip={gs} npu_skip={ns}", flush=True)
            m["done"][key] = {"ts": int(time.time())}; save(m)

        # write aggregate rows for this level (only if we have data, i.e. not fully cached-skipped)
        if all(len(acc[r]["worst"]) > 0 for r in range(5)):
            per_r = {}
            for r in range(5):
                w = acc[r]["worst"]
                row = {"contention_level": lvl_label,
                       "resnet_k": spec[1] if spec[0] == "resnet" else -1,
                       "split_ratio_npu": r, "strategy": strat_name(r),
                       "worst_sap": round(float(np.mean(w)), 4),
                       "worst_std": round(float(np.std(w)), 4),
                       "mean_sap": round(float(np.mean(acc[r]["mean"])), 4),
                       "gpu_skip": round(float(np.mean(acc[r]["gs"])), 1),
                       "npu_skip": round(float(np.mean(acc[r]["ns"])), 1),
                       "reps": len(w)}
                app(RES / "rev29_split_sweep.csv", SWEEP_COLS, row)
                per_r[r] = row
            # oracle row
            worsts = {r: per_r[r]["worst_sap"] for r in range(5)}
            o_r = max(worsts, key=worsts.get)
            app(RES / "rev29_oracle_by_contention.csv", ORACLE_COLS, {
                "contention_level": lvl_label,
                "gpu_skip_allgpu": per_r[0]["gpu_skip"],
                "allgpu_worst": per_r[0]["worst_sap"],
                "allnpu_worst": per_r[4]["worst_sap"],
                "oracle_worst": worsts[o_r], "oracle_ratio": o_r})
            print(f"  >> {lvl_label}: All-GPU={worsts[0]:.4f} All-NPU={worsts[4]:.4f} "
                  f"Oracle={worsts[o_r]:.4f}@r={o_r} (gpu_skip@AllGPU={per_r[0]['gpu_skip']}%)", flush=True)

    sanity(val, gpu_models[0], npu_models[0], "post")
    dispose_npu_for("yolo11s")
    print("=== rev29 done ===", flush=True)

if __name__ == "__main__":
    main()
