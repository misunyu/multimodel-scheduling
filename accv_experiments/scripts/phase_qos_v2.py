"""EXP-QOS v2 — Stream-priority QoS vs placement reversal (W2 defence).

All-GPU only (NPU reference line reused from rev22/rev20). Reuses the exact
rev22 measurement path: fg_worker + per_stream_sap + start_bg_custom +
FGModelGPUGeneric. Core scripts NOT modified.

PRIO = move the 4 FG streams onto ONE shared high-priority CUDA stream
(priority=-1) by wrapping FGModelGPUGeneric.predict in a torch.cuda.stream()
context. Baseline shares the torch default stream (stream 0) across the 4 FG
threads, so a single shared high-priority stream mirrors that concurrency
structure and only changes priority (+ removes the legacy-default-stream
implicit sync, which is intrinsic to using stream priority in PyTorch).

Co-tenant lever: resnet_k ResNet50 GPU-loops (rev22 SWEEP{k}); vlm_sat = L3_vlm.

Stages:
  pre   : topology probe, detection-identity (PRIO on/off) check, PRIO
          micro-benchmark (k2 infer p99 on/off), BASE k2 sanity (3 reps).
  sweep : PRIO full sweep, 6 points x 3 reps, All-GPU.

Outputs: results/exp_qos_results.csv  (+ prints; summary written separately)
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "minimal_pipeline"))

import _step_d_common as cm
from _step_d_common import (load_val, load_split_for_sid, fg_worker, per_stream_sap,
                            preload_background_models, stop_background, WARMUP_FRAMES,
                            npu_infer)
from phase_rev6_sweep import FGModelGPUGeneric, DETECTORS
import step_h2_robustness as h2
from step_h2_robustness import _bg_resnet50_loop, start_bg_custom
from step0_compare_devices import DATA, FPS

import threading
RES = Path("accv_experiments/results")
OUT_CSV = RES / "exp_qos_results.csv"
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
THREADS = 4; N_REPS = 3
PANEL4 = [2, 22, 3, 21]
# operating points: resnet_k ints, plus 'vlm_sat' -> L3_vlm
POINTS = [0, 1, 2, 3, 8, "vlm_sat"]
# baseline measured GPU skip% reference per point (rev22 means; vlm_sat from rev20)
BASE_SKIP_REF = {0: 0.0, 1: 40.6, 2: 55.7, 3: 66.5, 8: 80.3, "vlm_sat": 100.0}
PERIOD = 1000.0 / FPS

CSV_COLS = ["exp_id", "mechanism", "cotenant", "resnet_k_or_vlm", "baseline_skip_ref_pct",
            "repeat", "stream_id", "log_name",
            "sap", "sap_small", "sap_medium", "sap_large",
            "fg_gpu_skip_pct", "infer_ms_p50", "infer_ms_p99",
            "e2e_delivery_ms_p50", "e2e_delivery_ms_p99",
            "priority_value_used", "stream_topology",
            "run_timestamp", "notes"]


def app(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def bg_level_for(point):
    """Register/return the bg-level string for an operating point."""
    if point == "vlm_sat":
        return "L3_vlm"  # resnet50 + qwen2-vl (already in BG_VARIANTS)
    name = f"SWEEP{point}"
    h2.BG_VARIANTS[name] = [_bg_resnet50_loop] * point
    return name


# ----------------------- PRIO wrapping -----------------------
_PRIO_STREAM = None  # one shared high-priority stream for all FG streams

def make_prio_stream(priority=-1):
    global _PRIO_STREAM
    _PRIO_STREAM = torch.cuda.Stream(priority=priority)
    return _PRIO_STREAM

def wrap_prio(model, stream):
    """Wrap FGModelGPUGeneric.predict so its kernels run on `stream` (high
    priority). Synchronize the stream before returning so the GPU tensors are
    ready for the subsequent .cpu() in fg_worker (default stream) -> no race;
    validated by the detection-identity check."""
    orig = model.predict
    def predict(img_path):
        with torch.cuda.stream(stream):
            r = orig(img_path)
        stream.synchronize()
        return r
    model.predict = predict
    model._orig_predict = orig

def unwrap_prio(model):
    if hasattr(model, "_orig_predict"):
        model.predict = model._orig_predict


# ----------------------- measurement -----------------------
def measure_allgpu(sids, splits, gpu_models, bg):
    """One All-GPU multistream run. Returns per-stream dicts incl. infer/e2e
    p50/p99 (extra vs rev22, computed from raw fg_worker traces)."""
    n = len(sids)
    results = [defaultdict(list) for _ in range(n)]
    stop = threading.Event()
    bg_stops, bg_threads = start_bg_custom(bg)
    threads = []
    for i in range(n):
        t = threading.Thread(target=fg_worker,
                             args=(i, "GPU", splits[i], gpu_models[i], results[i], stop),
                             daemon=True)
        threads.append(t)
    for t in threads: t.start()
    for t in threads: t.join()
    stop_background(bg_stops, bg_threads)
    per = []
    for i in range(n):
        sap = per_stream_sap(splits[i], results[i])
        infer = results[i]["infer_ms"][WARMUP_FRAMES:]
        eff = results[i]["eff_ms"][WARMUP_FRAMES:]
        per.append({
            "stream_id": i, "sid": sids[i], "log_name": splits[i]["log_name"],
            "sap": sap["sap_5095"], "sap_small": sap["sap_small"],
            "sap_medium": sap["sap_medium"], "sap_large": sap["sap_large"],
            "fg_gpu_skip_pct": sap["frame_skip_pct"],
            "infer_p50": float(np.percentile(infer, 50)) if infer else 0.0,
            "infer_p99": float(np.percentile(infer, 99)) if infer else 0.0,
            "e2e_p50": float(np.percentile(eff, 50)) if eff else 0.0,
            "e2e_p99": float(np.percentile(eff, 99)) if eff else 0.0,
        })
    return per


def emit(mech, point, rep, per, notes=""):
    cotenant = "qwen2vl+resnet50" if point == "vlm_sat" else f"resnet50x{point}"
    rk = "vlm_sat" if point == "vlm_sat" else point
    prio = -1 if mech == "prio" else ""
    topo = "shared1" if mech == "prio" else ""
    ts = int(time.time())
    for s in per:
        app(OUT_CSV, CSV_COLS, {
            "exp_id": "EXP-QOS-v2", "mechanism": mech, "cotenant": cotenant,
            "resnet_k_or_vlm": rk, "baseline_skip_ref_pct": BASE_SKIP_REF[point],
            "repeat": rep, "stream_id": s["stream_id"], "log_name": s["log_name"],
            "sap": round(s["sap"], 4), "sap_small": round(s["sap_small"], 4),
            "sap_medium": round(s["sap_medium"], 4), "sap_large": round(s["sap_large"], 4),
            "fg_gpu_skip_pct": round(s["fg_gpu_skip_pct"], 1),
            "infer_ms_p50": round(s["infer_p50"], 2), "infer_ms_p99": round(s["infer_p99"], 2),
            "e2e_delivery_ms_p50": round(s["e2e_p50"], 2),
            "e2e_delivery_ms_p99": round(s["e2e_p99"], 2),
            "priority_value_used": prio, "stream_topology": topo,
            "run_timestamp": ts, "notes": notes,
        })
    worst = min(s["sap"] for s in per)
    print(f"  [{mech}] {('k'+str(point)) if point!='vlm_sat' else 'vlm_sat'} rep{rep}: "
          f"worst={worst:.4f} mean={np.mean([s['sap'] for s in per]):.4f} "
          f"skip={np.mean([s['fg_gpu_skip_pct'] for s in per]):.1f}% "
          f"inferp99={np.mean([s['infer_p99'] for s in per]):.1f}ms")
    return worst


# ----------------------- stages -----------------------
def topology_probe(gpu_models, splits):
    print("\n=== TOPOLOGY PROBE (3.1-1,2) ===")
    main_s = torch.cuda.current_stream()
    print(f"  main thread current_stream: {main_s} (default={torch.cuda.default_stream()})")
    seen = {}
    def w(i):
        seen[i] = (str(torch.cuda.current_stream()), torch.cuda.current_stream() == torch.cuda.default_stream())
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    shared_default = all(v[1] for v in seen.values())
    for i in sorted(seen): print(f"  worker thread {i}: stream={seen[i][0]} is_default={seen[i][1]}")
    print(f"  => baseline FG threads share default stream: {shared_default}")
    return {"main_stream": str(main_s), "default_stream": str(torch.cuda.default_stream()),
            "workers": {str(k): seen[k][0] for k in seen},
            "all_share_default": bool(shared_default),
            "prio_topology_choice": "shared1 (single shared priority=-1 stream, mirrors baseline)"}


def identity_check(gpu_models, splits, point=0):
    """3.3: PRIO must not change detections. Compare 20 frames off vs on."""
    print("\n=== DETECTION-IDENTITY CHECK (3.3) ===")
    sp = splits[0]
    paths = [str(DATA / sp["seq_dir"] / im["name"]) for im in sp["imgs"]]
    frames = paths[40:60]
    gm = gpu_models[0]
    unwrap_prio(gm)
    off = [gm.predict(p) for p in frames]
    s = make_prio_stream(-1); wrap_prio(gm, s)
    on = [gm.predict(p) for p in frames]
    unwrap_prio(gm)
    max_box = 0.0; max_score = 0.0; max_ncls = 0; ok = True
    for ro, rn in zip(off, on):
        bo = ro.boxes.xyxy.cpu().numpy(); bn = rn.boxes.xyxy.cpu().numpy()
        so = ro.boxes.conf.cpu().numpy(); sn = rn.boxes.conf.cpu().numpy()
        co = ro.boxes.cls.cpu().numpy().astype(int); cn = rn.boxes.cls.cpu().numpy().astype(int)
        if len(bo) != len(bn) or not np.array_equal(co, cn):
            ok = False; max_ncls = max(max_ncls, abs(len(bo) - len(bn))); continue
        if len(bo):
            max_box = max(max_box, float(np.abs(bo - bn).max()))
            max_score = max(max_score, float(np.abs(so - sn).max()))
    print(f"  20 frames: identical={ok} max_box_diff={max_box:.2e} max_score_diff={max_score:.2e}")
    return {"identical": bool(ok), "max_box_diff": max_box, "max_score_diff": max_score,
            "n_class_mismatch": max_ncls}


def microbench(gpu_models, splits, point=2):
    """3.1-4: FG infer p99 with PRIO off vs on at k2 (one full N=4 pass each)."""
    print("\n=== PRIO MICRO-BENCH (3.1-4): k2, infer p99 off vs on ===")
    bg = bg_level_for(point)
    sids = PANEL4
    for gm in gpu_models: unwrap_prio(gm)
    off = measure_allgpu(sids, splits, gpu_models, bg)
    s = make_prio_stream(-1)
    for gm in gpu_models: wrap_prio(gm, s)
    on = measure_allgpu(sids, splits, gpu_models, bg)
    for gm in gpu_models: unwrap_prio(gm)
    p99_off = float(np.mean([x["infer_p99"] for x in off]))
    p99_on = float(np.mean([x["infer_p99"] for x in on]))
    skip_off = float(np.mean([x["fg_gpu_skip_pct"] for x in off]))
    skip_on = float(np.mean([x["fg_gpu_skip_pct"] for x in on]))
    print(f"  infer p99: off={p99_off:.1f}ms on={p99_on:.1f}ms  (delta={p99_on-p99_off:+.1f})")
    print(f"  fg skip%:  off={skip_off:.1f}  on={skip_on:.1f}")
    effective = (p99_off - p99_on) > 1.0 or (skip_off - skip_on) > 1.0
    print(f"  => priority appears effective: {effective} "
          f"(else 'injected but no effect' — sweep proceeds regardless)")
    return {"infer_p99_off": p99_off, "infer_p99_on": p99_on,
            "skip_off": skip_off, "skip_on": skip_on, "effective": bool(effective)}


def base_sanity(gpu_models, splits, point=2):
    """4-2: re-measure BASE (no PRIO) at k2, 3 reps. Compare worst-stream sAP to
    existing rev22 mean 0.0765 within +/-0.006."""
    print("\n=== BASE SANITY (4-2): k2 All-GPU, no PRIO, 3 reps ===")
    bg = bg_level_for(point); sids = PANEL4
    for gm in gpu_models: unwrap_prio(gm)
    worsts = []
    for rep in range(N_REPS):
        per = measure_allgpu(sids, splits, gpu_models, bg)
        worsts.append(emit("base", point, rep, per, notes="k2 sanity re-measure"))
    mean_w = float(np.mean(worsts))
    ref = 0.0765  # rev22 k2 All-GPU worst mean
    diff = abs(mean_w - ref)
    ok = diff <= 0.006
    print(f"  re-measured worst mean={mean_w:.4f} (reps {[round(w,4) for w in worsts]}) "
          f"vs rev22 {ref:.4f}  |diff|={diff:.4f}  PASS={ok}")
    return {"remeasured_worst_mean": mean_w, "worsts": worsts, "rev22_ref": ref,
            "abs_diff": diff, "pass": bool(ok)}


def prio_sweep(gpu_models):
    print("\n=== PRIO FULL SWEEP: 6 points x 3 reps, All-GPU ===")
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    s = make_prio_stream(-1)
    for gm in gpu_models: wrap_prio(gm, s)
    for point in POINTS:
        bg = bg_level_for(point)
        for rep in range(N_REPS):
            per = measure_allgpu(PANEL4, splits, gpu_models, bg)
            emit("prio", point, rep, per)
    for gm in gpu_models: unwrap_prio(gm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pre", "sweep"])
    args = ap.parse_args()

    torch.set_num_threads(THREADS)
    print(f"=== EXP-QOS v2 stage={args.stage} threads={THREADS} period={PERIOD:.1f}ms ===")
    print("[preload] bg models max=L3 (resnet ORT + qwen2vl torch; tinyllama unused)…")
    t0 = time.time()
    preload_background_models(max_level="L3")
    print(f"[preload] {time.time()-t0:.1f}s")
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print(f"[panel] PANEL4={PANEL4} logs={[sp['log_name'] for sp in splits]}")
    print("[fg] creating 4 GPU YOLOv11s instances…")
    gpu_models = [FGModelGPUGeneric(DET["ultralytics_pt"]) for _ in range(4)]

    if args.stage == "pre":
        rep = {}
        rep["topology"] = topology_probe(gpu_models, splits)
        rep["identity"] = identity_check(gpu_models, splits)
        rep["microbench"] = microbench(gpu_models, splits)
        rep["base_sanity"] = base_sanity(gpu_models, splits)
        (RES / "exp_qos_pre.json").write_text(json.dumps(rep, indent=2))
        print("\n=== PRE COMPLETE ===")
        print(json.dumps(rep, indent=2))
        if not rep["base_sanity"]["pass"]:
            print("\n*** BASE SANITY FAILED — STOP. Do not run sweep. Report to human. ***")
    else:
        prio_sweep(gpu_models)
        print("\n=== SWEEP COMPLETE ===")


if __name__ == "__main__":
    main()
