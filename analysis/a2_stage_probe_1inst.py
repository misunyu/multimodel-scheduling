"""A2 Step 2 follow-up — minimally-perturbing stage probe (1 instrumented stream).

The fully-instrumented probe (a2_stage_probe.py) collapsed L2LM NPU DM from the
stock ~68% to ~23%: per-frame fine-grained timing perturbs the contention
interaction. Fix per gate: keep the stock contention environment by running 3
streams with the UNMODIFIED fg_worker and instrumenting only stream 0. Read
stage timings from stream 0; verify stream 0's DM stays ~stock (~68% under
L2LM) so its per-stage attribution is trustworthy.

Configs: none / synth_c24 / L2LM, >=3 reps. All-NPU N=4, PANEL4, threads=4.
Output: analysis/a2_stage_probe_1inst.csv (+ summary), console verdict.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "accv_experiments/scripts"
sys.path.insert(0, str(SD))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

import torch
from _step_d_common import load_val, load_split_for_sid, fg_worker, per_stream_sap, stop_background
from phase_rev6_sweep import load_npu_engines, set_active_npu_engines, dispose_npu_for, DETECTORS
import step_h2_robustness as h2
from a2_stage_probe import timed_npu_worker
from step0_compare_devices import FPS

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD = 1000.0 / FPS
REPS = 3
STRESS = ROOT / "analysis/a2_cpu_stress_worker.py"
PY = ROOT / ".venv/bin/python"
OUT = ROOT / "analysis/a2_stage_probe_1inst.csv"
STAGES = ["read_pre", "infer", "post", "other"]


def launch_stress(c):
    procs = [subprocess.Popen([str(PY), str(STRESS), str(core)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for core in range(c)]
    if c:
        time.sleep(1.0)
    return procs


def kill_stress(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()


def run_cfg(cfg, splits, npu_models, rep, writer):
    stress, bg_stops, bg_threads = [], [], []
    if cfg == "synth_c24":
        stress = launch_stress(24)
    elif cfg == "L2LM":
        bg_stops, bg_threads = h2.start_bg_custom("L2_lm")
    torch.set_num_threads(4)
    stop = threading.Event()

    rows0 = []                                    # instrumented stream 0
    stock_res = [defaultdict(list) for _ in range(3)]  # streams 1..3 (stock)
    threads = [threading.Thread(target=timed_npu_worker,
                                args=(0, splits[0], npu_models[0], stop, rows0), daemon=True)]
    for j in range(1, 4):
        threads.append(threading.Thread(target=fg_worker,
                       args=(j, "NPU", splits[j], npu_models[j], stock_res[j - 1], stop),
                       daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if cfg == "synth_c24":
        kill_stress(stress)
    elif cfg == "L2LM":
        stop_background(bg_stops, bg_threads)

    # stream 0 (instrumented) stage means + DM
    arr = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows0], float)  # read,infer,post,other,eff,miss
    s0_dm = 100.0 * arr[:, 5].mean() if len(arr) else 0.0
    # stock streams DM (from eff_ms > period)
    stock_dm = []
    for res in stock_res:
        eff = res["eff_ms"][30:] if len(res["eff_ms"]) > 30 else res["eff_ms"]
        if eff:
            stock_dm.append(100.0 * np.mean([e > PERIOD for e in eff]))
    stock_dm_mean = float(np.mean(stock_dm)) if stock_dm else 0.0

    for r in rows0:
        writer.writerow((cfg, rep) + r)
    means = {STAGES[i]: float(arr[:, i].mean()) for i in range(4)} if len(arr) else {s: 0 for s in STAGES}
    eff0 = float(arr[:, 4].mean()) if len(arr) else 0.0
    print(f"  [{cfg}/rep{rep}] s0_DM={s0_dm:.1f}% stock_DM={stock_dm_mean:.1f}% "
          f"eff0={eff0:.1f}ms | " + " ".join(f"{s}={means[s]:.2f}" for s in STAGES), flush=True)
    return means, eff0, s0_dm, stock_dm_mean


def main():
    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 NPU engines + L2 co-tenants ...", flush=True)
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    from _step_d_common import preload_background_models
    preload_background_models("L2")

    fh = open(OUT, "w", newline="")
    fw = csv.writer(fh)
    fw.writerow(["config", "rep", "stream_id", "fidx", "read_pre_ms", "infer_ms",
                 "post_ms", "other_ms", "eff_ms", "miss"])
    acc = defaultdict(lambda: defaultdict(list))
    dm_acc = defaultdict(lambda: {"s0": [], "stock": []})
    for cfg in ["none", "synth_c24", "L2LM"]:
        print(f"\n=== {cfg} ===", flush=True)
        for rep in range(REPS):
            means, eff0, s0dm, stockdm = run_cfg(cfg, splits, npu_models, rep, fw)
            for s in STAGES:
                acc[cfg][s].append(means[s])
            acc[cfg]["eff"].append(eff0)
            dm_acc[cfg]["s0"].append(s0dm)
            dm_acc[cfg]["stock"].append(stockdm)
    fh.close()

    print("\n" + "=" * 80)
    print("MINIMALLY-PERTURBING STAGE DECOMPOSITION (instrumented stream 0 of 4)")
    print(f"{'config':10s} {'s0_DM%':>7s} {'stock_DM%':>9s} | "
          + " ".join(f"{s:>10s}" for s in STAGES) + f"{'eff':>9s}")
    base = None
    for cfg in ["none", "synth_c24", "L2LM"]:
        s0 = np.mean(dm_acc[cfg]["s0"]); st = np.mean(dm_acc[cfg]["stock"])
        ms = {s: (np.mean(acc[cfg][s]), np.std(acc[cfg][s])) for s in STAGES}
        eff = np.mean(acc[cfg]["eff"])
        if cfg == "none":
            base = {s: ms[s][0] for s in STAGES}
        print(f"{cfg:10s} {s0:7.1f} {st:9.1f} | "
              + " ".join(f"{ms[s][0]:6.2f}±{ms[s][1]:3.2f}" for s in STAGES) + f"{eff:9.2f}")
    print("\nDELTA vs none (stream 0):")
    for cfg in ["synth_c24", "L2LM"]:
        d = {s: np.mean(acc[cfg][s]) - base[s] for s in STAGES}
        top = max(d, key=d.get)
        print(f"  {cfg:10s}: " + " ".join(f"{s}+{d[s]:.2f}" for s in STAGES)
              + f"  -> dominant: {top} (+{d[top]:.2f}ms)")
    # validity gate
    l2_s0 = np.mean(dm_acc["L2LM"]["s0"])
    print(f"\nVALIDITY: L2LM stream-0 DM = {l2_s0:.1f}% (stock reference ~68%). "
          f"{'OK — attribution trustworthy' if l2_s0 >= 55 else 'STILL PERTURBED — attribution suspect'}")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
