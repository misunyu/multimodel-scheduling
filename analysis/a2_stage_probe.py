"""A2 Task 1b — All-NPU per-frame stage-latency decomposition.

Diagnoses WHERE the NPU path stalls under different co-tenants. The NPU
foreground per-frame cost splits into host/device stages:
  (a) imread + letterbox preprocess   (host CPU + IO)
  (b) NPU on-chip inference m(x)       (device; host blocks on driver dispatch)
  (c) postprocess (dequant/decode/NMS) (host CPU, vendor SDK)
  (d) other/wait = eff - (a+b+c)       (GIL, thread scheduling, CUDA/driver sync)

We replicate _step_d_common.fg_worker's NPU branch + streaming clock VERBATIM
(new instrumented copy in analysis/, frozen harness untouched) and time each
stage per frame. Runs All-NPU N=4 (PANEL4) under 3 co-tenant configs:
  none        : no co-tenant (also yields the canonical threads=4 breakdown)
  synth_c24   : 24 CPU-pinned spin workers (pure user-space CPU saturation)
  L2LM        : ResNet50 + TinyLLaMA ORT-CUDA co-tenants (the co-tenant that
                breaks the NPU path per step_h2_bg_ablation)
x >=3 reps. Also samples host CPU user/system split (psutil); memory-bandwidth
HW counters are unavailable (perf_event_paranoid=4, no root) and reported as N/A.

Goal: localize the L2LM NPU-DM (~76%) vs synthetic-100%-CPU NPU-DM (~27%) gap to
a stage. Outputs: analysis/a2_stage_probe.csv (per-frame long),
analysis/a2_stage_summary.csv (config x stage mean/p50/p95), console verdict.
"""
from __future__ import annotations

import argparse
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

import cv2
import psutil
import torch
from _step_d_common import (load_val, load_split_for_sid, DATA, N_AHD, WARMUP_FRAMES,
                            preload_background_models)
from phase_rev6_sweep import load_npu_engines, set_active_npu_engines, dispose_npu_for, DETECTORS
from step0_compare_devices import FPS, IMG_SIZE, CONF, IOU
import step_h2_robustness as h2

PANEL4 = [2, 22, 3, 21]
DET = [d for d in DETECTORS if d["name"] == "yolo11s"][0]
PERIOD = 1000.0 / FPS
REPS = 3
STRESS = ROOT / "analysis/a2_cpu_stress_worker.py"
PY = ROOT / ".venv/bin/python"
OUT_FRAMES = ROOT / "analysis/a2_stage_probe.csv"
OUT_SUM = ROOT / "analysis/a2_stage_summary.csv"
STAGES = ["read_pre", "infer", "post", "other"]


def npu_infer_timed(img_path, frame_shape, m):
    """Mirror of _step_d_common.npu_infer with per-stage timers.
    Returns (xyxy, scores, coco_cls, timings_dict_ms)."""
    t = {}
    t0 = time.perf_counter()
    img = cv2.imread(str(img_path))
    x = m.preprocess(img)
    t["read_pre"] = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    out = m(x)
    t["infer"] = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    res = m.postprocess(out, conf_thres=CONF, iou_thres=IOU)
    box_cls = getattr(res, "box_cls", None)
    if box_cls is None or box_cls.shape[0] == 0:
        t["post"] = (time.perf_counter() - t2) * 1000.0
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, int), t)
    arr = box_cls.detach().cpu().numpy() if hasattr(box_cls, "detach") else np.asarray(box_cls)
    h0, w0 = frame_shape[:2]
    gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
    pad_x = (IMG_SIZE - w0 * gain) / 2.0
    pad_y = (IMG_SIZE - h0 * gain) / 2.0
    xyxy = arr[:, :4].astype(np.float32, copy=True)
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= gain
    np.clip(xyxy[:, [0, 2]], 0, w0, out=xyxy[:, [0, 2]])
    np.clip(xyxy[:, [1, 3]], 0, h0, out=xyxy[:, [1, 3]])
    t["post"] = (time.perf_counter() - t2) * 1000.0
    return xyxy, arr[:, 4].astype(np.float32), arr[:, 5].astype(int), t


def timed_npu_worker(stream_id, split, model, stop_event, out_rows):
    """Streaming sim (verbatim clock from fg_worker) recording per-frame stages."""
    imgs = split["imgs"]
    coco_mapping = split["coco_mapping"]
    seq_dir = split["seq_dir"]
    n_frame = len(imgs)
    t_total = n_frame / FPS
    frame_shape = (imgs[0]["height"], imgs[0]["width"])
    t_elapsed = 0.0
    last_fidx = -1
    fcount = 0
    while t_elapsed < t_total and not stop_event.is_set():
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last_fidx:
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS
        if fidx >= n_frame:
            break
        last_fidx = fidx
        img_path = DATA / seq_dir / imgs[fidx]["name"]
        wait_start = time.perf_counter()
        _, _, _, t = npu_infer_timed(img_path, frame_shape, model)
        eff = (time.perf_counter() - wait_start) * 1000.0
        rt = eff / 1000.0
        t_elapsed += rt
        other = eff - (t["read_pre"] + t["infer"] + t["post"])
        if fcount >= WARMUP_FRAMES:
            out_rows.append((stream_id, fidx, round(t["read_pre"], 3),
                             round(t["infer"], 3), round(t["post"], 3),
                             round(other, 3), round(eff, 3), int(eff > PERIOD)))
        fcount += 1


class CpuSplitSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self._ev = threading.Event()
        self.samples = []

    def run(self):
        psutil.cpu_times_percent(0.0)
        while not self._ev.is_set():
            ct = psutil.cpu_times_percent(self.interval)
            self.samples.append((ct.user, ct.system, ct.iowait))

    def stop(self):
        self._ev.set()
        self.join(timeout=2)
        if not self.samples:
            return {}
        a = np.array(self.samples)
        return {"cpu_user": round(float(a[:, 0].mean()), 1),
                "cpu_sys": round(float(a[:, 1].mean()), 1),
                "cpu_iowait": round(float(a[:, 2].mean()), 2)}


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


def run_config(cfg, splits, npu_models, rep, frame_writer):
    stress, bg_stops, bg_threads = [], [], []
    if cfg == "synth_c24":
        stress = launch_stress(24)
    elif cfg == "L2LM":
        bg_stops, bg_threads = h2.start_bg_custom("L2_lm")

    stop = threading.Event()
    per_stream_rows = [[] for _ in splits]
    threads = [threading.Thread(target=timed_npu_worker,
                                args=(i, splits[i], npu_models[i], stop, per_stream_rows[i]),
                                daemon=True) for i in range(len(splits))]
    samp = CpuSplitSampler()
    torch.set_num_threads(4)
    samp.start()
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    cpu = samp.stop()

    if cfg == "synth_c24":
        kill_stress(stress)
    elif cfg == "L2LM":
        from _step_d_common import stop_background
        stop_background(bg_stops, bg_threads)

    # aggregate this rep
    all_rows = []
    for i, rows in enumerate(per_stream_rows):
        for r in rows:
            all_rows.append((cfg, rep) + r)
            frame_writer.writerow((cfg, rep) + r)
    arr_eff = np.array([r[8] for r in all_rows], float) if all_rows else np.array([])
    dm = 100.0 * np.mean(arr_eff > PERIOD) if len(arr_eff) else 0.0
    print(f"  [{cfg}/rep{rep}] frames={len(all_rows)} eff_mean={arr_eff.mean():.1f}ms "
          f"DM={dm:.1f}% cpu_user={cpu.get('cpu_user')}% cpu_sys={cpu.get('cpu_sys')}% "
          f"({wall:.1f}s)", flush=True)
    return all_rows, cpu, dm


def plan():
    n = 3 * REPS
    print("A2 Task 1b stage-probe plan:")
    print(f"  All-NPU N=4 (PANEL4) x configs[none, synth_c24, L2LM] x {REPS} reps = {n} cells")
    print(f"  est ~20-35s/cell -> ~{n*28/60:.0f} min GPU(only L2LM)+NPU occupancy "
          f"(+ ~1.5 min NPU load, ~1 min L2 preload)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan(); return

    val = load_val()
    splits = [load_split_for_sid(val, s) for s in PANEL4]
    print("loading 4 NPU engines ...", flush=True)
    npu_models = load_npu_engines(DET, DET["multistream_mxq"], DET["multistream_mode"], 4)
    set_active_npu_engines(npu_models)
    print("preloading L2 co-tenants (ResNet50 + TinyLLaMA, ORT-CUDA) ...", flush=True)
    preload_background_models("L2")

    fh = open(OUT_FRAMES, "w", newline="")
    fw = csv.writer(fh)
    fw.writerow(["config", "rep", "stream_id", "fidx",
                 "read_pre_ms", "infer_ms", "post_ms", "other_ms", "eff_ms", "miss"])

    summary = []
    cpu_by_cfg = defaultdict(list)
    dm_by_cfg = defaultdict(list)
    stage_acc = defaultdict(lambda: defaultdict(list))
    for cfg in ["none", "synth_c24", "L2LM"]:
        print(f"\n=== config: {cfg} ===", flush=True)
        for rep in range(REPS):
            rows, cpu, dm = run_config(cfg, splits, npu_models, rep, fw)
            dm_by_cfg[cfg].append(dm)
            if cpu:
                cpu_by_cfg[cfg].append(cpu)
            for r in rows:
                for si, st in enumerate(STAGES):
                    stage_acc[cfg][st].append(r[4 + si])
                stage_acc[cfg]["eff"].append(r[8])
    fh.close()

    # summary csv
    with open(OUT_SUM, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "dm_pct_mean", "cpu_user_mean", "cpu_sys_mean",
                    "stage", "mean_ms", "p50_ms", "p95_ms", "share_pct"])
        for cfg in ["none", "synth_c24", "L2LM"]:
            dm = float(np.mean(dm_by_cfg[cfg]))
            cu = float(np.mean([c["cpu_user"] for c in cpu_by_cfg[cfg]])) if cpu_by_cfg[cfg] else -1
            csys = float(np.mean([c["cpu_sys"] for c in cpu_by_cfg[cfg]])) if cpu_by_cfg[cfg] else -1
            eff_mean = float(np.mean(stage_acc[cfg]["eff"]))
            for st in STAGES:
                a = np.array(stage_acc[cfg][st], float)
                w.writerow([cfg, round(dm, 1), round(cu, 1), round(csys, 1), st,
                            round(a.mean(), 3), round(np.percentile(a, 50), 3),
                            round(np.percentile(a, 95), 3),
                            round(100 * a.mean() / eff_mean, 1)])

    # console report
    print("\n" + "=" * 78)
    print("STAGE DECOMPOSITION (mean ms per frame, All-NPU N=4, post-warmup)")
    print(f"{'config':10s} {'DM%':>6s} {'user%':>6s} {'sys%':>6s} | "
          + " ".join(f"{s:>9s}" for s in STAGES) + f" {'eff':>8s}")
    base = {}
    for cfg in ["none", "synth_c24", "L2LM"]:
        dm = float(np.mean(dm_by_cfg[cfg]))
        cu = np.mean([c["cpu_user"] for c in cpu_by_cfg[cfg]]) if cpu_by_cfg[cfg] else -1
        cs = np.mean([c["cpu_sys"] for c in cpu_by_cfg[cfg]]) if cpu_by_cfg[cfg] else -1
        means = {st: float(np.mean(stage_acc[cfg][st])) for st in STAGES}
        eff = float(np.mean(stage_acc[cfg]["eff"]))
        if cfg == "none":
            base = means
        print(f"{cfg:10s} {dm:6.1f} {cu:6.1f} {cs:6.1f} | "
              + " ".join(f"{means[s]:9.2f}" for s in STAGES) + f" {eff:8.2f}")
    print("\nDELTA vs none (which stage inflates):")
    for cfg in ["synth_c24", "L2LM"]:
        means = {st: float(np.mean(stage_acc[cfg][st])) for st in STAGES}
        d = {st: means[st] - base[st] for st in STAGES}
        top = max(d, key=d.get)
        print(f"  {cfg:10s}: " + " ".join(f"{s}+{d[s]:.2f}" for s in STAGES)
              + f"   -> dominant: {top} (+{d[top]:.2f}ms)")
    print(f"\nwrote {OUT_FRAMES}\nwrote {OUT_SUM}")
    dispose_npu_for("yolo11s")


if __name__ == "__main__":
    main()
