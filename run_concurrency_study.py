#!/usr/bin/env python3
"""Headless N-view runner for the concurrency sweep (up to 8 concurrent models).

Unlike the 4-view GUI executor, this spawns one worker process per scheduled
view (view1..viewN), feeds vision/VLM views from the video stream at each view's
infps, lets LLM views self-generate, and after a warmup measures per-view
throughput, tokens/sec and deadline-miss rate. Writes the same performance-window
schema (with rate_factor / workload tags) as run_collection, merged into one JSON.

Usage:
    source runtime_env.sh
    $PYTHON_BIN run_concurrency_study.py --schedule schedules_concurrency.yaml \
        --duration 8 --out xgboost_model/performance_data/concurrency/performance.json
"""
from __future__ import annotations
import argparse, json, os, queue, sys, threading, time
from pathlib import Path
import multiprocessing as mp
import cv2, numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry as reg
import model_processors as MP

VIDEO = "stockholm_1280x720.mp4"


def _worker_for(model):
    kind = reg.kind_of(model); task = reg.get(model).get("task")
    if kind == "vision" and task == "detection":
        return "det"
    if kind == "vision":
        return "cls"
    if kind == "vlm":
        return "vlm"
    return "llm"


class ViewStat:
    def __init__(self, model, device, infps):
        self.model, self.device, self.infps = model, device, infps
        self.deadline_ms = (1000.0 / infps) * reg.DEADLINE_FACTOR if infps > 0 else float("inf")
        self.completed = self.on_time = 0
        self.tok_sum = 0.0
        self.lock = threading.Lock()
        self.measuring = False

    def reset(self):
        with self.lock:
            self.completed = self.on_time = 0; self.tok_sum = 0.0


def run_combo(name, combo, meta, duration, warmup, ctx):
    views = []  # (vid, model, device, infps, kind)
    for vid_key, cfg in combo.items():
        model = cfg["model"]; dev = reg.norm_device(cfg["execution"])
        infps = float(cfg.get("infps", 0) or 0)
        views.append((cfg.get("display", vid_key), model, dev, infps, _worker_for(model)))

    shutdown = ctx.Event()
    vq = ctx.Queue(maxsize=10)
    vr = ctx.Process(target=MP.video_reader_process, args=(VIDEO, vq, shutdown), daemon=True)
    vr.start()

    procs = []; frame_qs = {}; out_qs = {}; stats = {}
    for (vid, model, dev, infps, wk) in views:
        fq = ctx.Queue(maxsize=10); oq = ctx.Queue(maxsize=10)
        frame_qs[vid] = fq; out_qs[vid] = oq
        stats[vid] = ViewStat(model, dev, infps)
        if wk == "det":
            tgt, a = MP.run_detection_process, (fq, oq, shutdown, dev, vid, model)
        elif wk == "cls":
            tgt, a = MP.run_classification_process, (fq, oq, shutdown, dev, vid, model)
        elif wk == "vlm":
            tgt, a = MP.run_vlm_process, (fq, oq, shutdown, dev, vid, model, infps, 24)
        else:
            tgt, a = MP.run_llm_process, (fq, oq, shutdown, dev, vid, model, infps, 48)
        p = ctx.Process(target=tgt, args=a, daemon=True); p.start(); procs.append(p)

    stop = threading.Event()

    def feeder():
        last = None; interval = {v[0]: (1.0 / v[3] if v[3] > 0 else None) for v in views}
        last_ts = {v[0]: 0.0 for v in views}
        feed_views = [v[0] for v in views if v[4] in ("det", "cls", "vlm")]
        while not stop.is_set():
            try:
                while True:
                    last = vq.get_nowait()
            except queue.Empty:
                pass
            if last is None:
                try:
                    last = vq.get(timeout=0.5)
                except queue.Empty:
                    continue
            now = time.time()
            for vid in feed_views:
                iv = interval[vid]
                if iv is None or (now - last_ts[vid]) >= iv:
                    last_ts[vid] = now
                    try:
                        frame_qs[vid].put_nowait((last.copy(), now))
                    except (queue.Full, Exception):
                        pass
            time.sleep(0.002)

    def collector(vid):
        st = stats[vid]; oq = out_qs[vid]; is_gen = st.model in reg.LLM_MODELS
        while not stop.is_set():
            try:
                item = oq.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                break
            lat = item[-1] if isinstance(item, tuple) and len(item) >= 4 else None
            with st.lock:
                if st.measuring:
                    st.completed += 1
                    if lat is not None and lat <= st.deadline_ms:
                        st.on_time += 1
                    if is_gen:
                        st.tok_sum += float(item[2] or 0.0)

    ft = threading.Thread(target=feeder, daemon=True); ft.start()
    cts = [threading.Thread(target=collector, args=(v[0],), daemon=True) for v in views]
    for t in cts:
        t.start()

    time.sleep(warmup)                       # warmup
    for st in stats.values():
        st.reset(); st.measuring = True
    t0 = time.time()
    time.sleep(duration)                     # measurement window
    dt = time.time() - t0
    for st in stats.values():
        st.measuring = False

    # build window
    models_out = {}; miss_rates = []; tot_fps = 0.0; tot_tok = 0.0
    for (vid, model, dev, infps, wk) in views:
        st = stats[vid]
        completed = st.completed; on_time = st.on_time
        throughput = completed / dt if dt > 0 else 0.0
        tokens = (st.tok_sum / completed) if (completed and model in reg.LLM_MODELS) else 0.0
        offered = int(round(infps * dt)) if infps > 0 else completed
        offered = max(offered, completed)
        miss = (offered - on_time) / offered if offered > 0 else 0.0
        miss_rates.append(miss)
        if model in reg.LLM_MODELS:
            tot_tok += tokens
        else:
            tot_fps += throughput
        models_out[vid] = {"model": model, "execution": dev.upper(),
                           "throughput_fps": round(throughput, 2),
                           "tokens_per_s": round(tokens, 2),
                           "deadline_miss_count": int(offered - on_time),
                           "frames_total": int(offered), "on_time_count": int(on_time),
                           "deadline_miss_rate": round(miss, 4)}
    window = {
        "combination": name, "window_sec": dt,
        "rate_factor": meta.get("rate_factor"), "workload": meta.get("workload"),
        "n_models": meta.get("n_models"), "models": models_out,
        "total": {"total_throughput_fps": round(tot_fps, 2),
                  "total_tokens_per_s": round(tot_tok, 2),
                  "deadline_miss_rate": round(sum(miss_rates) / len(miss_rates), 4) if miss_rates else 0.0},
        "derived": {"deadline_miss_rate": round(sum(miss_rates) / len(miss_rates), 4) if miss_rates else 0.0},
    }

    # cleanup
    stop.set(); shutdown.set()
    for p in [vr] + procs:
        try:
            p.join(timeout=2)
        except Exception:
            pass
    for p in [vr] + procs:
        if p.is_alive():
            try:
                p.terminate(); p.join(timeout=1)
            except Exception:
                pass
    for q in list(frame_qs.values()) + list(out_qs.values()) + [vq]:
        try:
            q.close()
        except Exception:
            pass
    return window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    sched = yaml.safe_load(Path(args.schedule).read_text()) or {}
    meta = {}
    mp_path = Path(args.schedule + ".meta.json")
    if mp_path.exists():
        meta = json.loads(mp_path.read_text())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    windows = []
    combos = list(sched.items())
    print(f"[concurrency] {len(combos)} combos", flush=True)
    for i, (name, combo) in enumerate(combos):
        try:
            w = run_combo(name, combo, meta.get(name, {}), args.duration, args.warmup, ctx)
            w["schedule file"] = os.path.basename(args.schedule)
            windows.append(w)
            t = w["total"]
            print(f"[{i+1}/{len(combos)}] {name} N={w.get('n_models')} rate={w.get('rate_factor')} "
                  f"fps={t['total_throughput_fps']:.0f} miss={t['deadline_miss_rate']:.2f}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] {name} FAIL {type(e).__name__}: {e}", flush=True)
        out.write_text(json.dumps({"schedule file": os.path.basename(args.schedule), "data": windows}, indent=2))
    print(f"[concurrency] DONE {len(windows)}/{len(combos)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
