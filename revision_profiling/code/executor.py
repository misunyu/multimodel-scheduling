"""Headless executor for one placement (spec v2.2 section 2). Single process, one feeder thread + one worker thread
per instance, input queue capacity 2 (drop-newest), warm-up 5 s + window 10 s on a shared monotonic clock.

  python3 -B code/executor.py --schedule <yaml> --combination <id> --out <json> [--repeat 0] [--closed-loop]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent.parent
RUNTIME_NAMES = ("model_processors", "unified_viewer", "view_handlers", "image_processing", "utils", "timing_utils", "npu")

WARMUP_S = 5.0
WINDOW_S = 10.0
QUEUE_CAP = 2
OUT_QUEUE_CAP = 1024
N_IMAGES = 128
N_FRAMES = 64


def project_modules():
    """Modules loaded from inside the repository (site-packages of .venv excluded). Returns (dict, violations)."""
    mods, bad = {}, []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(REPO)
        except ValueError:
            if name in RUNTIME_NAMES:
                mods[name] = str(p)
                bad.append(f"{name}:{p}")
            continue
        if rel.parts[0] == ".venv" or (rel.parts[0] == "revision_profiling" and rel.parts[1] == "vendor"):
            continue
        mods[name] = str(rel)
        if p.parent != CODE_DIR:
            bad.append(f"{name}:{rel}")
    return mods, bad


def assert_module_paths(stage):
    mods, bad = project_modules()
    if bad:
        print(f"[FATAL] module path check failed at {stage}: {bad}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    return mods


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry():
    import yaml
    return yaml.safe_load((CODE_DIR / "model_registry.yaml").read_text())["models"]


def resolve_model(registry, name):
    if name not in registry:
        raise SystemExit(f"[FATAL] model {name!r} is not in model_registry.yaml")
    spec = dict(registry[name])
    spec["name"] = name
    spec["abs_path"] = str(REPO / spec["onnx"])
    files = {spec["onnx"]: spec["sha256"]}
    files.update(spec.get("extra_files") or {})
    spec["files_checked"] = {}
    for rel, want in files.items():
        got = sha256_file(REPO / rel)
        spec["files_checked"][rel] = {"sha256": got, "match": got == want}
        if got != want:
            raise SystemExit(f"[FATAL] sha256 mismatch for {rel}: registry {want}, file {got}")
    return spec


def load_inputs(kinds):
    """Everything up to decode happens here, outside the measured path: file I/O, JPEG decode, video frame decode."""
    import cv2
    out, info = {}, {}
    if "imagenet" in kinds:
        files = sorted((REPO / "imagenet-sample-images").glob("*.JPEG"))[:N_IMAGES]
        out["imagenet"] = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files]
        assert all(i is not None and i.dtype.name == "uint8" for i in out["imagenet"]) and len(files) == N_IMAGES
        info["imagenet"] = {"n": len(files), "first": files[0].name, "last": files[-1].name}
    if "video" in kinds:
        cap = cv2.VideoCapture(str(REPO / "stockholm_1280x720.mp4"))
        frames = []
        while len(frames) < N_FRAMES:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        assert len(frames) == N_FRAMES, len(frames)
        out["video"] = frames
        info["video"] = {"n": len(frames), "file": "stockholm_1280x720.mp4", "shape": list(frames[0].shape)}
    if "prompt" in kinds:
        out["prompt"] = json.loads((CODE_DIR / "lm_assets" / "prompts.json").read_text())
        info["prompt"] = {"file": "revision_profiling/code/lm_assets/prompts.json", "P": out["prompt"]["P"], "K": out["prompt"]["K"]}
    return out, info


def make_worker(log, spec, device, inputs, q_in, q_out):
    from lm_runner import LMWorker
    from worker_cls import ClsWorker
    from worker_yolo import YoloV4Worker
    post = spec["postprocess"]
    if post == "cls_argmax":
        return ClsWorker(log, spec, device, inputs["imagenet"], q_in, q_out)
    if post == "yolov4_decode_nms":
        return YoloV4Worker(log, spec, device, inputs["video"], q_in, q_out)
    if post == "lm_greedy":
        return LMWorker(log, spec, device, [inputs["prompt"][spec["name"]]["ids"]], q_in, q_out)
    raise SystemExit(f"[FATAL] unknown postprocess {post!r}")


class Drainer(threading.Thread):
    """Consumer of the output queues. Its failures are consumer_failed (after completed, never subtracted)."""

    def __init__(self, workers):
        super().__init__(name="drainer", daemon=True)
        self.workers = workers
        self.stop_flag = False
        self.consumed = {w.log.name: 0 for w in workers}

    def run(self):
        while True:
            idle = True
            for w in self.workers:
                while True:
                    try:
                        name, _res = w.out.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:  # noqa: BLE001
                        w.log.consumer_fail_ts.append(time.monotonic_ns())
                        break
                    try:
                        self.consumed[name] += 1
                        idle = False
                    except Exception:  # noqa: BLE001
                        w.log.consumer_fail_ts.append(time.monotonic_ns())
            if idle:
                if self.stop_flag:
                    return
                time.sleep(0.001)


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(name="gpu-sampler", daemon=True)
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def run(self):
        for line in self.proc.stdout:
            try:
                u, m = [float(x) for x in line.strip().split(",")]
            except ValueError:
                continue
            self.samples.append((time.monotonic_ns(), u, m))

    def stop(self):
        self.proc.terminate()


def sleep_until(t_ns):
    while True:
        d = t_ns - time.monotonic_ns()
        if d <= 0:
            return
        time.sleep(min(d / 1e9, 0.2) if d > 2_000_000 else d / 1e9)


def _wait_start(start_val, stop_val):
    while start_val.value == 0 and not stop_val.value:
        time.sleep(0.001)
    return start_val.value


def thread_cpu():
    import psutil
    p = psutil.Process()
    return {t.id: t.user_time + t.system_time for t in p.threads()}, sum(p.cpu_times()[:2])


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--combination", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--closed-loop", action="store_true", help="offline profile: feeder waits instead of dropping")
    ap.add_argument("--feeder", choices=("process", "thread"), default="process")
    args = ap.parse_args()

    import yaml
    import numpy as np
    import onnxruntime as ort
    from counters import InstanceLog, invalid_reasons, system_stats, window_stats
    import multiprocessing as mp
    from feeder import boundary_reader_main, feeder_loop, feeder_process_main
    from shmqueue import ShmQueue
    from worker_base import INFER_NICE, INFER_SCHED_IDLE, INTRA_OP_THREADS, lower_inference_priority

    assert_module_paths("startup")
    sched_path = Path(args.schedule).resolve()
    sched = yaml.safe_load(sched_path.read_text())
    meta = sched["meta"]
    comb = sched["combinations"][args.combination]
    registry = load_registry()
    run_key = f"{sched_path.stem}|{args.combination}|rep{args.repeat}"
    execution_id = time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"

    inst_cfg = [dict(v, name=k) for k, v in comb["instances"].items()]
    specs = {c["name"]: resolve_model(registry, c["model"]) for c in inst_cfg}
    inputs, input_info = load_inputs({s["input_source"] for s in specs.values()})

    # feeder processes are spawned before any model is loaded and idle until start_ns is published
    ctx = mp.get_context("spawn")
    start_val, stop_val = ctx.RawValue("q", 0), ctx.RawValue("b", 0)
    queues = {c["name"]: ShmQueue(ctx, QUEUE_CAP) for c in inst_cfg}
    n_inputs = {c["name"]: (1 if specs[c["name"]]["input_source"] == "prompt" else len(inputs[specs[c["name"]]["input_source"]]))
                for c in inst_cfg}
    feeders, hello = {}, {}
    for c in inst_cfg:
        name = c["name"]
        if args.feeder == "process":
            pc, cc = ctx.Pipe(duplex=False)
            p = ctx.Process(target=feeder_process_main, name=f"feeder:{name}", daemon=True,
                            args=(name, queues[name], c.get("rate"), n_inputs[name], start_val, stop_val, args.closed_loop, cc,
                                  str(CODE_DIR), str(REPO)))
            p.start()
            cc.close()
            a_ns = time.monotonic_ns()
            h = pc.recv()
            b_ns = time.monotonic_ns()
            h["clock_shared_ok"] = bool(a_ns - 5_000_000_000 <= h["mono_ns"] <= b_ns)   # child clock inside parent's bracket
            hello[name] = h
            feeders[name] = (p, pc)
        else:
            box = {}
            t = threading.Thread(name=f"feeder:{name}", daemon=True, target=lambda box=box, name=name, c=c: box.update(bufs=feeder_loop(
                queues[name], c.get("rate"), n_inputs[name], _wait_start(start_val, stop_val), lambda: stop_val.value, args.closed_loop)))
            t.start()
            feeders[name] = (t, box)

    names = [c["name"] for c in inst_cfg]
    rd_parent, rd_child = ctx.Pipe(duplex=False)
    reader = ctx.Process(target=boundary_reader_main, name="boundary-reader", daemon=True,
                         args=(names, [queues[n] for n in names], start_val, stop_val, int(WARMUP_S * 1e9), int(WINDOW_S * 1e9), rd_child))
    reader.start()
    rd_child.close()

    logs, workers, load_err = [], [], []

    def _load_all():
        # Sessions are created in a thread that lowered its own priority first: the ORT thread pools inherit it.
        try:
            lower_inference_priority()
            for c in inst_cfg:
                log = InstanceLog(c["name"])
                w = make_worker(log, specs[c["name"]], c["execution"], inputs, queues[c["name"]], queue.Queue(maxsize=OUT_QUEUE_CAP))
                assert len(w.inputs) == n_inputs[c["name"]]
                logs.append(log)
                workers.append(w)
        except BaseException as e:  # noqa: BLE001
            load_err.append(e)

    lt = threading.Thread(target=_load_all, name="loader")
    lt.start()
    lt.join()
    if load_err:
        stop_val.value = 1
        raise load_err[0]
    assert_module_paths("after_load")

    drainer = Drainer(workers)
    sampler = GpuSampler()
    drainer.start()
    sampler.start()
    for w in workers:
        w.start()
    load1 = os.getloadavg()
    start_ns = time.monotonic_ns() + 200_000_000
    start_val.value = start_ns

    t0 = start_ns + int(WARMUP_S * 1e9)
    t1 = t0 + int(WINDOW_S * 1e9)
    sleep_until(t0)
    cpu0, proc0 = thread_cpu()
    sleep_until(t1)
    cpu1, proc1 = thread_cpu()
    if not rd_parent.poll(20):
        raise SystemExit("[FATAL] boundary reader did not report")
    pend0, pend1 = rd_parent.recv()      # queue length + in-flight of every instance at t0 and t1 (read by the reader process)
    reader.join(timeout=5)

    stop_val.value = 1
    feeder_info = {}
    for w in workers:
        name, log = w.log.name, w.log
        h, x = feeders[name]
        if args.feeder == "process":
            msg = x.recv() if x.poll(20) else None
            h.join(timeout=10)
            if msg is None:
                raise SystemExit(f"[FATAL] feeder process {name} did not report")
            bufs = msg["bufs"]
            feeder_info[name] = {"pid": hello[name]["pid"], "clock": hello[name]["clock"], "clock_shared_ok": hello[name]["clock_shared_ok"],
                                 "modules": msg["modules"], "violations": msg["violations"], "cpu_sec_total": msg["cpu_sec_total"],
                                 "dont_write_bytecode": msg["dont_write_bytecode"], "exitcode": h.exitcode}
        else:
            h.join(timeout=10)
            bufs = x["bufs"]
            feeder_info[name] = {"thread": True, "violations": []}
        log.gen_ts, log.gen_ok, log.gen_delay, log.missed_grid, log.gen_id = [list(map(int, b)) for b in bufs]
    for w in workers:
        w.stop_flag = True
    for w in workers:
        w.join(timeout=120)
    alive = [w.log.name for w in workers if w.is_alive()]
    leftover_ids = {w.log.name: w.q.leftover_ids() for w in workers}
    leftover = {k: len(v) for k, v in leftover_ids.items()}
    drainer.stop_flag = True
    drainer.join(timeout=10)
    sampler.stop()
    mods = assert_module_paths("end")
    _, bad = project_modules()
    for name, fi in feeder_info.items():
        bad += [f"feeder[{name}]:{v}" for v in fi["violations"]]
        if args.feeder == "process" and not fi["clock_shared_ok"]:
            bad.append(f"feeder[{name}]:clock_not_shared")

    instances = []
    for w, c in zip(workers, inst_cfg):
        log, name = w.log, w.log.name
        period_ns = None if args.closed_loop else 1e9 / float(c["rate"])
        win = window_stats(log, t0, t1, start_ns, period_ns, pend0[name], pend1[name], leftover_ids[name], closed_loop=args.closed_loop)
        deq = np.asarray(log.deq_ts, dtype=np.int64)
        fin = np.sort(np.concatenate([np.asarray(log.done_ts, dtype=np.int64), np.asarray(log.fail_ts, dtype=np.int64)]))
        n = min(len(deq) - 1, len(fin))
        seq_ok = bool(len(deq) - len(fin) in (0, 1) and (n <= 0 or np.all(deq[1:n + 1] >= fin[:n])))
        enq_total = int(sum(log.gen_ok))
        total_ok = enq_total == len(log.done_ts) + len(log.fail_ts) + leftover[name] and name not in alive
        spec = specs[name]
        tid = lambda th: (round(cpu1.get(th.native_id, 0) - cpu0.get(th.native_id, 0), 4) if th.native_id else None)  # noqa: E731
        fi = feeder_info[name]
        instances.append({
            "name": name, "model": c["model"], "execution": c["execution"],
            "rate": c.get("rate"), "base_rate": c.get("base_rate"), "beta": c.get("beta"), "multiplier": c.get("multiplier"),
            "onnx": spec["onnx"], "files_checked": spec["files_checked"], "providers": w.providers,
            "request_unit": spec["request_unit"], "preprocess": spec["preprocess"], "postprocess": spec["postprocess"],
            "load_ms": w.load_ms, "first_request_ms": w.first_request_ms,
            "window": win,
            "inflight_sequential_ok": seq_ok,
            "total": {"generated": len(log.gen_ts), "enqueued": enq_total, "completed": len(log.done_ts), "failed": len(log.fail_ts),
                      "leftover_in_queue_at_stop": leftover[name], "missed_ticks": len(log.missed_grid),
                      "consumed_by_drainer": drainer.consumed[name]},
            "total_identity_ok": bool(total_ok),
            "consumer_failed_total": len(log.consumer_fail_ts),
            "fail_kinds": sorted(set(log.fail_kind))[:5],
            "cpu_sec_in_window": {"worker_thread": tid(w)}, "worker_nice": getattr(w, "nice", None),
            "feeder": fi,
            "last_result": w.last_result if isinstance(w.last_result, (int, list)) else None,
        })

    gs = [(ts, u, m) for ts, u, m in sampler.samples if t0 < ts <= t1 + 200_000_000]
    window_sec = (t1 - t0) / 1e9
    result = {
        "schema": "revision_profiling/v2.2+1b",
        "run_key": run_key,
        "execution_id": execution_id,
        "schedule_file": sched_path.name,
        "combination": args.combination,
        "repeat": args.repeat,
        "mode": "closed_loop" if args.closed_loop else "open_loop",
        "meta": meta,
        "placement": comb.get("placement"),
        "timing": {"warmup_sec": WARMUP_S, "window_sec_actual": window_sec, "feeder_start_ns": start_ns, "t0_ns": t0, "t1_ns": t1,
                   "snapshot_read_lag_ms_max": max(max(v["ts"] - t0 for v in pend0.values()), max(v["ts"] - t1 for v in pend1.values())) / 1e6,
                   "queue_capacity": QUEUE_CAP, "drop_policy": "drop-newest", "clock": "time.monotonic_ns = CLOCK_MONOTONIC, shared by the executor and the feeder processes",
                   "feeder_mode": args.feeder, "boundary_read": "separate reader process, lock-free shared-memory read"},
        "instances": instances,
        "system": system_stats(instances, window_sec),
        "gpu_samples": {"n": len(gs), "util_mean": float(np.mean([g[1] for g in gs])) if gs else None,
                        "util_max": max([g[1] for g in gs]) if gs else None,
                        "mem_used_mib_max": max([g[2] for g in gs]) if gs else None,
                        "series": [[round((ts - t0) / 1e9, 3), u, m] for ts, u, m in gs]},
        "process_cpu_sec_in_window": round(proc1 - proc0, 3),
        "inputs": input_info,
        "loaded_modules": mods,
        "module_violations": bad,
        "workers_alive_after_join": alive,
        "git_commit": git_commit(),
        "env": {"python": sys.version.split()[0], "onnxruntime": ort.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "cpu_count": os.cpu_count(), "host": platform.node(),
                "switch_interval": sys.getswitchinterval(), "inference_nice": INFER_NICE, "intra_op_num_threads": INTRA_OP_THREADS, "intra_op_allow_spinning": "0", "inference_sched_idle": INFER_SCHED_IDLE,
                "main_thread_nice": os.getpriority(os.PRIO_PROCESS, threading.get_native_id()), "loadavg_before": load1, "dont_write_bytecode": sys.dont_write_bytecode,
                "process_layout": "executor process: one worker thread per instance + drainer; "
                                  + ("one feeder process per instance (spawn)" if args.feeder == "process" else "one feeder thread per instance")},
    }
    result["invalid_reasons"] = invalid_reasons(instances, bad) + ([f"worker_alive:{a}" for a in alive])
    result["valid"] = not result["invalid_reasons"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev = {}
    for log in logs:
        for k in ("gen_ts", "gen_ok", "gen_delay", "gen_id", "missed_grid", "deq_ts", "deq_id", "done_ts", "done_id", "done_enq", "done_deq", "done_pre", "fail_ts", "fail_id"):
            ev[f"{log.name}|{k}"] = np.asarray(getattr(log, k), dtype=np.int64)
    np.savez_compressed(str(out.with_suffix("")) + ".events.npz", **ev)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out)
    s = result["system"]
    print(f"[done] {run_key} valid={result['valid']} F={s['throughput_fps']:.1f} gen={s['generated_rate_fps']:.1f} "
          f"drop={s['drop_rate_fps']:.1f} Dmacro={s['drop_macro']} gpu_util={result['gpu_samples']['util_mean']} "
          f"invalid={result['invalid_reasons']}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
