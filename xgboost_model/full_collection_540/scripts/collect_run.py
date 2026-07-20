"""Full re-collection for one platform, one set-group at a time.

Order is fixed by the caller (set-group argument); within a group the combination
order is randomized (seeded). An all-accelerator anchor is re-measured every 20 runs.
Everything is saved incrementally to a per-platform JSONL, so a crash resumes. Failed
combinations are skipped and logged.

Usage: collect_run.py <npu|gpu> <schedule.yaml> <comma-separated set ids> [duration]
"""
import os, sys, json, time, signal, subprocess, random
from pathlib import Path
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad")
OUTDIR = SC / "collect"
OUTDIR.mkdir(exist_ok=True)
TR = OUTDIR / "traces"
TR.mkdir(exist_ok=True)

accel = sys.argv[1]
schedule = Path(sys.argv[2])
set_ids = sys.argv[3].split(",")
duration = int(sys.argv[4]) if len(sys.argv) > 4 else 180

doc = yaml.safe_load(open(schedule))
meta = json.loads(Path(str(schedule) + ".meta.json").read_text())

# combinations of the requested set groups, randomized within the whole selection
combos = [c for c in doc if meta[c]["set"] in set_ids]
random.seed(20260715)
random.shuffle(combos)

# anchor: all-accelerator S3 combo (every model on the accelerator, LLM included)
anchor = None
for c in doc:
    if meta[c]["set"] == "S3" and all(e["execution"] != "cpu" for e in doc[c].values()):
        anchor = c
        break

results_path = OUTDIR / f"collect_{accel}_results.jsonl"
anchor_path = OUTDIR / f"collect_{accel}_anchor.jsonl"
done = set()
if results_path.exists():
    for l in open(results_path):
        try:
            done.add(json.loads(l)["combo"])
        except Exception:
            pass

py = os.environ.get("PYTHON_BIN", sys.executable)
results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)


def one_run(combo, tag):
    for f in results_dir.glob("performance_*.json"):
        f.unlink()
    trace = TR / f"{accel}_{tag}.jsonl"
    if trace.exists():
        trace.unlink()
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "offscreen", "RECORD_TIME": "1",
                "RESULT_TIME_FILE": str(trace), "RUN_ID": f"{accel}_{tag}",
                "DISABLE_VISUALIZATION": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "WARMUP_SECONDS": "20"})
    p = subprocess.Popen([py, "schedule_executor_main.py", "--schedule", str(schedule),
                          "--schedule_name", combo, "--duration", str(duration)],
                         cwd=str(ROOT), env=env, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + duration + 20 + 120
    while time.time() < deadline:
        if p.poll() is not None:
            break
        if any(json.loads(f.read_text() or "[]") for f in results_dir.glob("performance_*.json")):
            time.sleep(2)
            break
        time.sleep(1)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    time.sleep(3)
    w = None
    for f in sorted(results_dir.glob("performance_*.json")):
        blob = json.loads(f.read_text() or "[]")
        if blob:
            w = (blob if isinstance(blob, list) else [blob])[-1]
    return w


t0 = time.time()
n = 0
failed = []
print(f"### collect {accel}: sets={set_ids} {len(combos)} combos, anchor={anchor}, "
      f"{len(done)} already done", flush=True)
for i, combo in enumerate(combos, 1):
    if (n % 20) == 0:
        aw = one_run(anchor, f"anchor_{int(time.time())}")
        if aw:
            t = aw["total"]
            rec = {"at_run": n, "t_min": round((time.time() - t0) / 60, 1),
                   "y1": t["total_throughput_fps"], "y2": t["deadline_miss_rate"],
                   "y3": t["total_tokens_per_s"]}
            with open(anchor_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  [anchor@{n:3d}] y1={rec['y1']:7.2f} y2={rec['y2']:.4f} y3={rec['y3']:6.2f}", flush=True)
    if combo in done:
        continue
    w = one_run(combo, combo)
    n += 1
    if not w:
        failed.append(combo)
        print(f"  [{i:3d}/{len(combos)}] {combo} ({meta[combo]['set']}) FAILED", flush=True)
        continue
    t = w["total"]
    rec = {"combo": combo, "set": meta[combo]["set"], "stratum": meta[combo]["stratum"],
           "n_accel": meta[combo]["n_accel"], "rate": meta[combo]["rate_factor"],
           "y1": t["total_throughput_fps"], "y2": t["deadline_miss_rate"],
           "y3": t["total_tokens_per_s"], "window": w["window_sec"],
           "models": {v["model"]: (v["execution"], v["throughput_fps"], v["inference_count"])
                      for v in w["models"].values()}}
    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if i % 10 == 0 or i == len(combos):
        print(f"  [{i:3d}/{len(combos)}] {combo} ({meta[combo]['set']} n_acc={meta[combo]['n_accel']}) "
              f"y1={t['total_throughput_fps']:7.2f} y2={t['deadline_miss_rate']:.4f} "
              f"y3={t['total_tokens_per_s']:6.2f}  [{(time.time()-t0)/60:.0f}min]", flush=True)

print(f"### collect {accel} GROUP DONE: {n} new runs, {len(failed)} failed {failed}, "
      f"{(time.time()-t0)/60:.1f} min", flush=True)
