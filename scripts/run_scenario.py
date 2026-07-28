#!/usr/bin/env python3
"""Run one scenario end-to-end into a tracked run directory (v24 task 102 / v30 task 138).

Everything a run needs to be re-derivable is written DIRECTLY into runs/<utc>_<tag>/ --
there is no "copy it somewhere afterwards" step, because that missing step is what lost
the artefacts three times (the fig/ apparatus, the plot_*.py regenerators, and the raw
runs behind confirmed_values.json).

Per run:
    schedule_snapshot.yaml   the schedule actually executed
    metrics.csv              per-second V(t) series
    executor.log             stdout: candidate applications, hot-swaps, reverts
    aggregate.json           the executor's per-combination summary (copied from results/)
    run_manifest.json        hashes + provenance + the transition summary

usage:
  run_scenario.py --schedule schedules/q3_misprediction_cpu-gpu.yaml \\
      --mode 1 --tag q3_boundguard --combo-duration combination_stable=22 ... [--background]
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python3")
EXECUTOR = os.path.join(PROJECT, "schedule_executor_main.py")
RUNS = os.path.join(PROJECT, "runs")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def header_field(sched, key):
    for line in open(sched):
        if not line.startswith("#"):
            break
        m = re.match(rf"#\s*{key}\s*:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return None


def transition_summary(log_path):
    """Applied-candidate sequence and per-view hot-swap count from the executor log."""
    applied, swaps, reverts = [], 0, []
    if not os.path.isfile(log_path):
        return {}
    for line in open(log_path, errors="replace"):
        m = re.search(r"\[Executor\] Starting schedule: (\S+)", line)
        if m:
            applied.append(m.group(1))
        if "hot-swap complete" in line:
            swaps += 1
        if "exhausted -> revert to best" in line:
            reverts.append(line.strip())
    return {"applied_sequence": applied, "hotswaps_per_view": swaps,
            "reverts": reverts, "n_candidates_applied": len(applied)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--combo-duration", action="append", default=[])
    ap.add_argument("--combo-trigger", action="append", default=[])
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--buffer", default=None)
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rd = os.path.join(RUNS, f"{stamp}_{a.tag}_r{a.rep}")
    os.makedirs(rd, exist_ok=True)
    sched = os.path.join(rd, "schedule_snapshot.yaml")
    shutil.copy2(a.schedule, sched)
    metrics = os.path.join(rd, "metrics.csv")
    log = os.path.join(rd, "executor.log")

    cmd = [PYTHON, EXECUTOR, "--schedule", sched, "--duration", str(a.duration),
           "--adaptive-mode", str(a.mode), "--metrics-csv", metrics, "--auto_start_all"]
    for c in a.combo_duration:
        cmd += ["--combo-duration", c]
    for c in a.combo_trigger:
        cmd += ["--combo-trigger", c]
    if a.background:
        cmd += ["--background"]

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["FSRR_RATE_REPLICATE"] = "1"
    if a.buffer:
        env["FSRR_FRAME_BUFFER"] = str(a.buffer)

    before = set(glob.glob(os.path.join(PROJECT, "results", "performance_*.json")))
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=PROJECT, env=env, stdout=lf,
                             stderr=subprocess.STDOUT, timeout=a.duration * 6 + 300)
    wall = time.time() - t0
    # the executor writes its aggregate to results/; bring it INTO the run directory so
    # the run is self-contained (the aggregate is the per-view liveness evidence).
    new = sorted(set(glob.glob(os.path.join(PROJECT, "results", "performance_*.json"))) - before)
    if new:
        shutil.copy2(new[-1], os.path.join(rd, "aggregate.json"))

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None

    rk = header_field(a.schedule, "ranking file")
    manifest = {
        "run_dir": os.path.basename(rd),
        "tag": a.tag, "rep": a.rep,
        "schedule_source": os.path.relpath(a.schedule, PROJECT),
        "adaptive_mode": a.mode,
        "combo_durations": a.combo_duration, "combo_triggers": a.combo_trigger,
        "background_enabled": bool(a.background),
        "ranking_file": rk,
        "ranking_sha256": sha256(os.path.join(PROJECT, rk)) if rk and os.path.isfile(os.path.join(PROJECT, rk)) else None,
        "alpha_beta": header_field(a.schedule, "alpha / beta"),
        "lambda_header": header_field(a.schedule, "lambda"),
        "active_background_header": header_field(a.schedule, "active background"),
        "env_knobs": {k: v for k, v in env.items() if k.startswith("FSRR_")},
        "repo_commit": commit,
        "python": sys.version.split()[0],
        "executor_rc": rc, "wallclock_s": round(wall, 1),
        "transitions": transition_summary(log),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [{"file": f, "bytes": os.path.getsize(os.path.join(rd, f)),
                   "sha256": sha256(os.path.join(rd, f))}
                  for f in sorted(os.listdir(rd)) if os.path.isfile(os.path.join(rd, f))],
    }
    json.dump(manifest, open(os.path.join(rd, "run_manifest.json"), "w"), indent=2)
    print(f"[run] {rd}  rc={rc} wall={wall:.0f}s "
          f"applied={manifest['transitions'].get('applied_sequence')} "
          f"hotswaps={manifest['transitions'].get('hotswaps_per_view')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
