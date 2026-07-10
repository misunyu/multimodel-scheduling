#!/usr/bin/env python3
"""Hang-proof training-data collector.

Runs each schedule combination in its OWN executor subprocess with a hard
OS-level timeout and process-group kill, so a stuck worker (e.g. a slow NPU/VLM
generation that ignores the shutdown signal) can never stall the whole sweep.
Each combo contributes one contention window; all windows are merged into a
single training JSON.

Usage:
    source runtime_env.sh
    $PYTHON_BIN run_collection.py --schedule model_schedules.yaml --duration 8 \
        --out xgboost_model/performance_data/train/performance_collected.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _read_windows(results_dir: Path):
    files = sorted(results_dir.glob("performance_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    try:
        d = json.loads(files[-1].read_text())
    except Exception:
        return []
    return d if isinstance(d, list) else d.get("data", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="model_schedules.yaml")
    ap.add_argument("--duration", type=int, default=8)
    ap.add_argument("--out", default="xgboost_model/performance_data/train/performance_collected.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=12)
    args = ap.parse_args()

    python = os.environ.get("PYTHON_BIN", sys.executable)
    schedule = args.schedule
    combos = list((yaml.safe_load(Path(schedule).read_text()) or {}).keys())
    # Optional metadata sidecar: {combo: {rate_factor, workload}} for split + normalization.
    meta_path = Path(schedule).with_suffix(Path(schedule).suffix + ".meta.json")
    combo_meta = {}
    if meta_path.exists():
        try:
            combo_meta = json.loads(meta_path.read_text())
        except Exception:
            combo_meta = {}
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    per_combo_timeout = args.duration + args.warmup + args.buffer
    master = []
    print(f"[collect] {len(combos)} combos, ~{per_combo_timeout}s each")

    for i, name in enumerate(combos):
        for f in results_dir.glob("performance_*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        cmd = [python, "schedule_executor_main.py", "--schedule", schedule,
               "--schedule_name", name, "--duration", str(args.duration)]
        proc = subprocess.Popen(cmd, env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=per_combo_timeout)
        except subprocess.TimeoutExpired:
            pass
        # kill the whole process group (executor + spawned workers)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        time.sleep(2)  # let NPU/GPU drivers settle

        windows = _read_windows(results_dir)
        ok = False
        if windows:
            w = windows[0]
            w["schedule file"] = os.path.basename(schedule)
            m = combo_meta.get(name, {})
            if "rate_factor" in m:
                w["rate_factor"] = m["rate_factor"]
            if "workload" in m:
                w["workload"] = m["workload"]
            master.append(w)
            ok = True
        out.write_text(json.dumps({"schedule file": os.path.basename(schedule), "data": master}, indent=2))
        tot = windows[0]["total"] if ok else {}
        print(f"[{i+1}/{len(combos)}] {name}: {'ok fps=%.1f tok/s=%.1f' % (tot.get('total_throughput_fps',0), tot.get('total_tokens_per_s',0)) if ok else 'NO DATA'}",
              flush=True)

    print(f"[collect] DONE  {len(master)}/{len(combos)} windows -> {out}")


if __name__ == "__main__":
    main()
