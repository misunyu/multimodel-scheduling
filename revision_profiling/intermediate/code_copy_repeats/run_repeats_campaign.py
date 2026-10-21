"""Phase 4b campaign runner (rep1, rep2). Copy of intermediate/code_copy/run_intermediate_campaign.py; changes in ../CHANGES_REPEATS.md.
Run list from repeats_plan.jobs() (rep1 block, then rep2 block); one subprocess of the ORIGINAL code/executor.py per run.

  python3 -B revision_profiling/intermediate/code_copy_repeats/run_repeats_campaign.py --reason "start"   # resumable
  python3 -B revision_profiling/intermediate/code_copy_repeats/run_repeats_campaign.py --dry

A run whose result is missing, incomplete or invalid is executed; an existing bad result is kept as *.rejected.<execution_id>.json.
Up to 3 attempts per run and invocation (1 + 2 retries); after that the run goes to logs/failed_runs.csv.
At the start of each block the runtime files are hashed and asserted equal to the Phase 4 logs/runtime_sha256.json (-> logs/runtime_sha256_rep<r>.json),
and the offline profile is asserted equal to its git HEAD blob.
Logs (new files under revision_profiling/intermediate/logs/, the Phase 4 log files are not touched):
  exec_order_rep1.csv, exec_order_rep2.csv (one line per attempt), campaign_progress_repeats.log (every 10 min), gpu_1min_repeats.csv (every minute),
  campaign_sessions_repeats.log (start / block / end), campaign_runs_repeats/<run>.log (executor output), failed_runs.csv
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent                       # revision_profiling/intermediate
ROOT = IROOT.parent                       # revision_profiling
CODE_DIR = ROOT / "code"                  # original runtime (executor.py and its modules), unchanged
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(CODE_DIR))
import repeats_plan as P  # noqa: E402
from run_campaign import status  # noqa: E402  (original)

LOGS = IROOT / "logs"
RUNTIME_FILES = ("executor.py", "feeder.py", "counters.py", "shmqueue.py", "worker_base.py", "worker_cls.py", "worker_yolo.py", "lm_runner.py",
                 "image_processing.py", "model_registry.yaml")
PROFILE = ROOT / "offline_profile" / "offline_profile.json"
MAX_ATTEMPTS = 3
STATE = {"done": 0, "total": 0, "executed": 0, "retries": 0, "failed": 0, "current": "", "t0": time.time(), "stop": False}


def now():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def append_csv(path, row, header):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)


def gpu_query():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.graphics,power.draw,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        return [x.strip() for x in out.split(",")]
    except Exception:  # noqa: BLE001
        return ["", "", "", "", ""]


def monitor():
    last_progress = 0.0
    while not STATE["stop"]:
        g = gpu_query()
        la = os.getloadavg()
        append_csv(LOGS / "gpu_1min_repeats.csv", {"ts": now(), "epoch": round(time.time(), 1), "gpu_util": g[0], "gpu_temp_c": g[1], "gpu_clock_mhz": g[2],
                                                   "gpu_power_w": g[3], "gpu_mem_mib": g[4], "load1": round(la[0], 2), "load5": round(la[1], 2), "current": STATE["current"]},
                   ["ts", "epoch", "gpu_util", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "gpu_mem_mib", "load1", "load5", "current"])
        if time.time() - last_progress >= 600:
            last_progress = time.time()
            el = time.time() - STATE["t0"]
            rate = STATE["executed"] / el if el > 0 and STATE["executed"] else 0
            left = STATE["total"] - STATE["done"]
            eta = datetime.fromtimestamp(time.time() + left / rate).isoformat(sep=" ", timespec="minutes") if rate else "n/a"
            miss = STATE["failed"] / max(1, STATE["done"])
            with open(LOGS / "campaign_progress_repeats.log", "a") as f:
                f.write(f"{now()} done {STATE['done']}/{STATE['total']} executed_this_session {STATE['executed']} retries {STATE['retries']} "
                        f"failed {STATE['failed']} ({miss:.2%}{' WARNING >1%' if miss > 0.01 else ''}) eta {eta} "
                        f"gpu_util {g[0]}% temp {g[1]}C clock {g[2]}MHz load1 {la[0]:.2f} current {STATE['current']}\n")
        for _ in range(60):
            if STATE["stop"]:
                break
            time.sleep(1)


def reject(out, r):
    tag = (r or {}).get("execution_id", str(int(time.time())))
    os.replace(out, out.with_suffix(f".rejected.{tag}.json"))
    ev = Path(str(out.with_suffix("")) + ".events.npz")
    if ev.exists():
        os.replace(ev, Path(str(out.with_suffix("")) + f".rejected.{tag}.events.npz"))


def runtime_check(rep):
    """sha256 of the runtime files == Phase 4 logs/runtime_sha256.json; offline profile == its git HEAD blob."""
    p4 = json.loads((LOGS / "runtime_sha256.json").read_text())["sha256"]
    rt = {f: hashlib.sha256((CODE_DIR / f).read_bytes()).hexdigest() for f in RUNTIME_FILES}
    assert set(p4) == set(RUNTIME_FILES), "Phase 4 runtime list differs"
    for f in RUNTIME_FILES:
        assert rt[f] == p4[f], f"{f}: sha256 differs from Phase 4 logs/runtime_sha256.json"
    prof = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    head = hashlib.sha256(subprocess.run(["git", "-C", str(ROOT.parent), "show", f"HEAD:{PROFILE.relative_to(ROOT.parent)}"],
                                         capture_output=True, check=True).stdout).hexdigest()
    assert prof == head, "offline profile differs from git HEAD"
    (LOGS / f"runtime_sha256_rep{rep}.json").write_text(json.dumps({"checked_at": now(), "block": f"rep{rep}", "equal_to_phase4_runtime_sha256_json": True,
                                                                    "sha256": rt, "offline_profile_sha256": prof, "offline_profile_equal_to_git_HEAD": True}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reason", default="start")
    ap.add_argument("--limit", type=int, help="stop after this many executed runs (testing)")
    a = ap.parse_args()
    J = P.jobs()
    assert len(J) == 2880
    sch = json.loads((LOGS / "schedule_sha256.json").read_text())["sha256"]
    for rel, h in sch.items():
        assert P.sha(IROOT / rel) == h, f"schedule changed: {rel}"
    todo = [j for j in J if status(Path(j["out"]))[0] != "ok"]
    est = sum(P.P4.est_sec(j["N"]) for j in todo)
    print(f"[phase4b] {len(J)} runs planned, {len(todo)} to execute, estimated {est / 3600:.2f} h", flush=True)
    if a.dry:
        return
    os.environ["REVPROF_NICE"] = "0"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    (LOGS / "campaign_runs_repeats").mkdir(parents=True, exist_ok=True)
    with open(LOGS / "campaign_sessions_repeats.log", "a") as f:
        f.write(f"{now()} START reason={a.reason!r} planned={len(J)} todo={len(todo)} est_h={est / 3600:.2f} pid={os.getpid()}\n")
    STATE.update(total=len(J), done=len(J) - len(todo), t0=time.time())
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    header = ["exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep", "attempt", "start_ts", "end_ts",
              "start_epoch", "end_epoch", "seconds", "rc", "status", "invalid_reasons", "execution_id", "F", "D_macro"]
    block = None
    try:
        for j in todo:
            if j["block"] != block:
                block = j["block"]
                runtime_check(j["rep"])
                with open(LOGS / "campaign_sessions_repeats.log", "a") as f:
                    f.write(f"{now()} BLOCK {block} start (runtime sha256 = Phase 4, offline profile = git HEAD) todo_in_block="
                            f"{sum(1 for x in todo if x['block'] == block)}\n")
            out = Path(j["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            name = f"{j['family']}_x{j['mult']}_{j['combination']}_rep{j['rep']}"
            STATE["current"] = name
            ok = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                st, r = status(out)
                if st in ("incomplete", "invalid"):
                    reject(out, r)
                t, ts = time.time(), now()
                with open(LOGS / "campaign_runs_repeats" / (name + ".log"), "a") as lf:
                    rc = subprocess.run([sys.executable, "-B", "-W", "ignore", str(CODE_DIR / "executor.py"), "--schedule", j["schedule"],
                                         "--combination", j["combination"], "--repeat", str(j["rep"]), "--out", str(out)],
                                        stdout=lf, stderr=subprocess.STDOUT).returncode
                st, r = status(out)
                append_csv(LOGS / f"exec_order_rep{j['rep']}.csv", {
                    **{k: j[k] for k in ("exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep")},
                    "attempt": attempt, "start_ts": ts, "end_ts": now(), "start_epoch": round(t, 3), "end_epoch": round(time.time(), 3),
                    "seconds": round(time.time() - t, 2), "rc": rc, "status": st,
                    "invalid_reasons": ";".join(r["invalid_reasons"]) if r and "invalid_reasons" in r else "",
                    "execution_id": (r or {}).get("execution_id", ""),
                    "F": r["system"]["throughput_fps"] if st in ("ok", "invalid") else "", "D_macro": r["system"]["drop_macro"] if st in ("ok", "invalid") else ""}, header)
                if st == "ok":
                    ok = True
                    break
                STATE["retries"] += attempt < MAX_ATTEMPTS
            STATE["done"] += 1
            STATE["executed"] += 1
            if not ok:
                STATE["failed"] += 1
                st, r = status(out)
                if st in ("incomplete", "invalid"):
                    reject(out, r)
                append_csv(LOGS / "failed_runs.csv", {"ts": now(), "exec_order": j["exec_order"], "family": j["family"], "mult": j["mult"],
                                                      "combination": j["combination"], "rep": j["rep"], "attempts": MAX_ATTEMPTS, "last_status": st,
                                                      "invalid_reasons": ";".join(r["invalid_reasons"]) if r and "invalid_reasons" in r else ""},
                           ["ts", "exec_order", "family", "mult", "combination", "rep", "attempts", "last_status", "invalid_reasons"])
            if a.limit and STATE["executed"] >= a.limit:
                break
    finally:
        STATE["stop"] = True
        with open(LOGS / "campaign_sessions_repeats.log", "a") as f:
            f.write(f"{now()} END executed={STATE['executed']} retries={STATE['retries']} failed={STATE['failed']} "
                    f"hours={(time.time() - STATE['t0']) / 3600:.2f}\n")
    print(f"[phase4b] finished: executed {STATE['executed']}, retries {STATE['retries']}, failed {STATE['failed']}, "
          f"{(time.time() - STATE['t0']) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
