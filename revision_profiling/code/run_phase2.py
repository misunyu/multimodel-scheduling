"""Phase 2 campaign runner (v1.2 section 2.3). Order and run list come from phase2_plan.jobs(); one executor subprocess per run.

  python3 -B code/run_phase2.py --reason "start"            # resumable: valid results are skipped
  python3 -B code/run_phase2.py --dry

A run whose result is missing, incomplete or invalid is executed; an existing bad result is kept as *.rejected.<execution_id>.json.
Up to 3 attempts per run and invocation (1 + 2 retries); after that the run goes to logs/failed_runs.csv.
Logs: logs/exec_order.csv (one line per attempt), logs/campaign_progress.log (every 10 min), logs/gpu_1min.csv (every minute),
logs/campaign_sessions.log (start / resume / end with reason), logs/campaign_runs/<run>.log (executor output).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import phase2_plan as P
from run_campaign import status

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
LOGS = ROOT / "logs"
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
        append_csv(LOGS / "gpu_1min.csv", {"ts": now(), "epoch": round(time.time(), 1), "gpu_util": g[0], "gpu_temp_c": g[1], "gpu_clock_mhz": g[2],
                                           "gpu_power_w": g[3], "gpu_mem_mib": g[4], "load1": round(la[0], 2), "load5": round(la[1], 2), "current": STATE["current"]},
                   ["ts", "epoch", "gpu_util", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "gpu_mem_mib", "load1", "load5", "current"])
        if time.time() - last_progress >= 600:
            last_progress = time.time()
            el = time.time() - STATE["t0"]
            rate = STATE["executed"] / el if el > 0 and STATE["executed"] else 0
            left = STATE["total"] - STATE["done"]
            eta = datetime.fromtimestamp(time.time() + left / rate).isoformat(sep=" ", timespec="minutes") if rate else "n/a"
            miss = STATE["failed"] / max(1, STATE["done"])
            with open(LOGS / "campaign_progress.log", "a") as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reason", default="start")
    ap.add_argument("--limit", type=int, help="stop after this many executed runs (testing)")
    a = ap.parse_args()
    J = P.jobs()
    todo = [j for j in J if status(Path(j["out"]))[0] != "ok"]
    est = sum(P.est_sec(j["N"]) for j in todo)
    print(f"[phase2] {len(J)} runs planned, {len(todo)} to execute, estimated {est / 3600:.2f} h", flush=True)
    if a.dry:
        return
    os.environ["REVPROF_NICE"] = "0"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    (LOGS / "campaign_runs").mkdir(parents=True, exist_ok=True)
    with open(LOGS / "campaign_sessions.log", "a") as f:
        f.write(f"{now()} START reason={a.reason!r} planned={len(J)} todo={len(todo)} est_h={est / 3600:.2f} pid={os.getpid()}\n")
    STATE.update(total=len(J), done=len(J) - len(todo), t0=time.time())
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    header = ["exec_order", "block", "family_index", "family", "N", "mult", "combination", "rep", "attempt", "start_ts", "end_ts",
              "start_epoch", "end_epoch", "seconds", "rc", "status", "invalid_reasons", "execution_id", "F", "D_macro"]
    try:
        for j in todo:
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
                with open(LOGS / "campaign_runs" / (name + ".log"), "a") as lf:
                    rc = subprocess.run([sys.executable, "-B", "-W", "ignore", str(CODE_DIR / "executor.py"), "--schedule", j["schedule"],
                                         "--combination", j["combination"], "--repeat", str(j["rep"]), "--out", str(out)],
                                        stdout=lf, stderr=subprocess.STDOUT).returncode
                st, r = status(out)
                append_csv(LOGS / "exec_order.csv", {
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
        with open(LOGS / "campaign_sessions.log", "a") as f:
            f.write(f"{now()} END executed={STATE['executed']} retries={STATE['retries']} failed={STATE['failed']} "
                    f"hours={(time.time() - STATE['t0']) / 3600:.2f}\n")
    print(f"[phase2] finished: executed {STATE['executed']}, retries {STATE['retries']}, failed {STATE['failed']}, "
          f"{(time.time() - STATE['t0']) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
