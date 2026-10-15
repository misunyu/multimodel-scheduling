"""run_campaign.py for the CPU-NPU board: every combination of the given schedules through executor_npu.py, one
subprocess per run, same resume rule (run_campaign.status). Additionally records, per run, the free pages of the DMA32
zone before and after (the plda driver allocates its SG lists there; exhaustion made every LoadModel fail, REPORT_N1 §4)
into <out-dir>/dma32_log.csv, and stops the campaign when free pages fall below --dma32-min.

  python3 -B code/run_campaign_npu.py --out-dir <dir> [--window 30] [--repeats 1] <yaml ...>
  python3 -B code/run_campaign_npu.py --out-dir <dir> --plan <run_plan.csv> --session-log <log>     (Phase N2-A)

--plan: the runs and their order come from run_plan.csv (order, block, schedule, combination, repeat; schedule paths
relative to the CSV). A run fails if its JSON is not ok or any NPU sidecar of the run has Close rc != 0; the failed result
is renamed *.rejected.<execution_id>.json (sidecars too) and the run is retried once at once (<out-dir>/../retries.csv).
A failed retry stops the campaign. Session start/stop lines go to --session-log.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from run_campaign import status

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent


def dma32_free():
    f = False
    for line in open("/proc/zoneinfo"):
        if line.startswith("Node") and "zone" in line:
            f = line.split()[-1] == "DMA32"
        elif f and line.strip().startswith("nr_free_pages"):
            return int(line.split()[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedules", nargs="*")
    ap.add_argument("--plan")
    ap.add_argument("--session-log")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--first-repeat", type=int, default=0)
    ap.add_argument("--combinations", help="comma separated subset (same for every schedule)")
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--dma32-min", type=int, default=20000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return run_plan(a)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    only = set(a.combinations.split(",")) if a.combinations else None
    jobs = []
    for rep in range(a.first_repeat, a.first_repeat + a.repeats):
        for s in a.schedules:
            y = yaml.safe_load(Path(s).read_text())
            for comb in y["combinations"]:
                if only and comb not in only:
                    continue
                jobs.append((Path(s), comb, rep, out_dir / f"{Path(s).stem}__{comb}__rep{rep}.json"))
    todo = [j for j in jobs if status(j[3])[0] != "ok"]
    print(f"[campaign] {len(jobs)} runs, {len(todo)} to execute", flush=True)
    if a.dry:
        return
    dlog = out_dir / "dma32_log.csv"
    if not dlog.exists():
        dlog.write_text("run,dma32_free_before,dma32_free_after,rc,status,sec\n")
    t_all = time.time()
    for n, (s, comb, rep, out) in enumerate(todo, 1):
        st, r = status(out)
        if st in ("incomplete", "invalid"):
            tag = (r or {}).get("execution_id", str(int(time.time())))
            os.replace(out, out.with_suffix(f".rejected.{tag}.json"))
            ev = Path(str(out.with_suffix("")) + ".events.npz")
            if ev.exists():
                os.replace(ev, Path(str(out.with_suffix("")) + f".rejected.{tag}.events.npz"))
        d0 = dma32_free()
        if d0 is not None and d0 < a.dma32_min:
            print(f"[campaign] STOP: DMA32 free pages {d0} < {a.dma32_min}", flush=True)
            return
        t = time.time()
        cmd = [sys.executable, "-B", str(CODE_DIR / "executor_npu.py"), "--schedule", str(s), "--combination", comb,
               "--repeat", str(rep), "--out", str(out), "--window", str(a.window)] + (["--closed-loop"] if a.closed_loop else [])
        with open(log_dir / (out.stem + ".log"), "a") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        st, r = status(out)
        d1 = dma32_free()
        with open(dlog, "a") as fh:
            fh.write(f"{out.stem},{d0},{d1},{rc},{st},{time.time() - t:.1f}\n")
        print(f"[{n}/{len(todo)}] {out.stem} rc={rc} status={st} {time.time() - t:.1f}s dma32={d0}->{d1}"
              + (f" invalid={r['invalid_reasons']}" if r and not r["valid"] else ""), flush=True)
    print(f"[campaign] finished in {(time.time() - t_all) / 60:.1f} min", flush=True)


def sidecar_rcs(out):
    return {json.loads(p.read_text())["instance"]: json.loads(p.read_text()).get("close_rc")
            for p in out.parent.glob(out.stem + ".npu.*.json")}


def run_one(a, s, comb, rep, out, log_dir):
    cmd = [sys.executable, "-B", str(CODE_DIR / "executor_npu.py"), "--schedule", str(s), "--combination", comb,
           "--repeat", str(rep), "--out", str(out), "--window", str(a.window)]
    with open(log_dir / (out.stem + ".log"), "a") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    st, r = status(out)
    y = yaml.safe_load(Path(s).read_text())
    n_npu = sum(1 for i in y["combinations"][comb]["instances"].values() if i["execution"] != "cpu")
    rcs = sidecar_rcs(out)
    bad = []
    if rc != 0:
        bad.append(f"rc={rc}")
    if st != "ok":
        bad.append(f"status={st}" + (f" {r['invalid_reasons']}" if r else ""))
    if len(rcs) != n_npu or any(v != 0 for v in rcs.values()):
        bad.append(f"close_rc={rcs} (npu instances {n_npu})")
    return rc, st, r, bad


def reject(out, why, retries_csv, attempt):
    st, r = status(out)
    tag = (r or {}).get("execution_id", str(int(time.time())))
    for p in [out] + list(out.parent.glob(out.stem + ".npu.*.json")) + [Path(str(out.with_suffix("")) + ".events.npz")]:
        if p.exists():
            os.replace(p, p.parent / p.name.replace(out.stem, f"{out.stem}.rejected.{tag}", 1))
    new = not retries_csv.exists()
    with open(retries_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "run", "attempt", "rejected_execution_id", "reason"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), out.stem, attempt, tag, "; ".join(why)])


def run_plan(a):
    plan = Path(a.plan).resolve()
    base = plan.parent
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    retries_csv = out_dir.parent / "retries.csv"
    jobs = []
    with open(plan) as f:
        for row in csv.DictReader(f):
            s = base / row["schedule"]
            jobs.append((int(row["order"]), row["block"], s, row["combination"], int(row["repeat"]),
                         out_dir / f"{s.stem}__{row['combination']}__rep{row['repeat']}.json"))
    todo = [j for j in jobs if status(j[5])[0] != "ok"]
    sess = open(a.session_log, "a") if a.session_log else None

    def note(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {msg}"
        print(line, flush=True)
        if sess:
            sess.write(line + "\n")
            sess.flush()

    note(f"SESSION START plan={plan.name} total={len(jobs)} todo={len(todo)} first={todo[0][0] if todo else '-'}")
    dlog = out_dir / "dma32_log.csv"
    if not dlog.exists():
        dlog.write_text("order,block,run,attempt,dma32_free_before,dma32_free_after,rc,status,sec\n")
    t_all = time.time()
    for n, (order, block, s, comb, rep, out) in enumerate(todo, 1):
        for attempt in (1, 2):
            st, r = status(out)
            if st in ("incomplete", "invalid"):
                reject(out, [f"left from an earlier session: {st}"], retries_csv, 0)
            d0 = dma32_free()
            if d0 is not None and d0 < a.dma32_min:
                note(f"SESSION STOP DMA32 free pages {d0} < {a.dma32_min} before order {order} ({out.stem})")
                return 2
            t = time.time()
            rc, st, r, bad = run_one(a, s, comb, rep, out, log_dir)
            d1 = dma32_free()
            with open(dlog, "a") as fh:
                fh.write(f"{order},{block},{out.stem},{attempt},{d0},{d1},{rc},{st if not bad else 'failed'},{time.time() - t:.1f}\n")
            print(f"[{n}/{len(todo)}] #{order} {block} {out.stem} rc={rc} status={st} {time.time() - t:.1f}s dma32={d0}->{d1}"
                  + (f" FAILED {bad}" if bad else ""), flush=True)
            if not bad:
                break
            reject(out, bad, retries_csv, attempt)
            if attempt == 2:
                note(f"SESSION STOP retry failed at order {order} ({out.stem}): {bad}")
                return 3
    note(f"SESSION END finished {len(todo)} runs in {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
