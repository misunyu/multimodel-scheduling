"""Run every combination of the given schedule files through executor.py, one subprocess per run.

  python3 -B code/run_campaign.py --out-dir pilot/smoke/runs --repeats 1 pilot/smoke/schedules/*.yaml

Resume: a run (run_key = schedule stem | combination | repeat) is skipped only if its JSON exists, has the required
fields and passes conservation. An existing but invalid/incomplete result is kept under *.rejected.<execution_id>.json
and the run is executed again (once per invocation).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
REQUIRED = ("run_key", "execution_id", "instances", "system", "timing", "loaded_modules", "invalid_reasons", "valid", "env")


def status(path):
    """'ok' | 'missing' | 'incomplete' | 'invalid'."""
    if not path.exists():
        return "missing", None
    try:
        r = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return "incomplete", None
    if any(k not in r for k in REQUIRED):
        return "incomplete", r
    return ("ok" if r["valid"] else "invalid"), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedules", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--first-repeat", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rep in range(a.first_repeat, a.first_repeat + a.repeats):
        for s in a.schedules:
            y = yaml.safe_load(Path(s).read_text())
            for comb in y["combinations"]:
                jobs.append((Path(s), comb, rep, out_dir / f"{Path(s).stem}__{comb}__rep{rep}.json"))
    todo = [(s, c, r, o) for s, c, r, o in jobs if status(o)[0] != "ok"]
    print(f"[campaign] {len(jobs)} runs, {len(todo)} to execute, estimated {len(todo) * 22 / 60:.1f} min", flush=True)
    if a.dry:
        return
    t_all = time.time()
    for n, (s, comb, rep, out) in enumerate(todo, 1):
        st, r = status(out)
        if st in ("incomplete", "invalid"):
            tag = (r or {}).get("execution_id", str(int(time.time())))
            os.replace(out, out.with_suffix(f".rejected.{tag}.json"))
            ev = Path(str(out.with_suffix("")) + ".events.npz")
            if ev.exists():
                os.replace(ev, Path(str(out.with_suffix("")) + f".rejected.{tag}.events.npz"))
        t = time.time()
        with open(log_dir / (out.stem + ".log"), "a") as lf:
            rc = subprocess.run([sys.executable, "-B", str(CODE_DIR / "executor.py"), "--schedule", str(s), "--combination", comb,
                                 "--repeat", str(rep), "--out", str(out)], stdout=lf, stderr=subprocess.STDOUT).returncode
        st, r = status(out)
        print(f"[{n}/{len(todo)}] {out.stem} rc={rc} status={st} {time.time() - t:.1f}s"
              + (f" invalid={r['invalid_reasons']}" if r and not r["valid"] else ""), flush=True)
    print(f"[campaign] finished in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
