"""Isolated capacity per model x device, measured through the same worker path (preprocess + inference + postprocess)
with a closed-loop feeder (the worker is never idle, nothing is dropped). Output: offline_profile/offline_profile.json

  python3 -B code/offline_profile.py [--models a,b] [--devices cpu,gpu]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import subprocess
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
OUT = ROOT / "offline_profile"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models")
    ap.add_argument("--devices", default="gpu,cpu")
    a = ap.parse_args()
    reg = yaml.safe_load((CODE_DIR / "model_registry.yaml").read_text())["models"]
    models = a.models.split(",") if a.models else list(reg)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "schedules").mkdir(parents=True, exist_ok=True)
    for m in models:
        for dev in a.devices.split(","):
            out = OUT / "runs" / f"{m}_{dev}.json"
            if out.exists():
                continue
            sched = OUT / "schedules" / f"closed_{m}_{dev}.yaml"
            sched.write_text(yaml.safe_dump({
                "meta": {"family": m, "models": [m], "n_instances": 1, "beta": None, "multiplier": None, "rate_rule": "closed loop"},
                "combinations": {"combination_000": {"placement": dev[0].upper(), "instances": {
                    m: {"model": m, "execution": dev, "base_rate": None, "beta": None, "multiplier": None, "rate": None}}}}}, sort_keys=False))
            log = ROOT / "logs" / f"offline_{m}_{dev}.log"
            with open(log, "w") as lf:
                rc = subprocess.run([sys.executable, "-B", str(CODE_DIR / "executor.py"), "--schedule", str(sched),
                                     "--combination", "combination_000", "--out", str(out), "--closed-loop"],
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            print(m, dev, "rc", rc, flush=True)
    prof = {"definition": "closed-loop single instance, warm-up 5 s + window 10 s; throughput = completed / window; "
                          "service = dequeue -> output put (preprocess + inference + postprocess)", "models": {}}
    for m in reg:
        prof["models"][m] = {}
        for dev in ("gpu", "cpu"):
            p = OUT / "runs" / f"{m}_{dev}.json"
            if not p.exists():
                prof["models"][m][dev] = None
                continue
            r = json.loads(p.read_text())
            w = r["instances"][0]["window"]
            prof["models"][m][dev] = {"throughput_fps": r["system"]["throughput_fps"], "completed": w["completed"],
                                      "service_ms_p50": w["service_ms_p50"], "service_ms_p95": w["service_ms_p95"],
                                      "preprocess_ms_mean": w["preprocess_ms_mean"], "valid": r["valid"],
                                      "invalid_reasons": r["invalid_reasons"], "execution_id": r["execution_id"],
                                      "gpu_util_mean": r["gpu_samples"]["util_mean"]}
    (OUT / "offline_profile.json").write_text(json.dumps(prof, indent=1))
    for m, v in prof["models"].items():
        print(m, {d: (round(x["throughput_fps"], 2), round(x["service_ms_p50"], 3), x["valid"]) if x else None for d, x in v.items()})


if __name__ == "__main__":
    main()
