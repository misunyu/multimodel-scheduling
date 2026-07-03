"""STEP 1 of B3 (rev 4) — probe YOLO11 family mxq availability.

For each member of {YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x}:
  1. Look in ~/.cache/huggingface/hub/models--mobilint--YOLO11* for an existing .mxq.
  2. If absent, attempt construction via mblt_model_zoo.vision.<Cls>(infer_mode='global8'),
     which triggers an HF download. Wrap in a per-detector timeout so a stalled
     download cannot block the whole probe.
  3. Record OK/FAIL with file path + size + wall time to results/v11_mxq_probe.csv.
  4. Decide PLAN: A (5 of 5), A-partial, B (s only), or terminate (0 of 5).

This script is read-only with respect to other artifacts: it touches the HF
cache (download side-effect) and writes one CSV.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)

CACHE = Path.home() / ".cache" / "huggingface" / "hub"
TIMEOUT_S = 480  # per detector hard cap (download time)

DETECTORS = [
    {"name": "yolo11n", "cls": "YOLO11n", "repo": "mobilint/YOLO11n"},
    {"name": "yolo11s", "cls": "YOLO11s", "repo": "mobilint/YOLO11s"},
    {"name": "yolo11m", "cls": "YOLO11m", "repo": "mobilint/YOLO11m"},
    {"name": "yolo11l", "cls": "YOLO11l", "repo": "mobilint/YOLO11l"},
    {"name": "yolo11x", "cls": "YOLO11x", "repo": "mobilint/YOLO11x"},
]


def find_existing_mxq(name):
    pat = str(CACHE / f"models--mobilint--{name.upper()}/snapshots/*/aries/{name}.mxq")
    pat2 = str(CACHE / f"models--mobilint--{name.upper().replace('YOLO','YOLO')}/snapshots/*/aries/{name}.mxq")
    cands = glob.glob(pat) + glob.glob(pat2)
    # Try case-correct repo (Mobilint uses 'YOLO11n' not 'YOLO11N'); handle both
    pat3 = str(CACHE / f"models--mobilint--{name.replace('yolo','YOLO').replace('YOLO11', 'YOLO11')}/snapshots/*/aries/{name}.mxq")
    cands += glob.glob(pat3)
    return cands[0] if cands else None


class _Timeout(Exception): pass


def _alarm(signum, frame):
    raise _Timeout("download timeout")


def probe_one(det):
    name = det["name"]
    info = {"detector": name, "repo": det["repo"]}
    # 1. existing
    existing = find_existing_mxq(name)
    if existing and os.path.exists(existing):
        sz = os.path.getsize(existing)
        info.update({"status": "ok-cached", "path": existing, "size_bytes": sz,
                     "wall_sec": 0.0, "err": ""})
        return info
    # 2. attempt download via mblt_model_zoo
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        from mblt_model_zoo import vision as mv
        cls = getattr(mv, det["cls"])
        m = cls(infer_mode="global8", product="aries")
        # The constructor downloads (if needed) + loads. Locate the mxq we just got.
        signal.alarm(0)
        wall = time.time() - t0
        # Find the freshly cached file
        existing = find_existing_mxq(name)
        sz = os.path.getsize(existing) if existing else 0
        info.update({"status": "ok-downloaded", "path": existing or "(in-memory only)",
                     "size_bytes": sz, "wall_sec": round(wall, 1), "err": ""})
        try:
            m.dispose()
        except Exception:
            pass
    except _Timeout:
        info.update({"status": "fail-timeout", "path": "", "size_bytes": 0,
                     "wall_sec": TIMEOUT_S, "err": f"download exceeded {TIMEOUT_S}s"})
    except Exception as e:
        signal.alarm(0)
        info.update({"status": "fail-error", "path": "", "size_bytes": 0,
                     "wall_sec": round(time.time() - t0, 1),
                     "err": f"{type(e).__name__}: {str(e)[:200]}"})
    return info


def main():
    rows = []
    for det in DETECTORS:
        print(f"[probe] {det['name']:<10s}  repo={det['repo']}", end=" ", flush=True)
        info = probe_one(det)
        rows.append(info)
        print(f"-> {info['status']}  size={info['size_bytes']/1e6:.1f} MB  wall={info['wall_sec']}s")
        if info["err"]:
            print(f"    err: {info['err']}")

    # Save CSV
    out = RES / "v11_mxq_probe.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "repo", "status", "path",
                                          "size_bytes", "wall_sec", "err"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved {out}")

    # Decide PLAN
    ok = {r["detector"] for r in rows if r["status"].startswith("ok")}
    if ok >= {"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}:
        plan = "A"
    elif "yolo11s" in ok and len(ok) > 1:
        plan = "A-partial"
    elif ok == {"yolo11s"}:
        plan = "B"
    else:
        plan = "terminate"
    decision = {"plan": plan, "available": sorted(ok),
                "missing": sorted({d["name"] for d in DETECTORS} - ok)}
    with open(RES / "v11_plan_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\nPLAN: {plan}")
    print(f"  available: {decision['available']}")
    print(f"  missing:   {decision['missing']}")


if __name__ == "__main__":
    main()
