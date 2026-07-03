"""rev12 — measure missing L2_LM 24-log single-stream for all 4 detectors.

Pinned state per claude_code_experiments.md (rev12 spec):
  - NPU binary: legacy v11s mxq sha b2441f9d (verified by T-D0)
  - SDK/driver: same as rev10 (verified by GATE-Q PASS = -0.201)
  - detector family: YOLO11 s/m/l/x
  - dataset: Argoverse-HD val 24 logs

Only L2_LM × 24 logs is missing from rev10 coverage (rev10 A-3 measured only
4 sids at L2_LM). All other data (L0 single-stream, N=4 L1_light multi-stream)
is reused from rev10 since pinned state is unchanged.

Output:
  results/rev12_l2lm_single.csv  — 4 det × 2 dev × 24 sid = 192 rows
  results/manifest_rev12.json    — checkpoint
  results/rev12_l2lm_log.txt     — per-line log
"""

from __future__ import annotations

import csv, json, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _step_d_common import (load_split_for_sid, load_val,
                              preload_background_models)
from phase_rev6_sweep import (FGModelGPUGeneric, load_npu_engines,
                                set_active_npu_engines, dispose_npu_for,
                                measure_single_stream, DETECTORS)
from phase_rev10_measure import _native, short_sha

RES = Path("accv_experiments/results")
RES.mkdir(parents=True, exist_ok=True)
MANIFEST = RES / "manifest_rev12.json"
LOG = RES / "rev12_l2lm_log.txt"
OUT_CSV = RES / "rev12_l2lm_single.csv"

BG = "L2_lm"


def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except Exception: pass
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(_native(m), indent=2))


def mark_done(manifest, key):
    manifest.setdefault("done", {})[key] = {"ts": int(time.time())}
    save_manifest(manifest)


def is_done(manifest, key):
    return key in manifest.get("done", {})


def append_log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts}  {msg}\n"
    with open(LOG, "a") as f: f.write(line)
    print(line, end="", flush=True)


def probe_env():
    info = {}
    try:
        info["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip()
    except Exception as e: info["driver"] = f"unknown ({e})"
    try:
        gu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
              "--format=csv,noheader,nounits"], text=True).strip()
        u, t = gu.split(",")
        info["gpu_util_pct"] = int(u.strip()); info["gpu_temp_c"] = int(t.strip())
    except Exception:
        info["gpu_util_pct"] = -1; info["gpu_temp_c"] = -1
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


COLS = ["detector", "params_M", "phase", "sid", "device", "infer_mode",
        "mxq_path", "mxq_sha8", "bg_level",
        "sap_5095", "sap_50", "sap_s", "sap_m", "sap_l",
        "map_5095", "map_s", "map_m", "map_l",
        "latency_mean", "eff_e2e_mean", "frame_skip_pct", "wall_sec",
        "gpu_util_start", "gpu_temp_start", "ts_iso"]


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def run_l2lm(val, manifest):
    """L2_LM single-stream × all 4 dets × {GPU, NPU} × 24 sids."""
    for det in DETECTORS:
        name = det["name"]; pm = det["params_M"]
        try:
            sha8 = short_sha(det["baseline_mxq"])
        except Exception:
            sha8 = "err"
        gpu = FGModelGPUGeneric(det["ultralytics_pt"])
        npu = load_npu_engines(det, det["baseline_mxq"], det["baseline_mode"], 1)
        set_active_npu_engines(npu)
        append_log(f"[{name}] L2_LM mxq sha8={sha8} mode={det['baseline_mode']}")
        for sid in range(len(val["sequences"])):
            split = load_split_for_sid(val, sid)
            for device, model in [("GPU", gpu), ("NPU", npu[0])]:
                key = f"l2lm/{name}/{device}/{sid}"
                if is_done(manifest, key): continue
                env_pre = probe_env()
                try:
                    m = measure_single_stream(sid, split, device, model, BG)
                except Exception as e:
                    append_log(f"  FAIL {key}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue
                row = {"detector": name, "params_M": pm, "phase": "rev12_l2lm",
                        "sid": sid, "device": device,
                        "infer_mode": det["baseline_mode"] if device == "NPU" else "",
                        "mxq_path": det["baseline_mxq"] if device == "NPU" else "",
                        "mxq_sha8": sha8 if device == "NPU" else "",
                        "bg_level": BG,
                        "gpu_util_start": env_pre.get("gpu_util_pct", -1),
                        "gpu_temp_start": env_pre.get("gpu_temp_c", -1),
                        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in m.items()}}
                append_csv(OUT_CSV, row)
                mark_done(manifest, key)
        dispose_npu_for(name)
        del gpu


def main():
    manifest = load_manifest()
    env = probe_env()
    append_log(f"==== rev12 L2_LM start ==== driver={env.get('driver')} "
                f"gpu_util={env.get('gpu_util_pct')}% temp={env.get('gpu_temp_c')}C "
                f"git={env.get('git_commit','?')[:8]}")
    manifest["env_start"] = env; save_manifest(manifest)

    append_log("[rev12] preloading bg max=L2 (need L2_LM)…")
    t0 = time.time()
    preload_background_models(max_level="L2")
    append_log(f"[rev12] bg preload {time.time()-t0:.1f}s")

    val = load_val()
    append_log(f"[rev12] {len(val['sequences'])} logs")

    run_l2lm(val, manifest)
    append_log("==== rev12 L2_LM done ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
