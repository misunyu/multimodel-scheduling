#!/usr/bin/env python3
"""Generate placement + input-rate-sweep schedules for CPU / GPU / NPU.

For each feasible device placement of a working set (one detection model +
resnet50 + llama1b + qwen2_vl), emit one schedule combination per input-rate
level. Each model's per-view rate is `rate_factor * baseline_rate(model)`, where
baseline_rate (slowest allowed device throughput) comes from the static profile.

Rate levels follow the paper: 1x/2x/4x for training, 3x held out (plus optional
intermediate levels). A metadata sidecar (`<out>.meta.json`) records each
combination's rate_factor and workload id for the train/holdout split and for
per-workload throughput normalization.

Usage:
    python generate_schedules.py --out xgboost_model/schedules/model_schedules.yaml \
        --static_json xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import model_registry as reg


def _load_devices(device_conf_path: str):
    accels = []
    try:
        cfg = yaml.safe_load(Path(device_conf_path).read_text()) or {}
        devs = cfg.get("devices", {})
        if int(devs.get("gpu", {}).get("count", 0)) > 0:
            accels.append("gpu")
        if int(devs.get("npu", {}).get("count", 0)) > 0:
            accels.append("npu")
    except Exception:
        accels = ["gpu", "npu"]
    return accels


def _load_baseline_rates(static_json: str):
    blob = json.loads(Path(static_json).read_text())
    return {r["model"]: r.get("baseline_rate") for r in blob.get("total_data", [])}


def working_sets(models):
    """<=4-model working sets: each detection model + all non-detection models."""
    if len(models) <= 4:
        return [list(models)]
    det = [m for m in models if reg.get(m).get("task") == "detection"]
    other = [m for m in models if reg.get(m).get("task") != "detection"]
    return [([d] + other)[:4] for d in det]


def enumerate_placements(models, platform_devices):
    """All placements on a platform where BOTH devices are shareable.

    Each model is assigned independently to any device it is allowed on that is
    also part of the platform device set (2^N placements before per-model
    constraints). Used for the CPU+GPU and CPU+NPU platform studies where the
    accelerator (RTX 5090 / Mobilint MLA100) can co-host multiple models.
    """
    import itertools
    choices = []
    for m in models:
        allowed = [d for d in reg.allowed_devices(m) if d in platform_devices]
        choices.append(allowed or ["cpu"])
    return [dict(zip(models, combo)) for combo in itertools.product(*choices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xgboost_model/schedules/model_schedules.yaml")
    ap.add_argument("--device_conf", default="target_device.yaml")
    ap.add_argument("--static_json", default="xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json")
    ap.add_argument("--models", nargs="*", default=reg.model_names())
    ap.add_argument("--rate_factors", nargs="*", type=float, default=reg.RATE_FACTORS)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="platform device set, e.g. 'cpu gpu' or 'cpu npu'. Default: cpu + accelerators from device_conf")
    args = ap.parse_args()

    if args.devices:
        platform_devices = list(args.devices)
    else:
        platform_devices = ["cpu"] + _load_devices(args.device_conf)
    baseline = _load_baseline_rates(args.static_json)

    schedules = {}
    meta = {}
    idx = 0
    for ws in working_sets(args.models):
        workload_id = next((m for m in ws if reg.get(m).get("task") == "detection"), ws[0])
        for placement in enumerate_placements(ws, platform_devices):
            for rf in args.rate_factors:
                idx += 1
                name = f"combination_{idx}"
                entry = {}
                for j, (model, dev) in enumerate(placement.items()):
                    base = baseline.get(model) or 1.0
                    infps = round(rf * base, 3)
                    entry[f"{model}_{dev}"] = {
                        "model": model,
                        "execution": dev,
                        "display": f"view{j + 1}",
                        "infps": infps,
                    }
                schedules[name] = entry
                meta[name] = {"rate_factor": rf, "workload": workload_id}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("# model_schedules.yaml - auto-generated (placement x input-rate sweep)\n\n")
        f.write(yaml.dump(schedules, default_flow_style=False, sort_keys=False))
    meta_path = outp.with_suffix(outp.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {outp} ({len(schedules)} combos = placements x {len(args.rate_factors)} rates)")
    print(f"[OK] wrote {meta_path}")


if __name__ == "__main__":
    main()
