#!/usr/bin/env python3
"""Pick the best phase_c placement via XGBoost and overwrite phase_c in the
dynamic-load schedule YAML.

For BoundGuard (mode 1) and Stop-and-restart (mode 0) runs of
run_dynload_scenarios.sh, the placement used when V(t) > epsilon is whatever
phase_c currently holds in tests/dynamic_load_views_schedule_npu.yaml. This
script enumerates every feasible placement of the phase_b model set across
{cpu, npu0, npu1} (with npu0 and npu1 each hosting at most one model),
predicts T_hat and D_hat with the double-target NPU XGBoost model, and picks
arg max S_hat(x) = T_hat(x) - alpha * D_hat(x) with alpha = 0.3. The selected
placement is written back as phase_c.
"""
import argparse
import os
import sys
from itertools import product
from pathlib import Path

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from xgboost_model.deploy_selector_xgb_suite import (  # noqa: E402
    featurize_from_combo,
    load_models,
)

DEFAULT_SCHED = os.path.join(PROJECT_DIR, "tests", "dynamic_load_views_schedule_npu.yaml")
DEFAULT_MODEL_PREFIX = os.path.join(
    PROJECT_DIR, "xgboost_model", "artifacts", "npu", "xgb_model_npu_double"
)
DEFAULT_ALPHA = 0.3
DEVICES = ("cpu", "npu0", "npu1")


def _phase_entries(phase_cfg):
    """Return [(model, display, infps)] in the YAML's declared order."""
    out = []
    for entry in phase_cfg.values():
        if not isinstance(entry, dict):
            continue
        out.append((entry["model"], entry.get("display"), entry.get("infps")))
    return out


def _build_combo(entries, assignment):
    combo = {}
    for (model, display, infps), dev in zip(entries, assignment):
        key = f"{model}_{dev}"
        combo[key] = {
            "model": model,
            "execution": dev,
            "display": display,
            "infps": int(infps) if infps is not None else None,
        }
    return combo


def _enumerate(n_models):
    for assign in product(DEVICES, repeat=n_models):
        if assign.count("npu0") > 1 or assign.count("npu1") > 1:
            continue
        yield assign


def _split_yaml_header(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    header = []
    for line in raw.splitlines(keepends=True):
        if line.startswith("#") or line.strip() == "":
            header.append(line)
        else:
            break
    return "".join(header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=DEFAULT_SCHED)
    ap.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--source-phase", default="phase_b")
    ap.add_argument("--target-phase", default="phase_c")
    args = ap.parse_args()

    with open(args.schedule, "r", encoding="utf-8") as f:
        sched = yaml.safe_load(f) or {}
    if args.source_phase not in sched:
        print(f"[PickPhaseC] ERROR: '{args.source_phase}' missing in {args.schedule}")
        return 2

    entries = _phase_entries(sched[args.source_phase])
    if not entries:
        print("[PickPhaseC] ERROR: source phase has no entries")
        return 2

    b1, b2, feats, mode, _ = load_models("double", args.alpha, prefix=Path(args.model_prefix))
    if mode != "double" or b2 is None:
        print(f"[PickPhaseC] ERROR: expected double-target model, got mode={mode}")
        return 2

    best = None  # (score, assignment, t_hat, d_hat)
    scored = 0
    for assign in _enumerate(len(entries)):
        combo = _build_combo(entries, assign)
        X = featurize_from_combo(combo).reindex(columns=feats, fill_value=0.0)
        t_hat = float(b1.predict(X)[0])
        d_hat = float(b2.predict(X)[0])
        score = t_hat - args.alpha * d_hat
        scored += 1
        if best is None or score > best[0]:
            best = (score, assign, t_hat, d_hat)

    score, assign, t_hat, d_hat = best
    # NPU hardware fact on this rig: NPU0 has slower PCIe than NPU1. YOLO
    # models move ~10x more bytes per inference than ResNet (608x608 vs
    # 224x224), and on NPU0 the YOLO DMA transactions stall after a few
    # frames. Since the XGBoost features collapse NPU0/NPU1 into the same
    # "non-CPU" bucket, a placement that has yolo on NPU0 and resnet on
    # NPU1 is scored identically to the reverse. Pin YOLO to NPU1 and
    # ResNet to NPU0 when both are chosen — same score, stable runtime.
    assign = list(assign)
    yolo_npu_idx  = [i for i, (m, _, _) in enumerate(entries) if "yolo"   in m and assign[i] in ("npu0", "npu1")]
    resnet_npu_idx = [i for i, (m, _, _) in enumerate(entries) if "resnet" in m and assign[i] in ("npu0", "npu1")]
    for i in yolo_npu_idx:
        assign[i] = "npu1"
    for i in resnet_npu_idx:
        assign[i] = "npu0"
    assign = tuple(assign)

    print(f"[PickPhaseC] Evaluated {scored} placements (alpha={args.alpha})")
    print(f"[PickPhaseC] Best: T_hat={t_hat:.4f}, D_hat={d_hat:.4f}, S_hat={score:.4f}")
    for (model, display, infps), dev in zip(entries, assign):
        print(f"  {model:<16} -> {dev:<5}  display={display} infps={infps}")

    new_phase = {}
    for (model, display, infps), dev in zip(entries, assign):
        key = f"{model}_{dev}"
        entry = {"model": model, "execution": dev, "display": display}
        if infps is not None:
            entry["infps"] = int(infps)
        new_phase[key] = entry
    sched[args.target_phase] = new_phase

    header = _split_yaml_header(args.schedule)
    with open(args.schedule, "w", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(sched, f, default_flow_style=False, sort_keys=False)
    print(f"[PickPhaseC] Wrote '{args.target_phase}' into {args.schedule}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
