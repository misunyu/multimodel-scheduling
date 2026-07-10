#!/usr/bin/env python3
"""Static per-model, per-device profiler for the Mobilint + GPU stack.

Measures each registry model on CPU / GPU / NPU and writes the static profile
consumed by the XGBoost placement predictor (deploy_selector_xgb_suite.py).

Schema (sample_profiling_data.json):
{
  "devices": ["cpu", "gpu", "npu"],
  "total_data": [
    { "model": "yolo11s", "kind": "vision",
      "cpu_load": .., "cpu_infer": ..,   # ms
      "gpu_load": .., "gpu_infer": ..,
      "npu_load": .., "npu_infer": .. },
    { "model": "tinyllama", "kind": "llm",
      "cpu_load": .., "cpu_prefill_ms": .., "cpu_tokens_per_s": ..,
      "gpu_load": .., "gpu_prefill_ms": .., "gpu_tokens_per_s": ..,
      "npu_load": .., "npu_prefill_ms": .., "npu_tokens_per_s": .. },
    ...
  ]
}

Vision "infer" is the average device inference time in ms. LLM/VLM report
prefill (TTFT) ms and decode tokens/sec. Missing/failed device measurements are
recorded as null.

Usage:
    source runtime_env.sh
    $PYTHON_BIN profile_models.py --out xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import model_registry as reg

DEVICES = ["cpu", "gpu", "npu"]


def _sample_frame():
    cap = cv2.VideoCapture("stockholm_1280x720.mp4")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    return frame


def profile_vision(model_name, device, frame, warmup=3, iters=15):
    """Return (load_ms, avg_infer_ms) for a vision model on a device."""
    if device == "npu":
        from runtime.mobilint_vision import build_vision_npu
        t0 = time.time()
        model = build_vision_npu(model_name, infer_mode="global8")
        load_ms = (time.time() - t0) * 1000.0
        pre = model.preprocess(frame)
        for _ in range(warmup):
            model(pre)
        ts = []
        for _ in range(iters):
            t = time.time(); model(pre); ts.append((time.time() - t) * 1000.0)
        try:
            model.dispose()
        except Exception:
            pass
        return load_ms, float(np.mean(ts))
    else:
        import onnxruntime as ort
        from utils import resolve_onnx_path
        from model_processors import _yolo_letterbox, _resnet_preprocess
        so = ort.SessionOptions(); so.log_severity_level = 3
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        t0 = time.time()
        sess = ort.InferenceSession(resolve_onnx_path(model_name), sess_options=so, providers=providers)
        load_ms = (time.time() - t0) * 1000.0
        if device == "gpu" and "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("CUDA EP unavailable")
        inp = sess.get_inputs()[0].name
        if reg.is_detection(model_name):
            x, _ = _yolo_letterbox(frame, int(reg.get(model_name).get("input_size", 640)))
        else:
            x = _resnet_preprocess(frame)
        for _ in range(warmup):
            sess.run(None, {inp: x})
        ts = []
        for _ in range(iters):
            t = time.time(); sess.run(None, {inp: x}); ts.append((time.time() - t) * 1000.0)
        return load_ms, float(np.mean(ts))


def profile_generative(model_name, device, frame, max_new_tokens=64):
    """Return (load_ms, prefill_ms, tokens_per_s) for an LLM/VLM on a device."""
    from runtime.llm_engine import LLMEngine
    engine = LLMEngine(model_name, device, max_new_tokens=max_new_tokens)
    is_vlm = reg.kind_of(model_name) == "vlm"
    p = engine.profile(frame=(frame if is_vlm else None), max_new_tokens=max_new_tokens, warmup=1, iters=3)
    engine.dispose()
    return p["load_ms"], p["prefill_ms"], p["tokens_per_s"]


def _baseline_rate(model_name, kind, row, max_new_tokens):
    """Requests/sec on the slowest allowed device (min throughput across devices)."""
    rates = []
    for dev in reg.allowed_devices(model_name):
        if kind == "vision":
            infer = row.get(f"{dev}_infer")
            if infer and infer > 0:
                rates.append(1000.0 / infer)
        else:
            prefill = row.get(f"{dev}_prefill_ms")
            tps = row.get(f"{dev}_tokens_per_s")
            if prefill and tps and tps > 0:
                gen_ms = prefill + (max_new_tokens / tps) * 1000.0
                if gen_ms > 0:
                    rates.append(1000.0 / gen_ms)
    if not rates:
        return None
    return round(min(rates), 3)  # slowest allowed device


def main():
    ap = argparse.ArgumentParser(description="Profile all models on CPU/GPU/NPU.")
    ap.add_argument("--out", required=True, help="Output sample_profiling_data.json path")
    ap.add_argument("--models", nargs="*", default=reg.model_names(), help="Subset of models to profile")
    ap.add_argument("--devices", nargs="*", default=DEVICES, help="Subset of devices")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    frame = _sample_frame()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Merge with any existing profile so per-model runs accumulate.
    existing_rows = {}
    if outp.exists():
        try:
            prev = json.loads(outp.read_text())
            for r in prev.get("total_data", []):
                existing_rows[r.get("model")] = r
        except Exception:
            pass

    def _flush():
        merged = [existing_rows[m] for m in existing_rows]
        out = {"devices": DEVICES, "total_data": merged}
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name in args.models:
        spec = reg.get(name)
        kind = spec["kind"]
        row = {"model": name, "kind": kind}
        for dev in args.devices:
            try:
                if kind == "vision":
                    load_ms, infer_ms = profile_vision(name, dev, frame)
                    row[f"{dev}_load"] = round(load_ms, 3)
                    row[f"{dev}_infer"] = round(infer_ms, 3)
                    print(f"[{name} {dev}] load={load_ms:.1f}ms infer={infer_ms:.2f}ms")
                else:
                    load_ms, prefill_ms, tok_s = profile_generative(name, dev, frame, args.max_new_tokens)
                    row[f"{dev}_load"] = round(load_ms, 3)
                    row[f"{dev}_prefill_ms"] = round(prefill_ms, 3)
                    row[f"{dev}_tokens_per_s"] = round(tok_s, 3)
                    # For a unified "infer" proxy (ms per generation-ish), store prefill as infer.
                    row[f"{dev}_infer"] = round(prefill_ms, 3)
                    print(f"[{name} {dev}] load={load_ms:.1f}ms prefill={prefill_ms:.1f}ms tok/s={tok_s:.2f}")
            except Exception as e:
                print(f"[{name} {dev}] SKIP: {type(e).__name__}: {e}")
                row[f"{dev}_load"] = None
                row[f"{dev}_infer"] = None
                if kind != "vision":
                    row[f"{dev}_prefill_ms"] = None
                    row[f"{dev}_tokens_per_s"] = None

        # Baseline input rate (requests/sec) = throughput on the SLOWEST allowed
        # device. 1x in the rate sweep; higher factors induce overload.
        row["baseline_rate"] = _baseline_rate(name, kind, row, args.max_new_tokens)
        existing_rows[name] = row
        _flush()  # incremental: persist after each model
        print(f"[saved] {name}  baseline_rate={row['baseline_rate']}")

    print(f"[OK] wrote {outp}  ({len(existing_rows)} models)")


if __name__ == "__main__":
    main()
