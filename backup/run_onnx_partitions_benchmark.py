#!/usr/bin/env python3
"""
Run all ONNX partition models under models/resnet50_big/partitions/ with dummy inputs and report execution time.

Usage:
  python run_onnx_partitions_benchmark.py \
      [--dir models/resnet50_big/partitions] \
      [--runs 50] [--warmup 10] [--device cpu] [--batch 1] [--seed 42] \
      [--json-out results/onnx_partitions_benchmark.json] [--verbose]

Notes:
- Requires onnxruntime (pip install onnxruntime). If you have GPU and want CUDA:
  pip install onnxruntime-gpu and use --device cuda
- Input shapes are inferred from the model graph. Dynamic dims are replaced with provided --batch
  for the first dimension; other dynamic dims default to common values if resolvable, otherwise 224.
- Supports models with multiple inputs.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit("onnxruntime is required to run this script. Please install it: pip install onnxruntime\nError: " + str(e))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(REPO_ROOT, "models", "resnet50_big", "partitions")


def find_onnx_files(root_dir: str) -> List[str]:
    matches: List[str] = []
    for base, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".onnx"):
                matches.append(os.path.join(base, f))
    matches.sort()
    return matches


def _resolve_dim(dim: Any, batch_size: int, fallback: int = 224, dim_index: int = 0) -> int:
    # dim can be int or string/None for dynamic; choose reasonable default
    if isinstance(dim, int) and dim > 0:
        return dim
    # If it's the first dimension, use batch size
    if dim_index == 0:
        return batch_size
    # Common image size defaults
    return fallback


def build_dummy_inputs(session: "ort.InferenceSession", batch_size: int) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for i, inp in enumerate(session.get_inputs()):
        name = inp.name
        shape = list(inp.shape)
        # Replace None/'None' or dynamic dims with defaults
        resolved = []
        for j, d in enumerate(shape):
            resolved.append(_resolve_dim(d, batch_size=batch_size, fallback=224, dim_index=j))
        # Determine dtype from ONNX input type string
        type_str = (getattr(inp, 'type', '') or '').lower()
        dtype = np.float32
        if 'uint8' in type_str:
            dtype = np.uint8
        elif 'uint16' in type_str:
            dtype = np.uint16
        elif 'uint32' in type_str:
            dtype = np.uint32
        elif 'int8' in type_str:
            dtype = np.int8
        elif 'int16' in type_str:
            dtype = np.int16
        elif 'int32' in type_str:
            dtype = np.int32
        elif 'int64' in type_str:
            dtype = np.int64
        elif 'float16' in type_str:
            dtype = np.float16
        elif 'float' in type_str or type_str == '':
            dtype = np.float32
        elif 'bool' in type_str:
            dtype = np.bool_
        # Create random data in a reasonable range
        if dtype == np.uint8:
            arr = np.random.randint(0, 256, size=resolved, dtype=np.uint8)
        elif np.issubdtype(dtype, np.unsignedinteger):
            # For other unsigned integers, keep a small range
            info = np.iinfo(dtype)
            high = min(info.max, 100)
            arr = np.random.randint(0, high + 1, size=resolved, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = np.random.randint(0, 10, size=resolved, dtype=dtype)
        elif dtype == np.bool_:
            arr = np.random.rand(*resolved) > 0.5
        else:
            # Floats: values in [-1, 1]
            arr = (np.random.rand(*resolved).astype(np.float32) * 2.0) - 1.0
            if dtype == np.float16:
                arr = arr.astype(np.float16)
        inputs[name] = arr.astype(dtype, copy=False)
    return inputs


def run_benchmark(model_path: str, runs: int, warmup: int, providers: List[str], batch: int, verbose: bool) -> Dict[str, Any]:
    t0 = time.time()
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    setup_time = time.time() - t0

    # Build inputs once
    np.random.seed()
    inputs = build_dummy_inputs(session, batch)

    # Warmup
    for _ in range(warmup):
        session.run(None, inputs)

    # Timed runs
    times: List[float] = []
    for _ in range(runs):
        t1 = time.time()
        session.run(None, inputs)
        t2 = time.time()
        times.append((t2 - t1) * 1000.0)  # ms

    avg_ms = float(np.mean(times)) if times else None
    p50 = float(np.percentile(times, 50)) if times else None
    p90 = float(np.percentile(times, 90)) if times else None
    p95 = float(np.percentile(times, 95)) if times else None
    p99 = float(np.percentile(times, 99)) if times else None

    if verbose:
        print(f"- {os.path.basename(model_path)}: avg {avg_ms:.2f} ms | p50 {p50:.2f} | p95 {p95:.2f} (runs={runs}, warmup={warmup})")

    return {
        "model": model_path,
        "providers": providers,
        "setup_time_ms": setup_time * 1000.0,
        "runs": runs,
        "warmup": warmup,
        "batch": batch,
        "avg_ms": avg_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "all_times_ms": times,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ONNX partition models with dummy inputs and measure times.")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory containing ONNX partition models")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs per model")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs per model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for dynamic first dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--json-out", default="", help="If set, write JSON summary to this path")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-model logs")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    target_dir = os.path.abspath(args.dir)
    if not os.path.isdir(target_dir):
        raise SystemExit(f"Directory not found: {target_dir}")

    onnx_files = find_onnx_files(target_dir)
    if not onnx_files:
        raise SystemExit(f"No .onnx files found under: {target_dir}")

    print(f"Found {len(onnx_files)} ONNX models under {target_dir}")
    results: List[Dict[str, Any]] = []

    for path in onnx_files:
        print(f"Running {path} ...")
        try:
            res = run_benchmark(path, runs=args.runs, warmup=args.warmup, providers=providers, batch=args.batch, verbose=args.verbose)
            results.append(res)
        except Exception as e:
            print(f"ERROR running {path}: {e}")
            results.append({
                "model": path,
                "error": str(e),
                "providers": providers,
            })

    # Print summary
    print("\nSummary (avg_ms):")
    for r in results:
        if "avg_ms" in r and r["avg_ms"] is not None:
            print(f"- {os.path.basename(r['model'])}: {r['avg_ms']:.2f} ms (p95 {r['p95_ms']:.2f})")
        else:
            print(f"- {os.path.basename(r['model'])}: ERROR")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "dir": target_dir,
                "device": args.device,
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
