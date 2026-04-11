#!/usr/bin/env python3
"""
Headless ONNX inference worker.

Runs an ONNX model in a tight loop on either CPU or GPU, with optional
target inference rate. Used by the bounded-recovery sweep to add
non-displayed background load while the main qos_recovery scenario runs
with its own four view models.

The worker exits cleanly on SIGTERM/SIGINT so the sweep driver can
launch and kill it as a Popen subprocess between scenarios.

Usage:
    python scripts/headless_inference_worker.py \
        --model models_onnx/squeezenet1.0-12.onnx \
        --device cpu \
        --rate 30 \
        --duration 60
"""
import argparse
import os
import signal
import sys
import time

import numpy as np

# Lazy import: onnxruntime can be slow to import. We do it after argparse
# so --help is fast.

_KEEP_RUNNING = True


def _stop(*_args):
    global _KEEP_RUNNING
    _KEEP_RUNNING = False


def _resolve_input_shape(sess_input):
    """Replace dynamic dimensions in an ONNX input spec with a usable size.

    Symbolic dimensions (e.g. "batch_size", "N") and unknown dimensions
    (None / -1) get replaced with 1 for batch and a sensible default for
    spatial dims so we can construct a numpy tensor.
    """
    raw = list(sess_input.shape)
    out = []
    for i, dim in enumerate(raw):
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
            continue
        # Symbolic / unknown -> pick a sensible default
        if i == 0:
            out.append(1)            # batch
        elif len(raw) == 4 and i == 1:
            out.append(3)            # channels for image models
        else:
            out.append(224)          # generic spatial
    return tuple(out)


def _make_dummy(sess_input):
    shape = _resolve_input_shape(sess_input)
    dtype = sess_input.type
    if "float16" in dtype:
        np_dtype = np.float16
    elif "float" in dtype:
        np_dtype = np.float32
    elif "int64" in dtype:
        np_dtype = np.int64
    elif "int32" in dtype:
        np_dtype = np.int32
    else:
        np_dtype = np.float32
    if np.issubdtype(np_dtype, np.integer):
        return np.zeros(shape, dtype=np_dtype)
    return np.random.randn(*shape).astype(np_dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Target inferences per second (0 = unbounded)")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Run for this many seconds and then exit")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup inferences before timing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if not os.path.exists(args.model):
        print(f"[bg-worker] model not found: {args.model}", file=sys.stderr)
        return 2

    # Suppress noisy ORT logs unless --verbose was requested.
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1   # one thread per worker keeps CPU stress controllable
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    if args.device == "gpu":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(args.model, sess_options=so, providers=providers)
    except Exception as e:
        print(f"[bg-worker] failed to load {args.model} on {args.device}: {e}",
              file=sys.stderr)
        return 3

    feed = {sess_in.name: _make_dummy(sess_in) for sess_in in sess.get_inputs()}

    # warmup
    for _ in range(max(0, int(args.warmup))):
        try:
            sess.run(None, feed)
        except Exception as e:
            print(f"[bg-worker] warmup failed: {e}", file=sys.stderr)
            return 4

    interval = (1.0 / args.rate) if args.rate > 0 else 0.0
    end = time.time() + float(args.duration)
    n = 0
    started = time.time()
    if not args.quiet:
        print(f"[bg-worker] {os.path.basename(args.model)} on {args.device} "
              f"rate={args.rate} dur={args.duration}s pid={os.getpid()}",
              flush=True)

    while _KEEP_RUNNING and time.time() < end:
        loop_start = time.time()
        try:
            sess.run(None, feed)
        except Exception as e:
            print(f"[bg-worker] inference error: {e}", file=sys.stderr)
            break
        n += 1
        if interval > 0:
            sleep_for = interval - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    elapsed = max(time.time() - started, 1e-6)
    if not args.quiet:
        print(f"[bg-worker] exit pid={os.getpid()} "
              f"n={n} elapsed={elapsed:.2f}s avg_fps={n / elapsed:.2f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
