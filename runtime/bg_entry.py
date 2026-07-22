"""Standalone entry for one isolated background generative model.

Launched as its own process (``python -m runtime.bg_entry ...``) by
``BackgroundManager`` so that torch/transformers live ONLY here -- the vision
onnxruntime-gpu workers never share a process with torch's cuDNN. This module
imports nothing from the Qt / vision stack; it only needs the LLM engine.

Protocol:
  - loads the model on the given device,
  - writes ``--ready`` file once loaded (parent waits on it),
  - generates in a loop until SIGTERM/SIGINT, then disposes and exits.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True)  # gpu | cpu | npu
    ap.add_argument("--ready", required=True)    # path to touch when loaded
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    tag = f"[bg-llm {args.model}/{args.device}]"
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        from runtime.llm_engine import LLMEngine
        engine = LLMEngine(args.model, args.device, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"{tag} load FAILED: {type(e).__name__}: {e}", flush=True)
        # Still signal ready so the parent does not block for the full timeout.
        try:
            open(args.ready, "w").write("failed")
        except Exception:
            pass
        return 1

    print(f"{tag} loaded ({engine.load_ms:.0f} ms); generating continuously", flush=True)
    try:
        open(args.ready, "w").write("ready")
    except Exception:
        pass

    while not stop["flag"]:
        try:
            engine.infer(max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"{tag} infer error: {type(e).__name__}: {e}", flush=True)
            break
    try:
        engine.dispose()
    except Exception:
        pass
    print(f"{tag} stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
