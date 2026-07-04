"""A2 Step-final B helper — L2LM co-tenant as a SEPARATE, core-pinnable process.

Self-pins to the cores given in argv (comma list), preloads ResNet50 + TinyLLaMA
(ORT-CUDA, same as the in-process L2_lm co-tenant) and loops both until SIGTERM.
Used by a2_affinity.py to place the co-tenant on cores disjoint from the
foreground, to discriminate core-sharing scheduling contention (DM drops when
pinned apart) from memory-bandwidth/cache contention (DM persists).
"""
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "accv_experiments/scripts"))
sys.path.insert(0, str(ROOT / "accv_experiments/minimal_pipeline"))

cores = [int(c) for c in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] else None
if cores:
    try:
        os.sched_setaffinity(0, set(cores))
    except Exception as e:
        print(f"affinity failed: {e}", flush=True)

from _step_d_common import preload_background_models, _bg_resnet50_loop, _bg_tinyllama_loop

preload_background_models("L2")
print(f"[cotenant-proc] pinned={cores} L2 ready", flush=True)

stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: stop.set())
signal.signal(signal.SIGINT, lambda *a: stop.set())
threads = [threading.Thread(target=_bg_resnet50_loop, args=(stop,), daemon=True),
           threading.Thread(target=_bg_tinyllama_loop, args=(stop,), daemon=True)]
for t in threads:
    t.start()
# re-assert affinity after ORT spawns its threads (they inherit, but be safe)
if cores:
    try:
        os.sched_setaffinity(0, set(cores))
    except Exception:
        pass
stop.wait()
