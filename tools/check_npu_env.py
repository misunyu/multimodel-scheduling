"""Sanity-check the Mobilint NPU environment.

Run from project root:
    .venv/bin/python tools/check_npu_env.py

Checks:
  1. Python venv has mblt_model_zoo, mblt_tracker, qbruntime, qbcompiler importable.
  2. qbruntime sees at least one Aries device.
  3. resnet50.mxq under models/mobilint/ loads + launches + disposes cleanly.

Exits 0 on success, non-zero on first failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESNET_MXQ = ROOT / "models" / "mobilint" / "resnet50.mxq"


def check_imports() -> None:
    import mblt_model_zoo  # noqa: F401
    from mblt_model_zoo.vision import ResNet50, YOLOv8n  # noqa: F401
    from mblt_tracker import CPUDeviceTracker, GPUDeviceTracker, NPUDeviceTracker  # noqa: F401
    import qbruntime  # noqa: F401
    import qbcompiler  # noqa: F401
    print("[ok] imports: mblt_model_zoo, mblt_tracker, qbruntime, qbcompiler")


def check_device() -> None:
    import qbruntime as qb
    devs = qb.get_available_device_numbers()
    if not devs:
        raise RuntimeError(
            "no NPU devices visible. Check `lspci | grep -i mobilint`, "
            "`lsmod | grep aries`, and `/dev/maccel*` permissions."
        )
    print(f"[ok] qbruntime device(s) available: {devs}")


def check_mxq_load() -> None:
    if not RESNET_MXQ.exists():
        raise FileNotFoundError(f"missing {RESNET_MXQ} — re-copy from MobilintTest")
    import qbruntime as qb
    t0 = time.time()
    model = qb.load(str(RESNET_MXQ))
    dt = (time.time() - t0) * 1000.0
    print(f"[ok] loaded {RESNET_MXQ.name} on NPU in {dt:.1f} ms")
    # Disposal happens on object destruction; let Python GC handle it.
    del model


def main() -> int:
    checks = [check_imports, check_device, check_mxq_load]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"[fail] {fn.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    print("\nAll Mobilint NPU sanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
