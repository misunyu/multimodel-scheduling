"""Entry point for the CPU-NPU board (Phase N1). Runs the unchanged executor.main() with three module-level substitutions,
so executor.py itself is not edited:

  executor.make_worker  -> antara_* registry keys go to worker_npu.make_antara_worker; every other key goes to the
                           original make_worker (CPU-GPU path untouched)
  executor.GpuSampler   -> NullSampler (no GPU and no nvidia-smi on this board; gpu_samples is reported empty)
  executor.WINDOW_S     -> --window (Phase N1: 30 s; warm-up stays executor.WARMUP_S = 5 s)

Before main() it checks the placement: every execution in {cpu, npu0, npu1}, every NPU entry on an NPU, CPU entries on
the CPU, and at most one instance per NPU (residency). A violation aborts the run before anything is loaded.
Per NPU instance a sidecar <out stem>.npu.<instance>.json records the server pid, load time, driver-call durations
(diagnostic, measured inside the server) and the Close() return code.

  python3 -B code/executor_npu.py --schedule <yaml> --combination <id> --out <json> [--repeat 0] [--closed-loop] [--window 30]
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path

NPU_DEVICES = ("npu0", "npu1")
DEVICES = ("cpu",) + NPU_DEVICES


class NullSampler:
    samples = []

    def start(self):
        pass

    def stop(self):
        pass


def check_placement(sched_path, combination, registry):
    import yaml
    sched = yaml.safe_load(Path(sched_path).read_text())
    inst = sched["combinations"][combination]["instances"]
    used = {}
    for name, c in inst.items():
        dev, key = c["execution"], c["model"]
        if dev not in DEVICES:
            raise SystemExit(f"[FATAL] {name}: unknown device {dev!r}")
        kind = registry[key].get("device_kind")
        if kind == "npu" and dev not in NPU_DEVICES or kind == "cpu" and dev != "cpu":
            raise SystemExit(f"[FATAL] {name}: registry entry {key} ({kind}) placed on {dev}")
        if dev in NPU_DEVICES:
            if dev in used:
                raise SystemExit(f"[FATAL] residency: {name} and {used[dev]} both on {dev}")
            used[dev] = name


def main():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--schedule")
    ap.add_argument("--combination")
    ap.add_argument("--out")
    own, rest = ap.parse_known_args()
    import executor
    import worker_npu
    registry = executor.load_registry()
    check_placement(own.schedule, own.combination, registry)
    orig = executor.make_worker
    out_stem = str(Path(own.out).with_suffix(""))

    def make_worker(log, spec, device, inputs, q_in, q_out):
        if spec["name"].startswith("antara_"):
            w = worker_npu.make_antara_worker(log, spec, device, inputs, q_in, q_out)
            if hasattr(w, "client"):
                w.sidecar = f"{out_stem}.npu.{log.name}.json"
            return w
        return orig(log, spec, device, inputs, q_in, q_out)

    executor.make_worker = make_worker
    executor.GpuSampler = NullSampler
    executor.WINDOW_S = float(own.window)
    sys.argv = [sys.argv[0], "--schedule", own.schedule, "--combination", own.combination, "--out", own.out] + rest
    executor.main()


if __name__ == "__main__":
    main()
