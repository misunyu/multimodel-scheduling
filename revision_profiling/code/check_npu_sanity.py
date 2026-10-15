"""Sanity only (not a pass criterion), CPU-NPU board: ResNet50 top-1 on the 128 pre-decoded imagenet-sample-images
(image k is class k) and YOLOv3 box counts on the 64 pre-decoded video frames, for every antara_* registry entry on
every device it can run on. Uses the measured worker classes directly (no feeder / executor timing).
  python3 -B code/check_npu_sanity.py <out.json>
"""
import sys

sys.dont_write_bytecode = True
import json
import queue

from counters import InstanceLog
from executor import load_inputs, load_registry, resolve_model
from worker_npu import make_antara_worker


def main():
    reg = load_registry()
    inputs, _ = load_inputs({"imagenet", "video"})
    out = {}
    for key in sorted(k for k in reg if k.startswith("antara_")):
        spec = resolve_model(reg, key)
        devs = ["npu0", "npu1"] if spec["device_kind"] == "npu" else ["cpu"]
        for dev in devs:
            w = make_antara_worker(InstanceLog(key), spec, dev, inputs, queue.Queue(2), queue.Queue(8))
            n = len(w.inputs)
            res = [w.postprocess(w.infer(w.preprocess(i)), i) for i in range(n)]
            if hasattr(w, "client"):
                w.client.close()
            if spec["logical_model"] == "resnet50":
                r = {"top1_correct": sum(int(c == i) for i, c in enumerate(res)), "n": n}
            else:
                r = {"frames": n, "boxes_total": int(sum(res)), "frames_with_box": int(sum(1 for c in res if c > 0)),
                     "boxes_first8": res[:8]}
            out[f"{key}@{dev}"] = r
            print(key, dev, r, flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
