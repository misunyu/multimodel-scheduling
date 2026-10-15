"""Sanity only (not a pass criterion): top-1 of each classifier on the 128 pre-decoded imagenet-sample-images
(sorted by synset id, i.e. image k is class k) and YOLOv4 detection counts on the pre-decoded video frames.
  python3 -B code/check_models_sanity.py <out.json>"""
import sys

sys.dont_write_bytecode = True
import json
import queue

from counters import InstanceLog
from executor import assert_module_paths, load_inputs, load_registry, make_worker, resolve_model

reg = load_registry()
inputs, _ = load_inputs({"imagenet", "video"})
out = {}
for name, s in reg.items():
    if s["request_unit"] != "image":
        continue
    spec = resolve_model(reg, name)
    w = make_worker(InstanceLog(name), spec, "gpu", inputs, queue.Queue(2), queue.Queue(8))
    n = len(w.inputs)
    res = [w.postprocess(w.infer(w.preprocess(i)), i) for i in range(n)]
    if spec["postprocess"] == "cls_argmax":
        out[name] = {"top1_correct": sum(int(r == i) for i, r in enumerate(res)), "n": n}
    else:
        out[name] = {"detections_per_frame_mean": sum(res) / n, "frames_with_detection": sum(int(r > 0) for r in res), "n": n}
    print(name, out[name], flush=True)
assert_module_paths("end")
json.dump(out, open(sys.argv[1], "w"), indent=1)
