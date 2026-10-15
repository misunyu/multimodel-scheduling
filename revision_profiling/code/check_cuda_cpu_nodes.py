"""One ORT profiling pass per model on the CUDA EP: how many node executions (and how much kernel time) run on the CPU EP,
i.e. whether a 'GPU' session uses the CPU thread pool.   python3 -B code/check_cuda_cpu_nodes.py <out.json>"""
import sys

sys.dont_write_bytecode = True
import collections
import json
import os
import queue
import tempfile

import onnxruntime as ort

import worker_base
from counters import InstanceLog
from executor import load_inputs, load_registry, make_worker, resolve_model

reg = load_registry()
inputs, _ = load_inputs({"imagenet", "video", "prompt"})
tmp = tempfile.mkdtemp()
orig = worker_base.make_session


def profiled(path, device):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = worker_base.INTRA_OP_THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tmp, "prof")
    return ort.InferenceSession(path, sess_options=so, providers=["CUDAExecutionProvider"])


worker_base.make_session = profiled
out = {}
for name in reg:
    w = make_worker(InstanceLog(name), resolve_model(reg, name), "gpu", inputs, queue.Queue(2), queue.Queue(8))
    for i in range(3):
        w.postprocess(w.infer(w.preprocess(i)), i)
    ev = json.load(open(w.session.end_profiling()))
    n, dur, ops = collections.Counter(), collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            p = e["args"].get("provider", "?")
            n[p] += 1
            dur[p] += e["dur"]
            if p == "CPUExecutionProvider":
                ops[e["args"].get("op_name")] += 1
    tot = sum(dur.values()) or 1
    out[name] = {"node_executions": dict(n), "kernel_time_us": dict(dur), "cpu_ep_time_share": round(dur.get("CPUExecutionProvider", 0) / tot, 4),
                 "cpu_ep_ops": dict(ops)}
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
