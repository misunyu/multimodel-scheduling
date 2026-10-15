"""Worker thread: one request at a time. preprocess -> inference -> postprocess -> output queue put.

completed is recorded once, right after the output queue put succeeds (display is not involved).
failed = exception in preprocess/inference/postprocess, or output queue put failure. Mutually exclusive with completed.
"""
import sys

sys.dont_write_bytecode = True

import os
import queue
import threading
import time

import onnxruntime as ort


def make_session(onnx_path, device):
    """CPU EP or CUDA EP only (no fallback). Thread settings are the same for both (Phase 1b user decision)."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = INTRA_OP_THREADS                          # Phase 1b decision: 4 for every session
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # Phase 1b decision: no spinning
    if device == "cpu":
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    elif device == "gpu":
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDAExecutionProvider not active for {onnx_path}: {sess.get_providers()}")
    else:
        raise ValueError(f"unknown device {device!r}")
    return sess


INTRA_OP_THREADS = 4
# niceness of the inference threads (worker + ORT pools); 0 = unchanged. Set per run through REVPROF_NICE (Phase 1b section 2)
INFER_NICE = int(os.environ.get("REVPROF_NICE", "0"))   # Phase 1b section 2 decision: no nice
INFER_SCHED_IDLE = False  # tried: it starves the workers themselves (pilot/arrival/attempt6_sched_idle_dev)


def lower_inference_priority():
    """Applies to the calling thread only (Linux); threads it creates afterwards inherit it."""
    if INFER_NICE:
        os.nice(INFER_NICE)
    if INFER_SCHED_IDLE:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))


class Worker(threading.Thread):
    def __init__(self, log, spec, device, inputs, in_queue, out_queue):
        super().__init__(name=f"worker:{log.name}", daemon=True)
        self.log = log
        self.spec = spec
        self.device = device
        self.inputs = inputs
        self.q = in_queue
        self.out = out_queue
        self.stop_flag = False
        t = time.perf_counter()
        self.session = make_session(spec["abs_path"], device)
        self.providers = self.session.get_providers()
        self.setup()
        self.load_ms = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        self.last_result = self.postprocess(self.infer(self.preprocess(0)), 0)   # one request before the feeder starts
        self.first_request_ms = (time.perf_counter() - t) * 1000.0

    # hooks
    def setup(self):
        pass

    def preprocess(self, idx):
        raise NotImplementedError

    def infer(self, x):
        raise NotImplementedError

    def postprocess(self, y, idx):
        raise NotImplementedError

    def run(self):
        log, q, out = self.log, self.q, self.out
        lower_inference_priority()               # this thread was created by the main thread (normal priority)
        self.nice = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
        while not self.stop_flag:
            item = q.get(0.05)
            if item is None:
                continue
            idx, t_enq, t_deq = item          # the queue marked this request in-flight
            log.deq_ts.append(t_deq)
            log.deq_id.append(idx)
            try:
                x = self.preprocess(idx)
                t_pre = time.monotonic_ns()
                res = self.postprocess(self.infer(x), idx)
            except Exception as e:  # noqa: BLE001 - counted, not swallowed
                log.fail_ts.append(q.finish())
                log.fail_id.append(idx)
                log.fail_kind.append(f"exception:{e!r}"[:300])
                continue
            try:
                out.put_nowait((log.name, res))
            except queue.Full:
                log.fail_ts.append(q.finish())
                log.fail_id.append(idx)
                log.fail_kind.append("output_put_failed")
                continue
            log.done_ts.append(q.finish())    # completed: one timestamp, right after the output put succeeded
            log.done_id.append(idx)
            log.done_enq.append(t_enq)
            log.done_deq.append(t_deq)
            log.done_pre.append(t_pre - t_deq)
            self.last_result = res
