"""NPU server process: the only owner of one NeublaDriver handle (Phase N1 section 2-2).

Why a process. Measured on this board (revision_profiling_npu/code_notes/NPU_THREADING.md):
  - a driver initialised in one thread and called from another fails ("Failed to program DMA0 engine");
  - two drivers (npu0, npu1) used concurrently from two threads of one process fail ("Failed to sg_dma_read") and hang.
The old runtime ran every NPU pipeline in its own multiprocessing.Process (model_processors.py run_*_npu_process in the
main checkout). Here only the driver calls (SendInput -> Launch -> ReceiveOutputs) run in the server process; everything
else of a request (preprocess, front quantize ONNX, back ONNX, postprocess) stays in the executor's worker thread, so the
request is timed by the unchanged Worker.run loop (dequeue -> output put) and includes the IPC round trip.

Transport: one shared-memory input buffer, one output buffer (multiprocessing RawArray, uint8) and a Pipe for control.
The worker thread blocks in Connection.recv() (GIL released) while the NPU runs.

Driver API (npu.py lines 13-30 of the main checkout): NeublaDriver().Init(npu_id), LoadModel(o_path),
SendInput(bytes, n), Launch(), ReceiveOutputs() -> list[bytes], Close(); every call returns 0 on success.
The extension module is loaded by absolute path (registry `driver_so`, sha256-checked), not via sys.path.
"""
import sys

sys.dont_write_bytecode = True

import time

IN_CAP = 3 * 608 * 608            # largest NPU input (YOLOv3 608x608x3 uint8)
OUT_CAP = 8 * 1024 * 1024         # >= sum of NPU output buffers (checked per request)


def _load_driver_module(so_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("NeublaDriver", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_main(npu_id, so_path, o_path, in_buf, out_buf, conn):
    """Entry point of the server process (spawn). Protocol:
       -> ("run", n_bytes)    <- ("ok", [len of each output], npu_ns) | ("err", repr)
       -> ("close",)          <- ("closed", rc)"""
    import os
    import numpy as np
    drv = None
    try:
        mod = _load_driver_module(so_path)
        t = time.perf_counter()
        drv = mod.NeublaDriver()
        rc_i = drv.Init(int(npu_id))
        if rc_i != 0:
            raise RuntimeError(f"Init({npu_id}) returned {rc_i}")
        rc_l = drv.LoadModel(o_path)
        if rc_l != 0:
            raise RuntimeError(f"LoadModel({o_path}) returned {rc_l}")
        conn.send(("ready", {"pid": os.getpid(), "npu_id": int(npu_id), "o_path": o_path,
                             "load_ms": (time.perf_counter() - t) * 1000.0, "module_file": mod.__file__}))
    except Exception as e:  # noqa: BLE001
        conn.send(("load_failed", repr(e)))
        return
    inv = np.frombuffer(in_buf, dtype=np.uint8)
    outv = np.frombuffer(out_buf, dtype=np.uint8)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg[0] == "close":
            rc = drv.Close()
            conn.send(("closed", rc))
            return
        try:
            n = int(msg[1])
            data = inv[:n].tobytes()
            t0 = time.monotonic_ns()
            if drv.SendInput(data, n) != 0:
                raise RuntimeError("SendInput != 0")
            if drv.Launch() != 0:
                raise RuntimeError("Launch != 0")
            outs = drv.ReceiveOutputs()
            t1 = time.monotonic_ns()
            lens, off = [], 0
            for b in outs:
                k = len(b)
                if off + k > OUT_CAP:
                    raise RuntimeError(f"NPU output larger than OUT_CAP ({off + k})")
                outv[off:off + k] = np.frombuffer(b, dtype=np.uint8)
                lens.append(k)
                off += k
            conn.send(("ok", lens, t1 - t0))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", repr(e)[:300]))
    try:
        drv.Close()
    except Exception:  # noqa: BLE001
        pass


class NpuClient:
    """Executor-side handle of one server process. Used by exactly one worker thread."""

    def __init__(self, npu_id, so_path, o_path, load_timeout=300):
        import multiprocessing as mp
        import numpy as np
        ctx = mp.get_context("spawn")
        self.in_buf = ctx.RawArray("B", IN_CAP)
        self.out_buf = ctx.RawArray("B", OUT_CAP)
        self.inv = np.frombuffer(self.in_buf, dtype=np.uint8)
        self.outv = np.frombuffer(self.out_buf, dtype=np.uint8)
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=server_main, name=f"npu-server:{npu_id}", daemon=True,
                                args=(npu_id, so_path, o_path, self.in_buf, self.out_buf, child))
        self.proc.start()
        child.close()
        if not self.conn.poll(load_timeout):
            self.proc.kill()
            raise RuntimeError(f"NPU server {npu_id} did not report within {load_timeout} s")
        tag, info = self.conn.recv()
        if tag != "ready":
            self.proc.join(5)
            raise RuntimeError(f"NPU load failed on npu{npu_id}: {info}")
        self.info = info
        self.npu_ns = []          # driver-call duration per request, measured inside the server (diagnostic only)
        self.closed = None

    def run(self, arr_u8):
        """arr_u8: contiguous uint8 array. Returns list of uint8 arrays (copies), one per NPU output buffer."""
        flat = arr_u8.reshape(-1)
        n = flat.size
        self.inv[:n] = flat
        self.conn.send(("run", n))
        tag, *rest = self.conn.recv()
        if tag != "ok":
            raise RuntimeError(f"npu{self.info['npu_id']}: {rest}")
        lens, ns = rest
        self.npu_ns.append(ns)
        outs, off = [], 0
        for k in lens:
            outs.append(self.outv[off:off + k].copy())
            off += k
        return outs

    def close(self, timeout=30):
        if self.closed is not None:
            return self.closed
        try:
            self.conn.send(("close",))
            self.closed = self.conn.recv()[1] if self.conn.poll(timeout) else "timeout"
        except Exception as e:  # noqa: BLE001
            self.closed = f"error:{e!r}"
        self.proc.join(10)
        if self.proc.is_alive():
            self.proc.kill()
            self.closed = f"killed(after {self.closed})"
        return self.closed
