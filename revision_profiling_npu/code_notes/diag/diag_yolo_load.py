"""Diagnostic: which difference between the plain benchmark and npu_server makes LoadModel(yolov3_big) abort?"""
import sys; sys.dont_write_bytecode = True
SO = "/workspace/lib/NeublaDriver.cpython-38-x86_64-linux-gnu.so"
O = "/workspace/multimodel-scheduling/models/yolov3_big/npu_code/yolov3_big_neubla_p1.o"

def child(tag, bufs, q):
    import importlib.util
    spec = importlib.util.spec_from_file_location("NeublaDriver", SO); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import numpy as np
    d = m.NeublaDriver(); a = d.Init(0); b = d.LoadModel(O)
    x = np.zeros(3 * 608 * 608, np.uint8).tobytes(); d.SendInput(x, len(x)); d.Launch(); r = d.ReceiveOutputs()
    q.put((tag, a, b, [len(t) for t in r], d.Close()))

if __name__ == "__main__":
    import multiprocessing as mp
    mode = sys.argv[1]
    sys.path.insert(0, "/workspace/mms_npu_port/revision_profiling/code")
    if mode in ("C", "D"):
        import onnxruntime, cv2, numpy  # noqa: F401
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    if mode == "D":
        from npu_server import NpuClient
        try:
            c = NpuClient(0, SO, O); print("D ready", c.info); print("D close", c.close())
        except Exception as e:
            print("D failed", repr(e))
        sys.exit(0)
    bufs = (ctx.RawArray("B", 3 * 608 * 608), ctx.RawArray("B", 8 * 1024 * 1024)) if mode in ("B", "C") else None
    p = ctx.Process(target=child, args=(mode, bufs, q)); p.start(); p.join(120)
    print(mode, "exit", p.exitcode, q.get(timeout=2) if p.exitcode == 0 else "-")
