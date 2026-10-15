"""Diagnostic: per driver .so - output stability over 12 repeated calls and call latency for the 4 NPU binaries on npu0,
then npu0 + npu1 concurrently from two processes (ResNet small on npu0, YOLOv3 big on npu1). Random-but-fixed uint8 input."""
import sys; sys.dont_write_bytecode = True
import time

_MOD = {}
def load(so):
    if so in _MOD: return _MOD[so]
    import importlib.util
    spec = importlib.util.spec_from_file_location("NeublaDriver", so); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _MOD[so] = m; return m

def bench(so, npu, o, nbytes, n, q=None):
    import numpy as np
    mod = load(so); rng = np.random.default_rng(0); data = rng.integers(0, 255, nbytes, dtype=np.uint8).tobytes()
    d = mod.NeublaDriver(); assert d.Init(npu) == 0; assert d.LoadModel(o) == 0
    outs, ms = [], []
    for _ in range(n):
        t = time.perf_counter(); assert d.SendInput(data, nbytes) == 0; assert d.Launch() == 0; r = d.ReceiveOutputs(); ms.append((time.perf_counter() - t) * 1000)
        outs.append(b"".join(r))
    assert d.Close() == 0
    res = {"lens": [len(x) for x in r], "stable": sum(x == outs[0] for x in outs), "n": n, "ms_p50": float(np.median(ms[2:]))}
    if q is not None: q.put((npu, res))
    return res

if __name__ == "__main__":
    import multiprocessing as mp
    so = sys.argv[1]; M = "/workspace/multimodel-scheduling/models"
    O = {f"{m}_{v}": (f"{M}/{m}_{v}/npu_code/{m}_{v}_neubla_p1.o", 3*224*224 if m == "resnet50" else 3*608*608) for m in ("resnet50", "yolov3") for v in ("big", "small")}
    ctx = mp.get_context("spawn")
    for k, (o, nb) in O.items():
        q1 = ctx.Queue(); p = ctx.Process(target=bench, args=(so, 0, o, nb, 12, q1)); p.start(); p.join(90)
        if p.is_alive(): p.kill(); print(so.split("/")[2], k, "npu0 HANG", flush=True); continue
        print(so.split("/")[2], k, "npu0", q1.get(timeout=5)[1] if p.exitcode == 0 else f"exit {p.exitcode}", flush=True)
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=bench, args=(so, 0, *O["resnet50_small"], 40, q)), ctx.Process(target=bench, args=(so, 1, *O["yolov3_big"], 20, q))]
    t = time.perf_counter(); [p.start() for p in ps]; [p.join(120) for p in ps]
    print(so.split("/")[2], "concurrent", [q.get(timeout=5) for _ in ps], "exit", [p.exitcode for p in ps], "%.1fs" % (time.perf_counter() - t), flush=True)
