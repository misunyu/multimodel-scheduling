"""Diagnostic: step-by-step prints; sequence given on the command line, each item = <model>:<npu> in its own process."""
import sys; sys.dont_write_bytecode = True
SO = sys.argv[1]
M = "/workspace/multimodel-scheduling/models"

def child(model, npu):
    import importlib.util, numpy as np
    spec = importlib.util.spec_from_file_location("NeublaDriver", SO); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    o = f"{M}/{model}/npu_code/{model}_neubla_p1.o"; nb = 3 * 224 * 224 if model.startswith("resnet") else 3 * 608 * 608
    d = m.NeublaDriver(); print(model, npu, "Init", d.Init(npu), flush=True)
    print(model, npu, "LoadModel", d.LoadModel(o), flush=True)
    x = np.zeros(nb, np.uint8).tobytes()
    for k in range(3):
        d.SendInput(x, nb); d.Launch(); r = d.ReceiveOutputs()
    print(model, npu, "run ok", [len(t) for t in r], flush=True)
    print(model, npu, "Close", d.Close(), flush=True)

if __name__ == "__main__":
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    for item in sys.argv[2:]:
        model, npu = item.split(":")
        p = ctx.Process(target=child, args=(model, int(npu))); p.start(); p.join(120)
        print("->", item, "exit", p.exitcode, flush=True)
