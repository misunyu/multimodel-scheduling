if __name__ == "__main__":
    import sys; sys.dont_write_bytecode = True
    sys.path.insert(0, "/workspace/mms_npu_port/revision_profiling/code")
    import queue, numpy as np
    from counters import InstanceLog
    from executor import load_inputs, load_registry, resolve_model
    from worker_npu import make_antara_worker
    reg = load_registry(); inputs, _ = load_inputs({"imagenet"})
    spec = resolve_model(reg, "antara_resnet50_small_npu")
    w = make_antara_worker(InstanceLog("x"), spec, "npu0", inputs, queue.Queue(2), queue.Queue(8))
    N = 32
    single = [w.postprocess(w.infer(w.preprocess(i)), i) for i in range(N)]
    double = []
    for i in range(N):
        x = w.preprocess(i); w.infer(x); double.append(w.postprocess(w.infer(x), i))
    w.client.close()
    print("single", single[:12])
    print("single==k", sum(single[i] == i for i in range(N)), " single==k-1", sum(single[i] == i - 1 for i in range(1, N)))
    print("double==k", sum(double[i] == i for i in range(N)))
