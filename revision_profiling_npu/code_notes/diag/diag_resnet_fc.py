"""Diagnostic (not a measurement): ResNet50 NPU output = avgpool features (2048 uint8)? Host quantized FC as in
parallel_model_partition_test/npu.py lines 136-166 (vendor reference). Repeated-call stability. Read-only on all files."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/workspace/mms_npu_port/revision_profiling/code")
if __name__ == "__main__":
    import numpy as np, onnx
    from executor import load_inputs, load_registry, resolve_model
    from npu_server import NpuClient
    from worker_base import make_session
    from worker_cls import cls_preprocess
    REPO = "/workspace/mms_npu_port"
    reg = load_registry(); inputs, _ = load_inputs({"imagenet"}); imgs = inputs["imagenet"]
    host = onnx.load("/workspace/resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx")
    P = {i.name: onnx.numpy_helper.to_array(i) for i in host.graph.initializer}
    scale = P['/0/avgpool/GlobalAveragePool_output_0_scale'] * P['0.fc.weight_scale']
    zp_act = P['/0/avgpool/GlobalAveragePool_output_0_zero_point'].astype(np.int64); zp_w = P['0.fc.weight_zero_point'].astype(np.int64)
    scale_out = P['/0/fc/Gemm_output_0_scale']; zp_out = P['/0/fc/Gemm_output_0_zero_point']
    Wq = P['0.fc.weight_quantized'].T.astype(np.int64)
    print("fc weight", P['0.fc.weight_quantized'].shape, "bias?", [k for k in P if 'fc.bias' in k][:3])
    def fc(f):
        o = f.astype(np.int64) @ Wq - zp_act * Wq.sum(0) - zp_w * f.astype(np.int64).sum() + zp_act * zp_w * f.size
        return int(np.argmax(o * scale))
    for v in ("small", "big"):
        spec = resolve_model(reg, f"antara_resnet50_{v}_npu")
        front = make_session(REPO + "/" + spec["onnx"], "cpu")
        cl = NpuClient(0, str(__import__("pathlib").Path(REPO, spec["driver_so"]).resolve()), str(__import__("pathlib").Path(REPO, spec["npu_binary"]).resolve()))
        q = lambda i: np.ascontiguousarray(front.run(None, {"input": cls_preprocess(imgs[i])})[0])
        rep = [cl.run(q(5)) for _ in range(6)]
        print(v, "n_out", len(rep[0]), "len", [len(o) for o in rep[0]], "nonzero per call", [int((r[0] != 0).sum()) for r in rep],
              "same as call0", [bool(np.array_equal(r[0], rep[0][0])) for r in rep])
        top = []
        for i in range(32):
            f = cl.run(q(i))[0]
            top.append((fc(f[:2048]), int((f != 0).sum())))
        print(v, "hostFC top1 k==pred", sum(p == i for i, (p, _) in enumerate(top)), "/32; preds", [p for p, _ in top[:10]], "nonzero", [n for _, n in top[:10]])
        print(v, "close", cl.close())
