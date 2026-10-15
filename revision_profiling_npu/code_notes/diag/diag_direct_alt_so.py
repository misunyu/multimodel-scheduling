"""Diagnostic: direct NeublaDriver use in one thread (vendor pattern, no server / IPC). Is every second output corrupted?"""
import sys; sys.dont_write_bytecode = True
if __name__ == "__main__":
    import importlib.util, numpy as np, cv2, onnxruntime as ort
    spec = importlib.util.spec_from_file_location("NeublaDriver", sys.argv[1])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    M = "/workspace/multimodel-scheduling/models/resnet50_small"
    front = ort.InferenceSession(f"{M}/partitions/resnet50_small_neubla_p0.onnx", providers=["CPUExecutionProvider"])
    img = cv2.imread("/workspace/resnet/cat_285.jpg")
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224)) / 255.0
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    data = front.run(None, {"input": x})[0].tobytes()
    for so_name, so in [(sys.argv[1], sys.argv[1])]:
        d = mod.NeublaDriver(); assert d.Init(0) == 0; assert d.LoadModel(f"{M}/npu_code/resnet50_small_neubla_p1.o") == 0
        outs = []
        for k in range(8):
            assert d.SendInput(data, 3 * 224 * 224) == 0; assert d.Launch() == 0
            o = np.frombuffer(d.ReceiveOutputs()[0], dtype=np.uint8).copy(); outs.append(o)
        print(so_name, "nonzero", [int((o != 0).sum()) for o in outs], "eq call0", [bool(np.array_equal(o, outs[0])) for o in outs])
        assert d.Close() == 0
    # vendor driver copy (parallel_model_partition_test) is the same file (sha 5a43); certification copy differs (717585dd)
