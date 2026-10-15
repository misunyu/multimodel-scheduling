"""Phase N1 section 2-3: registry entries for the CPU-NPU (Antara) board. Read-only on every model file.

Writes
  revision_profiling_npu/registry/npu_registry_block.yaml  entries to append to code/model_registry.yaml (antara_* keys)
  revision_profiling_npu/registry/yolov3_dequant.json      DequantizeLinear scale / zero point of the three YOLOv3 NPU
                                                           outputs, read from the host quantized ONNX (initializers only)
  revision_profiling_npu/registry/files_sha256.json        sha256 of every registered file (pre/post run comparison)

The old runtime extracted front/back graphs at run time and wrote <host onnx>.front/.back next to the host model
(npu.py: yolo_prepare_onnx_model / resnet50_prepare_onnx_model). Here nothing is written outside the repository:
front (p0) and back (p2) partitions are the pre-split files under models/<variant>/partitions, and the three dequant
parameters are taken once from the host ONNX into a JSON file that the worker reads.

  python3 -B code/build_npu_registry.py
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent.parent
OUT = REPO / "revision_profiling_npu" / "registry"
ASSETS = "../multimodel-scheduling"                       # NPU board: models/ and the driver live in the main checkout (untracked)
DRIVER_SO = None                                          # set from --driver-so (decision recorded in REPORT_N1.md)
HOST_RESNET = "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
HOST_YOLO = "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
YOLO_DEQ = ["onnx::Transpose_684_DequantizeLinear", "onnx::Transpose_688_DequantizeLinear", "onnx::Transpose_692_DequantizeLinear"]
YOLO_BACK_IN = {"onnx::Transpose_684": [1, 255, 19, 19], "onnx::Transpose_688": [1, 255, 38, 38], "onnx::Transpose_692": [1, 255, 76, 76]}


def sha(rel):
    h = hashlib.sha256()
    with open(REPO / rel, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def resnet_fc_params(onnx, np):
    """Quantized FC (2048 -> 1000) run on the host after the NPU, as in the vendor reference
    (parallel_model_partition_test/npu.py lines 136-166): the NPU binary p1.o outputs the 2048 uint8 avgpool features,
    not the 1000 FC outputs that models/*/partitions/*_p2.onnx expects (measured: one 2048-byte output buffer)."""
    mdl = onnx.load(str(REPO / HOST_RESNET))
    P = {i.name: onnx.numpy_helper.to_array(i) for i in mdl.graph.initializer}
    keys = ["/0/avgpool/GlobalAveragePool_output_0_scale", "/0/avgpool/GlobalAveragePool_output_0_zero_point",
            "0.fc.weight_quantized", "0.fc.weight_scale", "0.fc.weight_zero_point", "0.fc.bias_quantized",
            "/0/fc/Gemm_output_0_scale", "/0/fc/Gemm_output_0_zero_point"]
    extra = [k for k in P if k.startswith("0.fc.bias") and k not in keys]
    out = {k.replace("/", "_").strip("_").replace(".", "_"): np.asarray(P[k]) for k in keys + extra}
    info = {k: [str(P[k].dtype), list(P[k].shape)] for k in keys + extra}
    return out, info


def main():
    import argparse
    import numpy as np
    import onnx
    global DRIVER_SO
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver-so", required=True, help="path relative to the repository root")
    DRIVER_SO = ap.parse_args().driver_so
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}

    def reg(rel):
        files[rel] = sha(rel)
        return files[rel]

    # CPU ONNX: big and small are the same file (checked here); one entry per model
    cpu = {}
    for m in ("resnet50", "yolov3"):
        a, b = f"{ASSETS}/models/{m}_big/model/{m}_big.onnx", f"{ASSETS}/models/{m}_small/model/{m}_small.onnx"
        assert reg(a) == reg(b), f"CPU ONNX of {m} big/small differ"
        cpu[m] = a
    # NPU binaries must differ between variants
    for m in ("resnet50", "yolov3"):
        sb = reg(f"{ASSETS}/models/{m}_big/npu_code/{m}_big_neubla_p1.o")
        ss = reg(f"{ASSETS}/models/{m}_small/npu_code/{m}_small_neubla_p1.o")
        assert sb != ss, f"NPU binaries of {m} big/small are identical"
    reg(DRIVER_SO)

    fc, fc_info = resnet_fc_params(onnx, np)
    fc_rel = "revision_profiling_npu/registry/resnet50_fc_params.npz"
    np.savez(str(REPO / fc_rel), **fc)
    (OUT / "resnet50_fc_params.info.json").write_text(json.dumps({"source": HOST_RESNET, "source_sha256": sha(HOST_RESNET),
                                                                  "arrays": fc_info, "npz_keys": sorted(fc)}, indent=1))
    reg(fc_rel)

    # YOLOv3 dequant parameters (initializers of the three DequantizeLinear nodes of the host quantized model)
    host_sha = sha(HOST_YOLO)
    mdl = onnx.load(str(REPO / HOST_YOLO))
    init = {i.name: onnx.numpy_helper.to_array(i) for i in mdl.graph.initializer}
    deq = {}
    for node in mdl.graph.node:
        if node.name in YOLO_DEQ:
            s, z = init[node.input[1]], init[node.input[2]]
            assert s.size == 1 and z.size == 1, (node.name, s.shape, z.shape)
            deq[node.name] = {"scale": float(s.reshape(-1)[0]), "zero_point": int(z.reshape(-1)[0]), "zero_point_dtype": str(z.dtype),
                              "scale_input": node.input[1], "zp_input": node.input[2]}
    assert list(deq) == YOLO_DEQ or set(deq) == set(YOLO_DEQ), deq
    deq_doc = {"source": HOST_YOLO, "source_sha256": host_sha,
               "order": YOLO_DEQ, "back_inputs": YOLO_BACK_IN, "params": {k: deq[k] for k in YOLO_DEQ},
               "note": "NPU output buffer k (k = 0, 1, 2) is dequantized with params[order[k]] and fed to back input k "
                       "(order of model_processors.py run_yolo_npu_process in the main checkout)."}
    deq_rel = "revision_profiling_npu/registry/yolov3_dequant.json"
    (REPO / deq_rel).write_text(json.dumps(deq_doc, indent=1))
    reg(deq_rel)

    entries = {}
    entries["antara_resnet50_cpu"] = {
        "onnx": cpu["resnet50"], "sha256": files[cpu["resnet50"]], "input_size": [224, 224], "layout": "NCHW",
        "preprocess": "cls_resize256_crop224_imagenet_norm", "postprocess": "cls_argmax", "input_source": "imagenet",
        "request_unit": "image", "logical_model": "resnet50", "variant": "common", "device_kind": "cpu"}
    entries["antara_yolov3_cpu"] = {
        "onnx": cpu["yolov3"], "sha256": files[cpu["yolov3"]], "input_size": [608, 608], "layout": "NCHW",
        "preprocess": "yolo_letterbox_div255", "postprocess": "yolov3_keras_nms_count", "input_source": "video",
        "request_unit": "image", "logical_model": "yolov3", "variant": "common", "device_kind": "cpu"}
    for m in ("resnet50", "yolov3"):
        for v in ("big", "small"):
            base = f"{ASSETS}/models/{m}_{v}"
            p0, p1, p2 = f"{base}/partitions/{m}_{v}_neubla_p0.onnx", f"{base}/npu_code/{m}_{v}_neubla_p1.o", f"{base}/partitions/{m}_{v}_neubla_p2.onnx"
            for r in (p0, p2):
                reg(r)
            e = {"onnx": p0, "sha256": files[p0],
                 "extra_files": {p1: files[p1], p2: files[p2], DRIVER_SO: files[DRIVER_SO]},
                 "npu_binary": p1, "back_onnx": p2, "driver_so": DRIVER_SO,
                 "logical_model": m, "variant": v, "device_kind": "npu", "layout": "NCHW", "npu_input_dtype": "uint8",
                 "input_source": "imagenet" if m == "resnet50" else "video", "request_unit": "image"}
            if m == "resnet50":
                e.update({"input_size": [224, 224], "preprocess": "cls_resize256_crop224_imagenet_norm",
                          "postprocess": "npu_resnet50_hostfc_argmax", "npu_input_bytes": 3 * 224 * 224,
                          "npu_output_bytes": 2048, "host_fc_params": fc_rel,
                          "back_onnx_note": "p2 (ArgMax over 1000 uint8 FC outputs) does not match p1.o's 2048-byte output; "
                                            "registered for provenance, not executed"})
                e["extra_files"][fc_rel] = files[fc_rel]
            else:
                e.update({"input_size": [608, 608], "preprocess": "yolo_letterbox_div255", "postprocess": "npu_yolov3_nms_count",
                          "npu_input_bytes": 3 * 608 * 608, "dequant_json": deq_rel})
                e["extra_files"][deq_rel] = files[deq_rel]
            entries[f"antara_{m}_{v}_npu"] = e
    block = "# --- Phase N1 (CPU-NPU board, Antara): antara_* entries generated by code/build_npu_registry.py ---\n"
    block += yaml.safe_dump(entries, sort_keys=False, default_flow_style=None, width=200)
    (OUT / "npu_registry_block.yaml").write_text(block)
    (OUT / "files_sha256.json").write_text(json.dumps({"files": files, "host_onnx_read_only": {HOST_YOLO: host_sha, HOST_RESNET: sha(HOST_RESNET)}}, indent=1))
    print(json.dumps({k: {"onnx": v["onnx"], "variant": v["variant"]} for k, v in entries.items()}, indent=1))
    print("dequant", {k: (v["scale"], v["zero_point"]) for k, v in deq.items()})


if __name__ == "__main__":
    main()
