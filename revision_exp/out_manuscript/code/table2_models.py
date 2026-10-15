"""Table 2 data: the 8 models of the registry (revision_profiling/code/model_registry.yaml). Reads the ONNX files with the onnx
package (metadata only, no inference) and checks the registry sha256.

  .venv/bin/python3 -B revision_exp/out_manuscript/code/table2_models.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]

import collections  # noqa: E402
import hashlib  # noqa: E402

import onnx  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

OUT = REPO / "revision_exp" / "out_manuscript"
REG = yaml.safe_load((REPO / "revision_profiling" / "code" / "model_registry.yaml").read_text())["models"]
TASK = {"mnasnet": "image classification", "resnet50": "image classification", "resnext50": "image classification", "shufflenet-v2-12": "image classification",
        "squeezenet1.0-12": "image classification", "vgg19": "image classification", "yolov4": "object detection", "gpt2": "text generation (LM)"}
OLD = {"gpt2": "654 MB", "vgg19": "575 MB"}     # values quoted in the instruction from the current manuscript Table 2
DT = {1: "fp32", 10: "fp16", 3: "int8", 2: "uint8", 7: "int64", 11: "fp64", 16: "bf16"}

rows = []
for name, r in REG.items():
    p = REPO / r["onnx"]
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    m = onnx.load(str(p), load_external_data=False)
    init_types = collections.Counter(DT.get(t.data_type, str(t.data_type)) for t in m.graph.initializer)
    ext = [t.name for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    ops = collections.Counter(n.op_type for n in m.graph.node)
    quant = sum(ops[o] for o in ("QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"))
    inp = [(i.name, [d.dim_value if d.dim_value else (d.dim_param or "?") for d in i.type.tensor_type.shape.dim], DT.get(i.type.tensor_type.elem_type))
           for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    opset = ";".join(f"{o.domain or 'ai.onnx'}:{o.version}" for o in m.opset_import)
    weight_prec = max(init_types, key=init_types.get) if init_types else ""
    rows.append({"model": name, "onnx_file": r["onnx"], "sha256_matches_registry": sha == r["sha256"], "size_MB(1e6 B)": round(len(data) / 1e6, 1),
                 "size_MiB": round(len(data) / 2 ** 20, 1), "external_data_tensors": len(ext), "opset": opset,
                 "producer": f"{m.producer_name} {m.producer_version}".strip(), "graph_inputs": str(inp),
                 "registry_input_size": "x".join(map(str, r.get("input_size", []))) if r.get("input_size") else "", "registry_layout": r.get("layout", ""),
                 "preprocess": r.get("preprocess"), "initializer_dtypes": dict(init_types), "precision(weights)": weight_prec,
                 "quantization_ops": quant, "task": TASK[name], "request_unit": r.get("request_unit"),
                 "lm_generation": (f"prompt {r['lm']['prompt_tokens']} tokens, {r['lm']['decode_steps']} greedy tokens, kind={r['lm']['kind']}" if "lm" in r else ""),
                 "manuscript_old_value": OLD.get(name, "")})
d = pd.DataFrame(rows)
d.to_csv(OUT / "table2_models.csv", index=False)
print(d.drop(columns=["graph_inputs"]).to_string())
for r in rows:
    print(r["model"], r["graph_inputs"])
