"""A2 additional co-tenants (kept separate from the frozen harness).

GPU co-tenants for Task 2 identity-invariance (alternative to ResNet50, chosen
for DIFFERENT compute profiles):
  - vit_b_16  : Vision Transformer, attention/matmul-bound (torch-CUDA, random
                weights — a co-tenant only needs to load the GPU, not be trained).
  - vgg19     : conv + very large FC layers, memory-bandwidth bound (ORT-CUDA,
                existing models/onnx/vgg19.onnx).
Both differ structurally from ResNet50's residual bottleneck convolutions.

CPU-only anchor for Task 1 (realistic, non-synthetic CPU pressure):
  - tinyllama_cpu : the L2LM language component run on CPU via ORT CPUEP only,
                    i.e. the host-CPU stress of L2LM without any GPU footprint.

Each preload_* returns nothing but populates _A2_PRELOADED; each _bg_*_loop is a
worker(stop_event) matching the harness's start_bg_custom contract.
"""
import numpy as np
import onnxruntime as ort
import torch

_A2_PRELOADED = {}


# ---------------- GPU co-tenants ----------------

def preload_vit(batch=1):
    if "vit" in _A2_PRELOADED:
        return
    from torchvision import models
    m = models.vit_b_16(weights=None).to("cuda").eval()
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    with torch.no_grad():
        m(x)
    _A2_PRELOADED["vit"] = (m, x)
    print(f"[a2-preload] ViT-B/16 torch-CUDA ready (batch={batch})", flush=True)


def _bg_vit_loop(stop_event):
    m, x = _A2_PRELOADED["vit"]
    with torch.no_grad():
        while not stop_event.is_set():
            m(x)


def preload_vgg19(batch=1):
    if "vgg19" in _A2_PRELOADED:
        return
    sess = ort.InferenceSession("models/onnx/vgg19.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    feed = {name: np.random.randn(batch, 3, 224, 224).astype(np.float32)}
    sess.run(None, feed)
    _A2_PRELOADED["vgg19"] = (sess, feed)
    print(f"[a2-preload] VGG19 ORT-CUDA ready (batch={batch})", flush=True)


def _bg_vgg19_loop(stop_event):
    sess, feed = _A2_PRELOADED["vgg19"]
    while not stop_event.is_set():
        sess.run(None, feed)


# ---------------- CPU-only anchor ----------------

def preload_tinyllama_cpu(intra_threads=4):
    if "tinyllama_cpu" in _A2_PRELOADED:
        return
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    sess = ort.InferenceSession("models/onnx/tiny-llama-chat-onnx/model.onnx",
                                sess_options=so, providers=["CPUExecutionProvider"])
    seq_len = 32
    feeds = {
        "input_ids": np.random.randint(1, 30000, size=(1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values."):
            feeds[inp.name] = np.zeros((1, 4, 0, 64), dtype=np.float32)
    sess.run(None, feeds)
    _A2_PRELOADED["tinyllama_cpu"] = (sess, feeds)
    print(f"[a2-preload] TinyLLaMA ORT-CPU ready (intra={intra_threads})", flush=True)


def _bg_tinyllama_cpu_loop(stop_event):
    sess, feeds = _A2_PRELOADED["tinyllama_cpu"]
    while not stop_event.is_set():
        sess.run(None, feeds)
