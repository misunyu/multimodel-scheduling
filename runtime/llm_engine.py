"""Unified LLM / VLM engine for GPU (transformers+CUDA), CPU and Mobilint NPU.

Handles the two generative models in the registry:
  - tinyllama (text-generation)
  - qwen2_vl  (image-text-to-text)

The same class runs on all three devices; only the checkpoint id and the
placement of host tensors differ. Throughput is reported as decode tokens/sec,
with prefill (time-to-first-token) measured separately during profiling.

NPU note: Mobilint models load with `trust_remote_code=True` and a specific
`revision`; the compiled work runs on the Aries chip while host tensors stay on
CPU. GPU/CPU use the original (non-quantized) HF checkpoints.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

import model_registry as reg

DEFAULT_PROMPT = "Describe what is happening in one short sentence."
LLM_PROMPTS = [
    "Summarize the benefits of edge AI accelerators in two sentences.",
    "Explain what object detection is to a beginner.",
    "List three common computer vision tasks.",
    "What is the difference between a CPU, a GPU and an NPU?",
]


def _pil_from_bgr(frame):
    from PIL import Image
    import cv2
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class LLMEngine:
    def __init__(self, model_name: str, device: str, max_new_tokens: int = 64):
        self.model_name = model_name
        self.device = reg.norm_device(device)
        self.spec = reg.get(model_name)
        self.kind = self.spec["kind"]  # llm | vlm
        self.max_new_tokens = int(max_new_tokens)
        self.load_ms = 0.0
        self._load()

    # ---- loading ----
    def _checkpoint(self):
        if self.device == "npu":
            cid = self.spec.get("npu_hf")
            local = self.spec.get("npu_local")
            if local and os.path.isdir(local):
                return local, self.spec.get("revision")
            return cid, self.spec.get("revision")
        return self.spec.get("gpu_hf"), None

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        cid, revision = self._checkpoint()
        host = "cuda" if self.device == "gpu" else "cpu"
        dtype = torch.float16 if self.device == "gpu" else torch.float32
        kw = dict(trust_remote_code=True)
        if revision:
            kw["revision"] = revision
        t0 = time.time()
        if self.kind == "vlm":
            from transformers import AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(cid, **kw)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self.model = AutoModelForImageTextToText.from_pretrained(cid, dtype=dtype, **kw)
        else:
            from transformers import AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(cid, **kw)
            self.processor = None
            self.model = AutoModelForCausalLM.from_pretrained(cid, dtype=dtype, **kw)
        if host == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.host = host
        self.load_ms = (time.time() - t0) * 1000.0

    # ---- input building ----
    def _build_inputs(self, prompt: str, frame=None):
        import torch
        if self.kind == "vlm":
            content = []
            if frame is not None:
                self._last_img = _pil_from_bgr(frame)
                content.append({"type": "image", "image": self._last_img})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
        else:
            messages = [{"role": "user", "content": prompt}]
            enc = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            # transformers >=5 returns a BatchEncoding; older returns a bare tensor.
            if hasattr(enc, "keys"):
                inputs = {k: enc[k] for k in enc.keys()}
            else:
                inputs = {"input_ids": enc}
        if self.host == "cuda":
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs

    def _generate(self, inputs, max_new_tokens):
        import torch
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self.host == "cuda":
            torch.cuda.synchronize()
        return out

    # ---- one inference (worker path) ----
    def infer(self, prompt: Optional[str] = None, frame=None,
              max_new_tokens: Optional[int] = None, measure_prefill: bool = False) -> dict:
        prompt = prompt if prompt is not None else (
            DEFAULT_PROMPT if self.kind == "vlm" else LLM_PROMPTS[0])
        n = int(max_new_tokens or self.max_new_tokens)
        inputs = self._build_inputs(prompt, frame=frame)
        n_in = int(inputs["input_ids"].shape[1])

        prefill_ms = 0.0
        if measure_prefill:
            t = time.time()
            _ = self._generate(inputs, 1)
            prefill_ms = (time.time() - t) * 1000.0

        t = time.time()
        out = self._generate(inputs, n)
        total_ms = (time.time() - t) * 1000.0
        n_out = int(out.shape[1]) - n_in
        n_out = max(n_out, 0)

        decode_s = (total_ms - prefill_ms) / 1000.0
        if measure_prefill and n_out > 1 and decode_s > 0.01:
            # decode-only rate (excludes time-to-first-token)
            tok_per_s = (n_out - 1) / decode_s
        else:
            # robust fallback: overall generation rate
            tok_per_s = n_out / max(total_ms / 1000.0, 1e-6)

        return {
            "n_in": n_in, "n_out": n_out,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "tokens_per_s": tok_per_s,
        }

    def profile(self, frame=None, max_new_tokens: Optional[int] = None, warmup: int = 1, iters: int = 3) -> dict:
        """Return averaged load/prefill/decode metrics for the static profile."""
        for _ in range(max(0, warmup)):
            self.infer(frame=frame, max_new_tokens=max_new_tokens, measure_prefill=False)
        prefills, tps, touts = [], [], []
        for _ in range(max(1, iters)):
            r = self.infer(frame=frame, max_new_tokens=max_new_tokens, measure_prefill=True)
            prefills.append(r["prefill_ms"]); tps.append(r["tokens_per_s"]); touts.append(r["n_out"])
        return {
            "load_ms": self.load_ms,
            "prefill_ms": float(np.mean(prefills)),
            "tokens_per_s": float(np.mean(tps)),
            "n_out": float(np.mean(touts)),
        }

    def dispose(self):
        try:
            import torch
            del self.model
            if self.host == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
