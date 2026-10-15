"""Language-model worker (spec v2.2 section 3).

One request = fixed prompt (P tokens) + K greedy decode steps. No KV cache: every step is a full forward
over P+k tokens. completed is recorded when the K-th token has been produced (and the output put succeeded).
  kind = plain          : inputs input_ids, attention_mask (gpt2.onnx)
  kind = empty_past_kv  : the graph requires past_key_values.*; they are fed with past length 0 on every step
"""
import sys

sys.dont_write_bytecode = True

import numpy as np

from worker_base import Worker


class LMWorker(Worker):
    def setup(self):
        lm = self.spec["lm"]
        self.kind = lm["kind"]
        self.P = int(lm["prompt_tokens"])
        self.K = int(lm["decode_steps"])
        names = [i.name for i in self.session.get_inputs()]
        assert "input_ids" in names and "attention_mask" in names, names
        self.logits_name = self.session.get_outputs()[0].name
        self.past = {}
        if self.kind == "empty_past_kv":
            shape = (1, int(lm["kv_heads"]), 0, int(lm["head_size"]))
            self.past = {n: np.zeros(shape, dtype=np.float32) for n in names if n.startswith("past_key_values.")}
            assert len(self.past) == 2 * int(lm["layers"]), len(self.past)
        else:
            assert names == ["input_ids", "attention_mask"], names
        for p in self.inputs:
            assert len(p) == self.P, (len(p), self.P)

    def preprocess(self, idx):
        return np.asarray(self.inputs[idx % len(self.inputs)], dtype=np.int64)[None, :]

    def infer(self, ids):
        new = []
        for _ in range(self.K):
            feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            feed.update(self.past)
            logits = self.session.run([self.logits_name], feed)[0]
            nxt = int(np.argmax(logits[0, -1]))
            new.append(nxt)
            ids = np.concatenate([ids, np.array([[nxt]], dtype=np.int64)], axis=1)
        return new

    def postprocess(self, y, idx):
        if len(y) != self.K:
            raise RuntimeError(f"generated {len(y)} tokens, expected {self.K}")
        return y
