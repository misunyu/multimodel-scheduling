# EXP-AUDIT-THREADWHAT — what the thread lever parallelizes (read-only, understanding only)

Question: in Table 2's lever, what does `torch.set_num_threads(nt)` (nt=4/24) parallelize in the
NPU host post-processing? Static code reading only; NOT for paper integration.

## Q3 — scope of the `set_num_threads` calls
`phase_rev18_stage1.py`: `THREAD_SETTINGS=[24,4]` (:48); `torch.set_num_threads(nt)` is a
**process-global** setting applied at `:76` (before building models / measuring) and re-asserted at
`:86` (per rep) — it does NOT wrap a specific op. It governs PyTorch/ATen **intra-op** threading for
the whole process while `measure_single_stream` runs the NPU path:
`measure_single_stream → fg_worker(NPU) → npu_infer` (`_step_d_common.py:142-144`: `m.preprocess`,
`m(x)`, `m.postprocess`) → `MBLT_Engine.postprocess` (`wrapper.py:126` → `self.postprocessor`) →
`YOLOAnchorlessPost.__call__` (`base.py:89`: dequant/decode + NMS). So `nt` applies to the entire
NPU post-processing (dequant, decode, NMS), all of which are torch tensor ops.

## Q1/Q2 — intra-op only; no additional parallelism (confirmed)
The post-processing module (`vision/utils/postprocess/*.py`) contains **no explicit parallelism**:
- No `torchvision` import anywhere (grep: none) → NMS is a **custom torch implementation**
  (`yolo_anchorless_post.py:128 def nms`, a `for xi in x:` loop over batch images), not
  `torchvision.ops.nms`.
- No `ThreadPoolExecutor` / `multiprocessing` / `joblib` / `Pool(` / `num_workers` anywhere in the
  postprocess (grep: none).
- No per-frame / per-stream / batch distribution: this experiment is single-stream (N=1), batch=1;
  the only loops (`for xi in x` in decode/nms) iterate over the batch (size 1) sequentially.
→ **Q1 = intra-op only.** The only parallelism `nt` controls is ATen intra-op threading *inside*
each tensor op of the decode/NMS (e.g. sigmoid, split/cat, the DFL matmul, the NMS IoU/sort). **Q2 =
no additional parallelization present.**

## Q4 — tensor sizes (code fact) + "small tensor" rationale (documented, not profiled)
Decode operates on a per-frame head tensor of shape **(b, 144, 8400)** (`yolo_anchorless_post.py:71`;
`no = nc + reg_max*4 = 80 + 16*4 = 144`, `:26`), reshaped to **(8400, 84)** per image
(`:101,:112,:115`), with the DFL over **reg_max=16** bins (`:25-27`). NMS then runs on ≤8400 (conf-
filtered) candidates. These are small tensors. The rev18 header/report states the mechanism: with
24 ATen threads these tiny ops incur thread dispatch/sync overhead that dominates the parallel gain
(~36 ms), whereas 4 threads run them in ~10 ms — "thread thrashing on tiny YOLO11 head tensors."
The tensor shapes above are code-confirmed; the "overhead > parallel gain" explanation is the
documented rationale (rev18 comment/report), not a fresh profile (no re-profiling done).

## One-paragraph summary
At nt=24 the parallelized unit is **ATen intra-op threading within each torch op of the NPU
post-processing** (dequant/decode/NMS over a per-frame ~(8400×84) head tensor; custom torch NMS, no
torchvision, no thread/process pools, no per-frame/stream fan-out). It runs slower than nt=4 because
the head tensors are small, so splitting these tiny ops across 24 threads makes dispatch/sync
overhead dominate the compute — the documented "thread thrashing" effect. Detections are unchanged
(same math), only delivery latency.

## Compliance
Read-only static reading; no measurement/re-run/profiling; claims cited to code lines, rationale
marked as documented (not re-profiled); paper .tex untouched; not for paper integration. Output =
this file only.
