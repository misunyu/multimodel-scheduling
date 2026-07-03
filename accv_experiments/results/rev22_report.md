# rev22 — VLM/GPU saturation sweep: gradual reversal threshold (+ per-stream + bit-identical)

_Both devices `torch.set_num_threads(4)`. Contention lever = number of ResNet50 GPU-loops (0–8), registered at runtime into `BG_VARIANTS` (dict mutation, no core file edit). ResNet50 is GPU-bound → isolates GPU contention, NPU host postprocess stays fast. x-axis = **measured GPU skip%**. 3 reps/point. mxq `b2441f9d`, global8, yolo11s. core scripts / paper unchanged._

## Safety gates (all pass)
- **Sanity pre/post**: NPU infer 14.1 / 14.0 ms, skip 0.0% → threads=4 fast state held throughout (large_gap −0.022 is the 5-sid subset signature, not 24-log). ✓
- **NPU stays clean across the whole sweep**: NPU skip% ≤ 0.1% at N=2,4 (≤4.7% only at N=8 high load) → the ResNet50 lever loads the GPU, **not** the NPU host postprocess. The GPU-contention axis is cleanly isolated. ✓
- **Bit-identical (PART C)**: 20 frames, threads=4 vs threads=24, **all identical**, max box diff `0.00e+00`, max score diff `0.00e+00`, det counts equal. Persisted to `rev22_bitident.csv`. ✓

## PART A — saturation sweep (N=4, worst-stream sAP)

| ResNet k | GPU skip% | NPU skip% | All-GPU worst | **All-NPU worst** | mixed2 worst |
|---|---|---|---|---|---|
| 0 | 0.1 | 0.0 | 0.1149 | 0.0836 | 0.1090 |
| 1 | 42.0 | 0.0 | 0.0879 | 0.0836 | 0.1057 |
| 2 | 57.9 | 0.0 | 0.0765 | 0.0836 | 0.0908 |
| 3 | 67.3 | 0.0 | 0.0739 | 0.0836 | 0.0827 |
| 4 | 71.5 | 0.1 | 0.0719 | 0.0836 | 0.0754 |
| 6 | 73.5 | 0.0 | 0.0700 | 0.0836 | 0.0712 |
| 8 | 79.2 | 0.0 | 0.0663 | 0.0836 | 0.0647 |

`results/rev22_vlm_sweep.csv` (std ≤ 0.003).

- **All-NPU worst is flat at 0.0836** across the entire contention range (NPU skip 0%, immune to GPU load).
- **All-GPU worst declines monotonically** 0.1149 → 0.0663 as GPU skip rises 0 → 79%.
- **mixed2 (2 NPU + 2 GPU)** sits between and declines with the GPU streams.

### ★ Crossover ≈ 48% GPU skip (NOT 100%)
All-GPU worst crosses All-NPU's flat 0.0836 between k=1 (GPU skip 42%, 0.0879) and k=2 (GPU skip 58%, 0.0765). Linear interpolation: **crossover at ~48% GPU skip** — the same value at N=2 and N=4:

| N | crossover (GPU skip%) |
|---|---|
| N=2 | ~48% |
| N=4 | ~48% |
| N=8 | already past at k=0 (8 foreground streams alone drive GPU skip 64% → All-NPU wins from the start) |

**This defeats the "trivial" reading.** The reversal is not "the GPU dies at 100% and then the NPU is obviously better." It is a **gradual crossing at moderate contention (~half the frames missing the budget)**, reproducible and N-robust. Past ~48% GPU skip, routing all cameras to the INT8 NPU gives the better worst-stream sAP; below it, the GPU is better.

## PART B — per-stream sAP (mixed2, N=4): which camera is blinded

`results/rev22_perstream_sap.csv`. mixed2 = sid2,sid22 → NPU; sid3,sid21 → GPU:

| ResNet k | GPU skip | NPU sid2 | NPU sid22 | GPU sid3 | GPU sid21 |
|---|---|---|---|---|---|
| 0 | 0% | 0.109 (sk0) | 0.191 (sk0) | 0.156 (sk0) | 0.115 (sk0) |
| 3 | ~42% | 0.109 (sk0) | 0.191 (sk0) | 0.103 (sk42) | 0.083 (sk40) |
| 8 | ~66% | 0.109 (sk0) | 0.191 (sk0) | 0.084 (sk65) | **0.065 (sk67)** |

- **NPU streams (sid2, sid22) are immune**: sAP flat at 0.109 / 0.191, skip 0%, at every contention level.
- **GPU streams (sid3, sid21) degrade monotonically** with contention (sid21: 0.115 → 0.065 as its skip 0 → 67%).
- The **worst stream is always a GPU-resident camera once contended**, while the **mean is buoyed by the immune NPU streams** — exactly the C3 effect: averaging hides the blinded GPU camera. (E.g. at k=8, worst 0.0647 vs the NPU streams' 0.109–0.191.)

## PART C — bit-identical detections (Table 2 causal claim, persisted)
`results/rev22_bitident.csv`: 20 frames, threads=4 vs 24 → **100% identical**, max box-coord diff `0.00e+00`, max score diff `0.00e+00`. The thread lever changes only speed (frame-skip), never the detections. This anchors the causal-separation claim: the size-stratified quantization signal (threads=4, skip 0: small −48%, large ≈0) and the staleness term (threads=24, skip 52%: large −0.096) come from the *same detections*, differing only in how many frames meet the streaming budget.

## Summary verdict
- **Gradual, non-trivial reversal threshold established**: worst-stream device preference flips from GPU to All-NPU at **~48% GPU skip** (≈ moderate contention), reproducibly, robust across N=2/4 and already passed at N=8. This is the single strongest evidence against the "trivial at full saturation" objection.
- **Mechanism shown per-stream**: NPU streams are contention-immune (flat); GPU streams degrade with load; the worst is a blinded GPU camera the mean hides.
- **Detections bit-identical** across the thread lever → the quantization-vs-staleness separation is a property of the streaming clock, not of changed outputs.

### For the paper (results in hand; edits by you)
- **Sweep figure**: x = measured GPU skip% (0→80%), y = worst sAP; All-GPU declining curve, All-NPU flat line at 0.0836, crossover marked at ~48%. Caption: reversal occurs at moderate contention, not only at saturation.
- **Table 3 / C3**: per-stream decomposition (PART B) + worst-vs-mean gap as the "mean hides the blinded camera" quantification.
- **Table 2 / mechanism**: cite bit-identical (diff 0) + the skip=0 vs skip=52% per-size contrast for the causal quantization/staleness split.

## Files
- `results/rev22_vlm_sweep.csv`, `results/rev22_perstream_sap.csv`, `results/rev22_bitident.csv`, `results/rev22_sanity.csv`, `results/rev22_report.md`
- `accv_experiments/scripts/phase_rev22_sweep.py`. core scripts / prior artifacts / paper unchanged.
