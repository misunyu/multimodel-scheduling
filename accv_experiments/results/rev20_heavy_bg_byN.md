# rev20 (Option B) — heavy-background reversal search (threads=4, fair, both devices)

_Both devices `torch.set_num_threads(4)`. Per-cell GPU skip% AND NPU skip% recorded. 3 reps/cell. Sanity (pre+post): NPU infer ~14 ms, skip 0% → threads=4 fast state confirmed throughout. mxq `b2441f9d`, global8, yolo11s. core scripts / paper unchanged._

## Full matrix — worst-stream sAP (mean of 3 reps), with device skips

| bg | N | All-GPU | Isolated | Cont-aware | All-NPU | Oracle | (All-GPU GPUskip / All-NPU NPUskip) |
|---|---|---|---|---|---|---|---|
| L1_light | 2 | **0.1149** | 0.0833 | 0.1090 | 0.0834 | 0.1149 | 0% / 0.5% |
| L1_light | 4 | 0.0915 | 0.0836 | **0.0985** | 0.0832 | 0.1086 | 35% / 1% |
| L1_light | 8 | 0.0570 | 0.0723 | 0.0529 | **0.0826** | — | 72% / 1% |
| L2_lm | 2 | **0.0751** | 0.0442 | 0.0751 | 0.0453 | 0.0763 | 70% / 73% |
| L2_lm | 4 | **0.0618** | 0.0413 | 0.0587 | 0.0422 | 0.0626 | 83% / 76% |
| L2_lm | 8 | 0.0441 | 0.0364 | 0.0369 | **0.0537** | — | 87% / 61% |
| L3_vlm | 2 | 0.0181 | 0.0209 | 0.0182 | **0.0829** | 0.0829 | 100% / 0.8% |
| L3_vlm | 4 | 0.0159 | 0.0239 | 0.0147 | **0.0833** | 0.0834 | 100% / 0.9% |
| L3_vlm | 8 | 0.0136 | 0.0177 | 0.0185 | **0.0826** | — | 99% / 2% |

`results/rev20_5strat_heavybg.csv`. std ≤ 0.004 everywhere (reproducible).

## H-gates at the hypothesis-critical heavy-bg N=4 cells

| cell | best strategy | H2 (Cont-aware > Isolated) | H3 (NPU-side > All-GPU) | H4 (reproducible) | all 4? |
|---|---|---|---|---|---|
| L1_light N=4 | Cont-aware (0.0985) | ✓ | ✓ | ✓ | **(H1 suspect — see below)** |
| L2_lm N=4 | All-GPU (0.0618) | ✓ | ✗ | ✓ | no (H3 fails) |
| L3_vlm N=4 | **All-NPU (0.0833)** | ✗ | ✓ (strong) | ✓ | no (H2 fails) |

**No cell passes all of H1–H4 as written** (H2 requires the *partial* Cont-aware > Isolated *and* H3 requires NPU-side advantage in the same cell — they never co-occur).

## What the data actually shows — the reversal is real, but it is All-NPU, and background-type-dependent

The decisive variable is **what the background competes for**:

All-NPU − All-GPU (worst sAP, N=4):
- L1_light (ResNet50, light GPU): **−0.008** — GPU wins (low contention).
- L2_lm (TinyLlama, touches CPU): **−0.020** — GPU wins, because the LM also slows the NPU's host postprocess (NPU skip 76%!), erasing the NPU advantage.
- L3_vlm (Qwen2-VL, heavy GPU): **+0.067** — **NPU wins 5×** (GPU 100% skipped, NPU 1% skip).

And at L3_vlm the reversal appears **even at N=2** (All-NPU 0.0829 vs All-GPU 0.0181, +0.065). It is not an N=8-only effect; under a GPU-competing co-tenant the GPU saturates immediately and any camera left on the GPU dies.

Why the *partial* Cont-aware story collapses at L3: with the GPU saturated, **any** stream left on the GPU is lost. Cont-aware (small-rich→NPU, large-rich→GPU) loses its large-rich GPU streams (worst 0.0147); Isolated (large-rich→NPU, small-rich→GPU) loses its small-rich GPU streams (0.0239). Only **All-NPU** keeps every stream off the saturated GPU (0.0833). The question "which camera to offload" degenerates to "offload all."

## ⚠️ Reproducibility caveat (GPU path)
L1_light N=4 All-GPU differs between rev19 (0.1149, GPU skip 0%) and rev20 (0.0915, GPU skip 35%) under nominally identical threads=4. The GPU-path skip is not fully reproducible at the N=4/L1_light boundary (GPU per-frame sits near the 33 ms budget, so small scheduling differences flip skip). This makes the L1_light N=4 "Cont-aware best" reading fragile (H1 suspect) — do **not** headline it. The L3_vlm result is robust (huge margin, std ≤0.0008, GPU saturation unambiguous).

## Final verdict

- **The partial contention-aware reversal (Cont-aware beats both extremes at deployable N) is NOT supported** at the clean operating point, at any background/N tested. Where the reversal is real (L3_vlm) Cont-aware is not the winner; where Cont-aware edges Isolated (L1/L2) it does not beat All-GPU or the margin is fragile.
- **The binary device-flip reversal IS strongly supported under a heavy GPU-competing background (L3_vlm):** All-NPU is 4–5× the worst-stream sAP of any GPU-using placement, at N=2/4/8, reproducibly. The isolated single-camera rule (keep small-rich on the GPU) is then **catastrophic** — those streams die on the saturated GPU.
- **The effect is background-type-dependent**, which is itself a finding: a co-located GPU workload (VLM) forces All-NPU; a CPU-touching workload (LM) slows the NPU's host postprocess and removes the advantage; a light GPU workload leaves the GPU best at low N.

## Recommendation (human decision)

Re-anchor the headline honestly on the **binary, deployment-driven reversal under heavy GPU contention**:

> "When a heterogeneous edge platform runs a GPU-resident co-tenant (e.g. a vision-language model), the detector's GPU path saturates and the isolated single-camera prescription — keep small-object-rich cameras on the GPU — becomes catastrophic; routing all cameras to the INT8 NPU recovers 4–5× the worst-stream sAP. The device decision a single-camera evaluation makes is exactly reversed under realistic co-tenancy."

Drop: the partial contention-aware (small-rich→NPU) as the deployable optimum; the N=4/L1_light headline; the marginal "near-oracle contention-aware" claim. Keep and strengthen: the two size-opposite mechanisms (rev17/18), the isolated-rule reversal (now as All-NPU under heavy GPU bg), the worst-stream-vs-mean argument. State the background-type dependence as a scoping condition, not hide it.

Alternatively, if the partial contention-aware story is essential, it is not recoverable from these measurements — it was an artifact of the mixed-thread state (rev19), and the clean operating point does not reproduce it.

## Files
- `results/rev20_5strat_heavybg.csv` (full matrix, per-device skips, 3 reps)
- `results/rev20_sanity.csv` (pre/post threads=4 single-stream sanity)
- `results/rev20_heavy_bg_byN.md` (this report)
- `accv_experiments/scripts/phase_rev20_heavybg.py`. core scripts / prior artifacts / paper unchanged.
