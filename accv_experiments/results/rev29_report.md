# rev29 — Condition-dependent optimal placement (All-GPU / split / All-NPU / Oracle)

**Question.** Line 389 claims mixed/split placement never beats the best homogeneous placement at
*robust* operating points (the only Table-3 cell where split won was the skip-unstable, 33ms-boundary
cell that was excluded). On a **stable** ResNet50-pressure sweep, does the optimal placement become
**condition-dependent** (low→All-GPU, moderate→split, saturation→All-NPU)?

**Verdict: 3-regime CONFIRMED, with honest nuance.** There exists a **stable** moderate-contention
operating point (GPU frame-skip ≈ 38%, RES1) where **split (1 large-object stream on NPU) strictly
beats BOTH homogeneous placements in all 3 reps**. The strong form of the line-389 claim is therefore
**refuted** — but the strictly-wins band is **narrow** (essentially one level), and at higher
contention split *converges to* (ties, does not beat) All-NPU. Isolated single-stream evaluation
prescribes All-GPU everywhere and thus mis-ranks at both moderate and high contention.

> Framing: Oracle / split are a **diagnostic upper bound** (which placement is best given the
> condition), NOT a scheduler proposal. The message is "the optimal placement is condition-dependent
> and isolated evaluation cannot see it" (reinforces C1), not "we propose a better placement policy".

---

## Protocol (identical to existing sweeps; solo run)
- Argoverse-HD forward (24 val logs), **YOLOv11s**, streaming sAP, both devices **threads=4**,
  NPU = MLA100 **global8** (matches rev27 N=4 loading). N=4 fixed.
- Contention lever = **ResNet50 GPU-pressure** (stable; ORT-CUDA, GPU-bound) at k=0/1/2/4/8, plus
  one **VLM-saturation** point (L3_vlm) for contrast. 3 reps/cell, GPU+NPU skip% recorded per cell.
- Split = ratio r∈{0..4} = #streams on NPU; streams moved to NPU in a **pre-declared,
  mechanism-motivated order: large-object streams first** ([3,21] then [22,2]) — staleness is
  large-biased, so large streams are worst-hit on GPU and cheapest to quantize on NPU.
- **Sanity (pre/post): NPU infer 10.2/10.1 ms, NPU skip 0.0%, large_gap −0.0223 (stable).**
  RES0 All-GPU worst 0.1148 @ gpu_skip 0.2% reproduces rev22 v11s (0.1149) → comparable.

## Worst-stream sAP by contention level and split ratio
| level | gpu_skip@AllGPU | r0 All-GPU | r1 split | r2 split | r3 split | r4 All-NPU | **Oracle** | best |
|---|---|---|---|---|---|---|---|---|
| RES0 | 0.2%  | **0.1148** | 0.1149 | 0.0835 | 0.0835 | 0.0835 | 0.1149@r1\* | **All-GPU** |
| RES1 | 37.8% | 0.0887 | **0.0910** | 0.0836 | 0.0836 | 0.0834 | **0.0910@r1** | **split (r1)** |
| RES2 | 51.1% | 0.0813 | 0.0795 | 0.0836 | 0.0835 | 0.0834 | 0.0836@r2 | All-NPU (split2 ties) |
| RES4 | 59.4% | 0.0766 | 0.0726 | 0.0836 | 0.0836 | 0.0835 | 0.0836@r2 | All-NPU (split2 ties) |
| RES8 | 71.8% | 0.0694 | 0.0665 | 0.0835 | 0.0835 | 0.0836 | 0.0836@r4 | **All-NPU** |
| VLM  | 96.8% | 0.0187 | 0.0223 | 0.0244 | 0.0219 | 0.0832 | 0.0832@r4 | **All-NPU** |

\* RES0 Oracle@r1 (0.1149) exceeds All-GPU (0.1148) by 0.0001 — **statistical tie / noise**; the
honest reading at idle is **All-GPU optimal**.

## The split win is real and stable (RES1, paired across 3 reps)
| rep | All-GPU (r0) | split1 (r1) | All-NPU (r4) | r1−r0 | r1−r4 |
|---|---|---|---|---|---|
| 0 | 0.0870 | 0.0888 | 0.0835 | +0.0018 | +0.0053 |
| 1 | 0.0880 | 0.0922 | 0.0831 | +0.0042 | +0.0091 |
| 2 | 0.0912 | 0.0920 | 0.0836 | +0.0008 | +0.0084 |
| mean | 0.0887 | **0.0910** | 0.0834 | **+0.0023 (3/3>0)** | **+0.0076 (3/3>0)** |

- split1 > All-GPU in **every** rep (+1.0…+4.8%, mean +2.6%) and > All-NPU in every rep (mean +9%).
- **Stable operating point** (NOT the 33ms boundary): split1 GPU-skip ≈ 18–20%, worst_std ≈ 0.0015;
  All-GPU GPU-skip ≈ 35–40%. No skip thrashing. The split superiority does **not** rely on the
  excluded skip-unstable cell.
- Mechanism: moving the worst-hit (large-object) stream to NPU removes it from large-biased GPU
  staleness AND drops the remaining GPU streams' skip from ~38% → ~19% (3 streams share the GPU),
  while the moved stream pays only the tiny large-object quantization loss on NPU.

## 3-regime boundaries (GPU frame-skip on All-GPU)
- **Low (skip ≲ ~30%) → All-GPU optimal** (RES0). NPU-side far worse (0.084 vs 0.115).
- **Moderate (skip ≈ 38%) → split strictly optimal** (RES1): split1 > All-GPU > All-NPU.
- **High / saturation (skip ≳ ~50%) → All-NPU optimal** (RES2–VLM): All-GPU collapses; split2/3
  converge to (tie, do not beat) All-NPU; under VLM saturation any GPU-resident stream is crushed
  (skip 97–100%) so All-NPU dominates by ~3.5×.
- Homogeneous crossover (All-GPU=All-NPU) sits at skip ≈ 45–50% (between RES1 and RES2); the split
  window sits **just before** it, exactly where the mechanism predicts.

## Honest nuances / caveats
1. **Narrow window.** Split *strictly* beats both only at RES1 (one stable level). Margin over
   All-GPU is modest (+0.0023; 3/3 reps, ~1.7×SEM) though over All-NPU it is clear (+0.0076).
2. **High-contention split = All-NPU, not better.** At RES2+ the Oracle's "split2" merely ties the
   NPU floor (0.0836 ≈ All-NPU 0.0834) — the worst stream is an NPU stream; split adds nothing
   beyond going All-NPU. Do not claim split beats All-NPU there.
3. **Oracle = upper bound over the 5 measured (large-first) ratios**, not all 16 placements; it is a
   diagnostic, not a scheduler. Large-first ordering is conservative (pre-declared by mechanism).
4. **Comparability.** rev29 uses global8 NPU engines (per directive / rev27); internally consistent.
   RES0 reproduces rev22 v11s All-GPU. Sanity pre=post (no drift).

## Implication for the paper (claim correction)
- Line 389 strong form ("mixed never beats best homogeneous at robust points") is **refuted** by a
  stable counterexample (RES1). Suggested reframing: *"the optimal placement is condition-dependent —
  low contention favors All-GPU, a moderate stable band (GPU-skip ≈ 38%) favors a 1-stream split that
  beats both homogeneous placements, and saturation favors All-NPU; isolated single-stream evaluation
  prescribes All-GPU throughout and therefore mis-ranks under both moderate and saturated co-tenancy."*
- This is a **diagnostic** strengthening of C1 (evaluation failure), not a scheduler proposal, and it
  pre-empts the strawman/load-balancing objection by showing the excluded split cell was right on the
  mechanism even at a stable point.

## Artifacts
- `rev29_split_sweep.csv`, `rev29_oracle_by_contention.csv`, `rev29_sanity.csv`,
  `rev29_stdout.log` (per-rep), `manifest_rev29.json`.
- Script: `scripts/phase_rev29_split.py` (core untouched; existing results / paper unchanged).
