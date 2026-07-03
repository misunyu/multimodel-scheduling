# rev23 — supplementary tables/curves (extraction only, 0 new measurements)

_All values extracted from rev20 (heavy-bg matrix) and rev22 (saturation sweep, both threads=4). No new measurement. mxq `b2441f9d`, global8, yolo11s. core scripts / paper unchanged. rev22 sanity (pre/post): NPU infer ~14 ms, skip 0% — threads=4 held._

## TASK 1 — N=2/4/8 saturation curves ✓ (already measured in rev22, extracted)

`results/rev23_sweep_byN.csv`. worst sAP vs measured GPU skip%, All-GPU vs All-NPU (flat), per N:

| N | All-NPU worst (flat) | crossover (GPU skip%) | note |
|---|---|---|---|
| 2 | 0.0836 | **~48%** | All-GPU 0.1149→0.0793 over GPU skip 0→55% |
| 4 | 0.0836 | **~48%** | All-GPU 0.1149→0.0663 over GPU skip 0→79% |
| 8 | 0.0817 | **already past at k=0** | 8 foreground streams alone drive GPU skip 64% → All-NPU wins the entire measured range (0.0507–0.0701 All-GPU) |

**Crossover is stable at ~48% GPU skip for N=2 and N=4**; at N=8 the foreground load alone exceeds the crossover, so the NPU wins throughout. This is the 3-curve evidence the text's "stable across N" claim needs. NPU skip ≤ 0.1% at N=2/4 (≤4.7% at N=8) — GPU-contention axis cleanly isolated.

## TASK 2 — per-point std / error bars ✓ (extracted)

`results/rev23_sweep_std.csv` (mean, std, n_reps=3 per point). **Max std across all sweep points = 0.0063**; most points ≤ 0.003. Error bars are far smaller than the All-GPU↔All-NPU separation and the crossover gap → the curves are well-resolved.

## TASK 3 — partial placements (supplementary) ✓ (extracted from rev20)

`results/rev23_partial_placements.csv` (3 co-tenants × N{2,4,8} × {Isolated, Cont-aware, All-GPU, All-NPU, Oracle}, worst+mean+std+device skips).

"Partial never wins outright" — best strategy per co-tenant at N=4:
| co-tenant | best (worst sAP) |
|---|---|
| L1_light | Cont-aware 0.0985 ⚠ |
| L2_lm | All-GPU 0.0618 |
| L3_vlm | All-NPU 0.0833 |

So at the robust heavy points the winner is always an **extreme** (All-GPU at low/medium GPU pressure, All-NPU at high). The only point where a partial (Cont-aware) edges ahead is **L1_light N=4** — which carries the rev19/rev20 GPU-skip reproducibility caveat (GPU sits at the 33 ms budget boundary; skip flips 0↔35% between runs). So the honest supplementary statement: **partial placements never win at any robust (heavy-contention) operating point; the lone partial "win" is at a fragile low-contention boundary and should not be headlined.**

## TASK 4 — per-stream sAP (C3, blinded-stream) ✓ (extracted from rev22)

`results/rev23_perstream.csv` (84 rows: All-GPU, All-NPU, mixed2 placements, N=4, per resnet_k). mixed2 = sid2,sid22→NPU; sid3,sid21→GPU:

| GPU load (resnet_k / GPU skip) | NPU sid2 | NPU sid22 | GPU sid3 | GPU sid21 |
|---|---|---|---|---|
| k=0 / 0% | 0.109 (sk0) | 0.191 (sk0) | 0.156 (sk0) | 0.115 (sk0) |
| k=3 / ~42% | 0.109 (sk0) | 0.191 (sk0) | 0.103 (sk42) | 0.083 (sk40) |
| k=8 / ~66% | 0.109 (sk0) | 0.191 (sk0) | 0.084 (sk65) | **0.065 (sk67)** |

- **NPU streams immune** (flat 0.109/0.191, skip 0% at all loads).
- **GPU streams degrade** monotonically with contention; the **worst stream is always a GPU-resident camera**, while the **mean is buoyed by the immune NPU streams** → the C3 "mean hides the blinded camera" effect, shown per-stream with the explicit sid→device mapping.
- All-GPU and All-NPU per-stream also captured (rev23_perstream.csv) for completeness.

## TASK 5 — bit-identical detection table (Table 2 anchor) ✓ (extracted from rev22)

`results/rev23_bitident_table.csv`: 20 frames, threads=4 vs threads=24:
- **identical = 100%** of frames, det counts equal.
- **max box-coord diff = 0.00e+00**, **max score diff = 0.00e+00**.

→ The thread lever changes only frame-skip, never detections. The quantization/staleness separation rests on this + the skip-0 (threads=4) vs skip-52% (threads=24) per-size contrast.

### ⚠ Correction carried forward (do NOT build an "offline mAP invariance" table)
offline mAP is **not** thread-invariant (threads=4 mAP_large 0.599 vs threads=24 0.529) because the implemented offline mAP penalizes skipped frames as zero recall. The causal-separation evidence is the **bit-identical per-processed-frame detections** + the **skip 0% vs 52% contrast**, not an mAP-invariance claim. This matches the current Table 2 framing.

## Summary

| TASK | status | output | new measurement? |
|---|---|---|---|
| 1 N=2/4/8 curves | ✓ extracted | rev23_sweep_byN.csv | no |
| 2 per-point std | ✓ extracted | rev23_sweep_std.csv | no |
| 3 partial placements | ✓ extracted | rev23_partial_placements.csv | no |
| 4 per-stream sAP | ✓ extracted | rev23_perstream.csv | no |
| 5 bit-identical | ✓ extracted | rev23_bitident_table.csv | no |

**Zero new measurements** — rev22 had already swept N=2/8 and captured per-stream + bit-identical; rev20 had the partial placements. Key supplementary facts now persisted: crossover ~48% GPU skip stable at N=2,4 (N=8 past it); sweep std ≤0.0063; partials never win at robust points; NPU streams contention-immune; detections bit-identical (diff 0).

## Files
- `results/rev23_sweep_byN.csv`, `rev23_sweep_std.csv`, `rev23_partial_placements.csv`, `rev23_perstream.csv`, `rev23_bitident_table.csv`, `rev23_report.md`
- core scripts / prior artifacts / paper unchanged.
