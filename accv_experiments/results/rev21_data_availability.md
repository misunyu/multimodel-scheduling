# rev21 — data-availability audit: mean sAP & bit-identical evidence (extraction only)

_No measurement. Pure lookup of existing artifacts (rev18/19/20). threads=4 operating point. core scripts / prior artifacts / paper unchanged._

## PART A — mean sAP availability ✓ AVAILABLE

`rev20_5strat_heavybg.csv` schema: `rep, bg, N, strategy, placement, worst_sap, **mean_sap**, gpu_skip, npu_skip`.
→ **mean_sap is present** for every (bg, N, strategy) cell, 3 reps. No re-measurement needed.
(`rev19_main_comparison_threads4.csv` also has `mean_sap`.)

Extracted → `results/rev21_mean_worst_extract.csv`. N=4 (threads=4):

| bg | strategy | worst | mean | mean − worst |
|---|---|---|---|---|
| L1_light | All-GPU | 0.0915 | 0.1308 | 0.0393 |
| L1_light | All-NPU | 0.0832 | 0.1260 | 0.0428 |
| L1_light | Oracle | 0.1086 | 0.1327 | 0.0240 |
| L2_lm | All-GPU | 0.0618 | 0.1000 | 0.0382 |
| L2_lm | All-NPU | 0.0422 | 0.0876 | 0.0454 |
| L3_vlm | All-GPU | 0.0159 | 0.0468 | 0.0309 |
| L3_vlm | Isolated | **0.0239** | **0.0869** | **0.0630** |
| L3_vlm | Cont-aware | 0.0147 | 0.0829 | 0.0682 |
| L3_vlm | All-NPU | **0.0833** | 0.1260 | 0.0427 |
| L3_vlm | Oracle | 0.0834 | 0.1197 | 0.0363 |

### ★ C3 evidence (mean hides the blinded camera) — AVAILABLE
**L3_vlm N=4, Isolated**: worst sAP **0.0239** but mean sAP **0.0869** — mean is **3.6×** the worst; mean hides a Δ0.063 collapse of the GPU-blinded streams. Same effect on All-GPU (worst 0.0159, mean 0.0468 = 2.9×) and Cont-aware (0.0147 vs 0.0829 = 5.6×). A mean-sAP report would call these placements "fine" while their worst camera is dead.

Contrast All-NPU: worst 0.0833, mean 0.1260 (1.5× — no blinded camera). So the mean-vs-worst gap is itself diagnostic: large gap = a blinded stream hidden by averaging.

**Limitation**: rev20 stores per-strategy worst+mean but **not per-stream sAP** (no `s{i}_sap` columns), so I cannot point to *which* stream is blinded from rev20 alone. The worst-vs-mean gap already makes the C3 point; per-stream decomposition for VLM would need a re-measure (flagged, not done).

## PART B — bit-identical & offline-mAP-invariance

### B1 — bit-identical detections: VERIFIED but NOT persisted
The rev19 safety gate ran threads=24 vs threads=4 on 10 frames and reported: **142 vs 142 detections, max box-coord diff 0.00e+00, max score diff 0.00e+00**. This is the bit-identical evidence (the thread lever changes only speed). **However it was printed to the rev19 run log, not saved to a CSV artifact.** `rev18_postproc_levers.csv` holds per-size sAP at both thread counts but no detection-hash record.
→ Status: **result known (diff = 0), but needs a 1-shot re-run to persist as a citable CSV** (`rev21_bitident_check.csv` currently holds the mAP comparison, not the per-detection hash). Re-run is trivial (~30 s) but is a measurement, so deferred per "extraction only."

### B2 — offline mAP off==on: ⚠️ DOES NOT HOLD (and why)
Compared offline mAP at threads=4 (`rev19_table1_threads4.csv`) vs threads=24 (`rev10_gen_single.csv`), NPU L0 24-log:

| metric | threads=4 | threads=24 | diff |
|---|---|---|---|
| mAP_small | 0.0094 | 0.0086 | +0.0008 |
| mAP_medium | 0.1966 | 0.1722 | +0.0244 |
| mAP_large | 0.5993 | 0.5293 | **+0.0700** |
| mAP_5095 | 0.2279 | 0.2014 | +0.0265 |
| sAP_large (contrast) | 0.4773 | 0.3810 | +0.0963 |

**Offline mAP is NOT thread-invariant.** Reason: the implemented `per_stream_map_offline` evaluates over *all* post-warmup imgIds, so frames the slow (threads=24) NPU **skips** count as zero recall — i.e. the offline mAP here already bakes in the throughput/skip penalty. It is "quantization + skip-recall," not pure quantization.

→ The intended causal-separation table ("offline mAP identical in both settings; only sAP changes") **cannot be built from existing data**, because this offline-mAP definition is not skip-free. To get a true skip-free quantization mAP you must score **only processed frames against their own GT** in both settings (a re-computation / re-measure, not available now).

What IS the valid causal-separation evidence (from existing data):
- **Per-processed-frame detections are bit-identical** across thread counts (B1, diff 0) → detection *quality* is unchanged by the lever.
- **At threads=4, skip = 0%** → sAP carries no staleness, so the threads=4 per-size gap (small −48%, large ≈0, rev19) **is** the pure quantization signal; the threads=24 extra loss (large −0.096) is the skip/staleness term. The separation comes from the **skip=0 vs skip=52% contrast**, not from an offline-mAP invariance.

## PART C — verdict

| need | status | source / action |
|---|---|---|
| mean sAP (Table 3 column, all cells) | **✓ available** | `rev20_5strat_heavybg.csv` `mean_sap`; extracted to `rev21_mean_worst_extract.csv` |
| C3 "mean hides blinded camera" | **✓ available** | VLM N=4 Isolated worst 0.0239 / mean 0.0869 (3.6×) |
| per-stream decomposition (which stream blinded) | ✗ re-measure | rev20 lacks `s{i}_sap`; need per-stream VLM re-run |
| bit-identical detections (diff=0) | **◐ known, not persisted** | rev19 gate reported diff 0.00e+00; re-run ~30 s to save CSV |
| offline mAP off==on | **✗ does not hold** | mAP includes skip-recall; not a valid invariance table |
| pure-quantization separation evidence | **✓ available (alt form)** | threads=4 skip=0 per-size gap (small −48%, large ≈0) vs threads=24 skip=52% |

### Immediately usable table drafts (existing data)

**Table 3 — worst & mean sAP (threads=4, N=4), reversal story All-GPU↔All-NPU↔Oracle:**

| co-tenant | All-GPU worst/mean | All-NPU worst/mean | Oracle worst/mean |
|---|---|---|---|
| L1_light | 0.0915 / 0.1308 | 0.0832 / 0.1260 | 0.1086 / 0.1327 |
| L2_lm | 0.0618 / 0.1000 | 0.0422 / 0.0876 | 0.0626 / 0.0986 |
| L3_vlm | 0.0159 / 0.0468 | **0.0833 / 0.1260** | 0.0834 / 0.1197 |
(Isolated/Cont-aware available in CSV for supplementary.)

**Table 2 support — causal separation (existing data, reframed):**

| setting | NPU skip | per-size sAP gap (S/M/L) | interpretation |
|---|---|---|---|
| threads=4 (skip-free) | 0% | −48% / −20% / **≈0** | pure INT8 quantization |
| threads=24 (skip 52%) | 52% | −56% / −36% / −20% | quantization + staleness |
(Per-frame detections bit-identical across the two — diff 0, to be persisted.)

### To pair with the next measurement (saturation sweep)
1. Per-stream sAP for VLM cells (decompose blinded stream) — add `s{i}_sap` capture.
2. Persist bit-identical detection-hash CSV (threads 4 vs 24).
3. (Optional) skip-free offline mAP (score only processed frames) for a clean quantization-only mAP at threads=24.

## Files
- `results/rev21_data_availability.md` (this file)
- `results/rev21_mean_worst_extract.csv` (PART A, all cells)
- `results/rev21_bitident_check.csv` (PART B2 offline-mAP comparison — note: shows NON-invariance)
- Measurement: 0. core scripts / prior artifacts / paper unchanged.
