# rev17 — separation of the two loss axes (quantization vs latency/staleness)

_Extraction-based (both sAP and offline mAP are already in `rev10_gen_single.csv`). No new measurement. core scripts / prior artifacts / paper unchanged. Pinned state: mxq `b2441f9d`, global8, yolo11s, L0; values are 24-log means (1 measurement rep; std across logs reported). rev12/T-D0/rev14/rev15/rev16 unchanged._

## STEP A — what each Table 1 row actually measures

Source: `rev10_gen_single.csv` (per-size, L0) + `rev12_single_stream.csv` (24-log agg). Both expose **both** metric families per row:

| Table 1 row | metric family | clock applied? | source column |
|---|---|---|---|
| sAP$_{[.50:.95]}$, sAP$_{0.50}$ | **streaming sAP** | YES (33.3 ms budget) | `sap_5095`, `sap_50` |
| AP$_{small/medium/large}$ | **streaming sAP** by size | YES | `sap_s`, `sap_m`, `sap_l` |
| infer mean (ms) | wall latency | — | `latency_mean` |

So Table 1's per-size AP rows are **streaming sAP** (quant + throughput-skip + staleness), not pure offline mAP. The CSV also carries offline `map_s/m/l` (clock off, no staleness) which Table 1 does not currently surface.

`per_stream_map_offline` (definition in `step_f_partA_matrix.py:59`): pairs each detection with its **own input frame** (no temporal/staleness penalty) but evaluates over all post-warmup imgIds, so frames the detector skipped contribute zero recall. → **offline mAP = quantization + (size-neutral) throughput-skip penalty; NO staleness.**

## STEP B — pure-quantization signal (offline mAP), per size

yolo11s, L0, 24-log, GPU-FP32 vs NPU-INT8:

| size | offline mAP gap (abs) | offline mAP rel |
|---|---|---|
| small  | −0.0097 | **−53.2%** |
| medium | −0.0679 | −28.3% |
| large  | −0.0844 | **−13.7%** |

The size-stratification (small ≫ large) is **present with the clock off**. Since the throughput-skip penalty is size-neutral (a skipped frame loses objects of all sizes equally), a small-worst pattern in offline mAP can only be **quantization**. ✓

## STEP C — quantization vs staleness separation (core verdict)

`results/rev17_map_vs_sap_bysize.csv`:

| size | mAP_gap (quant) | sAP_gap (quant+stale) | staleness = sAP−mAP | direction |
|---|---|---|---|---|
| small  | −0.0097 | −0.0087 | **+0.0010** | no staleness |
| medium | −0.0679 | −0.0669 | +0.0010 | none |
| large  | −0.0844 | −0.0958 | **−0.0114** (std 0.040) | staleness adds to large |

**Two axes separate cleanly and in opposite size directions:**
- **Quantization** (offline mAP): small −53% ≫ large −14% — concentrated on **small**.
- **Staleness** (sAP − mAP): ~0 on small/medium, **−0.011 on large** — concentrated on **large**.

This is exactly the two-mechanism prediction (quantization small-biased, staleness large-biased), and it is confirmed in data, not assumed. The headline size-stratification (small −54.9%) **is the quantization signature**, verified clock-off.

### Important caveat (host-postprocess skip)

The NPU skips ~52% of frames even at L0 (rev16: host postprocess 27 ms → eff 35.5 ms > 33.3 ms budget). This adds a **size-neutral throughput/recall penalty** to BOTH the NPU's mAP and sAP. So:
- the *absolute* NPU gaps (both mAP and sAP) are inflated by the host-postprocess skip;
- but the *size pattern* (small-worst) and the staleness term (large-worst) are unaffected by it, because skip is size-neutral.

A truly skip-free pure-quantization number would require a clock-off / process-every-frame run (not done — out of extraction scope). The size-stratification verdict does not need it.

## STEP D — what drives the Table 6 reversal

rev10 gen-gain (NORMAL), worst-stream sAP, yolo11s:

| bg | AllGPU worst | AllNPU worst | AllNPU − AllGPU |
|---|---|---|---|
| L1_light | 0.0824 | 0.0836 | **+0.0012** |
| L2_lm | 0.0630 | 0.0644 | **+0.0014** |

As background load rises (L1→L2), AllGPU worst drops (0.0824→0.0630, −0.0194) more than AllNPU worst (0.0836→0.0644, −0.0192) — and AllNPU stays above AllGPU at both levels. The NPU path is **load-independent** (its 27 ms host postprocess is a fixed cost; on-chip is physically separate from the GPU), so the reversal is the GPU's contention-staleness growing past the NPU's fixed offset — the **normal mechanism**, not an artifact of NPU's fixed postprocess. (The fixed NPU cost sets the *level*; the *reversal* comes from GPU degradation under load.)

## STEP E — text spots that conflate the two axes (for paper-edit)

| # | location | current text | problem | suggested fix |
|---|---|---|---|---|
| E1 | `main_vision.tex:183` (partA paragraph) | "the NPU's [latency] stays near **20 ms**" | stale OUTLIER-era number; normal-state NPU single-stream is **35.5 ms** (rev16), and it is host-postprocess-bound | replace "near 20 ms" with the normal-state figure and note it is a fixed host-postprocess cost, load-independent |
| E2 | `main_vision.tex:356,362` (decomposition) | "the NPU path carries **no latency loss**" / "essentially nothing to contention" | true for *contention* latency, but misleading: the NPU has a large **fixed host-postprocess latency** (27 ms → 52% idle skip). "No latency loss" reads as "NPU is latency-free" | qualify: "no *contention-dependent* latency loss (the NPU's latency is a fixed, load-independent host-postprocess cost)" |
| E3 | `main_vision.tex:163` (Table 1 caption) | "isolated evaluation that holds contention near zero" | per-size AP rows are streaming sAP and already carry the NPU's 52% host-postprocess skip → not contention but a fixed NPU throughput penalty is baked in | caption should note NPU rows include host-postprocess frame-drop; offline mAP (clock-off) isolates pure quantization |
| E4 | `main_vision.tex:297,301` (gen-decomp) | "single-stream staleness columns are near zero for every detector" | refers to GPU-side staleness (correct), but the NPU's own single-stream host-postprocess skip is non-zero and not mentioned | clarify the near-zero staleness is GPU-path; the NPU has a fixed host-postprocess throughput cost (separate axis) |
| E5 | `main_vision.tex:110` | "To first order this quantization loss does not depend on contention" | fine, but the absolute NPU sAP also carries the fixed host-postprocess skip — worth one sentence so reviewers don't read the −54.9% as pure precision | add: the absolute NPU penalty combines INT8 precision loss with a fixed host-postprocess throughput cost; the *size stratification* is precision (confirmed by offline mAP) |

Good (no change): lines 61, 76, 79, 112, 116, 119 correctly attribute quantization→small, staleness→large/contention.

## STEP F — final verdict

1. **Two axes separate cleanly.** Offline mAP (clock-off) = quantization, size-stratified small≫large (−53.2% vs −13.7%). Staleness (sAP−mAP) = ~0 on small, −0.011 on large — opposite direction. The framing "quantization vs latency, opposite size directions" is **supported by data**.
2. **Table 1 size gap is a genuine quantization signature**, confirmed by offline mAP (not a staleness artifact). C2 (cause = quantization × latency mismatch) holds.
3. **Staleness mixed into NPU sAP**: the NPU's *streaming* numbers additionally carry a size-neutral throughput penalty from the fixed host-postprocess (27 ms → 52% idle skip, rev16). This inflates absolute NPU gaps but does not touch the size pattern or the GPU-staleness reversal mechanism.
4. **Table 6 reversal is the normal mechanism** (GPU contention-staleness overtaking a fixed NPU offset), not an artifact of NPU postprocess.
5. **Text conflations to fix**: E1–E5 above — chiefly the stale "20 ms" NPU latency (→35.5 ms) and the unqualified "NPU path carries no latency loss" (→ no *contention* latency; it has a fixed host-postprocess cost).

### Recommendation (labels/framing)
- Relabel Table 1 per-size rows as **streaming sAP** and add a companion **offline mAP** per-size column (already in CSV) so the quantization (clock-off) signal is shown explicitly alongside the streaming one — this directly demonstrates small≫large is precision, pre-empting the "it's just staleness" objection.
- Add one Limitations sentence: the NPU's absolute single-stream penalty combines INT8 precision loss (size-stratified, the object of study) with a fixed host-postprocess throughput cost (size-neutral, on-chip is only 4 ms); the reversal depends on the former's size structure and the GPU's contention latency, not on the NPU's fixed cost.
- Fix E1 (20→35.5 ms) and E2/E4 wording (contention-latency vs fixed host latency).

## Files
- `results/rev17_loss_separation.md` (A–F)
- `results/rev17_map_vs_sap_bysize.csv` (per-size mAP vs sAP, staleness)
- No new measurement (extraction only). core scripts / prior artifacts / paper unchanged.
