# rev19 — threads=4 operating point: state audit + Table 1/3 rebuild + ⚠️ headline impact

_Operating point fixed at `torch.set_num_threads(4)` (NPU L0 ~10 ms, skip 0%, 3-rep std≈0). Safety gate: NPU detection boxes are **bit-identical** across thread counts (max diff 0.00e+00) — the lever changes only speed. mxq `b2441f9d`, global8, yolo11s. No core eval script modified. paper unchanged._

## Safety gate (boxes unchanged by thread count)
10 frames, threads=24 vs threads=4: total dets 142 vs 142, max box-coord diff `0.00e+00`, max score diff `0.00e+00`. **The thread lever is a pure speed change.** ✓

## STEP 1 — thread-state audit (corrects rev14/rev15)

`results/rev19_thread_state_audit.csv`. State now judged on (threads/latency/skip), not latency alone:

| source | rev | NPU lat | skip% | verdict |
|---|---|---|---|---|
| rev7_single_stream | rev7 | 10.0 ms | 0.0% | **threads=4 NORMAL** |
| rev9_partA_npu | rev9 | 12.2 ms | 0.4% | **threads=4 NORMAL** |
| rev9_capacity | rev9 | — | 1.2% | **threads≈4 (NPU fast)** ⚠ mixed (see below) |
| rev9_main_cmp | rev9 | — | 1.4% | **threads≈4 (NPU fast)** ⚠ mixed |
| rev11_n8 | rev11 | — | 0.9% | **threads=4 NORMAL** |
| rev10_gen_single | rev10 | 35.5 ms | 52.4% | 24-thread MISCONFIG |
| rev10_gen_gain | rev10 | — | 40.7% | 24-thread MISCONFIG |
| rev12_* | rev12 | 35–150 ms | 52–100% | 24-thread MISCONFIG |
| rev6_baseline | rev6 | 36.4 ms | 55.0% | 24-thread MISCONFIG |

**This reverses the rev15 classification.** What rev15 called "OUTLIER, demote" (rev9 family) is the *correctly configured* state; what rev15 called "NORMAL" (rev10/rev12) is the 24-thread misconfiguration.

### ⚠️ Caveat discovered: rev9 was a MIXED thread state
rev9_capacity N=4 rows show **NPU skip ≈ 0–5% (fast) but GPU skip ≈ 45% (slow)** (e.g. All-GPU s0_skip=45.1%). So rev9 ran the **NPU path fast but the GPU path at 24-thread (slow)**. That asymmetry artificially penalized All-GPU and is what produced the N=4 reversal in rev9. It is **not** a clean both-devices-threads=4 measurement.

## STEP 2 — Table 1 rebuild (threads=4, clean, 3 reps)

`results/rev19_table1_threads4.csv` (3 reps, identical to 4 decimals → fully reproducible):

| metric | GPU | NPU | gap | rel |
|---|---|---|---|---|
| infer (ms) | 8.2 | 10.0 | — | — |
| skip % | 0.0 | **0.0** | — | — |
| sAP$_{[.50:.95]}$ | 0.1957 | 0.1859 | −0.0098 | −5.0% |
| AP$_{small}$ | 0.0159 | 0.0082 | −0.0077 | **−48.4%** |
| AP$_{medium}$ | 0.1839 | 0.1477 | −0.0362 | −19.7% |
| AP$_{large}$ | 0.4768 | 0.4773 | **+0.0005** | **+0.1%** |
| offline mAP$_{large}$ | 0.6137 | 0.5993 | −0.0144 | −2.3% |

**Pure INT8 quantization** (skip-free, correct operating point): steeply small-concentrated (−48% small), **≈ 0 on large**. The paper's "large −20.1% quantization" was the 24-thread frame-skip/staleness artifact, not quantization. NPU single-stream infer is **10 ms** (not 35.5 ms; not the rev7 transient either — reproducible).

## STEP 3/4 — main-comparison at the clean threads=4 operating point (BOTH devices fast)

`results/rev19_main_comparison_threads4.csv` (N=2,4, full Oracle enumeration, 3 reps) + `rev11_n8.csv` (N=8, 3 reps). Worst-stream sAP, L1_light:

| strategy | N=2 | N=4 | N=8 |
|---|---|---|---|
| **All-GPU** | **0.1149** | **0.1149** | 0.0568 |
| Isolated (large→NPU) | 0.0830 | 0.0831 | 0.0815 |
| Cont-aware (small→NPU) | 0.1089 | 0.1089 | 0.0620 |
| **All-NPU** | 0.0836 | 0.0835 | **0.0832** |
| Oracle | 0.1149 | 0.1149 | (≥0.0832; not enumerated) |

(`results/rev19_5strat_byN.csv`.)

### ⚠️⚠️ Headline impact — the N=4/L1_light reversal does NOT survive the clean operating point

- **N=2, N=4 (L1_light):** All-GPU is the **best** strategy and **equals Oracle** (0.1149). With both devices correctly threaded, the GPU absorbs 2–4 light-background streams without contention (4×8 ms ≈ frame budget), so moving any camera to the INT8 NPU only adds quantization loss. **No reversal at N≤4 / L1_light.**
- **N=8 (L1_light):** All-GPU collapses to 0.0568 (8×8 ms ≫ 33 ms → heavy GPU contention/skip) while All-NPU holds **0.0832**. Here the NPU strategies win → **reversal holds at N=8**.
- The paper's central deployable claim — "Contention-aware (partial small-rich→NPU) gives the best, near-oracle worst-stream at **N=4, L1_light**" (current text, line ~386, Table gen-gain) — is **contradicted** at the clean operating point: at N=4 All-GPU is best; Cont-aware is never the single best at any N (at N=8 All-NPU and Isolated both beat it).

### What DOES survive (the defensible core)
- **Two size-opposite mechanisms, cleanly isolated** (rev18/rev17 + STEP 2): INT8 quantization is small-concentrated (−48% small, ≈0 large); contention staleness is large-concentrated. The thread lever causally separates them (skip on/off). This is *stronger* evidence than before.
- **The isolated single-camera rule reverses under high contention**: at N=8, the Isolated rule (large-rich→NPU, keep small-rich on GPU, 0.0815) is beaten by All-NPU (0.0832) — i.e. the small-rich cameras the isolated rule keeps on GPU should also move to the NPU. So "small-rich → NPU under contention" survives, but as **All-NPU at N=8**, not partial Cont-aware at N=4.
- **Crossover exists**, between N=4 (All-GPU best) and N=8 (All-NPU best) — a cleaner contention-driven device flip.

## STEP 5 — paper-rebuild recommendation (human decision required)

1. **Table 1** → replace with threads=4 (STEP 2): infer NPU 10 ms / skip 0%; AP_large gap +0.1% (drop "large −20% quantization"); quantization is small −48%, medium −20%, large ≈0. State text: normal = threads=4 (10 ms, skip 0); 24-thread is a host misconfiguration; rev16 "host-bound" → "torch-thread thrashing on tiny head tensors, resolved by threads=4."
2. **Reversal claim must move from N=4/L1_light to N=8** (or heavier background). The current N=4/L1_light deployable-reversal headline does not hold when both devices are correctly threaded. Options for you:
   - **(A) Re-anchor on N=8 / All-NPU**: "under deployment-scale contention (N=8) the isolated large→NPU rule is beaten by All-NPU; the safety-critical small-rich cameras should move to the NPU." Honest, supported (0.0832 vs 0.0815 vs All-GPU 0.0568). Weaker margin (All-NPU vs Isolated +0.0017).
   - **(B) Re-introduce heavier background at N=4** (L2_LM/L3) to find where the reversal appears at small N — needs new threads=4 measurement at L2_LM (not yet done).
   - **(C) Keep contention-aware framing only if a regime is found where it is the single best** — not observed at L1_light any N; would need a background/N sweep.
3. **partA / Table 2 / gen-cstar**: these are rev9 (mixed NPU-fast/GPU-slow) — the per-camera gaps reflect that asymmetry, so demote to mechanism illustration, not deployable evidence.
4. **Decomposition (Table 7) "97% quantization / NPU path carries no latency loss"**: re-examine — at threads=4 the NPU's own host cost is gone, but the large-object loss attributed to quantization was largely staleness in the 24-thread state.

## Honesty note (per spec warning)
At the clean operating point the absolute NPU quantization loss is small (large ≈ 0) — reported as-is, not hidden. The consequence is larger than "small absolute loss": the specific N=4/L1_light contention-aware reversal the paper headlines is an artifact of a mixed/misconfigured thread state. The robust, honest residual is: (i) cleanly separated size-opposite losses, (ii) a contention-driven device flip that appears at N=8. **This requires a real decision on the paper's central claim — not an auto-edit — so rev19 stops here with the data and options.**

## Files
- `results/rev19_thread_state_audit.csv`, `results/rev19_table1_threads4.csv`,
  `results/rev19_main_comparison_threads4.csv`, `results/rev19_5strat_byN.csv`, this report.
- `accv_experiments/scripts/phase_rev19_measure.py`. core scripts / prior artifacts / paper unchanged.
