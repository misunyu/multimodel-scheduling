# FINDINGS — "GPU frame skip" definition + why sAP > 0 at 100% skip

Read-only verification for `paper/main_vision.tex`. No paper/code/config/result
modified. Every claim is grounded in a file path + quoted code or a logged CSV
row; anything not found in code is labeled **(inference)**.

(Part A repeats/condenses `FINDINGS_frameskip_definition.md`; Part B is new.)

---

## PART A — Definition of "GPU frame skip"

**Q1 — Whose skip? → the foreground YOLOv11s detector's own deadline-miss rate.**
> `_step_d_common.py:252-259`
> ```python
> # Frame skip = inferences whose effective latency exceeds frame period (33.3ms @30FPS)
> eff_post = result["eff_ms"][WARMUP_FRAMES:] ...
> period_ms = 1000.0 / FPS                      # 33.33 ms
> n_over = sum(1 for x in eff_post if x > period_ms)
> "frame_skip_pct": 100.0 * n_over / len(eff_post)
> ```
> `eff_ms` is populated only by the foreground worker (`fg_worker`,
> `_step_d_common.py:209,212`). The ResNet50 co-tenant is a bare
> `while: sess.run()` loop (`_step_d_common.py:342-345`) with no `result` dict —
> it never produces a `frame_skip_pct` and is **never counted**.

**Q2 — Aggregation over N=4 streams? → arithmetic mean over the foreground
streams on that device** (not worst / single / pooled).
> `phase_rev25_persize.py:102` (feeds `tab:persize-contention`):
> `gs = np.mean([s["frame_skip_pct"] for s in agg["per_stream"] if s["device"]=="GPU"])`
> `phase_rev22_sweep.py:52-55` (`dev_skips`, feeds `fig:sweeps` / `tab:main`): same `np.mean`.

- **Numerator** = that stream's post-warmup inferences with `eff_ms > 33.33`;
  **denominator** = number of post-warmup inferences performed
  (`len(eff_post)`), not GT-frame count. `WARMUP_FRAMES=30` dropped first.
- **Per-stream spread, ≈59% point** (`rev22_perstream_sap.csv`, N=4, k=2,
  All-GPU): 57.8 / 56.0 / 54.3 / 54.9 → mean 55.8, range ≈3.5 pp (mean ≈ worst).
- `tab:main`'s "GPU-skip / NPU-skip" and `fig:sweeps`'s x-axis are the **same**
  metric under different placements — no inconsistency found.
- ResNet50 launched as co-tenant only: `preload_background_models("L1")` = resnet
  (`_step_d_common.py:293`); lever `[_bg_resnet50_loop]*k`
  (`phase_rev22_sweep.py:49`, `phase_rev25_persize.py:45`).

---

## PART B — Why sAP is nonzero at "100%" skip

### B0. "100%" is LITERAL here, not rounded

The saturated point in both tables uses the **L3_vlm (Qwen2-VL) co-tenant**, not
the ResNet50 sweep (`phase_rev25_persize.py:34` `POINTS[...]=("skip~100","L3_vlm")`).
The VLM co-tenant drives a *single* YOLOv11s inference to **~377–690 ms**
— roughly **11–20× the 33.3 ms budget** (`cstar.csv`, `GPU,L3_vlm` rows,
`eff_e2e_mean` column: 460.9 / 432.9 / 437.9 / 475.0 / … ms). When *every*
inference exceeds the period, `frame_skip_pct` is **exactly 100.0**:

- `rev25_persize_under_contention.csv:19,21,23` — `skip~100,All-GPU`,
  `gpu_skip = 100.0 / 100.0 / 100.0` (all 3 reps).
- `step_h2_bg_ablation.csv`, `L3_vlm Naive_allGPU` (N=4): per-stream skips
  `100.0 / 100.0 / 100.0 / 100.0`.

So "100%" in `tab:persize-contention` is **correct as written** for the VLM
operating point (it is genuinely 100.0%, not a rounded 99.x). The thing that
needs clarifying is therefore **not** a rounding caveat but *why accuracy
survives a literal 100% skip*. **(This corrects the task's "is 100% rounded?"
premise: at the VLM point it is literally 100%.)**

> Caveat: the logs round to 1 decimal (`round(gs,1)`), so I cannot from the CSV
> alone distinguish a true 100.0 from a 99.95+ rounding. But the per-frame
> latencies (11–20× budget) make a literal 100% — every inference over budget —
> the mechanistically forced reading. **(inference, but tightly constrained.)**

### B1. The evaluator carries forward the last delivered detection (verified)

`per_stream_sap` pairs each GT frame with the **most recently completed**
inference at that wall-clock time — i.e. the streaming "last delivered result":
> `_step_d_common.py:233-250`
> ```python
> tidx_p1 = 0
> for ii, img in enumerate(imgs):
>     t_gt = ii / FPS
>     while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t_gt:
>         tidx_p1 += 1                       # advance past all inferences done by t_gt
>     ...
>     tidx = tidx_p1 - 1                      # most recent COMPLETED inference
>     bb, sc, lb = results_parsed[tidx]       # <-- reused (carried forward) if no fresh one
>     if input_fidx[tidx] == ii: in_time += 1 # "fresh": result computed from frame ii itself
>     for k in range(len(bb)):                # its boxes are scored against GT frame ii
>         ccf.append({...})
> ```
`timestamps[t]` is inference *completion* time. If the detector is behind,
`tidx` stays pinned to an **older** inference (computed on an earlier frame) and
its boxes are scored against many subsequent GT frames = **stale carry-forward**.
A frame contributes an *empty* prediction only before the first inference
finishes (`tidx_p1==0 → miss`, `:240-242`) — i.e. essentially never after warmup.

**So "100% skip" does NOT mean "no predictions."** The detector still completes
inferences (just ~12–20 frames late); their (stale) boxes are carried forward
and scored → nonzero sAP.

### B2. Residual sAP = stale boxes matching slow/large objects (verified by size)

At ~400 ms latency each inference spans ~12 frame periods, so a detection
computed on frame `ii-12` is reused for ~12 consecutive GT frames. Large/static
objects barely move over 12 frames → stale box still overlaps GT → residual sAP;
small/fast objects move out → near-zero. The logged size breakdown matches:

`rev25_persize_under_contention.csv` (All-GPU, per-size **absolute** sAP):

| point | gpu_skip | small | medium | large |
|---|---|---|---|---|
| skip0 (uncontended) | 0.0 | 0.0078 | 0.1681 | 0.4587 |
| skip~100 (saturated) | 100.0 | ~0.0015 | ~0.052 | **~0.133** |

Residual fraction retained at saturation: large ≈29%, medium ≈31%, small ≈19%.
The residual accuracy is **concentrated on large objects** (0.133 absolute) —
exactly the staleness signature of `tab:persize-contention`. Nothing contradicts
this. Same story in `tab:main`'s VLM All-GPU cell:
`step_h2_bg_ablation.csv` `L3_vlm Naive_allGPU` worst = **0.0171** (one rep;
`tab:main` reports 0.016 as mean-of-3), with all four streams at 100.0% skip —
nonzero worst-stream sAP under a literal 100% skip.

### B3. Fresh-vs-stale fraction: ~0 fresh (mechanistic; not separately logged)

`in_time` (fresh count) is computed at `_step_d_common.py:245` but is **not**
persisted: `measure_multistream`'s per-stream dict
(`phase_rev6_sweep.py:217-223`) keeps only sap fields + `frame_skip_pct`, not
`in_time`/`miss`/`n_eval`. So the exact fresh fraction is **not retrievable from
the rev22/rev25/step_h2 CSVs.** **(inference:)** at ~400 ms latency (≈12× the
33 ms period), the inference launched on frame `ii` finishes ~12 frames later,
so the result available at GT-frame `ii`'s matching time was computed on frame
`≈ii-12` → `input_fidx[tidx] == ii` is essentially never true → **fresh ≈ 0**
while sAP > 0 from carried-forward boxes. This is forced by latency ≫ period but
is not a directly logged number.

---

## Two concrete edits (worded to match the code)

**(i) Define "GPU frame skip" — for `tab:persize-contention` (applies verbatim to
`tab:main` / `fig:sweeps`):**
> GPU frame skip is the foreground YOLOv11s detector's deadline-miss rate — the
> fraction of its post-warmup inferences whose latency exceeds the 33.3 ms frame
> period — averaged over the $N{=}4$ foreground streams; the ResNet50 co-tenant
> is the contention source only and is not counted.

**(ii) Clarify nonzero sAP at saturation — for the `tab:persize-contention`
caption or the saturation sentence (`main_vision.tex:283`):**
> At the saturated point every inference misses the 33.3 ms deadline (frame skip
> $=100\%$); accuracy remains nonzero because the streaming evaluator carries the
> most recent detection forward, and stale detections still match slow-moving and
> large/static objects. The residual sAP is therefore concentrated on large
> objects.

Note: do **not** change "100%" to "$\approx$100%" for the VLM/saturated row — it
is literally 100.0% there (per-frame latency is 11–20× the budget). The "$\approx$"
caveat would only apply to a ResNet50-swept point that rounds up from 99.x; the
saturated table row is the VLM point, which is exactly 100.0.

---

## Audit trail

- Skip formula: `_step_d_common.py:252-259`; `eff_ms` src `:209,212`; warmup `:220`.
- Carry-forward pairing: `_step_d_common.py:233-250` (`tidx=tidx_p1-1`, reuse
  `results_parsed[tidx]`; `in_time` at `:245`; empty-only-before-first at `:240`).
- Cross-stream mean: `phase_rev25_persize.py:102`, `phase_rev22_sweep.py:52-55`.
- Co-tenant loop (uncounted): `_step_d_common.py:342-345`.
- Saturated = VLM: `phase_rev25_persize.py:34`.
- Literal 100% + nonzero per-stream sAP: `rev25_persize_under_contention.csv:19-24`,
  `step_h2_bg_ablation.csv` (L3_vlm Naive_allGPU, skips 100/100/100/100,
  worst 0.0171), `cstar.csv` (GPU,L3_vlm eff_e2e 377–690 ms).
- `in_time` NOT persisted: `phase_rev6_sweep.py:217-223`.

**Found vs inferred:** Part A, the carry-forward path, literal-100% at the VLM
point, and the size-breakdown attribution are **found in code/CSV**. Inferences
(labeled inline): fresh≈0 fraction (in_time not logged) and the
true-100 vs 99.95-rounded distinction (1-decimal logs) — both tightly forced by
the measured latency ≫ period, but not separately logged numbers.
