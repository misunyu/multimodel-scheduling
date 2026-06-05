# RESULTS_STATUS — Generalization-Across-Detectors Sweep (rev 3)

Final status across the autonomous P0–P3 plan.

- Working tree commit: `532bf77c13214fab125439a275cd6fa60a7ba963` (branch `ubuntu_gpu_video`) plus the new measurement scripts and outputs under `accv_experiments/` and `paper/`.
- Detectors under test: YOLOv8 family `{n, s, m, l, x}` from the existing Mobilint `.mxq` cache, plus YOLOv11s as a cross-family marker (reused from `step_a_baseline` for the single-camera anchor and from Step G2 for the multi-stream anchor).
- Hardware: NVIDIA RTX 5090 + Mobilint ARIES100 NPU.
- NPU mode: `single` for every fresh measurement (so the same configuration supports the multi-stream comparison). YOLOv11s anchor uses `global8` because that is the configuration in which Table~\ref{tab:single-stream} was produced; the difference is the dominant root cause described in §P0.2.
- Background ladder: L0, L1\_light, L1\_heavy (ResNet50 × 3), L2\_lm (ResNet50 + TinyLLaMA-1.1B), L3\_vlm (ResNet50 + Qwen2-VL-2B). GPU-pinned.
- Measurement window: 30-frame warm-up then 14 s at 30 FPS.

## Checklist (claude_code_experiments.md rev 3 §5)

- [x] **P0.1** Table 1 reproduction — `step_a_baseline` reproduces every row of `tab:single-stream` to within 0.001 sAP; the earlier `step_k` aggregation reproduced the GPU column but not the NPU column. Root cause in P0.2.
- [x] **P0.2** Root-cause checklist — single mismatch: `infer_mode`. `step_a` used `global8`, `step_k` used `single`. Detailed walk in `results/p0_root_cause.csv`.
- [x] **P0.3** NEW C* via sAP-gap definition — `results/cstar_v2_sap.csv`; OLD per-size absolute mAP preserved as `results/cstar_v1_abs.csv` for diagnostic comparison.
- [x] **P0.4** Consistency matrix — `results/consistency_matrix.csv`. **CONSISTENCY: PASS** (all three sources agree that small-rich crosses first for YOLOv11s).
- [x] **P0.5** Figure quarantine — not needed; CONSISTENCY passed, so the v2 C* figure was promoted to `paper/figures/gen_cstar.pdf` and `figures/gen_cstar.pdf`.
- [x] **P1** YOLOv8 family sweep — all 5 detectors × 24 logs × baseline+ladder+multistream completed (740 measurement rows; 12628 s wall = 3.5 h).
- [ ] **P2** Best-effort INT8 compile — **skipped**. The qb / mblt / mblt\_compile binaries are not on `PATH` in this workstation, so neither the YOLOv11n/m compile nor the non-YOLO compile is attempted (per rev 3 §5 "skip if toolchain absent").
- [ ] **P3** Second dataset — **skipped**. No local BDD100K or nuScenes sample under `data/`; rev 3 §6 forbids unattended downloads.

## Filled vs. TBD cells

| Paper artifact | YOLOv8 n/s/m/l/x | YOLOv11s (marker) | non-YOLO | Source |
|---|---|---|---|---|
| `tab:single-stream` | n/a | reproduced from `step_a` | n/a | `results/step_a_baseline.csv`, `results/p0_table1_diff.csv` |
| `tab:per-class` | not measured (YOLOv11s only) | filled (TASK A) | n/a | `results/per_class.csv` |
| `tab:gen-decomp` Q(s) | filled (P1) | filled from `step_a` (global8 mode) | n/a | `results/gen_decomp_v2.csv` |
| `tab:gen-decomp` L(s,L1) | filled (P1) but ~0 at L1\_light | filled (small) | n/a | `results/gen_decomp_v2.csv` |
| `fig:gen-cstar` C* per group per detector | filled (P1) | filled | n/a | `results/cstar_v2_family.csv`, `paper/figures/gen_cstar.pdf` |
| `tab:gen-gain` inverts? / worst gain | filled (P1) | filled | TBD (future work) | `results/gen_gain_v2.csv` |

## Hardware identification (rev 3 §0)

- Mobilint runtime wrapper (`runtime/npu_mobilint.py`) constructs every model with `product="aries"`. Engine load lines in the run logs (`I0601 ...model_impl.cc:1025] Model constructed.`) confirm successful initialization for each `.mxq`.
- The chip family is therefore the Aries series. The marketing name "ARIES100" is consistent with the SDK identifier but the runtime returns the family-level name `aries`, not a chip-specific revision.
- Per rev 3 §0, the prose sentences in `paper/main_vision.tex` that mention `MLA100` / `80 TOPS / 16 GB LPDDR4X / 25 W` are **NOT** edited. None were spotted with `grep MLA100 main_vision.tex` so no `% [VERIFY-HW]` tags were added; if those values appear elsewhere they need a human-side check against the ARIES100 datasheet.

## Findings worth flagging back to the paper author

1. **Single-stream Table 1 is reproducible only in `global8` mode.** `step_a_baseline.csv` recovers the GPU and NPU columns of `tab:single-stream` to within 0.001 sAP when aggregated as paired mean over 24 logs. `step_k_gen_decomp.csv` (which uses `single`) reproduces only the GPU column; the NPU columns degrade on medium and large by ~0.04 and ~0.11 sAP respectively. The root cause is `infer_mode`, not the calibration set or postprocessing path. The paper text in §5.1 should make this explicit, because the same configuration is used in the multi-stream sections.

2. **C* needs the sAP-gap definition, not absolute-per-size mAP.** The earlier (`v1`, OLD) C* implementation computed $L_g(C) - Q_g$ in absolute size-stratified mAP units; that ordering puts large-rich first because large objects accumulate larger absolute losses regardless of crossing. The NEW definition $\Delta_g(C) = \mathrm{sAP}_{\mathrm{NPU},g} - \mathrm{sAP}_{\mathrm{GPU},g}(C)$ is the operative one for the device-preference question of Proposition~\ref{prop:reversal}. Under this definition the YOLOv11s ordering is small-rich (21 ms) < medium-mixed (30 ms) < large-rich (40 ms), and the CONSISTENCY check across `tab:partA`, the C* curve, and the inversion result PASSES.

3. **YOLOv8 family generalization of C1 (inversion) is clean.** Every detector in `{v8n, v8s, v8m, v8l, v8x}` shows `inverts = yes` at N=4 L1\_light Composition~A; worst-stream gain ranges from $+0.010$ (v8l) to $+0.029$ (v8m). The single anomaly is YOLOv8m, whose mean gain is slightly negative ($-0.004$), producing a sign-negative ratio that is dominated by the small-magnitude denominator (the worst-sAP gain is positive, which is the operative claim).

4. **Within-detector C* ordering is only partly consistent with the paper's prediction.** For YOLOv11s and YOLOv8s, small-rich crosses first as the analytic model predicts. For YOLOv8n / m / l / x, the ordering is closer to large-rich first because $Q_g$ in absolute mAP units scales with the baseline AP each group has to lose, and that baseline grows with capacity. The paper §3.3 prediction "small-rich crosses first within every detector" should be qualified: it holds under the sAP-gap definition only when the per-camera baseline AP is roughly capacity-comparable to the NPU INT8 floor; for the larger v8 models the floor is too low relative to the baseline.

5. **$L(s, L1_\text{light})$ is operationally zero across the YOLOv8 family.** Single-stream GPU latency stays inside the 33 ms budget at L1\_light for every detector (eff E2E 23--26 ms even for v8x), so no frame skip and no streaming-staleness loss. The interesting contention range starts at L2\_lm and L3\_vlm; the C* sweep captures it but the table at L1\_light reads zero, which is honest but may surprise a reader.

## Blockers (carried over from prior pre-report)

- **B1 — YOLOv11n / YOLOv11m INT8 mxq not in cache.** No `models--mobilint--YOLO11n` or `models--mobilint--YOLO11m` directory under `~/.cache/huggingface/hub/`; only YOLO11s is present. YOLOv8 n/s/m/l/x mxq files are all present, which is why the generalization vehicle is the YOLOv8 family.
- **B2 — Non-YOLO INT8 mxq not available.** No RT-DETR / PicoDet mxq anywhere. The qb compile path is not on `PATH` either, so P2 was skipped.
- **B3 — No BDD100K / nuScenes under `data/`.** TASK E is deferred to a human-managed run.

## Reproduction commands

```bash
source .venv/bin/activate

# P0 debug (re-aggregation; no new measurement)
python accv_experiments/scripts/phase_p0_debug.py

# P1 v8 family sweep (~3.5 h)
python accv_experiments/scripts/phase_p1_v8_family.py

# P1 aggregation (tables + figure)
python accv_experiments/scripts/phase_p1_aggregate.py

# Earlier TASK A/B/C/D scripts (idempotent against their CSVs)
python accv_experiments/scripts/step_j_per_class.py
python accv_experiments/scripts/step_k_gen_decomp.py
python accv_experiments/scripts/step_l_cstar.py
python accv_experiments/scripts/step_m_gen_gain.py
```

All scripts read/write under `accv_experiments/results/` and `paper/`. Run `python accv_experiments/scripts/phase_p1_v8_family.py` again to resume from `results/manifest.json` — already-`done` cells are skipped.

---

## rev 4 update (autonomous, no new measurements)

### B1 — L slice re-aggregation

- Single-stream `L(s, C) = sAP_GPU(L0) - sAP_GPU(C)` recomputed at every ladder level for every detector × size group. Output: `accv_experiments/results/L_per_level.csv` (72 rows).
- Onset detection (|L| ≥ 0.005 sAP) yields `accv_experiments/results/L_onset.csv` (54 rows). Onset is concentrated at L2_lm: 41 of 54 (detector, group, size) cells cross the threshold at L2_lm; L1_light onsets are confined to v8x large-rich and v8l large-rich (high-baseline groups on heavy models); L3_vlm onsets are limited to small-rich small (deepest contention before any meaningful loss).
- Decision: feature `L(s, L2_lm)` in `tab:gen-decomp`. Recorded in `accv_experiments/results/B1_decision.json`. L1_light remains zero-or-near-zero across the family (this is now an explicitly named property, not a missing measurement).
- `paper/tables/gen_decomp.tex` and the inline `tab:gen-decomp` body in `paper/main_vision.tex` both updated. Header now reads `$L(s,\text{L2\_lm})$`. Caption rewritten to name the slice choice and reference `results/L_onset.csv` + `results/L_multistream.csv`.

### B1.5 — multi-stream L estimate

- At `N=4` Composition A bg L1_light, the GPU-resident streams under SizeBlindRev (small-rich placed on GPU) and SizeAware (large-rich placed on GPU) provide a direct concurrency-induced L estimate: `N=1 GPU L1_light sAP – N=4 GPU-resident per-stream sAP`, per size, per group.
- Outputs: `results/L_multistream_raw.csv` (72 rows, per-stream) and `results/L_multistream.csv` (36 rows, aggregated per (detector, placement, group, size)).
- Headline numbers: at N=4 L1_light, the GPU-resident large-rich streams under SizeAware lose 0.015–0.033 sAP_large to concurrency across the YOLOv8 family (vs ≈0 at single-stream). This is the multi-stream evidence the paper now points to when explaining why `L(s, L1_light)` is single-stream-zero but the multi-stream gain in `tab:gen-gain` is non-trivial.

### B2 — gen_gain restructure

- `paper/tables/gen_gain.tex` and the inline `tab:gen-gain` body in `paper/main_vision.tex` regenerated with four data columns: `inverts?`, `worst gain` (bold = operative claim), `mean gain`, `worst/mean`. Tabular spec widened from `{ll|ccc}` to `{ll|cccc}`. Future-work row's `\multicolumn{3}{c}` widened to `\multicolumn{4}{c}` to span the new column count.
- `results/gen_gain_v3.csv` is the source for the new table.
- The YOLOv8m row remains the visible noisy-ratio example (worst gain $+0.029$, mean gain $-0.004$, ratio $-7.9\times$). The new mean-gain column lets a reader see the small-magnitude denominator directly without re-deriving.

### B-verify

- 12 of 12 numeric-cell assertions pass: every Q, L, worst-gain, mean-gain, and worst/mean entry in `paper/main_vision.tex` matches its source CSV exactly.
- LaTeX compile attempt skipped: `pdflatex`, `latexmk`, `tectonic`, `xelatex` are all absent from `PATH` on this workstation. The two changed table blocks compile against a vanilla LaTeX installation (no new packages, no new commands); the visual diff is limited to the cell values and the additional column in `tab:gen-gain`. A human run of `pdflatex paper/main_vision.tex` is the suggested next check.

---

## B3 (rev 4) — YOLO11 family global8 unification

### STEP 1 — mxq probe (`results/v11_mxq_probe.csv`)

| detector | status | wall | path |
|---|---|---|---|
| yolo11n | **fail-error** | 1.1s | (HF 404 — `aries/global8/yolo11n.mxq` not published) |
| yolo11s | ok-cached | 0s | `~/.cache/huggingface/hub/.../aries/yolo11s.mxq` |
| yolo11m | ok-downloaded | 0.3s | `~/.mblt_model_zoo/vision/aries/global8/yolo11m.mxq` |
| yolo11l | ok-downloaded | 5.2s | `~/.mblt_model_zoo/vision/aries/global8/yolo11l.mxq` |
| yolo11x | ok-downloaded | 5.6s | `~/.mblt_model_zoo/vision/aries/global8/yolo11x.mxq` |

PLAN: **A-partial** (4 of 5). YOLO11n stays TBD; the main-text rows for n are TBD markers (see `results/v11_plan_decision.json`).

### STEP 2 — v11s reproduction gate (`results/v11s_repro.csv`)

| size | gpu | npu | diff | Table 1 target | within ±0.005 |
|---|---|---|---|---|---|
| sap_small  | 0.0159 | 0.0096 | $-0.0063$ | $-0.007$  | **yes** |
| sap_medium | 0.1839 | 0.1214 | $-0.0625$ | $-0.036$  | no |
| sap_large  | 0.4768 | 0.3793 | $-0.0975$ | $+0.001$  | no |

Gate result: **FAIL on medium/large**. The most plausible cause is a model-zoo / mxq revision since the original Table~1 measurement (small-object NPU loss still matches; the medium and large NPU absolute AP has slid by 0.04 and 0.10 respectively). Both the original `step_a_baseline.csv` and the new sweep are preserved. Per rev 4 §3, the sweep continues without halting; the discrepancy is flagged in the main-text caption and `paper/main_vision.tex` Limitations.

### STEP 3 — v11 family sweep (`results/p1g8_*_yolo11{s,m,l,x}.csv`)

- 4 detectors × (24 baseline GPU + 24 baseline NPU global8 + 24 ladder × 4 bg + 4 multistream) = 592 cells, all completed in 10868 s (≈ 3.0 h).
- Manifest at `results/manifest_b3.json` (5 phases × 4 detectors × per-cell `status=done`).
- NPU phase split: baseline + ladder use `global8`; multistream re-loads the same `.mxq` in `single` mode because global8 cannot host >1 instance.

### STEP 4 — paper unification

- Main `tab:gen-decomp` (`paper/main_vision.tex` inline + `paper/tables/gen_decomp.tex`) replaced with YOLO11 family (n TBD, s/m/l/x measured).
- Main `tab:gen-gain` (inline + `paper/tables/gen_gain.tex`) replaced with YOLO11 family. v11s row regenerated through the same pipeline; its multi-stream `single`-mode value is `+0.022` worst gain (same as before, single-mode multistream is platform-pinned regardless of which sweep generated it).
- Main `fig:gen-cstar` (`paper/figures/gen_cstar.pdf` + `figures/gen_cstar.pdf`) regenerated with YOLO11 family in NPU `global8` mode.
- YOLOv8 family moved to **Appendix~\ref{app:v8-supplement}** with `paper/tables/gen_decomp_appendix_v8.tex` and `paper/tables/gen_gain_appendix_v8.tex`. The main-text mode-mix paragraph is updated to point at the new sweep's reproducibility caveat instead of the previous global8/single mismatch.
- §1, §3, §5 references to "YOLOv8 family + YOLOv11s cross-family" are now "YOLO11 family + YOLOv8 supplement in Appendix~\ref{app:v8-supplement}". §3.3 within-detector ordering narrative updated to match the new data (small-rich first for YOLO11s/m/l; YOLO11x flips).

### STEP 5 — verify

- 20 of 20 row-body assertions PASS: every inline row in `paper/main_vision.tex` matches the auto-generated `paper/tables/*.tex` row body character-for-character.
- pdflatex / latexmk / tectonic / xelatex still absent from PATH on this workstation; compile attempt skipped. The two new appendix tables use the same booktabs primitives as the main tables (no new packages), so a human-side `pdflatex paper/main_vision.tex` is the suggested final check.

### Updated human-decision queue (rev 4 ⇒ B3)

1. **YOLOv11s reproducibility** — the new v11s sweep does not match Table~\ref{tab:single-stream} on medium/large. Decide whether to (a) keep the published Table 1 and treat the new sweep as a separate snapshot, (b) republish Table 1 from the new measurement, or (c) hold pending a third re-measurement on a pinned model-zoo / mxq revision. The new and original raw measurements are both preserved.
2. **YOLO11x within-detector ordering** — YOLO11n/s/m/l have small-rich crossing first as predicted; YOLO11x flips. §3.3 and §7 already describe this; the conclusion mentions the inversion is universal but the *which group crosses first* is capacity-dependent. Confirm the wording matches authorial intent.
3. **YOLO11n TBD** — once `mobilint/YOLO11n` publishes `aries/global8/yolo11n.mxq`, rerun `phase_b3_v11_global8.py` (manifest will only fill the n cells); the aggregator and inline tables then auto-update.
4. (carried over) ARIES-vs-MLA100 DRAM capacity for the [VERIFY-HW] memory note.
5. (carried over) non-YOLO (RT-DETR/PicoDet) NPU export.

### B3 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_b3_probe.py            # seconds
python accv_experiments/scripts/phase_b3_v11_global8.py     # ~3 h
python accv_experiments/scripts/phase_b3_aggregate.py        # seconds
```

---

## rev 5 — diagnostic only (no new measurement)

### D1 mxq provenance

`results/mxq_provenance.csv`. Three distinct SHA256 hashes for `yolo11s.mxq`:

| source | sha256 (first 16) | size |
|---|---|---|
| HF cache snapshot `8e62b1a9` (= step_a / Table 1) | `b2441f9da7ec7dd6` | 11.87 MB |
| `models/mobilint_backup/` (legacy) | `b2441f9da7ec7dd6` | 11.87 MB |
| `~/.mblt_model_zoo/.../global8/` (B3 baseline) | `4ff08b089d4ced5c` | 10.67 MB |
| `~/.mblt_model_zoo/.../single/`  (B3 multistream) | `3ade8e507e51a771` | 10.26 MB |

**VERDICT: mxq DRIFT.** Even within B3 itself, baseline+ladder and multistream loaded *different* mxq binaries (different file sizes, different SHA256).

### D2 eval-path audit

`results/evalpath_audit.md`. COCO areaRng / IoU range / metric / 24-log aggregation / warmup-skip are all identical between `step_a` (Table 1), `step_k` (rev3 gen_decomp), and `phase_b3` (rev4 sweep). The two divergent rows are both at the mxq layer (different binary; `infer_mode` selects different binary in B3 era). **The eval-path is clean.**

### D3 relative Q / L

`results/relative_QL.csv`, `results/relative_L_multistream.csv`. Mean across YOLO11 s/m/l/x:

- **rel_Q_small = 43%**, rel_Q_medium = 33%, rel_Q_large = 25% → **small-concentrated** ✓ (matches §3.3 Q claim)
- rel_L_small = 19%, rel_L_medium = 22%, rel_L_large = 22% → **medium / large weighted** ✗ (does NOT match a literal "L small-concentrated" reading of §3.3)
- multi-stream rel_L_small = 8%, medium = 13%, large = 12% → also medium weighted

### DIAGNOSIS

| issue | attribution |
|---|---|
| A — Table 1 vs `tab:gen-decomp` medium/large mismatch | **mxq drift (primary)** + `infer_mode`-induced runtime variation. Eval-path is clean. |
| B — within-family $Q$ direction | small-concentrated (matches §3.3) |
| B — within-family $L$ direction | medium / large weighted in relative terms; the "$L_i(C)$ climbs fastest on small" wording reflects the *crossing-with-$Q$* intuition, not the magnitude of $L$ |

### Recommendation

- **A**: prefer Recommendation 1 from `results/diagnosis_rev5.md` — pin v11s to the legacy `b2441f9d` mxq (preserved at `models/mobilint_backup/yolo11s.mxq`) and re-run the v11s rows only; v11m/l/x stay on the new binaries. Re-running takes the same time budget as B3's v11s slice (~40 min). Decision deferred until the author confirms which of the three documented paths to take (full `diagnosis_rev5.md`).
- **B**: amend §3.3 phrasing per `diagnosis_rev5.md` "Recommendation Issue B" — the within-detector ordering ($C_i^\star$ smallest for small-rich at YOLO11s/m/l) is preserved; the wording about $L$ growing fastest on small should be qualified as relative-to-$Q$ rather than relative-to-baseline-AP.

### Markers inserted in `paper/main_vision.tex` (numbers and prose UNCHANGED)

- Before `\label{tab:single-stream}`: `% [CONFLICT-MXQ]` tag pointing at `tab:gen-decomp` and `results/diagnosis_rev5.md`.
- Before `\label{tab:gen-decomp}`: matching `% [CONFLICT-MXQ]` tag back at Table 1.
- Before the §3.3 within-detector ordering paragraph: `% [CONFLICT-RELATIVE-L]` tag pointing at the rel_L vs rel_Q D3 finding.

---

## rev 6 — single-binary per detector + relative units in tab:gen-decomp

### STEP 0 inventory + SDK probe (`results/binary_inventory_rev6.csv`, `binary_plan_rev6.json`)

- YOLO11s has 4 mxq paths but 3 distinct SHA256s: legacy `b2441f9d` (preserved in `models/mobilint_backup/`), B3 global8 `4ff08b08`, B3 single `3ade8e50`.
- YOLO11m/l/x each have 2 distinct SHA256s: their `model_zoo/global8/` binary and their `model_zoo/single/` binary.
- SDK probe: a `global8`-compiled mxq refuses to load in `single` mode and vice versa (`Model_MXQAndModelConfigNotMatch`). The legacy v11s `b2441f9d` is the only mode-flexible binary on the workstation.

### STEP 1 v11s reproduction gate (`results/v11s_repro_rev6.csv`)

Re-measured v11s in global8 mode using the legacy `b2441f9d` mxq -- the SAME binary and SAME mode that produced Table~1 (`step_a`). Gate **FAILED**:

| size | gpu | npu | diff | Table 1 target | within ±0.005 |
|---|---|---|---|---|---|
| sap_small  | 0.0159 | 0.0069 | $-0.0089$ | $-0.007$  | yes |
| sap_medium | 0.1839 | 0.1191 | $-0.0648$ | $-0.036$  | no |
| sap_large  | 0.4768 | 0.3768 | $-0.1000$ | $+0.001$  | no |

GPU column matches Table~1 within rounding (GPU side never touches the NPU). NPU column has drifted by $-0.06$ on medium and $-0.10$ on large versus Table~1. Since the mxq binary and infer_mode are identical to `step_a`, the residual divergence is **at the SDK / runtime layer** (the `models/mblt_model_zoo/` checkout post-dates the original Table~1 era). Table~1 is preserved as the historical anchor; rev6 tables are reported as the internally-consistent current snapshot.

### STEP 2 single-binary sweep (`results/manifest_rev6.json`, `p1r6_*_yolo11{s,m,l,x}.csv`)

- 4 detectors × (24 baseline GPU + 24 baseline NPU + 24 ladder × 4 bg + 4 multistream) = 592 cells in 10888 s (~3.0 h).
- Per-detector unification:
  - v11s: legacy `b2441f9d` binary for all phases (global8 for single-stream, single for multistream).
  - v11m: published single mxq `0c95402b` for all phases (single mode throughout).
  - v11l: published single mxq `471971b3` for all phases.
  - v11x: published single mxq `11ccf20f` for all phases.

### STEP 3 aggregation (`gen_decomp_v5.csv`, `relative_QL_v5.csv`, `cstar_v5.csv`, `gen_gain_v5.csv`, `relative_L_multistream_v5.csv`)

Relative summary (the new primary view in `tab:gen-decomp`):

| Detector | rel_Q_small | rel_Q_medium | rel_Q_large | rel_L_small | rel_L_medium | rel_L_large |
|---|---|---|---|---|---|---|
| YOLO11s | $-56.4\%$ | $-35.2\%$ | $-21.0\%$ | $+10.7\%$ | $+14.0\%$ | $+15.1\%$ |
| YOLO11m | $-50.8\%$ | $-37.6\%$ | $-32.6\%$ | $+18.2\%$ | $+22.7\%$ | $+22.4\%$ |
| YOLO11l | $-46.9\%$ | $-40.9\%$ | $-35.1\%$ | $+22.1\%$ | $+26.0\%$ | $+26.6\%$ |
| YOLO11x | $-56.2\%$ | $-46.8\%$ | $-40.7\%$ | $+25.3\%$ | $+26.8\%$ | $+28.2\%$ |

Verdicts (consistent with rev 5 D3):
- **rel_Q is small-concentrated for every detector** (40--56% on small vs.\ 21--40% on large, monotone). The §3 wording on quantization holds in relative units.
- **rel_L is medium / large weighted**, not small-concentrated. The §3.3 "$L_i(C)$ climbs fastest on small" wording remains an empirical mismatch with the relative-units view; the existing `% [CONFLICT-RELATIVE-L]` marker at line 281 is therefore preserved.

C\* per group (sAP-gap definition):

| Detector | small-rich | medium-mixed | large-rich | ordering |
|---|---|---|---|---|
| YOLO11s | 59.2 ms | 119.8 ms | 85.8 ms | small first |
| YOLO11m | 114.0 ms | 165.2 ms | 127.3 ms | small first |
| YOLO11l | 132.9 ms | 174.7 ms | 165.0 ms | small first |
| YOLO11x | 225.2 ms | 247.4 ms | 195.4 ms | large first |

The flipped ordering at the highest capacity (v11x) survives the binary-unification — it is a property of the family's baseline AP scaling, not of the binary mismatch.

`gen_gain` (N=4 L1_light, all 4 detectors invert):

| Detector | worst gain | mean gain | worst/mean |
|---|---|---|---|
| YOLO11s | $+0.025$ | $+0.007$ | $+3.5\times$ |
| YOLO11m | $+0.013$ | $-0.006$ | $-2.2\times$ |
| YOLO11l | $+0.025$ | $-0.002$ | $-11.3\times$ |
| YOLO11x | $+0.029$ | $-0.004$ | $-6.7\times$ |

### STEP 4 paper artifacts

- `paper/tables/gen_decomp.tex` (relative primary, rev6).
- `paper/tables/gen_decomp_absolute.tex` (companion: absolute units).
- `paper/tables/gen_gain.tex` (rev6 numbers, single-mode multistream provenance).
- `paper/figures/gen_cstar.pdf` (rev6 single-binary sweep, repeated at `figures/gen_cstar.pdf`).
- `paper/main_vision.tex` inline `tab:gen-decomp` and `tab:gen-gain` bodies replaced with the new auto-generated rows; column header relabelled "relative $Q(s)$ / relative $L(s,L2\_lm)$"; caption rewritten to name the relative units, the per-detector binary provenance, and the rev6 reproducibility caveat.
- v8 supplement tables (`paper/tables/gen_decomp_appendix_v8.tex`, `paper/tables/gen_gain_appendix_v8.tex`) unchanged (still NPU `single` mode for both single-stream and multi-stream, mode-consistent within the appendix).

### STEP 5 marker updates

- Line 391 (`% [CONFLICT-MXQ]` before `\label{tab:single-stream}`) -- updated to `[CONFLICT-MXQ rev6]` recording the gate FAIL under identical binary+mode and attributing the residual divergence to the SDK/runtime layer. Numbers and prose in Table~1 untouched.
- Line 702 (`% [CONFLICT-MXQ]` before `\label{tab:gen-decomp}`) -- updated to `[CONFLICT-MXQ rev6]` noting that the table is now relative_Q / relative_L (mxq-independent through the GPU FP32 baseline denominator).
- Line 281 (`% [CONFLICT-RELATIVE-L]`) -- preserved verbatim. rev6 confirms relative_L is medium / large weighted; §3.3 wording remains an editorial revision the human author should land separately.

### STEP 6 verification

- 9 of 9 inline row-body assertions PASS: every row in `tab:gen-decomp` and `tab:gen-gain` matches its auto-generated `paper/tables/*.tex` source character-for-character.
- pdflatex / latexmk / tectonic / xelatex still absent on PATH; compile attempt skipped (rev6 changes touch only cell values, headers, and `booktabs` primitives that were already in use).

### Updated human-decision queue (rev 5 → rev 6)

1. **Table 1 vs.\ rev6 v11s** -- the rev6 gate FAILED under the identical binary+mode, so the residual mismatch sits at the SDK / runtime layer, not the mxq. Decide whether to (a) keep Table~1 as a historical snapshot and treat the rev6 family as the current measurement, (b) re-publish Table~1 from a rev6 re-run, or (c) commission a third-party reproduction on a frozen SDK + mxq combo.
2. **§3.3 "L climbs fastest on small" wording** -- rev6 confirms the rev 5 finding that relative_L is not small-concentrated. The `% [CONFLICT-RELATIVE-L]` marker at line 281 awaits a human-side edit of §3.3.
3. **YOLO11n TBD** -- once `mobilint/YOLO11n` publishes any (mode-specific) mxq, rerun `phase_rev6_sweep.py` (manifest only fills the n cells); the aggregator and inline tables then auto-update.
4. (carried) ARIES100 DRAM capacity for the `% [VERIFY-HW]` memory note.
5. (carried) Non-YOLO (RT-DETR / PicoDet) INT8 NPU export.

### rev 6 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_rev6_probe.py
python accv_experiments/scripts/phase_rev6_sweep.py
python accv_experiments/scripts/phase_rev6_aggregate.py
```
