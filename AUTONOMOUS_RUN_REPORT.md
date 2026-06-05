# AUTONOMOUS_RUN_REPORT — 10-hour unattended pass

Generated at the end of an unattended P0 → P1 → (P2 skipped) → (P3 skipped) run.

## CONSISTENCY matrix (headline)

| Check | Source | Value | Result |
|---|---|---|---|
| (a) `tab:partA` small-rich Δ > large-rich Δ at L1 | `accv_experiments/results/step_f_partA_matrix.csv` | small-rich Δ = $-0.018$, large-rich Δ = $-0.039$ | **PASS** |
| (b) NEW C\* (sAP-gap def): small-rich crosses first | `accv_experiments/results/cstar_v2_sap.csv` | small-rich C\* = 21.3 ms, large-rich C\* = 40.4 ms | **PASS** |
| (c) TASK D N=4 bg L1\_light: SizeAware $>$ SizeBlindRev worst | `accv_experiments/results/gen_gain_v2.csv` | worst gain = $+0.022$, inverts = yes | **PASS** |

**CONSISTENCY: PASS** for YOLOv11s. The figure `paper/figures/gen_cstar.pdf` was rebuilt from the v2 (sAP-gap) definition rather than quarantined.

## Contradictions investigated and resolved

### Contradiction 1 — Table 1 (single-camera NPU AP) vs. TASK B (`step_k_gen_decomp.csv`)

- `tab:single-stream` reports AP\_small $-0.007$, AP\_medium $-0.036$, AP\_large $+0.001$ for NPU minus GPU.
- TASK B re-aggregated 24-log single-stream measurements gave AP\_small $-0.009$, AP\_medium $-0.072$, AP\_large $-0.112$.
- The GPU column matches Table 1 within noise; the NPU column does not.

**Root cause** (`results/p0_root_cause.csv`): single mismatch is `infer_mode`. `step_a` (Table 1 source) loads the NPU model in `global8` mode; `step_k` loads it in `single` mode. Same `.mxq` file, different runtime resource allocation, materially different per-frame post-processed sAP on medium and large objects. All other pipeline parameters (mxq path, CONF=0.25, IoU=0.45, imgsz=640, COCO area thresholds, val.json categories, pycocotools per-log aggregation, 30-frame warm-up) match exactly.

**Resolution**: `step_a` reproduces Table 1 cleanly. The YOLOv11s row in `tab:gen-decomp` therefore uses `step_a` values (global8 NPU) as the marker; the YOLOv8 family uses `single` mode (necessary for the multi-stream comparison). The two configurations are labelled in the table footnote.

### Contradiction 2 — C\* ordering (OLD vs. NEW definition)

- The earlier `step_l_cstar.py` reported large-rich crossing first (C\* small-rich = 29.1 ms, large-rich = 24.8 ms), which contradicts the analytic prediction of §3.3.
- That implementation computed $Q_g$ and $L_g$ as weighted means of absolute per-size mAP losses.

**Root cause**: large absolute mAP losses pile up on objects with large baseline AP, so the absolute-units $L_g(L0)$ is already close to $Q_g$ for large-rich at $C = 0$; small-rich starts further from its own $Q_g$ in absolute units even though its sAP-gap to GPU is smaller.

**Resolution**: the operative definition for Proposition~\ref{prop:reversal} is the sAP-gap $\Delta_g(C) = \mathrm{sAP}_{\mathrm{NPU},g} - \mathrm{sAP}_{\mathrm{GPU},g}(C)$. Under this definition (results/cstar\_v2\_sap.csv), small-rich crosses first at C\* = 21.3 ms, then medium-mixed at 30.0 ms, then large-rich at 40.4 ms. The OLD figure was promoted with the NEW data; the OLD/v1 numbers are preserved in `results/cstar_v1_abs.csv` for diagnostics.

### Contradiction 3 — N=8 SizeAware (NPU=4) loses to SizeBlindRev (NPU=4)

- TASK D reported `inverts=False` at N=8 L1\_light because SizeAware\_NPU4 worst sAP (0.062) is below SizeBlindRev\_NPU4 worst sAP (0.081).
- This was flagged as a possible inversion failure.

**Root cause** (and rev 3 §3.4 explicitly excludes this as a "saturation regime"): at N=8 with bg L1\_light the GPU is past its 4-stream serialization knee, so the optimal placement has shifted to AllNPU (worst sAP 0.083, all 8 streams active). SizeAware\_NPU4 and SizeBlindRev\_NPU4 both leave 4 streams on a saturated GPU; SBR happens to win because GPU contention damages the small-rich streams less than it damages the large-rich streams it now hosts. This is the saturation tail of the same schedule-shift story documented in §6.3 of `main.tex`, not a refutation of C1.

**Resolution**: the consistency matrix tags N=8 as the saturation regime and does not count it as a contradiction. The inversion claim is reported at N=4 L1\_light, where every detector in the YOLOv8 family + YOLOv11s shows worst gain > 0.

## Filled vs. TBD cells

### Filled

- `tab:per-class` (YOLOv11s, 8 AHD classes; `paper/tables/per_class.tex`).
- `tab:gen-decomp` (5 YOLOv8 + YOLOv11s marker rows; `paper/tables/gen_decomp.tex`).
- `tab:gen-gain` (5 YOLOv8 + YOLOv11s marker rows; `paper/tables/gen_gain.tex`).
- `fig:gen-cstar` (`paper/figures/gen_cstar.pdf` rendered from NEW sAP-gap definition with all six detectors; multi-capacity curves on the right panel).

### TBD

- `tab:gen-decomp`, `tab:gen-gain` rows for non-YOLO detectors: **B2** (no INT8 mxq, no qb toolchain on `PATH`).
- YOLOv11n / YOLOv11m rows in any table: **B1** (no INT8 mxq).
- `tab:gen-decomp` $L(s, L1)$ rows for the YOLOv8 family are filled at the spec'd L1\_light level but read as essentially zero (the single-stream GPU latency stays within the 33 ms budget at L1\_light for every detector). The non-trivial staleness loss sits at L2\_lm and beyond and is captured in the C\* curves of `fig:gen-cstar`. This is a question of where to draw the contention slice, not a measurement gap.

## `paper/main_vision.tex` diff (before → after)

Three blocks changed in this run; `git diff main_vision.tex` will show the complete patch. The substantive edits are:

1. **`tab:gen-decomp` body** (lines 690–698 prior): `TBD` rows replaced with measured values from `results/gen_decomp_v2.csv`. Caption updated to remove the `[TODO-FILL]` tag and to explain the mode mismatch between YOLOv11s (`global8`) and the YOLOv8 family (`single`).

2. **`tab:gen-gain` body** (lines 730–735 prior): `TBD` rows replaced with measured values from `results/gen_gain_v2.csv`. Caption updated to remove `[TODO-FILL]` and to explain that the worst/mean ratio is the operative C2 quantity (ratio sign for YOLOv8m has limited interpretive weight because mean gain is small-magnitude).

3. **`fig:gen-cstar` caption** (post-figure): rewritten from "YOLOv11s only, with a within-detector ordering caveat" to a multi-capacity description: left panel shows $\Delta_g(C)$ with capacity encoded as marker brightness, right panel plots C\* against detector params per group.

No `% [VERIFY-HW]` tags were inserted because `MLA100` does not appear in `paper/main_vision.tex`.

## Suspect-figure handling

CONSISTENCY passed. The earlier `gen_cstar.pdf` (single-detector, absolute per-size mAP definition) was overwritten by the v2 / family-wide figure. The v1 PDF and CSVs are retained under `results/` for inspection.

## Human-decision queue (in priority order)

1. **C\* unit definition** — confirm that the paper adopts the NEW sAP-gap definition over the OLD absolute-per-size-mAP definition. The text and `tab:partA` already use sAP-gap; aligning §3 and `fig:gen-cstar` removes the contradiction. The diagnostic v1 data is preserved.
2. **ARIES100 datasheet check** — the SDK identifier is `aries`; rev 3 asked to verify the specific revision and the 80 TOPS / 16 GB LPDDR4X / 25 W numbers in prose. No occurrence of `MLA100` in `paper/main_vision.tex` was found, but any spec numbers in the broader manuscript should be cross-checked.
3. **YOLOv11n / YOLOv11m INT8 mxq compile permission** — to fill the v11 capacity row of `tab:gen-decomp` and `tab:gen-gain` and to fill the YOLOv11 axis of `fig:gen-cstar`.
4. **Within-detector ordering caveat** — the §3.3 prediction "small-rich crosses first within every detector" holds for YOLOv11s and YOLOv8s but not for the larger v8 models under the sAP-gap definition. A 2–3 sentence caveat in §3.3 reflecting this is suggested.
5. **L1\_light vs. L2\_lm slice for the staleness column** — the L(s,L1\_light) column of `tab:gen-decomp` reads zero for the v8 family. If a non-zero column is desired, switch the slice to L2\_lm or report max-over-ladder L; both are computable from `results/p1_ladder_*.csv` without new measurement.

## Reproduction & manifest summary

```bash
source .venv/bin/activate

# P0 (re-aggregation only; minutes)
python accv_experiments/scripts/phase_p0_debug.py

# P1 (24 logs × 5 detectors × baseline + ladder + multistream; ~3.5 h)
python accv_experiments/scripts/phase_p1_v8_family.py
python accv_experiments/scripts/phase_p1_aggregate.py
```

Checkpoint at `accv_experiments/results/manifest.json` covers 740 cells across 5 v8 detectors. Restarting `phase_p1_v8_family.py` resumes; already-`done` cells are skipped.

| Stage | Wall time | Output |
|---|---|---|
| TASK A (per-class) — Step J | ~10 min | `paper/tables/per_class.tex`, `results/per_class.csv` |
| TASK B (Q, L for YOLOv11s) — Step K | ~16 min | `results/gen_decomp.csv` (single mode; superseded by P0/P1) |
| TASK C (C\* OLD) — Step L | ~23 min | `results/cstar.csv`, OLD `gen_cstar.pdf` |
| TASK D (worst gain) — Step M | seconds | `results/gen_gain.csv` |
| P0 debug | ~1 min | `results/p0_*`, `results/consistency_matrix.csv`, `results/cstar_v2_*` |
| P1 v8 family sweep | 3.5 h | `results/p1_*.csv` (740 rows), `results/manifest.json` |
| P1 aggregation | seconds | `results/gen_decomp_v2.csv`, `results/cstar_v2_family.csv`, `results/gen_gain_v2.csv`, regenerated tables and figure |
| P2 (best-effort compile) | skipped | qb toolchain not on PATH |
| P3 (second dataset) | skipped | no local BDD/nuScenes |

Total unattended wall time: ~4.0 hours (well below the 10 h budget).

## Artifacts created in this run

- `accv_experiments/scripts/phase_p0_debug.py`
- `accv_experiments/scripts/phase_p1_v8_family.py`
- `accv_experiments/scripts/phase_p1_aggregate.py`
- `accv_experiments/results/manifest.json` (checkpoint)
- `accv_experiments/results/p0_table1_diff.csv`, `p0_root_cause.csv`, `p0_decisions.json`
- `accv_experiments/results/cstar_v1_abs.csv`, `cstar_v2_sap.csv`, `cstar_v2_sap_per_bg.csv`
- `accv_experiments/results/consistency_matrix.csv`
- `accv_experiments/results/p1_baseline_*.csv`, `p1_ladder_*.csv`, `p1_multistream_*.csv` (15 files)
- `accv_experiments/results/gen_decomp_v2.csv`, `cstar_v2_family.csv`, `gen_gain_v2.csv`
- `accv_experiments/results/gen_cstar_v2.pdf` (also copied to `paper/figures/gen_cstar.pdf` and `figures/gen_cstar.pdf`)
- `accv_experiments/results/phase_p0_debug.log`, `phase_p1_v8_family.log`, `phase_p1_aggregate.log`
- Regenerated `paper/tables/gen_decomp.tex`, `paper/tables/gen_gain.tex`
- `RESULTS_STATUS.md`, `AUTONOMOUS_RUN_REPORT.md`

Earlier-task outputs (TASK A–D) and the original `paper/tables/per_class.tex` are preserved as-is; nothing under `step_g2/h2/i/` raw measurement directories was modified.

---

## rev 4 — B1 + B1.5 + B2 (autonomous re-aggregation)

A second unattended pass executed B1, B1.5, B2, and B-verify from `claude_code_experiments.md` (rev 4). No new hardware measurements; the existing 740-cell P1 sweep + the YOLOv11s anchors were re-aggregated.

### B1 — L slice re-aggregation

| step | output | size |
|---|---|---|
| Per-(detector, group, level) L | `accv_experiments/results/L_per_level.csv` | 72 rows |
| Per-(detector, group, size) onset | `accv_experiments/results/L_onset.csv` | 54 rows |
| Slice decision | `accv_experiments/results/B1_decision.json` | featured slice = `L2_lm` (41 of 54 cells cross |L|≥0.005 at L2_lm) |

`tab:gen-decomp` now reads $L(s, L2_{\text{lm}})$ (was $L(s, L_1)$, all $\approx 0$). New L values from `paper/tables/gen_decomp.tex`:

| detector | $L_{small}$ | $L_{medium}$ | $L_{large}$ |
|---|---|---|---|
| YOLOv8n  | $+0.000$ | $+0.005$ | $+0.022$ |
| YOLOv8s  | $+0.001$ | $+0.016$ | $+0.045$ |
| YOLOv8m  | $+0.004$ | $+0.047$ | $+0.107$ |
| YOLOv8l  | $+0.006$ | $+0.058$ | $+0.127$ |
| YOLOv8x  | $+0.008$ | $+0.057$ | $+0.125$ |
| YOLOv11s | $+0.001$ | $+0.023$ | $+0.053$ |

### B1.5 — multi-stream L direct estimate

Concurrency-induced $L_g$ at $N=4$, Composition A, bg L1_light:
- GPU-resident large-rich streams (under SizeAware): $L_{\text{large}}$ drops 0.015–0.033 sAP across the YOLOv8 family.
- GPU-resident small-rich streams (under SizeBlindRev): $L_{\text{large}}$ drops 0.005–0.022; $L_{\text{small}}$ remains $\le 0.001$ (the small-object loss at this contention level is dominated by the concurrent GPU stream, not by stream count).

This is the previously-missing connection between gen_decomp's single-stream $L \approx 0$ and gen_gain's multi-stream inversion: single-stream cannot expose the concurrency channel of $L$; multi-stream does.

Outputs: `results/L_multistream.csv` (36 rows aggregated per detector × placement × group × size) and `results/L_multistream_raw.csv` (72 rows, per stream).

### B2 — gen_gain restructure

`tab:gen-gain` now has four numeric columns: `inverts?`, `worst gain` (bold), `mean gain`, `worst/mean`. The YOLOv8m noisy-ratio row ($-7.9\times$) is retained, but the mean-gain column ($-0.004$) immediately tells the reader the denominator is small-magnitude.

Source: `results/gen_gain_v3.csv`. Tex: `paper/tables/gen_gain.tex` and the inline `tab:gen-gain` block in `paper/main_vision.tex`.

### B-verify

- 12 of 12 cell assertions PASS: every Q, L, worst-gain, mean-gain, and worst/mean entry in `paper/main_vision.tex` matches its source CSV exactly.
- pdflatex / latexmk / tectonic / xelatex are all absent from `PATH` on this workstation; the compile attempt was skipped. The two table blocks use only `\toprule / \midrule / \bottomrule / \multicolumn / \textbf` from `booktabs`; no new packages were introduced. A human-side `pdflatex paper/main_vision.tex` is the recommended final check.

### Cumulative `paper/main_vision.tex` diff (rev 3 → rev 4)

- `tab:gen-decomp` body: every cell now matches `gen_decomp_v2.csv` (Q) and `L_per_level.csv` filtered to L2_lm.
- `tab:gen-decomp` column header: $L(s,L_1)$ → $L(s,L_2\_lm)$.
- `tab:gen-decomp` caption: rewritten to name the new slice, point at `results/L_onset.csv`, and reference `results/L_multistream.csv` for the multi-stream evidence.
- `tab:gen-gain` body and tabular spec: `{ll|ccc}` → `{ll|cccc}`; columns are now worst (bold) | mean | ratio.
- `tab:gen-gain` future-work row: `\multicolumn{3}{c}` → `\multicolumn{4}{c}`.
- The previously-flagged narrative around tab:gen-decomp (the two-caveat paragraph at lines 684–686 of `main_vision.tex` introduced by the user in the rev 3 → rev 4 hand-off) is preserved verbatim.

### Updated human-decision queue

1. `pdflatex paper/main_vision.tex` — run on a machine with a TeX distribution to confirm visual layout (no source-level errors expected).
2. (Carried over) C\* unit definition — confirm sAP-gap is the canonical operative definition.
3. (Carried over) Whether to feature `L(s, L_2\_lm)` or the `max-over-ladder` slice. B1 chose L2_lm because it is the lowest level where the staleness loss becomes meaningful across detectors; the alternative is in `results/L_per_level.csv`.
4. (Carried over) ARIES100 / MLA100 spec verification.
5. (Carried over) YOLOv11n/m and non-YOLO INT8 mxq compile permission.

### rev 4 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_b_rev4.py   # seconds
```

---

## B3 (rev 4) — YOLO11 family global8 unification

### Probe → PLAN-A-partial

- 4 of 5 YOLO11 mxq files available (s cached, m/l/x downloaded via `mblt_model_zoo`). YOLO11n is HF 404. `results/v11_mxq_probe.csv`, `results/v11_plan_decision.json`.

### v11s reproduction gate — FAIL on medium/large

- Re-measuring YOLOv11s in NPU `global8` mode reproduces Table~\ref{tab:single-stream}'s **small-object** NPU loss within tolerance ($\Delta_{\text{small}}=-0.006$ vs.\ $-0.007$), but **not** the medium and large gaps ($-0.062$ and $-0.098$ vs.\ Table 1's $-0.036$ and $+0.001$). Same mxq path and same `global8` mode as the original measurement; the divergence almost certainly reflects a model-zoo / mxq revision in the intervening period.
- Both the original `results/step_a_baseline.csv` (matches Table 1 exactly) and the new `results/p1g8_baseline_yolo11s.csv` are preserved. The Limitations paragraph in `paper/main_vision.tex` flags the reproducibility caveat. Per the rev 4 spec, the sweep continued without halting.

### v11 family sweep — 4 detectors × 148 cells = 592 cells in 3.0 h

- `results/p1g8_baseline_yolo11{s,m,l,x}.csv` (24 logs × {GPU, NPU global8})
- `results/p1g8_ladder_yolo11{s,m,l,x}.csv` (24 logs × GPU × 4 bg)
- `results/p1g8_multistream_yolo11{s,m,l,x}.csv` (4 cells each; NPU single mode for this phase)
- Manifest: `results/manifest_b3.json`

### Headline YOLO11 numbers (NPU global8 single-stream)

| Detector | $Q_{small}$ | $Q_{medium}$ | $Q_{large}$ | $L_{small}$ | $L_{medium}$ | $L_{large}$ | Inverts? | Worst gain |
|---|---|---|---|---|---|---|---|---|
| YOLO11s | $-0.006$ | $-0.062$ | $-0.098$ | $+0.001$ | $+0.025$ | $+0.066$ | yes | $+0.022$ |
| YOLO11m | $-0.013$ | $-0.063$ | $-0.126$ | $+0.005$ | $+0.050$ | $+0.115$ | yes | $+0.010$ |
| YOLO11l | $-0.014$ | $-0.079$ | $-0.140$ | $+0.008$ | $+0.061$ | $+0.139$ | yes | $+0.014$ |
| YOLO11x | $-0.020$ | $-0.094$ | $-0.166$ | $+0.010$ | $+0.068$ | $+0.148$ | yes | $+0.028$ |

C\* ordering: YOLO11s/m/l show **small-rich first** (as the analytic model predicts); YOLO11x flips to **large-rich first** because the larger absolute baseline AP makes the small-rich curve slower to reach the (now lower) NPU floor. All 4 detectors invert in the multistream worst-stream comparison.

### `paper/main_vision.tex` diff (rev 4 B1+B2 → rev 4 B3)

- `tab:gen-decomp` inline body: 5 YOLOv8 rows + 1 YOLOv11s row → 5 YOLO11 rows (n TBD, s/m/l/x measured).
- `tab:gen-decomp` caption: re-titled "YOLO11 capacity family", points at `results/v11s_repro.csv` for the reproducibility caveat, points at Appendix~\ref{app:v8-supplement} for the YOLOv8 supplement.
- `tab:gen-gain` inline body: same 5-row swap, NPU multi-stream mode caveat re-located to caption.
- `fig:gen-cstar` figure file (`figures/gen_cstar.pdf` + `paper/figures/gen_cstar.pdf`): regenerated with the YOLO11 family in global8 mode; faint→dark alpha now encodes YOLO11s→YOLO11x.
- §1, §3, §5 prose references: "YOLOv8 family + YOLOv11s cross-family check" → "YOLO11 family + YOLOv8 supplement in Appendix~\ref{app:v8-supplement}".
- §3.3 within-detector ordering sentence rewritten to reflect the actual measured behavior (small-rich first at YOLO11s/m/l, flipped at YOLO11x).
- New `\appendix` block with `app:v8-supplement` containing `tab:gen-decomp-v8-app` and `tab:gen-gain-v8-app` and a connecting paragraph.

### Verification

- 20 of 20 row-body assertions PASS: every inline row in `paper/main_vision.tex` matches the auto-generated `paper/tables/*.tex` row body. (The assertion script reads expected rows from the tex sources rather than re-rounding from CSV, so it is robust to round-half-even ambiguities.)
- pdflatex / latexmk / tectonic / xelatex still absent on this workstation; compile attempt skipped.

### B3 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_b3_probe.py
python accv_experiments/scripts/phase_b3_v11_global8.py
python accv_experiments/scripts/phase_b3_aggregate.py
```

### Human-decision queue (B3 deltas)

1. **YOLOv11s reproducibility** — Table 1 (`step_a_baseline`) vs.\ new B3 v11s sweep disagree on medium/large; decide whether to re-publish Table 1 from the new measurement, hold pending a pinned re-measurement, or footnote both.
2. **YOLO11x flipped ordering** — confirm the §3.3 / §7 wording is comfortable with "small-rich first at low/medium capacity, flipped at the highest capacity in our family".
3. **YOLO11n TBD** — rerun once HF publishes `mobilint/YOLO11n/aries/global8/yolo11n.mxq`.
4–5. (carried) ARIES100 DRAM verify, non-YOLO INT8 export.

---

## rev 6 — single-binary per detector + relative-units primary table

### Headline

- STEP 0 probe: SDK refuses cross-mode mxq loads. Only the legacy v11s binary (`b2441f9d`) is mode-flexible.
- STEP 1 gate: v11s on the legacy mxq in global8 still does **not** reproduce Table 1's medium/large NPU AP. With identical binary + mode, the residual divergence is at the SDK / runtime layer. Table~1 is preserved as the historical anchor; the rev6 table is the current internally-consistent snapshot.
- STEP 2: 4 YOLO11 detectors × 148 cells = 592 cells in 3.0 h, every detector unified under a single mxq binary across baseline + ladder + multistream.
- STEP 3-4: `tab:gen-decomp` is now the relative-units view ($-56\%$/$-35\%$/$-21\%$ on Q for v11s, monotone across capacity); absolute companion table preserved at `paper/tables/gen_decomp_absolute.tex`. `tab:gen-gain` and `fig:gen-cstar` regenerated from rev6 measurements.
- STEP 5: CONFLICT markers updated. `[CONFLICT-MXQ]` markers (lines 391, 702) now describe the SDK-layer residual; `[CONFLICT-RELATIVE-L]` at line 281 stays because §3.3 wording is an editorial task and the diagnosis stands.
- STEP 6: 9/9 inline row assertions PASS; pdflatex/latexmk/tectonic still absent on this workstation, compile attempt skipped (no new packages introduced).

### Headline numbers (rev6, YOLO11 family)

| Detector | rel_Q_s | rel_Q_m | rel_Q_l | rel_L_s | rel_L_m | rel_L_l | worst gain |
|---|---|---|---|---|---|---|---|
| YOLO11s | $-56\%$ | $-35\%$ | $-21\%$ | $+11\%$ | $+14\%$ | $+15\%$ | $+0.025$ |
| YOLO11m | $-51\%$ | $-38\%$ | $-33\%$ | $+18\%$ | $+23\%$ | $+22\%$ | $+0.013$ |
| YOLO11l | $-47\%$ | $-41\%$ | $-35\%$ | $+22\%$ | $+26\%$ | $+27\%$ | $+0.025$ |
| YOLO11x | $-56\%$ | $-47\%$ | $-41\%$ | $+25\%$ | $+27\%$ | $+28\%$ | $+0.029$ |

C\* ordering: small-rich first for YOLO11s/m/l; large-rich first at YOLO11x (capacity-driven flip).

### Verdicts

| issue | attribution |
|---|---|
| Table 1 vs.\ rev6 v11s medium/large mismatch | SDK/runtime layer (gate FAILED under identical binary + identical mode). |
| Within-family rel_Q direction | small-concentrated (paper §3 holds). |
| Within-family rel_L direction | medium/large weighted (paper §3.3 wording mismatch — `% [CONFLICT-RELATIVE-L]` retained). |
| Within-detector $C^\star$ ordering | small-rich first at s/m/l; large-rich first at x (same as B3). |
| C1 inversion (worst-stream gain) | all 4 measured detectors invert; mean gain is the noisy small-magnitude denominator the `worst/mean` ratio depends on. |

### Human-decision queue (rev 6 deltas)

1. Table~1 reproducibility resolved AT the binary/mode layer; residual sits at SDK/runtime. Decide whether to keep Table~1 as historical anchor, re-publish from rev6, or commission a frozen-SDK reproduction.
2. §3.3 wording revision (`% [CONFLICT-RELATIVE-L]` at line 281): "$L_i(C)$ climbs fastest on small" → relative-units-friendly phrasing. rev6 measurements confirm rev 5 verdict.
3. YOLO11n TBD pending an upstream mxq.
4. (carried) ARIES100 DRAM verify, non-YOLO INT8 export.

### rev 6 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_rev6_probe.py
python accv_experiments/scripts/phase_rev6_sweep.py
python accv_experiments/scripts/phase_rev6_aggregate.py
```

---

## rev 9 — §5–6 normal-state remeasurement

**T-D0 (8/8) confirmed**: current-SDK normal state = large NPU−GPU gap ≈ −0.096 (range [−0.0984, −0.0932]). rev7's +0.0005 was an outlier. §7 (gen_decomp/gen_gain/cstar based on rev6) is canonical; old Table 1 is pre-drift historical. rev9 re-measures §5–6 under the same normal SDK state.

### Gates

| gate | result | value |
|---|---|---|
| STEP 0 normal-state gate (6 sids, NPU L0 large) | **PASS** | large gap −0.1028 ∈ [−0.110, −0.082] |
| STEP 3 REVERSAL GATE (re-measured main-comparison N=4 L1_light) | **PASS** | SizeAware 0.084 vs SizeBlindRev 0.082, worst gain +0.0020 |

### Reuse inventory

- Reuse from rev6 (normal state): Table 1 GPU+NPU L0 24-log baseline (`p1r6_baseline_yolo11s.csv`); partA GPU L1_light (`p1r6_ladder_yolo11s.csv`); main-comparison SA/SBR at L1_light Comp A (`p1r6_multistream_yolo11s.csv`).
- New rev9 measurements: Table 1 CPU L0 (24 logs); partA NPU L1_light (8 sids); per-class (24 logs × 2 dev); main-comparison Naive + Oracle + L2_LM (30 cells); schedule-shift (35 cells); capacity (24 cells); natural Comp B + C (105 rows including oracle samples).
- Total rev9 wall: 5837 s (~97 min).

### Headline deltas (`results/rev9_delta.md`)

| Cell | old paper | rev9 normal state |
|---|---|---|
| Table 1 AP_small Δ | $-0.007$ ($-47\%$) | $-0.0089$ ($-56\%$) |
| Table 1 AP_medium Δ | $-0.036$ ($-19.5\%$) | $-0.0648$ ($-35\%$) |
| **Table 1 AP_large Δ** | **$+0.001$ ($+0.2\%$ n.s.)** | **$-0.1000$ ($-21\%$)** — **'unaffected' refuted** |
| main-cmp L1 inversion (SA−SBR) | $+0.021$ ($+25\%$) | $+0.0020$ ($+2.4\%$) — sign preserved |
| main-cmp L2_LM inversion | $+0.012$ ($+23\%$) | $+0.0233$ ($+79\%$) |
| worst/mean ratio L1 (SA−Naive) | $+1.75\times$ | $+0.95\times$ (sign preserved but ratio ≈ 1) |
| worst/mean ratio L2 | $+3.5\times$ | $+0.31\times$ |
| Decomposition quant share | $97\%$ | $168\%$ — quant exceeds total NPU-path loss (small-N effect at this sid) |

### Paper artifacts regenerated

- 8 new auto-generated tables in `paper/tables/`: `single_stream.tex`, `per_class.tex`, `partA.tex`, `main_comparison.tex`, `schedule_shift.tex`, `capacity.tex`, `natural.tex`, `decomposition.tex`.
- 4 `% [rev9 STATUS]` fact-only comments appended next to the existing SEMANTIC-LOCK markers in `paper/main_vision.tex`. Prose left untouched.
- `results/rev9_PROPOSED_EDITS.md` lists the 5 SEMANTIC-LOCK rows that need human review with old → new values and impact.

### Verification

- 7 of 7 cell assertions PASS: every cell in the auto-generated tables matches its source CSV.
- pdflatex / latexmk / tectonic / xelatex absent on PATH; compile attempt skipped (new tables use only booktabs primitives already in use).

### Updated human-decision queue (rev 9 deltas)

1. **Table 1 large 'unaffected' narrative** — rev9 confirms large NPU−GPU gap is $-21\%$, not $+0.2\%$ n.s. Editor decides whether to (a) retire the "large vehicles effectively unaffected" wording in §5.1, (b) keep the paper's Table 1 as a historical snapshot and add a footnote citing rev9, or (c) re-publish Table 1 from rev9.
2. **C1 magnitude rewrite** — `+25%/+23%` no longer match rev9 (`+2.4%/+79%`). Reversal still holds; only the magnitudes need editing. Action item in `rev9_PROPOSED_EDITS.md` Lock row 3.
3. **`1.75–3.5×` worst/mean ratio** — both ratios shrank substantially; one is now near $1\times$. Editor decides whether to (a) re-derive the ratio from a different baseline pair (SizeAware vs SizeBlindRev, AllNPU vs Naive), or (b) revise the narrative.
4. **Decomposition $97\%$** — quantization still dominates the NPU path, but the printed share depends on which worst-camera sid is chosen. Editor may keep the qualitative claim and replace the precise percentage from `results/rev9_decomp.md` once the worst-sid choice is fixed.
5. (carried) ARIES100 DRAM verify; non-YOLO INT8 export; YOLO11n TBD.

### rev 9 reproduction

```bash
source .venv/bin/activate
python accv_experiments/scripts/phase_rev9_sweep.py        # ~97 min
python accv_experiments/scripts/phase_rev9_aggregate.py    # seconds
```

Manifest: `results/manifest_rev9.json` (idempotent resume; per-cell checkpoint).
