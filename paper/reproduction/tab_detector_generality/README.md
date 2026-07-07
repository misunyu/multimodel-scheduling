# Reproduction: Table `tab:detector-generality`

**Paper label:** `tab:detector-generality`
**Paper caption:** "Cross-detector reproduction of the placement reversal. The table
summarizes isolated size-bin preference, VLM worst-stream sAP, and the contention-sweep
crossover point."
**Paper location:** Limitations / Generality discussion, Sec. 5.2 (around line 489–509 of
`paper/main_vision.tex`).

This table demonstrates that the isolated-vs-contention placement reversal (headline result
on YOLOv11s) is not an artifact of one detector: the same pattern reproduces across five YOLO
variants — YOLOv12s, YOLOv11s, YOLOv10s, YOLOv8s, YOLOv8n — with detector-dependent
large-object INT8 loss and crossover point.

## Inputs (`./data/`)

Per-detector CSVs copied verbatim from `accv_experiments/results/`:

- `rev27_yolo12s_{single_stream,contention,sweep}.csv`
- `rev27_yolov10s_{single_stream,contention,sweep}.csv`
- `rev27_yolov8s_{single_stream,contention,sweep}.csv`
- `rev26_yolov8n_{single_stream,contention,sweep}.csv`
- `rev19_table1_threads4.csv` — pooled single-stream eval used for the YOLOv11s (main) row.

Input meaning:
- `*_single_stream.csv`: isolated per-size (small/medium/large) GPU vs NPU sAP with
  `rel_gap_pct = (npu-gpu)/gpu` in percent (negative = GPU better).
- `*_contention.csv`: worst-stream sAP per background co-tenant (`L1_light`/`L2_lm`/`L3_vlm`)
  and strategy (`All-GPU`/`All-NPU`). Only the `L3_vlm` (VLM co-tenant) rows feed this table.
- `*_sweep.csv`: contention sweep of worst-stream sAP vs `gpu_skip` for `All-GPU` (and the
  flat `All-NPU` reference), used to interpolate the crossover point.
- `rev19_table1_threads4.csv`: 24-log pooled GPU/NPU sAP by size; the YOLOv11s row values are
  hardcoded constants in the generator (from `tab:single-stream` / `tab:main`), so this file
  is copied for completeness / provenance but the main-row numbers are not recomputed from it.

## Run

```
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs

- `detector_generality.tex` — the LaTeX `tabular` for the table.
- `detector_generality.csv` — per-detector computed values (raw precision).

## Transform

For each non-main detector:
- **isolated small / large** = `rel_gap_pct` for the `small` / `large` rows of
  `*_single_stream.csv`. Rendered as `GPU (x%)` when GPU is better and significant, or `tie`
  for near-zero (|rel| < 5) large-object gaps.
- **VLM worst NPU/GPU** = mean `worst_sap` over the `L3_vlm` rows for `All-NPU` and `All-GPU`
  in `*_contention.csv` (reversal: NPU >> GPU under the VLM co-tenant).
- **crossover %** = the `gpu_skip` value where the `All-GPU` worst-sAP curve in `*_sweep.csv`
  crosses the flat `All-NPU` level, linearly interpolated between bracketing sweep points.

The YOLOv11s (main) row uses hardcoded constants from `tab:single-stream` / `tab:main`
(`small=-48.4`, `large=+0.1`, VLM `0.083/0.016`, crossover `48.0`).

## Verification vs paper `tab:detector-generality`

All underlying data values match the paper table exactly:

| Detector | isolated small | isolated large | VLM worst NPU/GPU | crossover | verdict |
|----------|---------------|----------------|-------------------|-----------|---------|
| YOLOv12s | GPU (-36%)    | -1.2%          | 0.080 / 0.011 (=8.0/1.1) | ~45% | MATCH |
| YOLOv11s | GPU (-48%)    | +0.1%          | 0.083 / 0.016 (=8.3/1.6) | ~48% | MATCH |
| YOLOv10s | GPU (-45%)    | -4.6%          | 0.082 / 0.016 (=8.2/1.6) | ~30% | MATCH |
| YOLOv8s  | GPU (-44%)    | -0.3%          | 0.080 / 0.015 (=8.0/1.5) | ~31% | MATCH |
| YOLOv8n  | GPU (-36%)    | -8.1%          | 0.060 / 0.009 (=6.0/0.9) | ~36% | MATCH |

**Verdict: MATCH** (all five detectors).

### Cosmetic rendering notes (not data mismatches)

The `detector_generality.tex` emitted by the generator differs from the paper's committed,
hand-finalized table only in display formatting — these differences are produced by the
**source** generator (`paper/results_data/regen_table_detector_generality.py`, copied verbatim)
and are not introduced by this reproduction:

1. **VLM column scaling:** generator prints raw sAP `0.080/0.011`; paper prints ×1000-style
   `8.0/1.1`. Same numbers.
2. **Percent precision:** generator `sig_label` uses integer `%` (e.g. `-1%`, `-5%`); paper
   shows one decimal (`-1.2%`, `-4.6%`). Same underlying `rel_gap_pct` (in the CSV at full
   precision).
3. **YOLOv10s large label:** generator renders `tie` because `|−4.6| < 5`; the paper prints
   `GPU (-4.6%)`. The value (−4.6%) is identical; only the tie/GPU threshold label differs.
4. **Detector name:** generator `DETS` uses `YOLO12s`; paper prints `YOLOv12s`.

The full-precision data is in `detector_generality.csv` and matches the paper cell-for-cell.

### `isolated large` column — exact notation-rule difference

This is the one column where the generated `.tex` and the paper table can disagree on the
**label** (not the value). The two use different rules:

| | rounding of the % | GPU / tie decision rule |
|---|---|---|
| **Generator** (`sig_label`) | `rel_gap_pct` rounded to **integer** (`f"{x:.0f}%"`) | label `tie` when `abs(rel) < 5`, else `GPU`/`NPU` by sign |
| **Paper** (hand-finalized) | shown to **one decimal** (`-4.6%`) | uses a **stricter/statistical** notion of "tie" (near-zero *and* n.s.), so a `-4.6%` gap is written `GPU (-4.6%)` |

Consequences for the large column:
- **Value is always identical** (`rel_gap_pct` in `detector_generality.csv`: −1.2 / +0.1 / −4.6 / −0.3 / −8.1).
- **Rounding** makes the generator print `-1%` / `-0%` / `-5%` / `-8%` where the paper prints `-1.2%` / `-0.3%` / `-4.6%` / `-8.1%`.
- **Label** diverges for **YOLOv10s only**: generator `tie` (because `|−4.6| < 5`) vs paper `GPU (−4.6%)`. All other rows agree on the label.

(The `isolated small` column uses the same integer rounding but the paper also rounds small to
integers, so small labels/values already coincide: −36 / −48 / −45 / −44 / −36.)

### ✅ Action item — camera-ready notation unification (표기 통일 검토)

**For camera-ready, unify the `isolated large` notation** between the generator and the paper
table so they render identically: pick one rule for (a) percent precision (1 decimal is what the
paper uses) and (b) the tie/GPU threshold (the paper's stricter rule, under which YOLOv10s is
`GPU (−4.6%)`). This is a **display-only** reconciliation — the underlying values already match,
so no re-measurement is needed. `main_vision.tex` is intentionally **not** modified here.
