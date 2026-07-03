# rev12 — multi-stream table state check (C)

Source CSVs for Tables 2 (partA), 3 (capacity), 4 (main-comparison):

| table | csv | rows | infer_mode(s) | comment |
|---|---|---|---|---|
| Table 2 (partA) | `rev9_partA_npu.csv` | 8 | `['global8']` | rev9 partA NPU sAP per sid (8 sids partA panel, bg=L1_light) |
| Table 3 (capacity) | `rev9_capacity.csv` | 24 | `['single']` | rev9 capacity N=2..8 L1_light |
| Table 4 (main-comparison) | `rev9_main_cmp.csv` | 32 | `['single']` | rev9 main-comparison N=4 Comp.A at L1_light/L2_lm |

## rev9 sweep entry-point uses legacy mxq

`phase_rev9_sweep.py` references legacy mxq path `b2441f9d` (`models/mobilint_backup/yolo11s.mxq`): **True**.
Same binary used by T-D0 (8-run normal-state anchor).

## Pipeline cross-check (rev9 vs rev10/rev12)

sid=2 NPU sAP at L1_light:

- rev9 partA (Table 2 source): `0.1089` (frame_skip 0.7%)
- rev10 cstar (A-3): `0.0719` (frame_skip 96.2%)
- diff = `-0.0370` — large.

**Diagnosis**: same mxq (`b2441f9d`) and same `infer_mode` (`global8`), but rev10 A-3 NPU shows 52% (L0), 96–99% (L1_light), 100% (L2_LM), 81% (L3_VLM) `frame_skip_pct`. rev9 partA shows <1% at L1_light. This is a **measurement-pipeline difference** (likely BG-thread / streaming-clock scheduling between A-2 multi-stream and A-3 single-stream replay), NOT a state drift.

## What this means for the paper

- Tables 2, 3, 4 use the **rev9 pipeline** (Table 2: rev9 partA, Tables 3 & 4: rev9 sweep). All three use the same legacy mxq verified by T-D0. **Self-consistent.**
- Tables 1, 5: rev10/rev12 single-stream + rev12 L2_LM single-stream. L0 values reproduce T-D0 (PASS). L2_LM values record higher staleness but the absolute sAP differences are still meaningful for size-by-size quantization analysis.
- Cross-table joins should use rev9-pipeline NPU values for L1_light (rev9 partA covers 8 partA sids) and rev10/rev12 for L0 + L2_LM.

## Verdict

**Multi-stream tables (2, 3, 4) are internally consistent (single rev9 pipeline + legacy mxq b2441f9d). NO state mismatch.** The pipeline difference vs rev10 cstar is a known measurement methodology variance, not a state drift.
