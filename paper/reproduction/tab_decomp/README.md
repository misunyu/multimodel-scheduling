# Reproduction: Table `tab:decomp` (paper Table "Quantization vs. Staleness")

Paper label: `tab:decomp`

Caption (paper):
> Quantization versus staleness losses by object size (YOLOv11s, single stream).
> The quantization row is the NPU--GPU gap at 4 post-processing threads from
> Table `tab:single-stream`, where deadline misses are negligible. The staleness
> row is the change on the same NPU path when the thread count increases from 4
> to 24.

Paper location: Section 4.1 ("Isolated Evaluation Mainly Captures Quantization
Loss"), immediately after Table `tab:single-stream` (Table 1). This is the
loss-decomposition table.

## Inputs (in `./data/`)

- `rev19_table1_threads4.csv` — per-size sAP for the GPU (FP32) and NPU (INT8)
  paths at `threads=4` (deadline misses negligible), 3 reps each. Columns used:
  `device` (GPU/NPU), `sap_s`, `sap_m`, `sap_l`.
- `rev18_postproc_levers.csv` — NPU per-size sAP at `threads=4` and `threads=24`
  (the 24-thread setting induces ~55% deadline misses while predictions stay
  fixed, isolating pure delivery-time staleness), 3 reps each. Columns used:
  `threads`, `npu_sap_s`, `npu_sap_m`, `npu_sap_l` (plus `gpu_sap_*` for the
  FP32 baseline A_b, which is identical to the GPU row of rev19).

## Run

```
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs (written next to `generate.py`)

- `table2.tex` — the LaTeX `tabular` for the paper cells.
- `decomp.csv` — per (mechanism, size) row: `dsap` (raw ΔsAP, 3 dp),
  `dsap_raw` (4 dp), `rel_pct`, `A_b_gpu` (FP32 GPU baseline), `source`.
- `*_stale_backup.*` — one-time backups of any pre-existing output.

## Transform (computation, unchanged from source)

For each object size b ∈ {small, medium, large}:

- **Quantization** = NPU(4t) − GPU(4t), per-size sAP mean over reps
  (from `rev19_table1_threads4.csv`).
- **Staleness** = NPU(24t) − NPU(4t), per-size sAP mean over reps
  (from `rev18_postproc_levers.csv`).
- **rel%** = 100 × (raw ΔsAP numerator) / A_b, where A_b is the FP32-GPU
  per-size sAP baseline (`g.sap_*.mean()`).

The stored `dsap` is the raw sAP delta (e.g. `-0.008`). The paper prints these
as **points (sAP ×100)**, so `-0.008` displays as `-0.8`, `-0.098` as `-9.8`,
etc. The `rel%` values are identical between this package and the paper.

## Verification vs. paper Table `tab:decomp`

| Component | Size | Paper (pts, rel%) | Reproduced (dsap ×100, rel%) | |
|-----------|------|-------------------|------------------------------|---|
| Quantization | small  | −0.8 (−48.4%) | −0.8 (−48.4%) | MATCH |
| Quantization | medium | −3.6 (−19.7%) | −3.6 (−19.7%) | MATCH |
| Quantization | large  | +0.1 (+0.1%)  | +0.1 (+0.1%)  | MATCH |
| Staleness    | small  | −0.1 (−8.0%)  | −0.1 (−8.0%)  | MATCH |
| Staleness    | medium | −3.0 (−16.0%) | −3.0 (−16.1%) | MATCH* |
| Staleness    | large  | −9.8 (−20.5%) | −9.8 (−20.5%) | MATCH |

Verdict: **MATCH**.

\* Medium-staleness rel% rounding note (also emitted by the script's own
reconciliation check): the full-precision numerator gives
100 × (−0.02953) / 0.1839 = **−16.059% → −16.1%**, while the paper prints
**−16.0%** (rounding a 4-dp-rounded numerator, −0.0295, before the ratio). This
is a 0.1pp display-rounding artifact; the ΔsAP point value (−3.0) is identical.
