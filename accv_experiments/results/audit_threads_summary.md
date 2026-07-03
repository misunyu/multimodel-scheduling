# EXP-AUDIT-THREADS — Table 2 (tab:decomp) thread-lever settings (read-only)

Source of Table 2: `accv_experiments/scripts/phase_rev18_stage1.py` →
`results/rev18_postproc_levers.csv` (host post-processing thread lever; bit-identity from
rev22 `rev22_bitident.csv`). No measurement run.

## Q1/Q2 — thread settings (confirmed, code + stored data)
- Lever param: `torch.set_num_threads(nt)` (process-level), `nt` from `THREAD_SETTINGS=[24,4]`
  (`phase_rev18_stage1.py:48`, applied at `:76` and `:86`).
- **Staleness OFF (frame skip 0)**: **threads = 4**. Stored: `npu_skip_pct = 0.0%`,
  `npu_lat ≈ 10.1 ms` (< 33.3 ms budget) — `rev18_postproc_levers.csv` rows `threads=4`.
- **Staleness ON (frames dropped)**: **threads = 24**. Stored: `npu_skip_pct ≈ 54–56%`,
  `npu_lat ≈ 36 ms` (> 33.3 ms budget) — rows `threads=24`.

Direction note: staleness is turned ON by **raising** threads 4→24 (not lowering). Per the rev18
header/report, the ~27 ms NPU host post-process is torch CPU-thread thrashing on tiny YOLO11 head
tensors: `set_num_threads(24)` → ~36 ms (misses budget, drops frames); `=4` → ~10 ms (keeps up,
0% skip). Same math, only thread count.

## Q3 — only thread count varies (bit-identical consistent)
In `rev18_postproc_levers.csv`, across both settings the GPU per-size sAP is byte-identical
(`gpu_sap_s/m/l = 0.0159 / 0.1839 / 0.4768` in every row), same detector (yolo11s), same 24 logs,
same 3 reps, same conf/iou. Only `torch.set_num_threads` differs. Per-processed-frame NPU detections
are bit-identical across thread counts (rev18 report; independently verified in
`rev22_bitident.csv`: 20 frames, max box/score diff = 0). The thread count changes only delivery
timing (frame skip), not the detections — consistent with the paper's bit-identical claim.

## Table 2 number provenance (matches)
- Quantization row (staleness off = threads=4, skip 0): small (0.0082−0.0159)/0.0159 = **−48.4%**,
  medium (0.1477−0.1839)/0.1839 = **−19.7%**, large +0.0005/0.4768 = **+0.1%** (`large_gap=+0.0005`).
- Staleness row (Δ = threads=24 − threads=4): large npu_sap_l 0.378−0.477 = **−0.099 ≈ −0.098**,
  small ≈ 0 (−0.0013 per `rev18_stage1_report.md:45`).
Both rows reproduce the Table 2 values (−48.4% / +0.1% / −0.098).

## Fact statement (for separate channel; not LaTeX)
The staleness-decomposition uses host post-processing threads = 4 for the staleness-off setting
(NPU frame skip 0%, ~10 ms) and threads = 24 for the staleness-on setting (~55% frame skip, ~36 ms);
only the thread count changes and the per-frame detections are bit-identical.

## Compliance
Read-only; code lines + stored CSV/log only; no re-run; off/on values cited from code
(`THREAD_SETTINGS=[24,4]`) and confirmed by stored skip data; paper .tex untouched. Output = this file.
