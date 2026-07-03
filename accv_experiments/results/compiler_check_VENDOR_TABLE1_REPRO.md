# Vendor INT8 (b2441f9d) — Table 1 reproduction at the correct operating point

Highlighted separately per request. This is the clean re-measurement of the PAPER's vendor NPU
blob, used to validate the EXP-FT-COMPILER-CHECK harness — NOT the self-compiled model.

Protocol: NPU isolated single-stream, 24 Argoverse-HD logs, **threads=4 (frame skip 0.0%)**, 3 reps,
mode=global8. Model: `models/mobilint_backup/yolo11s.mxq` (sha256 b2441f9d…), the Table 1 source.

| size | vendor re-measure | published Table 1 (NPU) | |Δ| |
|---|---|---|---|
| AP_small  | 0.0082 | 0.009 | 0.0008 |
| AP_medium | 0.1477 | 0.148 | 0.0003 |
| AP_large  | 0.4773 | 0.478 | 0.0007 |
| (all)     | 0.1859 | —     | —      |
| skip %    | 0.0    | ~0    | —      |
| infer ms  | 10.1   | 10.0  | —      |

→ Vendor blob reproduces published Table 1 within Δ ≤ 0.001 on every size. The harness and
operating point are validated; the published Table 1 NPU figures are confirmed reproducible.

Caveat for reuse: `p1r6_baseline_yolo11s.csv` NPU rows are a 24-thread misconfiguration
(skip ~55%, staleness-contaminated large=0.377) and must NOT be cited as the Table 1 NPU values.
The correct, threads=4 figures are the ones above (and in `compiler_check_results.csv`,
mxq_label=vendor_b2441f9d).

Source rows: `compiler_check_results.csv` (mxq_label=vendor_b2441f9d, 72 rows).
