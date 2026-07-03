# EXP-NORM v2 — Environment Snapshot (pre-start)

Recorded after EXP-QOS v2 completion.

## Gate (EXP-NORM precondition)
- EXP-QOS v2 artifacts present: `exp_qos_results.csv`, `exp_qos_summary.md` ✓
- MPS-MINI was NOT executed (Outcome C, trigger not met) → no MPS daemon to clean.
- `CUDA_MPS_*` env vars: never set this session.
- GPU compute processes at start: **0** (verified `nvidia-smi --query-compute-apps`).
- MPS daemon: **absent** (verified `ps aux | grep mps`).

## Config (unchanged from EXP-QOS / rev22 baseline)
- driver 580.159.03 / CUDA 13.0, torch 2.12.0+cu130, ort 1.20.1, ultralytics 8.4.56 (no drift).
- N=4 panel = [2,22,3,21] (same logs as Table 5 / tab:metric-sensitivity).
- threads=4, 30-frame warm-up + log window, 3 reps.
- High-contention point = resnet_k=8 (rev22 SWEEP8, GPU skip 80.3%) == Table 5 "GPU skip ≈80%".
- **No protection mechanism (PRIO/MPS off).** All measurements plain.

## Table-5 identity confirmation
tab:metric-sensitivity worst-stream = All-GPU 0.066 / All-NPU 0.084 == rev22 k8 means
(0.0663 / 0.0836). Table 5 high-contention condition == resnet_k=8. No mismatch → proceed.
