# HUMAN_QUEUE — items requiring author decision

Listed in priority order. Each item lists the evidence CSV and the prose lines
(if any) it affects. Automatic loop will not edit these without human approval.

## P0 — pending T0 outcome
- **REVERSAL GATE (rev7)** — autonomous loop will report PASS/FAIL after rev7 main-comparison completes. If FAIL: thesis-level reassessment required before any §5–6 prose update.

## P1 — semantic-change candidates (auto-populated by T2 firewall)
- (none yet — pending rev7 numeric deltas)

## P2 — carried over from rev 5/6/7
1. **Table 1 reproducibility under current SDK** — rev6 confirmed SDK-layer drift on top of mxq drift. Decide whether to (a) keep historical Table 1 as snapshot and reference rev7 as canonical, (b) re-publish Table 1 from rev7, or (c) commission a frozen-SDK reproduction. Evidence: `results/v11s_repro_rev6.csv`, `results/rev7_pin.json`, `results/rev7_single_stream.csv` (pending).
2. **§3.3 "L climbs fastest on small" wording** — rev5/6 confirmed relative_L is medium/large weighted. `% [CONFLICT-RELATIVE-L]` at main_vision.tex line 281 awaits human edit. Evidence: `results/relative_QL.csv`, `results/relative_QL_v5.csv`.
3. **YOLO11n TBD** — pending upstream `mobilint/YOLO11n` mxq publication. T4 will probe HF every 4 cycles. Evidence: `results/v11_mxq_probe.csv`, `results/v11n_probe_history.csv` (pending).
4. **ARIES100 on-board DRAM capacity** — `% [VERIFY-HW]` memory note in main_vision.tex §4.1. Human-side datasheet verification required.
5. **Non-YOLO INT8 export (RT-DETR / PicoDet)** — no qb toolchain on PATH on this workstation. Carries to future work.

## Q (sticky context)
- Autonomous loop will refuse to edit prose lines flagged with `% [SEMANTIC-LOCK: ...]` and route any number change that would flip a sign or break the lock to this queue.
- Binary pin: `b2441f9d` (legacy v11s). SDK pin: rev6 mblt_model_zoo. These pins are inherited by every rev7+ measurement so a third party can reproduce.

## P0 (NEW — auto-loaded by T3 rev7_delta analysis, 2026-06-02)

### A — Worst/mean ratio sign flip at L1_light (semantic-lock breach)

- **Source**: `results/rev7_delta.md`, line "Worst-vs-mean ratio".
- **Paper claim** (abstract line 90, intro line 137, conclusion line 851):
  `worst-stream sAP gap is 1.75–3.5× larger than mean-stream sAP gap`.
- **rev7 measurement**: at L1_light the *mean* gain (SizeAware − Naive) is now **negative** (−0.007), making the ratio negative (−0.11×). At L2_LM the mean gain is also negative (−0.022). The C2 "worst > mean by 1.75-3.5×" narrative no longer holds in the rev7 sweep on the current SDK.
- **Auto-loop will NOT edit**. Human decision required:
  - (a) Re-derive the worst/mean ratio under a different baseline pair (e.g., SizeAware vs SizeBlindRev rather than vs Naive),
  - (b) revise the paper claim to reflect rev7,
  - (c) flag rev7's mean-gain anomaly as measurement noise and re-measure.
- Evidence: `results/rev7_main_cmp.csv` rows for L1_light/L2_lm × {Naive,SizeAware}.

### B — Inversion magnitude shrunk at L1_light (semantic-lock partial breach)

- **Paper claim** (§5.3 line 543): `SizeAware improves worst-stream sAP over SizeBlindRev by +25% at L1_light (0.105 vs 0.084) and +23% at L2_LM (0.065 vs 0.053)`.
- **rev7**: L1_light now `0.089 vs 0.083` → **+7%** (still positive; reversal preserved). L2_LM: `0.051 vs 0.028` → **+84%** (much larger than +23%, but SizeBlindRev at L2_LM crashed to 0.028 in rev7 — investigate whether this is real or a corner-case).
- Sign preserved (reversal still holds), but the magnitudes shifted materially. Per SEMANTIC-LOCK rule, auto-loop will not substitute these four numbers in §5.3. Human decision:
  - keep the +25% / +23% narrative as a historical snapshot and add a rev7 footnote, or
  - rewrite §5.3 around the rev7 magnitudes (+7% / +84%).
- Evidence: `results/rev7_main_cmp.csv`, `results/rev7_delta.md`.

### C — Quantization share moved from 97% to 100% on the NPU path

- **Paper claim** (abstract line 92, §5.7 caption line 793): `97% of the residual loss is INT8 quantization`.
- **rev7 measurement**: 100% (within rounding). The narrative direction is preserved (quantization dominates the NPU path), but the precise number has shifted. Auto-loop will NOT auto-edit because the value is under SEMANTIC-LOCK; a human one-line change (`97%` → `100%`) is trivial and likely safe.
- Evidence: `results/rev7_decomp.csv` derived in `results/rev7_delta.md`.

### D — SDK / driver state divergence between rev6 and rev7 measurements

- Same `b2441f9d` mxq binary, same `mblt_model_zoo` checkout, but rev6 sweep at 14:14 today produced Δ_large = −0.10 sAP while rev7 sweep at 17:45 reproduced Δ_large = +0.0005 sAP (matching Table 1). The NPU output drifted within the same session without any visible code change.
- Likely cause: NPU driver / runtime state (warm-up, throttling, firmware) — not the .mxq file. Reproducibility of the measurement on a third-party machine requires pinning the driver+firmware as well as the mxq and SDK.
- Recommendation: record this finding in §4 (Setup) and treat rev7 as the canonical snapshot.
