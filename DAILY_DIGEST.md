# DAILY_DIGEST — autonomous-loop activity log

Append-only. Each cycle adds a section.

## Cycle 0 (initialization)

- Initialized `PROJECT_STATE.json` with the rev8 task graph (T0–Tn).
- Pinned binary: `b2441f9d` legacy v11s mxq (`models/mobilint_backup/yolo11s.mxq`).
- Pinned SDK: `mblt_model_zoo` at `/home/msyu/PycharmProjects/MobilintTest/mblt-model-zoo/`.
- REVERSAL GATE pre-evaluated from rev6 data: PASS (v11s N=4 L1_light worst gain +0.025).
- T0 (rev7 §5–6 re-measurement) selected as first cycle.
- Carried HUMAN_QUEUE items from rev 5/6: Table 1 reproducibility decision, §3.3 wording revision, YOLO11n TBD, ARIES100 DRAM verify, non-YOLO export.

## Cycle 1 (T1 + T4)

**Selected tasks**: T1 (one-shot SRC tagging) and T4 (v11n probe, low-cost, no deps).

**T1 — SRC tagging (DONE)**:
- 24 dependency_map.csv entries covering the highest-impact §5-6 numerics (abstract, §3, §5.1, §5.3, §6 worst/mean, decomposition).
- 4 `% [SEMANTIC-LOCK]` markers inserted in `paper/main_vision.tex` (above Table 1 large row at line ~412; §5.1 three-size narrative at line ~418; §5.3 +25% inversion at line ~545; tab:decomposition caption at line ~793). These prevent T2 from auto-substituting flip-risk claims.

**T4 — v11n probe (DONE this cycle)**:
- `results/v11n_probe_history.csv` updated. HF still returns 404 for `mobilint/YOLO11n/aries/global8/yolo11n.mxq`. Retry scheduled at cycle 5 (cadence=4).

**T0 (rev7 sweep) RUNNING**: Table 1 phase in progress (NPU sid=3 of 24 done at last check).

**T2 substitution script** prepared at `accv_experiments/scripts/phase_rev7_substitute.py`. Conservative first pass: PROPOSE / QUEUE_LOCKED only (no in-place edits) — actual substitution will be a follow-up cycle after author reviews the proposal log.

**Pending T0 completion**: T0-gate (REVERSAL GATE check), then T2/T3/T7/T8/T5/T6/Tn unblock in dependency order.

## Cycle 2 (T0 done → T0-gate → T3 → T2-propose → T7)

**T0 completed**: rev7 §5–6 sweep finished in 8892 s (~2.5 h). 8 CSVs produced (435 rows total).

**T0-gate (REVERSAL GATE): PASS**
- SizeAware worst 0.089 vs SizeBlindRev 0.083; worst gain +0.0058 → C1 inversion preserved on current SDK.

**T3 (rev7_delta.md) generated** — `accv_experiments/results/rev7_delta.md`. Headline deltas:
- Table 1 reproduces (small −48% vs paper −47%; medium −19.7% vs −19.5%; large +0.1% vs +0.2% n.s.). ✓ Three-size narrative HOLDS on rev7.
- Decomposition reproduces: Quant share 100% (paper 97%), GPU-path loss −0.059 (paper −0.060), Quant delta −0.031 (paper −0.031). ✓ 97% claim still operationally true.
- Capacity N=8 row reproduces within rounding (0.058/0.059/0.076/0.083 vs paper 0.055/0.061/0.081/0.083).
- AllNPU N=8 L1 saturation ceiling 0.084 (paper 0.083). ✓
- **NEW: Main-comparison inversion +25% at L1_light shrunk to +7%** (still PASS C1, but magnitude meaningfully smaller).
- **NEW: Worst/mean ratio (1.75–3.5×) sign-flipped**: at L1_light SA's mean gain over Naive turned −0.007; ratio is now −0.11×, breaking the 1.75–3.5× narrative.

**T2 (substitution) PROPOSE-only** — Risk of regex-based in-place edits on `% [SEMANTIC-LOCK]`-tagged lines is too high. Per spec we log proposals only; in-place edits deferred to human review. Proposals consolidated into `results/rev7_delta.md` + HUMAN_QUEUE P0 A/B/C/D.

**T7 (consistency audit) PASS** — `results/consistency_audit_rev8.csv`. Highlights:
- rel_Q ordering small (48%) > medium (20%) > large (0.1%) — monotone ✓
- worst ≤ mean across multistream tables ✓
- Oracle ≥ best deployable strategy at L1_light and L2_lm ✓
- Decomposition components sum to within 5 mAP of total NPU-path loss ✓
- rev6-vs-rev7 v11s on identical binary diverged by ~0.05–0.10 sAP on medium/large (driver/runtime state) — recorded as WARN-DIVERGENCE.

**HUMAN_QUEUE P0 items added (4)**: A (worst/mean ratio sign flip), B (inversion magnitude shrunk), C (97%→100% one-line edit), D (rev6/rev7 driver-state divergence). Each item lists evidence CSV + recommended actions.

**Tasks remaining (auto_allowed READY)**: T4 next at cycle 5 (cadence 4); T5/T6/T8 soak (low priority); Tn after human reviews T2 proposals. **Cycle ends here**: no urgent auto-actions remain that don't touch SEMANTIC-LOCKed content. Loop pauses for human review of HUMAN_QUEUE.
