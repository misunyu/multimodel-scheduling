# EXP-CROSSOVER-COLLAPSE — Summary

**Question.** Each detector's All-GPU→All-NPU worst-stream crossover sits at a *different* raw
contention level (GPU frame-skip ≈ 29/31/36/44/48%). Do these crossovers **collapse onto a
universal normalized point L(C*)/Q ≈ 1** — i.e. "NPU wins once staleness loss L equals the
quantization deficit Q"?

**Verdict: NOT SUPPORTED (no clean collapse) when Q is the independent isolated deficit.**
With Q = the load-independent *single-stream* NPU deficit, the normalized crossover
L(C*)/Q spans **0.81–2.66 (CV 44%)** — wider in relative terms than the raw skip spread
(CV 19%). Normalizing by the isolated Q does **not** concentrate the crossovers. The crossover
collapses onto L/Q≡1 **only** if Q is taken as the contention-aware N=4 worst-stream deficit
(D_sweep) — but that identity is **tautological** (L(C*) ≡ D_sweep by construction at the
crossover) and is exactly the trap the design forbids.

**Why this matters (reinforces the paper thesis).** The isolated single-stream deficit does
*not* predict where the multi-camera crossover happens: N=4 worst-stream aggregation amplifies
the NPU deficit over its isolated value by a **detector-dependent 1.0–2.7×**. You cannot read
the deployment crossover off an isolated benchmark — which is the core claim of the paper.

---

## STEP 0 — data-availability gate → **Path A** (no new measurement)
All five detectors already have BOTH a full worst-stream contention sweep AND an isolated
single-stream measurement on disk. Nothing launched; HARD-CONSTRAINT (no silent 5-detector
sweep) is moot.

| detector | sweep (All-GPU/All-NPU vs skip) | isolated single-stream |
|---|---|---|
| v8n  | `rev26_yolov8n_sweep.csv` | `rev26_yolov8n_single_stream.csv` (+ stdout all-size) |
| v8s  | `rev27_yolov8s_sweep.csv` | `rev27_yolov8s_single_stream.csv` (+ stdout) |
| v10s | `rev27_yolov10s_sweep.csv` | `rev27_yolov10s_single_stream.csv` (+ stdout) |
| v12s | `rev27_yolo12s_sweep.csv` | `rev27_yolo12s_single_stream.csv` (+ stdout) |
| v11s | `rev22_vlm_sweep.csv` (N=4 rows) | `rev12_single_stream_overall.csv` |

## Definitions (tautology-safe)
- **Q_iso** = isolated, load-independent NPU deficit = sAP_GPU − sAP_NPU at **N=1, all-size**
  (NOT from the contention sweep). For v8n/v8s/v10s/v12s recovered from the step-1 pycocotools
  `area=all` blocks in stdout (streaming-sAP block = phase-0 GPU / phase-2 NPU), **validated**
  against the saved per-size CSV means (all match < 0.003). For v11s from `rev12` overall CSV.
- **L(C)** = staleness loss = sAP_GPU^worst(0) − sAP_GPU^worst(C), GPU-only, from the sweep.
- At the crossover C* (where All-GPU^worst = All-NPU^worst, flat), L(C*) ≡ D_sweep =
  sAP_GPU^worst(0) − NPU_flat. This D_sweep is the N=4 worst-stream C=0 deficit.
- Test statistic **R = L(C*)/Q_iso = D_sweep / Q_iso**; collapse ⟺ R ≈ 1 for all detectors.

## Results
| detector | GPU^worst(0) | NPU_flat | D_sweep=L(C*) | crossover skip* | Q_iso (N=1,all) | **R=D/Q** |
|---|---|---|---|---|---|---|
| v8n  | 0.0764 | 0.0604 | 0.0160 | 36.3% | 0.0153 | **1.04** |
| v8s  | 0.1025 | 0.0799 | 0.0226 | 30.6% | 0.0132 | **1.72** |
| v10s | 0.1034 | 0.0817 | 0.0217 | 29.2% | 0.0174 | **1.25** |
| v12s | 0.1063 | 0.0808 | 0.0255 | 43.8% | 0.0096 | **2.66** |
| v11s | 0.1149 | 0.0836 | 0.0313 | 48.0% | 0.0387 | **0.81** |

- Raw crossover skip* range 29–48% (CV 19%). **R range 0.81–2.66 (CV 44%)** — no collapse.
- 4 single-harness detectors (v8n/v8s/v10s/v12s): R = 1.04–2.66, all ≥1, mean 1.67 (CV 37%):
  worst-stream amplifies the isolated deficit 1.0–2.7×.

## Caveats (honest)
1. **Tautology guard (central).** L(C*)/Q ≡ 1 is automatic if Q := D_sweep. The non-trivial test
   uses an *independent* Q (single-stream); under it, collapse fails. Do not report the trivial 1.
2. **Flatness.** All-NPU^worst is load-stable (flat across the sweep); All-GPU^worst is monotone
   in skip — so C* is well defined and interpolated linearly between grid points.
3. **v11s cross-harness.** Its Q_iso (0.0387) comes from a separate campaign (rev12), the other
   four from one rev26/27 harness; v11s is the only R<1 point and should not be over-weighted.
4. **All-size vs per-size.** The sweep crossover is all-size worst-stream; Eq. 3 (L_b>Q_b) is a
   per-size-bin mechanism — this experiment tests the *aggregate* crossover, not Eq. 3 per bin.

## Artifacts
- `crossover_collapse.csv` (the table above), `crossover_collapse_summary.md` (this file).
- Inputs (unmodified): the five sweep CSVs, four single_stream CSVs, `rev12_single_stream_overall.csv`,
  and step-1 stdout logs (`rev26_stdout.log`, `rev27_yolov8s_stdout.log`,
  `rev27_yolov10s_CORRECTED_stdout.log`, `rev27c_yolo12s_stdout.log`).
- No core script modified; no results/ file overwritten; paper .tex untouched.
