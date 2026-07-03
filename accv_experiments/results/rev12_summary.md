# rev12 summary — Tables 1·5·6 in single pinned state

_Aggregation only. Re-measurement of L2_LM 24-log added for staleness; L0 and N=4 reused from rev10 (same pinned state per T-D0 + GATE-Q PASS). `paper/main_vision.tex` and `paper/tables/*.tex` NOT modified._

## State pin

- mxq sha8 (yolo11s anchor): `b2441f9d`
- driver: `580.159.03`
- SDK: `models/mblt_model_zoo  (T-D0 verified state)`
- git: `dc478b1f88cd`
- ts: `2026-06-05T16:18:49`

Source CSVs:
  - L0_single_stream: `accv_experiments/results/rev10_gen_single.csv (rev10 A-1, GATE-Q PASS rel=-0.201)`
  - L2_LM_single_stream: `accv_experiments/results/rev12_l2lm_single.csv (rev12, fresh)`
  - N4_multi_stream: `accv_experiments/results/rev10_gen_gain.csv (rev10 A-2)`
  - TD0_anchor: `accv_experiments/results/repro_driverstate.csv (8-run cold/warm/loaded)`

## T1 — single-stream yolo11s L0 (24 logs, per size)

### Acceptance gate (large gap = -0.096 ± 0.002): **PASS**

- measured: `-0.0958`, rel = `-0.201`

| size | GPU mean±std | NPU mean±std | gap (mean ± std) | rel gap | Wilcoxon p |
|---|---|---|---|---|---|
| small | `0.0159±0.0162` | `0.0072±0.0066` | `-0.0087 ± 0.0107` | `-0.549` | `4.61e-05` |
| medium | `0.1839±0.1274` | `0.1170±0.0903` | `-0.0669 ± 0.0441` | `-0.364` | `1.19e-07` |
| large | `0.4768±0.1643` | `0.3810±0.1823` | `-0.0958 ± 0.0601` | `-0.201` | `3.93e-06` |

## T2 — gen-decomp (4 detectors × {L0, L2_LM})

### Acceptance gate (yolo11s Q_large = -0.096 ± 0.002 ≈ T1): **PASS**

- yolo11s Q_large = `-0.0958`, rel = `-0.201`

| detector | params (M) | INT8 export? | Q_S | Q_M | Q_L (rel) | GPU stale_L (L0→L2_LM) | NPU stale_L |
|---|---|---|---|---|---|---|---|
| yolo11s | 9.4 | ✓ | `-0.009` | `-0.067` | `-0.096 (-0.20)` | `+0.069` | `+0.190` |
| yolo11m | 20.1 | ✓ | `-0.014` | `-0.081` | `-0.166 (-0.32)` | `+0.116` | `+0.156` |
| yolo11l | 25.3 | ✓ | `-0.017` | `-0.096` | `-0.186 (-0.35)` | `+0.138` | `+0.150` |
| yolo11x | 56.9 | ✓ | `-0.024` | `-0.120` | `-0.225 (-0.41)` | `+0.150` | `+0.135` |

_Q_X = NPU L0 sAP_X - GPU L0 sAP_X (quantization loss per size). stale_L = device L0 sAP_L - device L2_LM sAP_L (positive = degrades under bg)._

## T3 — gen-gain (4 detectors × N=4 / L1_light / Comp.A, 3 reps)

### Acceptance gate (all 4 worst_gain > 0): **PASS**

| detector | worst_gain (mean ± std) | mean_gain (mean ± std) | inverts worst (3/3)? |
|---|---|---|---|
| yolo11s | `+0.0168 ± 0.0052` | `+0.0013 ± 0.0031` | ✓ |
| yolo11m | `+0.0176 ± 0.0038` | `-0.0033 ± 0.0020` | ✓ |
| yolo11l | `+0.0153 ± 0.0028` | `-0.0087 ± 0.0024` | ✓ |
| yolo11x | `+0.0280 ± 0.0015` | `-0.0047 ± 0.0015` | ✓ |

_Gain = Contention-aware (SizeAware) − Isolated (SizeBlindRev)._

## T4 — N=8 8-run reproducibility

**SKIPPED per spec recommendation.** Use rev11 3-run wording: "across three runs, per-strategy std ≤ 0.0021" (rev11_n8.md verified).

---

_End. main_vision.tex and paper/tables/* NOT modified._
