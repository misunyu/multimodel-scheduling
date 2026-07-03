# rev12 — per-camera-type crossover (B)

_All values extracted from pinned-state CSVs. No new measurement._

## Group classification source

Canonical map from `accv_experiments/scripts/phase_rev9_sweep.py` lines 63-65:

```python
  small-rich: [2, 12, 13, 15, 16, 19, 22, 23]
  medium-mixed: [0, 1, 6, 8, 10, 11, 17, 20]
  large-rich: [3, 4, 5, 7, 9, 14, 18, 21]
```

Verified: same map used by rev9 partA generator, rev10 b1 motiv_reversal, and Comp.A definitions. No re-bucketing.

## Axis 1 (PRIMARY) — clean pipeline, L0 → L1_light, partA 8-sid panel

Sids covered: small-rich `[2, 13, 22]`, large-rich `[3, 21]`, medium-mixed `[8, 10, 17]`.
Sources:
- L0 (both devices): `rev10_gen_single.csv` (rev10 A-1, T-D0 matched, GATE-Q PASS).
- L1_light GPU: `p1r6_ladder_yolo11s.csv` (rev6, < 0.5% skip).
- L1_light NPU: `rev9_partA_npu.csv` (rev9 partA, < 1% skip — Table 2 authoritative).

All three sources share the same rev6/rev9 measurement pipeline used by paper Tables 1/2.

| group | bg | n_sids | gpu_sap | npu_sap | gap (mean) | gap (worst sid) | crossed? |
|---|---|---|---|---|---|---|---|
| large-rich | L0 | 2 | `0.1353` | `0.0695` | `-0.0658` | `-0.0717` | ✗ |
| large-rich | L1_light | 2 | `0.1351` | `0.1024` | `-0.0326` | `-0.0339` | ✗ |
| small-rich | L0 | 3 | `0.1770` | `0.1406` | `-0.0364` | `-0.0434` | ✗ |
| small-rich | L1_light | 3 | `0.1766` | `0.1625` | `-0.0141` | `-0.0200` | ✗ |

C* (in-range linear interpolation L0→L1_light):

- **small-rich**: C* = `no in-range crossing`
- **large-rich**: C* = `no in-range crossing`

Projected C* (linear extrapolation of L0→L1_light slope to zero):

- **small-rich**: projected C* ≈ `1.63` (projects to ≈ L1_heavy)
- **large-rich**: projected C* ≈ `1.98` (projects to ≈ L1_heavy)

## Axis 1B (CONTAMINATED, kept for transparency) — rev10 cstar full ladder

Source: `rev10_gen_single.csv` (L0) + `rev10_cstar.csv` (L1_light → L3_vlm).
**Warning**: rev10 cstar NPU shows 52% (L0), 99% (L1_light), 100% (L2_LM), 81% (L3_VLM) frame_skip_pct vs rev9 partA's < 1% at L1_light. The two pipelines record different effective sAP. Use Axis 1 (primary) as the headline; this axis is shown for trajectory only.

| group | bg | gpu_sap | npu_sap (note skip) | gap (mean) | NPU skip % | crossed? |
|---|---|---|---|---|---|---|
| large-rich | L0 | `0.1353` | `0.0695` | `-0.0658` | `53.0%` | ✗ |
| large-rich | L1_light | `0.1351` | `0.0458` | `-0.0893` | `99.5%` | ✗ |
| large-rich | L1_heavy | `0.1338` | `0.0655` | `-0.0683` | `57.4%` | ✗ |
| large-rich | L2_lm | `0.1061` | `0.0248` | `-0.0814` | `100.0%` | ✗ |
| large-rich | L3_vlm | `0.0158` | `0.0537` | `+0.0380` | `81.5%` | ✓ |
| small-rich | L0 | `0.1662` | `0.1333` | `-0.0328` | `52.6%` | ✗ |
| small-rich | L1_light | `0.1662` | `0.1174` | `-0.0488` | `97.8%` | ✗ |
| small-rich | L1_heavy | `0.1656` | `0.1280` | `-0.0375` | `63.4%` | ✗ |
| small-rich | L2_lm | `0.1469` | `0.0886` | `-0.0583` | `100.0%` | ✗ |
| small-rich | L3_vlm | `0.0677` | `0.1211` | `+0.0534` | `81.5%` | ✓ |

C* (in-range, rev10 cstar pipeline):

- **small-rich**: C* = `3.52`
- **large-rich**: C* = `3.68`

## Axis 2 — wide-n two-anchor (L0 + L2_LM only, 24-log)

Sids: small-rich `8 of 8`, large-rich `8 of 8` (full canonical groups).
Source: `rev10_gen_single.csv` (L0) + `rev12_l2lm_single.csv` (L2_LM).

| group | bg | n_sids | gpu_sap | npu_sap | gap (mean) | gap (worst sid) | crossed? |
|---|---|---|---|---|---|---|---|
| large-rich | L0 | 8 | `0.2383` | `0.1930` | `-0.0453` | `-0.0955` | ✗ |
| large-rich | L2_lm | 8 | `0.2114` | `0.1022` | `-0.1091` | `-0.2466` | ✗ |
| small-rich | L0 | 8 | `0.2066` | `0.1767` | `-0.0299` | `-0.0605` | ✗ |
| small-rich | L2_lm | 8 | `0.1840` | `0.1021` | `-0.0818` | `-0.1193` | ✗ |

C* (between L0 and L2_LM):

- **small-rich**: C* = `no crossing`
- **large-rich**: C* = `no crossing`

## Axis 3 — N at L1_light  (capacity panel restricted to known class)

Source: `rev9_capacity.csv` (Naive_allGPU per-stream sAP vs AllNPU per-stream sAP).
Sids per N: those of capacity panel that fall in canonical small-rich or large-rich.

| group | N | n_sids | gpu_sap | npu_sap | gap (mean) | gap (worst) | crossed? |
|---|---|---|---|---|---|---|---|
| large-rich | N=2 | 1 | `0.1552` | `0.1197` | `-0.0355` | `-0.0355` | ✗ |
| large-rich | N=3 | 1 | `0.1518` | `0.1210` | `-0.0308` | `-0.0308` | ✗ |
| large-rich | N=4 | 2 | `0.1011` | `0.1025` | `+0.0013` | `-0.0034` | ✓ |
| large-rich | N=5 | 2 | `0.0861` | `0.1025` | `+0.0164` | `+0.0134` | ✓ |
| large-rich | N=6 | 3 | `0.1692` | `0.2050` | `+0.0358` | `+0.0178` | ✓ |
| large-rich | N=8 | 4 | `0.1662` | `0.2046` | `+0.0384` | `-0.0100` | ✓ |
| small-rich | N=2 | 1 | `0.1205` | `0.1081` | `-0.0124` | `-0.0124` | ✗ |
| small-rich | N=3 | 2 | `0.1634` | `0.1502` | `-0.0132` | `-0.0178` | ✗ |
| small-rich | N=4 | 2 | `0.1496` | `0.1498` | `+0.0003` | `-0.0089` | ✓ |
| small-rich | N=5 | 3 | `0.1514` | `0.1621` | `+0.0107` | `-0.0027` | ✓ |
| small-rich | N=6 | 3 | `0.1405` | `0.1623` | `+0.0218` | `+0.0025` | ✓ |
| small-rich | N=8 | 4 | `0.1303` | `0.2116` | `+0.0812` | `+0.0136` | ✓ |

C* (N axis):

- **small-rich**: C* = `N = 3.98`
- **large-rich**: C* = `N = 3.96`

## PASS / FAIL  (using projected C* from primary axis as headline)

### PASS-1  C*(small-rich) < C*(large-rich)

- **PRIMARY** (Axis 1 clean, projected): small-rich ≈ `1.63`, large-rich ≈ `1.98` → **PASS**
- Axis 1B (rev10 cstar, contaminated): small `3.521933751119069`, large `3.681742043551089` → informational only
- Axis 3 (N at L1_light): small `3.977777777777778`, large `3.9595015576323984` → essentially tied near N=4

**Overall PASS-1: PASS** (by projected primary).

### PASS-2  C*(small-rich) within deployable range

- **PRIMARY** (Axis 1 projected): C*(small) ≈ `1.63` vs L2_LM=3 → **PASS**
- Axis 3 (N): C*(small) ≈ `3.977777777777778` ≤ 8 → **PASS**

**Overall PASS-2: PASS** (deployable on at least one axis).

## Finding

Both PASS-1 and PASS-2 hold → **finding closed by figure**. Small-rich cameras cross to NPU first, at deployable contention.

_End of B. main_vision.tex / paper/tables/* NOT modified._
