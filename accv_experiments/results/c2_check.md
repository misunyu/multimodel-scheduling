# C2 reconstruction direction check (rev9 data; no decisions)

_Pure fact extraction. Whether a direction is adopted is for the human / Claude prose pass. Numbers below are read off rev9 CSVs._

## STEP B — direction 3 (mean-best ≠ worst-best deployable placement)

Deployable placement set (excluding Oracle): `Naive_allGPU` (All-GPU), `SizeBlindRev` (Isolated), `SizeAware` (Contention-aware), `AllNPU`.

### main-comparison (N=4 Composition A)

| bg | mean-best placement | mean sAP | worst-best placement | worst sAP | disagreement |
|---|---|---|---|---|---|
| L1_light | `SizeBlindRev` | `0.1273` | `Naive_allGPU` | `0.0875` | **YES** |
| L2_lm | `AllNPU` | `0.1097` | `AllNPU` | `0.0707` | **no** |

### schedule-shift (N=8 Composition A, same deployable set)

| bg | mean-best placement | mean sAP | worst-best placement | worst sAP | disagreement |
|---|---|---|---|---|---|
| L0 | `SizeAware_NPU6` | `0.2127` | `SizeAware_NPU4` | `0.0938` | **YES** |
| L1_light | `AllNPU` | `0.2075` | `SizeAware_NPU6` | `0.0822` | **YES** |
| L1_heavy | `AllNPU` | `0.2061` | `SizeAware_NPU6` | `0.0809` | **YES** |
| L2_lm | `AllNPU` | `0.1727` | `AllNPU` | `0.0535` | **no** |
| L3_vlm | `AllNPU` | `0.2073` | `SizeAware_NPU6` | `0.0836` | **YES** |

## STEP C — direction 4 (capacity at threshold 0.10; N=8 survival)

### Max N at which worst-stream sAP ≥ 0.10 (bg L1_light)

| strategy | ≥0.10 | ≥0.08 | ≥0.05 |
|---|---|---|---|
| All-GPU | 3 | 4 | 8 |
| Contention-aware | 3 | 5 | 8 |
| Isolated | 3 | 6 | 8 |
| All-NPU | 3 | 8 | 8 |

- old paper claim at threshold 0.10: All-GPU sustained N=3, Contention-aware sustained N=5.
- rev9: All-GPU sustains N=3, Contention-aware sustains N=3. Direction 4 does NOT hold (no capacity gain).

### N=8 non-trivial throughput survival (per-stream `s{i}_skip` < 50%)

| placement | bg | streams with skip<50% / 8 |
|---|---|---|
| `Naive_allGPU` | L1_light | **3/8** |
| `AllNPU` | L1_light | **8/8** |

- old paper claim at N=8 bg L1_light: All-GPU 2/8 alive, All-NPU 8/8 alive.
- rev9: All-GPU 3/8, All-NPU 8/8.

## STEP D — Contention-aware worst-sAP gain (vs Isolated, vs All-GPU)

| bg | SA worst | Isolated worst | All-GPU worst | SA − Isolated | SA − All-GPU |
|---|---|---|---|---|---|
| L1_light | `0.0840` | `0.0820` | `0.0875` | `+0.0020` | `-0.0035` |
| L2_lm | `0.0527` | `0.0294` | `0.0600` | `+0.0233` | `-0.0073` |

## STEP E — factual summary (no prose decisions)

### Direction 3 — does mean-best differ from worst-best (deployable)?

- N=4 L1_light: **YES** (disagreement holds)
- N=4 L2_LM: **no**
- N=8 (any bg): YES at L0, L1_light, L1_heavy, L3_vlm

### Direction 4 — capacity gain at threshold 0.10

- All-GPU max N: rev9 = **3** (paper claim 3)
- Contention-aware max N: rev9 = **3** (paper claim 5)
- Direction 4 NOT supported at this threshold (no gain in N)
- N=8 L1_light survival: All-GPU 3/8, All-NPU 8/8 (paper claim 2/8 vs 8/8)

### Direction 5 — L1 weak vs L2_LM strong contention-aware advantage

- Contention-aware worst sAP gain vs Isolated: L1_light `+0.0020` vs L2_LM `+0.0233`
- Contention-aware worst sAP gain vs All-GPU:  L1_light `-0.0035` vs L2_LM `-0.0073`
- The L1 (weak) → L2_LM (strong) contrast on the SA−Isolated axis is supported by the data (11.6× larger gain at L2_LM).

