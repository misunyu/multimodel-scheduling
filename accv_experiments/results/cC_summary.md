# Path-C diagnostic — SizeAware worst-stream underperformance

_Aggregation only. No measurement, no prose / table edits._

## C-1  Per-camera anchors and SizeAware offload decision (Composition A, N=4)

Composition A sids: 2 (small-rich), 22 (small-rich), 3 (large-rich), 21 (large-rich).

| sid | role | GPU L0 sAP | NPU L0 sAP | GPU L1 sAP | NPU L1 sAP | $Q_i$ | $L_i$(L1) | dev-gap(L1) | SA at L1 | SA at L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | small-rich | 0.121 | 0.085 | 0.121 | 0.109 | +0.037 | +0.001 | -0.012 | `NPU` | `NPU` |
| 22 | small-rich | 0.211 | 0.177 | 0.211 | 0.191 | +0.034 | -0.000 | -0.020 | `NPU` | `NPU` |
| 3 | large-rich | 0.156 | 0.084 | 0.155 | 0.121 | +0.072 | +0.001 | -0.034 | `GPU` | `GPU` |
| 21 | large-rich | 0.115 | 0.058 | 0.115 | 0.084 | +0.057 | +0.000 | -0.031 | `GPU` | `GPU` |

Legend: $Q_i$ = GPU L0 sAP − NPU L0 sAP (positive = NPU loses to quantization). $L_i$(L1) = GPU L0 sAP − GPU L1 sAP (positive = GPU degrades under L1 contention). dev-gap(L1) = NPU L1 sAP − GPU L1 sAP (positive = NPU is the better device at L1).

### SizeAware placement decision

At L1_light, SizeAware routes sids `[2, 22]` to **NPU** and sids `[3, 21]` to **GPU**.

- sid 2 (small-rich) — $Q$=+0.037, $L$(L1)=+0.001, dev-gap(L1)=-0.012 → **offload HURTS this sid** (NPU L1 < GPU L1).
- sid 22 (small-rich) — $Q$=+0.034, $L$(L1)=-0.000, dev-gap(L1)=-0.020 → **offload HURTS this sid** (NPU L1 < GPU L1).

## C-2  Worst-stream camera per strategy (N=4 Composition A)

### bg = L1_light

| strategy | worst sid | role | device under strategy | worst sAP |
|---|---|---|---|---|
| Naive_allGPU | 21 | large-rich | `GPU` | `0.0875` |
| SizeAware | 21 | large-rich | `GPU` | `0.0840` |
| SizeBlindRev | 21 | large-rich | `NPU` | `0.0820` |
| AllNPU | 21 | large-rich | `NPU` | `0.0836` |
| Oracle (`Oracle_k3_0_1_2`) | 2 | small-rich | `NPU` | `0.1096` |

- Oracle placement: `{2: 'NPU', 22: 'NPU', 3: 'NPU', 21: 'GPU'}` (worst sAP `0.1096`).
- SizeAware placement: `{2: 'NPU', 22: 'NPU', 3: 'GPU', 21: 'GPU'}` (worst sAP `0.0840`).
- Sids placed differently: `[3]`. Oracle: { 3→NPU } ; SizeAware: { 3→GPU }.

### bg = L2_lm

| strategy | worst sid | role | device under strategy | worst sAP |
|---|---|---|---|---|
| Naive_allGPU | 21 | large-rich | `GPU` | `0.0600` |
| SizeAware | 21 | large-rich | `GPU` | `0.0527` |
| SizeBlindRev | 21 | large-rich | `NPU` | `0.0294` |
| AllNPU | 21 | large-rich | `NPU` | `0.0707` |
| Oracle (`Oracle_k3_0_1_2`) | 21 | large-rich | `GPU` | `0.0736` |

- Oracle placement: `{2: 'NPU', 22: 'NPU', 3: 'NPU', 21: 'GPU'}` (worst sAP `0.0736`).
- SizeAware placement: `{2: 'NPU', 22: 'NPU', 3: 'GPU', 21: 'GPU'}` (worst sAP `0.0527`).
- Sids placed differently: `[3]`. Oracle: { 3→NPU } ; SizeAware: { 3→GPU }.

## C-3  Diagnostic facts (no decision)

### Q-A  Is the worst camera under SizeAware one of the cameras SA offloaded to the NPU?

- bg = L1_light: **no** — worst sid 21 (large-rich) was left on GPU by SA
- bg = L2_lm: **no** — worst sid 21 (large-rich) was left on GPU by SA

### Q-B  Oracle vs SizeAware placement divergence

- bg = L1_light: differing sids `[3]`. Oracle: `{ 3→NPU }`; SA: `{ 3→GPU }`.
- bg = L2_lm: differing sids `[3]`. Oracle: `{ 3→NPU }`; SA: `{ 3→GPU }`.

### Q-C  For each camera SizeAware offloaded to NPU at L1_light: $L_i$(L1) vs $Q_i$

If $L_i$(L1) $<$ $Q_i$, offloading the camera **loses sAP** because the streaming penalty it would have paid on the GPU is smaller than the quantization penalty it now pays on the NPU.

| sid | role | $Q_i$ | $L_i$(L1) | $L_i$<$Q_i$? | dev-gap(L1) | net at L1 |
|---|---|---|---|---|---|---|
| 2 | small-rich | `+0.037` | `+0.001` | yes (offload hurts) | `-0.012` | `-0.012` |
| 22 | small-rich | `+0.034` | `-0.000` | yes (offload hurts) | `-0.020` | `-0.020` |

Summary: 0 offloaded camera(s) HELP (NPU beats GPU at L1); 2 offloaded camera(s) HURT (GPU beats NPU at L1).

### Q-D  AllNPU worst sAP at N=4 vs other strategies

- bg = L1_light: deployable ranking by worst sAP → `Naive_allGPU` (0.0875), `SizeAware` (0.0840), `AllNPU` (0.0836), `SizeBlindRev` (0.0820)
- bg = L2_lm: deployable ranking by worst sAP → `AllNPU` (0.0707), `Naive_allGPU` (0.0600), `SizeAware` (0.0527), `SizeBlindRev` (0.0294)

### Q-E  AllNPU vs SizeAware worst sAP at N=4 (rev9 normal state)

- bg = L1_light: AllNPU=0.0836, SA=0.0840, AllNPU−SA = -0.0004 → **no — SizeAware ≥ AllNPU**
- bg = L2_lm: AllNPU=0.0707, SA=0.0527, AllNPU−SA = +0.0180 → **YES — AllNPU beats SizeAware**

### Q-F  Cross-link: AllNPU at the capacity table

| N | AllNPU | Naive | SizeAware | SizeBlindRev |
|---|---|---|---|---|
| 2 | `0.1081` | `0.1205` | `0.1085` | `0.1203` |
| 3 | `0.1090` | `0.1176` | `0.1077` | `0.1158` |
| 4 | `0.0836` | `0.0870` | `0.0924` | `0.0833` |
| 5 | `0.0836` | `0.0702` | `0.0862` | `0.0823` |
| 6 | `0.0836` | `0.0658` | `0.0749` | `0.0834` |
| 8 | `0.0829` | `0.0543` | `0.0615` | `0.0791` |

