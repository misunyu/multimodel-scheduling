# rev7 delta — old (Table 1 / paper §5–6) vs new (rev7 measurements)

_generated 2026-06-02T20:15:07_

## Table 1 (single-stream YOLOv11s)

| size | old Δ abs | old Δ % | new Δ abs | new Δ % | flip risk |
|---|---|---|---|---|---|
| small  | $-0.007$ | $-47\%$ | $-0.0076$ | $-48.0\%$ | low (sign preserved) |
| medium | $-0.036$ | $-19.5\%$ | $-0.0362$ | $-19.7\%$ | low |
| large  | $+0.001$ | $+0.2\%$ (n.s.) | $+0.0005$ | $+0.1\%$ | low — Δ still ≈0 |

## Main-comparison (N=4 size-diverse Composition A)

| bg | strategy | old worst sAP | new worst sAP | old mean sAP | new mean sAP |
|---|---|---|---|---|---|
| L1_light | Naive_allGPU | 0.084 | 0.088 | 0.125 | 0.132 |
| L1_light | SizeAware    | 0.105 | **0.089** | 0.137 | 0.125 |
| L1_light | SizeBlindRev | 0.084 | 0.083 | 0.133 | 0.129 |
| L2_LM    | Naive_allGPU | 0.058 | 0.059 | 0.100 | 0.101 |
| L2_LM    | SizeAware    | 0.065 | **0.051** | 0.102 | 0.079 |
| L2_LM    | SizeBlindRev | 0.053 | 0.028 | 0.101 | 0.079 |

### Inversion magnitude (SA − SBR worst sAP)
- L1_light: old $+0.021$ ($+25\%$) → new $+0.0058$ ($+7.0\%$). **SIGN PRESERVED**
- L2_LM: old $+0.012$ ($+23\%$) → new $+0.0234$ ($+83.6\%$). **SIGN PRESERVED**

### Worst-vs-mean ratio (SA − Naive)
- L1_light: old worst $+0.021$, mean $+0.012$, ratio $1.75\times$ → new worst $+0.0008$, mean $-0.0073$, ratio $-0.11\times$
- L2_LM:    old worst $+0.007$, mean $+0.002$, ratio $3.5\times$ → new worst $-0.0079$, mean $-0.0223$, ratio $+0.35\times$

## Decomposition

| component | old | new |
|---|---|---|
| Quantization (GPU L0 → NPU L0 N=1)         | $-0.031$ | $-0.0313$ |
| Bg on NPU (NPU L0 → L1 N=1)                | $\approx 0$ | $-0.0000$ |
| Concurrent NPU (NPU L0 → N=8 AllNPU L0)    | $-0.001$ | $-0.0000$ |
| Interaction (N=8 AllNPU L0 → L1)           | $-0.001$ | $-0.0001$ |
| GPU-path (GPU L0 → GPU N=8 L1 Naive)       | $-0.060$ | $-0.0594$ |

**Quantization share of NPU-path loss**: old $97\%$ → new $100\%$.

## Capacity table (N=8 L1_light)

| strategy | old worst sAP | new worst sAP |
|---|---|---|
| Naive_allGPU   | 0.055 | 0.058 |
| SizeAware      | 0.061 | 0.059 |
| SizeBlindRev   | 0.081 | 0.076 |
| AllNPU         | 0.083 | 0.083 |

Saturation ceiling at $N=8$ bg L1_light AllNPU: old $0.083$ → new $0.084$.

## REVERSAL GATE

- new sa_worst 0.0887, sbr_worst 0.0829, worst gain +0.0058 → **PASS**
