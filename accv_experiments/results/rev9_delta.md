# rev9 delta — old paper / Table 1 vs rev9 normal-state measurements

_generated 2026-06-04T15:07:00_

All cells reuse rev6 baselines where the cell key matches (24-log GPU+NPU L0; GPU L1_light ladder; Comp A multistream). New rev9 measurements fill the rest.

## Table 1 (single-stream Δ NPU − GPU)

| size | old paper | rev9 (normal state) | flip? |
|---|---|---|---|
| small  | $-0.007$ ($-47.1\%$) | $-0.0089$ ($-56.4\%$) | low |
| medium | $-0.036$ ($-19.5\%$) | $-0.0648$ ($-35.2\%$) | low |
| large  | $+0.001$ ($+0.2\%$ n.s.) | $-0.1000$ ($-21.0\%$) | **HIGH — magnitude moved from ≈0 to non-negligible** |

## main-comparison (N=4 size-diverse)

| bg | strategy | old worst sAP | rev9 worst sAP | old mean sAP | rev9 mean sAP |
|---|---|---|---|---|---|
| L1_light | Naive_allGPU | 0.084 | 0.087 | 0.125 | 0.127 |
| L1_light | SizeAware    | 0.105 | **0.084** | 0.137 | 0.123 |
| L1_light | SizeBlindRev | 0.084 | 0.082 | 0.133 | 0.127 |
| L2_LM    | Naive_allGPU | 0.058 | 0.060 | 0.100 | 0.103 |
| L2_LM    | SizeAware    | 0.065 | **0.053** | 0.102 | 0.079 |
| L2_LM    | SizeBlindRev | 0.053 | 0.029 | 0.101 | 0.081 |

### Inversion magnitude (SizeAware − SizeBlindRev worst sAP)
- L1_light: old $+0.021$ ($+25\%$) → rev9 $+0.0020$ ($+2.4\%$). **SIGN PRESERVED**
- L2_LM:    old $+0.012$ ($+23\%$) → rev9 $+0.0233$ ($+79.3\%$). **SIGN PRESERVED**

### Worst-vs-mean ratio (SizeAware − Naive)
- L1_light: old worst $+0.021$, mean $+0.012$, ratio $1.75\times$ → rev9 worst $-0.0035$, mean $-0.0037$, ratio $+0.95\times$
- L2_LM:    old worst $+0.007$, mean $+0.002$, ratio $3.5\times$ → rev9 worst $-0.0073$, mean $-0.0239$, ratio $+0.31\times$

## Decomposition (N=8 size-diverse worst camera)

| component | old | rev9 |
|---|---|---|
| Quantization | $-0.031$ ($97\%$) | $-0.057$ ($169\%$) |
| Total NPU-path | $-0.032$ | $-0.034$ |
| GPU-path (Naive N=8 L1) | $-0.060$ | $-0.059$ |

## REVERSAL GATE

- sa_worst=0.0840, sbr_worst=0.0820, worst gain $+0.0020$ → **PASS**
