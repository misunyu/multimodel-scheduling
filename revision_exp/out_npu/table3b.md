# Table 3(b) - CPU-NPU, 3x hold-out, N>=3 (4 workloads, 60 candidates), alpha = 0.3, D = D_macro

| method | workloads | avg_score | avg_score_std_over_trials | regret | top1 | top5 | mae_T | mae_D | sample_contains_oracle |
|---|---|---|---|---|---|---|---|---|---|
| Oracle | 4 | 0.9898 |  | 0.0 | 4/4 | 4/4 | N/A | N/A |  |
| Predictive (Ours) | 4 | 0.9898 |  | 0.0 | 4/4 | 4/4 | 0.1372 | 0.11 |  |
| Latency-based | 4 | 0.9898 |  | 0.0 | 4/4 | N/A | N/A | N/A |  |
| Per-instance predictor | 4 | 0.793 |  | 0.1968 | 1/4 | 1/4 | 0.2309 | 0.1333 |  |
| Best-of-5 | 4 | 0.8277 | 0.1017 | 0.1621 | 0.350 (mean hit rate) | N/A | N/A | N/A | 0.35 |
| All-CPU (reference) | 4 | 0.3666 |  | 0.6232 | 0/4 | N/A | N/A | N/A |  |

Best-of-5: 1,000 candidate-sampling trials (5 of the measured candidates, best measured taken); Top-1 is the mean hit rate (not n/4). All-CPU is a reference row. MAE over the 60 primary candidates. Per-instance MAE(T) uses its own T_hat (F_hat / predicted maximum).
