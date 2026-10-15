# Table 3(a) - CPU-GPU, 3x hold-out, N>=3, alpha=0.3

| strategy | workloads | avg_score | avg_score_std | top1 | top5 | regret | mae_T | mae_D | sample_contains_oracle | selected_all_gpu | not_evaluated | manuscript_Table_3a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Latency-based | 18 | 0.6763 |  | 3/18 (0.17) | N/A | 0.3104 |  |  |  | 3 |  | 0.63 / hit 0.17 |
| Ours (retrained) | 18 | 0.9505 |  | 13/18 (0.72) | 17/18 (0.94) | 0.0362 | 0.0599 | 0.0905 |  | 15 |  | 0.95 / Top-1 0.72 / Top-5 0.94 (MAE 0.06 / 0.09, gap 0.03) |
| Ours (existing artifact + new evaluator) | 18 | 0.9453 |  | 13/18 (0.72) | 17/18 (0.94) | 0.0415 | 0.0601 | 0.0904 |  | 16 |  |  |
| Best-of-5 (1,000 trials) | 18 | 0.7299 | 0.0388 | 0.307 (mean rate) | N/A | 0.2568 |  |  | 0.307 |  |  | 0.78 +- 0.040 / hit 0.30 |
| All-GPU | 17 | 0.9426 |  | 13/17 (0.76) | N/A | 0.0433 |  |  |  | 17 | model_schedules_g_m_t_x3.yaml |  |
| Oracle | 18 | 0.9867 |  | 18/18 (1.00) | 18/18 (1.00) | 0.0 |  |  |  | 13 |  | 0.99 |
| All-CPU | 18 | -0.1945 |  | 0/18 (0.00) | N/A | 1.1813 |  |  |  | 0 |  |  |
| Latency-based (stored rule replay) | 18 | 0.6175 |  | 3/18 (0.17) | N/A | 0.3692 |  |  |  | 1 |  |  |
| Ours (retrained, D without weight) | 18 | 0.9453 |  | 13/18 (0.72) | 17/18 (0.94) | 0.0415 | 0.0599 | 0.09 |  | 16 |  |  |
