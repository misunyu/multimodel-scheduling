# Table 3(a) - re-profiled CPU-GPU data, 3x hold-out, N>=3, alpha=0.3, drop=macro

| strategy | workloads | not_evaluated | avg_score | top1 | top5 | regret | oracle_avg_same_workloads | selected_all_gpu | mae_T | mae_D | avg_score_std_over_trials | sample_contains_oracle | previous_data_retrained(revision_exp/out, 18 workloads, D_agg) | manuscript_Table_3a(old data) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Latency-based | 12 | 0 | 0.9727 | 6/12 (0.50) | N/A | 0.0247 | 0.9974 | 12.0 |  |  |  |  | 0.6763 / 3/18 (0.17) / nan | 0.63 / hit 0.17 |
| Ours (retrained) | 12 | 0 | 0.993 | 7/12 (0.58) | 11/12 (0.92) | 0.0044 | 0.9974 | 8.0 | 0.0273 | 0.03 |  |  | 0.9505 / 13/18 (0.72) / 17/18 (0.94) | 0.95 / Top-1 0.72 / Top-5 0.94 (MAE 0.06 / 0.09) |
| Best-of-5 (1,000 candidate-sampling trials) | 12 | 0 | 0.9404 | 0.402 (mean rate) | N/A | 0.057 | 0.9974 |  |  |  | 0.021 | 0.402 | 0.7299 / 0.307 (mean rate) / nan | 0.78 +- 0.040 / hit 0.30 |
| All-GPU | 12 | 0 | 0.9727 | 6/12 (0.50) | N/A | 0.0247 | 0.9974 | 12.0 |  |  |  |  | 0.9426 / 13/17 (0.76) / nan |  |
| All-CPU | 12 | 0 | 0.3966 | 0/12 (0.00) | N/A | 0.6008 | 0.9974 | 0.0 |  |  |  |  | -0.1945 / 0/18 (0.00) / nan |  |
| Oracle | 12 | 0 | 0.9974 | 12/12 (1.00) | 12/12 (1.00) | 0.0 | 0.9974 | 4.0 |  |  |  |  | 0.9867 / 18/18 (1.00) / 18/18 (1.00) | 0.99 |
| Latency-based (1x open-loop latency) | 12 | 0 | 0.9727 | 6/12 (0.50) | N/A | 0.0247 | 0.9974 | 12.0 |  |  |  |  |  |  |
| Ours (retrained, D without weight) | 12 | 0 | 0.9946 | 7/12 (0.58) | 12/12 (1.00) | 0.0028 | 0.9974 | 8.0 | 0.0273 | 0.0303 |  |  | 0.9453 / 13/18 (0.72) / 17/18 (0.94) |  |

Notes: evaluation workloads = 3x, N>=3 (denominators in the table). Oracle and all measured scores use the 3-repeat mean labels. Best-of-5: 1,000 repetitions of CANDIDATE SAMPLING (5 placements drawn without replacement, the best measured one taken); it is not a repetition of measurements. D = D_macro (mean of the per-instance drop fractions), alpha = 0.3. The two right-most columns are earlier numbers on the OLD data with the old drop definition; they are not comparable row by row.
