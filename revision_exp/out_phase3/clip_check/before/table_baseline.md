# table_baseline

3x hold-out, N>=3, 12 workloads, alpha = 0.3, D = D_macro.

| method | workloads | MAE(T) | MAE(D) | Avg score | regret | Top-1 | Top-5 | Ours-All-GPU paired mean (ref.) |
|---|---|---|---|---|---|---|---|---|
| Latency-based (= All-GPU) | 12 | N/A | N/A | 0.9727 | 0.0247 | 6/12 | N/A |  |
| Per-instance predictor ([14]-inspired) | 12 | 0.0276 | 0.0236 | 0.9868 | 0.0106 | 6/12 | 11/12 | 0.0141 |
| XGBoost Full | 12 | 0.0273 | 0.03 | 0.993 | 0.0044 | 7/12 | 11/12 | 0.0203 |
| Best-of-5 | 12 | N/A | N/A | 0.9404 | 0.057 | 0.402 (mean hit rate over 1,000 samplings) | N/A |  |
| Oracle | 12 | N/A | N/A | 0.9974 | 0.0 | 12/12 | N/A |  |

Top-1 / Top-5: hits / evaluated workloads (3x, N>=3). Top-5 only for methods that rank every candidate. (i) Workloads whose oracle-to-second measured S margin is <= 0.0002: 6/12 (of which exact ties: 2). (ii) Workloads whose per-repeat oracle is not the same in all three repeats: 4/12 (results/INTEGRITY.md section 6). Top-1 accuracy is sensitive to small differences between the highest-ranked measured placements; we therefore report placement score and regret alongside Top-1 and Top-5. Best-of-5: 1,000 repetitions of candidate sampling, not of measurement; its Top-1 is the mean hit rate. Latency-based selects all-GPU for every model on the new offline profile. Per-instance predictor: inspired by the instance-wise interference prediction of Kim et al. [14]; not a reproduction of their system. MAE: N>=3 candidates (480); per-instance MAE(T) uses T_hat_select = F_hat / max F_hat (predicted maximum). Selection score: S_hat = clip(T_hat - 0.3 D_hat, 0, 1) for the direct predictors (Phase 2 rule); per-instance S_hat = T_hat_select - 0.3 D_hat_macro.
