# table_predictors

3x hold-out, N>=3, 12 workloads, alpha = 0.3, D = D_macro.

| method | workloads | MAE(T) | MAE(D) | Avg score | regret | Top-1 | Top-5 | Ours-All-GPU paired mean (ref.) |
|---|---|---|---|---|---|---|---|---|
| XGBoost Full | 12 | 0.0273 | 0.03 | 0.993 | 0.0044 | 7/12 | 11/12 | 0.0203 |
| Ridge | 12 | 0.03 | 0.0214 | 0.9941 | 0.0033 | 7/12 | 11/12 | 0.0214 |
| Random Forest | 12 | 0.0307 | 0.041 | 0.9948 | 0.0026 | 8/12 | 11/12 | 0.0221 |
| MLP (small) | 12 | 0.0597 | 0.0474 | 0.9729 | 0.0245 | 6/12 | 9/12 | 0.0002 |

Top-1 / Top-5: hits / evaluated workloads (3x, N>=3). Top-5 only for methods that rank every candidate. (i) Workloads whose oracle-to-second measured S margin is <= 0.0002: 6/12 (of which exact ties: 2). (ii) Workloads whose per-repeat oracle is not the same in all three repeats: 4/12 (results/INTEGRITY.md section 6). Top-1 accuracy is sensitive to small differences between the highest-ranked measured placements; we therefore report placement score and regret alongside Top-1 and Top-5. Best-of-5: 1,000 repetitions of candidate sampling, not of measurement; its Top-1 is the mean hit rate. Latency-based selects all-GPU for every model on the new offline profile. Per-instance predictor: inspired by the instance-wise interference prediction of Kim et al. [14]; not a reproduction of their system. MAE: N>=3 candidates (480); per-instance MAE(T) uses T_hat_select = F_hat / max F_hat (predicted maximum). Selection score (all methods, no clip, as in the manuscript definition): S_hat = T_hat - 0.3 D_hat for the direct predictors (T_hat, D_hat not clipped individually); per-instance S_hat = T_hat_select - 0.3 D_hat_macro. See CLIP_CHECK.md.
