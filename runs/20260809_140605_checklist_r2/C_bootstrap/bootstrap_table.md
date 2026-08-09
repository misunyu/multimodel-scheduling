| 지표 | 점추정 | 95% CI | bootstrap SD |
|---|---|---|---|
| `decomposed_top1_gpu` | +0.911 | [+0.844, +0.978] | 0.038 |
| `decomposed_top1_npu` | +0.622 | [+0.422, +0.800] | 0.098 |
| `diff_direct_minus_decomposed_gpu` | +0.022 | [+0.000, +0.067]  (CI가 0 제외: 아니오) | 0.022 |
| `diff_direct_minus_decomposed_npu` | +0.244 | [+0.044, +0.467]  (CI가 0 제외: 예) | 0.111 |
| `diff_direct_minus_greedy_load_factor_gpu` | +0.622 | [+0.489, +0.756]  (CI가 0 제외: 예) | 0.069 |
| `diff_direct_minus_greedy_load_factor_npu` | +0.667 | [+0.511, +0.800]  (CI가 0 제외: 예) | 0.076 |
| `diff_direct_minus_greedy_speedup_gpu` | +0.622 | [+0.489, +0.756]  (CI가 0 제외: 예) | 0.069 |
| `diff_direct_minus_greedy_speedup_npu` | +0.622 | [+0.422, +0.800]  (CI가 0 제외: 예) | 0.099 |
| `direct_group_mean_spearman_gpu` | +0.986 | [+0.982, +0.990] | 0.002 |
| `direct_group_mean_spearman_npu` | +0.972 | [+0.953, +0.986] | 0.009 |
| `direct_top1_gpu` | +0.933 | [+0.867, +1.000] | 0.034 |
| `direct_top1_npu` | +0.867 | [+0.756, +0.956] | 0.053 |
| `divergence_rate` | +0.244 | [+0.089, +0.422] | 0.081 |
| `greedy_load_factor_top1_lb_gpu` | +0.311 | [+0.156, +0.467] | 0.079 |
| `greedy_load_factor_top1_lb_npu` | +0.200 | [+0.089, +0.311] | 0.060 |
| `greedy_speedup_top1_lb_gpu` | +0.311 | [+0.156, +0.467] | 0.079 |
| `greedy_speedup_top1_lb_npu` | +0.244 | [+0.111, +0.378] | 0.065 |
