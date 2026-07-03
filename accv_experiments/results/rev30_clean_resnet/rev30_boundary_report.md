# rev30 boundary-instability report (N=4, k=1, 33.333 ms deadline)

Reps = 10.

| metric | mean | std | min | max |
|---|---|---|---|---|
| GPU frame skip (%) | 24.3 | 2.2 | 21.8 | 28.6 |
| All-GPU worst sAP | 0.0979 | 0.0037 | 0.0901 | 0.1042 |
| All-NPU worst sAP | 0.0830 | 0.0004 | 0.0824 | 0.0837 |
| Oracle worst sAP (r=1) | 0.1026 | 0.0031 | 0.0975 | 0.1089 |
| deadline margin (ms, GPU) | -5.71 | 0.64 | -6.50 | -4.49 |

**Oracle picked-ratio distribution over 10 reps** (ratio = #streams on NPU): {0: 1, 1: 9, 2: 0, 3: 0, 4: 0}

Interpretation: a wide GPU-skip spread and a multi-modal picked-ratio distribution at k=1
show the *optimal placement itself* is unstable at the deadline boundary — a quantified
finding, reported as mean±std, not a hidden weakness.

## Cross-check against the old scattered runs (same nominal point)
- rev21/rev20 (old tab:main): Oracle 0.1086, All-GPU 0.0915, skip 35.0
- rev29 (old Fig3b): Oracle 0.0910, All-GPU 0.0887, skip 37.8
- rev30 k=1: Oracle 0.1026±0.0031, All-GPU 0.0979±0.0037, skip 24.3±2.2
- Old Oracle values inside rev30 [min,max]=[0.0975,0.1089]? 0.1086:YES  0.0910:NO
