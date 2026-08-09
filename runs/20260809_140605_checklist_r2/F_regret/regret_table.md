| 방법 | 플랫폼 | 분모 | Top-1 | Oracle ratio 평균 | 중앙값 | p10 | 최소 | regret 평균 | ratio<0.9 |
|---|---|---|---|---|---|---|---|---|---|
| direct (P1) | GPU | 45 groups | 0.933 | **0.996** | 1.000 | 1.000 | 0.921 | 0.004 | 0 |
| direct (P1) | NPU | 45 groups | 0.867 | **0.993** | 1.000 | 0.976 | 0.910 | 0.007 | 0 |
| decomposed (P6) | GPU | 45 groups | 0.911 | **0.980** | 1.000 | 1.000 | 0.249 | 0.020 | 1 |
| decomposed (P6) | NPU | 45 groups | 0.622 | **0.951** | 1.000 | 0.883 | 0.455 | 0.049 | 6 |
| greedy (load_factor) | GPU | 41 measured groups | 0.342 | **0.575** | 0.531 | 0.074 | -0.154 | 0.425 | 27 |
| greedy (load_factor) | GPU | 45 groups (unmeasured = ratio 0) | 0.311 | **0.524** | 0.522 | 0.000 | -0.154 | 0.476 | 31 |
| greedy (speedup) | GPU | 33 measured groups | 0.424 | **0.734** | 0.782 | 0.397 | 0.089 | 0.266 | 19 |
| greedy (speedup) | GPU | 45 groups (unmeasured = ratio 0) | 0.311 | **0.538** | 0.567 | 0.000 | 0.000 | 0.462 | 31 |
| greedy (load_factor) | NPU | 36 measured groups | 0.250 | **0.590** | 0.569 | 0.021 | -0.156 | 0.410 | 23 |
| greedy (load_factor) | NPU | 45 groups (unmeasured = ratio 0) | 0.200 | **0.472** | 0.419 | 0.000 | -0.156 | 0.528 | 32 |
| greedy (speedup) | NPU | 33 measured groups | 0.333 | **0.793** | 0.926 | 0.443 | 0.029 | 0.207 | 15 |
| greedy (speedup) | NPU | 45 groups (unmeasured = ratio 0) | 0.244 | **0.581** | 0.686 | 0.000 | 0.000 | 0.419 | 27 |
