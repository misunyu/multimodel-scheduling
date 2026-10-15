# Pilot report (Phase N1 §5)

Runs: main 136, repeat 6. Criteria fixed in `PILOT_FAMILIES.md` before the runs.

## beta = 0.1

| criterion | pass | detail |
|---|---|---|
| A | PASS | 0 bad instance-runs |
| B | PASS | max relative spread 0.0032; violations [] |
| C1 | PASS | rbybys 1x R1 D_macro 0.0000, min completion 1.000 -> True; rbrsybys 1x R1 D_macro 0.0000, min completion 1.000 -> True |
| C2 | **FAIL** | rbybys 4x R1 max q_i 0.0000 -> False; rbrsybys 4x R1 max q_i 0.0000 -> False |
| D | PASS | min generated 42 |
| E | PASS | min realization 0.9968 |
| F | PASS | 0 problems |
| G(ref) | PASS | F rel range 0.0000, D_macro abs range 0.0000 |

Reference (not criteria):

- rbybys 1x: candidates with drop > 0: 0/13
- rbybys 1x: R2 D_macro: 0.0000
- rbrsybys 1x: candidates with drop > 0: 0/21
- rbrsybys 1x: R2 D_macro: 0.0000
- rbybys 4x: candidates with drop > 0: 11/13
- rbybys 4x: R2 D_macro: 0.4191
- rbrsybys 4x: candidates with drop > 0: 19/21
- rbrsybys 4x: R2 D_macro: 0.3630
- repeat (rbrsybys R1, 2x, n=3): F [26.7, 26.7, 26.7] (rel range 0.0000, SE 0.0000); D_macro [0.0, 0.0, 0.0] (abs range 0.0000, SE 0.0000); q_i range {'resnet50_big': [0.0, 0.0], 'resnet50_small': [0.0, 0.0], 'yolov3_big': [0.0, 0.0], 'yolov3_small': [0.0, 0.0]}

## beta = 0.25

| criterion | pass | detail |
|---|---|---|
| A | PASS | 0 bad instance-runs |
| B | PASS | max relative spread 0.0019; violations [] |
| C1 | PASS | rbybys 1x R1 D_macro 0.0000, min completion 1.000 -> True; rbrsybys 1x R1 D_macro 0.0000, min completion 1.000 -> True |
| C2 | PASS | rbybys 4x R1 max q_i 0.3668 -> True; rbrsybys 4x R1 max q_i 0.4065 -> True |
| D | PASS | min generated 107 |
| E | PASS | min realization 0.9981 |
| F | PASS | 0 problems |
| G(ref) | PASS | F rel range 0.0000, D_macro abs range 0.0000 |

Reference (not criteria):

- rbybys 1x: candidates with drop > 0: 11/13
- rbybys 1x: R2 D_macro: 0.2492
- rbrsybys 1x: candidates with drop > 0: 19/21
- rbrsybys 1x: R2 D_macro: 0.2313
- rbybys 4x: candidates with drop > 0: 13/13
- rbybys 4x: R2 D_macro: 0.7395
- rbrsybys 4x: candidates with drop > 0: 21/21
- rbrsybys 4x: R2 D_macro: 0.7334
- repeat (rbrsybys R1, 2x, n=3): F [66.8, 66.8, 66.8] (rel range 0.0000, SE 0.0000); D_macro [0.0, 0.0, 0.0] (abs range 0.0000, SE 0.0000); q_i range {'resnet50_big': [0.0, 0.0], 'resnet50_small': [0.0, 0.0], 'yolov3_big': [0.0, 0.0], 'yolov3_small': [0.0, 0.0]}

## Selection

- passing betas: [0.25]
- **selected beta: 0.25**
