# rev27c STEP 0 — YOLO12s export load + small-biased gate

## (0a) Load
YOLO12s global8 LOADS+RUNS. NPU raw output shapes: [(20, 20, 64), (20, 20, 80), (40, 40, 64), (40, 40, 80), (80, 80, 64), (80, 80, 80)]

## (0b) small-biased sanity (6 sids x 3 reps)

NPU infer 14.9ms, skip 0.0% (threads=4)

| size | GPU | NPU | gap | rel% |
|---|---|---|---|---|
| small | 0.0068 | 0.0048 | -0.0020 | -30.1% |
| medium | 0.1638 | 0.1440 | -0.0198 | -12.1% |
| large | 0.5087 | 0.4904 | -0.0183 | -3.6% |

## Gate verdict

**PASS — small-biased** (small -30.1% worst, large -3.6% mildest). Proceed to STEP 1-3.
