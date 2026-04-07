# NPU (Antara) Deploy Change Experiment Analysis

## Experiment Setup

- **Platform**: Antara NPU (2x NPU devices) + CPU, Docker environment
- **Models**: resnet50_big, resnet50_small, yolov3_big, yolov3_small (4 models)
- **QoS targets**: resnet50 infps=30, yolov3 infps=30
- **Duration**: 30s per combination, 2 combinations per mode
- **Schedule**: `gen_schedules_antara/model_schedules_vscore_test.yaml`

### Deployment Transition

| | combination_1 (GOOD) | combination_2 (WORSE) |
|---|---|---|
| resnet50_big | **NPU0** | CPU |
| resnet50_small | **NPU1** | CPU |
| yolov3_big | CPU | CPU |
| yolov3_small | CPU | CPU |

The experiment transitions from an NPU-accelerated deployment to an all-CPU deployment, simulating a degraded deployment decision.

## Results Summary

| Metric | Mode 0 (Stop-and-restart) | Mode 1 (Adaptive hot-swap) | Mode 2 (Reactive rollback) |
|---|---|---|---|
| **Samples** | 56 | 56 | 71 |
| **Avg V-Score** | 4.52 | 4.05 | **2.63** |
| **Max V-Score** | 8.61 | 6.33 | 6.28 |
| **V-Score spike** | 1.87 → 8.6 (max) | 1.69 → 6.3 (max) | 1.8 → 6.3 → rollback → **2.3 (baseline)** |
| **Avg Latency (ms)** | 704.55 | 642.52 | **370.00** |
| **Avg Drop Rate (FPS)** | 60.40 | 58.28 | **55.57** |
| **Avg Throughput (FPS)** | 14.17 | 15.92 | 13.54 |
| **Rollback triggered** | N/A | N/A | **Yes** (V(t) > 5, ε = 5.0 strictly exceeding) |

## Key Observations

### 1. V-Score Spike at Deploy Change

All three modes show a clear V-score increase when transitioning from NPU to CPU:
- **Mode 0** shows the most dramatic spike (~8.6 peak) due to full stop-and-restart downtime
- **Mode 1** has a smoother transition (~6.3 peak) since old workers continue serving during hot-swap
- **Mode 2** detects the spike and triggers rollback after 5s stabilisation

### 2. Reactive Rollback Behavior

Mode 2 (Reactive) successfully detected the QoS degradation:
- Rollback policy: **V(t) > 5 triggers rollback** (ε = 5.0, strictly exceeding threshold)
- After transitioning to combination_2, V(t) spiked to ~6.3 (exceeding ε = 5.0)
- Rollback fired ~6s after deploy change, reverting to combination_1
- Post-rollback, V(t) exponentially decayed back to baseline (~2.3) within ~8s
- This results in the lowest Avg V-Score (2.63) and Avg Latency (370ms) among all modes

### 3. Latency Impact

The all-CPU deployment shows ~3x higher total inference latency compared to NPU-accelerated:
- NPU phase (combination_1): ~300 ms total (4 models, ResNet on NPU)
- CPU phase (combination_2): ~940 ms total (4 models, all on CPU)
- Reactive rollback: after reverting, latency returns to ~300 ms (Avg 370ms overall)

This confirms that offloading ResNet models to NPU significantly reduces per-model latency, freeing CPU resources for YOLO models.

### 4. NPU vs GPU Comparison

Compared to the GPU experiment (same models on GPU+CPU):

| | GPU Experiment | NPU Experiment |
|---|---|---|
| V-Score (accelerated) | ~14 | ~1.8 |
| V-Score (all CPU) | ~54 | ~6.8 |
| V-Score spike ratio | 3.8x | 3.6x |
| Max V-Score (Mode 0) | 54.9 | 8.6 |
| Throughput (accelerated) | ~750 FPS | ~13 FPS |

The NPU experiment shows a similar V-score spike ratio (~3.6x) to the GPU experiment (~3.8x), validating that the reactive deployment mechanism works consistently across heterogeneous accelerators.

## Generated Files

| File | Description |
|---|---|
| `droprate_over_time_antara.pdf` | Drop rate time series (3 modes) |
| `latency_over_time_antara.pdf` | Total inference latency time series |
| `vscore_over_time_antara.pdf` | QoS violation score with spike and rollback |
| `adaptive_comparison_antara_20260403_173820.pdf` | Combined multi-page PDF |
| `comparison_summary_antara.pdf` | Summary table with statistics |
| `adaptive_metrics_mode{0,1,2}_antara_20260403_173820.csv` | Raw per-second metrics |

## Timestamp
Experiment run: 2026-04-03 17:38 KST
