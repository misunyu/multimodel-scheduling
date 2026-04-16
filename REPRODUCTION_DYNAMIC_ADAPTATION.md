# Reproduction: Dynamic Load Adaptation Graph

This document describes the data sources and execution scenarios used to generate `dynamic_load_adaptation.pdf`.

## 1. Experimental Setup
The graph is based on real-world experimental data comparing two runtime adaptation modes:
- **Mode 1 (Adaptive Hot-swap)**: Immediately applies a new placement without safeguards.
- **Mode 2 (Reactive/Rollback)**: Our **BoundGuard** approach, which triggers a rollback if the new placement fails to meet QoS within a stabilization period (5 seconds).

### Baseline and Scenario
- **Schedule**: `tests/model_schedules_test.yaml`
- **Transition**: `combination_1` (GPU-optimized) → `combination_2` (CPU-bound/Worse placement)
- **Objective**: Demonstrate that BoundGuard recovers from a poor placement caused by a sudden load change/misprediction, while a standard hot-swap strategy suffers from persistent QoS degradation.

## 2. Execution Commands
To reproduce the raw metrics (CSV), the following commands were used:

```bash
# Mode 1: Adaptive Hot-swap (Static Limitation)
python3 schedule_executor_main.py \
    --schedule tests/model_schedules_test.yaml \
    --duration 30 \
    --adaptive-mode 1 \
    --metrics-csv results/adaptive_metrics_mode1_20260403_135343.csv \
    --auto_start_all

# Mode 2: BoundGuard (Rollback Recovery)
python3 schedule_executor_main.py \
    --schedule tests/model_schedules_test.yaml \
    --duration 30 \
    --adaptive-mode 2 \
    --metrics-csv results/adaptive_metrics_mode2_20260403_135343.csv \
    --auto_start_all
```

*Note: The actual CSV files used in the current graph were generated on 2026-04-03 13:53:43.*

## 3. Data Processing & Plotting
The graph `dynamic_load_adaptation.pdf` was generated using the script `generate_real_dynamic_adaptation.py`.

### Processing Logic:
1. **Time Centering**: The transition point (`combination_1` → `combination_2`) is automatically detected from the CSV and mapped to **t=20s** on the x-axis.
2. **Metric**: The `v_score` (Violation Score) column is plotted.
3. **Annotations**:
    - **t=20s**: Vertical line indicating the "Input Rate Increase & Deploy Change".
    - **BoundGuard Recovery**: Annotation at the rollback point where the score drops back to the baseline.

### How to Regenerate:
Ensure the required CSV files exist in the `results/` directory, then run:
```bash
python3 generate_real_dynamic_adaptation.py
```

## 4. Key Observations in the Graph
- **Static/Hot-swap (Red Dashed Line)**: After the deployment change at t=20s, the violation score spikes to approximately 40-50 and remains high.
- **BoundGuard (Blue Solid Line)**: After the same change at t=20s, the score spikes briefly. However, at t≈28s (after a 5s monitoring window + system overhead), BoundGuard detects the QoS failure and rolls back to `combination_1`, restoring the score to the stable baseline (<15).
