#!/usr/bin/env python3
"""Schedule generation entry point (Mobilint NPU + GPU stack).

The legacy Neubla profiler GUI has been retired. Profiling and schedule
generation are now handled by two standalone tools:

  1. Profile every model on CPU / GPU / NPU (writes the static profile):
        source runtime_env.sh
        $PYTHON_BIN profile_models.py \
            --out xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json

  2. Generate placement combinations (writes model_schedules.yaml):
        $PYTHON_BIN generate_schedules.py \
            --out xgboost_model/schedules/model_schedules.yaml

Then collect training data and train the XGBoost model:
        $PYTHON_BIN schedule_executor_main.py --schedule model_schedules.yaml \
            --duration 10 --auto_start_all
        $PYTHON_BIN xgboost_model/deploy_selector_xgb_suite.py train ...

The GUI-driven best-deployment finder lives in best_deploy_finder_executor.py.
"""

import sys


def main():
    print(__doc__)
    print("[schedule_generator_main] Deprecated launcher. Use profile_models.py "
          "and generate_schedules.py (see above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
