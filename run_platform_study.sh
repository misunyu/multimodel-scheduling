#!/usr/bin/env bash
# Two-platform predictive-placement study (paper-style): CPU+GPU and CPU+NPU.
# For each platform: collect rate-sweep contention data -> train a platform-
# specific 3-target XGBoost -> evaluate (3-fold CV, 3x rate hold-out, paper-style
# Top-1/Top-5 placement selection). Results appended to $REPORT.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime_env.sh"
export RECORD_TIME=0 QT_QPA_PLATFORM=offscreen

STATIC=xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json
SD=xgboost_model/schedules
REPORT=platform_study_report.txt
FILT='reshape_check|^I0[0-9]| INFO\]|^\[20|Model (constructed|launched|disposed)'
: > "$REPORT"

for PLAT in cpu_gpu cpu_npu; do
  echo "==================== PLATFORM: $PLAT ====================" | tee -a "$REPORT"
  PERF="xgboost_model/performance_data/$PLAT"
  mkdir -p "$PERF"
  rm -f "$PERF"/performance*.json results/performance_*.json 2>/dev/null

  echo "[collect] $PLAT ..." | tee -a "$REPORT"
  "$PYTHON_BIN" run_collection.py --schedule "schedules_${PLAT}.yaml" --duration 8 \
      --out "$PERF/performance.json" > "/tmp/collect_${PLAT}.log" 2>&1
  echo "[collect] $PLAT done: $(grep -c ' ok ' "/tmp/collect_${PLAT}.log" 2>/dev/null) ok" | tee -a "$REPORT"

  echo "[train] $PLAT (rates 1x/2x/4x)" | tee -a "$REPORT"
  "$PYTHON_BIN" xgboost_model/deploy_selector_xgb_suite.py train --perf_dir "$PERF" \
      --schedule_dir "$SD" --static_json "$STATIC" \
      --model_out "xgboost_model/artifacts/deploy_${PLAT}" 2>&1 | grep -avE "$FILT" | grep -E "train|OK" | tee -a "$REPORT"

  echo "--- 3-fold CV ---" | tee -a "$REPORT"
  "$PYTHON_BIN" evaluate_model.py cv --perf_dir "$PERF" --schedule_dir "$SD" \
      --static_json "$STATIC" --folds 3 2>&1 | grep -avE "$FILT" | grep -E "target|y1|y2|y3|windows" | tee -a "$REPORT"

  echo "--- 3x rate hold-out ---" | tee -a "$REPORT"
  "$PYTHON_BIN" evaluate_model.py holdout --model_in "xgboost_model/artifacts/deploy_${PLAT}" \
      --test_perf "$PERF" --schedule_dir "$SD" --static_json "$STATIC" --rate_factors 3 2>&1 \
      | grep -avE "$FILT" | grep -E "target|y1|y2|y3|hold" | tee -a "$REPORT"

  echo "--- placement selection (Top-1/Top-5) ---" | tee -a "$REPORT"
  "$PYTHON_BIN" evaluate_model.py select --model_in "xgboost_model/artifacts/deploy_${PLAT}" \
      --test_perf "$PERF" --schedule_dir "$SD" --schedule_yaml "schedules_${PLAT}.yaml" \
      --static_json "$STATIC" --rate_factors 3 2>&1 | grep -avE "$FILT" | grep -E "Top|oracle|Spearman|groups" | tee -a "$REPORT"
  echo "" | tee -a "$REPORT"
done
echo "ALL_DONE" | tee -a "$REPORT"
