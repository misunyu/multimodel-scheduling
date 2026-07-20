"""B.3 final training: deploy_cpu_gpu from the full 540-combo CPU-GPU collection.
Same recipe as train_npu.py. Writes the FINAL deploy_cpu_gpu artifacts and the
out-of-fold predictions used for honest per-set validation in B.4.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from xgboost_model.deploy_selector_xgb_suite import (
    build_dataset, train_targets, write_coverage, out_of_fold_predictions)

SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad")
WIN = SC / "collect" / "windows_gpu"
SCHED_DIR = ROOT / "xgboost_model" / "schedules" / "collection"
STATIC = ROOT / "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json"
PREFIX = ROOT / "xgboost_model" / "artifacts" / "deploy_cpu_gpu"

X, Y, M = build_dataset(WIN, STATIC, SCHED_DIR, normalize=True)
print(f"dataset: {len(X)} rows, {X.shape[1]} features")
print(f"  view_counts: {sorted(set(M['models'].apply(lambda s: len(s.split(',')))))}")
print(f"  y3_valid (LLM/VLM on accelerator): {int(M['y3_valid'].sum())} / {len(M)}")
print(f"  rate_factors: {sorted(set(M['rate_factor'].dropna()))}")

oof = out_of_fold_predictions(X, Y, M, folds=3, seed=42)
oof_out = SC / "collect" / "oof_gpu.json"
oof_out.write_text(json.dumps({
    "pred_y1": oof["pred_y1"].tolist(), "pred_y2": oof["pred_y2"].tolist(),
    "pred_y3": oof["pred_y3"].tolist(),
    "y1_m": Y["y1_total_throughput_fps"].tolist(), "y2_m": Y["y2_deadline_miss_rate"].tolist(),
    "y3_m": Y["y3_total_tokens_per_s"].tolist(),
    "models": M["models"].tolist(), "rate": M["rate_factor"].tolist(),
    "y3_valid": M["y3_valid"].astype(bool).tolist(),
}))
print(f"out-of-fold saved -> {oof_out.name}")

train_targets(X, Y, PREFIX, M=M)
write_coverage(M, PREFIX)
print(f"trained -> {PREFIX}_y1/y2/y3.json + _coverage.json")

cov = json.loads(Path(str(PREFIX) + "_coverage.json").read_text())
print(f"coverage: {len(cov['model_sets'])} model_sets, view_counts={cov['view_counts']}, "
      f"models={len(cov['models'])}")
